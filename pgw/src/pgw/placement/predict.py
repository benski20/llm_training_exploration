"""Decode tok/s per user from the network path: the calibrated round-trip model (benchmarks/round_trip_model.py).

A decoded token of the client library is a chain of ``trips`` sequential round trips, so::

    ms per token = (trips x RTT + fixed_ms + link_ms) / tau

* ``trips`` and ``fixed_ms`` (lane, device, client work, serving at RTT 0) are
  ``round_trip_model.sdk_ms_per_token(model, 0)`` (H100 provider, the measured 7B calibration; 32B / 72B modelled);
  ``tests/placement/test_predict.py`` keeps them equal to the benchmark module;
* ``link_ms``: the bytes a token moves (every linear's input up, output down, the masked head's logits;
  ``benchmarks/wan_model.py``) over the user's access link, when it is given (consumer links; in-cloud clients have
  25 Gbit/s and it is negligible);
* ``tau``: tokens per pass with strict speculation (1 without).

The placement controller only needs the *difference* an endpoint makes, which is ``trips x delta RTT``: the model is
linear in RTT and the RTT term dominates beyond ~1 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MODELS", "ModelCost", "model_key", "ms_per_token", "tok_s"]


@dataclass(frozen=True)
class ModelCost:
    trips: int  # standard schedule, 4L + 1
    fixed_ms: float  # ms per pass at RTT 0 (lane, device, client, serving)
    up_mb: float  # MB up per row (K32 elements + the head's input)
    down_mb: float  # MB down per row (K32 elements + 152k logits)
    layers: int


#: ``round_trip_model.sdk_ms_per_token(model, 0)`` (trips, pass_ms) and ``scaling_model`` shapes (bytes per row).
MODELS: dict[str, ModelCost] = {
    "7B": ModelCost(113, 37.714, 3.340, 6.171, 28),
    "32B": ModelCost(257, 101.856, 11.031, 19.220, 64),
    "72B": ModelCost(321, 137.007, 17.359, 28.051, 80),
}


def model_key(model: str) -> str:
    """``qwen2.5-7b-instruct`` -> ``7B`` (the predictor's key); a key passes through."""
    m = model.strip()
    if m in MODELS:
        return m
    low = m.lower()
    for k in sorted(MODELS, key=len, reverse=True):
        if f"-{k.lower()}" in low:
            return k
    raise KeyError(f"no latency model for {model!r} (known: {', '.join(MODELS)})")


def ms_per_token(model: str, rtt_ms: float, *, link_mbit: tuple[float, float] | None = None, tau: float = 1.0,
                 fused: bool = False) -> float:
    """Modelled ms per decoded token. ``link_mbit``: (up, down) Mbit/s of the user's access link; ``fused``: the
    2L + 1 schedule (fused2: same RTT term per trip, 1.25x bytes, ~15 % more device time; an approximation)."""
    c = MODELS[model_key(model)]
    trips = 2 * c.layers + 1 if fused else c.trips
    fixed = c.fixed_ms * (1.08 if fused else 1.0)
    ms = trips * max(0.0, rtt_ms) + fixed
    if link_mbit is not None:
        bpe = 1.25 if fused else 1.0
        up, down = link_mbit
        ms += (c.up_mb * bpe * 8 / up + c.down_mb * bpe * 8 / down) * 1e3
    return ms / max(tau, 1e-9)


def tok_s(model: str, rtt_ms: float, **kw: object) -> float:
    """Modelled decode tok/s of one stream (see :func:`ms_per_token`)."""
    return 1e3 / ms_per_token(model, rtt_ms, **kw)  # type: ignore[arg-type]
