"""The client library's linear schedule per model: ``"standard"`` (default) or ``"fused"`` (opt-in, high RTT).

``standard`` is the measured default stack: per layer four masked provider round trips (qkv, o, gate_up, down) and
the masked int8 LM head: ``4L + 1`` sequential trips per forward pass, K32 ring, p31 checks.

``fused`` (``research/reviews/consumer-round-trips.md``) is fused2 with the final norm and the LM head folded into the
last trip and layer 0's qkv computed locally: **2L trips per pass** (0.5B 48 instead of 97, 7B 56 instead of 113),
ring K56A12C53 (7 bytes per element: 1.75x the bytes per element of K32), Freivalds checks over ``F_(2^61-1)`` on both
stages of every trip, masks from the encrypted pks-v3 factory by the top-digit K56 plan
(:mod:`pgw.factory.fused2_he`). Privacy and integrity are the standard schedule's (uniformly masked ring elements of
public shapes, single-use masks, every returned value range- and Freivalds-checked before use). Its outputs are a
**different pinned reference**: the fixed-point residual stream and the folded int8 weights round differently, so
tokens and logprobs are not bit-identical to ``standard`` (Qwen2.5-0.5B, 4088 wikitext tokens: ppl 19.90 vs 19.85,
the same top-1 token at 97.4 % of positions; ``benchmarks/results/fused2/quality_qwen05b_headfold_i8.json``).

The schedule is therefore chosen **once per model and client** (when the model starts) and never changes while the
client runs: every answer, every conversation turn and every retry of that client uses the same schedule. A new
client (process) may choose differently; an application that needs one reference across restarts pins
``schedule="standard"`` or ``"fused"`` instead of ``"auto"``.

``auto`` measures the link to the provider when the model starts (:func:`measure_link`: the round-trip time of small
authenticated probes, and the upload and download bandwidth of 512 KiB probes on ``POST /pgw/v1/link/probe``) and
estimates the time per token of both schedules from public shapes only (:func:`estimate_ms`): ``trips x RTT +
bytes up / upload + bytes down / download + client work per trip``. It picks ``fused`` when the provider serves it,
the RTT is at least ``schedule_auto_rtt_ms`` and the fused estimate is lower by at least ``schedule_auto_margin``;
otherwise ``standard``. Bandwidth matters because fused moves 1.75x the bytes per element: on a slow uplink with a
moderate RTT the extra bytes can cost more than the saved trips. The decision and its inputs are recorded in the
client's status (``client.status()["models"][...]["schedule_decision"]``). No content is sent by the probes (zero
bytes) and the provider learns nothing it does not see from the masked traffic itself (the trip count per token).
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["FUSED_RING", "LinkMeasurement", "ScheduleDecision", "decide", "estimate_ms", "measure_link"]

log = logging.getLogger(__name__)

#: The fused schedule's ring (K56, activations 12 bits, outputs capped at 2^53: p61 checks).
FUSED_RING = "K56A12C53"
STANDARD_RING_BYTES = 4
FUSED_RING_BYTES = 7
PROBE_PATH = "/pgw/v1/link/probe"
#: Client work per trip (ms), measured on one core (research/reviews/consumer-round-trips.md section 9, 32B widths,
#: scaled by the model's width): standard 0.105 ms, fused pair trip 0.58 ms at 32B (hidden 5120).
_CLIENT_MS_32B = {"standard": 0.105, "fused": 0.58}


@dataclass(frozen=True)
class LinkMeasurement:
    """What ``auto`` measured (public, no secrets)."""

    rtt_ms: float
    up_mbps: float | None
    down_mbps: float | None
    probes: int
    seconds: float


@dataclass(frozen=True)
class ScheduleDecision:
    """The schedule of one model for the life of the client, and why (public)."""

    requested: str
    schedule: str
    reason: str
    link: LinkMeasurement | None = None
    est_ms_standard: float | None = None
    est_ms_fused: float | None = None
    fused_served: bool | None = None
    decided_at: float = 0.0

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        if self.link is not None:
            d["link"] = asdict(self.link)
        return d


def _probe(client: Any, up: int, down: int) -> float:
    t0 = time.perf_counter()
    r = client.post(PROBE_PATH, content=bytes(up), headers={"PGW-Protocol": "1", "PGW-Probe-Reply-Bytes": str(down),
                                                            "Content-Type": "application/octet-stream"})
    r.read()
    dt = time.perf_counter() - t0
    if r.status_code != 200 or len(r.content) != down:
        raise RuntimeError(f"link probe answered {r.status_code}")
    return dt


def measure_link(session: Any, *, rounds: int = 5, size: int = 512 << 10, timeout_s: float = 30.0) -> LinkMeasurement:
    """RTT (median of ``rounds`` empty probes after one warm-up on a kept-alive connection) and bandwidth each way
    (``size`` bytes up, then down; the RTT subtracted). Raises on a provider without the probe route."""
    t0 = time.perf_counter()
    with session.http_client(timeout_s=timeout_s) as c:
        _probe(c, 0, 0)  # connection, TLS and the first request off the measurement
        rtts = [_probe(c, 0, 0) for _ in range(max(1, int(rounds)))]
        rtt = statistics.median(rtts)
        up_s = _probe(c, size, 0) - rtt
        down_s = _probe(c, 0, size) - rtt
    mbps = lambda s: None if s <= 1e-5 else size * 8 / s / 1e6  # noqa: E731 - small local helper
    return LinkMeasurement(rtt_ms=rtt * 1e3, up_mbps=mbps(up_s), down_mbps=mbps(down_s), probes=len(rtts) + 3,
                           seconds=time.perf_counter() - t0)


def pass_shapes(arch: Any) -> dict[str, dict[str, float]]:
    """Trips and elements up / down per one-row decode pass of both schedules (public shapes only)."""
    L, d = int(arch.num_layers), int(arch.hidden_size)
    inter = int(arch.intermediate_size)
    hd = int(getattr(arch, "head_dim", 0) or d // int(arch.num_heads))
    qkv = (int(arch.num_heads) + 2 * int(arch.num_kv_heads)) * hd
    vocab = int(arch.vocab_size)
    # standard: qkv (d -> qkv), o (d -> d), gate_up (d -> 2I), down (I -> d) per layer, then the head (d -> vocab)
    std_up = L * (d + d + d + inter) + d
    std_down = L * (qkv + d + 2 * inter + d) + vocab
    # fused (2L trips, local qkv0): per layer pair A (x: d, R: d -> H: d, Z: 2I), pair B (x: I, R: d -> H: d,
    # Z: qkv, or the head's vocab in the last layer)
    f_up = L * ((d + d) + (inter + d))
    f_down = L * ((d + 2 * inter) + d) + (L - 1) * qkv + vocab
    sb, fb = STANDARD_RING_BYTES, FUSED_RING_BYTES
    return {"standard": {"trips": 4 * L + 1, "up": std_up * sb, "down": std_down * sb},
            "fused": {"trips": 2 * L, "up": f_up * fb, "down": f_down * fb}}


def estimate_ms(arch: Any, link: LinkMeasurement, *, client_ms: dict[str, float] | None = None) -> dict[str, float]:
    """Estimated ms per decoded token (one row per pass) of each schedule on ``link``: trips x RTT + bytes / bandwidth
    each way + client work per trip (scaled from the 32B measurement by the hidden size; ``client_ms`` overrides).
    The provider's device time is the same order for both schedules and is left out."""
    shapes = pass_shapes(arch)
    scale = int(arch.hidden_size) / 5120.0
    cw = client_ms or {k: v * scale for k, v in _CLIENT_MS_32B.items()}
    out = {}
    for name, s in shapes.items():
        t = s["trips"] * (link.rtt_ms + cw[name])
        if link.up_mbps:
            t += s["up"] * 8 / (link.up_mbps * 1e6) * 1e3
        if link.down_mbps:
            t += s["down"] * 8 / (link.down_mbps * 1e6) * 1e3
        out[name] = t
    return out


def decide(requested: str, *, fused_served: bool, arch: Any = None, session: Any = None,
           rtt_threshold_ms: float = 3.0, margin: float = 0.1, link: LinkMeasurement | None = None
           ) -> ScheduleDecision:
    """The schedule of a model (see the module docstring). ``requested``: ``standard`` / ``fused`` / ``auto``.

    ``fused`` when the provider does not serve it raises :class:`pgw.sdk.errors.NotFoundError` (never a silent
    fallback to another reference); ``auto`` falls back to ``standard`` with the reason recorded."""
    from pgw.sdk.errors import NotFoundError

    now = time.time()
    if requested == "standard":
        return ScheduleDecision("standard", "standard", "requested", fused_served=fused_served, decided_at=now)
    if requested == "fused":
        if not fused_served:
            raise NotFoundError("schedule='fused' but the provider does not serve the fused schedule of this model "
                                "(PGW_SERVICE_FUSED2_FROM_PACK with PGW_SERVICE_FUSED2_HEAD=on and fused2 maskgen)")
        return ScheduleDecision("fused", "fused", "requested", fused_served=True, decided_at=now)
    if not fused_served:
        return ScheduleDecision("auto", "standard", "the provider does not serve the fused schedule",
                                fused_served=False, decided_at=now)
    if link is None:
        forced = os.environ.get("PGW_SDK_LINK_RTT_MS", "").strip()  # tests / operators: a known link, no probe
        if forced:
            link = LinkMeasurement(float(forced), None, None, 0, 0.0)
        else:
            try:
                link = measure_link(session)
            except Exception as exc:  # no probe route (an older provider) or a transient failure: the default
                return ScheduleDecision("auto", "standard", f"link not measured ({type(exc).__name__})",
                                        fused_served=True, decided_at=now)
    est = estimate_ms(arch, link)
    if link.rtt_ms < float(rtt_threshold_ms):
        why = f"RTT {link.rtt_ms:.2f} ms below the {rtt_threshold_ms:g} ms threshold"
        sched = "standard"
    elif est["fused"] <= est["standard"] * (1.0 - float(margin)):
        why = f"RTT {link.rtt_ms:.2f} ms: fused estimated {est['fused']:.0f} vs {est['standard']:.0f} ms per token"
        sched = "fused"
    else:
        why = (f"fused not faster enough on this link ({est['fused']:.0f} vs {est['standard']:.0f} ms per token: "
               "bandwidth or client work)")
        sched = "standard"
    d = ScheduleDecision("auto", sched, why, link, est["standard"], est["fused"], True, now)
    log.info("schedule auto: %s (%s)", sched, why)
    return d
