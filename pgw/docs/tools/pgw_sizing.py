#!/usr/bin/env python3
"""PGW sizing calculator (standard library only; runs on any operator workstation).

Computes, for a dense transformer served through PGW, the per-token wire volume, mask storage,
round trips, the network latency floor and mask-factory capacity, from the model shape and the
ring preset. Every output is tagged with how it was obtained:

* ``analytic``: exact arithmetic on the model shape (wire bytes, mask bytes, rounds);
* ``derived``: arithmetic on a measured number (e.g. a measured T4 rate scaled to slots/day);
* ``estimate``: rests on an assumption (RTT, scaling a measured rate to another model size).

Examples::

    python pgw_sizing.py --model llama-3-8b --ring K48A20
    python pgw_sizing.py --model llama-3-8b --tokens-per-day 2000000 --rtt-ms 1 --json
    python pgw_sizing.py --hidden 3584 --intermediate 18944 --layers 28 --heads 28 --kv-heads 4 --ring K40A12

The formulas are documented in ``docs/operations/sizing.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_CHECKS",
    "MODELS",
    "RINGS",
    "ModelShape",
    "Ring",
    "SizingReport",
    "load_measurements",
    "main",
    "size",
]

MB = 1_000_000  # decimal megabyte, as used in all PGW documents
GB = 1_000_000_000
TB = 1_000_000_000_000
#: Upper bound of the factory's ``checks`` setting (``pgw.factory.config.FactorySettings.checks``).
MAX_CHECKS = 8

#: Measured T4 mask-generation reference points (docs/measurements.json): (model key, K) -> s/slot.
_T4_REFERENCE = {("llama-3-8b", 48): "maskgen_8b_t4_k48", ("llama-3-8b", 32): "maskgen_8b_t4_k32",
                 ("llama-3-70b", 48): "maskgen_70b_t4_k48", ("llama-3-70b", 32): "maskgen_70b_t4_k32"}


@dataclass(frozen=True)
class Ring:
    """Ring preset ``Z_{2^K}`` with ``A``-bit activations."""

    k: int
    a: int

    @property
    def name(self) -> str:
        """Canonical name, e.g. ``K40A12``."""
        return f"K{self.k}A{self.a}"

    @property
    def bytes_per_element(self) -> int:
        """Packed size of one ring element on the wire and in the store."""
        return self.k // 8

    @property
    def limbs(self) -> int:
        """Number of int8 limb GEMMs per ring matmul on the GPU path."""
        return self.k // 8

    def max_in_features(self) -> int:
        """Largest inner dimension that decodes unambiguously: ``qmax*127*n < 2^(K-1)``."""
        qmax = 2 ** (self.a - 1) - 1
        return (2 ** (self.k - 1) - 1) // (qmax * 127)

    def soundness_bits(self, in_features: int, checks: int) -> int:
        """Freivalds + range-check bound ``c*h`` with ``h = K - ceil(log2(4*qmax*127*n))``."""
        qmax = 2 ** (self.a - 1) - 1
        h = self.k - math.ceil(math.log2(4 * qmax * 127 * in_features))
        return max(0, h) * checks


RINGS: dict[str, Ring] = {r.name: r for r in (Ring(32, 8), Ring(40, 12), Ring(48, 20))}


@dataclass(frozen=True)
class ModelShape:
    """Shape of a dense decoder-only transformer (Llama / Qwen2 family)."""

    name: str
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int | None = None

    def __post_init__(self) -> None:
        """Reject shapes that are not a valid dense transformer (clear error instead of nonsense)."""
        for fname in ("hidden", "intermediate", "layers", "heads", "kv_heads"):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{fname} must be a positive integer, got {v!r}")
        if self.head_dim is not None and (
            isinstance(self.head_dim, bool) or not isinstance(self.head_dim, int) or self.head_dim < 1
        ):
            raise ValueError(f"head_dim must be a positive integer, got {self.head_dim!r}")
        if self.head_dim is None and self.hidden % self.heads:
            raise ValueError(f"hidden ({self.hidden}) is not divisible by heads ({self.heads}); give head_dim")
        if self.heads % self.kv_heads:
            raise ValueError(f"heads ({self.heads}) must be a multiple of kv_heads ({self.kv_heads})")

    @property
    def hd(self) -> int:
        """Head dimension."""
        return self.head_dim if self.head_dim is not None else self.hidden // self.heads

    def linears(self) -> dict[str, tuple[int, int]]:
        """``{group: (m, n)}`` for the four masked groups of one layer (out rows, in features)."""
        q = self.heads * self.hd
        kv = self.kv_heads * self.hd
        return {
            "qkv": (q + 2 * kv, self.hidden),
            "o": (self.hidden, q),
            "gate_up": (2 * self.intermediate, self.hidden),
            "down": (self.hidden, self.intermediate),
        }

    @property
    def linear_params(self) -> int:
        """Number of weights in all masked linears (excludes embeddings and LM head)."""
        return self.layers * sum(m * n for m, n in self.linears().values())

    @property
    def sum_m(self) -> int:
        """Sum of output rows of the four groups (one layer)."""
        return sum(m for m, _ in self.linears().values())

    @property
    def sum_n(self) -> int:
        """Sum of input features of the four groups (one layer)."""
        return sum(n for _, n in self.linears().values())

    @property
    def max_n(self) -> int:
        """Largest inner dimension (decides the ring headroom and the Freivalds bound)."""
        return max(n for _, n in self.linears().values())


MODELS: dict[str, ModelShape] = {
    m.name: m
    for m in (
        ModelShape("qwen2.5-0.5b", 896, 4864, 24, 14, 2),
        ModelShape("qwen2.5-7b", 3584, 18944, 28, 28, 4),
        ModelShape("llama-3-8b", 4096, 14336, 32, 32, 8),
        ModelShape("llama-3-70b", 8192, 28672, 80, 64, 8),
    )
}


def load_measurements(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Measured/derived/estimated numbers from ``docs/measurements.json`` keyed by ``id``."""
    p = path or Path(__file__).resolve().parent.parent / "measurements.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return {e["id"]: e for e in data["entries"]}


@dataclass
class Figure:
    """One number with its unit and provenance."""

    value: float
    unit: str
    kind: str  # analytic | derived | estimate
    note: str = ""


@dataclass
class SizingReport:
    """Result of :func:`size`."""

    model: str
    ring: str
    figures: dict[str, Figure]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {"model": self.model, "ring": self.ring, "figures": {k: asdict(v) for k, v in self.figures.items()}}

    def format(self) -> str:
        """Human-readable table."""
        width = max(len(k) for k in self.figures)
        lines = [f"PGW sizing: {self.model}, ring {self.ring}", ""]
        for k, f in self.figures.items():
            val = f"{f.value:,.4g}" if isinstance(f.value, float) else f"{f.value:,}"
            note = f"  ({f.note})" if f.note else ""
            lines.append(f"  {k:<{width}}  {val:>14} {f.unit:<16} [{f.kind}]{note}")
        return "\n".join(lines)


def size(
    shape: ModelShape,
    ring: Ring,
    *,
    tokens_per_day: float = 0.0,
    rtt_ms: float = 1.0,
    checks: int = 2,
    segment_slots: int = 4096,
    batch_sessions: int = 1,
    measurements: dict[str, dict[str, Any]] | None = None,
) -> SizingReport:
    """Compute the sizing figures for ``shape`` at ``ring``.

    Args:
        shape: Model shape.
        ring: Ring preset.
        tokens_per_day: Prompt + completion tokens per day (each consumes one mask slot).
        rtt_ms: Gateway <-> service round-trip time in milliseconds.
        checks: Freivalds vectors per linear (factory ``checks``).
        segment_slots: Slots per mask segment (factory ``segment_slots``).
        batch_sessions: Concurrent sessions sharing each round trip (decode).
        measurements: Override for :func:`load_measurements`.

    Raises:
        ValueError: invalid argument or a shape the ring cannot decode.
    """
    if not all(math.isfinite(v) for v in (rtt_ms, tokens_per_day)):
        raise ValueError("rtt_ms and tokens_per_day must be finite")
    if rtt_ms < 0 or tokens_per_day < 0 or segment_slots < 1 or batch_sessions < 1:
        raise ValueError("arguments must be non-negative (segment_slots, batch_sessions >= 1)")
    if not 0 <= checks <= MAX_CHECKS:
        raise ValueError(f"checks must be between 0 and {MAX_CHECKS} (the factory's limit)")
    if shape.max_n > ring.max_in_features():
        raise ValueError(f"{ring.name} cannot decode inner dimension {shape.max_n} (max {ring.max_in_features()})")
    meas = measurements if measurements is not None else load_measurements()
    b = ring.bytes_per_element
    L = shape.layers
    up = shape.sum_n * L * b
    down = shape.sum_m * L * b
    mask = down  # W r rows; r itself is regenerated from the segment seed
    # per segment: seed (32 B) + u^T W8 (checks x n per linear), amortised per slot. Header JSON,
    # wrapped DEK and AEAD tags (16 B per chunk) are not counted: well under 0.1 % of a segment.
    seg_overhead = 32 + checks * shape.sum_n * L * b
    rounds = 4 * L
    f: dict[str, Figure] = {
        "linear_params": Figure(float(shape.linear_params), "weights", "analytic", "masked linears only"),
        "rounds_per_token": Figure(float(rounds), "round trips", "analytic",
                                   "4 groups x layers; prefill: same per prompt"),
        "wire_up_per_token": Figure(up / MB, "MB", "analytic", "x_hat payload"),
        "wire_down_per_token": Figure(down / MB, "MB", "analytic", "y_hat payload"),
        "wire_total_per_token": Figure((up + down) / MB, "MB", "analytic", "payload; framing adds < 0.1 %"),
        "mask_storage_per_slot": Figure((mask + seg_overhead / segment_slots) / MB, "MB", "analytic",
                                        "W r rows + amortised segment header/u^T W"),
        "slots_per_tb": Figure(TB / (mask + seg_overhead / segment_slots), "slots", "analytic"),
        "freivalds_bits_worst_linear": Figure(float(ring.soundness_bits(shape.max_n, checks)), "bits", "analytic",
                                              f"c={checks}, n={shape.max_n}; false accept <= 2^-bits per call"),
        "latency_floor_per_token": Figure(rounds * rtt_ms / 1000.0, "s", "estimate",
                                          f"rounds x RTT at {rtt_ms:g} ms; "
                                          f"shared by {batch_sessions} batched sessions"),
        "gateway_online_ring_elements": Figure(float(shape.sum_n * L + shape.sum_m * L), "elements/token",
                                               "analytic", "regenerate r and subtract W r"),
    }
    if tokens_per_day > 0:
        f["wire_per_day"] = Figure(tokens_per_day * (up + down) / TB, "TB/day", "analytic")
        f["mean_bandwidth"] = Figure(tokens_per_day * (up + down) * 8 / 86400 / 1e9, "Gbit/s", "analytic",
                                     "daily average, both directions")
        f["mask_storage_per_day"] = Figure(tokens_per_day * mask / TB, "TB/day", "analytic",
                                           "consumed and shredded; size the store for the refill window")
    # factory throughput: measured reference when available, else scale the closest measured point (estimate)
    key = (shape.name, ring.k)
    same_model_k48 = _T4_REFERENCE.get((shape.name, 48))
    if key in _T4_REFERENCE and _T4_REFERENCE[key] in meas:
        s_per_slot = float(meas[_T4_REFERENCE[key]]["value"])
        kind, note = "derived", "measured T4 rate"
    elif same_model_k48 is not None and same_model_k48 in meas:
        s_per_slot = float(meas[same_model_k48]["value"]) * ring.limbs / 6
        kind, note = "estimate", "measured K48 T4 rate of this model scaled by limbs"
    else:
        ref = MODELS["llama-3-8b"]
        ref_s = float(meas["maskgen_8b_t4_k48"]["value"])
        s_per_slot = ref_s * (shape.linear_params * ring.limbs) / (ref.linear_params * 6)
        kind, note = "estimate", "8B K48 T4 rate scaled by weights x limbs"
    f["factory_t4_s_per_slot"] = Figure(s_per_slot, "s/slot", kind, note)
    f["factory_t4_slots_per_hour"] = Figure(3600.0 / s_per_slot, "slots/h", kind, note)
    if tokens_per_day > 0:
        f["factory_t4_gpus_needed"] = Figure(tokens_per_day * s_per_slot / 86400.0, "T4 GPUs (24/7)", kind,
                                             "before headroom; add 30-50 % for refill bursts")
    cpu_tops = float(meas["cpu_fbgemm_tops"]["value"])
    cpu_s = 2.0 * shape.linear_params * ring.limbs / (cpu_tops * 1e12)
    f["factory_cpu4_s_per_slot"] = Figure(cpu_s, "s/slot", "derived",
                                          f"GEMM time at {cpu_tops:g} int8 TOPS (4 vCPU, measured)")
    return SizingReport(shape.name, ring.name, f)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PGW sizing calculator",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Presets: " + ", ".join(sorted(MODELS)))
    p.add_argument("--model", choices=sorted(MODELS), default=None)
    p.add_argument("--hidden", type=int)
    p.add_argument("--intermediate", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--kv-heads", type=int)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--ring", choices=sorted(RINGS), default="K40A12")
    p.add_argument("--tokens-per-day", type=float, default=0.0)
    p.add_argument("--rtt-ms", type=float, default=1.0)
    p.add_argument("--checks", type=int, default=2)
    p.add_argument("--segment-slots", type=int, default=4096)
    p.add_argument("--batch-sessions", type=int, default=1)
    p.add_argument("--measurements", type=Path, default=None, help="Alternative measurements.json")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code (0 ok, 2 usage error)."""
    a = _parse(argv)
    shape_flags = ("hidden", "intermediate", "layers", "heads", "kv_heads", "head_dim")
    if a.model:
        given = [n for n in shape_flags if getattr(a, n) is not None]
        if given:
            flags = ", --".join(g.replace("_", "-") for g in given)
            print(f"error: --model {a.model} cannot be combined with --{flags}", file=sys.stderr)
            return 2
        shape = MODELS[a.model]
    else:
        missing = [n for n in ("hidden", "intermediate", "layers", "heads", "kv_heads") if getattr(a, n) is None]
        if missing:
            flags = ", --".join(m.replace("_", "-") for m in missing)
            print(f"error: give --model or all of --{flags}", file=sys.stderr)
            return 2
    try:
        if not a.model:
            shape = ModelShape("custom", a.hidden, a.intermediate, a.layers, a.heads, a.kv_heads, a.head_dim)
        meas = load_measurements(a.measurements)
        rep = size(shape, RINGS[a.ring], tokens_per_day=a.tokens_per_day, rtt_ms=a.rtt_ms, checks=a.checks,
                   segment_slots=a.segment_slots, batch_sessions=a.batch_sessions, measurements=meas)
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(rep.to_dict(), indent=2) if a.json else rep.format())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
