"""Generate cross-language test vectors for pgw-core-rs from the Python reference (``pgw.core``).

Usage (from the ``pgw`` directory)::

    python rust/tools/gen_vectors.py            # rewrite tests/vectors/core_v1.json
    python rust/tools/gen_vectors.py --check    # exit 1 if the file differs from a fresh build

All inputs are derived deterministically from fixed public seeds, so regenerating produces the same
bytes. The vectors are a protocol artifact: a diff here means the Python core changed behaviour,
which is a protocol break (SPEC.md 7.3) unless SPEC and ``INTERFACE_NOTES.md`` say otherwise.

Encoding: ring values are decimal integers (canonical, ``[0, 2^K)``); byte strings and float64 bit
patterns are lowercase hex; labels are ``{"s": str} | {"i": int} | {"b": hex}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.core import (
    RingParams,
    SeedExpander,
    balanced_digits,
    freivalds_check,
    freivalds_precompute,
    mask_activations,
    pack_ring,
    quantize_activations,
    quantize_weight_rows,
    ringmm,
    soundness_bits,
    unmask_outputs,
)
from pgw.core.prng import Label, encode_labels

VECTOR_FILE = Path(__file__).resolve().parents[2] / "tests" / "vectors" / "core_v1.json"
FORMAT_VERSION = 1


def _seed(tag: str) -> bytes:
    return hashlib.sha256(b"pgw-test-vector/" + tag.encode()).digest()


def _lab(label: Label) -> dict[str, Any]:
    if isinstance(label, str):
        return {"s": label}
    if isinstance(label, int):
        return {"i": label}
    return {"b": bytes(label).hex()}


def _ints(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.reshape(-1).tolist()]


def _f64_bits(t: torch.Tensor | np.ndarray) -> list[str]:
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return [struct.pack(">d", float(v)).hex() for v in a.astype(np.float64).reshape(-1)]


def _f32_bits(t: torch.Tensor) -> list[str]:
    return [struct.pack(">f", float(v)).hex() for v in t.to(torch.float32).reshape(-1).tolist()]


def prng_vectors() -> dict[str, Any]:
    kat = bytes(range(32))
    seeds = {"kat": kat, "zero": bytes(32), "h1": _seed("prng-1")}
    label_sets: list[list[Label]] = [
        [],
        ["mask", "0.qkv", 0],
        ["r", "3.gate_up"],
        ["u", "27.down"],
        [1, "1", b"1"],
        [-(2**63), 2**63 - 1, -1],
        ["é中", b"\x00\xff"],
    ]
    enc = [{"labels": [_lab(x) for x in ls], "info": encode_labels(ls).hex()} for ls in label_sets]
    streams = []
    for name, seed in seeds.items():
        sx = SeedExpander(seed)
        for ls in label_sets[:4]:
            streams.append(
                {
                    "seed": seed.hex(),
                    "seed_name": name,
                    "labels": [_lab(x) for x in ls],
                    "key": sx._key(ls).hex(),
                    "bytes": sx.random_bytes(200, *ls).hex(),
                }
            )
    ring_uniform = []
    sx = SeedExpander(seeds["h1"])
    for k in (16, 32, 40, 48, 56):
        ring_uniform.append(
            {
                "seed": seeds["h1"].hex(),
                "labels": [_lab("u"), _lab(f"k{k}")],
                "k": k,
                "values": _ints(sx.ring_uniform(37, k, "u", f"k{k}")),
            }
        )
    rows = []
    for cols, start, count, k in [
        (1, 0, 5, 40),
        (3, 7, 4, 40),
        (13, 5, 2, 48),
        (896, 2, 3, 32),
        (5, 0, 40, 40),
        (4096, 100_000, 1, 40),
        (7, 2**26 + 3, 2, 48),
    ]:
        lab: list[Label] = ["r", "0.qkv"]
        rows.append(
            {
                "seed": kat.hex(),
                "labels": [_lab(x) for x in lab],
                "cols": cols,
                "row_start": start,
                "row_count": count,
                "k": k,
                "values": _ints(SeedExpander(kat).ring_uniform_rows(cols, k, start, count, *lab)),
            }
        )
    uniform_int = [
        {
            "seed": kat.hex(),
            "labels": [_lab("w")],
            "low": lo,
            "high": hi,
            "values": _ints(SeedExpander(kat).uniform_int(25, lo, hi, "w")),
        }
        for lo, hi in [(-127, 128), (0, 2), (-(2**31), 2**31), (5, 6)]
    ]
    derive = []
    for ls in ([("segment"), 1], ["c"], []):
        derive.append(
            {"seed": kat.hex(), "labels": [_lab(x) for x in ls], "derived": SeedExpander(kat).derive_seed(*ls).hex()}
        )
    return {
        "encode_labels": enc,
        "streams": streams,
        "ring_uniform": ring_uniform,
        "ring_uniform_rows": rows,
        "uniform_int": uniform_int,
        "derive_seed": derive,
    }


def ring_vectors() -> dict[str, Any]:
    params = []
    for name in ("K32A8", "K40A12", "K48A20", "K56A30", "K16A4"):
        p = RingParams.from_name(name)
        params.append(
            {
                "name": name,
                "k": p.k,
                "a": p.a,
                "qmax": p.qmax,
                "max_in_features": p.max_in_features(),
                "bounds": [
                    {"n": n, "output_bound": p.output_bound(n), "soundness_bits_c2": int(soundness_bits(p, n, 2))}
                    for n in (1, 896, 4096, 4864, 14336)
                    if n <= max(1, p.max_in_features())
                ],
            }
        )
    digits = []
    sx = SeedExpander(_seed("digits"))
    for k in (16, 32, 40, 48, 56):
        mask = (1 << k) - 1
        vals = [0, 1, 127, 128, 255, 256, mask, mask - 127, mask >> 1, (mask >> 1) + 1]
        vals += _ints(sx.ring_uniform(6, k, "d", k))
        t = torch.tensor(vals, dtype=torch.int64)
        for bits in (8, 26):
            ds = balanced_digits(t, k, bits)
            digits.append(
                {"k": k, "bits": bits, "values": vals, "digits": [[int(d[i]) for d in ds] for i in range(len(vals))]}
            )
    packing = []
    for k in (16, 32, 40, 48, 56, 64):
        mask = (1 << k) - 1
        vals = [0, 1, mask, 0x0102030405060708 & mask, (mask >> 1)]
        arr = np.array(vals, dtype=np.uint64)
        packing.append({"k": k, "values": vals, "packed": pack_ring(arr, k).hex()})
    return {"params": params, "balanced_digits": digits, "pack_ring": packing}


def linear_vectors() -> list[dict[str, Any]]:
    out = []
    cases = [
        ("K32A8", 3, 24, 16, False),
        ("K40A12", 4, 40, 24, True),
        ("K48A20", 2, 33, 9, False),
        ("K40A12", 1, 8, 5, False),
    ]
    for idx, (ring, rows, n, m, use_m128) in enumerate(cases):
        p = RingParams.from_name(ring)
        seed = _seed(f"linear-{idx}")
        sx = SeedExpander(seed)
        lid = f"{idx}.qkv"
        x = sx.uniform_int((rows, n), -(2**20), 2**20, "x").to(torch.float64) / 2**16
        w = sx.uniform_int((m, n), -(2**20), 2**20, "wf").to(torch.float64) / 2**18
        x[0, 0] = 0.5 * float(x[0].abs().max())  # an interior value at half the row maximum
        xq, scale_x = quantize_activations(x, p)
        w8, sw = quantize_weight_rows(w)
        if use_m128:
            w8[0, 0] = -128  # kernels must handle -128 exactly even though packs never produce it
        r = sx.ring_uniform_rows(n, p.k, 5, rows, "r", lid)
        u = sx.ring_uniform((2, m), p.k, "u", lid)
        x_hat = mask_activations(xq, r, p)
        y_hat = ringmm(x_hat, w8, p, backend="cpu_f64")
        wr = ringmm(r, w8, p, backend="cpu_f64")
        yq = unmask_outputs(y_hat, wr, p, in_features=None if use_m128 else n)
        assert torch.equal(yq, xq @ w8.to(torch.int64).t()), "python core is not exact"
        uw = freivalds_precompute(u, w8, p)
        tampered = y_hat.clone()
        tampered[0, 0] = (tampered[0, 0] + 1) & p.mask
        if rows > 1:
            tampered[rows - 1, m - 1] = (tampered[rows - 1, m - 1] + (1 << (p.k - 1))) & p.mask
        out.append(
            {
                "ring": ring,
                "rows": rows,
                "n": n,
                "m": m,
                "checks": 2,
                "seed": seed.hex(),
                "linear": lid,
                "mask_row_start": 5,
                "x_f64": _f64_bits(x),
                "w_f64": _f64_bits(w),
                "xq": _ints(xq),
                "sx_f64": _f64_bits(scale_x),
                "w8": _ints(w8),
                "sw_f32": _f32_bits(sw),
                "r": _ints(r),
                "u": _ints(u),
                "x_hat": _ints(x_hat),
                "y_hat": _ints(y_hat),
                "wr": _ints(wr),
                "yq": _ints(yq),
                "uw": _ints(uw),
                "x_hat_packed_sha256": hashlib.sha256(pack_ring(x_hat, p)).hexdigest(),
                "tampered_y_hat": _ints(tampered),
                "tampered_verdicts": [bool(v) for v in freivalds_check(x_hat, tampered, u, uw, p).tolist()],
            }
        )
    return out


def build() -> dict[str, Any]:
    """Build the full vector document."""
    return {
        "format": FORMAT_VERSION,
        "protocol": 1,
        "description": "pgw.core reference outputs; consumed by rust/pgw-core-rs/tests/vectors.rs",
        "prng": prng_vectors(),
        "ring": ring_vectors(),
        "linear": linear_vectors(),
    }


def render(doc: dict[str, Any]) -> str:
    """Stable JSON rendering (sorted keys, one trailing newline)."""
    return json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify instead of writing")
    ap.add_argument("--out", type=Path, default=VECTOR_FILE)
    args = ap.parse_args(argv)
    text = render(build())
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is out of date with pgw.core", file=sys.stderr)
            return 1
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
