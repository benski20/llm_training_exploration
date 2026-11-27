"""Narrow wire rings on the 0.5B CPU path: bytes per row before / after, exactness, carry statistics, pack CPU.

For each mode (``weight-free`` = m = max_j w_j, and carry modes m = 16 / 20 / 24) the narrow in-process backend
(``pgw.engine.narrow_backend``: mask, provider product on the packed masked values only, unmask, range + p31
Freivalds) decodes ``--tokens`` greedy tokens after a chat prompt with the masked int8 head, and the plain
reference (``PlainBackend``) replays the same tokens: every logit must be bit-identical. Reports bytes per prefill
row and per decode row against today's K32 wire (4 bytes per element), per linear kind, the measured mean
``sum |xq| / n`` per linear kind (the expected carry density at any m is ``E|xq| / 2^m``), and the CPU of bit
packing per decode row at 0.5B / 7B / 32B shapes (numpy reference and the native unpack of ``pgw_fast``).

    python benchmarks/narrow_wire_bench.py --hf-dir <Qwen2.5-0.5B-Instruct snapshot> --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SHAPES = {  # (layers, d, I, qkv out, vocab)
    "0.5B": (24, 896, 4864, 1152, 151936), "7B": (28, 3584, 18944, 4608, 152064),
    "32B": (64, 5120, 27648, 7168, 152064)}


def pack_cpu(reps: int = 5) -> dict[str, Any]:
    """ms per decode row (one row through every linear + head) to pack / unpack, by width, one core."""
    from pgw.hemask.wire import pack_bits, unpack_bits

    try:
        import pgw_fast
        native = getattr(pgw_fast, "he_unpack_shift", None)
    except ImportError:  # pragma: no cover
        native = None
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {}
    for model, (L, d, inter, q, vocab) in SHAPES.items():
        ins = L * (3 * d + inter) + d
        outs = L * (q + 2 * d + 2 * inter) + vocab
        res: dict[str, float] = {}
        for label, count in (("up", ins), ("down", outs)):
            count = (count + 7) // 8 * 8
            v = rng.integers(0, 1 << 32, count, dtype=np.uint64)
            # today's K32 wire: one cast each way
            t0 = time.perf_counter()
            for _ in range(reps):
                b = v.astype("<u4").tobytes()
            res[f"{label}_k32_pack_ms"] = (time.perf_counter() - t0) / reps * 1e3
            t0 = time.perf_counter()
            for _ in range(reps):
                np.frombuffer(b, "<u4").astype(np.uint64)
            res[f"{label}_k32_unpack_ms"] = (time.perf_counter() - t0) / reps * 1e3
            for bits in (28, 20, 16):
                t0 = time.perf_counter()
                for _ in range(reps):
                    pb = pack_bits(v, bits)
                res[f"{label}_{bits}b_pack_ms"] = (time.perf_counter() - t0) / reps * 1e3
                t0 = time.perf_counter()
                for _ in range(reps):
                    unpack_bits(pb, count, bits)
                res[f"{label}_{bits}b_unpack_ms"] = (time.perf_counter() - t0) / reps * 1e3
                if native is not None:
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        native(bytes(pb), count, bits, 0, 1)
                    res[f"{label}_{bits}b_unpack_native_ms"] = (time.perf_counter() - t0) / reps * 1e3
        out[model] = {k: round(x, 3) for k, x in res.items()}
    return out


def run_mode(pack: Any, ring: Any, in_bits: int | None, ids: torch.Tensor, tokens: int) -> dict[str, Any]:
    from pgw.engine.backends import PlainBackend
    from pgw.engine.masked_head import enable_masked_head
    from pgw.engine.model import TransformerModel
    from pgw.engine.narrow_backend import NarrowMaskedBackend

    be = NarrowMaskedBackend(pack, ring, in_bits=in_bits)
    absx: dict[str, list[float]] = {}

    def observe(lid: Any, xq: np.ndarray) -> None:
        a = absx.setdefault(lid.kind.value, [0.0, 0.0, 0.0])
        a[0] += float(np.abs(xq).sum())
        a[1] += float(xq.size)
        a[2] = max(a[2], float((np.abs(xq) > 0).mean()))

    be.observe = observe
    model = TransformerModel(pack, be)
    enable_masked_head(model)
    plain = TransformerModel(pack, PlainBackend(pack, ring))
    enable_masked_head(plain)
    w = be.wire
    exact = True
    with torch.no_grad():
        c1, c2 = model.new_cache(len(ids) + tokens + 1), plain.new_cache(len(ids) + tokens + 1)
        t0 = time.perf_counter()
        a = model.forward(ids, c1)
        t_pre = time.perf_counter() - t0
        exact &= bool(torch.equal(a, plain.forward(ids, c2)))
        pre = (w.up, w.down, w.k_up, w.k_down, w.carries, w.carry_slots)
        t0 = time.perf_counter()
        for _ in range(tokens):
            x = torch.tensor([int(a[-1].argmax())])
            a = model.forward(x, c1)
            exact &= bool(torch.equal(a, plain.forward(x, c2)))
        t_dec = time.perf_counter() - t0
    P = int(ids.numel())
    dec = [cur - p for cur, p in zip((w.up, w.down, w.k_up, w.k_down, w.carries, w.carry_slots), pre, strict=True)]
    plans = {}
    for key, lin in be._lins.items():
        kind = key.split(".", 1)[1]
        pl = plans.setdefault(kind, [0, 0, 0])
        pl[0] += lin.plan.in_bits * lin.plan.in_features
        pl[1] += int(lin.plan.out_bits.sum())
        pl[2] += lin.plan.out_features
    return {
        "in_bits": in_bits or "weight-free",
        "exact": exact,
        "prefill_rows": P,
        "prefill_bytes_per_row": {"up": pre[0] / P, "down": pre[1] / P, "k32_up": pre[2] / P,
                                  "k32_down": pre[3] / P},
        "decode_bytes_per_row": {"up": dec[0] / tokens, "down": dec[1] / tokens, "k32_up": dec[2] / tokens,
                                 "k32_down": dec[3] / tokens},
        "decode_saving_pct": {"up": 100 * (1 - dec[0] / dec[2]), "down": 100 * (1 - dec[1] / dec[3]),
                              "total": 100 * (1 - (dec[0] + dec[1]) / (dec[2] + dec[3]))},
        "carries_per_decode_row": dec[4] / tokens, "carry_slots_per_decode_row": dec[5] / tokens,
        "carry_overflows": w.carry_overflows,
        "mean_abs_xq": {k: v[0] / max(1.0, v[1]) for k, v in absx.items()},
        "mean_out_bits": {k: v[1] / v[2] for k, v in plans.items()},
        "per_linear_decode_and_prefill": w.per_linear,
        "cpu_s": {"prefill": t_pre, "decode_per_token": t_dec / tokens},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--modes", default="wf,16,20,24")
    ap.add_argument("--prompt", default="<|im_start|>user\nWhy is the sky blue? Answer in two sentences.<|im_end|>\n"
                                        "<|im_start|>assistant\n")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-pack-cpu", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    from transformers import AutoTokenizer

    from pgw.core.ring import RingParams
    from pgw.engine.masked_head import MaskedHeadPack
    from pgw.engine.pack import InMemoryPack

    ring = RingParams(32, 12, 27)
    pack = MaskedHeadPack(InMemoryPack.from_hf(Path(args.hf_dir)))
    ids = AutoTokenizer.from_pretrained(args.hf_dir)(args.prompt, return_tensors="pt").input_ids[0]
    out: dict[str, Any] = {"model": "Qwen2.5-0.5B-Instruct", "ring": ring.name, "tokens": args.tokens, "modes": {}}
    for mode in args.modes.split(","):
        bits = None if mode == "wf" else int(mode)
        r = run_mode(pack, ring, bits, ids, args.tokens)
        out["modes"][mode] = r
        print(mode, json.dumps({k: r[k] for k in ("exact", "decode_bytes_per_row", "decode_saving_pct",
                                                  "carries_per_decode_row", "carry_slots_per_decode_row")}),
              flush=True)
        if not r["exact"]:
            raise SystemExit(f"{mode}: narrow != plain")
    if not args.no_pack_cpu:
        out["pack_cpu_ms_per_decode_row"] = pack_cpu()
        print(json.dumps(out["pack_cpu_ms_per_decode_row"], indent=1))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
