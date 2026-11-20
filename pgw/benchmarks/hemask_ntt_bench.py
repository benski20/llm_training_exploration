"""Customer-side encrypted-factory products: NTT (``pgw.hemask.ntt``) vs the dense path, per slot.

Times the two products the gateway computes per 256-slot group, at rank 16 / q = 2^56 (K32):

* encrypt pad ``A[:n] @ S`` (8 int8 planes of the public matrix), one secret per group as
  ``lwe.encrypt_masks`` does today, plus the NTT with the public matrix transformed once
  (``TransformedOperand``, several groups per pass);
* decrypt correction ``H @ [S_1 | ... | S_G]`` (6 hint planes, bytes 1..6) for ``--groups`` groups per
  provider call, as ``HEMaskBackend._decrypt`` does.

Per-row costs are measured on ``--rows`` rows (every row costs the same) and extrapolated to the
model: ms/slot = cost per row and group * (sum of n, resp. m, over all linears) / 256.

    python benchmarks/hemask_ntt_bench.py --preset qwen2.5-7b --groups 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable

import numpy as np
import torch

from pgw.hemask import DEGREE, HEParams, fastmm, ntt, public_matrix, secret_matrix

PRESETS = {  # (layers, {kind: (m, n)})
    "qwen2.5-0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
}


def best(fn: Callable[[], object], reps: int) -> float:
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen2.5-7b", choices=sorted(PRESETS))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--groups", type=int, default=4, help="256-slot groups per provider call (decrypt)")
    ap.add_argument("--rows", type=int, default=1024, help="rows timed for the NTT products")
    ap.add_argument("--dense-rows", type=int, default=256, help="rows timed for the dense products")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--engine", default=None, help="dense engine (cpu_int8, cpu_f64; default: automatic)")
    ap.add_argument("--threads", type=int, default=0, help="threads for both paths (0: torch / pgw_ntt default)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.engine:
        os.environ["PGW_HEMASK_ENGINE"] = args.engine
    if args.threads:
        torch.set_num_threads(args.threads)
        os.environ["PGW_NTT_THREADS"] = str(args.threads)
    if not ntt.available():
        raise SystemExit("pgw_ntt is not built: python rust/tools/build_ntt.py")
    layers, kinds = PRESETS[args.preset]
    sum_n = layers * sum(n for _m, n in kinds.values())
    sum_m = layers * sum(m for m, _n in kinds.values())
    params = HEParams(k=32, rank=args.rank)
    g = args.groups
    rng = np.random.default_rng(0)
    rows = max(args.rows, args.dense_rows)
    a = public_matrix(bytes(32), rows, params)
    ap8 = fastmm.to_planes(a)
    hint = rng.integers(-128, 128, size=(rows, 6, params.dim), dtype=np.int8)
    secrets = [rng.integers(-1, 2, size=(args.rank, DEGREE)) for _ in range(max(g, 4))]
    smats = [secret_matrix(s) for s in secrets]
    s1, sg = smats[0], np.concatenate(smats[:g], axis=1)
    s4 = np.concatenate(smats[:4], axis=1)
    rn, rd = args.rows, args.dense_rows
    res: dict[str, object] = {
        "preset": args.preset,
        "rank": args.rank,
        "groups": g,
        "sum_n": sum_n,
        "sum_m": sum_m,
        "dense_engine": fastmm.engine(),
        "ntt": ntt.info(),
        "threads": torch.get_num_threads(),
    }
    # per row and group, seconds
    t: dict[str, float] = {}
    t["dense_encrypt"] = best(lambda: fastmm.big_planes_mm_small(ap8[:rd], s1), args.reps) / rd
    t["ntt_encrypt_planes_g1"] = best(lambda: ntt.big_planes_mm_small(ap8[:rn], s1), args.reps) / rn
    t0 = time.perf_counter()
    pub = ntt.TransformedOperand.from_planes(ap8[:rn])
    res["transform_a_s_per_row"] = (time.perf_counter() - t0) / rn
    res["transform_a_bytes_per_row"] = pub.nbytes / pub.rows
    t["ntt_encrypt_pre_g1"] = best(lambda: pub.mm(s1), args.reps) / rn
    t["ntt_encrypt_pre_g4"] = best(lambda: pub.mm(s4), args.reps) / (rn * 4)
    t["dense_decrypt"] = best(lambda: fastmm.big_planes_mm_small(hint[:rd], sg, first_byte=1), args.reps) / (rd * g)
    t["ntt_decrypt_planes"] = best(lambda: ntt.big_planes_mm_small(hint[:rn], sg, first_byte=1), args.reps) / (rn * g)
    pub_h = ntt.TransformedOperand.from_planes(hint[:rn], first_byte=1)
    t["ntt_decrypt_pre"] = best(lambda: pub_h.mm(sg), args.reps) / (rn * g)
    # bit-identity spot check on the timed shapes
    assert np.array_equal(ntt.big_planes_mm_small(ap8[:rd], s1), fastmm.big_planes_mm_small(ap8[:rd], s1))
    assert np.array_equal(
        ntt.big_planes_mm_small(hint[:rd], sg, first_byte=1) & np.uint64(2**56 - 1),
        fastmm.big_planes_mm_small(hint[:rd], sg, first_byte=1) & np.uint64(2**56 - 1),
    )
    per_slot = {k: 1e3 * v * (sum_n if "encrypt" in k else sum_m) / DEGREE for k, v in t.items()}
    res["us_per_row_group"] = {k: round(v * 1e6, 3) for k, v in t.items()}
    res["ms_per_slot"] = {k: round(v, 2) for k, v in per_slot.items()}
    total = {
        "dense": per_slot["dense_encrypt"] + per_slot["dense_decrypt"],
        "ntt_dropin": per_slot["ntt_encrypt_planes_g1"] + per_slot["ntt_decrypt_planes"],
        "ntt_A_transformed": per_slot["ntt_encrypt_pre_g1"] + per_slot["ntt_decrypt_planes"],
        "ntt_A_transformed_g4": per_slot["ntt_encrypt_pre_g4"] + per_slot["ntt_decrypt_planes"],
        "ntt_A_and_hint_transformed": per_slot["ntt_encrypt_pre_g4"] + per_slot["ntt_decrypt_pre"],
    }
    res["total_ms_per_slot"] = {k: round(v, 2) for k, v in total.items()}
    res["slots_per_hour"] = {k: round(3.6e6 / v) for k, v in total.items()}
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
