"""Gateway encryption of mask groups: today's dense path vs the public matrix in NTT form.

Times full encryptions (secret and noise sampling, pad, combine), not just the pad product:

* ``dense``: ``lwe.encrypt_masks(r, None, params, a_planes=...)`` per 256-slot group, as
  ``HEMaskBackend(encrypt="dense")`` does (fused native AMX/VNNI path on ``cpu_int8``, the GPU on
  ``cuda``, float64 slices on ``cpu_f64``);
* ``ntt_pre_g{G}``: ``mlwe_ring.PretransformedEncryptor`` with ``G`` groups per pass over ``A``
  (``HEMaskBackend(encrypt="ntt_pre")``), bit-identical ciphertexts.

Per-row cost is measured on ``--rows`` rows of the public matrix (every row costs the same) and
extrapolated to the model: ms/slot = cost per row and group x sum of linear input widths / 256.

    python benchmarks/hemask_encrypt_bench.py --preset qwen2.5-7b --rank 16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable

import numpy as np
import torch

from pgw.hemask import DEGREE, HEParams, public_matrix
from pgw.hemask.fastmm import engine, to_planes
from pgw.hemask.lwe import a_plane_count, encrypt_masks
from pgw.hemask.mlwe_ring import PretransformedEncryptor, encrypt_cost_ops

PRESETS = {  # (layers, {kind: (m, n)})
    "qwen2.5-0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "llama-70b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)}),
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
    ap.add_argument("--preset", default="qwen2.5-0.5b", choices=sorted(PRESETS))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--rows", type=int, default=2048, help="public-matrix rows timed for ntt_pre")
    ap.add_argument("--dense-rows", type=int, default=512, help="rows timed for the dense path")
    ap.add_argument("--groups", default="1,4,8", help="groups per pass for ntt_pre")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--engine", default=None, help="dense engine override (cpu_int8, cpu_f64, cuda)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.engine:
        os.environ["PGW_HEMASK_ENGINE"] = args.engine
    if args.threads:
        torch.set_num_threads(args.threads)
        os.environ["PGW_NTT_THREADS"] = str(args.threads)
    layers, kinds = PRESETS[args.preset]
    sum_n = layers * sum(n for _m, n in kinds.values())
    params = HEParams(k=args.k, rank=args.rank)
    rows = max(args.rows, args.dense_rows)
    a = public_matrix(bytes(32), rows, params)
    rng = np.random.default_rng(0)
    gs = [int(x) for x in args.groups.split(",")]
    r = rng.integers(0, 1 << args.k, size=(DEGREE * max(gs), rows), dtype=np.int64)
    res: dict[str, object] = {"preset": args.preset, "rank": args.rank, "k": args.k, "sum_n": sum_n,
                              "dense_engine": engine(), "threads": torch.get_num_threads(),
                              "cpu": os.cpu_count()}
    per_row_group: dict[str, float] = {}

    rd = args.dense_rows
    planes = to_planes(a[:rd])[:, :a_plane_count(params)]
    if engine() == "cuda":  # pragma: no cover - GPU only
        def dense() -> None:
            encrypt_masks(r[:DEGREE, :rd], None, params, a_planes=planes)
            torch.cuda.synchronize()
    else:
        def dense() -> None:
            encrypt_masks(r[:DEGREE, :rd], None, params, a_planes=planes)
    per_row_group["dense"] = best(dense, args.reps) / rd

    t0 = time.perf_counter()
    enc = PretransformedEncryptor(a[:args.rows], params)
    res["transform_s"] = round(time.perf_counter() - t0, 3)
    res["transform_bytes_per_row"] = enc.nbytes / args.rows
    rn = args.rows
    for g in gs:
        enc.groups_per_pass = g
        rr = r[:DEGREE * g, :rn]
        per_row_group[f"ntt_pre_g{g}"] = best(lambda rr=rr: enc.encrypt(rr), args.reps) / (rn * g)

    res["us_per_row_group"] = {k: round(v * 1e6, 3) for k, v in per_row_group.items()}
    ms = {k: 1e3 * v * sum_n / DEGREE for k, v in per_row_group.items()}
    res["encrypt_ms_per_slot"] = {k: round(v, 3) for k, v in ms.items()}
    res["encrypt_slots_per_s"] = {k: round(1e3 / v, 1) for k, v in ms.items()}
    res["speedup_vs_dense"] = {k: round(ms["dense"] / v, 1) for k, v in ms.items()}
    res["ops_model_one_row_group"] = encrypt_cost_ops(1, args.rank)
    res["transformed_A_bytes_model"] = 12 * max(n for _m, n in kinds.values()) * params.dim
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
