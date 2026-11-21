"""Microbenchmark of one decode-sized exact local linear (``local_linears``): the native fused kernel
(``pgw_fast.local_linear_into``) against the numpy + two-digit ``torch._int_mm`` form it replaces.

Usage:
    PYTHONPATH=src python benchmarks/local_linear_bench.py [--shapes 896x1152,896x896,3584x4608,3584x3584]
        [--rows 1] [--reps 200] [--torch-threads 1]

Prints the median microseconds per call (calling-thread CPU and wall) for each shape (``n x m``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="896x1152,896x896,4864x896,3584x4608,3584x3584,18944x3584")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="native: 1 = calling thread, 0 = pool when large")
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    from pgw.core import native
    from pgw.core.ring import DEFAULT_RING
    from pgw.core.ringmm import int8_digit_gemm

    ring = DEFAULT_RING
    rng = np.random.default_rng(0)
    out: dict[str, dict[str, float]] = {}
    for shp in args.shapes.split(","):
        n, m = (int(v) for v in shp.split("x"))
        w8 = rng.integers(-128, 128, size=(m, n), dtype=np.int8)
        w8t = np.ascontiguousarray(w8.T)
        w8t_t = torch.from_numpy(w8t)
        wnorm = float(np.sqrt((w8.astype(np.float64) ** 2).sum(axis=1)).max())
        scale = rng.random(m, dtype=np.float32) * np.float32(1e-3)
        x = rng.standard_normal((args.rows, n)).astype(np.float32)
        bound = int(ring.output_bound(n))
        res = np.empty((args.rows, m), dtype=np.float32)
        sx = np.empty(args.rows, dtype=np.float64)

        def nat() -> None:
            native.fast.local_linear_into(x, w8t, scale, None, ring.k, ring.a, ring.cap, wnorm, bound, sx, res,
                                          args.threads)

        zeros = np.zeros((args.rows, n), dtype=np.int64)

        def old() -> None:  # MaskedBackend._local_fast before L2 (minus the self-check bookkeeping)
            payload = np.empty(args.rows * n * 4, dtype=np.uint8)
            s = np.empty(args.rows, dtype=np.float64)
            native.fast.trip_prepare_into(x, zeros, ring.k, ring.a, ring.cap, wnorm, payload, s)
            xq = payload.view("<u4").astype(np.int64).reshape(args.rows, n)
            xq -= (xq >> 31) << 32
            lo = ((xq + 128) & 255) - 128
            hi = (xq - lo) >> 8
            d8 = torch.from_numpy(np.concatenate([lo, hi]).astype(np.int8))
            y = int8_digit_gemm(d8, w8t_t.t()).numpy()
            yq = y[: args.rows] + (y[args.rows :] << 8)
            _ = int(np.abs(yq).max()) > bound
            _ = (yq.astype(np.float64) * s[:, None]) * scale.astype(np.float64)[None, :]

        row: dict[str, float] = {}
        for name, fn in (("native", nat), ("numpy_int_mm", old)):
            for _ in range(5):
                fn()
            cpu, wall = [], []
            for _ in range(args.reps):
                c, t = time.thread_time(), time.perf_counter()
                fn()
                cpu.append(time.thread_time() - c)
                wall.append(time.perf_counter() - t)
            row[f"{name}_us_cpu"] = round(1e6 * statistics.median(cpu), 1)
            row[f"{name}_us_wall"] = round(1e6 * statistics.median(wall), 1)
        out[shp] = row
    print(json.dumps({"rows": args.rows, "torch_threads": args.torch_threads, "shapes": out}, indent=1))


if __name__ == "__main__":
    main()
