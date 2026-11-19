#!/usr/bin/env python3
"""Per-chunk finish of a masked linear call: Python path vs ``pgw_fast.finish_chunk``.

    PYTHONPATH=src:. python benchmarks/native_finish_bench.py [--reps 20] [--checks 7]

Times the loop body of ``MaskedBackend._finish`` for one chunk at the Qwen2.5-7B linear shapes (ring
K32A12C27, ``c`` Freivalds check vectors, with bias) for decode (rows=1) and prefill (rows=40), and
prints the median ms per call. Inputs are random; the range bound is set so that every call takes
the full success path (the Freivalds verdict does not change the work done).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from pgw.core.freivalds import freivalds_check_u64
from pgw.core.masking import unmask_outputs_u32
from pgw.core.ring import RingParams, dequantize_outputs

SHAPES = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}


def python_finish(
    y_hat: bytes,
    wr: np.ndarray,
    x_hat: np.ndarray,
    u: np.ndarray,
    uw: np.ndarray,
    sx: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    ring: RingParams,
) -> tuple[torch.Tensor, bool]:
    """The K=32 Python path of ``MaskedBackend._finish`` for one chunk (range check without a bound)."""
    rows, m = wr.shape
    y = np.frombuffer(y_hat, dtype="<u4").reshape(rows, m).astype(np.uint64)
    if not u.any():
        raise RuntimeError("u is zero")
    ok = freivalds_check_u64(x_hat.view(np.uint64), y, u, uw, ring)
    yq = unmask_outputs_u32(y_hat, wr, ring, None)
    _ = int(yq.max()), int(yq.min())  # the range check's work
    out = dequantize_outputs(
        torch.from_numpy(yq), torch.from_numpy(sx), torch.from_numpy(scale), torch.from_numpy(bias), torch.float32
    )
    return out, bool(ok.all())


def timeit(fn: Callable[[], Any], reps: int) -> float:
    """Median wall time in ms (after one warm-up call)."""
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--checks", type=int, default=7)
    args = ap.parse_args()
    import pgw_fast  # type: ignore[import-not-found,unused-ignore]

    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(0)
    c = args.checks
    print(
        f"ring {ring.name}, c={c}, torch threads {torch.get_num_threads()}, cpus {os.cpu_count()}, "
        f"median of {args.reps}"
    )
    print(
        f"{'linear':8} {'m x n':>12} {'rows':>4} {'python ms':>10} {'native ms':>10} {'native 1T':>10} {'speedup':>8}"
    )
    for rows in (1, 40):
        for name, (m, n) in SHAPES.items():
            y_hat = rng.integers(0, 1 << 32, size=rows * m, dtype=np.uint64).astype("<u4").tobytes()
            wr = rng.integers(0, 1 << 32, size=(rows, m), dtype=np.int64)
            x_hat = rng.integers(0, 1 << 32, size=(rows, n), dtype=np.int64)
            u = rng.integers(0, 1 << 32, size=(c, m), dtype=np.uint64)
            uw = rng.integers(0, 1 << 32, size=(c, n), dtype=np.uint64)
            sx = rng.uniform(1e-4, 1e-2, size=rows)
            scale = rng.uniform(-1e-3, 1e-3, size=m).astype(np.float32)
            bias = rng.normal(size=m).astype(np.float32)
            bound = 1 << 40  # everything in range: full success path
            a = (y_hat, wr, x_hat, u, uw, sx, scale, bias)
            tp = timeit(lambda: python_finish(*a, ring), args.reps)  # noqa: B023
            tn = timeit(lambda: pgw_fast.finish_chunk(*a, 32, bound), args.reps)  # noqa: B023
            t1 = timeit(lambda: pgw_fast.finish_chunk(*a, 32, bound, 1), args.reps)  # noqa: B023
            print(f"{name:8} {f'{m}x{n}':>12} {rows:>4} {tp:10.3f} {tn:10.3f} {t1:10.3f} {tp / tn:7.1f}x")


if __name__ == "__main__":
    main()
