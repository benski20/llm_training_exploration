#!/usr/bin/env python3
"""Gateway prepare step (quantize + mask + pack): Python path vs ``pgw_fast.quantize_mask_pack``.

    python benchmarks/native_prepare_bench.py [--ring K32A12C27] [--iters 200]

The Python path is what ``MaskedBackend._prepare`` runs per chunk on a K=32 ring:
``quantize_activations`` (capped) -> ``mask_pack_u32`` -> ``x_hat`` as int64.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from pgw.core.masking import mask_activations, mask_pack_u32
from pgw.core.packing import pack_ring
from pgw.core.ring import RingParams, quantize_activations


def _python(x: torch.Tensor, r: np.ndarray, ring: RingParams, wn: float | None) -> object:
    xq, sx = quantize_activations(x, ring, wn)
    if ring.k == 32:
        x32, payload = mask_pack_u32(xq.numpy(), r, ring)
        x_hat = torch.from_numpy(x32.astype(np.int64))
    else:
        x_hat = mask_activations(xq, torch.from_numpy(r), ring)
        payload = pack_ring(x_hat, ring)
    return payload, x_hat, sx


def _time(fn: object, iters: int) -> float:
    fn()  # type: ignore[operator]
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()  # type: ignore[operator]
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    import pgw_fast  # type: ignore[import-not-found,unused-ignore]

    ring = RingParams.from_name(args.ring)
    rng = np.random.default_rng(0)
    print(f"ring {ring.name}, torch threads {torch.get_num_threads()}")
    print(f"{'rows':>5} {'n':>6} {'python ms':>10} {'native ms':>10} {'speedup':>8}")
    for n in (3584, 18944):
        w8 = np.clip(np.rint(rng.normal(0, 40, size=(8, n))), -127, 127)
        wn = float(np.sqrt((w8 * w8).sum(axis=1)).max()) if ring.cap else None
        for rows in (1, 40):
            xn = rng.normal(0, 1, size=(rows, n)).astype(np.float32)
            xn[:, rng.choice(n, size=max(1, n // 500), replace=False)] *= 300  # outlier channels
            x = torch.from_numpy(xn)
            r = rng.integers(0, ring.modulus, size=(rows, n), dtype=np.int64)
            want = _python(x, r, ring, wn)
            got = pgw_fast.quantize_mask_pack(xn, r, ring.k, ring.a, ring.cap, float(wn or 0.0))
            assert got[0] == want[0] and np.array_equal(got[1], want[1].numpy())  # type: ignore[index,union-attr]
            iters = max(5, args.iters // rows)
            tp = _time(lambda x=x, r=r, wn=wn: _python(x, r, ring, wn), iters)
            tn = _time(
                lambda xn=xn, r=r, wn=wn: pgw_fast.quantize_mask_pack(
                    xn, r, ring.k, ring.a, ring.cap, float(wn or 0.0)
                ),
                iters,
            )
            print(f"{rows:>5} {n:>6} {tp:>10.3f} {tn:>10.3f} {tp / tn:>7.1f}x")


if __name__ == "__main__":
    main()
