#!/usr/bin/env python3
"""Mask fetch (``MaskReservation.take``) hot path: Python vs ``pgw_fast`` (store.rs).

    python benchmarks/native_store_bench.py [--reps 200]

* ``mask_rows``: ``SeedExpander(seed).ring_uniform_rows(cols, K, slot, rows, "r", lid)`` (the pad
  ``r``) vs ``pgw_fast.mask_rows`` and ``mask_rows_with_key`` (HKDF key cached by the caller).
* ``unpack_rows``: ``_read_wr``'s slice copy + ``unpack_ring`` vs ``pgw_fast.unpack_rows`` reading the
  rows straight out of the decrypted chunk ``bytearray``.

Prints the median time per call in ms.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable

from pgw.core.packing import unpack_ring
from pgw.core.prng import SeedExpander

try:
    import pgw_fast  # type: ignore[import-not-found,unused-ignore]
except ImportError:  # pragma: no cover
    pgw_fast = None

K = 32
OFF = 7  # first row read from the chunk


def _ms(fn: Callable[[], object], reps: int) -> float:
    for _ in range(min(10, reps)):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    if pgw_fast is None or not hasattr(pgw_fast, "mask_rows"):
        print("pgw_fast store functions not built: python rust/tools/build_fast.py")
        return 1
    seed = os.urandom(32)
    lid = "12.mlp.down"
    slot = 123_457

    print(f"mask_rows (K={K}, slot {slot}), ms/call (median of {args.reps})")
    print(f"{'cols':>6} {'rows':>4} {'python':>9} {'native':>9} {'with_key':>9} {'speedup':>8}")
    for cols in (3584, 18944):
        for rows in (1, 40):
            key = pgw_fast.mask_key(seed, ["r", lid])
            py = _ms(lambda c=cols, r=rows: SeedExpander(seed).ring_uniform_rows(c, K, slot, r, "r", lid), args.reps)
            nat = _ms(lambda c=cols, r=rows: pgw_fast.mask_rows(seed, ["r", lid], c, K, slot, r), args.reps)
            wk = _ms(lambda c=cols, r=rows, key=key: pgw_fast.mask_rows_with_key(key, c, K, slot, r), args.reps)
            print(f"{cols:>6} {rows:>4} {py:>9.4f} {nat:>9.4f} {wk:>9.4f} {py / nat:>7.1f}x")
    hk = _ms(lambda: pgw_fast.mask_key(seed, ["r", lid]), args.reps)
    print(f"mask_key alone (HKDF + call overhead): {hk:.4f} ms")

    print(f"\nunpack_rows (K={K}, from a 64-row chunk), ms/call (median of {args.reps})")
    print(f"{'m':>6} {'rows':>4} {'python':>9} {'native':>9} {'speedup':>8}")
    nb = K // 8
    chunk_rows = 64
    for m in (4608, 3584, 37888):
        buf = bytearray(os.urandom(chunk_rows * m * nb))
        row_bytes = m * nb
        for rows in (1, 40):

            def py_path(buf=buf, off=OFF, rows=rows, m=m, row_bytes=row_bytes) -> object:
                out = bytearray(rows * row_bytes)
                out[:] = buf[off * row_bytes : (off + rows) * row_bytes]
                return unpack_ring(out, (rows, m), K)

            py = _ms(py_path, args.reps)
            nat = _ms(lambda b=buf, r=rows, m=m: pgw_fast.unpack_rows(b, OFF, r, m, K), args.reps)
            print(f"{m:>6} {rows:>4} {py:>9.4f} {nat:>9.4f} {py / nat:>7.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
