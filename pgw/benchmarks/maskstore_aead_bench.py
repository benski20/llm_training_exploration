"""AES-256-GCM cost of the mask store per slot: the factory's segment writer and the gateway's open and retire.

The store-side rows of ``benchmarks/pks_slot_breakdown.py`` (``seal``, ``open``, ``retire``) on their own, with
the same production calls, at the real shapes of each model (one layer of each linear kind plus the LM head,
times the layer count; 1024-slot segments; ~1 MiB chunks), without the HE parts. Random uint32 ``W r``.

  * ``seal``      range check + pack + AES-256-GCM + ``pwrite`` (``pgw_fast.seal_ring_chunks``) + fsync
  * ``seal_nofsync``  the same without the fsync
  * ``open``      ``pgw_fast.chunk_open_into`` (pread + AES-256-GCM open) + ``unpack_rows_u32``
  * ``open_aead`` ``chunk_open_into`` alone
  * ``retire``    ``shred_file`` (crypto-erase + zero-fill + fsync + unlink; the default)
  * ``retire_nozero``  ``shred_file(zero_fill=False)`` (``PGW_MASKSTORE_SHRED_ZERO_FILL=false``)

Also the AEAD's raw rate (``--rate``): GB/s of ``seal_chunks`` / ``chunk_open_into`` on 1 MiB chunks, one thread.
CPU ms per slot (``time.process_time``: all threads), best of ``--reps``.

    PYTHONPATH=src:. python benchmarks/maskstore_aead_bench.py --model 7b --rate
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks.pks_slot_breakdown import SHAPES


def _best(fn: Any, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        c0 = time.process_time()
        fn()
        best = min(best, time.process_time() - c0)
    return best


def rate(tmpdir: str, mib: int = 256) -> dict[str, float]:
    """GB/s of the native seal and open on 1 MiB chunks, one thread (page-cache writes, no fsync)."""
    from pgw.core import native
    from pgw.maskstore.format import chunk_aad, chunk_nonce

    n = 1 << 20
    data = np.random.default_rng(1).integers(0, 256, n * 8, dtype=np.uint8)
    dek, prefix, digest = bytearray(os.urandom(32)), os.urandom(4), os.urandom(32)
    path = Path(tmpdir) / "rate.bin"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        jobs = [(i, i * (n + 16), i * n, n) for i in range(8)]
        os.ftruncate(fd, 8 * (n + 16))
        reps = max(1, mib // 8)

        def seal() -> None:
            for _ in range(reps):
                native.fast.seal_chunks(fd, dek, prefix, digest, data, jobs, 1)

        seal()
        t = min(_timed(seal) for _ in range(3))
        out = bytearray(n)
        key = bytes(dek)

        def open_() -> None:
            for _ in range(reps):
                for i, o, _, _ in jobs:
                    native.fast.chunk_open_into(key, chunk_nonce(prefix, i), chunk_aad(digest, i), fd, o, n + 16, out)

        open_()
        t2 = min(_timed(open_) for _ in range(3))
    finally:
        os.close(fd)
        path.unlink()
    gb = reps * 8 * n / 1e9
    return {"seal_GBps": round(gb / t, 2), "open_GBps": round(gb / t2, 2)}


def _timed(fn: Any) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def measure_kind(m: int, slots: int, reps: int, tmpdir: str) -> dict[str, float]:
    from pgw.core import native
    from pgw.maskstore.format import chunk_aad, chunk_nonce
    from pgw.maskstore.segment_io import shred_file

    wr = np.random.default_rng(m).integers(0, 1 << 32, (slots, m), dtype=np.uint32)
    nb = 4
    cs = max(1, (1 << 20) // (m * nb))
    path = Path(tmpdir) / f"seg-{m}.bin"
    dek = bytearray(os.urandom(32))
    prefix, digest = os.urandom(4), os.urandom(32)
    jobs = []
    off = 0
    for i, s0 in enumerate(range(0, slots, cs)):
        rows = min(cs, slots - s0)
        jobs.append((i, off, s0, rows))
        off += rows * m * nb + 16
    res: dict[str, float] = {}

    def seal(fsync: bool) -> None:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, off)
            assert native.fast.seal_ring_chunks(fd, dek, prefix, digest, wr, 32, jobs, 0)
            if fsync:
                os.fsync(fd)
        finally:
            os.close(fd)

    res["seal_nofsync"] = _best(lambda: seal(False), reps)
    res["seal"] = _best(lambda: seal(True), reps)

    def open_all(unpack: bool) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            for i, o, _, rows in jobs:
                buf = bytearray(rows * m * nb)
                native.fast.chunk_open_into(bytes(dek), chunk_nonce(prefix, i), chunk_aad(digest, i), fd, o,
                                            rows * m * nb + 16, buf)
                if unpack:
                    native.fast.unpack_rows_u32(buf, 0, rows, m, 32)
        finally:
            os.close(fd)

    res["open_aead"] = _best(lambda: open_all(False), reps)
    res["open"] = _best(lambda: open_all(True), reps)
    for name, zero in (("retire", True), ("retire_nozero", False)):
        best = float("inf")
        for _ in range(max(1, reps // 2)):
            p2 = path.with_suffix(".ret")
            p2.write_bytes(path.read_bytes())
            c0 = time.process_time()
            shred_file(p2, zero_fill=zero)
            best = min(best, time.process_time() - c0)
        res[name] = best
    path.unlink()
    return res


def measure(model: str, slots: int, reps: int, tmpdir: str) -> dict[str, Any]:
    layers, shapes = SHAPES[model]
    tot: dict[str, float] = {}
    for kind, (m, _n) in shapes.items():
        mult = 1 if kind == "lm_head" else layers
        for k, v in measure_kind(m, slots, reps, tmpdir).items():
            tot[k] = round(tot.get(k, 0.0) + v * 1e3 * mult / slots, 3)
    tot["open_plus_retire"] = round(tot["open"] + tot["retire"], 3)
    return {"model": model, "slots": slots, "ms_per_slot": tot}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="*", default=["0.5b"], choices=sorted(SHAPES))
    ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rate", action="store_true", help="also the raw seal / open GB/s")
    ap.add_argument("--tmpdir", default=None, help="a directory on the store's disk (fsync cost)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=a.tmpdir) as td:
        if a.rate:
            print(json.dumps({"rate": rate(td)}), flush=True)
        for model in a.model:
            print(json.dumps(measure(model, a.slots, a.reps, td)), flush=True)


if __name__ == "__main__":
    main()
