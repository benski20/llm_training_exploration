"""The pks-v3 hint build's GEMM ``H = W8 A mod 2^56`` and its per-block tail, with ``A``'s byte planes built once.

``A = a_mlwe_rows(public_ring_rows(a_seed, n, linear))`` (``n x 4096``, entries ``< 2^56``) is the same for every row
block of a linear. :func:`pgw.hemask.lwe.matmul_mod64` re-derives its byte planes (and transposes the ``8 n 4096``
bytes) for every 1024-row block, and multiplies 8 planes although the top byte is zero. Here the 7 planes are built
once per linear, natively and straight from the ``n/16`` ring rows (``pgw_pks.a_planes``; ``A`` is never
materialised as uint64), in column blocks of :data:`COL_BLOCK`. Each (row block, column block) is one int8 GEMM
(``torch._int_mm``: AMX / AVX-512 VNNI through oneDNN) plus a native recombination (``pgw_pks.combine_planes``).

Exactness: plane ``d`` holds ``byte_d(A) - 128`` (int8), and ``W8 byte_d(A) = W8 P_d + 128 rowsum(W8)``. Every int32
partial sum is at most ``127 * 128 * n + 128 * 127 * n < 2^31`` for ``n <= MAX_N``. The planes are recombined with
wrapping uint64 shifts mod ``2^64`` (a ring map onto ``2^56``). The result equals ``matmul_mod64(W8, A) & (2^56 - 1)``
bit for bit (``tests/hemask/test_hint_build.py``). The int8 path is used only where CPU int8 GEMMs are exact on
extreme operands (:func:`pgw.core.ringmm.cpu_int8_available`; oneDNN's AVX2 kernel saturates) and fast
(:func:`pgw.hemask.fastmm.engine` ``cpu_int8``); elsewhere the exact float64 path of ``matmul_mod64`` runs.

:func:`hint_rows` is the whole per-block tail of the build: ``H`` block -> stored hint rows
(``keyswitch(switch_hint(H), k)`` in the ``(rows/32, 2, 8192, 3)`` uint8 form), fused natively when ``pgw_pks`` has
``hint_keyswitch``.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError
from pgw.hemask import pks
from pgw.hemask.pks import NU, RANK

__all__ = ["HintGemm", "hint_rows", "threads"]

_M56 = np.uint64((1 << 56) - 1)
PLANES = 7  # A < 2^56: the top byte plane is zero
#: Largest inner dimension (exact int32 partial sums: ``2 * 127 * 128 * n < 2^31``).
MAX_N = 65536  # also the inner length the exactness probe of cpu_int8_available covers
#: Output columns per int8 GEMM (``7 * 512`` operand rows): AMX / VNNI kernels run up to 2x faster on such column
#: blocks than on all ``7 * 4096`` at once (measured, ``benchmarks/provision_hint_bench.py``).
COL_BLOCK = 512
#: Ring rows of ``A`` turned into planes per step (bounds the transient uint64 block: ``16 * 64 * 4096 * 8`` B).
_RING_STEP = 64


def threads(n: int | None = None) -> int:
    """Worker threads for the native kernels: ``n``, else ``PGW_PKS_THREADS``, else every CPU."""
    if n is not None:
        return max(1, int(n))
    raw = os.environ.get("PGW_PKS_THREADS", "").strip()
    return max(1, int(raw)) if raw.isdigit() else max(1, os.cpu_count() or 1)


def _native() -> Any:
    from pgw.hemask.pks_native import native

    nat = native()
    return nat if nat is not None and hasattr(nat, "combine_planes") and hasattr(nat, "hint_keyswitch") else None


def a_planes_ref(ring: np.ndarray, cb: int = NU) -> np.ndarray:
    """``(7 * 4096, n)`` int8 byte planes of ``A = a_mlwe_rows(ring)`` in column blocks of ``cb`` (row
    ``(col // cb) 7 cb + d cb + col % cb``: ``byte_d(A[:, col]) - 128``), in numpy (the reference of
    ``pgw_pks.a_planes``)."""
    n = ring.shape[0] * RANK
    planes = np.empty((PLANES * NU, n), dtype=np.int8)
    pv = np.empty((PLANES, NU, n), dtype=np.int8)
    for c0 in range(0, ring.shape[0], _RING_STEP):
        blk = pks.a_mlwe_rows(ring[c0:c0 + _RING_STEP])  # rows 16 c0 .. of A
        raw = np.ascontiguousarray(blk.T).view(np.uint8).reshape(NU, blk.shape[0], 8)  # (col, row, byte)
        for d in range(PLANES):
            pv[d, :, RANK * c0:RANK * c0 + blk.shape[0]] = (raw[:, :, d] ^ np.uint8(0x80)).view(np.int8)
    planes.reshape(NU // cb, PLANES, cb, n)[:] = pv.reshape(PLANES, NU // cb, cb, n).transpose(1, 0, 2, 3)
    return planes


class HintGemm:
    """``H = W8 A mod 2^56`` for one linear (``A`` from ``a_seed`` and the linear id), row block by row block."""

    def __init__(self, a_seed: bytes, n: int, linear: str, *, n_threads: int | None = None,
                 fast: bool | None = None):
        from pgw.core.ringmm import cpu_int8_available
        from pgw.hemask.fastmm import engine

        if n % RANK:
            raise ConfigError(f"{linear}: in_features {n} is not a multiple of {RANK}")
        self.n, self.linear, self.a_seed = int(n), linear, bytes(a_seed)
        self.threads = threads(n_threads)
        self.nat = _native()
        if fast is None:
            fast = engine() == "cpu_int8"
        # never without an exact CPU int8 GEMM (the probe covers extreme operands; e.g. oneDNN's AVX2 kernel
        # saturates), whatever was asked
        self.fast = bool(fast) and self.nat is not None and n <= MAX_N and cpu_int8_available()
        ring = pks.public_ring_rows(self.a_seed, self.n, linear)  # (n/16, 4096)
        if not self.fast:
            self._a: np.ndarray | None = pks.a_mlwe_rows(ring)
            self._planes: torch.Tensor | None = None
            return
        self._a = None
        # column block b: rows [7 cb b, 7 cb (b + 1)), row 7 cb b + d cb + j = byte_d(A[:, cb b + j]) - 128
        planes = (self.nat.a_planes(np.ascontiguousarray(ring), COL_BLOCK, self.threads)
                  if hasattr(self.nat, "a_planes") else a_planes_ref(ring, COL_BLOCK))
        self._planes = torch.from_numpy(planes)

    @property
    def nbytes(self) -> int:
        if self._planes is not None:
            return int(self._planes.numel())
        return int(self._a.nbytes) if self._a is not None else 0

    def __call__(self, w: np.ndarray) -> np.ndarray:
        """``H`` rows for the int8 weight rows ``w`` (``(rows, n)``): ``(rows, 4096)`` uint64 ``< 2^56``."""
        w = np.ascontiguousarray(w, dtype=np.int8)
        if w.ndim != 2 or w.shape[1] != self.n:
            raise ConfigError(f"{self.linear}: weight block must be (rows, {self.n})")
        if not self.fast:
            from pgw.hemask.lwe import matmul_mod64

            assert self._a is not None
            return matmul_mod64(w, self._a) & _M56
        assert self._planes is not None
        rows = w.shape[0]
        with warnings.catch_warnings():  # read-only weights (a checked download, a mapped pack): torch only reads
            warnings.simplefilter("ignore", UserWarning)
            wt = torch.from_numpy(w)
        off = (128 * w.astype(np.int64).sum(1)).astype(np.int32)
        out = np.empty((rows, NU), dtype=np.uint64)
        step = PLANES * COL_BLOCK
        for b in range(NU // COL_BLOCK):
            part = torch._int_mm(wt, self._planes[b * step:(b + 1) * step].t())  # (rows, 7 cb) int32
            out[:, b * COL_BLOCK:(b + 1) * COL_BLOCK] = self.nat.combine_planes(
                part.numpy().reshape(rows, PLANES, COL_BLOCK), off, self.threads)
        return out

    def close(self) -> None:
        self._planes = None
        self._a = None


def hint_rows(h: np.ndarray, k_hat: np.ndarray, *, n_threads: int | None = None) -> np.ndarray:
    """Stored hint rows ``(rows/32, 2, 8192, 3)`` uint8 of an ``H`` block (``(rows, 4096)`` uint64, entries taken mod
    ``2^56``): ``keyswitch(switch_hint(H), k)``, the switch fused into the native key switch when available."""
    nat = _native()
    if nat is not None:
        out: np.ndarray = nat.hint_keyswitch(np.ascontiguousarray(h, dtype=np.uint64),
                                             np.ascontiguousarray(k_hat, dtype=np.uint32), True, threads(n_threads))
        return out
    from pgw.hemask.pks_backend import switch_hint_fast
    from pgw.hemask.pks_native import keyswitch

    return keyswitch(switch_hint_fast(h), k_hat)
