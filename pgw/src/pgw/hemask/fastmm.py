"""Exact products mod ``2^64`` of a 64-bit matrix and a small int8 matrix, on int8 GEMM units.

Every product in the MLWE mask protocol has one operand of arbitrary 64-bit words (the public
matrix ``A``, the hint ``H``, a ciphertext ``b``) and one small operand (a ternary secret matrix
``S``, or int8 weights ``W8``). The 64-bit operand is split into its 8 bytes ("planes"), offset by
-128 to signed int8. One int8 GEMM (``torch._int_mm``: AMX/VNNI on CPU) then multiplies all planes
at once, and the planes are recombined with wrapping uint64 shifts::

    big = sum_d byte_d << 8d,   byte_d @ S = (byte_d - 128) @ S + 128 * colsum(S)

Partial sums stay in int32 (``|.| <= 128 * k`` for a ternary ``S``, ``127 * 128 * n`` for weights), so
everything is exact. On CPUs without fast int8 GEMMs (see :func:`pgw.core.ringmm.cpu_int8_fast`) the
callers fall back to the float64 bit-slice path.

Planes layout: ``(rows, 8, cols)`` int8, plane ``d`` holding ``byte_d ^ 0x80`` (== ``byte_d - 128``). It
takes exactly as many bytes as the uint64 matrix, so long-lived operands (hint, public matrix) are
stored in this form.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

__all__ = ["big_planes_mm_small", "int8_gemm_available", "planes_to_u64", "small_mm_big", "to_planes"]



def int8_gemm_available() -> bool:
    """True when exact and fast CPU int8 GEMMs exist (cached probe)."""
    from pgw.core.ringmm import cpu_int8_fast

    return cpu_int8_fast()


def to_planes(x: np.ndarray) -> np.ndarray:
    """``(rows, cols)`` uint64 -> ``(rows, 8, cols)`` int8 planes (byte ``d`` of each word, minus 128)."""
    x = np.ascontiguousarray(x, dtype="<u8")
    rows, cols = x.shape
    b = x.view(np.uint8).reshape(rows, cols, 8).transpose(0, 2, 1)
    return np.ascontiguousarray(b ^ np.uint8(0x80)).view(np.int8)


def planes_to_u64(p: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_planes`; with fewer than 8 planes the missing low bytes are zero."""
    rows, npl, cols = p.shape
    full = np.zeros((rows, cols, 8), dtype=np.uint8)
    full[:, :, 8 - npl:] = (p.view(np.uint8) ^ np.uint8(0x80)).transpose(0, 2, 1)
    return full.reshape(rows, cols * 8).view("<u8").astype(np.uint64)


def _combine(part: torch.Tensor, offset: torch.Tensor, first_byte: int = 0, small: bool = False) -> np.ndarray:
    """``sum_i (part[:, i] + offset) << 8 (first_byte + i)`` mod 2^64; ``part`` is ``(rows, planes, cols)`` int32.

    Bytes 0-3 and 4-7 are combined separately in int64 (every term is below ``2^30 * 2^24``, so
    nothing overflows); only the final ``hi << 32`` wraps, in numpy uint64 where wrap-around is well
    defined. With ``small`` (``|part| < 2^22``, true for ternary secrets), neighbouring planes are first
    paired in int32, which halves the int64 traffic.
    """
    rows, npl, cols = part.shape
    x = part + offset  # int32
    halves: list[torch.Tensor | None] = []
    for lo_b, hi_b in ((0, 4), (4, 8)):
        idx = [i for i in range(npl) if lo_b <= first_byte + i < hi_b]
        if not idx:
            halves.append(None)
            continue
        acc: torch.Tensor | None = None
        i = 0
        while i < len(idx):
            p = idx[i]
            shift = 8 * (first_byte + p - lo_b)
            if small and i + 1 < len(idx) and idx[i + 1] == p + 1:
                term = (x[:, p] + x[:, p + 1] * 256).to(torch.int64)  # |.| < 2^30: exact in int32
                i += 2
            else:
                term = x[:, p].to(torch.int64)
                i += 1
            term = term * (1 << shift) if shift else term
            acc = term if acc is None else acc + term
        halves.append(acc)
    lo_t, hi_t = halves
    out = np.zeros((rows, cols), dtype=np.uint64)
    if lo_t is not None:
        out += lo_t.numpy().view(np.uint64)
    if hi_t is not None:
        out += hi_t.numpy().view(np.uint64) << np.uint64(32)
    return out


def big_planes_mm_small(planes: np.ndarray, small: np.ndarray, *, block_rows: int = 2048,
                        first_byte: int | None = None) -> np.ndarray:
    """``U @ S mod 2^64`` with ``U`` given as byte planes ``(rows, P, k)`` and ``S`` int8 ``(k, c)``.

    ``P = 8`` is exact. With fewer planes, they hold bytes ``first_byte .. 7`` (default ``8 - P``) and
    the dropped low bytes are ignored: the result is then off by at most ``255 * 2^(8 (first_byte-1)) *
    sum|S|`` (used for decryption-only hints, whose error is not multiplied by the weights).
    """
    rows, npl, k = planes.shape
    if not 1 <= npl <= 8 or small.shape[0] != k:
        raise ValueError(f"shape mismatch: planes {planes.shape}, small {small.shape}")
    first = 8 - npl if first_byte is None else first_byte
    c = small.shape[1]
    st = torch.from_numpy(np.ascontiguousarray(np.asarray(small, dtype=np.int8).T))  # (c, k): column-major operand
    s_small = bool(np.abs(np.asarray(small, dtype=np.int64)).sum(0).max() <= (1 << 14))  # |partial| < 2^22
    offset = torch.from_numpy((128 * np.asarray(small, dtype=np.int64).sum(0)).astype(np.int32)).view(1, 1, -1)
    out = np.empty((rows, c), dtype=np.uint64)
    for r0 in range(0, rows, block_rows):
        blk = np.ascontiguousarray(planes[r0:r0 + block_rows])
        nb = blk.shape[0]
        with warnings.catch_warnings():  # read-only memory-mapped hints: torch only reads them
            warnings.simplefilter("ignore", UserWarning)
            lhs = torch.from_numpy(blk.reshape(nb * npl, k))
        part = torch._int_mm(lhs, st.t()).view(nb, npl, c)
        out[r0:r0 + nb] = _combine(part, offset, first, s_small)
    return out


def small_mm_big(small: np.ndarray, big: np.ndarray, *, block_cols: int = 1024) -> np.ndarray:
    """``W @ X mod 2^64`` for int8 ``W`` ``(m, n)`` and uint64 ``X`` ``(n, c)``."""
    m, n = small.shape
    if big.shape[0] != n:
        raise ValueError(f"shape mismatch: small {small.shape}, big {big.shape}")
    c = big.shape[1]
    wt = torch.from_numpy(np.ascontiguousarray(np.asarray(small, dtype=np.int8)))
    offset = torch.from_numpy((128 * np.asarray(small, dtype=np.int64).sum(1)).astype(np.int32)).view(-1, 1, 1)
    out = np.empty((m, c), dtype=np.uint64)
    for c0 in range(0, c, block_cols):
        xb = np.ascontiguousarray(big[:, c0:c0 + block_cols], dtype="<u8")
        cb = xb.shape[1]
        # planes of X^T stacked along the output axis: (8*cb, n), used as the column-major operand
        raw = np.ascontiguousarray(xb.T).view(np.uint8).reshape(cb, n, 8).transpose(2, 0, 1)
        pt = np.ascontiguousarray(raw ^ np.uint8(0x80)).view(np.int8).reshape(8 * cb, n)
        part = torch._int_mm(wt, torch.from_numpy(pt).t()).view(m, 8, cb)
        out[:, c0:c0 + cb] = _combine(part, offset)
    return out
