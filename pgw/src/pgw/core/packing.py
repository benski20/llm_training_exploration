"""Compact byte encoding of ring tensors (wire payloads and mask store records).

A ring tensor with ``K``-bit elements is serialised as ``K/8`` little-endian bytes per element in
row-major (C) order, with no header; shape and ``K`` travel in the enclosing message. For K=40 this
is 5 bytes/element instead of 8.
"""

from __future__ import annotations

import numpy as np
import torch

from pgw.core.errors import ProtocolError, ShapeError
from pgw.core.ring import RingParams

__all__ = ["pack_ring", "packed_size", "unpack_ring", "unpack_ring_u64"]


def _k(params: RingParams | int) -> int:
    k = params.k if isinstance(params, RingParams) else int(params)
    if k % 8 or not 8 <= k <= 64:
        raise ShapeError(f"bad ring width {k}")
    return k


def packed_size(shape: tuple[int, ...], params: RingParams | int) -> int:
    """Byte length of a packed tensor of ``shape``."""
    n = 1
    for d in shape:  # plain ints: np.prod costs microseconds per wire message
        n *= int(d)
    return n * (_k(params) // 8)


def pack_ring(t: torch.Tensor | np.ndarray, params: RingParams | int) -> bytes:
    """Encode int64 ring values (reduced mod ``2^K`` first) to ``K/8`` bytes each."""
    k = _k(params)
    nb = k // 8
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.dtype != np.int64 and a.dtype != np.uint64:
        raise ShapeError(f"ring tensors must be int64, got {a.dtype}")
    if nb in (2, 4):  # astype to a narrower unsigned type wraps mod 2^K: one pass, no byte shuffling
        return np.ascontiguousarray(a).astype("<u4" if nb == 4 else "<u2").tobytes()
    u = np.ascontiguousarray(a.astype(np.uint64, copy=False) & np.uint64((1 << k) - 1))
    by = u.astype("<u8", copy=False).view(np.uint8).reshape(-1, 8)
    return bytes(by[:, :nb].tobytes() if nb < 8 else by.tobytes())


def unpack_ring(data: bytes | bytearray | memoryview, shape: tuple[int, ...], params: RingParams | int) -> torch.Tensor:
    """Decode bytes from :func:`pack_ring` into an int64 tensor of ``shape``."""
    k = _k(params)
    nb = k // 8
    expected = packed_size(shape, k)
    if len(data) != expected:
        raise ProtocolError(f"ring payload has {len(data)} bytes, expected {expected} for shape {shape} K={k}")
    if nb in (2, 4, 8):
        vals = np.frombuffer(data, dtype={2: "<u2", 4: "<u4", 8: "<u8"}[nb]).astype(np.int64).reshape(shape)
        return torch.from_numpy(vals)
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, nb)
    full = np.zeros((raw.shape[0], 8), dtype=np.uint8)
    full[:, :nb] = raw
    vals = full.view("<u8").reshape(shape).astype(np.int64)
    return torch.from_numpy(vals)


def unpack_ring_u64(
    data: bytes | bytearray | memoryview, shape: tuple[int, ...], params: RingParams | int
) -> np.ndarray:
    """Like :func:`unpack_ring` but returns a uint64 ``np.ndarray`` (no torch tensor, one copy less).

    Used on the gateway's per-call hot path, where results stay in numpy until dequantization.
    """
    k = _k(params)
    nb = k // 8
    expected = packed_size(shape, k)
    if len(data) != expected:
        raise ProtocolError(f"ring payload has {len(data)} bytes, expected {expected} for shape {shape} K={k}")
    raw = np.frombuffer(data, dtype=np.uint8)
    if nb == 8:
        return raw.view("<u8").reshape(shape).copy()
    if nb in (2, 4):
        return raw.view("<u4" if nb == 4 else "<u2").astype(np.uint64).reshape(shape)
    full = np.zeros((raw.size // nb, 8), dtype=np.uint8)
    full[:, :nb] = raw.reshape(-1, nb)
    return full.view("<u8").reshape(shape)
