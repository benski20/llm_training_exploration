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

__all__ = ["pack_ring", "packed_size", "unpack_ring"]


def _k(params: RingParams | int) -> int:
    k = params.k if isinstance(params, RingParams) else int(params)
    if k % 8 or not 8 <= k <= 64:
        raise ShapeError(f"bad ring width {k}")
    return k


def packed_size(shape: tuple[int, ...], params: RingParams | int) -> int:
    """Byte length of a packed tensor of ``shape``."""
    return int(np.prod(shape, dtype=np.int64)) * (_k(params) // 8)


def pack_ring(t: torch.Tensor | np.ndarray, params: RingParams | int) -> bytes:
    """Encode int64 ring values (reduced mod ``2^K`` first) to ``K/8`` bytes each."""
    k = _k(params)
    nb = k // 8
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.dtype != np.int64 and a.dtype != np.uint64:
        raise ShapeError(f"ring tensors must be int64, got {a.dtype}")
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
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, nb)
    full = np.zeros((raw.shape[0], 8), dtype=np.uint8)
    full[:, :nb] = raw
    vals = full.view("<u8").reshape(shape).astype(np.int64)
    return torch.from_numpy(vals)
