"""fused2 pair-mask segments in the encrypted store: the p61 check words and their normative derivation.

Format: :mod:`pgw.maskstore.format` (``schedule = "fused2"``, ``check_field = "p61"``). This module holds the
pieces the store, the factory and the gateway share without importing the engine:

* :func:`p61_check_vectors`: the secret check vectors ``u`` of a segment and linear, int64 ``(checks, m)`` in
  ``[0, p)``, ``p = 2^61 - 1``: ``u64_le(stream("u-p61", linear_id)[8i : 8i + 8]) mod p`` (the derivation
  :class:`pgw.engine.fused2.Fused2MaskSource` uses with its own seed);
* :func:`pack_p61` / :func:`unpack_p61`: ``uW`` words as 8-byte little-endian integers below ``p`` (values up to
  ``2^61`` do not fit 7-byte K56 ring elements), range-checked both ways;
* :func:`fused2_soundness_bits`: per-call soundness of the p61 check with ``checks`` vectors (each stage), valid
  when ``2^(C+1) < p`` (every genuine output is range-checked to ``|y| <= 2^C``).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pgw.core.errors import ConfigError, MaskStoreError
from pgw.core.prng import SeedExpander
from pgw.core.ring import RingParams
from pgw.maskstore.format import FUSED2_SCHEDULE, P61, P61_WORD_BYTES

__all__ = [
    "FUSED2_SCHEDULE", "P61", "P61_TAG", "fused2_soundness_bits", "p61_check_vectors", "pack_p61", "unpack_p61",
]

#: Third element of a check token naming p61 vectors (``(segment_id, linear, "p61")``).
P61_TAG = "p61"


def p61_check_vectors(seed: bytes, checks: int, out_features: int, linear_key: str) -> torch.Tensor:
    """The secret p61 check vectors of a segment and linear: int64 ``(checks, m)`` in ``[0, 2^61 - 1)``."""
    if checks < 1:
        raise ConfigError("p61 check vectors need checks >= 1")
    raw = SeedExpander(seed).ring_uniform((checks, out_features), 64, "u-p61", linear_key)
    u = raw.numpy().view(np.uint64)
    out = torch.from_numpy((u % np.uint64(P61)).astype(np.int64))
    u.fill(0)
    return out


def pack_p61(uw: torch.Tensor | np.ndarray, shape: tuple[int, int], what: str = "uW") -> bytes:
    """``uW`` as 8-byte little-endian words; refuses a wrong shape or a value outside ``[0, p)``."""
    a = uw.detach().cpu().numpy() if isinstance(uw, torch.Tensor) else np.asarray(uw)
    if a.dtype != np.int64:
        raise MaskStoreError(f"{what} must be int64 words in [0, 2^61 - 1)")
    if tuple(a.shape) != tuple(shape):
        raise MaskStoreError(f"{what} has shape {tuple(a.shape)}, expected {tuple(shape)}")
    if a.size and (int(a.min()) < 0 or int(a.max()) >= P61):
        raise MaskStoreError(f"{what} has values outside F_p, p = 2^61 - 1")
    return np.ascontiguousarray(a, dtype="<i8").tobytes()


def unpack_p61(buf: bytes | bytearray | memoryview, shape: tuple[int, int], what: str = "uW") -> torch.Tensor:
    """Inverse of :func:`pack_p61` (int64 tensor, a copy); refuses words outside ``[0, p)``."""
    n = int(shape[0]) * int(shape[1])
    if len(buf) != n * P61_WORD_BYTES:
        raise MaskStoreError(f"{what}: {len(buf)} bytes, expected {n * P61_WORD_BYTES}")
    a = np.frombuffer(buf, dtype="<i8").astype(np.int64).reshape(shape)
    if a.size and (int(a.min()) < 0 or int(a.max()) >= P61):
        a.fill(0)
        raise MaskStoreError(f"{what} carries check words outside F_p")
    return torch.from_numpy(a)


def fused2_soundness_bits(ring: RingParams, checks: int) -> float:
    """``-log2`` of the false-accept probability of one stage's p61 check with ``checks`` vectors (0 if unsound).

    ``u`` is ``u64 mod p`` (point probability at most ``ceil(2^64 / p) / 2^64 <= (1 + 2^-3) / p``); an accepted
    wrong output differs from the genuine one by at most ``2^(C+1) < p`` per entry, so it is non-zero mod ``p``.
    """
    if checks < 1 or not ring.cap or ring.cap + 1 >= 61:
        return 0.0
    return checks * (math.log2(P61) - math.log2(1 + 2.0**-3))
