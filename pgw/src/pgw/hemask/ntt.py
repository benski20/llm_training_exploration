"""Exact NTT-based ring products for the customer side of the encrypted mask factory.

The customer computes two products by the small ternary secret ``S`` (``lwe.secret_matrix``: the
block-negacyclic ``dim x 256`` matrix of ``rank`` secret polynomials, groups concatenated along
columns):

* encrypt, the pad ``A[:n] @ S mod 2^64`` (``A`` the public matrix, ``n x dim`` 64-bit words, given as
  8 int8 byte planes or as uint64);
* decrypt, the correction ``H @ S mod 2^64`` (``H = W8 A`` the hint, ``m x dim``, stored as 6 byte planes
  holding bytes 1..6, i.e. ``first_byte=1``, with byte 0 treated as zero).

The dense path (:mod:`pgw.hemask.fastmm`, :func:`pgw.hemask.lwe.matmul_mod64`) multiplies by ``S``
as a matrix: ``dim * 256`` int8 MACs per row and byte plane. Here each row is a sum of ``rank``
negacyclic products in ``Z[X]/(X^256 + 1)``, computed exactly with NTTs over three ~2^29 primes and
CRT (native crate ``rust/pgw-ntt``, module ``pgw_ntt``), then reduced mod ``2^64``: the result is the
dense result **bit for bit** (``tests/hemask/test_ntt.py``). The wire and the provider are unchanged.

Exactness: an output coefficient is ``sum +- a * s`` with ``|a| < 2^b`` (``b = 64`` for ``A``, ``56`` for the
hint) and ``sum |s| = L1(S)`` (``<= 256 rank`` for a ternary secret), so ``|Z| <= (2^b - 1) L1``. CRT with a
centred last digit recovers every ``|Z| <= HALF = p0 p1 (p2 - 1) / 2 ~ 2^86``; at rank 16 the bound is
``2^76``. Every call checks the exact condition (:func:`exact`) and falls back to the dense product if
it fails (it cannot for the protocol's parameters).

Functions with the dense signatures (drop-in, fall back to the dense path when the native module is
missing, ``PGW_NTT=0`` / ``PGW_NATIVE=0``, or ``small`` is not a block-negacyclic secret matrix):

* :func:`big_planes_mm_small` (same as :func:`pgw.hemask.fastmm.big_planes_mm_small`);
* :func:`matmul_mod64` (same as :func:`pgw.hemask.lwe.matmul_mod64`; NTT for ``small_left=False``).

For operands reused across calls, :class:`TransformedOperand` keeps the forward NTTs (12 bytes per
coefficient instead of 8 for ``A``): the public matrix (every linear multiplies a prefix ``A[:n]``),
or a hint held in NTT form.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from pgw.core.errors import ConfigError
from pgw.hemask.fastmm import big_planes_mm_small as _dense_planes_mm_small
from pgw.hemask.lwe import matmul_mod64 as _dense_matmul_mod64

__all__ = [
    "DEGREE",
    "TransformedOperand",
    "available",
    "big_planes_mm_small",
    "exact",
    "info",
    "matmul_mod64",
    "secrets_of",
]

DEGREE = 256
_OFF = ("0", "false", "no", "off")


def _load() -> Any:
    if os.environ.get("PGW_NATIVE", "1").strip().lower() in _OFF:
        return None
    if os.environ.get("PGW_NTT", "1").strip().lower() in _OFF:
        return None
    try:
        import pgw_ntt  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return None
    return pgw_ntt


#: The loaded ``pgw_ntt`` module, or ``None`` (dense path).
native: Any = _load()


def available() -> bool:
    """True when the native NTT module is loaded (and not disabled)."""
    return native is not None


def info() -> dict[str, Any]:
    """Primes, CRT range and instruction set of the native module (``{}`` without it)."""
    if native is None:
        return {}
    d = dict(native.info())
    d["half"] = int(d["half"])
    return d


def _threads() -> int:
    try:
        return max(0, int(os.environ.get("PGW_NTT_THREADS", "0")))
    except ValueError:
        return 0


def _isa() -> str:
    return os.environ.get("PGW_NTT_ISA", "auto")


def secrets_of(small: np.ndarray) -> np.ndarray | None:
    """The ``(G, rank, 256)`` int8 secret polynomials of a block-negacyclic matrix ``small``
    (``rank*256 x G*256``, :func:`pgw.hemask.lwe.secret_matrix` blocks concatenated along columns), or
    ``None`` if ``small`` does not have exactly that structure (or the native module is missing)."""
    if native is None:
        return None
    s = np.asarray(small)
    if s.ndim != 2 or s.size == 0 or s.shape[0] % DEGREE or s.shape[1] % DEGREE:
        return None
    if s.dtype != np.int8:
        if not np.issubdtype(s.dtype, np.integer) or int(np.abs(s.astype(np.int64)).max()) > 127:
            return None
        s = s.astype(np.int8)
    out: np.ndarray | None = native.extract_secrets(np.ascontiguousarray(s))
    return out


def exact(big_bits: int, secrets: np.ndarray) -> bool:
    """True when products of a big operand below ``2^big_bits`` with ``secrets`` are exact
    (``(2^big_bits - 1) * max_g L1(s_g) <= HALF``; checked again natively on every call)."""
    if native is None:
        return False
    l1 = int(np.abs(secrets.astype(np.int64)).reshape(secrets.shape[0], -1).sum(1).max())
    return bool(native.exact_for(int(big_bits), l1))


def big_planes_mm_small(
    planes: np.ndarray, small: np.ndarray, *, block_rows: int = 2048, first_byte: int | None = None
) -> np.ndarray:
    """``U @ S mod 2^64`` with ``U`` as byte planes ``(rows, P, k)``; same contract and bits as
    :func:`pgw.hemask.fastmm.big_planes_mm_small` (planes hold bytes ``first_byte .. first_byte+P-1``,
    default ``8 - P``; missing bytes count as zero)."""
    _rows, npl, k = planes.shape
    first = 8 - npl if first_byte is None else first_byte
    if native is not None and 1 <= npl <= 8 and 0 <= first and first + npl <= 8 and np.asarray(small).shape[0] == k:
        sec = secrets_of(small)
        if sec is not None and exact(8 * (first + npl), sec):
            p = planes if planes.dtype == np.int8 else np.asarray(planes).view(np.int8)
            out: np.ndarray = native.mm_planes(np.ascontiguousarray(p), sec, first, _threads(), _isa())
            return out
    return _dense_planes_mm_small(planes, small, block_rows=block_rows, first_byte=first_byte)


def matmul_mod64(
    small: np.ndarray, big: np.ndarray, *, small_left: bool = True, fast: bool | None = None
) -> np.ndarray:
    """Exact ``small @ big`` (or ``big @ small``) mod ``2^64``; same contract and bits as
    :func:`pgw.hemask.lwe.matmul_mod64`. ``big @ S`` for a block-negacyclic secret ``S`` uses NTTs;
    everything else (weights times ``A`` for the hint build) goes to the dense path."""
    if native is not None and not small_left:
        b = np.asarray(big)
        if b.ndim == 2 and b.shape[1] == np.asarray(small).shape[0]:
            sec = secrets_of(small)
            if sec is not None and exact(64, sec):
                out: np.ndarray = native.mm_u64(np.ascontiguousarray(b, dtype=np.uint64), sec, _threads(), _isa())
                return out
    return _dense_matmul_mod64(small, big, small_left=small_left, fast=fast)


class TransformedOperand:
    """A big operand in NTT form (residues mod the three primes, forward-transformed), for operands
    multiplied by many secrets: the public matrix ``A`` (shared by all linears as prefixes ``A[:n]``
    and by all groups) or a hint kept in NTT form. ``12 * rows * dim`` bytes (rows padded to 16).

    ``mm(small, rows=n)`` equals ``big_planes_mm_small(planes[:n], small)`` (or
    ``matmul_mod64(small, a[:n], small_left=False)``) bit for bit.
    """

    def __init__(self, handle: Any):
        self._h = handle

    @classmethod
    def from_u64(cls, big: np.ndarray) -> TransformedOperand:
        if native is None:
            raise ConfigError("pgw_ntt is not available (build it with rust/tools/build_ntt.py)")
        return cls(native.Transformed.from_u64(np.ascontiguousarray(big, dtype=np.uint64), _threads(), _isa()))

    @classmethod
    def from_planes(cls, planes: np.ndarray, first_byte: int | None = None) -> TransformedOperand:
        if native is None:
            raise ConfigError("pgw_ntt is not available (build it with rust/tools/build_ntt.py)")
        first = 8 - planes.shape[1] if first_byte is None else first_byte
        p = planes if planes.dtype == np.int8 else np.asarray(planes).view(np.int8)
        return cls(native.Transformed.from_planes(np.ascontiguousarray(p), first, _threads(), _isa()))

    @property
    def rows(self) -> int:
        return int(self._h.rows)

    @property
    def rank(self) -> int:
        return int(self._h.rank)

    @property
    def nbytes(self) -> int:
        return int(self._h.nbytes)

    def mm_secrets(self, secrets: np.ndarray, rows: int | None = None) -> np.ndarray:
        """``U[:rows] @ S mod 2^64`` for secrets ``(G, rank, 256)`` int8."""
        if not exact(int(self._h.big_bits), secrets):
            raise ConfigError("secret too large for an exact NTT product")
        out: np.ndarray = self._h.mm(np.ascontiguousarray(secrets, dtype=np.int8), rows, _threads(), _isa())
        return out

    def mm(self, small: np.ndarray, rows: int | None = None) -> np.ndarray:
        """``U[:rows] @ small mod 2^64`` for a block-negacyclic secret matrix ``small``."""
        sec = secrets_of(small)
        if sec is None:
            raise ConfigError("small is not a block-negacyclic secret matrix (lwe.secret_matrix)")
        return self.mm_secrets(sec, rows)
