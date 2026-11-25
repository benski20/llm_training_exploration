"""Client-side check of a provider-computed pks-v3 hint operand ``H = W8 A mod 2^56`` (2-adic Freivalds).

Why this form. The stored pks-v3 hint is ``c1 = ModDown(sum_l A~_l k_l) mod q1`` with ``A~ = centred(round(q1 H /
2^56) mod q1)`` (:func:`pgw.hemask.pks.compact_hint`). Between ``W8`` and ``c1`` there are two exact roundings, so
``c1`` is not a linear function of ``W8`` and no random projection of ``c1`` can be compared with a projection of
``W8``: a check would need the first rounding's remainders (one 56-bit value per entry of ``H``, as large as ``H``
itself) and a range proof for them. The expensive part of the build is the GEMM ``H = W8 A`` (60+% of the CPU
time); everything after it (``switch_hint``, the key switch) is cheap and deterministic. So the provider computes
and streams ``H``, the client checks ``H`` exactly with the test below and computes the rest itself.

The test (per linear, ``W8`` int8 ``m x n``, ``A`` public ``n x 4096`` from the customer's seed, claimed ``H``
``m x 4096`` with entries ``< 2^56``): draw ``U in {0,1}^(m x d)`` and ``X in {0,1}^(4096 x d)`` from the OS CSPRNG,
never sent anywhere, and accept iff ``U^T H X == (U^T W8)(A X)  (mod 2^56)`` (a ``d x d`` matrix).

Soundness (``d = CHALLENGE_COLS = 42``): let ``E = H - W8 A mod 2^56 != 0``, ``2^v`` the largest power of two dividing
every entry (``v <= 55``) and ``E' = E / 2^v``, which is nonzero mod 2. ``U^T E X == 0 mod 2^56`` implies
``U^T E' X == 0 mod 2`` (divide by ``2^v``, reduce mod 2). Over GF(2): ``E' X = 0`` has probability at most
``2^-d`` (each of the ``d`` independent uniform columns ``x`` hits the kernel of the nonzero ``E'`` with probability
``2^-rank <= 1/2``); if ``Z = E' X != 0``, ``U^T Z = 0`` has probability at most ``2^-d`` likewise. So a wrong ``H``
passes with probability ``<= 2^(1-d) = 2^-41``, for ANY error the provider chooses (it never sees ``U``, ``X``; fresh
per linear and per attempt). This is the mod-2^k pitfall handled properly: a single uniform vector over ``Z_2^56``
misses ``E = 2^55 E'`` with probability 1/2, so ``d`` binary columns (one bit each) are needed; the entries above
bit 0 of the challenges add nothing against such errors, hence binary challenges (cheapest exact arithmetic).

Arithmetic: everything mod ``2^64`` (uint64 wrap, a ring map onto mod ``2^56``); the products are float64 GEMMs on
28-bit slices (every partial sum ``< 4096 * 2^28 = 2^40 < 2^53``, exact) and float32 GEMMs for ``U^T W8`` (sums
``< 127 * 4096 < 2^24`` per block). ``A X`` is computed from the customer's ring rows without expanding ``A``
(:func:`a_times_x`).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from pgw.core.errors import ConfigError
from pgw.core.prng import SeedExpander, new_seed
from pgw.hemask import pks
from pgw.hemask.pks import NU, RANK, D

__all__ = ["CHALLENGE_COLS", "H_BYTES", "SOUNDNESS_BITS", "LinearHCheck", "a_times_x", "decode_h56", "encode_h56"]

#: Binary challenge columns on each side: a wrong ``H`` passes with probability ``<= 2^(1 - CHALLENGE_COLS)``.
CHALLENGE_COLS = 42
SOUNDNESS_BITS = CHALLENGE_COLS - 1
#: Bytes per entry of ``H`` on the wire (56 bits little endian).
H_BYTES = 7
_U64 = np.uint64
_M56 = _U64((1 << 56) - 1)
_M28 = _U64((1 << 28) - 1)
#: Rows of ``W8`` per float32 block of ``U^T W8`` (``127 * rows < 2^24`` keeps every float32 sum exact).
_W_BLOCK = 4096


def encode_h56(h: np.ndarray) -> bytes:
    """``(rows, 4096)`` uint64 (< 2^56) -> ``rows * 4096 * 7`` bytes (little endian, row major)."""
    h = np.ascontiguousarray(h, dtype="<u8")
    return np.ascontiguousarray(h.view(np.uint8).reshape(*h.shape, 8)[..., :H_BYTES]).tobytes()


def decode_h56(data: bytes | bytearray | memoryview, rows: int) -> np.ndarray:
    """Inverse of :func:`encode_h56` (``(rows, 4096)`` uint64, every entry ``< 2^56`` by construction)."""
    if len(data) != rows * NU * H_BYTES:
        raise ConfigError(f"H chunk has {len(data)} bytes, expected {rows * NU * H_BYTES}")
    raw = np.frombuffer(data, dtype=np.uint8).reshape(rows, NU, H_BYTES)
    full = np.zeros((rows, NU, 8), dtype=np.uint8)

    def part(b0: int, b1: int) -> None:
        full[b0:b1, :, :H_BYTES] = raw[b0:b1]

    _row_parallel(part, rows)
    return full.view("<u8").reshape(rows, NU).astype(np.uint64, copy=False)


_POOL: ThreadPoolExecutor | None = None


def _row_parallel(fn: Any, rows: int, min_rows: int = 64) -> list[Any]:
    """``fn(b0, b1)`` over row ranges in a shared thread pool (numpy and BLAS release the GIL)."""
    global _POOL
    nt = max(1, os.cpu_count() or 1)
    if nt == 1 or rows < 2 * min_rows:
        return [fn(0, rows)]
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=nt, thread_name_prefix="pgw-hcheck")
    step = max(min_rows, -(-rows // nt))
    return list(_POOL.map(lambda b0: fn(b0, min(rows, b0 + step)), range(0, rows, step)))


def _bits(sx: SeedExpander, shape: tuple[int, int], label: str) -> np.ndarray:
    n = shape[0] * shape[1]
    raw = np.frombuffer(sx.random_bytes((n + 7) // 8, label), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n].reshape(shape)


def _mm_u64_small(big: np.ndarray, small: np.ndarray) -> np.ndarray:
    """``big @ small mod 2^64`` for ``big`` uint64 entries ``< 2^56`` and ``small`` entries in ``{-1, 0, 1}``
    (float64 matrix), exact: two 28-bit slices, every float64 partial sum ``< inner * 2^28``."""
    inner = big.shape[1]
    if inner > (1 << 24):
        raise ConfigError("inner dimension too large for exact slicing")
    lo = (big & _M28).astype(np.float64)
    hi = (big >> _U64(28)).astype(np.float64)
    r_lo = np.rint(lo @ small).astype(np.int64).astype(_U64)
    del lo
    r_hi = np.rint(hi @ small).astype(np.int64).astype(_U64)
    return (r_hi << _U64(28)) + r_lo


def _x_cat(x: np.ndarray) -> np.ndarray:
    """``(4096, 16 d)`` float64: block ``k`` maps the customer's ring row ``a_C`` (coefficient ``16 u + s`` is
    component ``s`` at ``u``) to row ``16 C + k`` of ``A X`` (see :func:`a_times_x`)."""
    d = x.shape[1]
    xb = x.astype(np.float64).reshape(RANK, D, d)  # xb[l, u] = X[256 l + u]
    xs = np.empty_like(xb)  # x''[w] = x[w + 1] (w < 255), x''[255] = -x[0]: <X a, x> = <a, x''>
    xs[:, :-1] = xb[:, 1:]
    xs[:, -1] = -xb[:, 0]
    z = np.empty((D, RANK, RANK, d), dtype=np.float64)  # [u, s, k, j]: ring position 16 u + s, output row 16 C + k
    for k in range(RANK):
        for s in range(RANK):
            z[:, s, k] = xb[k - s] if s <= k else xs[k - s + RANK]
    return z.reshape(NU, RANK * d)


def a_times_x(a_seed: bytes, linear: str, n: int, x: np.ndarray, *, block: int = 256) -> np.ndarray:
    """``A X mod 2^64`` (``(n, d)`` uint64) for ``A = a_mlwe_rows(public_ring_rows(a_seed, n, linear))`` and binary
    ``X`` (``(4096, d)``), from the ``n/16`` ring rows (``A`` itself is 16x larger and never built).

    Row ``16 C + k`` of ``A`` has component ``l`` (256 coefficients) equal to component ``k - l`` of ``a_C``
    (``l <= k``) or ``X`` times component ``k - l + 16`` (``l > k``, negacyclic in ``R_256``);
    ``<X c, x> = <c, x''>`` with ``x''`` the shifted, sign-flipped slice.
    Equal to ``matmul_mod64(a_mlwe_rows(...), X)`` mod ``2^56`` (``tests/provision/test_hintcheck.py``)."""
    d = x.shape[1]
    zc = _x_cat(x)
    ring = pks.public_ring_rows(a_seed, n, linear)  # (n/16, 4096) < 2^56
    out = np.empty((n // RANK, RANK, d), dtype=_U64)
    for c0 in range(0, ring.shape[0], block):
        out[c0:c0 + block] = _mm_u64_small(ring[c0:c0 + block], zc).reshape(-1, RANK, d)
    return out.reshape(n, d)


class LinearHCheck:
    """The check of one linear's ``H``, fed in row blocks (any order, each row once) of ``W8`` and ``H``.

    ``add_weights(r0, w)`` and ``add_h(r0, h)`` accumulate ``U^T W8`` and ``U^T (H X)``; :meth:`passed` compares.
    The challenges are drawn here from the OS CSPRNG and never leave the object."""

    def __init__(self, a_seed: bytes, linear: str, m: int, n: int, *, cols: int = CHALLENGE_COLS,
                 _challenge_seed: bytes | None = None):
        if n % RANK:
            raise ConfigError(f"{linear}: in_features {n} is not a multiple of {RANK}")
        sx = SeedExpander(new_seed() if _challenge_seed is None else _challenge_seed)
        self.linear, self.m, self.n, self.cols = linear, int(m), int(n), int(cols)
        self._u = _bits(sx, (self.m, self.cols), "u")
        x = _bits(sx, (NU, self.cols), "x")
        self._xf = x.astype(np.float64)
        self._ax = a_times_x(a_seed, linear, self.n, x)  # (n, d)
        self._uw = np.zeros((self.cols, self.n), dtype=np.int64)
        self._gl = np.zeros((self.cols, self.cols), dtype=_U64)
        self._w_rows = np.zeros(self.m, dtype=bool)
        self._h_rows = np.zeros(self.m, dtype=bool)

    def _mark(self, seen: np.ndarray, r0: int, rows: int, what: str) -> None:
        if r0 < 0 or r0 + rows > self.m or seen[r0:r0 + rows].any():
            raise ConfigError(f"{self.linear}: {what} rows [{r0}, {r0 + rows}) out of range or repeated")
        seen[r0:r0 + rows] = True

    def add_weights(self, r0: int, w: np.ndarray) -> None:
        """Rows ``[r0, r0 + len(w))`` of ``W8`` (int8)."""
        w = np.asarray(w)
        if w.ndim != 2 or w.shape[1] != self.n:
            raise ConfigError(f"{self.linear}: weight block must be (rows, {self.n})")
        self._mark(self._w_rows, r0, w.shape[0], "weight")
        for b0 in range(0, w.shape[0], _W_BLOCK):
            wb = w[b0:b0 + _W_BLOCK].astype(np.float32)
            ub = self._u[r0 + b0:r0 + b0 + wb.shape[0]].T.astype(np.float32)
            self._uw += np.rint(ub @ wb).astype(np.int64)

    def add_h(self, r0: int, h: np.ndarray) -> None:
        """Rows ``[r0, r0 + len(h))`` of the claimed ``H`` (uint64, entries ``< 2^56``)."""
        h = np.asarray(h, dtype=_U64)
        if h.ndim != 2 or h.shape[1] != NU:
            raise ConfigError(f"{self.linear}: H block must be (rows, {NU})")
        if (h > _M56).any():
            raise ConfigError(f"{self.linear}: H entry out of range")
        self._mark(self._h_rows, r0, h.shape[0], "H")

        def part(b0: int, b1: int) -> np.ndarray:
            hx = _mm_u64_small(h[b0:b1], self._xf)  # (rows, d)
            ub = self._u[r0 + b0:r0 + b1].astype(_U64)
            out: np.ndarray = ub.T @ hx  # wraps mod 2^64
            return out

        for g in _row_parallel(part, h.shape[0]):
            self._gl += g

    @property
    def complete(self) -> bool:
        return bool(self._w_rows.all() and self._h_rows.all())

    def passed(self) -> bool:
        """``U^T H X == (U^T W8)(A X) mod 2^56``; ``False`` unless every row of ``W8`` and ``H`` was fed."""
        if not self.complete:
            return False
        gr = self._uw.astype(_U64) @ self._ax  # (d, d), wraps mod 2^64 (negative entries as two's complement)
        ok = bool(((self._gl - gr) & _M56 == 0).all())
        self.wipe()
        return ok

    def wipe(self) -> None:
        """Drop the challenges (a check object is used for one verdict)."""
        for a in (self._u, self._xf, self._ax, self._uw, self._gl):
            a.fill(0)


def check_linear(a_seed: bytes, linear: str, w8: np.ndarray, h_blocks: Iterable[tuple[int, np.ndarray]]) -> bool:
    """Convenience: the whole check of one linear (``h_blocks`` yields ``(r0, rows)``)."""
    m, n = w8.shape
    ck = LinearHCheck(a_seed, linear, m, n)
    ck.add_weights(0, w8)
    for r0, h in h_blocks:
        ck.add_h(r0, h)
    return ck.passed()
