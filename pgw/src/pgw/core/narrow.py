"""Narrow wire rings for masked linears (opt-in prototype, off by default; ``research/reviews/consumer-bandwidth.md``).

Today every masked value travels as a ``K``-bit ring element (K32A12C27: 4 bytes up per input, 4 bytes down per
output). This module sends fewer bits, exactly, with the same privacy and the same integrity check:

* **Output widths** (sound for every client). A genuine output of a capped ring satisfies, per output row ``j``,
  ``|y_j| <= B_j`` with ``B_j = min(qmax ||W8_j||_1, ceil(2^C ||W8_j||_2 / wnorm) - 1)`` (``wnorm`` the largest row
  norm: the gateway's cap check ``wnorm ||xq||_2 (1 + 1e-9) <= 2^C`` makes the bound strict). Every value in
  ``[-B_j, B_j]`` is determined by its residue mod ``2^w_j`` with ``w_j = bitlen(2 B_j)`` (at most ``C + 1``), so the
  provider returns ``y_hat_j mod 2^w_j``, the client subtracts ``(W8 r)_j mod 2^w_j`` and centres in ``w_j`` bits.
  The widths depend only on ``W8`` and the ring (public), never on data.
* **Input width** ``m = in_bits``. The client sends ``x_hat = (xq + r) mod 2^m`` with ``r`` uniform in ``[0, 2^m)``,
  so ``x_hat`` is exactly uniform in ``Z_(2^m)`` for every ``xq``. The provider multiplies the integers
  ``x_hat`` in ``[0, 2^m)`` (its K32 kernel is unchanged: ``W8 x_hat mod 2^32``, then the low ``w_j`` bits). With
  the carry vector ``c = (xq + r - x_hat) / 2^m`` (in ``{-1, 0, 1}`` when ``2^m > qmax``):
  ``W8 xq = W8 x_hat - W8 r + 2^m W8 c``. For rows with ``w_j <= m`` the carry term vanishes mod ``2^w_j``
  (``m = max w_j`` is the *weight-free* mode: no extra client work, any K32 mask can be truncated). For
  ``m < w_j`` the client adds ``2^m (W8 c)_j``, a sparse column sum over ``W8`` (the client must hold the int8
  weights: the factory pack); ``P(c_i != 0) = |xq_i| / 2^m <= qmax / 2^m`` independently per element, so the work
  is padded to a data-independent budget (:func:`carry_budget`) and its time reveals nothing. ``c`` never leaves
  the client.
* **Integrity.** Decoded rows are range-checked against ``B_j`` and verified by the prime-field Freivalds check
  (``p = 2^31 - 1``) on the unmasked integers exactly as today: an accepted wrong row differs from the genuine one by
  at most ``2^(w_j) <= 2^(C+1) < p`` per entry, so the soundness bound is unchanged. (The ring check over
  ``Z_(2^K)`` needs headroom above the outputs and does not apply to narrow rings: the p31 check is required.)

Bit packing uses the maskgen v2 layout (:func:`pgw.hemask.wire.pack_bits`); output rows are grouped by width in a
fixed public order (:attr:`NarrowPlan.order`). Nothing here changes any existing path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError, ProtocolError, ShapeError
from pgw.core.ring import RingParams

__all__ = [
    "NarrowPlan",
    "carry_budget",
    "carry_term",
    "decode_outputs",
    "mask_correction",
    "mask_inputs",
    "pack_down",
    "pack_up",
    "plan_linear",
    "provider_apply",
    "unpack_down",
    "unpack_up",
]

_U64 = np.uint64
#: Over-estimate of the float64 error of ``2^C * sqrt(a / b)`` (actual: a few ulps, ~1e-15 relative).
_REL_SLACK = 2.0 ** -40


def _bitlen(v: np.ndarray) -> np.ndarray:
    """Exact ``int.bit_length`` of non-negative int64 values below ``2^53`` (``frexp`` exponent)."""
    v = np.asarray(v, dtype=np.int64)
    if v.size and (int(v.min()) < 0 or int(v.max()) >= 1 << 53):
        raise ConfigError("bit lengths need values in [0, 2^53)")
    _, e = np.frexp(v.astype(np.float64))
    return np.where(v > 0, e, 0).astype(np.int64)


def _pad8(count: int) -> int:
    return (count + 7) // 8 * 8


@dataclass(frozen=True)
class NarrowPlan:
    """Public wire widths of one linear (derived from ``W8`` and the ring only)."""

    ring_name: str
    in_features: int
    in_bits: int
    out_bits: np.ndarray  # int64 (m,): bits per returned value of output row j
    bound: np.ndarray  # int64 (m,): |y_j| <= bound_j for every genuine output
    order: np.ndarray  # int64 (m,): wire order of the output rows (grouped by width, stable)
    groups: tuple[tuple[int, int, int], ...]  # (bits, start, stop) over ``order``
    carry_rows: np.ndarray  # int64: output rows with out_bits > in_bits (need the carry term)

    @property
    def out_features(self) -> int:
        return int(self.out_bits.size)

    def up_bytes(self, rows: int) -> int:
        """Bytes of the packed ``x_hat`` of ``rows`` rows."""
        return _pad8(rows * self.in_features) * self.in_bits // 8

    def down_bytes(self, rows: int) -> int:
        """Bytes of the packed ``y_hat`` of ``rows`` rows."""
        return sum(_pad8(rows * (b - a)) * w // 8 for w, a, b in self.groups)

    def k_bytes(self, rows: int, k: int = 32) -> tuple[int, int]:
        """(up, down) bytes of the same call on today's ``K``-bit wire."""
        return rows * self.in_features * k // 8, rows * self.out_features * k // 8


def plan_linear(w8: np.ndarray | torch.Tensor, ring: RingParams, in_bits: int | None = None) -> NarrowPlan:
    """The narrow plan of a linear with int8 weight ``w8`` ``(m, n)`` under ``ring``.

    ``in_bits=None``: the weight-free mode (``max_j w_j``: no carry work). Otherwise ``ring.a <= in_bits <=
    max_j w_j`` (carry-corrected rows ``w_j > in_bits`` need the client to hold ``W8``).
    """
    w = (w8.detach().cpu().numpy() if isinstance(w8, torch.Tensor) else np.asarray(w8)).astype(np.int64)
    if w.ndim != 2:
        raise ShapeError("weight must be 2-D")
    m, n = w.shape
    l1 = np.abs(w).sum(axis=1)
    bound = ring.qmax * l1  # |xq_i| <= qmax for every adaptive bit width <= A
    if ring.cap:
        nsq = (w * w).sum(axis=1)  # exact integers (< 2^53)
        nmax = int(nsq.max()) if nsq.size else 0
        if nmax:
            rho = np.sqrt(nsq.astype(np.float64) / float(nmax))
            b2 = np.ceil(float(1 << ring.cap) * rho * (1.0 + _REL_SLACK)).astype(np.int64) - 1
            bound = np.minimum(bound, np.maximum(b2, 0))
        bound = np.minimum(bound, (1 << ring.cap) - 1)
    elif int(bound.max(initial=0)) >= 1 << (ring.k - 1):
        raise ConfigError(f"{ring.name}: outputs of width {n} can wrap the ring")
    out_bits = np.maximum(_bitlen(2 * bound), 1)
    top = int(out_bits.max(initial=1))
    if top > ring.k:
        raise ConfigError("an output needs more bits than the ring")
    mb = top if in_bits is None else int(in_bits)
    if not (ring.a <= mb <= ring.k) or mb > max(top, ring.a):
        raise ConfigError(f"in_bits must be in [{ring.a}, {top}], got {mb}")
    order = np.argsort(out_bits, kind="stable").astype(np.int64)
    sb = out_bits[order]
    groups = []
    start = 0
    for i in range(1, m + 1):
        if i == m or sb[i] != sb[start]:
            groups.append((int(sb[start]), start, i))
            start = i
    carry_rows = np.nonzero(out_bits > mb)[0].astype(np.int64)
    return NarrowPlan(ring.name, n, mb, out_bits, bound.astype(np.int64), order, tuple(groups), carry_rows)


# ---- client: mask, carries, corrections, decode ---------------------------------------------------------------


def mask_inputs(xq: np.ndarray, r: np.ndarray, plan: NarrowPlan) -> tuple[np.ndarray, np.ndarray]:
    """``(x_hat, c)``: ``x_hat = (xq + r) mod 2^m`` (uint64) and the carries ``c = (xq + r - x_hat) / 2^m`` (int8)."""
    xq = np.asarray(xq, dtype=np.int64)
    r = np.asarray(r, dtype=np.int64)
    if xq.shape != r.shape or xq.ndim != 2 or xq.shape[1] != plan.in_features:
        raise ShapeError("xq and r must be (T, n) of the plan's width")
    if r.size and (int(r.min()) < 0 or int(r.max()) >= 1 << plan.in_bits):
        raise ConfigError("narrow masks must lie in [0, 2^in_bits)")
    s = xq + r
    xh = s & ((1 << plan.in_bits) - 1)
    c = (s - xh) >> plan.in_bits
    if c.size and int(np.abs(c).max()) > 1:
        raise ConfigError("carries outside {-1, 0, 1}: inputs exceed 2^in_bits")
    return xh.astype(_U64), c.astype(np.int8)


def _row_masks(plan: NarrowPlan) -> np.ndarray:
    return ((np.ones(plan.out_features, dtype=_U64) << plan.out_bits.astype(_U64)) - _U64(1)).astype(_U64)


def mask_correction(w8t: torch.Tensor, r: np.ndarray, plan: NarrowPlan) -> np.ndarray:
    """``(W8 r)_j mod 2^w_j`` (uint64 ``(T, m)``) for masks ``r`` in ``[0, 2^m)`` (the factory's product)."""
    from pgw.engine.backends import exact_int_matmul

    wr = exact_int_matmul(torch.from_numpy(np.ascontiguousarray(r, dtype=np.int64)), w8t, plan.in_bits + 1)
    return (wr.numpy().astype(_U64)) & _row_masks(plan)[None, :]


def carry_budget(n: int, in_bits: int, qmax: int, fail_log2: float = -40.0) -> int:
    """Smallest ``k`` with ``P(Bin(n, qmax / 2^m) > k) <= 2^fail_log2``: a data-independent bound on the number
    of non-zero carries of one row for any input (each carry is non-zero with probability ``|xq_i| / 2^m <=
    qmax / 2^m``, independently)."""
    p = min(1.0, qmax / float(1 << in_bits))
    if p >= 1.0:
        return n
    lp, lq = math.log(p), math.log1p(-p)
    target = fail_log2 * math.log(2.0)

    def logpmf(i: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq

    tail = -math.inf  # log P(X > k), summed from the top
    k = n
    while k > 0:
        nxt = logpmf(k)
        new_tail = nxt if tail == -math.inf else max(tail, nxt) + math.log1p(math.exp(-abs(tail - nxt)))
        if new_tail > target:  # P(X > k - 1) would exceed the target: k is the budget
            return k
        tail = new_tail
        k -= 1
    return 0


def carry_term(w8t: torch.Tensor, c: np.ndarray, plan: NarrowPlan, budget: int) -> tuple[np.ndarray, int]:
    """``2^m (W8 c)_j mod 2^w_j`` on the carry rows (uint64 ``(T, m)``, zero elsewhere) and the overflow count.

    Every row gathers exactly ``budget`` columns of ``W8`` (padding with zero carries), so the work does not
    depend on the data unless a row has more than ``budget`` non-zero carries (probability <= the budget's
    target; such rows are still exact, and counted).
    """
    t = c.shape[0]
    out = np.zeros((t, plan.out_features), dtype=_U64)
    rows = plan.carry_rows
    if not rows.size or not t:
        return out, 0
    w = w8t.cpu().numpy()  # (n, m) int8
    wr = w[:, rows]  # carry rows only (static, public selection)
    over = 0
    for i in range(t):
        idx = np.nonzero(c[i])[0]
        if idx.size > budget:
            over += 1
            take = idx
        else:
            take = np.zeros(budget, dtype=np.int64)
            take[: idx.size] = idx
        sign = np.zeros(take.size, dtype=np.int64)
        sign[: idx.size] = c[i, idx]
        acc = (sign[:, None] * wr[take].astype(np.int64)).sum(axis=0)
        out[i, rows] = (acc << plan.in_bits).astype(_U64)
    out &= _row_masks(plan)[None, :]
    return out, over


def decode_outputs(y_hat: np.ndarray, corr: np.ndarray, carry: np.ndarray | None, plan: NarrowPlan) -> np.ndarray:
    """Exact ``W8 xq`` (int64 ``(T, m)``) from the returned residues; range-checked against the plan's bounds."""
    msk = _row_masks(plan)[None, :]
    v = (np.asarray(y_hat, dtype=_U64) - corr) & msk
    if carry is not None:
        v = (v + carry) & msk
    half = (_U64(1) << (plan.out_bits.astype(_U64) - _U64(1)))[None, :]
    y = v.astype(np.int64) - np.where(v >= half, (msk.astype(np.int64) + 1), 0)
    if y.size and bool((np.abs(y) > plan.bound[None, :]).any()):
        raise IntegrityError("narrow output outside its row bound (tampered or wrong mask)")
    return y


# ---- wire ------------------------------------------------------------------------------------------------------


def pack_up(x_hat: np.ndarray, plan: NarrowPlan) -> bytes:
    """``x_hat`` ``(T, n)`` as ``in_bits``-bit values (maskgen v2 bit layout, zero-padded to 8 values)."""
    from pgw.hemask.wire import pack_bits

    flat = np.ascontiguousarray(x_hat, dtype=_U64).reshape(-1)
    buf = np.zeros(_pad8(flat.size), dtype=_U64)
    buf[: flat.size] = flat
    return bytes(pack_bits(buf, plan.in_bits))


def unpack_up(data: bytes | bytearray | memoryview, rows: int, plan: NarrowPlan) -> np.ndarray:
    """Provider side of :func:`pack_up` (length-checked)."""
    from pgw.hemask.wire import unpack_bits

    if len(data) != plan.up_bytes(rows):
        raise ProtocolError(f"narrow payload has {len(data)} bytes, expected {plan.up_bytes(rows)}")
    count = rows * plan.in_features
    vals = unpack_bits(data, _pad8(count), plan.in_bits)
    if vals[count:].any():
        raise ProtocolError("narrow payload: non-zero padding")
    return vals[:count].reshape(rows, plan.in_features)


def pack_down(y: np.ndarray, plan: NarrowPlan) -> bytes:
    """The low ``w_j`` bits of ``y`` ``(T, m)``, rows grouped by width in the plan's public order."""
    from pgw.hemask.wire import pack_bits

    ya = np.asarray(y, dtype=_U64)
    parts = []
    for w, a, b in plan.groups:
        g = np.ascontiguousarray(ya[:, plan.order[a:b]]).reshape(-1)
        buf = np.zeros(_pad8(g.size), dtype=_U64)
        buf[: g.size] = g & _U64((1 << w) - 1)
        parts.append(bytes(pack_bits(buf, w)))
    return b"".join(parts)


def unpack_down(data: bytes | bytearray | memoryview, rows: int, plan: NarrowPlan) -> np.ndarray:
    """Client side of :func:`pack_down`: uint64 ``(T, m)`` in natural row order (length-checked)."""
    from pgw.hemask.wire import unpack_bits

    if len(data) != plan.down_bytes(rows):
        raise ProtocolError(f"narrow reply has {len(data)} bytes, expected {plan.down_bytes(rows)}")
    mv = memoryview(data)
    out = np.empty((rows, plan.out_features), dtype=_U64)
    pos = 0
    for w, a, b in plan.groups:
        cnt = rows * (b - a)
        nb = _pad8(cnt) * w // 8
        vals = unpack_bits(mv[pos:pos + nb], _pad8(cnt), w)
        if vals[cnt:].any():
            raise ProtocolError("narrow reply: non-zero padding")
        out[:, plan.order[a:b]] = vals[:cnt].reshape(rows, b - a)
        pos += nb
    return out


def provider_apply(w8t: torch.Tensor, x_hat: np.ndarray, plan: NarrowPlan) -> bytes:
    """The provider: ``W8 x_hat`` for integers ``x_hat`` in ``[0, 2^m)`` (exact; equal to today's K32 product mod
    ``2^32`` reduced further), the low ``w_j`` bits of each output, packed. Sees only ``x_hat``."""
    from pgw.engine.backends import exact_int_matmul

    xt = torch.from_numpy(np.ascontiguousarray(x_hat).astype(np.int64))
    y = exact_int_matmul(xt, w8t, plan.in_bits + 1)
    return pack_down(y.numpy().astype(_U64), plan)
