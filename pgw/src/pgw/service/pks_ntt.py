# mypy: disable-error-code="no-untyped-def,untyped-decorator"
"""Fused Triton NTTs for the ``pks-v3`` provider key switch (:func:`pgw.service.pks_gpu._keyswitch_t`).

The torch key switch runs ``16 + G`` negacyclic NTTs of 8192 points per 32 output rows and prime as ~90 eager int64
operators each (every butterfly stage reads and writes the whole polynomial several times). Here one Triton program
holds one polynomial (8192 residues, 32 KiB as uint32) on chip and runs all 13 butterfly stages there, so a transform
reads its input once and writes its output once:

* :func:`forward` (one launch per prime): gathers the centred reply-key operand ``A~`` straight from the switched hint
  rows (``A~[b, l, 32 u + j] = hs[32 b + j, 256 l + u]``), reduces it mod ``p`` and runs the forward transform of
  :meth:`pgw.hemask.pks.Ntt.fwd` (Cooley-Tukey, bit-reversed output); stores uint32 residues.
* :func:`mac_inverse` (one launch per prime): per (reply ciphertext, group) the pointwise products with the 16 key
  polynomials, their sum, and the inverse transform of :meth:`pgw.hemask.pks.Ntt.inv` (Gentleman-Sande, scaled by
  ``1/8192``).

Arithmetic (all integer, exact for every prime of :data:`pgw.hemask.pks.QP`, ``2^23 < p < 2^30``):

* twiddle products: Shoup, ``w' = floor(w 2^32 / p)``: ``q = umulhi(a, w')``, ``r = a w - q p mod 2^32`` in
  ``[0, 2p)`` for any ``a < 2^32``, one conditional subtraction;
* key products: Montgomery (``R = 2^32``): ``REDC(a b) = (a b - m p) / 2^32``, ``m = (a b mod 2^32) p^-1 mod 2^32``, in
  ``(-p, p)``, one conditional addition. The factor ``2^-32`` of the summed products is undone exactly by the final
  scale ``2^32 / 8192 mod p`` (a Shoup product);
* the input reduction ``hs mod p`` (``hs < q1 < 2^48``) as ``(hs >> 24) 2^24 + (hs mod 2^24)`` with both halves
  below ``2p`` (``p > 2^23``) and a Shoup product by ``2^24 mod p``.

Every value is a canonical residue in ``[0, p)``, so the output equals the torch path and :class:`pks.Ntt` bit for
bit (a linear map mod ``p`` has one exact result; ``tests/service/test_pks_ntt.py`` checks it on random and edge-case
inputs in Triton's interpreter, ``TRITON_INTERPRET=1``, on CPU tensors; the provider self-tests it against the torch
path before first use, :func:`pgw.service.pks_gpu.ntt_mode`).
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

from pgw.hemask import pks
from pgw.hemask.pks import NR, PACK, RANK, D

__all__ = ["Tables", "forward", "mac_inverse", "tables"]

LOG_NR = NR.bit_length() - 1  # 13
_M32 = (1 << 32) - 1


@triton.jit
def _shoup(a, w, ws, p):
    """``a w mod p`` for uint32 ``a`` (any value), ``w < p < 2^31``, ``ws = floor(w 2^32 / p)``."""
    q = tl.umulhi(a, ws)
    r = a * w - q * p  # wraps mod 2^32; the true value is in [0, 2p)
    return tl.where(r >= p, r - p, r)


@triton.jit
def _redc(a, b, p, pinv):
    """``a b 2^-32 mod p`` in ``[0, p)`` for uint32 ``a, b < p < 2^30`` (``pinv = p^-1 mod 2^32``)."""
    lo = a * b
    hi = tl.umulhi(a, b)
    m = lo * pinv
    mh = tl.umulhi(m, p)  # a b - m p = (hi - mh) 2^32 exactly (the low words cancel)
    t = hi.to(tl.int32) - mh.to(tl.int32)
    t = tl.where(t < 0, t + p.to(tl.int32), t)
    return t.to(tl.uint32)


@triton.jit
def _fwd_stage(x, Z, ZS, p, S: tl.constexpr, N: tl.constexpr):
    """Cooley-Tukey stage ``S`` of :meth:`pks.Ntt.fwd` (``2^S`` blocks of length ``N / 2^(S+1)``) on one polynomial
    ``x`` (``(N,)`` uint32 residues): ``a, b <- a + w b, a - w b`` with the block's twiddle ``z[2^S + block]``."""
    NB: tl.constexpr = 1 << S
    L: tl.constexpr = N >> (S + 1)
    u, v = tl.split(tl.permute(tl.reshape(x, (NB, 2, L)), (0, 2, 1)))
    tw = NB + tl.arange(0, NB)
    t = _shoup(v, tl.load(Z + tw)[:, None], tl.load(ZS + tw)[:, None], p)
    a = u + t
    a = tl.where(a >= p, a - p, a)
    b = u + p - t
    b = tl.where(b >= p, b - p, b)
    return tl.reshape(tl.permute(tl.join(a, b), (0, 2, 1)), (N,))


@triton.jit
def _inv_stage(x, ZI, ZIS, p, S: tl.constexpr, N: tl.constexpr):
    """Gentleman-Sande stage ``S`` of :meth:`pks.Ntt.inv` (blocks of length ``2^S``):
    ``a, b <- a + b, (a - b) w`` with the block's twiddle ``zi[N / 2^(S+1) + block]``."""
    L: tl.constexpr = 1 << S
    NB: tl.constexpr = N >> (S + 1)
    u, v = tl.split(tl.permute(tl.reshape(x, (NB, 2, L)), (0, 2, 1)))
    tw = NB + tl.arange(0, NB)
    a = u + v
    a = tl.where(a >= p, a - p, a)
    d = u + p - v
    d = tl.where(d >= p, d - p, d)
    b = _shoup(d, tl.load(ZI + tw)[:, None], tl.load(ZIS + tw)[:, None], p)
    return tl.reshape(tl.permute(tl.join(a, b), (0, 2, 1)), (N,))


@triton.jit
def _fwd_stages(x, Z, ZS, p, N: tl.constexpr, LOGN: tl.constexpr):
    """All stages of :meth:`pks.Ntt.fwd` (bit-reversed output)."""
    for s in tl.static_range(LOGN):
        x = _fwd_stage(x, Z, ZS, p, s, N)
    return x


@triton.jit
def _inv_stages(x, ZI, ZIS, p, N: tl.constexpr, LOGN: tl.constexpr):
    """All stages of :meth:`pks.Ntt.inv` without the final ``1/N`` (bit-reversed input)."""
    for s in tl.static_range(LOGN):
        x = _inv_stage(x, ZI, ZIS, p, s, N)
    return x


@triton.jit
def _fwd_kernel(HS, AH, Z, ZS, p, half, qp, c24, c24s, N: tl.constexpr, LOGN: tl.constexpr,
                PK: tl.constexpr, RK: tl.constexpr, DD: tl.constexpr):
    """Program ``(b, l)``: ``AH[b, l] = fwd(A~[b, l] mod p)`` with ``A~`` gathered from the hint rows ``HS``
    (``(32 nb, 16 256)`` int64 in ``[0, q1)``; ``half = (q1 + 1) / 2``, ``qp = q1 mod p``, ``c24 = 2^24 mod p``)."""
    pid = tl.program_id(0)
    b = pid // RK
    li = pid % RK
    i = tl.arange(0, N)
    off = (b * PK + i % PK).to(tl.int64) * (RK * DD) + li * DD + i // PK
    h = tl.load(HS + off)
    hi = (h >> 24).to(tl.uint32)  # < 2^24 < 2p
    lo = (h & 0xFFFFFF).to(tl.uint32)  # < 2^24 < 2p
    pu = p.to(tl.uint32)
    hi = tl.where(hi >= pu, hi - pu, hi)
    lo = tl.where(lo >= pu, lo - pu, lo)
    r = _shoup(hi, c24.to(tl.uint32), c24s.to(tl.uint32), pu) + lo
    r = tl.where(r >= pu, r - pu, r)
    # centred A~ = h - q1 for h >= (q1 + 1) / 2: residue r - (q1 mod p)
    qu = qp.to(tl.uint32)
    r = tl.where(h >= half, tl.where(r >= qu, r - qu, r + pu - qu), r)
    x = _fwd_stages(r, Z, ZS, pu, N, LOGN)
    tl.store(AH + pid.to(tl.int64) * N + i, x)


@triton.jit
def _mac_inv_kernel(AH, OPS, OUT, ZI, ZIS, p, pinv, scale, scales, G, N: tl.constexpr, LOGN: tl.constexpr,
                    RK: tl.constexpr):
    """Program ``(b, g)``: ``OUT[g, b] = inv(sum_l AH[b, l] OPS[g, l])`` (``OPS``: ``(G, 16, N)`` uint32 key
    residues of this prime; ``scale = 2^32 / N mod p`` undoes the Montgomery factor and applies ``1/N``)."""
    g = tl.program_id(0)  # groups vary fastest: the programs of one reply ciphertext share its AH rows in L2
    b = tl.program_id(1)
    i = tl.arange(0, N)
    pu = p.to(tl.uint32)
    acc = tl.zeros((N,), dtype=tl.uint32)
    for li in range(RK):  # a loop, not unrolled: one key product in flight per value (registers)
        a = tl.load(AH + (b.to(tl.int64) * RK + li) * N + i)
        k = tl.load(OPS + (g * RK + li) * N + i)
        acc = acc + _redc(a, k, pu, pinv.to(tl.uint32))
        acc = tl.where(acc >= pu, acc - pu, acc)
    x = _inv_stages(acc, ZI, ZIS, pu, N, LOGN)
    x = _shoup(x, scale.to(tl.uint32), scales.to(tl.uint32), pu)
    tl.store(OUT + (g.to(tl.int64) * tl.num_programs(1) + b) * N + i, x)


def _shoup_of(w: np.ndarray | int, p: int) -> np.ndarray:
    return np.array((np.asarray(w, dtype=object) << 32) // p, dtype=np.uint64)


class Tables:
    """Per prime and device: the twiddles of :class:`pks.Ntt` and their Shoup companions (uint32 tensors) and the
    scalar constants of the two kernels."""

    def __init__(self, p: int, dev: Any):
        if not (1 << 23) < p < (1 << 30):
            raise ValueError(f"fused NTT needs 2^23 < p < 2^30, got {p}")
        nt = pks.Ntt.of(p, NR)
        self.p = p

        def u32(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.uint64).astype(np.uint32)).to(dev)

        self.z, self.zs = u32(nt.z), u32(_shoup_of(nt.z, p))
        self.zi, self.zis = u32(nt.zi), u32(_shoup_of(nt.zi, p))
        q1 = pks.Q1[0] * pks.Q1[1]
        self.half, self.qp = (q1 + 1) // 2, q1 % p
        self.c24 = (1 << 24) % p
        self.c24s = int(_shoup_of(self.c24, p))
        self.pinv = pow(p, -1, 1 << 32)
        self.scale = (int(nt.ninv) << 32) % p
        self.scales = int(_shoup_of(self.scale, p))


@cache
def _tables_cached(p: int, dev: str) -> Tables:
    return Tables(p, torch.device(dev))


def tables(p: int, dev: Any) -> Tables:
    return _tables_cached(p, str(torch.device(dev)))


def _warps() -> int:
    return int(os.environ.get("PGW_SERVICE_PKS_NTT_WARPS", "8"))


def forward(hs_t: torch.Tensor, p: int) -> torch.Tensor:
    """``(nb, 16, 8192)`` uint32: ``fwd(A~ mod p)`` of the reply ciphertexts of ``hs_t`` (``(32 nb, 4096)`` int64
    switched hint rows in ``[0, q1)``), equal to ``pks.Ntt.of(p).fwd(A~ % p)``."""
    m = int(hs_t.shape[0])
    if m % PACK or hs_t.shape[1] != RANK * D or hs_t.dtype != torch.int64:
        raise ValueError("fused NTT: bad hint rows")
    nb = m // PACK
    hs = hs_t.contiguous()
    t = tables(p, hs.device)
    out = torch.empty((nb, RANK, NR), dtype=torch.uint32, device=hs.device)
    _fwd_kernel[(nb * RANK,)](hs, out, t.z, t.zs, p, t.half, t.qp, t.c24, t.c24s, N=NR, LOGN=LOG_NR, PK=PACK,
                              RK=RANK, DD=D, num_warps=_warps())
    return out


def mac_inverse(ah: torch.Tensor, ops_p: torch.Tensor, p: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """``(G, nb, 8192)`` uint32: ``inv(sum_l ah[:, l] ops_p[g, l] mod p)`` for ``ah`` from :func:`forward` and the
    reply keys of this prime ``ops_p`` (``(G, 16, 8192)`` uint32 in ``[0, p)``), written to ``out`` if given."""
    nb, groups = int(ah.shape[0]), int(ops_p.shape[0])
    if ah.shape[1:] != (RANK, NR) or ops_p.shape[1:] != (RANK, NR) or ah.dtype != torch.uint32 \
            or ops_p.dtype != torch.uint32:
        raise ValueError("fused NTT: bad operands")
    t = tables(p, ah.device)
    if out is None:
        out = torch.empty((groups, nb, NR), dtype=torch.uint32, device=ah.device)
    elif out.shape != (groups, nb, NR) or out.dtype != torch.uint32 or not out.is_contiguous():
        raise ValueError("fused NTT: bad output")
    _mac_inv_kernel[(groups, nb)](ah.contiguous(), ops_p.contiguous(), out, t.zi, t.zis, p, t.pinv, t.scale,
                                  t.scales, groups, N=NR, LOGN=LOG_NR, RK=RANK, num_warps=_warps())
    return out


# ---- the key switch's tail: body, ModDown, CRT and the reply rounding in one pass -------------------------------
#
# The same int64 arithmetic as the torch tail of :func:`pgw.service.pks_gpu._keyswitch_t` (every operand
# non-negative before a ``%``, every product below 2^63), fused: per reply value it reads the four residues and one
# value of ``y`` and writes the reply value (the torch tail ran ~60 int64 operators over the whole reply).


@triton.jit
def _round_q1(y, Q1: tl.constexpr):
    """``floor((y q1 + 2^55) / 2^56) mod q1`` for ``0 <= y < 2^56`` (14-bit digits, as ``_round_mul_shift``)."""
    m14 = (1 << 14) - 1
    s = (y & m14) * Q1
    s = (s >> 14) + ((y >> 14) & m14) * Q1
    s = (s >> 14) + ((y >> 28) & m14) * Q1
    s = (s >> 14) + ((y >> 42) & m14) * Q1 + (1 << 13)
    s = s >> 14
    return tl.where(s >= Q1, s - Q1, s)


@triton.jit
def _tail_kernel(RES, Y, OUT, NBLK, G, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                 PM: tl.constexpr, Q1: tl.constexpr, YB: tl.constexpr, INV23: tl.constexpr, PMINV0: tl.constexpr,
                 PMINV1: tl.constexpr, INV01: tl.constexpr, N: tl.constexpr, PK: tl.constexpr, DD: tl.constexpr,
                 BLOCK: tl.constexpr):
    """Program ``(i block, b, g)``: ``OUT[g, b, i]`` from ``RES[k, g, b, i]`` (the inverse transforms, uint32 per
    prime of ``QP``) and ``body[g, b, 32 u + j] = round_q1(Y[32 b + j, 256 g + u])``."""
    pid = tl.program_id(0)
    b = tl.program_id(1)
    g = tl.program_id(2)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    yoff = (b * PK + i % PK).to(tl.int64) * (G * DD) + g * DD + i // PK
    body = _round_q1(tl.load(Y + yoff), Q1)
    plane = (g.to(tl.int64) * NBLK + b) * N + i
    step = G.to(tl.int64) * NBLK * N
    r0 = tl.load(RES + plane).to(tl.int64)
    r1 = tl.load(RES + step + plane).to(tl.int64)
    r2 = tl.load(RES + 2 * step + plane).to(tl.int64)
    r3 = tl.load(RES + 3 * step + plane).to(tl.int64)
    # res_k = (body pm - acc_k) mod p_k
    r0 = (body % P0 * (PM % P0) + P0 - r0) % P0
    r1 = (body % P1 * (PM % P1) + P1 - r1) % P1
    r2 = (body % P2 * (PM % P2) + P2 - r2) % P2
    r3 = (body % P3 * (PM % P3) + P3 - r3) % P3
    # ModDown: cp = CRT(r2, r3) mod P (centred), o_k = (r_k - cp) / P mod p_k
    k = (r3 + P3 - r2 % P3) % P3 * INV23 % P3
    cp = r2 + P2 * k  # [0, P)
    neg = cp >= (PM + 1) // 2
    c0 = cp % P0 + tl.where(neg, P0 - PM % P0, 0)
    c1 = cp % P1 + tl.where(neg, P1 - PM % P1, 0)
    o0 = (r0 + 2 * P0 - c0) % P0 * PMINV0 % P0
    o1 = (r1 + 2 * P1 - c1) % P1 * PMINV1 % P1
    # CRT to [0, q1), then floor((c 2^YB + floor(q1/2)) / q1) mod 2^YB by 12-bit long division
    c = o0 + P0 * ((o1 + P1 - o0 % P1) % P1 * INV01 % P1)
    quo = tl.zeros((BLOCK,), dtype=tl.int64)
    rem = c
    for t in tl.static_range(YB // 12):
        a = rem << 12
        if t == YB // 12 - 1:
            a = a + Q1 // 2
        quo = (quo << 12) + a // Q1
        rem = a % Q1
    tl.store(OUT + plane, quo & ((1 << YB) - 1))


_TAIL_BLOCK = 1024


def keyswitch(hs_t: torch.Tensor, ops_u: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """The key switch of :func:`pgw.service.pks_gpu._keyswitch_t` with the fused kernels: ``hs_t`` ``(R, 4096)``
    int64 in ``[0, q1)``, ``ops_u`` ``(4, G, 16, 8192)`` uint32 (:func:`ops_u32`), ``y_t`` ``(R, 256 G)`` int64 below
    ``2^56``; returns ``(G, R/32, 8192)`` int64, equal bit for bit."""
    from pgw.hemask.pks import PKSParams

    par = PKSParams()
    m = int(hs_t.shape[0])
    nb, groups = m // PACK, int(ops_u.shape[1])
    if par.yb % 12 or y_t.shape != (m, D * groups) or y_t.dtype != torch.int64 or ops_u.shape[0] != len(pks.QP):
        raise ValueError("fused key switch: bad shapes")
    dev = hs_t.device
    res = torch.empty((len(pks.QP), groups, nb, NR), dtype=torch.uint32, device=dev)
    for k, p in enumerate(pks.QP):
        ah = forward(hs_t, p)
        mac_inverse(ah, ops_u[k], p, out=res[k])
        del ah
    out = torch.empty((groups, nb, NR), dtype=torch.int64, device=dev)
    p0, p1, p2, p3 = pks.QP
    pm = par.pmod
    _tail_kernel[(NR // _TAIL_BLOCK, nb, groups)](
        res, y_t.contiguous(), out, nb, groups, P0=p0, P1=p1, P2=p2, P3=p3, PM=pm, Q1=par.q1, YB=par.yb,
        INV23=pow(p2, -1, p3), PMINV0=pow(pm, -1, p0), PMINV1=pow(pm, -1, p1), INV01=pow(p0, -1, p1), N=NR, PK=PACK,
        DD=D, BLOCK=_TAIL_BLOCK, num_warps=4)
    return out


def ops_u32(ops_t: torch.Tensor) -> torch.Tensor:
    """``(4, G, 16, 8192)`` uint32 per-prime key residues from the ``(G, 4, 16, 8192)`` int64 reply keys."""
    return ops_t.transpose(0, 1).to(torch.int32).view(torch.uint32).contiguous()
