# mypy: disable-error-code="no-untyped-def,untyped-decorator"
"""Fused Triton kernels of the ``pks-v3`` device item (:func:`pgw.service.pks_gpu.item`) besides the NTTs.

* :func:`combine` (``PGW_SERVICE_PKS_GPU_SWITCH``): the recombination of the 7 int8-plane GEMM partials mod ``2^56``
  (:func:`pgw.service.kernels._combine_planes`, then ``& (2^56 - 1)``) and, for the hint rows, the exact switch
  ``round(h q1 / 2^56) mod q1`` (:func:`pgw.service.pks_gpu._round_mul_shift` with 14-bit digits, as
  :func:`pgw.hemask.pks.switch_hint`), in one pass: per value it reads the 7 int32 partials (28 B) and writes one
  int64 (8 B). The eager torch path ran ~41 int64 operators per column block (~628 B per value).
* :func:`ring_rows` (``PGW_SERVICE_PKS_GPU_RING``): the customer's public ring rows ``a_C`` (:func:`pgw.hemask.pks.
  public_ring_rows`) generated on the device: ChaCha20 (RFC 8439 block function, zero nonce, 32-bit block counter
  from 0) keyed by the stream key of ``(A_LABEL, linear)`` under the public ``a_seed``, 8 keystream bytes per
  element (little-endian), reduced mod ``2^56``. The host derives the 32-byte key (HKDF, as the host path) and the
  device expands it; nothing goes up but the key, and the provider learns nothing new (``a_seed`` and the stream
  are public, the host path computes the same bytes).

Arithmetic, all integer and exact:

* recombination: ``lo = sum_{d<4} p_d 2^(8d)`` and ``hi = sum_{d>=4} p_{d} 2^(8(d-4))`` are exact in int64
  (``|p_d| < 2^31``), and mod ``2^56`` the value is ``lo + (c mod 2^56) + ((hi mod 2^24) << 32)``: every term is
  below ``2^56`` in magnitude, so nothing wraps before the final ``& (2^56 - 1)`` (two's complement, i.e. mod
  ``2^56``). Equal to ``(sum_d p_d 2^(8d) + c mod 2^64) mod 2^56``, the torch path;
* switch: 14-bit digits of ``h`` times ``q1 < 2^48`` (every partial below ``2^62``), as :func:`_round_mul_shift`
  for ``shift = 56`` (the carry ``2^55`` enters digit 3), then one conditional subtraction of ``q1``;
* ChaCha20: uint32 additions (mod ``2^32``), xors and rotations, as the RFC.

``tests/service/test_pks_fused.py`` checks both against the torch path, :func:`pgw.hemask.pks.switch_hint` and
:func:`pgw.hemask.pks.public_ring_rows` in Triton's interpreter (``TRITON_INTERPRET=1``) on CPU tensors; the
provider self-tests each on its device before first use and falls back to the torch / host path for good on any
mismatch or error (:mod:`pgw.service.pks_gpu`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

from pgw.hemask.pks import NU, PKSParams

__all__ = ["chacha_key", "combine", "compile_for", "ring_rows"]

_PAR = PKSParams()
PLANES = 7
_BR, _BC = 16, 128  # rows x columns per program of the combine kernel


@triton.jit
def _round_q1(h, Q1: tl.constexpr):
    """``floor((h q1 + 2^55) / 2^56) mod q1`` for ``0 <= h < 2^56`` (14-bit digits, as ``_round_mul_shift``)."""
    m14 = (1 << 14) - 1
    s = (h & m14) * Q1
    s = (s >> 14) + ((h >> 14) & m14) * Q1
    s = (s >> 14) + ((h >> 28) & m14) * Q1
    s = (s >> 14) + ((h >> 42) & m14) * Q1 + (1 << 13)
    s = s >> 14
    return tl.where(s >= Q1, s - Q1, s)


@triton.jit
def _combine_kernel(PART, CONST, OUT, R, C, ostride, SWITCH: tl.constexpr, Q1: tl.constexpr, BR: tl.constexpr,
                    BC: tl.constexpr):
    """Program ``(row block, column block)``: ``OUT[r, c] = combine(PART[r, d C + c] for d < 7, CONST[r]) mod 2^56``
    (``PART``: ``(R, 7 C)`` int32 GEMM partials, plane-major; ``CONST``: ``(R, 1)`` int64), switched to ``q1`` when
    ``SWITCH``. ``R``, ``C`` are only compared (Triton may make an argument equal to 1 a constant)."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    r64 = rows.to(tl.int64)
    base = r64[:, None] * (7 * C) + cols[None, :]
    lo = tl.load(PART + base, mask=mask, other=0).to(tl.int64)
    lo += tl.load(PART + base + C, mask=mask, other=0).to(tl.int64) << 8
    lo += tl.load(PART + base + 2 * C, mask=mask, other=0).to(tl.int64) << 16
    lo += tl.load(PART + base + 3 * C, mask=mask, other=0).to(tl.int64) << 24
    hi = tl.load(PART + base + 4 * C, mask=mask, other=0).to(tl.int64)
    hi += tl.load(PART + base + 5 * C, mask=mask, other=0).to(tl.int64) << 8
    hi += tl.load(PART + base + 6 * C, mask=mask, other=0).to(tl.int64) << 16
    c = tl.load(CONST + r64, mask=rows < R, other=0)
    m56 = (1 << 56) - 1
    h = (lo + (c & m56)[:, None] + ((hi & 0xFFFFFF) << 32)) & m56
    if SWITCH:
        h = _round_q1(h, Q1)
    tl.store(OUT + r64[:, None] * ostride + cols[None, :], h, mask=mask)


def combine(part: torch.Tensor, const: torch.Tensor, out: torch.Tensor, switch: bool) -> torch.Tensor:
    """``out = (sum_d part[:, d c:(d+1) c] 2^(8d) + const mod 2^64) mod 2^56`` (``(R, c)`` int64, a view with unit
    column stride), switched to ``round(. q1 / 2^56) mod q1`` when ``switch``: equal bit for bit to
    ``_combine_planes(...) & (2^56 - 1)`` (then ``_round_mul_shift(., q1, 56) % q1``). ``part``: ``(R, 7 c)`` int32
    (contiguous), ``const``: ``(R, 1)`` int64."""
    r, c7 = part.shape
    c = c7 // PLANES
    if part.dtype != torch.int32 or c7 != PLANES * c or not part.is_contiguous() or const.dtype != torch.int64 \
            or out.shape != (r, c) or out.dtype != torch.int64 or out.stride(1) != 1 or const.shape[0] != r:
        raise ValueError("fused combine: bad operands")
    _combine_kernel[(triton.cdiv(r, _BR), triton.cdiv(c, _BC))](
        part, const.contiguous(), out, r, c, out.stride(0), SWITCH=bool(switch), Q1=_PAR.q1, BR=_BR, BC=_BC,
        num_warps=4)
    return out


# ---- ChaCha20 ring rows -------------------------------------------------------------------------------------


@triton.jit
def _rotl(x, r: tl.constexpr):
    return (x << r) | (x >> (32 - r))


@triton.jit
def _qr(a, b, c, d):
    a = a + b
    d = _rotl(d ^ a, 16)
    c = c + d
    b = _rotl(b ^ c, 12)
    a = a + b
    d = _rotl(d ^ a, 8)
    c = c + d
    b = _rotl(b ^ c, 7)
    return a, b, c, d


@triton.jit
def _chacha_ring_kernel(KEY, OUT, nblocks, BLK: tl.constexpr):
    """Program ``i``: ChaCha20 blocks ``i BLK .. i BLK + BLK - 1`` of the stream ``KEY`` (8 uint32 words, zero nonce),
    each 8 little-endian uint64 ring elements ``OUT[8 blk + k] = (w[2k] | w[2k+1] << 32) mod 2^56``."""
    blk = tl.program_id(0) * BLK + tl.arange(0, BLK)
    live = blk < nblocks
    z = tl.zeros((BLK,), dtype=tl.uint32)
    k0 = tl.load(KEY + 0)
    k1 = tl.load(KEY + 1)
    k2 = tl.load(KEY + 2)
    k3 = tl.load(KEY + 3)
    k4 = tl.load(KEY + 4)
    k5 = tl.load(KEY + 5)
    k6 = tl.load(KEY + 6)
    k7 = tl.load(KEY + 7)
    ctr = blk.to(tl.uint32)
    x0 = z + 0x61707865
    x1 = z + 0x3320646E
    x2 = z + 0x79622D32
    x3 = z + 0x6B206574
    x4 = z + k0
    x5 = z + k1
    x6 = z + k2
    x7 = z + k3
    x8 = z + k4
    x9 = z + k5
    x10 = z + k6
    x11 = z + k7
    x12 = ctr
    x13 = z
    x14 = z
    x15 = z
    for _ in range(10):  # 10 double rounds (a loop, not unrolled: one round's code)
        x0, x4, x8, x12 = _qr(x0, x4, x8, x12)
        x1, x5, x9, x13 = _qr(x1, x5, x9, x13)
        x2, x6, x10, x14 = _qr(x2, x6, x10, x14)
        x3, x7, x11, x15 = _qr(x3, x7, x11, x15)
        x0, x5, x10, x15 = _qr(x0, x5, x10, x15)
        x1, x6, x11, x12 = _qr(x1, x6, x11, x12)
        x2, x7, x8, x13 = _qr(x2, x7, x8, x13)
        x3, x4, x9, x14 = _qr(x3, x4, x9, x14)
    x0 += 0x61707865
    x1 += 0x3320646E
    x2 += 0x79622D32
    x3 += 0x6B206574
    x4 += k0
    x5 += k1
    x6 += k2
    x7 += k3
    x8 += k4
    x9 += k5
    x10 += k6
    x11 += k7
    x12 += ctr
    o = OUT + blk.to(tl.int64) * 8
    tl.store(o + 0, (x0.to(tl.int64) | ((x1 & 0xFFFFFF).to(tl.int64) << 32)), mask=live)
    tl.store(o + 1, (x2.to(tl.int64) | ((x3 & 0xFFFFFF).to(tl.int64) << 32)), mask=live)
    tl.store(o + 2, (x4.to(tl.int64) | ((x5 & 0xFFFFFF).to(tl.int64) << 32)), mask=live)
    tl.store(o + 3, (x6.to(tl.int64) | ((x7 & 0xFFFFFF).to(tl.int64) << 32)), mask=live)
    tl.store(o + 4, (x8.to(tl.int64) | ((x9 & 0xFFFFFF).to(tl.int64) << 32)), mask=live)
    tl.store(o + 5, (x10.to(tl.int64) | ((x11 & 0xFFFFFF).to(tl.int64) << 32)), mask=live)
    tl.store(o + 6, (x12.to(tl.int64) | ((x13 & 0xFFFFFF).to(tl.int64) << 32)), mask=live)
    tl.store(o + 7, (x14.to(tl.int64) | ((x15 & 0xFFFFFF).to(tl.int64) << 32)), mask=live)


_CHACHA_BLK = 128


def chacha_key(a_seed: bytes, linear: str) -> bytes:
    """The ChaCha20 key of the public ring-row stream of ``linear`` (:func:`pgw.hemask.pks.public_ring_rows`)."""
    from pgw.core.prng import SeedExpander
    from pgw.hemask.pks import A_LABEL

    return SeedExpander(bytes(a_seed)).stream_key(A_LABEL, linear)


def ring_rows(a_seed: bytes, n: int, linear: str, dev: Any) -> torch.Tensor:
    """``(n/16, 4096)`` int64 on ``dev``: :func:`pgw.hemask.pks.public_ring_rows` generated by the device."""
    from pgw.hemask.pks import RANK

    if n % RANK or n <= 0:
        raise ValueError(f"in_features {n} is not a positive multiple of {RANK}")
    rows = n // RANK
    nblocks = rows * NU // 8
    if nblocks > 1 << 32:
        raise ValueError("ring rows beyond the 2^32-block ChaCha20 counter range")
    key = torch.from_numpy(np.frombuffer(chacha_key(a_seed, linear), dtype="<u4").astype(np.uint32)).to(dev)
    out = torch.empty((rows, NU), dtype=torch.int64, device=dev)
    _chacha_ring_kernel[(triton.cdiv(nblocks, _CHACHA_BLK),)](key, out, nblocks, BLK=_CHACHA_BLK, num_warps=4)
    return out


def compile_for(arch: int, shared: dict[str, int] | None = None) -> dict[str, int]:
    """Compile both kernels for CUDA compute capability ``arch`` without a GPU (not under the interpreter); cubin
    bytes per kernel (see :func:`pgw.service.pks_ntt.compile_for`)."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    jobs = {
        "combine_switch": (_combine_kernel, {"PART": "*i32", "CONST": "*i64", "OUT": "*i64", "R": "i32", "C": "i32",
                                             "ostride": "i32"}, dict(SWITCH=True, Q1=_PAR.q1, BR=_BR, BC=_BC)),
        "combine": (_combine_kernel, {"PART": "*i32", "CONST": "*i64", "OUT": "*i64", "R": "i32", "C": "i32",
                                      "ostride": "i32"}, dict(SWITCH=False, Q1=_PAR.q1, BR=_BR, BC=_BC)),
        "chacha_ring": (_chacha_ring_kernel, {"KEY": "*u32", "OUT": "*i64", "nblocks": "i32"},
                        dict(BLK=_CHACHA_BLK)),
    }
    out = {}
    for name, (fn, sig, cx) in jobs.items():
        src = ASTSource(fn=fn, signature={**sig, **{k: "constexpr" for k in cx}}, constexprs=cx)
        k = triton.compile(src, target=GPUTarget("cuda", arch, 32), options={"num_warps": 4})
        out[name] = len(k.asm["cubin"])
        if shared is not None:
            shared[name] = int(k.metadata.shared)
    return out
