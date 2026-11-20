"""Triton kernel body of :mod:`pgw.factory.gpu_prng` (imported only when the kernel is first used).

Kept in its own module (like :mod:`pgw.core._triton_gemv_kernel`) because the Triton interpreter needs
``triton.language`` among the kernel's module globals, and so that importing :mod:`pgw.factory.gpu_prng`
never needs triton. The kernel is wrapped with ``triton.jit`` by :func:`pgw.factory.gpu_prng._kernel`; its
store helper :func:`_emit` is jitted when this module is first imported (``TRITON_INTERPRET`` must be chosen
before that, as for every Triton kernel).

One program lane computes one 64-byte ChaCha20 block (RFC 8439 block function, 20 rounds) of the stream
``ChaCha20(key, counter = block0 + b, nonce = 0^12)``, i.e. eight little-endian 64-bit words
``w_j = x[2j] | x[2j+1] << 32``. Word ``j`` of relative block ``b`` is ring element
``e = 8 b + j - s0`` of the requested ``(rows, cols)`` slice (``s0`` = 64-bit words to skip in the first
block when the slice does not start on a block boundary), reduced mod ``2^K``. Outputs (either or both):

* ``r``: int64 ring elements, row-major ``(rows, cols)`` (``e`` is the flat index);
* ``d``: ``LIMBS = ceil(K/8)`` int8 planes of balanced base-256 digits (exactly
  :func:`pgw.core.ring.balanced_digits` with ``bits = 8``), plane ``i`` at ``d + i * plane_stride``, element
  ``(row, col)`` at ``row * row_stride + col``. Positions outside the slice are not written (callers
  zero-fill padding).

All ChaCha arithmetic is unsigned 32-bit (wrapping add, xor, rotate by shifts); no float anywhere.
"""
# mypy: ignore-errors

import triton
import triton.language as tl


def chacha_ring_kernel(
    key_ptr, r_ptr, d_ptr,
    block0, s0, total, cols, nblk, row_stride, plane_stride,
    K: tl.constexpr, LIMBS: tl.constexpr, BLOCK: tl.constexpr, WRITE_R: tl.constexpr, WRITE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)  # relative ChaCha block index
    b_ok = b < nblk
    zero = tl.zeros([BLOCK], dtype=tl.uint32)
    # key: 8 little-endian words (int32 bit patterns)
    k0 = tl.load(key_ptr + 0).to(tl.uint32, bitcast=True) + zero
    k1 = tl.load(key_ptr + 1).to(tl.uint32, bitcast=True) + zero
    k2 = tl.load(key_ptr + 2).to(tl.uint32, bitcast=True) + zero
    k3 = tl.load(key_ptr + 3).to(tl.uint32, bitcast=True) + zero
    k4 = tl.load(key_ptr + 4).to(tl.uint32, bitcast=True) + zero
    k5 = tl.load(key_ptr + 5).to(tl.uint32, bitcast=True) + zero
    k6 = tl.load(key_ptr + 6).to(tl.uint32, bitcast=True) + zero
    k7 = tl.load(key_ptr + 7).to(tl.uint32, bitcast=True) + zero
    # block counter (host guarantees block0 + nblk <= 2^32: no carry into the nonce words)
    ctr = ((block0 + b) & 0xFFFFFFFF).to(tl.uint32)
    c0 = zero + 0x61707865
    c1 = zero + 0x3320646E
    c2 = zero + 0x79622D32
    c3 = zero + 0x6B206574
    x0, x1, x2, x3 = c0, c1, c2, c3
    x4, x5, x6, x7 = k0, k1, k2, k3
    x8, x9, x10, x11 = k4, k5, k6, k7
    x12, x13, x14, x15 = ctr, zero, zero, zero
    for _ in tl.static_range(10):
        # column round: (0,4,8,12) (1,5,9,13) (2,6,10,14) (3,7,11,15)
        x0 += x4; x12 ^= x0; x12 = (x12 << 16) | (x12 >> 16)  # noqa: E702
        x8 += x12; x4 ^= x8; x4 = (x4 << 12) | (x4 >> 20)  # noqa: E702
        x0 += x4; x12 ^= x0; x12 = (x12 << 8) | (x12 >> 24)  # noqa: E702
        x8 += x12; x4 ^= x8; x4 = (x4 << 7) | (x4 >> 25)  # noqa: E702
        x1 += x5; x13 ^= x1; x13 = (x13 << 16) | (x13 >> 16)  # noqa: E702
        x9 += x13; x5 ^= x9; x5 = (x5 << 12) | (x5 >> 20)  # noqa: E702
        x1 += x5; x13 ^= x1; x13 = (x13 << 8) | (x13 >> 24)  # noqa: E702
        x9 += x13; x5 ^= x9; x5 = (x5 << 7) | (x5 >> 25)  # noqa: E702
        x2 += x6; x14 ^= x2; x14 = (x14 << 16) | (x14 >> 16)  # noqa: E702
        x10 += x14; x6 ^= x10; x6 = (x6 << 12) | (x6 >> 20)  # noqa: E702
        x2 += x6; x14 ^= x2; x14 = (x14 << 8) | (x14 >> 24)  # noqa: E702
        x10 += x14; x6 ^= x10; x6 = (x6 << 7) | (x6 >> 25)  # noqa: E702
        x3 += x7; x15 ^= x3; x15 = (x15 << 16) | (x15 >> 16)  # noqa: E702
        x11 += x15; x7 ^= x11; x7 = (x7 << 12) | (x7 >> 20)  # noqa: E702
        x3 += x7; x15 ^= x3; x15 = (x15 << 8) | (x15 >> 24)  # noqa: E702
        x11 += x15; x7 ^= x11; x7 = (x7 << 7) | (x7 >> 25)  # noqa: E702
        # diagonal round: (0,5,10,15) (1,6,11,12) (2,7,8,13) (3,4,9,14)
        x0 += x5; x15 ^= x0; x15 = (x15 << 16) | (x15 >> 16)  # noqa: E702
        x10 += x15; x5 ^= x10; x5 = (x5 << 12) | (x5 >> 20)  # noqa: E702
        x0 += x5; x15 ^= x0; x15 = (x15 << 8) | (x15 >> 24)  # noqa: E702
        x10 += x15; x5 ^= x10; x5 = (x5 << 7) | (x5 >> 25)  # noqa: E702
        x1 += x6; x12 ^= x1; x12 = (x12 << 16) | (x12 >> 16)  # noqa: E702
        x11 += x12; x6 ^= x11; x6 = (x6 << 12) | (x6 >> 20)  # noqa: E702
        x1 += x6; x12 ^= x1; x12 = (x12 << 8) | (x12 >> 24)  # noqa: E702
        x11 += x12; x6 ^= x11; x6 = (x6 << 7) | (x6 >> 25)  # noqa: E702
        x2 += x7; x13 ^= x2; x13 = (x13 << 16) | (x13 >> 16)  # noqa: E702
        x8 += x13; x7 ^= x8; x7 = (x7 << 12) | (x7 >> 20)  # noqa: E702
        x2 += x7; x13 ^= x2; x13 = (x13 << 8) | (x13 >> 24)  # noqa: E702
        x8 += x13; x7 ^= x8; x7 = (x7 << 7) | (x7 >> 25)  # noqa: E702
        x3 += x4; x14 ^= x3; x14 = (x14 << 16) | (x14 >> 16)  # noqa: E702
        x9 += x14; x4 ^= x9; x4 = (x4 << 12) | (x4 >> 20)  # noqa: E702
        x3 += x4; x14 ^= x3; x14 = (x14 << 8) | (x14 >> 24)  # noqa: E702
        x9 += x14; x4 ^= x9; x4 = (x4 << 7) | (x4 >> 25)  # noqa: E702
    x0 += c0
    x1 += c1
    x2 += c2
    x3 += c3
    x4 += k0
    x5 += k1
    x6 += k2
    x7 += k3
    x8 += k4
    x9 += k5
    x10 += k6
    x11 += k7
    x12 += ctr
    # x13..x15: initial words are zero
    e0 = b * 8 - s0  # flat element index of word 0 of each block
    _emit(x0, x1, e0 + 0, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride, K, LIMBS, WRITE_R, WRITE_D)
    _emit(x2, x3, e0 + 1, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride, K, LIMBS, WRITE_R, WRITE_D)
    _emit(x4, x5, e0 + 2, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride, K, LIMBS, WRITE_R, WRITE_D)
    _emit(x6, x7, e0 + 3, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride, K, LIMBS, WRITE_R, WRITE_D)
    _emit(x8, x9, e0 + 4, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride, K, LIMBS, WRITE_R, WRITE_D)
    _emit(x10, x11, e0 + 5, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride, K, LIMBS, WRITE_R, WRITE_D)
    _emit(x12, x13, e0 + 6, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride, K, LIMBS, WRITE_R, WRITE_D)
    _emit(x14, x15, e0 + 7, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride, K, LIMBS, WRITE_R, WRITE_D)


@triton.jit
def _emit(
    lo, hi, e, b_ok, total, cols, r_ptr, d_ptr, row_stride, plane_stride,
    K: tl.constexpr, LIMBS: tl.constexpr, WRITE_R: tl.constexpr, WRITE_D: tl.constexpr,
):
    """Store ring element ``(hi << 32 | lo) mod 2^K`` at flat index ``e`` (r and/or digit planes)."""
    ok = b_ok & (e >= 0) & (e < total)
    if K > 32:  # K <= 63 (host check): the top bits of hi are dropped, the int64 stays non-negative
        v = (lo.to(tl.int64) & 0xFFFFFFFF) | ((hi & ((1 << (K - 32)) - 1)).to(tl.int64) << 32)
    elif K == 32:
        v = lo.to(tl.int64) & 0xFFFFFFFF
    else:
        v = (lo & ((1 << K) - 1)).to(tl.int64)
    if WRITE_R:
        tl.store(r_ptr + e, v, mask=ok)
    if WRITE_D:
        row = e // cols
        col = e - row * cols
        base = d_ptr + row * row_stride + col
        cur = v
        for i in tl.static_range(LIMBS):
            dg = cur & 255
            dg = tl.where(dg >= 128, dg - 256, dg)
            tl.store(base + i * plane_stride, dg.to(tl.int8), mask=ok)
            cur = (cur - dg) >> 8
