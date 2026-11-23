"""Triton kernel bodies of :mod:`pgw.core.ring64` (the fused2 pair op; imported only when first used).

Kept apart (as :mod:`pgw.core._triton_gemv_kernel`) so that importing :mod:`pgw.core.ring64` never needs
triton and the interpreter sees ``triton.language`` among the kernels' module globals.

Every operation on ring values is a wrapping **unsigned 64-bit** multiply or add (LLVM ``mul``/``add`` on
``i64``, PTX ``mul.lo.u64``/``add.u64``, numpy ``uint64`` in the interpreter): ``Z_{2^K}`` (``K <= 64``) is a
quotient of ``Z_{2^64}``, so any reduction order gives the same bits and a final mask gives the residue
mod ``2^K``. No float anywhere.
"""
# mypy: ignore-errors

import triton.language as tl


def unpack_kernel(src_ptr, dst_ptr, numel, BPE: tl.constexpr, BLOCK: tl.constexpr):
    """``dst[i] = sum_b src[i * BPE + b] << 8b`` (little-endian packed ring elements -> int64)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    ok = offs < numel
    base = src_ptr + offs.to(tl.int64) * BPE
    y = tl.zeros((BLOCK,), dtype=tl.uint64)
    for b in tl.static_range(BPE):
        byte = tl.load(base + b, mask=ok, other=0).to(tl.uint64)
        y = y | (byte << (8 * b))
    tl.store(dst_ptr + offs, y.to(tl.int64, bitcast=True), mask=ok)


def gemv64_sum_kernel(
    x_ptr, w_ptr, p_ptr,
    B, N, M, n_per_split,
    stride_x, stride_w, stride_ps,
    BLOCK_R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Split-K partial ``p[s, r, j] = sum_{i in split s} x[r, i] * W[j, i]`` mod 2^64 (CUDA cores).

    ``x`` int64 ring elements (any value; read as uint64), ``W`` int8 sign-extended to 64 bits. Partial ``s``
    covers columns ``[s * n_per_split, (s + 1) * n_per_split)``; masked entries load 0.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    r_ok = offs_r < B
    m_ok = offs_m < M
    k_lo = pid_k * n_per_split
    k_hi = k_lo + n_per_split
    k_hi = tl.where(k_hi < N, k_hi, N)
    w_rows = w_ptr + offs_m.to(tl.int64)[:, None] * stride_w
    x_rows = x_ptr + offs_r.to(tl.int64)[:, None] * stride_x
    acc = tl.zeros((BLOCK_R, BLOCK_M), dtype=tl.uint64)
    for n0 in range(k_lo, k_hi, BLOCK_N):
        offs_n = n0 + offs_n0
        n_ok = offs_n < k_hi
        w = tl.load(w_rows + offs_n[None, :], mask=m_ok[:, None] & n_ok[None, :], other=0)
        x = tl.load(x_rows + offs_n[None, :], mask=r_ok[:, None] & n_ok[None, :], other=0)
        wu = w.to(tl.int64).to(tl.uint64, bitcast=True)  # sign extension: W8 mod 2^64
        xu = x.to(tl.uint64, bitcast=True)
        acc += tl.sum(xu[:, None, :] * wu[None, :, :], axis=2)  # uint64: wraps mod 2^64
    p_ptrs = p_ptr + pid_k.to(tl.int64) * stride_ps + offs_r.to(tl.int64)[:, None] * M + offs_m[None, :]
    tl.store(p_ptrs, acc.to(tl.int64, bitcast=True), mask=r_ok[:, None] & m_ok[None, :])


def gemv64_dot_kernel(
    x_ptr, w_ptr, p_ptr,
    B, N, M, n_per_split,
    stride_x, stride_w, stride_ps,
    BLOCK_R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NDIG: tl.constexpr,
):
    """Tensor-core variant: ``x`` as ``NDIG`` balanced signed base-256 digit planes, one int8 ``tl.dot`` each.

    ``x = sum_i d_i 256^i (mod 2^(8 NDIG))``, ``d_i in [-128, 127]`` (peeled with wrapping uint64 subtract and
    logical shift). Each tile's digit product starts from a zero int32 accumulator and is bounded by
    ``128 * 128 * BLOCK_N < 2^31`` (exact); it is sign-extended to 64 bits, shifted by ``8 i`` and added into
    the uint64 accumulator (wrapping). So the result is ``x W^T mod 2^(8 NDIG)`` on the low bits, which the
    epilogue masks to ``2^K`` (``K <= 8 NDIG``).
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    r_ok = offs_r < B
    m_ok = offs_m < M
    k_lo = pid_k * n_per_split
    k_hi = k_lo + n_per_split
    k_hi = tl.where(k_hi < N, k_hi, N)
    w_rows = w_ptr + offs_m.to(tl.int64)[:, None] * stride_w
    x_rows = x_ptr + offs_r.to(tl.int64)[:, None] * stride_x
    acc = tl.zeros((BLOCK_R, BLOCK_M), dtype=tl.uint64)
    for n0 in range(k_lo, k_hi, BLOCK_N):
        offs_n = n0 + offs_n0
        n_ok = offs_n < k_hi
        w = tl.load(w_rows + offs_n[None, :], mask=m_ok[:, None] & n_ok[None, :], other=0)
        wt = tl.trans(w)
        v = tl.load(x_rows + offs_n[None, :], mask=r_ok[:, None] & n_ok[None, :], other=0)
        v = v.to(tl.uint64, bitcast=True)
        for i in tl.static_range(NDIG):
            d = ((v & 0xFF).to(tl.int32) ^ 0x80) - 0x80  # balanced digit in [-128, 127]
            part = tl.dot(d.to(tl.int8), wt, out_dtype=tl.int32)  # exact: |.| <= 128 * 128 * BLOCK_N
            acc += part.to(tl.int64).to(tl.uint64, bitcast=True) << (8 * i)
            v = (v - d.to(tl.int64).to(tl.uint64, bitcast=True)) >> 8  # exact: x - d is a multiple of 256
    p_ptrs = p_ptr + pid_k.to(tl.int64) * stride_ps + offs_r.to(tl.int64)[:, None] * M + offs_m[None, :]
    tl.store(p_ptrs, acc.to(tl.int64, bitcast=True), mask=r_ok[:, None] & m_ok[None, :])


def epilogue_kernel(
    p_ptr, S, numel, M,
    r_ptr, hmul_ptr, h_ptr, out_ptr,
    HAS_R: tl.constexpr, HAS_HMUL: tl.constexpr, STORE_H: tl.constexpr, BPE: tl.constexpr, BLOCK: tl.constexpr,
):
    """``y = (r + hmul * sum_s p[s]) mod 2^K`` per element of a ``(B, M)`` output (flat ``e = row * M + j``).

    Writes ``y`` as int64 to ``h`` (``STORE_H``: the next stage's input) and as ``BPE`` little-endian bytes to
    ``out`` (the packed reply), ``K = 8 * BPE`` (reduced by a logical shift pair, no mask constant).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    ok = offs < numel
    o64 = offs.to(tl.int64)
    y = tl.zeros((BLOCK,), dtype=tl.uint64)
    for s in range(0, S):
        y += tl.load(p_ptr + s * numel + o64, mask=ok, other=0).to(tl.uint64, bitcast=True)
    if HAS_HMUL:
        hm = tl.load(hmul_ptr + (offs % M), mask=ok, other=0).to(tl.uint64, bitcast=True)
        y = y * hm
    if HAS_R:
        y += tl.load(r_ptr + o64, mask=ok, other=0).to(tl.uint64, bitcast=True)
    if BPE < 8:
        y = (y << (64 - 8 * BPE)) >> (64 - 8 * BPE)  # uint64: logical shifts keep the low K bits
    if STORE_H:
        tl.store(h_ptr + o64, y.to(tl.int64, bitcast=True), mask=ok)
    base = out_ptr + o64 * BPE
    for b in tl.static_range(BPE):
        tl.store(base + b, ((y >> (8 * b)) & 0xFF).to(tl.uint8), mask=ok)
