"""Triton kernel bodies of :mod:`pgw.core.ring64` (the fused2 pair op; imported only when first used).

Kept apart (as :mod:`pgw.core._triton_gemv_kernel`) so that importing :mod:`pgw.core.ring64` never needs
triton and the interpreter sees ``triton.language`` among the kernels' module globals.

Every operation on ring values is a wrapping **unsigned 64-bit** multiply or add (LLVM ``mul``/``add`` on
``i64``, PTX ``mul.lo.u64``/``add.u64``, numpy ``uint64`` in the interpreter): ``Z_{2^K}`` (``K <= 64``) is a
quotient of ``Z_{2^64}``, so any reduction order gives the same bits and a final mask gives the residue
mod ``2^K``. No float anywhere.
"""
# mypy: ignore-errors

import triton
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


@triton.jit
def _store_planes(hp_ptr, y, c0, col, stride, ok):
    """Store ``y`` (uint64) as 8 balanced signed base-256 digits ``d_i in [-128, 127]`` at
    ``hp[(c0 + i) * stride + col]`` (int8): ``y == sum_i d_i 256^i (mod 2^64)`` exactly."""
    v = y
    for i in tl.static_range(8):
        d = ((v & 0xFF).to(tl.int32) ^ 0x80) - 0x80
        tl.store(hp_ptr + (c0 + i) * stride + col, d.to(tl.int8), mask=ok)
        v = (v - d.to(tl.int64).to(tl.uint64, bitcast=True)) >> 8  # exact: v - d is a multiple of 256


def epilogue_kernel(
    p_ptr, S, numel, M,
    r_ptr, hmul_ptr, h_ptr, out_ptr, hp_ptr,
    HAS_R: tl.constexpr, HAS_HMUL: tl.constexpr, STORE_H: tl.constexpr, BPE: tl.constexpr, BLOCK: tl.constexpr,
    STORE_PLANES: tl.constexpr = False,
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
    if STORE_PLANES:  # the next stage's input as 8 balanced int8 digit planes, row c = row * 8 + i
        _store_planes(hp_ptr, y, (o64 // M) * 8, o64 % M, M, ok)
    base = out_ptr + o64 * BPE
    for b in tl.static_range(BPE):
        tl.store(base + b, ((y >> (8 * b)) & 0xFF).to(tl.uint8), mask=ok)


def unpack_planes_kernel(src_ptr, planes_ptr, r_ptr, nx, n, total, BPE: tl.constexpr, BLOCK: tl.constexpr):
    """Fused unpack of a lane body for the ``planes`` variant.

    Elements ``e < nx`` (the ``(rows, n)`` input ``x``, row-major) are written as 8 balanced int8 digit planes,
    ``planes[(row * 8 + i) * n + col]``; elements ``nx <= e < total`` (the residual ``R``) as int64 at
    ``r[e - nx]``. Packed elements are ``BPE`` little-endian bytes.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    o64 = offs.to(tl.int64)
    base = src_ptr + o64 * BPE
    y = tl.zeros((BLOCK,), dtype=tl.uint64)
    for b in tl.static_range(BPE):
        byte = tl.load(base + b, mask=ok, other=0).to(tl.uint64)
        y = y | (byte << (8 * b))
    isx = o64 < nx
    ro = tl.where(isx, 0, o64 - nx)
    tl.store(r_ptr + ro, y.to(tl.int64, bitcast=True), mask=ok & (o64 >= nx))
    xo = tl.where(isx, o64, 0)
    _store_planes(planes_ptr, y, (xo // n) * 8, xo % n, n, ok & isx)


def gemm_planes_kernel(
    w_ptr, d_ptr, p_ptr, r_ptr, hmul_ptr, hp_ptr, out_ptr,
    B, N, M, n_per_split,
    stride_w, stride_d, stride_ps,
    RP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    FUSED: tl.constexpr, HAS_R: tl.constexpr, HAS_HMUL: tl.constexpr, STORE_PLANES: tl.constexpr,
    BPE: tl.constexpr,
):
    """Tensor-core ring GEMV of the ``planes`` variant: ``y[r, j] = sum_k W[j, k] x[r, k] (mod 2^64)``.

    ``x`` arrives as 8 balanced int8 digit planes per row (``d[(r * 8 + i), k]``, ``x = sum_i d_i 256^i``), so the
    whole product is ONE int8 ``tl.dot`` per tile, ``W (BLOCK_M, BLOCK_K) @ D^T (BLOCK_K, 8 RP)``, accumulated in
    int32 over the program's whole split: ``|sum| <= 128 * 128 * n_per_split < 2^31`` (the host keeps
    ``n_per_split <= 2^17 - 1``), so every accumulator is the exact integer ``sum_k W[j, k] d_i[r, k]``. The 8
    columns of a row are recombined once, ``sum_i acc_i << 8 i`` in wrapping uint64. Weight bytes are read once
    (the memory-bound part); digit planes are 8 RP bytes per input element (L2-resident).

    ``FUSED`` (one split): ``* hmul``, ``+ R``, ``mod 2^K`` here, the packed reply bytes written, and (``STORE_PLANES``)
    the result as the next stage's digit planes. Else the uint64 partial goes to ``p[split, r, j]``.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    NC: tl.constexpr = 8 * RP
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, NC)
    offs_k0 = tl.arange(0, BLOCK_K)
    m_ok = offs_m < M
    k_lo = pid_k * n_per_split
    k_hi = k_lo + n_per_split
    k_hi = tl.where(k_hi < N, k_hi, N)
    w_rows = w_ptr + offs_m.to(tl.int64)[:, None] * stride_w
    d_rows = d_ptr + offs_c.to(tl.int64)[:, None] * stride_d
    acc = tl.zeros((BLOCK_M, NC), dtype=tl.int32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + offs_k0
        k_ok = offs_k < k_hi
        w = tl.load(w_rows + offs_k[None, :], mask=m_ok[:, None] & k_ok[None, :], other=0)
        d = tl.load(d_rows + offs_k[None, :], mask=k_ok[None, :], other=0)
        acc = tl.dot(w, tl.trans(d), acc, out_dtype=tl.int32)  # exact int32 (bound above)
    # recombine per row without a reshape of the MMA result: column c = row * 8 + i carries digit i of the row
    a64 = acc.to(tl.int64).to(tl.uint64, bitcast=True) << ((offs_c % 8) * 8).to(tl.uint64)[None, :]
    zero = tl.zeros_like(a64)
    m64 = offs_m.to(tl.int64)
    if FUSED and HAS_HMUL:
        hm = tl.load(hmul_ptr + offs_m, mask=m_ok, other=0).to(tl.uint64, bitcast=True)
    for r in tl.static_range(RP):
        y = tl.sum(tl.where((offs_c // 8 == r)[None, :], a64, zero), axis=1)  # (BLOCK_M,) wrapping uint64
        ok = m_ok & (offs_m * 0 + r < B)
        e = r * M + m64  # element index row * M + j
        if FUSED:
            if HAS_HMUL:
                y = y * hm
            if HAS_R:
                y += tl.load(r_ptr + e, mask=ok, other=0).to(tl.uint64, bitcast=True)
            if BPE < 8:
                y = (y << (64 - 8 * BPE)) >> (64 - 8 * BPE)
            if STORE_PLANES:
                _store_planes(hp_ptr, y, r * 8, m64, M, ok)
            base = out_ptr + e * BPE
            for b in tl.static_range(BPE):
                tl.store(base + b, ((y >> (8 * b)) & 0xFF).to(tl.uint8), mask=ok)
        else:
            tl.store(p_ptr + pid_k.to(tl.int64) * stride_ps + e, y.to(tl.int64, bitcast=True), mask=ok)
