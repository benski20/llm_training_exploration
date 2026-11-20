"""Triton kernel body of :mod:`pgw.core.triton_gemv` (imported only when the kernel is first used).

Kept in its own module because the Triton interpreter needs ``triton.language`` among the kernel's
module globals, and so that importing :mod:`pgw.core.triton_gemv` never needs triton. The function
is decorated with ``triton.jit`` by :func:`pgw.core.triton_gemv._kernel` (per interpreter mode).
"""
# mypy: ignore-errors

import triton.language as tl


def ring_gemv_kernel(
    x_ptr, w_ptr, y_ptr,
    B, N, M, NW, n_per_split, kmask,
    stride_x, stride_w, stride_y,
    BLOCK_R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ATOMIC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    r_ok = offs_r < B
    m_ok = offs_m < M
    k_lo = pid_k * n_per_split
    k_hi = k_lo + n_per_split
    k_hi = tl.where(k_hi < NW, k_hi, NW)
    w_rows = w_ptr + offs_m.to(tl.int64)[:, None] * stride_w
    x_rows = x_ptr + offs_r.to(tl.int64)[:, None] * stride_x
    acc = tl.zeros((BLOCK_R, BLOCK_M), dtype=tl.uint32)
    for n0 in range(k_lo, k_hi, BLOCK_N):
        offs_n = n0 + offs_n0
        # weight columns in [N, NW) (padding) meet zero x entries, whatever they hold
        w = tl.load(w_rows + offs_n[None, :], mask=m_ok[:, None] & (offs_n < k_hi)[None, :], other=0)
        x = tl.load(x_rows + offs_n[None, :], mask=r_ok[:, None] & (offs_n < N)[None, :], other=0)
        wu = w.to(tl.int32).to(tl.uint32, bitcast=True)  # sign extension: W8 mod 2^32
        xu = x.to(tl.uint32, bitcast=True)
        acc += tl.sum(xu[:, None, :] * wu[None, :, :], axis=2)  # uint32: wraps mod 2^32
    # kmask: 2^K - 1 (int32 -1 = all ones for K = 32, and for split-K partial sums: masked after)
    acc = acc & kmask.to(tl.uint32, bitcast=True)
    y_ptrs = y_ptr + offs_r.to(tl.int64)[:, None] * stride_y + offs_m[None, :]
    y_ok = r_ok[:, None] & m_ok[None, :]
    if ATOMIC:
        # wrapping unsigned add (red.global.add.u32): partial sums combine mod 2^32 in any order
        tl.atomic_add(y_ptrs.to(tl.pointer_type(tl.uint32)), acc, mask=y_ok)
    else:
        tl.store(y_ptrs, acc.to(tl.int32, bitcast=True), mask=y_ok)


def ring_gemv_dot_kernel(
    x_ptr, w_ptr, y_ptr,
    B, N, M, NW, n_per_split, kmask,
    stride_x, stride_w, stride_y,
    BLOCK_R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ATOMIC: tl.constexpr,
):
    """Tensor-core variant: ``x`` as four balanced signed base-256 digits, four int8 ``tl.dot`` per tile.

    ``x = sum_i d_i 256^i (mod 2^32)`` with ``d_i`` in ``[-128, 127]`` (digits peeled with wrapping
    unsigned arithmetic); each ``|sum_n d_i W| <= 128 * 128 * n_per_split < 2^31`` (the host caps
    ``n_per_split``), so the int32 dot accumulators are exact, and the digit products are recombined
    with unsigned shifts and adds mod 2^32. Masked ``x`` entries are 0, hence all-zero digits: weight
    columns past ``N`` contribute nothing.
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
    k_hi = tl.where(k_hi < NW, k_hi, NW)
    w_rows = w_ptr + offs_m.to(tl.int64)[:, None] * stride_w
    x_rows = x_ptr + offs_r.to(tl.int64)[:, None] * stride_x
    acc0 = tl.zeros((BLOCK_R, BLOCK_M), dtype=tl.int32)
    acc1 = tl.zeros((BLOCK_R, BLOCK_M), dtype=tl.int32)
    acc2 = tl.zeros((BLOCK_R, BLOCK_M), dtype=tl.int32)
    acc3 = tl.zeros((BLOCK_R, BLOCK_M), dtype=tl.int32)
    for n0 in range(k_lo, k_hi, BLOCK_N):
        offs_n = n0 + offs_n0
        w = tl.load(w_rows + offs_n[None, :], mask=m_ok[:, None] & (offs_n < k_hi)[None, :], other=0)
        wt = tl.trans(w)
        v = tl.load(x_rows + offs_n[None, :], mask=r_ok[:, None] & (offs_n < N)[None, :], other=0)
        v = v.to(tl.uint32, bitcast=True)
        d = ((v & 0xFF).to(tl.int32, bitcast=True) ^ 0x80) - 0x80  # balanced digit in [-128, 127]
        acc0 = tl.dot(d.to(tl.int8), wt, acc0, out_dtype=tl.int32)
        v = (v - d.to(tl.uint32, bitcast=True)) >> 8  # exact: (x - d0) mod 2^32 is a multiple of 256
        d = ((v & 0xFF).to(tl.int32, bitcast=True) ^ 0x80) - 0x80
        acc1 = tl.dot(d.to(tl.int8), wt, acc1, out_dtype=tl.int32)
        v = (v - d.to(tl.uint32, bitcast=True)) >> 8
        d = ((v & 0xFF).to(tl.int32, bitcast=True) ^ 0x80) - 0x80
        acc2 = tl.dot(d.to(tl.int8), wt, acc2, out_dtype=tl.int32)
        v = (v - d.to(tl.uint32, bitcast=True)) >> 8
        d = ((v & 0xFF).to(tl.int32, bitcast=True) ^ 0x80) - 0x80  # top digit: congruence mod 2^32 only
        acc3 = tl.dot(d.to(tl.int8), wt, acc3, out_dtype=tl.int32)
    acc = (acc0.to(tl.uint32, bitcast=True) + (acc1.to(tl.uint32, bitcast=True) << 8)
           + (acc2.to(tl.uint32, bitcast=True) << 16) + (acc3.to(tl.uint32, bitcast=True) << 24))
    acc = acc & kmask.to(tl.uint32, bitcast=True)
    y_ptrs = y_ptr + offs_r.to(tl.int64)[:, None] * stride_y + offs_m[None, :]
    y_ok = r_ok[:, None] & m_ok[None, :]
    if ATOMIC:
        tl.atomic_add(y_ptrs.to(tl.pointer_type(tl.uint32)), acc, mask=y_ok)
    else:
        tl.store(y_ptrs, acc.to(tl.int32, bitcast=True), mask=y_ok)

