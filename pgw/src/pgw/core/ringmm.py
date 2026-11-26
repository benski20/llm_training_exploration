"""Exact ring matrix multiplication ``Y = X @ W8^T mod 2^K``.

``X`` is ``(B, n)`` int64 in ``[0, 2^K)`` (masked activations or mask vectors ``r``), ``W8`` is the
public ``(m, n)`` int8 weight. The result is exact modulo ``2^K`` on every backend.

Backends (``backend=`` argument):

``cuda_int8``
    ``K/8`` int8 tensor-core GEMMs via ``torch._int_mm`` on balanced base-256 digits. ``_int_mm``
    needs more than 16 rows and dimensions that are multiples of 8, so inputs are zero-padded.
    int32 accumulation is exact while ``128 * 128 * n < 2^31`` (``n < 131072``); longer inner
    dimensions are split into chunks.
``cpu_f64``
    Wide balanced digits (as many bits as float64 allows for this ``n``), one float64 GEMM per digit.
    For ``n <= 2^12`` and ``K <= 48`` that is 2 GEMMs. All partial sums are integers below 2^53, so the
    BLAS result is exact regardless of summation order.
``cpu_f32``
    Balanced base-256 digits, float32 GEMMs over inner-dimension chunks of 1024 (partial sums stay
    within 2^24, hence exact), accumulated in int64.
``auto``
    ``cuda_int8`` for CUDA tensors, else ``cpu_f64``.

:class:`PreparedWeight` caches the per-backend form of a weight for repeated calls (the service and
the factory call the same weight many times).
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams, balanced_digits

__all__ = [
    "INT_MM_MAX_INNER",
    "INT_MM_MAX_INNER_CPU",
    "Backend",
    "PreparedWeight",
    "chunk_rows_for",
    "cpu_int8_available",
    "cpu_int8_fast",
    "cuda_int8_available",
    "eager_work_budget",
    "int8_digit_gemm",
    "int8_ring_matmul",
    "int_mm_max_inner",
    "offset_byte_consts",
    "offset_byte_core",
    "offset_byte_rows",
    "offset_byte_work_bytes",
    "pad_weight_cuda",
    "padded_shape",
    "ringmm",
    "ringmm_reference",
    "select_backend",
    "weight_rowsum",
]

log = logging.getLogger(__name__)

Backend = Literal["auto", "cuda_int8", "cpu_f64", "cpu_f32"]
_BACKENDS = ("auto", "cuda_int8", "cpu_f64", "cpu_f32")

#: ``128 * 128 * n < 2^31`` => ``n <= 131071``; rounded down to a multiple of ``_CUDA_ALIGN``.
INT_MM_MAX_INNER = 131_040
#: CPU int8 GEMMs (oneDNN/fbgemm) usually compute u8 x s8 with a ``+128`` compensation term whose
#: int32 intermediates reach ``255 * 128 * n``; ``n <= 65536`` keeps those below ``2^31`` too.
#: :func:`cpu_int8_available` probes exactly this chunk length with extreme operands.
INT_MM_MAX_INNER_CPU = 65_536
#: CUDA operand padding. cuBLASLt int8 GEMMs (``torch._int_mm``) on Turing (T4) with CUDA 13 reject
#: output/inner dimensions that are multiples of 8 but not of 16/32 (CUBLAS_STATUS_NOT_SUPPORTED, measured
#: on Kaggle: m in {8, 24, 40} and (m=16, k=24) fail); multiples of 32 work everywhere.
_CUDA_ALIGN = 32
_INT_MM_MIN_ROWS = 32
_F32_CHUNK = 1024  # 128 * 128 * 1024 = 2^24: every partial sum is an exactly representable integer


def cuda_int8_available() -> bool:
    """True if CUDA is present and ``torch._int_mm`` works on it."""
    if not torch.cuda.is_available():
        return False
    try:
        # Probe the layout every caller uses: mat2 column-major (a transposed view of a row-major
        # (m, n) weight). cuBLASLt int8 GEMMs on some GPUs/CUDA versions (e.g. T4 with CUDA 13)
        # reject a row-major mat2 with CUBLAS_STATUS_NOT_SUPPORTED.
        a = torch.ones((32, _CUDA_ALIGN), dtype=torch.int8, device="cuda")
        return bool(int(torch._int_mm(a, a.t()).sum()) == 32 * 32 * _CUDA_ALIGN)
    except Exception as exc:  # pragma: no cover - GPU only
        log.warning("CUDA present but torch._int_mm failed: %r", exc)
        return False


def int_mm_max_inner(device: torch.device | str) -> int:
    """Longest inner dimension one exact ``torch._int_mm`` call may take on ``device``."""
    return INT_MM_MAX_INNER if torch.device(device).type == "cuda" else min(INT_MM_MAX_INNER, INT_MM_MAX_INNER_CPU)


def int8_digit_gemm(d8: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact ``d8 @ w^T`` as int64 for int8 ``d8 (p, n)`` and ``w (m, n)`` (any layout).

    The inner dimension is split into chunks of :func:`int_mm_max_inner` so int32 accumulation
    (and CPU compensation terms) never overflow. No padding: on CUDA the caller passes shapes
    ``torch._int_mm`` accepts.
    """
    n = int(d8.shape[1])
    chunk = int_mm_max_inner(d8.device)
    if n <= chunk:
        return torch._int_mm(d8, w.t()).to(torch.int64)
    y: torch.Tensor | None = None
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        part = torch._int_mm(d8[:, c0:c1].contiguous(), w[:, c0:c1].t()).to(torch.int64)
        y = part if y is None else y + part
    assert y is not None
    return y


def int8_ring_matmul(x: torch.Tensor, w8: torch.Tensor, k: int) -> torch.Tensor:
    """``x @ w8^T mod 2^k`` via ``k/8`` int8 GEMMs (``torch._int_mm``) on balanced base-256 digits.

    The single implementation of the digit GEMM shared by the service, the factory and the engine.

    Args:
        x: int64 ``(B, n)`` (interpreted mod ``2^k``) on the same device as ``w8``.
        w8: int8 ``(m, n)`` (row-major, or the transposed view of an ``(n, m)`` buffer).
        k: Ring width in bits (multiple of 8 in ``[8, 56]``).

    Returns:
        int64 ``(B, m)`` in ``[0, 2^k)``. No padding is applied (on CUDA the caller pads).
    """
    if k % 8 != 0 or not 8 <= k <= 56:
        raise ConfigError(f"ring width must be a multiple of 8 in [8, 56], got {k}")
    mask = (1 << k) - 1
    b = int(x.shape[0])
    digits = balanced_digits(x & mask, k, 8)
    # All digits stacked as rows of ONE GEMM: the weight is read once instead of k/8 times, which is
    # what matters for small batches (decode), where the GEMM is memory-bound.
    d8 = torch.cat(digits, dim=0).to(torch.int8)
    y = int8_digit_gemm(d8, w8)
    return _combine_digit_rows(y, b, len(digits), 8, k)


def _combine_digit_rows(y: torch.Tensor, b: int, limbs: int, bits: int, k: int) -> torch.Tensor:
    """``sum_i (y[i*b:(i+1)*b] << bits*i) mod 2^k`` for the stacked digit-row products ``y``."""
    mask = (1 << k) - 1
    acc = y[:b] & mask
    for i in range(1, limbs):
        acc = (acc + _shift_mod(y[i * b : (i + 1) * b], bits * i, k)) & mask
    return acc


@functools.cache
def cpu_int8_available() -> bool:
    """True if CPU ``torch._int_mm`` exists and is bit-exact, including on extreme operands.

    The probe covers every way the engine, service and factory call it: all-(-128)/all-127
    operands over the full CPU chunk length (:data:`INT_MM_MAX_INNER_CPU`; they overflow int16 or
    saturating int32 intermediates in some kernels), row counts 1, 2, 5 and 33 (GEMV and GEMM
    paths; oneDNN picks kernels by shape), both layouts of the second operand (transposed view and
    row-major), and random ring products with odd shapes checked against Python integers. The
    result is cached per process (``cpu_int8_available.cache_clear()`` re-probes).
    """
    try:
        n_probe = INT_MM_MAX_INNER_CPU
        for rows in (1, 2, 5, 33):
            for av, bv in ((-128, -128), (-128, 127), (127, 127), (127, -128)):
                a = torch.full((rows, n_probe), av, dtype=torch.int8)
                b = torch.full((9, n_probe), bv, dtype=torch.int8)
                expect = av * bv * n_probe
                for got in (torch._int_mm(a, b.t()), torch._int_mm(a, b.t().contiguous())):
                    if got.dtype != torch.int32 or not bool((got.to(torch.int64) == expect).all()):
                        log.warning("CPU torch._int_mm is inexact on extreme inputs; CPU int8 GEMMs disabled")
                        return False
        gen = torch.Generator().manual_seed(1234)
        for rows in (1, 2, 7, 40):
            x = torch.randint(0, 1 << 48, (rows, 77), dtype=torch.int64, generator=gen)
            w = torch.randint(-128, 128, (13, 77), dtype=torch.int8, generator=gen)
            if int8_ring_matmul(x, w, 48).tolist() != ringmm_reference(x, w, 48).tolist():
                log.warning("CPU torch._int_mm ring product mismatch; CPU int8 GEMMs disabled")
                return False
            u = x[:, :13]  # u @ W8: second operand row-major
            if int8_ring_matmul(u, w.t(), 48).tolist() != ringmm_reference(u, w.t().contiguous(), 48).tolist():
                log.warning("CPU torch._int_mm ring product mismatch (row-major layout); CPU int8 GEMMs disabled")
                return False
        return True
    except Exception as exc:  # torch build without CPU _int_mm
        log.info("CPU torch._int_mm unavailable (%s); CPU int8 GEMMs disabled", type(exc).__name__)
        return False


def cpu_int8_fast() -> bool:
    """True if CPU int8 GEMMs are exact (:func:`cpu_int8_available`) *and* fast on this CPU.

    Some CPUs (no VNNI/AMX, or a torch build without an optimized kernel) run ``torch._int_mm``
    through a slow reference loop that is orders of magnitude slower than a float64 GEMM. The int8
    path does about twice the multiplies of the float64 path (8-bit vs ~20-bit digits), so it is
    kept only if it is at least 2x faster per call on a decode-sized product. The timing is cached
    per process (``_cpu_int8_timing_ok.cache_clear()`` re-measures).
    """
    return cpu_int8_available() and _cpu_int8_timing_ok()


@functools.cache
def _cpu_int8_timing_ok() -> bool:
    import time

    gen = torch.Generator().manual_seed(7)
    a = torch.randint(-128, 128, (8, 1024), dtype=torch.int8, generator=gen)
    w = torch.randint(-128, 128, (512, 1024), dtype=torch.int8, generator=gen)
    af, wf = a.to(torch.float64), w.to(torch.float64).t().contiguous()

    def once(fn: Callable[[], object]) -> float:
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    # interleaved, best of many: on a loaded machine a burst of other work during one side's few runs used to
    # pick the float64 path for the whole process (plain prompt passes then ~10x slower at 128 rows)
    i8, f64 = (lambda: torch._int_mm(a, w.t())), (lambda: af @ wf)
    i8(), f64()
    t_int8 = t_f64 = float("inf")
    for rnd in range(4):  # a slow verdict is re-measured a little later (up to ~1 s once per process)
        if rnd:
            time.sleep(0.1)
        for _ in range(15):
            t_int8 = min(t_int8, once(i8))
            t_f64 = min(t_f64, once(f64))
        if t_int8 * 2.0 <= t_f64:
            break
    fast = t_int8 * 2.0 <= t_f64
    log.info("CPU int8 GEMM %.3f ms vs float64 %.3f ms: %s", t_int8 * 1e3, t_f64 * 1e3,
             "using cpu_int8" if fast else "int8 is slow on this CPU, using cpu_f64")
    return fast


def select_backend(device: torch.device | str, backend: Backend = "auto") -> str:
    """Resolve ``backend`` for tensors on ``device``."""
    if backend not in _BACKENDS:
        raise ConfigError(f"unknown ringmm backend {backend!r}; choose one of {_BACKENDS}")
    dev = torch.device(device)
    if backend == "auto":
        return "cuda_int8" if dev.type == "cuda" else "cpu_f64"
    if backend == "cuda_int8" and dev.type != "cuda":
        raise BackendUnavailableError("cuda_int8 backend needs CUDA tensors")
    if backend in ("cpu_f64", "cpu_f32") and dev.type != "cpu":
        raise BackendUnavailableError(f"{backend} backend needs CPU tensors")
    return backend


def _f64_digit_bits(n: int, k: int) -> int:
    """Widest balanced digit such that ``2^(b-1) * 128 * n <= 2^52`` (one bit of margin)."""
    b = 53 - 7 - max(1, math.ceil(math.log2(max(2, n))))
    b = max(8, min(b, k))
    limbs = -(-k // b)
    return -(-k // limbs)  # balance digit width across limbs


def _shift_mod(y: torch.Tensor, shift: int, k: int) -> torch.Tensor:
    """``(y << shift) mod 2^k`` without signed int64 overflow."""
    if shift >= k:
        return torch.zeros_like(y)
    return (y & ((1 << (k - shift)) - 1)) << shift


def _offset_byte_mm(x: torch.Tensor, k: int, w: torch.Tensor, rowsum: torch.Tensor, m: int) -> torch.Tensor:
    """Exact ``x @ w^T mod 2^K`` with one int8 GEMM over offset byte planes (CUDA path).

    ``x`` holds ring elements in ``[0, 2^K)`` as int64; its little-endian bytes ``u_i`` are the
    unsigned base-256 digits. With ``d_i = u_i - 128`` (int8)::

        W x = sum_i 2^(8i) W d_i + C * rowsum(W)   (mod 2^K),   C = 128 * sum_i 2^(8i)

    Every ``|W d_i| <= 128 * 128 * n < 2^31`` for ``n < 131072`` (int32-exact; larger ``n`` is
    chunked). Each term is reduced mod ``2^(K - 8i)`` before the shift, so the int64 sum cannot
    overflow. About a dozen device ops per call (the balanced-digit path took ~45).

    Args:
        x: ``(b, n)`` int64 in ``[0, 2^K)``, on ``w``'s device.
        k: Ring bits (multiple of 8, ``<= 56``).
        w: ``(mp, np_)`` int8 weight, zero padded (``mp, np_`` multiples of 32).
        rowsum: ``(mp,)`` int64 row sums of ``w``.
        m: Output features (the unpadded ``mp``).
    """
    if k % 8 or not 8 <= k <= 56:
        raise ShapeError(f"offset-byte matmul needs K a multiple of 8 in [8, 56], got {k}")
    sh, pm = offset_byte_consts(k, x.device)
    b, n = int(x.shape[0]), int(x.shape[1])
    mp, np_ = int(w.shape[0]), int(w.shape[1])
    step = chunk_rows_for(eager_work_budget(), lambda r: offset_byte_work_bytes(k, r, n, np_, mp), b)
    if step >= b:
        return offset_byte_core(x, k, w, rowsum, sh, pm, offset_byte_rows(k, b))[:, :m]
    # rows in chunks (each output row depends only on its own input row: bit-identical), bounded memory
    out = torch.empty((b, m), dtype=torch.int64, device=x.device)
    for r0 in range(0, b, step):
        xc = x[r0:r0 + step]
        out[r0:r0 + step] = offset_byte_core(xc, k, w, rowsum, sh, pm, offset_byte_rows(k, int(xc.shape[0])))[:, :m]
    return out


def offset_byte_rows(k: int, b: int) -> int:
    """Padded GEMM rows of the eager offset-byte path: ``K/8 * b`` rounded up to 8, at least 32."""
    return max(_INT_MM_MIN_ROWS, -(-((k // 8) * b) // 8) * 8)


def offset_byte_consts(k: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-plane shifts ``8 i`` and masks ``2^(K - 8 i) - 1`` (int64 ``(K/8, 1, 1)``) on ``device``.

    Built eagerly (a host-to-device copy); CUDA-graph captures create them before capturing.
    """
    nb = k // 8
    sh = torch.arange(0, 8 * nb, 8, dtype=torch.int64, device=device).view(nb, 1, 1)
    pm = torch.tensor([(1 << (k - 8 * i)) - 1 for i in range(nb)], dtype=torch.int64, device=device).view(nb, 1, 1)
    return sh, pm


def offset_byte_core(
    x: torch.Tensor, k: int, w: torch.Tensor, rowsum: torch.Tensor, sh: torch.Tensor, pm: torch.Tensor, p: int
) -> torch.Tensor:
    """Device-only body of :func:`_offset_byte_mm`: returns ``(b, mp)`` int64 in ``[0, 2^K)``.

    No host synchronization and no host-to-device copies, so it can be captured in a CUDA graph.
    ``p >= (K/8) * b`` is the padded GEMM row count (the caller picks a shape ``torch._int_mm``
    accepts); ``sh``/``pm`` come from :func:`offset_byte_consts` (kept for the callers; the per-plane
    shifts and masks are the same constants, passed as kernel scalars). Columns past ``m`` are padding.

    Working memory (:func:`offset_byte_work_bytes`): the int32 GEMM result and two ``(b, mp)`` int64
    buffers. The planes are recombined in place (``acc += (plane_i & (2^(K-8i) - 1)) << 8i``), not as
    ``K/8`` stacked int64 temporaries (three ``(K/8) b mp`` int64 tensors at once before 2026-11-26: 24
    bytes per digit row and output column, e.g. 7.5 GB for a 2048-digit-row call on a 152k-row LM head).
    Same integers in the same int64 wrap-around arithmetic, so the result is bit-identical.
    """
    nb = k // 8
    b, n = x.shape
    mp, np_ = w.shape
    rows = nb * b
    if p < rows:
        raise ShapeError(f"offset-byte matmul: {p} padded rows < {rows} digit rows")
    planes = (x.contiguous().view(torch.uint8).view(b, n, 8)[:, :, :nb] ^ 0x80).view(torch.int8)
    d8 = torch.zeros((p, np_), dtype=torch.int8, device=x.device)
    d8[:rows, :n].view(nb, b, n).copy_(planes.permute(2, 0, 1))
    del planes
    if np_ <= INT_MM_MAX_INNER:
        y = torch._int_mm(d8, w.t())  # (p, mp) int32, exact: |sum| <= 128 * 128 * n < 2^31
    else:  # inner chunks summed in int64 (each chunk's int32 partial is exact)
        y = torch._int_mm(d8[:, :INT_MM_MAX_INNER].contiguous(), w[:, :INT_MM_MAX_INNER].t())[:rows].to(torch.int64)
        for c0 in range(INT_MM_MAX_INNER, np_, INT_MM_MAX_INNER):
            c1 = min(np_, c0 + INT_MM_MAX_INNER)
            y.add_(torch._int_mm(d8[:, c0:c1].contiguous(), w[:, c0:c1].t())[:rows])
    del d8
    acc = torch.empty((b, mp), dtype=torch.int64, device=x.device)
    acc.copy_(y[:b])  # int32 -> int64 sign extension, as ``.to(torch.int64)``
    acc.bitwise_and_((1 << k) - 1)
    if nb > 1:
        tmp = torch.empty((b, mp), dtype=torch.int64, device=x.device)
        for i in range(1, nb):
            tmp.copy_(y[i * b:(i + 1) * b])
            tmp.bitwise_and_((1 << (k - 8 * i)) - 1)  # reduced mod 2^(K - 8i) before the shift: no overflow
            tmp.bitwise_left_shift_(8 * i)
            acc.add_(tmp)
        del tmp
    del y
    c = (128 * sum(1 << (8 * i) for i in range(nb))) % (1 << k)
    acc.add_(rowsum, alpha=c)  # int64 wrap-around == mod 2^64, then reduced mod 2^K
    return acc.bitwise_and_((1 << k) - 1)


def offset_byte_work_bytes(k: int, b: int, n: int, np_: int, mp: int, p: int | None = None) -> int:
    """Peak bytes :func:`offset_byte_core` allocates for a ``(b, n)`` call on a padded ``(mp, np_)`` weight
    (its result included; the input and the weight are not). Used to size row chunks and by
    :mod:`pgw.service.memplan`; checked against measured allocations on CPU (``tests/core/test_ringmm.py``).
    cuBLAS workspaces are outside (a fixed per-stream reservation, counted by the memory plan).
    """
    nb = k // 8
    rows = nb * b
    p = offset_byte_rows(k, b) if p is None else int(p)
    ph_planes = b * n * nb + p * np_  # byte planes + the padded digit operand
    if np_ <= INT_MM_MAX_INNER:
        ph_gemm = p * np_ + 4 * p * mp
        y_bytes = 4 * p * mp
    else:
        ch = INT_MM_MAX_INNER
        ph_gemm = p * np_ + 8 * rows * mp + 4 * p * mp + p * ch + (p * ch if np_ % ch else 0)
        y_bytes = 8 * rows * mp
    ph_combine = y_bytes + (16 if nb > 1 else 8) * b * mp
    return max(ph_planes, ph_gemm, ph_combine)


def chunk_rows_for(budget: int, work: Callable[[int], int], rows: int, *, align: int = 8) -> int:
    """Largest row count ``r <= rows`` with ``work(r) <= budget`` (at least 1), rounded down to ``align`` when
    above it. ``work`` must be non-decreasing in ``r``."""
    if rows <= 1 or work(rows) <= budget:
        return max(1, rows)
    lo, hi = 1, rows  # work(lo) may exceed the budget: one row is the floor
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if work(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo // align * align if lo >= align else lo


#: Working-memory budget of one eager offset-byte call before it is split into row chunks (bytes). A call's
#: rows are independent (every output row depends only on its own input row), so chunking is bit-identical.
#: ``PGW_SERVICE_EAGER_WORK_MB`` overrides it (read by :func:`eager_work_budget`).
EAGER_WORK_BYTES = 512 << 20
ENV_EAGER_WORK_MB = "PGW_SERVICE_EAGER_WORK_MB"


def eager_work_budget() -> int:
    """Bytes of transient device memory one eager ring-matmul call may use (:data:`EAGER_WORK_BYTES`)."""
    import os

    raw = (os.environ.get(ENV_EAGER_WORK_MB) or "").strip()
    if not raw:
        return EAGER_WORK_BYTES
    try:
        mb = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_EAGER_WORK_MB} must be an integer number of MiB, got {raw!r}") from exc
    if mb < 16:
        raise ConfigError(f"{ENV_EAGER_WORK_MB} must be at least 16 (MiB), got {mb}")
    return mb << 20


def padded_shape(m: int, n: int) -> tuple[int, int]:
    """``(m, n)`` rounded up to the CUDA operand alignment (:func:`pad_weight_cuda`)."""
    return -(-m // _CUDA_ALIGN) * _CUDA_ALIGN, -(-n // _CUDA_ALIGN) * _CUDA_ALIGN


def pad_weight_cuda(w8: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """``w8 (m, n)`` zero padded to multiples of 32 in both dimensions on ``device`` (CUDA layout).

    Device memory: exactly one padded copy. A host weight is padded on the host (only when its shape needs it)
    and copied once; a CUDA weight already on ``device``, contiguous and aligned, is returned as is (no copy).
    Before 2026-11-26 every call made a zeroed padded tensor *and* a full unpadded device copy of ``w8`` (two
    copies of the largest linear at once during load).
    """
    dev = torch.device(device)
    m, n = w8.shape
    mp, np_ = padded_shape(int(m), int(n))
    aligned = (mp, np_) == (int(m), int(n))
    if dev.type == "cuda" and w8.device.type == "cuda" and aligned and w8.is_contiguous() \
            and w8.device.index == (dev.index if dev.index is not None else torch.cuda.current_device()):
        return w8
    if dev.type == "cuda" and w8.device.type == "cpu":
        if aligned:
            return w8.contiguous().to(dev)
        host = torch.zeros((mp, np_), dtype=torch.int8)
        host[:m, :n] = w8
        return host.to(dev)
    w = torch.zeros((mp, np_), dtype=torch.int8, device=dev)
    w[:m, :n] = w8
    return w


def weight_rowsum(w8: torch.Tensor, chunk_rows: int | None = None, *, chunk_bytes: int = 32 << 20) -> torch.Tensor:
    """Exact int64 row sums of an int8 weight, in row chunks: never an int64 copy of the whole weight (that is
    8x the weight: 4.1 GiB for a 152k x 3584 LM head, which ran a 7B provider out of memory on a 16 GB T4).
    Note that ``w8.sum(1, dtype=torch.int64)`` *does* make that copy (torch casts the input first).
    Chunks hold at most ``chunk_bytes`` of int64 (or ``chunk_rows`` rows)."""
    rows, n = int(w8.shape[0]), int(w8.shape[1]) if w8.ndim == 2 else 1
    if chunk_rows is None:
        chunk_rows = max(1, chunk_bytes // max(1, 8 * n))
    out = torch.empty(rows, dtype=torch.int64, device=w8.device)
    for r0 in range(0, rows, chunk_rows):
        out[r0 : r0 + chunk_rows] = w8[r0 : r0 + chunk_rows].to(torch.int64).sum(1)
    return out


@dataclass
class PreparedWeight:
    """A public int8 weight prepared for repeated exact ring matmuls on one device/backend.

    Args:
        w8: ``(m, n)`` int8 weight (values in ``[-127, 127]``; -128 is also handled exactly).
        device: Target device. Defaults to the weight's device.
        backend: See module docstring.
        cache_float: For CPU float backends, keep a float copy of the weight (4-8x memory) instead
            of converting on every call. Recommended unless memory is tight.
    """

    w8: torch.Tensor
    device: torch.device | str | None = None
    backend: Backend = "auto"
    cache_float: bool = True
    _resolved: str = field(init=False, default="")
    _w_dev: torch.Tensor | None = field(init=False, default=None, repr=False)
    _w_float: torch.Tensor | None = field(init=False, default=None, repr=False)
    _w_rowsum: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.w8.dtype != torch.int8 or self.w8.ndim != 2:
            raise ShapeError(f"weight must be a 2-D int8 tensor, got {self.w8.dtype} {tuple(self.w8.shape)}")
        dev = torch.device(self.device) if self.device is not None else self.w8.device
        self.device = dev
        self._resolved = select_backend(dev, self.backend)
        if self._resolved == "cuda_int8":
            self._w_dev = pad_weight_cuda(self.w8, dev)
        else:
            self._w_dev = self.w8.to(dev)
            if self.cache_float:
                self._w_float = self._float_weight()

    @property
    def out_features(self) -> int:
        """``m``."""
        return int(self.w8.shape[0])

    @property
    def in_features(self) -> int:
        """``n``."""
        return int(self.w8.shape[1])

    @property
    def resolved_backend(self) -> str:
        """The concrete backend in use."""
        return self._resolved

    def _float_weight(self) -> torch.Tensor:
        assert self._w_dev is not None
        dt = torch.float64 if self._resolved == "cpu_f64" else torch.float32
        return self._w_dev.to(dt).t().contiguous()  # (n, m)

    def matmul(self, x: torch.Tensor, params: RingParams | int) -> torch.Tensor:
        """Return ``x @ w8^T mod 2^K`` as int64 ``(B, m)`` in ``[0, 2^K)`` on ``x``'s device.

        Args:
            x: int64 ``(B, n)``; any int64 values are accepted and interpreted mod ``2^K``.
            params: Ring parameters or the ring bit width ``K``.
        """
        k = params.k if isinstance(params, RingParams) else int(params)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ShapeError(f"x must be (B, {self.in_features}), got {tuple(x.shape)}")
        if x.dtype != torch.int64:
            raise ShapeError(f"x must be int64 ring elements, got {x.dtype}")
        if x.shape[0] == 0:
            return torch.zeros((0, self.out_features), dtype=torch.int64, device=x.device)
        mask = (1 << k) - 1
        xin = x.to(self.device) & mask
        if self._resolved == "cuda_int8":
            out = self._mm_cuda(xin, k)
        elif self._resolved == "cpu_f64":
            out = self._mm_f64(xin, k)
        else:
            out = self._mm_f32(xin, k)
        return out.to(x.device)

    __call__ = matmul

    # ---- backends --------------------------------------------------------------------------
    def _mm_cuda(self, x: torch.Tensor, k: int) -> torch.Tensor:  # pragma: no cover - GPU only
        assert self._w_dev is not None
        if k % 8 == 0 and k <= 56:
            return _offset_byte_mm(x, k, self._w_dev, self._rowsum(), self.out_features)
        b, n = x.shape
        m = self.out_features
        mp, np_ = self._w_dev.shape
        digits = balanced_digits(x, k, 8)
        limbs = len(digits)
        # Stack the k/8 digit planes as rows of one GEMM so the weight is streamed once per call.
        rows = limbs * b
        p = max(_INT_MM_MIN_ROWS, -(-rows // 8) * 8)
        d8 = torch.zeros((p, np_), dtype=torch.int8, device=x.device)
        d8[:rows, :n] = torch.cat(digits, dim=0)
        y = torch.zeros((p, mp), dtype=torch.int64, device=x.device)
        for c0 in range(0, np_, INT_MM_MAX_INNER):
            c1 = min(np_, c0 + INT_MM_MAX_INNER)
            y += torch._int_mm(d8[:, c0:c1].contiguous(), self._w_dev[:, c0:c1].t()).to(torch.int64)
        return _combine_digit_rows(y, b, limbs, 8, k)[:, :m]

    def _rowsum(self) -> torch.Tensor:
        if self._w_rowsum is None:
            assert self._w_dev is not None
            self._w_rowsum = weight_rowsum(self._w_dev)
        return self._w_rowsum

    def _mm_f64(self, x: torch.Tensor, k: int) -> torch.Tensor:
        wf = self._w_float if self._w_float is not None else self._float_weight()
        bits = _f64_digit_bits(self.in_features, k)
        b = int(x.shape[0])
        digits = balanced_digits(x, k, bits)
        # one GEMM over the stacked digit rows (weight streamed once)
        y = torch.round(torch.cat(digits, dim=0).to(torch.float64) @ wf).to(torch.int64)
        return _combine_digit_rows(y, b, len(digits), bits, k)

    def _mm_f32(self, x: torch.Tensor, k: int) -> torch.Tensor:
        wf = self._w_float if self._w_float is not None else self._float_weight()
        n = self.in_features
        mask = (1 << k) - 1
        acc: torch.Tensor | None = None
        for i, d in enumerate(balanced_digits(x, k, 8)):
            df = d.to(torch.float32)
            y = torch.zeros((x.shape[0], self.out_features), dtype=torch.int64)
            for c0 in range(0, n, _F32_CHUNK):
                c1 = min(n, c0 + _F32_CHUNK)
                y += torch.round(df[:, c0:c1] @ wf[c0:c1]).to(torch.int64)
            term = _shift_mod(y, 8 * i, k)
            acc = term if acc is None else (acc + term) & mask
        assert acc is not None
        return acc & mask


def ringmm(x: torch.Tensor, w8: torch.Tensor, params: RingParams | int, backend: Backend = "auto") -> torch.Tensor:
    """One-shot exact ``x @ w8^T mod 2^K`` (see :class:`PreparedWeight` for repeated use).

    The computation runs on ``x``'s device; ``w8`` is moved there if needed.
    """
    pw = PreparedWeight(w8, device=x.device, backend=backend, cache_float=False)
    return pw.matmul(x, params)


def ringmm_reference(x: torch.Tensor | np.ndarray, w8: torch.Tensor | np.ndarray, k: int) -> np.ndarray:
    """Slow arbitrary-precision reference (Python ints). For tests only; returns int64 ndarray."""
    xa = np.asarray(torch.as_tensor(x).cpu().numpy(), dtype=np.int64).astype(object)
    wa = np.asarray(torch.as_tensor(w8).cpu().numpy(), dtype=np.int64).astype(object)
    out: np.ndarray = np.asarray((xa @ wa.T) % (1 << k)).astype(np.int64)
    return out
