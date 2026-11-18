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
    "cpu_int8_available",
    "cpu_int8_fast",
    "cuda_int8_available",
    "int8_digit_gemm",
    "int8_ring_matmul",
    "int_mm_max_inner",
    "ringmm",
    "ringmm_reference",
    "select_backend",
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

    def best(fn: Callable[[], object]) -> float:
        fn()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    t_int8 = best(lambda: torch._int_mm(a, w.t()))
    t_f64 = best(lambda: af @ wf)
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

    def __post_init__(self) -> None:
        if self.w8.dtype != torch.int8 or self.w8.ndim != 2:
            raise ShapeError(f"weight must be a 2-D int8 tensor, got {self.w8.dtype} {tuple(self.w8.shape)}")
        dev = torch.device(self.device) if self.device is not None else self.w8.device
        self.device = dev
        self._resolved = select_backend(dev, self.backend)
        m, n = self.w8.shape
        if self._resolved == "cuda_int8":
            mp, np_ = -(-m // _CUDA_ALIGN) * _CUDA_ALIGN, -(-n // _CUDA_ALIGN) * _CUDA_ALIGN
            w = torch.zeros((mp, np_), dtype=torch.int8, device=dev)
            w[:m, :n] = self.w8.to(dev)
            self._w_dev = w
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
