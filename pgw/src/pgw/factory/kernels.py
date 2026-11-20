"""Exact ring GEMM kernels used by the mask factory.

The factory computes ``W8 r mod 2^K`` for huge numbers of mask rows ``r``, so it wants the fastest
*exact* path available on the host:

``cuda_int8``
    ``K/8`` int8 tensor-core GEMMs (``torch._int_mm``) on balanced base-256 digits of ``r``, int32
    accumulation, inner dimension chunked so no partial sum can overflow
    (:func:`pgw.core.ringmm.int8_ring_matmul`, shared with the service and the engine). Inputs are
    zero-padded to the shapes cuBLASLt needs (rows >= 32 and multiples of 8).
``cpu_int8``
    The same algorithm on CPU. Recent PyTorch builds implement ``torch._int_mm`` on CPU via
    oneDNN/fbgemm (VNNI / AMX where present): measured ~1 TOPS on 4 cores versus ~0.1 TFLOPS for
    float64, i.e. ~4x faster than ``cpu_f64`` for K40 despite needing 5 instead of 2 GEMMs. Because
    some int8 CPU kernels saturate intermediate int16 sums on older ISAs, the backend is only
    selected after an exactness probe with adversarial (extreme-value) inputs passes
    (:func:`pgw.core.ringmm.cpu_int8_available`).
``cpu_f64`` / ``cpu_f32``
    Fallbacks from :mod:`pgw.core.ringmm` (always exact, slower).

All kernels return canonical int64 ring elements in ``[0, 2^K)``.

:func:`ring_vecmat` computes the Freivalds precomputation ``u W8 mod 2^K`` (``c`` tiny, inner
dimension ``m``) on any device with float64 GEMMs over row chunks, without materialising a
transposed or float copy of the whole weight.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams, balanced_digits
from pgw.core.ringmm import (
    PreparedWeight,
    _combine_digit_rows,
    _shift_mod,
    cpu_int8_available,
    cpu_int8_fast,
    int8_digit_gemm,
    int8_ring_matmul,
)

__all__ = [
    "FACTORY_BACKENDS",
    "FactoryBackend",
    "RingKernel",
    "cpu_int8_available",
    "resolve_backend",
    "resolve_device",
    "ring_vecmat",
]

log = logging.getLogger(__name__)

FactoryBackend = Literal["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"]
FACTORY_BACKENDS: tuple[str, ...] = ("auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32")

_CUDA_MIN_ROWS = 32
#: Output/inner padding on CUDA (cuBLASLt int8 on T4 + CUDA 13 rejects dims that are not multiples of 32).
_CUDA_ALIGN = 32
#: Elements of a float64 weight chunk materialised at once by :func:`ring_vecmat` (128 MiB).
_VECMAT_CHUNK_ELEMS = 1 << 24


def resolve_device(device: str | torch.device) -> torch.device:
    """``"auto"`` -> ``cuda`` when available, else ``cpu``; otherwise validate the device string."""
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ConfigError(f"invalid device {device!r}") from exc
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError(f"device {device!r} requested but CUDA is not available")
    if dev.type not in ("cpu", "cuda"):
        raise ConfigError(f"unsupported device type {dev.type!r}; use cpu or cuda")
    return dev


def resolve_backend(device: torch.device, backend: str = "auto") -> str:
    """Pick the concrete kernel for ``device``.

    Raises:
        ConfigError: unknown backend name.
        BackendUnavailableError: backend incompatible with the device or not available.
    """
    if backend not in FACTORY_BACKENDS:
        raise ConfigError(f"unknown factory backend {backend!r}; choose one of {FACTORY_BACKENDS}")
    if backend == "auto":
        if device.type == "cuda":
            return "cuda_int8"
        return "cpu_int8" if cpu_int8_fast() else "cpu_f64"
    if backend == "cuda_int8" and device.type != "cuda":
        raise BackendUnavailableError("cuda_int8 needs a CUDA device")
    if backend in ("cpu_int8", "cpu_f64", "cpu_f32") and device.type != "cpu":
        raise BackendUnavailableError(f"{backend} needs device 'cpu'")
    if backend == "cpu_int8" and not cpu_int8_available():
        raise BackendUnavailableError("cpu_int8 requested but torch._int_mm is not exact/available on this CPU")
    return backend


@dataclass
class RingKernel:
    """``W8 r mod 2^K`` for one device-resident weight.

    Cheap to construct (no weight copies unless CUDA needs padding to multiples of 8). One instance
    is created per linear per segment; it is valid as long as ``w8`` is.

    Args:
        w8: int8 ``(m, n)`` weight already on the target device.
        backend: Concrete backend from :func:`resolve_backend`.
    """

    w8: torch.Tensor
    backend: str
    _w: torch.Tensor | None = field(init=False, default=None, repr=False)
    _fallback: PreparedWeight | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.w8.dtype != torch.int8 or self.w8.ndim != 2:
            raise ShapeError(f"weight must be 2-D int8, got {self.w8.dtype} {tuple(self.w8.shape)}")
        m, n = self.w8.shape
        if self.backend == "cuda_int8":
            mp, np_ = -(-m // _CUDA_ALIGN) * _CUDA_ALIGN, -(-n // _CUDA_ALIGN) * _CUDA_ALIGN
            if (mp, np_) == (m, n) and self.w8.is_contiguous():
                self._w = self.w8
            else:
                w = torch.zeros((mp, np_), dtype=torch.int8, device=self.w8.device)
                w[:m, :n] = self.w8
                self._w = w
        elif self.backend == "cpu_int8":
            self._w = self.w8 if self.w8.is_contiguous() else self.w8.contiguous()
        elif self.backend in ("cpu_f64", "cpu_f32"):
            self._fallback = PreparedWeight(self.w8, device="cpu", backend=self.backend, cache_float=True)  # type: ignore[arg-type]
        else:
            raise ConfigError(f"RingKernel needs a concrete backend, got {self.backend!r}")

    @property
    def out_features(self) -> int:
        """``m``."""
        return int(self.w8.shape[0])

    @property
    def in_features(self) -> int:
        """``n``."""
        return int(self.w8.shape[1])

    def matmul(self, r: torch.Tensor, params: RingParams) -> torch.Tensor:
        """Return ``r @ W8^T mod 2^K`` (int64 ``(B, m)``) on the weight's device.

        Args:
            r: int64 ``(B, n)`` ring elements on any device (moved to the weight's device).
            params: Ring parameters.
        """
        if r.ndim != 2 or r.shape[1] != self.in_features:
            raise ShapeError(f"r must be (B, {self.in_features}), got {tuple(r.shape)}")
        if r.dtype != torch.int64:
            raise ShapeError(f"r must be int64, got {r.dtype}")
        b = int(r.shape[0])
        if b == 0:
            return torch.zeros((0, self.out_features), dtype=torch.int64, device=self.w8.device)
        if self._fallback is not None:
            return self._fallback.matmul(r.to("cpu"), params)
        assert self._w is not None
        dev = self._w.device
        if self.backend == "cpu_int8":
            return int8_ring_matmul(r.to(dev), self._w, params.k)
        # cuda_int8: pad rows and inner dimension
        n = self.in_features
        p = max(_CUDA_MIN_ROWS, -(-b // 8) * 8)
        np_ = int(self._w.shape[1])
        if p == b and np_ == n:
            xp = r.to(dev, non_blocking=True)
        else:
            xp = torch.zeros((p, np_), dtype=torch.int64, device=dev)
            xp[:b, :n] = r.to(dev, non_blocking=True)
        return int8_ring_matmul(xp, self._w, params.k)[:b, : self.out_features]

    __call__ = matmul

    @property
    def supports_planes(self) -> bool:
        """True if :meth:`matmul_planes` is available (the int8 digit-GEMM backends)."""
        return self.backend in ("cuda_int8", "cpu_int8")

    def plane_shape(self, rows: int) -> tuple[int, int]:
        """``(P, NP)``: padded rows and inner dimension of one digit plane for a batch of ``rows``.

        The stacked operand of :meth:`matmul_planes` is int8 ``(ceil(K/8) * P, NP)``, zero outside the
        ``rows x n`` block of each plane: the same operand :meth:`matmul` builds from ``r`` (on CUDA rows
        are padded to at least 32 and a multiple of 8, columns to the padded weight).
        """
        if not self.supports_planes:
            raise ConfigError(f"digit-plane GEMM needs an int8 backend, not {self.backend}")
        assert self._w is not None
        if self.backend == "cuda_int8":
            return max(_CUDA_MIN_ROWS, -(-rows // 8) * 8), int(self._w.shape[1])
        return rows, self.in_features

    def matmul_planes(self, d8: torch.Tensor, rows: int, params: RingParams) -> torch.Tensor:
        """``r @ W8^T mod 2^K`` (int64 ``(rows, m)`` on the weight's device) from the digit planes of ``r``.

        ``d8`` is the stacked balanced base-256 digit operand of :meth:`plane_shape` (built on the device
        by :class:`pgw.factory.gpu_prng.DeviceRingStream`, so ``r`` never crosses the host bus). Bit-identical
        to :meth:`matmul` on the same ``r``: it is the second half of :func:`pgw.core.ringmm.int8_ring_matmul`.
        The int64 digit products are zeroed after they are combined.
        """
        if not self.supports_planes:
            raise ConfigError(f"digit-plane GEMM needs an int8 backend, not {self.backend}")
        assert self._w is not None
        n_l = -(-params.k // 8)
        p, np_ = self.plane_shape(rows)
        if d8.dtype != torch.int8 or tuple(d8.shape) != (n_l * p, np_) or d8.device != self._w.device:
            raise ShapeError(f"digit planes must be int8 {(n_l * p, np_)} on {self._w.device}, got {d8.dtype} "
                             f"{tuple(d8.shape)} on {d8.device}")
        if rows == 0:
            return torch.zeros((0, self.out_features), dtype=torch.int64, device=self._w.device)
        y = int8_digit_gemm(d8, self._w)
        try:
            acc = _combine_digit_rows(y, p, n_l, 8, params.k)
        finally:
            y.zero_()
        if p == rows and int(acc.shape[1]) == self.out_features:
            return acc
        return acc[:rows, : self.out_features].contiguous()


def _vecmat_digit_bits(inner: int, k: int) -> int:
    """Widest balanced digit with ``2^(b-1) * 128 * inner <= 2^52``, balanced across limbs."""
    b = 53 - 7 - max(1, math.ceil(math.log2(max(2, inner))))
    b = max(8, min(b, k))
    limbs = -(-k // b)
    return -(-k // limbs)


def ring_vecmat(
    u: torch.Tensor, w8: torch.Tensor, params: RingParams, *, method: Literal["auto", "int8", "f64"] = "auto"
) -> torch.Tensor:
    """Freivalds precomputation ``u @ W8 mod 2^K`` on ``w8``'s device.

    Args:
        u: int64 ``(c, m)`` check vectors (any device; moved to ``w8``'s device).
        w8: int8 ``(m, n)`` weight.
        params: Ring parameters.
        method: ``auto`` (int8 on CPUs where it is exact, else float64), ``int8`` or ``f64``.

    Returns:
        int64 ``(c, n)`` in ``[0, 2^K)`` on the CPU.

    On CPUs with an exact ``torch._int_mm`` this uses int8 digit GEMMs. Otherwise it splits ``u``
    into wide balanced digits and multiplying row chunks of ``W8`` in
    float64 so every partial sum is an integer below ``2^53`` (exact on any BLAS); only one chunk
    (at most 128 MiB) of float weight exists at a time.
    """
    if u.ndim != 2 or w8.ndim != 2 or u.shape[1] != w8.shape[0]:
        raise ShapeError(f"u must be (c, {w8.shape[0]}), got {tuple(u.shape)}")
    c, m = u.shape
    n = int(w8.shape[1])
    k = params.k
    mask = params.mask
    dev = w8.device
    if c == 0:
        return torch.zeros((0, n), dtype=torch.int64)
    if method not in ("auto", "int8", "f64"):
        raise ConfigError(f"unknown vecmat method {method!r}")
    if method == "int8" and not (dev.type == "cpu" and cpu_int8_available()):
        raise BackendUnavailableError("int8 vecmat needs a CPU with an exact torch._int_mm")
    if method == "int8" or (method == "auto" and dev.type == "cpu" and cpu_int8_fast()):
        # u @ W8 == int8 digit GEMMs against the transposed view (no copy): ~4x faster than float64
        return int8_ring_matmul(u.to(torch.int64), w8.t(), k)
    rows = max(1, min(m, _VECMAT_CHUNK_ELEMS // max(1, n)))
    bits = _vecmat_digit_bits(rows, k)
    # split on the weight's device (no host round trip of the digits)
    digits = [d.to(torch.float64) for d in balanced_digits(u.to(dev, torch.int64) & mask, k, bits)]
    acc = torch.zeros((c, n), dtype=torch.int64, device=dev)
    for r0 in range(0, m, rows):
        r1 = min(m, r0 + rows)
        wf = w8[r0:r1].to(torch.float64)
        for i, d in enumerate(digits):
            y = torch.round(d[:, r0:r1] @ wf).to(torch.int64)
            acc = (acc + _shift_mod(y, bits * i, k)) & mask
        del wf
    return acc.cpu()
