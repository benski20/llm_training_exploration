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

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams, balanced_digits

__all__ = ["Backend", "PreparedWeight", "cuda_int8_available", "ringmm", "ringmm_reference", "select_backend"]

log = logging.getLogger(__name__)

Backend = Literal["auto", "cuda_int8", "cpu_f64", "cpu_f32"]
_BACKENDS = ("auto", "cuda_int8", "cpu_f64", "cpu_f32")

_INT_MM_MAX_N = 131_064  # 128*128*n < 2^31 => n <= 131071; rounded down to a multiple of 8
_INT_MM_MIN_ROWS = 32
_F32_CHUNK = 1024  # 128 * 128 * 1024 = 2^24: every partial sum is an exactly representable integer


def cuda_int8_available() -> bool:
    """True if CUDA is present and ``torch._int_mm`` works on it."""
    if not torch.cuda.is_available():
        return False
    try:
        a = torch.ones((32, 8), dtype=torch.int8, device="cuda")
        torch._int_mm(a, a.t().contiguous())
        return True
    except Exception as exc:  # pragma: no cover - GPU only
        log.warning("CUDA present but torch._int_mm failed: %r", exc)
        return False


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
            mp, np_ = -(-m // 8) * 8, -(-n // 8) * 8
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
        p = max(_INT_MM_MIN_ROWS, -(-b // 8) * 8)
        xp = torch.zeros((p, np_), dtype=torch.int64, device=x.device)
        xp[:b, :n] = x
        mask = (1 << k) - 1
        acc = torch.zeros((p, mp), dtype=torch.int64, device=x.device)
        for i, d in enumerate(balanced_digits(xp, k, 8)):
            d8 = d.to(torch.int8)
            y = torch.zeros((p, mp), dtype=torch.int64, device=x.device)
            for c0 in range(0, np_, _INT_MM_MAX_N):
                c1 = min(np_, c0 + _INT_MM_MAX_N)
                y += torch._int_mm(d8[:, c0:c1].contiguous(), self._w_dev[:, c0:c1].t()).to(torch.int64)
            acc = (acc + _shift_mod(y, 8 * i, k)) & mask
        return acc[:b, :m]

    def _mm_f64(self, x: torch.Tensor, k: int) -> torch.Tensor:
        wf = self._w_float if self._w_float is not None else self._float_weight()
        bits = _f64_digit_bits(self.in_features, k)
        mask = (1 << k) - 1
        acc: torch.Tensor | None = None
        for i, d in enumerate(balanced_digits(x, k, bits)):
            y = torch.round(d.to(torch.float64) @ wf).to(torch.int64)
            term = _shift_mod(y, bits * i, k)
            acc = term if acc is None else (acc + term) & mask
        assert acc is not None
        return acc & mask

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
