"""Compute kernels: exact ``W8 x_hat mod 2^K`` for one linear on the service device.

Backends (all exact modulo ``2^K``; see SPEC.md section 5):

``cuda_int8``
    ``K/8`` int8 tensor-core GEMMs (``torch._int_mm``) on balanced base-256 digits of ``x_hat``,
    int32 accumulation, inner dimension chunked so no partial sum overflows. Rows are padded to
    at least 32 and to a multiple of 8, ``n`` and ``m`` to multiples of 8 (cuBLASLt constraints).
    Implemented by :class:`pgw.core.PreparedWeight`.
``cpu_int8``
    The same digit algorithm on CPU via ``torch._int_mm`` (oneDNN/fbgemm, VNNI/AMX where present).
    Selected only after an exactness probe with adversarial operands passes
    (:func:`pgw.core.ringmm.cpu_int8_available`), because some int8 CPU kernels saturate int16
    intermediates. The digit GEMM is :func:`pgw.core.ringmm.int8_ring_matmul` (shared with the
    engine and the factory; CPU inner chunks of ``INT_MM_MAX_INNER_CPU``).
``cpu_f64`` / ``cpu_f32``
    Float GEMMs on wide balanced digits (:class:`pgw.core.PreparedWeight`). Always exact.
``auto``
    ``cuda_int8`` on CUDA devices; on CPU ``cpu_int8`` if the probe passes, else ``cpu_f64``.

Measured: Llama-3-8B, T4, K=48: 0.45 s/token at batch 1, 0.014 s/token amortized at batch 32.
Future: fused CUDA/Triton kernel doing digit split + K/8 GEMMs + shift-accumulate in one pass.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams
from pgw.core.ringmm import (
    PreparedWeight,
    cpu_int8_available,
    cpu_int8_fast,
    cuda_int8_available,
    int8_ring_matmul,
)
from pgw.core.types import LinearSpec

__all__ = [
    "SERVICE_BACKENDS",
    "ComputeKernel",
    "ServiceBackend",
    "cpu_int8_available",
    "int8_ring_matmul",
    "resolve_backend",
    "resolve_device",
]

log = logging.getLogger(__name__)

ServiceBackend = Literal["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"]
SERVICE_BACKENDS: tuple[str, ...] = ("auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32")

def resolve_device(device: str | torch.device) -> torch.device:
    """``"auto"`` -> ``cuda`` (device 0) when available, else ``cpu``. Validates explicit devices.

    Raises:
        ConfigError: malformed or unsupported device string.
        BackendUnavailableError: a CUDA device is requested but not present.
    """
    if isinstance(device, str) and device.strip().lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ConfigError(f"invalid device {device!r}") from exc
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise BackendUnavailableError(f"device {device!s} requested but CUDA is not available")
        idx = 0 if dev.index is None else dev.index
        if idx >= torch.cuda.device_count():
            raise BackendUnavailableError(f"device {device!s} requested but only {torch.cuda.device_count()} GPU(s)")
        return torch.device("cuda", idx)
    if dev.type != "cpu":
        raise ConfigError(f"unsupported device type {dev.type!r}; use cpu or cuda[:i]")
    return dev


def resolve_backend(device: torch.device, backend: str = "auto") -> str:
    """Concrete backend for ``device``.

    Raises:
        ConfigError: unknown backend name.
        BackendUnavailableError: backend incompatible with the device or unavailable on this host.
    """
    if backend not in SERVICE_BACKENDS:
        raise ConfigError(f"unknown service backend {backend!r}; choose one of {SERVICE_BACKENDS}")
    if backend == "auto":
        if device.type == "cuda":
            return "cuda_int8"
        return "cpu_int8" if cpu_int8_fast() else "cpu_f64"
    if backend == "cuda_int8":
        if device.type != "cuda":
            raise BackendUnavailableError("cuda_int8 needs a CUDA device")
        if not cuda_int8_available():  # pragma: no cover - GPU only
            raise BackendUnavailableError("torch._int_mm does not work on this CUDA device")
        return backend
    if device.type != "cpu":
        raise BackendUnavailableError(f"{backend} needs device 'cpu'")
    if backend == "cpu_int8" and not cpu_int8_available():
        raise BackendUnavailableError("cpu_int8 requested but CPU torch._int_mm is unavailable or inexact")
    return backend


class ComputeKernel:
    """Exact ring matmul for one public int8 linear weight resident on one device.

    Args:
        spec: Linear spec (``id``, ``out_features m``, ``in_features n``).
        w8: int8 ``(m, n)`` weight (any device; copied to ``device``).
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:i"``.
        backend: See module docstring.
        cache_float: CPU float backends keep a float copy of the weight (faster, 4-8x memory).

    Thread-safety: :meth:`run` may be called from several threads (weights are read-only), but the
    service serializes GPU work per device through :class:`pgw.service.BatchScheduler`.
    """

    def __init__(
        self,
        spec: LinearSpec,
        w8: torch.Tensor,
        device: str | torch.device = "auto",
        backend: str = "auto",
        *,
        cache_float: bool = True,
    ):
        if w8.dtype != torch.int8 or w8.ndim != 2:
            raise ShapeError(f"{spec.id}: weight must be 2-D int8, got {w8.dtype} {tuple(w8.shape)}")
        if tuple(w8.shape) != (spec.out_features, spec.in_features):
            raise ShapeError(
                f"{spec.id}: weight shape {tuple(w8.shape)} does not match spec "
                f"({spec.out_features}, {spec.in_features})"
            )
        self.spec = spec
        self.device = resolve_device(device)
        self.backend = resolve_backend(self.device, backend)
        self._pw: PreparedWeight | None = None
        self._w: torch.Tensor | None = None
        # Own the memory: a pack's tensors may be memory-mapped views of files.
        wd = w8.detach().to(self.device)
        if wd.data_ptr() == w8.data_ptr():
            wd = wd.clone()
        wd = wd.contiguous()
        if self.backend == "cpu_int8":
            self._w = wd
        else:
            self._pw = PreparedWeight(
                wd, device=self.device, backend=self.backend,  # type: ignore[arg-type]
                cache_float=cache_float,
            )

    # ---- metadata ------------------------------------------------------------------------
    @property
    def linear(self) -> str:
        """Canonical linear id (``"3.qkv"``)."""
        return self.spec.id.key

    @property
    def kind(self) -> str:
        """Linear kind (``qkv``, ``o``, ``gate_up``, ``down``); used as a metrics label."""
        return self.spec.id.kind.value

    @property
    def out_features(self) -> int:
        """``m``."""
        return self.spec.out_features

    @property
    def in_features(self) -> int:
        """``n``."""
        return self.spec.in_features

    @property
    def device_bytes(self) -> int:
        """Device memory held by this kernel (weights and cached float copies)."""
        if self._w is not None:
            return int(self._w.numel() * self._w.element_size())
        assert self._pw is not None
        total = 0
        for t in (self._pw._w_dev, self._pw._w_float):
            if t is not None:
                total += int(t.numel() * t.element_size())
        return total

    # ---- compute -------------------------------------------------------------------------
    def run(self, x_hat: torch.Tensor, ring: RingParams) -> torch.Tensor:
        """``(B, n)`` int64 -> ``(B, m)`` int64 in ``[0, 2^K)``; input and output on CPU.

        Raises:
            ShapeError: wrong shape or dtype.
        """
        if x_hat.ndim != 2 or x_hat.shape[1] != self.in_features:
            raise ShapeError(
                f"{self.linear}: x_hat must be (B, {self.in_features}), got {tuple(x_hat.shape)}",
                details={"linear": self.linear},
            )
        if x_hat.dtype != torch.int64:
            raise ShapeError(f"{self.linear}: x_hat must be int64, got {x_hat.dtype}", details={"linear": self.linear})
        if x_hat.shape[0] == 0:
            return torch.zeros((0, self.out_features), dtype=torch.int64)
        if self._w is not None:
            out = int8_ring_matmul(x_hat.to(self.device), self._w, ring.k)
        else:
            assert self._pw is not None
            xin = x_hat.to(self.device, non_blocking=True) if self.device.type == "cuda" else x_hat
            out = self._pw.matmul(xin, ring)
        return out.cpu() if out.device.type != "cpu" else out

    __call__ = run

    def mulmod64(self, b: np.ndarray, nbytes: int = 8) -> np.ndarray:
        """``W8 b mod 2^64`` for MLWE mask ciphertexts ``b`` (``(n, cols)`` uint64 -> ``(m, cols)``).

        Used by ``POST /pgw/v1/maskgen`` (:mod:`pgw.hemask`). On CUDA, ``b`` is split into eight
        unsigned byte planes, offset to signed int8, multiplied with int8 tensor-core GEMMs
        (``|partial| <= 127 * 128 * n < 2^31``) and recombined with wrapping int64 shifts. With
        ``nbytes < 8`` (coefficients below ``2^(8 nbytes)``, e.g. q = 2^56) the zero top planes are skipped.
        """
        if not 1 <= nbytes <= 8:
            raise ShapeError("nbytes must be in [1, 8]")
        if b.ndim != 2 or b.shape[0] != self.in_features or b.dtype != np.uint64:
            raise ShapeError(f"{self.linear}: maskgen operand must be ({self.in_features}, cols) uint64, "
                             f"got {b.dtype} {tuple(b.shape)}")
        w8 = self._w if self._w is not None else self._pw.w8  # type: ignore[union-attr]
        if self.device.type != "cuda":
            from pgw.hemask.fastmm import engine, small_mm_big
            from pgw.hemask.lwe import matmul_mod64

            if engine() == "cpu_f64":
                return matmul_mod64(w8.cpu().numpy(), b, fast=False)
            return small_mm_big(w8.cpu().numpy(), b, nbytes=nbytes)
        return self._mulmod64_cuda(b, nbytes)  # pragma: no cover - GPU only

    def _mulmod64_cuda(self, b: np.ndarray, nbytes: int = 8) -> np.ndarray:  # pragma: no cover - GPU only
        assert self._pw is not None and self._pw._w_dev is not None
        wpad = self._pw._w_dev  # (mp, np_) int8, zero padded to multiples of 32
        n, cols = b.shape
        m = self.out_features
        mp, np_ = wpad.shape
        colp = -(-cols // 32) * 32
        bt = torch.from_numpy(np.ascontiguousarray(b).view(np.int64)).to(self.device)
        rowsum = wpad.to(torch.int64).sum(1, keepdim=True)  # for the -128 digit offset
        acc = torch.zeros((mp, colp), dtype=torch.int64, device=self.device)
        plane = torch.zeros((colp, np_), dtype=torch.int8, device=self.device)
        for d in range(nbytes):  # coefficients < 2^(8 nbytes): the top planes are zero
            byte = (bt >> (8 * d)) & 0xFF  # arithmetic shift is fine: the mask keeps the byte
            plane[:cols, :n] = (byte - 128).to(torch.int8).t()
            part = torch._int_mm(wpad, plane.t()).to(torch.int64) + 128 * rowsum
            acc += part << (8 * d)  # int64 two's-complement wrap == mod 2^64
        out = acc[:m, :cols].contiguous().cpu().numpy().view(np.uint64)
        return np.asarray(out)

    def synchronize(self) -> None:
        """Wait for queued device work (no-op on CPU)."""
        if self.device.type == "cuda":  # pragma: no cover - GPU only
            torch.cuda.synchronize(self.device)

    def warmup(self, ring: RingParams, rows: int = 1) -> float:
        """Run one dummy product (allocator / cuBLAS warm-up); returns seconds taken."""
        t0 = time.perf_counter()
        self.run(torch.zeros((rows, self.in_features), dtype=torch.int64), ring)
        self.synchronize()
        return time.perf_counter() - t0

    def __repr__(self) -> str:
        return (
            f"ComputeKernel({self.linear}, m={self.out_features}, n={self.in_features}, "
            f"device={self.device}, backend={self.backend})"
        )
