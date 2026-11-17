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
    (:func:`cpu_int8_available`), because some int8 CPU kernels saturate int16 intermediates.
``cpu_f64`` / ``cpu_f32``
    Float GEMMs on wide balanced digits (:class:`pgw.core.PreparedWeight`). Always exact.
``auto``
    ``cuda_int8`` on CUDA devices; on CPU ``cpu_int8`` if the probe passes, else ``cpu_f64``.

Measured: Llama-3-8B, T4, K=48: 0.45 s/token at batch 1, 0.014 s/token amortized at batch 32.
Future: fused CUDA/Triton kernel doing digit split + K/8 GEMMs + shift-accumulate in one pass.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Literal

import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams, balanced_digits
from pgw.core.ringmm import PreparedWeight, cuda_int8_available
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

#: ``128 * 128 * n < 2^31`` => ``n <= 131071``; rounded down to a multiple of 8.
_INT_MM_MAX_INNER = 131_064
#: CPU int8 GEMMs (oneDNN/fbgemm) usually compute u8 x s8 with a ``+128`` compensation term, whose
#: int32 intermediates reach ``255 * 128 * n``; ``n <= 65536`` keeps those below ``2^31`` too.
#: :func:`cpu_int8_available` probes exactly this chunk length with extreme operands.
_INT_MM_MAX_INNER_CPU = 65_536


def _shift_mod(y: torch.Tensor, shift: int, k: int) -> torch.Tensor:
    """``(y << shift) mod 2^k`` without signed int64 overflow."""
    if shift >= k:
        return torch.zeros_like(y)
    return (y & ((1 << (k - shift)) - 1)) << shift


def int8_ring_matmul(x: torch.Tensor, w8: torch.Tensor, k: int) -> torch.Tensor:
    """``x @ w8^T mod 2^k`` via ``k/8`` ``torch._int_mm`` GEMMs on balanced base-256 digits.

    Args:
        x: int64 ``(B, n)`` (interpreted mod ``2^k``) on the same device as ``w8``.
        w8: int8 ``(m, n)``, contiguous.
        k: Ring width in bits (multiple of 8).

    Returns:
        int64 ``(B, m)`` in ``[0, 2^k)``. No padding is applied (CPU accepts any shape).
    """
    if k % 8 != 0 or not 8 <= k <= 56:
        raise ConfigError(f"ring width must be a multiple of 8 in [8, 56], got {k}")
    mask = (1 << k) - 1
    n = x.shape[1]
    wt = w8.t()
    chunk = _INT_MM_MAX_INNER if x.device.type == "cuda" else min(_INT_MM_MAX_INNER, _INT_MM_MAX_INNER_CPU)
    acc: torch.Tensor | None = None
    for i, d in enumerate(balanced_digits(x & mask, k, 8)):
        d8 = d.to(torch.int8)
        if n <= chunk:
            y = torch._int_mm(d8, wt).to(torch.int64)
        else:
            y = torch.zeros((x.shape[0], w8.shape[0]), dtype=torch.int64, device=x.device)
            for c0 in range(0, n, chunk):
                c1 = min(n, c0 + chunk)
                y += torch._int_mm(d8[:, c0:c1].contiguous(), w8[:, c0:c1].t()).to(torch.int64)
        term = _shift_mod(y, 8 * i, k)
        acc = term if acc is None else (acc + term) & mask
    assert acc is not None
    return acc & mask


@functools.cache
def cpu_int8_available() -> bool:
    """True if CPU ``torch._int_mm`` exists and is bit-exact, including on extreme operands.

    The probe multiplies all-(-128)/all-127 operands over the full CPU chunk length
    (``_INT_MM_MAX_INNER_CPU``; overflows int16 or saturating int32 intermediates in some kernels),
    for a single row (GEMV paths) and several rows, then checks a random ring product with odd
    shapes against Python integers. The result is cached per process.
    """
    try:
        n_probe = _INT_MM_MAX_INNER_CPU
        for rows in (1, 5):
            for av, bv in ((-128, -128), (-128, 127), (127, 127), (127, -128)):
                a = torch.full((rows, n_probe), av, dtype=torch.int8)
                b = torch.full((9, n_probe), bv, dtype=torch.int8)
                got = torch._int_mm(a, b.t())
                if got.dtype != torch.int32 or not bool((got.to(torch.int64) == av * bv * n_probe).all()):
                    log.warning("CPU torch._int_mm is inexact on extreme inputs; cpu_int8 disabled")
                    return False
        gen = torch.Generator().manual_seed(4321)
        x = torch.randint(0, 1 << 48, (7, 77), dtype=torch.int64, generator=gen)
        w = torch.randint(-128, 128, (13, 77), dtype=torch.int8, generator=gen)
        got = int8_ring_matmul(x, w.contiguous(), 48)
        xs, ws = x.tolist(), w.tolist()
        ref = [[sum(a * b for a, b in zip(xr, wr, strict=True)) % (1 << 48) for wr in ws] for xr in xs]
        if got.tolist() != ref:
            log.warning("CPU torch._int_mm ring product mismatch; cpu_int8 disabled")
            return False
        return True
    except Exception as exc:  # torch build without CPU _int_mm
        log.info("CPU torch._int_mm unavailable (%s); cpu_int8 disabled", type(exc).__name__)
        return False


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
        return "cpu_int8" if cpu_int8_available() else "cpu_f64"
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
