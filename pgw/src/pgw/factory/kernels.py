"""Exact ring GEMM kernels used by the mask factory.

The factory computes ``W8 r mod 2^K`` for huge numbers of mask rows ``r``, so it wants the fastest
*exact* path available on the host:

``cuda_int8``
    ``K/8`` int8 tensor-core GEMMs (``torch._int_mm``) on balanced base-256 digits of ``r``, int32
    accumulation, inner dimension chunked at 131064 so no partial sum can overflow. Inputs are
    zero-padded to the shapes cuBLASLt needs (rows >= 32 and multiples of 8).
``cpu_int8``
    The same algorithm on CPU. Recent PyTorch builds implement ``torch._int_mm`` on CPU via
    oneDNN/fbgemm (VNNI / AMX where present): measured ~1 TOPS on 4 cores versus ~0.1 TFLOPS for
    float64, i.e. ~4x faster than ``cpu_f64`` for K40 despite needing 5 instead of 2 GEMMs. Because
    some int8 CPU kernels saturate intermediate int16 sums on older ISAs, the backend is only
    selected after an exactness probe with adversarial (extreme-value) inputs passes.
``cpu_f64`` / ``cpu_f32``
    Fallbacks from :mod:`pgw.core.ringmm` (always exact, slower).

All kernels return canonical int64 ring elements in ``[0, 2^K)``.

:func:`ring_vecmat` computes the Freivalds precomputation ``u W8 mod 2^K`` (``c`` tiny, inner
dimension ``m``) on any device with float64 GEMMs over row chunks, without materialising a
transposed or float copy of the whole weight.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams, balanced_digits
from pgw.core.ringmm import PreparedWeight

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

#: 128 * 128 * n < 2^31  =>  n <= 131071; rounded down to a multiple of 8.
INT_MM_MAX_INNER = 131_064
_CUDA_MIN_ROWS = 32
#: Elements of a float64 weight chunk materialised at once by :func:`ring_vecmat` (128 MiB).
_VECMAT_CHUNK_ELEMS = 1 << 24


def _shift_mod(y: torch.Tensor, shift: int, k: int) -> torch.Tensor:
    """``(y << shift) mod 2^k`` without signed int64 overflow."""
    if shift >= k:
        return torch.zeros_like(y)
    return (y & ((1 << (k - shift)) - 1)) << shift


def _int8_digit_gemm(d8: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact ``d8 @ w^T`` as int64 for int8 ``d8 (p, n)`` and ``w (m, n)`` with inner chunking."""
    n = d8.shape[1]
    if n <= INT_MM_MAX_INNER:
        return torch._int_mm(d8, w.t()).to(torch.int64)
    y: torch.Tensor | None = None
    for c0 in range(0, n, INT_MM_MAX_INNER):
        c1 = min(n, c0 + INT_MM_MAX_INNER)
        part = torch._int_mm(d8[:, c0:c1].contiguous(), w[:, c0:c1].t()).to(torch.int64)
        y = part if y is None else y + part
    assert y is not None
    return y


def int8_ring_mm(x: torch.Tensor, w: torch.Tensor, k: int) -> torch.Tensor:
    """``x @ w^T mod 2^k`` via ``k/8`` int8 GEMMs on balanced base-256 digits (``torch._int_mm``).

    Args:
        x: int64 ``(B, n)`` ring elements (any int64 values; interpreted mod ``2^k``).
        w: int8 ``(m, n)`` on the same device as ``x``.
        k: Ring bit width (multiple of 8).

    No padding is done here; callers on CUDA must pass shapes ``_int_mm`` accepts.
    """
    mask = (1 << k) - 1
    acc: torch.Tensor | None = None
    for i, d in enumerate(balanced_digits(x & mask, k, 8)):
        y = _int8_digit_gemm(d.to(torch.int8), w)
        term = _shift_mod(y, 8 * i, k)
        acc = term if acc is None else (acc + term) & mask
    assert acc is not None
    return acc & mask


@functools.cache
def cpu_int8_available() -> bool:
    """True if ``torch._int_mm`` exists on CPU and is bit-exact, including on extreme values.

    The probe uses all-(-128)/all-127 operands (which overflow int16 intermediates in some
    non-VNNI kernels), odd shapes, and a random ring product checked against Python integers.
    """
    try:
        for av, bv in ((-128, -128), (-128, 127), (127, 127), (127, -128)):
            a = torch.full((5, 4100), av, dtype=torch.int8)
            b = torch.full((9, 4100), bv, dtype=torch.int8)
            got = torch._int_mm(a, b.t())
            if got.dtype != torch.int32 or not bool((got.to(torch.int64) == av * bv * 4100).all()):
                log.warning("CPU torch._int_mm is not exact on extreme inputs; using float fallback")
                return False
        gen = torch.Generator().manual_seed(1234)
        x = torch.randint(0, 1 << 40, (7, 77), dtype=torch.int64, generator=gen)
        w = torch.randint(-128, 128, (13, 77), dtype=torch.int8, generator=gen)
        got = int8_ring_mm(x, w, 40)
        xs, ws = x.tolist(), w.tolist()
        ref = [[sum(a * b for a, b in zip(xr, wr, strict=True)) % (1 << 40) for wr in ws] for xr in xs]
        if got.tolist() != ref:
            log.warning("CPU torch._int_mm ring product mismatch; using float fallback")
            return False
        return True
    except Exception as exc:  # torch build without CPU _int_mm
        log.info("CPU torch._int_mm unavailable (%s); using float fallback", type(exc).__name__)
        return False


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
        return "cpu_int8" if cpu_int8_available() else "cpu_f64"
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
            mp, np_ = -(-m // 8) * 8, -(-n // 8) * 8
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
            return int8_ring_mm(r.to(dev), self._w, params.k)
        # cuda_int8: pad rows and inner dimension
        n = self.in_features
        p = max(_CUDA_MIN_ROWS, -(-b // 8) * 8)
        np_ = int(self._w.shape[1])
        if p == b and np_ == n:
            xp = r.to(dev, non_blocking=True)
        else:
            xp = torch.zeros((p, np_), dtype=torch.int64, device=dev)
            xp[:b, :n] = r.to(dev, non_blocking=True)
        return int8_ring_mm(xp, self._w, params.k)[:b, : self.out_features]

    __call__ = matmul


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
    if method == "int8" or (method == "auto" and dev.type == "cpu" and cpu_int8_available()):
        # u @ W8 == int8 digit GEMMs against the transposed view (no copy): ~4x faster than float64
        return int8_ring_mm(u.to(torch.int64), w8.t(), k)
    rows = max(1, min(m, _VECMAT_CHUNK_ELEMS // max(1, n)))
    bits = _vecmat_digit_bits(rows, k)
    digits = [d.to(dev, torch.float64) for d in balanced_digits(u.to(torch.int64) & mask, k, bits)]
    acc = torch.zeros((c, n), dtype=torch.int64, device=dev)
    for r0 in range(0, m, rows):
        r1 = min(m, r0 + rows)
        wf = w8[r0:r1].to(torch.float64)
        for i, d in enumerate(digits):
            y = torch.round(d[:, r0:r1] @ wf).to(torch.int64)
            acc = (acc + _shift_mod(y, bits * i, k)) & mask
        del wf
    return acc.cpu()
