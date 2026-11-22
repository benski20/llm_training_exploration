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

The int8 backends split ``r`` into its balanced base-256 digit planes with one offset add and a byte
gather (:func:`ring_digit_planes`: the balanced digits of ``v`` are the little-endian bytes of
``v + 128 * sum_i 256^i``, each minus 128) instead of a compare/select/shift loop per digit, and combine
the per-plane int32 GEMM outputs in place (:func:`combine_digit_planes`). Both are bit-identical to
:func:`pgw.core.ringmm.int8_ring_matmul` (the canonical result is unique); on 4 CPU cores they made the
factory's "GEMM" stage about 3x faster, because the digit split and the int64 combine, not
``torch._int_mm``, dominated it.

:func:`ring_vecmat` computes the Freivalds precomputation ``u W8 mod 2^K`` (``c`` tiny, inner
dimension ``m``) on any device with float64 GEMMs over row chunks, without materialising a
transposed or float copy of the whole weight.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Literal

import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams, balanced_digits
from pgw.core.ringmm import (
    PreparedWeight,
    _shift_mod,
    cpu_int8_available,
    cpu_int8_fast,
    int8_digit_gemm,
    int8_ring_matmul,
    int_mm_max_inner,
)

__all__ = [
    "FACTORY_BACKENDS",
    "FactoryBackend",
    "RingKernel",
    "combine_digit_planes",
    "cpu_int8_available",
    "resolve_backend",
    "resolve_device",
    "ring_digit_planes",
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
#: CPU digit-plane combine works on row blocks of about this many output bytes (stays in cache: ~2x faster
#: than whole-tensor passes for wide linears on 4 vCPUs).
_COMBINE_BLOCK_BYTES = 4 << 20
_LITTLE_ENDIAN = sys.byteorder == "little"


def _offset(k: int) -> int:
    """``128 * sum_i 256^i`` over the ``ceil(K/8)`` digit planes."""
    return 128 * sum(1 << (8 * i) for i in range(-(-k // 8)))


def ring_digit_planes(r: torch.Tensor, k: int, out: torch.Tensor, p: int | None = None) -> torch.Tensor:
    """Write the balanced base-256 digit planes of ``r`` into the stacked int8 operand ``out``; return ``out``.

    ``r`` is int64 ``(B, n)`` on ``out``'s device, read modulo ``2^K``. ``out`` is int8 ``(L * P, NP)`` with
    ``L = ceil(K/8)``, ``P >= B`` (default ``P = len(out) / L``) and ``NP >= n``; plane ``i`` lands in
    ``out[i*P : i*P + B, :n]`` and nothing else is written (zero the padding first when ``P > B`` or
    ``NP > n``). The planes equal ``balanced_digits(r & mask, K, 8)`` cast to int8 for canonical ``r``
    (the factory's ``r`` is canonical by construction, so no range scan or mask pass is made; for other
    values they equal those of ``r mod 2^(8L)``, which give the same product mod ``2^K``).

    Balanced digits ``d_i`` in ``[-128, 128)`` with ``sum d_i 256^i == v`` are the bytes ``e_i`` of
    ``v + C`` (``C = 128 * sum_i 256^i``) minus 128, and ``e_i - 128`` as int8 is ``e_i ^ 0x80``. The int64
    temporary ``r + C`` is zeroed before it is released.
    """
    b, n = (int(x) for x in r.shape)
    n_l = -(-k // 8)
    p = int(out.shape[0]) // n_l if p is None else int(p)
    if out.dtype != torch.int8 or out.ndim != 2 or not out.is_contiguous() or out.shape[0] < n_l * p or p < b \
            or out.shape[1] < n or out.device != r.device:
        raise ShapeError(f"digit planes must be contiguous int8 ({n_l} x >= {b}, >= {n}) on {r.device}, "
                         f"got {out.dtype} {tuple(out.shape)} on {out.device}")
    if r.dtype != torch.int64:
        raise ShapeError(f"r must be int64, got {r.dtype}")
    if b == 0:
        return out
    region = out[: n_l * p].view(n_l, p, int(out.shape[1]))[:, :b, :n]
    if not _LITTLE_ENDIAN or k > 56:  # pragma: no cover - big-endian hosts; K > 56 has no int8 path
        for i, d in enumerate(balanced_digits(r & ((1 << k) - 1), k, 8)):
            region[i].copy_(d)
        return out
    t = torch.empty((b, n), dtype=torch.int64, device=r.device)  # contiguous whatever r's strides
    torch.add(r, _offset(k), out=t)  # canonical r < 2^56: no int64 overflow
    try:
        ru = region.view(torch.uint8)
        ru.copy_(t.view(torch.uint8).view(b, n, 8)[:, :, :n_l].permute(2, 0, 1))
        ru.bitwise_xor_(0x80)
    finally:
        t.zero_()
    return out


def combine_digit_planes(y: torch.Tensor, rows: int, p: int, n_l: int, k: int, m: int, inner: int) -> torch.Tensor:
    """``sum_i 256^i y_i mod 2^K`` for the stacked per-plane products ``y`` (``(n_l * p, >= m)``, int32 or int64).

    Returns a new contiguous int64 ``(rows, m)`` in ``[0, 2^K)`` (rows ``[0, rows)`` and columns ``[0, m)`` of
    each plane; the rest is padding). ``inner`` bounds the GEMM's inner dimension: ``|y_i| <= 2^14 * inner``.
    When ``sum_i |y_i| 256^i`` provably fits in int64 (K = 32 for any width, K = 40 for ``inner < 2^16``) the
    planes are summed with in-place ``add_(alpha=256^i)`` and masked once; otherwise each partial sum is
    reduced before it is shifted (Horner from the top plane), so no intermediate overflows. Either way the
    result is the unique canonical value, identical to :func:`pgw.core.ringmm._combine_digit_rows`.
    """
    mp = int(y.shape[1])
    yv = y.view(n_l, p, mp)
    out = torch.empty((rows, m), dtype=torch.int64, device=y.device)
    if rows == 0:
        return out
    fast = (1 << 15) * max(1, inner) * (1 << (8 * (n_l - 1))) < (1 << 63)  # 2x margin covers sum 256^i/256^(L-1)
    blk = rows if y.device.type != "cpu" else max(16, _COMBINE_BLOCK_BYTES // max(1, 8 * m))
    for r0 in range(0, rows, blk):
        r1 = min(rows, r0 + blk)
        a = out[r0:r1]
        if fast:
            a.copy_(yv[0, r0:r1, :m])
            for i in range(1, n_l):
                a.add_(yv[i, r0:r1, :m], alpha=1 << (8 * i))
            a.bitwise_and_((1 << k) - 1)
        else:
            a.copy_(yv[n_l - 1, r0:r1, :m])
            a.bitwise_and_((1 << max(0, k - 8 * (n_l - 1))) - 1)
            for i in range(n_l - 2, -1, -1):
                a.mul_(256)  # a < 2^(K - 8(i+1)): no overflow
                a.add_(yv[i, r0:r1, :m])
                a.bitwise_and_((1 << (k - 8 * i)) - 1)
    return out


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
        k = params.k
        if k % 8 != 0 or not 8 <= k <= 56:
            raise ConfigError(f"ring width must be a multiple of 8 in [8, 56], got {k}")
        # digit planes straight into the (padded on CUDA) GEMM operand; same operand and result as
        # int8_ring_matmul on a zero-padded copy of r
        p, np_ = self.plane_shape(b)
        n_l = k // 8
        alloc = torch.empty if (p, np_) == (b, self.in_features) else torch.zeros
        d8 = alloc((n_l * p, np_), dtype=torch.int8, device=dev)
        x = r if r.device == dev else r.to(dev, non_blocking=True)
        try:
            ring_digit_planes(x, k, d8, p)
            return self._gemm_planes(d8, b, p, np_, params)
        finally:
            d8.zero_()
            if x is not r:
                x.zero_()  # device copy of r (stream-ordered after the GEMM)

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
        to :meth:`matmul` on the same ``r`` and to :func:`pgw.core.ringmm.int8_ring_matmul` (whose second half
        it is, with the combine of :func:`combine_digit_planes`). The digit products are zeroed after they are
        combined.
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
        return self._gemm_planes(d8, rows, p, np_, params)

    def _gemm_planes(self, d8: torch.Tensor, rows: int, p: int, np_: int, params: RingParams) -> torch.Tensor:
        assert self._w is not None
        n_l = -(-params.k // 8)
        # one exact int32 GEMM when the inner dimension allows it (no int64 copy of the products)
        y = torch._int_mm(d8, self._w.t()) if np_ <= int_mm_max_inner(d8.device) else int8_digit_gemm(d8, self._w)
        try:
            return combine_digit_planes(y, rows, p, n_l, params.k, self.out_features, np_)
        finally:
            y.zero_()


def _vecmat_digit_bits(inner: int, k: int) -> int:
    """Widest balanced digit with ``2^(b-1) * 128 * inner <= 2^52``, balanced across limbs."""
    b = 53 - 7 - max(1, math.ceil(math.log2(max(2, inner))))
    b = max(8, min(b, k))
    limbs = -(-k // b)
    return -(-k // limbs)


def ring_vecmat(
    u: torch.Tensor, w8: torch.Tensor, params: RingParams, *,
    method: Literal["auto", "int8", "f64", "kernel"] = "auto",
) -> torch.Tensor:
    """Freivalds precomputation ``u @ W8 mod 2^K`` on ``w8``'s device.

    Args:
        u: int64 ``(c, m)`` check vectors (any device; moved to ``w8``'s device).
        w8: int8 ``(m, n)`` weight.
        params: Ring parameters.
        method: ``auto`` (CUDA: ``kernel``; CPUs: int8 where it is exact and fast, else float64), ``int8``,
            ``f64`` or ``kernel`` (the factory's own digit-plane GEMM, :class:`RingKernel`, on a transposed copy of
            ``W8``: ``cuda_int8`` on a GPU, ``cpu_int8`` on a CPU; the same exact kernel as ``W8 r``, instead of
            float64 GEMMs that run at 1/32-1/64 rate on T4/L4/A10G-class GPUs).

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
    if method not in ("auto", "int8", "f64", "kernel"):
        raise ConfigError(f"unknown vecmat method {method!r}")
    if (method == "kernel" or (method == "auto" and dev.type == "cuda")) and k % 8 == 0 and 8 <= k <= 56:
        backend = "cuda_int8" if dev.type == "cuda" else "cpu_int8"
        if backend == "cpu_int8" and not cpu_int8_available():
            raise BackendUnavailableError("kernel vecmat on a CPU needs an exact torch._int_mm")
        # u @ W8 = u @ (W8^T)^T: the kernel computes x @ weight^T for weight = W8^T (n, m)
        wt = w8.t().contiguous()
        try:
            dev_out = RingKernel(wt, backend).matmul(u.to(torch.int64), params)
            out = dev_out.cpu()
            if dev_out.device.type != "cpu":
                dev_out.zero_()
            return out
        finally:
            del wt
    if method == "kernel":
        raise ConfigError(f"kernel vecmat needs K a multiple of 8 in [8, 56], got {k}")
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
