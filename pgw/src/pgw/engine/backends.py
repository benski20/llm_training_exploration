"""Linear backends: how the engine evaluates ``y = W x + b`` for a masked linear group.

* :class:`PlainBackend` (alias :data:`QuantPlainBackend`): exact integer reference. Quantizes the
  activations exactly as the masked protocol does and computes ``xq @ W8^T`` exactly.
* :class:`FloatBackend`: dequantized float weights, no activation quantization (quality baseline).
* :class:`MaskedBackend`: production. One-time-masked remote compute with unmasking, range and
  Freivalds checks. Bit-identical to :class:`PlainBackend` (SPEC 9.3).

All backends return float32 ``(T, m)`` including the bias, computed as
``y = yq * sx[t] * sw[j] + b[j]`` in float64 then cast (SPEC section 5) for the integer backends.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import inspect
import json
import logging
import math
import os
import secrets
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.core import native
from pgw.core.errors import (
    ConfigError,
    FingerprintMismatchError,
    IntegrityError,
    MaskExhaustedError,
    MaskReuseError,
    MaskStoreError,
    PGWError,
    ProtocolError,
    RingOverflowError,
    ServiceUnavailableError,
    ShapeError,
)
from pgw.core.freivalds import checks_to_u64, freivalds_check_u64
from pgw.core.masking import mask_activations, mask_pack_u32, unmask_outputs_u32, unmask_outputs_u64
from pgw.core.native import native_available
from pgw.core.packing import pack_ring, unpack_ring_u64
from pgw.core.ring import RingParams, balanced_digits, center, dequantize_outputs, quantize_activations
from pgw.core.ringmm import INT_MM_MAX_INNER, cpu_int8_fast, int8_digit_gemm
from pgw.core.types import LinearId

__all__ = [
    "BackendStats",
    "FloatBackend",
    "LinearBackend",
    "LinearCall",
    "MaskedBackend",
    "PlainBackend",
    "QuantPlainBackend",
    "exact_int_matmul",
]

log = logging.getLogger(__name__)


@dataclass
class LinearCall:
    """One linear evaluation request.

    Attributes:
        id: Linear group.
        x: float32 ``(T, n)`` activations (T tokens, possibly from several sequences).
        session: Opaque session handle; the masked backend uses it as that session's mask
            reservation (:class:`pgw.maskstore.MaskReservation` or compatible). ``None`` for
            backends that do not need it.
        meta: Free-form per-call annotations (never sent anywhere).
    """

    id: LinearId
    x: torch.Tensor
    session: object | None = None
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class BackendStats:
    """Cumulative counters of one backend instance (cheap, lock-protected)."""

    calls: int = 0
    rows: int = 0
    round_trips: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    freivalds_checked: int = 0
    freivalds_skipped: int = 0
    integrity_failures: int = 0
    remote_seconds: float = 0.0
    local_seconds: float = 0.0
    #: Round trips of decode-sized calls (at most ``decode_max_rows`` rows in total) and their time.
    decode_calls: int = 0
    decode_remote_seconds: float = 0.0
    decode_local_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **kw: float) -> None:
        """Increment counters."""
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def add_many(self, calls: int, rows: int, round_trips: int, bytes_up: int, bytes_down: int,
                 checked: int, skipped: int, remote_s: float, local_s: float, decode: bool = False) -> None:
        """Per-round-trip update of the masked backend (one lock, no keyword dispatch)."""
        with self._lock:
            if decode:
                self.decode_calls += round_trips
                self.decode_remote_seconds += remote_s
                self.decode_local_seconds += local_s
            self.calls += calls
            self.rows += rows
            self.round_trips += round_trips
            self.bytes_up += bytes_up
            self.bytes_down += bytes_down
            self.freivalds_checked += checked
            self.freivalds_skipped += skipped
            self.remote_seconds += remote_s
            self.local_seconds += local_s

    def snapshot(self) -> dict[str, float]:
        """Plain dict copy."""
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class LinearBackend(abc.ABC):
    """Evaluates masked linear groups. Output: float32 ``(T, m)`` including the bias.

    Implementations must be safe to call from one event loop / thread per session; batching across
    sessions happens through :meth:`forward_many`.
    """

    ring: RingParams

    @abc.abstractmethod
    def forward(self, call: LinearCall) -> torch.Tensor:
        """Evaluate one call synchronously."""

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """Evaluate several calls (same or different linears). Default: sequential."""
        return [self.forward(c) for c in calls]

    async def aforward(self, call: LinearCall) -> torch.Tensor:
        """Async variant. Default: run :meth:`forward` in a worker thread."""
        return await asyncio.to_thread(self.forward, call)

    async def aforward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """Async batched variant. Default: run :meth:`forward_many` in a worker thread."""
        return await asyncio.to_thread(self.forward_many, calls)

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (connections, reservations)."""


# ---- exact integer product -----------------------------------------------------------------------
def _digits_needed(value_bits: int) -> int:
    """Fewest balanced base-256 digits that represent every ``|v| <= 2^(value_bits-1) - 1`` exactly."""
    vmax = (1 << (value_bits - 1)) - 1
    n = 1
    while 127 * ((256**n - 1) // 255) < vmax:
        n += 1
    return n


def exact_int_matmul(xq: torch.Tensor, w8t: torch.Tensor, value_bits: int) -> torch.Tensor:
    """Exact ``xq @ w8t`` for int64 ``xq (T, n)`` with ``|xq| < 2^(value_bits-1)`` and int8 ``w8t (n, m)``.

    Uses int8 GEMMs (``torch._int_mm``, int32 accumulation) on balanced base-256 digits of ``xq``,
    so the result is an exact int64 regardless of BLAS summation order. The digit GEMM and the CPU
    exactness probe are the shared ones in :mod:`pgw.core.ringmm` (:func:`int8_digit_gemm`,
    :func:`cpu_int8_available`). Falls back to float64 GEMMs (exact while partial sums stay below
    2^53) when CPU int8 GEMMs are unavailable or inexact.
    """
    t, n = xq.shape
    m = w8t.shape[1]
    if t == 0:
        return torch.zeros((0, m), dtype=torch.int64, device=xq.device)
    dev = xq.device
    use_int = (dev.type == "cpu" and cpu_int8_fast()) or dev.type == "cuda"
    if not use_int:
        bound = (1 << (value_bits - 1)) * 128 * n
        if bound >= 1 << 53:  # pragma: no cover - only for exotic rings
            raise ConfigError("float64 exact fallback cannot represent this product; need int8 GEMMs")
        return torch.round(xq.to(torch.float64) @ w8t.to(torch.float64)).to(torch.int64)
    ndig = _digits_needed(value_bits)
    digits = balanced_digits(xq, 8 * ndig, 8) if ndig > 1 else [xq]
    acc = torch.zeros((t, m), dtype=torch.int64, device=dev)
    pad_rows = dev.type == "cuda"
    for i, d in enumerate(digits):
        d8 = d.to(torch.int8)
        if pad_rows:  # pragma: no cover - GPU only
            y = torch.zeros((t, m), dtype=torch.int64, device=dev)
            for c0 in range(0, n, INT_MM_MAX_INNER):
                c1 = min(n, c0 + INT_MM_MAX_INNER)
                y += _cuda_int_mm_padded(d8[:, c0:c1].contiguous(), w8t[c0:c1])
        else:
            y = int8_digit_gemm(d8, w8t.t())
        acc += y << (8 * i)
    return acc


def _cuda_int_mm_padded(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pragma: no cover - GPU only
    """Exact ``a (t, n) @ b (n, m)`` on CUDA, padded to shapes ``torch._int_mm`` accepts.

    ``mat2`` is passed column-major (the transpose of a row-major ``(m, n)`` buffer): cuBLASLt int8
    GEMMs on Turing (T4) with recent CUDA reject a row-major ``mat2`` (CUBLAS_STATUS_NOT_SUPPORTED).
    When ``b`` is already such a view with aligned shapes, no copy of the weight is made.
    """
    t, n = a.shape
    m = b.shape[1]
    # rows: > 16 and a multiple of 8; inner and output dims: multiples of 32 (cuBLASLt on T4 / CUDA 13)
    tp, np_, mp = max(32, -(-t // 8) * 8), -(-n // 32) * 32, -(-m // 32) * 32
    if tp == t and np_ == n and a.is_contiguous():
        ap = a
    else:
        ap = torch.zeros((tp, np_), dtype=torch.int8, device=a.device)
        ap[:t, :n] = a
    if np_ == n and mp == m and b.stride() == (1, n):
        bt = b  # already the column-major view of a contiguous (m, n) buffer
    else:
        bcm = torch.zeros((mp, np_), dtype=torch.int8, device=a.device)
        bcm[:m, :n] = b.t()
        bt = bcm.t()
    return torch._int_mm(ap, bt)[:t, :m].to(torch.int64)


# ---- weight access ---------------------------------------------------------------------------------
class _WeightCache:
    """Lazily loads linear weights from a pack and keeps a backend-specific form resident."""

    def __init__(self, pack: Any, device: torch.device, convert: Any, *, loader: Any = None):
        self.pack = pack
        self.device = device
        self.convert = convert
        #: ``fn(lid) -> record`` handed to ``convert``; defaults to ``pack.load_linear`` (full weights).
        self.loader = loader if loader is not None else (lambda lid: pack.load_linear(lid, device="cpu"))
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, lid: LinearId) -> Any:
        key = lid.key
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        with self._lock:
            hit = self._cache.get(key)
            if hit is None:
                hit = self.convert(self.loader(lid), self.device)
                self._cache[key] = hit
        return hit

    def preload(self) -> None:
        for spec in self.pack.linear_specs():
            self.get(spec.id)


@dataclass(frozen=True)
class _IntWeight:
    w8t: torch.Tensor  # int8 (n, m); on CUDA the transposed view of a contiguous (m, n) weight
    scale: torch.Tensor  # float32 (m,)
    bias: torch.Tensor | None
    in_features: int
    out_features: int
    wnorm: float = 0.0  # largest Euclidean norm of a W8 row (capped rings bound outputs with it)


def _row_norm_max(w8: torch.Tensor) -> float:
    """``max_j ||W8_j||_2`` in float64. Each row's float64 sum of squares is exact (below ``2^53``),
    so this equals ``math.sqrt(float(row_norm_sq_max(w8)))`` bit for bit (tested)."""
    return float(w8.to(torch.float64).pow(2).sum(dim=1).sqrt().max())


def _to_int_weight(lw: Any, device: torch.device) -> _IntWeight:
    w8 = lw.w8
    if w8.dtype != torch.int8 or w8.ndim != 2:
        raise ShapeError(f"{lw.id}: weight must be 2-D int8")
    # CUDA: keep W8 row-major and use its transposed view, the (column-major) layout cuBLASLt int8
    # GEMMs accept on every GPU; CPU: a contiguous (n, m) copy (fastest for the CPU kernels).
    w8t = w8.contiguous().to(device).t() if device.type == "cuda" else w8.t().contiguous().to(device)
    return _IntWeight(
        w8t,
        lw.scale.to(torch.float32).to(device),
        None if lw.bias is None else lw.bias.to(torch.float32).to(device),
        int(w8.shape[1]),
        int(w8.shape[0]),
        _row_norm_max(w8),
    )


class PlainBackend(LinearBackend):
    """Exact local quantized product: ``quantize(x) @ W8^T`` then rescale.

    Numerically identical to :class:`MaskedBackend` (same quantization, exact integers), so it is
    the reference for end-to-end exactness tests and the backend for ``dev`` mode.

    Args:
        pack: Open :class:`pgw.modelpack.ModelPack` or any :class:`pgw.engine.pack.EnginePack`
            (linears are loaded lazily and cached).
        ring: Ring parameters (determine activation quantization).
        device: Where to run the integer product.
        preload: Load every linear at construction instead of on first use.
    """

    def __init__(self, pack: object, ring: RingParams, device: str = "cpu", *, preload: bool = False):
        self.pack = pack
        self.ring = ring
        self.device = torch.device(device)
        self.stats = BackendStats()
        self._weights = _WeightCache(pack, self.device, _to_int_weight)
        self._checked: set[str] = set()
        if preload:
            self._weights.preload()

    def _weight(self, lid: LinearId) -> _IntWeight:
        w = self._weights.get(lid)
        if lid.key not in self._checked:
            self.ring.check_in_features(w.in_features)
            self._checked.add(lid.key)
        return w  # type: ignore[no-any-return]

    def forward(self, call: LinearCall) -> torch.Tensor:
        """Quantize, exact integer product, dequantize."""
        t0 = time.perf_counter()
        w = self._weight(call.id)
        x = call.x
        if x.ndim != 2 or x.shape[1] != w.in_features:
            raise ShapeError(f"{call.id}: x must be (T, {w.in_features}), got {tuple(x.shape)}")
        xq, sx = quantize_activations(x.to(self.device), self.ring, w.wnorm if self.ring.cap else None)
        yq = exact_int_matmul(xq.to(self.device), w.w8t, self.ring.a)
        y = dequantize_outputs(yq, sx, w.scale, w.bias, torch.float32)
        self.stats.add(calls=1, rows=int(x.shape[0]), local_seconds=time.perf_counter() - t0)
        return y.to(x.device)

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """Concatenate rows of calls on the same linear into one product (exactness unaffected)."""
        return _grouped_forward(self, calls)


def _grouped_forward(backend: Any, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
    """One product per linear for all calls on it (rows concatenated, results split back)."""
    if len(calls) <= 1:
        return [backend.forward(c) for c in calls]
    out: list[torch.Tensor | None] = [None] * len(calls)
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(calls):
        groups.setdefault(c.id.key, []).append(i)
    for idxs in groups.values():
        if len(idxs) == 1:
            out[idxs[0]] = backend.forward(calls[idxs[0]])
            continue
        xs = [calls[i].x for i in idxs]
        y = backend.forward(LinearCall(calls[idxs[0]].id, torch.cat(xs, dim=0)))
        for i, part in zip(idxs, torch.split(y, [int(x.shape[0]) for x in xs]), strict=True):
            out[i] = part
    return [o for o in out if o is not None]


#: Name used in design docs for the exact integer reference backend.
QuantPlainBackend = PlainBackend


@dataclass(frozen=True)
class _FloatWeight:
    wt: torch.Tensor  # (n, m) float
    bias: torch.Tensor | None
    in_features: int


class FloatBackend(LinearBackend):
    """Float reference: ``x @ (W8 * sw)^T + b`` without activation quantization (quality baseline).

    Args:
        pack: Model pack.
        ring: Kept for interface symmetry (unused for the arithmetic).
        device: Device of the float GEMMs.
        dtype: Weight/compute dtype (float32 default).
    """

    def __init__(self, pack: object, ring: RingParams, device: str = "cpu", *, dtype: torch.dtype = torch.float32,
                 batch_rows: bool = False):
        self.pack = pack
        self.ring = ring
        self.device = torch.device(device)
        self.dtype = dtype
        #: Concatenate sequences' rows into one GEMM per linear (draft models: faster; float results
        #: then depend on the batch, which is fine for drafts but not for a reference).
        self.batch_rows = batch_rows
        self.stats = BackendStats()

        def conv(lw: Any, dev: torch.device) -> _FloatWeight:
            w = (lw.w8.to(torch.float32) * lw.scale.to(torch.float32)[:, None]).to(dtype)
            b = None if lw.bias is None else lw.bias.to(torch.float32).to(dev)
            return _FloatWeight(w.t().contiguous().to(dev), b, int(lw.w8.shape[1]))

        self._weights = _WeightCache(pack, self.device, conv)

    def forward(self, call: LinearCall) -> torch.Tensor:
        """Float GEMM plus bias."""
        w: _FloatWeight = self._weights.get(call.id)
        x = call.x
        if x.ndim != 2 or x.shape[1] != w.in_features:
            raise ShapeError(f"{call.id}: x must be (T, {w.in_features}), got {tuple(x.shape)}")
        y = (x.to(self.device, self.dtype) @ w.wt).to(torch.float32)
        if w.bias is not None:
            y = y + w.bias
        self.stats.add(calls=1, rows=int(x.shape[0]))
        return y.to(x.device)

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """One GEMM per linear for all sequences when ``batch_rows`` (weights read once per batch)."""
        if self.batch_rows:
            return _grouped_forward(self, calls)
        return [self.forward(c) for c in calls]


def _weight_only_int8(w8: torch.Tensor, scale: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """``x -> x @ (w8 * scale)^T`` with int8 weights and float activations (``None`` if unsupported).

    Weight-only: activations stay float, so activation outliers (common in Qwen/Llama) do not hurt
    accuracy, unlike per-tensor dynamic activation quantization.
    """
    fn = getattr(torch, "_weight_int8pack_mm", None)
    if fn is None:
        return None
    w = w8.contiguous()
    sc = scale.to(torch.float32).contiguous()
    try:
        fn(torch.zeros((1, int(w.shape[1])), dtype=torch.float32), w, sc)
    except Exception:
        return None
    return lambda x: fn(x.to(torch.float32).contiguous(), w, sc)


class Int8DraftBackend(LinearBackend):
    """Fast approximate CPU backend for a speculative-decoding *draft* model.

    Uses the pack's own int8 weights and per-row scales with float activations (weight-only int8):
    a quarter of the float32 weight traffic, which bounds draft speed on CPU. Its numerics differ
    slightly from :class:`FloatBackend`, which is fine for a draft: drafts are only proposals, and
    every emitted token is chosen by the masked target. Falls back to float when unsupported.
    """

    def __init__(self, pack: object, ring: RingParams):
        self.pack = pack
        self.ring = ring
        self.device = torch.device("cpu")
        self.stats = BackendStats()

        def conv(lw: Any, dev: torch.device) -> Any:
            fn = _weight_only_int8(lw.w8, lw.scale)
            b = None if lw.bias is None else lw.bias.to(torch.float32)
            if fn is not None:
                return ("q", fn, b, int(lw.w8.shape[1]))
            w = lw.w8.to(torch.float32) * lw.scale.to(torch.float32)[:, None]
            return ("f", w.t().contiguous(), b, int(lw.w8.shape[1]))

        self._weights = _WeightCache(pack, self.device, conv)

    @staticmethod
    def available() -> bool:
        return _weight_only_int8(torch.ones((8, 8), dtype=torch.int8), torch.ones(8)) is not None

    @staticmethod
    def faster_than_float(pack: Any, reps: int = 5) -> bool:
        """Time int8 weight-only vs float32 on the pack's largest first-layer linear (1 row, as in
        drafting). The int8 kernel's speed varies a lot between CPUs (slower than float32 on some)."""
        import time as _t

        specs = [sp for sp in pack.linear_specs() if sp.id.layer == 0]
        if not specs:
            return False
        sp = max(specs, key=lambda x: x.out_features * x.in_features)
        lw = pack.load_linear(sp.id)
        fn = _weight_only_int8(lw.w8, lw.scale)
        if fn is None:
            return False
        wt = (lw.w8.to(torch.float32) * lw.scale.to(torch.float32)[:, None]).t().contiguous()
        x = torch.randn(1, int(lw.w8.shape[1]))

        def best(f: Callable[[], Any]) -> float:
            f()
            times = []
            for _ in range(reps):
                t0 = _t.perf_counter()
                f()
                times.append(_t.perf_counter() - t0)
            return min(times)

        return best(lambda: fn(x)) < 0.8 * best(lambda: x @ wt)

    def forward(self, call: LinearCall) -> torch.Tensor:
        kind, w, b, n = self._weights.get(call.id)
        x = call.x
        if x.ndim != 2 or x.shape[1] != n:
            raise ShapeError(f"{call.id}: x must be (T, {n}), got {tuple(x.shape)}")
        xf = x.to("cpu", torch.float32)
        y: torch.Tensor = w(xf) if kind == "q" else xf @ w
        if b is not None:
            y = y + b
        self.stats.add(calls=1, rows=int(x.shape[0]))
        return y.to(x.device)

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """One GEMM per linear for all sequences (weights read once per batch)."""
        return _grouped_forward(self, calls)


def int8_lm_head(weight: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """A weight-only int8 ``h -> h @ weight^T`` for a draft model's output layer (``None`` if unavailable)."""
    w = weight.to(torch.float32)
    sc = w.abs().amax(dim=1) / 127.0
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    w8 = torch.clamp(torch.round(w / sc[:, None]), -127, 127).to(torch.int8)
    return _weight_only_int8(w8, sc)


# ---- masked backend ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class _MaskedMeta:
    """What the gateway needs per linear: no int8 weights, only scales, bias and shapes."""

    scale: torch.Tensor
    bias: torch.Tensor | None
    in_features: int
    out_features: int
    wnorm: float | None = None  # capped rings only: max W8 row norm (weights read once, then dropped)


@dataclass
class _Chunk:
    """One wire item: contiguous rows of one call that live in a single mask extent."""

    call_index: int
    row0: int
    rows: int
    lid: LinearId
    x_hat: torch.Tensor | None  # None when only the packed payload is kept (native lean path)
    wr: torch.Tensor
    u: np.ndarray | None  # uint64/uint32 (c, m) read-only view, or None when this chunk is not checked
    uw: np.ndarray | None  # (c, n)
    payload: bytes | None = None  # packed x_hat
    token: Any = None  # names the check vectors (equal tokens: identical u/uw); None = unknown
    wr_wiped: bool = False


def _wipe(t: torch.Tensor) -> None:
    """Zero a buffer in place. CPU tensors go through numpy: torch's multithreaded fill is up to
    ~10x slower for these sizes when the CPU is busy (OpenMP threads contend), same result."""
    if t.device.type == "cpu" and t.is_contiguous():
        t.numpy().fill(0)
    else:
        t.zero_()


def _wr_int64(wr: torch.Tensor) -> np.ndarray:
    """``W r`` as an int64 array: the tensor's own buffer, or a copy of a uint32 one (zero it after use)."""
    a = wr.numpy()
    return a.astype(np.int64) if a.dtype == np.uint32 else a


@dataclass
class _Prepared:
    calls: Sequence[LinearCall]
    chunks: list[_Chunk]
    sx: list[np.ndarray]  # float64 (T,) per call
    metas: list[_MaskedMeta]
    request: Any  # ComputeRequest, or None when every call has zero rows
    t_prepare: float

    def wipe(self) -> None:
        """Zero the ``W r`` buffers (secret-derived) of every chunk; idempotent."""
        for ch in self.chunks:
            if not ch.wr_wiped:
                _wipe(ch.wr)
                ch.wr_wiped = True


class MaskedBackend(LinearBackend):
    """One-time-masked remote compute.

    Per call: quantize ``x`` per token -> take the next ``T`` unused mask rows of this session's
    reservation for ``call.id`` -> ``x_hat = xq + r`` -> (optional outbound noise monitor) ->
    ``ComputeClient.compute`` -> ``yq = centered(y_hat - W r)`` with range check -> Freivalds check
    (always, or with probability ``freivalds_rate``) -> rescale + bias.

    The mask reservation is taken from ``call.session`` (SPEC 9.4); ``mask_source`` is used when a
    call carries no session (single-session tools). Calls are split at segment-extent boundaries
    because Freivalds vectors differ per segment. All calls of one :meth:`forward_many` go out as a
    single ``ComputeRequest`` (one round trip), with items split by ``max_rows_per_item``.

    Before any mask row is taken the backend checks that every reservation has enough rows left,
    so a :class:`MaskExhaustedError` never wastes masks of other sessions in the same batch.

    Args:
        pack: Model pack (only scales, biases and shapes are kept; int8 weights are not needed).
        mask_source: Default reservation for calls without ``session`` (optional).
        client: :class:`pgw.wire.ComputeClient` or compatible (``compute_sync`` / async ``compute``).
        ring: Ring parameters; must match the mask store segment ring.
        fingerprint: Model fingerprint sent with every request.
        freivalds_rate: Fraction of calls checked (1.0 = every call; SPEC default).
        noise_monitor: Optional :class:`pgw.observability.OutboundNoiseMonitor`.
        max_rows_per_item: Split items above this many rows (SPEC 6.5 default 8192).
        deadline_ms: Optional per-request deadline sent to the service.
        on_integrity_failure: Callback ``fn(error)`` (e.g. trip the gateway circuit breaker).
        min_soundness_bits: Refuse (before any mask row is taken) segments whose Freivalds vectors
            give fewer bits of soundness for a linear (:func:`pgw.core.freivalds.soundness_bits`).
            The gateway passes its ``min_soundness_bits`` setting (default 20); 0 disables.

    Raises (from forward):
        MaskExhaustedError, IntegrityError, FingerprintMismatchError, ServiceUnavailableError.
    """

    def __init__(
        self,
        pack: object,
        mask_source: object,
        client: object,
        ring: RingParams,
        fingerprint: str,
        *,
        freivalds_rate: float = 1.0,
        noise_monitor: object | None = None,
        max_rows_per_item: int = 8192,
        deadline_ms: int | None = None,
        on_integrity_failure: Any = None,
        min_soundness_bits: float = 0.0,
    ):
        if not 0.0 <= freivalds_rate <= 1.0:
            raise ConfigError("freivalds_rate must be in [0, 1]")
        if freivalds_rate < 1.0:
            log.warning("Freivalds checks sampled at rate %.3f (SPEC default is 1.0)", freivalds_rate)
        if max_rows_per_item <= 0:
            raise ConfigError("max_rows_per_item must be positive")
        pack_fp = getattr(pack, "fingerprint", None)
        if isinstance(pack_fp, str) and pack_fp != fingerprint:
            raise FingerprintMismatchError(
                "MaskedBackend fingerprint does not match the pack",
                details={"pack": pack_fp[:12], "given": fingerprint[:12]},
            )
        self.pack = pack
        self.mask_source = mask_source
        self.client = client
        self.ring = ring
        self._u32 = ring.k == 32  # uint32 masking/unmasking fast path (bit-identical; K=32 only)
        self._native_consts: dict[str, tuple[np.ndarray, np.ndarray | None, int]] = {}
        self.fingerprint = fingerprint
        self.freivalds_rate = freivalds_rate
        self.noise_monitor = noise_monitor
        self.max_rows_per_item = max_rows_per_item
        self.deadline_ms = deadline_ms
        self.on_integrity_failure = on_integrity_failure
        if min_soundness_bits < 0:
            raise ConfigError("min_soundness_bits must be >= 0")
        self.min_soundness_bits = float(min_soundness_bits)
        self._soundness_memo: dict[tuple[int, int], float] = {}
        #: Check-vector tokens (segment, linear) already found sound enough.
        self._sound_ok: set[Any] = set()
        #: Reservations whose ring and fingerprint were validated (weak: never keeps one alive).
        self._res_ok: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
        self._native_prepare = native_available("quantize_mask_pack")
        self._native_finish = native_available("finish_chunk")
        self._native_group = native_available("finish_group")
        self.stats = BackendStats()
        #: Calls with at most this many rows in total count as decode in ``stats.decode_*`` (the
        #: engine's ``inline_max_rows`` default; the runtime may set it to its configured value).
        self.decode_max_rows = 8
        #: ``row_norm_sq_max`` per linear key from :meth:`warm_weight_norms` (packs without the field).
        self._norm_sq: dict[str, int] = {}
        #: Full int8 weights read on a request path to learn a norm (0 with norms in the manifest or
        #: after :meth:`warm_weight_norms`; anything else is a cold-start penalty worth alerting on).
        self.hot_weight_loads = 0

        def conv(lw: Any, dev: torch.device) -> _MaskedMeta:
            n, m = int(lw.in_features), int(lw.out_features)
            ring.check_in_features(n)
            wnorm = None
            if ring.cap:  # capped rings bound each call's outputs with the largest weight-row norm
                wnorm = math.sqrt(float(self._norm_sq_of(lw)))
            return _MaskedMeta(
                lw.scale.to(torch.float32).clone(),
                None if lw.bias is None else lw.bias.to(torch.float32).clone(),
                n,
                m,
                wnorm,
            )

        # Only scales, bias and shapes: never read the int8 weights the gateway does not need.
        load_meta = getattr(pack, "load_linear_meta", None)
        loader = load_meta if callable(load_meta) else None
        self._meta = _WeightCache(pack, torch.device("cpu"), conv, loader=loader)

    # ---- weight-row norms (capped rings) --------------------------------------------------------
    def _norm_sq_of(self, lw: Any) -> int:
        """``row_norm_sq_max`` of one linear: the manifest value, a warmed value, the weight the loader
        already returned (in-memory packs), or, last resort, one read of the full int8 weight."""
        from pgw.modelpack.loader import row_norm_sq_max

        sq = getattr(lw, "row_norm_sq_max", None)
        if sq is None:
            sq = self._norm_sq.get(lw.id.key)
        if sq is None:
            w8 = getattr(lw, "w8", None)
            if isinstance(w8, torch.Tensor):
                sq = row_norm_sq_max(w8)
            else:
                self.hot_weight_loads += 1
                if self.hot_weight_loads == 1:
                    log.warning("capped ring: reading int8 weights on a request to learn row norms "
                                "(pack without row_norm_sq_max and warm_weight_norms() not run)")
                full = self.pack.load_linear(lw.id, device="cpu")  # type: ignore[attr-defined]
                sq = row_norm_sq_max(full.w8)
                del full
        return int(sq)

    def warm_weight_norms(self, *, cache_dir: str | Path | None = None) -> int:
        """Make every linear's metadata (and, for capped rings, weight-row norm) known before serving.

        Packs that carry ``row_norm_sq_max`` in the manifest need no weights at all. For older packs
        the norms are read from a sidecar ``wnorm.<fingerprint>.json`` (next to the pack, or in
        ``cache_dir`` / ``$PGW_WNORM_CACHE_DIR`` / ``~/.cache/pgw/wnorm`` when the pack directory
        is read-only), else computed once from the int8 weights (chunked, exact int64) and written
        to that sidecar, so the next start reads no weights. Call it at startup, in a worker thread,
        before the model is marked ready; requests afterwards never read a weight.

        Trust: a sidecar is trusted exactly like the pack directory. It is only checked for format,
        version, fingerprint and the manifest bound ``0 <= v <= 127^2 * in_features``, not against the
        weights, so the cache directory must be writable only by the service account (like the pack
        dir). ``pgw doctor --pack P --deep`` recomputes the norms and compares them with the manifest
        and any sidecar in the pack or default cache directory (install-time check).

        Returns:
            The number of linears whose norm was computed from the weights in this call.
        """
        from pgw.modelpack.loader import row_norm_sq_max

        computed = 0
        manifest = getattr(self.pack, "manifest", None)
        has_meta = callable(getattr(self.pack, "load_linear_meta", None))
        if self.ring.cap and manifest is not None and has_meta:
            missing = [e for e in manifest.linears if e.row_norm_sq_max is None and e.id not in self._norm_sq]
            if missing:
                found = _read_norm_sidecar(self.pack, cache_dir)
                if found is not None and all(e.id in found for e in missing):
                    for e in missing:
                        self._norm_sq[e.id] = found[e.id]
                else:
                    t0 = time.perf_counter()
                    norms: dict[str, int] = {}
                    for e in manifest.linears:
                        lw = self.pack.load_linear(e.id, device="cpu")  # type: ignore[attr-defined]
                        norms[e.id] = row_norm_sq_max(lw.w8)
                        del lw
                    computed = len(norms)
                    for e in missing:
                        self._norm_sq[e.id] = norms[e.id]
                    where = _write_norm_sidecar(self.pack, norms, cache_dir)
                    log.info("computed weight-row norms of %d linears in %.1f s (sidecar: %s)",
                             computed, time.perf_counter() - t0, where or "not written")
        self._meta.preload()
        return computed

    # ---- helpers -------------------------------------------------------------------------------
    def _reservation(self, call: LinearCall) -> Any:
        res = call.session if call.session is not None else self.mask_source
        if res is None:
            raise ConfigError(f"{call.id}: masked call without a mask reservation (session)")
        try:
            if self._res_ok.get(id(res)) is res:  # validated before (ring and fingerprint never change)
                return res
        except TypeError:  # pragma: no cover - defensive
            pass
        if not hasattr(res, "take"):
            raise ConfigError(f"{call.id}: masked call without a mask reservation (session)")
        res_ring = getattr(res, "ring", None)
        if res_ring is not None:
            name = res_ring if isinstance(res_ring, str) else getattr(res_ring, "name", None)
            if name is not None and name != self.ring.name:
                raise ConfigError(f"mask reservation ring {name} != backend ring {self.ring.name}")
        res_fp = getattr(res, "fingerprint", None)
        if isinstance(res_fp, str) and res_fp != self.fingerprint:
            raise FingerprintMismatchError(
                "mask reservation belongs to a different model fingerprint",
                details={"reservation": res_fp[:12], "backend": self.fingerprint[:12]},
            )
        with contextlib.suppress(TypeError):  # not weak-referenceable: validated on every call
            self._res_ok[id(res)] = res
        return res

    def _checks_u64(self, res: Any, lid: Any) -> tuple[np.ndarray, np.ndarray]:
        """The current extent's check vectors as uint64 arrays (zero-copy where supported)."""
        fn = getattr(res, "checks_u64", None)
        if fn is not None:
            return fn(lid)  # type: ignore[no-any-return]
        u, uw = res.checks(lid)
        return checks_to_u64(u, uw, self.ring)

    def _current_checks(self, res: Any, lid: Any) -> tuple[Any, np.ndarray, np.ndarray]:
        """``(token, u, uw)`` of the current extent; ``token`` names the vectors (``None``: unknown)."""
        fn = getattr(res, "checks_compact", None)
        if fn is not None:
            return fn(lid)  # type: ignore[no-any-return]
        return (None, *self._checks_u64(res, lid))

    def _check_soundness(self, lid: Any, n: int, u: Any, token: Any = None) -> None:
        """Raise if the current extent's check vectors give less than ``min_soundness_bits``."""
        if self.min_soundness_bits <= 0:
            return
        if token is not None and token in self._sound_ok:
            return
        checks = 0 if u is None else int(u.shape[0])
        bits = self._soundness_bits(n, checks)
        if bits < self.min_soundness_bits:
            raise MaskStoreError(
                f"{lid}: mask segment has {checks} Freivalds vector(s) = {bits:.0f} bits of soundness for "
                f"n={n} at {self.ring.name}; at least {self.min_soundness_bits:.0f} are required "
                "(regenerate masks with more checks)",
                details={"linear": str(lid), "checks": checks, "min_soundness_bits": self.min_soundness_bits},
            )
        if token is not None:
            if len(self._sound_ok) > 65536:  # bounded: tokens are (segment, linear) pairs
                self._sound_ok.clear()
            self._sound_ok.add(token)

    def _soundness_bits(self, n: int, checks: int) -> float:
        key = (n, checks)
        hit = self._soundness_memo.get(key)
        if hit is None:
            from pgw.core.freivalds import soundness_bits

            hit = self._soundness_memo[key] = float(soundness_bits(self.ring, n, checks))
        return hit

    def _should_check(self) -> bool:
        if self.freivalds_rate >= 1.0:
            return True
        if self.freivalds_rate <= 0.0:
            return False
        return secrets.randbelow(1 << 30) < int(self.freivalds_rate * (1 << 30))

    def _precheck(self, calls: Sequence[LinearCall]) -> tuple[list[Any], list[_MaskedMeta]]:
        """Resolve reservations, validate shapes and fail fast before any mask row is taken.

        Raises :class:`MaskExhaustedError` if any reservation would run out, so one exhausted
        session never consumes (wastes) rows of the other sessions in the same batch.
        """
        reservations = [self._reservation(c) for c in calls]
        metas: list[_MaskedMeta] = []
        mget = self._meta.get
        # (id(reservation), linear key) -> [reservation, linear id, in_features, rows needed]
        need: dict[tuple[int, str], list[Any]] = {}
        for c, res in zip(calls, reservations, strict=True):
            meta: _MaskedMeta = mget(c.id)
            x = c.x
            if x.ndim != 2 or x.shape[1] != meta.in_features:
                raise ShapeError(f"{c.id}: x must be (T, {meta.in_features}), got {tuple(x.shape)}")
            metas.append(meta)
            key = (id(res), c.id.key)
            hit = need.get(key)
            if hit is None:
                need[key] = [res, c.id, meta.in_features, int(x.shape[0])]
            else:
                hit[3] += int(x.shape[0])
        # Growing reservations (gateway windows) extend here, before any row of the batch is taken.
        for res, lid, _, rows in need.values():
            ensure = getattr(res, "ensure", None)
            if ensure is not None:
                ensure(lid, rows)
        for res, lid, _, rows in need.values():
            left = int(res.remaining(lid))
            if left < rows:
                raise MaskExhaustedError(
                    "session mask reservation exhausted",
                    details={"linear": lid.key, "needed": rows, "remaining": left},
                )
        if self.min_soundness_bits > 0:
            for res, lid, n, _ in need.values():
                token, u, _uw = self._current_checks(res, lid)
                self._check_soundness(lid, n, u, token)
        return reservations, metas

    def _native_prepare_ok(self, calls: Sequence[LinearCall]) -> bool:
        if not self._native_prepare or self.ring.k % 8 or self.ring.k > 56:
            return False
        if self.ring.cap:  # capped rings need each linear's weight-row norm
            return all(self._meta.get(c.id).wnorm is not None for c in calls)
        return True

    def _prepare(self, calls: Sequence[LinearCall]) -> _Prepared:
        if self._native_prepare_ok(calls):
            return self._prepare_native(calls)
        return self._prepare_python(calls)

    def _take(self, res: Any, lid: LinearId, max_rows: int, want_checks: bool) -> tuple[int, Any, Any, Any]:
        """``(rows, r, W r, (token, u, uw) | None)``: the next rows of one extent (at most ``max_rows``)."""
        fn = getattr(res, "take_checked", None)
        if fn is not None:
            return fn(lid, max_rows, want_checks)  # type: ignore[no-any-return]
        boundary = int(res.rows_until_boundary(lid))
        if boundary <= 0:
            raise MaskExhaustedError("mask reservation has no rows left", details={"linear": lid.key})
        rows = min(max_rows, boundary)
        # vectors of the extent the next rows come from: read before take()
        chk = (None, *self._checks_u64(res, lid)) if want_checks else None
        r, wr = res.take(lid, rows)
        return rows, r, wr, chk

    def _prepare_native(self, calls: Sequence[LinearCall]) -> _Prepared:
        """:meth:`_prepare_python` with quantize + mask + pack fused in ``pgw_fast`` (bit-identical)."""
        from pgw.wire.messages import ComputeItem, ComputeRequest

        t0 = time.perf_counter()
        reservations, metas = self._precheck(calls)
        # Same guarantee as the Python path: a non-finite activation in a later call must not
        # consume mask rows of earlier calls, so check every call before taking any row.
        xs: list[np.ndarray] = []
        for call in calls:
            x = call.x
            if x.dtype != torch.float32 or x.device.type != "cpu":
                x = x.to("cpu", torch.float32)
            xn = x.detach().numpy()
            if not xn.flags.c_contiguous:
                xn = np.ascontiguousarray(xn)
            if xn.size and not np.isfinite(xn).all():
                raise RingOverflowError("non-finite activation values cannot be quantized")
            xs.append(xn)
        ring = self.ring
        kbits, abits, cap = ring.k, ring.a, int(ring.cap or 0)
        noise = self.noise_monitor
        # x_hat is only materialized where something reads it (an old finish kernel, or the noise
        # monitor on the calls it samples); the Freivalds check reads the request payload (the same
        # values, packed). Monitors with plan_call/guard_packed run their every-call checks (range,
        # duplicate rows) on the packed payload of non-sampled calls.
        group_finish = self._native_group
        plan_call: Any = None
        guard_packed: Any = None
        if noise is not None and group_finish:
            plan_call = getattr(noise, "plan_call", None)
            guard_packed = getattr(noise, "guard_packed", None)
            if plan_call is None or guard_packed is None:
                plan_call = guard_packed = None
        qmp = native.fast.quantize_mask_pack
        want_soundness = self.min_soundness_bits > 0
        max_item = self.max_rows_per_item
        sx_parts: list[list[np.ndarray]] = [[] for _ in calls]
        chunks: list[_Chunk] = []
        try:
            for ci, (call, res, meta, xn) in enumerate(zip(calls, reservations, metas, xs, strict=True)):
                t = int(xn.shape[0])
                lid = call.id
                wn = float(meta.wnorm or 0.0)
                row = 0
                while row < t:
                    check = self._should_check()
                    # The sampling draw comes before any row is taken (an alarmed monitor burns none).
                    sampled = bool(plan_call()) if plan_call is not None else noise is not None
                    want_xhat = not group_finish or sampled
                    rows, r, wr, chk = self._take(res, lid, min(t - row, max_item), check or want_soundness)
                    token = u = uw = None
                    try:
                        if chk is not None:
                            token, u, uw = chk
                            self._check_soundness(lid, meta.in_features, u, token)
                            if not check:
                                token = u = uw = None
                        if tuple(r.shape) != (rows, meta.in_features) or tuple(wr.shape) != (rows, meta.out_features):
                            raise ShapeError(
                                f"{lid}: mask shapes {tuple(r.shape)}/{tuple(wr.shape)} do not match the linear"
                            )
                        r_np = r.numpy()
                        try:
                            if want_xhat:
                                payload, xh, sx = qmp(xn[row : row + rows], r_np, kbits, abits, cap, wn)
                            else:
                                payload, xh, sx = qmp(xn[row : row + rows], r_np, kbits, abits, cap, wn, False)
                        except ValueError as exc:
                            if str(exc).startswith("ring_overflow"):
                                raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
                            raise ShapeError(str(exc)) from exc
                        x_hat = None if xh is None else torch.from_numpy(xh)
                        if noise is not None:
                            if guard_packed is not None and not sampled:
                                guard_packed(payload, rows, meta.in_features)
                            elif x_hat is not None:  # xq = centered(x_hat - r), for its checks
                                xq = center(x_hat - r.to(torch.int64), ring)
                                try:
                                    if plan_call is not None:
                                        noise.guard(x_hat, xq, planned=True)  # type: ignore[attr-defined]
                                    else:
                                        noise.guard(x_hat, xq)  # type: ignore[attr-defined]
                                finally:
                                    _wipe(xq)
                            if group_finish:
                                x_hat = None  # the group finish reads the payload: every chunk alike
                    except BaseException:
                        _wipe(wr)
                        raise
                    finally:
                        _wipe(r)  # one-time pad: never keep it around
                    chunks.append(_Chunk(ci, row, rows, lid, x_hat, wr, u, uw, payload, token))
                    sx_parts[ci].append(sx)
                    row += rows
        except BaseException:
            for ch in chunks:
                _wipe(ch.wr)
            raise
        sxs = [np.concatenate(p) if len(p) > 1 else (p[0] if p else np.zeros(0, dtype=np.float64))
               for p in sx_parts]
        request = None
        if chunks:
            request = ComputeRequest(
                request_id=uuid.uuid4().hex,
                fingerprint=self.fingerprint,
                ring=self.ring.name,
                items=tuple(ComputeItem(ch.lid.key, ch.rows, ch.payload) for ch in chunks),  # type: ignore[arg-type]
                deadline_ms=self.deadline_ms,
            )
        return _Prepared(calls, chunks, sxs, metas, request, time.perf_counter() - t0)

    def _prepare_python(self, calls: Sequence[LinearCall]) -> _Prepared:
        from pgw.wire.messages import ComputeItem, ComputeRequest

        t0 = time.perf_counter()
        reservations, metas = self._precheck(calls)
        # Quantize everything first: a non-finite activation in a later call must not consume
        # mask rows of earlier calls.
        quantized: list[torch.Tensor] = []
        sxs: list[np.ndarray] = []
        for call in calls:
            wn = self._meta.get(call.id).wnorm if self.ring.cap else None
            xq, sx = quantize_activations(call.x.detach().to("cpu"), self.ring, wn)
            quantized.append(xq)
            sxs.append(np.ascontiguousarray(sx.numpy(), dtype=np.float64))
        chunks: list[_Chunk] = []
        want_soundness = self.min_soundness_bits > 0
        try:
            for ci, (call, res, meta, xq) in enumerate(zip(calls, reservations, metas, quantized, strict=True)):
                t = int(xq.shape[0])
                row = 0
                while row < t:
                    check = self._should_check()
                    rows, r, wr, chk = self._take(res, call.id, min(t - row, self.max_rows_per_item),
                                                  check or want_soundness)
                    token = u = uw = None
                    try:
                        if chk is not None:
                            token, u, uw = chk
                            self._check_soundness(call.id, meta.in_features, u, token)
                            if not check:
                                token = u = uw = None
                        if tuple(r.shape) != (rows, meta.in_features) or tuple(wr.shape) != (rows, meta.out_features):
                            raise ShapeError(
                                f"{call.id}: mask shapes {tuple(r.shape)}/{tuple(wr.shape)} do not match the linear"
                            )
                        if self._u32:
                            x32, payload = mask_pack_u32(xq[row : row + rows].numpy(), r.numpy(), self.ring)
                            x_hat = torch.from_numpy(x32.astype(np.int64))
                        else:
                            payload = None
                            x_hat = mask_activations(xq[row : row + rows], r.to(torch.int64), self.ring)
                    except BaseException:
                        _wipe(wr)
                        raise
                    finally:
                        _wipe(r)  # one-time pad: never keep it around
                    chunk = _Chunk(ci, row, rows, call.id, x_hat, wr, u, uw, payload, token)
                    chunks.append(chunk)
                    if self.noise_monitor is not None:
                        self.noise_monitor.guard(x_hat, xq[row : row + rows])  # type: ignore[attr-defined]
                    row += rows
        except BaseException:
            for ch in chunks:
                _wipe(ch.wr)
            raise
        finally:
            for xq in quantized:
                _wipe(xq)
        request = None
        if chunks:
            items = tuple(ComputeItem(ch.lid.key, ch.rows, ch.payload if ch.payload is not None
                                      else pack_ring(ch.x_hat, self.ring))  # type: ignore[arg-type]
                          for ch in chunks)
            request = ComputeRequest(
                request_id=uuid.uuid4().hex,
                fingerprint=self.fingerprint,
                ring=self.ring.name,
                items=items,
                deadline_ms=self.deadline_ms,
            )
        return _Prepared(calls, chunks, sxs, metas, request, time.perf_counter() - t0)

    def _finish(self, prep: _Prepared, response: Any, remote_s: float) -> list[torch.Tensor]:
        t0 = time.perf_counter()
        req = prep.request
        outs: list[list[torch.Tensor]] = [[] for _ in prep.calls]
        bytes_down = 0
        checked = skipped = 0
        use_native = self._native_finish and self.ring.k % 8 == 0 and self.ring.k <= 56
        try:
            if req is not None:
                if getattr(response, "request_id", None) != req.request_id:
                    raise ProtocolError("compute response id does not match the request")
                results = tuple(response.items)
                if len(results) != len(prep.chunks):
                    raise ProtocolError(
                        f"compute response has {len(results)} items, expected {len(prep.chunks)}"
                    )
            else:
                results = ()
            for ch, res in zip(prep.chunks, results, strict=True):
                if res.linear != ch.lid.key or int(res.rows) != ch.rows:
                    raise ProtocolError(
                        "compute response item does not match the request",
                        details={"expected": ch.lid.key, "got": str(res.linear)},
                    )
                bytes_down += len(res.y_hat)
                if ch.u is not None:
                    checked += 1
                else:
                    skipped += 1
            if use_native and self._native_group and all(ch.payload is not None for ch in prep.chunks):
                self._finish_groups_native(prep, results, outs)
            else:
                for ch, res in zip(prep.chunks, results, strict=True):
                    meta: _MaskedMeta = prep.metas[ch.call_index]
                    if use_native:
                        outs[ch.call_index].append(self._finish_chunk_native(prep, ch, meta, res.y_hat))
                    else:
                        outs[ch.call_index].append(self._finish_chunk_python(prep, ch, meta, res.y_hat))
        except IntegrityError as exc:
            self.stats.add(integrity_failures=1)
            log.error(
                "integrity check failed: request=%s code=%s details=%s",
                req.request_id if req is not None else "-", exc.code, exc.details,
            )
            if self.on_integrity_failure is not None:
                try:
                    self.on_integrity_failure(exc)
                except Exception:  # pragma: no cover - callback must not mask the error
                    log.exception("on_integrity_failure callback raised")
            raise
        finally:
            prep.wipe()
        result = []
        for call, meta, parts in zip(prep.calls, prep.metas, outs, strict=True):
            if len(parts) == 1:
                y = parts[0]
            else:
                y = torch.cat(parts, dim=0) if parts else torch.zeros((0, meta.out_features), dtype=torch.float32)
            dev = call.x.device
            result.append(y if dev.type == "cpu" else y.to(dev))
        bytes_up = sum(len(it.x_hat) for it in req.items) if req is not None else 0
        rows = sum(ch.rows for ch in prep.chunks)
        self.stats.add_many(
            len(prep.calls),
            rows,
            1 if req is not None else 0,
            bytes_up,
            bytes_down,
            checked,
            skipped,
            remote_s,
            prep.t_prepare + time.perf_counter() - t0,
            0 < rows <= self.decode_max_rows,
        )
        return result

    def _finish_chunk_python(self, prep: _Prepared, ch: _Chunk, meta: _MaskedMeta, y_hat_bytes: bytes) -> torch.Tensor:
        """Reference per-chunk finish (no native extension): Freivalds, unmask, range check, rescale."""
        if self._u32:
            y_hat = None
        else:
            y_hat = unpack_ring_u64(y_hat_bytes, (ch.rows, meta.out_features), self.ring)
        if ch.u is not None and ch.uw is not None:
            if y_hat is None:
                y_hat = np.frombuffer(y_hat_bytes, dtype="<u4").reshape(ch.rows, meta.out_features).astype(np.uint64)
            if not ch.u.any():
                # The reservation was closed (and its cached vectors wiped) while this call
                # was in flight: an all-zero u would make the check vacuous.
                raise MaskReuseError("mask reservation closed during a compute call")
            assert ch.x_hat is not None
            ok = freivalds_check_u64(ch.x_hat.numpy().view(np.uint64), y_hat, ch.u.astype(np.uint64, copy=False),
                                     ch.uw.astype(np.uint64, copy=False), self.ring)
            if not bool(ok.all()):
                raise IntegrityError(
                    "compute service result failed the Freivalds check",
                    details={"linear": ch.lid.key, "bad_rows": int((~ok).sum()), "rows": ch.rows},
                )
        wr = _wr_int64(ch.wr)
        try:
            if self._u32:
                yq = torch.from_numpy(unmask_outputs_u32(y_hat_bytes, wr, self.ring, meta.in_features))
            else:
                assert y_hat is not None
                yq = torch.from_numpy(unmask_outputs_u64(y_hat, wr, self.ring, meta.in_features))
        finally:
            wr.fill(0)
        _wipe(ch.wr)
        ch.wr_wiped = True
        sx = torch.from_numpy(prep.sx[ch.call_index][ch.row0 : ch.row0 + ch.rows])
        return dequantize_outputs(yq, sx, meta.scale, meta.bias, torch.float32)

    def _consts(self, lid: LinearId, meta: _MaskedMeta) -> tuple[np.ndarray, np.ndarray | None, int]:
        key = lid.key
        consts = self._native_consts.get(key)
        if consts is None:
            consts = self._native_consts[key] = (
                np.ascontiguousarray(meta.scale.numpy(), dtype=np.float32),
                None if meta.bias is None else np.ascontiguousarray(meta.bias.numpy(), dtype=np.float32),
                int(self.ring.output_bound(meta.in_features)),
            )
        return consts

    def _raise_status(self, status: int, bad: int, lid: LinearId, rows: int, bound: int) -> None:
        if status == 3:
            raise MaskReuseError("mask reservation closed during a compute call")
        if status == 1:
            raise IntegrityError(
                "compute service result failed the Freivalds check",
                details={"linear": lid.key, "bad_rows": int(bad), "rows": rows},
            )
        if status == 2:
            raise IntegrityError(
                "decoded linear output outside the representable range (corrupt or tampered result)",
                details={"bound": bound, "ring": self.ring.name},
            )

    def _finish_groups_native(self, prep: _Prepared, results: Sequence[Any], outs: list[list[torch.Tensor]]) -> None:
        """Finish all chunks with ``pgw_fast.finish_group``: chunks of one linear whose rows come from
        extents with the same check vectors (same segment; typically one chunk per session) are
        verified, unmasked and rescaled in one pass that reads ``u``/``uW`` once. Every chunk keeps
        its own ``W r``; nothing of one session's masks is applied to another's rows."""
        groups: dict[tuple[Any, ...], list[int]] = {}
        if len(prep.chunks) == 1:  # single-session decode
            groups[()] = [0]
        else:
            for i, ch in enumerate(prep.chunks):
                if ch.u is None:
                    gk: tuple[Any, ...] = (ch.lid.key, False)
                elif ch.token is not None:
                    gk = (ch.lid.key, True, ch.token)
                else:
                    gk = (ch.lid.key, True, i)  # vectors of unknown origin: never shared
                idxs = groups.get(gk)
                if idxs is None:
                    groups[gk] = [i]
                else:
                    idxs.append(i)
        fin = native.fast.finish_group
        k = self.ring.k
        for idxs in groups.values():
            first = prep.chunks[idxs[0]]
            meta: _MaskedMeta = prep.metas[first.call_index]
            scale, bias, bound = self._consts(first.lid, meta)
            checked = first.u is not None and first.uw is not None
            parts = []
            in_place: list[int] = []  # chunks whose own W r buffer goes to the kernel (it zeroes them)
            for i in idxs:
                ch = prep.chunks[i]
                sx = prep.sx[ch.call_index]
                if ch.row0 or ch.rows != sx.shape[0]:
                    sx = sx[ch.row0 : ch.row0 + ch.rows]
                wn = ch.wr.numpy()
                if wn.dtype in (np.int64, np.uint32) and wn.flags.c_contiguous and wn.flags.writeable:
                    in_place.append(i)
                else:  # e.g. a strided test double: a copy, zeroed by the kernel; prep.wipe() zeroes ch.wr
                    wn = np.ascontiguousarray(wn, dtype=np.int64)
                parts.append((results[i].y_hat, wn, ch.payload if checked else None, sx))
            try:
                out, status, bad = fin(parts, first.u if checked else None, first.uw if checked else None,
                                       scale, bias, k, bound, True)
            except ValueError as exc:
                raise ProtocolError(f"compute response item is malformed: {exc}") from exc
            for i in in_place:  # the kernel zeroed every W r it was given (any returned status)
                prep.chunks[i].wr_wiped = True
            if status in (1, 2) and checked and first.u is not None and not first.u.any():
                status = 3  # the reservation was closed (vectors wiped) while the kernel ran: not tampering
            if status:
                self._raise_status(status, bad, first.lid, sum(prep.chunks[i].rows for i in idxs), bound)
            ot = torch.from_numpy(out)
            if len(idxs) == 1:
                outs[first.call_index].append(ot)
                continue
            off = 0
            for i in idxs:
                ch = prep.chunks[i]
                outs[ch.call_index].append(ot[off : off + ch.rows])
                off += ch.rows

    def _finish_chunk_native(self, prep: _Prepared, ch: _Chunk, meta: _MaskedMeta, y_hat: bytes) -> torch.Tensor:
        """Freivalds, unmask, range check and dequantize one chunk in ``pgw_fast`` (bit-identical)."""
        scale, bias, bound = self._consts(ch.lid, meta)
        sx = np.ascontiguousarray(prep.sx[ch.call_index][ch.row0 : ch.row0 + ch.rows], dtype=np.float64)
        u = uw = x_hat = None
        if ch.u is not None and ch.uw is not None:
            u, uw = ch.u.astype(np.uint64, copy=False), ch.uw.astype(np.uint64, copy=False)
            x_hat = ch.x_hat.numpy() if ch.x_hat is not None else None
        checked = u is not None
        wr = _wr_int64(ch.wr)
        try:
            out, status, bad = native.fast.finish_chunk(
                y_hat, wr, x_hat, u, uw, sx, scale, bias, self.ring.k, bound,
            )
        except ValueError as exc:
            raise ProtocolError(f"compute response item is malformed: {exc}") from exc
        finally:
            wr.fill(0)
            _wipe(ch.wr)
            ch.wr_wiped = True
        if status in (1, 2) and checked and ch.u is not None and not ch.u.any():
            status = 3  # the reservation was closed (vectors wiped) while the kernel ran: not tampering
        self._raise_status(status, bad, ch.lid, ch.rows, bound)
        return torch.from_numpy(out)

    @staticmethod
    def _wrap_transport(exc: Exception) -> PGWError:
        if isinstance(exc, PGWError):
            return exc
        return ServiceUnavailableError(f"compute request failed: {type(exc).__name__}")

    # ---- public API ---------------------------------------------------------------------------
    def forward(self, call: LinearCall) -> torch.Tensor:
        """One masked round trip."""
        return self.forward_many([call])[0]

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """All calls in one ``ComputeRequest`` (one round trip)."""
        if not calls:
            return []
        prep = self._prepare(calls)
        if prep.request is None:
            return self._finish(prep, None, 0.0)
        t0 = time.perf_counter()
        try:
            if hasattr(self.client, "compute_sync"):
                response = self.client.compute_sync(prep.request)
            else:
                response = _run_coro_blocking(self.client.compute(prep.request))  # type: ignore[attr-defined]
        except BaseException as exc:
            prep.wipe()
            if isinstance(exc, Exception):
                raise self._wrap_transport(exc) from exc
            raise
        return self._finish(prep, response, time.perf_counter() - t0)

    async def aforward(self, call: LinearCall) -> torch.Tensor:
        """Async single call."""
        return (await self.aforward_many([call]))[0]

    async def aforward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """Async: local crypto in a worker thread, network on the event loop.

        Cancellation (e.g. a client disconnect) wipes the secret-derived buffers of the request;
        the mask rows it took stay consumed and are never reused.
        """
        if not calls:
            return []
        prep = await asyncio.to_thread(self._prepare, calls)
        response, remote_s = await self.send_prepared(prep)
        fut = asyncio.ensure_future(asyncio.to_thread(self._finish, prep, response, remote_s))
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            # The worker thread is still reading the buffers: let it finish (it wipes them itself)
            # instead of zeroing them underneath it, which could turn a cancel into a false
            # integrity failure.
            with contextlib.suppress(BaseException):
                await fut
            prep.wipe()
            raise

    # ---- split-hop interface (engine fuses finish + plaintext step + prepare in one thread hop) ----
    #: The engine checks this flag and then drives :meth:`prepare_calls` / :meth:`send_prepared` /
    #: :meth:`finish_prepared` itself, saving two thread hand-offs per round trip.
    split_hop = True

    def prepare_calls(self, calls: Sequence[LinearCall]) -> _Prepared:
        """Synchronous local half of a round trip (quantize, mask, pack). Wipe with :meth:`discard`."""
        return self._prepare(calls)

    async def send_prepared(self, prep: _Prepared) -> tuple[Any, float]:
        """Network half: send ``prep.request``; returns ``(response, remote_seconds)``.

        On any failure (including cancellation) the prepared secrets are wiped.
        """
        if prep.request is None:
            return None, 0.0
        t0 = time.perf_counter()
        try:
            compute = getattr(self.client, "compute", None)
            if compute is not None and inspect.iscoroutinefunction(compute):
                response = await compute(prep.request)
            else:
                response = await asyncio.to_thread(self.client.compute_sync, prep.request)  # type: ignore[attr-defined]
        except BaseException as exc:
            prep.wipe()
            if isinstance(exc, Exception):
                raise self._wrap_transport(exc) from exc
            raise
        return response, time.perf_counter() - t0

    def finish_prepared(self, prep: _Prepared, response: Any, remote_s: float) -> list[torch.Tensor]:
        """Synchronous local half after the response (verify, unmask, rescale); always wipes ``prep``."""
        return self._finish(prep, response, remote_s)

    @staticmethod
    def discard(prep: _Prepared | None) -> None:
        """Wipe a prepared request that will not be sent or finished."""
        if prep is not None:
            prep.wipe()


def _run_coro_blocking(coro: Any) -> Any:
    """Run a coroutine to completion from synchronous code (no running loop in this thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise ConfigError("synchronous MaskedBackend.forward called inside an event loop; use aforward")


# ---- weight-row norm sidecar (packs written before the manifest carried row_norm_sq_max) ----------
_NORM_SIDECAR_FORMAT = "pgw-wnorm"
_NORM_SIDECAR_VERSION = 1


def _norm_sidecar_dirs(pack: Any, cache_dir: str | Path | None) -> list[Path]:
    dirs: list[Path] = []
    root = getattr(pack, "root", None)
    if isinstance(root, Path):
        dirs.append(root)
    if cache_dir is not None:
        dirs.append(Path(cache_dir))
    else:
        env = os.environ.get("PGW_WNORM_CACHE_DIR")
        if env:
            dirs.append(Path(env))
        else:
            base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
            dirs.append(Path(base) / "pgw" / "wnorm")
    return dirs


def _read_norm_sidecar(pack: Any, cache_dir: str | Path | None) -> dict[str, int] | None:
    """Validated norms from the first readable sidecar for this pack's fingerprint, else ``None``."""
    fp = str(pack.fingerprint)
    widths = {e.id: int(e.in_features) for e in pack.manifest.linears}
    for d in _norm_sidecar_dirs(pack, cache_dir):
        p = d / f"wnorm.{fp}.json"
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        try:
            if doc.get("format") != _NORM_SIDECAR_FORMAT or doc.get("fingerprint") != fp:
                raise ValueError("wrong format or fingerprint")
            version = doc.get("version")
            if not isinstance(version, int) or isinstance(version, bool) or version != _NORM_SIDECAR_VERSION:
                raise ValueError(f"unsupported version {version!r}")
            out: dict[str, int] = {}
            for key, v in doc["row_norm_sq_max"].items():  # same bound as the manifest (127^2 * n)
                n = widths.get(key)
                if n is None or not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 127 * 127 * n:
                    raise ValueError(f"bad entry {key!r}")
                out[key] = v
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            log.warning("ignoring invalid weight-norm sidecar %s: %s", p, exc)
            continue
        return out
    return None


def _write_norm_sidecar(pack: Any, norms: dict[str, int], cache_dir: str | Path | None) -> Path | None:
    """Write the sidecar atomically into the first writable directory; returns its path or ``None``."""
    fp = str(pack.fingerprint)
    body = json.dumps({"format": _NORM_SIDECAR_FORMAT, "version": _NORM_SIDECAR_VERSION, "fingerprint": fp,
                       "row_norm_sq_max": norms}, sort_keys=True)
    for d in _norm_sidecar_dirs(pack, cache_dir):
        p = d / f"wnorm.{fp}.json"
        tmp = d / f".wnorm.{fp}.{uuid.uuid4().hex}.tmp"
        try:
            d.mkdir(parents=True, exist_ok=True)
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, p)
            return p
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
    return None
