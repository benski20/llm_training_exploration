"""Linear backends: how the engine evaluates ``y = W x + b`` for a masked linear group.

* :class:`PlainBackend` (alias :data:`QuantPlainBackend`): exact integer reference. Quantizes the
  activations exactly as the masked protocol does and computes ``xq @ W8^T`` exactly.
* :class:`FloatBackend`: dequantized float weights, no activation quantization (quality baseline).
* :class:`MaskedBackend`: production. One-time-masked remote compute with unmasking, range and
  Freivalds checks. Bit-identical to :class:`PlainBackend` (SPEC 9.3).

All backends return float32 ``(T, m)`` including the bias, computed as
``y = yq * sx[t] * sw[j] + b[j]`` in float64 then cast (SPEC section 5) for the integer backends.

Chat-speed integration in :class:`MaskedBackend` (WP-A4b; every option keeps the bits of the plain
reference and sends the provider nothing new):

* TripEngine: the native trip kernels (``pgw_fast.trip_*``) with constant-time width selection, prefill
  blocks on the native pool, and a deferred Freivalds check joined before the session's next payload
  (and never deferred for the model's last linear, so before logits leave);
* ``sync_trip_ok``: synchronous trips for decode passes when a trip-lane client is configured;
* E2 exact local linears (qkv and o on the gateway, decided once at start-up);
* speculation padding rows (uniform random, no mask taken, dummy finish) and dummy calls.
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
from pgw.core.factory_yield import note_activity as _note_activity
from pgw.core.freivalds import P31, checks_to_u64, freivalds_check_u64, is_prime_token
from pgw.core.masking import mask_activations, mask_pack_u32, unmask_outputs_u32, unmask_outputs_u64
from pgw.core.native import native_available
from pgw.core.packing import pack_ring, unpack_ring_u64
from pgw.core.ring import RingParams, balanced_digits, center, dequantize_outputs, quantize_activations
from pgw.core.ringmm import INT_MM_MAX_INNER, cpu_int8_fast, int8_digit_gemm
from pgw.core.types import LinearId
from pgw.maskstore import store as _store_flags  # TRIP_LEASE_AHEAD (set by the SDK at start-up)

__all__ = [
    "DUMMY",
    "LOCAL_ATTENTION_KINDS",
    "LOCAL_LINEAR_MODES",
    "PAD_ROWS",
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
    #: Freivalds output halves run on the native helper thread and joined later (deferred checks).
    deferred_checks: int = 0
    #: Speculation padding rows sent (uniform random, no mask taken) and dummy (all-padding) calls.
    pad_rows: int = 0
    dummy_calls: int = 0
    #: Calls and rows of linears computed exactly on the gateway (E2 local linears; no round trip).
    local_calls: int = 0
    local_rows: int = 0
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

    def add_local(self, calls: int, rows: int, seconds: float) -> None:
        """``add(local_calls=, local_rows=, local_seconds=)`` without keyword dispatch."""
        with self._lock:
            self.local_calls += calls
            self.local_rows += rows
            self.local_seconds += seconds

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
    #: Each output row is the same bits whatever other rows share the call (per-row activation scales and an
    #: exact integer product). Conversation KV-prefix reuse (:mod:`pgw.engine.prefix_cache`) needs it.
    rows_exact: bool = False

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
    pad_rows = dev.type == "cuda"
    if not pad_rows and ndig > 1:
        # CPU: one GEMM over the digits stacked as rows. Decode GEMVs are memory-bound, so this reads
        # the weight once instead of once per digit (~1.9x at 1 row; same integers, so the same bits).
        y = int8_digit_gemm(torch.cat(digits, 0).to(torch.int8), w8t.t())
        acc = y[:t].clone()
        for i in range(1, ndig):
            acc += y[i * t:(i + 1) * t] << (8 * i)
        return acc
    acc = torch.zeros((t, m), dtype=torch.int64, device=dev)
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

    ``rows_exact``: activations are quantized per row and the product is exact integers.

    Numerically identical to :class:`MaskedBackend` (same quantization, exact integers), so it is
    the reference for end-to-end exactness tests and the backend for ``dev`` mode.

    Args:
        pack: Open :class:`pgw.modelpack.ModelPack` or any :class:`pgw.engine.pack.EnginePack`
            (linears are loaded lazily and cached).
        ring: Ring parameters (determine activation quantization).
        device: Where to run the integer product.
        preload: Load every linear at construction instead of on first use.
    """

    rows_exact = True

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


def _native_int8_linear(w8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, ring: RingParams,
                        threads: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """``x -> dequant(quant(x) @ w8^T)`` through ``pgw_fast.local_linear_into`` (the exact local-linear
    kernel: the target's quantizer, an exact int16 x int8 product, columns spread over the native pool).

    Its outputs are a deterministic function of ``x`` whatever the thread count (exact integer sums) and
    each row depends only on its own input row. ``threads``: 0 = the native pool, 1 = calling thread.
    """
    from pgw.modelpack.loader import row_norm_sq_max

    n, m = int(w8.shape[1]), int(w8.shape[0])
    ring.check_in_features(n)
    w8t = w8.to(torch.int8).t().contiguous().numpy()
    sc = np.ascontiguousarray(scale.to(torch.float32).numpy(), dtype=np.float32)
    bs = None if bias is None else np.ascontiguousarray(bias.to(torch.float32).numpy(), dtype=np.float32)
    wnorm = math.sqrt(float(row_norm_sq_max(w8))) if ring.cap else 0.0
    bound = int(ring.output_bound(n))
    fn = native.fast.local_linear_into
    k, a, cap = ring.k, ring.a, int(ring.cap or 0)
    block = 64  # the kernel's row limit

    def run(x: torch.Tensor) -> torch.Tensor:
        xf = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy(), dtype=np.float32)
        rows = int(xf.shape[0])
        out = np.empty((rows, m), dtype=np.float32)
        for r0 in range(0, rows, block):
            r1 = min(rows, r0 + block)
            sx = np.empty(r1 - r0, dtype=np.float64)
            status = fn(xf[r0:r1], w8t, sc, bs, k, a, cap, wnorm, bound, sx, out[r0:r1], threads)
            if status:
                raise RingOverflowError(f"native int8 draft linear: status {status}")
        return torch.from_numpy(out)

    return run


class NativeInt8DraftBackend(LinearBackend):
    """Draft-model backend on the native exact local-linear kernel (``pgw_fast.local_linear_into``).

    The pack's int8 weights (no float copy: 1 byte per weight resident) with the target ring's activation
    quantizer and an exact integer product, the columns of every product spread over the native pool's
    cores. A draft only proposes (the masked target decides every token), so the int8 activation
    quantization is fine; what matters is that the draft's cost and its outputs do not depend on anything
    but its inputs: the kernel runs over public sizes only and its sums are exact, so the drafts are the
    same bits for any thread count (tested) and the work per step depends only on the rows fed.

    Args:
        pack: The draft model's pack.
        ring: Ring whose quantizer and output bound are used (the target's ring is fine).
        threads: 0 (default) = the native pool for products of at least ``pool_min_macs``; 1 = one core.
    """

    def __init__(self, pack: object, ring: RingParams, *, threads: int = 0, pool_min_macs: int = 1 << 16):
        if not self.available():
            raise ConfigError("NativeInt8DraftBackend needs pgw_fast with local_linear_into")
        if ring.a > 15 or ring.k % 8 or not 8 <= ring.k <= 56:
            raise ConfigError(f"NativeInt8DraftBackend needs A <= 15 and a byte-sized K, not {ring.name}")
        self.pack = pack
        self.ring = ring
        self.device = torch.device("cpu")
        self.stats = BackendStats()
        self.threads = int(threads)

        def conv(lw: Any, dev: torch.device) -> Any:
            n, m = int(lw.w8.shape[1]), int(lw.w8.shape[0])
            th = self.threads if (self.threads != 0 or n * m >= pool_min_macs) else 1
            return _native_int8_linear(lw.w8, lw.scale, lw.bias, ring, th), n

        self._weights = _WeightCache(pack, self.device, conv)

    @staticmethod
    def available() -> bool:
        return native_available("local_linear_into")

    def preload(self) -> None:
        self._weights.preload()

    def forward(self, call: LinearCall) -> torch.Tensor:
        fn, n = self._weights.get(call.id)
        x = call.x
        if x.ndim != 2 or x.shape[1] != n:
            raise ShapeError(f"{call.id}: x must be (T, {n}), got {tuple(x.shape)}")
        y = fn(x)
        self.stats.add(calls=1, rows=int(x.shape[0]))
        return y.to(x.device)

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """One product per linear for all sequences (weights read once per batch)."""
        return _grouped_forward(self, calls)


def native_int8_lm_head(weight: torch.Tensor, ring: RingParams, *,
                        threads: int = 0) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """A draft model's output layer ``h -> h @ weight^T`` as a per-row int8 weight on the native
    local-linear kernel (``None`` without it): 1 byte per weight instead of 4, all cores."""
    if not NativeInt8DraftBackend.available() or ring.a > 15:
        return None
    w = weight.to(torch.float32)
    sc = w.abs().amax(dim=1) / 127.0
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    w8 = torch.clamp(torch.round(w / sc[:, None]), -127, 127).to(torch.int8)
    fn = _native_int8_linear(w8, sc, None, ring, threads)
    return lambda h: fn(h.reshape(-1, h.shape[-1])).reshape(*h.shape[:-1], -1)


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
    #: TripEngine with a deferred check: the Freivalds input half ``uw . x_hat`` (rows, c) from prepare.
    rhs: np.ndarray | None = None
    #: Speculation padding appended to this chunk's wire item: rows and their packed payload (uniform
    #: random ring elements; no mask row taken, so nothing to unmask).
    pad: int = 0
    pad_payload: bytes | None = None


@dataclass
class _Filler:
    """A wire item made only of padding rows: a ``dummy`` call (or a padded call without real rows)."""

    call_index: int
    lid: LinearId
    rows: int
    payload: bytes
    checks: tuple[Any, Any] | None = None  # (u, uw) when the reservation's current vectors are known


@dataclass
class _FixedItem:
    """A padded or dummy call on the fixed-structure path (L3, ``MaskedBackend.fixed_pad``): one wire item of
    ``rows`` = real + padding rows, prepared by ONE ``trip_prepare_into`` over all rows (padding rows: zero
    activations masked with fresh ChaCha20 ``r``) and finished by the same kernel invocations whatever the
    split, so the gateway's work per call depends on the public row count only."""

    call_index: int
    lid: LinearId
    real: int  # real rows (masks taken from the reservation), first
    rows: int  # real + padding (the wire item's rows)
    out_rows: int  # rows of the call's result (real rows, or the call's own rows for a dummy call)
    payload: bytes
    sx: np.ndarray  # float64 (rows,)
    wr: np.ndarray  # int64 (rows, m): real rows' W r, zeros for padding rows
    parts: list[tuple[int, Any, Any]]  # (rows, u, uw) per extent of the real rows (u None: unchecked)
    dummy: bool = False  # a dummy call: its one taken row (``real``) carries zero activations; result zeros

    def wipe(self) -> None:
        self.wr.fill(0)


@dataclass
class _PendingCheck:
    """A Freivalds output half running on the native helper thread (deferred check)."""

    reservations: tuple[Any, ...]  # strong references: their ids stay unique while this is pending
    check: Any  # pgw_fast.TripCheck
    lid: LinearId
    rows: int
    request_id: str


@dataclass
class _FailedCheck:
    """A failed deferred check harvested outside its own session's path (another session's join, the
    pending-check limit, :meth:`MaskedBackend.close`). It was already reported (log, stats, callback);
    the error is raised in its own session's next call. Weak references where the reservation allows
    them, so an abandoned session is not kept alive (a dead reference: the session is gone)."""

    refs: tuple[Any, ...]  # weakref.ref, or the reservation itself when it takes no weak references
    error: PGWError

    def owned_by(self, ids: set[int]) -> bool:
        for r in self.refs:
            obj = r() if isinstance(r, weakref.ref) else r
            if obj is not None and id(obj) in ids:
                return True
        return False

    def alive(self) -> bool:
        return any((r() if isinstance(r, weakref.ref) else r) is not None for r in self.refs)


def _weak_or_strong(obj: Any) -> Any:
    try:
        return weakref.ref(obj)
    except TypeError:
        return obj


def _vacuous(u: Any, status: int) -> bool:
    """True if the Freivalds vectors ``u`` are all zero (a check against them proves nothing).

    Cheap for live vectors: the first word of a uniform ``u`` is non-zero with probability
    ``1 - 2^-K``, so the full scan runs only after a failure or for wiped vectors."""
    if status in (1, 2) or not u.flat[0]:
        return not bool(u.any())
    return False


def _uuid4_hex() -> str:
    """``uuid.uuid4().hex`` (16 OS-random bytes with the version-4 and RFC 4122 variant bits): the same
    format and distribution, without building a ``UUID`` object."""
    b = bytearray(os.urandom(16))
    b[6] = (b[6] & 0x0F) | 0x40
    b[8] = (b[8] & 0x3F) | 0x80
    return b.hex()


def _wipe(t: torch.Tensor) -> None:
    """Zero a buffer in place. CPU tensors go through numpy: torch's multithreaded fill is up to
    ~10x slower for these sizes when the CPU is busy (OpenMP threads contend), same result."""
    if t.device.type == "cpu" and t.is_contiguous():
        t.numpy().fill(0)
    else:
        t.zero_()


def _env_flag(name: str, default: bool) -> bool:
    """Boolean environment switch (``0/false/no/off`` and ``1/true/yes/on``; anything else: default)."""
    v = os.environ.get(name, "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


def _env_int_default(name: str, default: int) -> int:
    """Integer environment setting (unset, empty or unparsable: ``default``)."""
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


#: Native trip kernels the TripEngine needs (chat-speed WP-A4a).
_TRIP_FUNCS = ("trip_prepare_into", "trip_finish_into", "trip_freivalds_submit")

#: ``LinearCall.meta`` keys of speculation padding (fixed-R verify passes, WP-A4b / WP-A7):
#: ``PAD_ROWS``: this many padding rows are appended to the call's wire item (uniform random ring
#: elements, indistinguishable from ``xq + r mod 2^K``; no mask row is taken for them and their outputs
#: are dropped after a dummy finish). ``DUMMY``: every row of the call is padding (mask-free dummy pass
#: on the real code path: ``x`` is quantized and masked with a fresh local random ``r``); its result is
#: all zeros.
PAD_ROWS = "pad_rows"
DUMMY = "dummy"

#: Linear kinds of the ``attention`` local-linear policy (E2: qkv and o computed on the gateway).
LOCAL_ATTENTION_KINDS: frozenset[str] = frozenset({"qkv", "o"})
#: R1 (research/plans/fewer-round-trips.md), all default off: ``attention_down`` also computes down on
#: the gateway (one round trip per layer: gate_up), ``all`` computes every linear there (no round trip),
#: ``auto_kinds`` decides per linear kind from a start-up calibration (local iff cheaper than one trip).
LOCAL_ATTENTION_DOWN_KINDS: frozenset[str] = frozenset({"qkv", "o", "down"})
LOCAL_ALL_KINDS: frozenset[str] = frozenset({"qkv", "o", "gate_up", "down"})
LOCAL_LINEAR_MODES = ("off", "attention", "auto", "attention_down", "all", "auto_kinds")
_LOCAL_FIXED_KINDS: dict[str, frozenset[str]] = {
    "attention": LOCAL_ATTENTION_KINDS, "attention_down": LOCAL_ATTENTION_DOWN_KINDS, "all": LOCAL_ALL_KINDS}


def _is_lane_client(client: Any) -> bool:
    """True for a :class:`pgw.wire.lane.LaneComputeClient` (also behind a ``__getattr__`` wrapper)."""
    cfg = getattr(client, "lane_config", None)
    if cfg is None:
        return False
    try:
        from pgw.wire.lane import LaneConfig
    except ImportError:  # pragma: no cover - lane module always present
        return False
    return isinstance(cfg, LaneConfig)


def _pad_rows(call: LinearCall) -> int:
    meta = call.meta
    if not meta:
        return 0
    v = meta.get(PAD_ROWS, 0)
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise ConfigError(f"{call.id}: meta[{PAD_ROWS!r}] must be a non-negative int, got {v!r}")
    return v


def _is_dummy(call: LinearCall) -> bool:
    return bool(call.meta) and bool(call.meta.get(DUMMY, False))


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
    #: Chunks prepared by the TripEngine (native trip kernels): finished by ``trip_finish_into``.
    trip: bool = False
    #: Mask reservations of the calls (deferred checks are keyed by them).
    reservations: tuple[Any, ...] = ()
    #: Padding-only items (dummy calls), and the order of the request's items when there are any:
    #: ``("c", chunk index)`` or ``("f", filler index)``; ``None`` = one item per chunk, in order.
    fillers: list[_Filler] = field(default_factory=list)
    items_map: list[tuple[str, int]] | None = None
    #: ``("g", group index)`` items: chunks of one padded call merged into one wire item (chunk indices,
    #: consecutive, in row order; the padding rows follow the last chunk's rows).
    groups: list[list[int]] = field(default_factory=list)
    #: ``("x", index)`` items: calls on the fixed-structure padding path (L3).
    fixed: list[_FixedItem] = field(default_factory=list)
    #: Prepared by the one-call trip path (:attr:`MaskedBackend.trip_fast`): finished by ``_finish_one``.
    one: bool = False
    #: Prepared by the native per-trip path (:attr:`MaskedBackend.trip_native`): its ``pgw_fast.TripTicket``
    #: (finished by ``_finish_lease``) and whether its Freivalds output half is deferred.
    ticket: Any = None
    defer: bool = False
    #: Prepared by the native per-trip path for several calls (:attr:`MaskedBackend.trip_native_multi`): one
    #: ``pgw_fast.TripTicket`` per call (one item each, in call order) and whether each one's check is deferred.
    tickets: list[Any] | None = None
    defers: list[bool] = field(default_factory=list)

    def wipe(self) -> None:
        """Zero the ``W r`` buffers (secret-derived) of every chunk; idempotent."""
        if self.ticket is not None:
            self.ticket.wipe()
        if self.tickets is not None:
            for tk in self.tickets:
                tk.wipe()
        for fx in self.fixed:
            fx.wipe()
        for ch in self.chunks:
            if not ch.wr_wiped:
                _wipe(ch.wr)
                ch.wr_wiped = True
            if ch.rhs is not None:
                ch.rhs.fill(0)
                ch.rhs = None


class MaskedBackend(LinearBackend):
    """One-time-masked remote compute.

    Per call: quantize ``x`` per token -> take the next ``T`` unused mask rows of this session's
    reservation for ``call.id`` -> ``x_hat = xq + r`` -> (optional outbound noise monitor) ->
    ``ComputeClient.compute`` -> ``yq = centered(y_hat - W r)`` with range check -> Freivalds check
    (always, or with probability ``freivalds_rate``) -> rescale + bias.

    ``rows_exact``: per-row quantization, an exact ring product and per-row rescaling (local linears too).

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
        trip_engine: Prepare and finish with the native trip kernels (constant-time activation width
            selection; chat-speed WP-A4a/A4b). ``None``: on with a trip-lane client
            (:class:`pgw.wire.lane.LaneComputeClient`), else ``PGW_TRIP_ENGINE`` (default off).
            Bit-identical; a per-linear self-check falls back to today's kernels.
        deferred_check: With the TripEngine, run each trip's Freivalds output half on the native
            helper thread and join it before the session's next payload (the model's last linear is
            always checked synchronously). ``None``: on with the TripEngine (``PGW_DEFERRED_CHECK``).
        sync_trips: :meth:`sync_trip_ok` says yes for decode passes of at most ``sync_trip_max_rows``
            rows (the engine's synchronous decode loop). ``None``: on with a trip-lane client and the
            TripEngine (``PGW_SYNC_TRIPS``).
        local_linears: E2 policy, ``off`` (default), ``attention``, ``auto``, or (R1, default off)
            ``attention_down``, ``all``, ``auto_kinds`` (:meth:`configure_local_linears`); ``local_rtt_s``
            is the round trip ``auto`` / ``auto_kinds`` compare with. ``local_max_rows``: local linears
            apply only to passes of at most this many rows in total (decode); larger passes (prefill)
            send every linear. ``None``: every pass (today's behaviour).

    Speculation padding (fixed-R verify passes): ``LinearCall.meta[PAD_ROWS]`` / ``meta[DUMMY]``, see
    :data:`PAD_ROWS`. Padding rows are uniform random ring elements, take no mask row and are dropped
    after a dummy finish that still verifies them.

    Raises (from forward):
        MaskExhaustedError, IntegrityError, FingerprintMismatchError, ServiceUnavailableError.
    """

    rows_exact = True

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
        prepare_many: bool | None = None,
        mask_prefetch: bool | None = None,
        trip_engine: bool | None = None,
        deferred_check: bool | None = None,
        sync_trips: bool | None = None,
        local_linears: str = "off",
        local_rtt_s: float | None = None,
        local_max_rows: int | None = None,
        prime_check: bool | None = None,
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
        self._fixed_pad_ok: bool | None = None
        self._native_finish = native_available("finish_chunk")
        self._native_group = native_available("finish_group")
        #: Several wire items of one round trip go through one native ``quantize_mask_pack_many``
        #: call (spread over a persistent pool when large); the same bytes as one call per item.
        #: ``PGW_PREPARE_MANY=0`` (or ``prepare_many=False``) keeps one native call per item.
        self.prepare_many = (_env_flag("PGW_PREPARE_MANY", True) if prepare_many is None
                             else bool(prepare_many)) and native_available("quantize_mask_pack_many")
        #: Single-use mask prefetch (:meth:`_maybe_prefetch`): after a multi-session step, or a call
        #: of ``3..prefetch_max_rows`` rows, a background thread stages the next rows of the same
        #: (reservation, linear), so the next step takes them without deriving them. Staging never
        #: changes which rows a call gets. ``PGW_MASK_PREFETCH=0`` (or ``mask_prefetch=False``) turns
        #: it off.
        self.mask_prefetch = (_env_flag("PGW_MASK_PREFETCH", True) if mask_prefetch is None
                              else bool(mask_prefetch))
        #: Calls with more rows than this (prefill) are not prefetched for: the next call of that
        #: linear is a decode step.
        self.prefetch_max_rows = 8
        #: Prompt passes: the engine stages the next remote linear's rows ahead (:meth:`prefetch_prompt`).
        #: Off by default (``PGW_PROMPT_PREFETCH=1``); needs :attr:`mask_prefetch`.
        self.prompt_prefetch = self.mask_prefetch and _env_flag("PGW_PROMPT_PREFETCH", False)
        #: Steps staged ahead per (reservation, linear); the reservation's byte budget also bounds it.
        self.prefetch_depth = 2
        self._prefetch_pool: Any = None
        self._prefetch_pending = 0
        self._prefetch_lock = threading.Lock()
        #: Rows staged by the prefetcher (diagnostics).
        self.prefetched_rows = 0
        self.stats = BackendStats()
        #: Calls with at most this many rows in total count as decode in ``stats.decode_*`` (the
        #: engine's ``inline_max_rows`` default; the runtime may set it to its configured value).
        self.decode_max_rows = 8
        #: ``row_norm_sq_max`` per linear key from :meth:`warm_weight_norms` (packs without the field).
        self._norm_sq: dict[str, int] = {}
        #: Full int8 weights read on a request path to learn a norm (0 with norms in the manifest or
        #: after :meth:`warm_weight_norms`; anything else is a cold-start penalty worth alerting on).
        self.hot_weight_loads = 0

        # ---- chat-speed WP-A4b: TripEngine, deferred checks, synchronous trips, local linears ----
        # a trip-lane client, or the blocking stream lane (``lane = "stream"``): synchronous trips are fast
        lane_client = _is_lane_client(client) or getattr(client, "sync_stream", False) is True
        self._trip_native = all(native_available(f) for f in _TRIP_FUNCS)
        #: TripEngine: prepare and finish with the native trip kernels (``pgw_fast.trip_*``, WP-A4a):
        #: the activation bit width is selected in constant time (every row evaluates the same public
        #: number of widths), prefill blocks run on the native pool, and the Freivalds check can be
        #: deferred. Bit-identical to the other paths (tested, plus a per-linear self-check with
        #: fallback). Default: on with a trip-lane client (faster trips make gateway timing more
        #: visible), else ``PGW_TRIP_ENGINE`` (off).
        self.trip_engine = ((_env_flag("PGW_TRIP_ENGINE", lane_client) if trip_engine is None else bool(trip_engine))
                            and self._trip_native and ring.k % 8 == 0 and 8 <= ring.k <= 56)
        #: Accept mask segments in the prime-field check format (``check_field = "p31"``, SPEC 5.1): one
        #: secret vector over ``F_p``, ``p = 2^31 - 1``, checked on the unmasked ``xq`` / ``yq`` (31 bits per
        #: call instead of the ring check's ``c * h``; 1/7 of the check-vector reads at K32A12C27). Off by
        #: default (``PGW_PRIME_CHECK``): such segments are then refused before any row is used. On, it
        #: needs the native trip kernels with ``trip_prime_output`` and turns the TripEngine on; the
        #: prime check always runs synchronously (never deferred) and only on the TripEngine paths
        #: (padded / fixed-structure calls and the non-trip fallbacks refuse prime extents, fail closed).
        self.prime_check = _env_flag("PGW_PRIME_CHECK", False) if prime_check is None else bool(prime_check)
        if self.prime_check:
            if not (self._trip_native and native_available("trip_prime_output") and ring.k <= 32
                    and ring.k % 8 == 0):
                raise ConfigError("prime_check needs the native trip kernels (pgw_fast.trip_prime_output) "
                                  "and a ring of at most 32 bits")
            self.trip_engine = True
        #: Deferred Freivalds check (TripEngine only): the output half ``u . y_hat`` of a trip runs on
        #: the native helper thread while the gateway continues with the plaintext step. It is joined
        #: before the same session's next payload is built (so before its next send), and the last
        #: linear of the model is always checked synchronously (so before any logits leave). A failed
        #: check raises :class:`IntegrityError` there (see :attr:`defer_min_out` for what the session's
        #: state may already hold then). A session that ends or aborts with checks pending should be
        #: released with :meth:`release_session_checks`. Default: with the TripEngine (``PGW_DEFERRED_CHECK``).
        self.deferred_check = self.trip_engine and (
            _env_flag("PGW_DEFERRED_CHECK", True) if deferred_check is None else bool(deferred_check))
        #: :meth:`sync_trip_ok` answers yes for decode-sized passes (the engine then runs a whole decode
        #: pass on its plaintext thread over :meth:`forward_many`). Default: with a trip-lane client and
        #: the TripEngine (``PGW_SYNC_TRIPS``).
        self.sync_trips = self.trip_engine and (
            _env_flag("PGW_SYNC_TRIPS", lane_client) if sync_trips is None else bool(sync_trips))
        #: Largest pass (rows over all sequences) :meth:`sync_trip_ok` accepts (the lane's row limit).
        self.sync_trip_max_rows = 8
        #: ``PGW_SYNC_MULTI`` (default off; on in the SDK): :meth:`sync_trip_ok` also accepts a larger pass
        #: (the decode step of concurrent answers, at most :attr:`sync_multi_max_rows` rows) when the lane client
        #: serves it as one multi call (``multi_rows()``, lane format 2). The pass then runs wholly on the
        #: plaintext thread over :meth:`forward_many` (no event-loop and executor hand-offs per trip). Same
        #: calls, rows, order and requests as the split-hop path, so the same masks, bytes and outputs.
        self.sync_multi = self.sync_trips and _env_flag("PGW_SYNC_MULTI", False)
        self.sync_multi_max_rows = 64
        #: Preferred blocks per pipelined prefill round trip (for ``TransformerModel.pipeline_blocks``):
        #: the TripEngine prepares and finishes each block on the native pool, so 4 blocks overlap
        #: masking, compute and verification better than halves. Rows are independent: same bits.
        self.pipeline_blocks = 4
        #: Rows x features from which a TripEngine call spreads over the native pool (prefill).
        self.trip_pool_min_elems = 1 << 16
        #: ``PGW_TRIP_SPLIT`` (default 0 = off): on the native per-trip path, split each trip's prepare (and
        #: its ``r`` keystream) by input-column blocks and its finish / prime output half by output columns
        #: over this many tasks of the persistent native pool, for trips of at least
        #: ``PGW_TRIP_SPLIT_MIN_ELEMS`` (default 8192) rows x features. Fixed partition, exact merges: the
        #: bytes do not depend on the core count (``tests/native/test_trip_split.py``).
        self.trip_split = max(0, _env_int_default("PGW_TRIP_SPLIT", 0))
        self.trip_split_min_elems = max(1, _env_int_default("PGW_TRIP_SPLIT_MIN_ELEMS", 8192))
        #: Deferred checks still pending above this many are all joined at the next prepare.
        self.max_pending_checks = 256
        #: Defer only checks of linears with at least this many outputs (public, per linear): handing
        #: the output half to the helper thread costs ~10-20 us here, more than the half itself for
        #: m <= 4608 (7B qkv, o, down); it pays for gate_up (m = 37888, ~36 us). 0 = every linear.
        #: Lowering it lets unverified qkv / o outputs reach the session's plaintext state (its KV
        #: cache) before the join raises. Nothing is sent before the join (no privacy loss), but that
        #: session's cache is then poisoned: after an :class:`IntegrityError` the gateway must discard
        #: the session (never resume it). The default defers only gate_up, whose output is consumed
        #: within the same layer (down's prepare joins it), so no cache entry depends on it.
        self.defer_min_out = 16384
        self._trip_checked: dict[tuple[int, int, float], bool] = {}
        self._defer_memo: dict[tuple[Any, int], bool] = {}
        self._final_ordinal: int | None = None
        self._pending: list[_PendingCheck] = []
        #: Failed deferred checks harvested outside their own session's path: raised in that
        #: session's next call (bounded by :attr:`max_pending_checks`, oldest dropped; all reported).
        self._failed: list[_FailedCheck] = []
        self._pending_lock = threading.Lock()
        #: Linear kinds computed exactly on the gateway (E2): empty = every linear is remote.
        self._local_kinds: frozenset[str] = frozenset()
        self._local_w: _WeightCache | None = None
        #: :meth:`_local_fast` usable at all (native trip prepare kernel, a 32-bit ring, at most 15
        #: activation bits for two int8 digits, exact CPU int8 GEMMs); per shape it is self-checked.
        #: Decided on first use (the CPU int8 probe times a GEMM once per process).
        self._local_fast_on: bool | None = None
        self._local_fast_checked: dict[tuple[int, int, float, int], bool] = {}
        self._local_zero_r: dict[int, np.ndarray] = {}
        #: Single-session decode prefetch (:meth:`_maybe_prefetch`): on with synchronous trips (the gateway
        #: thread then waits in a GIL-free ``recv`` during every trip, which is when the prefetch thread
        #: derives); ``PGW_PREFETCH_DECODE`` overrides. Rows staged per (reservation, linear) at a time.
        self.prefetch_decode = self.mask_prefetch and _env_flag("PGW_PREFETCH_DECODE", self.sync_trips)
        self.decode_prefetch_rows = 8
        self._decode_pf: dict[tuple[int, str], int] = {}
        #: One-call trips on a lean path (T3, ``PGW_TRIP_FAST``, default off): a pass of one call with at
        #: most :attr:`max_rows_per_item` rows (decode) skips the batch bookkeeping. Same masks in the same
        #: order, same check and sampling draws, same payload bytes, the same checks on every row (the
        #: soundness check of the extent's vectors runs inside the take, before any row is taken), same
        #: outputs and wiping; uuid4-format request ids as on every path.
        self.trip_fast = self.trip_engine and _env_flag("PGW_TRIP_FAST", False)
        #: With :attr:`trip_fast` (``PGW_NOISE_ASYNC_SAMPLED``, default off): a call the noise monitor samples
        #: runs its every-call checks (duplicate rows) inline, as every call does, and its statistics
        #: (``inspect_planned``) on a worker thread, on copies, in call order; the next prepare waits for
        #: them before anything else is taken or sent, so an alarm blocks the next payload (fail closed).
        #: The copies are wiped after the inspection.
        self.noise_async_sampled = self.trip_fast and _env_flag("PGW_NOISE_ASYNC_SAMPLED", False)
        #: With :attr:`trip_fast` (``PGW_TRIP_NATIVE``, default off): one native call per trip on each side
        #: of the lane call. The reservation hands a lease (``pgw_fast.TripLease``: the next rows of one
        #: linear in one decrypted ``W r`` chunk, the extent's ``r`` key and check vectors) and each trip
        #: makes ``lease.prepare`` (take the rows, derive ``r``, quantize, mask, pack, Freivalds input
        #: half) and ``ticket.finish`` (unmask, range check, Freivalds, rescale, wipe). Same rows in the
        #: same order (each once), same check and sampling draws, same payload bytes and request ids, the
        #: same checks on every row and the same outputs; reservations without leases (in-memory ring
        #: segments, chained tiers) and calls a lease cannot serve take the :attr:`trip_fast` path.
        self.trip_native = (self.trip_fast and _env_flag("PGW_TRIP_NATIVE", False)
                            and native_available("TripLease") and self.ring.k % 8 == 0 and self.ring.k <= 56)
        #: With :attr:`trip_native` and a lane client (``PGW_TRIP_DRIVER``, default off): a leased trip's lane call
        #: and finish are one native call (``TripTicket.lane_finish`` via ``LaneComputeClient.trip_ticket``): the
        #: reply is read straight into the finish (no response objects, no zero-filled reply buffer, no copy).
        #: Same payload bytes, header and request id, same checks, outputs, statistics and fallback (a call the
        #: lane does not serve goes to the wrapped client with the same request, finished as before).
        self.trip_driver = self.trip_native and _env_flag("PGW_TRIP_DRIVER", False)
        #: With :attr:`trip_native` (``PGW_DECODE_NATIVE``, default off): a decode pass of one sequence on the native
        #: float path runs its decoder layers in ``pgw_fast.DecodeDriver`` (:mod:`pgw.engine.decode_native`): the
        #: float operations and every trip (lease prepare, duplicate-row check, native lane trip, finish with every
        #: check) with the GIL released, Python once per call (``PGW_DECODE_NATIVE_LAYERS`` layers, 0 = all). Same
        #: rows, payload bytes, request-id format, checks and outputs; trips the lane does not serve take the
        #: ordinary fallback with the same request.
        self.decode_native = (self.trip_native and _env_flag("PGW_DECODE_NATIVE", False)
                              and native_available("DecodeDriver"))
        #: With :attr:`trip_native` (``PGW_TRIP_NATIVE_MULTI``, default off; on in the SDK): a trip of several
        #: one-chunk calls (the decode step of concurrent answers) takes one lease per call
        #: (:meth:`_prepare_trip_leases`) instead of the batch path's Python takes. Same rows in the same order,
        #: same draws, payload bytes, items and request-id format, the same checks on every row, same outputs.
        self.trip_native_multi = self.trip_native and _env_flag("PGW_TRIP_NATIVE_MULTI", False)
        self._verifiers: dict[tuple[str, int], Any] = {}
        #: id(lease) -> (min_soundness_bits, size of the soundness cache) when its extent was verified
        self._lease_sound: dict[int, tuple[float, int]] = {}
        self._noise_pool: Any = None
        self._noise_pending: Any = None
        #: The lease-ahead thread (``PGW_TRIP_LEASE_AHEAD``, :meth:`_lease_ahead_submit`) and its queued jobs.
        self._ahead_pool: Any = None
        self._ahead_lock = threading.Lock()
        self._ahead_jobs: set[tuple[int, str]] = set()
        self._closing = False
        #: :meth:`_local_lean_one` constants per linear key (:attr:`trip_fast`).
        self._local_lean: dict[str, tuple[Any, ...]] = {}
        #: :meth:`_local_fast` on ``pgw_fast.local_linear_into`` (one fused native call; ``PGW_LOCAL_NATIVE=0``
        #: keeps the numpy + ``torch._int_mm`` form). Same self-check per shape against the reference.
        self._local_native = _env_flag("PGW_LOCAL_NATIVE", True) and native_available("local_linear_into")
        #: ``lid.key -> (w8t tensor, its numpy view)``; the tensor is held so the identity check is sound.
        self._local_w8n: dict[str, tuple[torch.Tensor, np.ndarray]] = {}
        #: Exact local products on a gateway GPU through captured CUDA graphs (``local_linears_device`` cuda/auto,
        #: K = 32; :mod:`pgw.engine.local_gpu`), or ``None`` (CPU kernels). Chosen at start-up after a self-check.
        self._local_gpu: Any = None
        #: ``lid.key -> f(x) -> y``: lean 1-row decode calls of the GPU's native kernel
        #: (:meth:`pgw.engine.local_gpu.LocalGpuProducts.one_row`), self-checked like the GPU path; empty = none.
        self._local_one_row: dict[str, Any] = {}
        #: Wall time of :meth:`forward_local` (local linears only, unlike ``stats.local_seconds``, which also
        #: counts the masked trips' gateway work): 1-row steps (decode) and the rest (prompt passes).
        self.local_linear_timing: dict[str, float] = {"decode_calls": 0, "decode_s": 0.0, "other_calls": 0,
                                                       "other_s": 0.0}
        self._local_gpu_mismatch = ""  # the failing case of the last GPU self-check (diagnostics)
        #: ``n * m`` from which a native local linear spreads its columns over the native pool. A decode step
        #: reads every local weight once from DRAM (the model's int8 weights do not stay in cache between
        #: tokens), so even the 0.5B's 1 MB qkv gains from several cores' bandwidth (``PGW_LOCAL_POOL_MIN_MACS``;
        #: the column split changes no bit: every output is the same exact integer sum).
        self.local_pool_min_macs = max(1, _env_int_default("PGW_LOCAL_POOL_MIN_MACS", 1 << 19))
        #: The local-linear decision and its calibration inputs (public configuration; logged at start).
        self.local_policy: dict[str, Any] = {"mode": "off", "kinds": ()}
        if local_max_rows is not None and int(local_max_rows) < 1:
            raise ConfigError("local_max_rows must be >= 1 or None")
        #: Passes of more rows (in total) than this send every linear (R1; None: no cap).
        self.local_max_rows: int | None = None if local_max_rows is None else int(local_max_rows)

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
        if local_linears != "off":
            self.configure_local_linears(local_linears, rtt_s=local_rtt_s)

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
        if self.trip_engine:  # TripEngine self-checks at start-up, not on the first request
            self.warm_trip_engine()
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

    def _check_soundness(self, lid: Any, n: int, u: Any, token: Any = None, record: bool = True) -> None:
        """Raise if the current extent's check vectors give less than ``min_soundness_bits``."""
        if u is not None and int(getattr(u, "size", 0) or 0) > 0 and _vacuous(u, 0):
            # Wiped vectors (a reservation closed under us): Freivalds with u = 0 accepts anything.
            raise MaskReuseError("mask reservation check vectors are wiped (reservation closed)")
        prime = is_prime_token(token)
        if prime and not self.prime_check:
            raise MaskStoreError(
                f"{lid}: mask segment uses the prime-field Freivalds check (check_field='p31'), which this "
                "gateway does not accept (prime_check is off)", details={"linear": str(lid)})
        if self.min_soundness_bits <= 0:
            return
        if token is not None and token in self._sound_ok:
            return
        checks = 0 if u is None else int(u.shape[0])
        bits = self._soundness_bits(n, checks, prime)
        if bits < self.min_soundness_bits:
            raise MaskStoreError(
                f"{lid}: mask segment has {checks} Freivalds vector(s) = {bits:.0f} bits of soundness for "
                f"n={n} at {self.ring.name}; at least {self.min_soundness_bits:.0f} are required "
                "(regenerate masks with more checks)",
                details={"linear": str(lid), "checks": checks, "min_soundness_bits": self.min_soundness_bits},
            )
        if token is not None and record:
            if len(self._sound_ok) > 65536:  # bounded: tokens are (segment, linear) pairs
                self._sound_ok.clear()
            self._sound_ok.add(token)

    def _soundness_bits(self, n: int, checks: int, prime: bool = False) -> float:
        key = (n, checks, prime)
        hit = self._soundness_memo.get(key)  # type: ignore[call-overload]
        if hit is None:
            from pgw.core.freivalds import prime_soundness_bits, soundness_bits

            bits = prime_soundness_bits(self.ring, n, checks) if prime else soundness_bits(self.ring, n, checks)
            hit = self._soundness_memo[key] = float(bits)  # type: ignore[index]
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
        ensured: set[tuple[int, str]] = set()
        for key, (res, lid, _, rows) in need.items():
            ensure = getattr(res, "ensure", None)
            if ensure is not None:
                ensure(lid, rows)  # returns only with >= rows reserved (else raises): no remaining() needed
                ensured.add(key)
        for key, (res, lid, _, rows) in need.items():
            if key in ensured:
                continue
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
        _note_activity()  # PGW_FACTORY_YIELD_FILE: the gateway's factory steps aside while passes run
        # Deferred checks of these sessions are joined first: no mask row is taken and no payload
        # is built (so nothing is sent) before the previous trip's results proved correct.
        if self._noise_pending is not None:
            self._join_noise()
        if self._pending or self._failed:
            self._join_pending(calls)
        if self.trip_fast and len(calls) == 1:
            c = calls[0]
            if (not c.meta and c.x.ndim == 2 and 0 < int(c.x.shape[0]) <= self.max_rows_per_item
                    and self._use_trip(calls)):
                if self.trip_native:
                    prep = self._prepare_trip_lease(c)
                    if prep is not None:
                        return prep
                prep = self._prepare_trip_one(c)
                if self.mask_prefetch:
                    self._maybe_prefetch(calls)
                return prep
        if any(c.meta and (PAD_ROWS in c.meta or DUMMY in c.meta) for c in calls):
            if self.fixed_pad:
                prep = self._prepare_fixed(calls)
                real = [c for c in calls if not (c.meta and (PAD_ROWS in c.meta or DUMMY in c.meta))]
            else:
                prep = self._prepare_padded(calls)
                real = [c for c in calls if not _is_dummy(c)]
        else:
            if self.trip_native_multi and len(calls) > 1 and self._use_trip(calls):
                prep_m = self._prepare_trip_leases(calls)
                if prep_m is not None:  # leased trips: the leases' parts are decrypted ahead (no prefetch)
                    return prep_m
            prep = self._prepare_core(calls)
            real = calls  # type: ignore[assignment]
        if self.mask_prefetch:
            self._maybe_prefetch(real)
        return prep

    def _prepare_core(self, calls: Sequence[LinearCall]) -> _Prepared:
        if self._use_trip(calls):
            return self._prepare_trip(calls)
        return self._prepare_native(calls) if self._native_prepare_ok(calls) else self._prepare_python(calls)

    # ---- TripEngine (native trip kernels, WP-A4a) ---------------------------------------------------
    def _use_trip(self, calls: Sequence[LinearCall]) -> bool:
        """The TripEngine serves these calls (enabled, every linear's self-check passed)."""
        if not self.trip_engine:
            return False
        for c in calls:
            meta: _MaskedMeta = self._meta.get(c.id)
            if self.ring.cap and meta.wnorm is None:  # pragma: no cover - capped metas always carry it
                return False
            if not self._trip_self_check(meta):
                return False
        return True

    def _trip_self_check(self, meta: _MaskedMeta) -> bool:
        """Once per linear shape: the trip kernels reproduce the reference bit for bit on public test
        data (prepare: payload and scales; finish: outputs), else the TripEngine is switched off for
        this backend (every path falls back to today's kernels) and an error is logged."""
        key = (meta.in_features, meta.out_features, float(meta.wnorm or 0.0))
        hit = self._trip_checked.get(key)
        if hit is not None:
            return hit
        try:
            ok = _trip_kernels_agree(self.ring, *key)
        except Exception as exc:  # any surprise: today's path
            log.error("trip kernel self-check raised %s for n=%d m=%d", type(exc).__name__, key[0], key[1])
            ok = False
        self._trip_checked[key] = ok
        if not ok:
            log.error("trip kernels disagree with the reference for n=%d m=%d at %s: TripEngine off "
                      "(deferred checks and synchronous trips off too)", key[0], key[1], self.ring.name)
            self.trip_engine = self.deferred_check = self.sync_trips = False
        return ok

    def warm_trip_engine(self) -> bool:
        """Run the TripEngine self-check for every linear now (start-up) instead of on first use."""
        if not self.trip_engine:
            return False
        for spec in self.pack.linear_specs():  # type: ignore[attr-defined]
            if not self._trip_self_check(self._meta.get(spec.id)):
                return False
        return True

    def _final_linear_ordinal(self) -> int:
        fo = self._final_ordinal
        if fo is None:
            try:
                fo = max((s.id.ordinal for s in self.pack.linear_specs()), default=-1)  # type: ignore[attr-defined]
            except Exception:  # unknown order: never defer
                fo = -1
            self._final_ordinal = fo
        return fo

    def _defer_ok(self, lid: Any) -> bool:
        """A trip of ``lid`` may have its Freivalds check deferred: never the model's last linear (its
        output leads straight to logits), never a linear outside the standard schedule, and only for
        linears with at least :attr:`defer_min_out` outputs (a public per-linear choice)."""
        if not self.deferred_check or not isinstance(lid, LinearId):
            return False
        ck = (lid, int(self.defer_min_out))
        hit = self._defer_memo.get(ck)
        if hit is None:  # a public per-linear decision: computed once per (linear, threshold)
            hit = (lid.ordinal < self._final_linear_ordinal()
                   and int(self._meta.get(lid).out_features) >= int(self.defer_min_out))
            if len(self._defer_memo) > 65536:
                self._defer_memo.clear()
            self._defer_memo[ck] = hit
        return hit

    def _prepare_trip(self, calls: Sequence[LinearCall]) -> _Prepared:
        """:meth:`_prepare_native` on ``pgw_fast.trip_prepare_into`` (constant-time width selection).

        Same masks taken in the same order, same check and noise-sampling draws, same payload bytes
        and scales. A chunk whose check is deferred also gets the Freivalds input half
        ``uw . x_hat`` (computed in the same pass) for the output half after the reply.
        """
        t0 = time.perf_counter()
        reservations, metas = self._precheck(calls)
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
        nb = kbits // 8
        noise = self.noise_monitor
        plan_call: Any = None
        guard_packed: Any = None
        if noise is not None:
            plan_call = getattr(noise, "plan_call", None)
            guard_packed = getattr(noise, "guard_packed", None)
            if plan_call is None or guard_packed is None:
                plan_call = guard_packed = None
        prep_into = native.fast.trip_prepare_into
        want_soundness = self.min_soundness_bits > 0
        max_item = self.max_rows_per_item
        pool_min = int(self.trip_pool_min_elems)
        sx_parts: list[list[np.ndarray]] = [[] for _ in calls]
        chunks: list[_Chunk] = []
        try:
            for ci, (call, res, meta, xn) in enumerate(zip(calls, reservations, metas, xs, strict=True)):
                t = int(xn.shape[0])
                lid = call.id
                n = meta.in_features
                wn = float(meta.wnorm or 0.0)
                defer_lid = self._defer_ok(lid)
                row = 0
                while row < t:
                    check = self._should_check()
                    sampled = bool(plan_call()) if plan_call is not None else noise is not None
                    rows, r, wr, chk = self._take(res, lid, min(t - row, max_item), check or want_soundness,
                                                  prime_ok=True)
                    token = u = uw = None
                    rhs = None
                    r64 = None
                    try:
                        if chk is not None:
                            token, u, uw = chk
                            self._check_soundness(lid, n, u, token)
                            if not check:
                                token = u = uw = None
                        if tuple(r.shape) != (rows, n) or tuple(wr.shape) != (rows, meta.out_features):
                            raise ShapeError(
                                f"{lid}: mask shapes {tuple(r.shape)}/{tuple(wr.shape)} do not match the linear"
                            )
                        r_np = r.numpy()
                        r64 = r_np if r_np.dtype == np.int64 and r_np.flags.c_contiguous else \
                            np.ascontiguousarray(r_np, dtype=np.int64)
                        buf = np.empty(rows * n * nb, dtype=np.uint8)
                        sx = np.empty(rows, dtype=np.float64)
                        xh = np.empty((rows, n), dtype=np.int64) if sampled else None
                        uw_in = None
                        prime = u is not None and is_prime_token(token)
                        if (prime or defer_lid) and u is not None and uw is not None and int(u.shape[0]) > 0:
                            uw_in = uw
                            rhs = np.empty((rows, int(u.shape[0])), dtype=np.uint64)
                        try:
                            if prime:  # prime input half uw . xq mod p (synchronous output half at finish)
                                prep_into(xn[row : row + rows], r64, kbits, abits, cap, wn, buf, sx, xh, uw_in, rhs,
                                          0 if rows * n >= pool_min else 1, field=P31)
                            else:
                                prep_into(xn[row : row + rows], r64, kbits, abits, cap, wn, buf, sx, xh, uw_in, rhs,
                                          0 if rows * n >= pool_min else 1)
                        except ValueError as exc:
                            if str(exc).startswith("ring_overflow"):
                                raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
                            raise ShapeError(str(exc)) from exc
                        payload = buf.tobytes()
                        if noise is not None:
                            if guard_packed is not None and not sampled:
                                guard_packed(payload, rows, n)
                            elif xh is not None:
                                x_hat = torch.from_numpy(xh)
                                xq = center(x_hat - r.to(torch.int64), ring)
                                try:
                                    if plan_call is not None:
                                        noise.guard(x_hat, xq, planned=True)  # type: ignore[attr-defined]
                                    else:
                                        noise.guard(x_hat, xq)  # type: ignore[attr-defined]
                                finally:
                                    _wipe(xq)
                    except BaseException:
                        _wipe(wr)
                        if rhs is not None:
                            rhs.fill(0)
                        raise
                    finally:
                        _wipe(r)  # one-time pad: never keep it around
                        if r64 is not None and r64 is not r_np:
                            r64.fill(0)
                    chunks.append(_Chunk(ci, row, rows, lid, None, wr, u, uw, payload, token, rhs=rhs))
                    sx_parts[ci].append(sx)
                    row += rows
        except BaseException:
            for ch in chunks:
                _wipe(ch.wr)
                if ch.rhs is not None:
                    ch.rhs.fill(0)
            raise
        prep = self._native_request(calls, chunks, sx_parts, metas, t0)
        prep.trip = True
        prep.reservations = tuple(reservations)
        return prep

    def _join_noise(self) -> None:
        """Wait for the previous sampled call's statistics (:attr:`noise_async_sampled`); an alarm then
        raises in the next ``plan_call`` (fail closed)."""
        fut, self._noise_pending = self._noise_pending, None
        if fut is not None:
            try:
                fut.result()
            except Exception:  # the monitor's own failure: fail closed like the inline path would
                log.exception("outbound-noise inspection failed")
                raise

    def _noise_submit(self, noise: Any, xh: np.ndarray, r64: np.ndarray) -> None:
        """Run ``noise.inspect_planned`` on copies of ``x_hat`` and ``r`` on the noise worker (wiped after)."""
        xh_c = xh.copy()
        r_c = r64.copy()
        ring = self.ring

        def run() -> None:
            xq = None
            try:
                x_hat = torch.from_numpy(xh_c)
                xq = center(x_hat - torch.from_numpy(r_c), ring)
                noise.inspect_planned(x_hat, xq)
            finally:
                if xq is not None:
                    _wipe(xq)
                xh_c.fill(0)
                r_c.fill(0)

        pool = self._noise_pool
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor

            pool = self._noise_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-noise")
        self._noise_pending = pool.submit(run)

    def _sound_verifier(self, lid: Any, n: int, record: bool = True) -> Any:
        """``verify(checks)`` for ``take_checked``: :meth:`_check_soundness` of the extent's vectors (``record``:
        remember a passing extent; off for the lease-ahead thread, which leaves the plaintext thread's state as is)."""
        def verify(chk: Any) -> None:
            token, u, _uw = chk
            self._check_soundness(lid, n, u, token, record=record)
        return verify

    def _prepare_trip_one(self, call: LinearCall) -> _Prepared:
        """:meth:`_prepare_trip` for one call of at most :attr:`max_rows_per_item` rows (:attr:`trip_fast`).

        Same order as the batch path: shapes and finite activations are checked and the reservation is
        extended before any row is taken; the Freivalds draw, then the noise-sampling draw, then the take
        (whose extent's check vectors pass the soundness check before a row is consumed); then the fused
        native prepare. Rows crossing an extent boundary are taken chunk by chunk, as there.
        """
        t0 = time.perf_counter()
        res = self._reservation(call)
        lid = call.id
        meta: _MaskedMeta = self._meta.get(lid)
        x = call.x
        n = meta.in_features
        if x.shape[1] != n:
            raise ShapeError(f"{lid}: x must be (T, {n}), got {tuple(x.shape)}")
        t = int(x.shape[0])
        if x.dtype != torch.float32 or x.device.type != "cpu":
            x = x.to("cpu", torch.float32)
        xn = x.detach().numpy()
        if not xn.flags.c_contiguous:
            xn = np.ascontiguousarray(xn)
        if not np.isfinite(xn).all():
            raise RingOverflowError("non-finite activation values cannot be quantized")
        want_soundness = self.min_soundness_bits > 0
        verify_ok = getattr(res, "take_checked_verify", False) is True
        ensure = getattr(res, "ensure", None)
        if ensure is not None:
            ensure(lid, t)
        else:
            left = int(res.remaining(lid))
            if left < t:
                raise MaskExhaustedError("session mask reservation exhausted",
                                         details={"linear": lid.key, "needed": t, "remaining": left})
        if want_soundness and not verify_ok:  # reservations without the hook: the batch path's pre-check
            token, u, _uw = self._current_checks(res, lid)
            self._check_soundness(lid, n, u, token)
        verify = self._sound_verifier(lid, n) if want_soundness and verify_ok else None
        ring = self.ring
        kbits, abits, cap = ring.k, ring.a, int(ring.cap or 0)
        nb = kbits // 8
        m = meta.out_features
        noise = self.noise_monitor
        plan_call = guard_packed = None
        if noise is not None:
            plan_call = getattr(noise, "plan_call", None)
            guard_packed = getattr(noise, "guard_packed", None)
            if plan_call is None or guard_packed is None:
                plan_call = guard_packed = None
        # off the critical path only where the inspection makes no random draw of its own (all rows
        # inspected), so the monitor's draws stay in the inline order
        noise_async = (self.noise_async_sampled and plan_call is not None
                       and hasattr(noise, "inspect_planned")
                       and t * n <= int(getattr(noise, "_max_values", 0) or 0))
        prep_into = native.fast.trip_prepare_into
        wn = float(meta.wnorm or 0.0)
        defer_lid = self._defer_ok(lid)
        threads = 0 if t * n >= int(self.trip_pool_min_elems) else 1
        max_item = self.max_rows_per_item
        chunks: list[_Chunk] = []
        sx_list: list[np.ndarray] = []
        row = 0
        try:
            while row < t:
                check = self._should_check()
                sampled = bool(plan_call()) if plan_call is not None else noise is not None
                want = check or want_soundness
                if verify is not None:
                    rows, r, wr, chk = res.take_checked(lid, min(t - row, max_item), want, verify)
                else:
                    rows, r, wr, chk = self._take(res, lid, min(t - row, max_item), want, prime_ok=True)
                token = u = uw = None
                rhs = None
                r64 = None
                r_np = None
                try:
                    if chk is not None:
                        token, u, uw = chk
                        if verify is None:
                            self._check_soundness(lid, n, u, token)
                        if not check:
                            token = u = uw = None
                    if tuple(r.shape) != (rows, n) or tuple(wr.shape) != (rows, m):
                        raise ShapeError(
                            f"{lid}: mask shapes {tuple(r.shape)}/{tuple(wr.shape)} do not match the linear"
                        )
                    r_np = r.numpy()
                    r64 = r_np if r_np.dtype == np.int64 and r_np.flags.c_contiguous else \
                        np.ascontiguousarray(r_np, dtype=np.int64)
                    buf = np.empty(rows * n * nb, dtype=np.uint8)
                    sx = np.empty(rows, dtype=np.float64)
                    xh = np.empty((rows, n), dtype=np.int64) if sampled else None
                    uw_in = None
                    prime = u is not None and is_prime_token(token)
                    if (prime or defer_lid) and u is not None and uw is not None and int(u.shape[0]) > 0:
                        uw_in = uw
                        rhs = np.empty((rows, int(u.shape[0])), dtype=np.uint64)
                    try:
                        if prime:  # prime input half uw . xq mod p (synchronous output half at finish)
                            prep_into(xn if rows == t else xn[row : row + rows], r64, kbits, abits, cap, wn, buf,
                                      sx, xh, uw_in, rhs, threads, field=P31)
                        else:
                            prep_into(xn if rows == t else xn[row : row + rows], r64, kbits, abits, cap, wn, buf,
                                      sx, xh, uw_in, rhs, threads)
                    except ValueError as exc:
                        if str(exc).startswith("ring_overflow"):
                            raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
                        raise ShapeError(str(exc)) from exc
                    payload = buf.tobytes()
                    if noise is not None:
                        if guard_packed is not None and (not sampled or noise_async):
                            guard_packed(payload, rows, n)  # every-call checks (duplicate rows)
                            if sampled and xh is not None:
                                self._noise_submit(noise, xh, r64)
                        elif xh is not None:
                            x_hat = torch.from_numpy(xh)
                            xq = center(x_hat - r.to(torch.int64), ring)
                            try:
                                if plan_call is not None:
                                    noise.guard(x_hat, xq, planned=True)  # type: ignore[attr-defined]
                                else:
                                    noise.guard(x_hat, xq)  # type: ignore[attr-defined]
                            finally:
                                _wipe(xq)
                except BaseException:
                    _wipe(wr)
                    if rhs is not None:
                        rhs.fill(0)
                    raise
                finally:
                    _wipe(r)  # one-time pad: never keep it around
                    if r64 is not None and r64 is not r_np:
                        r64.fill(0)
                chunks.append(_Chunk(0, row, rows, lid, None, wr, u, uw, payload, token, rhs=rhs))
                sx_list.append(sx)
                row += rows
        except BaseException:
            for ch in chunks:
                _wipe(ch.wr)
                if ch.rhs is not None:
                    ch.rhs.fill(0)
            raise
        from pgw.wire.messages import ComputeItem, ComputeRequest

        request = ComputeRequest(
            request_id=_uuid4_hex(),  # uuid4 format, as on every path
            fingerprint=self.fingerprint,
            ring=self.ring.name,
            items=tuple(ComputeItem(ch.lid.key, ch.rows, ch.payload) for ch in chunks),  # type: ignore[arg-type]
            deadline_ms=self.deadline_ms,
        )
        sxs = [sx_list[0] if len(sx_list) == 1 else np.concatenate(sx_list)]
        prep = _Prepared((call,), chunks, sxs, [meta], request, time.perf_counter() - t0)
        prep.trip = True
        prep.one = True
        prep.reservations = (res,)
        return prep

    # ---- native per-trip path (PGW_TRIP_NATIVE) ------------------------------------------------------
    def _trip_lease_for(self, res: Any, leases: dict[str, Any], lid: LinearId, meta: _MaskedMeta, t: int) -> Any:
        """The lease serving the next ``t`` rows of ``lid`` (the attached one, or a new one: the window's
        ``ensure`` / exhaustion pre-check and the extent's soundness check run before any row is taken), or
        ``None`` (the ordinary path)."""
        key = lid.key
        lease = leases.get(key)
        sound = (float(self.min_soundness_bits), len(self._sound_ok))
        if lease is not None and self._lease_sound.get(id(lease)) != sound:
            # soundness policy changed since this lease's extent was verified: verify again before any row
            lease.detach()
        if lease is None or key == res.trip_step_key or not lease.ready(t):
            prev = lease
            n = meta.in_features
            vkey = (key, n)
            verify = self._verifiers.get(vkey) if self.min_soundness_bits > 0 else None
            if verify is None and self.min_soundness_bits > 0:
                verify = self._verifiers[vkey] = self._sound_verifier(lid, n)
            ahead = _store_flags.TRIP_LEASE_AHEAD and getattr(res, "lease_ahead", None) is not None
            kw: dict[str, Any] = {}
            if self.prime_check:  # prime-field extents are leased only to a gateway that runs their check
                kw["prime_ok"] = True
            if ahead:
                kw["tag"] = (sound, bool(self.prime_check))
            lease = res.trip_lease(lid, t, verify, float(meta.wnorm or 0.0), **kw)
            if lease is None:
                return None
            if len(self._lease_sound) > 65536:
                self._lease_sound.clear()
            self._lease_sound[id(lease)] = (float(self.min_soundness_bits), len(self._sound_ok))
            if ahead and lease is not prev:
                self._lease_ahead_submit(res, lid, meta, t, kw["tag"])
        return lease

    def _lease_ahead_submit(self, res: Any, lid: LinearId, meta: _MaskedMeta, t: int, tag: Any) -> None:
        """Prepare ``lid``'s next trip lease of ``res`` on the lease-ahead thread (``PGW_TRIP_LEASE_AHEAD``;
        ``MaskReservation.lease_ahead``): rows the session already holds, attached only at its first row under
        the same ``tag`` (the soundness policy, checked there without recording), else zeroed. Timing only."""
        rk = (id(res), lid.key)
        with self._ahead_lock:
            if rk in self._ahead_jobs or self._closing:
                return
            self._ahead_jobs.add(rk)
            pool = self._ahead_pool
            if pool is None:
                from concurrent.futures import ThreadPoolExecutor

                pool = self._ahead_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-lease-ahead")
        n = meta.in_features
        verify = self._sound_verifier(lid, n, record=False) if self.min_soundness_bits > 0 else None
        wnorm = float(meta.wnorm or 0.0)
        prime_ok = bool(self.prime_check)

        def run() -> None:
            try:
                res.lease_ahead(lid, t, verify, wnorm, prime_ok=prime_ok, tag=tag)
            except Exception as exc:  # the lease path meets any error again and raises it there
                log.debug("trip lease ahead failed: %s", type(exc).__name__)
            finally:
                with self._ahead_lock:
                    self._ahead_jobs.discard(rk)

        try:
            pool.submit(run)
        except RuntimeError:  # shut down (closing)
            with self._ahead_lock:
                self._ahead_jobs.discard(rk)

    def _prepare_trip_lease(self, call: LinearCall) -> _Prepared | None:
        """:meth:`_prepare_trip_one` as one native call (:attr:`trip_native`), or ``None`` (no row taken,
        nothing drawn): the reservation hands no lease for this call, so the caller takes that path.

        Same order: shapes, then (in ``trip_lease``) the window's ``ensure`` / the exhaustion pre-check and
        the extent's soundness check before any row is taken, then the Freivalds draw, the noise-sampling
        draw, and ``lease.prepare`` (non-finite activations raise there before the row is taken).
        """
        t0 = time.perf_counter()
        res = self._reservation(call)
        leases = getattr(res, "trip_leases", None)
        if leases is None:
            return None
        lid = call.id
        meta: _MaskedMeta = self._meta.get(lid)
        x = call.x
        n = meta.in_features
        if x.shape[1] != n:
            raise ShapeError(f"{lid}: x must be (T, {n}), got {tuple(x.shape)}")
        t = int(x.shape[0])
        key = lid.key
        lease = self._trip_lease_for(res, leases, lid, meta, t)
        if lease is None:
            return None
        if x.dtype != torch.float32 or x.device.type != "cpu":
            x = x.to("cpu", torch.float32)
        xn = x.detach().numpy()
        if not xn.flags.c_contiguous:
            xn = np.ascontiguousarray(xn)
        noise = self.noise_monitor
        plan_call = guard_packed = None
        if noise is not None:
            plan_call = getattr(noise, "plan_call", None)
            guard_packed = getattr(noise, "guard_packed", None)
            if plan_call is None or guard_packed is None:
                plan_call = guard_packed = None
        check = self._should_check()
        sampled = bool(plan_call()) if plan_call is not None else noise is not None
        # prime-field leases check synchronously inside ticket.finish (never deferred)
        defer = check and not lease.field and self._defer_ok(lid)
        threads = 0 if t * n >= int(self.trip_pool_min_elems) else 1
        noise_async = False
        dup = None
        if noise is not None:
            noise_async = (self.noise_async_sampled and plan_call is not None
                           and hasattr(noise, "inspect_planned")
                           and t * n <= int(getattr(noise, "_max_values", 0) or 0))
            if guard_packed is not None and (not sampled or noise_async):
                # the every-call duplicate fingerprints are computed inside lease.prepare (same keys)
                params = getattr(noise, "packed_dup_params", None)
                dup = params(n) if params is not None else None
        try:
            if self.trip_split >= 2 and t * n >= self.trip_split_min_elems:
                got = lease.prepare(xn, check, sampled, threads, dup, self.trip_split)
            else:
                got = lease.prepare(xn, check, sampled, threads, dup)
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith(("ring_overflow", "nonfinite")):
                raise RingOverflowError(msg.split(":", 1)[-1].strip()) from exc
            raise ShapeError(msg) from exc
        if got is None:  # pragma: no cover - ready() was checked under the GIL just before
            raise MaskReuseError("mask lease could not serve a call it reported ready for")
        payload, ticket, xh, xq, keys = got
        try:
            if noise is not None:
                if guard_packed is not None and (not sampled or noise_async):
                    if keys is not None:
                        noise.guard_packed_keys(keys, t)  # every-call checks (duplicate rows)
                    else:
                        guard_packed(payload, t, n)  # every-call checks (duplicate rows)
                    if sampled and xh is not None:
                        self._noise_submit_xq(noise, xh, xq)
                        xq = None
                elif xh is not None:
                    if plan_call is not None:
                        noise.guard(torch.from_numpy(xh), torch.from_numpy(xq), planned=True)  # type: ignore[attr-defined]
                    else:
                        noise.guard(torch.from_numpy(xh), torch.from_numpy(xq))  # type: ignore[attr-defined]
        except BaseException:
            ticket.wipe()
            raise
        finally:
            if xq is not None:
                xq.fill(0)
        from pgw.wire.messages import ComputeItem, ComputeRequest

        request = ComputeRequest(
            request_id=_uuid4_hex(),  # uuid4 format, as on every path
            fingerprint=self.fingerprint,
            ring=self.ring.name,
            items=(ComputeItem(key, t, payload),),
            deadline_ms=self.deadline_ms,
        )
        prep = _Prepared((call,), [], [], [meta], request, time.perf_counter() - t0)
        prep.trip = True
        prep.one = True
        prep.ticket = ticket
        prep.defer = defer
        prep.reservations = (res,)
        return prep

    def _prepare_trip_leases(self, calls: Sequence[LinearCall]) -> _Prepared | None:
        """:meth:`_prepare_trip` for a trip of several one-chunk calls (one per session and linear: the decode
        step of concurrent answers) on the native per-trip path (:attr:`trip_native_multi`), or ``None``
        (nothing taken, nothing drawn): the caller then takes :meth:`_prepare_trip`.

        Same order and the same request as :meth:`_prepare_trip`: shapes and finite activations of every call
        are checked, and every call's lease is resolved (in ``trip_lease``: the window's ``ensure`` / the
        exhaustion pre-check and the extent's soundness check) before any row is taken; then, call by call,
        the Freivalds draw, the noise-sampling draw and ``lease.prepare`` (the same rows, ``r``, payload
        bytes and scales as ``trip_prepare_into``), the noise monitor's every-call check (or, for a sampled
        call, its statistics inline) before the next call is prepared. One item per call, in call order, one
        uuid4-format request id. Each call's ``ticket.finish`` gives :meth:`_finish_trip`'s outputs, statuses
        and deferred checks (:meth:`_finish_leases`)."""
        t0 = time.perf_counter()
        if len(calls) < 2:
            return None
        max_item = int(self.max_rows_per_item)
        reservations: list[Any] = []
        metas: list[_MaskedMeta] = []
        seen: set[tuple[int, str]] = set()
        mget = self._meta.get
        for c in calls:
            x = c.x
            if c.meta or x.ndim != 2:
                return None
            t = int(x.shape[0])
            if not 0 < t <= max_item:
                return None
            res = self._reservation(c)
            if getattr(res, "trip_leases", None) is None:
                return None
            meta: _MaskedMeta = mget(c.id)
            if x.shape[1] != meta.in_features:
                raise ShapeError(f"{c.id}: x must be (T, {meta.in_features}), got {tuple(x.shape)}")
            key = (id(res), c.id.key)
            if key in seen:  # two calls of one session and linear: rows taken in order by the batch path
                return None
            seen.add(key)
            reservations.append(res)
            metas.append(meta)
        xs: list[np.ndarray] = []
        for c in calls:
            x = c.x
            if x.dtype != torch.float32 or x.device.type != "cpu":
                x = x.to("cpu", torch.float32)
            xn = x.detach().numpy()
            if not xn.flags.c_contiguous:
                xn = np.ascontiguousarray(xn)
            xs.append(xn)
        for xn in xs:  # as _prepare_trip: no row of any session is taken for a batch with a non-finite value
            if xn.size and not np.isfinite(xn).all():
                raise RingOverflowError("non-finite activation values cannot be quantized")
        leases: list[Any] = []
        for c, res, meta, xn in zip(calls, reservations, metas, xs, strict=True):
            lease = self._trip_lease_for(res, res.trip_leases, c.id, meta, int(xn.shape[0]))
            if lease is None:  # no row taken yet (leases only reserve): the batch path serves the whole trip
                return None
            leases.append(lease)
        noise = self.noise_monitor
        plan_call = guard_packed = guard_keys = dup_params = None
        if noise is not None:
            plan_call = getattr(noise, "plan_call", None)
            guard_packed = getattr(noise, "guard_packed", None)
            if plan_call is None or guard_packed is None:
                plan_call = guard_packed = None
            else:
                guard_keys = getattr(noise, "guard_packed_keys", None)
                dup_params = getattr(noise, "packed_dup_params", None)
        pool_min = int(self.trip_pool_min_elems)
        split = int(self.trip_split)
        split_min = int(self.trip_split_min_elems)
        tickets: list[Any] = []
        defers: list[bool] = []
        items: list[Any] = []
        from pgw.wire.messages import ComputeItem, ComputeRequest

        if split < 2 and native.native_available("trip_prepare_many"):
            return self._prepare_trip_leases_many(calls, reservations, metas, xs, leases, t0)
        try:
            for c, lease, meta, xn in zip(calls, leases, metas, xs, strict=True):
                t = int(xn.shape[0])
                n = meta.in_features
                lid = c.id
                check = self._should_check()
                sampled = bool(plan_call()) if plan_call is not None else noise is not None
                defer = check and not lease.field and self._defer_ok(lid)
                threads = 0 if t * n >= pool_min else 1
                dup = None
                if guard_packed is not None and not sampled and guard_keys is not None and dup_params is not None:
                    dup = dup_params(n)  # the every-call fingerprints, computed inside lease.prepare (same keys)
                try:
                    if split >= 2 and t * n >= split_min:
                        got = lease.prepare(xn, check, sampled, threads, dup, split)
                    else:
                        got = lease.prepare(xn, check, sampled, threads, dup)
                except ValueError as exc:
                    msg = str(exc)
                    if msg.startswith(("ring_overflow", "nonfinite")):
                        raise RingOverflowError(msg.split(":", 1)[-1].strip()) from exc
                    raise ShapeError(msg) from exc
                if got is None:  # pragma: no cover - ready() was checked under the GIL just before
                    raise MaskReuseError("mask lease could not serve a call it reported ready for")
                payload, ticket, xh, xq, keys = got
                tickets.append(ticket)
                defers.append(defer)
                try:
                    if noise is not None:
                        if guard_packed is not None and not sampled:
                            if keys is not None:
                                guard_keys(keys, t)  # type: ignore[misc]  # every-call checks (duplicate rows)
                            else:
                                guard_packed(payload, t, n)
                        elif xh is not None:
                            if plan_call is not None:
                                noise.guard(torch.from_numpy(xh), torch.from_numpy(xq), planned=True)  # type: ignore[attr-defined]
                            else:
                                noise.guard(torch.from_numpy(xh), torch.from_numpy(xq))  # type: ignore[attr-defined]
                finally:
                    if xq is not None:
                        xq.fill(0)
                    if xh is not None:
                        xh.fill(0)
                items.append(ComputeItem(lid.key, t, payload))
        except BaseException:
            for tk in tickets:
                tk.wipe()
            raise
        request = ComputeRequest(
            request_id=_uuid4_hex(),  # uuid4 format, as on every path
            fingerprint=self.fingerprint,
            ring=self.ring.name,
            items=tuple(items),
            deadline_ms=self.deadline_ms,
        )
        prep = _Prepared(tuple(calls), [], [], metas, request, time.perf_counter() - t0)
        prep.trip = True
        prep.tickets = tickets
        prep.defers = defers
        prep.reservations = tuple(reservations)
        return prep

    #: Rows x features of a several-answer trip from which ``trip_prepare_many`` / ``trip_finish_many`` spread its
    #: items over the native pool (below: in order on the calling thread, still one GIL release).
    trip_many_parallel_elems = 1 << 15

    def _prepare_trip_leases_many(self, calls: Sequence[LinearCall], reservations: list[Any],
                                  metas: list[_MaskedMeta], xs: list[np.ndarray], leases: list[Any],
                                  t0: float) -> _Prepared:
        """:meth:`_prepare_trip_leases` after the leases with one ``pgw_fast.trip_prepare_many`` call: the
        Freivalds and noise-sampling draws of every call first, in call order (the same sequence as the per-call
        loop), then every item prepared in one native call (each lease takes its own rows: the same rows, ``r``,
        payload bytes and scales in any schedule), then the noise monitor's every-call checks (or a sampled call's
        statistics) in call order, all before the request exists. On failure every ticket of the trip is wiped
        (rows already taken stay consumed; nothing was sent)."""
        noise = self.noise_monitor
        plan_call = guard_packed = guard_keys = dup_params = None
        if noise is not None:
            plan_call = getattr(noise, "plan_call", None)
            guard_packed = getattr(noise, "guard_packed", None)
            if plan_call is None or guard_packed is None:
                plan_call = guard_packed = None
            else:
                guard_keys = getattr(noise, "guard_packed_keys", None)
                dup_params = getattr(noise, "packed_dup_params", None)
        flags = bytearray()
        defers: list[bool] = []
        dups: list[Any] = []
        for c, lease, meta in zip(calls, leases, metas, strict=True):
            check = self._should_check()
            sampled = bool(plan_call()) if plan_call is not None else noise is not None
            defers.append(check and not lease.field and self._defer_ok(c.id))
            flags.append((1 if check else 0) | (2 if sampled else 0))
            dup = None
            if guard_packed is not None and not sampled and guard_keys is not None and dup_params is not None:
                dup = dup_params(meta.in_features)
            dups.append(dup)
        try:
            got = native.fast.trip_prepare_many(leases, xs, bytes(flags), dups, int(self.trip_many_parallel_elems))
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith("not_ready"):
                raise MaskReuseError("mask lease could not serve a call it reported ready for") from exc
            if msg.startswith(("ring_overflow", "nonfinite")):
                raise RingOverflowError(msg.split(":", 1)[-1].strip()) from exc
            raise ShapeError(msg) from exc
        from pgw.wire.messages import ComputeItem, ComputeRequest

        tickets = [g[1] for g in got]
        items: list[Any] = []
        try:
            for c, meta, f, (payload, _tk, xh, xq, keys) in zip(calls, metas, flags, got, strict=True):
                t = int(c.x.shape[0])
                try:
                    if noise is not None:
                        if guard_packed is not None and not f & 2:
                            if keys is not None:
                                guard_keys(keys, t)  # type: ignore[misc]  # every-call checks (duplicate rows)
                            else:
                                guard_packed(payload, t, meta.in_features)
                        elif xh is not None:
                            if plan_call is not None:
                                noise.guard(torch.from_numpy(xh), torch.from_numpy(xq), planned=True)  # type: ignore[attr-defined]
                            else:
                                noise.guard(torch.from_numpy(xh), torch.from_numpy(xq))  # type: ignore[attr-defined]
                finally:
                    if xq is not None:
                        xq.fill(0)
                    if xh is not None:
                        xh.fill(0)
                items.append(ComputeItem(c.id.key, t, payload))
        except BaseException:
            for tk in tickets:
                tk.wipe()
            for g in got:  # the statistics of later sampled calls never ran: zero their copies too
                for a in (g[2], g[3]):
                    if a is not None:
                        a.fill(0)
            raise
        request = ComputeRequest(
            request_id=_uuid4_hex(),  # uuid4 format, as on every path
            fingerprint=self.fingerprint,
            ring=self.ring.name,
            items=tuple(items),
            deadline_ms=self.deadline_ms,
        )
        prep = _Prepared(tuple(calls), [], [], metas, request, time.perf_counter() - t0)
        prep.trip = True
        prep.tickets = tickets
        prep.defers = defers
        prep.reservations = tuple(reservations)
        return prep

    def _finish_leases(self, prep: _Prepared, response: Any, remote_s: float) -> list[torch.Tensor]:
        """:meth:`_finish` (:meth:`_finish_trip`) for a trip of :meth:`_prepare_trip_leases`: the same response
        checks, one ``ticket.finish`` per item in item order (Freivalds when drawn, synchronously or deferred to
        the helper thread; unmask, range check, rescale, ``W r`` zeroed), the same statuses, pending checks,
        statistics and integrity reporting."""
        t0 = time.perf_counter()
        req = prep.request
        tickets = prep.tickets or []
        outs: list[torch.Tensor] = []
        bytes_down = checked = skipped = 0
        rows_total = 0
        submitted: list[tuple[int, Any]] = []
        pool_min = int(self.trip_pool_min_elems)
        split = int(self.trip_split)
        split_min = int(self.trip_split_min_elems)
        try:
            if getattr(response, "request_id", None) != req.request_id:
                raise ProtocolError("compute response id does not match the request")
            results = tuple(response.items)
            if len(results) != len(tickets):
                raise ProtocolError(f"compute response has {len(results)} items, expected {len(tickets)}")
            for i, (call, _meta, tk, item) in enumerate(zip(prep.calls, prep.metas, tickets, results, strict=True)):
                lid = call.id
                rows = int(tk.rows)
                if item.linear != lid.key or int(item.rows) != rows:
                    raise ProtocolError(
                        "compute response item does not match the request",
                        details={"expected": lid.key, "got": str(item.linear)},
                    )
                y = item.y_hat
                bytes_down += len(y)
                if tk.checked:
                    if not prep.defers[i]:
                        checked += 1
                else:
                    skipped += 1
            fin: list[Any] | None = None
            if split < 2 and native.native_available("trip_finish_many"):
                # every item in one native call (the deferred checks submitted first, in item order; the kernels
                # on the native pool when the outputs are large): the same outputs and statuses as one finish per
                # item; the statuses are then handled in item order below, as with separate calls
                ys, scs, bss, bds = [], [], [], []
                for call, meta, item in zip(prep.calls, prep.metas, results, strict=True):
                    y = item.y_hat
                    ys.append(y if isinstance(y, bytes) else np.frombuffer(y, dtype=np.uint8))
                    scale, bias, bound = self._consts(call.id, meta)
                    scs.append(scale)
                    bss.append(bias)
                    bds.append(int(bound))
                try:
                    fin = native.fast.trip_finish_many(tickets, ys, scs, bss, bds, list(prep.defers),
                                                       int(self.trip_many_parallel_elems))
                except ValueError as exc:
                    raise ProtocolError(f"compute response item is malformed: {exc}") from exc
            for i, (call, meta, tk, item) in enumerate(zip(prep.calls, prep.metas, tickets, results, strict=True)):
                lid = call.id
                rows = int(tk.rows)
                rows_total += rows
                scale, bias, bound = self._consts(lid, meta)
                if fin is not None:
                    status, bad, out, pending = fin[i]
                else:
                    y = item.y_hat
                    yb = y if isinstance(y, bytes) else np.frombuffer(y, dtype=np.uint8)
                    m = meta.out_features
                    threads = 0 if rows * m >= pool_min else 1
                    try:
                        if split >= 2 and rows * m >= split_min:
                            status, bad, out, pending = tk.finish(yb, scale, bias, bound, prep.defers[i], threads,
                                                                  split)
                        else:
                            status, bad, out, pending = tk.finish(yb, scale, bias, bound, prep.defers[i], threads)
                    except ValueError as exc:
                        raise ProtocolError(f"compute response item is malformed: {exc}") from exc
                if status:
                    if pending is not None:  # its verdict no longer matters: wait (it reads y_hat and u)
                        pending.wait()
                    for _, p in submitted:  # this request fails: no later check of it can matter either
                        p.wait()
                    if fin is not None:  # the later items' checks were submitted too
                        for later in fin[i + 1:]:
                            if later[3] is not None:
                                later[3].wait()
                    self._raise_status(status, bad, lid, rows, bound)
                if pending is not None:
                    submitted.append((i, pending))
                y_out = torch.from_numpy(out)
                dev = call.x.device
                outs.append(y_out if dev.type == "cpu" else y_out.to(dev))
            if submitted:
                entries = [_PendingCheck(prep.reservations, p, prep.calls[i].id, int(tickets[i].rows), req.request_id)
                           for i, p in submitted]
                with self._pending_lock:
                    self._pending.extend(entries)
                self.stats.add(deferred_checks=len(entries))
        except IntegrityError as exc:
            self.stats.add(integrity_failures=1)
            log.error("integrity check failed: request=%s code=%s details=%s", req.request_id, exc.code, exc.details)
            if self.on_integrity_failure is not None:
                try:
                    self.on_integrity_failure(exc)
                except Exception:  # pragma: no cover - callback must not mask the error
                    log.exception("on_integrity_failure callback raised")
            raise
        finally:
            prep.wipe()
        bytes_up = sum(len(it.x_hat) for it in req.items)
        self.stats.add_many(len(prep.calls), rows_total, 1, bytes_up, bytes_down, checked, skipped, remote_s,
                            prep.t_prepare + time.perf_counter() - t0, 0 < rows_total <= self.decode_max_rows)
        return outs

    def _noise_submit_xq(self, noise: Any, xh: np.ndarray, xq: np.ndarray) -> None:
        """:meth:`_noise_submit` with ``xq`` already computed (owned here: wiped after the inspection)."""
        def run() -> None:
            try:
                noise.inspect_planned(torch.from_numpy(xh), torch.from_numpy(xq))
            finally:
                xh.fill(0)
                xq.fill(0)

        pool = self._noise_pool
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor

            pool = self._noise_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-noise")
        self._noise_pending = pool.submit(run)

    def _finish_lease(self, prep: _Prepared, response: Any, remote_s: float,
                      drive: bool = False) -> list[torch.Tensor] | None:
        """:meth:`_finish_one` for a trip of the native per-trip path: the same response checks, one
        ``ticket.finish`` (Freivalds when drawn, synchronously or deferred to the helper thread; unmask,
        range check, rescale, W r zeroed), the same statistics and integrity reporting.

        ``drive`` (:attr:`trip_driver`): the lane call itself runs here, fused with the finish
        (``client.trip_ticket``); ``None`` if the lane did not serve the call (nothing wiped: the caller sends
        the request through the fallback client and calls this again with its response)."""
        t0 = time.perf_counter()
        req = prep.request
        tk = prep.ticket
        call = prep.calls[0]
        lid = call.id
        meta: _MaskedMeta = prep.metas[0]
        rows = int(tk.rows)
        bytes_down = 0
        pending = None
        fin = None
        if drive:
            scale, bias, bound = self._consts(lid, meta)
            threads = 0 if rows * meta.out_features >= int(self.trip_pool_min_elems) else 1
            split = (self.trip_split if self.trip_split >= 2
                     and rows * meta.out_features >= self.trip_split_min_elems else 0)
            try:
                got = self.client.trip_ticket(req, tk, scale, bias, bound, prep.defer, threads, split)
            except ValueError as exc:
                prep.wipe()
                raise ProtocolError(f"compute response item is malformed: {exc}") from exc
            except BaseException:
                prep.wipe()
                raise
            if got is None:
                return None
            fin, lane_s, bytes_down = got
            remote_s = lane_s
            t0 = time.perf_counter()
        try:
            if fin is not None:
                status, bad, out, pending = fin
                if status:
                    if pending is not None:
                        pending.wait()
                        pending = None
                    self._raise_status(status, bad, lid, rows, bound)
                if pending is not None:
                    with self._pending_lock:
                        self._pending.append(_PendingCheck(prep.reservations, pending, lid, rows, req.request_id))
                    self.stats.add(deferred_checks=1)
            else:
                if getattr(response, "request_id", None) != req.request_id:
                    raise ProtocolError("compute response id does not match the request")
                results = tuple(response.items)
                if len(results) != 1:
                    raise ProtocolError(f"compute response has {len(results)} items, expected 1")
                item = results[0]
                if item.linear != lid.key or int(item.rows) != rows:
                    raise ProtocolError(
                        "compute response item does not match the request",
                        details={"expected": lid.key, "got": str(item.linear)},
                    )
                y = item.y_hat
                bytes_down = len(y)
                yb = y if isinstance(y, bytes) else np.frombuffer(y, dtype=np.uint8)
                scale, bias, bound = self._consts(lid, meta)
                threads = 0 if rows * meta.out_features >= int(self.trip_pool_min_elems) else 1
                try:
                    if self.trip_split >= 2 and rows * meta.out_features >= self.trip_split_min_elems:
                        status, bad, out, pending = tk.finish(yb, scale, bias, bound, prep.defer, threads,
                                                              self.trip_split)
                    else:
                        status, bad, out, pending = tk.finish(yb, scale, bias, bound, prep.defer, threads)
                except ValueError as exc:
                    raise ProtocolError(f"compute response item is malformed: {exc}") from exc
                if status:
                    if pending is not None:  # its verdict no longer matters: wait (it reads y_hat and u)
                        pending.wait()
                        pending = None
                    self._raise_status(status, bad, lid, rows, bound)
                if pending is not None:
                    with self._pending_lock:
                        self._pending.append(_PendingCheck(prep.reservations, pending, lid, rows, req.request_id))
                    self.stats.add(deferred_checks=1)
        except IntegrityError as exc:
            self.stats.add(integrity_failures=1)
            log.error("integrity check failed: request=%s code=%s details=%s", req.request_id, exc.code, exc.details)
            if self.on_integrity_failure is not None:
                try:
                    self.on_integrity_failure(exc)
                except Exception:  # pragma: no cover - callback must not mask the error
                    log.exception("on_integrity_failure callback raised")
            raise
        finally:
            prep.wipe()
        checked = 1 if tk.checked and pending is None else 0
        skipped = 0 if tk.checked else 1
        self.stats.add_many(1, rows, 1, len(req.items[0].x_hat), bytes_down, checked, skipped,
                            remote_s, prep.t_prepare + time.perf_counter() - t0, 0 < rows <= self.decode_max_rows)
        y_out = torch.from_numpy(out)
        dev = call.x.device
        return [y_out if dev.type == "cpu" else y_out.to(dev)]

    def _finish_trip(self, prep: _Prepared, results: Sequence[Any], outs: list[list[torch.Tensor]]) -> None:
        """Finish every chunk with ``pgw_fast.trip_finish_into`` (same outputs and statuses as
        ``finish_group``). A chunk with a deferred check submits the Freivalds output half to the native
        helper thread first and finishes (unmask, range check, rescale) meanwhile; the check is
        registered as pending and joined before this session's next payload is built."""
        fin = native.fast.trip_finish_into
        k = self.ring.k
        pool_min = int(self.trip_pool_min_elems)
        submitted: list[tuple[_Chunk, Any]] = []
        for ch, res in zip(prep.chunks, results, strict=True):
            meta: _MaskedMeta = prep.metas[ch.call_index]
            scale, bias, bound = self._consts(ch.lid, meta)
            sx = prep.sx[ch.call_index]
            if ch.row0 or ch.rows != sx.shape[0]:
                sx = sx[ch.row0 : ch.row0 + ch.rows]
            y = res.y_hat
            yb = y if isinstance(y, bytes) else np.frombuffer(y, dtype=np.uint8)
            wn = ch.wr.numpy()
            in_place = wn.dtype in (np.int64, np.uint64, np.uint32) and wn.flags.c_contiguous and wn.flags.writeable
            if not in_place:  # e.g. a strided test double: a copy (zeroed by the kernel); prep.wipe() zeroes ch.wr
                wn = np.ascontiguousarray(wn, dtype=np.int64)
            out = np.empty((ch.rows, meta.out_features), dtype=np.float32)
            threads = 0 if ch.rows * meta.out_features >= pool_min else 1
            pending = None
            try:
                if ch.u is not None and ch.rhs is not None and is_prime_token(ch.token):
                    # prime-field check (synchronous): range + u . yq == rhs (mod p) before W r is wiped
                    status, bad = native.fast.trip_prime_output(yb, wn, ch.u, ch.rhs, k, bound)
                    ch.rhs.fill(0)
                    ch.rhs = None
                    if status == 0:
                        status, bad = fin(yb, wn, sx, scale, bias, k, bound, out, threads=threads)
                    else:
                        wn.fill(0)  # never unmasked: zero the pad-derived W r here (fail closed)
                        out.fill(0)
                elif ch.u is not None and ch.rhs is not None:
                    pending = native.fast.trip_freivalds_submit(yb, ch.u, ch.rhs, k)
                    ch.rhs.fill(0)  # the helper holds its own copy (wiped after the check)
                    ch.rhs = None
                    status, bad = fin(yb, wn, sx, scale, bias, k, bound, out, threads=threads)
                elif ch.u is not None and ch.uw is not None:
                    status, bad = fin(yb, wn, sx, scale, bias, k, bound, out,
                                      x_hat=bytes(ch.payload),  # type: ignore[arg-type]
                                      u=ch.u, uw=ch.uw, threads=threads)
                else:
                    status, bad = fin(yb, wn, sx, scale, bias, k, bound, out, threads=threads)
            except ValueError as exc:
                raise ProtocolError(f"compute response item is malformed: {exc}") from exc
            finally:
                if in_place:
                    ch.wr_wiped = True  # the kernel zeroed W r (any returned status)
            if pending is None and ch.u is not None and _vacuous(ch.u, status):
                status = 3  # vectors wiped (reservation closed): an all-zero u verifies nothing
            if status:
                if pending is not None:  # its verdict no longer matters: wait (it reads y_hat and u)
                    pending.wait()
                for _, p in submitted:  # this request fails: no later check of it can matter either
                    p.wait()
                self._raise_status(status, bad, ch.lid, ch.rows, bound)
            if pending is not None:
                submitted.append((ch, pending))
            outs[ch.call_index].append(torch.from_numpy(out))
        if submitted:
            entries = [_PendingCheck(prep.reservations, p, ch.lid, ch.rows,
                                     prep.request.request_id if prep.request is not None else "-")
                       for ch, p in submitted]
            with self._pending_lock:
                self._pending.extend(entries)
            self.stats.add(deferred_checks=len(entries))

    def _join_pending(self, calls: Sequence[LinearCall] | None) -> None:
        """Wait for the deferred checks of the sessions of ``calls`` (``None``: every pending check).

        Other sessions' checks are never raised here: finished ones are harvested without waiting
        (which also drops their references to the reservations, e.g. of an aborted request), and above
        :attr:`max_pending_checks` the oldest are waited for. A failed one is reported at once (log,
        ``integrity_failures``, ``on_integrity_failure``) and raised in its own session's next call.

        Raises:
            IntegrityError: a deferred check of these sessions failed (the service returned a wrong result).
            MaskReuseError: the check vectors were wiped meanwhile (reservation closed).
        """
        with self._pending_lock:
            if not self._pending and not self._failed:
                return
            mine: list[_PendingCheck]
            others: list[_PendingCheck] = []
            stashed: list[_FailedCheck]
            if calls is None:
                mine = self._pending
                stashed = [f for f in self._failed if f.alive()]
                self._failed = []
            else:
                ids = {id(c.session if c.session is not None else self.mask_source) for c in calls}
                mine = []
                for p in self._pending:
                    (mine if any(id(r) in ids for r in p.reservations) else others).append(p)
                stashed = []
                keep_failed: list[_FailedCheck] = []
                for f in self._failed:
                    (stashed if f.owned_by(ids) else keep_failed).append(f)
                self._failed = keep_failed
            # other sessions: harvest what already finished; wait for the oldest above the limit
            excess = max(0, len(others) - int(self.max_pending_checks))
            harvest = others[:excess]
            still: list[_PendingCheck] = []
            for p in others[excess:]:
                (harvest if p.check.done() else still).append(p)
            self._pending = still
        if harvest:
            for p, err in self._wait_checks(harvest):
                if err is not None:
                    self._report_deferred(err, p)
                    self._stash_failure(p, err)
        err_mine: PGWError | None = None
        for p, err in self._wait_checks(mine):  # wait for every one (each reads buffers it keeps alive)
            if err is not None:
                self._report_deferred(err, p)
                if err_mine is None:
                    err_mine = err
        if stashed and err_mine is None:
            err_mine = stashed[0].error
        if err_mine is not None:
            raise err_mine

    def _wait_checks(self, checks: Sequence[_PendingCheck]) -> list[tuple[_PendingCheck, PGWError | None]]:
        """Wait for ``checks``; their verdicts (``None``: verified, counted in ``freivalds_checked``)."""
        out: list[tuple[_PendingCheck, PGWError | None]] = []
        ok = 0
        for p in checks:
            status, bad = p.check.wait()
            err: PGWError | None = None
            if status == 3:
                err = MaskReuseError("mask reservation closed during a compute call")
            elif status == 1:
                err = IntegrityError("compute service result failed the Freivalds check",
                                     details={"linear": p.lid.key, "bad_rows": int(bad), "rows": p.rows,
                                              "deferred": True})
            elif status:  # 4: the helper failed internally (fail closed)
                err = IntegrityError("deferred Freivalds check could not run",
                                     details={"linear": p.lid.key, "status": int(status), "deferred": True})
            else:
                ok += 1
            out.append((p, err))
        if ok:
            self.stats.add(freivalds_checked=ok)
        return out

    def _report_deferred(self, err: PGWError, p: _PendingCheck) -> None:
        """Log, count and call ``on_integrity_failure`` for a failed deferred check (once per failure)."""
        if not isinstance(err, IntegrityError):
            log.warning("deferred check could not complete: request=%s code=%s", p.request_id, err.code)
            return
        self.stats.add(integrity_failures=1)
        log.error("integrity check failed (deferred): request=%s code=%s details=%s",
                  p.request_id, err.code, err.details)
        if self.on_integrity_failure is not None:
            try:
                self.on_integrity_failure(err)
            except Exception:  # pragma: no cover - callback must not mask the error
                log.exception("on_integrity_failure callback raised")

    def _stash_failure(self, p: _PendingCheck, err: PGWError) -> None:
        f = _FailedCheck(tuple(_weak_or_strong(r) for r in p.reservations), err)
        with self._pending_lock:
            self._failed = [x for x in self._failed if x.alive()]
            self._failed.append(f)
            over = len(self._failed) - max(1, int(self.max_pending_checks))
            if over > 0:  # already reported; only the raise in that session's next call is lost
                del self._failed[:over]

    def join_checks(self) -> None:
        """Join every deferred check now (raises like :meth:`forward_many` on a failed one)."""
        self._join_pending(None)

    def release_session_checks(self, session: Any) -> bool:
        """Gateway hook for a session that ends or aborts (e.g. between gate_up and down): wait for
        its deferred checks and drop every reference to it. Failures are reported (log, stats,
        ``on_integrity_failure``), never raised: no output of that session is used any more.

        Returns True when every check of the session joined here passed and none had failed before
        (a session's KV cache may then be kept for the next turn of its conversation)."""
        with self._pending_lock:
            mine = [p for p in self._pending if any(r is session for r in p.reservations)]
            if mine:
                self._pending = [p for p in self._pending if not any(r is session for r in p.reservations)]
            failed_before = any(f.owned_by({id(session)}) for f in self._failed)
            self._failed = [f for f in self._failed if not f.owned_by({id(session)}) and f.alive()]
        ok = not failed_before
        for p, err in self._wait_checks(mine):
            if err is not None:
                ok = False
                self._report_deferred(err, p)
        return ok

    @property
    def pending_checks(self) -> int:
        """Deferred checks not joined yet (diagnostics)."""
        with self._pending_lock:
            return len(self._pending)

    # ---- synchronous trips (engine hook) -----------------------------------------------------------
    def sync_trip_ok(self, rows: int) -> bool:
        """Engine hook (``TransformerModel.sync_decode``): a pass of ``rows`` rows may run as one
        plaintext-thread call over :meth:`forward_many`. Yes for decode-sized passes when synchronous
        trips are on (a trip-lane client and the TripEngine). :meth:`forward_many` never touches the
        caller's event loop (the client's ``compute_sync`` runs on the lane or its private loop)."""
        rows = int(rows)
        if not self.sync_trips or rows <= 0:
            return False
        if rows <= int(self.sync_trip_max_rows):
            return True
        if not self.sync_multi:
            return False
        cap = getattr(self.client, "multi_rows", None)
        return callable(cap) and rows <= min(int(cap()), int(self.sync_multi_max_rows))

    # ---- speculation padding rows -------------------------------------------------------------------
    def _random_r(self, rows: int, n: int) -> np.ndarray:
        """Fresh ring elements ``(rows, n)``, never a store mask: the ChaCha20 stream of a fresh OS-random key,
        read exactly as the store reads a mask's ``r`` (:meth:`SeedExpander.ring_uniform_rows_with_key`). Same
        distribution as real ``r`` (each is the low K bits of a ChaCha20 stream under a secret key) and the
        same expansion cost per row, so a padding row's prepare time does not stand out from a real row's
        (L3: the gateway's turnaround per trip must not depend on how many rows are padding)."""
        from pgw.core.prng import SeedExpander

        key = bytearray(os.urandom(32))
        try:
            return SeedExpander.ring_uniform_rows_with_key(key, n, self.ring.k, 0, rows).numpy()
        finally:
            key[:] = bytes(32)

    def _filler_payload(self, x: np.ndarray | None, rows: int, meta: _MaskedMeta) -> bytes:
        """Payload of padding rows: ``x`` (zeros for padding rows of a real call) quantized and masked
        with a fresh local random ``r``, so the bytes are uniform ring elements, and the prepare work
        is the real kernel's. ``r`` is not kept (nothing to unmask: the outputs are dropped)."""
        n = meta.in_features
        if x is None:
            x = np.zeros((rows, n), dtype=np.float32)
        r = self._random_r(rows, n)
        try:
            ring = self.ring
            wn = float(meta.wnorm or 0.0)
            try:
                if self.trip_engine:
                    buf = np.empty(rows * n * (ring.k // 8), dtype=np.uint8)
                    native.fast.trip_prepare_into(x, r, ring.k, ring.a, int(ring.cap or 0), wn, buf,
                                                  np.empty(rows, dtype=np.float64))
                    return buf.tobytes()
                if self._native_prepare_ok_meta(meta):
                    payload, _, _ = native.fast.quantize_mask_pack(x, r, ring.k, ring.a, int(ring.cap or 0), wn, False)
                    return bytes(payload)
            except ValueError as exc:
                if str(exc).startswith("ring_overflow"):
                    raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
                raise ShapeError(str(exc)) from exc
            xq, _ = quantize_activations(torch.from_numpy(x), ring, meta.wnorm if ring.cap else None)
            try:
                return pack_ring(mask_activations(xq, torch.from_numpy(r), ring), ring)
            finally:
                _wipe(xq)
        finally:
            r.fill(0)

    def _native_prepare_ok_meta(self, meta: _MaskedMeta) -> bool:
        return (self._native_prepare and self.ring.k % 8 == 0 and self.ring.k <= 56
                and (not self.ring.cap or meta.wnorm is not None))

    def _prepare_padded(self, calls: Sequence[LinearCall]) -> _Prepared:
        """Prepare calls that carry speculation padding (``meta[PAD_ROWS]``) or are dummies (``meta[DUMMY]``).

        Real rows are prepared exactly as without padding (same masks, same bytes). Padding rows are
        appended to the wire item of their call's last chunk, so the provider sees one item of
        ``real + pad`` rows; they take no mask row. A dummy call (or a padded call without real rows)
        becomes an item of padding rows only, built on the real prepare kernel with a local random
        ``r``. Padding payloads are made before any mask row is taken.
        """
        metas_all: list[_MaskedMeta] = []
        pads: list[int] = []
        fill_payload: dict[int, bytes] = {}
        for i, c in enumerate(calls):
            meta: _MaskedMeta = self._meta.get(c.id)
            metas_all.append(meta)
            if c.x.ndim != 2 or c.x.shape[1] != meta.in_features:
                raise ShapeError(f"{c.id}: x must be (T, {meta.in_features}), got {tuple(c.x.shape)}")
            if _is_dummy(c):
                rows = int(c.x.shape[0]) + _pad_rows(c)
                if rows:
                    xd = c.x.detach().to("cpu", torch.float32).numpy()
                    if _pad_rows(c):
                        xd = np.concatenate([xd, np.zeros((_pad_rows(c), meta.in_features), dtype=np.float32)])
                    xd = np.ascontiguousarray(xd)
                    if not np.isfinite(xd).all():
                        raise RingOverflowError("non-finite activation values cannot be quantized")
                    fill_payload[i] = self._filler_payload(xd, rows, meta)
                pads.append(rows)
            else:
                p = _pad_rows(c)
                if p:
                    fill_payload[i] = self._filler_payload(None, p, meta)
                pads.append(p)
        real_idx = [i for i, c in enumerate(calls) if not _is_dummy(c) and int(c.x.shape[0]) > 0]
        real_calls = [calls[i] for i in real_idx]
        if real_calls:
            core = self._prepare_core(real_calls)
        else:
            core = _Prepared([], [], [], [], None, 0.0)
        try:
            t0 = time.perf_counter()
            pos = {j: i for j, i in enumerate(real_idx)}  # real call index -> original index
            chunks: list[_Chunk] = []
            fillers: list[_Filler] = []
            items_map: list[tuple[str, int]] = []
            by_call: dict[int, list[_Chunk]] = {}
            for ch in core.chunks:
                ch.call_index = pos[ch.call_index]
                by_call.setdefault(ch.call_index, []).append(ch)
            sxs: list[np.ndarray] = [np.zeros(0, dtype=np.float64) for _ in calls]
            for j, i in enumerate(real_idx):
                sxs[i] = core.sx[j]
            groups: list[list[int]] = []
            for i, c in enumerate(calls):
                own = by_call.get(i, [])
                if own and pads[i] and not _is_dummy(c):
                    own[-1].pad = pads[i]
                    own[-1].pad_payload = fill_payload[i]
                if len(own) > 1 and not _is_dummy(c):
                    # A padded call whose real rows span mask extents (segment or window boundaries) goes out
                    # as ONE item: where its rows split depends on how many masks the session used so far,
                    # which (with padding) depends on the drafts; the item size is the public R.
                    idx = []
                    for ch in own:
                        idx.append(len(chunks))
                        chunks.append(ch)
                    items_map.append(("g", len(groups)))
                    groups.append(idx)
                else:
                    for ch in own:
                        items_map.append(("c", len(chunks)))
                        chunks.append(ch)
                if pads[i] and not own:  # dummy, or padding of a call without real rows
                    # Verified like real rows (the reservation's current check vectors, at the same rate):
                    # an unverified dummy would let a provider that corrupts results learn which passes
                    # are dummies (no integrity failure) and so how many passes were real.
                    checks = self._filler_checks(c)
                    items_map.append(("f", len(fillers)))
                    fillers.append(_Filler(i, c.id, pads[i], fill_payload[i], checks))
            request = None
            if items_map:
                from pgw.wire.messages import ComputeItem, ComputeRequest

                items = []
                def body_of(ch: _Chunk) -> bytes:
                    b = ch.payload if ch.payload is not None else pack_ring(ch.x_hat, self.ring)  # type: ignore[arg-type]
                    return bytes(b) + ch.pad_payload if ch.pad else b  # type: ignore[operator]

                for kind, j in items_map:
                    if kind == "c":
                        ch = chunks[j]
                        items.append(ComputeItem(ch.lid.key, ch.rows + ch.pad, body_of(ch)))
                    elif kind == "g":
                        part = [chunks[x] for x in groups[j]]
                        items.append(ComputeItem(part[0].lid.key, sum(ch.rows + ch.pad for ch in part),
                                                 b"".join(bytes(body_of(ch)) for ch in part)))
                    else:
                        f = fillers[j]
                        items.append(ComputeItem(f.lid.key, f.rows, f.payload))
                request = ComputeRequest(
                    request_id=core.request.request_id if core.request is not None else uuid.uuid4().hex,
                    fingerprint=self.fingerprint,
                    ring=self.ring.name,
                    items=tuple(items),
                    deadline_ms=self.deadline_ms,
                )
            reservations = list(core.reservations)
            for c in calls:
                if _is_dummy(c) or int(c.x.shape[0]) == 0:
                    res = c.session if c.session is not None else self.mask_source
                    if res is not None:
                        reservations.append(res)
            prep = _Prepared(calls, chunks, sxs, metas_all, request, core.t_prepare + time.perf_counter() - t0,
                             trip=core.trip, reservations=tuple(reservations), fillers=fillers, items_map=items_map,
                             groups=groups)
        except BaseException:
            core.wipe()
            raise
        self.stats.add(pad_rows=sum(p for p, c in zip(pads, calls, strict=True) if not _is_dummy(c)),
                       dummy_calls=sum(1 for c in calls if _is_dummy(c)))
        return prep

    @property
    def fixed_pad(self) -> bool:
        """Padded and dummy calls take the fixed-structure path (L3): needs the native trip kernels."""
        ok = self._fixed_pad_ok
        if ok is None:
            ok = self._fixed_pad_ok = bool(
                self.ring.k % 8 == 0 and self.ring.k <= 56 and native_available("trip_prepare_into")
                and native_available("trip_finish_into"))
        return ok

    @property
    def dummy_takes_row(self) -> bool:
        """A dummy call takes one mask row (zero activations) on the fixed-structure path, so its store take
        matches a real call's (L3 timing); the public charge schedule always holds that slot."""
        return self.fixed_pad

    @property
    def supports_fixed_pad(self) -> bool:
        """Engine hook (``TransformerModel.supports_pad_rows``): padding rows cost the same gateway work as real
        rows here, so strict padded speculation may use this backend."""
        return self.fixed_pad

    def _prepare_fixed(self, calls: Sequence[LinearCall]) -> _Prepared:
        """Calls with ``meta[PAD_ROWS]`` / ``meta[DUMMY]`` on the fixed-structure path, the others as usual.

        Per fixed call the work is the same whatever its split into real and padding rows: one prepare kernel
        over all its rows (padding rows: zero activations, fresh ChaCha20 ``r`` read like a store mask), one
        Freivalds pass over all its rows (the reservation's current check vectors; drawn at the normal rate),
        one finish of the real rows and one of the padding rows (together: all rows). The provider gets one item
        of all the rows. The generator keeps a pass's real rows inside one mask extent; if they still span two,
        each extent's rows are checked against their own vectors (one more check pass, never a weaker check).
        """
        fixed_idx = [i for i, c in enumerate(calls) if c.meta and (PAD_ROWS in c.meta or DUMMY in c.meta)]
        fset = set(fixed_idx)
        normal_idx = [i for i in range(len(calls)) if i not in fset]
        metas_all: list[_MaskedMeta] = []
        for c in calls:
            meta: _MaskedMeta = self._meta.get(c.id)
            if c.x.ndim != 2 or c.x.shape[1] != meta.in_features:
                raise ShapeError(f"{c.id}: x must be (T, {meta.in_features}), got {tuple(c.x.shape)}")
            metas_all.append(meta)
        # validate the fixed calls' activations and reservations before any mask row is taken
        xs: dict[int, np.ndarray] = {}
        for i in fixed_idx:
            c = calls[i]
            dummy = _is_dummy(c)
            if dummy:
                # a dummy call takes ONE mask row (zero activations), like the pending token's row of a real
                # call: the per-call store take then happens on every call, real or dummy (L3 timing). The
                # public charge schedule always leaves this row (each pass charges >= 1 slot and dummy passes
                # come after the real ones), so it costs the factory nothing: the slot is charged either way.
                need = 1 if int(c.x.shape[0]) + _pad_rows(c) >= 1 and (
                    c.session is not None or self.mask_source is not None) else 0
            else:
                need = int(c.x.shape[0])
            if need:
                # the same conversion and finiteness check as a real row (a dummy's row is zeros)
                xsrc = self._zero_row(metas_all[i].in_features) if dummy else c.x
                xn = xsrc.detach().to("cpu", torch.float32).numpy()
                if not np.isfinite(xn).all():
                    raise RingOverflowError("non-finite activation values cannot be quantized")
                xs[i] = xn
            if need:
                res = self._reservation(c)
                ensure = getattr(res, "ensure", None)
                if ensure is not None:
                    ensure(c.id, need)
                if int(res.remaining(c.id)) < need:
                    raise MaskExhaustedError("session mask reservation exhausted",
                                             details={"linear": c.id.key, "needed": need})
        core = self._prepare_core([calls[i] for i in normal_idx]) if normal_idx else \
            _Prepared([], [], [], [], None, 0.0)
        fixed: list[_FixedItem] = []
        try:
            t0 = time.perf_counter()
            for i in fixed_idx:
                fixed.append(self._fixed_item(i, calls[i], metas_all[i], xs.get(i)))
            pos = {j: i for j, i in enumerate(normal_idx)}
            by_call: dict[int, list[int]] = {}
            for ci, ch in enumerate(core.chunks):
                ch.call_index = pos[ch.call_index]
                by_call.setdefault(ch.call_index, []).append(ci)
            sxs: list[np.ndarray] = [np.zeros(0, dtype=np.float64) for _ in calls]
            for j, i in enumerate(normal_idx):
                sxs[i] = core.sx[j]
            fpos = {fx.call_index: k for k, fx in enumerate(fixed)}
            items_map: list[tuple[str, int]] = []
            for i in range(len(calls)):
                if i in fpos:
                    items_map.append(("x", fpos[i]))
                else:
                    items_map.extend(("c", ci) for ci in by_call.get(i, []))
            from pgw.wire.messages import ComputeItem, ComputeRequest

            items = []
            for kind, j in items_map:
                if kind == "x":
                    fx = fixed[j]
                    items.append(ComputeItem(fx.lid.key, fx.rows, fx.payload))
                else:
                    ch = core.chunks[j]
                    body = ch.payload if ch.payload is not None else pack_ring(ch.x_hat, self.ring)  # type: ignore[arg-type]
                    items.append(ComputeItem(ch.lid.key, ch.rows, body))
            request = ComputeRequest(
                request_id=core.request.request_id if core.request is not None else uuid.uuid4().hex,
                fingerprint=self.fingerprint, ring=self.ring.name, items=tuple(items), deadline_ms=self.deadline_ms,
            ) if items else None
            reservations = list(core.reservations)
            for i in fixed_idx:
                res = calls[i].session if calls[i].session is not None else self.mask_source
                if res is not None:
                    reservations.append(res)
            prep = _Prepared(calls, core.chunks, sxs, metas_all, request, core.t_prepare + time.perf_counter() - t0,
                             trip=core.trip, reservations=tuple(reservations), items_map=items_map, fixed=fixed)
        except BaseException:
            core.wipe()
            for fx in fixed:
                fx.wipe()
            raise
        self.stats.add(pad_rows=sum(fx.rows - (0 if fx.dummy else fx.real) for fx in fixed),
                       dummy_calls=sum(1 for i in fixed_idx if _is_dummy(calls[i])))
        return prep

    def _zero_row(self, n: int) -> torch.Tensor:
        """A cached ``(1, n)`` float32 zero row: a dummy call's activations (read only)."""
        cache = self.__dict__.setdefault("_zero_rows", {})
        z = cache.get(n)
        if z is None:
            z = cache[n] = torch.zeros((1, n), dtype=torch.float32)
        return z

    def _fixed_item(self, i: int, c: LinearCall, meta: _MaskedMeta, xn: np.ndarray | None) -> _FixedItem:
        n_in, m = meta.in_features, meta.out_features
        own = int(c.x.shape[0])
        dummy = _is_dummy(c)
        lid = c.id
        res = c.session if c.session is not None else self.mask_source
        rows = own + _pad_rows(c)
        # a dummy call takes one mask row (zero activations; see _prepare_fixed), so it runs a real call's take
        real = (1 if rows >= 1 and res is not None else 0) if dummy else own
        check = self._should_check()
        want = check or self.min_soundness_bits > 0
        r_all = np.empty((rows, n_in), dtype=np.int64)
        wr_all = np.zeros((rows, m), dtype=np.int64)
        parts: list[tuple[int, Any, Any]] = []
        try:
            got = 0
            while got < real:
                n, r, wr, chk = self._take(res, lid, real - got, want)
                try:
                    r_all[got:got + n] = r.numpy()
                    wr_all[got:got + n] = wr.numpy()
                finally:
                    _wipe(r)
                    _wipe(wr)
                u = uw = None
                if chk is not None:
                    token, u, uw = chk
                    self._check_soundness(lid, n_in, u, token)
                parts.append((n, u if check else None, uw if check else None))
                got += n
            if not parts:  # no row taken (no reservation): the current vectors (the same draw and lookup)
                u = uw = None
                if want and res is not None:
                    try:
                        token, u, uw = self._current_checks(res, lid)
                        self._ring_only(token, lid)
                        self._check_soundness(lid, n_in, u, token)
                    except (MaskStoreError, MaskReuseError):
                        raise
                    except Exception:  # no vectors (closed or checks = 0): nothing to check against
                        u = uw = None
                parts.append((0, u if check else None, uw if check else None))
                # ballast for the take a call with real rows makes (one row of r and W r): the gateway's work
                # per call should not tell a dummy pass from a real one
                ballast = self._random_r(1, n_in)
                np.copyto(ballast[:, :1], wr_all[:1, :1])
                ballast.fill(0)
            # padding rows do a real row's per-row work too: expand r (ChaCha20, as a take does), copy a W r
            # row (zeros) and wipe both, so the per-call cost does not grow with the number of real rows
            pr = self._random_r(rows - real, n_in)
            pw = np.zeros((rows - real, m), dtype=np.int64)
            r_all[real:] = pr
            wr_all[real:] = pw
            pr.fill(0)
            pw.fill(0)
            x_all = np.zeros((rows, n_in), dtype=np.float32)
            if real and xn is not None:
                x_all[:real] = xn
            nb = self.ring.k // 8
            buf = np.empty(rows * n_in * nb, dtype=np.uint8)
            sx = np.empty(rows, dtype=np.float64)
            ring = self.ring
            try:
                native.fast.trip_prepare_into(x_all, r_all, ring.k, ring.a, int(ring.cap or 0),
                                              float(meta.wnorm or 0.0), buf, sx)
            except ValueError as exc:
                if str(exc).startswith("ring_overflow"):
                    raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
                raise ShapeError(str(exc)) from exc
            payload = buf.tobytes()
            guard = getattr(self.noise_monitor, "guard_packed", None) if self.noise_monitor is not None else None
            if guard is not None:
                guard(payload, rows, n_in)
        except BaseException:
            wr_all.fill(0)
            raise
        finally:
            r_all.fill(0)
        return _FixedItem(i, lid, real, rows, own, payload, sx, wr_all, parts, dummy)

    def _finish_fixed(self, fx: _FixedItem, meta: _MaskedMeta, y_hat: Any) -> torch.Tensor:
        """Verify every row (Freivalds against the item's vectors), then unmask + range-check the real rows and
        run the same finish on the padding rows (verdict dropped): the same kernel work whatever the split."""
        m = meta.out_features
        nb = self.ring.k // 8
        yb = np.frombuffer(y_hat, dtype=np.uint8) if not isinstance(y_hat, np.ndarray) else y_hat
        if len(yb) != fx.rows * m * nb:
            raise ProtocolError("compute response item is malformed: wrong y_hat size")
        scale, bias, bound = self._consts(fx.lid, meta)
        fin = native.fast.trip_finish_into
        full = 1 << (self.ring.k - 1)  # never out of range: this pass only verifies
        scratch = np.empty((fx.rows, m), dtype=np.float32)
        xrow = meta.in_features * nb
        row0 = 0
        try:
            for n, u, uw in fx.parts:
                # the last part also covers the padding rows (they use the current, i.e. last, vectors)
                cnt = n if (row0 + n) < fx.real else fx.rows - row0
                a, b = row0, row0 + cnt
                ys = yb[a * m * nb: b * m * nb]
                xs = fx.payload if (a, b) == (0, fx.rows) else fx.payload[a * xrow: b * xrow]
                try:
                    if u is not None and uw is not None:
                        status, bad = fin(ys, fx.wr[a:b], fx.sx[a:b], scale, bias, self.ring.k, full, scratch[a:b],
                                          x_hat=xs, u=u, uw=uw, wipe_wr=False)
                    else:
                        status, bad = fin(ys, fx.wr[a:b], fx.sx[a:b], scale, bias, self.ring.k, full, scratch[a:b],
                                          wipe_wr=False)
                except ValueError as exc:
                    raise ProtocolError(f"compute response item is malformed: {exc}") from exc
                if u is not None and (status == 3 or _vacuous(u, status)):
                    raise MaskReuseError("mask reservation closed during a compute call")
                if status == 1:
                    raise IntegrityError("compute service result failed the Freivalds check",
                                         details={"linear": fx.lid.key, "bad_rows": int(bad), "rows": cnt})
                row0 = b
            scratch.fill(0)
            r = fx.real
            out = np.empty((r, m), dtype=np.float32)
            pad_out = np.empty((fx.rows - r, m), dtype=np.float32)
            st_r, _ = fin(yb[: r * m * nb], fx.wr[:r], fx.sx[:r], scale, bias, self.ring.k, bound, out)
            fin(yb[r * m * nb:], fx.wr[r:], fx.sx[r:], scale, bias, self.ring.k, bound, pad_out)
            pad_out.fill(0)
            if st_r == 2:
                raise IntegrityError(
                    "decoded linear output outside the representable range (corrupt or tampered result)",
                    details={"bound": bound, "ring": self.ring.name})
            if st_r not in (0, 2):
                self._raise_status(int(st_r), 0, fx.lid, r, bound)
        finally:
            fx.wipe()
        if fx.real == 0 or fx.dummy:
            out.fill(0)
            return torch.from_numpy(np.zeros((fx.out_rows, m), dtype=np.float32))
        return torch.from_numpy(out)

    def _filler_checks(self, call: LinearCall) -> tuple[Any, Any] | None:
        """Current check vectors of the call's reservation (padding of a call without real rows)."""
        res = call.session if call.session is not None else self.mask_source
        if res is None or not self._should_check():
            return None
        try:
            token, u, uw = self._current_checks(res, call.id)
            self._ring_only(token, call.id)
        except Exception:  # no vectors (closed or checks=0): the padding is not verified
            return None
        return u, uw

    def _dummy_finish(self, lid: LinearId, meta: _MaskedMeta, y_hat: Any, rows: int, payload: bytes,
                      checks: tuple[Any, Any] | None) -> None:
        """Finish work on padding rows whose outputs are dropped: the real finish kernel with ``W r = 0``
        (its range verdict is expected to fail and is ignored), and the Freivalds check of the padding
        rows against the item's check vectors when there are any (the provider must compute them right
        too). Raises like a real finish on a failed check."""
        m, n = meta.out_features, meta.in_features
        u = uw = None
        if checks is not None:
            u, uw = checks
            if u is not None and not isinstance(u, np.ndarray):  # torch from checks()
                u, uw = checks_to_u64(u, uw, self.ring)
        nb = self.ring.k // 8
        if self.ring.k % 8 == 0 and self.ring.k <= 56 and native_available("trip_finish_into"):
            scale, bias, bound = self._consts(lid, meta)
            yb = y_hat if isinstance(y_hat, bytes) else np.frombuffer(y_hat, dtype=np.uint8)
            if len(yb) != rows * m * nb:
                raise ProtocolError("compute response item is malformed: padding rows have the wrong size")
            out = np.empty((rows, m), dtype=np.float32)
            wr = np.zeros((rows, m), dtype=np.int64)
            sx = np.zeros(rows, dtype=np.float64)
            try:
                if u is not None and uw is not None:
                    status, bad = native.fast.trip_finish_into(
                        yb, wr, sx, scale, bias, self.ring.k, bound, out,
                        x_hat=bytes(payload), u=u, uw=uw)
                else:
                    status, bad = native.fast.trip_finish_into(yb, wr, sx, scale, bias, self.ring.k, bound, out)
            except ValueError as exc:
                raise ProtocolError(f"compute response item is malformed: {exc}") from exc
            out.fill(0)
            if u is not None and (status == 3 or _vacuous(u, status)):
                raise MaskReuseError("mask reservation closed during a compute call")
            if status == 1:
                raise IntegrityError("compute service result failed the Freivalds check",
                                     details={"linear": lid.key, "bad_rows": int(bad), "rows": rows, "padding": True})
            return
        y64 = unpack_ring_u64(bytes(y_hat), (rows, m), self.ring)
        if u is not None and uw is not None:
            if not u.any():
                raise MaskReuseError("mask reservation closed during a compute call")
            x64 = unpack_ring_u64(payload, (rows, n), self.ring)
            ok = freivalds_check_u64(x64, y64, u.astype(np.uint64, copy=False), uw.astype(np.uint64, copy=False),
                                     self.ring)
            if not bool(ok.all()):
                raise IntegrityError("compute service result failed the Freivalds check",
                                     details={"linear": lid.key, "bad_rows": int((~ok).sum()), "rows": rows,
                                              "padding": True})

    def _split_padded_results(self, prep: _Prepared, results: Sequence[Any],
                              outs: list[list[torch.Tensor]]) -> tuple[list[Any], int]:
        """Results aligned with ``prep.chunks`` (real rows only), after the dummy finish of every
        padding part and filler item. Returns them and the padding bytes received."""
        from pgw.wire.messages import ComputeResult

        nb = self.ring.k // 8
        real: list[Any] = []
        pad_bytes = 0
        assert prep.items_map is not None
        for (kind, j), res in zip(prep.items_map, results, strict=True):
            if kind == "x":
                fx = prep.fixed[j]
                if res.linear != fx.lid.key or int(res.rows) != fx.rows:
                    raise ProtocolError("compute response item does not match the request",
                                        details={"expected": fx.lid.key, "got": str(res.linear)})
                outs[fx.call_index].append(self._finish_fixed(fx, prep.metas[fx.call_index], res.y_hat))
                pad_bytes += len(res.y_hat)
                continue
            if kind == "g":
                part = [prep.chunks[x] for x in prep.groups[j]]
                total = sum(ch.rows + ch.pad for ch in part)
                if res.linear != part[0].lid.key or int(res.rows) != total:
                    raise ProtocolError("compute response item does not match the request",
                                        details={"expected": part[0].lid.key, "got": str(res.linear)})
                meta_g: _MaskedMeta = prep.metas[part[0].call_index]
                row_b = meta_g.out_features * nb
                yg = memoryview(res.y_hat)
                if len(yg) != total * row_b:
                    raise ProtocolError("compute response item is malformed: wrong y_hat size")
                off = 0
                for ch in part:
                    end = off + ch.rows * row_b
                    real.append(ComputeResult(res.linear, ch.rows, bytes(yg[off:end])))
                    off = end
                    if ch.pad:
                        pend = off + ch.pad * row_b
                        chk_g = (ch.u, ch.uw) if ch.u is not None else None
                        self._dummy_finish(ch.lid, meta_g, bytes(yg[off:pend]), ch.pad, ch.pad_payload or b"", chk_g)
                        pad_bytes += pend - off
                        off = pend
                continue
            if kind == "c":
                ch = prep.chunks[j]
                if res.linear != ch.lid.key or int(res.rows) != ch.rows + ch.pad:
                    raise ProtocolError("compute response item does not match the request",
                                        details={"expected": ch.lid.key, "got": str(res.linear)})
                if not ch.pad:
                    real.append(res)
                    continue
                meta: _MaskedMeta = prep.metas[ch.call_index]
                cut = ch.rows * meta.out_features * nb
                y = memoryview(res.y_hat)
                if len(y) != (ch.rows + ch.pad) * meta.out_features * nb:
                    raise ProtocolError("compute response item is malformed: wrong y_hat size")
                chk = (ch.u, ch.uw) if ch.u is not None else None
                self._dummy_finish(ch.lid, meta, bytes(y[cut:]), ch.pad, ch.pad_payload or b"", chk)
                pad_bytes += len(y) - cut
                real.append(ComputeResult(res.linear, ch.rows, bytes(y[:cut])))
            else:
                f = prep.fillers[j]
                meta = prep.metas[f.call_index]
                if res.linear != f.lid.key or int(res.rows) != f.rows:
                    raise ProtocolError("compute response item does not match the request",
                                        details={"expected": f.lid.key, "got": str(res.linear)})
                self._dummy_finish(f.lid, meta, res.y_hat, f.rows, f.payload, f.checks)
                pad_bytes += len(res.y_hat)
                rows_out = int(prep.calls[f.call_index].x.shape[0])  # the call's own rows; padding dropped
                outs[f.call_index].append(torch.zeros((rows_out, meta.out_features), dtype=torch.float32))
        return real, pad_bytes

    # ---- single-use mask prefetch -----------------------------------------------------------------
    def _maybe_prefetch(self, calls: Sequence[LinearCall]) -> None:
        """Stage the next rows of every (reservation, linear) of this step in the background.

        Active only for multi-session steps (two or more reservations) or calls of
        ``3..prefetch_max_rows`` rows (speculative verification); single-stream decode keeps its
        critical path free of a second thread. The staged rows are the reservation's next rows in
        order (a cache of what the next take would derive), held by that reservation only: they are
        never handed to another session, and ``close()`` zeroes the unused ones (they stay burned).
        Staging stops at the current extent's end, so it never moves a call across a boundary.
        """
        need: dict[tuple[int, str], list[Any]] = {}
        for c in calls:
            res = c.session if c.session is not None else self.mask_source
            if res is None or not hasattr(res, "prefetch"):
                continue
            key = (id(res), c.id.key)
            hit = need.get(key)
            if hit is None:
                need[key] = [res, c.id, int(c.x.shape[0])]
            else:
                hit[2] += int(c.x.shape[0])
        if not need:
            return
        multi = len({k[0] for k in need}) > 1
        cap = int(self.prefetch_max_rows)
        items = [(res, lid, rows * max(1, int(self.prefetch_depth))) for res, lid, rows in need.values()
                 if 0 < rows <= cap and (multi or rows >= 3)]
        if not items and self.prefetch_decode:
            # Single-session decode (L2): stage the next ``decode_prefetch_rows`` rows of this (reservation,
            # linear) once every ``decode_prefetch_rows`` calls, so the derivation runs on the prefetch
            # thread while trips are in flight (the gateway thread waits in recv, GIL free) and the next
            # tokens take staged rows. Same rows, same order (a cache of the next takes); nothing sent changes.
            depth = int(self.decode_prefetch_rows)
            for key, (res, lid, rows) in need.items():
                if rows > 2:
                    continue
                left = self._decode_pf.get(key, 0) - rows
                if left > 0:
                    self._decode_pf[key] = left
                    continue
                self._decode_pf[key] = max(1, depth)
                items.append((res, lid, depth))
            if len(self._decode_pf) > 4096:  # bounded (keys of ended sessions)
                self._decode_pf.clear()
        if not items:
            return
        with self._prefetch_lock:
            if self._prefetch_pending >= 2:  # the prefetcher is behind: never queue unbounded work
                return
            if self._prefetch_pool is None:
                from concurrent.futures import ThreadPoolExecutor

                self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-prefetch")
            self._prefetch_pending += 1
            pool = self._prefetch_pool
        try:
            pool.submit(self._run_prefetch, items)
        except RuntimeError:  # shut down
            with self._prefetch_lock:
                self._prefetch_pending -= 1

    def prefetch_prompt(self, items: Sequence[tuple[Any, LinearId, int]]) -> bool:
        """Prompt-pass mask prefetch (:attr:`prompt_prefetch`; the engine calls it, off by default): stage
        ``rows`` next rows of ``(reservation, linear)`` on the prefetch thread, e.g. the next remote linear of
        a prompt pass while this one's trip and the local work in between run. The staged rows are exactly
        the rows the next takes would derive (same rows, same order, still single-use and owned by that
        reservation); the row count is the prompt's, which the provider sees anyway. Returns whether a
        prefetch was queued (never more than two at a time)."""
        if not self.prompt_prefetch:
            return False
        todo = [(res, lid, int(rows)) for res, lid, rows in items
                if res is not None and hasattr(res, "prefetch") and int(rows) > 0]
        if not todo:
            return False
        with self._prefetch_lock:
            if self._prefetch_pending >= 2:
                return False
            if self._prefetch_pool is None:
                from concurrent.futures import ThreadPoolExecutor

                self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-prefetch")
            self._prefetch_pending += 1
            pool = self._prefetch_pool
        try:
            pool.submit(self._run_prefetch, todo)
        except RuntimeError:  # shut down
            with self._prefetch_lock:
                self._prefetch_pending -= 1
            return False
        return True

    def _run_prefetch(self, items: Sequence[tuple[Any, LinearId, int]]) -> None:
        staged = 0
        try:
            for res, lid, rows in items:
                try:
                    staged += int(res.prefetch(lid, rows))
                except Exception as exc:  # the next take derives the rows (and reports any error)
                    log.debug("mask prefetch for %s skipped: %s", lid, type(exc).__name__)
        finally:
            with self._prefetch_lock:
                self.prefetched_rows += staged
                self._prefetch_pending -= 1

    def wait_prefetch(self, timeout: float = 5.0) -> bool:
        """Wait until queued prefetches finished (tests and benchmarks); ``False`` on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._prefetch_lock:
                if self._prefetch_pending == 0:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)

    def close(self) -> None:
        """Stop the prefetch thread (staged rows belong to their reservations, which zero them)."""
        with self._prefetch_lock:
            pool, self._prefetch_pool = self._prefetch_pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        with self._ahead_lock:
            self._closing = True
            apool, self._ahead_pool = self._ahead_pool, None
        if apool is not None:  # leases prepared ahead belong to their reservations, which zero them at close
            apool.shutdown(wait=True, cancel_futures=True)
        npool, self._noise_pool = self._noise_pool, None
        if npool is not None:  # the pending inspection wipes its copies when it ends
            npool.shutdown(wait=True)
        self._noise_pending = None
        with self._pending_lock:  # deferred checks read buffers they keep alive: let them finish
            pending, self._pending = self._pending, []
            self._failed = []
        # their verdicts are still reported (log, ``integrity_failures``, ``on_integrity_failure``),
        # never raised from close()
        try:
            verdicts = self._wait_checks(pending)
        except Exception:  # pragma: no cover - the native wait does not raise
            log.exception("waiting for deferred checks at close failed")
            return
        for p, err in verdicts:
            if err is not None:
                self._report_deferred(err, p)

    # ---- E2: exact local linears ------------------------------------------------------------------
    def configure_local_linears(self, mode: str = "off", *, rtt_s: float | None = None,
                                reps: int = 5, device: str = "cpu") -> frozenset[str]:
        """Decide once, at start-up, which linears the gateway computes itself (E2; public configuration).

        ``off``: none. ``attention``: qkv and o of every layer, computed exactly from the pack's int8
        weights with the same quantization and integer product as the masked path (so the same bits);
        only gate_up and down are sent, halving the round trips. ``auto``: ``attention`` when the
        calibrated local time of one layer's qkv + o (one row, median of ``reps``) is below two round
        trips of ``rtt_s`` seconds; ``rtt_s=None`` takes the mean decode round trip measured so far
        (at least 16 trips), else stays ``off``. The decision never changes per token. Call before
        serving (not concurrently with calls). The local int8 weights are loaded here (start-up), so
        a request never reads a weight.

        ``device`` (``cpu``, ``cuda`` or ``auto``) is where the exact integer product of the local
        linears runs; quantization, the range check and dequantization stay on the CPU. A gateway GPU
        is used only after :meth:`_local_device_check` found its integer outputs identical to the CPU's
        on layer 0 (and, for ``auto``, faster); otherwise the CPU is kept. Decided once, at start-up.

        Returns:
            The linear kinds computed locally.
        """
        if mode not in LOCAL_LINEAR_MODES:
            raise ConfigError(f"local_linears must be one of {LOCAL_LINEAR_MODES}, got {mode!r}")
        policy: dict[str, Any] = {"mode": mode}
        kinds: frozenset[str] = frozenset()
        if mode != "off":
            if self._local_w is None:
                self._local_w = _WeightCache(self.pack, torch.device("cpu"), _to_int_weight)
            if mode in _LOCAL_FIXED_KINDS:
                kinds = _LOCAL_FIXED_KINDS[mode]
            elif mode == "auto_kinds":
                kinds = self._calibrate_kinds(rtt_s, max(1, int(reps)), policy)
            else:
                kinds = self._calibrate_local(rtt_s, max(1, int(reps)), policy)
        policy["device"] = "cpu"
        self._local_gpu = None
        self._local_one_row = {}
        if kinds and device != "cpu":
            gp = self._local_gpu_choice(device, kinds, max(1, int(reps)), policy)
            if gp is not None:
                self._local_gpu = gp
                policy["device"] = f"{gp.device} (graphs)"
                self._local_one_row = self._local_one_row_build(gp, kinds)
                policy["one_row_fast_path"] = len(self._local_one_row)
            else:
                dev = self._local_device_choice(device, max(1, int(reps)), policy)
                if dev is not None:
                    self._local_w = _WeightCache(self.pack, dev, _to_int_weight)
                    policy["device"] = str(dev)
        if kinds:
            for spec in self.pack.linear_specs():  # type: ignore[attr-defined]
                if spec.id.kind.value in kinds:
                    self._local_w.get(spec.id)  # type: ignore[union-attr]
        else:
            self._local_w = None
        self._local_kinds = kinds
        policy["kinds"] = tuple(sorted(kinds))
        policy["max_rows"] = self.local_max_rows
        self.local_policy = policy
        log.info("local linears: %s -> %s%s", mode, ", ".join(sorted(kinds)) or "none (every linear remote)",
                 f" on {policy['device']}" if kinds else "")
        return kinds

    def _local_gpu_choice(self, want: str, kinds: frozenset[str], reps: int, policy: dict[str, Any]) -> Any:
        """:class:`pgw.engine.local_gpu.LocalGpuProducts` for the local linears on the gateway GPU ``want``
        (``cuda``, ``cuda:N`` or ``auto``), or ``None`` (then :meth:`_local_device_choice` decides). Used only
        if every local linear's GPU outputs equal the CPU reference's on public test rows
        (:meth:`_local_gpu_agrees`) and, for ``auto``, layer 0's local linears are faster than on the CPU.
        ``PGW_LOCAL_GPU_GRAPHS=0`` disables it."""
        if not _env_flag("PGW_LOCAL_GPU_GRAPHS", True):
            policy["gpu_graphs_reason"] = "PGW_LOCAL_GPU_GRAPHS=0"
            return None
        if want not in ("auto", "cuda") and not want.startswith("cuda:"):
            raise ConfigError(f"local linear device must be cpu, cuda or auto, got {want!r}")
        if not torch.cuda.is_available():
            policy["gpu_graphs_reason"] = "no CUDA device"
            return None
        from pgw.engine.local_gpu import LocalGpuProducts, local_gpu_unavailable  # pragma: no cover - GPU only

        dev = torch.device("cuda" if want == "auto" else want)  # pragma: no cover - GPU only
        return self._local_gpu_build(dev, kinds, reps, policy, LocalGpuProducts,  # pragma: no cover - GPU only
                                     local_gpu_unavailable, need_faster=want == "auto")

    def _local_gpu_build(self, dev: torch.device, kinds: frozenset[str], reps: int, policy: dict[str, Any],
                         factory: Any, unavailable: Any, *, need_faster: bool) -> Any:
        """Build and self-check the GPU products of :meth:`_local_gpu_choice` (``factory`` and ``unavailable``
        are injectable for tests). Any failure keeps the CPU."""
        ring = self.ring
        why = unavailable(dev, ring.k)
        if why is not None:
            policy["gpu_graphs_reason"] = why
            log.warning("local linears: GPU graphs off (%s)", why)
            return None
        assert self._local_w is not None
        specs = [s for s in self.pack.linear_specs() if s.id.kind.value in kinds]  # type: ignore[attr-defined]
        t0 = time.perf_counter()
        try:
            rows_env = os.environ.get("PGW_LOCAL_GPU_GRAPH_ROWS", "1")
            graph_rows = tuple(int(v) for v in rows_env.split(",") if v.strip())
            gp = factory(dev, ring.k, ring.a, int(ring.cap or 0), graph_rows=graph_rows,
                         max_n=max(s.in_features for s in specs), max_m=max(s.out_features for s in specs))
            for spec in specs:
                gp.add(spec.id.key, self._local_w.get(spec.id).w8t.t())
            for spec in specs:
                meta: _MaskedMeta = self._meta.get(spec.id)
                if not self._local_gpu_agrees(gp, meta, self._local_w.get(spec.id), spec.id):
                    policy["gpu_graphs_reason"] = (f"outputs differ from the CPU reference ({spec.id}, "
                                                   f"{self._local_gpu_mismatch})")
                    log.error("local linears: GPU %s; using the CPU", policy["gpu_graphs_reason"])
                    return None
            t_cpu = t_dev = 0.0
            gen = torch.Generator().manual_seed(1)
            for spec in specs:
                if spec.id.layer != 0:
                    continue
                meta = self._meta.get(spec.id)
                call = LinearCall(spec.id, torch.randn(1, spec.in_features, generator=gen))
                w = self._local_w.get(spec.id)
                for slot in (0, 1):
                    ts = []
                    for _ in range(reps + 1):
                        t1 = time.perf_counter()
                        if slot == 0:
                            self._forward_local_cpu(call.x, meta, w, call)
                        else:
                            self._local_gpu_forward(gp, call.x, meta, spec.id)
                        ts.append(time.perf_counter() - t1)
                    med = sorted(ts[1:])[len(ts[1:]) // 2]
                    if slot == 0:
                        t_cpu += med
                    else:
                        t_dev += med
        except Exception as exc:  # a capture or kernel failure: the CPU is kept
            policy["gpu_graphs_reason"] = f"{dev} failed: {type(exc).__name__}: {str(exc)[:200]}"
            log.error("local linears: GPU graphs %s; using the CPU", policy["gpu_graphs_reason"])
            return None
        policy["gpu_graphs_check"] = {"linears": len(specs), "cpu_layer0_s": t_cpu, "gpu_layer0_s": t_dev,
                                      "graph_rows": list(gp.graph_rows), "fused_rows": gp.fused_rows,
                                      "native": bool(getattr(gp, "native", False)),
                                      "native_reason": getattr(gp, "native_reason", None),
                                      "setup_s": round(time.perf_counter() - t0, 3), "identical": True}
        if need_faster and not t_dev < t_cpu:
            policy["gpu_graphs_reason"] = f"{dev} not faster ({t_dev:.3g} s vs {t_cpu:.3g} s for layer 0)"
            return None
        policy["gpu_graphs_reason"] = "identical outputs"
        log.info("local linears: exact products on %s through CUDA graphs (%d linears; layer 0: %.3g ms GPU, "
                 "%.3g ms CPU)", dev, len(specs), 1e3 * t_dev, 1e3 * t_cpu)
        return gp

    def _local_one_row_build(self, gp: Any, kinds: frozenset[str]) -> dict[str, Any]:
        """Lean 1-row callables per local linear (``PGW_LOCAL_GPU_ONE_ROW=0`` disables them), each compared with
        the CPU reference bit for bit (and for the same refusals) on public test rows before use; a linear whose
        callable differs keeps the regular path."""
        if not _env_flag("PGW_LOCAL_GPU_ONE_ROW", True) or not hasattr(gp, "one_row"):
            return {}
        out: dict[str, Any] = {}
        for spec in self.pack.linear_specs():  # type: ignore[attr-defined]
            if spec.id.kind.value not in kinds or not gp.has(spec.id.key):
                continue
            lid = spec.id
            meta: _MaskedMeta = self._meta.get(lid)
            scale, bias, bound = self._consts(lid, meta)

            def on_error(err: Any, lid: LinearId = lid, bound: int = bound) -> None:
                if isinstance(err, ValueError):
                    if str(err).startswith("ring_overflow"):
                        raise RingOverflowError(str(err).split(":", 1)[-1].strip()) from err
                    raise ShapeError(str(err)) from err
                self._raise_status(int(err), 0, lid, 1, bound)

            fn = gp.one_row(lid.key, float(meta.wnorm or 0.0) if self.ring.cap else 0.0, scale, bias, bound, on_error)
            if fn is None:
                continue
            w = self._local_w.get(lid)  # type: ignore[union-attr]
            rng = np.random.default_rng(0x0E1 ^ meta.in_features * 7919 + meta.out_features)
            x = (rng.standard_normal((4, meta.in_features)) * 3).astype(np.float32)
            x[1] *= np.float32(1e-3)
            x[1, int(rng.integers(0, meta.in_features))] = 50.0
            x[2] = 0.0
            x[3] *= np.float32(1e-30)
            nonfinite = x[:1].copy()
            nonfinite[0, 0] = np.nan
            ok = True
            for xs in (x[0:1], x[1:2], x[2:3], x[3:4], nonfinite):
                xt = torch.from_numpy(np.ascontiguousarray(xs))
                ref: Any
                got: Any
                try:
                    ref = self._local_reference(xt, meta, w, lid)
                except PGWError as exc:
                    ref = type(exc)
                try:
                    got = fn(xt)
                except PGWError as exc:
                    got = type(exc)
                if isinstance(ref, type) or isinstance(got, type):
                    ok = ref is got
                else:
                    ok = ref.dtype == got.dtype and ref.shape == got.shape and torch.equal(ref, got)
                if not ok:
                    log.error("local linears: 1-row GPU fast path differs from the CPU reference for %s; regular "
                              "path kept", lid)
                    break
            if ok:
                out[lid.key] = fn
        return out

    def _local_gpu_agrees(self, gp: Any, meta: _MaskedMeta, w: _IntWeight, lid: LinearId) -> bool:
        """Self-check: the GPU path gives the CPU reference's outputs bit for bit, and the same refusals, on
        public test rows (Gaussian, spike, zero, tiny; 1, 3 and 12 rows: the graph and the eager paths)."""
        rng = np.random.default_rng(0x10CA1 ^ meta.in_features * 7919 + meta.out_features)
        x = (rng.standard_normal((3, meta.in_features)) * 3).astype(np.float32)
        x[1] *= np.float32(1e-3)
        x[1, int(rng.integers(0, meta.in_features))] = 50.0
        x[2] = 0.0
        tiny = (rng.standard_normal((1, meta.in_features)) * 1e-30).astype(np.float32)
        many = (rng.standard_normal((12, meta.in_features)) * 2).astype(np.float32)
        self._local_gpu_mismatch = ""
        cases = [x[:1], x[1:2], x[2:3], tiny, x, many]
        blk = int(getattr(gp, "prefill_block_rows", 0) or 0)
        if blk:  # prompt blocks: full blocks, a padded short block, and a spike in the last block
            long = (rng.standard_normal((2 * blk + 5, meta.in_features)) * 2).astype(np.float32)
            long[-2] = x[1]
            cases.append(long)
        for case, xs in enumerate(cases):
            xt = torch.from_numpy(xs.copy())
            ref: Any
            got: Any
            try:
                ref = self._local_reference(xt, meta, w, lid)
            except PGWError as exc:
                ref = type(exc)
            try:
                got = self._local_gpu_forward(gp, xt, meta, lid)
            except PGWError as exc:
                got = type(exc)
            if isinstance(ref, type) or isinstance(got, type):
                if ref is not got:
                    self._local_gpu_mismatch = f"case {case} ({xs.shape[0]} rows): {ref} vs {got}"
                    return False
            elif ref.dtype != got.dtype or ref.shape != got.shape or not torch.equal(ref, got):
                self._local_gpu_mismatch = f"case {case} ({xs.shape[0]} rows): outputs differ"
                return False
        return True

    def _local_gpu_forward(self, gp: Any, x: torch.Tensor, meta: _MaskedMeta, lid: LinearId) -> torch.Tensor:
        """One local linear through :class:`pgw.engine.local_gpu.LocalGpuProducts` (the CPU path's bits)."""
        xn = x.detach()
        if xn.dtype != torch.float32:
            xn = xn.to(torch.float32)
        xf = xn.numpy()
        if not xf.flags.c_contiguous:
            xf = np.ascontiguousarray(xf)
        if xf.size and not np.isfinite(xf).all():
            raise RingOverflowError("non-finite activation values cannot be quantized")
        rows = int(xf.shape[0])
        scale, bias, bound = self._consts(lid, meta)
        out = np.empty((rows, meta.out_features), dtype=np.float32)
        if rows == 0:
            return torch.from_numpy(out)
        sx = np.empty(rows, dtype=np.float64)
        wn = float(meta.wnorm or 0.0) if self.ring.cap else 0.0
        blk = int(getattr(gp, "prefill_block_rows", 0) or 0)
        try:
            if blk and rows > blk:  # a prompt pass in blocks of graph replays (rows are independent: same bits)
                status = 0
                for a in range(0, rows, blk):
                    b = min(rows, a + blk)
                    status = gp.forward(lid.key, xf[a:b], wn, scale, bias, bound, out[a:b], sx[a:b])
                    if status:
                        break
            else:
                status = gp.forward(lid.key, xf, wn, scale, bias, bound, out, sx)
        except ValueError as exc:
            if str(exc).startswith("ring_overflow"):
                raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
            raise ShapeError(str(exc)) from exc
        finally:
            sx.fill(0)
        if status:
            out.fill(0)
            self._raise_status(int(status), 0, lid, rows, bound)
        return torch.from_numpy(out)

    def _local_device_choice(self, want: str, reps: int, policy: dict[str, Any]) -> torch.device | None:
        """The device of the local integer products (``None``: the CPU); see :meth:`configure_local_linears`."""
        if want not in ("auto", "cuda") and not want.startswith("cuda:"):
            raise ConfigError(f"local linear device must be cpu, cuda or auto, got {want!r}")
        if not torch.cuda.is_available():
            policy["device_reason"] = "no CUDA device"
            if want != "auto":
                log.warning("local linears: device %s requested but CUDA is not available; using the CPU", want)
            return None
        dev = torch.device("cuda" if want == "auto" else want)  # pragma: no cover - GPU only
        return self._local_device_check(dev, reps, policy, need_faster=want == "auto")  # pragma: no cover

    def _local_device_check(self, dev: torch.device, reps: int, policy: dict[str, Any], *,
                            need_faster: bool) -> torch.device | None:
        """``dev`` if its exact integer products equal the CPU's on layer 0's local linears (1 and 7 rows,
        seeded inputs; with ``need_faster`` also a lower median one-row time), else ``None``."""
        assert self._local_w is not None
        dev_w = _WeightCache(self.pack, dev, _to_int_weight)
        specs = [s for s in self.pack.linear_specs() if s.id.layer == 0  # type: ignore[attr-defined]
                 and s.id.kind.value in LOCAL_ATTENTION_KINDS]
        gen = torch.Generator().manual_seed(0)
        t_cpu = t_dev = 0.0
        try:
            for spec in specs:
                meta: _MaskedMeta = self._meta.get(spec.id)
                for rows in (1, 7):
                    x = torch.randn(rows, spec.in_features, generator=gen)
                    xq, _ = quantize_activations(x, self.ring, meta.wnorm if self.ring.cap else None)
                    ref = self._local_int_product(self._local_w.get(spec.id), xq)
                    got = self._local_int_product(dev_w.get(spec.id), xq)
                    if not torch.equal(ref, got):
                        policy["device_reason"] = f"{dev} integer products differ from the CPU's ({spec.id})"
                        log.error("local linears: %s; using the CPU", policy["device_reason"])
                        return None
                xq, _ = quantize_activations(torch.randn(1, spec.in_features, generator=gen), self.ring,
                                             meta.wnorm if self.ring.cap else None)
                for cache, slot in ((self._local_w, 0), (dev_w, 1)):
                    times = []
                    for _ in range(reps + 1):
                        t0 = time.perf_counter()
                        self._local_int_product(cache.get(spec.id), xq)
                        times.append(time.perf_counter() - t0)
                    med = sorted(times[1:])[len(times[1:]) // 2]
                    if slot == 0:
                        t_cpu += med
                    else:
                        t_dev += med
        except Exception as exc:  # a missing kernel or a cuBLAS refusal: the CPU is kept
            policy["device_reason"] = f"{dev} failed: {type(exc).__name__}: {str(exc)[:200]}"
            log.error("local linears: %s; using the CPU", policy["device_reason"])
            return None
        policy["device_check"] = {"cpu_s": t_cpu, "device_s": t_dev, "identical": True}
        if need_faster and not t_dev < t_cpu:
            policy["device_reason"] = f"{dev} not faster ({t_dev:.3g} s vs {t_cpu:.3g} s per layer)"
            return None
        policy["device_reason"] = f"{dev} identical and {t_cpu / max(t_dev, 1e-12):.2f}x the CPU's speed"
        return dev

    def _local_int_product(self, w: _IntWeight, xq: torch.Tensor) -> torch.Tensor:
        """Exact ``xq @ W8^T`` (int64, on the CPU) with the weight's device doing the integer GEMM."""
        dev = w.w8t.device
        if dev.type == "cpu":
            return exact_int_matmul(xq, w.w8t, self.ring.a)
        xd = xq.to(dev)  # pragma: no cover - GPU only
        try:  # pragma: no cover - GPU only
            return exact_int_matmul(xd, w.w8t, self.ring.a).cpu()
        finally:  # pragma: no cover - GPU only
            _wipe(xd)

    def _calibrate_local(self, rtt_s: float | None, reps: int, policy: dict[str, Any]) -> frozenset[str]:
        if rtt_s is None:
            st = self.stats.snapshot()
            if st["decode_calls"] >= 16:
                rtt_s = st["decode_remote_seconds"] / st["decode_calls"]
        policy["rtt_s"] = rtt_s
        specs = [s for s in self.pack.linear_specs() if s.id.layer == 0  # type: ignore[attr-defined]
                 and s.id.kind.value in LOCAL_ATTENTION_KINDS]
        if rtt_s is None or not specs:
            policy["reason"] = "no round-trip measurement" if rtt_s is None else "no attention linears"
            return frozenset()
        gen = torch.Generator().manual_seed(0)
        total = 0.0
        for spec in specs:
            x = torch.randn(1, spec.in_features, generator=gen)
            call = LinearCall(spec.id, x)
            self._forward_local_one(call)  # warm-up (loads the weight)
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                self._forward_local_one(call)
                times.append(time.perf_counter() - t0)
            total += sorted(times)[len(times) // 2]
        policy["local_s"] = total
        on = total < 2.0 * float(rtt_s)
        policy["reason"] = "local qkv+o {:.3g} s {} 2 round trips {:.3g} s".format(
            total, "<" if on else ">=", 2.0 * float(rtt_s))
        return LOCAL_ATTENTION_KINDS if on else frozenset()

    def _calibrate_kinds(self, rtt_s: float | None, reps: int, policy: dict[str, Any]) -> frozenset[str]:
        """``auto_kinds`` (R1): each linear kind is local iff its calibrated local time (layer 0, one
        row, median of ``reps``) is below one round trip of ``rtt_s`` seconds. Every remote linear is a
        round trip of its own (a nonlinear step sits between any two consecutive linears), so the kinds
        are independent. Decided once at start-up from public data (random inputs); never per token."""
        if rtt_s is None:
            st = self.stats.snapshot()
            if st["decode_calls"] >= 16:
                rtt_s = st["decode_remote_seconds"] / st["decode_calls"]
        policy["rtt_s"] = rtt_s
        if rtt_s is None:
            policy["reason"] = "no round-trip measurement"
            return frozenset()
        gen = torch.Generator().manual_seed(0)
        times: dict[str, float] = {}
        for spec in self.pack.linear_specs():  # type: ignore[attr-defined]
            kind = spec.id.kind.value
            if spec.id.layer != 0 or kind not in LOCAL_ALL_KINDS:
                continue
            call = LinearCall(spec.id, torch.randn(1, spec.in_features, generator=gen))
            self._forward_local_one(call)  # warm-up (loads the weight)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                self._forward_local_one(call)
                ts.append(time.perf_counter() - t0)
            times[kind] = sorted(ts)[len(ts) // 2]
        policy["local_s_by_kind"] = dict(sorted(times.items()))
        policy["reason"] = f"per kind: local iff below one round trip of {float(rtt_s):.3g} s"
        return frozenset(k for k, t in times.items() if t < float(rtt_s))

    def local_rows_ok(self, rows: int) -> bool:
        """Engine hook (R1): local linears apply to a pass step of ``rows`` rows in total. Depends only
        on the row count, which the provider sees for any remote pass anyway (SPEC 3.4)."""
        cap = self.local_max_rows
        return cap is None or int(rows) <= cap

    def local_linear(self, lid: Any) -> bool:
        """Engine hook: ``lid`` is computed exactly on the gateway (no mask, no round trip)."""
        kinds = self._local_kinds
        if not kinds:
            return False
        kind = getattr(lid, "kind", None)
        return kind is not None and getattr(kind, "value", kind) in kinds

    #: Decode-sized local calls (at most this many rows) take :meth:`_local_fast` when its one-time
    #: self-check per linear shape passed (0 disables; ``PGW_LOCAL_FAST=0`` too).
    local_fast_max_rows = 8
    #: With the native kernel (``pgw_fast.local_linear_into``): calls of any row count take it, in blocks of at
    #: most this many rows (its limit). Each output row depends only on its own input row (per-row activation
    #: scale, exact integer product), so the blocks give the reference's bits (self-checked with 1, 3 and 12
    #: rows); a prompt (prefill) pass of local linears no longer goes through the two-digit int8 GEMM.
    local_native_block_rows = 64

    def _local_fast_ok(self, meta: _MaskedMeta, w: _IntWeight, rows: int, lid: LinearId) -> bool:
        if rows <= 0 or (rows > self.local_fast_max_rows and not (self._local_native and self.local_fast_max_rows)):
            return False
        on = self._local_fast_on
        if on is None:
            ring = self.ring
            native_ok = self._local_native and ring.k % 8 == 0 and 8 <= ring.k <= 56
            if not native_ok:
                self._local_native = False
            on = self._local_fast_on = (_env_flag("PGW_LOCAL_FAST", True) and native_available("trip_prepare_into")
                                        and ring.a <= 15
                                        and (native_ok or (ring.k == 32 and cpu_int8_fast())))
        if not on or (rows > self.local_fast_max_rows and not self._local_native):
            return False
        key = (meta.in_features, meta.out_features, float(meta.wnorm or 0.0), id(w))
        hit = self._local_fast_checked.get(key)
        if hit is None:
            hit = self._local_fast_checked[key] = self._local_fast_agrees(meta, w, lid)
        return hit

    def _local_fast_agrees(self, meta: _MaskedMeta, w: _IntWeight, lid: LinearId) -> bool:
        """Self-check: :meth:`_local_fast` gives the reference outputs bit for bit on public test rows
        (Gaussian, spike, zero, tiny; 1 and 3 rows), and refuses the same rows."""
        rng = np.random.default_rng(0x5EED ^ meta.in_features * 7919 + meta.out_features)
        x = (rng.standard_normal((3, meta.in_features)) * 3).astype(np.float32)
        x[1] *= np.float32(1e-3)
        x[1, int(rng.integers(0, meta.in_features))] = 50.0
        x[2] = 0.0
        tiny = (rng.standard_normal((1, meta.in_features)) * 1e-30).astype(np.float32)
        cases = [x[:1], x, tiny]
        if self._local_native:  # the native kernel's many-row path (columns split instead of features)
            cases.append(np.concatenate([x, (rng.standard_normal((9, meta.in_features)) * 2).astype(np.float32)]))
        try:
            for xs in cases:
                ref = self._local_reference(torch.from_numpy(xs.copy()), meta, w, lid)
                got = self._local_fast(np.ascontiguousarray(xs), meta, w, lid)
                if ref.dtype != got.dtype or ref.shape != got.shape or not torch.equal(ref, got):
                    raise ValueError("outputs differ")
        except Exception as exc:
            log.error("local fast path self-check failed for n=%d m=%d (%s): reference path kept",
                      meta.in_features, meta.out_features, exc)
            return False
        return True

    def _local_fast(self, xf: np.ndarray, meta: _MaskedMeta, w: _IntWeight, lid: LinearId) -> torch.Tensor:
        """The exact local product for a few rows: the TripEngine's prepare kernel quantizes (with a zero
        mask its payload is the quantized ring element; checked against the reference by
        ``_trip_kernels_agree`` like every TripEngine use), two balanced base-256 digits go through one
        int8 GEMM, and the numpy rescale is :func:`dequantize_outputs`' small-input path. Same integers,
        so the same bits as :meth:`_local_reference` (self-checked per shape)."""
        ring = self.ring
        rows, n = xf.shape
        if self._local_native and w.w8t.device.type == "cpu" and w.w8t.is_contiguous():
            # pgw_fast.local_linear_into: the same quantizer (zero mask), the exact int8 product as one
            # int16 x int8 GEMV with int64 flushes, the same range check and rescale; no temporaries
            scale, bias, bound = self._consts(lid, meta)
            hit = self._local_w8n.get(lid.key)
            if hit is None or hit[0] is not w.w8t:  # same linear key, other weights (reload, other pack)
                hit = self._local_w8n[lid.key] = (w.w8t, w.w8t.numpy())
            w8n = hit[1]
            res = np.empty((rows, meta.out_features), dtype=np.float32)
            sx = np.empty(rows, dtype=np.float64)
            try:
                status = native.fast.local_linear_into(
                    xf, w8n, scale, bias, ring.k, ring.a, int(ring.cap or 0),
                    float(meta.wnorm or 0.0) if ring.cap else 0.0, bound,
                    sx, res, 0 if n * meta.out_features >= self.local_pool_min_macs else 1)
            except ValueError as exc:
                if str(exc).startswith("ring_overflow"):
                    raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
                raise ShapeError(str(exc)) from exc
            if status:
                self._raise_status(int(status), 0, lid, rows, bound)
            return torch.from_numpy(res)
        zeros = self._local_zero_r.get(n)
        if zeros is None or zeros.shape[0] < rows:
            zeros = self._local_zero_r[n] = np.zeros((self.local_fast_max_rows, n), dtype=np.int64)
        payload = np.empty(rows * n * 4, dtype=np.uint8)
        sx = np.empty(rows, dtype=np.float64)
        try:
            native.fast.trip_prepare_into(xf, zeros[:rows], ring.k, ring.a, int(ring.cap or 0),
                                          float(meta.wnorm or 0.0) if ring.cap else 0.0, payload, sx)
        except ValueError as exc:
            if str(exc).startswith("ring_overflow"):
                raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
            raise ShapeError(str(exc)) from exc
        xq = payload.view("<u4").astype(np.int64).reshape(rows, n)
        xq -= (xq >> 31) << 32  # centered: |xq| < 2^(A-1)
        lo = ((xq + 128) & 255) - 128
        hi = (xq - lo) >> 8
        payload.fill(0)
        d8 = torch.from_numpy(np.concatenate([lo, hi]).astype(np.int8))
        xq.fill(0)
        lo.fill(0)
        hi.fill(0)
        y = int8_digit_gemm(d8, w.w8t.t()).numpy()
        d8.numpy().fill(0)
        yq = y[:rows] + (y[rows:] << 8)
        bound = int(ring.output_bound(meta.in_features))
        if int(np.abs(yq).max()) > bound:
            self._raise_status(2, 0, lid, rows, bound)
        out = dequantize_outputs(torch.from_numpy(yq), torch.from_numpy(sx), meta.scale, meta.bias, torch.float32)
        return out

    def _forward_local_one(self, call: LinearCall) -> torch.Tensor:
        assert self._local_w is not None
        w: _IntWeight = self._local_w.get(call.id)
        meta: _MaskedMeta = self._meta.get(call.id)
        x = call.x
        if x.ndim != 2 or x.shape[1] != meta.in_features:
            raise ShapeError(f"{call.id}: x must be (T, {meta.in_features}), got {tuple(x.shape)}")
        gp = self._local_gpu
        if gp is not None and x.device.type == "cpu" and gp.has(call.id.key):
            return self._local_gpu_forward(gp, x, meta, call.id)
        return self._forward_local_cpu(x, meta, w, call)

    def _forward_local_cpu(self, x: torch.Tensor, meta: _MaskedMeta, w: _IntWeight, call: LinearCall) -> torch.Tensor:
        """The CPU kernels of :meth:`_forward_local_one` (the native fast path, else the reference)."""
        if x.device.type == "cpu" and self._local_fast_ok(meta, w, int(x.shape[0]), call.id):
            xn = np.ascontiguousarray(x.detach().to(torch.float32).numpy())
            if not np.isfinite(xn).all():
                raise RingOverflowError("non-finite activation values cannot be quantized")
            blk = int(self.local_native_block_rows)
            if xn.shape[0] <= blk or not self._local_native:
                return self._local_fast(xn, meta, w, call.id)
            return torch.cat([self._local_fast(np.ascontiguousarray(xn[i : i + blk]), meta, w, call.id)
                              for i in range(0, xn.shape[0], blk)])
        return self._local_reference(x, meta, w, call.id)

    def _local_reference(self, x: torch.Tensor, meta: _MaskedMeta, w: _IntWeight, lid: LinearId) -> torch.Tensor:
        xf = x.detach().to("cpu", torch.float32)
        if xf.numel() and not bool(torch.isfinite(xf).all()):
            raise RingOverflowError("non-finite activation values cannot be quantized")
        # the masked path's quantization (same weight-row norm) and the exact integer product
        xq, sx = quantize_activations(xf, self.ring, meta.wnorm if self.ring.cap else None)
        try:
            yq = self._local_int_product(w, xq)
        finally:
            _wipe(xq)
        # the masked path's range check (status 2 of its finish): same refusal on the same outputs
        bound = int(self.ring.output_bound(meta.in_features))
        if yq.numel() and int(yq.abs().max()) > bound:
            self._raise_status(2, 0, lid, int(yq.shape[0]), bound)
        y = dequantize_outputs(yq, sx, meta.scale, meta.bias, torch.float32)
        return y if x.device.type == "cpu" else y.to(x.device)

    def _local_lean_one(self, call: LinearCall) -> torch.Tensor | None:
        """:attr:`trip_fast`: one CPU local call on the native kernel with its per-linear constants cached
        (built by the ordinary path after the per-shape self-check passed); ``None``: not eligible (the
        ordinary path runs). Same kernel, arguments, checks and bits as :meth:`_local_fast`."""
        key = call.id.key
        ent = self._local_lean.get(key)
        if ent is None:
            return None
        w, w8n, meta, scale, bias, bound, wn, thr = ent
        lw = self._local_w
        x = call.x
        if (lw is None or lw._cache.get(key) is not w or x.ndim != 2 or x.device.type != "cpu"
                or (self._local_gpu is not None and self._local_gpu.has(key))):
            return None
        rows, n = int(x.shape[0]), int(x.shape[1])
        if n != meta.in_features or not 0 < rows <= int(self.local_native_block_rows):
            return None
        xn = x.detach()
        if xn.dtype != torch.float32:
            xn = xn.to(torch.float32)
        xn = xn.numpy()
        if not xn.flags.c_contiguous:
            xn = np.ascontiguousarray(xn)
        if not np.isfinite(xn).all():
            raise RingOverflowError("non-finite activation values cannot be quantized")
        ring = self.ring
        res = np.empty((rows, meta.out_features), dtype=np.float32)
        sx = np.empty(rows, dtype=np.float64)
        try:
            status = native.fast.local_linear_into(xn, w8n, scale, bias, ring.k, ring.a, int(ring.cap or 0), wn,
                                                   bound, sx, res, thr)
        except ValueError as exc:
            if str(exc).startswith("ring_overflow"):
                raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
            raise ShapeError(str(exc)) from exc
        if status:
            self._raise_status(int(status), 0, call.id, rows, bound)
        return torch.from_numpy(res)

    def _local_lean_learn(self, call: LinearCall) -> None:
        """Cache :meth:`_local_lean_one`'s constants for ``call``'s linear once the ordinary path would take
        the native kernel for it (self-check passed, CPU weights, no GPU product)."""
        lw = self._local_w
        if lw is None or not self._local_native or len(self._local_lean) > 4096:
            return
        key = call.id.key
        w = lw._cache.get(key)
        if w is None or w.w8t.device.type != "cpu" or not w.w8t.is_contiguous():
            return
        if self._local_gpu is not None and self._local_gpu.has(key):
            return
        meta: _MaskedMeta = self._meta.get(call.id)
        if not self._local_fast_ok(meta, w, 1, call.id) or not self._local_native:
            return
        scale, bias, bound = self._consts(call.id, meta)
        hit = self._local_w8n.get(key)
        if hit is None or hit[0] is not w.w8t:
            hit = self._local_w8n[key] = (w.w8t, w.w8t.numpy())
        wn = float(meta.wnorm or 0.0) if self.ring.cap else 0.0
        thr = 0 if meta.in_features * meta.out_features >= self.local_pool_min_macs else 1
        self._local_lean[key] = (w, hit[1], meta, scale, bias, bound, wn, thr)

    def forward_local(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """Engine hook: evaluate calls of local linears on the gateway (bit-identical to the masked path)."""
        t0 = time.perf_counter()
        if len(calls) == 1 and self._local_one_row:  # decode on the gateway GPU: one lean native call
            c = calls[0]
            fn = self._local_one_row.get(c.id.key)  # only local linears have one
            x = c.x
            if (fn is not None and x.dtype == torch.float32 and x.ndim == 2 and x.shape[0] == 1
                    and x.device.type == "cpu" and not x.requires_grad and x.is_contiguous()):
                y = fn(x)
                dt = time.perf_counter() - t0
                self.stats.add(local_calls=1, local_rows=1, local_seconds=dt)
                lt = self.local_linear_timing
                lt["decode_calls"] += 1
                lt["decode_s"] += dt
                return [y]
        for c in calls:
            if not self.local_linear(c.id):
                raise ConfigError(f"{c.id} is not a local linear")
        if self.trip_fast and len(calls) == 1:
            y = self._local_lean_one(calls[0])
            if y is not None:
                self.stats.add_local(1, int(calls[0].x.shape[0]), time.perf_counter() - t0)
                return [y]
            out = _grouped_forward(_LocalView(self), calls)
            self._local_lean_learn(calls[0])
            self.stats.add_local(1, int(calls[0].x.shape[0]), time.perf_counter() - t0)
            return out
        out = _grouped_forward(_LocalView(self), calls)
        dt = time.perf_counter() - t0
        rows = sum(int(c.x.shape[0]) for c in calls)
        self.stats.add(local_calls=len(calls), local_rows=rows, local_seconds=dt)
        lt = self.local_linear_timing
        kind = "decode" if rows == len(calls) == 1 else "other"
        lt[kind + "_calls"] += len(calls)
        lt[kind + "_s"] += dt
        return out

    def _take(self, res: Any, lid: LinearId, max_rows: int, want_checks: bool,
              prime_ok: bool = False) -> tuple[int, Any, Any, Any]:
        """``(rows, r, W r, (token, u, uw) | None)``: the next rows of one extent (at most ``max_rows``).

        Prime-field vectors (:func:`pgw.core.freivalds.is_prime_token`) are refused unless the caller
        runs the prime check (``prime_ok``, the TripEngine paths): fail closed, the rows are burned."""
        fn = getattr(res, "take_checked", None)
        if fn is not None:
            out = fn(lid, max_rows, want_checks)
        else:
            boundary = int(res.rows_until_boundary(lid))
            if boundary <= 0:
                raise MaskExhaustedError("mask reservation has no rows left", details={"linear": lid.key})
            rows = min(max_rows, boundary)
            # vectors of the extent the next rows come from: read before take()
            cf = getattr(res, "check_field", None)
            if want_checks and callable(cf) and cf(lid) is not None:
                chk = res.checks_compact(lid)
            else:
                chk = (None, *self._checks_u64(res, lid)) if want_checks else None
            r, wr = res.take(lid, rows)
            out = (rows, r, wr, chk)
        if out[3] is not None and not prime_ok:
            self._ring_only(out[3][0], lid)
        return out  # type: ignore[no-any-return]

    def _ring_only(self, token: Any, lid: Any) -> None:
        """Raise if ``token`` names prime-field vectors on a path that runs only the ring check."""
        if is_prime_token(token):
            raise ConfigError(f"{lid}: prime-field check vectors on a path without the prime check "
                              "(padded / fixed-structure calls or a non-TripEngine fallback)")

    def _prepare_native(self, calls: Sequence[LinearCall]) -> _Prepared:
        """:meth:`_prepare_python` with quantize + mask + pack fused in ``pgw_fast`` (bit-identical)."""
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
        if self.prepare_many and (len(calls) > 1 or sum(int(x.shape[0]) for x in xs) > 1):
            self._native_many(calls, reservations, metas, xs, plan_call, guard_packed, chunks, sx_parts)
            return self._native_request(calls, chunks, sx_parts, metas, t0)
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
        return self._native_request(calls, chunks, sx_parts, metas, t0)

    def _native_request(self, calls: Sequence[LinearCall], chunks: list[_Chunk], sx_parts: list[list[np.ndarray]],
                        metas: list[_MaskedMeta], t0: float) -> _Prepared:
        from pgw.wire.messages import ComputeItem, ComputeRequest

        sxs = [np.concatenate(p) if len(p) > 1 else (p[0] if p else np.zeros(0, dtype=np.float64))
               for p in sx_parts]
        request = None
        if chunks:
            request = ComputeRequest(
                # uuid4 like every other compute request (filler-only and Python-path trips too): a
                # different id format per path would tell the provider which trips carry real rows
                request_id=uuid.uuid4().hex,
                fingerprint=self.fingerprint,
                ring=self.ring.name,
                items=tuple(ComputeItem(ch.lid.key, ch.rows, ch.payload) for ch in chunks),  # type: ignore[arg-type]
                deadline_ms=self.deadline_ms,
            )
        return _Prepared(calls, chunks, sxs, metas, request, time.perf_counter() - t0)

    def _native_many(self, calls: Sequence[LinearCall], reservations: list[Any], metas: list[_MaskedMeta],
                     xs: list[np.ndarray], plan_call: Any, guard_packed: Any, chunks: list[_Chunk],
                     sx_parts: list[list[np.ndarray]]) -> None:
        """The per-item loop of :meth:`_prepare_native` with one ``quantize_mask_pack_many`` call.

        Chunks, their rows, the check and noise-sampling draws, and every byte are the same as
        the per-item loop's: masks are taken chunk by chunk in the same order, then all chunks are
        quantized, masked and packed in one native call (several sessions, or row blocks of a
        prefill, in parallel). The difference is on failure only: a ring overflow or a noise alarm
        in a chunk also burns the rows already taken for the later chunks of the batch (the
        per-item loop stops taking at the failing chunk). Burned rows are never used again.
        """
        ring = self.ring
        noise = self.noise_monitor
        group_finish = self._native_group
        want_soundness = self.min_soundness_bits > 0
        max_item = self.max_rows_per_item
        # [call index, first row, rows, linear, r, W r, token, u, uw, sampled, want x_hat, wnorm, in_features]
        pend: list[list[Any]] = []
        try:
            try:
                for ci, (call, res, meta, xn) in enumerate(zip(calls, reservations, metas, xs, strict=True)):
                    t = int(xn.shape[0])
                    lid = call.id
                    wn = float(meta.wnorm or 0.0)
                    row = 0
                    while row < t:
                        check = self._should_check()
                        sampled = bool(plan_call()) if plan_call is not None else noise is not None
                        rows, r, wr, chk = self._take(res, lid, min(t - row, max_item), check or want_soundness)
                        p = [ci, row, rows, lid, r, wr, None, None, None, sampled, not group_finish or sampled,
                             wn, meta.in_features]
                        pend.append(p)
                        if chk is not None:
                            token, u, uw = chk
                            self._check_soundness(lid, meta.in_features, u, token)
                            if check:
                                p[6], p[7], p[8] = token, u, uw
                        if tuple(r.shape) != (rows, meta.in_features) or tuple(wr.shape) != (rows, meta.out_features):
                            raise ShapeError(
                                f"{lid}: mask shapes {tuple(r.shape)}/{tuple(wr.shape)} do not match the linear"
                            )
                        row += rows
                try:
                    outs = native.fast.quantize_mask_pack_many(
                        [xs[p[0]][p[1] : p[1] + p[2]] for p in pend], [p[4].numpy() for p in pend],
                        ring.k, ring.a, int(ring.cap or 0), [p[11] for p in pend], [p[10] for p in pend],
                    )
                except ValueError as exc:
                    if str(exc).startswith("ring_overflow"):
                        raise RingOverflowError(str(exc).split(":", 1)[-1].strip()) from exc
                    raise ShapeError(str(exc)) from exc
                for p, (payload, xh, sx) in zip(pend, outs, strict=True):
                    x_hat = None if xh is None else torch.from_numpy(xh)
                    if noise is not None:
                        if guard_packed is not None and not p[9]:
                            guard_packed(payload, p[2], p[12])
                        elif x_hat is not None:  # xq = centered(x_hat - r), for its checks
                            xq = center(x_hat - p[4].to(torch.int64), ring)
                            try:
                                if plan_call is not None:
                                    noise.guard(x_hat, xq, planned=True)  # type: ignore[attr-defined]
                                else:
                                    noise.guard(x_hat, xq)  # type: ignore[attr-defined]
                            finally:
                                _wipe(xq)
                        if group_finish:
                            x_hat = None  # the group finish reads the payload: every chunk alike
                    chunks.append(_Chunk(p[0], p[1], p[2], p[3], x_hat, p[5], p[7], p[8], payload, p[6]))
                    sx_parts[p[0]].append(sx)
            finally:
                for p in pend:
                    _wipe(p[4])  # one-time pads: never keep them around
        except BaseException:
            for p in pend:
                _wipe(p[5])
            raise

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

    def _finish_one(self, prep: _Prepared, response: Any, remote_s: float) -> list[torch.Tensor]:
        """:meth:`_finish` for a request of the one-call trip path: the same response checks (id, item
        count, linear and rows per item), the same per-chunk finish (:meth:`_finish_trip`: Freivalds when
        drawn, unmask, range check, rescale, W r wiped), the same statistics and integrity reporting."""
        t0 = time.perf_counter()
        req = prep.request
        chunks = prep.chunks
        outs: list[list[torch.Tensor]] = [[]]
        bytes_down = checked = skipped = 0
        try:
            if getattr(response, "request_id", None) != req.request_id:
                raise ProtocolError("compute response id does not match the request")
            results = tuple(response.items)
            if len(results) != len(chunks):
                raise ProtocolError(f"compute response has {len(results)} items, expected {len(chunks)}")
            for ch, res in zip(chunks, results, strict=True):
                if res.linear != ch.lid.key or int(res.rows) != ch.rows:
                    raise ProtocolError(
                        "compute response item does not match the request",
                        details={"expected": ch.lid.key, "got": str(res.linear)},
                    )
                bytes_down += len(res.y_hat)
                if ch.u is None:
                    skipped += 1
                elif ch.rhs is None or is_prime_token(ch.token):  # a deferred check counts once it passed
                    checked += 1
            self._finish_trip(prep, results, outs)
        except IntegrityError as exc:
            self.stats.add(integrity_failures=1)
            log.error("integrity check failed: request=%s code=%s details=%s", req.request_id, exc.code, exc.details)
            if self.on_integrity_failure is not None:
                try:
                    self.on_integrity_failure(exc)
                except Exception:  # pragma: no cover - callback must not mask the error
                    log.exception("on_integrity_failure callback raised")
            raise
        finally:
            prep.wipe()
        parts = outs[0]
        y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        dev = prep.calls[0].x.device
        rows = sum(ch.rows for ch in chunks)
        self.stats.add_many(1, rows, 1, sum(len(it.x_hat) for it in req.items), bytes_down, checked, skipped,
                            remote_s, prep.t_prepare + time.perf_counter() - t0, 0 < rows <= self.decode_max_rows)
        return [y if dev.type == "cpu" else y.to(dev)]

    def _finish(self, prep: _Prepared, response: Any, remote_s: float) -> list[torch.Tensor]:
        if prep.ticket is not None:
            return self._finish_lease(prep, response, remote_s)
        if prep.tickets is not None:
            return self._finish_leases(prep, response, remote_s)
        if prep.one:
            return self._finish_one(prep, response, remote_s)
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
                n_items = len(prep.items_map) if prep.items_map is not None else len(prep.chunks)
                if len(results) != n_items:
                    raise ProtocolError(
                        f"compute response has {len(results)} items, expected {n_items}"
                    )
                if prep.items_map is not None:  # speculation padding: dummy finish, real rows remain
                    results, pad_bytes = self._split_padded_results(prep, results, outs)  # type: ignore[assignment]
                    bytes_down += pad_bytes
            else:
                results = ()
            for ch, res in zip(prep.chunks, results, strict=True):
                if res.linear != ch.lid.key or int(res.rows) != ch.rows:
                    raise ProtocolError(
                        "compute response item does not match the request",
                        details={"expected": ch.lid.key, "got": str(res.linear)},
                    )
                bytes_down += len(res.y_hat)
                if ch.u is None:
                    skipped += 1
                elif not (prep.trip and ch.rhs is not None) or is_prime_token(ch.token):  # deferred: counted when it passed
                    checked += 1
            if prep.trip:
                self._finish_trip(prep, results, outs)
            elif use_native and self._native_group and all(ch.payload is not None for ch in prep.chunks):
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
        rows = sum(ch.rows for ch in prep.chunks) + sum(fx.rows for fx in prep.fixed)
        for fx in prep.fixed:  # fixed-structure padding items (L3): one check per item when drawn
            if any(u is not None for _, u, _ in fx.parts):
                checked += 1
            else:
                skipped += 1
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
            if checked and first.u is not None and _vacuous(first.u, status):
                status = 3  # vectors wiped (reservation closed): an all-zero u verifies nothing
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
        if checked and ch.u is not None and _vacuous(ch.u, status):
            status = 3  # vectors wiped (reservation closed): an all-zero u verifies nothing
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
        drive = (self.trip_driver and prep.ticket is not None and hasattr(self.client, "trip_ticket")
                 and getattr(self.client, "lane_eligible_for_driver", True))
        if drive:
            done = self._finish_lease(prep, None, 0.0, drive=True)
            if done is not None:
                return done
        try:
            if drive and not getattr(self.client, "lane_eligible_for_driver", True):
                # the lane's channel cannot take the driver (e.g. the Python UDS channel): use the ordinary
                # lane path, which still serves the call on the lane (same request bytes and id)
                response = self.client.compute_sync(prep.request)
            elif drive:
                response = self.client.fallback_sync(prep.request)
            elif hasattr(self.client, "compute_sync"):
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


class _LocalView:
    """``forward`` = the masked backend's exact local product (for :func:`_grouped_forward`)."""

    def __init__(self, backend: MaskedBackend):
        self.forward = backend._forward_local_one


def _trip_kernels_agree(ring: RingParams, n: int, m: int, wnorm: float) -> bool:
    """Self-check of the TripEngine for one linear shape on public, deterministic test data.

    ``trip_prepare_into`` must give the reference payload and scales (``quantize_activations``,
    ``mask_activations``, ``pack_ring``) for a Gaussian row, a spike row and a zero row, and
    ``trip_finish_into`` the reference outputs (``dequantize_outputs`` of ``centered(y_hat - W r)``)
    and the range verdict of an out-of-range row.
    """
    rng = np.random.default_rng(0x9E3779B9 ^ (n * 1_000_003 + m))
    rows = 3
    k, nb = ring.k, ring.k // 8
    x = (rng.standard_normal((rows, n)) * 3).astype(np.float32)
    x[1] *= np.float32(1e-3)
    x[1, int(rng.integers(0, n))] = 50.0
    x[2] = 0.0
    r = rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64)
    wn = float(wnorm) if ring.cap else 0.0
    try:
        xq, sx_ref = quantize_activations(torch.from_numpy(x), ring, wn if ring.cap else None)
    except RingOverflowError:
        # No width fits some test row at this norm. Fail closed: the trip kernel must refuse the same
        # rows, and the comparison runs on the rows that do fit; it needs at least one non-zero row.
        buf = np.empty(rows * n * nb, dtype=np.uint8)
        sx = np.empty(rows, dtype=np.float64)
        try:
            native.fast.trip_prepare_into(x, r, k, ring.a, int(ring.cap or 0), wn, buf, sx)
        except ValueError as exc:
            if not str(exc).startswith("ring_overflow"):
                return False
        else:
            return False  # accepted what the reference refuses
        keep = []
        for i in range(rows):
            try:
                quantize_activations(torch.from_numpy(x[i : i + 1]), ring, wn if ring.cap else None)
            except RingOverflowError:
                continue
            keep.append(i)
        if not any(x[i].any() for i in keep):
            return False  # nothing non-trivial fits: no evidence the kernels agree
        x, r, rows = np.ascontiguousarray(x[keep]), np.ascontiguousarray(r[keep]), len(keep)
        xq, sx_ref = quantize_activations(torch.from_numpy(x), ring, wn if ring.cap else None)
    want = pack_ring(mask_activations(xq, torch.from_numpy(r), ring), ring)
    buf = np.empty(rows * n * nb, dtype=np.uint8)
    sx = np.empty(rows, dtype=np.float64)
    native.fast.trip_prepare_into(x, r, k, ring.a, int(ring.cap or 0), wn, buf, sx)
    if buf.tobytes() != want or not np.array_equal(sx.view(np.uint64), sx_ref.numpy().view(np.uint64)):
        return False
    bound = int(ring.output_bound(n))
    lim = min(bound, 1 << (k - 2), 1 << 20)
    yq = rng.integers(-lim, lim + 1, size=(rows, m), dtype=np.int64)
    wr = rng.integers(0, 1 << k, size=(rows, m), dtype=np.int64)
    y_hat = pack_ring(torch.from_numpy((wr + yq) & ring.mask), ring)
    scale = (rng.random(m) + 0.5).astype(np.float32) * np.float32(1e-3)
    bias = rng.standard_normal(m).astype(np.float32)
    ref = dequantize_outputs(torch.from_numpy(yq), torch.from_numpy(sx), torch.from_numpy(scale),
                             torch.from_numpy(bias), torch.float32).numpy()
    out = np.empty((rows, m), dtype=np.float32)
    status, _ = native.fast.trip_finish_into(y_hat, wr.copy(), sx, scale, bias, k, bound, out)
    if status != 0 or not np.array_equal(out.view(np.uint32), ref.view(np.uint32)):
        return False
    if bound < (1 << (k - 1)) - 1:  # an out-of-range output must be refused
        bad = yq.copy()
        bad[0, 0] = bound + 1
        y_bad = pack_ring(torch.from_numpy((wr + bad) & ring.mask), ring)
        status, _ = native.fast.trip_finish_into(y_bad, wr.copy(), sx, scale, bias, k, bound, out)
        if status != 2 or out.any():
            return False
    return True


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
