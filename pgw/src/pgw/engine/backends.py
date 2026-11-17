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
import inspect
import logging
import secrets
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from pgw.core.errors import (
    ConfigError,
    FingerprintMismatchError,
    IntegrityError,
    MaskExhaustedError,
    PGWError,
    ProtocolError,
    ServiceUnavailableError,
    ShapeError,
)
from pgw.core.freivalds import freivalds_check
from pgw.core.masking import mask_activations, unmask_outputs
from pgw.core.packing import pack_ring, unpack_ring
from pgw.core.ring import RingParams, balanced_digits, dequantize_outputs, quantize_activations
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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **kw: float) -> None:
        """Increment counters."""
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

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
_INT_MM_MAX_N = 131_064
_cpu_int_mm_ok: bool | None = None


def _cpu_int_mm_available() -> bool:
    global _cpu_int_mm_ok
    if _cpu_int_mm_ok is None:
        try:
            a = torch.ones((1, 8), dtype=torch.int8)
            _cpu_int_mm_ok = bool(torch._int_mm(a, torch.ones((8, 8), dtype=torch.int8)).sum() == 64)
        except Exception:  # pragma: no cover - depends on the torch build
            _cpu_int_mm_ok = False
            log.info("torch._int_mm unavailable on CPU; using float64 exact fallback")
    return _cpu_int_mm_ok


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
    so the result is an exact int64 regardless of BLAS summation order. Falls back to float64 GEMMs
    (exact while partial sums stay below 2^53) when int8 GEMMs are unavailable.
    """
    t, n = xq.shape
    m = w8t.shape[1]
    if t == 0:
        return torch.zeros((0, m), dtype=torch.int64, device=xq.device)
    dev = xq.device
    use_int = (dev.type == "cpu" and _cpu_int_mm_available()) or dev.type == "cuda"
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
        y = torch.zeros((t, m), dtype=torch.int64, device=dev)
        for c0 in range(0, n, _INT_MM_MAX_N):
            c1 = min(n, c0 + _INT_MM_MAX_N)
            a = d8[:, c0:c1].contiguous()
            b = w8t[c0:c1]
            if pad_rows:  # pragma: no cover - GPU only
                y += _cuda_int_mm_padded(a, b)
            else:
                y += torch._int_mm(a, b).to(torch.int64)
        acc += y << (8 * i)
    return acc


def _cuda_int_mm_padded(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pragma: no cover - GPU only
    t, n = a.shape
    m = b.shape[1]
    tp, np_, mp = max(32, -(-t // 8) * 8), -(-n // 8) * 8, -(-m // 8) * 8
    ap = torch.zeros((tp, np_), dtype=torch.int8, device=a.device)
    ap[:t, :n] = a
    bp = torch.zeros((np_, mp), dtype=torch.int8, device=a.device)
    bp[:n, :m] = b
    return torch._int_mm(ap, bp)[:t, :m].to(torch.int64)


# ---- weight access ---------------------------------------------------------------------------------
class _WeightCache:
    """Lazily loads linear weights from a pack and keeps a backend-specific form resident."""

    def __init__(self, pack: Any, device: torch.device, convert: Any):
        self.pack = pack
        self.device = device
        self.convert = convert
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
                lw = self.pack.load_linear(lid, device="cpu")
                hit = self.convert(lw, self.device)
                self._cache[key] = hit
        return hit

    def preload(self) -> None:
        for spec in self.pack.linear_specs():
            self.get(spec.id)


@dataclass(frozen=True)
class _IntWeight:
    w8t: torch.Tensor  # int8 (n, m)
    scale: torch.Tensor  # float32 (m,)
    bias: torch.Tensor | None
    in_features: int
    out_features: int


def _to_int_weight(lw: Any, device: torch.device) -> _IntWeight:
    w8 = lw.w8
    if w8.dtype != torch.int8 or w8.ndim != 2:
        raise ShapeError(f"{lw.id}: weight must be 2-D int8")
    return _IntWeight(
        w8.t().contiguous().to(device),
        lw.scale.to(torch.float32).to(device),
        None if lw.bias is None else lw.bias.to(torch.float32).to(device),
        int(w8.shape[1]),
        int(w8.shape[0]),
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
        xq, sx = quantize_activations(x.to(self.device), self.ring)
        yq = exact_int_matmul(xq, w.w8t, self.ring.a)
        y = dequantize_outputs(yq, sx, w.scale, w.bias, torch.float32)
        self.stats.add(calls=1, rows=int(x.shape[0]), local_seconds=time.perf_counter() - t0)
        return y.to(x.device)

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        """Concatenate rows of calls on the same linear into one product (exactness unaffected)."""
        if len(calls) <= 1:
            return [self.forward(c) for c in calls]
        out: list[torch.Tensor | None] = [None] * len(calls)
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(calls):
            groups.setdefault(c.id.key, []).append(i)
        for idxs in groups.values():
            if len(idxs) == 1:
                out[idxs[0]] = self.forward(calls[idxs[0]])
                continue
            xs = [calls[i].x for i in idxs]
            y = self.forward(LinearCall(calls[idxs[0]].id, torch.cat(xs, dim=0)))
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

    def __init__(self, pack: object, ring: RingParams, device: str = "cpu", *, dtype: torch.dtype = torch.float32):
        self.pack = pack
        self.ring = ring
        self.device = torch.device(device)
        self.dtype = dtype
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


# ---- masked backend ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class _MaskedMeta:
    """What the gateway needs per linear: no int8 weights, only scales, bias and shapes."""

    scale: torch.Tensor
    bias: torch.Tensor | None
    in_features: int
    out_features: int


@dataclass
class _Chunk:
    """One wire item: contiguous rows of one call that live in a single mask extent."""

    call_index: int
    row0: int
    rows: int
    lid: LinearId
    x_hat: torch.Tensor
    wr: torch.Tensor
    u: torch.Tensor | None
    uw: torch.Tensor | None


@dataclass
class _Prepared:
    calls: Sequence[LinearCall]
    chunks: list[_Chunk]
    sx: list[torch.Tensor]
    metas: list[_MaskedMeta]
    request: Any  # ComputeRequest, or None when every call has zero rows
    t_prepare: float

    def wipe(self) -> None:
        """Zero the ``W r`` buffers (secret-derived) of every chunk; idempotent."""
        for ch in self.chunks:
            ch.wr.zero_()


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
        self.fingerprint = fingerprint
        self.freivalds_rate = freivalds_rate
        self.noise_monitor = noise_monitor
        self.max_rows_per_item = max_rows_per_item
        self.deadline_ms = deadline_ms
        self.on_integrity_failure = on_integrity_failure
        self.stats = BackendStats()

        def conv(lw: Any, dev: torch.device) -> _MaskedMeta:
            n, m = int(lw.w8.shape[1]), int(lw.w8.shape[0])
            ring.check_in_features(n)
            return _MaskedMeta(
                lw.scale.to(torch.float32).clone(),
                None if lw.bias is None else lw.bias.to(torch.float32).clone(),
                n,
                m,
            )

        self._meta = _WeightCache(pack, torch.device("cpu"), conv)

    # ---- helpers -------------------------------------------------------------------------------
    def _reservation(self, call: LinearCall) -> Any:
        res = call.session if call.session is not None else self.mask_source
        if res is None or not hasattr(res, "take"):
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
        return res

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
        for c in calls:
            meta: _MaskedMeta = self._meta.get(c.id)
            x = c.x
            if x.ndim != 2 or x.shape[1] != meta.in_features:
                raise ShapeError(f"{c.id}: x must be (T, {meta.in_features}), got {tuple(x.shape)}")
            metas.append(meta)
        need: dict[tuple[int, str], int] = {}
        for c, res in zip(calls, reservations, strict=True):
            key = (id(res), c.id.key)
            need[key] = need.get(key, 0) + int(c.x.shape[0])
        for c, res in zip(calls, reservations, strict=True):
            left = int(res.remaining(c.id))
            if left < need[(id(res), c.id.key)]:
                raise MaskExhaustedError(
                    "session mask reservation exhausted",
                    details={"linear": c.id.key, "needed": need[(id(res), c.id.key)], "remaining": left},
                )
        return reservations, metas

    def _prepare(self, calls: Sequence[LinearCall]) -> _Prepared:
        from pgw.wire.messages import ComputeItem, ComputeRequest

        t0 = time.perf_counter()
        reservations, metas = self._precheck(calls)
        # Quantize everything first: a non-finite activation in a later call must not consume
        # mask rows of earlier calls.
        quantized: list[torch.Tensor] = []
        sxs: list[torch.Tensor] = []
        for call in calls:
            xq, sx = quantize_activations(call.x.detach().to("cpu"), self.ring)
            quantized.append(xq)
            sxs.append(sx)
        chunks: list[_Chunk] = []
        try:
            for ci, (call, res, meta, xq) in enumerate(zip(calls, reservations, metas, quantized, strict=True)):
                t = int(xq.shape[0])
                row = 0
                while row < t:
                    boundary = int(res.rows_until_boundary(call.id))
                    if boundary <= 0:
                        raise MaskExhaustedError("mask reservation has no rows left", details={"linear": call.id.key})
                    rows = min(t - row, boundary, self.max_rows_per_item)
                    check = self._should_check()
                    u = uw = None
                    if check:  # vectors of the extent the next rows come from: read before take()
                        u, uw = res.checks(call.id)
                    r, wr = res.take(call.id, rows)
                    try:
                        if tuple(r.shape) != (rows, meta.in_features) or tuple(wr.shape) != (rows, meta.out_features):
                            raise ShapeError(
                                f"{call.id}: mask shapes {tuple(r.shape)}/{tuple(wr.shape)} do not match the linear"
                            )
                        x_hat = mask_activations(xq[row : row + rows], r.to(torch.int64), self.ring)
                    except BaseException:
                        wr.zero_()
                        raise
                    finally:
                        r.zero_()  # one-time pad: never keep it around
                    chunk = _Chunk(ci, row, rows, call.id, x_hat, wr, u, uw)
                    chunks.append(chunk)
                    if self.noise_monitor is not None:
                        self.noise_monitor.guard(x_hat, xq[row : row + rows])  # type: ignore[attr-defined]
                    row += rows
        except BaseException:
            for ch in chunks:
                ch.wr.zero_()
            raise
        finally:
            for xq in quantized:
                xq.zero_()
        request = None
        if chunks:
            items = tuple(ComputeItem(ch.lid.key, ch.rows, pack_ring(ch.x_hat, self.ring)) for ch in chunks)
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
                meta: _MaskedMeta = prep.metas[ch.call_index]
                if res.linear != ch.lid.key or int(res.rows) != ch.rows:
                    raise ProtocolError(
                        "compute response item does not match the request",
                        details={"expected": ch.lid.key, "got": str(res.linear)},
                    )
                bytes_down += len(res.y_hat)
                y_hat = unpack_ring(res.y_hat, (ch.rows, meta.out_features), self.ring)
                if ch.u is not None and ch.uw is not None:
                    ok = freivalds_check(ch.x_hat, y_hat, ch.u.to(torch.int64), ch.uw.to(torch.int64), self.ring)
                    if not bool(ok.all()):
                        raise IntegrityError(
                            "compute service result failed the Freivalds check",
                            details={"linear": ch.lid.key, "bad_rows": int((~ok).sum()), "rows": ch.rows},
                        )
                    checked += 1
                else:
                    skipped += 1
                yq = unmask_outputs(y_hat, ch.wr.to(torch.int64), self.ring, meta.in_features)
                ch.wr.zero_()
                sx = prep.sx[ch.call_index][ch.row0 : ch.row0 + ch.rows]
                outs[ch.call_index].append(dequantize_outputs(yq, sx, meta.scale, meta.bias, torch.float32))
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
            y = torch.cat(parts, dim=0) if parts else torch.zeros((0, meta.out_features), dtype=torch.float32)
            result.append(y.to(call.x.device))
        bytes_up = sum(len(it.x_hat) for it in req.items) if req is not None else 0
        self.stats.add(
            calls=len(prep.calls),
            rows=sum(ch.rows for ch in prep.chunks),
            round_trips=1 if req is not None else 0,
            bytes_up=bytes_up,
            bytes_down=bytes_down,
            freivalds_checked=checked,
            freivalds_skipped=skipped,
            remote_seconds=remote_s,
            local_seconds=prep.t_prepare + time.perf_counter() - t0,
        )
        return result

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
        if prep.request is None:
            return self._finish(prep, None, 0.0)
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
        try:
            return await asyncio.to_thread(self._finish, prep, response, time.perf_counter() - t0)
        except asyncio.CancelledError:
            prep.wipe()
            raise


def _run_coro_blocking(coro: Any) -> Any:
    """Run a coroutine to completion from synchronous code (no running loop in this thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise ConfigError("synchronous MaskedBackend.forward called inside an event loop; use aforward")
