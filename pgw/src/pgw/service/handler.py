"""The service's :class:`pgw.wire.ComputeHandler`: auth, admission control, quotas and compute.

Request path (``POST /pgw/v1/compute``, after the wire router validated headers, size limits and
payload lengths against :meth:`ServiceHandler.linear_dims`):

1. admission: draining -> ``unavailable``; more than ``max_inflight_requests`` -> ``unavailable``
   with ``Retry-After``; per-tenant concurrency and row-rate quotas -> ``rate_limited``;
2. every item is unpacked to int64 ring elements; items of one request for the same linear are
   submitted to the :class:`BatchScheduler` as one group (:meth:`BatchScheduler.submit_parts`: one
   kernel call for the group instead of one per item), different linears concurrently (and
   items may share a batch with other tenants' rows);
3. results are packed and returned in request order with server timing.

Nothing here logs or retains payload bytes; the service keeps no state between requests.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from pgw.core.errors import (
    ErrorCode,
    PGWError,
    ProtocolError,
    RateLimitedError,
    ServiceUnavailableError,
)
from pgw.core.packing import pack_ring, unpack_ring
from pgw.core.ring import RingParams
from pgw.service.auth import TenantAuthenticator, TenantIdentity
from pgw.service.batcher import BatchScheduler, BatchTiming
from pgw.service.config import ServiceSettings
from pgw.service.metrics import ServiceMetrics
from pgw.service.registry import ModelRegistry
from pgw.wire.messages import (
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    LinearInfo,
    ModelInfo,
    ModelsResponse,
    ServerTiming,
)
from pgw.wire.server import RequestContext

__all__ = ["ServiceHandler", "TokenBucket"]

log = logging.getLogger(__name__)

_UNKNOWN_TENANT = "unknown"

#: Default for :attr:`ServiceHandler.group_min_rows` (``PGW_SERVICE_GROUP_MIN_ROWS`` overrides it).
GROUP_MIN_ROWS = 3

#: Default for :attr:`ServiceHandler.maskgen_concurrency` when the service has a CUDA device
#: (``PGW_SERVICE_MASKGEN_CONCURRENCY`` overrides it). Each item runs on its own CUDA stream.
MASKGEN_CONCURRENCY = 2

#: Default for :attr:`ServiceHandler.maskgen_concurrency` on CPU-only services: one CPU maskgen
#: call already uses every intra-op thread and the combine pool, so two at once only oversubscribe
#: the cores and double the peak working memory.
MASKGEN_CONCURRENCY_CPU = 1


def _uses_cuda(settings: Any) -> bool:
    """True if any device the service is configured for is (or resolves to) a CUDA device."""
    devs = list(getattr(settings, "devices", None) or []) or [getattr(settings, "device", "auto")]
    for d in devs:
        s = str(d).strip().lower()
        if s == "auto":
            if torch.cuda.is_available():
                return True
        elif s.startswith("cuda"):
            return True
    return False


def _maskgen_concurrency_default(settings: Any = None) -> int:
    fallback = MASKGEN_CONCURRENCY if settings is not None and _uses_cuda(settings) else MASKGEN_CONCURRENCY_CPU
    raw = os.environ.get("PGW_SERVICE_MASKGEN_CONCURRENCY", "").strip()
    if not raw:
        return fallback
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("ignoring PGW_SERVICE_MASKGEN_CONCURRENCY=%r (not an integer)", raw)
        return fallback


def _maskgen_versions_default() -> tuple[int, ...]:
    """Maskgen wire versions served: v1 always; v2 unless ``PGW_SERVICE_MASKGEN_V2`` is false."""
    raw = os.environ.get("PGW_SERVICE_MASKGEN_V2", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return (1,)
    if raw and raw not in ("1", "true", "yes", "on"):
        log.warning("ignoring PGW_SERVICE_MASKGEN_V2=%r (not a boolean)", raw)
    return (1, 2)


def _maskgen_item_v2(kern: Any, it: Any) -> bytes | bytearray:
    """Wire v2 item: bit-packed ``b`` in, ``round(W8 b mod 2^bb)`` bit-packed at ``yb`` bits out.

    The product runs on the kernel's wire entry point with whole-byte words: 8-byte words on CUDA
    (zero-copy views of the unpacked values; the one extra byte plane is cheap on a GPU), and
    ``ceil(bb/8)`` bytes on CPU, where every GEMM plane costs more than the byte copy. Both give
    ``y`` modulo a multiple of ``2^bb``, which is all that :func:`pgw.hemask.wire.pack_y_v2` needs.
    """
    import numpy as np

    from pgw.hemask.wire import le_words, pack_y_v2, unpack_bits

    n, m, cols = kern.in_features, kern.out_features, it.cols
    b = unpack_bits(it.b, n * cols, it.b_bits)  # values < 2^bb
    fast = getattr(kern, "mulmod64_wire", None)
    nb = -(-it.b_bits // 8)
    if fast is None:  # kernels without the wire entry point
        y = kern.mulmod64(b.reshape(n, cols), nb)
    else:
        dev = getattr(kern, "device", None)
        if getattr(dev, "type", "cpu") == "cuda":
            nb = 8
        raw = b if nb == 8 else np.ascontiguousarray(b.view(np.uint8).reshape(n * cols, 8)[:, :nb])
        # 8-byte words: a read-only view, pack_y_v2's first pass makes the working copy
        y = le_words(fast(memoryview(raw).cast("B"), cols, 8 * nb, 0), m * cols, nb, copy=False)
    return pack_y_v2(y, it.b_bits, it.y_bits, inplace=True)


def _maskgen_fused(kern: Any) -> bool:
    """True when ``kern`` takes the device-side CUDA maskgen path (:mod:`pgw.service.maskgen_gpu`)."""
    dev = getattr(kern, "device", None)
    if getattr(dev, "type", "cpu") != "cuda" or getattr(kern, "_pw", None) is None:
        return False
    from pgw.service.maskgen_gpu import fused_enabled

    return fused_enabled()


def _maskgen_item(kern: Any, it: Any, decode_busy: Any = None) -> bytes | bytearray:
    """``y`` bytes of one maskgen item (runs in a worker thread).

    CUDA kernels run the item end to end on the device (:func:`pgw.service.maskgen_gpu.run_item`:
    device-side v2 unpack/pack, ``ceil(bb/8)`` planes, fused combine; byte-identical), unless
    ``PGW_SERVICE_MASKGEN_FUSED=0``. ``decode_busy`` feeds the opt-in decode yield of that path.
    """
    if _maskgen_fused(kern):
        from pgw.service.maskgen_gpu import run_item

        return run_item(kern, it, decode_busy=decode_busy)
    if it.v2:
        return _maskgen_item_v2(kern, it)
    fast = getattr(kern, "mulmod64_wire", None)
    if fast is not None:  # raw bytes in, packed bytes out (device-side unpack/pack on CUDA)
        return bytes(fast(it.b, it.cols, it.q_bits, it.drop_bits))
    from pgw.hemask.wire import pack_y, unpack_b  # kernels without the wire entry point

    b = unpack_b(it.b, (kern.in_features, it.cols), it.q_bits)
    return pack_y(kern.mulmod64(b, it.q_bits // 8), it.drop_bits, it.q_bits)


def _group_min_rows_default() -> int:
    raw = os.environ.get("PGW_SERVICE_GROUP_MIN_ROWS", "").strip()
    if not raw:
        return GROUP_MIN_ROWS
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("ignoring PGW_SERVICE_GROUP_MIN_ROWS=%r (not an integer)", raw)
        return GROUP_MIN_ROWS


class TokenBucket:
    """Classic token bucket (monotonic clock).

    Args:
        rate: Tokens per second.
        burst: Bucket capacity.
    """

    def __init__(self, rate: float, burst: float, *, clock: Any = time.monotonic):
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate = float(rate)
        self.burst = float(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._t = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + (now - self._t) * self.rate)
        self._t = now

    def take(self, n: float) -> float:
        """Charge ``n`` tokens. Returns 0 on success, else seconds to wait before retrying.

        A request larger than ``burst`` is admitted once the bucket is full and then charged in
        full: the balance goes negative ("debt") and later requests wait until it is repaid. So
        oversized requests cannot exceed the configured long-run rate (charging only ``burst``
        would let a tenant send ``max_request`` rows per ``burst / rate`` seconds).
        """
        if n < 0:
            raise ValueError("token count must be >= 0")
        self._refill()
        need = min(float(n), self.burst)  # balance required before admission
        if self._tokens >= need:
            self._tokens -= float(n)
            return 0.0
        return (need - self._tokens) / self.rate


@dataclass
class _TenantState:
    inflight: int = 0
    bucket: TokenBucket | None = None


class ServiceHandler:
    """Implements :class:`pgw.wire.ComputeHandler` on top of the registry and batch scheduler.

    Args:
        settings: Service settings (limits, quotas, offload threshold).
        registry: Loaded models.
        batcher: Started :class:`BatchScheduler`.
        authenticator: Tenant authenticator, or ``None`` to skip authentication entirely
            (only valid together with ``settings.allow_unauthenticated``).
        metrics: Service metrics.
        maskgen_concurrency: Encrypted maskgen items computed at the same time (each on its own CUDA
            stream on GPUs; the rest wait). Bounds the working memory of large maskgen requests.
            Default :data:`MASKGEN_CONCURRENCY` with a CUDA device, :data:`MASKGEN_CONCURRENCY_CPU`
            on CPU-only services (``PGW_SERVICE_MASKGEN_CONCURRENCY`` overrides both). A slot is
            held until the worker thread finishes, also when the waiting request is cancelled.
        group_min_rows: Items of one request for the same linear are computed as one group (one
            kernel call) when they have at least this many rows in all, or when the linear has no
            fused one-row kernel. Below it, on a kernel serving one-row calls with the fused Triton
            GEMV, two one-row items run as two GEMV calls, which can beat one digit GEMM on 2 rows
            (T4). 0 turns grouping off (one kernel call per item, as before grouping existed).
            Default :data:`GROUP_MIN_ROWS` (``PGW_SERVICE_GROUP_MIN_ROWS``).
        maskgen_versions: Maskgen wire versions served and advertised (``vs`` in every reply). v1
            is always served; v2 (bit-packed, :mod:`pgw.hemask.wire`) unless turned off here or with
            ``PGW_SERVICE_MASKGEN_V2=false``. Clients only send v2 after a reply advertised it.
    """

    def __init__(
        self,
        settings: ServiceSettings,
        registry: ModelRegistry,
        batcher: BatchScheduler,
        authenticator: TenantAuthenticator | None,
        metrics: ServiceMetrics,
        *,
        group_min_rows: int | None = None,
        maskgen_concurrency: int | None = None,
        maskgen_versions: tuple[int, ...] | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.batcher = batcher
        self.authenticator = authenticator
        self.metrics = metrics
        self.draining = False
        self.inflight = 0
        self._tenants: dict[str, _TenantState] = {}
        self._idle = asyncio.Event()
        self._idle.set()
        self.maskgen_concurrency = (max(1, int(maskgen_concurrency)) if maskgen_concurrency is not None
                                    else _maskgen_concurrency_default(settings))
        self._maskgen_slots = asyncio.Semaphore(self.maskgen_concurrency)  # bounded, not one global lock
        #: per CUDA device: ``maskgen_concurrency`` slots each (multi-GPU services scale maskgen with devices)
        self._maskgen_dev_slots: dict[str, asyncio.Semaphore] = {}
        if maskgen_versions is None:
            maskgen_versions = _maskgen_versions_default()
        if 1 not in maskgen_versions or not set(maskgen_versions) <= {1, 2}:
            raise ValueError(f"maskgen_versions must contain 1 and only 1 or 2, got {maskgen_versions!r}")
        self.maskgen_versions: tuple[int, ...] = tuple(sorted(set(maskgen_versions)))
        self.group_min_rows = max(0, int(group_min_rows)) if group_min_rows is not None else _group_min_rows_default()
        #: requests answered by :meth:`compute_inline` (the stream channel's frame lane)
        self.inline_requests = 0
        #: the native trip lane's shared admission (``pgw_fast.LaneService``) while a lane runs: draining,
        #: in-flight caps and tenant row buckets are then one budget for the lane and this handler
        #: (see :mod:`pgw.service.lane`); ``None`` = this handler's own counters (no lane)
        self.lane_admission: Any = None
        #: lane calls served through :meth:`lane_compute` (calls without a CUDA graph)
        self.lane_requests = 0

    # ---- ComputeHandler protocol -------------------------------------------------------------
    def authenticate(self, authorization: str | None, client_cert_subject: str | None) -> TenantIdentity:
        """Delegate to :class:`TenantAuthenticator` (anonymous when auth is disabled)."""
        if self.authenticator is None:
            return TenantIdentity("anonymous", "anonymous")
        return self.authenticator.authenticate(authorization, client_cert_subject)

    def linear_dims(self, fingerprint: str, ring: str) -> Mapping[str, tuple[int, int]]:
        """``{linear: (m, n)}`` of a served model/ring (see :meth:`ModelRegistry.linear_dims`)."""
        return self.registry.linear_dims(fingerprint, ring)

    async def models(self) -> ModelsResponse:
        """Served models with their rings, linears and limits."""
        s = self.settings
        infos = []
        for m in self.registry.models():
            linears = tuple(LinearInfo(sp.id.key, sp.out_features, sp.in_features) for sp in m.specs)
            infos.append(ModelInfo(m.name, m.fingerprint, tuple(m.rings), linears, s.max_rows_per_item,
                                   s.max_request_bytes))
        return ModelsResponse(tuple(infos), server=s.server_name)

    async def compute(self, request: ComputeRequest, ctx: RequestContext) -> ComputeResponse:
        """Evaluate every item exactly; results in request order."""
        ident: TenantIdentity | None = ctx.identity if isinstance(ctx.identity, TenantIdentity) else None
        tenant = ident.tenant_id if ident is not None else "anonymous"
        shared = self._admit(tenant, request.total_rows, enter=True)
        st = self._tenants.get(tenant)
        if st is None:
            st = self._tenants.setdefault(tenant, _TenantState())
        self.inflight += 1
        st.inflight += 1
        self._idle.clear()
        self.metrics.inflight(self.inflight)
        try:
            return await self._compute(request, tenant)
        finally:
            self.inflight -= 1
            st.inflight -= 1
            if shared is not None:
                shared.exit(tenant)
            self.metrics.inflight(self.inflight)
            if self.inflight == 0:
                self._idle.set()

    def inline_ready(self, request: ComputeRequest) -> bool:
        """True when :meth:`compute_inline` can answer ``request`` right now: one item, whose kernel
        call the batcher would run inline (idle device, at most ``batch_inline_max_rows`` rows) and
        whose payloads the regular path would not offload to a thread. No side effects (quotas are
        charged by :meth:`compute_inline`). Raises for an unknown model or linear."""
        if self.draining or len(request.items) != 1:
            return False
        it = request.items[0]
        ring = RingParams.from_name(request.ring)
        m, _n = self.registry.get(request.fingerprint).linear_dims[it.linear]
        rows = int(it.rows)
        offload = self.settings.codec_offload_bytes
        if len(it.x_hat) > offload or rows * m * ring.bytes_per_element > offload:
            return False
        return self.batcher.can_run_inline(request.fingerprint, ring, it.linear, rows)

    def compute_inline(self, request: ComputeRequest, ctx: RequestContext) -> ComputeResponse:
        """:meth:`compute` for a request :meth:`inline_ready` accepted, run synchronously (the stream
        channel's frame lane): the same admission, tenant limits and quotas, in-flight accounting,
        kernel call (inline on the idle device), metrics and response."""
        ident: TenantIdentity | None = ctx.identity if isinstance(ctx.identity, TenantIdentity) else None
        tenant = ident.tenant_id if ident is not None else "anonymous"
        shared = self._admit(tenant, request.total_rows, enter=True)
        st = self._tenants.get(tenant)
        if st is None:
            st = self._tenants.setdefault(tenant, _TenantState())
        self.inflight += 1
        st.inflight += 1
        self._idle.clear()
        self.metrics.inflight(self.inflight)
        try:
            ring = RingParams.from_name(request.ring)
            it = request.items[0]
            m, n = self.registry.get(request.fingerprint).linear_dims[it.linear]
            rows = int(it.rows)
            x = unpack_ring(it.x_hat, (rows, n), ring)
            y, timing = self.batcher.run_inline(request.fingerprint, ring, it.linear, x, compact=True)
            if y.shape != (rows, m):  # pragma: no cover - kernel contract
                raise PGWError("kernel returned a wrong shape", code=ErrorCode.INTERNAL)
            self.inline_requests += 1
            return self._single_response(request, tenant, pack_ring(y, ring), timing)
        finally:
            self.inflight -= 1
            st.inflight -= 1
            if shared is not None:
                shared.exit(tenant)
            self.metrics.inflight(self.inflight)
            if self.inflight == 0:
                self._idle.set()

    def _single_response(self, request: ComputeRequest, tenant: str, data: bytes,
                         bt: BatchTiming) -> ComputeResponse:
        """Response (and metrics) of a one-item request whose result packs to ``data``."""
        it0 = request.items[0]
        self.metrics.rows(tenant, it0.linear.split(".", 1)[1], int(it0.rows))
        self.metrics.bytes_io(tenant, request.payload_bytes, len(data))
        return ComputeResponse(request.request_id, (ComputeResult(it0.linear, int(it0.rows), data),), ServerTiming(
            queue_ms=round(bt.queue_ms, 3), compute_ms=round(bt.compute_ms, 3), batch_rows=bt.batch_rows))

    async def maskgen(self, request: Any, identity: Any) -> Any:
        """``POST /pgw/v1/maskgen``: ``W8 b mod 2^64`` for MLWE-encrypted masks (see :mod:`pgw.hemask`).

        The provider sees only pseudorandom ciphertexts. Admission and tenant limits count every
        256-column group as 256 rows, like the compute path. Each item goes from its raw ``b`` bytes
        to its packed ``y`` bytes in one worker-thread call (on CUDA end to end on the device,
        :mod:`pgw.service.maskgen_gpu`; on CPU :meth:`ComputeKernel.mulmod64_wire`, wire v2 items
        also unpack and bit-pack on the CPU, :func:`_maskgen_item_v2`). The items of a request run
        concurrently; at most :attr:`maskgen_concurrency` items run at once per CUDA device (and
        in the shared CPU pool), across all requests. The route stamps the
        reply with the request's version and :attr:`maskgen_versions`.
        """
        from pgw.hemask.wire import MaskgenResponse, MaskgenResult

        ident: TenantIdentity | None = identity if isinstance(identity, TenantIdentity) else None
        tenant = ident.tenant_id if ident is not None else "anonymous"
        model = self.registry.get(request.fingerprint)
        dims = model.linear_dims
        for it in request.items:
            if it.linear not in dims:
                raise ProtocolError(f"unknown linear {it.linear!r}", details={"linear": it.linear})
            n = dims[it.linear][1]
            want = it.b_nbytes(n)  # v1: q/8 bytes per coefficient; v2: bb bits
            if len(it.b) != want:
                raise ProtocolError(f"maskgen {it.linear}: b has {len(it.b)} bytes, expected {want}")
        self._admit(tenant, sum(it.cols for it in request.items))
        self.inflight += 1
        self._idle.clear()
        try:
            # Items run concurrently (each under its device's slots), replies keep request order.
            ys = await asyncio.gather(*(self._maskgen_in_slot(model.kernel(it.linear), it) for it in request.items))
            out = [MaskgenResult(it.linear, it.cols, y) for it, y in zip(request.items, ys, strict=True)]
            return MaskgenResponse(request.request_id, tuple(out))
        finally:
            self.inflight -= 1
            if self.inflight == 0:
                self._idle.set()

    async def _maskgen_in_slot(self, kern: Any, it: Any) -> bytes | bytearray:
        """Run one maskgen item in a worker thread under :attr:`_maskgen_slots`.

        The slot is released when the thread finishes, not when the awaiting request goes away:
        a cancelled request (client disconnect) cannot stop ``to_thread`` work, so releasing on
        cancellation would let more than :attr:`maskgen_concurrency` items run at once.
        """
        slots = self._maskgen_slots_for(kern)
        await slots.acquire()
        try:
            fut = asyncio.ensure_future(asyncio.to_thread(_maskgen_item, kern, it, *self._maskgen_extra(kern)))
        except BaseException:
            slots.release()
            raise

        def _done(f: asyncio.Future[bytes | bytearray]) -> None:
            slots.release()
            if not f.cancelled():
                f.exception()  # retrieved: no "exception was never retrieved" for abandoned items

        fut.add_done_callback(_done)
        return await asyncio.shield(fut)

    def _maskgen_slots_for(self, kern: Any) -> asyncio.Semaphore:
        """The slot pool of ``kern``'s device: one per CUDA device, the shared pool otherwise."""
        dev = getattr(kern, "device", None)
        per_dev = getattr(self, "_maskgen_dev_slots", None)
        if per_dev is None or getattr(dev, "type", "cpu") != "cuda":
            return self._maskgen_slots
        key = str(dev)
        sem = per_dev.get(key)
        if sem is None:
            sem = per_dev[key] = asyncio.Semaphore(self.maskgen_concurrency)
        return sem

    def _maskgen_extra(self, kern: Any) -> tuple[Any, ...]:
        """Extra ``_maskgen_item`` arguments: the decode-busy probe for fused CUDA items."""
        if not _maskgen_fused(kern):
            return ()
        batcher = getattr(self, "batcher", None)
        if batcher is None:
            return ()
        return (lambda: batcher.queued_rows > 0,)

    def on_request_done(self, ctx: RequestContext, status: int, code: str | None) -> None:
        """Metrics hook called by the wire router after every compute reply."""
        ident = ctx.identity if isinstance(ctx.identity, TenantIdentity) else None
        tenant = ident.tenant_id if ident is not None else _UNKNOWN_TENANT
        self.metrics.request_done(tenant, code or "ok", time.monotonic() - ctx.received_at)

    def precheck(self) -> None:
        """Cheap admission run by the wire router *before* the request body is read."""
        s = self.settings
        if self.draining:
            raise ServiceUnavailableError("service is draining", details={"retry_after_s": s.retry_after_s})
        if self._inflight_total() >= s.max_inflight_requests:
            raise ServiceUnavailableError(
                "too many in-flight requests", details={"retry_after_s": s.retry_after_s}
            )

    def _inflight_total(self) -> int:
        """In-flight requests counted against ``max_inflight_requests``: this handler's, or with a lane
        the larger of this handler's and the shared count (lane calls included)."""
        shared = self.lane_admission
        return self.inflight if shared is None else max(self.inflight, int(shared.inflight()))

    async def lane_compute(self, fingerprint: str, linear: str, ring: RingParams, rows: int, body: bytes,
                           tenant: str) -> bytes:
        """One lane call that has no CUDA graph, on the regular kernel path (the lane admitted it and
        checked the header and body length; :mod:`pgw.service.lane` drains its metrics). Returns the
        packed ``y_hat``: the same bytes :meth:`compute` returns for the same one-item request."""
        self.inflight += 1
        self._idle.clear()
        try:
            m, n = self.registry.get(fingerprint).linear_dims[linear]
            x = unpack_ring(body, (rows, n), ring)
            y, _bt = await self.batcher.submit_timed(fingerprint, ring, linear, x, compact=True)
            if y.shape != (rows, m):  # pragma: no cover - kernel contract
                raise PGWError("kernel returned a wrong shape", code=ErrorCode.INTERNAL)
            self.lane_requests += 1
            return pack_ring(y, ring)
        finally:
            self.inflight -= 1
            if self.inflight == 0:
                self._idle.set()

    # ---- internals -----------------------------------------------------------------------
    def _admit(self, tenant: str, rows: int, *, enter: bool = False) -> Any:
        """Admission: draining, global and tenant in-flight caps, tenant row bucket. Raises on refusal.

        With a lane running the decision is the shared one (:attr:`lane_admission`); with ``enter`` the
        request is then counted in the shared in-flight totals and the returned object's
        ``exit(tenant)`` must be called when it finishes. Returns ``None`` without a lane (the caller's
        own counters apply).
        """
        s = self.settings
        if self.draining:
            raise ServiceUnavailableError("service is draining", details={"retry_after_s": s.retry_after_s})
        shared = self.lane_admission
        if shared is not None:
            code, retry_ms = shared.admit(tenant, int(rows), enter)
            if code == 0:
                return shared if enter else None
            retry = round(retry_ms / 1000.0, 3)
            if code == 3:  # rate_limited: tenant concurrency cap or row bucket
                raise RateLimitedError("tenant limit reached", details={"retry_after_s": retry})
            raise ServiceUnavailableError("service is draining or overloaded", details={"retry_after_s": retry})
        if self.inflight >= s.max_inflight_requests:
            raise ServiceUnavailableError(
                "too many in-flight requests", details={"retry_after_s": s.retry_after_s}
            )
        st = self._tenants.get(tenant)
        if s.tenant_max_inflight is not None and st is not None and st.inflight >= s.tenant_max_inflight:
            raise RateLimitedError(
                "tenant concurrency limit reached",
                details={"retry_after_s": s.retry_after_s, "limit": s.tenant_max_inflight},
            )
        if s.tenant_rows_per_s is not None:
            if st is None:
                st = self._tenants.setdefault(tenant, _TenantState())
            if st.bucket is None:
                burst = s.tenant_burst_rows or max(1, math.ceil(2 * s.tenant_rows_per_s))
                st.bucket = TokenBucket(s.tenant_rows_per_s, burst)
            wait = st.bucket.take(rows)
            if wait > 0:
                raise RateLimitedError("tenant row rate limit exceeded", details={"retry_after_s": round(wait, 3)})
        return None

    async def _compute(self, request: ComputeRequest, tenant: str) -> ComputeResponse:
        ring = RingParams.from_name(request.ring)
        model = self.registry.get(request.fingerprint)
        dims = model.linear_dims
        offload = self.settings.codec_offload_bytes
        bytes_in = request.payload_bytes

        if len(request.items) == 1:  # decode calls: no extra task per item, no per-kind tallies
            it = request.items[0]
            m, n = dims[it.linear]
            rows = int(it.rows)
            if len(it.x_hat) > offload:
                x = await asyncio.to_thread(unpack_ring, it.x_hat, (rows, n), ring)
            else:
                x = unpack_ring(it.x_hat, (rows, n), ring)
            # compact: 32-bit results for K <= 32 on CUDA (packed as-is, half the device-to-host bytes)
            y, bt = await self.batcher.submit_timed(request.fingerprint, ring, it.linear, x, compact=True)
            if y.shape != (rows, m):  # pragma: no cover - kernel contract
                raise PGWError("kernel returned a wrong shape", code=ErrorCode.INTERNAL)
            if rows * m * ring.bytes_per_element > offload:
                data = await asyncio.to_thread(pack_ring, y, ring)
            else:
                data = pack_ring(y, ring)
            return self._single_response(request, tenant, data, bt)
        groups = self._groups(request, model)
        if len(groups) == 1:
            results = await self._run_group(request, groups[0], ring, dims, offload)
        else:
            tasks = [asyncio.ensure_future(self._run_group(request, g, ring, dims, offload)) for g in groups]
            try:
                per_group = await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                raise
            results = [r for g in per_group for r in g]
        results.sort(key=lambda r: r[0])
        out = tuple(r[1] for r in results)
        bytes_out = sum(len(r.y_hat) for r in out)
        kinds: dict[str, int] = {}
        for it in request.items:
            kind = it.linear.split(".", 1)[1]
            kinds[kind] = kinds.get(kind, 0) + int(it.rows)
        for kind, rows in kinds.items():
            self.metrics.rows(tenant, kind, rows)
        self.metrics.bytes_io(tenant, bytes_in, bytes_out)
        timing = ServerTiming(
            queue_ms=round(max((r[2].queue_ms for r in results), default=0.0), 3),
            compute_ms=round(sum(r[2].compute_ms for r in results if r[3]), 3),
            batch_rows=max((r[2].batch_rows for r in results), default=0),
        )
        return ComputeResponse(request.request_id, out, timing)

    def _groups(self, request: ComputeRequest, model: Any) -> list[list[int]]:
        """Item indices per kernel call: same-linear items grouped (first-appearance order)."""
        if self.group_min_rows <= 0:  # grouping off
            return [[i] for i in range(len(request.items))]
        by_linear: dict[str, list[int]] = {}
        for i, it in enumerate(request.items):
            by_linear.setdefault(it.linear, []).append(i)
        groups: list[list[int]] = []
        for linear, idx in by_linear.items():
            if len(idx) == 1:
                groups.append(idx)
                continue
            rows = sum(int(request.items[i].rows) for i in idx)
            if rows >= self.group_min_rows or not self._has_fused_gemv(model, linear):
                groups.append(idx)
            else:  # e.g. two one-row items on a fused-GEMV kernel: two GEMV calls
                groups.extend([i] for i in idx)
        return groups

    @staticmethod
    def _has_fused_gemv(model: Any, linear: str) -> bool:
        try:
            return bool(getattr(model.kernel(linear), "triton_gemv", False))
        except Exception:  # pragma: no cover - the linear was validated by the router
            return False

    async def _run_group(
        self, request: ComputeRequest, idx: list[int], ring: RingParams,
        dims: Mapping[str, tuple[int, int]], offload: int,
    ) -> list[tuple[int, ComputeResult, BatchTiming, bool]]:
        """Compute the items ``idx`` (all for one linear) with one batcher submission.

        Returns ``(item index, result, timing, first of its group)`` per item.
        """
        linear = request.items[idx[0]].linear
        m, n = dims[linear]
        xs: list[torch.Tensor] = []
        for i in idx:
            it = request.items[i]
            rows = int(it.rows)
            if len(it.x_hat) > offload:
                xs.append(await asyncio.to_thread(unpack_ring, it.x_hat, (rows, n), ring))
            else:
                xs.append(unpack_ring(it.x_hat, (rows, n), ring))
        if len(xs) == 1:
            y0, timing = await self.batcher.submit_timed(request.fingerprint, ring, linear, xs[0], compact=True)
            ys = [y0]
        else:
            ys, timing = await self.batcher.submit_parts(request.fingerprint, ring, linear, xs, compact=True)
        out: list[tuple[int, ComputeResult, BatchTiming, bool]] = []
        for j, (i, y) in enumerate(zip(idx, ys, strict=True)):
            rows = int(request.items[i].rows)
            if y.shape != (rows, m):  # pragma: no cover - kernel contract
                raise PGWError("kernel returned a wrong shape", code=ErrorCode.INTERNAL)
            if rows * m * ring.bytes_per_element > offload:
                data = await asyncio.to_thread(pack_ring, y, ring)
            else:
                data = pack_ring(y, ring)
            out.append((i, ComputeResult(linear, rows, data), timing, j == 0))
        return out

    async def wait_idle(self, timeout: float | None) -> bool:
        """Wait until no request is in flight; True if idle within ``timeout``."""
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False
