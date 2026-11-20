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
        group_min_rows: Items of one request for the same linear are computed as one group (one
            kernel call) when they have at least this many rows in all, or when the linear has no
            fused one-row kernel. Below it, on a kernel serving one-row calls with the fused Triton
            GEMV, two one-row items run as two GEMV calls, which can beat one digit GEMM on 2 rows
            (T4). 0 turns grouping off (one kernel call per item, as before grouping existed).
            Default :data:`GROUP_MIN_ROWS` (``PGW_SERVICE_GROUP_MIN_ROWS``).
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
        self._maskgen_lock = asyncio.Lock()  # one encrypted GEMM at a time (large, memory-heavy)
        self.group_min_rows = max(0, int(group_min_rows)) if group_min_rows is not None else _group_min_rows_default()

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
        self._admit(tenant, request.total_rows)
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
            self.metrics.inflight(self.inflight)
            if self.inflight == 0:
                self._idle.set()

    async def maskgen(self, request: Any, identity: Any) -> Any:
        """``POST /pgw/v1/maskgen``: ``W8 b mod 2^64`` for MLWE-encrypted masks (see :mod:`pgw.hemask`).

        The provider sees only pseudorandom ciphertexts. Admission and tenant limits count every
        256-column group as 256 rows, like the compute path.
        """
        from pgw.hemask.wire import MaskgenResponse, MaskgenResult, pack_y, unpack_b

        ident: TenantIdentity | None = identity if isinstance(identity, TenantIdentity) else None
        tenant = ident.tenant_id if ident is not None else "anonymous"
        model = self.registry.get(request.fingerprint)
        dims = model.linear_dims
        for it in request.items:
            if it.linear not in dims:
                raise ProtocolError(f"unknown linear {it.linear!r}", details={"linear": it.linear})
            n = dims[it.linear][1]
            want = it.b_bytes_per_element * n * it.cols
            if len(it.b) != want:
                raise ProtocolError(f"maskgen {it.linear}: b has {len(it.b)} bytes, expected {want}")
        self._admit(tenant, sum(it.cols for it in request.items))
        self.inflight += 1
        self._idle.clear()
        try:
            out = []
            for it in request.items:
                kern = model.kernel(it.linear)
                b = await asyncio.to_thread(unpack_b, it.b, (kern.in_features, it.cols), it.q_bits)
                async with self._maskgen_lock:
                    y = await asyncio.to_thread(kern.mulmod64, b, it.q_bits // 8)
                out.append(MaskgenResult(it.linear, it.cols,
                                         await asyncio.to_thread(pack_y, y, it.drop_bits, it.q_bits)))
            return MaskgenResponse(request.request_id, tuple(out))
        finally:
            self.inflight -= 1
            if self.inflight == 0:
                self._idle.set()

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
        if self.inflight >= s.max_inflight_requests:
            raise ServiceUnavailableError(
                "too many in-flight requests", details={"retry_after_s": s.retry_after_s}
            )

    # ---- internals -----------------------------------------------------------------------
    def _admit(self, tenant: str, rows: int) -> None:
        s = self.settings
        if self.draining:
            raise ServiceUnavailableError("service is draining", details={"retry_after_s": s.retry_after_s})
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

    async def _compute(self, request: ComputeRequest, tenant: str) -> ComputeResponse:
        ring = RingParams.from_name(request.ring)
        model = self.registry.get(request.fingerprint)
        dims = model.linear_dims
        offload = self.settings.codec_offload_bytes
        bytes_in = request.payload_bytes

        async def one(i: int) -> tuple[ComputeResult, BatchTiming]:
            it = request.items[i]
            m, n = dims[it.linear]
            rows = int(it.rows)
            if len(it.x_hat) > offload:
                x = await asyncio.to_thread(unpack_ring, it.x_hat, (rows, n), ring)
            else:
                x = unpack_ring(it.x_hat, (rows, n), ring)
            # compact: 32-bit results for K <= 32 on CUDA (packed as-is, half the device-to-host bytes)
            y, timing = await self.batcher.submit_timed(request.fingerprint, ring, it.linear, x, compact=True)
            if y.shape != (rows, m):  # pragma: no cover - kernel contract
                raise PGWError("kernel returned a wrong shape", code=ErrorCode.INTERNAL)
            if rows * m * ring.bytes_per_element > offload:
                data = await asyncio.to_thread(pack_ring, y, ring)
            else:
                data = pack_ring(y, ring)
            return ComputeResult(it.linear, rows, data), timing

        if len(request.items) == 1:  # decode calls: no extra task per item, no per-kind tallies
            res, bt = await one(0)
            it0 = request.items[0]
            self.metrics.rows(tenant, it0.linear.split(".", 1)[1], int(it0.rows))
            self.metrics.bytes_io(tenant, bytes_in, len(res.y_hat))
            return ComputeResponse(request.request_id, (res,), ServerTiming(
                queue_ms=round(bt.queue_ms, 3), compute_ms=round(bt.compute_ms, 3), batch_rows=bt.batch_rows))
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
