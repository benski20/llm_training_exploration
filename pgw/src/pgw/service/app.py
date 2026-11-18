"""ASGI application of the compute service.

Routes (SPEC.md section 6.2):
    POST /pgw/v1/compute   msgpack ComputeRequest -> ComputeResponse
    GET  /pgw/v1/models    ModelsResponse
    GET  /healthz          liveness;  GET /readyz  readiness (packs loaded, device OK, not draining)
    GET  /metrics          Prometheus (optionally only on a separate port, see :mod:`pgw.service.server`)
Run with hypercorn for HTTP/2 (``pgw service serve`` / ``python -m pgw.service``).

Lifecycle: on startup the app loads every configured pack (``verify="full"``, linears only) in a
worker thread, checks the devices and starts the batch scheduler; ``/readyz`` turns 200 only after
that. On shutdown (SIGTERM via the server) it first reports not-ready, waits for in-flight requests
(``drain_timeout_s``), drains the scheduler and exits. The app never logs request payloads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from pgw.core.fingerprint import short_fingerprint
from pgw.service.auth import TenantAuthenticator
from pgw.service.batcher import BatchScheduler
from pgw.service.config import ServiceSettings
from pgw.service.handler import ServiceHandler
from pgw.service.metrics import ServiceMetrics
from pgw.service.registry import ModelRegistry

__all__ = ["ServiceState", "create_app", "create_metrics_app"]

log = logging.getLogger(__name__)

_DEVICE_CHECK_TTL_S = 10.0

Scope = MutableMapping[str, Any]
ASGIApp = Callable[[Scope, Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]], Awaitable[None]]


@dataclass
class ServiceState:
    """Runtime state shared by the routes (``app.state.pgw``)."""

    settings: ServiceSettings
    registry: ModelRegistry
    metrics: ServiceMetrics
    batcher: BatchScheduler
    handler: ServiceHandler
    loaded: bool = False
    load_error: str | None = None
    started_at: float = field(default_factory=time.time)
    _device_ok: dict[str, bool] = field(default_factory=dict)
    _device_checked_at: float = 0.0

    @property
    def draining(self) -> bool:
        """True once shutdown started."""
        return self.handler.draining

    def begin_drain(self) -> None:
        """Stop admitting new compute requests (``/readyz`` -> 503)."""
        if not self.handler.draining:
            log.info("compute service draining: no new requests admitted")
        self.handler.draining = True

    async def device_ok(self) -> dict[str, bool]:
        """Cached device probe (re-run at most every few seconds, off the event loop)."""
        now = time.monotonic()
        if not self._device_ok or now - self._device_checked_at > _DEVICE_CHECK_TTL_S:
            self._device_ok = await asyncio.to_thread(self.registry.check_devices)
            self._device_checked_at = now
        return self._device_ok

    async def readiness(self) -> tuple[bool, dict[str, Any]]:
        """``(ready, details)`` for ``/readyz``."""
        details: dict[str, Any] = {
            "loaded": self.loaded,
            "draining": self.draining,
            "models": [
                {"name": m.name, "fp": m.short_fingerprint, "rings": list(m.rings), "devices": m.devices()}
                for m in self.registry.models()
            ],
            "inflight": self.handler.inflight,
            "queued_rows": self.batcher.queued_rows,
        }
        if self.load_error:
            details["error"] = self.load_error
        ready = self.loaded and not self.draining and len(self.registry) > 0 and self.batcher.running
        if self.loaded:
            dev = await self.device_ok()
            details["devices"] = dev
            ready = ready and bool(dev) and all(dev.values())
        return ready, details


def _client_cert_header_middleware(app: ASGIApp, header: str) -> ASGIApp:
    """Copy a TLS-terminating proxy's verified client-cert subject header into the ASGI TLS
    extension (``scope["extensions"]["tls"]["client_cert_name"]``) where the wire router reads it.

    Only enable when every request reaches the service through that proxy; otherwise clients
    could forge the header.
    """
    hname = header.lower().encode("latin-1")

    async def mw(scope: Scope, receive: Callable[[], Awaitable[Any]], send: Callable[[Any], Awaitable[None]]) -> None:
        if scope.get("type") == "http":
            values = [v for k, v in scope.get("headers") or () if k.lower() == hname]
            value: str | None = None
            if len(values) == 1:
                value = values[0].decode("latin-1").strip()
            elif len(values) > 1:
                # A proxy that appends instead of overwriting would let clients smuggle a subject:
                # ambiguous identity is no identity.
                log.warning("ignoring %d copies of the trusted client-cert header", len(values))
            scope = dict(scope)
            ext = dict(scope.get("extensions") or {})
            tls = dict(ext.get("tls") or {})
            tls.pop("client_cert_name", None)
            tls.pop("client_cert_subject", None)
            if value:
                tls["client_cert_name"] = value
            ext["tls"] = tls
            scope["extensions"] = ext
        await app(scope, receive, send)

    return mw


def _build_state(
    settings: ServiceSettings,
    registry: ModelRegistry | None,
    authenticator: TenantAuthenticator | None,
    metrics: ServiceMetrics | None,
) -> ServiceState:
    s = settings
    if registry is None:
        registry = ModelRegistry(
            s.device, s.backend, devices=s.devices or None, placement=s.placement, rings=s.rings,
            verify=s.verify_packs, cache_float=s.cache_float_weights, warmup=s.warmup,
        )
    has_creds = s.tokens_file is not None or s.mtls_tenants_file is not None
    if authenticator is None and (has_creds or not s.allow_unauthenticated):
        authenticator = TenantAuthenticator(
            s.tokens_file, allow_unauthenticated=s.allow_unauthenticated, mtls_tenants_file=s.mtls_tenants_file
        )
    metrics = metrics or ServiceMetrics(enabled=s.metrics_enabled)
    batcher = BatchScheduler(
        registry, batch_window_ms=s.batch_window_ms, max_batch_rows=s.max_batch_rows, cpu_workers=s.cpu_workers,
        gpu_workers_per_device=s.gpu_workers_per_device, max_queued_rows=s.max_queued_rows, metrics=metrics,
    )
    handler = ServiceHandler(s, registry, batcher, authenticator, metrics)
    return ServiceState(settings=s, registry=registry, metrics=metrics, batcher=batcher, handler=handler)


def create_app(
    settings: ServiceSettings | None = None,
    *,
    registry: ModelRegistry | None = None,
    authenticator: TenantAuthenticator | None = None,
    metrics: ServiceMetrics | None = None,
) -> Any:
    """Build the FastAPI app (loads packs on startup, starts the batcher).

    Args:
        settings: Service settings (default: from the environment).
        registry: Pre-built registry (tests); configured ``packs`` are still loaded into it.
        authenticator: Pre-built authenticator (default: from ``tokens_file`` / ``mtls_tenants_file``).
        metrics: Pre-built metrics.

    Returns:
        A FastAPI application; ``app.state.pgw`` is the :class:`ServiceState`.

    Raises:
        ConfigError: no credentials configured while unauthenticated access is disabled, etc.
    """
    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse, Response

    from pgw.wire.server import WireLimits, create_router

    s = settings or ServiceSettings()
    state = _build_state(s, registry, authenticator, metrics)

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        t0 = time.perf_counter()
        try:
            for p in s.packs:
                m = await asyncio.to_thread(state.registry.load, p)
                log.info("serving %s fp=%s", m.name, short_fingerprint(m.fingerprint))
        except Exception as exc:
            state.load_error = f"{type(exc).__name__}: {getattr(exc, 'message', str(exc))}"
            log.error("failed to load model packs: %s", state.load_error)
            raise
        await state.batcher.start()
        await state.device_ok()
        state.loaded = True
        log.info(
            "compute service ready: %d model(s), devices=%s, startup %.2fs",
            len(state.registry), [d["device"] for d in state.registry.device_summary()], time.perf_counter() - t0,
        )
        try:
            yield
        finally:
            await shutdown(state)

    app = FastAPI(
        title="PGW compute service", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None,
    )
    app.state.pgw = state
    limits = WireLimits(
        max_request_bytes=s.max_request_bytes, max_items=s.max_items, max_rows_per_item=s.max_rows_per_item
    )
    wire_router = create_router(state.handler, limits=limits, max_deadline_s=s.max_deadline_s,
                                max_inflight_bytes=max(s.max_inflight_bytes, s.max_request_bytes))
    app.include_router(wire_router)
    from pgw.hemask.server import add_maskgen_route

    # Encrypted mask generation: W8 times MLWE ciphertexts (the provider never sees the masks).
    add_maskgen_route(app, state.handler)
    from pgw.wire.server import ComputeFastPath

    # Innermost middleware: compute requests skip routing and framework middleware (hot path). Any
    # middleware added below (e.g. the trusted client-cert header) still runs first.
    app.add_middleware(ComputeFastPath, endpoint=getattr(wire_router, "pgw_compute_endpoint", None))

    async def healthz(request: Request) -> Response:
        return PlainTextResponse("ok")

    async def readyz(request: Request) -> Response:
        ready, details = await state.readiness()
        return JSONResponse({"ready": ready, **details}, status_code=200 if ready else 503)

    async def metrics_route(request: Request) -> Response:
        body, ctype = state.metrics.render()
        return Response(content=body, media_type=ctype.split(";")[0], headers={"Content-Type": ctype})

    app.add_route("/healthz", healthz, methods=["GET"], include_in_schema=False)
    app.add_route("/readyz", readyz, methods=["GET"], include_in_schema=False)
    if s.metrics_on_main_port and s.metrics_enabled:
        app.add_route("/metrics", metrics_route, methods=["GET"], include_in_schema=False)

    if s.trusted_client_cert_header:
        app.add_middleware(_HeaderToTLSExt, header=s.trusted_client_cert_header)
    return app


class _HeaderToTLSExt:
    """Starlette-compatible middleware class wrapping :func:`_client_cert_header_middleware`."""

    def __init__(self, app: ASGIApp, header: str):
        self._mw = _client_cert_header_middleware(app, header)

    async def __call__(self, scope: Scope, receive: Callable[[], Awaitable[Any]],
                       send: Callable[[Any], Awaitable[None]]) -> None:
        await self._mw(scope, receive, send)


async def shutdown(state: ServiceState) -> None:
    """Graceful drain: refuse new work, wait for in-flight requests, drain the scheduler."""
    state.begin_drain()
    timeout = state.settings.drain_timeout_s
    t0 = time.monotonic()
    if not await state.handler.wait_idle(timeout):
        log.warning("drain timeout: %d request(s) still in flight", state.handler.inflight)
    remaining = max(0.0, timeout - (time.monotonic() - t0))
    await state.batcher.stop(drain=True, timeout=remaining)
    log.info("compute service stopped")


def create_metrics_app(state: ServiceState) -> Any:
    """Minimal ASGI app for a separate, non-public port: ``/metrics``, ``/healthz`` and ``/readyz``.

    ``/healthz`` and ``/readyz`` have the same semantics as on the main listener. They exist here so
    that orchestrator probes work when the main port requires client certificates (kubelet probes
    cannot present one).
    """
    from prometheus_client import CONTENT_TYPE_LATEST

    async def app(scope: Scope, receive: Callable[[], Awaitable[Any]], send: Callable[[Any], Awaitable[None]]) -> None:
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path, method = scope.get("path"), scope.get("method")
        if path == "/metrics" and method == "GET":
            body, ctype = state.metrics.render()
            status = 200
        elif path == "/healthz" and method == "GET":
            body, ctype, status = b"ok", "text/plain; charset=utf-8", 200
        elif path == "/readyz" and method == "GET":
            ready, details = await state.readiness()
            body = json.dumps({"ready": ready, **details}, default=str).encode()
            ctype, status = "application/json", 200 if ready else 503
        else:
            body, ctype, status = b"not found\n", "text/plain; charset=utf-8", 404
        ctype = ctype or CONTENT_TYPE_LATEST
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", ctype.encode()), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    return app
