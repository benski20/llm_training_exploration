"""ASGI application of the compute service.

Routes (SPEC.md section 6.2):
    POST /pgw/v1/compute   msgpack ComputeRequest -> ComputeResponse
    POST /pgw/v1/lane/open JSON {fingerprint} -> trip-lane ticket and tables (when ``lane`` is on;
                           see :mod:`pgw.service.lane`)
    GET  /pgw/v1/models    ModelsResponse
    GET  /healthz          liveness;  GET /readyz  readiness (packs loaded, device OK, not draining)
    GET  /metrics          Prometheus (optionally only on a separate port, see :mod:`pgw.service.server`)
    GET/POST/DELETE /pgw/v1/provision/...   client provisioning (when ``provision`` is set; see
                           :mod:`pgw.service.provision`)
    GET  /pgw/v1/enroll/ca, POST /pgw/v1/enroll   gateway enrollment (when ``enrollment_file`` is set;
                           see :mod:`pgw.enroll.service`)
    POST /pgw/v1/sdk/enroll, GET /pgw/v1/sdk/usage, POST /pgw/v1/sdk/report   SDK keys, usage and the
                           content-free request log (when ``accounts_db`` is set; admin API under /pgw/v1/admin
                           with ``accounts_admin_api``; see :mod:`pgw.service.accounts`)
    GET  /platform/...     the developer console (``PGW_PLATFORM_ENABLED=1``; :mod:`pgw.platform`)
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
    #: the native trip lane (``lane != "off"``), started with the batcher
    lane: Any = None
    #: :class:`pgw.service.provision.ProvisionService` (routes mounted always; they answer only with ``provision``)
    provision: Any = None
    #: request-body bytes buffered at once, shared by the compute routes, ``/pgw/v1/maskgen`` and the bulk
    #: maskgen channel (``max_inflight_bytes``, at least one request's ``max_request_bytes``)
    body_budget: Any = None
    #: SDK keys, usage metering and quotas (``accounts_db``; :mod:`pgw.service.accounts`)
    accounts: Any = None

    @property
    def draining(self) -> bool:
        """True once shutdown started."""
        return self.handler.draining

    def begin_drain(self) -> None:
        """Stop admitting new compute requests (``/readyz`` -> 503)."""
        if not self.handler.draining:
            log.info("compute service draining: no new requests admitted")
        self.handler.draining = True
        if self.lane is not None:
            self.lane.set_draining()

    async def device_ok(self) -> dict[str, bool]:
        """Cached device probe (re-run at most every few seconds, off the event loop)."""
        now = time.monotonic()
        if not self._device_ok or now - self._device_checked_at > _DEVICE_CHECK_TTL_S:
            self._device_ok = await asyncio.to_thread(self.registry.check_devices)
            self._device_checked_at = now
        return self._device_ok

    async def readiness(self, *, public: bool = False) -> tuple[bool, dict[str, Any]]:
        """``(ready, details)`` for ``/readyz``.

        ``public`` (the main listener, reachable by every tenant and without credentials): no load figures
        (in-flight requests, queued rows, lane connections). Polled often, they show when other tenants send work
        and how many rows (security review 2026-11, S5); the metrics port keeps them."""
        details: dict[str, Any] = {
            "loaded": self.loaded,
            "draining": self.draining,
            "models": [
                {"name": m.name, "fp": m.short_fingerprint, "rings": list(m.rings), "devices": m.devices(),
                 **({"masked_head": True, "base_fp": m.base_fingerprint[:12]} if m.base_fingerprint else {})}
                for m in self.registry.models()
            ],
            "inflight": self.handler.inflight,
            "queued_rows": self.batcher.queued_rows,
        }
        prog = getattr(self.handler, "maskgen_progress", None)
        if prog is not None:
            # public: one coarse bit (a wedged device answers /readyz but finishes no maskgen item); the numbers
            # only on the operator's metrics port (tenants see their own at /pgw/v1/maskgen/status)
            details["maskgen"] = prog.public_state()
            if not public:
                details["maskgen_progress"] = prog.operator_view()
        if self.load_error:
            details["error"] = self.load_error
        if self.lane is not None:
            details["lane"] = self.lane.status()
        if public:
            details.pop("inflight", None)
            details.pop("queued_rows", None)
            if isinstance(details.get("lane"), dict):
                details["lane"] = {k: v for k, v in details["lane"].items() if k != "connections"}
        ready = self.loaded and not self.draining and len(self.registry) > 0 and self.batcher.running
        if self.loaded:
            dev = await self.device_ok()
            details["devices"] = dev
            ready = ready and bool(dev) and all(dev.values())
        return ready, details


def _client_cert_header_middleware(app: ASGIApp, header: str, pem_header: str | None = None) -> ASGIApp:
    """Copy a TLS-terminating proxy's verified client-cert subject header into the ASGI TLS
    extension (``scope["extensions"]["tls"]["client_cert_name"]``) where the wire router reads it,
    and, with ``pem_header``, the certificate itself (URL-encoded PEM) into ``client_cert_chain``.

    Only enable when every request reaches the service through that proxy; otherwise clients
    could forge the header. Anything already in those scope keys is dropped.
    """
    hname = header.lower().encode("latin-1")
    pname = pem_header.lower().encode("latin-1") if pem_header else None

    def single(scope: Scope, name: bytes) -> str | None:
        values = [v for k, v in scope.get("headers") or () if k.lower() == name]
        if len(values) == 1:
            return values[0].decode("latin-1").strip() or None
        if len(values) > 1:
            # A proxy that appends instead of overwriting would let clients smuggle a subject:
            # ambiguous identity is no identity.
            log.warning("ignoring %d copies of the trusted client-cert header", len(values))
        return None

    async def mw(scope: Scope, receive: Callable[[], Awaitable[Any]], send: Callable[[Any], Awaitable[None]]) -> None:
        if scope.get("type") in ("http", "websocket"):
            value = single(scope, hname)
            pem: str | None = None
            if value and pname is not None:
                from urllib.parse import unquote

                raw = single(scope, pname)
                pem = unquote(raw) if raw else None
            scope = dict(scope)
            ext = dict(scope.get("extensions") or {})
            tls = dict(ext.get("tls") or {})
            tls.pop("client_cert_name", None)
            tls.pop("client_cert_subject", None)
            tls.pop("client_cert_chain", None)
            if value:
                tls["client_cert_name"] = value
                if pem:
                    tls["client_cert_chain"] = [pem]
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
            cuda_graphs=s.cuda_graphs, cuda_graph_buckets=s.cuda_graph_buckets, triton_gemv=s.triton_gemv,
            tp_devices=s.tp_devices or None,
            masked_head=s.masked_head,
        )
    has_creds = s.tokens_file is not None or s.mtls_tenants_file is not None
    if authenticator is None and (has_creds or not s.allow_unauthenticated):
        authenticator = TenantAuthenticator(
            s.tokens_file, allow_unauthenticated=s.allow_unauthenticated, mtls_tenants_file=s.mtls_tenants_file,
            allow_empty=s.enrollment_file is not None or s.accounts_db is not None,
            require_client_cert=s.compute_requires_client_cert,
            reload_interval_s=s.tokens_reload_interval_s,
        )
    metrics = metrics or ServiceMetrics(enabled=s.metrics_enabled)
    batcher = BatchScheduler(
        registry, batch_window_ms=s.batch_window_ms, max_batch_rows=s.max_batch_rows, cpu_workers=s.cpu_workers,
        gpu_workers_per_device=s.gpu_workers_per_device, max_queued_rows=s.max_queued_rows, metrics=metrics,
        inline_max_rows=s.batch_inline_max_rows,
    )
    handler = ServiceHandler(s, registry, batcher, authenticator, metrics)
    state = ServiceState(settings=s, registry=registry, metrics=metrics, batcher=batcher, handler=handler)
    from pgw.service.lane import LaneManager

    state.lane = LaneManager.from_settings(s, state)
    return state


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
        if state.accounts is not None:
            state.accounts.start()
        await state.device_ok()
        if state.lane is not None:
            await state.lane.start()
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
        max_request_bytes=s.max_request_bytes, max_items=s.max_items, max_rows_per_item=s.max_rows_per_item,
        stream=s.stream_channel,
    )
    from pgw.wire.server import ByteBudget

    state.body_budget = ByteBudget(max(s.max_inflight_bytes, s.max_request_bytes))
    wire_router = create_router(state.handler, limits=limits, max_deadline_s=s.max_deadline_s,
                                body_budget=state.body_budget)
    app.include_router(wire_router)
    from pgw.service.lane import add_lane_routes

    add_lane_routes(app, state)
    from pgw.hemask.server import add_maskgen_route

    # Encrypted mask generation: W8 times MLWE ciphertexts (the provider never sees the masks).
    # The same body budget as the compute routes: a maskgen body (up to 512 MiB) counts against the service's
    # max_inflight_bytes (a request larger than the budget alone is refused with 'unavailable').
    add_maskgen_route(app, state.handler, body_budget=state.body_budget)
    from pgw.service.provision import add_provision_routes

    # Client provisioning (public artifacts and pks-v3 hint operands; answers 'unavailable' unless s.provision).
    add_provision_routes(app, state)
    from pgw.wire.server import ComputeFastPath

    # Innermost middleware: compute requests skip routing and framework middleware (hot path). Any
    # middleware added below (e.g. the trusted client-cert header) still runs first.
    app.add_middleware(ComputeFastPath, endpoint=getattr(wire_router, "pgw_compute_endpoint", None),
                       stream=getattr(wire_router, "pgw_stream_app", None))

    async def healthz(request: Request) -> Response:
        return PlainTextResponse("ok")

    async def readyz(request: Request) -> Response:
        ready, details = await state.readiness(public=True)
        return JSONResponse({"ready": ready, **details}, status_code=200 if ready else 503)

    async def metrics_route(request: Request) -> Response:
        body, ctype = state.metrics.render()
        return Response(content=body, media_type=ctype.split(";")[0], headers={"Content-Type": ctype})

    if s.enrollment_file is not None or s.accounts_db is not None:
        from pgw.enroll.service import add_enroll_routes

        def _served() -> list[dict[str, Any]]:
            # enrollment names packs by their own (manifest) fingerprint: a masked-head entry is listed as its base
            # pack (the gateway derives the head fingerprint itself), once
            out: dict[str, dict[str, Any]] = {}
            for m in state.registry.models():
                fp = m.base_fingerprint or m.fingerprint
                e = out.setdefault(fp, {"name": m.name, "fingerprint": fp, "rings": list(m.rings),
                                        "pack_path": m.pack_path, "masked_head": False})
                e["masked_head"] = bool(e["masked_head"] or m.base_fingerprint)
            return list(out.values())

        enroll_svc = add_enroll_routes(app, s, state.handler.authenticator, _served)
        if s.accounts_db is not None:
            from pgw.enroll.service import ca_fingerprint
            from pgw.service.accounts import Accounts, AccountsStore, add_admin_routes, add_sdk_routes

            acc = Accounts(AccountsStore(s.accounts_db), s, registry=state.registry,
                           authenticator=state.handler.authenticator)
            acc.ca_sha256 = ca_fingerprint(enroll_svc.ca_pem) if enroll_svc.ca_pem else None

            def _lane_hint() -> dict[str, Any] | None:
                # the SDK uses the TCP+TLS trip lane when enrollment names it (port as clients reach it)
                lane = state.lane
                if lane is None or getattr(lane, "transport", None) != "tcp" or getattr(lane, "svc", None) is None:
                    return None
                return {"transport": "tcp", "port": int(s.lane_advertised_port or lane.svc.port)}

            acc.lane_hint = _lane_hint
            state.accounts = acc
            state.handler.accounts = acc
            if state.handler.authenticator is not None:
                state.handler.authenticator.tenant_gate = acc
            add_sdk_routes(app, acc, enroll_svc)
            from pgw.service.accounts import add_report_route

            add_report_route(app, acc, state.handler)
            if s.accounts_admin_api:
                add_admin_routes(app, acc)
    _mount_platform(app, state)

    app.add_route("/healthz", healthz, methods=["GET"], include_in_schema=False)
    app.add_route("/readyz", readyz, methods=["GET"], include_in_schema=False)
    if s.metrics_on_main_port and s.metrics_enabled:
        app.add_route("/metrics", metrics_route, methods=["GET"], include_in_schema=False)

    if s.trusted_client_cert_header:
        app.add_middleware(_HeaderToTLSExt, header=s.trusted_client_cert_header,
                           pem_header=s.trusted_client_cert_pem_header)
    return app


def _mount_platform(app: Any, state: ServiceState) -> None:
    """The developer console (:mod:`pgw.platform`) at ``/platform`` when ``PGW_PLATFORM_ENABLED=1``: on this
    service's own accounts (the store behind the admin API), or on another provider's admin API with
    ``PGW_PLATFORM_ACCOUNTS_URL`` (+ ``_TOKEN_FILE``, ``_CA_FILE``)."""
    from pgw.platform import PlatformSettings

    ps = PlatformSettings()
    if not ps.enabled:
        return
    from pgw.core.errors import ConfigError
    from pgw.platform import HttpAccountsClient, mount_platform

    client: Any
    if ps.accounts_url:
        if ps.accounts_token_file is None:
            raise ConfigError("PGW_PLATFORM_ACCOUNTS_URL needs PGW_PLATFORM_ACCOUNTS_TOKEN_FILE (a global admin token)")
        token = ps.accounts_token_file.read_text(encoding="utf-8").strip()
        client = HttpAccountsClient(ps.accounts_url, token, sso_org=ps.oidc_org,
                                    verify=str(ps.accounts_ca_file) if ps.accounts_ca_file else True)
    elif state.accounts is not None:
        from pgw.service.platform_accounts import ServiceAccountsClient

        client = ServiceAccountsClient(state.accounts, sso_org=ps.oidc_org)
    else:
        raise ConfigError("PGW_PLATFORM_ENABLED needs accounts (PGW_SERVICE_ACCOUNTS_DB) or PGW_PLATFORM_ACCOUNTS_URL")
    mount_platform(app, client, ps)
    log.info("developer console mounted at %s", "/" + ps.path_prefix.strip("/"))


class _HeaderToTLSExt:
    """Starlette-compatible middleware class wrapping :func:`_client_cert_header_middleware`."""

    def __init__(self, app: ASGIApp, header: str, pem_header: str | None = None):
        self._mw = _client_cert_header_middleware(app, header, pem_header)

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
    if state.lane is not None:
        # lane calls finish (graph replays need no event loop) before the listener stops
        await state.lane.stop(max(0.5, timeout - (time.monotonic() - t0)))
    remaining = max(0.0, timeout - (time.monotonic() - t0))
    await state.batcher.stop(drain=True, timeout=remaining)
    if state.accounts is not None:
        await asyncio.to_thread(state.accounts.stop)  # flush metered usage
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
