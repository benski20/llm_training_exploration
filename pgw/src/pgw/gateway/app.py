"""ASGI application of the gateway (OpenAI-compatible API, admin and ops endpoints).

Public (API key, scope ``inference``): ``POST /v1/chat/completions``, ``POST /v1/completions``
(``stream=true`` gives SSE ``data:`` chunks ending with ``data: [DONE]``), ``GET /v1/models``,
``GET /v1/models/{name}``.
Admin (admin token, an API key with scope ``admin``, or a console session): ``GET /admin/status``,
``GET /admin/masks``, ``GET /admin/keys``, ``POST /admin/keys``, ``DELETE /admin/keys/{key_id}``,
``POST /admin/models/{name}/reload``, ``POST /admin/models/{name}/reset``, ``GET /admin/overview``,
``GET /admin/usage``, ``GET /admin/audit``.
Ops (no auth unless ``metrics_require_admin``): ``/healthz``, ``/readyz``, ``/metrics``, ``/dashboard``.
Console (:mod:`pgw.console`): ``/console/`` (static app) and ``/console/api/*``.

Errors use the OpenAI shape ``{"error": {"message", "type", "code", "param"}}`` with ``code`` set
to a :class:`pgw.core.ErrorCode` value. Nothing here logs prompts or completions; every request
leaves one content-free audit record.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.responses import StreamingResponse

from pgw.console.sessions import COOKIE_NAME, CSRF_HEADER, ConsoleSessions, LoginRateLimiter
from pgw.core.errors import AuthError, ErrorCode, PGWError, RateLimitedError
from pgw.core.http_body import read_limited
from pgw.engine import SamplingParams, TokenEvent
from pgw.gateway.audit import AuditLog
from pgw.gateway.auth import ANONYMOUS_KEY, ApiKey, ApiKeyStore, KeyNotFoundError, QuotaTracker, RateLimiter
from pgw.gateway.config import GatewaySettings
from pgw.gateway.egress import disable_library_telemetry
from pgw.gateway.metrics import GatewayMetrics
from pgw.gateway.openai_render import TokenText as _TokenText
from pgw.gateway.openai_render import chat_logprobs as _chat_logprobs
from pgw.gateway.openai_render import chat_messages as _chat_messages
from pgw.gateway.openai_render import sampling_params as _sampling
from pgw.gateway.openai_render import stop_ids as _stop_ids
from pgw.gateway.openai_render import visible as _visible
from pgw.gateway.openai_schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
)
from pgw.gateway.runtime import ActiveSession, InferenceRuntime, InvalidRequestError, ModelRuntime

__all__ = ["GatewayState", "create_app", "error_body"]

log = logging.getLogger(__name__)

#: Optional client conversation id (``prefix_cache``): narrows the conversation key of KV-prefix reuse below the
#: API key and OpenAI ``user`` (gateway-local; never sent to the compute service).
CONVERSATION_HEADER = "x-pgw-conversation-id"

_OPENAI_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "invalid_request_error",
    413: "invalid_request_error",
    426: "invalid_request_error",
    429: "rate_limit_error",
}


def error_body(exc: PGWError, *, param: str | None = None) -> dict[str, Any]:
    """OpenAI-shaped error body for a :class:`PGWError` (internal errors are not detailed)."""
    status = exc.http_status
    if exc.code == ErrorCode.INTERNAL:
        msg = "internal error"
    else:
        msg = exc.message
    etype = _OPENAI_TYPE.get(status, "server_error" if status >= 500 else "invalid_request_error")
    if exc.code == ErrorCode.RATE_LIMITED and exc.details.get("limit") == "daily_tokens":
        etype = "insufficient_quota"
    return {"error": {"message": msg, "type": etype, "code": str(exc.code), "param": param}}


def _retry_after(exc: PGWError) -> str | None:
    v = exc.details.get("retry_after_s")
    if isinstance(v, int | float) and v > 0:
        return str(max(1, int(v + 0.999)))
    return None


@dataclass
class _KeyLease:
    """Per-key admission for one API request: concurrency, rate limit and quota reservation."""

    state: GatewayState
    key: ApiKey
    reserved: int
    done: bool = False

    def finish(self, actual_tokens: int, masks_reserved: int = 0) -> None:
        """Settle rate limit and quota.

        The charge is ``max(tokens, burn_weight x mask slots reserved)``: reserved slots are
        destroyed whether used or not, so a key that asks for huge ``max_tokens`` and stops early
        pays for the masks it burned (the store is shared by every tenant).
        """
        if self.done:
            return
        self.done = True
        st = self.state
        weight = float(st.settings.mask_burn_weight)
        actual_tokens = max(int(actual_tokens), round(weight * int(masks_reserved)))
        st.rate_limiter.record_tokens(self.key, actual_tokens, reserved=self.reserved)
        st.quotas.settle(self.key, self.reserved, actual_tokens)
        n = st.key_inflight.get(self.key.key_id, 1) - 1
        if n > 0:
            st.key_inflight[self.key.key_id] = n
        else:
            st.key_inflight.pop(self.key.key_id, None)


@dataclass
class GatewayState:
    """Everything the routes share (``app.state.pgw``)."""

    settings: GatewaySettings
    runtime: InferenceRuntime
    keys: ApiKeyStore
    rate_limiter: RateLimiter
    quotas: QuotaTracker
    audit: AuditLog
    metrics: GatewayMetrics
    started_at: float = field(default_factory=time.time)
    key_inflight: dict[str, int] = field(default_factory=dict)
    #: Console sign-in sessions (``None`` when the console is disabled).
    console: ConsoleSessions | None = None
    login_limiter: LoginRateLimiter = field(default_factory=LoginRateLimiter)
    #: Console single sign-on (:class:`pgw.console.oidc.OIDCClient`; ``None`` when not configured).
    oidc: Any = None
    #: Limits SSO sign-in starts per client (bounds the pending sign-in store).
    oidc_start_limiter: LoginRateLimiter = field(default_factory=lambda: LoginRateLimiter(attempts=30))
    #: global cap on SSO callbacks (key "*"): each one costs a token request to the identity provider
    oidc_callback_limiter: LoginRateLimiter = field(default_factory=lambda: LoginRateLimiter(attempts=600))
    audit_reader: Any = None
    #: Streaming API requests since start-up.
    stream_requests: int = 0

    def lease(self, key: ApiKey, est_tokens: int) -> _KeyLease:
        """Admit a request for ``key`` (raises :class:`RateLimitedError`)."""
        limit = key.max_concurrent_requests
        if limit is not None and self.key_inflight.get(key.key_id, 0) >= limit:
            raise RateLimitedError(
                "too many concurrent requests for this API key",
                details={"retry_after_s": 1.0, "limit": "concurrency"},
            )
        self.rate_limiter.check_request(key, est_tokens)
        try:
            self.quotas.reserve(key, est_tokens)
        except PGWError:
            self.rate_limiter.record_tokens(key, 0, reserved=est_tokens)
            raise
        self.key_inflight[key.key_id] = self.key_inflight.get(key.key_id, 0) + 1
        return _KeyLease(self, key, est_tokens)


# =================================================================================================
# helpers
# =================================================================================================
def _sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode() + b"\n\n"


_SSE_DONE = b"data: [DONE]\n\n"


@dataclass
class _Collected:
    text: str = ""
    finish: str | None = None
    tokens: int = 0
    events: list[TokenEvent] = field(default_factory=list)


def _completion_logprobs(events: list[TokenEvent], tt: _TokenText, offset0: int) -> dict[str, Any]:
    tokens, lps, tops, offs = [], [], [], []
    off = offset0
    for ev in events:
        tok = tt(ev.token_id)
        tokens.append(tok)
        lps.append(ev.logprob)
        tops.append({tt(i): lp for i, lp in (ev.top_logprobs or [])} if ev.top_logprobs is not None else None)
        offs.append(off)
        off += len(tok)
    return {"tokens": tokens, "token_logprobs": lps, "top_logprobs": tops, "text_offset": offs}


def _managed_stream(
    body: AsyncIterator[bytes], on_close: Callable[[], Awaitable[None]], headers: dict[str, str]
) -> Any:
    """SSE ``StreamingResponse`` whose ``on_close`` runs on completion, error or client disconnect."""

    class _ManagedStream(StreamingResponse):
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                with contextlib.suppress(Exception):
                    await body.aclose()  # type: ignore[attr-defined]
                await on_close()

    return _ManagedStream(body, media_type="text/event-stream", headers=headers)


#: Body limit of the gateway's small admin JSON routes (security review 2026-11, S13; the API itself has
#: ``max_body_bytes`` through :class:`_BodyLimit`).
_ADMIN_BODY_LIMIT = 64 << 10


class _BodyLimit:
    """Pure-ASGI middleware rejecting request bodies above ``limit`` bytes (413)."""

    def __init__(self, app: Any, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > self.limit:
                        await self._reject(send)
                        return
                except ValueError:
                    pass
        seen = 0
        limit = self.limit
        state = {"too_large": False, "replaced": False}

        async def guarded() -> Any:
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > limit:
                    state["too_large"] = True
                    raise _TooLargeError()
            return msg

        async def guarded_send(msg: Any) -> None:
            # The framework may turn the receive error into its own 400; answer 413 instead.
            if state["too_large"]:
                if msg["type"] == "http.response.start" and not state["replaced"]:
                    state["replaced"] = True
                    await self._reject(send)
                return
            await send(msg)

        try:
            await self.app(scope, guarded, guarded_send)
        except _TooLargeError:
            if not state["replaced"]:
                state["replaced"] = True
                await self._reject(send)

    async def _reject(self, send: Any) -> None:
        body = json.dumps({"error": {"message": "request body too large", "type": "invalid_request_error",
                                     "code": str(ErrorCode.PAYLOAD_TOO_LARGE), "param": None}}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class _TooLargeError(Exception):
    pass


_FALLBACK_DASHBOARD = """<!doctype html><html><head><meta charset="utf-8"><title>PGW gateway</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}
@media (prefers-color-scheme:dark){body{background:#111;color:#eee}}pre{white-space:pre-wrap}</style></head>
<body><h1>PGW gateway</h1><p>Readiness (refreshes every 5 s):</p><pre id="r">loading...</pre>
<script>async function f(){try{const r=await fetch('readyz');document.getElementById('r').textContent=
JSON.stringify(await r.json(),null,2)}catch(e){document.getElementById('r').textContent='unreachable'}}
f();setInterval(f,5000);</script></body></html>"""


def _cert_renewers(runtime: Any) -> list[Any]:
    """One :class:`pgw.enroll.client.CertRenewer` per distinct client certificate of the masked models
    (renewed at 2/3 of its lifetime; the live TLS contexts and compute channels pick up the new pair)."""
    import functools

    from pgw.enroll.client import CertRenewer, renew_client_cert

    out: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for rt in getattr(runtime, "runtimes", {}).values():
        c = getattr(rt, "config", None)
        if (c is None or getattr(c, "mode", None) != "masked" or c.client_cert_file is None
                or c.client_key_file is None or not str(c.service_url or "").startswith("https://")):
            continue
        pair = (str(Path(c.client_cert_file).resolve()), str(Path(c.client_key_file).resolve()))
        if pair in seen:
            continue
        try:
            token = c.resolved_service_token()
        except PGWError as exc:
            log.error("model %s: no client certificate renewal (%s)", c.name, exc.message)
            continue
        if token is None:
            log.error("model %s: no client certificate renewal (renewal needs the service token)", c.name)
            continue
        seen.add(pair)
        out.append(CertRenewer(Path(c.client_cert_file), functools.partial(
            renew_client_cert, str(c.service_url), token.get_secret_value(), Path(c.client_cert_file),
            Path(c.client_key_file), ca_file=c.service_ca_file)))
    return out


def _dashboard_html() -> str:
    try:
        from pgw.observability.dashboard import dashboard_html

        return dashboard_html("PGW gateway")
    except ImportError as exc:  # pragma: no cover - observability is part of the package
        log.warning("dashboard unavailable (%s); serving the minimal fallback page", exc)
        return _FALLBACK_DASHBOARD


# =================================================================================================
# app
# =================================================================================================
def create_app(
    settings: GatewaySettings | None = None,
    *,
    runtime: InferenceRuntime | None = None,
    key_store: ApiKeyStore | None = None,
    rate_limiter: RateLimiter | None = None,
    quotas: QuotaTracker | None = None,
    audit: AuditLog | None = None,
    metrics: GatewayMetrics | None = None,
    mask_store: Any = None,
    transports: dict[str, Any] | None = None,
    client_factory: Any = None,
    console_static_dir: Path | None = None,
    oidc_transport: Any = None,
) -> Any:
    """Build the FastAPI app.

    Public (API key): ``POST /v1/chat/completions``, ``POST /v1/completions``, ``GET /v1/models``,
    ``GET /v1/models/{name}``.
    Admin (admin token, ``admin`` scope, or a console session cookie; viewers GET only, writes need
    the CSRF header): ``GET /admin/status``, ``GET /admin/masks``,
    ``GET /admin/keys``, ``POST /admin/keys``, ``DELETE /admin/keys/{key_id}``,
    ``POST /admin/models/{name}/reload``, ``POST /admin/models/{name}/reset``,
    ``GET /admin/overview``, ``GET /admin/usage``, ``GET /admin/audit``.
    Ops: ``/healthz``, ``/readyz``, ``/metrics``, ``/dashboard``.
    Console (``console_enabled``): ``/console/`` and ``/console/api/*`` (see :mod:`pgw.console`).
    Errors use the OpenAI error shape with ``code`` = :class:`pgw.core.ErrorCode` value.

    All keyword arguments are optional injection points (tests, embedding); by default everything
    is built from ``settings`` (``PGW_GATEWAY_*`` env when ``None``). Model runtimes start in the
    app lifespan; a model that cannot start (pack, fingerprint, service, policy) aborts startup.
    """
    settings = settings or GatewaySettings()
    disable_library_telemetry()
    metrics = metrics or GatewayMetrics()
    if runtime is None:
        runtime = InferenceRuntime(
            settings, mask_store=mask_store, metrics=metrics, client_factory=client_factory, transports=transports
        )
    else:
        settings.validate_policy(list(rt.config for rt in runtime.runtimes.values()))
    state = GatewayState(
        settings=settings,
        runtime=runtime,
        keys=key_store if key_store is not None else ApiKeyStore(settings.api_keys_file),
        rate_limiter=rate_limiter or RateLimiter(settings.requests_per_minute, settings.tokens_per_minute),
        quotas=quotas or QuotaTracker(settings.daily_token_quota, settings.quota_state_file, deferred=True),
        audit=audit or AuditLog(settings.audit_log_file),
        metrics=metrics,
    )
    if settings.console_enabled:
        state.console = ConsoleSessions(
            settings.admin_token.get_secret_value() if settings.admin_token is not None else None,
            settings.console_viewer_tokens_file,
        )
        if settings.console_oidc_enabled:
            from pgw.console.oidc import OIDCClient

            state.oidc = OIDCClient.from_settings(settings, transport=oidc_transport)
    req_counter = metrics.get("pgw_gateway_requests_total")
    runtime.add_safety_listener(
        lambda event, model, reason: state.audit.event(event, action="safety", model=model, reason=reason)
    )

    async def _quota_flusher() -> None:
        while True:
            await asyncio.sleep(settings.quota_flush_interval_s)
            await asyncio.to_thread(state.quotas.flush)

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        state.audit.event("gateway.start", action="start")
        try:
            await runtime.start()
        except BaseException:
            await runtime.stop()
            raise
        loop = asyncio.get_running_loop()
        background = [loop.create_task(_quota_flusher(), name="pgw-quota-flush")]
        background += [loop.create_task(r.run(), name="pgw-cert-renew") for r in _cert_renewers(runtime)]
        try:
            yield
        finally:
            runtime.begin_drain()
            for task in background:
                task.cancel()
            for task in background:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            try:
                await runtime.stop()
            finally:
                await asyncio.to_thread(state.quotas.flush)
                state.audit.event("gateway.stop", action="stop")
                state.audit.close()

    dev = settings.allow_dev_modes
    app = FastAPI(
        title="PGW gateway",
        version="1",
        lifespan=lifespan,
        docs_url="/docs" if dev else None,
        redoc_url=None,
        openapi_url="/openapi.json" if dev else None,
    )
    app.state.pgw = state
    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type"])
    app.add_middleware(_BodyLimit, limit=settings.max_body_bytes)

    # ---- errors -----------------------------------------------------------------------------------
    def _err(exc: PGWError, request_id: str | None = None) -> Any:
        headers = {}
        ra = _retry_after(exc)
        if ra is not None and exc.http_status in (429, 503):
            headers["Retry-After"] = ra
        if request_id:
            headers["x-request-id"] = request_id
        return JSONResponse(error_body(exc), status_code=exc.http_status, headers=headers)

    @app.exception_handler(PGWError)
    async def _pgw_error(request: Request, exc: PGWError) -> Any:
        if exc.code == ErrorCode.INTERNAL:
            log.error("internal error on %s: %s", request.url.path, type(exc).__name__)
        status = 404 if isinstance(exc, KeyNotFoundError) else exc.http_status
        resp = _err(exc)
        resp.status_code = status
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> Any:
        errs = exc.errors()
        first = errs[0] if errs else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or None
        msg = f"invalid request: {first.get('msg', 'validation failed')}"
        body = {"error": {"message": msg, "type": "invalid_request_error", "code": str(ErrorCode.PROTOCOL),
                          "param": loc}}
        return JSONResponse(body, status_code=400)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> Any:
        log.error("unhandled error on %s: %s", request.url.path, type(exc).__name__)
        return JSONResponse(error_body(PGWError("internal error")), status_code=500)

    # ---- auth ---------------------------------------------------------------------------------------
    def _bearer(request: Request) -> str | None:
        h = request.headers.get("authorization")
        if not h:
            return None
        scheme, _, tok = h.partition(" ")
        if scheme.lower() != "bearer" or not tok.strip():
            raise AuthError("Authorization header must be 'Bearer <key>'")
        return tok.strip()

    def _client(request: Request) -> str | None:
        from pgw.core.client_ip import client_ip

        return client_ip(request) if request.client is not None else None

    def _auth_failed(request: Request, exc: PGWError, endpoint: str) -> None:
        state.audit.event(
            "auth.failed", endpoint=endpoint, code=str(exc.code), client=_client(request),
            key_id=str(exc.details.get("key_id")) if exc.details.get("key_id") else None,
        )

    def require_key(request: Request, endpoint: str) -> ApiKey:
        try:
            tok = _bearer(request)
            if tok is None:
                if settings.allow_anonymous and settings.allow_dev_modes:
                    return ANONYMOUS_KEY
                raise AuthError("missing API key (Authorization: Bearer <key>)")
            key = state.keys.verify(tok)
            if not key.has_scope("inference"):
                raise PGWError("API key lacks the 'inference' scope", code=ErrorCode.PERMISSION_DENIED,
                               details={"key_id": key.key_id})
            return key
        except PGWError as exc:
            _auth_failed(request, exc, endpoint)
            req_counter.labels(model="", endpoint=endpoint, status=str(exc.http_status)).inc()
            raise

    def require_admin(request: Request, endpoint: str) -> str:
        """Returns the actor id for the audit log."""
        try:
            tok = _bearer(request)
            if tok is None:
                sid = request.cookies.get(COOKIE_NAME)
                if sid and state.console is not None:
                    return state.console.authorize(sid, request.method, request.headers.get(CSRF_HEADER)).actor
                raise AuthError("missing admin credentials")
            admin = settings.admin_token
            if admin is not None and hmac.compare_digest(tok.encode(), admin.get_secret_value().encode()):
                return "admin-token"
            if tok.startswith("pgw-"):
                key = state.keys.verify(tok)
                if key.has_scope("admin"):
                    return key.key_id
                raise PGWError("API key lacks the 'admin' scope", code=ErrorCode.PERMISSION_DENIED,
                               details={"key_id": key.key_id})
            raise AuthError("invalid admin credentials")
        except PGWError as exc:
            _auth_failed(request, exc, endpoint)
            raise

    def _model_for_key(key: ApiKey, model: str) -> ModelRuntime:
        if not key.allows_model(model):
            raise PGWError(f"model {model!r} does not exist", code=ErrorCode.MODEL_NOT_FOUND,
                           details={"model": model})
        return runtime.get(model)

    # ---- inference core -------------------------------------------------------------------------------
    async def _admit(
        rt: ModelRuntime, key: ApiKey, prompts: list[list[int]], params: SamplingParams, rid: str,
        conversation: str | None = None, prefix_key: str | None = None,
    ) -> tuple[_KeyLease, list[ActiveSession]]:
        est = sum(len(p) + params.max_tokens for p in prompts)
        rt.check_available()
        lease = state.lease(key, est)
        try:
            # all admission slots at once (no hold-and-wait); masks are reserved only afterwards
            sessions = await rt.open_sessions(prompts, params, request_id=rid, conversation=conversation,
                                              prefix_key=prefix_key)
        except BaseException:
            lease.finish(0)
            raise
        return lease, sessions

    async def _collect(sess: ActiveSession, stop_ids: frozenset[int]) -> _Collected:
        out = _Collected()
        parts: list[str] = []
        async for ev in sess.stream():
            out.tokens += 1
            parts.append(ev.text)
            if _visible(ev, stop_ids):
                out.events.append(ev)
            if ev.finish_reason is not None:
                out.finish = str(ev.finish_reason)
        out.text = "".join(parts)
        return out

    def _audit_request(
        endpoint: str, rid: str, key: ApiKey, rt: ModelRuntime | None, status: int, code: str | None,
        t0: float, sessions: list[ActiveSession], stream: bool,
    ) -> None:
        req_counter.labels(model=rt.name if rt else "", endpoint=endpoint, status=str(status)).inc()
        if stream:
            state.stream_requests += 1
        state.audit.event(
            "inference",
            request_id=rid, key_id=key.key_id, model=rt.name if rt else None, endpoint=endpoint, status=status,
            code=code, stream=stream,
            prompt_tokens=sum(len(s.prompt_ids) for s in sessions),
            completion_tokens=sum(s.completion_tokens for s in sessions),
            masks_reserved=sum(s.masks_reserved for s in sessions),
            masks_consumed=sum(s.masks_consumed for s in sessions),
            masks_burned=sum(s.masks_burned for s in sessions),
            prefix_reused=sum(s.prefix_reused for s in sessions) or None,
            queue_ms=round(max((s.queue_s for s in sessions), default=0.0) * 1e3, 1),
            ttft_ms=round(sessions[0].ttft_s * 1e3, 1) if sessions and sessions[0].ttft_s is not None else None,
            finish_reason=",".join(s.finish_reason or "" for s in sessions) if sessions else None,
            duration_ms=round((time.perf_counter() - t0) * 1e3, 1),
            mode=rt.mode if rt else None, ring=rt.ring.name if rt else None,
            prompts=len(sessions) if len(sessions) > 1 else None,
        )

    async def _handle(
        request: Request,
        endpoint: str,
        req: ChatCompletionRequest | CompletionRequest,
        build_prompts: Callable[[ModelRuntime], Awaitable[tuple[list[list[int]], list[str]]]],
        logprobs: int | None,
        render: Callable[..., Any],
        render_stream: Callable[..., AsyncIterator[bytes]],
        key: ApiKey | None = None,
    ) -> Any:
        t0 = time.perf_counter()
        prefix = "chatcmpl-" if endpoint == "chat" else "cmpl-"
        rid = prefix + uuid.uuid4().hex[:24]
        if key is None:
            key = require_key(request, endpoint)
        rt: ModelRuntime | None = None
        sessions: list[ActiveSession] = []
        lease: _KeyLease | None = None
        try:
            rt = _model_for_key(key, req.model)
            prompts, prompt_texts = await build_prompts(rt)
            if req.stream and len(prompts) > 1:
                raise InvalidRequestError("streaming supports a single prompt")
            max_tokens = min(rt.clamp_max_tokens(len(p), req.requested_max_tokens) for p in prompts)
            params = _sampling(req, max_tokens, logprobs)
            # Strict speculation orders the requests of one conversation behind the previous one's dummy
            # passes. The API has no conversation id: requests of one API key (and OpenAI ``user``) count as
            # one conversation (gateway-local key, never sent to the service).
            conversation = f"key:{key.key_id}\x00user:{getattr(req, 'user', None) or ''}"
            # Conversation KV-prefix reuse (prefix_cache, opt-in): the same key, optionally narrowed by a client
            # conversation id so one key's parallel chats keep their own cached turns (never wider than the key)
            prefix_key = None
            if rt.prefix_cache is not None:
                conv_id = request.headers.get(CONVERSATION_HEADER, "")
                if len(conv_id) > 128:
                    raise InvalidRequestError(f"{CONVERSATION_HEADER} is longer than 128 characters")
                prefix_key = f"{conversation}\x00conv:{conv_id}"
            lease, sessions = await _admit(rt, key, prompts, params, rid, conversation, prefix_key)
        except PGWError as exc:
            _audit_request(endpoint, rid, key, rt, exc.http_status, str(exc.code), t0, sessions, req.stream)
            raise
        assert rt is not None and lease is not None
        stop_ids = _stop_ids(rt, params)
        created = int(time.time())
        fp_short = rt.fingerprint[:12] if rt.fingerprint else None
        model_rt: ModelRuntime = rt
        the_lease: _KeyLease = lease

        def _usage() -> dict[str, Any]:
            p = sum(len(s.prompt_ids) for s in sessions)
            c = sum(s.completion_tokens for s in sessions)
            out: dict[str, Any] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
            if model_rt.prefix_cache is not None:  # OpenAI's field for prompt tokens served from a cache
                out["prompt_tokens_details"] = {"cached_tokens": sum(s.prefix_reused for s in sessions)}
            return out

        if not req.stream:
            status, code = 200, None
            try:
                tasks = [asyncio.ensure_future(_collect(s, stop_ids)) for s in sessions]
                try:
                    results = await asyncio.gather(*tasks)
                except BaseException:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                payload = render(rid, created, model_rt, fp_short, results, prompt_texts, _usage())
                return JSONResponse(payload, headers={"x-request-id": rid})
            except PGWError as exc:
                status, code = exc.http_status, str(exc.code)
                for s in sessions:
                    await s.close()
                return _err(exc, rid)
            except BaseException:
                status, code = 500, str(ErrorCode.INTERNAL)
                for s in sessions:
                    await s.close()
                raise
            finally:
                the_lease.finish(_usage()["total_tokens"], sum(s.masks_reserved for s in sessions))
                _audit_request(endpoint, rid, key, model_rt, status, code, t0, sessions, False)

        sess = sessions[0]
        outcome: dict[str, Any] = {"status": 200, "code": None}

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in render_stream(rid, created, model_rt, fp_short, sess, stop_ids,
                                                 prompt_texts[0], req.include_usage, _usage):
                    yield chunk
            except PGWError as exc:
                outcome["status"], outcome["code"] = exc.http_status, str(exc.code)
                yield _sse(error_body(exc))
            except asyncio.CancelledError:
                outcome["code"] = "cancelled"
                raise
            except Exception as exc:
                outcome["status"], outcome["code"] = 500, str(ErrorCode.INTERNAL)
                log.error("stream failed: %s", type(exc).__name__, extra={"request_id": rid})
                yield _sse(error_body(PGWError("internal error")))
            yield _SSE_DONE

        async def on_close() -> None:
            await sess.close()
            the_lease.finish(_usage()["total_tokens"], sess.masks_reserved)
            _audit_request(endpoint, rid, key, model_rt, int(outcome["status"]), outcome["code"], t0, sessions, True)

        return _managed_stream(body(), on_close, {"x-request-id": rid, "Cache-Control": "no-cache",
                                                  "X-Accel-Buffering": "no"})

    # ---- /v1/chat/completions ---------------------------------------------------------------------
    def _render_chat(rid: str, created: int, rt: ModelRuntime, fp: str | None, results: list[_Collected],
                     prompt_texts: list[str], usage: dict[str, int]) -> dict[str, Any]:
        r = results[0]
        choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": r.text},
            "logprobs": _chat_logprobs(r.events, _TokenText(rt)) if chat_logprobs_enabled(r) else None,
            "finish_reason": r.finish,
        }
        return {"id": rid, "object": "chat.completion", "created": created, "model": rt.name,
                "system_fingerprint": fp, "choices": [choice], "usage": usage}

    def chat_logprobs_enabled(r: _Collected) -> bool:
        return any(ev.logprob is not None for ev in r.events)

    async def _stream_chat(rid: str, created: int, rt: ModelRuntime, fp: str | None, sess: ActiveSession,
                           stop_ids: frozenset[int], prompt_text: str, include_usage: bool,
                           usage: Callable[[], dict[str, int]]) -> AsyncIterator[bytes]:
        base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": rt.name,
                "system_fingerprint": fp}
        extra = {"usage": None} if include_usage else {}
        tt = _TokenText(rt)
        yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                         "logprobs": None, "finish_reason": None}], **extra})
        async for ev in sess.stream():
            vis = _visible(ev, stop_ids)
            if not ev.text and ev.finish_reason is None:
                if not (vis and ev.logprob is not None):
                    continue
            delta: dict[str, Any] = {"content": ev.text} if ev.text else {}
            lp = _chat_logprobs([ev], tt) if (vis and ev.logprob is not None) else None
            yield _sse({**base, "choices": [{"index": 0, "delta": delta, "logprobs": lp,
                                             "finish_reason": str(ev.finish_reason) if ev.finish_reason else None}],
                        **extra})
        if include_usage:
            yield _sse({**base, "choices": [], "usage": usage()})

    async def _run_chat(req: ChatCompletionRequest, request: Request, key: ApiKey | None = None) -> Any:
        """Chat completion; ``key`` given = already authenticated (console playground)."""

        async def prompts(rt: ModelRuntime) -> tuple[list[list[int]], list[str]]:
            msgs = _chat_messages(req)
            ids = await asyncio.to_thread(rt.encode_chat, msgs)
            return [ids], [""]

        lp = (req.top_logprobs or 0) if req.logprobs else None
        return await _handle(request, "chat", req, prompts, lp, _render_chat, _stream_chat, key=key)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
        return await _run_chat(req, request)

    # ---- /v1/completions -------------------------------------------------------------------------
    def _render_completion(rid: str, created: int, rt: ModelRuntime, fp: str | None, results: list[_Collected],
                           prompt_texts: list[str], usage: dict[str, int]) -> dict[str, Any]:
        choices = []
        tt = _TokenText(rt)
        for i, (r, ptext) in enumerate(zip(results, prompt_texts, strict=True)):
            text = (ptext + r.text) if ptext else r.text
            lp = _completion_logprobs(r.events, tt, len(ptext)) if chat_logprobs_enabled(r) else None
            choices.append({"index": i, "text": text, "logprobs": lp, "finish_reason": r.finish})
        return {"id": rid, "object": "text_completion", "created": created, "model": rt.name,
                "system_fingerprint": fp, "choices": choices, "usage": usage}

    async def _stream_completion(rid: str, created: int, rt: ModelRuntime, fp: str | None, sess: ActiveSession,
                                 stop_ids: frozenset[int], prompt_text: str, include_usage: bool,
                                 usage: Callable[[], dict[str, int]]) -> AsyncIterator[bytes]:
        base = {"id": rid, "object": "text_completion", "created": created, "model": rt.name,
                "system_fingerprint": fp}
        extra = {"usage": None} if include_usage else {}
        tt = _TokenText(rt)
        offset = len(prompt_text)
        if prompt_text:
            yield _sse({**base, "choices": [{"index": 0, "text": prompt_text, "logprobs": None,
                                             "finish_reason": None}], **extra})
        async for ev in sess.stream():
            vis = _visible(ev, stop_ids)
            if not ev.text and ev.finish_reason is None and not (vis and ev.logprob is not None):
                continue
            lp = _completion_logprobs([ev], tt, offset) if (vis and ev.logprob is not None) else None
            offset += len(ev.text)
            yield _sse({**base, "choices": [{"index": 0, "text": ev.text, "logprobs": lp,
                                             "finish_reason": str(ev.finish_reason) if ev.finish_reason else None}],
                        **extra})
        if include_usage:
            yield _sse({**base, "choices": [], "usage": usage()})

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request) -> Any:
        async def prompts(rt: ModelRuntime) -> tuple[list[list[int]], list[str]]:
            items = req.prompt_items()
            if len(items) > settings.max_prompts_per_request:
                raise InvalidRequestError(f"at most {settings.max_prompts_per_request} prompts per request")
            if req.suffix:
                raise InvalidRequestError("suffix is not supported")
            ids: list[list[int]] = []
            echo: list[str] = []
            for item in items:
                if isinstance(item, str):
                    ids.append(await asyncio.to_thread(rt.encode_text, item))
                    echo.append(item)
                else:
                    ids.append(rt.check_token_ids(item))
                    echo.append(await asyncio.to_thread(rt.decode_ids, item) if req.echo else "")
            return ids, (echo if req.echo else [""] * len(items))

        return await _handle(request, "completions", req, prompts, req.logprobs, _render_completion,
                             _stream_completion)

    # ---- /v1/models -------------------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models(request: Request) -> Any:
        key = require_key(request, "models")
        created = int(state.started_at)
        cards = [ModelCard(id=n, created=created) for n in runtime.names() if key.allows_model(n)]
        return ModelList(data=cards).model_dump()

    @app.get("/v1/models/{name}")
    async def get_model(name: str, request: Request) -> Any:
        key = require_key(request, "models")
        _model_for_key(key, name)
        return ModelCard(id=name, created=int(state.started_at)).model_dump()

    # ---- admin ---------------------------------------------------------------------------------
    @app.get("/admin/status")
    async def admin_status(request: Request) -> Any:
        require_admin(request, "admin.status")
        statuses = await asyncio.to_thread(lambda: [rt.status() for rt in runtime.runtimes.values()])
        from pgw.gateway.gpu_report import gpu_report

        try:
            gpu = await asyncio.to_thread(gpu_report, dict(runtime.runtimes))
        except Exception as exc:  # a report must never break the status page
            gpu = {"error": type(exc).__name__}
        return {
            "version": 1,
            "uptime_s": round(time.time() - state.started_at, 1),
            "production_policy": not settings.allow_dev_modes,
            "models": statuses,
            "gpu": gpu,
            "api_keys": len(state.keys.list()),
            "egress_allowlist": sorted("{}://{}:{}".format(*o) for o in settings.egress_origins(
                [rt.config for rt in runtime.runtimes.values()])),
            "audit_log": str(settings.audit_log_file) if settings.audit_log_file else None,
            "telemetry": "disabled",
        }

    @app.get("/admin/masks")
    async def admin_masks(request: Request) -> Any:
        require_admin(request, "admin.masks")
        return await asyncio.to_thread(runtime.masks_status)

    @app.get("/admin/keys")
    async def admin_keys(request: Request) -> Any:
        require_admin(request, "admin.keys")
        keys = state.keys.list()
        usage = state.quotas.usage([k.key_id for k in keys])
        return {"keys": [{**k.public(), "tokens_today": usage.get(k.key_id, 0)} for k in keys]}

    @app.post("/admin/keys", status_code=201)
    async def admin_create_key(request: Request) -> Any:
        actor = require_admin(request, "admin.keys")
        try:
            body = json.loads(await read_limited(request, _ADMIN_BODY_LIMIT))
        except ValueError as exc:
            raise InvalidRequestError("body must be JSON") from exc
        if not isinstance(body, dict) or not isinstance(body.get("name"), str):
            raise InvalidRequestError("body needs a string 'name'")

        def _opt_int(name: str) -> int | None:
            v = body.get(name)
            if v is None:
                return None
            if not isinstance(v, int) or isinstance(v, bool):
                raise InvalidRequestError(f"{name} must be an integer")
            return v

        def _str_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
            v = body.get(name)
            if v is None:
                return default
            if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v) or len(v) > 256:
                raise InvalidRequestError(f"{name} must be a list of non-empty strings")
            return tuple(v)

        scopes = _str_list("scopes", ("inference",)) or ("inference",)
        models = _str_list("models", ())
        try:
            rec, plaintext = state.keys.create(
                body["name"],
                scopes=scopes,
                models=models,
                requests_per_minute=_opt_int("requests_per_minute"),
                tokens_per_minute=_opt_int("tokens_per_minute"),
                daily_token_quota=_opt_int("daily_token_quota"),
                max_concurrent_requests=_opt_int("max_concurrent_requests"),
            )
        except PGWError as exc:
            raise InvalidRequestError(exc.message) from exc
        state.audit.event("admin.key_created", action="key_create", target=rec.key_id, key_id=actor,
                          scopes=",".join(rec.scopes))
        return JSONResponse({"key": plaintext, "record": rec.public()}, status_code=201)

    @app.delete("/admin/keys/{key_id}")
    async def admin_revoke_key(key_id: str, request: Request) -> Any:
        actor = require_admin(request, "admin.keys")
        state.keys.revoke(key_id)
        state.rate_limiter.forget(key_id)
        state.audit.event("admin.key_revoked", action="key_revoke", target=key_id, key_id=actor)
        return {"revoked": key_id}

    @app.post("/admin/models/{name}/reload")
    async def admin_reload(name: str, request: Request) -> Any:
        actor = require_admin(request, "admin.models")
        runtime.get(name)
        state.audit.event("admin.model_reload", action="model_reload", target=name, key_id=actor)
        rt = await runtime.reload(name)
        return {"reloaded": name, "fingerprint": rt.fingerprint, "ready": rt.ready()[0]}

    @app.post("/admin/models/{name}/reset")
    async def admin_reset(name: str, request: Request) -> Any:
        actor = require_admin(request, "admin.models")
        rt = runtime.get(name)
        body: Any = {}
        raw = await read_limited(request, _ADMIN_BODY_LIMIT)
        if raw:
            try:
                body = json.loads(raw)
            except ValueError as exc:
                raise InvalidRequestError("body must be JSON") from exc
        if not isinstance(body, dict):
            raise InvalidRequestError("body must be a JSON object")
        flags = {k: body.get(k, True) for k in ("breaker", "noise")}
        if not all(isinstance(v, bool) for v in flags.values()):
            raise InvalidRequestError("'breaker' and 'noise' must be booleans")
        out = rt.reset_safety(breaker=flags["breaker"], noise=flags["noise"])
        state.audit.event("admin.safety_reset", action="safety_reset", target=name, key_id=actor,
                          reason=json.dumps(out, sort_keys=True))
        return {"model": name, **out}

    # ---- ops -----------------------------------------------------------------------------------------
    @app.get("/healthz")
    async def healthz() -> Any:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(models: str | None = None) -> Any:
        """Process readiness (Kubernetes): 200 once started and not draining.

        Per-model readiness (masks exhausted, breaker open, service unreachable) is reported in the
        body and in ``pgw_gateway_model_ready`` but does not fail the probe: a single unhealthy
        model must not take the whole gateway out of the load balancer (requests for it fail fast
        with typed errors). ``?models=all`` restores the strict check (503 unless every model is
        ready), e.g. for an external monitor.
        """
        details = {}
        all_models_ok = bool(runtime.runtimes)
        for rt in runtime.runtimes.values():
            ok, reason = await asyncio.to_thread(rt.ready)
            details[rt.name] = {"ready": ok, "reason": reason, "state": rt.state}
            all_models_ok = all_models_ok and ok
        process_ok = runtime.ready()
        ok = process_ok and (all_models_ok if models == "all" else True)
        return JSONResponse(
            {"ready": ok, "process_ready": process_ok, "all_models_ready": all_models_ok, "models": details},
            status_code=200 if ok else 503,
        )

    async def _metrics_body() -> Any:
        body, ctype = await asyncio.to_thread(metrics.render)
        return Response(body, media_type=ctype)

    @app.get("/metrics")
    async def metrics_route(request: Request) -> Any:
        if settings.metrics_require_admin:
            require_admin(request, "metrics")
        return await _metrics_body()

    # Internal plain-HTTP ops listener (probes and scraping when the API requires client certs).
    ops_app = FastAPI(title="PGW gateway ops", docs_url=None, redoc_url=None, openapi_url=None)
    ops_app.add_api_route("/healthz", healthz, methods=["GET"])
    ops_app.add_api_route("/readyz", readyz, methods=["GET"])
    ops_app.add_api_route("/metrics", _metrics_body, methods=["GET"])
    app.state.ops_app = ops_app

    if settings.dashboard_enabled:
        html = _dashboard_html()

        @app.get("/dashboard")
        async def dashboard() -> Any:
            return HTMLResponse(html, headers={"Content-Security-Policy": "default-src 'self' 'unsafe-inline'"})

    # ---- console (operator web UI and its data endpoints) ---------------------------------------
    from pgw.console.routes import ConsoleHeaders, install_console

    install_console(app, state, require_admin=require_admin, run_chat=_run_chat, static_dir=console_static_dir)
    app.add_middleware(ConsoleHeaders)
    return app
