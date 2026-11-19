"""Console routes: sign-in API, playground, operator data endpoints and the static single-page app.

Installed into the gateway app by :func:`install_console` (called from
:func:`pgw.gateway.app.create_app`). The contract is ``docs/api/console-api.md``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hmac
import html
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError

import pgw
from pgw.console.oidc import BINDING_COOKIE, BINDING_COOKIE_PATH, OIDCClient, OIDCError, as_auth_error
from pgw.console.sessions import COOKIE_NAME, CSRF_HEADER, ConsoleSessions, cookie_secure
from pgw.console.usage import WINDOWS, AuditReader, parse_ts_ms
from pgw.core.errors import AuthError, ErrorCode, PGWError
from pgw.gateway.auth import ApiKey
from pgw.gateway.openai_schemas import ChatCompletionRequest
from pgw.gateway.runtime import InvalidRequestError

#: Audit-log scans (overview, usage, audit) run on their own thread so a few busy console tabs cannot
#: occupy the default executor that inference uses for masking and reservation work.
_SCAN_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-console-scan")


async def _scan(fn: Any, *args: Any) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_SCAN_POOL, fn, *args)

__all__ = ["CONSOLE_SECURITY_HEADERS", "DEFAULT_STATIC_DIR", "PLAYGROUND_KEY", "ConsoleHeaders", "install_console"]

#: Built static single-page app (output of ``pgw/console``).
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "static"

#: Identity the playground runs as: quota-limited and audited like any key (never in the key store).
PLAYGROUND_KEY = ApiKey(key_id="console-playground", hashed="", name="console playground", scopes=("inference",),
                        max_concurrent_requests=4)

CONSOLE_SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                               "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}
_IMMUTABLE = "public, max-age=31536000, immutable"
_MAX_AUDIT_LIMIT = 1000

_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".json": "application/json", ".map": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
    ".ico": "image/x-icon", ".webp": "image/webp", ".woff2": "font/woff2", ".woff": "font/woff",
    ".txt": "text/plain; charset=utf-8", ".webmanifest": "application/manifest+json",
}

_NOT_BUILT = """<!doctype html><html><head><meta charset="utf-8"><title>PGW Console</title></head>
<body><h1>PGW Console is not built</h1><p>The console assets are missing from this installation
(<code>pgw/console/static/</code>). Build the console (<code>pgw/console</code>) or install a release
package. The gateway API and <a href="/dashboard">/dashboard</a> work without it.</p></body></html>"""


class ConsoleHeaders:
    """Pure-ASGI middleware adding the console security headers to ``/console*`` responses and
    ``Cache-Control: no-store`` to console API and ``/admin/*`` responses (unless already set)."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        console = path == "/console" or path.startswith("/console/")
        api = path.startswith(("/console/api/", "/admin/"))
        if not (console or api):
            await self.app(scope, receive, send)
            return

        async def wrapped(msg: Any) -> None:
            if msg["type"] == "http.response.start":
                headers = list(msg.get("headers", []))
                present = {k.lower() for k, _ in headers}
                if console:
                    for k, v in CONSOLE_SECURITY_HEADERS.items():
                        if k.lower().encode() not in present:
                            headers.append((k.lower().encode(), v.encode()))
                if api and b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                msg = {**msg, "headers": headers}
            await send(msg)

        await self.app(scope, receive, wrapped)


_SSO_ERROR = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PGW Console: {title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body><main><h1>{title}</h1><p>{message}</p><p><a href="/console/login">Back to sign-in</a></p></main></body></html>"""
_SSO_TITLES = {400: "Sign-in could not be completed", 401: "Sign-in failed", 403: "Access denied",
               429: "Too many sign-in attempts", 503: "Single sign-on is unavailable"}


def _not_found() -> JSONResponse:
    return JSONResponse({"error": {"message": "unknown console API endpoint", "type": "not_found_error",
                                   "code": str(ErrorCode.PROTOCOL), "param": None}}, status_code=404)


def _client(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _static_file(root: Path, rel: str) -> Path | None:
    """``root/rel`` if it is a regular file inside ``root`` (no traversal, no hidden files)."""
    if not rel or "\\" in rel or "\0" in rel:
        return None
    parts = rel.split("/")
    if any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    try:
        root_r = root.resolve()
        cand = (root_r / rel).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not cand.is_relative_to(root_r) or not cand.is_file():
        return None
    return cand


def install_console(
    app: FastAPI,
    state: Any,
    *,
    require_admin: Callable[[Request, str], str],
    run_chat: Callable[[ChatCompletionRequest, Request, ApiKey], Awaitable[Any]],
    static_dir: Path | None = None,
) -> None:
    """Register ``/admin/overview``, ``/admin/usage``, ``/admin/audit`` and, when
    ``settings.console_enabled``, the ``/console`` routes (sign-in API, playground, static app)."""
    settings = state.settings
    runtime = state.runtime
    reader = AuditReader(settings.audit_log_file)
    state.audit_reader = reader
    sessions: ConsoleSessions | None = state.console

    def _key_name(key_id: str) -> str | None:
        if key_id == PLAYGROUND_KEY.key_id:
            return PLAYGROUND_KEY.name
        rec = state.keys.get(key_id) if len(key_id) == 12 else None
        return rec.name if rec is not None else None

    # ---- operator data -------------------------------------------------------------------------------
    @app.get("/admin/overview")
    async def admin_overview(request: Request) -> Any:
        require_admin(request, "admin.overview")
        statuses = await asyncio.to_thread(lambda: [rt.status() for rt in runtime.runtimes.values()])
        day = await _scan(reader.usage, "24h", _key_name)
        consumed_24h = {m: int(a.get("masks_consumed", 0)) for m, a in day["by_model"].items()}
        models = []
        integrity = checked = 0
        masked = [s for s in statuses if s.get("mode") == "masked"]
        for s in statuses:
            avail = s.get("masks_available")
            rate = consumed_24h.get(s["name"], 0) / 86_400.0
            runway = round(avail / rate, 1) if isinstance(avail, int) and rate > 0 else None
            models.append({"name": s["name"], "state": s.get("state"), "ring": s.get("ring"), "ready": s.get("ready"),
                           "reason": s.get("reason"), "masks_available": avail, "masks_runway_s": runway,
                           "mode": s.get("mode")})
            backend = s.get("backend") or {}
            integrity += int(backend.get("integrity_failures", 0) or 0)
            checked += int(backend.get("freivalds_checked", 0) or 0)
        rings = sorted({str(s.get("ring")) for s in masked})
        rates = [float(s["freivalds_rate"]) for s in masked if s.get("freivalds_rate") is not None]
        noise_models = {s["name"]: s.get("noise_monitor") or {} for s in masked}
        totals = day["totals"]
        return {
            "version": pgw.__version__,
            "uptime_s": round(time.time() - state.started_at, 1),
            "models": models,
            "requests_24h": totals["requests"],
            "tokens_24h": totals["prompt_tokens"] + totals["completion_tokens"],
            "errors_24h": totals["errors"],
            "integrity_failures_total": integrity,
            "freivalds_checked_total": checked,
            "stream_requests_total": state.stream_requests,
            "privacy": {
                "masked": bool(statuses) and len(masked) == len(statuses),
                "ring": ", ".join(rings) if rings else None,
                "freivalds_rate": min(rates) if rates else None,
                "noise_monitor": {
                    "enabled": bool(settings.noise_monitor),
                    "fail_closed": bool(settings.noise_monitor_fail_closed),
                    "alarm": any(bool(n.get("alarm")) for n in noise_models.values()),
                    "models": noise_models,
                },
            },
            "production_policy": not settings.allow_dev_modes,
        }

    @app.get("/admin/usage")
    async def admin_usage(request: Request, window: str = "24h") -> Any:
        require_admin(request, "admin.usage")
        if window not in WINDOWS:
            raise InvalidRequestError(f"window must be one of {', '.join(WINDOWS)}")
        return await _scan(reader.usage, window, _key_name)

    @app.get("/admin/audit")
    async def admin_audit(request: Request, limit: int = 100, before: str | None = None, key_id: str | None = None,
                          model: str | None = None, status: str | None = None, event: str | None = None) -> Any:
        require_admin(request, "admin.audit")
        if not 1 <= limit <= _MAX_AUDIT_LIMIT:
            raise InvalidRequestError(f"limit must be between 1 and {_MAX_AUDIT_LIMIT}")
        before_ms: int | None = None
        if before:
            before_ms = parse_ts_ms(before)
            if before_ms is None:
                try:
                    before_ms = int(float(before) * 1000)
                except (ValueError, OverflowError) as exc:
                    raise InvalidRequestError("before must be an audit timestamp or Unix seconds") from exc
        return await _scan(lambda: reader.events(
            limit=limit, before_ms=before_ms, key_id=key_id or None, model=model or None, status=status or None,
            event=event or None))

    if not settings.console_enabled:
        return
    root = static_dir if static_dir is not None else DEFAULT_STATIC_DIR

    # ---- sign-in -------------------------------------------------------------------------------------
    oidc: OIDCClient | None = state.oidc
    token_login = bool(settings.console_token_login_enabled)

    def _sessions() -> ConsoleSessions:
        """The session store, if anybody can sign in (admin token configured or SSO)."""
        if sessions is None or not (sessions.login_enabled or oidc is not None):
            raise PGWError("console login is disabled: no admin token is configured (PGW_GATEWAY_ADMIN_TOKEN)",
                           code=ErrorCode.UNAVAILABLE)
        return sessions

    def _me(s: Any) -> dict[str, Any]:
        assert sessions is not None
        return {"role": s.role, "csrf": s.csrf, "expires_at": int(sessions.expires_at(s)), "version": pgw.__version__,
                "auth": s.auth, "user": s.user}

    def _secure(request: Request) -> bool:
        return cookie_secure(request.url.scheme, request.url.hostname, _client(request))

    def _start_session(request: Request, resp: Response, store: ConsoleSessions, sess_sid: str) -> None:
        resp.set_cookie(COOKIE_NAME, sess_sid, max_age=int(store.max_age_s), path="/", httponly=True,
                        samesite="strict", secure=_secure(request))

    @app.get("/console/api/auth-methods")
    async def console_auth_methods() -> Any:
        return {"token": token_login and sessions is not None and sessions.login_enabled,
                "oidc": {"enabled": oidc is not None, "label": oidc.label if oidc is not None else "SSO"}}

    @app.post("/console/api/login")
    async def console_login(request: Request) -> Any:
        if not token_login:
            raise PGWError("token sign-in is disabled on this gateway; sign in with SSO",
                           code=ErrorCode.PERMISSION_DENIED)
        store = _sessions()
        if not store.login_enabled:
            raise PGWError("console login is disabled: no admin token is configured (PGW_GATEWAY_ADMIN_TOKEN)",
                           code=ErrorCode.UNAVAILABLE)
        client = _client(request)
        state.login_limiter.check(client)
        try:
            body = await request.json()
        except ValueError as exc:
            raise InvalidRequestError("body must be JSON: {\"token\": \"...\"}") from exc
        token = body.get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token or len(token) > 4096:
            raise InvalidRequestError("body needs a string 'token'")
        role = store.authenticate(token.strip())
        if role is None:
            state.audit.event("console.login_failed", action="console_login", client=client, method="token",
                              code=str(ErrorCode.UNAUTHENTICATED))
            raise AuthError("invalid console token")
        old = request.cookies.get(COOKIE_NAME)
        if old:
            store.delete(old)
        sid, sess = store.create(role, token=token.strip())
        state.audit.event("console.login", action="console_login", key_id=sess.actor, client=client, reason=role,
                          method="token")
        resp = JSONResponse({"role": role, "csrf": sess.csrf})
        _start_session(request, resp, store, sid)
        return resp

    # ---- single sign-on (OIDC) -------------------------------------------------------------------------
    def _sso_error_page(exc: PGWError) -> HTMLResponse:
        status = exc.http_status
        page = _SSO_ERROR.format(title=html.escape(_SSO_TITLES.get(status, "Sign-in failed")),
                                 message=html.escape(exc.message))
        headers = {"Cache-Control": "no-store"}
        retry = exc.details.get("retry_after_s")
        if status == 429 and isinstance(retry, int | float):
            headers["Retry-After"] = str(max(1, round(retry)))
        return HTMLResponse(page, status_code=status, headers=headers)

    def _clear_binding(request: Request, resp: Response) -> None:
        resp.delete_cookie(BINDING_COOKIE, path=BINDING_COOKIE_PATH, httponly=True, samesite="lax",
                           secure=_secure(request))

    @app.get("/console/api/oidc/login")
    async def console_oidc_login(request: Request) -> Any:
        if oidc is None:
            return _not_found()
        try:
            _sessions()
            state.oidc_start_limiter.check(_client(request))
            url, binding = await oidc.start()
        except PGWError as exc:
            return _sso_error_page(exc)
        resp = RedirectResponse(url, status_code=302, headers={"Cache-Control": "no-store"})
        resp.set_cookie(BINDING_COOKIE, binding, max_age=int(oidc.pending.ttl_s), path=BINDING_COOKIE_PATH,
                        httponly=True, samesite="lax", secure=_secure(request))
        return resp

    @app.get("/console/api/oidc/callback")
    async def console_oidc_callback(request: Request) -> Any:
        if oidc is None:
            return _not_found()
        client = _client(request)
        q = request.query_params
        try:
            store = _sessions()
            state.login_limiter.check(client)
            try:
                ident = await oidc.finish(code=q.get("code"), state=q.get("state"), error=q.get("error"),
                                          iss=q.get("iss"), binding=request.cookies.get(BINDING_COOKIE))
            except OIDCError as exc:
                state.audit.event("console.login_failed", action="console_login", client=client, method="oidc",
                                  code=str(exc.code), reason=exc.reason, subject=exc.subject, email=exc.email)
                raise
            except PGWError:
                raise
            except Exception as exc:  # never leak details of a failed sign-in
                err = as_auth_error(exc)
                state.audit.event("console.login_failed", action="console_login", client=client, method="oidc",
                                  code=str(err.code), reason="oidc_error")
                raise err from exc
        except PGWError as exc:
            resp: Response = _sso_error_page(exc)
            _clear_binding(request, resp)
            return resp
        old = request.cookies.get(COOKIE_NAME)
        if old:
            store.delete(old)
        sid, sess = store.create(ident.role, auth="oidc", user=ident.user())
        state.audit.event("console.login", action="console_login", key_id=sess.actor, client=client,
                          reason=ident.role, method="oidc", subject=ident.sub, email=ident.email)
        resp = RedirectResponse("/console/", status_code=302, headers={"Cache-Control": "no-store"})
        _start_session(request, resp, store, sid)
        _clear_binding(request, resp)
        return resp

    @app.post("/console/api/logout", status_code=204, response_class=Response)
    async def console_logout(request: Request) -> Response:
        store = _sessions()
        sid = request.cookies.get(COOKIE_NAME)
        sess = store.get(sid, touch=False) if sid else None
        if sess is not None:
            csrf = request.headers.get(CSRF_HEADER)
            if not csrf or not hmac.compare_digest(csrf.encode(), sess.csrf.encode()):
                raise PGWError(f"missing or invalid CSRF token ({CSRF_HEADER})", code=ErrorCode.PERMISSION_DENIED)
            store.delete(sid)
            state.audit.event("console.logout", action="console_logout", key_id=sess.actor, client=_client(request))
        resp = Response(status_code=204)
        resp.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="strict", secure=_secure(request))
        return resp

    @app.get("/console/api/me")
    async def console_me(request: Request) -> Any:
        store = _sessions()
        s = store.get(request.cookies.get(COOKIE_NAME))
        if s is None:
            raise AuthError("not signed in")
        return _me(s)

    # ---- playground ----------------------------------------------------------------------------------
    @app.post("/console/api/playground/chat")
    async def console_playground_chat(request: Request) -> Any:
        actor = require_admin(request, "console.playground")
        if not actor.startswith("console-admin") and actor != "admin-token":
            raise PGWError("the playground needs an admin console session or the admin token",
                           code=ErrorCode.PERMISSION_DENIED)
        try:
            body = await request.json()
        except ValueError as exc:
            raise InvalidRequestError("body must be JSON") from exc
        try:
            req = ChatCompletionRequest.model_validate(body)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        return await run_chat(req, request, PLAYGROUND_KEY)

    @app.api_route("/console/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def console_api_missing(rest: str) -> Any:
        return _not_found()

    # ---- static app ----------------------------------------------------------------------------------
    def _index() -> Path | None:
        return _static_file(root, "index.html")

    def _serve_index() -> Response:
        index = _index()
        if index is None:
            return HTMLResponse(_NOT_BUILT, headers={"Cache-Control": "no-store"})
        return FileResponse(index, media_type=_MEDIA_TYPES[".html"], headers={"Cache-Control": "no-cache"})

    @app.get("/console", include_in_schema=False)
    async def console_redirect() -> Any:
        return RedirectResponse("/console/", status_code=308)

    @app.get("/console/", include_in_schema=False)
    async def console_root() -> Any:
        return _serve_index()

    @app.get("/console/{path:path}", include_in_schema=False)
    async def console_static(path: str) -> Any:
        f = _static_file(root, path)
        if f is not None:
            if f.name == "index.html":
                return _serve_index()
            media = _MEDIA_TYPES.get(f.suffix.lower(), "application/octet-stream")
            cache = _IMMUTABLE if path.startswith("assets/") else "no-cache"
            return FileResponse(f, media_type=media, headers={"Cache-Control": cache})
        last = path.rsplit("/", 1)[-1]
        if path.startswith("assets/") or "." in last or ".." in path.split("/") or "\\" in path:
            return Response("not found", status_code=404, media_type="text/plain",
                            headers={"Cache-Control": "no-store"})
        return _serve_index()  # client-side route
