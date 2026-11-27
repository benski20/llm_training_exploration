"""The developer platform web app (FastAPI, server-rendered, mounted by the provider service).

Pages (under the mount prefix, ``/platform`` by default): ``/login``, ``/`` (overview), ``/quickstart``,
``/keys`` (+ ``/keys/new``, ``/keys/{id}/revoke``), ``/models``, ``/usage`` (+ ``/usage.csv``),
``/requests``, ``/settings``; static files under ``/assets/``.

Security model
    * Sign-in with an organisation access token (checked by the accounts service) or single sign-on
      (OIDC, the console's relying party). Sessions are the console's in-memory store: random 256-bit
      cookie ``pgw_platform`` (HttpOnly, SameSite=Strict, Secure except plain-HTTP loopback), a CSRF token
      per session that every form echoes, idle and absolute expiry.
    * Every accounts call is scoped to the session's organisation; no organisation id is taken from a request.
    * Viewers are read-only (no key creation or revocation, enforced server side).
    * Strict CSP (no inline script or style, no external origin), ``no-store`` on every page.
    * No prompt or response content exists in this app's data model (see :mod:`pgw.platform.client`).
"""

from __future__ import annotations

import asyncio
import hmac
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from pgw.console.sessions import ConsoleSession, ConsoleSessions, LoginRateLimiter, cookie_secure
from pgw.core.errors import ConfigError, PGWError, RateLimitedError
from pgw.core.settings import PGWSettings
from pgw.platform import pages
from pgw.platform.client import AccountsClient, AccountsError, Principal
from pgw.platform.ui import document, shell

__all__ = ["COOKIE_NAME", "SECURITY_HEADERS", "PlatformSettings", "create_platform_app", "mount_platform"]

COOKIE_NAME = "pgw_platform"
BINDING_COOKIE = "pgw_platform_oidc"
DESIGN_DIR = Path(__file__).resolve().parent.parent / "design"
STATIC_DIR = Path(__file__).resolve().parent / "static"

SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                               "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
                               "form-action 'self'",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Cross-Origin-Opener-Policy": "same-origin",
}
_ASSETS: dict[str, tuple[Path, str]] = {
    "pgw.css": (DESIGN_DIR / "pgw.css", "text/css; charset=utf-8"),
    "platform.css": (STATIC_DIR / "platform.css", "text/css; charset=utf-8"),
    "platform.js": (STATIC_DIR / "platform.js", "text/javascript; charset=utf-8"),
    "theme.js": (STATIC_DIR / "theme.js", "text/javascript; charset=utf-8"),
    "favicon.svg": (STATIC_DIR / "favicon.svg", "image/svg+xml"),
    **{f"fonts/{f}": (DESIGN_DIR / "fonts" / f, "font/woff2") for f in (
        "ibm-plex-sans-latin-wght-normal.woff2", "ibm-plex-sans-latin-ext-wght-normal.woff2",
        "ibm-plex-mono-latin-400-normal.woff2", "ibm-plex-mono-latin-500-normal.woff2")},
}
_MAX_FORM = 64 * 1024
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PAGE_SIZE = 50


class PlatformSettings(PGWSettings):
    """``PGW_PLATFORM_*`` settings of the developer platform."""

    model_config = SettingsConfigDict(env_prefix="PGW_PLATFORM_", extra="ignore", frozen=True)

    enabled: bool = False
    #: Where the provider service mounts the platform.
    path_prefix: str = "/platform"
    #: Allow sign-in with an organisation access token (disable to require SSO).
    token_login: bool = True
    session_idle_s: float = Field(default=8 * 3600.0, gt=0)
    session_max_age_s: float = Field(default=24 * 3600.0, gt=0)
    #: Accounts API of a separate provider process (``HttpAccountsClient``); unset when the service passes
    #: its in-process accounts object.
    accounts_url: str | None = None
    accounts_token_file: Path | None = None
    accounts_ca_file: Path | None = None
    # single sign-on (the console's OIDC relying party)
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret_file: Path | None = None
    oidc_redirect_url: str | None = None
    oidc_scopes: str = "openid email profile"
    oidc_role_claim: str = "groups"
    oidc_admin_values: list[str] = Field(default_factory=lambda: ["pgw-admins"])
    oidc_viewer_values: list[str] = Field(default_factory=lambda: ["pgw-viewers"])
    oidc_allowed_domains: list[str] = Field(default_factory=list)
    oidc_ca_file: Path | None = None
    oidc_label: str = "SSO"
    #: Organisation of single-sign-on users with :class:`HttpAccountsClient` (the IdP decides who may sign in).
    oidc_org: str | None = None

    def build_oidc(self) -> Any:
        """The OIDC client, or ``None`` when SSO is not configured."""
        if not self.oidc_issuer:
            return None
        if not (self.oidc_client_id and self.oidc_client_secret_file and self.oidc_redirect_url):
            raise ConfigError("platform SSO needs oidc_client_id, oidc_client_secret_file and oidc_redirect_url")
        from pgw.console.oidc import OIDCClient

        try:
            secret = Path(self.oidc_client_secret_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"cannot read oidc_client_secret_file ({exc.strerror or type(exc).__name__})") from exc
        if not secret or any(c.isspace() for c in secret):
            raise ConfigError("oidc_client_secret_file must contain exactly one secret")
        return OIDCClient(issuer=self.oidc_issuer, client_id=self.oidc_client_id, client_secret=secret,
                          redirect_url=self.oidc_redirect_url, scopes=self.oidc_scopes, role_claim=self.oidc_role_claim,
                          admin_values=self.oidc_admin_values, viewer_values=self.oidc_viewer_values,
                          allowed_domains=self.oidc_allowed_domains, ca_file=self.oidc_ca_file, label=self.oidc_label)

    def build_accounts(self) -> AccountsClient | None:
        """:class:`HttpAccountsClient` when ``accounts_url`` is set (token read from ``accounts_token_file``)."""
        if not self.accounts_url:
            return None
        from pgw.platform.client import HttpAccountsClient

        if self.accounts_token_file is None:
            raise ConfigError("accounts_url needs accounts_token_file")
        try:
            token = Path(self.accounts_token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"cannot read accounts_token_file ({exc.strerror or type(exc).__name__})") from exc
        verify: Any = str(self.accounts_ca_file) if self.accounts_ca_file else True
        return HttpAccountsClient(self.accounts_url, token, sso_org=self.oidc_org, verify=verify)


class _Headers:
    """Security headers on every response; ``no-store`` on everything but static assets."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        asset = "/assets/" in scope.get("path", "")

        async def wrapped(msg: Any) -> None:
            if msg["type"] == "http.response.start":
                headers = list(msg.get("headers", []))
                present = {k.lower() for k, _ in headers}
                for k, v in SECURITY_HEADERS.items():
                    if k.lower().encode() not in present:
                        headers.append((k.lower().encode(), v.encode()))
                if b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-cache" if asset else b"no-store"))
                msg = {**msg, "headers": headers}
            await send(msg)

        await self.app(scope, receive, wrapped)


def _clean_name(raw: str) -> str | None:
    name = " ".join(raw.split())
    if not 1 <= len(name) <= 64 or any(unicodedata.category(c).startswith("C") for c in name):
        return None
    return name


def create_platform_app(accounts: AccountsClient | None, settings: PlatformSettings | None = None, *,
                        oidc: Any = None, clock: Callable[[], float] = time.time) -> Any:
    """Build the platform app. Mount it at ``settings.path_prefix`` (see :func:`mount_platform`).

    Args:
        accounts: The accounts API (``None``: sign-in shows that the platform is not configured).
        settings: Platform settings (default: from the environment).
        oidc: An :class:`~pgw.console.oidc.OIDCClient` (default: from settings, ``None`` without SSO).
        clock: Wall clock (tests).
    """



    s = settings or PlatformSettings()
    if oidc is None:
        oidc = s.build_oidc()
    sessions = ConsoleSessions(None, idle_s=s.session_idle_s, max_age_s=s.session_max_age_s, clock=clock)
    limiter = LoginRateLimiter()
    sso_start_limiter = LoginRateLimiter(attempts=30)
    app = FastAPI(title="PGW platform", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(_Headers)
    app.state.platform_sessions = sessions

    # ---- helpers -------------------------------------------------------------------------------------
    def prefix(request: Request) -> str:
        return str(request.scope.get("root_path", "")).rstrip("/")

    def client(request: Request) -> str:
        from pgw.core.client_ip import client_ip

        return client_ip(request)

    def secure(request: Request) -> bool:
        return cookie_secure(request.url.scheme, request.url.hostname, client(request))

    def html(request: Request, body: str, *, title: str, status: int = 200) -> HTMLResponse:
        return HTMLResponse(document(title=title, body=body, prefix=prefix(request)), status_code=status)

    def principal_of(sess: ConsoleSession) -> Principal:
        u = sess.user or {}
        return Principal(org_id=u.get("org") or "", role=sess.role, name=u.get("name"), email=u.get("email"))

    def current(request: Request) -> ConsoleSession | None:
        return sessions.get(request.cookies.get(COOKIE_NAME))

    def to_login(request: Request) -> Response:
        return RedirectResponse(f"{prefix(request)}/login", status_code=303)

    async def call(fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kw)

    async def optional(fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        """``None`` when the accounts service does not offer this data (``unsupported``)."""
        try:
            return await call(fn, *args, **kw)
        except AccountsError as exc:
            if exc.kind == "unsupported":
                return None
            raise

    async def ctx_for(request: Request, sess: ConsoleSession) -> pages.Ctx:
        p = principal_of(sess)
        assert accounts is not None
        org = await call(accounts.org, p.org_id)
        return pages.Ctx(prefix=prefix(request), principal=p, org=org, csrf=sess.csrf, now=clock())

    def render(request: Request, ctx: pages.Ctx, sess: ConsoleSession, *, active: str, title: str, content: str,
               status: int = 200) -> HTMLResponse:
        p = ctx.principal
        body = shell(prefix=ctx.prefix, active=active, title=title, content=content, org_name=ctx.org.name,
                     plan=ctx.org.plan, user=p.email or p.name or "", role=p.role, csrf=sess.csrf)
        return html(request, body, title=title, status=status)

    def unavailable(request: Request, exc: AccountsError) -> HTMLResponse:
        status = 404 if exc.kind == "not_found" else 503 if exc.kind == "unavailable" else 400
        title = "Not found" if status == 404 else "Something went wrong"
        msg = exc.message if status != 503 else "The accounts service is unavailable. Try again in a moment."
        page_ = pages.error_page(prefix=prefix(request), title=title, message=msg)
        return html(request, page_, title=title, status=status)

    async def form(request: Request) -> dict[str, list[str]]:
        from pgw.core.http_body import read_limited

        try:
            body = await read_limited(request, _MAX_FORM, what="form")  # bounded also for chunked bodies
        except PGWError as exc:
            raise AccountsError("Form too large.", kind="invalid") from exc
        try:
            return parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True, max_num_fields=64)
        except ValueError as exc:
            raise AccountsError("Too many form fields.", kind="invalid") from exc

    def same_origin(request: Request) -> bool:
        """Refuse cross-site form posts (defence in depth next to SameSite cookies and CSRF tokens).

        Browsers send ``Sec-Fetch-Site``; otherwise a non-opaque ``Origin`` must name this host (the scheme
        is not compared: TLS may end at a proxy in front of the service).
        """
        site = request.headers.get("sec-fetch-site")
        if site is not None:
            return site in ("same-origin", "none")
        origin = request.headers.get("origin")
        if not origin or origin == "null":
            return True
        return urlsplit(origin).netloc == request.headers.get("host", request.url.netloc)

    def csrf_ok(sess: ConsoleSession, f: dict[str, list[str]]) -> bool:
        tok = (f.get("csrf") or [""])[0]
        return bool(tok) and hmac.compare_digest(tok.encode(), sess.csrf.encode())

    def forbidden(request: Request, message: str) -> HTMLResponse:
        return html(request, pages.error_page(prefix=prefix(request), title="Not allowed", message=message),
                    title="Not allowed", status=403)

    async def guarded_post(request: Request) -> tuple[ConsoleSession, dict[str, list[str]]] | Response:
        sess = current(request)
        if sess is None:
            return to_login(request)
        if not same_origin(request):
            return forbidden(request, "Cross-origin form submission refused.")
        try:
            f = await form(request)
        except AccountsError as exc:
            return forbidden(request, exc.message)
        if not csrf_ok(sess, f):
            return forbidden(request, "Your session token did not match. Reload the page and try again.")
        return sess, f

    # ---- static ----------------------------------------------------------------------------------------
    @app.get("/assets/{path:path}", include_in_schema=False)
    async def assets(path: str) -> Any:
        entry = _ASSETS.get(path)
        if entry is None or not entry[0].is_file():
            return Response("not found", status_code=404, media_type="text/plain")
        return FileResponse(entry[0], media_type=entry[1])

    # ---- sign-in ---------------------------------------------------------------------------------------
    def login_view(request: Request, *, error: str | None = None, status: int = 200) -> HTMLResponse:
        token_enabled = s.token_login and accounts is not None
        unavailable_msg = None
        if accounts is None:
            unavailable_msg = "The accounts service is not configured on this provider."
        elif not token_enabled and oidc is None:
            unavailable_msg = "No sign-in method is enabled on this provider."
        body = pages.login_page(prefix=prefix(request), token_enabled=token_enabled,
                                sso_label=oidc.label if (oidc is not None and accounts is not None) else None,
                                error=error, unavailable=unavailable_msg)
        return html(request, body, title="Sign in", status=status)

    @app.get("/login", include_in_schema=False)
    async def login_get(request: Request) -> Any:
        if current(request) is not None:
            return RedirectResponse(f"{prefix(request)}/", status_code=303)
        return login_view(request)

    def start_session(request: Request, resp: Response, p: Principal, *, auth: str) -> None:
        old = request.cookies.get(COOKIE_NAME)
        if old:
            sessions.delete(old)
        sid, _sess = sessions.create(p.role, auth=auth,  # type: ignore[arg-type]
                                     user={"org": p.org_id, "name": p.name, "email": p.email, "sub": None})
        resp.set_cookie(COOKIE_NAME, sid, max_age=int(s.session_max_age_s), path=prefix(request) or "/",
                        httponly=True, samesite="strict", secure=secure(request))

    @app.post("/login", include_in_schema=False)
    async def login_post(request: Request) -> Any:
        if accounts is None or not s.token_login:
            return login_view(request, error="Token sign-in is disabled.", status=403)
        if not same_origin(request):
            return login_view(request, error="Cross-origin sign-in refused.", status=403)
        try:
            with limiter.attempt(client(request)) as att:
                token = ((await form(request)).get("token") or [""])[0].strip()
                p = await call(accounts.authenticate, token) if 0 < len(token) <= 4096 else None
                if p is None or not p.org_id:
                    att.fail()
                    return login_view(request, error="That token was not recognised.", status=401)
        except RateLimitedError:
            return login_view(request, error="Too many sign-in attempts. Wait a minute and try again.", status=429)
        except AccountsError:
            return login_view(request, error="The accounts service is unavailable. Try again in a moment.",
                              status=503)
        resp = RedirectResponse(f"{prefix(request)}/", status_code=303)
        start_session(request, resp, p, auth="token")
        return resp

    @app.get("/sso/start", include_in_schema=False)
    async def sso_start(request: Request) -> Any:
        if oidc is None or accounts is None:
            return Response("not found", status_code=404, media_type="text/plain")
        try:
            sso_start_limiter.check(client(request))
            url, binding = await oidc.start()
        except PGWError as exc:
            return login_view(request, error=exc.message, status=exc.http_status)
        resp = RedirectResponse(url, status_code=302)
        resp.set_cookie(BINDING_COOKIE, binding, max_age=int(oidc.pending.ttl_s), path=f"{prefix(request)}/sso/",
                        httponly=True, samesite="lax", secure=secure(request))
        return resp

    @app.get("/sso/callback", include_in_schema=False)
    async def sso_callback(request: Request) -> Any:
        if oidc is None or accounts is None:
            return Response("not found", status_code=404, media_type="text/plain")
        q = request.query_params
        try:
            with limiter.attempt(client(request)) as att:
                try:
                    ident = await oidc.finish(code=q.get("code"), state=q.get("state"), error=q.get("error"),
                                              iss=q.get("iss"), binding=request.cookies.get(BINDING_COOKIE))
                    p = await call(accounts.principal_for_sso, sub=ident.sub, email=ident.email, role=ident.role)
                except PGWError:
                    att.fail()
                    raise
                except Exception as exc:  # never leak details of a failed sign-in
                    att.fail()
                    raise PGWError("single sign-on failed") from exc
                if p is None or not p.org_id:
                    att.fail()
                    return login_view(request, error="Your account is not a member of an organisation on this "
                                      "provider.", status=403)
        except PGWError as exc:
            msg = exc.message if isinstance(exc, RateLimitedError) else "Single sign-on failed. Try again."
            return login_view(request, error=msg, status=exc.http_status)
        resp = RedirectResponse(f"{prefix(request)}/", status_code=303)
        start_session(request, resp, Principal(p.org_id, p.role, p.name or ident.name, p.email or ident.email),
                      auth="oidc")
        resp.delete_cookie(BINDING_COOKIE, path=f"{prefix(request)}/sso/", httponly=True, samesite="lax",
                           secure=secure(request))
        return resp

    @app.post("/logout", include_in_schema=False)
    async def logout(request: Request) -> Any:
        r = await guarded_post(request)
        if isinstance(r, Response):
            return r
        sessions.delete(request.cookies.get(COOKIE_NAME))
        resp = RedirectResponse(f"{prefix(request)}/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME, path=prefix(request) or "/", httponly=True, samesite="strict",
                           secure=secure(request))
        return resp

    # ---- pages -----------------------------------------------------------------------------------------
    def page(active: str, title: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: session check, context, error handling and the shell around a page body."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            async def handler(request: Request) -> Any:
                sess = current(request)
                if sess is None or accounts is None:
                    return to_login(request)
                try:
                    ctx = await ctx_for(request, sess)
                    out = await fn(request, ctx, sess)
                except AccountsError as exc:
                    return unavailable(request, exc)
                if isinstance(out, Response):
                    return out
                content, status = out if isinstance(out, tuple) else (out, 200)
                return render(request, ctx, sess, active=active, title=title, content=content, status=status)

            handler.__name__ = fn.__name__
            return handler

        return deco

    def rng_of(request: Request) -> str:
        r = request.query_params.get("range", "30d")
        return r if r in pages.RANGES else "30d"

    @page("overview", "Overview")
    async def overview(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        rng = rng_of(request)
        n = pages.RANGES[rng]
        cur_days = pages.day_span(ctx.now, n)
        prev_days = pages.day_span(ctx.now, n, offset=n)
        org = ctx.principal.org_id
        rows, keys, recent, models = await asyncio.gather(
            call(accounts.usage, org, since=prev_days[0], until=cur_days[-1]), call(accounts.list_keys, org),
            optional(accounts.requests, org, limit=6), call(accounts.models, org))
        cur = [r for r in rows if r.day >= cur_days[0]]
        prev = [r for r in rows if r.day < cur_days[0]]
        return pages.overview(ctx, rng=rng, usage=cur, prev=prev, keys=keys, recent=recent, models=models)

    @page("quickstart", "Quickstart")
    async def quickstart(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        models, keys = await asyncio.gather(call(accounts.models, ctx.principal.org_id),
                                            call(accounts.list_keys, ctx.principal.org_id))
        return pages.quickstart_page(ctx, models=models, has_keys=any(k.active for k in keys))

    @page("keys", "API keys")
    async def keys(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        flash = "Key revoked." if request.query_params.get("revoked") else None
        return pages.keys_page(ctx, keys=await call(accounts.list_keys, ctx.principal.org_id), flash=flash)

    @page("keys", "Create API key")
    async def key_new(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        if not ctx.admin:
            return forbidden(request, "Viewers cannot create keys.")
        return pages.key_new_page(ctx, models=await call(accounts.models, ctx.principal.org_id))

    @page("keys", "API keys")
    async def key_create(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        if not same_origin(request):
            return forbidden(request, "Cross-origin form submission refused.")
        f = await form(request)
        if not csrf_ok(sess, f):
            return forbidden(request, "Your session token did not match. Reload the page and try again.")
        if not ctx.admin:
            return forbidden(request, "Viewers cannot create keys.")
        org = ctx.principal.org_id
        models = await call(accounts.models, org)
        raw_name = (f.get("name") or [""])[0]
        env = (f.get("env") or ["live"])[0]
        scoped = list(dict.fromkeys(f.get("models") or []))
        name = _clean_name(raw_name)
        error = None
        if name is None:
            error = "Enter a name of 1 to 64 printable characters."
        elif env not in ("live", "test"):
            error = "Choose the Live or Test environment."
        elif not pages.models_scope_valid(scoped, models):
            error = "Choose models from the list."
        if error is not None:
            return pages.key_new_page(ctx, models=models, error=error, name=raw_name[:64],
                                      env=env if env in ("live", "test") else "live", scoped=scoped), 400
        actor = ctx.principal.email or f"console-{ctx.principal.role}:{sess.public_id}"
        created = await call(accounts.create_key, org, name=name, env=env, models=scoped or None, actor=actor)
        keys_now = await call(accounts.list_keys, org)
        return pages.keys_page(ctx, keys=keys_now, created=created)

    @page("keys", "Revoke key")
    async def key_revoke_confirm(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        if not ctx.admin:
            return forbidden(request, "Viewers cannot revoke keys.")
        key_id = request.path_params["key_id"]
        key = next((k for k in await call(accounts.list_keys, ctx.principal.org_id) if k.id == key_id), None)
        if key is None or not _KEY_ID_RE.match(key_id):
            raise AccountsError("No such key.", kind="not_found")
        if not key.active:
            return RedirectResponse(ctx.url("/keys"), status_code=303)
        return pages.key_revoke_page(ctx, key)

    @page("keys", "Revoke key")
    async def key_revoke(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        if not same_origin(request):
            return forbidden(request, "Cross-origin form submission refused.")
        f = await form(request)
        if not csrf_ok(sess, f):
            return forbidden(request, "Your session token did not match. Reload the page and try again.")
        if not ctx.admin:
            return forbidden(request, "Viewers cannot revoke keys.")
        key_id = request.path_params["key_id"]
        if not _KEY_ID_RE.match(key_id):
            raise AccountsError("No such key.", kind="not_found")
        actor = ctx.principal.email or f"console-{ctx.principal.role}:{sess.public_id}"
        await call(accounts.revoke_key, ctx.principal.org_id, key_id, actor=actor)
        return RedirectResponse(ctx.url("/keys", revoked="1"), status_code=303)

    @page("models", "Models")
    async def models_view(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        return pages.models_page(ctx, models=await call(accounts.models, ctx.principal.org_id))

    @page("usage", "Usage")
    async def usage(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        rng = rng_of(request)
        by = request.query_params.get("by", "day")
        by = by if by in ("day", "key", "model") else "day"
        days = pages.day_span(ctx.now, pages.RANGES[rng])
        org = ctx.principal.org_id
        rows, keys_l, models = await asyncio.gather(call(accounts.usage, org, since=days[0], until=days[-1]),
                                                    call(accounts.list_keys, org), call(accounts.models, org))
        return pages.usage_page(ctx, rng=rng, by=by, usage=rows, keys=keys_l, models=models)

    async def usage_csv(request: Request) -> Any:
        sess = current(request)
        if sess is None or accounts is None:
            return to_login(request)
        p = principal_of(sess)
        rng = rng_of(request)
        days = pages.day_span(clock(), pages.RANGES[rng])
        try:
            rows, keys_l = await asyncio.gather(call(accounts.usage, p.org_id, since=days[0], until=days[-1]),
                                                call(accounts.list_keys, p.org_id))
        except AccountsError as exc:
            return unavailable(request, exc)
        return Response(pages.usage_csv(rows, keys_l), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="pgw-usage-{days[0]}-{days[-1]}.csv"'})

    @page("requests", "Requests")
    async def requests_view(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        q = request.query_params
        key_id = q.get("key") or None
        model = q.get("model") or None
        status = q.get("status") if q.get("status") in ("ok", "error") else None
        before = q.get("before") or None
        for v in (key_id, model, before):
            if v is not None and (len(v) > 128 or not v.isprintable()):
                raise AccountsError("Invalid filter.", kind="invalid")
        org = ctx.principal.org_id
        page_, keys_l, models = await asyncio.gather(
            optional(accounts.requests, org, limit=_PAGE_SIZE, before=before, key_id=key_id, model=model,
                     status=status),
            call(accounts.list_keys, org), call(accounts.models, org))
        return pages.requests_page(ctx, page=page_, keys=keys_l, models=models, key_id=key_id, model=model,
                                   status=status, before=before)

    @page("settings", "Settings")
    async def settings_view(request: Request, ctx: pages.Ctx, sess: ConsoleSession) -> Any:
        assert accounts is not None
        audit_fn = getattr(accounts, "audit", None)
        audit, keys_l = await asyncio.gather(
            optional(audit_fn, ctx.principal.org_id, limit=20) if audit_fn is not None else asyncio.sleep(0),
            call(accounts.list_keys, ctx.principal.org_id))
        return pages.settings_page(ctx, auth=sess.auth, expires_at=sessions.expires_at(sess), audit=audit,
                                   keys=keys_l)

    app.add_api_route("/", overview, methods=["GET"], include_in_schema=False)
    app.add_api_route("/quickstart", quickstart, methods=["GET"], include_in_schema=False)
    app.add_api_route("/keys", keys, methods=["GET"], include_in_schema=False)
    app.add_api_route("/keys", key_create, methods=["POST"], include_in_schema=False)
    app.add_api_route("/keys/new", key_new, methods=["GET"], include_in_schema=False)
    app.add_api_route("/keys/{key_id}/revoke", key_revoke_confirm, methods=["GET"], include_in_schema=False)
    app.add_api_route("/keys/{key_id}/revoke", key_revoke, methods=["POST"], include_in_schema=False)
    app.add_api_route("/models", models_view, methods=["GET"], include_in_schema=False)
    app.add_api_route("/usage", usage, methods=["GET"], include_in_schema=False)
    app.add_api_route("/usage.csv", usage_csv, methods=["GET"], include_in_schema=False)
    app.add_api_route("/requests", requests_view, methods=["GET"], include_in_schema=False)
    app.add_api_route("/settings", settings_view, methods=["GET"], include_in_schema=False)

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Any) -> Any:
        return html(request, pages.error_page(prefix=prefix(request), title="Page not found",
                                     message="There is nothing at this address."), title="Not found", status=404)

    return app


def mount_platform(app: Any, accounts: AccountsClient | None, settings: PlatformSettings | None = None,
                   **kw: Any) -> Any:
    """Mount the platform on ``app`` at ``settings.path_prefix``; returns the platform sub-app."""
    s = settings or PlatformSettings()
    sub = create_platform_app(accounts, s, **kw)
    p = "/" + s.path_prefix.strip("/")

    from starlette.responses import RedirectResponse

    async def _platform_root() -> Any:
        return RedirectResponse(p + "/", status_code=308)

    app.add_api_route(p, _platform_root, methods=["GET"], include_in_schema=False)

    app.mount(p, sub)
    return sub
