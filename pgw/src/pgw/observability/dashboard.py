"""Self-contained ops dashboard page (HTML + JS polling ``/metrics`` and ``/admin/status``).

The page has no external assets (works air-gapped, compatible with a strict
``Content-Security-Policy: default-src 'self' 'unsafe-inline'``). It polls the Prometheus text
exposition of the process that serves it and derives:

* mask inventory: slots remaining, burn rate (slots/s), estimated time to exhaustion, low water,
* throughput: completion and prompt tokens/s, API requests/s and error share,
* latency: TTFT, TPOT and masked round trip (mean and p50/p95 from histogram deltas),
* integrity: Freivalds / range-check failures (total and in the last interval),
* noise monitor: alarm state per model, and (with an admin token) the monitor's statistics,
* compute service and factory families when the page is served by those processes.

The admin token (optional, for ``/admin/status`` and an admin-protected ``/metrics``) is kept in
``sessionStorage`` only and sent as a bearer token to the same origin. All data is inserted with
``textContent``; label values are never interpreted as HTML.

The JavaScript between ``/*PGW_CORE_BEGIN*/`` and ``/*PGW_CORE_END*/`` is pure (no DOM) so it can be
unit-tested with Node.
"""

from __future__ import annotations

import functools
import html
import json
import re
from importlib import resources
from typing import Any

__all__ = ["DASHBOARD_CSP", "core_js", "dashboard_html", "mount_dashboard"]

#: CSP the page is designed for (inline script/style, same-origin fetches only).
DASHBOARD_CSP = "default-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'"


@functools.cache
def _asset(name: str) -> str:
    """Packaged asset text (``src/pgw/observability/static``)."""
    return resources.files("pgw.observability").joinpath("static", name).read_text(encoding="utf-8")


def _js_string_config(cfg: dict[str, Any]) -> str:
    # json.dumps output is valid JS; escape "</" and HTML-significant characters so it cannot
    # terminate the <script> element.
    s = json.dumps(cfg, separators=(",", ":"))
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _check_same_origin(url: str) -> None:
    r"""Reject URLs that could resolve to another origin (the admin token is sent along).

    Browsers treat ``\`` like ``/`` and strip tabs/newlines, so ``/\evil.example`` or
    ``/<TAB>/evil.example`` would be protocol-relative; any ``scheme:`` prefix is rejected too.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("dashboard URLs must be non-empty strings")
    bad = "\\" in url or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url)
    if bad or url.startswith("//") or "://" in url or _SCHEME.match(url):
        raise ValueError(f"dashboard URLs must be same-origin relative paths, got {url!r}")


def core_js() -> str:
    """The DOM-free JavaScript core (parsing and derived metrics), for tests and reuse."""
    return _asset("dashboard_core.js")


def dashboard_html(
    title: str = "PGW",
    *,
    metrics_url: str = "metrics",
    status_url: str | None = "admin/status",
    refresh_seconds: float = 5.0,
) -> str:
    """Return the dashboard page (no external assets; works air-gapped).

    Args:
        title: Page title (HTML-escaped).
        metrics_url: Prometheus exposition URL, relative to the page (default: sibling ``metrics``,
            i.e. ``/metrics`` for a page at ``/dashboard``).
        status_url: Admin status URL (queried only once an admin token is entered); ``None``
            disables the model table.
        refresh_seconds: Poll interval.

    Raises:
        ValueError: A URL is absolute to another origin, or the interval is not positive.
    """
    for u in (metrics_url, status_url):
        if u is not None:
            _check_same_origin(u)
    if not refresh_seconds > 0:
        raise ValueError("refresh_seconds must be positive")
    cfg = {"metricsUrl": metrics_url, "statusUrl": status_url, "refreshSeconds": float(refresh_seconds)}
    subs = {
        "__CORE__": core_js().strip(),
        "__CONFIG__": _js_string_config(cfg),
        "__TITLE__": html.escape(title, quote=True),
    }
    return re.sub(r"__(?:CORE|CONFIG|TITLE)__", lambda m: subs[m.group(0)], _asset("dashboard.html"))


def mount_dashboard(
    app: Any,
    path: str = "/dashboard",
    *,
    title: str = "PGW",
    metrics_url: str | None = None,
    status_url: str | None = "admin/status",
    refresh_seconds: float = 5.0,
) -> None:
    """Add the dashboard route to a FastAPI (or Starlette) app.

    The page is rendered once. It is served with ``Cache-Control: no-store``, ``X-Frame-Options``
    and :data:`DASHBOARD_CSP`. ``metrics_url`` defaults to the sibling ``metrics`` of ``path``.
    """
    if not path.startswith("/"):
        raise ValueError(f"path must start with '/', got {path!r}")
    from starlette.responses import HTMLResponse

    page = dashboard_html(
        title, metrics_url=metrics_url or "metrics", status_url=status_url, refresh_seconds=refresh_seconds
    )
    headers = {
        "Content-Security-Policy": DASHBOARD_CSP,
        "Cache-Control": "no-store",
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }

    if hasattr(app, "add_api_route"):  # FastAPI: endpoint without parameters

        async def dashboard() -> Any:
            return HTMLResponse(page, headers=headers)

        app.add_api_route(path, dashboard, methods=["GET"], include_in_schema=False)
    else:  # Starlette: endpoint receives the request

        async def dashboard_endpoint(request: Any) -> Any:
            return HTMLResponse(page, headers=headers)

        app.add_route(path, dashboard_endpoint, methods=["GET"])
