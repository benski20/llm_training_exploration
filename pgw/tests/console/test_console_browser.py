"""Real-browser smoke test of the built console against a real gateway (dev 'plain' mode, tiny model).

A gateway runs under uvicorn on a free loopback port in a background thread; headless Chromium
(Playwright) signs in through the UI and visits every page. The test fails on any browser console
error, page error, CSP violation or failed ``/admin`` / ``/console/api`` request (the pre-login
``GET /console/api/me`` 401 is the only expected one). A second browser context signs in with a
viewer token and must not see any admin action.

Skipped when the Python ``playwright`` package or its Chromium build is not installed.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from pgw.console.routes import DEFAULT_STATIC_DIR
from tests.console.conftest import VIEWER_TOKEN
from tests.gateway.conftest import ADMIN_TOKEN, plain_settings

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright (Python) is not installed")

TIMEOUT_MS = 15_000

# Collects CSP violations in the page (Playwright init scripts are not subject to the page's CSP).
_CSP_LISTENER = """
window.__cspViolations = [];
document.addEventListener("securitypolicyviolation", (e) => {
  window.__cspViolations.push(e.violatedDirective + " " + (e.blockedURI || ""));
});
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"playwright cannot start: {exc}")
    try:
        try:
            b = pw.chromium.launch()
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Chromium is not available: {exc}")
        yield b
        b.close()
    finally:
        pw.stop()


@pytest.fixture(scope="module")
def gateway_url(tiny_pack_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    import uvicorn

    from pgw.gateway import ApiKeyStore, create_app

    if not (DEFAULT_STATIC_DIR / "index.html").is_file():
        pytest.skip("console is not built (src/pgw/console/static/index.html missing)")
    tmp = tmp_path_factory.mktemp("console_browser")
    viewers = tmp / "viewers.txt"
    viewers.write_text(f"{VIEWER_TOKEN}\n", encoding="utf-8")
    settings = plain_settings(tiny_pack_dir, tmp, console_viewer_tokens_file=viewers,
                              api_keys_file=tmp / "api_keys.json", quota_state_file=tmp / "quota.json")
    app = create_app(settings, key_store=ApiKeyStore(tmp / "api_keys.json"))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, log_level="warning",
                                           lifespan="on"))
    thread = threading.Thread(target=server.run, name="console-browser-gateway", daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("gateway did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=30)


@dataclass
class Watch:
    """Problems seen by one page: console errors, page errors and failed API calls."""

    signed_in: bool = False
    problems: list[str] = field(default_factory=list)

    def attach(self, page: Any) -> None:
        page.on("console", self._console)
        page.on("pageerror", lambda e: self.problems.append(f"pageerror: {e}"))
        page.on("response", self._response)
        page.on("requestfailed", self._failed)

    @staticmethod
    def _api(url: str) -> bool:
        path = urlsplit(url).path
        return path.startswith(("/admin", "/console/api", "/v1"))

    def _console(self, m: Any) -> None:
        if m.type != "error":
            return
        where = urlsplit((m.location or {}).get("url") or "").path
        if "Failed to load resource" in m.text and where.startswith(("/admin", "/console/api", "/v1")):
            return  # the browser's echo of an HTTP error status; ``_response`` judges those
        self.problems.append(f"console.{m.type}: {m.text} ({where})")

    def _response(self, r: Any) -> None:
        if not self._api(r.url) or r.status < 400:
            return
        path = urlsplit(r.url).path
        if r.status == 401 and path == "/console/api/me" and not self.signed_in:
            return  # expected: the app probes the session before sign-in
        self.problems.append(f"HTTP {r.status} {r.request.method} {path}")

    def _failed(self, req: Any) -> None:
        err = req.failure or ""
        if self._api(req.url) and "ERR_ABORTED" not in err:  # aborted = unmounted page cancelled a poll
            self.problems.append(f"request failed: {req.method} {urlsplit(req.url).path} ({err})")


def _new_page(browser: Any) -> tuple[Any, Any, Watch]:
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    ctx.set_default_timeout(TIMEOUT_MS)
    ctx.add_init_script(_CSP_LISTENER)
    page = ctx.new_page()
    w = Watch()
    w.attach(page)
    return ctx, page, w


def _sign_in(page: Any, w: Watch, base: str, token: str) -> None:
    page.goto(f"{base}/console/")
    page.wait_for_selector(".login-card")
    page.fill("input[type=password]", token)
    w.signed_in = True  # the 401 probe happened before this point
    page.click("button[type=submit]")
    page.wait_for_selector(".shell .topbar")


def _nav(page: Any, label: str, path: str) -> None:
    """Client-side navigation through the sidebar (no reload, so no aborted polls)."""
    page.click(f".sidebar .nav-link[title='{label}']")
    page.wait_for_url(re.compile(rf"/console{re.escape(path.rstrip('/'))}/?$"))
    page.wait_for_selector(".page")
    page.wait_for_load_state("networkidle")


def _check(page: Any, w: Watch) -> None:
    csp = page.evaluate("window.__cspViolations")
    assert csp == [], f"CSP violations: {csp}"
    assert w.problems == [], "\n".join(w.problems)


def test_admin_walkthrough(browser: Any, gateway_url: str) -> None:
    ctx, page, w = _new_page(browser)
    try:
        _sign_in(page, w, gateway_url, ADMIN_TOKEN)
        assert page.inner_text(".topbar").count("Admin") >= 1
        # overview
        page.wait_for_selector(".stats-grid .stat-value")
        page.wait_for_load_state("networkidle")

        # playground: send a message, wait for the streamed reply
        _nav(page, "Playground", "/playground")
        mt = page.get_by_label("Max tokens")
        mt.fill("6")
        page.fill("#pg-input", "Hello")
        page.click("button[aria-label='Send message']")
        reply = page.locator(".msg-assistant").last
        page.wait_for_function(
            "() => { const m = [...document.querySelectorAll('.msg-assistant')].pop();"
            " return m && !m.querySelector('.caret') && (m.querySelector('.msg-body') || m.querySelector('.msg-error')); }",
            timeout=60_000,
        )
        assert reply.locator(".msg-error").count() == 0, reply.inner_text()
        assert reply.locator(".msg-body").inner_text().strip(), "empty streamed reply"

        _nav(page, "Models", "/models")
        page.wait_for_selector(".model-actions button:has-text('Reload')")  # admin model actions are visible
        _nav(page, "Masks", "/masks")

        # keys: create, see the secret once, revoke
        _nav(page, "API keys", "/keys")
        page.click("button:has-text('Create key')")
        page.fill("#create-key input[placeholder='billing-app']", "browser-smoke")
        page.click("button[type=submit][form=create-key]")
        secret = page.locator("code[aria-label='New API key']")
        secret.wait_for()
        assert secret.inner_text().startswith("pgw-")
        page.click("button:has-text('I have stored the key')")
        assert page.locator("code[aria-label='New API key']").count() == 0
        row = page.locator("tr", has_text="browser-smoke")
        row.wait_for()
        row.locator("button:has-text('Revoke')").click()
        page.click("dialog[open] button:has-text('Revoke key')")
        row.locator("text=revoked").wait_for()
        page.wait_for_load_state("networkidle")

        for label, path in (("Audit log", "/audit"), ("Security", "/security"), ("Overview", "/")):
            _nav(page, label, path)
        _check(page, w)
    finally:
        ctx.close()


def test_viewer_sees_no_admin_actions(browser: Any, gateway_url: str) -> None:
    ctx, page, w = _new_page(browser)
    try:
        _sign_in(page, w, gateway_url, VIEWER_TOKEN)
        assert "Viewer" in page.inner_text(".topbar")
        page.wait_for_load_state("networkidle")
        _nav(page, "API keys", "/keys")
        page.wait_for_selector("text=Read-only")
        assert page.locator("button:has-text('Create key')").count() == 0
        assert page.locator("button:has-text('Revoke')").count() == 0
        _nav(page, "Models", "/models")
        page.wait_for_selector(".model-head")
        assert page.locator(".model-actions").count() == 0
        _nav(page, "Playground", "/playground")
        page.wait_for_selector("text=Read-only session")
        assert page.locator("#pg-input").is_disabled()
        for label, path in (("Masks", "/masks"), ("Audit log", "/audit"), ("Security", "/security")):
            _nav(page, label, path)
        _check(page, w)
        # the server enforces the role too: a viewer session cannot create keys
        csrf = page.evaluate("fetch('/console/api/me').then(r => r.json()).then(j => j.csrf)")
        status = page.evaluate(
            "(csrf) => fetch('/admin/keys', {method: 'POST', headers: {'content-type': 'application/json',"
            " 'X-PGW-CSRF': csrf}, body: JSON.stringify({name: 'x'})}).then(r => r.status)", csrf)
        assert status == 403
        # the watchers themselves work: the refused call and an inline script are reported
        page.wait_for_timeout(200)
        assert any("HTTP 403 POST /admin/keys" in p for p in w.problems), w.problems
        page.evaluate("() => { const s = document.createElement('script'); s.textContent = '1';"
                      " document.body.appendChild(s); }")
        page.wait_for_function("() => window.__cspViolations.length > 0")
    finally:
        ctx.close()
