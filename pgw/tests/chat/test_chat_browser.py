"""Real-browser smoke test of the built chat app (headless Chromium, Python Playwright) against ``pgw chat``'s server
with the fake backend: token hand-over through the fragment, no CSP violation or console error, send / stream /
stop / regenerate / delete, the privacy screen, typed error states, and a tab without the token that can do
nothing. Skipped when Playwright or its Chromium is missing, or the app is not built."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pgw.chat.server import DEFAULT_STATIC_DIR, ChatServer
from pgw.chat.store import DiskStore
from pgw.chat.testing import FakeBackend

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright (Python) is not installed")

TIMEOUT_MS = 15_000
_CSP_LISTENER = """
window.__csp = [];
document.addEventListener("securitypolicyviolation", (e) => window.__csp.push(e.violatedDirective + " " + e.blockedURI));
"""


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    if not (DEFAULT_STATIC_DIR / "index.html").is_file():
        pytest.skip("chat web app not built (scripts/build-chat.sh)")
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"playwright cannot start: {exc}")
    try:
        try:
            b = pw.chromium.launch()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"chromium cannot start: {exc}")
        yield b
        b.close()
    finally:
        pw.stop()


@pytest.fixture
def server(tmp_path: Path) -> Iterator[ChatServer]:
    s = ChatServer(FakeBackend(delay=0.01), DiskStore(tmp_path / "conv"), default_model="qwen2.5-1.5b-instruct",
                   history_dir=tmp_path / "conv").start()
    try:
        yield s
    finally:
        s.shutdown()


def _page(browser: Any) -> tuple[Any, Any, list[str]]:
    ctx = browser.new_context()
    ctx.add_init_script(_CSP_LISTENER)
    page = ctx.new_page()
    page.set_default_timeout(TIMEOUT_MS)
    problems: list[str] = []
    page.on("console", lambda m: problems.append(f"console: {m.text}") if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.on("requestfailed", lambda r: problems.append(f"failed: {r.url}") if "/api/chat" not in r.url else None)
    return ctx, page, problems


def test_chat_flow_in_a_real_browser(browser: Any, server: ChatServer) -> None:
    ctx, page, problems = _page(browser)
    external: list[str] = []
    page.on("request", lambda r: external.append(r.url) if not r.url.startswith(server.url) else None)
    page.goto(server.open_url)
    page.wait_for_selector(".welcome-title")
    assert "token" not in page.url  # moved to sessionStorage, out of the address bar and history
    assert page.evaluate("sessionStorage.getItem('pgw-chat-token')") == server.token
    page.wait_for_selector(".indicator .indicator-label:text('Private')")

    page.fill("textarea", "What is masking?")
    page.keyboard.press("Enter")
    page.wait_for_selector(".msg-assistant .private-tag:not(.is-live)")
    text = page.inner_text(".msg-assistant .prose")
    assert "Every linear layer ran on masked data" in text
    assert page.locator(".msg-assistant pre code").inner_text() == "print('hello from a canned answer')"
    assert page.url.startswith(server.url + "c/c_")
    page.wait_for_selector(".conv-title:text('What is masking?')")

    # stop a slow answer
    page.fill("textarea", "/slow")
    page.keyboard.press("Enter")
    page.wait_for_selector(".is-live .prose")
    page.click(".send-btn.is-stop")
    page.wait_for_selector(".meta-item:text('Stopped')")

    # regenerate the last answer
    page.click("[aria-label='Regenerate answer']")
    page.locator(".msg-assistant .private-tag:not(.is-live)").nth(1).wait_for()  # (string evaluation is CSP-blocked)

    # a typed error from the library
    page.fill("textarea", "/error integrity")
    page.keyboard.press("Enter")
    page.wait_for_selector(".notice-bad .notice-title:text('Integrity check failed: answer discarded')")

    # privacy screen
    page.click(".indicator")
    page.wait_for_selector(".privacy .fact[data-fact='device']")
    assert page.locator(".privacy .fact").count() >= 5
    page.keyboard.press("Escape")

    # delete the conversation
    page.click(".conv-item.is-current .conv-del")
    page.click(".conv-confirm .danger")
    page.wait_for_selector(".welcome-title")
    assert server.store.summaries() == []

    assert page.evaluate("window.__csp") == []
    assert not problems, problems
    assert not external, external  # nothing but this server is ever contacted
    ctx.close()


def test_a_tab_without_the_token_can_do_nothing(browser: Any, server: ChatServer) -> None:
    ctx, page, problems = _page(browser)
    page.goto(server.url)
    page.wait_for_selector("text=Open PGW Chat from your terminal")
    status = page.evaluate("fetch('/api/conversations').then(r => r.status)")
    assert status == 401
    status = page.evaluate("""fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
                               body: JSON.stringify({model: 'qwen2.5-1.5b-instruct', message: 'x'})}).then(r => r.status)""")
    assert status == 401
    assert server.backend.requests == []  # type: ignore[attr-defined]
    assert page.evaluate("window.__csp") == []
    assert not [p for p in problems if "401" not in p], problems  # the refused calls themselves log a 401
    ctx.close()


def test_inline_script_is_blocked_by_the_csp(browser: Any, server: ChatServer) -> None:
    ctx, page, _ = _page(browser)
    page.goto(server.open_url)
    page.wait_for_selector(".welcome-title")
    page.evaluate("""() => { const s = document.createElement('script'); s.textContent = 'window.__ran = 1';
                             document.body.appendChild(s); }""")
    page.wait_for_timeout(200)
    assert page.evaluate("window.__ran") is None
    assert any("script-src" in v for v in page.evaluate("window.__csp"))
    ctx.close()
