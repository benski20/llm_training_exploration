"""End to end on CPU: ``pgw chat``'s server in front of the client library against a real provider process.

The provider is the tiny CPU test provider of the SDK tests (:mod:`tests.sdk.provider_main`: TLS, TCP+TLS lane, masked
LM head, accounts). A message sent through the chat server's HTTP API (and, when Chromium is available, typed into
the real web app) must stream back exactly the answer the SDK gives for the same conversation: bit-identical text,
turn after turn. The chat server must not put the prompt in any log.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from pgw.chat.server import DEFAULT_STATIC_DIR, ChatServer
from pgw.chat.store import DiskStore
from tests.chat.conftest import parse_sse, text_of
from tests.sdk.conftest import MODEL, Provider, _restore_native_env, pki, provider, sdk_pack_dir  # noqa: F401

pytestmark = pytest.mark.timeout(1200)

GREEDY: dict[str, Any] = {"temperature": 0, "extra_body": {"ignore_eos": True}}
MAX_TOKENS = 10


@pytest.fixture(scope="module")
def chat_env(provider: Provider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:  # noqa: F811
    from pgw import OpenAI
    from pgw.chat.backend import SDKBackend
    from pgw.sdk import LocalPackProvisioner

    root = tmp_path_factory.mktemp("chat-e2e")
    client = OpenAI(api_key=provider.new_key("chat-e2e"), provisioner=LocalPackProvisioner({MODEL: provider.pack_dir}),
                    cache_dir=root / "cache", state_dir=root / "state", mask_scheme="plain", mask_workers=1,
                    mask_target_slots=400, mask_low_water_slots=200, mask_segment_slots=128, timeout=300,
                    ready_timeout_s=600)
    backend = SDKBackend(client, request_overrides=GREEDY)
    history = root / "state" / "chat" / "conversations"
    server = ChatServer(backend, DiskStore(history), default_model=MODEL, max_tokens=MAX_TOKENS,
                        history_dir=history).start()
    http = httpx.Client(base_url=f"http://127.0.0.1:{server.port}", timeout=600, trust_env=False)
    env = {"client": client, "server": server, "http": http, "root": root}
    try:
        yield env
    finally:
        http.close()
        server.shutdown()
        client.close()


def _h(server: ChatServer) -> dict[str, str]:
    return {"Authorization": f"Bearer {server.token}", "Origin": f"http://127.0.0.1:{server.port}",
            "Content-Type": "application/json"}


def _sdk(client: Any, messages: list[dict[str, str]]) -> str:
    chunks = client.chat.completions.create(model=MODEL, messages=messages, stream=True, max_tokens=MAX_TOKENS,
                                            **GREEDY)
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)


def test_chat_through_the_server_is_bit_identical_to_the_sdk(chat_env: dict[str, Any],
                                                              caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    server, http, client = chat_env["server"], chat_env["http"], chat_env["client"]
    http.post("/api/warmup", headers=_h(server), json={"model": MODEL})
    prompt = "Hello there, who are you? (private-e2e-marker-9137)"
    r = http.post("/api/chat", headers=_h(server), json={"model": MODEL, "message": prompt})
    assert r.status_code == 200, r.text
    ev = parse_sse(r.text)
    assert ev[0]["type"] == "start" and ev[-1]["type"] == "done", ev[-1]
    deltas = [e for e in ev if e["type"] == "delta"]
    assert len(deltas) >= 2, "the answer must arrive streamed, token by token"
    got = text_of(ev)
    msg = ev[-1]["message"]
    assert msg["verified"] is True and msg["content"] == got
    assert msg["usage"]["completion_tokens"] == MAX_TOKENS
    want = _sdk(client, [{"role": "user", "content": prompt}])
    assert got == want  # bit for bit
    # the second turn sends the whole conversation: still identical to the SDK
    cid = ev[0]["conversation_id"]
    r2 = http.post("/api/chat", headers=_h(server), json={"model": MODEL, "message": "And then?",
                                                          "conversation_id": cid})
    got2 = text_of(parse_sse(r2.text))
    want2 = _sdk(client, [{"role": "user", "content": prompt}, {"role": "assistant", "content": got},
                          {"role": "user", "content": "And then?"}])
    assert got2 == want2
    st = http.get(f"/api/status?model={MODEL}", headers=_h(server)).json()
    assert st["phase"] == "ready" and st["verified_requests"] >= 2 and st["integrity_failures"] == 0
    assert st["masked_head"] is True
    assert "private-e2e-marker-9137" not in caplog.text
    conv_file = Path(chat_env["root"]) / "state" / "chat" / "conversations" / f"{cid}.json"
    assert conv_file.stat().st_mode & 0o777 == 0o600


def test_typed_into_the_web_app(chat_env: dict[str, Any]) -> None:
    """The built app in headless Chromium: type, send, see the streamed answer; it equals the SDK's."""
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright (Python) is not installed")
    if not (DEFAULT_STATIC_DIR / "index.html").is_file():
        pytest.skip("chat web app not built")
    server, http, client = chat_env["server"], chat_env["http"], chat_env["client"]
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"playwright cannot start: {exc}")
    try:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"chromium cannot start: {exc}")
        page = browser.new_page()
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server.open_url)
        page.wait_for_selector(".indicator")
        page.fill("textarea", "Tell me a story")
        page.keyboard.press("Enter")
        page.wait_for_selector(".msg-assistant .private-tag:not(.is-live)", timeout=300_000)
        cid = page.evaluate("location.pathname").split("/")[-1]
        conv = http.get(f"/api/conversations/{cid}", headers=_h(server)).json()
        answer = conv["messages"][-1]["content"]
        assert conv["messages"][-1]["verified"] is True
        assert answer == _sdk(client, [{"role": "user", "content": "Tell me a story"}])
        assert not errors, errors
        browser.close()
    finally:
        pw.stop()
