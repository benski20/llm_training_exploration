"""Security checks of the ``pgw chat`` localhost server: bind address, per-launch token, Host / Origin / fetch-site
checks (DNS rebinding, CSRF), content types, body limits, CSP and the other headers, no content in logs, private
files."""

from __future__ import annotations

import logging
import os
import socket
import stat
import time
from pathlib import Path

import httpx
import pytest

from pgw.chat.server import CSP, ChatServer
from pgw.chat.store import DiskStore, MemoryStore
from pgw.chat.testing import FakeBackend
from tests.chat.conftest import MODEL, Running, parse_sse, text_of


def test_refuses_any_bind_address_but_loopback() -> None:
    for host in ("0.0.0.0", "::", "192.168.1.10", "localhost"):  # noqa: S104 - refused, never bound
        with pytest.raises(ValueError, match=r"127\.0\.0\.1"):
            ChatServer(FakeBackend(), MemoryStore(), host=host)


def test_binds_loopback_and_is_not_reachable_on_other_addresses(running: Running) -> None:
    host, port = running.server.httpd.server_address[:2]
    assert host == "127.0.0.1"
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except OSError:
        pass
    for a in addrs - {"127.0.0.1"}:
        if a.startswith("127."):
            continue
        with socket.socket() as s:
            s.settimeout(1.0)
            assert s.connect_ex((a, port)) != 0, f"reachable on {a}"


def test_token_is_random_per_launch_and_long() -> None:
    a = ChatServer(FakeBackend(), MemoryStore())
    b = ChatServer(FakeBackend(), MemoryStore())
    try:
        assert a.token != b.token and len(a.token) >= 43
        assert a.open_url.endswith("#token=" + a.token)  # in the fragment: never sent to a server
        assert "token" not in a.url
    finally:
        a.httpd.server_close()
        b.httpd.server_close()
    with pytest.raises(ValueError):
        ChatServer(FakeBackend(), MemoryStore(), token="short")


@pytest.mark.parametrize("path", ["/api/info", "/api/conversations", f"/api/status?model={MODEL}"])
def test_every_api_call_needs_the_token(running: Running, path: str) -> None:
    assert running.http.get(path).status_code == 401
    assert running.http.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert running.http.get(path, headers={"Authorization": running.server.token}).status_code == 401  # no scheme
    r = running.http.get(path, headers={"Authorization": f"Bearer {running.server.token}"})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"


def test_post_without_token_is_refused_and_starts_nothing(running: Running) -> None:
    h = running.h()
    del h["Authorization"]
    r = running.http.post("/api/chat", headers=h, json={"model": MODEL, "message": "hi"})
    assert r.status_code == 401
    assert running.backend.requests == []


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1", "localhost", "127.0.0.1:1", "attacker.test:{port}",
                                  "127.0.0.1.nip.io:{port}"])
def test_dns_rebinding_host_header_is_refused_everywhere(running: Running, host: str) -> None:
    host = host.format(port=running.server.port)
    for path in ("/", "/api/info", "/assets/app-abcdef12.js"):
        r = running.http.get(path, headers={"Host": host, "Authorization": f"Bearer {running.server.token}"})
        assert r.status_code == 421, (host, path)
        assert r.headers["content-security-policy"] == CSP


def test_localhost_name_is_accepted(running: Running) -> None:
    r = running.http.get("/api/info", headers={"Host": f"localhost:{running.server.port}",
                                               "Authorization": f"Bearer {running.server.token}"})
    assert r.status_code == 200


@pytest.mark.parametrize("origin", ["https://evil.example", "http://127.0.0.1:1", "null", "http://localhost"])
def test_foreign_origin_is_refused(running: Running, origin: str) -> None:
    for method, path in (("GET", "/api/info"), ("POST", "/api/chat"), ("DELETE", "/api/conversations")):
        r = running.http.request(method, path, headers=running.h(Origin=origin), json={"model": MODEL, "message": "x"})
        assert r.status_code == 403, (origin, method, path)
    assert running.backend.requests == []


def test_state_changing_calls_need_origin_and_json(running: Running) -> None:
    h = running.h()
    del h["Origin"]
    assert running.http.post("/api/chat", headers=h, json={"model": MODEL, "message": "x"}).status_code == 403
    assert running.http.delete("/api/conversations", headers=h).status_code == 403
    h = running.h(**{"Content-Type": "text/plain"})
    assert running.http.post("/api/chat", headers=h, content=b'{"model":"x","message":"y"}').status_code == 415
    h = running.h(**{"Content-Type": "application/x-www-form-urlencoded"})
    assert running.http.post("/api/chat", headers=h, content=b"message=y").status_code == 415
    assert running.backend.requests == []


def test_cross_site_fetch_metadata_is_refused(running: Running) -> None:
    for site in ("cross-site", "same-site"):
        r = running.http.get("/api/info", headers=running.h(**{"Sec-Fetch-Site": site}))
        assert r.status_code == 403
    assert running.http.get("/api/info", headers=running.h(**{"Sec-Fetch-Site": "same-origin"})).status_code == 200


def test_cors_preflight_is_refused(running: Running) -> None:
    r = running.http.options("/api/chat", headers={"Origin": "https://evil.example",
                                                   "Access-Control-Request-Method": "POST"})
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers


def test_security_headers_on_every_response(running: Running) -> None:
    for r in (running.http.get("/"), running.http.get("/assets/app-abcdef12.js"),
              running.http.get("/api/info", headers=running.h()), running.http.get("/nope.js"),
              running.http.get("/api/info")):
        assert r.headers["content-security-policy"] == CSP
        assert "script-src 'self'" in CSP and "unsafe-inline" not in CSP and "default-src 'none'" in CSP
        assert "frame-ancestors 'none'" in CSP and "connect-src 'self'" in CSP
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["cross-origin-opener-policy"] == "same-origin"
        assert "access-control-allow-origin" not in r.headers
    assert running.http.get("/assets/app-abcdef12.js").headers["cache-control"].endswith("immutable")


def test_static_files_and_traversal(running: Running) -> None:
    assert running.http.get("/").status_code == 200
    assert running.http.get("/c/c_0123456789abcdef01234567").text.startswith("<!doctype html>")
    for p in ("/../server.py", "/%2e%2e/%2e%2e/etc/passwd", "/assets/../../conftest.py", "/.hidden", "/etc/passwd"):
        assert running.http.get(p).status_code == 404, p
    assert running.http.post("/", headers=running.h(), content=b"{}").status_code == 405


def test_body_limits_and_validation(running: Running) -> None:
    big = "x" * 40_000
    r = running.http.post("/api/chat", headers=running.h(), json={"model": MODEL, "message": big})
    assert r.status_code == 413
    r = running.http.post("/api/chat", headers=running.h(), content=b"x" * (2 << 20))
    assert r.status_code == 413
    for body in ({"model": "gpt-4", "message": "hi"}, {"model": MODEL}, {"model": MODEL, "message": "  "},
                 {"model": MODEL, "message": "hi", "temperature": 3}, {"model": MODEL, "message": "hi", "seed": "x"},
                 {"model": MODEL, "message": "hi", "max_tokens": 0},
                 {"model": MODEL, "message": "hi", "conversation_id": "../../etc"}):
        r = running.http.post("/api/chat", headers=running.h(), json=body)
        assert r.status_code in (400, 404), (body, r.status_code)
    assert running.backend.requests == []


def test_chat_streams_and_stores_privately(running: Running, tmp_path: Path) -> None:
    ev = running.chat(message="Hello there")
    assert ev[0]["type"] == "start" and ev[-1]["type"] == "done"
    assert text_of(ev).startswith("Here is a short answer")
    msg = ev[-1]["message"]
    assert msg["verified"] is True and msg["status"] == "done" and msg["content"] == text_of(ev)
    cid = ev[0]["conversation_id"]
    root = tmp_path / "state" / "chat" / "conversations"
    f = root / f"{cid}.json"
    assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600
    conv = running.http.get(f"/api/conversations/{cid}", headers=running.h()).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"] and conv["title"] == "Hello there"
    # the second turn sends the whole conversation to the SDK
    running.chat(conversation_id=cid, message="And again")
    assert running.backend.requests[-1]["n_messages"] == 3
    assert running.http.delete(f"/api/conversations/{cid}", headers=running.h()).json() == {"deleted": True}
    assert not f.exists()


def test_no_content_in_logs_or_stderr(running: Running, caplog: pytest.LogCaptureFixture,
                                      capfd: pytest.CaptureFixture[str]) -> None:
    marker = "my-very-private-diagnosis-4711"
    caplog.set_level(logging.DEBUG)
    running.chat(message=marker)
    running.chat(message="/error integrity")
    out, err = capfd.readouterr()
    assert marker not in caplog.text and marker not in out and marker not in err
    assert running.server.token not in caplog.text and running.server.token not in err


def test_stop_and_closed_tab_stop_generation(running: Running) -> None:
    import threading

    got: dict[str, list] = {}

    def go() -> None:
        r = running.http.post("/api/chat", headers=running.h(), json={"model": MODEL, "message": "/slow"},
                              timeout=60)
        got["ev"] = parse_sse(r.text)

    t = threading.Thread(target=go)
    t.start()
    gen = None
    for _ in range(100):
        time.sleep(0.05)
        with running.server._gmu:
            gens = list(running.server._gens)
        if gens:
            gen = gens[0]
            break
    assert gen is not None
    time.sleep(0.5)
    assert running.http.post(f"/api/chat/{gen}/stop", headers=running.h(), content=b"{}").json() == {"stopped": True}
    t.join(10)
    assert got["ev"][-1]["type"] == "done" and got["ev"][-1]["message"]["status"] == "stopped"
    assert running.backend.streams[-1].closed.is_set()
    # a closed tab: the client goes away mid-answer
    with httpx.Client(base_url=running.base, timeout=30, trust_env=False) as c, c.stream(
            "POST", "/api/chat", headers=running.h(), json={"model": MODEL, "message": "/slow"}) as r:
        for line in r.iter_lines():
            if '"delta"' in line:
                break
    deadline = time.monotonic() + 10
    while not running.backend.streams[-1].closed.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert running.backend.streams[-1].closed.is_set()


def test_typed_errors_are_mapped(running: Running) -> None:
    for kind, want in (("integrity", "integrity"), ("stalled", "provider_stalled"), ("connection", "connection"),
                       ("not_ready", "not_ready"), ("auth", "auth")):
        ev = running.chat(message=f"/error {kind}")
        assert ev[-1]["type"] == "error" and ev[-1]["error"]["kind"] == want
    ev = running.chat(message="/error integrity")
    assert ev[-1]["error"]["retryable"] is False and "discarded" in ev[-1]["error"]["message"]


def test_one_answer_per_conversation_and_regenerate(running: Running) -> None:
    ev = running.chat(message="first")
    cid = ev[0]["conversation_id"]
    ev2 = running.chat(conversation_id=cid, regenerate=True)
    assert ev2[-1]["type"] == "done"
    conv = running.http.get(f"/api/conversations/{cid}", headers=running.h()).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]


def test_disk_store_refuses_bad_ids_and_symlinks(tmp_path: Path) -> None:
    st = DiskStore(tmp_path / "h")
    c = st.create(MODEL)
    assert st.get("../../x") is None and st.get("c_" + "0" * 24) is None
    with pytest.raises(ValueError):
        st._path("../evil")
    (tmp_path / "outside.json").write_text('{"id": "c_%s", "messages": []}' % ("1" * 24))
    os.symlink(tmp_path / "outside.json", tmp_path / "h" / ("c_" + "1" * 24 + ".json"))
    assert [s["id"] for s in st.summaries()] == [c["id"]]
    assert st.delete_all() == 1
