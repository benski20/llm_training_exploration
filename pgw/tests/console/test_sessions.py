"""Console sign-in: login/logout/me, roles, CSRF, Bearer compatibility, rate limit, expiry, cookie flags."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pgw.console.sessions import ConsoleSessions, LoginRateLimiter, cookie_secure, is_loopback_host
from tests.console.conftest import LOCAL, VIEWER_TOKEN, client, csrf, login, make_console_app
from tests.gateway.conftest import ADMIN, ADMIN_TOKEN


def _audit(tmp_path: Path) -> list[dict]:
    return [json.loads(ln) for ln in (tmp_path / "audit.jsonl").read_text().splitlines() if ln.strip()]


async def test_login_me_logout(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path)) as c:
        assert (await c.get("/console/api/me")).status_code == 401
        r = await c.post("/console/api/login", json={"token": ADMIN_TOKEN})
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "admin" and len(body["csrf"]) >= 32
        cookie = r.headers["set-cookie"]
        assert cookie.startswith("pgw_console=")
        low = cookie.lower()
        assert "httponly" in low and "samesite=strict" in low and "path=/" in low
        assert "secure" not in low  # plain HTTP to loopback
        sid = cookie.split(";")[0].split("=", 1)[1]
        assert len(sid) >= 43  # 256-bit id
        me = (await c.get("/console/api/me")).json()
        assert me["role"] == "admin" and me["csrf"] == body["csrf"] and me["version"]
        assert isinstance(me["expires_at"], int)
        # logout needs the CSRF token
        assert (await c.post("/console/api/logout")).status_code == 403
        r = await c.post("/console/api/logout", headers=csrf(body["csrf"]))
        assert r.status_code == 204 and "pgw_console=" in r.headers["set-cookie"]
        assert (await c.get("/console/api/me")).status_code == 401
        # the old cookie is dead server-side too
        c.cookies.set("pgw_console", sid)
        assert (await c.get("/console/api/me")).status_code == 401
        assert (await c.get("/admin/status")).status_code == 401
    events = [e for e in _audit(tmp_path) if e["event"].startswith("console.")]
    assert [e["event"] for e in events] == ["console.login", "console.logout"]
    assert events[0]["key_id"].startswith("console-admin:") and events[0]["client"] == "127.0.0.1"
    assert ADMIN_TOKEN not in (tmp_path / "audit.jsonl").read_text()
    assert sid not in (tmp_path / "audit.jsonl").read_text()


async def test_bad_token_is_audited_without_token(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path)) as c:
        r = await c.post("/console/api/login", json={"token": "wrong-token-value-xyz"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthenticated"
        assert "set-cookie" not in r.headers
        assert (await c.post("/console/api/login", json={})).status_code == 400
        assert (await c.post("/console/api/login", content=b"nope")).status_code == 400
    text = (tmp_path / "audit.jsonl").read_text()
    assert "console.login_failed" in text and "wrong-token-value-xyz" not in text


async def test_viewer_is_read_only(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path)) as c:
        tok = await login(c, VIEWER_TOKEN)
        assert (await c.get("/console/api/me")).json()["role"] == "viewer"
        for path in ("/admin/status", "/admin/masks", "/admin/keys", "/admin/overview", "/admin/usage",
                     "/admin/audit"):
            r = await c.get(path)
            assert r.status_code == 200, (path, r.text)
        r = await c.post("/admin/keys", json={"name": "x"}, headers=csrf(tok))
        assert r.status_code == 403 and "read-only" in r.json()["error"]["message"]
        assert (await c.delete("/admin/keys/0123456789ab", headers=csrf(tok))).status_code == 403
        assert (await c.post("/admin/models/tiny/reset", headers=csrf(tok))).status_code == 403
        r = await c.post("/console/api/playground/chat", headers=csrf(tok),
                         json={"model": "tiny", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 403
        # viewers can sign themselves out
        assert (await c.post("/console/api/logout", headers=csrf(tok))).status_code == 204


async def test_admin_cookie_writes_need_csrf(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path)) as c:
        tok = await login(c)
        r = await c.post("/admin/keys", json={"name": "x"})
        assert r.status_code == 403 and "CSRF" in r.json()["error"]["message"]
        assert (await c.post("/admin/keys", json={"name": "x"}, headers=csrf("wrong"))).status_code == 403
        r = await c.post("/admin/keys", json={"name": "x"}, headers=csrf(tok))
        assert r.status_code == 201, r.text
        kid = r.json()["record"]["key_id"]
        assert (await c.delete(f"/admin/keys/{kid}")).status_code == 403
        assert (await c.delete(f"/admin/keys/{kid}", headers=csrf(tok))).status_code == 200
        assert (await c.post("/admin/models/tiny/reset", headers=csrf(tok))).status_code == 200
    created = next(e for e in _audit(tmp_path) if e["event"] == "admin.key_created")
    assert created["key_id"].startswith("console-admin:")


async def test_bearer_admin_unchanged(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path)) as c:
        assert (await c.get("/admin/overview", headers=ADMIN)).status_code == 200
        r = await c.post("/admin/keys", json={"name": "x"}, headers=ADMIN)  # no CSRF needed for Bearer
        assert r.status_code == 201
        assert (await c.get("/admin/status")).status_code == 401
        # the viewer token is a console credential only, not a Bearer admin token
        assert (await c.get("/admin/status", headers={"Authorization": f"Bearer {VIEWER_TOKEN}"})).status_code == 401
        # a garbage cookie is rejected
        c.cookies.set("pgw_console", "forged")
        assert (await c.get("/admin/status")).status_code == 401


async def test_login_rate_limited(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path)) as c:
        for _ in range(10):
            assert (await c.post("/console/api/login", json={"token": "bad-token"})).status_code == 401
        r = await c.post("/console/api/login", json={"token": ADMIN_TOKEN})
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_login_rate_limiter_window() -> None:
    now = [0.0]
    lim = LoginRateLimiter(2, 60.0, clock=lambda: now[0])
    lim.check("a")
    lim.check("a")
    lim.check("b")
    with pytest.raises(Exception, match="too many"):
        lim.check("a")
    now[0] = 61.0
    lim.check("a")


async def test_session_idle_and_absolute_expiry(tiny_pack_dir: Path, tmp_path: Path) -> None:
    app = make_console_app(tiny_pack_dir, tmp_path)
    store = app.state.pgw.console
    now = [1_000_000.0]
    store.clock = lambda: now[0]
    async with client(app) as c:
        await login(c)
        now[0] += 7.9 * 3600
        assert (await c.get("/console/api/me")).status_code == 200  # refreshes the idle timer
        now[0] += 7.9 * 3600
        assert (await c.get("/admin/status")).status_code == 200
        now[0] += 8.1 * 3600
        assert (await c.get("/console/api/me")).status_code == 401  # idle
        await login(c)
        for _ in range(23):
            now[0] += 3600
            assert (await c.get("/console/api/me")).status_code == 200
        now[0] += 3600
        assert (await c.get("/console/api/me")).status_code == 401  # absolute 24 h


def test_session_store_cap_and_hashing() -> None:
    s = ConsoleSessions("a-very-long-admin-token", max_sessions=3)
    sids = [s.create("admin")[0] for _ in range(5)]
    assert len(s) == 3 and s.get(sids[0]) is None and s.get(sids[-1]) is not None
    assert all(sid not in repr(s._sessions) for sid in sids)
    assert s.authenticate("a-very-long-admin-token") == "admin" and s.authenticate("x") is None
    assert ConsoleSessions(None).authenticate("anything") is None


async def test_cookie_secure_flag(tiny_pack_dir: Path, tmp_path: Path) -> None:
    for base, secure in (("https://127.0.0.1", True), ("http://gateway.example", True), (LOCAL, False),
                         ("http://localhost:8080", False)):
        app = make_console_app(tiny_pack_dir, tmp_path / base.split("//")[1].replace(":", "_"))
        async with client(app, base) as c:
            r = await c.post("/console/api/login", json={"token": ADMIN_TOKEN})
            assert ("secure" in r.headers["set-cookie"].lower()) is secure, base
    assert cookie_secure("http", "127.0.0.1", "10.0.0.5")  # remote client via a loopback Host header
    assert not cookie_secure("http", "[::1]", "::1")
    assert is_loopback_host("localhost") and not is_loopback_host("example.com")


async def test_login_disabled_without_admin_token(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path, admin_token=None)) as c:
        r = await c.post("/console/api/login", json={"token": VIEWER_TOKEN})
        assert r.status_code == 503 and "disabled" in r.json()["error"]["message"]
        assert (await c.get("/console/api/me")).status_code == 503


async def test_console_disabled(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path, console_enabled=False)) as c:
        assert (await c.get("/console/")).status_code == 404
        assert (await c.post("/console/api/login", json={"token": ADMIN_TOKEN})).status_code == 404
        assert (await c.get("/admin/overview", headers=ADMIN)).status_code == 200
        c.cookies.set("pgw_console", "anything")
        assert (await c.get("/admin/status")).status_code == 401


async def test_admin_token_file(tiny_pack_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``admin_token_file`` (a mounted secret) signs in like ``admin_token``; misuse fails at start-up."""
    import os

    from pydantic import ValidationError

    from pgw.gateway import GatewaySettings

    tok = tmp_path / "admin-token"
    tok.write_text(ADMIN_TOKEN + "\n", encoding="utf-8")
    async with client(make_console_app(tiny_pack_dir, tmp_path, admin_token=None, admin_token_file=tok)) as c:
        await login(c)
    for k in list(os.environ):
        if k.startswith("PGW_GATEWAY_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("PGW_GATEWAY_ADMIN_TOKEN_FILE", str(tok))
    token = GatewaySettings().admin_token
    assert token is not None and token.get_secret_value() == ADMIN_TOKEN
    monkeypatch.setenv("PGW_GATEWAY_ADMIN_TOKEN", ADMIN_TOKEN)
    with pytest.raises(ValidationError, match="mutually exclusive"):
        GatewaySettings()
    monkeypatch.delenv("PGW_GATEWAY_ADMIN_TOKEN")
    (tmp_path / "two").write_text("a" * 20 + " " + "b" * 20, encoding="utf-8")
    (tmp_path / "short").write_text("short", encoding="utf-8")
    for name, msg in (("missing", "cannot read"), ("two", "exactly one token"), ("short", "at least")):
        monkeypatch.setenv("PGW_GATEWAY_ADMIN_TOKEN_FILE", str(tmp_path / name))
        with pytest.raises(ValidationError, match=msg):
            GatewaySettings()
