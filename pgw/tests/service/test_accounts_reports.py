"""Integration additions to the provider accounts: the enrollment reply's ``lane`` / ``masked_head``, content-free
request reports (``POST /pgw/v1/sdk/report``), the request log and model catalogue in the admin API, the usage
split, schema migration, and the developer console mounted on the service."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx

from pgw.service.accounts import AccountsStore, _parse_report
from tests.service.conftest import running_app
from tests.service.test_accounts import _new_key, _post, _settings

ADMIN = "pgwa-test-admin-token-0123456789"


def _admin_settings(tmp_path: Path, pack_dir: Path, **kw: Any) -> Any:
    import hashlib

    return _settings(tmp_path, pack_dir, accounts_admin_api=True,
                     accounts_admin_token_sha256=[hashlib.sha256(ADMIN.encode()).hexdigest()], **kw)


async def _call(app: Any, method: str, path: str, *, token: str, **kw: Any) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
        return await h.request(method, path, headers={"authorization": f"Bearer {token}"}, **kw)


def test_report_records_are_parsed_by_allow_list() -> None:
    now = 1_764_000_000.0
    rec = {"ts": now, "model": "qwen2.5-7b-instruct", "status": "ok", "prompt_tokens": 12, "completion_tokens": 3,
           "latency_ms": 50.5, "ttft_ms": 10.0, "stream": True, "checks_ok": True,
           "prompt": "secret text", "messages": [{"content": "secret"}], "error_code": "Bad Code!"}
    out = _parse_report(rec, now)
    assert out is not None and "secret" not in json.dumps(out)
    assert set(out) == {"ts", "model", "status", "error_code", "prompt_tokens", "completion_tokens", "latency_ms",
                        "ttft_ms", "stream", "checks_ok"}
    assert out["error_code"] is None and out["checks_ok"] is True
    assert _parse_report({**rec, "status": "error", "error_code": "integrity_check_failed"}, now)["checks_ok"] is False
    for bad in ({**rec, "ts": now - 30 * 86400}, {**rec, "model": "a b"}, {**rec, "prompt_tokens": -1},
                {**rec, "completion_tokens": True}, "text", None):
        assert _parse_report(bad, now) is None


def test_old_database_gets_the_report_columns(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE usage (day TEXT NOT NULL, key_id TEXT NOT NULL, model TEXT NOT NULL, "
                    "trips INTEGER NOT NULL DEFAULT 0, rows INTEGER NOT NULL DEFAULT 0, tokens REAL NOT NULL DEFAULT 0,"
                    " maskgen_cols INTEGER NOT NULL DEFAULT 0, maskgen_requests INTEGER NOT NULL DEFAULT 0, "
                    "bytes_in INTEGER NOT NULL DEFAULT 0, bytes_out INTEGER NOT NULL DEFAULT 0, "
                    "PRIMARY KEY (day, key_id, model));")
    c.close()
    st = AccountsStore(db)
    cols = {r[1] for r in st._conn().execute("PRAGMA table_info(usage)")}
    assert {"requests", "prompt_tokens", "completion_tokens", "errors", "checks_ok"} <= cols


async def test_enroll_reply_lane_and_masked_head_then_reports_reach_usage_log_and_console(
        tmp_path: Path, tiny_pack_dir: Path) -> None:
    s = _admin_settings(tmp_path, tiny_pack_dir, masked_head="also")
    key, rec = _new_key(s)
    async with running_app(s) as app:
        acc = app.state.pgw.accounts
        acc.lane_hint = lambda: {"transport": "tcp", "port": 8444}  # what a running TCP lane announces
        r = await _post(app, key.encode(), {"gateway_name": "host-1", "client_name": "host-1", "sdk_version": "0.1"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["lane"] == {"transport": "tcp", "port": 8444}
        assert [m["masked_head"] for m in d["models"]] == [True]
        tok = d["service_token"]
        now = __import__("time").time()
        body = {"requests": [
            {"ts": now, "model": "tiny-qwen2", "status": "ok", "prompt_tokens": 10, "completion_tokens": 4,
             "latency_ms": 12.0, "ttft_ms": 3.0, "stream": False, "checks_ok": True, "content": "do not store"},
            {"ts": now, "model": "tiny-qwen2", "status": "error", "error_code": "integrity_check_failed",
             "prompt_tokens": 10, "completion_tokens": 1, "latency_ms": 5.0, "stream": True, "checks_ok": True}]}
        r = await _call(app, "POST", "/pgw/v1/sdk/report", token=tok, json=body)
        assert r.status_code == 200 and r.json() == {"accepted": 2}, r.text
        assert (await _call(app, "POST", "/pgw/v1/sdk/report", token="nope", json=body)).status_code == 401
        acc.flush()
        usage = (await _call(app, "GET", "/pgw/v1/admin/orgs/acme/usage", token=ADMIN)).json()["usage"]
        (row,) = usage
        assert (row["requests"], row["prompt_tokens"], row["completion_tokens"], row["errors"], row["verified"]) == \
            (2, 20, 5, 1, 1)
        assert row["reported_by"] == "client"
        log = (await _call(app, "GET", "/pgw/v1/admin/orgs/acme/requests", token=ADMIN)).json()
        assert [x["status"] for x in log["requests"]] == ["error", "ok"]  # newest first
        assert log["requests"][0]["verified"] is False and log["requests"][1]["verified"] is True
        assert "do not store" not in json.dumps(log) and all(x["verification"] == "client-reported"
                                                             for x in log["requests"])
        f = (await _call(app, "GET", "/pgw/v1/admin/orgs/acme/requests?status=ok", token=ADMIN)).json()
        assert len(f["requests"]) == 1
        models = (await _call(app, "GET", "/pgw/v1/admin/orgs/acme/models", token=ADMIN)).json()["models"]
        assert [m["id"] for m in models] == ["tiny-qwen2"]
        # the in-process console client sees the same
        from pgw.service.platform_accounts import ServiceAccountsClient

        cl = ServiceAccountsClient(acc)
        page = cl.requests("acme", limit=10)
        assert len(page.records) == 2 and page.records[1].verified is True
        token, _ = AccountsStore(s.accounts_db).create_admin_token("console", "acme")
        assert cl.authenticate(token).org_id == "acme"  # type: ignore[union-attr]
        assert cl.authenticate(ADMIN) is None  # a global bootstrap token is not a console sign-in
        rows = cl.usage("acme", since="2000-01-01", until="2100-01-01")
        assert rows[0].requests == 2 and rows[0].verified == 1


async def test_console_is_mounted_behind_its_setting(tmp_path: Path, tiny_pack_dir: Path, monkeypatch: Any) -> None:
    s = _admin_settings(tmp_path, tiny_pack_dir)
    async with running_app(s) as app:
        assert (await _call(app, "GET", "/platform/login", token="x")).status_code == 404
    monkeypatch.setenv("PGW_PLATFORM_ENABLED", "1")
    _new_key(s)
    token, _ = AccountsStore(s.accounts_db).create_admin_token("console", "acme")
    async with running_app(s) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://svc") as h:
            assert (await h.get("/platform/login")).status_code == 200
            r = await h.post("/platform/login", data={"token": token})
            assert r.status_code == 303
            page = await h.get("/platform/keys")
            assert page.status_code == 200 and "Untitled key" not in page.text
