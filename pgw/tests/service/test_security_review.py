"""Regression tests of the 2026-11 security review (``research/reviews/security-review-2026-11.md``), provider side:
bounded bodies (S1), the public readiness view (S5), the report rate limit (S8) and certificate-name collisions
(S6)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from pgw.core.errors import ErrorCode, PayloadTooLargeError, PGWError
from pgw.core.http_body import read_limited
from pgw.service.app import create_metrics_app
from pgw.service.auth import TenantAuthenticator
from tests.service.conftest import running_app
from tests.service.test_accounts import _new_key, _post, _settings

ADMIN = "pgwa-test-admin-token-0123456789"


async def _endless_chunked(app: Any, path: str, *, chunk: int = 16 << 10, cap: int = 64 << 20,
                           headers: list[tuple[bytes, bytes]] | None = None) -> tuple[int, int]:
    """POST an (almost) endless chunked body (no Content-Length) straight through ASGI; returns (status, bytes the
    app pulled before answering)."""
    pulled = 0
    status: list[int] = []
    done = False

    async def receive() -> dict[str, Any]:
        nonlocal pulled
        if done or pulled >= cap:
            return {"type": "http.disconnect"}
        pulled += chunk
        return {"type": "http.request", "body": b"x" * chunk, "more_body": pulled < cap}

    async def send(msg: dict[str, Any]) -> None:
        nonlocal done
        if msg["type"] == "http.response.start":
            status.append(msg["status"])
            done = True

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"host", b"svc"), (b"transfer-encoding", b"chunked"),
                         (b"content-type", b"application/json"), *(headers or [])],
             "client": ("10.0.0.9", 1234), "server": ("svc", 80)}
    await app(scope, receive, send)
    return status[0], pulled


async def test_read_limited_stops_at_the_limit() -> None:
    from starlette.requests import Request

    sent = 0

    async def receive() -> dict[str, Any]:
        nonlocal sent
        sent += 1000
        return {"type": "http.request", "body": b"a" * 1000, "more_body": True}

    req = Request({"type": "http", "method": "POST", "headers": [], "path": "/"}, receive)
    with pytest.raises(PayloadTooLargeError):
        await read_limited(req, 4096)
    assert sent == 5000
    big = Request({"type": "http", "method": "POST", "headers": [(b"content-length", b"999999")], "path": "/"},
                  receive)
    with pytest.raises(PayloadTooLargeError):
        await read_limited(big, 4096)


async def test_unauthenticated_json_routes_do_not_buffer_chunked_bodies(tmp_path: Path, tiny_pack_dir: Path) -> None:
    """PoC of S1: before the fix ``request.body()`` read the whole chunked body (no Content-Length to check) before
    the size check, so one unauthenticated client could make the provider buffer gigabytes."""
    s = _settings(tmp_path, tiny_pack_dir, accounts_admin_api=True,
                  accounts_admin_token_sha256=[hashlib.sha256(ADMIN.encode()).hexdigest()])
    _new_key(s)
    async with running_app(s) as app:
        for path in ("/pgw/v1/sdk/enroll", "/pgw/v1/admin/attest"):
            status, pulled = await _endless_chunked(app, path)
            assert status == 413, path
            assert pulled <= (64 << 10) + (16 << 10), (path, pulled)  # stopped right after the 64 KiB limit


async def test_public_readyz_has_no_load_figures(tmp_path: Path, tiny_pack_dir: Path) -> None:
    """S5: in-flight requests and queued rows, polled by anyone, show other tenants' traffic."""
    s = _settings(tmp_path, tiny_pack_dir)
    async with running_app(s) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            r = await h.get("/readyz")
        body = r.json()
        assert r.status_code == 200 and body["ready"] and body["models"]
        assert "inflight" not in body and "queued_rows" not in body
        mapp = create_metrics_app(app.state.pgw)  # the non-public port keeps them
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mapp), base_url="http://m") as h:
            full = (await h.get("/readyz")).json()
        assert "inflight" in full and "queued_rows" in full


async def test_report_route_is_rate_limited_per_key(tmp_path: Path, tiny_pack_dir: Path) -> None:
    """S8: one installation could grow the request log without bound (256 records per call, no limit)."""
    s = _settings(tmp_path, tiny_pack_dir, accounts_report_records_per_s=1.0)
    key, _ = _new_key(s)
    async with running_app(s) as app:
        r = await _post(app, key.encode(), {"installation_name": "host-1"})
        tok = r.json()["service_token"]
        now = time.time()
        recs = [{"ts": now, "model": "tiny-qwen2", "status": "ok", "prompt_tokens": 1, "completion_tokens": 1}] * 256
        codes = []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            for _ in range(17):  # burst 4096 = 16 full batches
                rr = await h.post("/pgw/v1/sdk/report", json={"requests": recs},
                                  headers={"authorization": f"Bearer {tok}"})
                codes.append(rr.status_code)
        assert codes[:16] == [200] * 16 and codes[16] == 429
        assert json.loads(rr.text)["error"]["code"] == str(ErrorCode.RATE_LIMITED) and "retry-after" in rr.headers


def test_colliding_client_certificate_name_is_refused(tmp_path: Path) -> None:
    """S6: CN = pgw-gw.<tenant>.<name>: tenant "a" + name "b.c" and tenant "a.b" + name "c" give the same CN; mapping
    it again would move the other tenant's certificate identity."""
    from pgw.enroll.service import EnrollmentService

    (tmp_path / "tokens").write_text("")
    settings = SimpleNamespace(tokens_file=tmp_path / "tokens", enrollment_file=None, enrollment_audit_file=None,
                               enrollment_rate_limit_per_minute=10, enrollment_ca_file=None,
                               client_issuing_ca_cert_file=tmp_path / "unused.pem", mtls_tenants_file=None,
                               public_url="https://svc.example:8443")
    auth = TenantAuthenticator(None, allow_empty=True, mtls_subjects={"pgw-gw.a.b.c": "a"})
    svc = EnrollmentService(settings, auth, list)
    with pytest.raises(PGWError) as ei:
        svc.issue_credentials("a.b", "c", "-----BEGIN CERTIFICATE REQUEST-----")
    assert ei.value.code == ErrorCode.PERMISSION_DENIED
    assert (tmp_path / "tokens").read_text() == ""  # refused before any credential was issued
    assert auth.tenant_for_subject("pgw-gw.a.b.c") == "a"


async def test_provisioning_enforces_the_sdk_key_scope_quota_and_row_rate(
        tmp_path: Path, tiny_pack_dir: Path, pack_fp: str) -> None:
    """S10: provisioning authenticated like compute but never applied the key's policy: a key scoped to another model
    could fetch this model's artifacts and start hint jobs (provider CPU/GPU) past its quota and row rate."""
    s = _settings(tmp_path, tiny_pack_dir, provision=True, provision_chunk_mb=1)
    other, orec = _new_key(s, name="other", models=["tiny-qwen2", "some-other-model"])
    mine, _ = _new_key(s, name="mine", models=["tiny-qwen2"], daily_token_quota=10**9, rows_per_s=1.0, burst_rows=256)
    p = "/pgw/v1/provision"
    async with running_app(s) as app:
        toks = {}
        for name, key in (("other", other), ("mine", mine)):
            r = await _post(app, key.encode(), {"installation_name": f"host-{name}"})
            assert r.status_code == 200, r.text
            toks[name] = r.json()["service_token"]
        import asyncio
        import sqlite3

        with sqlite3.connect(s.accounts_db) as db:  # the key's scope narrows after enrollment
            db.execute("UPDATE sdk_keys SET models = ? WHERE key_id = ?", (json.dumps(["some-other-model"]),
                                                                           orec.key_hex))
        await asyncio.sleep(1.0)  # accounts_refresh_s = 0.2
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            def hd(name: str) -> dict[str, str]:
                return {"authorization": f"Bearer {toks[name]}", "pgw-protocol": "1"}

            seed = {"a_seed": "ab" * 32, "masked_head": False}
            # out of scope: no catalog entry, no manifest, no objects, no hint job
            assert (await h.get(f"{p}/models", headers=hd("other"))).json()["models"] == []
            assert (await h.get(f"{p}/{pack_fp}/manifest", headers=hd("other"))).status_code == 403
            r = await h.get(f"{p}/{pack_fp}/object", params={"id": "manifest", "chunk": 0}, headers=hd("other"))
            assert r.status_code == 403
            assert (await h.post(f"{p}/{pack_fp}/hint", json=seed, headers=hd("other"))).status_code == 403
            # in scope
            assert [m["name"] for m in (await h.get(f"{p}/models", headers=hd("mine"))).json()["models"]] == [
                "tiny-qwen2"]
            assert (await h.get(f"{p}/{pack_fp}/manifest", headers=hd("mine"))).status_code == 200
            r = await h.post(f"{p}/{pack_fp}/hint", json=seed, headers=hd("mine"))
            assert r.status_code == 200, r.text
            job = r.json()
            lin = job["linears"][0]["key"]
            # H chunks count against the key's row bucket (burst 256 = one chunk, then 1 row/s)
            r1 = await h.get(f"{p}/hint/{job['job']}/h", params={"linear": lin, "chunk": 0}, headers=hd("mine"))
            r2 = await h.get(f"{p}/hint/{job['job']}/h", params={"linear": lin, "chunk": 0}, headers=hd("mine"))
            assert r1.status_code == 200 and r2.status_code == 429 and int(r2.headers["retry-after"]) >= 1
            # the daily quota applies too
            app.state.pgw.accounts.over_quota = lambda tenant: "key"
            r = await h.get(f"{p}/{pack_fp}/manifest", headers=hd("mine"))
            assert r.status_code == 429 and r.json()["code"] == str(ErrorCode.RATE_LIMITED)


def test_provisioning_client_waits_out_the_row_rate_but_not_a_daily_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.provision import session as ps

    sleeps: list[float] = []
    monkeypatch.setattr(ps.time, "sleep", sleeps.append)
    replies = iter([(429, "2"), (429, "1"), (200, None)])

    def handler(req: httpx.Request) -> httpx.Response:
        code, ra = next(replies)
        return httpx.Response(code, json={"code": "rate_limited", "message": "slow down"},
                              headers={"Retry-After": ra} if ra else {})

    sess = ps.ProvisionSession("http://svc", transport=httpx.MockTransport(handler), retries=0)
    assert sess.request("GET", "/x").status_code == 200 and sleeps == [2.0, 1.0]
    quota = ps.ProvisionSession("http://svc", retries=3, transport=httpx.MockTransport(
        lambda r: httpx.Response(429, json={"code": "rate_limited", "message": "quota"}, headers={"Retry-After": "40000"})))
    with pytest.raises(PGWError) as ei:
        quota.request("GET", "/x")
    assert ei.value.code == ErrorCode.RATE_LIMITED
