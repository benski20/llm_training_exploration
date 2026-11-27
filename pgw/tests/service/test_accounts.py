"""Provider accounts: SDK keys (codec, store, lifecycle), POST /pgw/v1/sdk/enroll, scope, usage, limits, admin API."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from pgw.core import RING_K40_A12
from pgw.core.errors import ConfigError, ErrorCode, PGWError
from pgw.enroll.sdk_key import SDKKey, SDKKeyError, secret_digest
from pgw.service import ServiceSettings, hash_token
from pgw.service.accounts import AccountsStore, KeyRejectedError, attest_mac, day_of, revoke_key_everywhere
from tests.service.conftest import asgi_client, make_req, rand_x, running_app

URL = "http://127.0.0.1:9"
HTTPS_URL = "https://203.0.113.7:22443"


# ---------------------------------------------------------------------------------------------- key codec
def test_sdk_key_roundtrip_vector_and_damage() -> None:
    k = SDKKey("test", HTTPS_URL, bytes(range(32)), bytes.fromhex("0011223344556677"), bytes(range(100, 124)))
    s = k.encode()
    # the contract's test vector (docs/sdk/CONTRACT.md, appendix A1)
    assert s == ("pgw_test_aeaacaqdaqcqmbyibefawdanbyhraeiscmkbkfqxdamrugy4dupb6aareizuivlgo5sgkzthnbuwu23mnvxg64"
                 "drojzxi5lwo54hs6t3nb2hi4dthixs6mrqgmxdalrrgezs4nz2gizdinbtpqytwli")
    assert k.digest.hex() == "93825b55e599ec56935c715e19e9f1fe0dd0f1109564917224c183bbd5cb9ca3"
    assert SDKKey.parse(" " + s + "\n") == k
    assert k.secret.hex() not in repr(k) and k.secret.hex() not in str(k) and s not in str(k)
    body = s[len("pgw_test_"):]
    for i in range(0, len(body), 7):  # any single-character change is caught
        bad = body[:i] + ("a" if body[i] != "a" else "b") + body[i + 1:]
        with pytest.raises(SDKKeyError):
            SDKKey.parse("pgw_test_" + bad)
    with pytest.raises(SDKKeyError, match="damaged"):
        SDKKey.parse("pgw_live_" + body)  # env is covered by the checksum
    with pytest.raises(SDKKeyError) as ei:
        SDKKey.parse("pgw_test_" + body[:-3])
    assert body[:-3] not in ei.value.message
    for junk in ("", "sk-abc", "pgw_test_", "PGW_TEST_" + body.upper()):
        with pytest.raises(SDKKeyError):
            SDKKey.parse(junk)
    with pytest.raises(ConfigError, match="pin"):
        SDKKey("live", HTTPS_URL, bytes(32), bytes(8), bytes(24))


# --------------------------------------------------------------------------------------------------- store
def _store(tmp_path: Path, clock: Any = time.time) -> AccountsStore:
    return AccountsStore(tmp_path / "accounts.db", clock=clock)


def test_store_hashes_at_rest_and_verifies_in_constant_time(tmp_path: Path) -> None:
    st = _store(tmp_path)
    st.create_org("acme", "Acme Corp", daily_token_quota=1000)
    key, rec = st.create_key("acme", "prod", HTTPS_URL, os.urandom(32), models=["m1", "m0", "m1"])
    assert rec.models == ["m0", "m1"] and rec.key_id == key.key_id_str
    assert stat.S_IMODE((tmp_path / "accounts.db").stat().st_mode) == 0o600
    assert sqlite3.connect(tmp_path / "accounts.db").execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    st.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("accounts.db"))
    assert key.secret not in raw and key.encode().encode() not in raw and secret_digest(key.key_id, key.secret) in raw
    got, org = st.verify_key(key)
    assert got.key_hex == rec.key_hex and org["daily_token_quota"] == 1000
    forged = SDKKey(key.env, key.service_url, key.ca_sha256, key.key_id, os.urandom(24))
    with pytest.raises(KeyRejectedError) as ei:
        st.verify_key(forged)
    assert ei.value.reason == "bad_secret" and ei.value.message == "invalid, revoked or expired SDK key"
    with pytest.raises(KeyRejectedError) as ei:
        st.verify_key(SDKKey.new("live", HTTPS_URL, os.urandom(32)))
    assert ei.value.reason == "unknown_key"
    with pytest.raises(KeyRejectedError) as ei:
        st.verify_key(SDKKey("test", key.service_url, key.ca_sha256, key.key_id, key.secret))
    assert ei.value.reason in ("bad_secret", "wrong_env")
    with pytest.raises(ConfigError):
        st.create_org("acme")
    with pytest.raises(ConfigError):
        st.create_org("Bad Org")
    with pytest.raises(PGWError):
        st.create_key("nobody", "k", HTTPS_URL, os.urandom(32))


def test_key_lifecycle_expiry_revocation_org_disable(tmp_path: Path) -> None:
    now = [1_000_000.0]
    st = _store(tmp_path, clock=lambda: now[0])
    st.create_org("acme")
    k1, _r1 = st.create_key("acme", "short", URL, None, expires_in_s=60)
    k2, r2 = st.create_key("acme", "long", URL, None)
    assert st.verify_key(k1)[0].state(now[0]) == "active"
    now[0] += 61
    with pytest.raises(KeyRejectedError) as ei:
        st.verify_key(k1)
    assert ei.value.reason == "expired"
    st.set_org_disabled("acme", True)
    with pytest.raises(KeyRejectedError) as ei:
        st.verify_key(k2)
    assert ei.value.reason == "org_disabled"
    st.set_org_disabled("acme", False)
    st.verify_key(k2)
    tokens = tmp_path / "tenants.txt"
    tokens.write_text("")
    st.add_installation(r2.key_hex, "i1", r2.tenant, "tk_aaaaaaaaaaaa", cert_serial=None, cert_cn=None, client="x")
    tokens.write_text(f"{r2.tenant}:tk_aaaaaaaaaaaa:{hash_token('t')}\nother:tk_bbbbbbbbbbbb:{hash_token('u')}\n")
    out = revoke_key_everywhere(st, r2.key_id, tokens, actor="test")
    assert out["state"] == "revoked" and out["tokens_removed"] == 1
    assert "tk_aaaaaaaaaaaa" not in tokens.read_text() and "tk_bbbbbbbbbbbb" in tokens.read_text()
    with pytest.raises(KeyRejectedError) as ei:
        st.verify_key(k2)
    assert ei.value.reason == "revoked"
    assert revoke_key_everywhere(st, r2.key_id, tokens)["tokens_removed"] == 0  # idempotent
    assert st.list_installations("acme")[0]["revoked_at"] is not None
    events = [e["event"] for e in st.audit_events("acme")]
    assert events[:2] == ["key.revoked", "org.enabled"] and "key.created" in events


# ------------------------------------------------------------------------------------------------ service
def _settings(tmp_path: Path, pack_dir: Path, **kw: Any) -> ServiceSettings:
    tokens = tmp_path / "tenants.txt"
    if not tokens.exists():
        tokens.write_text("")
    base: dict[str, Any] = dict(packs=[pack_dir], device="cpu", tokens_file=tokens, rings=["K32A12C27", "K40A12"],
                                accounts_db=tmp_path / "accounts.db", public_url=URL, batch_window_ms=1.0,
                                accounts_refresh_s=0.2, accounts_flush_s=0.2)
    base.update(kw)
    return ServiceSettings(**base)


def _new_key(s: ServiceSettings, *, org: str = "acme", **kw: Any) -> tuple[SDKKey, Any]:
    st = AccountsStore(s.accounts_db)  # type: ignore[arg-type]
    if not any(o["org_id"] == org for o in st.list_orgs()):
        st.create_org(org)
    return st.create_key(org, kw.pop("name", "k"), URL, None, **kw)


async def _post(app: Any, key: str | None, body: Any, client: str = "10.0.0.1") -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=(client, 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://svc") as h:
        headers = {"authorization": f"Bearer {key}"} if key is not None else {}
        return await h.post("/pgw/v1/sdk/enroll", json=body, headers=headers)


async def _models(app: Any, token: str) -> int:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
        r = await h.get("/pgw/v1/models", headers={"authorization": f"Bearer {token}", "x-pgw-protocol": "1"})
    return r.status_code


async def test_sdk_enroll_issues_working_credentials_and_revocation_refuses(
        tmp_path: Path, tiny_pack_dir: Path, pack_fp: str) -> None:
    s = _settings(tmp_path, tiny_pack_dir)
    key, rec = _new_key(s, models=["tiny-qwen2"])
    async with running_app(s) as app:
        r = await _post(app, key.encode(), {"installation_name": "host-1"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["tenant"] == f"sdk.{rec.key_hex}" and d["service_url"] == URL
        assert d["models"][0]["fingerprint"] == pack_fp and d["models"][0]["name"] == "tiny-qwen2"
        assert d["sdk"]["key_id"] == rec.key_id and d["sdk"]["org_id"] == "acme"
        assert d["sdk"]["models_scope"] == ["tiny-qwen2"] and d["sdk"]["installation_id"].startswith("in_")
        tok = d["service_token"]
        assert f":{hash_token(tok)}:" in s.tokens_file.read_text()  # type: ignore[union-attr]
        assert await _models(app, tok) == 200
        # a second installation with the same (reusable) key gets its own token
        r2 = await _post(app, key.encode(), {"gateway_name": "host-2"})
        assert r2.status_code == 200 and r2.json()["service_token"] != tok
        # revoke: the key and every token issued with it are refused at once
        acc = app.state.pgw.accounts
        out = acc.revoke_key(rec.key_id, actor="test")
        assert out["tokens_removed"] == 2
        assert await _models(app, tok) == 401
        assert await _models(app, r2.json()["service_token"]) == 401
        r3 = await _post(app, key.encode(), {"installation_name": "host-3"})
        assert r3.status_code == 401 and r3.json()["error"]["message"] == "invalid, revoked or expired SDK key"
        # unknown, malformed and missing keys look the same
        bodies = [(await _post(app, k, {"installation_name": "h"})).json()
                  for k in (SDKKey.new("live", URL, None).encode(), "pgw_live_zzzz", None)]
        assert all(b == r3.json() for b in bodies)
    st = AccountsStore(s.accounts_db)  # type: ignore[arg-type]
    events = st.audit_events("acme")
    text = json.dumps(st.audit_events(None, limit=1000))
    assert key.encode() not in text and tok not in text and key.secret.hex() not in text
    assert [e["event"] for e in events].count("sdk.enroll.ok") == 2
    assert any(e["event"] == "sdk.enroll.failed" and e["detail"]["reason"] == "revoked" for e in events)


async def test_revocation_from_another_process_takes_effect_within_refresh(
        tmp_path: Path, tiny_pack_dir: Path) -> None:
    s = _settings(tmp_path, tiny_pack_dir, tokens_reload_interval_s=60.0)
    key, rec = _new_key(s)
    async with running_app(s) as app:
        tok = (await _post(app, key.encode(), {"installation_name": "h"})).json()["service_token"]
        assert await _models(app, tok) == 200
        # the CLI's path: database only, tokens file untouched (the service reloads it only every 60 s here)
        AccountsStore(s.accounts_db).revoke_key(rec.key_id)  # type: ignore[arg-type]
        import asyncio

        for _ in range(50):
            await asyncio.sleep(0.1)
            if await _models(app, tok) == 401:
                break
        assert await _models(app, tok) == 401


async def test_model_scope_refused_at_enroll_and_compute(tmp_path: Path, tiny_pack_dir: Path, pack_fp: str) -> None:
    s = _settings(tmp_path, tiny_pack_dir)
    key_tiny, rec_tiny = _new_key(s, name="tiny", models=["tiny-qwen2"])
    key_other, _ = _new_key(s, name="other", models=["qwen2.5-7b-instruct"])
    async with running_app(s) as app:
        r = await _post(app, key_tiny.encode(), {"installation_name": "h", "models": ["qwen2.5-7b-instruct"]})
        assert r.status_code == 403 and r.json()["error"]["code"] == "permission_denied"
        r = await _post(app, key_other.encode(), {"installation_name": "h"})
        assert r.status_code == 503  # nothing in its scope is served
        tok = (await _post(app, key_tiny.encode(), {"installation_name": "h"})).json()["service_token"]
        ring = RING_K40_A12
        c = asgi_client(app, token=tok, max_retries=0)
        await c.compute(make_req(pack_fp, ring, [("0.o", rand_x(1, 64, ring))]))
        # narrow the key's scope in the database: compute on the (now) foreign model is refused
        db = sqlite3.connect(s.accounts_db)  # type: ignore[arg-type]
        db.execute("UPDATE sdk_keys SET models = ? WHERE key_id = ?", (json.dumps(["qwen2.5-7b-instruct"]),
                                                                        rec_tiny.key_hex))
        db.commit()
        db.close()
        app.state.pgw.accounts.refresh()
        with pytest.raises(PGWError) as ei:
            await c.compute(make_req(pack_fp, ring, [("0.o", rand_x(1, 64, ring))]))
        assert ei.value.code == ErrorCode.PERMISSION_DENIED
        await c.aclose()


async def test_usage_is_metered_and_reported(tmp_path: Path, tiny_pack_dir: Path, pack_fp: str) -> None:
    s = _settings(tmp_path, tiny_pack_dir)
    key, rec = _new_key(s)
    async with running_app(s) as app:
        tok = (await _post(app, key.encode(), {"installation_name": "h"})).json()["service_token"]
        acc = app.state.pgw.accounts
        info = acc._model(pack_fp)
        ring = RING_K40_A12
        ref = f"0.{info.ref_kind}"
        dims = app.state.pgw.registry.get(pack_fp).linear_dims
        c = asgi_client(app, token=tok, max_retries=0)
        for i in range(3):  # 3 single-item trips on the reference linear, 2 rows each
            await c.compute(make_req(pack_fp, ring, [(ref, rand_x(2, dims[ref][1], ring, seed=i))]))
        # one multi-item trip: 1 row of the reference linear + 4 rows of another linear
        other = next(k for k in dims if not k.endswith("." + info.ref_kind))
        await c.compute(make_req(pack_fp, ring, [(ref, rand_x(1, dims[ref][1], ring)),
                                                 (other, rand_x(4, dims[other][1], ring))]))
        await c.aclose()
        # an unrelated tenant is not metered
        acc.flush()
        rows = acc.store.usage(key_id=rec.key_id)
        assert len(rows) == 1
        u = rows[0]
        assert u["day"] == day_of(time.time()) and u["model"] == "tiny-qwen2"
        assert u["trips"] == 4 and u["rows"] == 3 * 2 + 1 + 4
        assert u["tokens"] == pytest.approx(7 / info.ref_count)
        assert u["bytes_in"] > 0 and u["bytes_out"] > 0
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            r = await h.get("/pgw/v1/sdk/usage", headers={"authorization": f"Bearer {key.encode()}"})
            assert r.status_code == 200, r.text
            d = r.json()
            assert d["key_id"] == rec.key_id and d["usage"][0]["trips"] == 4
            assert d["today"]["tokens"] == pytest.approx(7 / info.ref_count, abs=1e-3)
            bad = await h.get("/pgw/v1/sdk/usage", headers={"authorization": f"Bearer {tok}"})
            assert bad.status_code == 401  # a service token is not an SDK key


async def test_rate_limit_and_daily_quota_enforced(tmp_path: Path, tiny_pack_dir: Path, pack_fp: str) -> None:
    s = _settings(tmp_path, tiny_pack_dir)
    key_rate, _ = _new_key(s, name="rate", rows_per_s=0.01, burst_rows=4)
    key_quota, rec_q = _new_key(s, name="quota", daily_token_quota=1)
    async with running_app(s) as app:
        ring = RING_K40_A12
        acc = app.state.pgw.accounts
        ref = f"0.{acc._model(pack_fp).ref_kind}"
        n = app.state.pgw.registry.get(pack_fp).linear_dims[ref][1]
        tok_r = (await _post(app, key_rate.encode(), {"installation_name": "h"})).json()["service_token"]
        c = asgi_client(app, token=tok_r, max_retries=0)
        await c.compute(make_req(pack_fp, ring, [(ref, rand_x(4, n, ring))]))  # the whole burst
        with pytest.raises(PGWError) as ei:
            await c.compute(make_req(pack_fp, ring, [(ref, rand_x(1, n, ring))]))
        assert ei.value.code == ErrorCode.RATE_LIMITED
        await c.aclose()
        tok_q = (await _post(app, key_quota.encode(), {"installation_name": "h"})).json()["service_token"]
        q = asgi_client(app, token=tok_q, max_retries=0)
        rows = 2 * acc._model(pack_fp).ref_count  # 2 tokens' worth of reference rows > quota 1
        await q.compute(make_req(pack_fp, ring, [(ref, rand_x(rows, n, ring))]))
        with pytest.raises(PGWError) as ei:
            await q.compute(make_req(pack_fp, ring, [(ref, rand_x(1, n, ring))]))
        assert ei.value.code == ErrorCode.RATE_LIMITED
        await q.aclose()
        r = await _post(app, key_quota.encode(), {"installation_name": "h2"})
        assert r.status_code == 429 and "Retry-After" in r.headers
    events = AccountsStore(s.accounts_db).audit_events("acme")  # type: ignore[arg-type]
    import asyncio

    for _ in range(20):  # written by a background thread
        if any(e["event"] == "quota.exceeded" and e["key_id"] == rec_q.key_id for e in events):
            break
        await asyncio.sleep(0.05)
        events = AccountsStore(s.accounts_db).audit_events("acme")  # type: ignore[arg-type]
    assert any(e["event"] == "quota.exceeded" and e["key_id"] == rec_q.key_id for e in events)


async def test_enroll_limits_per_key(tmp_path: Path, tiny_pack_dir: Path) -> None:
    s = _settings(tmp_path, tiny_pack_dir, sdk_enrolls_per_key_per_hour=2, enrollment_rate_limit_per_minute=3)
    key, _ = _new_key(s, name="a")
    key2, _ = _new_key(s, name="b", max_installations=1)
    async with running_app(s) as app:
        codes = [(await _post(app, key.encode(), {"installation_name": f"h{i}"})).status_code for i in range(3)]
        assert codes == [200, 200, 429]
        codes = [(await _post(app, key2.encode(), {"installation_name": f"h{i}"})).status_code for i in range(2)]
        assert codes == [200, 429]
        # failed attempts per address (bad keys) hit the per-IP limiter
        codes = [(await _post(app, SDKKey.new("live", URL, None).encode(), {"installation_name": "x"},
                              client="10.9.9.9")).status_code for _ in range(4)]
        assert codes == [401, 401, 401, 429]


def test_settings_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tokens_file"):
        ServiceSettings(accounts_db=tmp_path / "a.db", public_url=URL)
    with pytest.raises(ValueError, match="enrollment_ca_file"):
        ServiceSettings(accounts_db=tmp_path / "a.db", public_url=HTTPS_URL, tokens_file=tmp_path / "t")
    with pytest.raises(ValueError, match="accounts_db"):
        ServiceSettings(accounts_admin_api=True)
    with pytest.raises(ValueError):
        ServiceSettings(sdk_key_env="Live")


# ------------------------------------------------------------------------------------------------- admin API
async def test_admin_api_scopes_and_attest(tmp_path: Path, tiny_pack_dir: Path) -> None:
    boot = "pgwa-bootstrap-" + os.urandom(8).hex()
    boot_digest = hashlib.sha256(boot.encode()).digest()
    s = _settings(tmp_path, tiny_pack_dir, accounts_admin_api=True, accounts_admin_token_sha256=[boot_digest.hex()])
    st = AccountsStore(s.accounts_db)  # type: ignore[arg-type]
    async with running_app(s) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            hb = {"authorization": f"Bearer {boot}"}
            assert (await h.get("/pgw/v1/admin/orgs")).status_code == 401
            assert (await h.get("/pgw/v1/admin/orgs", headers={"authorization": "Bearer nope"})).status_code == 401
            r = await h.post("/pgw/v1/admin/orgs", json={"org_id": "acme", "daily_token_quota": 100}, headers=hb)
            assert r.status_code == 200 and r.json()["org_id"] == "acme"
            await h.post("/pgw/v1/admin/orgs", json={"org_id": "other"}, headers=hb)
            r = await h.post("/pgw/v1/admin/orgs/acme/keys", json={"name": "prod", "models": ["tiny-qwen2"]},
                             headers=hb)
            assert r.status_code == 200, r.text
            key = r.json()["key"]
            assert SDKKey.parse(key).service_url == URL and r.json()["record"]["models"] == ["tiny-qwen2"]
            assert (await _post(app, key, {"installation_name": "h"})).status_code == 200
            org_tok, _ = st.create_admin_token("console", "acme")
            ho = {"authorization": f"Bearer {org_tok}"}
            assert [o["org_id"] for o in (await h.get("/pgw/v1/admin/orgs", headers=ho)).json()["orgs"]] == ["acme"]
            assert (await h.get("/pgw/v1/admin/orgs/other/keys", headers=ho)).status_code == 403
            assert (await h.post("/pgw/v1/admin/orgs", json={"org_id": "x"}, headers=ho)).status_code == 403
            keys = (await h.get("/pgw/v1/admin/orgs/acme/keys", headers=ho)).json()["keys"]
            assert len(keys) == 1 and keys[0]["installations"] == 1 and "digest" not in json.dumps(keys)
            inst = (await h.get("/pgw/v1/admin/orgs/acme/installations", headers=ho)).json()["installations"]
            assert inst[0]["name"] == "h"
            r = await h.post(f"/pgw/v1/admin/orgs/acme/keys/{keys[0]['key_id']}/revoke", headers=ho)
            assert r.status_code == 200 and r.json()["state"] == "revoked"
            assert (await _post(app, key, {"installation_name": "h2"})).status_code == 401
            ev = (await h.get("/pgw/v1/admin/orgs/acme/audit?limit=5", headers=ho)).json()["events"]
            assert [e["event"] for e in ev[:2]] == ["sdk.enroll.failed", "key.revoked"] and key not in json.dumps(ev)
            assert (await h.get("/pgw/v1/admin/orgs/acme/usage", headers=ho)).status_code == 200
            nonce = os.urandom(32)
            a = (await h.post("/pgw/v1/admin/attest", json={"nonce": nonce.hex()})).json()
            assert a["macs"] == [attest_mac(boot_digest, nonce, bytes.fromhex(a["ca_sha256"])).hex()]


async def test_admin_api_off_by_default(tmp_path: Path, tiny_pack_dir: Path) -> None:
    async with running_app(_settings(tmp_path, tiny_pack_dir)) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            assert (await h.get("/pgw/v1/admin/orgs")).status_code == 404
            assert (await h.post("/pgw/v1/enroll", json={})).status_code in (404, 405)  # no codes without a file


# ------------------------------------------------------------------------------------------------------ CLI
def test_cli_keys_lifecycle_and_usage(tmp_path: Path) -> None:
    from pgw.cli.main import cli
    from tests.enroll_helpers import make_pki

    pki = make_pki(tmp_path / "pki")
    db = tmp_path / "accounts.db"
    tokens = tmp_path / "tenants.txt"
    tokens.write_text("")
    runner = CliRunner()
    r = runner.invoke(cli, ["service", "keys", "create", "--org", "acme", "--create-org", "--name", "prod",
                            "--models", "qwen2.5-7b-instruct", "--expires", "90d", "--daily-token-quota", "1000000",
                            "--service-url", HTTPS_URL, "--ca-file", str(pki["ca"]), "--accounts-db", str(db),
                            "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    from pgw.enroll.service import ca_fingerprint, load_ca_pem

    k = SDKKey.parse(out["key"])
    assert k.ca_sha256 == ca_fingerprint(load_ca_pem(pki["ca"])) and k.service_url == HTTPS_URL
    assert out["models"] == ["qwen2.5-7b-instruct"] and out["limits"]["daily_token_quota"] == 1000000
    r = runner.invoke(cli, ["service", "keys", "create", "--org", "acme", "--name", "ci", "--service-url",
                            HTTPS_URL, "--ca-file", str(pki["ca"]), "--accounts-db", str(db),
                            "--out", str(tmp_path / "ci.key")])
    assert r.exit_code == 0 and "pgw_live_" not in r.stdout
    assert stat.S_IMODE((tmp_path / "ci.key").stat().st_mode) == 0o600
    r = runner.invoke(cli, ["service", "keys", "list", "--accounts-db", str(db), "--json"])
    listed = json.loads(r.stdout)
    assert {x["name"] for x in listed} == {"prod", "ci"} and out["key"] not in r.stdout
    r = runner.invoke(cli, ["service", "keys", "revoke", out["key_id"], "--accounts-db", str(db),
                            "--tokens-file", str(tokens)])
    assert r.exit_code == 0 and json.loads(r.stdout)["state"] == "revoked"
    AccountsStore(db).add_usage({(day_of(time.time()), k.key_id.hex(), "qwen2.5-7b-instruct"):
                                 {"trips": 5, "rows": 10, "tokens": 2.0}})
    r = runner.invoke(cli, ["service", "usage", "--accounts-db", str(db), "--org", "acme", "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)[0]["trips"] == 5
    r = runner.invoke(cli, ["service", "usage", "--accounts-db", str(db), "--by", "model"])
    assert "trips=5" in r.stdout
    r = runner.invoke(cli, ["service", "keys", "create", "--org", "acme", "--name", "x", "--service-url", HTTPS_URL,
                            "--accounts-db", str(db)])
    assert r.exit_code != 0 and "pin" in r.output  # https without a CA to pin
    r = runner.invoke(cli, ["service", "admin-token", "create", "--name", "console", "--org", "acme",
                            "--accounts-db", str(db)])
    assert r.exit_code == 0 and json.loads(r.stdout)["token"].startswith("pgwa-")
    r = runner.invoke(cli, ["service", "audit", "--accounts-db", str(db), "--org", "acme"])
    assert r.exit_code == 0 and out["key"] not in r.stdout


# ------------------------------------------------------------------------------- real loopback TLS service
@pytest.mark.timeout(240)
def test_enroll_with_sdk_key_over_real_tls_gives_working_credentials(tiny_pack_dir: Path, tmp_path: Path,
                                                                     pack_fp: str, pack_weights: Any) -> None:
    """A real uvicorn TLS service with accounts on: the client side pins the CA from the key, enrolls with a
    locally generated key, and the written credentials (token + client certificate) compute exactly; after
    revocation the same credentials are refused, and a wrong pin never sends the key."""
    import asyncio
    import ssl

    from pydantic import SecretStr

    from pgw.core import unpack_ring
    from pgw.enroll.client import EnrollmentError, enroll_with_sdk_key
    from pgw.enroll.service import ca_fingerprint, load_ca_pem
    from pgw.gateway.config import load_models_file
    from pgw.wire import ClientConfig, ComputeClient
    from tests.enroll_helpers import enroll_service
    from tests.service.conftest import expected

    mtls = tmp_path / "mtls.txt"
    mtls.write_text("")
    with enroll_service(tiny_pack_dir, tmp_path / "svc", verify_client_certs=True, mtls_tenants_file=mtls,
                        require_client_cert_for_compute=True, accounts_db=tmp_path / "svc" / "accounts.db",
                        accounts_refresh_s=0.2) as svc:
        st = AccountsStore(svc.settings.accounts_db)
        st.create_org("acme")
        fp = ca_fingerprint(load_ca_pem(svc.dirs["ca"]))
        key, rec = st.create_key("acme", "prod", svc.url, fp, models=["tiny-qwen2"])
        # wrong pin: refused before the key is sent
        rogue_key, rogue = st.create_key("acme", "rogue", svc.url,
                                         ca_fingerprint(load_ca_pem(svc.dirs["rogue_ca"])))
        with pytest.raises(EnrollmentError, match="does not match"):
            enroll_with_sdk_key(rogue_key.encode(), tmp_path / "rogue")
        assert not any(e["key_id"] == rogue.key_id and e["event"].startswith("sdk.") for e in st.audit_events())
        res = enroll_with_sdk_key(key.encode(), tmp_path / "etc", installation_name="laptop-1")
        files = res.files
        assert res.tenant == rec.tenant and res.client_cert
        assert stat.S_IMODE(files["token"].stat().st_mode) == 0o600
        assert stat.S_IMODE(files["client_key"].stat().st_mode) == 0o600
        (m,) = load_models_file(files["models"])
        assert m.fingerprint == pack_fp and m.service_url == svc.url and m.client_cert_file == files["client_cert"]
        token = files["token"].read_text().strip()
        assert key.encode() not in "".join(p.read_text() for p in files.values())
        w8 = pack_weights["0.o"]

        async def compute() -> None:
            conf = ClientConfig(base_url=svc.url, token=SecretStr(token), ca_file=files["ca"],
                                client_cert_file=files["client_cert"], client_key_file=files["client_key"],
                                max_retries=0)
            c = ComputeClient(conf)
            try:
                ring = RING_K40_A12
                x = rand_x(2, w8.shape[1], ring)
                resp = await c.compute(make_req(pack_fp, ring, [("0.o", x)]))
                y = unpack_ring(resp.items[0].y_hat, (2, w8.shape[0]), ring)
                assert (y == expected(x, w8, ring)).all()
            finally:
                await c.aclose()

        asyncio.run(compute())
        ctx = ssl.create_default_context(cafile=str(files["ca"]))
        ctx.load_cert_chain(str(files["client_cert"]), str(files["client_key"]))
        hdr = {"authorization": f"Bearer {token}", "PGW-Protocol": "1"}
        assert httpx.get(f"{svc.url}/pgw/v1/models", headers=hdr, verify=ctx).status_code == 200
        # revoke from "another process" (the CLI path): refused within the refresh interval, certificate too
        revoke_key_everywhere(st, rec.key_id, svc.settings.tokens_file)
        deadline = time.time() + 10
        while time.time() < deadline:
            if httpx.get(f"{svc.url}/pgw/v1/models", headers=hdr, verify=ctx).status_code == 401:
                break
            time.sleep(0.1)
        assert httpx.get(f"{svc.url}/pgw/v1/models", headers=hdr, verify=ctx).status_code == 401
        assert httpx.get(f"{svc.url}/pgw/v1/models", headers={"PGW-Protocol": "1"},
                         verify=ctx).status_code in (401, 403)
        with pytest.raises(EnrollmentError, match="did not accept this SDK key"):
            enroll_with_sdk_key(key.encode(), tmp_path / "etc2")


@pytest.mark.timeout(240)
def test_enroll_response_lists_region_endpoints(tiny_pack_dir: Path, tmp_path: Path) -> None:
    """Region routing (docs/sdk/wan.md): ``endpoints`` goes into the SDK enrollment response and the client keeps the
    validated list (the enrolled URL first) in enrollment.json; without the setting the response has none."""
    import json as _json

    from pgw.enroll.client import enroll_with_sdk_key
    from pgw.enroll.service import ca_fingerprint, load_ca_pem
    from tests.enroll_helpers import enroll_service

    eps = [{"region": "eu-west", "url": "https://eu.example.net:8443", "models": [], "priority": 1}]
    with enroll_service(tiny_pack_dir, tmp_path / "svc", accounts_db=tmp_path / "svc" / "accounts.db",
                        accounts_refresh_s=0.2, endpoints=eps) as svc:
        st = AccountsStore(svc.settings.accounts_db)
        st.create_org("acme")
        key, _rec = st.create_key("acme", "prod", svc.url, ca_fingerprint(load_ca_pem(svc.dirs["ca"])))
        res = enroll_with_sdk_key(key.encode(), tmp_path / "etc", installation_name="laptop-1")
        meta = _json.loads(res.files["meta"].read_text())
    assert meta["endpoints"] == [{"region": "enrolled", "url": svc.url, "models": [], "priority": 0},
                                 {"region": "eu-west", "url": "https://eu.example.net:8443", "models": [],
                                  "priority": 1}]
    with enroll_service(tiny_pack_dir, tmp_path / "svc2", accounts_db=tmp_path / "svc2" / "accounts.db",
                        accounts_refresh_s=0.2) as svc:
        st = AccountsStore(svc.settings.accounts_db)
        st.create_org("acme")
        key, _rec = st.create_key("acme", "prod", svc.url, ca_fingerprint(load_ca_pem(svc.dirs["ca"])))
        res = enroll_with_sdk_key(key.encode(), tmp_path / "etc2", installation_name="laptop-2")
        assert "endpoints" not in _json.loads(res.files["meta"].read_text())


def test_lane_counters_are_metered_and_revoked_tenants_reported(tmp_path: Path, shared_registry: Any,
                                                                pack_fp: str) -> None:
    """The native lane counts per (tenant, kind); its drain is attributed to the model of the tenant's tickets."""
    from types import SimpleNamespace

    from pgw.service.accounts import Accounts

    st = _store(tmp_path)
    st.create_org("acme")
    _key, rec = st.create_key("acme", "k", URL, None)
    acc = Accounts(st, SimpleNamespace(accounts_refresh_s=1.0, accounts_flush_s=1.0, tokens_file=None),
                   registry=shared_registry)
    info = acc._model(pack_fp)
    t = rec.tenant
    acc.admit(t, pack_fp, 0)
    acc.lane_opened(t, pack_fp)
    drain = {"requests": [(t, "ok", [], 0.0, 5), (t, "rate_limited", [], 0.0, 2), ("other", "ok", [], 0.0, 9)],
             "rows": [(t, info.ref_kind, 3 * info.ref_count), (t, "zz", 4), ("other", info.ref_kind, 99)],
             "bytes": [(t, 100, 200)]}
    assert acc.record_lane(drain) == []
    acc.flush()
    (u,) = st.usage(key_id=rec.key_id)
    assert (u["model"], u["trips"], u["rows"], u["tokens"]) == ("tiny-qwen2", 5, 3 * info.ref_count + 4, 3.0)
    assert (u["bytes_in"], u["bytes_out"]) == (100, 200)
    acc.revoke_key(rec.key_id)
    assert acc.record_lane({}) == [t]  # its lane connections get closed (reported once)
    assert acc.record_lane({}) == []
    with pytest.raises(PGWError) as ei:
        acc.admit(t, pack_fp, 0)
    assert ei.value.code == ErrorCode.UNAUTHENTICATED
    acc.admit("not-an-sdk-tenant", pack_fp, 10**9)  # other tenants pass untouched
