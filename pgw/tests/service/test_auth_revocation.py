"""Token ids, expiry, revocation with mtime reload, require-client-cert, and closing revoked channels."""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest

from pgw.core.errors import AuthError, ConfigError, PGWError
from pgw.service import ServiceSettings
from pgw.service.auth import (
    TenantAuthenticator,
    TokenRecord,
    hash_token,
    issue_token_record,
    read_token_records,
    revoke_token,
)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_both_formats_load(tmp_path: Path) -> None:
    f = tmp_path / "tokens"
    f.write_text(f"# comment\nold:{hash_token('a')}\nnew:tk_1:{hash_token('b')}\n"
                 f"exp:tk_2:{hash_token('c')}:100\nto:tk_3:{hash_token('d')}::token-only\n")
    recs = read_token_records(f)
    assert [r.token_id for r in recs] == [f"legacy-{hash_token('a')[:8]}", "tk_1", "tk_2", "tk_3"]
    assert recs[0].legacy and recs[2].not_after == 100 and recs[3].token_only and recs[3].not_after is None
    a = TenantAuthenticator(f, wall_clock=lambda: 50.0)
    assert a.authenticate("Bearer a").tenant_id == "old"
    ident = a.authenticate("Bearer b")
    assert ident.tenant_id == "new" and ident.token_id == "tk_1" and ident.still_valid()  # noqa: S105
    assert a.authenticate("Bearer c").tenant_id == "exp"
    late = TenantAuthenticator(f, wall_clock=lambda: 100.0)
    with pytest.raises(AuthError, match="expired"):
        late.authenticate("Bearer c")
    for bad in (f"x:tk:{hash_token('e')}:soon", f"x:tk:{hash_token('e')}::admin", "x:bad id:" + "0" * 64,
                f"x:tk_1:{hash_token('e')}\nx:tk_1:{hash_token('f')}"):
        f.write_text(bad + "\n")
        with pytest.raises(ConfigError):
            read_token_records(f)


def test_record_line_roundtrip() -> None:
    r = TokenRecord("t", "tk_9", "ab" * 32, 1234.0, True)
    assert r.line() == f"t:tk_9:{'ab' * 32}:1234:token-only"
    assert TokenRecord("t", "tk_9", "ab" * 32).line() == f"t:tk_9:{'ab' * 32}"


def test_revoke_takes_effect_within_one_reload_interval(tmp_path: Path) -> None:
    f = tmp_path / "tokens"
    f.write_text("# keep me\n")
    tok, rec = issue_token_record("acme", f)
    tok2, _ = issue_token_record("acme", f)
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    clock = Clock()
    a = TenantAuthenticator(f, reload_interval_s=2.0, clock=clock)
    ident = a.authenticate(f"Bearer {tok}")
    assert ident.token_id == rec.token_id
    revoked = revoke_token(f, rec.token_id)
    assert revoked.token_id == rec.token_id
    assert f.read_text().startswith("# keep me\n") and rec.digest_hex not in f.read_text()
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 1_000_000))
    a.authenticate(f"Bearer {tok}")  # still cached: the interval has not passed
    clock.t += 2.0
    with pytest.raises(AuthError):
        a.authenticate(f"Bearer {tok}")
    assert not ident.still_valid()
    assert a.authenticate(f"Bearer {tok2}").tenant_id == "acme"  # the others still work
    with pytest.raises(ConfigError, match="no token"):
        revoke_token(f, rec.token_id)


def test_malformed_reload_keeps_previous_set(tmp_path: Path) -> None:
    f = tmp_path / "tokens"
    tok, _ = issue_token_record("acme", f)
    clock = Clock()
    a = TenantAuthenticator(f, clock=clock)
    f.write_text("garbage line\n")
    clock.t += 10
    assert a.authenticate(f"Bearer {tok}").tenant_id == "acme"


def test_enrollment_added_tokens_survive_until_the_file_has_them(tmp_path: Path) -> None:
    f = tmp_path / "tokens"
    f.write_text("")
    clock = Clock()
    a = TenantAuthenticator(f, allow_empty=True, clock=clock)
    tok, rec = issue_token_record("acme", f)
    a.add_token_hash(rec.digest_hex, "acme", token_id=rec.token_id)
    assert a.authenticate(f"Bearer {tok}").token_id == rec.token_id
    clock.t += 10
    assert a.authenticate(f"Bearer {tok}").token_id == rec.token_id  # reloaded from the file


def test_require_client_cert(tmp_path: Path) -> None:
    f = tmp_path / "tokens"
    tok, _ = issue_token_record("acme", f)
    tok_only, _ = issue_token_record("acme", f, token_only=True)
    m = tmp_path / "mtls"
    m.write_text("acme:pgw-gw.acme.gw-1\nbeta:pgw-gw.beta.gw-2\n")
    a = TenantAuthenticator(f, mtls_tenants_file=m, require_client_cert=True)
    with pytest.raises(PGWError) as ei:
        a.authenticate(f"Bearer {tok}")
    assert ei.value.http_status == 403
    assert a.authenticate(f"Bearer {tok}", "CN=pgw-gw.acme.gw-1").tenant_id == "acme"
    with pytest.raises(PGWError) as ei:
        a.authenticate(f"Bearer {tok}", "CN=pgw-gw.beta.gw-2")
    assert ei.value.http_status == 403
    assert a.authenticate(f"Bearer {tok_only}").tenant_id == "acme"  # enrolled without a CSR
    assert TenantAuthenticator(f).authenticate(f"Bearer {tok}").tenant_id == "acme"  # flag off


def test_settings_default_and_validation(tmp_path: Path) -> None:
    from tests.enroll_helpers import make_pki

    pki = make_pki(tmp_path / "pki")
    base = dict(client_issuing_ca_cert_file=pki["client_ca"], client_issuing_ca_key_file=pki["client_ca_key"])
    assert not ServiceSettings().compute_requires_client_cert
    assert ServiceSettings(**base, trusted_client_cert_header="x-cert").compute_requires_client_cert
    tls = {"cert_file": pki["server"], "key_file": pki["server_key"], "ca_file": pki["client_ca"]}
    assert ServiceSettings(**base, tls=tls).compute_requires_client_cert
    assert not ServiceSettings(**base).compute_requires_client_cert  # certificates cannot be verified
    assert not ServiceSettings(**base, tls=tls, require_client_cert_for_compute=False).compute_requires_client_cert
    with pytest.raises(ValueError, match="require_client_cert_for_compute"):
        ServiceSettings(require_client_cert_for_compute=True)


async def test_flag_on_routes(tmp_path: Path, tiny_pack_dir: Path) -> None:
    """Token without cert -> 403; token + mapped cert -> ok; enroll CA route works without a cert."""
    import httpx

    from tests.enroll_helpers import make_pki
    from tests.service.conftest import running_app

    pki = make_pki(tmp_path / "pki")
    f = tmp_path / "tokens"
    tok, _ = issue_token_record("acme", f)
    m = tmp_path / "mtls"
    m.write_text("acme:pgw-gw.acme.gw-1\n")
    s = ServiceSettings(packs=[tiny_pack_dir], device="cpu", tokens_file=f, mtls_tenants_file=m,
                        trusted_client_cert_header="x-client-cert", enrollment_file=tmp_path / "e.json",
                        public_url="http://127.0.0.1:9", enrollment_ca_file=pki["ca"],
                        client_issuing_ca_cert_file=pki["client_ca"], client_issuing_ca_key_file=pki["client_ca_key"],
                        rings=["K32A12C27"])
    assert s.compute_requires_client_cert
    hdr = {"authorization": f"Bearer {tok}", "x-pgw-protocol": "1"}
    async with running_app(s) as app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                                        base_url="http://svc") as h:
        assert (await h.get("/pgw/v1/models", headers=hdr)).status_code == 403
        r = await h.get("/pgw/v1/models", headers={**hdr, "x-client-cert": "CN=pgw-gw.acme.gw-1"})
        assert r.status_code == 200
        assert (await h.get("/pgw/v1/enroll/ca")).status_code == 200


async def test_revoked_token_channel_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A compute stream opened with a token is closed with 1008 once the token is revoked."""
    from fastapi import FastAPI

    import pgw.wire.server as wire_server
    from pgw.wire.messages import STREAM_PATH
    from pgw.wire.server import create_router

    monkeypatch.setattr(wire_server, "_CREDENTIAL_RECHECK_S", 0.05)
    f = tmp_path / "tokens"
    tok, rec = issue_token_record("acme", f)
    auth = TenantAuthenticator(f, reload_interval_s=0.01)

    class H:
        def authenticate(self, authorization: str | None, subject: str | None) -> object:
            return auth.authenticate(authorization, subject)

        def linear_dims(self, fingerprint: str, ring: str) -> dict[str, tuple[int, int]]:
            return {}

        async def compute(self, request: object, ctx: object) -> object:
            raise NotImplementedError

        async def models(self) -> object:
            raise NotImplementedError

    app = FastAPI()
    app.include_router(create_router(H()))  # type: ignore[arg-type]
    sent: list[dict[str, object]] = []
    inbox: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    await inbox.put({"type": "websocket.connect"})

    async def receive() -> dict[str, object]:
        return await inbox.get()

    async def send(msg: dict[str, object]) -> None:
        sent.append(msg)
        if msg["type"] == "websocket.close":
            await inbox.put({"type": "websocket.disconnect", "code": msg["code"]})

    scope = {"type": "websocket", "path": STREAM_PATH, "raw_path": STREAM_PATH.encode(), "query_string": b"",
             "headers": [(b"authorization", f"Bearer {tok}".encode()), (b"x-pgw-protocol", b"1")],
             "scheme": "ws", "server": ("svc", 80), "client": ("10.0.0.1", 1), "root_path": "",
             "subprotocols": [], "asgi": {"version": "3.0"}}
    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.sleep(0.2)
    assert [m["type"] for m in sent] == ["websocket.accept"]  # open while the token is valid
    revoke_token(f, rec.token_id)
    await asyncio.wait_for(task, timeout=5)
    closes = [m for m in sent if m["type"] == "websocket.close"]
    assert len(closes) == 1 and closes[0]["code"] == 1008 and closes[0]["reason"] == "credentials revoked"
