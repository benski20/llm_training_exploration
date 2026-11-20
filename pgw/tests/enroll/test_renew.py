"""Client-certificate renewal: ``POST /pgw/v1/enroll/renew`` rules and the gateway's fake-clock renewer."""

from __future__ import annotations

import functools
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography import x509

from pgw.core.errors import PGWError
from pgw.enroll.client import CertRenewer, _key_and_csr, renew_client_cert
from pgw.enroll.service import EnrollmentService
from pgw.enroll.store import EnrollmentStore
from pgw.service import ServiceSettings
from pgw.service.auth import TenantAuthenticator
from tests.enroll_helpers import make_pki

URL = "https://compute.example.net:8443"


class Env:
    def __init__(self, tmp: Path, validity_days: float = 30.0):
        self.pki = make_pki(tmp / "pki")
        (tmp / "tenants.txt").write_text("")
        (tmp / "mtls.txt").write_text("")
        self.audit = tmp / "enroll.audit.jsonl"
        self.settings = ServiceSettings(
            tokens_file=tmp / "tenants.txt", enrollment_file=tmp / "enroll.json", public_url=URL,
            enrollment_ca_file=self.pki["ca"], client_issuing_ca_cert_file=self.pki["client_ca"],
            client_issuing_ca_key_file=self.pki["client_ca_key"], mtls_tenants_file=tmp / "mtls.txt",
            trusted_client_cert_header="x-client-cert", client_cert_validity_days=validity_days)
        self.auth = TenantAuthenticator(None, allow_empty=True)
        self.svc = EnrollmentService(self.settings, self.auth,
                                     lambda: [{"name": "m", "fingerprint": "ab" * 32, "rings": ["K32A12C27"]}])
        self.dir = tmp / "gw"
        self.dir.mkdir()
        self.cert, self.key = self.dir / "client.crt", self.dir / "secrets" / "client.key"

    def enroll(self, tenant: str = "acme", name: str = "gw-1") -> str:
        _, secret = EnrollmentStore(self.settings.enrollment_file).create(tenant, 3600)  # type: ignore[arg-type]
        key_pem, csr = _key_and_csr(name)
        out = self.svc.enroll({"secret": secret.hex(), "gateway_name": name, "csr_pem": csr}, "10.0.0.1")
        self.cert.write_text(out["client_cert_pem"])
        self.key.parent.mkdir(exist_ok=True)
        self.key.write_bytes(key_pem)
        return str(out["service_token"])

    def subject(self, path: Path | None = None) -> str:
        return x509.load_pem_x509_certificate((path or self.cert).read_bytes()).subject.rfc4514_string()

    def transport(self) -> httpx.MockTransport:
        """Hands the request to the service as the TLS layer would: with the subject of the cert on disk."""

        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/pgw/v1/enroll/renew"
            try:
                out = self.svc.renew(json.loads(req.content), "10.0.0.1", req.headers.get("authorization"),
                                     self.subject())
            except PGWError as exc:
                return httpx.Response(exc.http_status, json={"error": {"code": str(exc.code), "message": exc.message}})
            return httpx.Response(200, json=out)

        return httpx.MockTransport(handler)


def _serial(p: Path) -> int:
    return int(x509.load_pem_x509_certificate(p.read_bytes()).serial_number)


def test_renew_rules(tmp_path: Path) -> None:
    env = Env(tmp_path)
    token = env.enroll()
    cur = env.cert.read_text()
    subj = env.subject()
    other = env.enroll("beta", "gw-2")
    beta_cert = env.cert.read_text()
    _, csr = _key_and_csr("x")
    body = {"csr_pem": csr, "cert_pem": cur}

    def renew(b: Any = body, tok: str | None = token, s: str | None = subj) -> Any:
        return env.svc.renew(b, "10.0.0.9", f"Bearer {tok}" if tok else None, s)

    with pytest.raises(PGWError) as ei:
        renew(tok=None)
    assert ei.value.http_status == 401
    with pytest.raises(PGWError) as ei:
        renew(s=None)  # no certificate in the handshake (e.g. it expired)
    assert ei.value.http_status == 403
    with pytest.raises(PGWError) as ei:
        renew(tok=other)  # beta's token with acme's certificate
    assert ei.value.http_status == 403
    with pytest.raises(PGWError) as ei:
        renew(b={"csr_pem": csr, "cert_pem": beta_cert})  # body cert is not the TLS one
    assert ei.value.http_status == 403
    with pytest.raises(PGWError) as ei:
        renew(b={"csr_pem": "junk", "cert_pem": cur})
    assert ei.value.http_status == 400
    out = renew()
    new = x509.load_pem_x509_certificate(out["client_cert_pem"].encode())
    old = x509.load_pem_x509_certificate(cur.encode())
    assert new.subject == old.subject and new.serial_number != old.serial_number
    events = [json.loads(line) for line in env.audit.read_text().splitlines()]
    ok = [e for e in events if e["event"] == "enroll.renewed"]
    assert len(ok) == 1 and ok[0]["tenant"] == "acme"
    assert ok[0]["cert_serial"] == f"{new.serial_number:x}" and ok[0]["prev_cert_serial"] == f"{old.serial_number:x}"
    assert token not in json.dumps(events)


def test_expired_cert_is_never_renewed(tmp_path: Path) -> None:
    env = Env(tmp_path, validity_days=1.0 / 86400)  # 1 s
    token = env.enroll()
    time.sleep(1.2)
    _, csr = _key_and_csr("x")
    with pytest.raises(PGWError) as ei:
        env.svc.renew({"csr_pem": csr, "cert_pem": env.cert.read_text()}, "10.0.0.1", f"Bearer {token}",
                      env.subject())
    assert ei.value.http_status == 403 and "expired" in ei.value.message


async def test_fake_clock_renews_a_2s_cert_before_expiry(tmp_path: Path) -> None:
    from pgw.wire.client import client_cert_validity

    env = Env(tmp_path, validity_days=2.0 / 86400)
    token = env.enroll()
    w = client_cert_validity(env.cert)
    assert w is not None
    nb, na = w
    old_serial = _serial(env.cert)
    clock = [time.time()]

    async def fake_sleep(s: float) -> None:
        clock[0] += s

    renew = functools.partial(renew_client_cert, URL, token, env.cert, env.key, transport=env.transport())
    r = CertRenewer(env.cert, renew, clock=lambda: clock[0], sleep=fake_sleep)
    due = r.renew_at()
    assert due is not None and nb < due < na and abs(due - (nb + (na - nb) * 2 / 3)) < 1e-6
    assert await r.run_once()
    assert r.renewals == 1 and clock[0] < na  # renewed before the old certificate expired
    assert _serial(env.cert) != old_serial
    assert stat.S_IMODE(env.key.stat().st_mode) == 0o600
    w2 = client_cert_validity(env.cert)
    assert w2 is not None and w2[1] >= na


async def test_renewer_retries_with_backoff(tmp_path: Path) -> None:
    env = Env(tmp_path, validity_days=1.0)
    env.enroll()
    clock = [time.time()]
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock[0] += s

    calls = [0]

    def flaky() -> None:
        calls[0] += 1
        if calls[0] < 3:
            raise PGWError("service down")

    r = CertRenewer(env.cert, flaky, clock=lambda: clock[0], sleep=fake_sleep, retry_min_s=10, retry_max_s=100)
    assert await r.run_once()
    assert calls[0] == 3 and r.failures == 2
    assert 5 <= sleeps[1] <= 15 and 10 <= sleeps[2] <= 30  # jittered exponential backoff


def test_renew_client_cert_rejects_cert_for_another_key(tmp_path: Path) -> None:
    env = Env(tmp_path)
    token = env.enroll()
    before = (env.cert.read_bytes(), env.key.read_bytes())
    other_cert = env.svc.renew({"csr_pem": _key_and_csr("z")[1], "cert_pem": env.cert.read_text()}, "1.1.1.1",
                               f"Bearer {token}", env.subject())["client_cert_pem"]
    t = httpx.MockTransport(lambda req: httpx.Response(200, json={"client_cert_pem": other_cert}))
    with pytest.raises(PGWError, match="not for this gateway"):
        renew_client_cert(URL, token, env.cert, env.key, transport=t)
    assert (env.cert.read_bytes(), env.key.read_bytes()) == before  # nothing changed
    assert not [p for p in os.listdir(env.dir) if ".tmp" in p]


async def test_renew_route(tmp_path: Path) -> None:
    from starlette.applications import Starlette

    from pgw.enroll.service import add_enroll_routes
    from pgw.service.app import _HeaderToTLSExt

    env = Env(tmp_path)
    token = env.enroll()
    app = Starlette()
    add_enroll_routes(app, env.settings, env.auth, lambda: [])
    app.add_middleware(_HeaderToTLSExt, header="x-client-cert")
    body = {"csr_pem": _key_and_csr("x")[1], "cert_pem": env.cert.read_text()}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
        r = await h.post("/pgw/v1/enroll/renew", json=body, headers={"authorization": f"Bearer {token}"})
        assert r.status_code == 403  # no client certificate
        r = await h.post("/pgw/v1/enroll/renew", json=body,
                         headers={"authorization": f"Bearer {token}", "x-client-cert": env.subject()})
        assert r.status_code == 200, r.text
        assert r.headers["cache-control"] == "no-store" and "BEGIN CERTIFICATE" in r.json()["client_cert_pem"]
