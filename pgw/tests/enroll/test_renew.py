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

    def der(self, path: Path | None = None) -> bytes:
        return _der((path or self.cert).read_text())

    def transport(self) -> httpx.MockTransport:
        """Hands the request to the service as the TLS layer would: with the subject of the cert on disk."""

        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/pgw/v1/enroll/renew"
            try:
                out = self.svc.renew(json.loads(req.content), "10.0.0.1", req.headers.get("authorization"),
                                     self.subject(), self.der())
            except PGWError as exc:
                return httpx.Response(exc.http_status, json={"error": {"code": str(exc.code), "message": exc.message}})
            return httpx.Response(200, json=out)

        return httpx.MockTransport(handler)


def _der(pem: str) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return x509.load_pem_x509_certificate(pem.encode()).public_bytes(serialization.Encoding.DER)


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

    def renew(b: Any = body, tok: str | None = token, s: str | None = subj, der: bytes | None = _der(cur)) -> Any:
        return env.svc.renew(b, "10.0.0.9", f"Bearer {tok}" if tok else None, s, der)

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
    with pytest.raises(PGWError) as ei:
        renew(der=None)  # the subject alone is not enough: the handshake certificate itself is needed
    assert ei.value.http_status == 403 and "certificate itself" in ei.value.message
    # a second certificate with the same subject (CN) and tenant, valid and issued by the same CA: the body
    # certificate must be byte-identical to the one presented in the handshake, not just share its subject
    sibling = env.svc.renew(body, "10.0.0.9", f"Bearer {token}", subj, _der(cur))["client_cert_pem"]
    assert x509.load_pem_x509_certificate(sibling.encode()).subject.rfc4514_string() == subj
    with pytest.raises(PGWError) as ei:
        renew(der=_der(sibling))  # handshake: the sibling; body: the original
    assert ei.value.http_status == 403
    with pytest.raises(PGWError) as ei:
        renew(b={"csr_pem": csr, "cert_pem": sibling})  # handshake: the original; body: the sibling
    assert ei.value.http_status == 403
    reasons = [json.loads(line).get("reason") for line in env.audit.read_text().splitlines()]
    assert "no_client_cert_der" in reasons and reasons.count("cert_mismatch") >= 3
    out = renew()
    new = x509.load_pem_x509_certificate(out["client_cert_pem"].encode())
    old = x509.load_pem_x509_certificate(cur.encode())
    assert new.subject == old.subject and new.serial_number != old.serial_number
    events = [json.loads(line) for line in env.audit.read_text().splitlines()]
    ok = [e for e in events if e["event"] == "enroll.renewed"]
    assert len(ok) == 2 and ok[-1]["tenant"] == "acme"
    ok = ok[-1:]
    assert ok[0]["cert_serial"] == f"{new.serial_number:x}" and ok[0]["prev_cert_serial"] == f"{old.serial_number:x}"
    assert token not in json.dumps(events)


def test_expired_cert_is_never_renewed(tmp_path: Path) -> None:
    env = Env(tmp_path, validity_days=1.0 / 86400)  # 1 s
    token = env.enroll()
    time.sleep(1.2)
    _, csr = _key_and_csr("x")
    with pytest.raises(PGWError) as ei:
        env.svc.renew({"csr_pem": csr, "cert_pem": env.cert.read_text()}, "10.0.0.1", f"Bearer {token}",
                      env.subject(), env.der())
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
                               f"Bearer {token}", env.subject(), env.der())["client_cert_pem"]
    t = httpx.MockTransport(lambda req: httpx.Response(200, json={"client_cert_pem": other_cert}))
    with pytest.raises(PGWError, match="not for this gateway"):
        renew_client_cert(URL, token, env.cert, env.key, transport=t)
    assert (env.cert.read_bytes(), env.key.read_bytes()) == before  # nothing changed
    assert not [p for p in os.listdir(env.dir) if ".tmp" in p]


def test_enroll_limiter_counts_failures_only(tmp_path: Path) -> None:
    from pgw.core.errors import RateLimitedError
    from pgw.enroll.service import _IPRateLimiter

    env = Env(tmp_path)
    env.svc.limiter = _IPRateLimiter(1, global_per_minute=3)
    for i in range(3):
        env.enroll(name=f"gw-{i}")  # successes from one address never count
    with pytest.raises(PGWError):
        env.svc.enroll({"secret": "00" * 16, "gateway_name": "x"}, "10.0.0.1")
    with pytest.raises(RateLimitedError):
        env.svc.enroll({"secret": "00" * 16, "gateway_name": "x"}, "10.0.0.1")
    for ip in ("10.0.0.2", "10.0.0.3"):
        with pytest.raises(PGWError):
            env.svc.enroll({"secret": "00" * 16, "gateway_name": "x"}, ip)
    with pytest.raises(RateLimitedError):  # global ceiling: rotating addresses does not help
        env.svc.enroll({"secret": "00" * 16, "gateway_name": "x"}, "10.0.0.4")


def test_enroll_limiter_admission_is_atomic(tmp_path: Path) -> None:
    """N concurrent wrong guesses cannot all pass the check before their failures are counted."""
    import threading

    from pgw.core.errors import RateLimitedError
    from pgw.enroll.service import _IPRateLimiter

    env = Env(tmp_path)
    env.svc.limiter = _IPRateLimiter(2, global_per_minute=100)
    gate = threading.Event()
    entered = threading.Semaphore(0)
    orig = env.svc._enroll

    def slow(body: Any, client: str) -> Any:
        entered.release()
        gate.wait(10)
        return orig(body, client)

    env.svc._enroll = slow  # type: ignore[method-assign]
    results: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        try:
            env.svc.enroll({"secret": "00" * 16, "gateway_name": "x"}, "10.0.0.7")
            r = "ok"
        except RateLimitedError:
            r = "limited"
        except PGWError:
            r = "failed"
        with lock:
            results.append(r)

    ts = [threading.Thread(target=attempt) for _ in range(10)]
    for t in ts:
        t.start()
    for _ in range(2):
        assert entered.acquire(timeout=10)
    for _ in range(100):  # the other 8 are refused without waiting for the 2 in flight
        with lock:
            if len(results) == 8:
                break
        time.sleep(0.01)
    with lock:
        assert results == ["limited"] * 8
    gate.set()
    for t in ts:
        t.join()
    assert sorted(results) == ["failed"] * 2 + ["limited"] * 8
    with pytest.raises(RateLimitedError):  # both failures were counted
        env.svc.enroll({"secret": "00" * 16, "gateway_name": "x"}, "10.0.0.7")
    # a successful attempt releases its slot without counting
    env.svc._enroll = orig  # type: ignore[method-assign]
    env.svc.limiter = _IPRateLimiter(1, global_per_minute=100)
    for i in range(3):
        env.enroll(name=f"ok-{i}")


async def test_renew_route(tmp_path: Path) -> None:
    from starlette.applications import Starlette

    from pgw.enroll.service import add_enroll_routes
    from pgw.service.app import _HeaderToTLSExt

    env = Env(tmp_path)
    token = env.enroll()
    app = Starlette()
    add_enroll_routes(app, env.settings, env.auth, lambda: [])
    app.add_middleware(_HeaderToTLSExt, header="x-client-cert", pem_header="x-client-cert-pem")
    from urllib.parse import quote

    body = {"csr_pem": _key_and_csr("x")[1], "cert_pem": env.cert.read_text()}
    auth = {"authorization": f"Bearer {token}"}
    pem_hdr = {"x-client-cert-pem": quote(env.cert.read_text())}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
        r = await h.post("/pgw/v1/enroll/renew", json=body, headers=auth)
        assert r.status_code == 403  # no client certificate
        r = await h.post("/pgw/v1/enroll/renew", json=body, headers={**auth, "x-client-cert": env.subject()})
        assert r.status_code == 403  # the subject alone: the certificate itself is needed
        r = await h.post("/pgw/v1/enroll/renew", json=body, headers={**auth, **pem_hdr})
        assert r.status_code == 403  # a certificate without the verified subject header is ignored
        r = await h.post("/pgw/v1/enroll/renew", json=body,
                         headers={**auth, "x-client-cert": env.subject(), **pem_hdr})
        assert r.status_code == 200, r.text
        assert r.headers["cache-control"] == "no-store" and "BEGIN CERTIFICATE" in r.json()["client_cert_pem"]


def test_renew_over_real_mtls(tiny_pack_dir: Path, tmp_path: Path) -> None:
    """Native TLS (uvicorn): enroll, compute needs the certificate, renewal checks the handshake certificate
    byte for byte (a same-subject sibling in the body is refused), and the renewed pair works."""
    import ssl

    from pgw.enroll.client import enroll_gateway
    from tests.enroll_helpers import enroll_service

    mtls = tmp_path / "mtls.txt"
    mtls.write_text("")
    with enroll_service(tiny_pack_dir, tmp_path / "svc", verify_client_certs=True, mtls_tenants_file=mtls,
                        require_client_cert_for_compute=True) as svc:
        assert svc.settings.compute_requires_client_cert
        code, _ = svc.create_code("acme")
        cfg = tmp_path / "etc"
        res = enroll_gateway(code, cfg, gateway_name="gw-1")
        files = res.files
        token = files["token"].read_text().strip()
        assert "acme:pgw-gw.acme.gw-1" in mtls.read_text()  # persisted for restarts

        def ctx(cert: Path | None = files["client_cert"]) -> ssl.SSLContext:
            c = ssl.create_default_context(cafile=str(files["ca"]))
            if cert is not None:
                c.load_cert_chain(str(cert), str(files["client_key"]))
            return c

        hdr = {"authorization": f"Bearer {token}", "PGW-Protocol": "1"}
        assert httpx.get(f"{svc.url}/pgw/v1/models", headers=hdr, verify=ctx(None)).status_code == 403
        assert httpx.get(f"{svc.url}/pgw/v1/models", headers=hdr, verify=ctx()).status_code == 200
        # a sibling certificate: same CN, same CA, valid, but not the one presented in the handshake
        first = files["client_cert"].read_text()
        sibling = httpx.post(f"{svc.url}/pgw/v1/enroll/renew", headers=hdr, verify=ctx(),
                             json={"csr_pem": _key_and_csr("x")[1], "cert_pem": first})
        assert sibling.status_code == 200, sibling.text
        sib_pem = sibling.json()["client_cert_pem"]
        r = httpx.post(f"{svc.url}/pgw/v1/enroll/renew", headers=hdr, verify=ctx(),
                       json={"csr_pem": _key_and_csr("y")[1], "cert_pem": sib_pem})
        assert r.status_code == 403
        # the real renewer: presents the current pair, sends the same certificate, installs the new one
        serial = renew_client_cert(svc.url, token, files["client_cert"], files["client_key"], ca_file=files["ca"])
        assert _serial(files["client_cert"]) == serial
        assert httpx.get(f"{svc.url}/pgw/v1/models", headers=hdr, verify=ctx()).status_code == 200
