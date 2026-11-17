"""Real-network tests: hypercorn with TLS + mTLS + HTTP/2, separate metrics port, SIGTERM drain."""

from __future__ import annotations

import asyncio
import datetime as dt
import ipaddress
import os
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch

pytest.importorskip("hypercorn")
pytest.importorskip("h2")

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from pydantic import SecretStr

from pgw.core import RING_K40_A12, ErrorCode, PGWError, unpack_ring
from pgw.service import ServiceSettings, build_hypercorn_config, hash_token, serve_async
from pgw.wire import ClientConfig, ComputeClient
from tests.service.conftest import TOKEN_A, expected, make_req, rand_x

REPO = Path(__file__).resolve().parents[2]


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _key_pem(key: Any) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _issue(ca_key: Any, ca: Any, cn: str, server: bool) -> tuple[Any, Any]:
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    b = (
        x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(ca.subject).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.ExtendedKeyUsage(
            [ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
    )
    if server:
        b = b.add_extension(x509.SubjectAlternativeName(
            [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
    return key, b.sign(ca_key, hashes.SHA256())


def _ca(cn: str) -> tuple[Any, Any]:
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(_name(cn)).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                     content_commitment=False, key_encipherment=False, data_encipherment=False,
                                     key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


@pytest.fixture(scope="module")
def pki(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("svc-pki")
    ca_key, ca = _ca("pgw-service-test-ca")
    rogue_key, rogue_ca = _ca("rogue-ca")
    out: dict[str, Path] = {"ca": d / "ca.pem"}
    out["ca"].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    for role, issuer_key, issuer in (("server", ca_key, ca), ("client", ca_key, ca), ("rogue", rogue_key, rogue_ca)):
        k, c = _issue(issuer_key, issuer, f"pgw-{role}", server=role == "server")
        out[role] = d / f"{role}.pem"
        out[f"{role}_key"] = d / f"{role}.key"
        out[role].write_bytes(c.public_bytes(serialization.Encoding.PEM))
        out[f"{role}_key"].write_bytes(_key_pem(k))
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _ServerThread:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.loop = asyncio.new_event_loop()
        self.stop_event: asyncio.Event | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)

        async def main() -> None:
            self.stop_event = asyncio.Event()
            await serve_async(self.settings, shutdown_event=self.stop_event)

        try:
            self.loop.run_until_complete(main())
        except BaseException as exc:  # surfaced by the test
            self.error = exc

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        if self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(timeout=30)
        assert not self.thread.is_alive()


def _wait_ready(url: str, *, verify: Any = False, timeout: float = 60.0) -> None:
    t0 = time.monotonic()
    last: BaseException | None = None
    while time.monotonic() - t0 < timeout:
        try:
            r = httpx.get(f"{url}/readyz", verify=verify, timeout=2.0)
            if r.status_code == 200:
                return
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(0.1)
    raise TimeoutError(f"service at {url} not ready: {last!r}")


@pytest.fixture(scope="module")
def mtls_server(tiny_pack_dir: Path, pki: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    tok = tmp_path_factory.mktemp("tok") / "tenants.txt"
    tok.write_text(f"tenant-a:{hash_token(TOKEN_A)}\n")
    port, mport = _free_port(), _free_port()
    s = ServiceSettings(
        host="127.0.0.1", port=port, metrics_port=mport, metrics_on_main_port=False, packs=[tiny_pack_dir],
        device="cpu", tokens_file=tok, batch_window_ms=1.0,
        tls={"cert_file": pki["server"], "key_file": pki["server_key"], "ca_file": pki["ca"],
             "require_client_cert": True},
    )
    srv = _ServerThread(s)
    srv.start()
    url = f"https://127.0.0.1:{port}"
    ctx = ssl.create_default_context(cafile=str(pki["ca"]))
    ctx.load_cert_chain(str(pki["client"]), str(pki["client_key"]))
    _wait_ready(url, verify=ctx)
    yield {"url": url, "metrics": f"http://127.0.0.1:{mport}", "ctx": ctx, "server": srv}
    srv.stop()
    assert srv.error is None, srv.error


async def test_mtls_http2_compute_exact(mtls_server: dict[str, Any], pki: dict[str, Path], pack_fp: str,
                                        pack_weights: dict[str, torch.Tensor]) -> None:
    cfg = ClientConfig(base_url=mtls_server["url"], token=SecretStr(TOKEN_A), ca_file=pki["ca"],
                       client_cert_file=pki["client"], client_key_file=pki["client_key"])
    async with ComputeClient(cfg) as c:
        await c.connect(pack_fp, "K40A12")
        ring = RING_K40_A12
        xs = [rand_x(1 + i, 128, ring, seed=i) for i in range(6)]
        resps = await asyncio.gather(*(c.compute(make_req(pack_fp, ring, [("1.down", x)], rid=f"m{i}"))
                                       for i, x in enumerate(xs)))
        for x, r in zip(xs, resps, strict=True):
            y = unpack_ring(r.items[0].y_hat, (x.shape[0], 64), ring)
            assert torch.equal(y, expected(x, pack_weights["1.down"], ring))
    # negotiated HTTP/2 over TLS
    async with httpx.AsyncClient(http2=True, verify=mtls_server["ctx"]) as h:
        r = await h.get(f"{mtls_server['url']}/healthz")
        assert r.status_code == 200 and r.http_version == "HTTP/2"
        assert (await h.get(f"{mtls_server['url']}/metrics")).status_code == 404  # not on the public port


async def test_mtls_rejects_missing_or_foreign_client_cert(mtls_server: dict[str, Any], pki: dict[str, Path]) -> None:
    no_cert = ssl.create_default_context(cafile=str(pki["ca"]))
    with pytest.raises(httpx.HTTPError):
        async with httpx.AsyncClient(verify=no_cert) as h:
            await h.get(f"{mtls_server['url']}/healthz")
    rogue = ssl.create_default_context(cafile=str(pki["ca"]))
    rogue.load_cert_chain(str(pki["rogue"]), str(pki["rogue_key"]))
    with pytest.raises(httpx.HTTPError):
        async with httpx.AsyncClient(verify=rogue) as h:
            await h.get(f"{mtls_server['url']}/healthz")


async def test_token_still_required_over_mtls(mtls_server: dict[str, Any], pki: dict[str, Path], pack_fp: str) -> None:
    cfg = ClientConfig(base_url=mtls_server["url"], ca_file=pki["ca"], client_cert_file=pki["client"],
                       client_key_file=pki["client_key"], max_retries=0)
    async with ComputeClient(cfg) as c:
        with pytest.raises(PGWError) as ei:
            await c.compute(make_req(pack_fp, RING_K40_A12, [("0.o", rand_x(1, 64, RING_K40_A12))]))
        assert ei.value.code == ErrorCode.UNAUTHENTICATED


def test_separate_metrics_port(mtls_server: dict[str, Any]) -> None:
    r = httpx.get(f"{mtls_server['metrics']}/metrics", timeout=5)
    assert r.status_code == 200
    assert "pgw_service_requests_total" in r.text and "pgw_service_inflight" in r.text


def test_build_hypercorn_config(tmp_path: Path, pki: dict[str, Path]) -> None:
    s = ServiceSettings(host="::1", port=1234, allow_unauthenticated=True)
    cfg = build_hypercorn_config(s)
    assert cfg.bind == ["[::1]:1234"] and not cfg.ssl_enabled and cfg.accesslog is None
    s2 = ServiceSettings(port=1, allow_unauthenticated=True, tls={"cert_file": tmp_path / "x.pem", "key_file": tmp_path / "y"})
    from pgw.core import ConfigError

    with pytest.raises(ConfigError):
        build_hypercorn_config(s2)
    s3 = ServiceSettings(tls={"cert_file": pki["server"], "key_file": pki["server_key"], "ca_file": pki["ca"],
                              "require_client_cert": True}, allow_unauthenticated=True)
    cfg3 = build_hypercorn_config(s3)
    assert cfg3.ssl_enabled and cfg3.verify_mode == ssl.CERT_REQUIRED and "h2" in cfg3.alpn_protocols


@pytest.mark.slow
def test_sigterm_graceful_shutdown_subprocess(tiny_pack_dir: Path, tmp_path: Path, pack_fp: str) -> None:
    port = _free_port()
    env = {
        **os.environ,
        "PGW_SERVICE_HOST": "127.0.0.1",
        "PGW_SERVICE_PORT": str(port),
        "PGW_SERVICE_PACKS": f'["{tiny_pack_dir}"]',
        "PGW_SERVICE_DEVICE": "cpu",
        "PGW_SERVICE_ALLOW_UNAUTHENTICATED": "true",
        "PGW_SERVICE_BATCH_WINDOW_MS": "800",
        "PGW_SERVICE_DRAIN_TIMEOUT_S": "10",
        "PYTHONPATH": str(REPO / "src"),
    }
    proc = subprocess.Popen([sys.executable, "-m", "pgw.service"], env=env, cwd=tmp_path,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{port}"
        _wait_ready(url)
        ring = RING_K40_A12
        x = rand_x(2, 64, ring)
        result: dict[str, Any] = {}

        def call() -> None:
            async def go() -> None:
                async with ComputeClient(ClientConfig(base_url=url, http2=False, max_retries=0)) as c:
                    result["resp"] = await c.compute(make_req(pack_fp, ring, [("0.o", x)]))
            asyncio.run(go())

        t = threading.Thread(target=call)
        t.start()
        time.sleep(0.3)  # request is now waiting in the 800 ms batch window
        proc.send_signal(signal.SIGTERM)
        t.join(timeout=30)
        out, _ = proc.communicate(timeout=30)
        assert proc.returncode == 0, out.decode(errors="replace")[-2000:]
        assert "resp" in result, "in-flight request must complete during drain"
        assert result["resp"].items[0].rows == 2
    finally:
        if proc.poll() is None:
            proc.kill()
