"""Real network test: hypercorn over TLS 1.2+/HTTP/2 with mutual TLS (SPEC 6.1, 6.7)."""

from __future__ import annotations

import asyncio
import datetime as dt
import ipaddress
import socket
import ssl
import threading
from pathlib import Path

import httpx
import pytest

pytest.importorskip("hypercorn")
pytest.importorskip("h2")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: E402

from pgw.core import RING_K40_A12, PGWError  # noqa: E402
from pgw.wire import ClientConfig, ComputeClient, build_ssl_context  # noqa: E402
from tests.wire.conftest import FP, FakeHandler, expected_y, make_app, make_request  # noqa: E402


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _write_key(path: Path, key) -> None:
    path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )


def _issue(ca_key, ca_cert, cn: str, *, server: bool):
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    b = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
    )
    if server:
        b = b.add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
    return key, b.sign(ca_key, hashes.SHA256())


@pytest.fixture(scope="module")
def pki(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("pki")
    ca_key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    ca = (
        x509.CertificateBuilder()
        .subject_name(_name("pgw-test-ca"))
        .issuer_name(_name("pgw-test-ca"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                          key_encipherment=False, data_encipherment=False, key_agreement=False,
                          encipher_only=False, decipher_only=False),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    paths = {k: d / f"{k}.pem" for k in ("ca", "server", "server_key", "client", "client_key")}
    paths["ca"].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    for role in ("server", "client"):
        key, cert = _issue(ca_key, ca, f"pgw-test-{role}", server=role == "server")
        paths[role].write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        _write_key(paths[f"{role}_key"], key)
    return paths


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(pki):
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    handler = FakeHandler()
    app = make_app(handler)
    port = _free_port()
    cfg = Config()
    cfg.bind = [f"127.0.0.1:{port}"]
    cfg.certfile = str(pki["server"])
    cfg.keyfile = str(pki["server_key"])
    cfg.ca_certs = str(pki["ca"])
    cfg.verify_mode = ssl.CERT_REQUIRED
    cfg.accesslog = None
    cfg.errorlog = None
    loop = asyncio.new_event_loop()
    stop = asyncio.Event()
    started = threading.Event()

    async def main() -> None:
        started.set()
        await serve(app, cfg, shutdown_trigger=stop.wait)  # type: ignore[arg-type]

    t = threading.Thread(target=lambda: loop.run_until_complete(main()), daemon=True)
    t.start()
    started.wait(5)
    for _ in range(100):  # wait until the port accepts connections
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            import time

            time.sleep(0.05)
    yield f"https://127.0.0.1:{port}", handler
    loop.call_soon_threadsafe(stop.set)
    t.join(10)


async def test_mtls_http2_end_to_end(pki, server):
    url, handler = server
    cfg = ClientConfig(base_url=url, ca_file=pki["ca"], client_cert_file=pki["client"],
                       client_key_file=pki["client_key"], request_timeout_s=20)
    # HTTP/2 is negotiated via ALPN with the client's SSL context
    async with httpx.AsyncClient(http2=True, verify=build_ssl_context(cfg), base_url=url) as raw:
        r = await raw.get("/pgw/v1/models")
        assert r.status_code == 200 and r.http_version == "HTTP/2"

    client = ComputeClient(cfg)
    await client.connect(FP, "K40A12")
    req, xs = make_request(RING_K40_A12, [("0.qkv", 3), ("0.down", 2)])
    # several concurrent requests multiplexed over the pool
    resps = await asyncio.gather(*(client.compute(req) for _ in range(4)))
    from pgw.core import unpack_ring
    from tests.wire.conftest import DIMS

    for resp in resps:
        for it, x, res in zip(req.items, xs, resp.items, strict=True):
            y = unpack_ring(res.y_hat, (it.rows, DIMS[it.linear][0]), RING_K40_A12)
            assert (y == expected_y(RING_K40_A12, it.linear, x, handler.weights)).all()
    # blocking API on the same client
    assert client.compute_sync(req) == resps[0]
    await client.aclose()


async def test_server_rejects_client_without_certificate(pki, server):
    url, _ = server
    cfg = ClientConfig(base_url=url, ca_file=pki["ca"], max_retries=1, backoff_base_s=0.001, request_timeout_s=5)
    client = ComputeClient(cfg)
    with pytest.raises(PGWError) as ei:
        await client.models()
    assert ei.value.retryable or ei.value.code.value in ("unavailable", "deadline_exceeded")
    await client.aclose()


async def test_client_rejects_untrusted_server(pki, server, tmp_path):
    url, _ = server
    other_ca_key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    other = (
        x509.CertificateBuilder().subject_name(_name("other")).issuer_name(_name("other"))
        .public_key(other_ca_key.public_key()).serial_number(1)
        .not_valid_before(now - dt.timedelta(minutes=1)).not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(other_ca_key, hashes.SHA256())
    )
    ca2 = tmp_path / "other.pem"
    ca2.write_bytes(other.public_bytes(serialization.Encoding.PEM))
    cfg = ClientConfig(base_url=url, ca_file=ca2, client_cert_file=pki["client"], client_key_file=pki["client_key"],
                       max_retries=0, request_timeout_s=5)
    client = ComputeClient(cfg)
    with pytest.raises(PGWError):
        await client.models()
    await client.aclose()
