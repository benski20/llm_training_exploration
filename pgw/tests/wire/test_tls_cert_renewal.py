"""Real TLS handshakes: an expired client certificate is not presented (token-only fallback), and a renewed
pair loaded with :func:`reload_client_cert` is what the next connection presents."""

from __future__ import annotations

import asyncio
import datetime as dt
import ssl
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from pgw.wire.client import ClientConfig, build_ssl_context, reload_client_cert
from tests.enroll_helpers import make_pki


def _client_cert(pki: dict[str, Path], d: Path, name: str, *, not_before: dt.datetime, not_after: dt.datetime,
                 ) -> tuple[Path, Path, int]:
    ca = x509.load_pem_x509_certificate(pki["client_ca"].read_bytes())
    ca_key = serialization.load_pem_private_key(pki["client_ca_key"].read_bytes(), password=None)
    key = ec.generate_private_key(ec.SECP256R1())
    serial = x509.random_serial_number()
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "pgw-gw.acme.gw-1")]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(serial)
            .not_valid_before(not_before).not_valid_after(not_after)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),  # type: ignore[arg-type]
                           critical=False)
            .sign(ca_key, hashes.SHA256()))  # type: ignore[arg-type]
    c, k = d / f"{name}.crt", d / f"{name}.key"
    c.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    k.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    return c, k, serial


class _Server:
    """TLS listener with optional client certificates (as the compute service); records each peer's serial."""

    def __init__(self, pki: dict[str, Path]):
        self.ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.ctx.load_cert_chain(str(pki["server"]), str(pki["server_key"]))
        self.ctx.load_verify_locations(cafile=str(pki["client_ca"]))
        self.ctx.verify_mode = ssl.CERT_OPTIONAL
        self.serials: list[int | None] = []
        self.port = 0
        self.server: asyncio.base_events.Server | None = None

    async def __aenter__(self) -> _Server:
        async def on_conn(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            der = w.get_extra_info("ssl_object").getpeercert(binary_form=True)
            self.serials.append(x509.load_der_x509_certificate(der).serial_number if der else None)
            w.write(b"ok")
            await w.drain()
            w.close()

        self.server = await asyncio.start_server(on_conn, "127.0.0.1", 0, ssl=self.ctx)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def connect(self, ctx: ssl.SSLContext) -> bytes:
        r, w = await asyncio.open_connection("127.0.0.1", self.port, ssl=ctx, server_hostname="127.0.0.1")
        try:
            return await asyncio.wait_for(r.read(2), timeout=5)
        finally:
            w.close()


def _cfg(pki: dict[str, Path], cert: Path, key: Path) -> ClientConfig:
    return ClientConfig(base_url="https://127.0.0.1:1", ca_file=pki["ca"], client_cert_file=cert, client_key_file=key)


async def test_expired_cert_fails_handshake_raw_but_falls_back_to_token_only(tmp_path: Path) -> None:
    pki = make_pki(tmp_path / "pki")
    now = dt.datetime.now(dt.UTC)
    cert, key, _ = _client_cert(pki, tmp_path, "old", not_before=now - dt.timedelta(days=31),
                                not_after=now - dt.timedelta(minutes=1))
    async with _Server(pki) as srv:
        # before the fix: the expired certificate is presented and the service rejects the handshake
        raw = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(pki["ca"]))
        raw.load_cert_chain(str(cert), str(key))
        with pytest.raises((ssl.SSLError, ConnectionError, asyncio.IncompleteReadError)):
            data = await srv.connect(raw)
            if data != b"ok":  # TLS 1.3 reports the client-cert rejection on the first read
                raise ConnectionError("rejected")
        # after: the certificate is not presented; the connection works (token-only authentication)
        assert await srv.connect(build_ssl_context(_cfg(pki, cert, key))) == b"ok"
        assert srv.serials[-1] is None


async def test_cert_inside_expiry_margin_is_not_presented(tmp_path: Path) -> None:
    pki = make_pki(tmp_path / "pki")
    now = dt.datetime.now(dt.UTC)
    cert, key, serial = _client_cert(pki, tmp_path, "c", not_before=now - dt.timedelta(days=30),
                                     not_after=now + dt.timedelta(seconds=30))
    async with _Server(pki) as srv:
        assert await srv.connect(build_ssl_context(_cfg(pki, cert, key))) == b"ok"
        assert srv.serials == [None]
        later = build_ssl_context(_cfg(pki, cert, key), now=now.timestamp() - 3600)
        assert await srv.connect(later) == b"ok"
        assert srv.serials[-1] == serial


async def test_reloaded_pair_is_presented_on_the_next_connection(tmp_path: Path) -> None:
    pki = make_pki(tmp_path / "pki")
    now = dt.datetime.now(dt.UTC)
    cert, key, s1 = _client_cert(pki, tmp_path, "gw", not_before=now - dt.timedelta(minutes=1),
                                 not_after=now + dt.timedelta(hours=1))
    ctx = build_ssl_context(_cfg(pki, cert, key))
    async with _Server(pki) as srv:
        assert await srv.connect(ctx) == b"ok"
        new_c, new_k, s2 = _client_cert(pki, tmp_path, "new", not_before=now - dt.timedelta(minutes=1),
                                        not_after=now + dt.timedelta(hours=2))
        new_c.replace(cert)
        new_k.replace(key)
        assert reload_client_cert(cert, key) >= 1
        assert await srv.connect(ctx) == b"ok"  # the same (live) context, the new certificate
    assert srv.serials == [s1, s2]
