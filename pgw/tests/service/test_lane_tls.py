"""The service trip lane over TCP with TLS 1.3 (``lane = "tcp"``, package L1) and the native A1 client channel.

Covers: the ticket names the TLS port; a native ``TripChannel.connect_service_tls`` trip is byte-identical to
the exact integer product and to the stream path's reply; the server is verified (a wrong CA fails); there is no
plaintext TCP (an unencrypted ``open`` frame gets no lane reply); tickets are single use; a ticket asked for
with a client certificate is bound to it (no certificate or another certificate is refused); a required client
certificate is enforced in the handshake; the native UDS client speaks the same A1 frames; settings checks.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import ipaddress
import json
import secrets
import socket
from pathlib import Path
from typing import Any

import pytest
import torch
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from pgw.core import RingParams, pack_ring
from pgw.core.errors import ConfigError
from pgw.core.settings import TLSConfig
from pgw.service import ServiceSettings, create_app
from pgw.service.lane import LANE_OPEN_PATH, lane_native_available
from pgw.wire.messages import PROTOCOL_HEADER, PROTOCOL_VERSION

from .conftest import asgi_client, expected, http_client, make_req, rand_x, running_app

fast = pytest.importorskip("pgw_fast")
pytestmark = pytest.mark.skipif(
    not lane_native_available() or not hasattr(fast.LaneService, "port")
    or not hasattr(fast.TripChannel, "connect_service_tls"),
    reason="pgw_fast without the TCP+TLS trip lane")

K32 = RingParams.from_name("K32A8")


# ---- PKI ---------------------------------------------------------------------------------------------
def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _ca(cn: str) -> tuple[Any, Any]:
    now = dt.datetime.now(dt.UTC)
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(_name(cn)).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False),
                           critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    return cert, key


def _leaf(ca: tuple[Any, Any], cn: str, *, server: bool) -> tuple[bytes, bytes]:
    now = dt.datetime.now(dt.UTC)
    key = ec.generate_private_key(ec.SECP256R1())
    b = (x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(ca[0].subject).public_key(key.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
         .not_valid_after(now + dt.timedelta(days=1))
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca[1].public_key()), critical=False))
    if server:
        b = b.add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                            critical=False)
    else:
        b = b.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
    cert = b.sign(ca[1], hashes.SHA256())
    return (cert.public_bytes(serialization.Encoding.PEM),
            key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption()))


def _pem(cert: Any) -> bytes:
    return bytes(cert.public_bytes(serialization.Encoding.PEM))


def _der(pem: bytes) -> bytes:
    return bytes(x509.load_pem_x509_certificate(pem).public_bytes(serialization.Encoding.DER))


@pytest.fixture(scope="module")
def pki(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    d = tmp_path_factory.mktemp("lane-tls-pki")
    ca, other, clients = _ca("svc-ca"), _ca("other-ca"), _ca("client-ca")
    server_pem, server_key = _leaf(ca, "127.0.0.1", server=True)
    out: dict[str, Any] = {"ca_pem": _pem(ca[0]), "other_pem": _pem(other[0]), "client_ca_pem": _pem(clients[0]),
                           "server_pem": server_pem, "server_key": server_key,
                           "a": _leaf(clients, "gw-a", server=False), "b": _leaf(clients, "gw-b", server=False),
                           "rogue": _leaf(other, "gw-rogue", server=False)}
    for name in ("ca_pem", "client_ca_pem", "server_pem", "server_key"):
        (d / name).write_bytes(out[name])
        out[name + "_file"] = d / name
    return out


# ---- the service app over TCP+TLS ----------------------------------------------------------------------
def _settings(tiny_pack_dir: Path, pki: dict[str, Any], **kw: Any) -> ServiceSettings:
    base: dict[str, Any] = {
        "packs": [tiny_pack_dir], "device": "cpu", "verify_packs": "fingerprint", "warmup": False,
        "rings": ["K32A8", "K40A12"], "host": "127.0.0.1", "lane": "tcp", "lane_port": 0,
        "allow_unauthenticated": True,
        "tls": TLSConfig(cert_file=pki["server_pem_file"], key_file=pki["server_key_file"]),
    }
    base.update(kw)
    return ServiceSettings(**base)


def _headers() -> dict[str, str]:
    return {PROTOCOL_HEADER: str(PROTOCOL_VERSION), "content-type": "application/json"}


async def _open(app: Any, fp: str) -> dict[str, Any]:
    async with http_client(app) as c:
        r = await c.post(LANE_OPEN_PATH, content=json.dumps({"fingerprint": fp}), headers=_headers())
        assert r.status_code == 200, r.text
        return r.json()  # type: ignore[no-any-return]


def _connect(info: dict[str, Any], ca_pem: bytes, ticket: bytes | None = None,
             client: tuple[bytes, bytes] | None = None) -> Any:
    cert, key = client if client is not None else (None, None)
    return fast.TripChannel.connect_service_tls(
        "127.0.0.1", int(info["port"]), "127.0.0.1", ca_pem, ticket or bytes.fromhex(info["ticket"]), 5.0,
        int(info["max_rows"]), 50, cert, key)


async def test_tls_lane_is_bit_identical_to_the_stream_path(tiny_pack_dir: Path, pki: dict[str, Any], pack_fp: str,
                                                           pack_weights: dict[str, torch.Tensor]) -> None:
    async with running_app(_settings(tiny_pack_dir, pki, lane_max_rows=4)) as app:
        info = await _open(app, pack_fp)
        assert info["transport"] == "tcp" and "socket" not in info and info["alpn"] == "pgw-lane/1"
        assert 0 < info["port"] < 65536 and info["port"] == app.state.pgw.lane.svc.port
        assert app.state.pgw.lane.status()["transport"] == "tcp"
        dims = dict(app.state.pgw.registry.get(pack_fp).linear_dims)
        ch = await asyncio.to_thread(_connect, info, pki["ca_pem"])
        client = asgi_client(app)
        try:
            assert ch.max_rows == info["max_rows"]
            for li, name in enumerate(info["linears"][:3]):
                m, n = dims[name]
                for rows in (1, 3, 4):
                    x = rand_x(rows, n, K32, seed=li * 10 + rows)
                    x[0] = K32.modulus - 1
                    out = bytearray(rows * m * 4)
                    st, _cu = await asyncio.to_thread(ch.trip, 0, li, rows, secrets.token_bytes(16), 0,
                                                      pack_ring(x, K32), out, 5.0)
                    assert st == 0
                    assert bytes(out) == pack_ring(expected(x, pack_weights[name], K32), K32)
                    resp = await client.compute(make_req(pack_fp, K32, [(name, x)]))
                    assert bytes(out) == resp.items[0].y_hat
            # rows above the lane's maximum: a non-closing refusal (the client sends that call elsewhere)
            m, n = dims[info["linears"][0]]
            st, _ = await asyncio.to_thread(ch.trip, 0, 0, 5, secrets.token_bytes(16), 0,
                                            pack_ring(rand_x(5, n, K32), K32), bytearray(5 * m * 4), 5.0)
            assert st == 8 and not ch.closed  # BUSY (fallback), connection kept
        finally:
            await client.aclose()
            ch.close()


async def test_tls_lane_verifies_the_server_and_has_no_plaintext(tiny_pack_dir: Path, pki: dict[str, Any],
                                                                pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, pki)) as app:
        info = await _open(app, pack_fp)
        with pytest.raises(OSError):  # another CA: the handshake fails before the ticket is sent
            await asyncio.to_thread(_connect, info, pki["other_pem"])

        def plaintext() -> bytes:
            from pgw.service.lane import encode_open

            with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=5) as s:
                s.sendall(encode_open(bytes.fromhex(info["ticket"])))
                try:
                    return s.recv(64)
                except OSError:
                    return b""

        got = await asyncio.to_thread(plaintext)
        assert not got.startswith(b"PGWR")  # no lane reply to an unencrypted frame (a TLS alert at most)
        # the ticket was never seen in plaintext by the lane, so it still opens exactly one TLS connection
        ch = await asyncio.to_thread(_connect, info, pki["ca_pem"])
        ch.close()
        with pytest.raises(ConnectionRefusedError):  # single use
            await asyncio.to_thread(_connect, info, pki["ca_pem"])
        with pytest.raises(ConnectionRefusedError):  # forged
            await asyncio.to_thread(_connect, info, pki["ca_pem"], secrets.token_bytes(32))


class _InjectClientCert:
    """ASGI wrapper publishing a verified client certificate in the scope, as the TLS server shims do."""

    def __init__(self, app: Any, pem: bytes):
        self.app, self.pem = app, pem
        self.state = app.state
        self.router = app.router

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope = dict(scope)
            scope["extensions"] = {"tls": {"client_cert_name": "CN=gw-a", "client_cert_chain": [self.pem.decode()]}}
        await self.app(scope, receive, send)


async def test_ticket_is_bound_to_the_client_certificate(tiny_pack_dir: Path, pki: dict[str, Any],
                                                         pack_fp: str) -> None:
    settings = _settings(tiny_pack_dir, pki, tls=TLSConfig(cert_file=pki["server_pem_file"],
                                                           key_file=pki["server_key_file"],
                                                           ca_file=pki["client_ca_pem_file"]))
    async with running_app(settings) as app:
        wrapped = _InjectClientCert(app, pki["a"][0])
        infos = [await _open(wrapped, pack_fp) for _ in range(4)]
        with pytest.raises(ConnectionRefusedError):  # bound ticket, no certificate
            await asyncio.to_thread(_connect, infos[0], pki["ca_pem"])
        with pytest.raises(ConnectionRefusedError):  # bound ticket, another valid certificate of the same CA
            await asyncio.to_thread(_connect, infos[1], pki["ca_pem"], None, pki["b"])
        with pytest.raises(OSError):  # a certificate the service's client CA did not issue: handshake fails
            await asyncio.to_thread(_connect, infos[2], pki["ca_pem"], None, pki["rogue"])
        ch = await asyncio.to_thread(_connect, infos[3], pki["ca_pem"], None, pki["a"])
        ch.close()
        # without a certificate on lane/open the ticket is unbound (token-only gateways)
        plain = await _open(app, pack_fp)
        ch = await asyncio.to_thread(_connect, plain, pki["ca_pem"])
        ch.close()


async def test_subject_without_certificate_gets_no_tcp_ticket(tiny_pack_dir: Path, pki: dict[str, Any],
                                                             pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, pki)) as app:
        mgr = app.state.pgw.lane
        from pgw.core.errors import ServiceUnavailableError

        with pytest.raises(ServiceUnavailableError, match="bind"):
            mgr.open_ticket(None, pack_fp, client_cert_der=None, client_cert_seen=True)
        assert mgr.open_ticket(None, pack_fp, client_cert_der=_der(pki["a"][0]))["transport"] == "tcp"


def test_required_client_certificate_is_enforced_in_the_handshake(pki: dict[str, Any]) -> None:
    svc = fast.LaneService("", max_rows=8, executor=lambda *a: bytes(a[3] * 3 * 4), tls_addr="127.0.0.1:0",
                           tls_cert_pem=pki["server_pem"], tls_key_pem=pki["server_key"],
                           tls_client_ca_pem=pki["client_ca_pem"], require_client_cert=True)
    try:
        svc.set_model(0, [(2, 3, "qkv")], [("K32A8", 4)])
        tickets = [secrets.token_bytes(32) for _ in range(2)]
        for t in tickets:
            svc.add_ticket(t, "t", 0, 60.0)
        info = {"port": svc.port, "max_rows": 8, "ticket": tickets[0].hex()}
        with pytest.raises(OSError):
            _connect(info, pki["ca_pem"])
        ch = _connect(info, pki["ca_pem"], tickets[1], pki["b"])
        out = bytearray(12)
        st, _ = ch.trip(0, 0, 1, secrets.token_bytes(16), 0, bytes(8), out, 5.0)
        assert st == 0 and bytes(out) == bytes(12)
        ch.close()
    finally:
        svc.stop(2.0)


def test_native_uds_client_speaks_the_service_frames(tmp_path: Path) -> None:
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="pgwlt"))
    calls: list[tuple[int, int, bytes]] = []

    def ex(_model: int, linear: int, _ring: int, rows: int, body: bytes, _tenant: str, _left: float) -> bytes | int:
        calls.append((linear, rows, body))
        if rows == 2:
            return 4  # deadline_exceeded
        return bytes(range(rows * 3 * 4))

    svc = fast.LaneService(str(d / "l.sock"), max_rows=4, executor=ex)
    try:
        svc.set_model(0, [(2, 3, "qkv")], [("K32A8", 4)])
        t = secrets.token_bytes(32)
        svc.add_ticket(t, "t", 0, 60.0)
        with pytest.raises(ConnectionRefusedError):
            fast.TripChannel.connect_service_uds(str(d / "l.sock"), secrets.token_bytes(32), 5.0, 4)
        ch = fast.TripChannel.connect_service_uds(str(d / "l.sock"), t, 5.0, 4)
        out = bytearray(12)
        st, _ = ch.trip(0, 0, 1, secrets.token_bytes(16), 0, b"\x01" * 8, out, 5.0)
        assert st == 0 and bytes(out) == bytes(range(12)) and calls[-1] == (0, 1, b"\x01" * 8)
        st, _ = ch.trip(0, 0, 2, secrets.token_bytes(16), 0, b"\x02" * 16, bytearray(24), 5.0)
        assert st == 7 and not ch.closed  # DEADLINE, connection kept
        # a body of the wrong length for the linear: the service refuses (fatal), the client maps it to a
        # closing status and drops the connection
        st, _ = ch.trip(0, 0, 1, secrets.token_bytes(16), 0, b"\x01" * 9, out, 5.0)
        assert st == 10 and ch.closed
    finally:
        svc.stop(2.0)
        import shutil

        shutil.rmtree(d, ignore_errors=True)


def test_tcp_lane_settings(tiny_pack_dir: Path, pki: dict[str, Any]) -> None:
    from pydantic import SecretStr

    from pgw.service.lane import LaneManager

    s = _settings(tiny_pack_dir, pki, tls=TLSConfig(cert_file=pki["server_pem_file"],
                                                    key_file=pki["server_key_file"], key_password=SecretStr("x")))
    with pytest.raises(ConfigError, match="unencrypted"):
        LaneManager.from_settings(s, None)
    assert create_app(_settings(tiny_pack_dir, pki)).state.pgw.lane.transport == "tcp"
