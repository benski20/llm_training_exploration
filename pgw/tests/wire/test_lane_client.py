"""Trip lane client (chat-speed WP-A3) against a real stream service and the reference lane server.

* lane vs WebSocket: byte-identical ``y_hat`` (and equal to the exact ring product) for every linear and
  rows 1..8, with the native channel and the Python channel;
* fallback byte identity: a refused, dropped or ineligible lane call reaches the stream service as the
  same ``x_hat`` bytes under the same request id;
* TLS on the TCP lane, and TCP without TLS is refused (the ticket is never sent in the clear).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import ipaddress
import socket
import ssl
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("hypercorn")
pytest.importorskip("websockets")

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from fastapi import Request, Response

from pgw.core import RING_K40_A12, RingParams, pack_ring, ringmm_reference, unpack_ring
from pgw.wire import ClientConfig, ComputeClient, LinearInfo, decode_message, encode_message
from pgw.wire import lane as L
from pgw.wire.codec import CONTENT_TYPE
from pgw.wire.messages import LANE_OPEN_PATH, ComputeItem, ComputeRequest, LaneOpenRequest
from tests.wire.conftest import DIMS, FP, FakeHandler, make_app, random_x_hat

RING = RING_K40_A12
LINEARS = tuple(LinearInfo(k, m, n) for k, (m, n) in DIMS.items())
NATIVE = L.native_channel_class() is not None
CHANNELS = [pytest.param(True, id="native", marks=pytest.mark.skipif(not NATIVE, reason="pgw_fast lane missing")),
            pytest.param(False, id="python")]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _stub_compute(handler: FakeHandler):
    def compute(li: LinearInfo, rows: int, ring: str, x_hat: bytes) -> bytes:
        rp = RingParams.from_name(ring)
        x = unpack_ring(x_hat, (rows, li.in_features), rp)
        import torch

        return pack_ring(torch.from_numpy(ringmm_reference(x, handler.weights[li.id], rp.k)), rp)

    return compute


class _Service:
    """hypercorn: the regular router (stream + HTTP) with ``FakeHandler``, plus ``lane_open`` answered by a
    :class:`pgw.wire.lane.LaneStubServer` computing with the same weights."""

    def __init__(self, tmp: Path, *, lane_open_ok: bool = True, transport: str = "uds",
                 ssl_context: ssl.SSLContext | None = None, max_rows: int = 8, ttl_ms: int = 60_000):
        self.handler = FakeHandler()
        self.lane = L.LaneStubServer(
            _stub_compute(self.handler), fingerprint=FP, rings=("K32A8", "K40A12"), linears=LINEARS,
            transport=transport, path=str(tmp / "lane.sock") if transport == "uds" else None,
            ssl_context=ssl_context, max_rows=max_rows, ttl_ms=ttl_ms)
        self.lane_opens = 0
        app = make_app(self.handler)

        @app.post(LANE_OPEN_PATH)
        async def lane_open(request: Request) -> Response:
            self.lane_opens += 1
            if not lane_open_ok:
                return Response(status_code=503)
            req = decode_message(await request.body(), LaneOpenRequest)
            return Response(encode_message(self.lane.issue_ticket(req)), media_type=CONTENT_TYPE)

        self.port = _free_port()
        self.loop = asyncio.new_event_loop()
        self.stop = asyncio.Event()

        async def main() -> None:
            from hypercorn.asyncio import serve
            from hypercorn.config import Config

            cfg = Config()
            cfg.bind = [f"127.0.0.1:{self.port}"]
            cfg.accesslog = None
            cfg.errorlog = None
            await serve(app, cfg, shutdown_trigger=self.stop.wait)  # type: ignore[arg-type]

        self.thread = threading.Thread(target=lambda: self.loop.run_until_complete(main()), daemon=True)
        self.thread.start()
        for _ in range(200):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.lane.close()
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(10)


@pytest.fixture
def service(tmp_path: Path) -> Iterator[_Service]:
    s = _Service(tmp_path)
    yield s
    s.close()


def _request(lid: str, rows: int, seed: int, ring: RingParams = RING) -> ComputeRequest:
    x = random_x_hat(rows, DIMS[lid][1], ring, seed)
    return ComputeRequest(uuid.uuid4().hex, FP, ring.name, (ComputeItem(lid, rows, pack_ring(x, ring)),),
                          deadline_ms=30_000)


def _expected(service: _Service, req: ComputeRequest) -> bytes:
    it = req.items[0]
    return _stub_compute(service.handler)(LinearInfo(it.linear, *DIMS[it.linear]), it.rows, req.ring, it.x_hat)


def _clients(service: _Service, native: bool, **kw: Any) -> tuple[ComputeClient, L.LaneComputeClient]:
    plain = ComputeClient(ClientConfig(base_url=service.url))
    lane = L.LaneComputeClient(ComputeClient(ClientConfig(base_url=service.url)),
                               L.LaneConfig(native=native, retry_s=60.0, **kw))
    plain.connect_sync(FP, RING.name)
    lane.connect_sync(FP, RING.name)
    return plain, lane


@pytest.mark.parametrize("native", CHANNELS)
def test_lane_equals_stream_bytewise(service: _Service, native: bool) -> None:
    plain, lc = _clients(service, native)
    try:
        n = 0
        for ring in (RING, RingParams.from_name("K32A8")):
            for lid in DIMS:
                for rows in range(1, 9):
                    req = _request(lid, rows, seed=rows * 10 + n, ring=ring)
                    a = plain.compute_sync(req)
                    b = lc.compute_sync(req)
                    n += 1
                    assert a.items[0].y_hat == b.items[0].y_hat == _expected(service, req)
                    assert b.request_id == req.request_id and b.items[0].rows == rows
        assert lc.lane_stats.trips == n and lc.lane_stats.fallbacks == 0
        assert len(service.lane.trips) == n
        assert len(service.handler.calls) == n  # only the plain client's calls reached the stream service
        assert lc.lane_stats.native is native
        # the lane carried exactly the request bytes
        for (hdr, body), req in zip(service.lane.trips, service.handler.calls, strict=True):
            assert body == req.items[0].x_hat and hdr["request_id"].hex() == req.request_id
    finally:
        plain.close()
        lc.close()


@pytest.mark.parametrize("native", CHANNELS)
def test_compute_into_caller_buffer(service: _Service, native: bool) -> None:
    _, lc = _clients(service, native)
    try:
        req = _request("0.gate_up", 3, seed=5)
        m = DIMS["0.gate_up"][0]
        out = np.zeros(3 * m * RING.k // 8, dtype=np.uint8)
        resp = lc.compute_into(req, out)
        assert out.tobytes() == _expected(service, req) == bytes(resp.items[0].y_hat)
        with pytest.raises(ValueError):
            lc.compute_into(req, np.zeros(7, dtype=np.uint8))
    finally:
        lc.close()


@pytest.mark.parametrize("native", CHANNELS)
@pytest.mark.parametrize("how", ["busy", "deadline", "draining", "drop"])
def test_fallback_resends_identical_bytes(service: _Service, native: bool, how: str) -> None:
    _, lc = _clients(service, native)
    try:
        warm = _request("0.o", 1, seed=1)
        lc.compute_sync(warm)  # opens the lane
        assert len(service.lane.trips) == 1
        if how == "drop":
            service.lane.drop_next = 1
        else:
            service.lane.refuse_next = [{"busy": L.Status.BUSY, "deadline": L.Status.DEADLINE,
                                         "draining": L.Status.DRAINING}[how]]
        req = _request("0.qkv", 2, seed=9)
        resp = lc.compute_sync(req)
        assert resp.items[0].y_hat == _expected(service, req)
        hdr, lane_body = service.lane.trips[-1]
        stream_req = service.handler.calls[-1]
        # same bytes, same id: the service sees nothing it had not seen on the lane
        assert stream_req.request_id == req.request_id == hdr["request_id"].hex()
        assert stream_req.items[0].x_hat == lane_body == req.items[0].x_hat
        assert encode_message(stream_req) == encode_message(req)
        assert lc.lane_stats.fallbacks == 1
        dead = how in ("draining", "drop")
        # a refusal that keeps the connection: the next call takes the lane again; else the lane cools down
        before = len(service.lane.trips)
        lc.compute_sync(_request("0.o", 1, seed=2))
        assert len(service.lane.trips) == before + (0 if dead else 1)
        assert (lc.lane_ticket is None) is dead
    finally:
        lc.close()


@pytest.mark.parametrize("native", CHANNELS)
def test_ineligible_calls_use_the_stream(service: _Service, native: bool) -> None:
    _, lc = _clients(service, native)
    try:
        lc.compute_sync(_request("0.o", 1, seed=1))
        n = len(service.lane.trips)
        too_many = _request("0.o", 9, seed=2)
        two = ComputeRequest(uuid.uuid4().hex, FP, RING.name, (_request("0.o", 1, 3).items[0],
                                                              _request("0.qkv", 1, 4).items[0]))
        odd_id = ComputeRequest("req-not-hex", FP, RING.name, _request("0.o", 1, 5).items)
        for req in (too_many, two, odd_id):
            resp = lc.compute_sync(req)
            assert service.handler.calls[-1] == req
            assert len(resp.items) == len(req.items)
        assert len(service.lane.trips) == n
        assert set(lc.lane_stats.fallback_reasons) == {"ineligible:rows", "ineligible:items", "ineligible:request_id"}
    finally:
        lc.close()


def test_no_lane_offered_cools_down(tmp_path: Path) -> None:
    svc = _Service(tmp_path, lane_open_ok=False)
    try:
        _, lc = _clients(svc, False)
        for i in range(3):
            req = _request("0.o", 1, seed=i)
            assert lc.compute_sync(req).items[0].y_hat == _expected(svc, req)
        assert svc.lane_opens == 1 and lc.lane_stats.open_failures == 1
        assert len(svc.handler.calls) == 3 and not svc.lane.trips
        lc.close()
    finally:
        svc.close()


def test_ticket_expiry(tmp_path: Path) -> None:
    """Connections outlive their ticket; a ticket that expired before any connection was opened is
    replaced by a new ``lane_open`` on the next call (that call itself takes the regular path)."""
    svc = _Service(tmp_path, ttl_ms=300)
    try:
        _, lc = _clients(svc, False)
        lc.compute_sync(_request("0.o", 1, seed=1))
        time.sleep(0.4)
        lc.compute_sync(_request("0.o", 1, seed=2))  # the open connection keeps serving
        assert lc.lane_stats.trips == 2 and svc.lane_opens == 1
        lc.lane_close()
        assert lc.open_sync(FP, RING.name) and svc.lane_opens == 2
        time.sleep(0.4)  # expired before any connection
        req = _request("0.o", 1, seed=3)
        assert lc.compute_sync(req).items[0].y_hat == _expected(svc, req)
        assert svc.handler.calls[-1] == req and lc.lane_stats.fallback_reasons == {"expired": 1}
        lc.compute_sync(_request("0.o", 1, seed=4))
        assert svc.lane_opens == 3 and lc.lane_stats.trips == 3
        lc.close()
    finally:
        svc.close()


@pytest.mark.parametrize("native", CHANNELS)
def test_bad_ticket_falls_back(service: _Service, native: bool) -> None:
    _, lc = _clients(service, native)
    try:
        assert lc.open_sync(FP, RING.name)
        lc._lane.secret[:] = b"\x00" * 32  # type: ignore[union-attr]  # not a ticket the server issued
        req = _request("0.o", 1, seed=3)
        assert lc.compute_sync(req).items[0].y_hat == _expected(service, req)
        assert service.handler.calls[-1] == req and not service.lane.trips
        assert lc.lane_ticket is None and service.lane.hellos == 0
        reasons = lc.lane_stats.fallback_reasons
        assert sum(v for k, v in reasons.items() if k.startswith("error:")) == 1
    finally:
        lc.close()


@pytest.mark.parametrize("native", CHANNELS)
def test_concurrent_callers_share_the_pool(service: _Service, native: bool) -> None:
    _, lc = _clients(service, native, max_channels=2)
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        try:
            for i in range(10):
                req = _request("0.gate_up", 1 + (i % 4), seed=k * 100 + i)
                assert lc.compute_sync(req).items[0].y_hat == _expected(service, req)
        except BaseException as e:
            errors.append(e)

    try:
        ts = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        assert not errors
        assert lc.lane_stats.trips == 40 and lc.lane_stats.channels_opened <= 2
    finally:
        lc.close()


def test_async_compute(service: _Service) -> None:
    async def main() -> None:
        lc = L.LaneComputeClient(ComputeClient(ClientConfig(base_url=service.url)), L.LaneConfig())
        try:
            await lc.connect(FP, RING.name)
            for i in range(3):
                req = _request("0.down", 2, seed=i)
                assert (await lc.compute(req)).items[0].y_hat == _expected(service, req)
            assert lc.lane_stats.trips == 3
        finally:
            await lc.aclose()

    asyncio.run(main())


# ------------------------------------------------------------------------------- TLS on TCP
def _pki(d: Path) -> dict[str, Path]:
    def name(cn: str) -> x509.Name:
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    now = dt.datetime.now(dt.UTC)
    paths: dict[str, Path] = {}
    for tag in ("ca", "other_ca"):
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca = (x509.CertificateBuilder().subject_name(name(f"lane-{tag}")).issuer_name(name(f"lane-{tag}"))
              .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=1))
              .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
              .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                           content_commitment=False, key_encipherment=False, data_encipherment=False,
                                           key_agreement=False, encipher_only=False, decipher_only=False),
                             critical=True)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
              .sign(ca_key, hashes.SHA256()))
        paths[tag] = d / f"{tag}.pem"
        paths[tag].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
        if tag == "ca":
            key = ec.generate_private_key(ec.SECP256R1())
            cert = (x509.CertificateBuilder().subject_name(name("lane-server")).issuer_name(ca.subject)
                    .public_key(key.public_key()).serial_number(x509.random_serial_number())
                    .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=1))
                    .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                    .add_extension(x509.SubjectAlternativeName(
                        [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                        critical=False)
                    .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                                   critical=False)
                    .sign(ca_key, hashes.SHA256()))
            paths["server"] = d / "server.pem"
            paths["server_key"] = d / "server_key.pem"
            paths["server"].write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            paths["server_key"].write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                                              serialization.PrivateFormat.PKCS8,
                                                              serialization.NoEncryption()))
    return paths


@pytest.fixture(scope="module")
def pki(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return _pki(tmp_path_factory.mktemp("lane-pki"))


def _server_ctx(pki: dict[str, Path]) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(str(pki["server"]), str(pki["server_key"]))
    ctx.set_alpn_protocols([L.LANE_ALPN])
    return ctx


def _tcp_stub(**kw: Any) -> tuple[L.LaneStubServer, FakeHandler]:
    h = FakeHandler()
    return L.LaneStubServer(_stub_compute(h), fingerprint=FP, rings=(RING.name,), linears=LINEARS,
                            transport="tcp", **kw), h


@pytest.mark.parametrize("native", CHANNELS)
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_tls_tcp_lane(pki: dict[str, Path], native: bool, host: str) -> None:
    stub, h = _tcp_stub(ssl_context=_server_ctx(pki))
    try:
        cfg = ClientConfig(base_url=f"https://{host}:1", ca_file=pki["ca"])
        ticket = stub.issue_ticket(LaneOpenRequest("r", FP, RING.name))
        ch = L.open_channel(ticket, bytearray(ticket.ticket), cfg, L.LaneConfig(native=native))
        assert ch.max_rows == 8 and (type(ch).__name__ == "TripChannel") is native
        for rows in (1, 5):
            req = _request("0.gate_up", rows, seed=rows)
            m, _n = DIMS["0.gate_up"]
            out = bytearray(rows * m * RING.k // 8)
            st, _ = ch.trip(0, 2, rows, bytes.fromhex(req.request_id), 0, req.items[0].x_hat, out, 5.0)
            assert st == L.Status.OK
            assert bytes(out) == _stub_compute(h)(LINEARS[2], rows, RING.name, req.items[0].x_hat)
        ch.close()
        assert ch.closed and stub.hellos == 1
    finally:
        stub.close()


@pytest.mark.parametrize("native", CHANNELS)
def test_tls_wrong_ca_is_refused(pki: dict[str, Path], native: bool) -> None:
    stub, _ = _tcp_stub(ssl_context=_server_ctx(pki))
    try:
        cfg = ClientConfig(base_url="https://127.0.0.1:1", ca_file=pki["other_ca"])
        ticket = stub.issue_ticket(LaneOpenRequest("r", FP, RING.name))
        with pytest.raises((OSError, ValueError)):
            L.open_channel(ticket, bytearray(ticket.ticket), cfg, L.LaneConfig(native=native))
        assert stub.hellos == 0
    finally:
        stub.close()


@pytest.mark.parametrize("native", CHANNELS)
def test_tcp_without_tls_is_refused(pki: dict[str, Path], native: bool) -> None:
    # the client refuses a TCP lane unless the service URL is https:// with verification on
    stub, _ = _tcp_stub(insecure_plaintext_tcp=True)
    try:
        ticket = stub.issue_ticket(LaneOpenRequest("r", FP, RING.name))
        for cfg in (ClientConfig(base_url="http://127.0.0.1:1"),
                    ClientConfig(base_url="https://127.0.0.1:1", ca_file=pki["ca"], verify_tls=False)):
            with pytest.raises(ValueError, match="TLS"):
                L.open_channel(ticket, bytearray(ticket.ticket), cfg, L.LaneConfig(native=native))
        # a plaintext listener behind an https:// URL: the TLS handshake fails, the ticket is never sent
        cfg = ClientConfig(base_url="https://127.0.0.1:1", ca_file=pki["ca"])
        with pytest.raises((OSError, ValueError)):
            L.open_channel(ticket, bytearray(ticket.ticket), cfg, L.LaneConfig(native=native, connect_timeout_s=2))
        assert stub.hellos == 0
    finally:
        stub.close()
    # and the reference server itself refuses to listen on plaintext TCP
    with pytest.raises(ValueError, match="TLS"):
        L.LaneStubServer(lambda *a: b"", fingerprint=FP, rings=(RING.name,), linears=LINEARS, transport="tcp")
