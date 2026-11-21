"""Bulk maskgen channel (pgw.hemask.bulk, pgw.service.maskgen_bulk): same bytes and rules as POST /pgw/v1/maskgen."""

from __future__ import annotations

import asyncio
import os
import socket
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives import serialization
from pydantic import SecretStr

from pgw.core.errors import ErrorCode, PGWError, ServiceUnavailableError
from pgw.hemask import DEGREE, HEParams, matmul_mod64
from pgw.hemask.bulk import (
    BULK_VERSION,
    REPLY_HEADER,
    REPLY_MAGIC,
    REQUEST_HEADER,
    REQUEST_MAGIC,
    BulkMaskgenClient,
    recv_exact,
)
from pgw.hemask.client import RemoteMulmod64, switch_down, switch_pack, unpack_shift
from pgw.hemask.wire import (
    MaskgenResponse,
    MaskgenResult,
    encode_maskgen_response,
    encode_maskgen_response_parts,
    pack_bits,
    unpack_bits,
)
from pgw.service import ServiceSettings, hash_token
from pgw.wire import ClientConfig, ComputeClient
from pgw.wire.client import build_ssl_context
from tests.service.conftest import TOKEN_A, TOKEN_B
from tests.service.test_server import _ca, _free_port, _issue, _key_pem, _ServerThread, _wait_ready


# ---- encoding and native kernels ---------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 7, 300, 70_000])
def test_response_parts_are_byte_identical(n: int) -> None:
    y1 = os.urandom(n)
    y2 = memoryview(os.urandom(n + 8)).toreadonly()
    for versions in ((), (1, 2)):
        r = MaskgenResponse("rid", (MaskgenResult("0.qkv", 256, y1), MaskgenResult("1.down", 512, y2)), 2, versions)
        assert b"".join(bytes(p) for p in encode_maskgen_response_parts(r)) == encode_maskgen_response(r)


@pytest.mark.parametrize(("q", "bb"), [(56, 53), (56, 56), (64, 61), (64, 64), (64, 57), (56, 49)])
def test_switch_pack_matches_numpy(q: int, bb: int) -> None:
    rng = np.random.default_rng(q * 100 + bb)
    for count in (8, 4096, 1 << 18):
        hi = (1 << q) - 1 if q < 64 else 2**64 - 1
        b = rng.integers(0, hi, size=(count // 8, 8), dtype=np.uint64, endpoint=True)
        b.reshape(-1)[:8] = hi  # the carry corner: b + d past 2^q (and 2^64)
        ref = pack_bits(switch_down(b, q, bb, np.random.default_rng(7)), bb)
        assert bytes(switch_pack(b, q, bb, np.random.default_rng(7))) == bytes(ref)


def test_switch_pack_fresh_dither_is_a_valid_switch() -> None:
    """Without an rng (production) every value is floor((b + d) / 8) for some d in [0, 8)."""
    b = np.random.default_rng(1).integers(0, 1 << 56, size=1 << 17, dtype=np.uint64)
    got = unpack_bits(bytes(switch_pack(b, 56, 53)), b.size, 53)
    lo = b >> np.uint64(3)
    hi = ((b + np.uint64(7)) >> np.uint64(3)) & np.uint64((1 << 53) - 1)
    assert np.all((got == lo) | (got == hi))
    assert 0.3 < float(np.mean(got != lo)) < 0.6  # E[P(round up)] = 7/16 for uniform b


@pytest.mark.parametrize("bits", [8, 13, 35, 43, 53, 56, 64])
def test_unpack_shift_matches_numpy(bits: int) -> None:
    rng = np.random.default_rng(bits)
    for count in (8, 1 << 18):
        v = rng.integers(0, (1 << bits) - 1 if bits < 64 else 2**64 - 1, size=count, dtype=np.uint64, endpoint=True)
        data = bytes(pack_bits(v, bits))
        for shift in (0, 56 - bits if bits <= 56 else 0, 3):
            assert np.array_equal(unpack_shift(data, count, bits, shift), unpack_bits(data, count, bits) << np.uint64(shift))


# ---- a stub handler behind the listener (plain TCP on loopback) ----------------------------------------
class _Handler:
    """Duck-typed ServiceHandler: token auth, a scripted maskgen that returns ``y = b`` sized replies."""

    maskgen_versions = (1, 2)

    def __init__(self) -> None:
        self.draining = False
        self.calls = 0

    def authenticate(self, authorization: str | None, subject: str | None) -> str:
        if authorization != f"Bearer {TOKEN_A}":
            raise PGWError("bad token", code=ErrorCode.UNAUTHENTICATED)
        return "tenant-a"

    def precheck(self) -> None:
        if self.draining:
            raise ServiceUnavailableError("draining", details={"retry_after_s": 1.0})

    async def maskgen(self, req: Any, identity: Any) -> MaskgenResponse:
        assert identity == "tenant-a"
        self.calls += 1
        it = req.items[0]
        y = memoryview(bytes(it.b[: it.y_nbytes(4)])).toreadonly()  # m = 4 outputs
        return MaskgenResponse(req.request_id, (MaskgenResult(it.linear, it.cols, y),))


@pytest.fixture
def stub_server() -> Iterator[tuple[Any, _Handler, asyncio.AbstractEventLoop]]:
    from pgw.service.maskgen_bulk import MaskgenBulkServer

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    h = _Handler()
    srv = MaskgenBulkServer(h, loop, None, host="127.0.0.1", port=0, max_conns=2, max_request_bytes=1 << 20)
    srv.start()
    yield srv, h, loop
    srv.close()
    loop.call_soon_threadsafe(loop.stop)
    t.join(5)


def _req_body(n: int = 8, cols: int = DEGREE) -> tuple[bytes, Any]:
    from pgw.hemask.wire import MaskgenItem, MaskgenRequest, encode_maskgen_request

    b = os.urandom(n * cols * 8)
    req = MaskgenRequest("rid-1", "f" * 64, (MaskgenItem("0.o", cols, b),), 1, (1, 2))
    return encode_maskgen_request(req), req


def test_bulk_round_trip_and_reply_bytes(stub_server: Any) -> None:
    srv, h, _ = stub_server
    body, req = _req_body()
    cli = BulkMaskgenClient("127.0.0.1", srv.port, ssl_context=None, server_hostname=None,
                            authorization=f"Bearer {TOKEN_A}")
    try:
        for _ in range(3):  # one connection, reused
            data = cli.call(body, req.request_id, bytes)
            want = encode_maskgen_response(MaskgenResponse(req.request_id, (MaskgenResult(
                "0.o", DEGREE, req.items[0].b[: req.items[0].y_nbytes(4)]),), 1, (1, 2)))
            assert data == want
        assert h.calls == 3 and cli._open == 1
    finally:
        cli.close()


def test_bulk_errors_are_the_services(stub_server: Any) -> None:
    srv, h, _ = stub_server
    body, req = _req_body()
    bad = BulkMaskgenClient("127.0.0.1", srv.port, ssl_context=None, server_hostname=None, authorization="Bearer x")
    with pytest.raises(PGWError) as ei:
        bad.call(body, req.request_id, bytes)
    assert ei.value.code == ErrorCode.UNAUTHENTICATED and not ei.value.details.get("bulk_transport")
    bad.close()
    h.draining = True
    cli = BulkMaskgenClient("127.0.0.1", srv.port, ssl_context=None, server_hostname=None,
                            authorization=f"Bearer {TOKEN_A}")
    with pytest.raises(PGWError) as ei:
        cli.call(body, req.request_id, bytes)
    assert ei.value.code == ErrorCode.UNAVAILABLE and ei.value.retryable
    h.draining = False
    with pytest.raises(PGWError) as ei:  # a body the decoder refuses: protocol error, connection stays usable
        cli.call(b"\x93garbage", "x", bytes)
    assert ei.value.code == ErrorCode.PROTOCOL
    assert cli.call(body, req.request_id, bytes)
    cli.close()
    assert h.calls == 1


def test_bulk_refuses_oversized_and_malformed_frames(stub_server: Any) -> None:
    srv, _h, _ = stub_server
    with socket.create_connection(("127.0.0.1", srv.port)) as s:  # body over max_request_bytes: refused unread
        meta = msgpack.packb({"id": "big"})
        s.sendall(REQUEST_HEADER.pack(REQUEST_MAGIC, BULK_VERSION, 0, len(meta), 2 << 20) + meta)
        magic, _v, status, _r, n = REPLY_HEADER.unpack(recv_exact(s, REPLY_HEADER.size))
        assert magic == REPLY_MAGIC and status == 413
        recv_exact(s, n)
        assert s.recv(1) == b""  # closed
    with socket.create_connection(("127.0.0.1", srv.port)) as s:
        s.sendall(b"GET / HTTP/1.1\r\n\r\n\x00\x00")
        _m, _v, status, _r, n = REPLY_HEADER.unpack(recv_exact(s, REPLY_HEADER.size))
        assert status == 400
    with socket.create_connection(("127.0.0.1", srv.port)) as s:  # meta that is not a map
        meta = msgpack.packb([1, 2])
        s.sendall(REQUEST_HEADER.pack(REQUEST_MAGIC, BULK_VERSION, 0, len(meta), 0) + meta)
        _m, _v, status, _r, n = REPLY_HEADER.unpack(recv_exact(s, REPLY_HEADER.size))
        assert status == 400


def test_bulk_connection_cap(stub_server: Any) -> None:
    srv, _h, _ = stub_server
    held = [socket.create_connection(("127.0.0.1", srv.port)) for _ in range(2)]
    try:
        import time

        time.sleep(0.2)
        with socket.create_connection(("127.0.0.1", srv.port)) as s:
            _m, _v, status, _r, _n = REPLY_HEADER.unpack(recv_exact(s, REPLY_HEADER.size))
            assert status == 503
        assert srv.refused == 1
    finally:
        for s in held:
            s.close()


def test_bulk_waiter_gets_a_slot_freed_by_an_error(stub_server: Any) -> None:
    """A caller waiting for a connection proceeds when another call drops its connection after an error."""
    srv, _h, _ = stub_server
    body, req = _req_body()
    cli = BulkMaskgenClient("127.0.0.1", srv.port, ssl_context=None, server_hostname=None,
                            authorization=f"Bearer {TOKEN_A}", max_conns=1, timeout_s=30.0)
    try:
        sock = cli._take()  # the only slot
        out: list[Any] = []
        t = threading.Thread(target=lambda: out.append(cli.call(body, req.request_id, bytes)))
        t.start()
        import time

        time.sleep(0.2)
        cli._drop(sock)  # what call() does after an error reply: the slot is free, the idle queue stays empty
        t.join(10)
        assert not t.is_alive() and out and out[0]
    finally:
        cli.close()


def test_remote_falls_back_to_http_when_bulk_is_down() -> None:
    class _Http:
        def __init__(self) -> None:
            self.bodies: list[bytes] = []

        def post_raw_sync(self, path: str, body: bytes, rid: str, decode: Any, *, timeout_s: float) -> Any:
            self.bodies.append(body)
            return np.zeros((4, DEGREE), dtype=np.uint64)

    http = _Http()
    dead = BulkMaskgenClient("127.0.0.1", _free_port(), ssl_context=None, server_hostname=None)
    remote = RemoteMulmod64(http, "f" * 64, {"0.o": 4}, q_bits=56, bulk=dead)
    y = remote("0.o", np.zeros((8, DEGREE), dtype=np.uint64))
    assert y.shape == (4, DEGREE) and len(http.bodies) == 1 and not dead.usable()
    remote("0.o", np.zeros((8, DEGREE), dtype=np.uint64))  # backing off: straight to HTTP
    assert len(http.bodies) == 2


# ---- the real service: TLS with client certificates, real kernels -------------------------------------------
@pytest.fixture(scope="module")
def bulk_service(tiny_pack_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    d = tmp_path_factory.mktemp("bulk-svc")
    ca_key, ca = _ca("pgw-bulk-ca")
    pki: dict[str, Path] = {"ca": d / "ca.pem"}
    pki["ca"].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    for role, cn, server in (("server", "pgw-server", True), ("a", "gw-a", False)):
        k, c = _issue(ca_key, ca, cn, server=server)
        pki[role], pki[f"{role}_key"] = d / f"{role}.pem", d / f"{role}.key"
        pki[role].write_bytes(c.public_bytes(serialization.Encoding.PEM))
        pki[f"{role}_key"].write_bytes(_key_pem(k))
    tok = d / "tenants.txt"
    tok.write_text(f"tenant-a:{hash_token(TOKEN_A)}\ntenant-b:{hash_token(TOKEN_B)}\n")
    mt = d / "mtls.txt"
    mt.write_text("tenant-a:gw-a\n")
    port, bport = _free_port(), _free_port()
    s = ServiceSettings(host="127.0.0.1", port=port, packs=[tiny_pack_dir], device="cpu", rings=["K32A12C27"],
                        tokens_file=tok, mtls_tenants_file=mt, batch_window_ms=1.0, maskgen_bulk_port=bport,
                        require_client_cert_for_compute=True,
                        tls={"cert_file": pki["server"], "key_file": pki["server_key"], "ca_file": pki["ca"]})
    srv = _ServerThread(s)
    srv.start()
    url = f"https://127.0.0.1:{port}"
    ctx = ssl.create_default_context(cafile=str(pki["ca"]))
    ctx.load_cert_chain(str(pki["a"]), str(pki["a_key"]))
    _wait_ready(url, verify=ctx)
    yield {"url": url, "bulk_port": bport, "pki": pki}
    srv.stop()
    assert srv.error is None, srv.error


def _remote(svc: dict[str, Any], pack_fp: str, outs: dict[str, int], *, cert: bool = True, token: str = TOKEN_A,
            v2: bool = False, split: int = 1) -> tuple[RemoteMulmod64, ComputeClient, BulkMaskgenClient]:
    pki = svc["pki"]
    cfg = ClientConfig(base_url=svc["url"], token=SecretStr(token), ca_file=pki["ca"],
                       client_cert_file=pki["a"] if cert else None, client_key_file=pki["a_key"] if cert else None)
    client = ComputeClient(cfg)
    ctx = build_ssl_context(cfg, register=False)
    bulk = BulkMaskgenClient("127.0.0.1", svc["bulk_port"], ssl_context=ctx, server_hostname="127.0.0.1",
                             authorization=f"Bearer {token}")
    p = HEParams(k=32, rank=16)
    kw: dict[str, Any] = {"params": p, "row_norms": dict.fromkeys(outs, 50.0)} if v2 else {}
    return RemoteMulmod64(client, pack_fp, outs, q_bits=p.q, bulk=bulk, bulk_split=split, **kw), client, bulk


@pytest.mark.parametrize("split", [1, 2])
def test_bulk_equals_http_on_the_real_service(bulk_service: dict[str, Any], pack_fp: str,
                                              pack_weights: dict[str, torch.Tensor], split: int) -> None:
    outs = {k: int(v.shape[0]) for k, v in pack_weights.items()}
    remote, client, bulk = _remote(bulk_service, pack_fp, outs, split=split)
    try:
        rng = np.random.default_rng(3)
        for lid, w8 in list(pack_weights.items())[:3]:
            b = rng.integers(0, 1 << 56, size=(int(w8.shape[1]), 2 * DEGREE), dtype=np.uint64)
            y_bulk = remote(lid, b)
            remote.bulk = None
            y_http = remote(lid, b)
            remote.bulk = bulk
            assert np.array_equal(y_bulk, y_http)
            ref = matmul_mod64(w8.numpy().astype(np.int64), b) & np.uint64((1 << 56) - 1)
            refr = ((ref + np.uint64(1 << 15)) >> np.uint64(16) << np.uint64(16)) & np.uint64((1 << 56) - 1)
            assert np.array_equal(y_bulk, refr)  # v1 replies are deterministic: exact
        assert remote.calls_bulk == 3 * split
    finally:
        bulk.close()
        client.close()


def test_bulk_v2_on_the_real_service(bulk_service: dict[str, Any], pack_fp: str,
                                     pack_weights: dict[str, torch.Tensor]) -> None:
    outs = {k: int(v.shape[0]) for k, v in pack_weights.items()}
    remote, client, bulk = _remote(bulk_service, pack_fp, outs, v2=True)
    try:
        lid, w8 = next(iter(pack_weights.items()))
        b = np.random.default_rng(4).integers(0, 1 << 56, size=(int(w8.shape[1]), DEGREE), dtype=np.uint64)
        remote(lid, b)  # learns the server's versions
        y = remote(lid, b)
        assert remote.calls_v2 >= 1 and remote.calls_bulk == 2
        ref = matmul_mod64(w8.numpy().astype(np.int64), b)
        diff = ((y - ref) & np.uint64((1 << 56) - 1)).astype(np.int64)
        diff = np.where(diff >= 1 << 55, diff - (1 << 56), diff)
        bound = 4 * float(np.linalg.norm(w8.numpy().astype(np.float64), axis=1).max()) * 8 + (1 << 21)
        assert float(np.abs(diff).max()) < bound  # switch + reply rounding only
    finally:
        bulk.close()
        client.close()


def test_bulk_enforces_the_compute_auth_rules(bulk_service: dict[str, Any], pack_fp: str,
                                              pack_weights: dict[str, torch.Tensor]) -> None:
    outs = {k: int(v.shape[0]) for k, v in pack_weights.items()}
    lid, w8 = next(iter(pack_weights.items()))
    b = np.zeros((int(w8.shape[1]), DEGREE), dtype=np.uint64)
    # token alone (no client certificate): require_client_cert_for_compute refuses it, as over HTTP
    remote, client, bulk = _remote(bulk_service, pack_fp, outs, cert=False)
    try:
        with pytest.raises(PGWError) as ei:
            bulk.call(remote._request(lid, b, None)[1], "rid", bytes)
        assert ei.value.code in (ErrorCode.UNAUTHENTICATED, ErrorCode.PERMISSION_DENIED)
    finally:
        bulk.close()
        client.close()
    # a certificate mapped to tenant-a with tenant-b's token: refused (cross-check), as over HTTP
    remote, client, bulk = _remote(bulk_service, pack_fp, outs, token=TOKEN_B)
    try:
        with pytest.raises(PGWError) as ei:
            bulk.call(remote._request(lid, b, None)[1], "rid", bytes)
        assert ei.value.code in (ErrorCode.UNAUTHENTICATED, ErrorCode.PERMISSION_DENIED)
    finally:
        bulk.close()
        client.close()
    # no TLS at all: the handshake fails, which is a transport failure (the caller resends over HTTP)
    plain = BulkMaskgenClient("127.0.0.1", bulk_service["bulk_port"], ssl_context=None, server_hostname=None,
                              authorization=f"Bearer {TOKEN_A}", timeout_s=5)
    with pytest.raises(ServiceUnavailableError) as ei2:
        plain.call(b"x" * 64, "rid", bytes)
    assert ei2.value.details.get("bulk_transport")
    plain.close()


def test_settings_refuse_plain_bulk_off_loopback() -> None:
    with pytest.raises(ValueError, match="loopback"):
        ServiceSettings(host="0.0.0.0", port=8443, maskgen_bulk_port=8444)  # noqa: S104
    with pytest.raises(ValueError, match="differ"):
        ServiceSettings(host="127.0.0.1", port=8443, maskgen_bulk_port=8443)
