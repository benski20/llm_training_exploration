"""Service trip lane (chat-speed WP-A1) on CPU: the native listener with the kernel-path executor.

Covers the ticket (``POST /pgw/v1/lane/open``), the peer check, header and length enforcement, the
shared quota, deadlines, draining, revocation, refusal -> fallback statuses, the metrics drain, and
bit-exactness: a lane reply is byte-identical to the stream path's reply for the same body and to
the exact integer product. The GPU graph path is in ``test_lane_graph.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RingParams, pack_ring, unpack_ring
from pgw.service import ServiceSettings, create_app, hash_token
from pgw.service.lane import (
    LANE_OPEN_PATH,
    REPLY_HEADER,
    LaneStatus,
    encode_call,
    encode_open,
    lane_native_available,
    parse_reply,
)
from pgw.wire.messages import PROTOCOL_HEADER, PROTOCOL_VERSION

from .conftest import TOKEN_A, TOKEN_B, asgi_client, expected, http_client, make_req, rand_x, running_app

pytestmark = pytest.mark.skipif(not lane_native_available(), reason="pgw_fast without the trip lane")

K32 = RingParams.from_name("K32A8")
K40 = RingParams.from_name("K40A12")


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="pgwl"))  # short: AF_UNIX paths are limited to ~108 bytes
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _settings(tiny_pack_dir: Path, sock_dir: Path, tokens: Path | None = None, **kw: Any) -> ServiceSettings:
    base: dict[str, Any] = {
        "packs": [tiny_pack_dir], "device": "cpu", "verify_packs": "fingerprint", "warmup": False,
        "rings": ["K32A8", "K40A12"], "lane": "uds", "lane_socket": sock_dir / "lane.sock",
    }
    if tokens is None:
        base["allow_unauthenticated"] = True
    else:
        base["tokens_file"] = tokens
        base["tokens_reload_interval_s"] = 0.05
    base.update(kw)
    return ServiceSettings(**base)


def _headers(token: str | None) -> dict[str, str]:
    h = {PROTOCOL_HEADER: str(PROTOCOL_VERSION), "content-type": "application/json"}
    if token:
        h["authorization"] = f"Bearer {token}"
    return h


async def _open(app: Any, fp: str, token: str | None = None) -> tuple[int, dict[str, Any]]:
    async with http_client(app) as c:
        r = await c.post(LANE_OPEN_PATH, content=json.dumps({"fingerprint": fp}), headers=_headers(token))
        return r.status_code, r.json()


class Lane:
    """Minimal synchronous lane client (the real one is WP-A3's ``pgw.wire.lane``)."""

    def __init__(self, path: str, ticket: bytes):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(10)
        self.s.connect(path)
        self.s.sendall(encode_open(ticket))
        self.open_status = self.reply()[0].status

    def _recv(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("lane closed")
            buf += chunk
        return bytes(buf)

    def reply(self) -> tuple[Any, bytes]:
        h = parse_reply(self._recv(REPLY_HEADER.size))
        return h, self._recv(h.body_len)

    def call(self, linear: int, rows: int, ring: int, body: bytes, deadline_ms: int = 0,
             rid: bytes | None = None) -> tuple[Any, bytes]:
        rid = rid or secrets.token_bytes(16)
        self.s.sendall(encode_call(rid, linear, rows, ring, body, deadline_ms))
        h, out = self.reply()
        assert h.request_id == rid
        return h, out

    def closed(self) -> bool:
        try:
            return self.s.recv(1) == b""
        except OSError:
            return True

    def close(self) -> None:
        self.s.close()


async def _lane(app: Any, fp: str, token: str | None = None) -> tuple[Lane, dict[str, Any]]:
    status, info = await _open(app, fp, token)
    assert status == 200, info
    lane = await asyncio.to_thread(Lane, info["socket"], bytes.fromhex(info["ticket"]))
    assert lane.open_status == LaneStatus.OK
    return lane, info


def _dims(app: Any, fp: str) -> dict[str, tuple[int, int]]:
    return dict(app.state.pgw.registry.get(fp).linear_dims)


async def test_lane_is_bit_identical_to_stream_path(tiny_pack_dir: Path, sock_dir: Path, pack_fp: str,
                                                   pack_weights: dict[str, torch.Tensor]) -> None:
    async with running_app(_settings(tiny_pack_dir, sock_dir)) as app:
        lane, info = await _lane(app, pack_fp)
        assert info["version"] == 1 and info["transport"] == "uds" and info["max_rows"] == 8
        assert info["rings"] == ["K32A8", "K40A12"]
        dims = _dims(app, pack_fp)
        assert info["linears"] == list(dims)
        client = asgi_client(app)
        try:
            for li, name in enumerate(info["linears"][:4]):
                m, n = dims[name]
                for ri, ring in enumerate((K32, K40)):
                    for rows in (1, 2, 5, 8):
                        x = rand_x(rows, n, ring, seed=li * 100 + ri * 10 + rows)
                        if rows == 5:
                            x[0] = ring.modulus - 1  # extreme operands
                            x[1] = 0
                        body = pack_ring(x, ring)
                        h, out = await asyncio.to_thread(lane.call, li, rows, ri, body)
                        assert h.status == LaneStatus.OK and h.body_len == rows * m * ring.bytes_per_element
                        want = pack_ring(expected(x, pack_weights[name], ring), ring)
                        assert out == want  # exact integer product, bit for bit
                        resp = await client.compute(make_req(pack_fp, ring, [(name, x)]))
                        assert out == resp.items[0].y_hat  # == the stream path's bytes
                        if rows == 2:  # one-time-padded input: unmasking gives the plain product
                            w8 = pack_weights[name]
                            r = rand_x(rows, n, ring, seed=7)
                            xh = (x + r) % ring.modulus
                            _, yh = await asyncio.to_thread(lane.call, li, rows, ri, pack_ring(xh, ring))
                            y = (unpack_ring(yh, (rows, m), ring) - expected(r, w8, ring)) % ring.modulus
                            assert torch.equal(y, expected(x, w8, ring))
            assert app.state.pgw.handler.lane_requests == 4 * 2 * 4 + 4 * 2  # plus the masked calls
        finally:
            await client.aclose()
            lane.close()


async def test_ticket_auth_and_open_errors(tiny_pack_dir: Path, sock_dir: Path, tokens_file: Path,
                                           pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, sock_dir, tokens_file)) as app:
        st, body = await _open(app, pack_fp)  # no token
        assert st == 401, body
        st, body = await _open(app, "ab" * 32, TOKEN_A)
        assert st == 404 and body["code"] == "model_not_found"
        st, info = await _open(app, pack_fp, TOKEN_A)
        assert st == 200
        ticket = bytes.fromhex(info["ticket"])
        lane = await asyncio.to_thread(Lane, info["socket"], ticket)
        assert lane.open_status == LaneStatus.OK
        again = await asyncio.to_thread(Lane, info["socket"], ticket)  # single use
        assert again.open_status == LaneStatus.UNAUTHENTICATED and again.closed()
        forged = await asyncio.to_thread(Lane, info["socket"], secrets.token_bytes(32))
        assert forged.open_status == LaneStatus.UNAUTHENTICATED
        assert os.stat(info["socket"]).st_mode & 0o777 == 0o600
        for c in (lane, again, forged):
            c.close()


async def test_wrong_peer_is_refused(tiny_pack_dir: Path, sock_dir: Path, pack_fp: str,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_SERVICE_LANE_PEER_UIDS", str(os.geteuid() + 4242))
    async with running_app(_settings(tiny_pack_dir, sock_dir)) as app:
        st, info = await _open(app, pack_fp)
        assert st == 200
        lane = await asyncio.to_thread(Lane, info["socket"], bytes.fromhex(info["ticket"]))
        assert lane.open_status == LaneStatus.PEER_REFUSED and lane.closed()
        lane.close()


async def test_header_and_length_enforcement(tiny_pack_dir: Path, sock_dir: Path, pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, sock_dir, lane_max_rows=2)) as app:
        dims = _dims(app, pack_fp)
        name0 = next(iter(dims))
        _m, n = dims[name0]
        lane, _ = await _lane(app, pack_fp)
        x = rand_x(3, n, K32)
        # rows above lane_max_rows: fallback (resend on the stream channel), connection kept
        h, _ = await asyncio.to_thread(lane.call, 0, 3, 0, pack_ring(x, K32))
        assert h.status == LaneStatus.FALLBACK and h.status.fallback and not h.status.fatal
        h, _out = await asyncio.to_thread(lane.call, 0, 2, 0, pack_ring(x[:2], K32))
        assert h.status == LaneStatus.OK
        # a body one byte short: fatal, closed
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, pack_ring(x[:1], K32)[:-1])
        assert h.status == LaneStatus.BAD_LENGTH and lane.closed()
        lane.close()
        # linear and ring outside the tables
        lane, info = await _lane(app, pack_fp)
        h, _ = await asyncio.to_thread(lane.call, len(info["linears"]), 1, 0, b"\0" * (n * 4))
        assert h.status == LaneStatus.UNKNOWN_LINEAR and lane.closed()
        lane.close()
        lane, _ = await _lane(app, pack_fp)
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 2, b"\0" * (n * 4))
        assert h.status == LaneStatus.UNKNOWN_LINEAR
        lane.close()


async def test_shared_quota_and_metrics(tiny_pack_dir: Path, sock_dir: Path, tokens_file: Path,
                                        pack_fp: str) -> None:
    s = _settings(tiny_pack_dir, sock_dir, tokens_file, tenant_rows_per_s=0.5, tenant_burst_rows=4)
    async with running_app(s) as app:
        dims = _dims(app, pack_fp)
        name0 = next(iter(dims))
        _m, n = dims[name0]
        lane, _ = await _lane(app, pack_fp, TOKEN_A)
        x = rand_x(3, n, K32)
        h, _ = await asyncio.to_thread(lane.call, 0, 3, 0, pack_ring(x, K32))
        assert h.status == LaneStatus.OK
        h, _ = await asyncio.to_thread(lane.call, 0, 3, 0, pack_ring(x, K32))
        assert h.status == LaneStatus.RATE_LIMITED and h.retry_after_ms > 0 and not h.status.fallback
        # the stream channel draws on the same bucket: tenant A is limited there too, B is not
        client_a = asgi_client(app, TOKEN_A, max_retries=0)
        client_b = asgi_client(app, TOKEN_B)
        try:
            from pgw.core.errors import RateLimitedError

            with pytest.raises(RateLimitedError):
                await client_a.compute(make_req(pack_fp, K32, [(name0, x[:2])]))
            await client_b.compute(make_req(pack_fp, K32, [(name0, x[:2])]))
        finally:
            await client_a.aclose()
            await client_b.aclose()
        lane.close()
        d = app.state.pgw.lane.drain_metrics()
        req = {(t, st): c for t, st, _b, _s, c in d["requests"]}
        assert req[("tenant-a", "ok")] == 1 and req[("tenant-a", "rate_limited")] == 1
        assert ("tenant-a", name0.split(".", 1)[1], 3) in d["rows"]
        page = app.state.pgw.metrics.render()[0].decode()
        assert 'pgw_service_requests_total{code="rate_limited",tenant="tenant-a"}' in page or \
            'pgw_service_requests_total{tenant="tenant-a",code="rate_limited"}' in page


async def test_deadline_and_executor_errors(tiny_pack_dir: Path, sock_dir: Path, pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, sock_dir)) as app:
        handler = app.state.pgw.handler
        dims = _dims(app, pack_fp)
        _m, n = dims[next(iter(dims))]
        body = pack_ring(rand_x(1, n, K32), K32)
        orig = handler.lane_compute

        async def slow(*a: Any) -> bytes:
            await asyncio.sleep(0.3)
            return await orig(*a)

        lane, _ = await _lane(app, pack_fp)
        n0 = handler.lane_requests
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body)  # idle device: inline on the loop
        assert h.status == LaneStatus.OK and handler.lane_requests == n0 + 1
        inline = handler.lane_compute_inline
        handler.lane_compute_inline = lambda *a: None  # a busy device: the queued kernel path
        handler.lane_compute = slow
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body, 50)
        assert h.status == LaneStatus.DEADLINE_EXCEEDED
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body)  # default deadline (service cap)
        assert h.status == LaneStatus.OK

        async def broken(*a: Any) -> bytes:
            raise RuntimeError("boom")

        handler.lane_compute = broken
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body)
        assert h.status == LaneStatus.INTERNAL and h.status.fallback
        handler.lane_compute = orig
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body)  # connection still usable
        assert h.status == LaneStatus.OK

        def broken_inline(*a: Any) -> bytes:
            raise RuntimeError("boom")

        handler.lane_compute_inline = broken_inline
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body)
        assert h.status == LaneStatus.INTERNAL
        handler.lane_compute_inline = inline
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body)
        assert h.status == LaneStatus.OK
        lane.close()


async def test_drain_refuses_with_fallback_status(tiny_pack_dir: Path, sock_dir: Path, pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, sock_dir)) as app:
        dims = _dims(app, pack_fp)
        _m, n = dims[next(iter(dims))]
        lane, _ = await _lane(app, pack_fp)
        sock = Path(app.state.pgw.lane.status()["socket"])
        app.state.pgw.begin_drain()
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, pack_ring(rand_x(1, n, K32), K32))
        assert h.status == LaneStatus.UNAVAILABLE and h.status.fallback and h.retry_after_ms == 1000
        st, body = await _open(app, pack_fp)
        assert st == 503 and body["code"] == "unavailable"
        lane.close()
    assert not sock.exists()  # shutdown removed the socket


async def test_revoked_token_closes_lane(tiny_pack_dir: Path, sock_dir: Path, tokens_file: Path,
                                         pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, sock_dir, tokens_file)) as app:
        dims = _dims(app, pack_fp)
        _m, n = dims[next(iter(dims))]
        body = pack_ring(rand_x(1, n, K32), K32)
        lane, _ = await _lane(app, pack_fp, TOKEN_A)
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body)
        assert h.status == LaneStatus.OK
        tokens_file.write_text(f"tenant-b:{hash_token(TOKEN_B)}\n")  # tenant A's token revoked
        for _ in range(100):
            await asyncio.sleep(0.05)
            app.state.pgw.handler.authenticate(f"Bearer {TOKEN_B}", None)  # nudges the reload
            h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, body)
            if h.status != LaneStatus.OK:
                break
        assert h.status == LaneStatus.UNAUTHENTICATED and lane.closed()
        lane.close()


class _Ident:
    """A token identity whose validity the test controls; records the threads that check it."""

    def __init__(self, tenant: str, token_id: str, threads: set[int]) -> None:
        self.tenant_id, self.token_id, self.valid, self._threads = tenant, token_id, True, threads

    def still_valid(self) -> bool:
        import threading

        self._threads.add(threading.get_ident())
        return self.valid


async def test_revocation_tracks_every_token_and_checks_off_the_loop(tiny_pack_dir: Path, sock_dir: Path,
                                                                     pack_fp: str) -> None:
    import threading

    async with running_app(_settings(tiny_pack_dir, sock_dir)) as app:
        mgr = app.state.pgw.lane
        threads: set[int] = set()
        many = [_Ident("tenant-x", f"tok{i}", threads) for i in range(200)]
        for ident in many:
            mgr.open_ticket(ident, pack_fp)
            mgr.open_ticket(ident, pack_fp)  # repeated opens do not grow the set
        mgr.open_ticket(_Ident("tenant-y", "toky", threads), pack_fp)
        assert len(mgr._identities["tenant-x"]) == 200 and mgr.svc.pending_tickets() == 401
        many[0].valid = False  # the oldest token: outside any recent-identities window
        await mgr._check_revocations()
        assert mgr.svc.pending_tickets() == 1  # tenant-x's tickets dropped, tenant-y's kept
        assert "tok0" not in mgr._identities["tenant-x"] and len(mgr._identities["tenant-x"]) == 199
        assert threading.get_ident() not in threads  # still_valid never ran on the event loop


async def test_without_native_lane_service_runs_on_stream_channel(
        tiny_pack_dir: Path, sock_dir: Path, pack_fp: str, monkeypatch: pytest.MonkeyPatch,
        pack_weights: dict[str, torch.Tensor]) -> None:
    import pgw.service.lane as lane_mod

    monkeypatch.setattr(lane_mod, "fast", None)
    async with running_app(_settings(tiny_pack_dir, sock_dir)) as app:
        assert app.state.pgw.lane.status()["running"] is False
        assert app.state.pgw.handler.lane_admission is None
        st, body = await _open(app, pack_fp)
        assert st == 503 and body["code"] == "unavailable"
        name0 = next(iter(_dims(app, pack_fp)))
        x = rand_x(2, _dims(app, pack_fp)[name0][1], K32)
        client = asgi_client(app)
        try:
            resp = await client.compute(make_req(pack_fp, K32, [(name0, x)]))
        finally:
            await client.aclose()
        assert resp.items[0].y_hat == pack_ring(expected(x, pack_weights[name0], K32), K32)
        assert not (sock_dir / "lane.sock").exists()


async def test_lane_off_and_tcp(tiny_pack_dir: Path, sock_dir: Path, pack_fp: str, tmp_path: Path) -> None:
    async with running_app(_settings(tiny_pack_dir, sock_dir, lane="off")) as app:
        assert app.state.pgw.lane is None
        st, _body = await _open(app, pack_fp)
        assert st == 503
    cert = tmp_path / "c.pem"
    cert.write_text("x")
    # the TCP+TLS lane is built (tests/service/test_lane_tls.py); the manager is created, the listener starts
    # with the app
    app = create_app(_settings(tiny_pack_dir, sock_dir, lane="tcp", lane_port=9555,
                               tls={"cert_file": cert, "key_file": cert}))
    assert app.state.pgw.lane.transport == "tcp"


def test_wire_mirror_matches_rust_layout() -> None:
    """The Python constants mirror proto.rs (the Rust side has the same vectors in its tests)."""
    rid = bytes(range(16))
    frame = encode_call(rid, 3, 2, 1, b"\x01" * 8, 250)
    assert frame[:4] == b"PGWL" and len(frame) == 48 + 8
    assert frame[4:6] == b"\x01\x00" and frame[6:8] == b"\x02\x00" and frame[8:24] == rid
    assert frame[24:28] == (3).to_bytes(4, "little") and frame[28:30] == b"\x02\x00"
    assert frame[30:32] == b"\x01\x00" and frame[32:36] == (250).to_bytes(4, "little")
    assert frame[36:40] == (8).to_bytes(4, "little") and frame[40:48] == bytes(8)
    assert len(encode_open(bytes(32))) == 48 + 32
    reply = REPLY_HEADER.pack(b"PGWR", 1, 3, rid, 1, 2, 3, 0)
    r = parse_reply(reply)
    assert r.status == LaneStatus.RATE_LIMITED and (r.queue_us, r.compute_us, r.retry_after_ms) == (1, 2, 3)
    assert [s.fatal for s in LaneStatus] == [s.value >= 16 for s in LaneStatus]
    assert {s for s in LaneStatus if s.fallback} == {LaneStatus.FALLBACK, LaneStatus.UNAVAILABLE, LaneStatus.INTERNAL,
                                                    LaneStatus.BUSY}
    assert LaneStatus.BUSY.fatal  # the connection-cap refusal closes the connection


def test_native_device_lock_is_a_lock_drop_in() -> None:
    from pgw.core.native import fast
    from pgw.service.gpu_exec import DeviceExecutor

    lock = fast.LaneDeviceLock()
    ex = DeviceExecutor("cpu")
    ex.share_lock(lock)
    assert ex.lock is lock
    with ex.lock:
        assert lock.locked()
        assert not lock.acquire(timeout=0.01)
        assert not lock.acquire(blocking=False)
    assert not lock.locked()
    with pytest.raises(RuntimeError):
        lock.release()
