"""The blocking stream lane (:mod:`pgw.wire.sync_stream`, gateway ``lane = "stream"``).

* Over a real hypercorn / uvicorn socket (and wss with mTLS): the same responses as the regular path, and
  the service receives the same ``ComputeRequest`` objects (the provider sees nothing new); no client
  event loop is ever started (no thread hand-off per trip).
* Fallback: an error reply, a broken connection, an unreachable service: the same request object on the
  regular path (byte-identical body, same request id).
* The frame parser: pings answered with pongs, fragmented replies reassembled, a close frame fails the call.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

import pytest

pytest.importorskip("websockets")

from pgw.core import RING_K40_A12, ServiceUnavailableError
from pgw.wire import ClientConfig, ComputeClient
from pgw.wire.codec import encode_message
from pgw.wire.sync_stream import BlockingStreamChannel, SyncStreamClient
from tests.wire.conftest import FP, FakeHandler, make_request
from tests.wire.test_stream import _check, _free_port, _Server
from tests.wire.test_tls_http2 import pki, server  # noqa: F401 - pytest fixtures

RING = RING_K40_A12
_SPECS = [("0.qkv", 1), ("0.o", 1), ("0.gate_up", 1), ("0.down", 1), ("0.gate_up", 3000), ("0.qkv", 7),
          ("0.down", 2500), ("0.o", 1)]


@pytest.fixture(params=["hypercorn", "uvicorn"])
def serve(request: pytest.FixtureRequest) -> Any:
    servers: list[_Server] = []

    def start(handler: Any) -> _Server:
        s = _Server(handler, None, request.param)
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.close()


def _client(url: str, **kw: Any) -> ComputeClient:
    return ComputeClient(ClientConfig(base_url=url, request_timeout_s=20, backoff_base_s=0.001, **kw))


def _wire(req: Any) -> bytes:
    return encode_message(req)


def test_sync_lane_matches_the_regular_path(serve: Any) -> None:
    srv = serve(FakeHandler())
    reqs = [make_request(RING, [spec], rid=f"r{i}", seed=i) for i, spec in enumerate(_SPECS)]
    reqs.append(make_request(RING, [("0.qkv", 2), ("0.down", 1)], rid="multi", seed=99))
    seen: dict[bool, list[bytes]] = {}
    outs: dict[bool, list[Any]] = {}
    for lane in (True, False):
        srv.handler.calls.clear()
        inner = _client(srv.url)
        inner.connect_sync(None, RING.name)
        client: Any = SyncStreamClient(inner) if lane else inner
        outs[lane] = []
        for req, xs in reqs:
            resp = client.compute_sync(req)
            _check(resp, req, xs, srv.handler)
            outs[lane].append(resp)
        seen[lane] = [_wire(c) for c in srv.handler.calls]
        if lane:
            st = client.sync_stream_stats
            assert st.calls == len(reqs) and st.fallbacks == 0 and st.opens == 1
            assert inner.stats.stream_requests == len(reqs) and inner.stats.requests == len(reqs)
        client.close()
    assert outs[True] == outs[False]
    assert seen[True] == seen[False] == [_wire(r) for r, _ in reqs]  # the service received the same requests


def test_sync_lane_starts_no_event_loop(serve: Any) -> None:
    srv = serve(FakeHandler())
    inner = _client(srv.url)
    client = SyncStreamClient(inner)
    for i in range(5):
        req, xs = make_request(RING, [("0.gate_up", 1)], rid=f"n{i}", seed=i)
        _check(client.compute_sync(req), req, xs, srv.handler)
    assert inner._loop is None  # every trip ran on this thread: no private loop, no hand-off
    assert client.sync_stream_stats.calls == 5
    client.close()


def test_sync_lane_threads_get_their_own_connections(serve: Any) -> None:
    srv = serve(FakeHandler(delay_s=0.002))
    client = SyncStreamClient(_client(srv.url), max_channels=3)
    errors: list[BaseException] = []

    def work(k: int) -> None:
        try:
            for i in range(6):
                req, xs = make_request(RING, [_SPECS[(i + k) % 4]], rid=f"t{k}-{i}", seed=10 * k + i)
                _check(client.compute_sync(req), req, xs, srv.handler)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    th = [threading.Thread(target=work, args=(k,)) for k in range(3)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert not errors
    st = client.sync_stream_stats
    assert st.calls + st.fallbacks == 18 and st.opens <= 3
    client.close()


class _RefuseSecond(FakeHandler):
    def __init__(self) -> None:
        super().__init__()
        self.prechecks = 0

    def precheck(self) -> None:
        self.prechecks += 1
        if self.prechecks == 3:  # handshake = 1, first call = 2, second call = 3
            raise ServiceUnavailableError("service is draining", details={"retry_after_s": 0.0})


def test_error_reply_resends_the_same_request_on_the_regular_path(serve: Any) -> None:
    srv = serve(_RefuseSecond())
    inner = _client(srv.url, max_retries=3, backoff_max_s=0.01)
    client = SyncStreamClient(inner)
    req0, xs0 = make_request(RING, [("0.qkv", 1)], rid="warm")
    _check(client.compute_sync(req0), req0, xs0, srv.handler)
    req, xs = make_request(RING, [("0.down", 1)], rid="again", seed=5)
    _check(client.compute_sync(req), req, xs, srv.handler)
    st = client.sync_stream_stats
    assert st.fallbacks == 1 and st.last_fallback == "ErrorResponse"
    # the lane attempt was refused before compute; the regular path sent the identical request once
    assert [_wire(c) for c in srv.handler.calls] == [_wire(req0), _wire(req)]
    client.close()


def test_broken_connection_falls_back_then_reconnects(serve: Any) -> None:
    srv = serve(FakeHandler())
    client = SyncStreamClient(_client(srv.url))
    req, xs = make_request(RING, [("0.qkv", 1)], rid="a")
    _check(client.compute_sync(req), req, xs, srv.handler)
    ch = client._idle[0]
    ch._sock.close()  # the connection dies under the lane
    req2, xs2 = make_request(RING, [("0.o", 1)], rid="b", seed=2)
    _check(client.compute_sync(req2), req2, xs2, srv.handler)
    assert client.sync_stream_stats.fallbacks == 1
    req3, xs3 = make_request(RING, [("0.down", 1)], rid="c", seed=3)
    _check(client.compute_sync(req3), req3, xs3, srv.handler)
    st = client.sync_stream_stats
    assert st.opens == 2 and st.calls == 2
    client.close()


def test_idle_connection_is_replaced(serve: Any) -> None:
    srv = serve(FakeHandler())
    client = SyncStreamClient(_client(srv.url), max_idle_s=0.0)
    for i in range(3):
        req, xs = make_request(RING, [("0.qkv", 1)], rid=f"i{i}", seed=i)
        _check(client.compute_sync(req), req, xs, srv.handler)
    st = client.sync_stream_stats
    assert st.calls == 3 and st.opens == 3 and st.idle_reopens == 2 and st.fallbacks == 0
    client.close()


def test_unreachable_service_cools_down_and_uses_the_regular_path() -> None:
    port = _free_port()
    inner = _client(f"http://127.0.0.1:{port}", max_retries=0, connect_timeout_s=0.5)
    client = SyncStreamClient(inner, cool_down_s=60)
    req, _ = make_request(RING, [("0.qkv", 1)], rid="x")
    with pytest.raises(ServiceUnavailableError):
        client.compute_sync(req)
    with pytest.raises(ServiceUnavailableError):
        client.compute_sync(req)
    st = client.sync_stream_stats
    assert st.open_failures == 1 and st.fallbacks == 2 and st.calls == 0  # the second call did not retry the lane
    client.close()


def test_custom_transport_never_takes_the_lane() -> None:
    import httpx

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    inner = ComputeClient(ClientConfig(base_url="http://127.0.0.1:9", max_retries=0, request_timeout_s=2),
                          transport=httpx.MockTransport(refuse))
    client = SyncStreamClient(inner)
    req, _ = make_request(RING, [("0.qkv", 1)], rid="y")
    with pytest.raises(Exception):
        client.compute_sync(req)
    assert client.sync_stream_stats.opens == 0 and client.sync_stream_stats.open_failures == 0
    client.close()


def test_wss_with_mtls(pki: Any, server: Any) -> None:  # noqa: F811 - fixtures
    url, handler = server
    cfg = ClientConfig(base_url=url, ca_file=pki["ca"], client_cert_file=pki["client"],
                       client_key_file=pki["client_key"], request_timeout_s=20)
    inner = ComputeClient(cfg)
    inner.connect_sync(FP, RING.name)
    client = SyncStreamClient(inner)
    for i in range(3):
        req, xs = make_request(RING, [("0.gate_up", 2)], rid=f"tls{i}", seed=i)
        _check(client.compute_sync(req), req, xs, handler)
    st = client.sync_stream_stats
    assert st.calls == 3 and st.fallbacks == 0 and st.opens == 1
    client.close()


# ---- frame parser ----------------------------------------------------------------------------------
def _bare_channel() -> tuple[BlockingStreamChannel, socket.socket]:
    from pgw.wire.client import _masked_binary_head, _ws_apply_mask

    a, b = socket.socketpair()
    ch = BlockingStreamChannel.__new__(BlockingStreamChannel)
    ch._sock = a
    ch._rx = bytearray()
    ch._head = _masked_binary_head
    ch._mask = _ws_apply_mask()
    ch.max_response_bytes = 1 << 20
    ch.closed = False
    ch.last_used = 0.0
    return ch, b


def _server_frame(op: int, payload: bytes, fin: bool = True) -> bytes:
    n = len(payload)
    b0 = (0x80 if fin else 0) | op
    if n < 126:
        return bytes((b0, n)) + payload
    return bytes((b0, 126)) + n.to_bytes(2, "big") + payload


def _read_client_frame(sock: socket.socket) -> tuple[int, bytes]:
    from websockets.frames import Frame
    from websockets.streams import StreamReader

    reader = StreamReader()
    reader.feed_data(sock.recv(1 << 16))
    gen = Frame.parse(reader.read_exact, mask=True)
    try:
        next(gen)
    except StopIteration as stop:
        f = stop.value
        return int(f.opcode), bytes(f.data)
    raise AssertionError("incomplete client frame")


def test_parser_answers_pings_and_reassembles_fragments() -> None:
    ch, srv = _bare_channel()
    payload = bytes(range(256)) * 3
    srv.sendall(_server_frame(0x9, b"hi") + _server_frame(0x2, payload[:100], fin=False)
                + _server_frame(0xA, b"") + _server_frame(0x0, payload[100:]))
    assert ch._read_message() == payload
    op, data = _read_client_frame(srv)
    assert op == 0xA and data == b"hi"  # pong with the ping's payload
    srv.close()
    ch.close()


def test_parser_refuses_close_masked_and_text_frames() -> None:
    from pgw.wire.sync_stream import _ChannelClosedError

    for frame in (_server_frame(0x8, b""), bytes((0x82, 0x80 | 4)) + b"abcd" + b"wxyz", _server_frame(0x1, b"x")):
        ch, srv = _bare_channel()
        srv.sendall(frame)
        with pytest.raises(_ChannelClosedError):
            ch._read_message()
        srv.close()
        ch.close()


def test_request_frame_is_what_websockets_would_parse() -> None:
    ch, srv = _bare_channel()
    body = bytes(range(200)) * 50
    ch._send_frame(0x82, body)
    srv.settimeout(2)
    from websockets.frames import Frame
    from websockets.streams import StreamReader

    data = b""
    while len(data) < len(body) + 8:  # 2 + 2 length bytes + 4 mask bytes
        data += srv.recv(1 << 16)
    reader = StreamReader()
    reader.feed_data(data)
    gen = Frame.parse(reader.read_exact, mask=True)
    try:
        next(gen)
        raise AssertionError("incomplete frame")
    except StopIteration as stop:
        f = stop.value
    assert int(f.opcode) == 0x2 and f.fin and bytes(f.data) == body
    srv.close()
    ch.close()
