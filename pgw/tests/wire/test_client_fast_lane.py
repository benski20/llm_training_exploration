"""The client's lean stream lane (:meth:`pgw.wire.ComputeClient.compute_fast`).

* Differential: the same calls with the lane on and off give equal responses and statistics, over a
  real hypercorn or uvicorn socket (replies large enough to span many socket reads included).
* Fallback: a failed lean attempt hands over to the regular retry loop, which resends the
  byte-identical body under the same request id (same backoff, retry budget, deadline).
* The lane's receive parser against hand-made frame streams cut at every position, and its
  request frames against ``websockets``' own parser.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

pytest.importorskip("websockets")

from pgw.core import RING_K40_A12, ErrorCode, PGWError, ServiceUnavailableError
from pgw.wire import ClientConfig, ComputeClient
from pgw.wire.codec import encode_message
from tests.wire.conftest import FakeHandler, make_request
from tests.wire.test_stream import _check, _Server

RING = RING_K40_A12


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


_SPECS = [("0.qkv", 1), ("0.o", 1), ("0.gate_up", 1), ("0.down", 1), ("0.gate_up", 4000), ("0.qkv", 7),
          ("0.down", 2500), ("0.o", 1)]


async def test_lane_matches_the_regular_path(serve: Any) -> None:
    srv = serve(FakeHandler())
    reqs = [make_request(RING, [spec], rid=f"r{i}", seed=i) for i, spec in enumerate(_SPECS)]
    results = {}
    for lane in (True, False):
        client = _client(srv.url, fast_lane=lane)
        await client.connect(None, RING.name)
        out = []
        for req, xs in reqs:
            resp = await client.compute(req)
            _check(resp, req, xs, srv.handler)
            out.append(resp)
        st = client.stats
        results[lane] = (out, st)
        if lane:  # the first call opened the channel on the regular path; the rest took the lane
            assert st.lane_requests == len(reqs) - 1 and st.stream_opens == 1
        else:
            assert st.lane_requests == 0
        await client.aclose()
    (on, s_on), (off, s_off) = results[True], results[False]
    assert on == off  # FakeHandler reports fixed timings: whole responses are equal
    for f in ("requests", "attempts", "retries", "failures", "bytes_sent", "bytes_received", "stream_requests"):
        assert getattr(s_on, f) == getattr(s_off, f), f


async def test_lane_concurrent_calls_are_matched_by_id(serve: Any) -> None:
    srv = serve(FakeHandler(delay_s=0.005))
    client = _client(srv.url)
    await client.connect(None, RING.name)
    first, xs0 = make_request(RING, [("0.qkv", 1)], rid="first")
    _check(await client.compute(first), first, xs0, srv.handler)  # opens the channel
    reqs = [make_request(RING, [_SPECS[i % len(_SPECS)]], rid=f"c{i}", seed=i) for i in range(24)]
    resps = await asyncio.gather(*(client.compute(r) for r, _ in reqs))
    for (req, xs), resp in zip(reqs, resps, strict=True):
        assert resp.request_id == req.request_id
        _check(resp, req, xs, srv.handler)
    assert client.stats.lane_requests == 24
    await client.aclose()


class _RefuseThird(FakeHandler):
    """Refuses the third admission check (handshake = 1, first call = 2, second call = 3) as draining."""

    def __init__(self) -> None:
        super().__init__()
        self.prechecks = 0

    def precheck(self) -> None:
        self.prechecks += 1
        if self.prechecks == 3:
            raise ServiceUnavailableError("service is draining", details={"retry_after_s": 0.0})


async def test_lane_failure_resends_the_identical_body(serve: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    srv = serve(_RefuseThird())
    client = _client(srv.url, max_retries=3, backoff_max_s=0.01)
    await client.connect(None, RING.name)
    req0, xs0 = make_request(RING, [("0.qkv", 1)], rid="warm")
    _check(await client.compute(req0), req0, xs0, srv.handler)
    resumed: list[tuple[bytes, str, Any]] = []
    orig = client._call

    async def spy(*a: Any, **kw: Any) -> Any:
        resumed.append((kw["body"], kw["request_id"], kw.get("resume")))
        return await orig(*a, **kw)

    monkeypatch.setattr(client, "_call", spy)
    req, xs = make_request(RING, [("0.down", 2)], rid="second")
    _check(await client.compute(req), req, xs, srv.handler)
    assert len(resumed) == 1
    body, rid, exc = resumed[0]
    assert body == encode_message(req) and rid == "second"
    assert isinstance(exc, PGWError) and exc.code == ErrorCode.UNAVAILABLE
    st = client.stats
    assert st.lane_requests == 1 and st.retries == 1 and st.stream_opens == 2
    assert st.attempts == 4  # models (connect), warm-up, the lean attempt, the resend
    seen = [c for c in srv.handler.calls if c.request_id == "second"]
    assert len(seen) == 1 and seen[0].items[0].x_hat == req.items[0].x_hat
    await client.aclose()


async def test_lane_failure_without_retries_raises_the_first_error(serve: Any) -> None:
    srv = serve(_RefuseThird())
    client = _client(srv.url, max_retries=0)
    await client.connect(None, RING.name)
    req0, _ = make_request(RING, [("0.qkv", 1)], rid="warm")
    await client.compute(req0)
    req, _ = make_request(RING, [("0.qkv", 1)], rid="x")
    with pytest.raises(PGWError) as ei:
        await client.compute(req)
    assert ei.value.code == ErrorCode.UNAVAILABLE and ei.value.details.get("attempts") == 1
    assert client.stats.failures == 1 and client.stats.retries == 0
    await client.aclose()


async def test_lane_retry_budget_applies(serve: Any) -> None:
    srv = serve(_RefuseThird())
    client = _client(srv.url, max_retries=3, retry_budget_burst=0.0, retry_budget_ratio=0.0)
    await client.connect(None, RING.name)
    await client.compute(make_request(RING, [("0.qkv", 1)], rid="warm")[0])
    with pytest.raises(PGWError) as ei:
        await client.compute(make_request(RING, [("0.qkv", 1)], rid="x")[0])
    assert ei.value.details.get("retry_budget_exhausted") is True
    assert client.stats.retries_denied == 1
    await client.aclose()


async def test_lane_deadline(serve: Any) -> None:
    srv = serve(FakeHandler())
    client = _client(srv.url, max_retries=0)
    await client.connect(None, RING.name)
    await client.compute(make_request(RING, [("0.qkv", 1)], rid="warm")[0])
    srv.handler.delay_s = 1.0
    req, _ = make_request(RING, [("0.qkv", 1)], rid="slow", deadline_ms=100)
    with pytest.raises(PGWError) as ei:
        await client.compute(req)
    assert ei.value.code == ErrorCode.DEADLINE_EXCEEDED
    ch = next(iter(client._channels.values()))
    assert not ch.pending  # the waiter was removed
    await client.aclose()


async def test_lane_connection_lost_mid_call_is_retried(serve: Any) -> None:
    srv = serve(FakeHandler())
    client = _client(srv.url, max_retries=3, backoff_max_s=0.01)
    await client.connect(None, RING.name)
    await client.compute(make_request(RING, [("0.qkv", 1)], rid="warm")[0])
    srv.handler.delay_s = 0.3
    ch = next(iter(client._channels.values()))
    req, xs = make_request(RING, [("0.o", 1)], rid="lost")
    task = asyncio.ensure_future(client.compute(req))
    await asyncio.sleep(0.05)
    ch.ws.transport.abort()  # the connection drops while the call waits for its reply
    srv.handler.delay_s = 0.0
    _check(await task, req, xs, srv.handler)
    assert client.stats.lane_requests == 1 and client.stats.retries == 1 and client.stats.stream_opens == 2
    await client.aclose()


def test_lane_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_WIRE_FAST_LANE", "0")
    assert ClientConfig(base_url="http://x").fast_lane is False
    monkeypatch.delenv("PGW_WIRE_FAST_LANE")
    assert ClientConfig(base_url="http://x").fast_lane is True


def test_sync_client_loop_uses_proc_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.core import proc_tuning

    made: list[Any] = []
    real = proc_tuning.new_event_loop

    def spy() -> Any:
        loop = real()
        made.append(loop)
        return loop

    monkeypatch.setattr(proc_tuning, "new_event_loop", spy)
    client = _client("http://127.0.0.1:9")
    loop = client._ensure_loop()
    assert made == [loop]
    client.close()


# ---- receive parser ------------------------------------------------------------------------------
class _Transport:
    def __init__(self) -> None:
        self.out: list[bytes] = []
        self.closing = False

    def write(self, data: bytes) -> None:
        self.out.append(bytes(data))

    def writelines(self, parts: Any) -> None:
        self.out.append(b"".join(bytes(p) for p in parts))

    def is_closing(self) -> bool:
        return self.closing

    def set_write_buffer_limits(self, *a: Any) -> None:
        pass

    def get_write_buffer_size(self) -> int:
        return 0

    def pause_reading(self) -> None:
        pass

    def resume_reading(self) -> None:
        pass

    def can_write_eof(self) -> bool:
        return True

    def write_eof(self) -> None:
        pass

    def close(self) -> None:
        self.closing = True

    def abort(self) -> None:
        self.closing = True

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default


def _open_connection(max_size: int = 1 << 24) -> tuple[Any, list[Any], _Transport]:
    from websockets.client import ClientProtocol
    from websockets.protocol import State
    from websockets.uri import parse_uri

    from pgw.wire.client import _direct_connection_class

    cls = _direct_connection_class()
    assert cls is not None
    from websockets.utils import accept_key

    proto = ClientProtocol(parse_uri("ws://test/pgw/v1/stream"), max_size=max_size)
    req = proto.connect()
    proto.send_request(req)
    proto.receive_data(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                       b"Sec-WebSocket-Accept: " + accept_key(req.headers["Sec-WebSocket-Key"]).encode() + b"\r\n\r\n")
    (response,) = proto.events_received()
    proto.data_to_send()
    assert proto.state is State.OPEN  # as after the opening handshake
    conn = cls(proto, ping_interval=None)
    conn.response = response
    tr = _Transport()
    conn.connection_made(tr)
    got: list[Any] = []
    conn.pgw_max_size = max_size
    conn.pgw_rx = True
    conn.pgw_sink = lambda m: got.append(m if isinstance(m, str) else bytes(m))
    return conn, got, tr


def _server_frame(opcode: Any, data: bytes, fin: bool = True) -> bytes:
    from websockets.frames import Frame

    return Frame(opcode, data, fin=fin).serialize(mask=False)


@pytest.mark.parametrize("keep", [32 << 20, 1000], ids=["reused-buffer", "growing-buffer"])
@pytest.mark.parametrize("seed", range(6))
async def test_receive_parser_any_chunking(seed: int, keep: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from websockets.frames import Opcode

    import pgw.wire.client as wc

    monkeypatch.setattr(wc, "_RX_BUFFER_KEEP", keep)

    rng = random.Random(seed)
    payloads = [bytes(rng.getrandbits(8) for _ in range(n)) for n in (0, 1, 125, 126, 300, 65535, 65536, 70001, 5)]
    stream = b""
    for i, p in enumerate(payloads):
        stream += _server_frame(Opcode.BINARY, p)
        if i % 3 == 1:
            stream += _server_frame(Opcode.PING, b"pp%d" % i)
    conn, got, tr = _open_connection()
    pos = 0
    while pos < len(stream):  # random cuts, including inside headers and right after them
        step = rng.choice([1, 2, 3, 7, 10, 64, 1000, 40000, 100000])
        conn.data_received(stream[pos:pos + step])
        pos += step
    assert got == payloads
    assert conn._pgw_on and not conn._pgw_off
    pongs = [b for b in tr.out if b and b[0] & 0x0F == 0xA]  # websockets answered every ping
    assert len(pongs) == 3


async def test_receive_parser_hands_unusual_frames_to_websockets() -> None:
    from websockets.frames import Opcode

    # a text frame: the stream goes back to websockets for good, the channel sees it and closes
    conn, got, _ = _open_connection()
    conn.data_received(_server_frame(Opcode.BINARY, b"one") + _server_frame(Opcode.TEXT, b"hi")
                       + _server_frame(Opcode.BINARY, b"two"))
    assert got == [b"one", "", b"two"] and conn._pgw_off
    # fragmented binary messages are reassembled by the regular path
    conn, got, _ = _open_connection()
    conn.data_received(_server_frame(Opcode.BINARY, b"ab", fin=False) + _server_frame(Opcode.CONT, b"cd"))
    assert got == [b"abcd"] and conn._pgw_off
    # too large: websockets fails the connection, nothing is delivered
    conn, got, _ = _open_connection(max_size=100)
    conn.data_received(_server_frame(Opcode.BINARY, b"x" * 101))
    assert got == [] and conn._pgw_off and conn.protocol.state.name in ("CLOSING", "CLOSED")


async def test_receive_parser_rejects_masked_server_frames() -> None:
    from websockets.frames import Frame, Opcode

    conn, got, _ = _open_connection()
    conn.data_received(Frame(Opcode.BINARY, b"masked").serialize(mask=True))
    assert got == [] and conn._pgw_off and conn.protocol.state.name in ("CLOSING", "CLOSED")


@pytest.mark.parametrize("n", [0, 1, 125, 126, 65535, 65536, 200000])
def test_request_frame_parses_as_websockets_does(n: int) -> None:
    from websockets.frames import Frame
    from websockets.streams import StreamReader

    from pgw.wire.client import _masked_binary_head, _ws_apply_mask

    body = bytes(range(256)) * (n // 256 + 1)
    body = body[:n]
    mask = b"\x01\x02\x03\x04"
    frame = _masked_binary_head(n) + mask + _ws_apply_mask()(body, mask)
    reader = StreamReader()
    reader.feed_data(frame)
    parser = Frame.parse(reader.read_exact, mask=True, max_size=None)
    try:
        next(parser)
    except StopIteration as stop:
        parsed = stop.value
    assert parsed.data == body and parsed.fin and parsed.opcode.name == "BINARY"
    assert not reader.buffer
