"""The stream channel's frame lane (:class:`pgw.wire.FrameLane`): differential byte-identity.

The stream app is driven at the ASGI level with a fake server protocol that offers a lane, so every
frame can be answered either by the lane (synchronously, inside the "frame callback") or by the
regular path (ASGI receive queue, eager task, ``_process``). Both must produce byte-identical reply
frames, the same handler calls, contexts and metrics-hook results, and the same refusals.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from pgw.core import ErrorCode, PGWError, RateLimitedError, RingParams, ServiceUnavailableError
from pgw.wire import ComputeRequest, ComputeResponse, RequestContext, WireLimits, create_router
from pgw.wire.codec import encode_message, encode_message_parts
from pgw.wire.messages import STREAM_PATH
from pgw.wire.server import FRAME_LANE_EXTENSION, FrameLane, ws_binary_header
from tests.wire.conftest import FakeHandler, make_request

RING = RingParams.from_name("K40A12")


class LaneHandler(FakeHandler):
    """FakeHandler that also answers synchronously (the frame lane hooks), with the same results."""

    def __init__(self, **kw: Any):
        super().__init__(**kw)
        self.ready = True
        self.inline_calls = 0
        self.prechecks = 0
        self.refuse: PGWError | None = None

    def precheck(self) -> None:
        self.prechecks += 1
        if self.refuse is not None:
            raise self.refuse

    def inline_ready(self, request: ComputeRequest) -> bool:
        return self.ready and len(request.items) == 1

    def compute_inline(self, request: ComputeRequest, ctx: RequestContext) -> ComputeResponse:
        self.inline_calls += 1
        coro = FakeHandler.compute(self, request, ctx)  # no delay configured: completes without suspending
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value  # type: ignore[no-any-return]
        raise AssertionError("compute suspended")  # pragma: no cover


class _Conn:
    """One stream connection to the router's stream app, with or without a frame lane."""

    def __init__(self, handler: Any, *, lane: bool, limits: WireLimits | None = None, ready: bool = True,
                 headers: Sequence[tuple[bytes, bytes]] = (), **router_kw: Any):
        self.handler = handler
        lim = limits or WireLimits()
        self.router = create_router(handler, limits=lim, **router_kw)
        self.inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.replies: asyncio.Queue[bytes] = asyncio.Queue()
        self.events: list[dict[str, Any]] = []
        self.accepted = asyncio.Event()
        self.lane_ready = ready
        self.lane_frames = 0
        self.lane: FrameLane | None = FrameLane(lambda: self.lane_ready, self._write) if lane else None
        ext = {FRAME_LANE_EXTENSION: self.lane} if self.lane is not None else {}
        self.scope = {"type": "websocket", "path": STREAM_PATH, "headers": list(headers), "extensions": ext}
        self.task: asyncio.Task[None] | None = None

    def _write(self, parts: Sequence[bytes | bytearray | memoryview]) -> bool:
        if not self.lane_ready:
            return False
        self.replies.put_nowait(b"".join(bytes(p) for p in parts))
        return True

    async def _receive(self) -> dict[str, Any]:
        return await self.inbox.get()

    async def _send(self, msg: dict[str, Any]) -> None:
        self.events.append(msg)
        if msg["type"] == "websocket.accept":
            self.accepted.set()
        elif msg["type"] == "websocket.send":
            self.replies.put_nowait(msg["bytes"])
        elif msg["type"] == "websocket.close":
            self.accepted.set()

    async def open(self) -> None:
        self.inbox.put_nowait({"type": "websocket.connect"})
        self.task = asyncio.ensure_future(self.router.pgw_stream_app(self.scope, self._receive, self._send))
        await asyncio.wait_for(self.accepted.wait(), 5)
        await asyncio.sleep(0)  # let the app publish its lane handler

    async def frame(self, body: bytes) -> bytes:
        """What the server protocol does with one complete binary message; returns the reply frame."""
        lane = self.lane
        if lane is not None and lane.handler is not None and lane.ready():
            parts = lane.handler(body)
            if parts is not None:
                self.lane_frames += 1
                assert lane.write(parts)
                return await asyncio.wait_for(self.replies.get(), 5)
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": body})
        return await asyncio.wait_for(self.replies.get(), 5)

    async def close(self) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})
        if self.task is not None:
            await asyncio.wait_for(self.task, 5)


def _frames() -> list[bytes]:
    """Valid single- and multi-item requests and requests every check refuses (in a fixed order)."""
    out: list[bytes] = []
    for i, (lid, rows) in enumerate([("0.qkv", 1), ("0.o", 1), ("0.gate_up", 2), ("0.down", 8), ("0.qkv", 3)]):
        out.append(encode_message(make_request(RING, [(lid, rows)], rid=f"ok{i}", seed=i)[0]))
    out.append(encode_message(make_request(RING, [("0.qkv", 1), ("0.down", 1)], rid="multi")[0]))
    out.append(encode_message(make_request(RING, [("0.qkv", 1)], fp="cd" * 32, rid="badfp")[0]))
    out.append(encode_message(make_request(RING, [("0.qkv", 1)], rid="deadline", deadline_ms=0)[0]))
    out.append(encode_message(make_request(RingParams.from_name("K32A8"), [("0.o", 1)], rid="k32")[0]))
    req, _ = make_request(RING, [("0.qkv", 2)], rid="short")
    it = req.items[0]
    out.append(encode_message(ComputeRequest(req.request_id, req.fingerprint, req.ring,
                                             (type(it)(it.linear, it.rows, it.x_hat[:-1]),))))
    req, _ = make_request(RING, [("0.qkv", 1)], rid="unknown")
    it = req.items[0]
    out.append(encode_message(ComputeRequest(req.request_id, req.fingerprint, req.ring,
                                             (type(it)("9.qkv", it.rows, it.x_hat),))))
    out.append(encode_message(make_request(RING, [("0.o", 9)], rid="toomany")[0]))  # rows above the limit
    return out


async def _run(frames: list[bytes], *, lane: bool, **kw: Any) -> tuple[list[bytes], LaneHandler, _Conn]:
    h = LaneHandler()
    conn = _Conn(h, lane=lane, limits=WireLimits(max_rows_per_item=8, frame_lane=True), **kw)
    await conn.open()
    replies = [await conn.frame(b) for b in frames]
    await conn.close()
    return replies, h, conn


async def test_lane_replies_are_byte_identical_to_the_regular_path() -> None:
    frames = _frames()
    on, h_on, c_on = await _run(frames, lane=True)
    off, h_off, c_off = await _run(frames, lane=False)
    assert on == off  # every reply frame, results and errors alike
    assert c_off.lane_frames == 0 and h_off.inline_calls == 0
    # the 1-item valid requests (5 on K40A12, 1 on K32A8) and the deadline refusal took the lane;
    # multi-item and undecodable or invalid requests did not
    assert h_on.inline_calls == 6 and c_on.lane_frames == 7
    assert h_on.done == h_off.done  # metrics hook: same statuses and codes
    assert [r.request_id for r in h_on.calls] == [r.request_id for r in h_off.calls]
    for a, b in zip(h_on.contexts, h_off.contexts, strict=True):
        assert (a.request_id, a.identity, a.attempt, a.body_bytes) == (b.request_id, b.identity, b.attempt,
                                                                       b.body_bytes)
        assert (a.deadline is None) == (b.deadline is None)
    assert c_on.lane is not None and c_on.lane.handler is None  # cleared when the channel ended


async def test_lane_maps_handler_errors_like_the_regular_path() -> None:
    frames = [encode_message(make_request(RING, [("0.qkv", 1)], rid=f"e{i}")[0]) for i in range(3)]
    results = []
    for lane in (True, False):
        h = LaneHandler()
        conn = _Conn(h, lane=lane)
        await conn.open()
        out = []
        for body, err in zip(frames, [RateLimitedError("slow down", details={"retry_after_s": 0.5}),
                                      RuntimeError("secret internals"), None], strict=True):
            h.fail_with = err
            out.append(await conn.frame(body))
        await conn.close()
        results.append((out, h.done, conn.lane_frames))
    (on, done_on, n_on), (off, done_off, n_off) = results
    assert on == off and done_on == done_off and n_on == 3 and n_off == 0
    assert b"secret" not in on[1]


async def test_lane_refusals_take_the_regular_path() -> None:
    body = encode_message(make_request(RING, [("0.qkv", 1)], rid="r")[0])
    # precheck refuses (draining): the lane steps aside, the regular path replies with the refusal
    replies = []
    for lane in (True, False):
        h = LaneHandler()
        conn = _Conn(h, lane=lane)
        await conn.open()
        h.refuse = ServiceUnavailableError("service is draining", details={"retry_after_s": 1.0})
        replies.append(await conn.frame(body))
        assert conn.lane_frames == 0 and h.inline_calls == 0
        await conn.close()
    assert replies[0] == replies[1]
    # the handler cannot answer inline (busy device), or the connection is not writable now
    for ready, handler_ready in ((True, False), (False, True)):
        h = LaneHandler()
        h.ready = handler_ready
        conn = _Conn(h, lane=True, ready=ready)
        await conn.open()
        assert (await conn.frame(body)) == await _ok_reply(body)
        assert conn.lane_frames == 0 and h.inline_calls == 0 and len(h.calls) == 1
        await conn.close()


async def _ok_reply(body: bytes) -> bytes:
    h = LaneHandler()
    conn = _Conn(h, lane=False)
    await conn.open()
    out = await conn.frame(body)
    await conn.close()
    return out


async def test_lane_charges_the_byte_budget_like_the_regular_path() -> None:
    body = encode_message(make_request(RING, [("0.down", 8)], rid="big")[0])
    replies = []
    for lane in (True, False):
        h = LaneHandler()
        conn = _Conn(h, lane=lane, max_inflight_bytes=len(body) - 1)
        await conn.open()
        replies.append(await conn.frame(body))
        await conn.close()
        assert h.calls == []
    assert replies[0] == replies[1] and b"unavailable" in replies[0]


async def test_lane_uses_the_connection_identity() -> None:
    class Auth(LaneHandler):
        def authenticate(self, authorization: str | None, client_cert_subject: str | None) -> Any:
            from pgw.core import AuthError

            if authorization != "Bearer good-token":
                raise AuthError("invalid or missing bearer token")
            return {"tenant": "t1"}

    h = Auth()
    conn = _Conn(h, lane=True, headers=[(b"authorization", b"Bearer good-token")])
    await conn.open()
    await conn.frame(encode_message(make_request(RING, [("0.qkv", 1)], rid="a")[0]))
    assert conn.lane_frames == 1 and h.contexts[-1].identity == {"tenant": "t1"}
    await conn.close()
    # refused at the handshake: the lane is never armed
    h2 = Auth()
    bad = _Conn(h2, lane=True, headers=[(b"authorization", b"Bearer nope")])
    await bad.open()
    assert bad.events[-1]["type"] == "websocket.close" and bad.events[-1]["code"] == 1008
    assert bad.lane is not None and bad.lane.handler is None


async def test_lane_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_SERVICE_FRAME_LANE", "0")
    assert WireLimits().frame_lane is False
    h = LaneHandler()
    conn = _Conn(h, lane=True, limits=WireLimits())
    await conn.open()
    assert conn.lane is not None and conn.lane.handler is None
    await conn.frame(encode_message(make_request(RING, [("0.qkv", 1)], rid="a")[0]))
    assert h.inline_calls == 0
    await conn.close()
    monkeypatch.setenv("PGW_SERVICE_FRAME_LANE", "1")
    assert WireLimits().frame_lane is True


async def test_unreadable_frame_still_closes_the_channel() -> None:
    h = LaneHandler()
    conn = _Conn(h, lane=True)
    await conn.open()
    reply = await conn.frame(b"\x00garbage")
    assert b"protocol" in reply
    await asyncio.wait_for(conn.task, 5)  # type: ignore[arg-type]
    assert any(e["type"] == "websocket.close" for e in conn.events)
    assert conn.lane is not None and conn.lane.handler is None


@pytest.mark.parametrize("n", [0, 1, 125, 126, 127, 65535, 65536, 70000])
def test_ws_binary_header_matches_websockets(n: int) -> None:
    from websockets.frames import Frame, Opcode

    data = bytes(range(256)) * (n // 256 + 1)
    data = data[:n]
    assert ws_binary_header(n) + data == Frame(Opcode.BINARY, data).serialize(mask=False)


def test_encode_message_parts_concatenate_to_encode_message() -> None:
    from pgw.wire.messages import ComputeResult, ErrorResponse, ServerTiming

    req, _ = make_request(RING, [("0.qkv", 3), ("0.down", 1)], rid="p", deadline_ms=1234)
    resp = ComputeResponse("p", (ComputeResult("0.qkv", 3, b"\x01" * 15 * 24),
                                 ComputeResult("0.down", 1, memoryview(b"\x02" * 80))), ServerTiming(0.5, 1.25, 4))
    err = ErrorResponse(code=str(ErrorCode.RATE_LIMITED), message="m", request_id="p")
    for msg in (req, resp, err):
        parts = encode_message_parts(msg)
        assert b"".join(bytes(p) for p in parts) == encode_message(msg)
    # payloads are passed through, not copied
    assert any(p is req.items[0].x_hat for p in encode_message_parts(req))
