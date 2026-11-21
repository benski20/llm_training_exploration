"""Blocking stream lane: compute calls over the service's WebSocket stream channel, sent and received with
plain blocking socket calls on the calling thread (gateway option ``lane = "stream"``).

The native trip lane (:mod:`pgw.wire.lane`) is UDS-only, so a gateway that reaches its provider over a
network link (e.g. the Modal TCP tunnel) used the asyncio stream channel for every round trip: the engine
hands each trip from its plaintext thread to the event loop and back (or, on the synchronous path, to
the client's private loop thread and back). With a remote round trip of milliseconds those hand-offs are
a measurable share of the gateway's per-trip work. This lane removes them: the plaintext thread writes
the request frame itself and blocks in ``recv`` until the reply frame is in, so a whole decode pass runs
on one thread (``TransformerModel.sync_decode`` with the backend's synchronous trips and TripEngine).

What goes on the wire is unchanged. The connection is an ordinary WebSocket to :data:`STREAM_PATH` on the
configured ``service_url`` (same TLS context, ALPN ``http/1.1``, bearer token and protocol header as the
asyncio channel), and every request is the msgpack ``ComputeRequest`` the regular path sends, as one
masked binary frame, exactly the bytes :meth:`ComputeClient.compute_fast` writes. The provider sees the
same bytes, the same request ids and the same call schedule; only the gateway's own thread hand-offs are
gone (timing gets faster, never data-dependent).

Any call the lane cannot complete (no connection, a broken or timed-out connection, an error reply, an
invalid reply) is resent on the wrapped client's regular path as the **same** ``ComputeRequest`` object:
byte-identical body under the same request id, never a new mask (the trip lane's fallback contract). A
failed connect cools the lane down for ``cool_down_s``; a connection idle for longer than ``max_idle_s``
is replaced before use (a silently dropped tunnel would otherwise cost a full timeout).
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from pgw.core.errors import PGWError
from pgw.wire.codec import decode_any, encode_message
from pgw.wire.messages import (
    COMPUTE_PATH,
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    STREAM_PATH,
    ComputeRequest,
    ComputeResponse,
)

__all__ = ["BlockingStreamChannel", "SyncStreamClient", "SyncStreamStats", "build_sync_stream_client"]

log = logging.getLogger(__name__)

_RECV = 1 << 18


class _ChannelClosedError(ConnectionError):
    pass


def _frame_head(b: bytes | bytearray | memoryview) -> tuple[int, int, int, int] | None:
    from pgw.wire.client import _frame_head as fh

    return fh(b)


class BlockingStreamChannel:
    """One WebSocket to :data:`STREAM_PATH`, one call at a time, blocking sockets only (no thread, no loop).

    Raises (constructor): OSError, ssl.SSLError, ConnectionError (the upgrade was refused).
    """

    def __init__(self, config: Any, *, ssl_context: ssl.SSLContext | None = None, timeout_s: float = 5.0):
        from websockets.client import ClientProtocol
        from websockets.uri import parse_uri

        from pgw.wire.client import _masked_binary_head, _ws_apply_mask

        base = str(config.base_url).rstrip("/")
        parts = urlsplit(base)
        tls = parts.scheme == "https"
        host = parts.hostname or "127.0.0.1"
        port = parts.port or (443 if tls else 80)
        self._head = _masked_binary_head
        self._mask = _ws_apply_mask()
        self.max_response_bytes = int(config.max_response_bytes)
        sock = socket.create_connection((host, port), timeout=timeout_s)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            for opt, val in (("TCP_KEEPIDLE", 10), ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 3)):
                if hasattr(socket, opt):
                    with contextlib.suppress(OSError):
                        sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
            if tls:
                if ssl_context is None:  # pragma: no cover - the client always passes its context
                    raise ConnectionError("TLS endpoint without an SSL context")
                sock = ssl_context.wrap_socket(sock, server_hostname=host)
            ws_url = ("wss" if tls else "ws") + "://" + base.split("://", 1)[1] + STREAM_PATH
            proto = ClientProtocol(parse_uri(ws_url), max_size=self.max_response_bytes)
            req = proto.connect()
            req.headers[PROTOCOL_HEADER] = str(PROTOCOL_VERSION)
            token = getattr(config, "token", None)
            if token is not None:
                req.headers["Authorization"] = f"Bearer {token.get_secret_value()}"
            req.headers["User-Agent"] = str(getattr(config, "user_agent", "pgw-wire/1"))
            proto.send_request(req)
            for chunk in proto.data_to_send():
                sock.sendall(chunk)
            while True:
                data = sock.recv(_RECV)
                if not data:
                    raise ConnectionError("stream upgrade: connection closed")
                proto.receive_data(data)
                if proto.events_received():
                    break
            if proto.handshake_exc is not None:
                raise ConnectionError(f"stream upgrade refused ({type(proto.handshake_exc).__name__})")
            # Anything after the 101 response stays with websockets' reader; the service sends nothing
            # before the first request, so the parser below starts on a frame boundary.
            rest = bytes(proto.reader.buffer) if getattr(proto, "reader", None) is not None else b""
        except BaseException:
            sock.close()
            raise
        self._sock = sock
        self._rx = bytearray(rest)
        self.closed = False
        self.last_used = time.monotonic()

    # ---- frames ---------------------------------------------------------------------------------
    def _send_frame(self, opcode_byte: int, payload: bytes | bytearray | memoryview) -> None:
        mask = os.urandom(4)
        head = bytearray(self._head(len(payload)))
        head[0] = opcode_byte
        self._sock.sendall(bytes(head) + mask + self._mask(bytes(payload), mask))

    def _recv_more(self) -> None:
        data = self._sock.recv(_RECV)
        if not data:
            raise _ChannelClosedError("compute stream closed by the service")
        self._rx += data

    def _read_message(self) -> bytes:
        """The next complete binary message (pings answered, pongs dropped)."""
        frags: list[bytes] | None = None
        while True:
            parsed = _frame_head(self._rx)
            while parsed is None:
                self._recv_more()
                parsed = _frame_head(self._rx)
            hl, b0, b1, length = parsed
            if b1 & 0x80 or b0 & 0x70:
                raise _ChannelClosedError("compute stream: masked or extension frame from the service")
            if length > self.max_response_bytes:
                raise _ChannelClosedError("compute stream: reply above the size limit")
            while len(self._rx) < hl + length:
                self._recv_more()
            payload = bytes(self._rx[hl:hl + length])
            del self._rx[:hl + length]
            op, fin = b0 & 0x0F, bool(b0 & 0x80)
            if op == 0x9:  # ping: answer, keep waiting
                self._send_frame(0x8A, payload)
                continue
            if op == 0xA:
                continue
            if op == 0x8:
                raise _ChannelClosedError("compute stream closed by the service")
            if op == 0x2 and frags is None:
                if fin:
                    return payload
                frags = [payload]
                continue
            if op == 0x0 and frags is not None:
                frags.append(payload)
                if sum(len(f) for f in frags) > self.max_response_bytes:
                    raise _ChannelClosedError("compute stream: reply above the size limit")
                if fin:
                    return b"".join(frags)
                continue
            raise _ChannelClosedError("compute stream: unexpected frame")

    def call(self, body: bytes, timeout_s: float) -> Any:
        """Send one encoded request; the decoded reply (``ComputeResponse`` or ``ErrorResponse``).

        Raises: OSError, TimeoutError, ConnectionError, PGWError (undecodable reply). The channel is closed
        on any of them.
        """
        if self.closed:
            raise _ChannelClosedError("channel closed")
        try:
            self._sock.settimeout(max(1e-3, float(timeout_s)))
            self._send_frame(0x82, body)
            msg = self._read_message()
            obj = decode_any(msg, max_bytes=self.max_response_bytes)
        except BaseException:
            self.close()
            raise
        self.last_used = time.monotonic()
        return obj

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with contextlib.suppress(OSError):
                if not isinstance(self._sock, ssl.SSLSocket):
                    self._sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                self._sock.close()


@dataclass
class SyncStreamStats:
    """Counters of the blocking stream lane (no payloads)."""

    calls: int = 0
    fallbacks: int = 0
    opens: int = 0
    open_failures: int = 0
    idle_reopens: int = 0
    last_fallback: str | None = None


class SyncStreamClient:
    """A :class:`~pgw.wire.client.ComputeClient` whose ``compute_sync`` takes the blocking stream lane.

    Every other attribute (``compute``, ``connect``, ``model``, ``stats``, ...) is the wrapped client's, so
    the asyncio path (prefill, other sessions) is unchanged. ``sync_stream = True`` tells
    :class:`~pgw.engine.backends.MaskedBackend` that synchronous trips are fast (TripEngine and synchronous
    decode default on, as with the UDS lane).
    """

    sync_stream = True

    def __init__(self, client: Any, *, max_channels: int = 4, max_idle_s: float = 20.0, cool_down_s: float = 30.0,
                 connect_timeout_s: float | None = None):
        self._client = client
        self.sync_stream_stats = SyncStreamStats()
        self._max_channels = max(1, int(max_channels))
        self._max_idle_s = float(max_idle_s)
        self._cool_down_s = float(cool_down_s)
        cfg = client.config
        self._connect_timeout_s = float(connect_timeout_s if connect_timeout_s is not None else cfg.connect_timeout_s)
        self._lock = threading.Lock()
        self._idle: list[BlockingStreamChannel] = []
        self._open = 0
        self._off_until = 0.0
        self._fails = 0
        self._closed = False
        # Like the asyncio stream channel: a custom transport (tests, in-process services) carries no
        # WebSocket unless the client opted in (the gateway's egress guard over the real network does).
        self._usable = bool(getattr(cfg, "stream", True)) and (
            getattr(client, "_transport", None) is None or bool(getattr(client, "_stream_with_transport", False)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    # ---- channels ------------------------------------------------------------------------------
    def _ssl(self) -> ssl.SSLContext | None:
        cfg = self._client.config
        if not cfg.uses_tls:
            return None
        from pgw.wire.client import build_ssl_context

        ctx = build_ssl_context(cfg, register=False)  # per connection: picks up a reloaded client cert
        ctx.set_alpn_protocols(["http/1.1"])
        return ctx

    def _acquire(self) -> BlockingStreamChannel | None:
        now = time.monotonic()
        stale: list[BlockingStreamChannel] = []
        ch: BlockingStreamChannel | None = None
        with self._lock:
            if self._closed or not self._usable or now < self._off_until:
                return None
            while self._idle:
                c = self._idle.pop()
                if c.closed or now - c.last_used > self._max_idle_s:
                    stale.append(c)
                    self._open -= 1
                    continue
                ch = c
                break
            if ch is None:
                if self._open >= self._max_channels:
                    return None
                self._open += 1
        for c in stale:
            if not c.closed:
                self.sync_stream_stats.idle_reopens += 1
            c.close()
        if ch is not None:
            return ch
        try:
            ch = BlockingStreamChannel(self._client.config, ssl_context=self._ssl(), timeout_s=self._connect_timeout_s)
        except Exception as exc:
            with self._lock:
                self._open -= 1
                self._fails += 1
                cool = min(600.0, self._cool_down_s * 2.0 ** (self._fails - 1))  # repeated failures back off
                self._off_until = time.monotonic() + cool
            self.sync_stream_stats.open_failures += 1
            log.info("blocking stream lane unavailable (%s); stream channel for %.0fs", type(exc).__name__, cool)
            return None
        with self._lock:
            self._fails = 0
        self.sync_stream_stats.opens += 1
        return ch

    def _release(self, ch: BlockingStreamChannel) -> None:
        with self._lock:
            if ch.closed or self._closed:
                self._open -= 1
                ch.close()
                return
            self._idle.append(ch)

    # ---- calls ---------------------------------------------------------------------------------
    def _fallback(self, request: ComputeRequest, why: str) -> ComputeResponse:
        self.sync_stream_stats.fallbacks += 1
        self.sync_stream_stats.last_fallback = why
        return self._client.compute_sync(request)  # type: ignore[no-any-return]

    def compute_sync(self, request: ComputeRequest) -> ComputeResponse:
        """One blocking call on the lane; on any failure the same request on the regular path."""
        client = self._client
        if request.deadline_ms is not None and request.deadline_ms <= 0:
            return client.compute_sync(request)  # type: ignore[no-any-return]
        max_bytes, max_items, max_rows = client._effective_limits()
        if (len(request.items) > max_items or any(it.rows > max_rows for it in request.items)
                or sum(len(it.x_hat) + 96 for it in request.items) + 1024 > max_bytes):
            return client.compute_sync(request)  # type: ignore[no-any-return]  # the regular path splits it
        ch = self._acquire()
        if ch is None:
            return self._fallback(request, "no_channel")
        budget = request.deadline_ms / 1000.0 if request.deadline_ms is not None else client.config.request_timeout_s
        st = client.stats
        st.requests += 1
        st.attempts += 1
        st.stream_requests += 1
        body = encode_message(request)
        t0 = time.monotonic()
        try:
            obj = ch.call(body, budget)
        except (OSError, PGWError) as exc:  # TimeoutError and ConnectionError are OSErrors
            self._release(ch)
            st.requests -= 1  # the fallback counts the call again
            return self._fallback(request, type(exc).__name__)
        if isinstance(obj, ComputeResponse) and obj.request_id != request.request_id:
            ch.close()  # a reply to another request: the channel is out of step, never reuse it
        self._release(ch)
        st.bytes_sent += len(body)
        st.last_rtt_s = time.monotonic() - t0
        if not isinstance(obj, ComputeResponse) or obj.request_id != request.request_id:
            # an error reply (or anything else): the regular path resends the same bytes and maps the
            # outcome exactly as it always does (retry decision, budget, error codes)
            st.requests -= 1
            return self._fallback(request, type(obj).__name__)
        st.bytes_received += obj.payload_bytes
        from pgw.wire.server import validate_compute_response

        try:
            validate_compute_response(request, obj, client._out_features_for(request))
        except PGWError as exc:  # an invalid reply: not retryable (as on the regular path)
            st.failures += 1
            client._hook(COMPUTE_PATH, None, t0, str(exc.code))
            raise
        client._deposit_retry_token()  # like every regular request (shared retry budget)
        client._hook(COMPUTE_PATH, 200, t0, None)
        self.sync_stream_stats.calls += 1
        return obj

    def lane_close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, []
            self._open -= len(idle)
        for c in idle:
            c.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.lane_close()
        self._client.close()

    async def aclose(self) -> None:
        with self._lock:
            self._closed = True
        self.lane_close()
        await self._client.aclose()


def build_sync_stream_client(client: Any, **kw: Any) -> SyncStreamClient:
    """Wrap ``client`` (a :class:`~pgw.wire.client.ComputeClient`) with the blocking stream lane."""
    return SyncStreamClient(client, **kw)
