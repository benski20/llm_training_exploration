"""Async HTTP/2 client for the compute service (pooled, mTLS, retries, deadlines).

The client is used by :class:`pgw.engine.MaskedBackend` (``await client.compute(req)`` or the
blocking ``client.compute_sync(req)``). Retrying is always safe: a compute request is a pure
function of its body, and a retry resends the *identical* ``x_hat`` bytes under the same request id
(``PGW-Request-Id``), so the service learns nothing new and no extra mask is consumed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import logging
import math
import os
import random
import ssl
import struct
import threading
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TypeVar, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from pgw.core.errors import (
    ConfigError,
    ErrorCode,
    FingerprintMismatchError,
    PayloadTooLargeError,
    PGWError,
    ProtocolError,
    ServiceUnavailableError,
    ShapeError,
    VersionMismatchError,
)
from pgw.core.ring import RingParams
from pgw.wire.codec import (
    CONTENT_TYPE,
    JSON_CONTENT_TYPE,
    decode_any,
    decode_error_json,
    decode_message,
    encode_message,
)
from pgw.wire.messages import (
    ATTEMPT_HEADER,
    COMPUTE_PATH,
    DEADLINE_HEADER,
    MODELS_PATH,
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    REQUEST_ID_HEADER,
    STREAM_PATH,
    ComputeItem,
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    ErrorResponse,
    ModelInfo,
    ModelsResponse,
    ServerTiming,
)
from pgw.wire.server import validate_compute_response

__all__ = ["ClientConfig", "ClientStats", "ComputeClient", "build_ssl_context", "client_cert_validity",
           "reload_client_cert"]

log = logging.getLogger(__name__)

T = TypeVar("T")

# msgpack framing overhead budget used when splitting requests by size.
_REQUEST_OVERHEAD = 1024
_ITEM_OVERHEAD = 96
# Caps on non-compute response bodies.
_MAX_ERROR_BODY = 1 << 20
_MAX_MODELS_BODY = 16 << 20


def _fast_lane_default() -> bool:
    """Default of :attr:`ClientConfig.fast_lane`: on unless ``PGW_WIRE_FAST_LANE`` is ``0``/``false``/``off``."""
    return os.environ.get("PGW_WIRE_FAST_LANE", "").strip().lower() not in ("0", "false", "no", "off")


async def _read_capped(resp: httpx.Response, limit: int) -> bytes | None:
    """Read a streamed body; ``None`` (and stop reading) once it exceeds ``limit`` bytes."""
    cl = resp.headers.get("content-length")
    if cl is not None and cl.strip().isdigit() and int(cl) > limit:
        return None
    buf = bytearray()
    async for chunk in resp.aiter_bytes():
        buf += chunk
        if len(buf) > limit:
            return None
    return bytes(buf)


class ClientConfig(BaseModel):
    """Connection settings for one compute service endpoint.

    Embeddable in pydantic-settings classes (e.g. ``PGW_GATEWAY_SERVICE__BASE_URL``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(description="e.g. https://compute.example.net:8443")
    token: SecretStr | None = Field(None, description="Bearer token (tenant credential)")
    ca_file: Path | None = Field(None, description="CA bundle to verify the service (default: system store)")
    client_cert_file: Path | None = Field(None, description="mTLS client certificate (PEM)")
    client_key_file: Path | None = Field(None, description="mTLS client private key (PEM)")
    client_key_password: SecretStr | None = None
    verify_tls: bool = Field(True, description="Verify the server certificate (never disable in production)")
    http2: bool = True
    stream: bool = Field(
        True, description="Send compute calls over one persistent WebSocket (/pgw/v1/stream) instead of one HTTP "
        "request each (lower per-call latency). Falls back to HTTP for a while if the channel cannot be opened",
    )
    stream_retry_s: float = Field(60.0, ge=0, description="After a failed channel open, use HTTP this long")
    fast_lane: bool = Field(
        default_factory=_fast_lane_default,
        description="Lean stream lane: single-item calls on an open channel take one direct attempt (frame written "
        "straight to the socket, reply parsed from the socket buffers without intermediate copies) and fall back "
        "to the regular retry loop with a byte-identical resend. Same validation, retries, budget and stats. "
        "Default from PGW_WIRE_FAST_LANE (on unless 0)",
    )
    connect_timeout_s: float = Field(5.0, gt=0)
    request_timeout_s: float = Field(60.0, gt=0, description="Overall budget per compute() incl. retries")
    pool_timeout_s: float = Field(10.0, gt=0)
    max_connections: int = Field(16, gt=0)
    max_keepalive_connections: int = Field(8, ge=0)
    keepalive_expiry_s: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0)
    backoff_base_s: float = Field(0.05, ge=0)
    backoff_max_s: float = Field(2.0, ge=0)
    max_retry_after_s: float = Field(10.0, ge=0, description="Cap on honoured Retry-After")
    retry_budget_ratio: float = Field(
        0.2, ge=0, description="Shared retry budget: each request deposits this many retry tokens (0.2 = retries "
        "limited to about 20% of requests), so an outage does not multiply the load on the service",
    )
    retry_budget_burst: float = Field(10.0, ge=0, description="Retry tokens available at start / at most banked")
    max_request_bytes: int = Field(256 << 20, gt=4096)
    max_response_bytes: int = Field(1 << 30, gt=4096)
    max_items: int = Field(256, gt=0)
    max_rows_per_item: int = Field(8192, gt=0)
    max_concurrency: int = Field(8, gt=0, description="Parallel sub-requests when a request is split")
    user_agent: str = "pgw-wire/1"

    @field_validator("base_url")
    @classmethod
    def _url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def _pair(self) -> Self:
        if (self.client_cert_file is None) != (self.client_key_file is None):
            raise ValueError("client_cert_file and client_key_file must be set together")
        return self

    @property
    def uses_tls(self) -> bool:
        """True for ``https://`` endpoints."""
        return self.base_url.startswith("https://")


#: A client certificate that expires within this many seconds is not presented (token-only instead).
CERT_EXPIRY_MARGIN_S = 60.0

# Live client TLS contexts by (cert file, key file), so a renewed pair can be loaded into every context
# built from those files (including transports that callers built with :func:`build_ssl_context`).
_CERT_CONTEXTS: dict[tuple[str, str], weakref.WeakSet[ssl.SSLContext]] = {}
_CERT_CLIENTS: weakref.WeakSet[ComputeClient] = weakref.WeakSet()
_CERT_LOCK = threading.Lock()
# Serializes reload_client_cert calls (one renewal's reload never interleaves with another's).
_RELOAD_LOCK = threading.Lock()
_BACKGROUND: set[asyncio.Task[None]] = set()  # strong references to fire-and-forget pool closes


def client_cert_validity(cert_file: Path) -> tuple[float, float] | None:
    """``(not_before, not_after)`` (unix seconds) of the first certificate in ``cert_file``; ``None`` if
    unreadable."""
    try:
        from cryptography import x509

        cert = x509.load_pem_x509_certificates(Path(cert_file).read_bytes())[0]
        return float(cert.not_valid_before_utc.timestamp()), float(cert.not_valid_after_utc.timestamp())
    except (OSError, ValueError, IndexError):
        return None


def _usable_client_cert(config: ClientConfig, now: float) -> bool:
    """False (logged as an error) when the configured client certificate has expired or expires within
    :data:`CERT_EXPIRY_MARGIN_S` (or a tenth of its lifetime, for short-lived test certificates)."""
    assert config.client_cert_file is not None
    window = client_cert_validity(config.client_cert_file)
    if window is None:  # let load_cert_chain report a broken file
        return True
    not_before, not_after = window
    if not_after - now > min(CERT_EXPIRY_MARGIN_S, (not_after - not_before) / 10):
        return True
    log.error("client certificate %s %s; not presenting it (token-only authentication). Renew it or re-enroll "
              "the gateway", config.client_cert_file,
              "has expired" if not_after <= now else f"expires in {not_after - now:.0f}s")
    return False


def build_ssl_context(config: ClientConfig, *, now: float | None = None, register: bool = True) -> ssl.SSLContext:
    """TLS context for ``config``: TLS >= 1.2, server verification against ``ca_file`` (or the
    system store), and the mTLS client certificate when configured.

    A client certificate that has expired or expires within :data:`CERT_EXPIRY_MARGIN_S` is not
    presented (an error is logged): the connection falls back to token-only authentication, which a
    service with optional client certificates accepts, instead of failing every handshake. With
    ``register`` the context is registered so :func:`reload_client_cert` can load a renewed pair into it
    (contexts a :class:`ComputeClient` owns are not registered: it builds a new one and swaps instead).

    Raises:
        ConfigError: unreadable certificate/key files.
    """
    try:
        if config.ca_file is not None:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(config.ca_file))
        else:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        if not config.verify_tls:
            log.warning("TLS server verification is DISABLED for %s (development only)", config.base_url)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        if config.client_cert_file is not None and config.client_key_file is not None:
            if _usable_client_cert(config, time.time() if now is None else now):
                pw = config.client_key_password.get_secret_value() if config.client_key_password else None
                ctx.load_cert_chain(str(config.client_cert_file), str(config.client_key_file), password=pw)
            if register:
                key = (str(Path(config.client_cert_file).resolve()), str(Path(config.client_key_file).resolve()))
                with _CERT_LOCK:
                    _CERT_CONTEXTS.setdefault(key, weakref.WeakSet()).add(ctx)
    except (OSError, ssl.SSLError) as exc:
        raise ConfigError(f"cannot load TLS material: {type(exc).__name__}: {exc}") from exc
    return ctx


def reload_client_cert(cert_file: Path, key_file: Path, *, password: str | None = None) -> int:
    """Load a renewed client certificate/key pair (thread-safe; reloads are serialized).

    Every :class:`ComputeClient` using the pair builds a new TLS context from the files and swaps it in
    (live contexts are not mutated): its pooled HTTP connections are retired (the next request opens a new
    pool that presents the new certificate, so connections opened without a certificate, e.g. after it had
    expired, are not reused; the old pool is closed once calls still using it are past their budget) and
    its compute stream channels close once their in-flight calls are answered (never during a round trip).
    Contexts that callers built themselves with :func:`build_ssl_context` cannot be swapped: the pair is
    loaded into them, so their new handshakes present it. Connections already open keep their certificate
    (the service checks it at the handshake only).

    Returns:
        The number of TLS contexts updated or replaced.

    Raises:
        ConfigError: the new pair cannot be loaded (everything keeps the previous certificate).
    """
    key = (str(Path(cert_file).resolve()), str(Path(key_file).resolve()))
    with _RELOAD_LOCK:
        try:  # validate the pair once before touching anything live
            probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            probe.load_cert_chain(key[0], key[1], password=password)
        except (OSError, ssl.SSLError) as exc:
            raise ConfigError(f"cannot load the renewed client certificate: {type(exc).__name__}: {exc}") from exc
        with _CERT_LOCK:
            contexts = list(_CERT_CONTEXTS.get(key, ()))
            clients = [c for c in _CERT_CLIENTS if c._cert_key == key]
        swapped = 0
        for c in clients:  # build first: a failure leaves every client on its previous context
            c._reload_tls()
            swapped += 1
        try:
            for ctx in contexts:
                ctx.load_cert_chain(key[0], key[1], password=password)
        except (OSError, ssl.SSLError) as exc:
            raise ConfigError(f"cannot load the renewed client certificate: {type(exc).__name__}: {exc}") from exc
    log.info("loaded the renewed client certificate into %d TLS context(s) and %d client(s)", len(contexts),
             swapped)
    return len(contexts) + swapped


@dataclass
class ClientStats:
    """Counters (cumulative, process-local). Payloads are never recorded."""

    requests: int = 0
    attempts: int = 0
    retries: int = 0
    retries_denied: int = 0
    failures: int = 0
    splits: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_rtt_s: float = 0.0
    stream_requests: int = 0
    stream_opens: int = 0
    stream_open_failures: int = 0
    #: calls whose first attempt took the lean lane (:meth:`ComputeClient.compute_fast`)
    lane_requests: int = 0


_STATUS_CODES: dict[int, ErrorCode] = {
    400: ErrorCode.PROTOCOL,
    401: ErrorCode.UNAUTHENTICATED,
    403: ErrorCode.PERMISSION_DENIED,
    404: ErrorCode.MODEL_NOT_FOUND,
    409: ErrorCode.FINGERPRINT_MISMATCH,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    426: ErrorCode.VERSION_MISMATCH,
    429: ErrorCode.RATE_LIMITED,
    502: ErrorCode.UNAVAILABLE,
    503: ErrorCode.UNAVAILABLE,
    504: ErrorCode.DEADLINE_EXCEEDED,
}


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        v = float(value.strip())
    except ValueError:
        return None  # HTTP-date form is not used by PGW services
    return v if math.isfinite(v) and v >= 0 else None


@dataclass
class _Piece:
    """A row range of one original item, placed in a sub-request."""

    item_index: int
    row0: int
    rows: int


class _StreamReply(Exception):  # noqa: N818 - internal control flow
    """An error reply (or channel failure) on the compute stream, handled by the retry loop."""

    def __init__(self, error: PGWError):
        super().__init__(str(error))
        self.error = error


#: Largest receive buffer a lean channel keeps between frames; larger frames are assembled in a buffer
#: that grows with the data received (nothing allocated up front on a header's word).
_RX_BUFFER_KEEP = 32 << 20


def _frame_head(b: bytes | memoryview) -> tuple[int, int, int, int] | None:
    """``(header length, byte 0, byte 1, payload length)`` of a WebSocket frame header at the start of
    ``b`` (without the masking key: masked frames are refused by the caller); ``None`` if incomplete."""
    if len(b) < 2:
        return None
    b0, b1 = b[0], b[1]
    n = b1 & 0x7F
    if n < 126:
        return 2, b0, b1, n
    if n == 126:
        return (4, b0, b1, int.from_bytes(b[2:4], "big")) if len(b) >= 4 else None
    return (10, b0, b1, int.from_bytes(b[2:10], "big")) if len(b) >= 10 else None


def _ws_apply_mask() -> Callable[[Any, bytes], bytes]:
    try:
        from websockets.speedups import apply_mask
    except ImportError:  # pragma: no cover - pure-Python websockets build
        from websockets.utils import apply_mask
    return cast(Callable[[Any, bytes], bytes], apply_mask)


_U16 = struct.Struct("!BBH")
_U64 = struct.Struct("!BBQ")


def _masked_binary_head(n: int) -> bytes:
    """Header (without the masking key) of one masked, unfragmented binary client frame of ``n`` bytes."""
    if n < 126:
        return bytes((0x82, 0x80 | n))
    if n < 1 << 16:
        return _U16.pack(0x82, 0xFE, n)
    return _U64.pack(0x82, 0xFF, n)


@functools.cache
def _direct_connection_class() -> type[Any] | None:
    """A ``websockets`` client connection class that hands complete binary messages to a sink
    callback (``pgw_sink``) from ``data_received`` instead of queueing them for ``recv()``.

    Uses the documented ``create_connection`` factory hook and ``process_event``; handshake, pings,
    pongs, close frames and size limits stay with ``websockets``. ``None`` when this ``websockets``
    release does not have the hook (the channel then reads with a task).
    """
    try:
        from websockets.asyncio.client import ClientConnection, connect
        from websockets.frames import Frame, Opcode
    except ImportError:  # pragma: no cover - websockets < 13
        return None
    import inspect

    if not callable(getattr(ClientConnection, "process_event", None)) or (
        "create_connection" not in inspect.signature(connect).parameters
    ):  # pragma: no cover - other websockets releases
        return None

    from websockets.protocol import State

    class _DirectConnection(ClientConnection):
        pgw_direct = True
        pgw_sink: Callable[[bytes | bytearray | memoryview | str], None] | None = None
        #: parse the server's frames here (see :meth:`data_received`); set by the channel
        pgw_rx = False
        pgw_max_size: int | None = None
        _pgw_frags: list[bytes] | None = None
        _pgw_on = False  # own parsing active (frame boundary reached with websockets idle)
        _pgw_off = False  # handed back to websockets for good
        _pgw_head: bytes = b""  # partial frame header carried over to the next chunk
        _pgw_buf: memoryview | None = None  # payload being filled (a view of _pgw_rx)
        _pgw_rx: bytearray | None = None  # receive buffer, reused across frames
        _pgw_big: bytearray | None = None  # payload of a frame above _RX_BUFFER_KEEP
        _pgw_need = 0
        _pgw_have = 0

        def data_received(self, data: bytes) -> None:
            """Binary frames are parsed here and their payload handed to the sink with at most one copy
            (into one buffer of the frame's size when it spans several reads; none when it fits in one),
            instead of websockets' growing buffer and its copies per frame. Control frames are handed
            to websockets whole; anything else (text, fragments, oversize, masked or extension frames,
            a closing connection) hands the stream back to websockets for good."""
            if self._pgw_off or not self.pgw_rx:
                super().data_received(data)
                return
            if not self._pgw_on:
                p = self.protocol
                if (self.pgw_sink is None or p.state is not State.OPEN or p.reader.buffer
                        or getattr(p, "current_size", getattr(p, "cur_size", None)) is not None):
                    self._pgw_off = True
                    super().data_received(data)
                    return
                self._pgw_on = True
            self._pgw_feed(data)

        def _pgw_handoff(self, rest: bytes | bytearray | memoryview) -> None:
            self._pgw_off = True
            self._pgw_head = b""
            self._pgw_buf = None
            self._pgw_big = None
            self._pgw_rx = None
            super().data_received(bytes(rest))

        def _pgw_deliver(self, payload: bytes | bytearray | memoryview) -> None:
            sink = self.pgw_sink
            if sink is not None:
                sink(payload)

        def _pgw_feed(self, data: bytes) -> None:
            mv = memoryview(data)
            n = len(mv)
            pos = 0
            while pos < n:
                big = self._pgw_big
                if big is not None:  # a frame above the kept buffer size: grows with what arrives
                    take = min(self._pgw_need - len(big), n - pos)
                    big += mv[pos:pos + take]
                    pos += take
                    if len(big) == self._pgw_need:
                        self._pgw_big = None
                        self._pgw_deliver(big)
                    continue
                buf = self._pgw_buf
                if buf is not None:  # filling a payload that spans reads
                    take = min(len(buf) - self._pgw_have, n - pos)
                    buf[self._pgw_have:self._pgw_have + take] = mv[pos:pos + take]
                    self._pgw_have += take
                    pos += take
                    if self._pgw_have == len(buf):
                        self._pgw_buf = None
                        try:
                            self._pgw_deliver(buf)  # decoded (copied out) before the sink returns
                        finally:
                            buf.release()
                    continue
                head = self._pgw_head + bytes(mv[pos:pos + 10]) if self._pgw_head else mv[pos:pos + 10]
                parsed = _frame_head(head)
                if parsed is None:  # header incomplete: keep it for the next read
                    self._pgw_head = bytes(head)
                    return
                hl, b0, b1, length = parsed
                carried = len(self._pgw_head)
                raw_head = bytes(head[:hl])
                self._pgw_head = b""
                start = pos + hl - carried  # payload start in this read
                op = b0 & 0x0F
                p = self.protocol
                limit = self.pgw_max_size
                ws_limit = getattr(p, "max_message_size", getattr(p, "max_size", None))
                if ws_limit is not None:
                    limit = ws_limit if limit is None else min(limit, ws_limit)
                if b1 & 0x80 or b0 & 0x70 or p.state is not State.OPEN:  # masked, RSV bits, closing
                    self._pgw_handoff(raw_head + bytes(mv[start:]))
                    return
                if op == 0x2 and b0 & 0x80:  # complete binary message
                    if limit is not None and length > limit:
                        self._pgw_handoff(raw_head + bytes(mv[start:]))  # websockets fails it (too big)
                        return
                    if n - start >= length:
                        pos = start + length
                        self._pgw_deliver(mv[start:pos])
                    elif length > _RX_BUFFER_KEEP:  # nothing allocated up front for what has not arrived
                        self._pgw_big = bytearray()
                        self._pgw_need = length
                        pos = start
                    else:  # spans reads: assemble it in the connection's receive buffer (reused)
                        rx = self._pgw_rx
                        if rx is None or len(rx) < length:
                            rx = self._pgw_rx = bytearray(length)
                        self._pgw_buf = memoryview(rx)[:length]
                        self._pgw_have = 0
                        pos = start
                    continue
                if op >= 0x8 and length <= 125 and n - start >= length:  # whole control frame: websockets
                    pos = start + length
                    super().data_received(raw_head + bytes(mv[start:pos]))
                    if p.reader.buffer:  # pragma: no cover - it consumed less than a frame
                        self._pgw_off = True
                        return
                    continue
                self._pgw_handoff(raw_head + bytes(mv[start:]))
                return

        def process_event(self, event: Any) -> None:
            sink = self.pgw_sink
            if sink is None or not isinstance(event, Frame):  # the handshake response, or no channel yet
                super().process_event(event)
                return
            op = event.opcode
            if op is Opcode.BINARY:
                if event.fin:
                    sink(bytes(event.data) if type(event.data) is not bytes else event.data)
                else:
                    self._pgw_frags = [bytes(event.data)]
                return
            if op is Opcode.CONT and self._pgw_frags is not None:  # websockets bounds the total size
                self._pgw_frags.append(bytes(event.data))
                if event.fin:
                    data, self._pgw_frags = b"".join(self._pgw_frags), None
                    sink(data)
                return
            if op is Opcode.TEXT:
                sink("")  # not part of the protocol: the channel closes
                return
            super().process_event(event)  # control frames: pongs (keepalive), close, ...

    return _DirectConnection


class _StreamChannel:
    """One WebSocket to :data:`STREAM_PATH`; concurrent requests are matched to replies by request id.

    With a :func:`_direct_connection_class` connection, every complete binary message is decoded and
    handed to its waiter straight from the connection's ``data_received`` callback (no reader task,
    no message queue: one event-loop wake-up per reply instead of two). Other connections are read
    by a reader task.
    """

    def __init__(self, ws: Any, max_response_bytes: int, *, lean: bool = False):
        self.ws = ws
        self.max_response_bytes = max_response_bytes
        # request id -> waiters in send order (ids are unique in practice; duplicates are answered FIFO)
        self.pending: dict[str, list[asyncio.Future[Any]]] = {}
        self.closed = False
        self.retiring = False  # no new requests; closes once the in-flight ones are answered
        self._send_lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        if getattr(ws, "pgw_direct", False):
            ws.pgw_max_size = max_response_bytes
            ws.pgw_rx = lean
            ws.pgw_sink = self._on_message
            self.reader = loop.create_task(self._watch())
        else:
            self.reader = loop.create_task(self._read())
        #: lean lane: requests may be written to the socket directly (see :meth:`_write_direct`)
        self.lean = lean and getattr(ws, "pgw_direct", False) and all(
            hasattr(ws, a) for a in ("protocol", "transport", "send_in_progress", "paused"))
        self._apply_mask = _ws_apply_mask() if self.lean else None

    def _write_direct(self, body: bytes) -> bool:
        """Write ``body`` as one masked binary frame straight to the transport (buffer list, one copy
        for the mask) when websockets has nothing in progress on the connection; False otherwise
        (the caller then sends through websockets). The frame is what websockets would write."""
        ws = self.ws
        p = ws.protocol
        tr = ws.transport
        if (p.state is not _ws_open() or ws.send_in_progress is not None or ws.paused or self._send_lock.locked()
                or getattr(p, "expect_continuation_frame", False) or p.extensions or tr.is_closing()):
            return False
        mask = os.urandom(4)
        assert self._apply_mask is not None
        tr.writelines([_masked_binary_head(len(body)) + mask, self._apply_mask(body, mask)])
        return True

    def _dispatch(self, msg: bytes | bytearray | memoryview | str) -> str | None:
        """Hand one received message to its waiter; returns why the channel must close, if it must."""
        if isinstance(msg, str):
            return "compute stream sent a text frame"
        try:
            obj = decode_any(msg, max_bytes=self.max_response_bytes)
        except PGWError as exc:
            return f"undecodable compute stream frame ({exc.code})"
        rid = getattr(obj, "request_id", None)
        if rid is None and isinstance(obj, ErrorResponse):
            return f"compute stream closed by the service: {obj.code}: {obj.message}"
        for fut in self.pending.get(rid, ()) if rid else ():
            if not fut.done():
                fut.set_result(obj)
                break
        # nothing waiting: a late reply to a request that already gave up - drop it
        return None

    def _on_message(self, msg: bytes | bytearray | memoryview | str) -> None:
        """Direct dispatch (runs inside the connection's ``data_received``; must not raise)."""
        if self.closed:
            return
        try:
            why = self._dispatch(msg)
        except Exception as exc:  # never break the transport callback
            why = f"compute stream reply handling failed ({type(exc).__name__})"
        if why is not None:
            self._fail_all(why)
            self.ws.pgw_sink = None
            asyncio.get_running_loop().create_task(self._close_ws())

    async def _close_ws(self) -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.ws.close(), timeout=2)

    async def _watch(self) -> None:
        """Direct mode: fail the waiters once the connection is gone."""
        why = "compute stream closed"
        try:
            await self.ws.wait_closed()
            exc = getattr(self.ws.protocol, "close_exc", None) if hasattr(self.ws, "protocol") else None
            if exc is not None and type(exc).__name__ == "ConnectionClosedError":
                why = f"compute stream connection lost ({type(exc).__name__})"
        except Exception as exc:  # pragma: no cover - wait_closed does not raise
            why = f"compute stream connection lost ({type(exc).__name__})"
        finally:
            self._fail_all(why)

    async def _read(self) -> None:
        why = "compute stream closed"
        try:
            async for msg in self.ws:
                stop = self._dispatch(msg)
                if stop is not None:
                    why = stop
                    break
        except Exception as exc:  # connection errors
            why = f"compute stream connection lost ({type(exc).__name__})"
        finally:
            self._fail_all(why)

    def _fail_all(self, why: str) -> None:
        self.closed = True
        for futs in self.pending.values():
            for fut in futs:
                if not fut.done():
                    fut.set_exception(ServiceUnavailableError(why))
        self.pending.clear()

    async def request(self, request_id: str, body: bytes) -> Any:
        """Send one encoded ComputeRequest; returns the decoded ComputeResponse or ErrorResponse."""
        if self.closed:
            raise ServiceUnavailableError("compute stream closed")
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.pending.setdefault(request_id, []).append(fut)
        try:
            try:
                async with self._send_lock:
                    await self.ws.send(body)
            except Exception as exc:
                raise ServiceUnavailableError(f"compute stream send failed ({type(exc).__name__})") from exc
            return await fut
        finally:
            futs = self.pending.get(request_id)
            if futs is not None:
                if fut in futs:
                    futs.remove(fut)
                if not futs:
                    del self.pending[request_id]
            if self.retiring and not self.pending and not self.closed:
                asyncio.get_running_loop().create_task(self.aclose())

    async def request_fast(self, request_id: str, body: bytes, timeout: float) -> Any:
        """:meth:`request` for the lean lane: the frame is written directly when possible, and the
        deadline is a timer on the reply future (``TimeoutError`` when it fires)."""
        if self.closed:
            raise ServiceUnavailableError("compute stream closed")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        waiters = self.pending.get(request_id)
        if waiters is None:
            self.pending[request_id] = [fut]
        else:
            waiters.append(fut)
        timer = loop.call_later(max(0.0, timeout), _expire, fut)
        try:
            if not self._write_direct(body):
                try:
                    async with self._send_lock:
                        await self.ws.send(body)
                except Exception as exc:
                    raise ServiceUnavailableError(f"compute stream send failed ({type(exc).__name__})") from exc
            return await fut
        finally:
            timer.cancel()
            futs = self.pending.get(request_id)
            if futs is not None:
                if fut in futs:
                    futs.remove(fut)
                if not futs:
                    del self.pending[request_id]
            if self.retiring and not self.pending and not self.closed:
                t = loop.create_task(self.aclose())
                _BACKGROUND.add(t)
                t.add_done_callback(_BACKGROUND.discard)

    def retire(self) -> None:
        self.retiring = True
        if not self.pending and not self.closed:
            asyncio.get_running_loop().create_task(self.aclose())

    async def aclose(self) -> None:
        self._fail_all("compute stream closed by the client")
        try:
            await asyncio.wait_for(self.ws.close(), timeout=2)
        except Exception:  # noqa: S110 - best effort
            pass
        self.reader.cancel()
        with contextlib.suppress(BaseException):
            await self.reader


def _expire(fut: asyncio.Future[Any]) -> None:
    if not fut.done():
        fut.set_exception(TimeoutError())


@functools.cache
def _ws_open() -> Any:
    from websockets.protocol import State

    return State.OPEN


class ComputeClient:
    """Client used by :class:`pgw.engine.MaskedBackend`.

    * Retries only retryable errors (``unavailable``, ``rate_limited``, ``deadline_exceeded``,
      ``backend_unavailable``, transport errors) with jittered exponential backoff, honouring
      ``Retry-After``, within the request deadline. Retrying resends the same ``x_hat``
      (the server already saw it, so no extra leakage, and no extra mask is consumed).
    * Splits requests above ``max_request_bytes`` / ``max_items`` / ``max_rows_per_item`` (the
      smaller of the config and the server's advertised limits) into several, sends them
      concurrently, and reassembles the results in the original item order.
    * Verifies on :meth:`connect` that the server serves the expected fingerprint and ring.
    * Validates every response (id, item order, linear, rows, ``y_hat`` length).
    * Never logs payload bytes or credentials.

    Args:
        config: Endpoint settings.
        transport: Optional httpx transport (tests: ``httpx.ASGITransport`` / ``MockTransport``).
        on_attempt: Optional hook ``fn(path, status_or_None, seconds, error_code_or_None)`` called
            after every HTTP attempt (metrics).
        sleep: Injectable async sleep (tests).
        stream_with_transport: Also use the WebSocket compute channel when ``transport`` is set (the
            transport only wraps the real network to ``base_url``).
    """

    def __init__(
        self,
        config: ClientConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        on_attempt: Callable[[str, int | None, float, str | None], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        stream_with_transport: bool = False,
    ):
        self.config = config
        # A custom transport (tests, in-process) usually cannot carry the WebSocket channel; callers
        # whose transport only wraps the real network (e.g. the gateway's egress guard) opt in.
        self._stream_with_transport = stream_with_transport
        self.stats = ClientStats()
        self._transport = transport
        self._on_attempt = on_attempt
        self._sleep = sleep or asyncio.sleep
        self._ssl: ssl.SSLContext | None = build_ssl_context(config, register=False) if config.uses_tls else None
        if not config.uses_tls and transport is None:
            if config.token is not None:
                log.warning(
                    "bearer token configured for a plaintext http:// endpoint %s; the token is sent "
                    "unencrypted (use https:// in production)",
                    config.base_url,
                )
            else:
                log.warning("compute service endpoint %s is plaintext http:// (development only)", config.base_url)
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )
        self._clients_lock = threading.Lock()
        #: pools retired by a client-certificate reload, closed after a grace period (or by aclose/close)
        self._retired_clients: list[tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = []
        self._channels: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _StreamChannel] = (
            weakref.WeakKeyDictionary()
        )
        self._channel_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )
        self._stream_off_until = 0.0
        self._stream_ssl: ssl.SSLContext | None = None
        self._retry_tokens = float(config.retry_budget_burst)
        self._retry_lock = threading.Lock()
        self._model: ModelInfo | None = None
        #: ``(model, {linear: out_features})`` of :attr:`_model` (built once, not per call)
        self._out_features: tuple[ModelInfo, dict[str, int]] | None = None
        self._models: ModelsResponse | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self._closed = False
        self._cert_key: tuple[str, str] | None = None
        if config.client_cert_file is not None and config.client_key_file is not None:
            self._cert_key = (str(Path(config.client_cert_file).resolve()),
                              str(Path(config.client_key_file).resolve()))
            with _CERT_LOCK:
                _CERT_CLIENTS.add(self)

    # -------------------------------------------------------------------------------- plumbing
    def _http(self) -> httpx.AsyncClient:
        """The pooled ``AsyncClient`` of the running event loop (one pool per loop)."""
        if self._closed:
            raise ServiceUnavailableError("compute client is closed")
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            c = self._clients.get(loop)
            if c is None or c.is_closed:
                cfg = self.config
                headers = {
                    "User-Agent": cfg.user_agent,
                    PROTOCOL_HEADER: str(PROTOCOL_VERSION),
                    "Accept": CONTENT_TYPE,
                    # SPEC 6.2: no compression (uniform ring data does not compress).
                    "Accept-Encoding": "identity",
                }
                if cfg.token is not None:
                    headers["Authorization"] = f"Bearer {cfg.token.get_secret_value()}"
                kwargs: dict[str, Any] = {
                    "base_url": cfg.base_url,
                    "headers": headers,
                    "timeout": httpx.Timeout(
                        cfg.request_timeout_s, connect=cfg.connect_timeout_s, pool=cfg.pool_timeout_s
                    ),
                    "limits": httpx.Limits(
                        max_connections=cfg.max_connections,
                        max_keepalive_connections=cfg.max_keepalive_connections,
                        keepalive_expiry=cfg.keepalive_expiry_s,
                    ),
                    "follow_redirects": False,
                    "trust_env": False,
                }
                if self._transport is not None:
                    kwargs["transport"] = self._transport
                else:
                    kwargs["http2"] = cfg.http2
                    kwargs["verify"] = self._ssl if self._ssl is not None else True
                c = httpx.AsyncClient(**kwargs)
                self._clients[loop] = c
            return c

    def _stream_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        scheme, rest = base.split("://", 1)
        return ("wss" if scheme == "https" else "ws") + "://" + rest + STREAM_PATH

    async def _channel(self) -> _StreamChannel | None:
        """The open compute stream of the running loop, opening one if needed; ``None`` = use HTTP."""
        cfg = self.config
        if not cfg.stream or self._closed or (self._transport is not None and not self._stream_with_transport):
            return None
        loop = asyncio.get_running_loop()
        ch = self._channels.get(loop)
        if ch is not None and not ch.closed and not ch.retiring:
            return ch
        if time.monotonic() < self._stream_off_until:
            return None
        lock = self._channel_locks.setdefault(loop, asyncio.Lock())
        async with lock:
            ch = self._channels.get(loop)
            if ch is not None and not ch.closed and not ch.retiring:
                return ch
            if time.monotonic() < self._stream_off_until:
                return None
            headers = {PROTOCOL_HEADER: str(PROTOCOL_VERSION)}
            if cfg.token is not None:
                headers["Authorization"] = f"Bearer {cfg.token.get_secret_value()}"
            try:
                from websockets.asyncio.client import connect

                kw: dict[str, Any] = {
                    "additional_headers": headers, "user_agent_header": cfg.user_agent, "compression": None,
                    "max_size": cfg.max_response_bytes, "open_timeout": cfg.connect_timeout_s,
                    "ping_interval": 20, "ping_timeout": 20, "close_timeout": 2,
                }
                direct = _direct_connection_class()
                if direct is not None:
                    kw["create_connection"] = direct
                if self._ssl is not None:
                    stream_ssl = self._stream_ssl
                    if stream_ssl is None:  # the upgrade is HTTP/1.1: never let ALPN pick h2
                        stream_ssl = build_ssl_context(cfg, register=False)
                        stream_ssl.set_alpn_protocols(["http/1.1"])
                        self._stream_ssl = stream_ssl
                    kw["ssl"] = stream_ssl
                try:
                    ws = await connect(self._stream_url(), proxy=None, **kw)
                except TypeError:  # websockets < 15 has no proxy support (and never uses one)
                    ws = await connect(self._stream_url(), **kw)
            except Exception as exc:
                self.stats.stream_open_failures += 1
                self._stream_off_until = time.monotonic() + cfg.stream_retry_s
                log.info("compute stream unavailable (%s); using HTTP for %.0fs", type(exc).__name__,
                         cfg.stream_retry_s)
                return None
            self.stats.stream_opens += 1
            ch = _StreamChannel(ws, cfg.max_response_bytes, lean=cfg.fast_lane)
            self._channels[loop] = ch
            return ch

    def _reload_tls(self) -> None:
        """Swap in a new TLS context built from the (renewed) certificate files and retire what used the old
        one: the pooled HTTP clients (closed after a grace period covering calls still using them) and the
        compute stream channels. Thread-safe.

        Raises:
            ConfigError: the files cannot be loaded (nothing is changed).
        """
        if self.config.uses_tls:
            new = build_ssl_context(self.config, register=False)
            with self._clients_lock:
                self._ssl = new
                self._stream_ssl = None
                old = list(self._clients.items())
                self._clients.clear()
            # every call bounds itself (retries included) by request_timeout_s: past that no call uses the pool
            grace = self.config.request_timeout_s + self.config.connect_timeout_s
            for loop, c in old:
                self._close_later(loop, c, grace)
        self.retire_channels()

    def _close_later(self, loop: asyncio.AbstractEventLoop, c: httpx.AsyncClient, delay: float) -> None:
        with self._clients_lock:
            self._retired_clients.append((loop, c))

        def schedule() -> None:
            def close() -> None:
                with self._clients_lock:
                    if (loop, c) in self._retired_clients:
                        self._retired_clients.remove((loop, c))
                if not c.is_closed:
                    t = loop.create_task(c.aclose())
                    _BACKGROUND.add(t)
                    t.add_done_callback(_BACKGROUND.discard)

            loop.call_later(delay, close)

        if loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(schedule)
        except RuntimeError:  # pragma: no cover - loop closed concurrently
            pass

    def retire_channels(self) -> None:
        """Retire every open compute stream channel (thread-safe): each closes once its in-flight calls
        are answered and the next call opens a new channel (e.g. with a renewed client certificate)."""
        for loop, ch in list(self._channels.items()):
            if loop.is_closed() or ch.closed or ch.retiring:
                continue
            try:
                loop.call_soon_threadsafe(ch.retire)
            except RuntimeError:  # pragma: no cover - loop closed concurrently
                pass

    async def _drop_channel(self, ch: _StreamChannel) -> None:
        loop = asyncio.get_running_loop()
        if self._channels.get(loop) is ch:
            del self._channels[loop]
        await ch.aclose()

    async def _close_channels(self) -> None:
        current = asyncio.get_running_loop()
        items = list(self._channels.items())
        self._channels.clear()
        for loop, ch in items:
            if loop is current:
                await ch.aclose()
            elif not loop.is_closed() and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(ch.aclose(), loop)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.wrap_future(fut), timeout=5)

    def _effective_limits(self) -> tuple[int, int, int]:
        cfg = self.config
        max_bytes, max_rows = cfg.max_request_bytes, cfg.max_rows_per_item
        if self._model is not None:
            max_bytes = min(max_bytes, self._model.max_request_bytes)
            max_rows = min(max_rows, self._model.max_rows_per_item)
        return max_bytes, cfg.max_items, max_rows

    def _deposit_retry_token(self) -> None:
        with self._retry_lock:
            self._retry_tokens = min(float(self.config.retry_budget_burst),
                                     self._retry_tokens + float(self.config.retry_budget_ratio))

    def _take_retry_token(self) -> bool:
        with self._retry_lock:
            if self._retry_tokens >= 1.0:
                self._retry_tokens -= 1.0
                return True
            return False

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        cfg = self.config
        delay = random.uniform(0.0, min(cfg.backoff_max_s, cfg.backoff_base_s * (2 ** (attempt - 1))))  # noqa: S311
        if retry_after is not None:
            delay = max(delay, min(retry_after, cfg.max_retry_after_s))
        return delay

    @staticmethod
    def _error_from_http(resp: httpx.Response, content: bytes | None) -> PGWError:
        """Typed error for a non-200 reply. ``content`` is ``None`` when the body was too large
        to read (the error is then derived from the HTTP status alone)."""
        ctype = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        err: ErrorResponse | None = None
        try:
            if content and ctype == CONTENT_TYPE:
                err = decode_message(content, ErrorResponse, max_bytes=_MAX_ERROR_BODY)
            elif content and ctype == JSON_CONTENT_TYPE:
                err = decode_error_json(content, max_bytes=_MAX_ERROR_BODY)
        except PGWError:
            err = None
        if err is not None:
            exc = err.to_exception()
        else:
            code = _STATUS_CODES.get(resp.status_code)
            if code is None:
                code = ErrorCode.UNAVAILABLE if resp.status_code >= 500 else ErrorCode.PROTOCOL
            if resp.status_code == 500:
                code = ErrorCode.INTERNAL
            exc = PGWError(
                f"compute service returned HTTP {resp.status_code}",
                code=code,
                details={"http_status": resp.status_code},
            )
        ra = _parse_retry_after(resp.headers.get("retry-after"))
        if ra is not None:
            exc.details.setdefault("retry_after_s", ra)
        return exc

    async def _call(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        request_id: str,
        deadline: float,
        decode: Callable[[bytes], T],
        max_response_bytes: int,
        stream_check: Callable[[ComputeResponse], T] | None = None,
        resume: PGWError | None = None,
    ) -> T:
        """One logical call with retries until success, a non-retryable error or the deadline.

        Response bodies are streamed and capped (``max_response_bytes`` for 200 replies,
        ``_MAX_ERROR_BODY`` for errors), so a broken or hostile server cannot exhaust memory.

        ``resume``: attempt 1 was already made (by :meth:`compute_fast`, which also deposited the
        call's retry token) and failed with this error; the loop continues with it as if its own
        first attempt had failed (same retry decision, backoff and budget), resending ``body``.
        """
        if self._closed:
            raise ServiceUnavailableError("compute client is closed")
        http: httpx.AsyncClient | None = None  # opened on first HTTP use (stream calls never need it)
        attempt = 0
        if resume is not None:
            attempt = 1
            await self._retry_or_raise(resume, attempt, deadline, request_id)
        else:
            self._deposit_retry_token()
        cfg = self.config
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PGWError(
                    "compute deadline exceeded",
                    code=ErrorCode.DEADLINE_EXCEEDED,
                    details={"request_id": request_id, "attempts": attempt - 1},
                )
            self.stats.attempts += 1
            t0 = time.monotonic()
            status: int | None = None
            exc: PGWError
            chan = await self._channel() if stream_check is not None and body is not None else None
            try:
                if chan is not None:
                    self.stats.stream_requests += 1
                    try:
                        async with asyncio.timeout(remaining):
                            obj: Any = await chan.request(request_id, body)  # type: ignore[arg-type]
                    except ServiceUnavailableError as e:  # the channel failed: retry (reconnects or HTTP)
                        obj = e
                    self.stats.bytes_sent += len(body)  # type: ignore[arg-type]
                    self.stats.last_rtt_s = time.monotonic() - t0
                    if isinstance(obj, ComputeResponse):
                        self.stats.bytes_received += obj.payload_bytes
                        result = stream_check(obj)  # type: ignore[misc]
                        self._hook(path, 200, t0, None)
                        return result
                    if isinstance(obj, PGWError):
                        exc = obj
                    elif isinstance(obj, ErrorResponse):
                        exc = obj.to_exception()
                    else:
                        exc = ProtocolError("unexpected message on the compute stream")
                    if exc.code == ErrorCode.UNAVAILABLE:
                        chan.retire()  # e.g. draining: the next attempt opens a new connection
                    raise _StreamReply(exc)
                # The body (and its deadline_ms) stays byte-identical across retries; the server uses
                # what is left of *this* attempt's budget, so an abandoned attempt stops early.
                headers = {REQUEST_ID_HEADER: request_id, ATTEMPT_HEADER: str(attempt),
                           DEADLINE_HEADER: str(max(1, int(remaining * 1000)))}
                if body is not None:
                    headers["Content-Type"] = CONTENT_TYPE
                timeout = httpx.Timeout(
                    remaining, connect=min(cfg.connect_timeout_s, remaining), pool=min(cfg.pool_timeout_s, remaining)
                )
                if http is None:
                    http = self._http()
                # asyncio.timeout also bounds transports that ignore httpx timeouts (e.g. in-process).
                async with asyncio.timeout(remaining):
                    async with http.stream(method, path, content=body, headers=headers, timeout=timeout) as resp:
                        status = resp.status_code
                        limit = max_response_bytes if status == 200 else _MAX_ERROR_BODY
                        content = await _read_capped(resp, limit)
                if body is not None:
                    self.stats.bytes_sent += len(body)
                self.stats.bytes_received += len(content or b"")
                self.stats.last_rtt_s = time.monotonic() - t0
                if status == 200:
                    if content is None:
                        raise ProtocolError(
                            f"compute service response exceeds the limit of {max_response_bytes} bytes",
                            details={"request_id": request_id, "max_bytes": max_response_bytes},
                        )
                    result = decode(content)
                    self._hook(path, status, t0, None)
                    return result
                exc = self._error_from_http(resp, content)
            except (httpx.TimeoutException, TimeoutError) as e:
                exc = PGWError(
                    f"compute service timed out ({type(e).__name__})",
                    code=ErrorCode.DEADLINE_EXCEEDED,
                    details={"request_id": request_id},
                )
            except httpx.TransportError as e:
                exc = ServiceUnavailableError(
                    f"compute service transport error ({type(e).__name__})", details={"request_id": request_id}
                )
            except _StreamReply as e:
                exc = e.error
            except PGWError as e:  # decoding a 200 body failed: not retryable
                self._hook(path, status, t0, str(e.code))
                raise
            self._hook(path, status, t0, str(exc.code))
            await self._retry_or_raise(exc, attempt, deadline, request_id)

    async def _retry_or_raise(self, exc: PGWError, attempt: int, deadline: float, request_id: str) -> None:
        """After failed attempt ``attempt``: wait the backoff and return (retry), or raise ``exc``."""
        cfg = self.config
        retry_after = exc.details.get("retry_after_s")
        ra = float(retry_after) if isinstance(retry_after, int | float) else None
        if exc.retryable and attempt <= cfg.max_retries:
            delay = self._backoff(attempt, ra)
            if time.monotonic() + delay < deadline and not self._take_retry_token():
                self.stats.retries_denied += 1
                exc.details.setdefault("retry_budget_exhausted", True)
            elif time.monotonic() + delay < deadline:
                self.stats.retries += 1
                log.info(
                    "pgw wire retry",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt,
                        "code": str(exc.code),
                        "delay_s": round(delay, 4),
                    },
                )
                await self._sleep(delay)
                return
        exc.details.setdefault("attempts", attempt)
        raise exc

    def _hook(self, path: str, status: int | None, t0: float, code: str | None) -> None:
        if self._on_attempt is not None:
            try:
                self._on_attempt(path, status, time.monotonic() - t0, code)
            except Exception:
                log.exception("on_attempt hook failed")

    # -------------------------------------------------------------------------------- API
    async def connect(self, expected_fingerprint: str | None = None, ring: str | None = None) -> ModelsResponse:
        """Open the pool, fetch ``/pgw/v1/models`` and check fingerprint/ring/version.

        Raises:
            VersionMismatchError: the server does not speak protocol version 1.
            FingerprintMismatchError: ``expected_fingerprint`` is not served.
            ConfigError: ``ring`` is not served for that model.
        """
        models = await self.models()
        if PROTOCOL_VERSION not in models.versions:
            raise VersionMismatchError(
                f"service does not support protocol v{PROTOCOL_VERSION}", details={"versions": list(models.versions)}
            )
        info: ModelInfo | None
        if expected_fingerprint is not None:
            info = models.find(expected_fingerprint)
            if info is None:
                raise FingerprintMismatchError(
                    f"service does not serve model fingerprint {expected_fingerprint[:12]}",
                    details={"expected": expected_fingerprint, "served": [m.fingerprint for m in models.models]},
                )
        else:
            info = models.models[0] if len(models.models) == 1 else None
        if ring is not None:
            rname = RingParams.from_name(ring).name
            candidates = [info] if info is not None else list(models.models)
            if not any(rname in m.rings for m in candidates):
                raise ConfigError(f"service does not serve ring {rname} for this model", details={"ring": rname})
        self._model = info
        log.info(
            "pgw wire connected",
            extra={
                "server": models.server,
                "models": len(models.models),
                "fp": info.fingerprint[:12] if info else None,
            },
        )
        return models

    async def models(self) -> ModelsResponse:
        """``GET /pgw/v1/models`` (with retries)."""
        deadline = time.monotonic() + self.config.request_timeout_s
        res = await self._call(
            "GET",
            MODELS_PATH,
            body=None,
            request_id=uuid.uuid4().hex,
            deadline=deadline,
            decode=lambda b: decode_message(b, ModelsResponse, max_bytes=_MAX_MODELS_BODY),
            max_response_bytes=_MAX_MODELS_BODY,
        )
        self._models = res
        return res

    async def ready(self) -> bool:
        """``GET /readyz`` (single attempt): True on HTTP 200."""
        try:
            r = await self._http().get("/readyz", timeout=self.config.connect_timeout_s)
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    @property
    def model(self) -> ModelInfo | None:
        """The model selected by :meth:`connect` (``None`` before)."""
        return self._model

    async def compute(self, request: ComputeRequest) -> ComputeResponse:
        """Send one compute request (with retries); raises typed :class:`pgw.core.PGWError` on failure.

        Requests above the limits are split by items and rows, sent concurrently (at most
        ``max_concurrency`` at a time) and reassembled transparently. With :attr:`ClientConfig.fast_lane`
        single-item requests go through :meth:`compute_fast`.
        """
        if self.config.fast_lane and len(request.items) == 1:
            return await self.compute_fast(request)
        return await self._compute(request)

    def _lane_channel(self, request: ComputeRequest) -> _StreamChannel | None:
        """The open lean stream channel of the running loop for ``request``, or ``None`` when the
        request is not for the lane (several items, above the limits, no open channel yet)."""
        if self._closed or len(request.items) != 1:
            return None
        ch = self._channels.get(asyncio.get_running_loop())
        if ch is None or not ch.lean or ch.closed or ch.retiring:
            return None
        max_bytes, max_items, max_rows = self._effective_limits()
        it0 = request.items[0]
        budget = max_bytes - _REQUEST_OVERHEAD - len(request.request_id) - len(request.ring)
        if it0.rows > max_rows or len(it0.x_hat) + _ITEM_OVERHEAD > budget or max_items < 1:
            return None
        return ch

    def _out_features_for(self, req: ComputeRequest) -> dict[str, int] | None:
        """``{linear: out_features}`` of the connected model when ``req`` is for it (else ``None``)."""
        model = self._model
        if model is None or model.fingerprint != req.fingerprint:
            return None
        cached = self._out_features
        if cached is None or cached[0] is not model:
            cached = self._out_features = (model, {k: m for k, (m, _) in model.linear_dims().items()})
        return cached[1]

    async def compute_fast(self, request: ComputeRequest) -> ComputeResponse:
        """:meth:`compute` for a single-item request on an open stream channel, with one lean attempt.

        The request frame is written straight to the socket and the reply dispatched from the socket
        callback; the reply is validated exactly like on the regular path. Any failure of this first
        attempt hands over to the regular retry loop (:meth:`_call` with ``resume``), which resends the
        byte-identical body under the same request id (no new information for the service, no extra
        mask), with the same retry decision, backoff, retry budget, deadline and stats. Requests the
        lane does not take (several items, no open channel yet, above the limits) go to :meth:`compute`'s
        regular path.
        """
        ch = self._lane_channel(request)
        if ch is None:
            return await self._compute(request)
        self.stats.requests += 1
        budget = request.deadline_ms / 1000.0 if request.deadline_ms is not None else self.config.request_timeout_s
        deadline = time.monotonic() + budget
        body = encode_message(request)
        out_features = self._out_features_for(request)

        def check(resp: ComputeResponse) -> ComputeResponse:
            validate_compute_response(request, resp, out_features)
            return resp

        def decode(b: bytes) -> ComputeResponse:
            return check(decode_message(b, ComputeResponse, max_bytes=self.config.max_response_bytes))

        self._deposit_retry_token()
        self.stats.attempts += 1
        self.stats.stream_requests += 1
        self.stats.lane_requests += 1
        t0 = time.monotonic()
        exc: PGWError
        try:
            try:
                obj: Any = await ch.request_fast(request.request_id, body, deadline - t0)
            except ServiceUnavailableError as e:  # the channel failed: retry (reconnects or HTTP)
                obj = e
            except TimeoutError as e:
                obj = PGWError(f"compute service timed out ({type(e).__name__})", code=ErrorCode.DEADLINE_EXCEEDED,
                               details={"request_id": request.request_id})
            self.stats.bytes_sent += len(body)
            self.stats.last_rtt_s = time.monotonic() - t0
            if isinstance(obj, ComputeResponse):
                self.stats.bytes_received += obj.payload_bytes
                try:
                    check(obj)
                except PGWError as e:  # an invalid reply: not retryable
                    self._hook(COMPUTE_PATH, None, t0, str(e.code))
                    raise
                self._hook(COMPUTE_PATH, 200, t0, None)
                return obj
            if isinstance(obj, PGWError):
                exc = obj
            elif isinstance(obj, ErrorResponse):
                exc = obj.to_exception()
            else:
                exc = ProtocolError("unexpected message on the compute stream")
            if exc.code == ErrorCode.UNAVAILABLE:
                ch.retire()  # e.g. draining: the next attempt opens a new connection
            self._hook(COMPUTE_PATH, None, t0, str(exc.code))
            return await self._call(
                "POST", COMPUTE_PATH, body=body, request_id=request.request_id, deadline=deadline, decode=decode,
                max_response_bytes=self.config.max_response_bytes, stream_check=check, resume=exc,
            )
        except PGWError:
            self.stats.failures += 1
            raise

    async def _compute(self, request: ComputeRequest) -> ComputeResponse:
        """The regular path of :meth:`compute` (split, retry loop)."""
        self.stats.requests += 1
        budget = request.deadline_ms / 1000.0 if request.deadline_ms is not None else self.config.request_timeout_s
        deadline = time.monotonic() + budget
        try:
            parts = self._split(request)
            if len(parts) == 1 and parts[0][0] is request:
                return await self._send_one(request, deadline)
            self.stats.splits += 1
            sem = asyncio.Semaphore(self.config.max_concurrency)

            async def run(sub: ComputeRequest) -> ComputeResponse:
                async with sem:
                    return await self._send_one(sub, deadline)

            tasks = [asyncio.create_task(run(sub)) for sub, _ in parts]
            try:
                responses = await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return self._assemble(request, [(r, pieces) for r, (_, pieces) in zip(responses, parts, strict=True)])
        except PGWError:
            self.stats.failures += 1
            raise

    async def _send_one(self, req: ComputeRequest, deadline: float) -> ComputeResponse:
        body = encode_message(req)
        out_features = self._out_features_for(req)

        def check(resp: ComputeResponse) -> ComputeResponse:
            validate_compute_response(req, resp, out_features)
            return resp

        def decode(b: bytes) -> ComputeResponse:
            return check(decode_message(b, ComputeResponse, max_bytes=self.config.max_response_bytes))

        t0 = time.monotonic()
        resp = await self._call(
            "POST",
            COMPUTE_PATH,
            body=body,
            request_id=req.request_id,
            deadline=deadline,
            decode=decode,
            max_response_bytes=self.config.max_response_bytes,
            stream_check=check,
        )
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "pgw compute",
                extra={
                    "request_id": req.request_id,
                    "items": len(req.items),
                    "rows": req.total_rows,
                    "bytes_up": len(body),
                    "bytes_down": resp.payload_bytes,
                    "ms": round((time.monotonic() - t0) * 1e3, 3),
                },
            )
        return resp

    def _split(self, req: ComputeRequest) -> list[tuple[ComputeRequest, list[_Piece]]]:
        """Partition ``req`` into sub-requests within the limits (or ``[(req, ...)]`` if it fits)."""
        max_bytes, max_items, max_rows = self._effective_limits()
        budget = max_bytes - _REQUEST_OVERHEAD - len(req.request_id) - len(req.ring)
        req_items = req.items
        if len(req_items) == 1:  # decode calls
            it0 = req_items[0]
            fits = it0.rows <= max_rows and len(it0.x_hat) + _ITEM_OVERHEAD <= budget and max_items >= 1
        else:
            fits = (
                len(req_items) <= max_items
                and all(it.rows <= max_rows for it in req_items)
                and req.payload_bytes + _ITEM_OVERHEAD * len(req_items) <= budget
            )
        if fits:
            return [(req, [_Piece(i, 0, it.rows) for i, it in enumerate(req.items)])]
        groups: list[list[_Piece]] = [[]]
        used = 0
        for i, it in enumerate(req.items):
            if it.rows <= 0 or len(it.x_hat) % it.rows:
                raise ShapeError(f"items[{i}]: x_hat length {len(it.x_hat)} is not a multiple of rows={it.rows}")
            row_bytes = len(it.x_hat) // it.rows
            cap_rows = min(max_rows, (budget - _ITEM_OVERHEAD) // max(row_bytes, 1))
            if cap_rows < 1:
                raise PayloadTooLargeError(
                    f"items[{i}]: one row ({row_bytes} bytes) exceeds the request size limit {max_bytes}",
                    details={"row_bytes": row_bytes, "max_request_bytes": max_bytes},
                )
            row = 0
            while row < it.rows:
                room = budget - used - _ITEM_OVERHEAD
                fit_rows = min(cap_rows, it.rows - row, room // max(row_bytes, 1)) if room > 0 else 0
                if fit_rows < 1 or len(groups[-1]) >= max_items:
                    groups.append([])
                    used = 0
                    continue
                groups[-1].append(_Piece(i, row, fit_rows))
                used += fit_rows * row_bytes + _ITEM_OVERHEAD
                row += fit_rows
        groups = [g for g in groups if g]
        base_id = req.request_id[:100]
        out = []
        for gi, pieces in enumerate(groups):
            items = []
            for p in pieces:
                it = req.items[p.item_index]
                rb = len(it.x_hat) // it.rows
                items.append(ComputeItem(it.linear, p.rows, it.x_hat[p.row0 * rb : (p.row0 + p.rows) * rb]))
            sub = ComputeRequest(
                request_id=f"{base_id}.{gi}",
                fingerprint=req.fingerprint,
                ring=req.ring,
                items=tuple(items),
                deadline_ms=req.deadline_ms,
                version=req.version,
            )
            out.append((sub, pieces))
        log.debug("pgw compute split", extra={"request_id": req.request_id, "parts": len(out)})
        return out

    @staticmethod
    def _assemble(req: ComputeRequest, parts: list[tuple[ComputeResponse, list[_Piece]]]) -> ComputeResponse:
        chunks: list[list[bytes]] = [[] for _ in req.items]
        queue_ms = compute_ms = 0.0
        batch_rows = 0
        for resp, pieces in parts:
            for res, p in zip(resp.items, pieces, strict=True):
                chunks[p.item_index].append(res.y_hat)
            queue_ms = max(queue_ms, resp.timing.queue_ms)
            compute_ms += resp.timing.compute_ms
            batch_rows = max(batch_rows, resp.timing.batch_rows)
        results = tuple(ComputeResult(it.linear, it.rows, b"".join(chunks[i])) for i, it in enumerate(req.items))
        return ComputeResponse(req.request_id, results, ServerTiming(queue_ms, compute_ms, batch_rows))

    # -------------------------------------------------------------------------------- sync API
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                from pgw.core.proc_tuning import new_event_loop

                loop = new_event_loop()  # uvloop when installed (PGW_UVLOOP=0: asyncio's)
                t = threading.Thread(target=loop.run_forever, name="pgw-wire-client", daemon=True)
                t.start()
                self._loop, self._loop_thread = loop, t
            return self._loop

    def _run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        if self._closed:
            coro.close()
            raise ServiceUnavailableError("compute client is closed")
        loop = self._ensure_loop()
        if threading.current_thread() is self._loop_thread:
            coro.close()
            raise RuntimeError("compute_sync() called from the client's own event loop thread")
        fut: concurrent.futures.Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return fut.result()
        except BaseException:
            # e.g. KeyboardInterrupt in the calling thread: do not leave the request running.
            fut.cancel()
            raise

    async def post_raw(self, path: str, body: bytes, request_id: str, decode: Callable[[bytes], T], *,
                       timeout_s: float | None = None, max_response_bytes: int | None = None) -> T:
        """POST a msgpack body to another PGW endpoint with this client's TLS, auth, retries and limits.

        Used by extensions that live outside :mod:`pgw.wire` (e.g. ``/pgw/v1/maskgen``).
        """
        deadline = time.monotonic() + (timeout_s if timeout_s is not None else self.config.request_timeout_s)
        return await self._call("POST", path, body=body, request_id=request_id, deadline=deadline, decode=decode,
                                max_response_bytes=max_response_bytes or self.config.max_response_bytes)

    def post_raw_sync(self, path: str, body: bytes, request_id: str, decode: Callable[[bytes], T], *,
                      timeout_s: float | None = None, max_response_bytes: int | None = None) -> T:
        """Blocking :meth:`post_raw`."""
        return self._run_sync(self.post_raw(path, body, request_id, decode, timeout_s=timeout_s,
                                            max_response_bytes=max_response_bytes))

    def compute_sync(self, request: ComputeRequest) -> ComputeResponse:
        """Blocking wrapper (runs on a private event loop thread)."""
        return self._run_sync(self.compute(request))

    def connect_sync(self, expected_fingerprint: str | None = None, ring: str | None = None) -> ModelsResponse:
        """Blocking :meth:`connect`."""
        return self._run_sync(self.connect(expected_fingerprint, ring))

    async def aclose(self) -> None:
        """Close connections (all pools, including the private sync loop's)."""
        self._closed = True
        await self._close_channels()
        with self._clients_lock:
            items = list(self._clients.items()) + self._retired_clients
            self._clients.clear()
            self._retired_clients = []
        current = asyncio.get_running_loop()
        for loop, c in items:
            if loop is current:
                await c.aclose()
            elif not loop.is_closed() and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(c.aclose(), loop)
                try:
                    await asyncio.wait_for(asyncio.wrap_future(fut), timeout=5)
                except (TimeoutError, RuntimeError):  # pragma: no cover - best effort
                    log.warning("could not close a connection pool cleanly")
        if threading.current_thread() is not self._loop_thread:
            self._stop_loop()

    def close(self) -> None:
        """Synchronous close (call when no event loop is running in this thread).

        Closes the private loop's pool and stops its thread. Pools opened on other event loops
        are dropped; prefer :meth:`aclose` from async code.
        """
        self._closed = True
        loop = self._loop
        if loop is not None and not loop.is_closed() and loop.is_running():
            ch = self._channels.pop(loop, None)
            if ch is not None:
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(ch.aclose(), loop).result(timeout=5)
            with self._clients_lock:
                c = self._clients.pop(loop, None)
                pools = [r for lp, r in self._retired_clients if lp is loop] + ([c] if c is not None else [])
                self._retired_clients = [(lp, r) for lp, r in self._retired_clients if lp is not loop]
            for pool in pools:
                try:
                    asyncio.run_coroutine_threadsafe(pool.aclose(), loop).result(timeout=5)
                except Exception:  # pragma: no cover - best effort
                    log.warning("could not close the sync connection pool cleanly")
        self._stop_loop()

    def _stop_loop(self) -> None:
        with self._loop_lock:
            loop, t = self._loop, self._loop_thread
            self._loop = self._loop_thread = None
        if loop is not None and t is not None:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            if not loop.is_running() and not loop.is_closed():
                loop.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
