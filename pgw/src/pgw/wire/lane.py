"""Trip lane: one masked round trip per call over a fixed binary header (chat-speed plan WP-A3).

The lane is a second transport next to the WebSocket stream channel (:mod:`pgw.wire.client`). It
carries exactly the same masked bytes: the request body is the ``x_hat`` of a one-item
:class:`~pgw.wire.messages.ComputeRequest`, the reply body its ``y_hat``. Only the framing changes (a
36-byte header instead of msgpack), so the provider learns nothing new. Whenever the lane cannot serve a
call (refusal, broken connection, timeout, a call it does not take), the call is resent on the regular
path as the **same** ``ComputeRequest`` object: byte-identical ``x_hat`` under the same request id, so a
resend never reveals anything the lane attempt did not, and never takes a new mask.

Lifecycle::

    ticket = client.lane_open_sync(fp, ring)          # over the authenticated regular channel
    ch = open_channel(ticket, client.config, ...)      # UDS, or TCP that is always TLS; hello + ticket
    status, compute_us = ch.trip(ring_ix, linear_ix, rows, rid16, deadline_ms, x_hat, out, timeout_s)

:class:`LaneComputeClient` wraps a :class:`~pgw.wire.client.ComputeClient` and does all of that (lazy
open, a channel pool, cool-down after failures, fallback); it is a drop-in for the client (same
``compute_sync`` / ``compute``), and ``compute_into`` reads the reply straight into a caller-owned buffer.
The channel is the native ``pgw_fast.TripChannel`` (GIL released, ``writev``, hybrid spin-then-block
wait) when the extension has it, else :class:`PyTripChannel` (same bytes, plain blocking sockets).

Wire format (lane version 1, all integers little-endian; normative for ``rust/pgw-lane`` (WP-A1), with
golden vectors in ``tests/wire/lane_golden.bin``):

=========  ======================================================================================
HELLO      40 bytes, client -> server, once per connection, first bytes on the socket:
           ``magic "PGWH" | version u16 | kind u16 = 1 | ticket 32 bytes``
REQUEST    36 bytes + ``body_len`` bytes of ``x_hat``:
           ``magic "PGWT" | version u16 | kind u8 = 2 | ring u8 | linear u16 | rows u16 |
           deadline_ms u32 | body_len u32 | request_id 16 bytes``
REPLY      36 bytes + ``body_len`` bytes of ``y_hat`` (server -> client; also the hello answer):
           ``magic "PGWR" | version u16 | kind u8 (3 = HELLO_ACK, 4 = TRIP_REPLY) | status u8 |
           linear u16 | rows u16 | compute_us u32 | body_len u32 | request_id 16 bytes``
=========  ======================================================================================

* ``ring`` / ``linear`` index the ticket's ``rings`` / ``linears`` tables (public, per model).
* ``rows``: 1 .. the ticket's ``max_rows``. ``body_len = rows * n * K/8`` (request) and
  ``rows * m * K/8`` (OK reply); ``0`` in every refusal and in HELLO_ACK.
* ``deadline_ms``: the request's ``deadline_ms`` (a per-deployment constant of the gateway configuration,
  already in every msgpack request today); ``0`` = none.
* ``request_id``: the 16 bytes of the request id, which must be 32 lowercase hex characters (the
  ``uuid4().hex`` the backend uses); a fallback sends the same id as hex.
* HELLO_ACK: ``status`` (0 or a refusal), ``rows`` = the largest rows the connection serves, every other
  field zero. A trip reply echoes ``linear``, ``rows`` and ``request_id``; ``compute_us`` is the service's
  own compute time (server -> client only).
* Statuses: :class:`Status`. After a refusal the server has consumed the request body (except for a
  ``body_len`` above its limit, after which it closes) and the connection stays usable, except after the
  statuses in :data:`CLOSING_STATUSES`, after which the server closes it.
* No other field exists; :data:`REQUEST_FIELDS` is the allowlist (``tests/wire/test_lane_protocol.py``).

What the service side (``rust/pgw-lane`` and the ``lane_open`` route, WP-A1) must do for this client:

* answer ``POST /pgw/v1/lane_open`` (:class:`~pgw.wire.messages.LaneOpenRequest`, authenticated like a
  compute call) with a :class:`~pgw.wire.messages.LaneTicket`, or an ``unavailable`` error when it offers
  no lane; the ticket's tables are the model's linears and the served rings, in a fixed order;
* bind each accepted connection to the ticket's tenant, fingerprint and rings; check ``SO_PEERCRED`` on
  UDS; on TCP speak TLS 1.3 only (the native client offers nothing older) and select ALPN
  :data:`LANE_ALPN` when the client offers it;
* consume the body of every refused trip (so the connection stays in sync) unless closing, and echo
  ``linear``, ``rows`` and ``request_id`` in every trip reply;
* treat ``request_id`` as the msgpack request id (a fallback resends the same bytes under its hex form).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import enum
import ipaddress
import logging
import os
import re
import socket
import ssl
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from pgw.core.errors import PGWError
from pgw.core.ring import RingParams
from pgw.wire.messages import (
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    LaneOpenRequest,
    LaneTicket,
    LinearInfo,
    ServerTiming,
)

__all__ = [
    "CLOSING_STATUSES",
    "HELLO",
    "HELLO_FIELDS",
    "HELLO_MAGIC",
    "KIND_HELLO",
    "KIND_HELLO_ACK",
    "KIND_TRIP",
    "KIND_TRIP_REPLY",
    "LANE_ALPN",
    "LANE_VERSION",
    "LANE_VERSIONS",
    "REPLY",
    "REPLY_FIELDS",
    "REPLY_MAGIC",
    "REQUEST",
    "REQUEST_FIELDS",
    "REQUEST_MAGIC",
    "LaneComputeClient",
    "LaneConfig",
    "LaneProtocolError",
    "LaneRefusedError",
    "LaneStats",
    "LaneStubServer",
    "PyTripChannel",
    "Status",
    "TripChannelLike",
    "native_channel_class",
    "open_channel",
    "pack_hello",
    "pack_reply",
    "pack_request",
    "request_id_bytes",
    "unpack_hello",
    "unpack_reply",
    "unpack_request",
]

log = logging.getLogger(__name__)

#: Lane wire version implemented here.
LANE_VERSION = 1
LANE_VERSIONS: tuple[int, ...] = (LANE_VERSION,)
#: TLS ALPN protocol id of the TCP lane.
LANE_ALPN = "pgw-lane/1"

HELLO_MAGIC = b"PGWH"
REQUEST_MAGIC = b"PGWT"
REPLY_MAGIC = b"PGWR"
KIND_HELLO = 1
KIND_TRIP = 2
KIND_HELLO_ACK = 3
KIND_TRIP_REPLY = 4
TICKET_BYTES = 32
REQUEST_ID_BYTES = 16

#: ``magic | version | kind | ticket``.
HELLO = struct.Struct("<4sHH32s")
#: ``magic | version | kind | ring | linear | rows | deadline_ms | body_len | request_id``.
REQUEST = struct.Struct("<4sHBBHHII16s")
#: ``magic | version | kind | status | linear | rows | compute_us | body_len | request_id``.
REPLY = struct.Struct("<4sHBBHHII16s")

#: The complete, ordered field lists of each header (the allowlist: nothing else is ever sent).
HELLO_FIELDS: tuple[str, ...] = ("magic", "version", "kind", "ticket")
REQUEST_FIELDS: tuple[str, ...] = (
    "magic", "version", "kind", "ring", "linear", "rows", "deadline_ms", "body_len", "request_id")
REPLY_FIELDS: tuple[str, ...] = (
    "magic", "version", "kind", "status", "linear", "rows", "compute_us", "body_len", "request_id")

_U32 = 0xFFFFFFFF
_RID_RE = re.compile(r"[0-9a-f]{32}")  # used with fullmatch (``$`` would accept a trailing newline)


class Status(enum.IntEnum):
    """Status byte of a lane reply. Every non-OK status sends the call back to the regular path."""

    OK = 0
    BAD_HEADER = 1  #: malformed header (magic, version, kind, rows 0); the server closes
    BAD_TICKET = 2  #: hello: unknown, expired, used-up or wrong-tenant ticket; the server closes
    UNKNOWN_LINEAR = 3  #: ``linear`` outside the ticket's table
    BAD_RING = 4  #: ``ring`` outside the ticket's table
    TOO_MANY_ROWS = 5  #: ``rows`` above what the lane serves
    BAD_LENGTH = 6  #: ``body_len`` is not ``rows * n * K/8`` (closes when above the server's limit)
    DEADLINE = 7  #: cannot run within ``deadline_ms``
    BUSY = 8  #: admission / quota / one-call-per-device: use the regular path for this call
    DRAINING = 9  #: the service is shutting the lane down; the server closes
    INTERNAL = 10  #: the lane failed (e.g. a graph self-check mismatch disabled it); the server closes


#: Statuses after which the server closes the connection (the client drops it and cools down).
CLOSING_STATUSES: frozenset[int] = frozenset({Status.BAD_HEADER, Status.BAD_TICKET, Status.DRAINING,
                                              Status.INTERNAL})


class LaneProtocolError(ValueError):
    """A lane header that is not valid lane version 1 (or does not answer the request)."""


class LaneRefusedError(ConnectionError):
    """The server refused the hello (``status``: :class:`Status`)."""

    def __init__(self, status: int):
        super().__init__(f"lane hello refused (status {status})")
        self.status = int(status)


# ---------------------------------------------------------------------------------------------
# header codec
# ---------------------------------------------------------------------------------------------
def request_id_bytes(request_id: str) -> bytes | None:
    """The 16 header bytes of ``request_id`` (32 lowercase hex characters), or ``None`` when the id has
    another form (such a call is not sent on the lane)."""
    if type(request_id) is not str or _RID_RE.fullmatch(request_id) is None:
        return None
    return bytes.fromhex(request_id)


def _u(v: int, bits: int, what: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < (1 << bits):
        raise LaneProtocolError(f"{what} must be an unsigned {bits}-bit integer")
    return int(v)


def pack_hello(ticket: bytes | bytearray | memoryview) -> bytes:
    """The 40-byte hello carrying ``ticket``."""
    t = bytes(ticket)
    if len(t) != TICKET_BYTES:
        raise LaneProtocolError(f"ticket must be {TICKET_BYTES} bytes")
    return HELLO.pack(HELLO_MAGIC, LANE_VERSION, KIND_HELLO, t)


def unpack_hello(data: bytes | bytearray | memoryview) -> bytes:
    """The ticket of a hello (server side)."""
    if len(data) != HELLO.size:
        raise LaneProtocolError("hello has the wrong size")
    magic, version, kind, ticket = HELLO.unpack(data)
    if magic != HELLO_MAGIC or version != LANE_VERSION or kind != KIND_HELLO:
        raise LaneProtocolError("not a lane v1 hello")
    return cast(bytes, ticket)


def pack_request(ring: int, linear: int, rows: int, request_id: bytes, deadline_ms: int, body_len: int) -> bytes:
    """The 36-byte trip request header. Its arguments are the complete list of what it carries."""
    if len(request_id) != REQUEST_ID_BYTES:
        raise LaneProtocolError(f"request_id must be {REQUEST_ID_BYTES} bytes")
    if _u(rows, 16, "rows") == 0:
        raise LaneProtocolError("rows must be >= 1")
    return REQUEST.pack(REQUEST_MAGIC, LANE_VERSION, KIND_TRIP, _u(ring, 8, "ring"), _u(linear, 16, "linear"),
                        rows, _u(deadline_ms, 32, "deadline_ms"), _u(body_len, 32, "body_len"), bytes(request_id))


def unpack_request(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Fields of a trip request header (server side, and the golden-vector tests)."""
    if len(data) != REQUEST.size:
        raise LaneProtocolError("request header has the wrong size")
    d = dict(zip(REQUEST_FIELDS, REQUEST.unpack(data), strict=True))
    if d["magic"] != REQUEST_MAGIC or d["version"] != LANE_VERSION or d["kind"] != KIND_TRIP or d["rows"] == 0:
        raise LaneProtocolError("not a lane v1 trip request")
    return d


def pack_reply(kind: int, status: int, linear: int, rows: int, compute_us: int, body_len: int,
               request_id: bytes) -> bytes:
    """The 36-byte reply header (server side, and the golden-vector tests)."""
    if kind not in (KIND_HELLO_ACK, KIND_TRIP_REPLY):
        raise LaneProtocolError("reply kind must be HELLO_ACK or TRIP_REPLY")
    if len(request_id) != REQUEST_ID_BYTES:
        raise LaneProtocolError(f"request_id must be {REQUEST_ID_BYTES} bytes")
    return REPLY.pack(REPLY_MAGIC, LANE_VERSION, kind, _u(status, 8, "status"), _u(linear, 16, "linear"),
                      _u(rows, 16, "rows"), _u(compute_us, 32, "compute_us"), _u(body_len, 32, "body_len"),
                      bytes(request_id))


def unpack_reply(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Fields of a reply header; checks magic, version, kind and the HELLO_ACK / refusal invariants."""
    if len(data) != REPLY.size:
        raise LaneProtocolError("reply header has the wrong size")
    d = dict(zip(REPLY_FIELDS, REPLY.unpack(data), strict=True))
    if d["magic"] != REPLY_MAGIC or d["version"] != LANE_VERSION or d["kind"] not in (KIND_HELLO_ACK,
                                                                                      KIND_TRIP_REPLY):
        raise LaneProtocolError("not a lane v1 reply")
    if d["kind"] == KIND_HELLO_ACK and (d["linear"] or d["compute_us"] or d["body_len"]
                                        or d["request_id"] != bytes(REQUEST_ID_BYTES)):
        raise LaneProtocolError("hello ack with non-zero fields")
    if d["status"] != Status.OK and d["body_len"] != 0:
        raise LaneProtocolError("refusal with a body")
    return d


# ---------------------------------------------------------------------------------------------
# channels
# ---------------------------------------------------------------------------------------------
class TripChannelLike(Protocol):
    """What :class:`LaneComputeClient` needs from a channel (``pgw_fast.TripChannel`` and
    :class:`PyTripChannel` both provide it)."""

    @property
    def closed(self) -> bool: ...

    @property
    def max_rows(self) -> int: ...

    def trip(self, ring: int, linear: int, rows: int, request_id: bytes, deadline_ms: int, body: Any, out: Any,
             timeout_s: float) -> tuple[int, int]: ...

    def close(self) -> None: ...


def _recv_exact(sock: socket.socket, view: memoryview) -> None:
    got = 0
    n = len(view)
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionResetError("lane connection closed by the service")
        got += k


class PyTripChannel:
    """Pure-Python lane channel (the fallback when ``pgw_fast`` has no ``TripChannel``): same bytes on the
    wire, blocking sockets (no spin phase), the reply read straight into ``out``.

    One trip at a time per channel (a lock); any error closes the channel.
    """

    def __init__(self, sock: socket.socket, ticket: bytes | bytearray, timeout_s: float):
        self._sock: socket.socket | None = sock
        self._lock = threading.Lock()
        self._tls = isinstance(sock, ssl.SSLSocket)
        self._hdr = bytearray(REPLY.size)
        try:
            sock.settimeout(timeout_s)
            hello = bytearray(pack_hello(ticket))
            try:
                sock.sendall(hello)
            finally:
                hello[:] = bytes(len(hello))
            _recv_exact(sock, memoryview(self._hdr))
            d = unpack_reply(self._hdr)
            if d["kind"] != KIND_HELLO_ACK:
                raise LaneProtocolError("expected a hello ack")
            if d["status"] != Status.OK:
                raise LaneRefusedError(d["status"])
            if d["rows"] < 1:
                raise LaneProtocolError("hello ack serves no rows")
            self._max_rows = int(d["rows"])
        except BaseException:
            self.close()
            raise

    @classmethod
    def connect_uds(cls, path: str, ticket: bytes | bytearray, timeout_s: float) -> PyTripChannel:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout_s)
            sock.connect(path)
        except BaseException:
            sock.close()
            raise
        return cls(sock, ticket, timeout_s)

    @classmethod
    def connect_tls(cls, host: str, port: int, context: ssl.SSLContext, ticket: bytes | bytearray,
                    timeout_s: float) -> PyTripChannel:
        """TCP is always TLS: ``context`` must verify the server (``CERT_REQUIRED`` and a hostname check)."""
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise ValueError("the TCP lane needs a verifying TLS context")
        raw = socket.create_connection((host, port), timeout=timeout_s)
        try:
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock = context.wrap_socket(raw, server_hostname=host)
        except BaseException:
            raw.close()
            raise
        if sock.selected_alpn_protocol() not in (None, LANE_ALPN):
            sock.close()
            raise LaneProtocolError("the server did not select the lane ALPN protocol")
        return cls(sock, ticket, timeout_s)

    @property
    def closed(self) -> bool:
        return self._sock is None

    @property
    def max_rows(self) -> int:
        return self._max_rows

    def trip(self, ring: int, linear: int, rows: int, request_id: bytes, deadline_ms: int, body: Any, out: Any,
             timeout_s: float) -> tuple[int, int]:
        """One trip: send header + ``body``, read the reply into ``out`` (exactly ``len(out)`` bytes on OK).

        Returns ``(status, compute_us)``. Raises ``OSError`` / :class:`LaneProtocolError` (the channel is
        then closed)."""
        bview = memoryview(body).cast("B")
        oview = memoryview(out).cast("B")
        with self._lock:
            sock = self._sock
            if sock is None:
                raise ConnectionError("lane channel is closed")
            try:
                hdr = pack_request(ring, linear, rows, request_id, deadline_ms, len(bview))
                sock.settimeout(max(timeout_s, 1e-3))
                if self._tls:
                    sock.sendall(hdr + bview.tobytes())  # SSLSocket has no sendmsg
                else:
                    sent = sock.sendmsg([hdr, bview])
                    total = len(hdr) + len(bview)
                    if sent < total:  # finish a partial writev
                        rest = (hdr + bview.tobytes())[sent:]
                        sock.sendall(rest)
                _recv_exact(sock, memoryview(self._hdr))
                d = unpack_reply(self._hdr)
                if (d["kind"] != KIND_TRIP_REPLY or d["request_id"] != bytes(request_id) or d["linear"] != linear
                        or d["rows"] != rows):
                    raise LaneProtocolError("lane reply does not answer the request")
                if d["status"] != Status.OK:
                    if d["status"] in CLOSING_STATUSES:
                        self._close_locked()
                    return int(d["status"]), int(d["compute_us"])
                if d["body_len"] != len(oview):
                    raise LaneProtocolError("lane reply body has the wrong length")
                _recv_exact(sock, oview)
                return Status.OK, int(d["compute_us"])
            except BaseException:
                self._close_locked()
                raise

    def _close_locked(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()

    def close(self) -> None:
        if self._lock.acquire(timeout=5):
            try:
                self._close_locked()
            finally:
                self._lock.release()
        else:  # a trip is stuck: shut the socket down under it
            sock = self._sock
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)


def native_channel_class() -> Any | None:
    """``pgw_fast.TripChannel`` when the native extension has it (and ``PGW_NATIVE`` is not ``0``)."""
    if os.environ.get("PGW_NATIVE", "").strip().lower() in ("0", "false", "no", "off"):
        return None
    from pgw.core.native import fast

    return getattr(fast, "TripChannel", None) if fast is not None else None


# ---------------------------------------------------------------------------------------------
# configuration, stats
# ---------------------------------------------------------------------------------------------
class LaneConfig(BaseModel):
    """Client-side lane settings (the deployment switch is the gateway's ``lane`` setting)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    native: bool | None = Field(None, description="Native channel: None = when available, True = required, "
                                "False = the Python channel")
    spin_us: int = Field(50, ge=0, le=100_000, description="Native channel: busy-poll the reply this long "
                         "before blocking (0 = block at once)")
    connect_timeout_s: float = Field(2.0, gt=0)
    trip_timeout_s: float | None = Field(None, gt=0, description="Per-trip budget (None = the request's "
                                         "deadline_ms, else the client's request_timeout_s). A lane call that "
                                         "times out falls back with a fresh budget, so the worst case of one "
                                         "call is about this plus the regular path's own timeout")
    pool_wait_s: float = Field(0.01, ge=0, le=1.0, description="When every lane connection is busy and no "
                               "more may be opened, wait this long for one before taking the regular path")
    uds_dir: str | None = Field(None, description="Directory a UDS ticket's socket must be in. None = accept "
                                "a UDS ticket only from a service on this host (a loopback base URL)")
    retry_s: float = Field(30.0, ge=0, description="After a lane failure, use the regular path this long")
    max_channels: int = Field(4, ge=1, le=64, description="Lane connections per client (one trip at a time each)")
    transports: tuple[str, ...] = Field(("uds", "tcp"), description="Transports this client may use")


@dataclass
class LaneStats:
    """Counters (process-local; payloads never recorded)."""

    opens: int = 0
    open_failures: int = 0
    channels_opened: int = 0
    trips: int = 0
    fallbacks: int = 0
    #: fallback reason -> count (``status:<name>``, ``error:<type>``, ``ineligible:<why>``, ``off``)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    bytes_sent: int = 0
    bytes_received: int = 0
    last_rtt_s: float = 0.0
    native: bool = False

    def _fallback(self, reason: str) -> None:
        self.fallbacks += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1


@dataclass(frozen=True)
class _Lane:
    """An accepted ticket, resolved for this client."""

    ticket: LaneTicket
    secret: bytearray = field(repr=False)  # the ticket bytes (zeroized on close; never in a repr)
    expires: float  # monotonic
    rings: Mapping[str, int]
    linears: Mapping[str, tuple[int, int, int]]  # id -> (index, m, n)


_LOOPBACK_NAMES = frozenset({"localhost", "localhost.", "ip6-localhost"})


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _uds_allowed(path: str, base_url: str | None, uds_dir: str | None) -> bool:
    """A UDS ticket names a local socket the client will write the hello (ticket) and ``x_hat`` to. Accept
    it only inside the configured ``uds_dir``, or, without one, from a service on this host."""
    if not path or "\x00" in path or path.startswith("@"):
        return False  # abstract sockets have no file-system owner to check
    if uds_dir is not None:
        root = os.path.realpath(uds_dir)
        sock = os.path.realpath(path)
        return sock != root and os.path.commonpath([root, sock]) == root
    return base_url is not None and _is_loopback_host(urlsplit(base_url).hostname)


def _service_host(base_url: str) -> str:
    host = urlsplit(base_url).hostname
    if not host:
        raise ValueError("base_url has no host")
    return host


def open_channel(ticket: LaneTicket, secret: bytes | bytearray, client_config: Any, config: LaneConfig) -> Any:
    """Open one lane connection for ``ticket`` (native channel when available and usable).

    TCP is TLS only: the client's base URL must be ``https://`` with verification on; the native TLS
    channel is used when the CA is a file (``ca_file``) and no client certificate is configured, else the
    Python channel with the client's own TLS context.

    Raises:
        OSError, LaneProtocolError, LaneRefusedError, ValueError (TCP without TLS).
    """
    native = native_channel_class() if config.native is not False else None
    if config.native is True and native is None:
        raise ValueError("lane native channel required but pgw_fast has no TripChannel")
    timeout = float(config.connect_timeout_s)
    if ticket.transport == "uds":
        assert ticket.path is not None
        if native is not None:
            return native.connect_uds(ticket.path, secret, timeout, int(config.spin_us))
        return PyTripChannel.connect_uds(ticket.path, secret, timeout)
    if ticket.transport != "tcp" or ticket.port is None:
        raise LaneProtocolError(f"unknown lane transport {ticket.transport!r}")
    if not getattr(client_config, "uses_tls", False) or not getattr(client_config, "verify_tls", False):
        raise ValueError("the TCP lane is TLS only: needs an https:// service URL with TLS verification on")
    host = _service_host(client_config.base_url)
    ca_file = getattr(client_config, "ca_file", None)
    if native is not None and ca_file is not None and getattr(client_config, "client_cert_file", None) is None:
        ca_pem = Path(ca_file).read_bytes()
        return native.connect_tls(host, int(ticket.port), host, ca_pem, secret, timeout, int(config.spin_us))
    if config.native is True:
        raise ValueError("lane native TLS channel needs ca_file and no client certificate")
    from pgw.wire.client import build_ssl_context

    ctx = build_ssl_context(client_config, register=False)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.set_alpn_protocols([LANE_ALPN])
    return PyTripChannel.connect_tls(host, int(ticket.port), ctx, secret, timeout)


# ---------------------------------------------------------------------------------------------
# the client wrapper
# ---------------------------------------------------------------------------------------------
class LaneComputeClient:
    """A :class:`~pgw.wire.client.ComputeClient` that sends eligible calls on the trip lane.

    Eligible: one item, ``rows <= max_rows``, a known linear and ring of the ticket's model, a
    ``uuid4().hex``-form request id and ``len(x_hat) == rows * n * K/8``. Everything else, and every lane
    failure, goes to the wrapped client with the **same** request object (byte-identical body and id).
    After a failure that breaks the lane (connection error, closing status, refused hello, failed
    ``lane_open``) the lane is off for :attr:`LaneConfig.retry_s`. A failure seen on a lane that has since
    been replaced never touches the newer one. A UDS ticket is accepted only from a service on this host
    (loopback base URL) or for a socket inside :attr:`LaneConfig.uds_dir`.

    Every other attribute is the wrapped client's, so this is a drop-in where a client is expected.
    """

    def __init__(self, client: Any, config: LaneConfig | None = None, *,
                 channel_factory: Callable[[LaneTicket, bytearray], Any] | None = None):
        self._client = client
        self.lane_config = config or LaneConfig()
        self.lane_stats = LaneStats(native=native_channel_class() is not None and self.lane_config.native is not False)
        self._factory = channel_factory or (lambda t, s: open_channel(t, s, client.config, self.lane_config))
        self._lane: _Lane | None = None
        self._off_until = 0.0
        self._cv = threading.Condition()
        self._idle: list[Any] = []
        self._count = 0  # open channels (idle + in use)
        self._gen = 0  # bumped when the lane is dropped: channels of an older generation are closed on release
        self._open_lock = threading.Lock()
        self._opening = False  # an async lane_open is in flight
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    # ---------------------------------------------------------------- open / close
    @property
    def lane_ticket(self) -> LaneTicket | None:
        lane = self._lane
        return lane.ticket if lane is not None else None

    def _accept(self, ticket: LaneTicket, fingerprint: str, ring: str) -> _Lane:
        if ticket.lane_version not in LANE_VERSIONS:
            raise LaneProtocolError(f"unsupported lane version {ticket.lane_version}")
        if ticket.fingerprint != fingerprint or ring not in ticket.rings:
            raise LaneProtocolError("lane ticket is for another model or ring")
        if ticket.transport not in self.lane_config.transports:
            raise LaneProtocolError(f"lane transport {ticket.transport!r} not allowed by this client")
        if ticket.transport == "uds" and not _uds_allowed(
                ticket.path or "", getattr(getattr(self._client, "config", None), "base_url", None),
                self.lane_config.uds_dir):
            raise LaneProtocolError("UDS lane ticket from a remote service or outside lane uds_dir")
        if len(ticket.rings) > 256 or len(ticket.linears) > 0x10000:
            raise LaneProtocolError("lane ticket tables too large")
        rings = {r: i for i, r in enumerate(ticket.rings)}
        linears = {li.id: (i, li.out_features, li.in_features) for i, li in enumerate(ticket.linears)}
        # keep the credential only in a bytearray that is wiped when the lane is dropped
        public = dataclasses.replace(ticket, ticket=bytes(TICKET_BYTES))
        return _Lane(public, bytearray(ticket.ticket), time.monotonic() + ticket.ttl_ms / 1000.0, rings, linears)

    def _note_open(self, ticket: LaneTicket | BaseException, fingerprint: str, ring: str) -> bool:
        if isinstance(ticket, BaseException):
            self.lane_stats.open_failures += 1
            self._off_until = time.monotonic() + self.lane_config.retry_s
            log.info("trip lane unavailable (%s); regular channel for %.0fs", type(ticket).__name__,
                     self.lane_config.retry_s)
            return False
        try:
            lane = self._accept(ticket, fingerprint, ring)
        except LaneProtocolError as exc:
            return self._note_open(exc, fingerprint, ring)
        self._drop_lane(replace_with=lane)
        self.lane_stats.opens += 1
        return True

    def _have(self, fingerprint: str, ring: str) -> bool:
        lane = self._lane
        return lane is not None and lane.ticket.fingerprint == fingerprint and ring in lane.rings

    def open_sync(self, fingerprint: str, ring: str, *, if_needed: bool = False) -> bool:
        """Get a ticket (``lane_open``) for ``fingerprint`` / ``ring``. ``False``: no lane (cooling down).
        ``if_needed``: keep a lane already open for them (and do not retry while cooling down)."""
        with self._open_lock:
            if if_needed and (self._have(fingerprint, ring) or time.monotonic() < self._off_until):
                return self._have(fingerprint, ring)
            try:
                t: LaneTicket | BaseException = self._client.lane_open_sync(
                    fingerprint, ring, transports=tuple(self.lane_config.transports), lane_versions=LANE_VERSIONS)
            except (PGWError, OSError, RuntimeError) as exc:
                t = exc
            return self._note_open(t, fingerprint, ring)

    async def open(self, fingerprint: str, ring: str, *, if_needed: bool = False) -> bool:
        """Async :meth:`open_sync`. One ``lane_open`` at a time: while another coroutine's is in flight
        this returns at once (``False`` unless a lane is already open for them)."""
        with self._cv:
            if self._have(fingerprint, ring) and if_needed:
                return True
            if self._opening or (if_needed and time.monotonic() < self._off_until):
                return self._have(fingerprint, ring)
            self._opening = True
        try:
            try:
                t: LaneTicket | BaseException = await self._client.lane_open(
                    fingerprint, ring, transports=tuple(self.lane_config.transports), lane_versions=LANE_VERSIONS)
            except (PGWError, OSError, RuntimeError) as exc:
                t = exc
            with self._open_lock:
                if if_needed and self._have(fingerprint, ring):  # a sync caller opened it meanwhile
                    return True
                return self._note_open(t, fingerprint, ring)
        finally:
            with self._cv:
                self._opening = False

    def _drop_lane(self, *, cool_down: bool = False, only: _Lane | None = None,
                   replace_with: _Lane | None = None) -> None:
        """Drop the current lane (close its idle connections, wipe its ticket), or install
        ``replace_with``. ``only``: do nothing unless ``only`` is still the current lane, so a failure
        seen on an older lane never drops (or cools down) a newer one."""
        with self._cv:
            if only is not None and self._lane is not only:
                return
            lane, self._lane = self._lane, replace_with
            idle, self._idle = self._idle, []
            self._count -= len(idle)
            self._gen += 1
            self._cv.notify_all()
        for ch in idle:
            with contextlib.suppress(Exception):
                ch.close()
        if lane is not None:
            lane.secret[:] = bytes(len(lane.secret))
        if cool_down:
            self._off_until = time.monotonic() + self.lane_config.retry_s

    def lane_close(self) -> None:
        """Close the lane connections (the wrapped client stays open)."""
        self._drop_lane()

    def close(self) -> None:
        self._closed = True
        self._drop_lane()
        self._client.close()

    async def aclose(self) -> None:
        self._closed = True
        self._drop_lane()
        await self._client.aclose()

    # ---------------------------------------------------------------- channel pool
    def _acquire(self, lane: _Lane, timeout_s: float) -> tuple[Any, int] | None:
        end = time.monotonic() + min(timeout_s, self.lane_config.pool_wait_s)
        with self._cv:
            while True:
                if self._lane is not lane:
                    return None
                if self._idle:
                    return self._idle.pop(), self._gen
                if self._count < min(self.lane_config.max_channels, lane.ticket.max_connections):
                    if time.monotonic() >= lane.expires:  # cannot open more with this ticket
                        if self._count == 0:
                            return None
                    else:
                        self._count += 1
                        gen = self._gen
                        # a private copy: _drop_lane may wipe lane.secret while the factory runs unlocked
                        secret = bytearray(lane.secret)
                        break
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)
        try:
            ch = self._factory(lane.ticket, secret)
        except BaseException:
            with self._cv:
                self._count -= 1
                self._cv.notify()
            raise
        finally:
            secret[:] = bytes(len(secret))
        self.lane_stats.channels_opened += 1
        return ch, gen

    def _release(self, ch: Any, gen: int) -> None:
        with self._cv:
            if not ch.closed and gen == self._gen and not self._closed:
                self._idle.append(ch)
                self._cv.notify()
                return
            self._count -= 1  # in use when the lane was dropped, or closed: no longer counted
            self._cv.notify()
        with contextlib.suppress(Exception):
            ch.close()

    # ---------------------------------------------------------------- calls
    def _plan(self, request: ComputeRequest) -> tuple[_Lane | None, str, tuple[Any, ...]]:
        lane = self._lane
        if lane is None:
            return None, "off", ()
        if len(request.items) != 1:
            return None, "ineligible:items", ()
        it = request.items[0]
        rid = request_id_bytes(request.request_id)
        if rid is None:
            return None, "ineligible:request_id", ()
        if request.fingerprint != lane.ticket.fingerprint:
            return None, "ineligible:fingerprint", ()
        ring_ix = lane.rings.get(request.ring)
        lin = lane.linears.get(it.linear)
        if ring_ix is None or lin is None:
            return None, "ineligible:table", ()
        rows = int(it.rows)
        if not 1 <= rows <= lane.ticket.max_rows:
            return None, "ineligible:rows", ()
        nb = RingParams.from_name(request.ring).k // 8
        lin_ix, m, n = lin
        if len(it.x_hat) != rows * n * nb:
            return None, "ineligible:length", ()
        deadline = request.deadline_ms if request.deadline_ms is not None else 0
        if not 0 <= deadline <= _U32:
            return None, "ineligible:deadline", ()
        return lane, "", (ring_ix, lin_ix, rows, rid, deadline, it.x_hat, rows * m * nb)

    def _timeout(self, request: ComputeRequest) -> float:
        if self.lane_config.trip_timeout_s is not None:
            return float(self.lane_config.trip_timeout_s)
        if request.deadline_ms:
            return request.deadline_ms / 1000.0
        return float(getattr(self._client.config, "request_timeout_s", 60.0))

    def _ensure_open_sync(self, request: ComputeRequest) -> None:
        if (self._lane is None and not self._closed and time.monotonic() >= self._off_until
                and len(request.items) == 1):
            self.open_sync(request.fingerprint, request.ring, if_needed=True)

    def _try_lane(self, request: ComputeRequest, out: Any | None) -> ComputeResponse | None:
        """The lane attempt: the response, or ``None`` (the caller falls back with the same request)."""
        lane, reason, plan = self._plan(request)
        if lane is None:
            self.lane_stats._fallback(reason)
            return None
        ring_ix, lin_ix, rows, rid, deadline, body, out_len = plan
        if out is None:
            buf: Any = bytearray(out_len)
        else:
            buf = memoryview(out).cast("B")
            if len(buf) != out_len:
                raise ValueError(f"out must be {out_len} bytes, got {len(buf)}")
        timeout = self._timeout(request)
        try:
            got = self._acquire(lane, timeout)
        except Exception as exc:  # opening a connection failed: the lane is down for a while
            self._drop_lane(cool_down=True, only=lane)
            self.lane_stats._fallback(f"error:{type(exc).__name__}")
            return None
        if got is None:
            with self._cv:
                spent = self._lane is lane and self._count == 0 and time.monotonic() >= lane.expires
            if spent:  # ticket expired with no connection left: the next call asks for a new ticket
                self._drop_lane(only=lane)
                self.lane_stats._fallback("expired")
            else:
                self.lane_stats._fallback("busy:pool")
            return None
        ch, gen = got
        if rows > int(ch.max_rows):  # the hello ack may serve fewer rows than the ticket says
            self._release(ch, gen)
            self.lane_stats._fallback("ineligible:rows")
            return None
        t0 = time.monotonic()
        try:
            status, compute_us = ch.trip(ring_ix, lin_ix, rows, rid, deadline, body, buf, timeout)
        except Exception as exc:  # broken connection, timeout, protocol violation: channel closed
            self._release(ch, gen)
            self._drop_lane(cool_down=True, only=lane)
            self.lane_stats._fallback(f"error:{type(exc).__name__}")
            return None
        self.lane_stats.trips += 1
        self.lane_stats.bytes_sent += REQUEST.size + len(body)
        self.lane_stats.last_rtt_s = time.monotonic() - t0
        self._release(ch, gen)
        if status != Status.OK:
            name = Status(status).name if status in Status._value2member_map_ else str(status)
            if status in CLOSING_STATUSES:
                self._drop_lane(cool_down=True, only=lane)
            self.lane_stats._fallback(f"status:{name}")
            return None
        self.lane_stats.bytes_received += REPLY.size + out_len
        it = request.items[0]
        y = bytes(buf) if out is None else buf
        return ComputeResponse(request.request_id, (ComputeResult(it.linear, rows, y),),
                               ServerTiming(0.0, compute_us / 1000.0, rows))

    def trip_ticket(self, request: ComputeRequest, ticket: Any, scale: Any, bias: Any, bound: int, defer: bool,
                    threads: int, split: int) -> tuple[Any, float, int] | None:
        """The native per-trip driver (``PGW_TRIP_DRIVER``, opt-in): the lane call and the trip's finish in one
        native call (``pgw_fast.TripTicket.lane_finish``: header + payload out, the reply read straight into the
        finish, GIL released) on a native channel of this client's pool.

        Returns ``(finish_result, lane_seconds, reply_bytes)``, or ``None`` when the lane did not serve the call
        (the same eligibility, pool, refusal and error handling and statistics as :meth:`compute_sync`); the ticket
        is then untouched and the caller sends the same request with :meth:`fallback_sync`.
        """
        self._ensure_open_sync(request)
        lane, reason, plan = self._plan(request)
        if lane is None:
            self.lane_stats._fallback(reason)
            return None
        ring_ix, lin_ix, rows, rid, deadline, body, out_len = plan
        if not isinstance(body, bytes):
            self.lane_stats._fallback("ineligible:body")
            return None
        timeout = self._timeout(request)
        try:
            got = self._acquire(lane, timeout)
        except Exception as exc:  # opening a connection failed: the lane is down for a while
            self._drop_lane(cool_down=True, only=lane)
            self.lane_stats._fallback(f"error:{type(exc).__name__}")
            return None
        if got is None:
            with self._cv:
                spent = self._lane is lane and self._count == 0 and time.monotonic() >= lane.expires
            if spent:
                self._drop_lane(only=lane)
                self.lane_stats._fallback("expired")
            else:
                self.lane_stats._fallback("busy:pool")
            return None
        ch, gen = got
        cls = native_channel_class()
        if rows > int(ch.max_rows) or cls is None or not isinstance(ch, cls):
            self._release(ch, gen)
            self.lane_stats._fallback("ineligible:rows" if rows > int(ch.max_rows) else "ineligible:channel")
            return None
        t0 = time.monotonic()
        try:
            status, _compute_us, err, res = ticket.lane_finish(ch, ring_ix, lin_ix, rid, deadline, body, timeout,
                                                              scale, bias, bound, defer, threads, split)
        except BaseException:  # the finish rejected the reply (malformed): the channel itself is fine
            self._release(ch, gen)
            raise
        dt = time.monotonic() - t0
        if err is not None:  # broken connection, timeout, protocol violation: channel closed
            self._release(ch, gen)
            self._drop_lane(cool_down=True, only=lane)
            self.lane_stats._fallback(f"error:{err}")
            return None
        self.lane_stats.trips += 1
        self.lane_stats.bytes_sent += REQUEST.size + len(body)
        self.lane_stats.last_rtt_s = dt
        self._release(ch, gen)
        if status != Status.OK:
            name = Status(status).name if status in Status._value2member_map_ else str(status)
            if status in CLOSING_STATUSES:
                self._drop_lane(cool_down=True, only=lane)
            self.lane_stats._fallback(f"status:{name}")
            return None
        self.lane_stats.bytes_received += REPLY.size + out_len
        return res, dt, out_len

    def fallback_sync(self, request: ComputeRequest) -> ComputeResponse:
        """The wrapped client's blocking compute (after :meth:`trip_ticket` returned ``None``)."""
        return cast(ComputeResponse, self._client.compute_sync(request))

    def compute_sync(self, request: ComputeRequest) -> ComputeResponse:
        """Blocking compute: the lane when it takes the call, else the wrapped client (same request)."""
        self._ensure_open_sync(request)
        resp = self._try_lane(request, None)
        return resp if resp is not None else self._client.compute_sync(request)

    def compute_into(self, request: ComputeRequest, out: Any) -> ComputeResponse:
        """:meth:`compute_sync` with the reply's ``y_hat`` read into ``out`` (a writable, C-contiguous
        buffer of exactly ``rows * m * K/8`` bytes) when the lane serves it; the response's ``y_hat`` is
        then a memoryview of ``out``. A fallback returns the wrapped client's response, and ``out`` is then
        unspecified (a lane attempt that broke mid-reply may have written part of it): always take ``y_hat``
        from the returned response, never from ``out`` alone.
        """
        self._ensure_open_sync(request)
        resp = self._try_lane(request, out)
        return resp if resp is not None else self._client.compute_sync(request)

    async def compute(self, request: ComputeRequest) -> ComputeResponse:
        """Async compute: the lane trip runs in a worker thread; else the wrapped client."""
        if (self._lane is None and not self._closed and time.monotonic() >= self._off_until
                and len(request.items) == 1):
            await self.open(request.fingerprint, request.ring, if_needed=True)
        if self._lane is not None:
            resp = await asyncio.to_thread(self._try_lane, request, None)
            if resp is not None:
                return resp
        else:
            self.lane_stats._fallback("off")
        return cast(ComputeResponse, await self._client.compute(request))


# ---------------------------------------------------------------------------------------------
# reference server (tests and protocol conformance; the production listener is rust/pgw-lane, WP-A1)
# ---------------------------------------------------------------------------------------------
class LaneStubServer:
    """Reference lane server: threads, UDS or TCP+TLS, a synchronous ``compute`` callback.

    ``compute(linear: LinearInfo, rows: int, ring: str, x_hat: bytes) -> y_hat bytes``. Records every
    provider-visible trip in :attr:`trips` (header fields and body) and every hello. Not a production
    server: no admission, quotas or peer-credential checks.

    Test knobs: :attr:`refuse_next` (answer the next trips with these statuses), :attr:`drop_next`
    (close the connection instead of answering the next trip).
    """

    def __init__(self, compute: Callable[[LinearInfo, int, str, bytes], bytes], *, fingerprint: str,
                 rings: tuple[str, ...], linears: tuple[LinearInfo, ...], transport: str = "uds",
                 path: str | None = None, host: str = "127.0.0.1", ssl_context: ssl.SSLContext | None = None,
                 insecure_plaintext_tcp: bool = False, max_rows: int = 8, max_connections: int = 4,
                 ttl_ms: int = 60_000, max_body: int = 64 << 20, record: bool = True):
        if transport == "tcp" and ssl_context is None and not insecure_plaintext_tcp:
            raise ValueError("the TCP lane is TLS only")
        if transport == "uds" and path is None:
            raise ValueError("uds needs a path")
        self.compute = compute
        self.fingerprint, self.rings, self.linears = fingerprint, tuple(rings), tuple(linears)
        self.transport, self.path, self.host = transport, path, host
        self.ssl_context = ssl_context
        self.max_rows, self.max_connections, self.ttl_ms, self.max_body = max_rows, max_connections, ttl_ms, max_body
        self.record = record
        self.trips: list[tuple[dict[str, Any], bytes]] = []
        self.hellos = 0
        self.refuse_next: list[int] = []
        self.drop_next = 0
        self._tickets: dict[bytes, list[float | int]] = {}  # ticket -> [expires, uses left]
        self._lock = threading.Lock()
        self._conns: set[socket.socket] = set()
        self._stop = threading.Event()
        if transport == "uds":
            assert path is not None
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
            self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._srv.bind(path)
            self.port: int | None = None
        else:
            self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._srv.bind((host, 0))
            self.port = int(self._srv.getsockname()[1])
        self._srv.listen(16)
        self._thread = threading.Thread(target=self._accept_loop, name="lane-stub", daemon=True)
        self._thread.start()

    def issue_ticket(self, req: LaneOpenRequest) -> LaneTicket:
        if req.fingerprint != self.fingerprint or req.ring not in self.rings:
            raise ValueError("unknown model or ring")
        secret = os.urandom(TICKET_BYTES)
        with self._lock:
            self._tickets[secret] = [time.monotonic() + self.ttl_ms / 1000.0, self.max_connections]
        return LaneTicket(request_id=req.request_id, lane_version=LANE_VERSION, transport=self.transport,
                          ticket=secret, ttl_ms=self.ttl_ms, max_connections=self.max_connections,
                          max_rows=self.max_rows, fingerprint=self.fingerprint, rings=self.rings,
                          linears=self.linears, path=self.path if self.transport == "uds" else None,
                          port=self.port if self.transport == "tcp" else None)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _check_ticket(self, ticket: bytes) -> bool:
        with self._lock:
            ent = self._tickets.get(ticket)
            if ent is None or time.monotonic() > ent[0] or ent[1] <= 0:
                return False
            ent[1] = int(ent[1]) - 1
            return True

    def _serve(self, conn: socket.socket) -> None:
        sock: socket.socket = conn
        try:
            if self.transport == "tcp":
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self.ssl_context is not None:
                    sock = self.ssl_context.wrap_socket(conn, server_side=True)
            with self._lock:
                self._conns.add(sock)
            hello = bytearray(HELLO.size)
            _recv_exact(sock, memoryview(hello))
            zero = bytes(REQUEST_ID_BYTES)
            try:
                ticket = unpack_hello(hello)
            except LaneProtocolError:
                sock.sendall(pack_reply(KIND_HELLO_ACK, Status.BAD_HEADER, 0, 0, 0, 0, zero))
                return
            if not self._check_ticket(ticket):
                sock.sendall(pack_reply(KIND_HELLO_ACK, Status.BAD_TICKET, 0, 0, 0, 0, zero))
                return
            self.hellos += 1
            sock.sendall(pack_reply(KIND_HELLO_ACK, Status.OK, 0, self.max_rows, 0, 0, zero))
            hdr = bytearray(REQUEST.size)
            while not self._stop.is_set():
                try:
                    _recv_exact(sock, memoryview(hdr))
                except ConnectionError:
                    return
                try:
                    d = unpack_request(hdr)
                except LaneProtocolError:
                    sock.sendall(pack_reply(KIND_TRIP_REPLY, Status.BAD_HEADER, 0, 0, 0, 0, zero))
                    return
                rid = d["request_id"]
                if d["body_len"] > self.max_body:
                    sock.sendall(pack_reply(KIND_TRIP_REPLY, Status.BAD_LENGTH, d["linear"], d["rows"], 0, 0, rid))
                    return
                body = bytearray(d["body_len"])
                _recv_exact(sock, memoryview(body))
                with self._lock:
                    if self.record:
                        self.trips.append((d, bytes(body)))
                    refuse = self.refuse_next.pop(0) if self.refuse_next else None
                    drop = self.drop_next > 0
                    if drop:
                        self.drop_next -= 1
                if drop:
                    return

                def refuse_with(st: int, d: dict[str, Any] = d) -> None:
                    sock.sendall(pack_reply(KIND_TRIP_REPLY, st, d["linear"], d["rows"], 0, 0, d["request_id"]))

                if refuse is not None:
                    refuse_with(refuse)
                    if refuse in CLOSING_STATUSES:
                        return
                    continue
                if d["ring"] >= len(self.rings):
                    refuse_with(Status.BAD_RING)
                    continue
                if d["linear"] >= len(self.linears):
                    refuse_with(Status.UNKNOWN_LINEAR)
                    continue
                if d["rows"] > self.max_rows:
                    refuse_with(Status.TOO_MANY_ROWS)
                    continue
                ring = self.rings[d["ring"]]
                li = self.linears[d["linear"]]
                nb = RingParams.from_name(ring).k // 8
                if d["body_len"] != d["rows"] * li.in_features * nb:
                    refuse_with(Status.BAD_LENGTH)
                    continue
                t0 = time.perf_counter()
                try:
                    y = self.compute(li, d["rows"], ring, bytes(body))
                except Exception:
                    refuse_with(Status.INTERNAL)
                    return
                us = min(_U32, int((time.perf_counter() - t0) * 1e6))
                head = pack_reply(KIND_TRIP_REPLY, Status.OK, d["linear"], d["rows"], us, len(y), rid)
                if isinstance(sock, ssl.SSLSocket):
                    sock.sendall(head + bytes(y))
                else:
                    sent = sock.sendmsg([head, y])
                    if sent < len(head) + len(y):
                        sock.sendall((head + bytes(y))[sent:])
        except (OSError, ssl.SSLError):
            return
        finally:
            with self._lock:
                self._conns.discard(sock)
            with contextlib.suppress(OSError):
                sock.close()

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._srv.close()
        with self._lock:
            conns = list(self._conns)
        for c in conns:
            with contextlib.suppress(OSError):
                c.shutdown(socket.SHUT_RDWR)
        if self.transport == "uds" and self.path is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.path)


# ---------------------------------------------------------------------------------------------
# benchmark plug-in (benchmarks/lane_bench.py --transport pgw.wire.lane:bench_transport)
# ---------------------------------------------------------------------------------------------
class _BenchTransport:
    """One lane channel to a :class:`LaneStubServer` (UDS, zero-cost compute) in this process."""

    def __init__(self, shapes: Mapping[str, tuple[int, int]], ring_k: int, native: bool | None):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="pgw-lane-")
        ring = {32: "K32A12C27", 40: "K40A12", 48: "K48A20"}[ring_k]
        self._nb = ring_k // 8
        linears = tuple(LinearInfo(f"0.{k}", m, n) for k, (m, n) in shapes.items())
        self._index = {k: i for i, k in enumerate(shapes)}
        self._m = {k: m for k, (m, _n) in shapes.items()}
        zeros: dict[int, bytes] = {}

        def compute(li: LinearInfo, rows: int, _ring: str, _x: bytes) -> bytes:
            n = rows * li.out_features * self._nb
            z = zeros.get(n)
            if z is None:
                z = zeros[n] = bytes(n)
            return z

        self.server = LaneStubServer(compute, fingerprint="ab" * 32, rings=(ring,), linears=linears,
                                     transport="uds", path=str(Path(self._tmp.name) / "lane.sock"), record=False)
        ticket = self.server.issue_ticket(LaneOpenRequest("bench", "ab" * 32, ring))
        self.channel = open_channel(ticket, bytearray(ticket.ticket), None, LaneConfig(native=native))
        self._out: dict[int, bytearray] = {}

    def trip(self, linear: str, rows: int, body: bytes) -> memoryview:
        n = rows * self._m[linear] * self._nb
        out = self._out.get(n)
        if out is None:
            out = self._out[n] = bytearray(n)
        status, _ = self.channel.trip(0, self._index[linear], rows, os.urandom(REQUEST_ID_BYTES), 0, body, out, 30.0)
        if status != Status.OK:
            raise ConnectionError(f"lane refused ({status})")
        return memoryview(out)

    def close(self) -> None:
        self.channel.close()
        self.server.close()
        self._tmp.cleanup()


def bench_transport(shapes: Mapping[str, tuple[int, int]], ring_k: int) -> _BenchTransport:
    """``lane_bench.py`` plug-in: the trip lane client (native channel when built) over UDS to the Python
    reference server in a thread of the same process. Only the client half is production code."""
    return _BenchTransport(shapes, ring_k, None)


def bench_transport_python(shapes: Mapping[str, tuple[int, int]], ring_k: int) -> _BenchTransport:
    """:func:`bench_transport` with the pure-Python channel."""
    return _BenchTransport(shapes, ring_k, False)
