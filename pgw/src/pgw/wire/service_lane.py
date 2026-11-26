"""Bridge from the gateway's trip-lane client (:mod:`pgw.wire.lane`, WP-A3) to the service's lane
(:mod:`pgw.service.lane` and ``rust/pgw-lane``, WP-A1).

The two work packages were built against different framings of the same idea:

* the client package speaks lane format "A3" (``POST /pgw/v1/lane_open`` in msgpack, a ``PGWH`` hello
  and ``PGWT`` / ``PGWR`` 36-byte headers; its native channel is ``pgw_fast.TripChannel``);
* the service that ships (``pgw service serve`` with ``lane = "uds"`` or ``"tcp"``) speaks lane format "A1":
  ``POST /pgw/v1/lane/open`` in JSON, then ``PGWL`` 48-byte request headers and 40-byte ``PGWR`` reply
  headers over a Unix domain socket or over TCP with TLS 1.3 (``rust/pgw-lane/src/proto.rs``).

This module lets :class:`~pgw.wire.lane.LaneComputeClient` (pool, cool-down, byte-identical fallback,
statistics) drive the A1 service lane: :class:`ServiceLaneOpener` wraps a
:class:`~pgw.wire.client.ComputeClient` and answers ``lane_open`` from the A1 JSON route, and the channels
speak A1 frames: the native ``pgw_fast.TripChannel`` (``connect_service_uds`` / ``connect_service_tls``: GIL
released for the whole trip, one ``writev`` per request, spin-then-block wait for the reply) when the
extension has it, else :class:`ServiceLaneChannel` (UDS only, plain blocking socket).

TCP lane (gateway ``lane = "tls"``): TLS 1.3 only, to the **host of the client's service URL** (never a host
from the service) on the port the ticket names, the server verified against the client's pinned ``ca_file``
under that host name, ALPN ``pgw-lane/1``, and the client's mTLS certificate presented when it has one (the
service binds the ticket to the certificate that asked for it). The ticket itself comes from ``lane/open`` on
the regular channel (bearer token and client certificate checked as for compute); revoking the token closes
the lane connections. Without the native extension, a ``ca_file`` or with a password-protected client key,
the TCP lane is not used (the calls take the wrapped client's path). Privacy is unchanged: an A1
request header carries the same public fields as the stream channel's request (request id, linear, ring,
rows, deadline, body length) and the body is the same packed ``x_hat``; nothing else is sent.

A1 tickets are single-use (one connection each, valid ``ttl_s`` to open it), so every further connection
of the pool asks for a fresh ticket. A1 statuses are mapped onto the client's
:class:`~pgw.wire.lane.Status`: per-call refusals (``fallback``, ``unavailable``, ``rate_limited``,
``internal``) become a non-closing ``BUSY`` (that call goes to the stream channel), ``deadline_exceeded``
becomes ``DEADLINE``, and every status after which the A1 service closes the connection (16 and above,
``busy`` = 23 included) becomes a closing status, so the client drops the lane and cools down.

Lane format 2 (``multi`` frames, ``rust/pgw-lane/src/proto.rs``): a service that serves them says so in its
ticket (``"versions": [1, 2]``, ``"max_items"``, ``"max_multi_rows"``; a format-1 service sends neither and keeps
``"version": 1``, which every client checks). With :attr:`~pgw.wire.lane.LaneConfig.multi` on
(``PGW_LANE_MULTI=1``; the SDK sets it) a request of several items, the rows of several answers in one batched
trip, goes in one multi call on the native channel: the same items in the same order, the same bytes and request
id as the stream channel's request, under the same connection's tenant. A format-1 ticket, the Python channel or
a shm connection keep one item per call (several-item requests take the stream channel, as before).

``build_lane_client`` is what the gateway uses (``GatewayModelConfig.lane``: ``auto`` / ``required`` accept
UDS tickets only, ``tls`` TCP tickets only).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pgw.core.errors import ProtocolError
from pgw.wire.lane import (
    LaneComputeClient,
    LaneConfig,
    LaneRefusedError,
    Status,
    native_channel_class,
    tune_channel,
)
from pgw.wire.messages import LaneTicket, LinearInfo

__all__ = [
    "SERVICE_LANE_OPEN_PATH",
    "ServiceLaneChannel",
    "ServiceLaneOpener",
    "build_lane_client",
    "map_service_status",
]

#: The service's ticket route (:data:`pgw.service.lane.LANE_OPEN_PATH`).
SERVICE_LANE_OPEN_PATH = "/pgw/v1/lane/open"
_VERSION = 1
_REQUEST_MAGIC = b"PGWL"
_REPLY_MAGIC = b"PGWR"
#: ``magic, version, kind, request id, linear, rows, ring, deadline_ms, body_len, reserved`` (48 bytes)
_REQUEST = struct.Struct("<4sHH16sIHHIIQ")
#: ``magic, version, status, request id, queue_us, compute_us, retry_after_ms, body_len`` (40 bytes)
_REPLY = struct.Struct("<4sHH16sIIII")
_KIND_OPEN = 1
_KIND_CALL = 2
#: lane format 2: the multi frame (``rust/pgw-lane/src/proto.rs``) and its limits
_VERSION_MULTI = 2
_MAX_ITEMS = 64
_MAX_MULTI_ROWS = 256
_TICKET_LEN = 32
_MAX_TICKET_BODY = 1 << 20

#: A1 status -> client status (see the module docstring).
_STATUS_MAP: dict[int, int] = {
    0: Status.OK,
    1: Status.BUSY,  # fallback
    2: Status.BUSY,  # unavailable (e.g. draining): this call on the stream channel
    3: Status.BUSY,  # rate_limited
    4: Status.DEADLINE,  # deadline_exceeded
    5: Status.BUSY,  # internal (the connection stays usable)
    16: Status.BAD_HEADER,  # bad_header
    17: Status.INTERNAL,  # bad_length
    18: Status.INTERNAL,  # unknown_linear
    19: Status.BAD_TICKET,  # unauthenticated
    20: Status.BAD_TICKET,  # peer_refused
    21: Status.BAD_HEADER,  # version
    22: Status.INTERNAL,  # not_open
    23: Status.DRAINING,  # busy: too many lane connections; retry the lane later
}


def map_service_status(status: int) -> int:
    """The client :class:`~pgw.wire.lane.Status` for an A1 reply status (unknown: closing ``INTERNAL``)."""
    return int(_STATUS_MAP.get(int(status), Status.INTERNAL))


def _recv_into(sock: socket.socket, view: memoryview) -> None:
    got = 0
    while got < len(view):
        n = sock.recv_into(view[got:])
        if n == 0:
            raise ConnectionError("lane connection closed by the service")
        got += n


class ServiceLaneChannel:
    """One A1 lane connection (``TripChannelLike``; one trip at a time, plain blocking socket).

    Raises (constructor):
        OSError, LaneRefusedError (the ``open`` frame was refused), ProtocolError.
    """

    def __init__(self, path: str, ticket: bytes | bytearray, timeout_s: float, max_rows: int):
        if len(ticket) != _TICKET_LEN:
            raise ValueError("lane ticket must be 32 bytes")
        self._max_rows = int(max_rows)
        self._lock = threading.Lock()
        self._closed = False
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout_s)
            sock.connect(path)
            sock.sendall(_REQUEST.pack(_REQUEST_MAGIC, _VERSION, _KIND_OPEN, bytes(16), 0, 0, 0, 0, _TICKET_LEN, 0)
                         + bytes(ticket))
            hdr = bytearray(_REPLY.size)
            _recv_into(sock, memoryview(hdr))
            magic, ver, status, _rid, _q, _c, _retry, blen = _REPLY.unpack(hdr)
            if magic != _REPLY_MAGIC or ver != _VERSION or blen != 0:
                raise ProtocolError("not a lane open reply")
            if status != 0:
                raise LaneRefusedError(map_service_status(status))
        except BaseException:
            sock.close()
            raise
        self._sock = sock

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def max_rows(self) -> int:
        return self._max_rows

    def trip(self, ring: int, linear: int, rows: int, request_id: bytes, deadline_ms: int, body: Any, out: Any,
             timeout_s: float) -> tuple[int, int]:
        """One call: ``(client status, compute_us)``; an OK reply's ``y_hat`` is read into ``out``.

        Raises:
            OSError, ConnectionError, TimeoutError, ProtocolError (the channel is closed then).
        """
        with self._lock:
            if self._closed:
                raise ConnectionError("lane channel closed")
            try:
                return self._trip(ring, linear, rows, request_id, deadline_ms, body, out, timeout_s)
            except BaseException:
                self._close_locked()
                raise

    def _trip(self, ring: int, linear: int, rows: int, request_id: bytes, deadline_ms: int, body: Any, out: Any,
              timeout_s: float) -> tuple[int, int]:
        sock = self._sock
        sock.settimeout(max(1e-3, float(timeout_s)))
        mv = memoryview(body).cast("B")
        header = _REQUEST.pack(_REQUEST_MAGIC, _VERSION, _KIND_CALL, bytes(request_id), int(linear), int(rows),
                               int(ring), int(deadline_ms), len(mv), 0)
        sock.sendall(header + bytes(mv) if len(mv) < 4096 else header)
        if len(mv) >= 4096:
            sock.sendall(mv)
        hdr = bytearray(_REPLY.size)
        _recv_into(sock, memoryview(hdr))
        magic, ver, status, rid, _q, compute_us, _retry, blen = _REPLY.unpack(hdr)
        if magic != _REPLY_MAGIC or ver != _VERSION:
            raise ProtocolError("not a lane reply")
        if rid != bytes(request_id):
            raise ProtocolError("lane reply answers another request")
        mapped = map_service_status(status)
        if status == 0:
            dst = memoryview(out).cast("B")
            if blen != len(dst):
                raise ProtocolError("lane reply body has the wrong length")
            _recv_into(sock, dst)
            return int(Status.OK), int(compute_us)
        if blen:  # refusals carry no body; drain defensively to stay in sync
            _recv_into(sock, memoryview(bytearray(blen)))
        if status >= 16:  # the service closes the connection after these
            self._close_locked()
        return mapped, int(compute_us)

    def _close_locked(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._sock.close()
            except OSError:  # pragma: no cover - best effort
                pass

    def close(self) -> None:
        with self._lock:
            self._close_locked()


class ServiceLaneOpener:
    """A :class:`~pgw.wire.client.ComputeClient` whose ``lane_open`` asks the A1 route for a ticket.

    Every other attribute is the wrapped client's. The ticket's tables come from the service's JSON reply
    (indices in its order); the linear shapes from the model the client connected to.
    """

    def __init__(self, client: Any, lane_config: LaneConfig, *, tcp_ports: Any = None):
        self._client = client
        self._lane_config = lane_config
        #: TCP lane ports the gateway may connect to (its egress allowlist for the service host); None = any
        self._tcp_ports: frozenset[int] | None = None if tcp_ports is None else frozenset(int(p) for p in tcp_ports)
        self._used: set[bytes] = set()
        self._used_lock = threading.Lock()
        #: the event loop of the first async lane_open (the client's HTTP pool may be bound to it)
        self._home: asyncio.AbstractEventLoop | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _dims(self) -> dict[str, tuple[int, int]]:
        model = getattr(self._client, "model", None)
        if model is None:
            raise ProtocolError("lane_open before connect(): the linear shapes are unknown")
        return dict(model.linear_dims())

    def _ticket(self, raw: bytes, rid: str, fingerprint: str, ring: str) -> LaneTicket:
        try:
            d = json.loads(raw)
            if (not isinstance(d, dict) or int(d.get("version", 0)) != _VERSION
                    or d.get("transport") not in self._lane_config.transports):
                raise ProtocolError("unsupported lane ticket")
            secret = bytes.fromhex(str(d["ticket"]))
            if len(secret) != _TICKET_LEN or d.get("fingerprint") != fingerprint:
                raise ProtocolError("lane ticket is malformed or for another model")
            dims = self._dims()
            linears = tuple(LinearInfo(str(i), *dims[str(i)]) for i in d["linears"])
            rings = tuple(str(r) for r in d["rings"])
            ttl_ms = max(1, int(float(d.get("ttl_s", 30.0)) * 1000))
            # lane format 2 (multi frames): only when the service lists it; a format-1 ticket has no "versions"
            versions = d.get("versions", [_VERSION])
            multi: dict[str, int] = {}
            if isinstance(versions, list) and _VERSION_MULTI in [int(v) for v in versions]:
                multi = {"max_items": max(0, min(_MAX_ITEMS, int(d.get("max_items", 0)))),
                         "max_multi_rows": max(0, min(_MAX_MULTI_ROWS, int(d.get("max_multi_rows", 0))))}
            common = (rid, _VERSION, str(d["transport"]), secret, ttl_ms, int(self._lane_config.max_channels),
                      int(d["max_rows"]), fingerprint, rings, linears)
            if d["transport"] == "tcp":
                port = int(d["port"])
                if not 0 < port < 65536 or d.get("alpn", "pgw-lane/1") != "pgw-lane/1":
                    raise ProtocolError("lane ticket names a bad port or protocol")
                if self._tcp_ports is not None and port not in self._tcp_ports:
                    raise ProtocolError(f"lane ticket names port {port}, which is not on the gateway's egress "
                                        "allowlist for the service host (add https://<host>:<port>)")
                return LaneTicket(*common, port=port, **multi)
            return LaneTicket(*common, path=str(d["socket"]), **multi)
        except ProtocolError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            raise ProtocolError(f"lane ticket is malformed ({type(exc).__name__})") from exc

    async def lane_open(self, fingerprint: str, ring: str, *, transports: tuple[str, ...] = ("uds",),
                        lane_versions: tuple[int, ...] = (1,)) -> LaneTicket:
        """A ticket from ``POST /pgw/v1/lane/open`` (one attempt; authenticated like a compute call)."""
        if not set(transports) & set(self._lane_config.transports) or _VERSION not in lane_versions:
            raise ProtocolError("the service lane is lane version 1 over the configured transports only")
        if self._home is None:
            self._home = asyncio.get_running_loop()
        rid = uuid.uuid4().hex
        body = json.dumps({"fingerprint": fingerprint}).encode()
        deadline = time.monotonic() + float(getattr(self._client.config, "request_timeout_s", 60.0))
        result: LaneTicket = await self._client._call(
            "POST", SERVICE_LANE_OPEN_PATH, body=body, request_id=rid, deadline=deadline,
            decode=lambda raw: self._ticket(raw, rid, fingerprint, ring), max_response_bytes=_MAX_TICKET_BODY,
            max_retries=0)
        return result

    def lane_open_sync(self, fingerprint: str, ring: str, *, transports: tuple[str, ...] = ("uds",),
                       lane_versions: tuple[int, ...] = (1,)) -> LaneTicket:
        """Blocking :meth:`lane_open`.

        On the event loop the first :meth:`lane_open` ran on (the gateway's), when it still runs and this is
        another thread: the wrapped client's HTTP pool may be bound to that loop (an injected transport such as
        the gateway's egress filter is), so the client's private sync loop would fail on it (``RuntimeError``:
        bound to a different event loop), and every further lane connection with it."""
        coro = self.lane_open(fingerprint, ring, transports=transports, lane_versions=lane_versions)
        home = self._home
        if home is not None and home.is_running() and not home.is_closed():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not home:
                fut = asyncio.run_coroutine_threadsafe(coro, home)
                try:
                    return fut.result(timeout=float(getattr(self._client.config, "request_timeout_s", 60.0)) + 5.0)
                except BaseException:
                    fut.cancel()
                    raise
        return self._client._run_sync(coro)  # type: ignore[no-any-return]

    def channel(self, ticket: LaneTicket, secret: bytearray) -> Any:
        """``LaneComputeClient`` channel factory: the accepted ticket for its first connection, a fresh
        ticket (same transport and socket path or port, which the client already checked) for every further
        one."""
        key = hashlib.sha256(bytes(secret)).digest()
        with self._used_lock:
            fresh = key not in self._used
            self._used.add(key)
            if len(self._used) > 4096:  # digests only; old ones are never offered again
                self._used = {key}
        if fresh:
            use: bytes | bytearray = secret
        else:
            ring = ticket.rings[0] if ticket.rings else ""
            t = self.lane_open_sync(ticket.fingerprint, ring, transports=(ticket.transport,))
            if (t.transport, t.path, t.port) != (ticket.transport, ticket.path, ticket.port):
                raise ProtocolError("a fresh lane ticket names another socket")
            use = t.ticket
            with self._used_lock:
                self._used.add(hashlib.sha256(bytes(use)).digest())
        cfg = self._lane_config
        timeout = float(cfg.connect_timeout_s)
        native = native_channel_class()
        if native is not None and not hasattr(native, "connect_service_uds"):
            native = None  # an older extension: A3 frames only
        if ticket.transport == "tcp":
            if native is None:
                raise ProtocolError("the TCP service lane needs the native pgw_fast.TripChannel")
            assert ticket.port is not None
            return tune_channel(_connect_tls(native, self._client.config, ticket, use, timeout, int(cfg.spin_us)),
                                cfg)
        assert ticket.path is not None
        if native is not None and cfg.native is not False:
            return tune_channel(native.connect_service_uds(ticket.path, use, timeout, int(ticket.max_rows),
                                                           int(cfg.spin_us)), cfg)
        if cfg.native is True:
            raise ProtocolError("lane native channel required but pgw_fast has no service-lane TripChannel")
        return ServiceLaneChannel(ticket.path, use, timeout, ticket.max_rows)

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._client.aclose()


def _connect_tls(native: Any, config: Any, ticket: LaneTicket, secret: bytes | bytearray, timeout: float,
                 spin_us: int) -> Any:
    """The native TLS channel to the service host of ``config.base_url`` (never a host from the service),
    verified against ``config.ca_file`` as that host, presenting the client certificate when configured.

    Raises:
        ProtocolError: TLS off or unverified, no ``ca_file``, or a password-protected client key (the lane is
            then not used; the wrapped client serves every call).
    """
    base = str(getattr(config, "base_url", ""))
    parts = urlsplit(base)
    if parts.scheme != "https" or not getattr(config, "verify_tls", False):
        raise ProtocolError("the TCP lane is TLS only: needs an https:// service URL with TLS verification on")
    host = parts.hostname
    ca_file = getattr(config, "ca_file", None)
    if not host or ca_file is None:
        raise ProtocolError("the TCP lane needs the service's CA pinned in ca_file")
    cert_file = getattr(config, "client_cert_file", None)
    key_file = getattr(config, "client_key_file", None)
    cert_pem = key_pem = None
    if cert_file is not None and key_file is not None:
        if getattr(config, "client_key_password", None) is not None:
            raise ProtocolError("the TCP lane cannot use a password-protected client key")
        # read per connection: a renewed certificate is presented from the next connection on
        cert_pem, key_pem = Path(cert_file).read_bytes(), bytearray(Path(key_file).read_bytes())
    try:
        return native.connect_service_tls(host, int(ticket.port or 0), host, Path(ca_file).read_bytes(), secret,
                                          timeout, int(ticket.max_rows), spin_us, cert_pem,
                                          bytes(key_pem) if key_pem is not None else None)
    finally:
        if key_pem is not None:
            key_pem[:] = bytes(len(key_pem))


def build_lane_client(client: Any, *, uds_dir: str | None = None, pool_wait_s: float = 0.01,
                      max_channels: int = 4, transports: tuple[str, ...] = ("uds",), spin_us: int | None = None,
                      native: bool | None = False, tcp_ports: Any = None) -> LaneComputeClient:
    """Wrap ``client`` (a connected-or-connecting :class:`~pgw.wire.client.ComputeClient`, or a wrapper of one
    such as :class:`~pgw.wire.sync_stream.SyncStreamClient`) so eligible calls use the service's trip lane,
    everything else (and every lane failure) the wrapped client with the byte-identical request.
    ``transports``: ``("uds",)`` (a service on this host) or ``("tcp",)`` (TLS to the service URL's host).
    ``native``: None = the native channel when the extension has it, False (default, as before) = the Python
    channel on UDS. The TCP lane always uses the native channel (there is no Python TLS channel for A1 frames).
    ``tcp_ports``: the only TCP lane ports accepted from a ticket (the gateway passes the ports its egress
    allowlist allows on the service host); ``None`` = any port (tests and tools without an allowlist)."""
    if spin_us is None:  # PGW_LANE_SPIN_US (default 50): the native channel's busy-poll before blocking
        raw = os.environ.get("PGW_LANE_SPIN_US", "").strip()
        try:
            spin_us = min(100_000, max(0, int(raw))) if raw else 50
        except ValueError as exc:
            raise ValueError(f"PGW_LANE_SPIN_US must be an integer, got {raw!r}") from exc
    # opt-in latency knobs of the native channel (defaults off): PGW_LANE_SPIN_AUTO_US (adaptive spin cap, us)
    # and PGW_LANE_SHM=1 (shared-memory transport of a UDS connection, when the service offers it)
    raw = os.environ.get("PGW_LANE_SPIN_AUTO_US", "").strip()
    try:
        spin_auto_us = min(100_000, max(0, int(raw))) if raw else 0
    except ValueError as exc:
        raise ValueError(f"PGW_LANE_SPIN_AUTO_US must be an integer, got {raw!r}") from exc
    shm = os.environ.get("PGW_LANE_SHM", "0").strip().lower() in ("1", "true", "yes", "on")
    # opt-in (the SDK sets it): several-item requests in one lane-format-2 multi call when the service offers it
    multi = os.environ.get("PGW_LANE_MULTI", "0").strip().lower() in ("1", "true", "yes", "on")
    bad = set(transports) - {"uds", "tcp"}
    if bad or not transports:
        raise ValueError(f"unknown lane transports {sorted(bad)}")
    cfg = LaneConfig(native=native, uds_dir=uds_dir, pool_wait_s=pool_wait_s, max_channels=max_channels,
                     transports=tuple(transports), spin_us=spin_us, spin_auto_us=spin_auto_us, shm=shm, multi=multi)
    opener = ServiceLaneOpener(client, cfg, tcp_ports=tcp_ports)
    return LaneComputeClient(opener, cfg, channel_factory=opener.channel)
