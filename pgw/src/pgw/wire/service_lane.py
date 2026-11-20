"""Bridge from the gateway's trip-lane client (:mod:`pgw.wire.lane`, WP-A3) to the service's lane
(:mod:`pgw.service.lane` and ``rust/pgw-lane``, WP-A1).

The two work packages were built against different framings of the same idea:

* the client package speaks lane format "A3" (``POST /pgw/v1/lane_open`` in msgpack, a ``PGWH`` hello
  and ``PGWT`` / ``PGWR`` 36-byte headers; its native channel is ``pgw_fast.TripChannel``);
* the service that ships (``pgw service serve`` with ``lane = "uds"``) speaks lane format "A1":
  ``POST /pgw/v1/lane/open`` in JSON, then ``PGWL`` 48-byte request headers and 40-byte ``PGWR`` reply
  headers over a Unix domain socket (``rust/pgw-lane/src/proto.rs``).

This module lets :class:`~pgw.wire.lane.LaneComputeClient` (pool, cool-down, byte-identical fallback,
statistics) drive the A1 service lane: :class:`ServiceLaneOpener` wraps a
:class:`~pgw.wire.client.ComputeClient` and answers ``lane_open`` from the A1 JSON route, and
:class:`ServiceLaneChannel` is a ``TripChannelLike`` speaking A1 frames. Privacy is unchanged: an A1
request header carries the same public fields as the stream channel's request (request id, linear, ring,
rows, deadline, body length) and the body is the same packed ``x_hat``; nothing else is sent.

A1 tickets are single-use (one connection each, valid ``ttl_s`` to open it), so every further connection
of the pool asks for a fresh ticket. A1 statuses are mapped onto the client's
:class:`~pgw.wire.lane.Status`: per-call refusals (``fallback``, ``unavailable``, ``rate_limited``,
``internal``) become a non-closing ``BUSY`` (that call goes to the stream channel), ``deadline_exceeded``
becomes ``DEADLINE``, and every status after which the A1 service closes the connection (16 and above,
``busy`` = 23 included) becomes a closing status, so the client drops the lane and cools down.

``build_lane_client`` is what the gateway uses (``GatewayModelConfig.lane``). Only the UDS transport is
built on the service side; the gateway's lane client accepts nothing else.
"""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import threading
import time
import uuid
from typing import Any

from pgw.core.errors import ProtocolError
from pgw.wire.lane import LaneComputeClient, LaneConfig, LaneRefusedError, Status
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

    def __init__(self, client: Any, lane_config: LaneConfig):
        self._client = client
        self._lane_config = lane_config
        self._used: set[bytes] = set()
        self._used_lock = threading.Lock()

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
            if not isinstance(d, dict) or int(d.get("version", 0)) != _VERSION or d.get("transport") != "uds":
                raise ProtocolError("unsupported lane ticket")
            secret = bytes.fromhex(str(d["ticket"]))
            if len(secret) != _TICKET_LEN or d.get("fingerprint") != fingerprint:
                raise ProtocolError("lane ticket is malformed or for another model")
            dims = self._dims()
            linears = tuple(LinearInfo(str(i), *dims[str(i)]) for i in d["linears"])
            rings = tuple(str(r) for r in d["rings"])
            ttl_ms = max(1, int(float(d.get("ttl_s", 30.0)) * 1000))
            return LaneTicket(rid, _VERSION, "uds", secret, ttl_ms, int(self._lane_config.max_channels),
                              int(d["max_rows"]), fingerprint, rings, linears, path=str(d["socket"]))
        except ProtocolError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            raise ProtocolError(f"lane ticket is malformed ({type(exc).__name__})") from exc

    async def lane_open(self, fingerprint: str, ring: str, *, transports: tuple[str, ...] = ("uds",),
                        lane_versions: tuple[int, ...] = (1,)) -> LaneTicket:
        """A ticket from ``POST /pgw/v1/lane/open`` (one attempt; authenticated like a compute call)."""
        if "uds" not in transports or _VERSION not in lane_versions:
            raise ProtocolError("the service lane is UDS lane version 1 only")
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
        """Blocking :meth:`lane_open`."""
        return self._client._run_sync(  # type: ignore[no-any-return]
            self.lane_open(fingerprint, ring, transports=transports, lane_versions=lane_versions))

    def channel(self, ticket: LaneTicket, secret: bytearray) -> ServiceLaneChannel:
        """``LaneComputeClient`` channel factory: the accepted ticket for its first connection, a fresh
        ticket (same socket path, which the client already checked) for every further one."""
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
            t = self.lane_open_sync(ticket.fingerprint, ring)
            if t.path != ticket.path:
                raise ProtocolError("a fresh lane ticket names another socket")
            use = t.ticket
            with self._used_lock:
                self._used.add(hashlib.sha256(bytes(use)).digest())
        assert ticket.path is not None
        return ServiceLaneChannel(ticket.path, use, float(self._lane_config.connect_timeout_s), ticket.max_rows)

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._client.aclose()


def build_lane_client(client: Any, *, uds_dir: str | None = None, pool_wait_s: float = 0.01,
                      max_channels: int = 4) -> LaneComputeClient:
    """Wrap ``client`` (a connected-or-connecting :class:`~pgw.wire.client.ComputeClient`) so eligible calls
    use the service's UDS trip lane, everything else (and every lane failure) the stream channel with the
    byte-identical request. Only the ``uds`` transport is accepted (the TCP+TLS lane is not built)."""
    cfg = LaneConfig(native=False, uds_dir=uds_dir, pool_wait_s=pool_wait_s, max_channels=max_channels,
                     transports=("uds",))
    opener = ServiceLaneOpener(client, cfg)
    return LaneComputeClient(opener, cfg, channel_factory=opener.channel)
