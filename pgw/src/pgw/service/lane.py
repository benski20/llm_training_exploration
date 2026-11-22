"""Service side of the native trip lane (chat-speed plan WP-A1; listener in ``rust/pgw-lane``).

The lane is a second front end next to the WebSocket stream channel for decode-sized calls (1 to
``lane_max_rows`` rows, one item): a fixed binary header of public fields and the same packed
``x_hat`` / ``y_hat`` bytes. It never carries anything the stream channel does not, so the provider
learns nothing new (plan section 3); it only removes framing, msgpack and Python from the trip.

Flow:

1. The gateway calls ``POST /pgw/v1/lane/open`` (same authentication as compute) with a model
   fingerprint and receives a single-use 32-byte ticket (valid :data:`TICKET_TTL_S`), the socket
   path, and the model's linear and ring tables (the header's indices point into them).
2. It connects to the Unix domain socket (only processes of the allowed uids are accepted,
   SO_PEERCRED), sends an ``open`` frame with the ticket, then ``call`` frames. With ``lane = "tcp"``
   the ticket names a TCP port instead (on the service's host address): every connection completes a
   TLS 1.3 handshake first (the service's certificate and key; client certificates verified against
   ``tls.ca_file``, required with ``tls.require_client_cert``), and a ticket issued to a ``lane/open``
   request that presented a client certificate is bound to that certificate's SHA-256, so only a
   connection presenting the same certificate can spend it. Same frames, same checks.
3. Each call is checked (header, table indices, exact body length, deadline cap) before its body
   is read, then admitted against the **shared** admission state (draining, global and per-tenant
   in-flight caps, per-tenant row bucket: one budget with the stream channel) and executed by
   a CUDA graph replayed natively (no Python) when one was captured and passed its self-check,
   otherwise by the regular kernel path through the event loop (:meth:`LaneManager._py_execute`).
4. Refusals ``fallback``, ``unavailable``, ``internal`` and ``busy`` tell the client to resend the
   byte-identical body on the stream channel under the same request id. ``busy`` (the service has
   its maximum of lane connections open) is fatal: it is the only reply on that connection, which
   the service then closes; retry the lane after ``retry_after_ms``. The whole ``open`` frame
   (header and ticket) must arrive within 10 s of connecting.

Fallbacks: no ``pgw_fast`` extension (or one without the lane) -> no lane, ``lane/open`` answers
``unavailable`` and gateways stay on the stream channel. No CUDA, no Triton GEMV, a ring other than
K = 32, a torch without ``CUDAGraph.raw_cuda_graph_exec`` or a failed graph self-check -> the lane
serves through the kernel path instead of graphs. (A self-check mismatch disables graph replay only,
not the lane: the kernel path it falls back to is the reference the check compares against.)

The constants and helpers at the top mirror the Rust wire format (``rust/pgw-lane/src/proto.rs``)
for tests and as the reference for the client package (WP-A3).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import enum
import logging
import os
import secrets
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pgw.core.errors import (
    ConfigError,
    ErrorCode,
    PGWError,
    ProtocolError,
    RateLimitedError,
    ServiceUnavailableError,
)
from pgw.core.native import fast
from pgw.core.packing import pack_ring
from pgw.core.ring import RingParams

__all__ = [
    "LANE_OPEN_PATH",
    "REPLY_HEADER",
    "REQUEST_HEADER",
    "TICKET_TTL_S",
    "LaneManager",
    "LaneStatus",
    "add_lane_routes",
    "encode_call",
    "encode_open",
    "lane_native_available",
    "parse_reply",
]

log = logging.getLogger(__name__)

# ---- wire format mirror (rust/pgw-lane/src/proto.rs) -----------------------------------------
LANE_VERSION = 1
REQUEST_MAGIC = b"PGWL"
REPLY_MAGIC = b"PGWR"
#: ``magic, version, kind, request id, linear, rows, ring, deadline_ms, body_len, reserved``
REQUEST_HEADER = struct.Struct("<4sHH16sIHHIIQ")
#: ``magic, version, status, request id, queue_us, compute_us, retry_after_ms, body_len``
REPLY_HEADER = struct.Struct("<4sHH16sIIII")
KIND_OPEN = 1
KIND_CALL = 2
TICKET_LEN = 32
#: Seconds a ticket stays valid for its ``open`` frame.
TICKET_TTL_S = 30.0
LANE_OPEN_PATH = "/pgw/v1/lane/open"
#: Seconds between drains of the lane's counters into Prometheus (and revocation checks).
DRAIN_INTERVAL_S = 0.5


class LaneStatus(enum.IntEnum):
    """Reply status codes (``Status`` in ``proto.rs``). At or above 16 the service closes the connection."""

    OK = 0
    FALLBACK = 1
    UNAVAILABLE = 2
    RATE_LIMITED = 3
    DEADLINE_EXCEEDED = 4
    INTERNAL = 5
    BAD_HEADER = 16
    BAD_LENGTH = 17
    UNKNOWN_LINEAR = 18
    UNAUTHENTICATED = 19
    PEER_REFUSED = 20
    VERSION = 21
    NOT_OPEN = 22
    BUSY = 23

    @property
    def fatal(self) -> bool:
        """The service closes the connection after this status."""
        return self.value >= 16

    @property
    def fallback(self) -> bool:
        """Resend the byte-identical body on the stream channel."""
        return self in (LaneStatus.FALLBACK, LaneStatus.UNAVAILABLE, LaneStatus.INTERNAL, LaneStatus.BUSY)


assert REQUEST_HEADER.size == 48 and REPLY_HEADER.size == 40


def encode_open(ticket: bytes) -> bytes:
    """An ``open`` frame (header + ticket)."""
    if len(ticket) != TICKET_LEN:
        raise ValueError("ticket must be 32 bytes")
    return REQUEST_HEADER.pack(REQUEST_MAGIC, LANE_VERSION, KIND_OPEN, bytes(16), 0, 0, 0, 0, TICKET_LEN, 0) + ticket


def encode_call(request_id: bytes, linear: int, rows: int, ring: int, body: bytes, deadline_ms: int = 0) -> bytes:
    """A ``call`` frame (header + packed ``x_hat``)."""
    if len(request_id) != 16:
        raise ValueError("request id must be 16 bytes")
    return REQUEST_HEADER.pack(REQUEST_MAGIC, LANE_VERSION, KIND_CALL, request_id, linear, rows, ring, deadline_ms,
                               len(body), 0) + body


@dataclass(frozen=True)
class Reply:
    """A parsed reply header."""

    status: LaneStatus
    request_id: bytes
    queue_us: int
    compute_us: int
    retry_after_ms: int
    body_len: int


def parse_reply(header: bytes) -> Reply:
    """Parse a 40-byte reply header.

    Raises:
        ProtocolError: bad magic, version or status.
    """
    magic, ver, status, rid, q, c, retry, blen = REPLY_HEADER.unpack(header)
    if magic != REPLY_MAGIC or ver != LANE_VERSION:
        raise ProtocolError("not a lane reply")
    try:
        st = LaneStatus(status)
    except ValueError as exc:
        raise ProtocolError(f"unknown lane status {status}") from exc
    return Reply(st, rid, q, c, retry, blen)


def lane_native_available() -> bool:
    """True if the ``pgw_fast`` extension provides the lane."""
    return fast is not None and hasattr(fast, "LaneService")


#: lane status name -> request-metric code (the stream channel's error codes where one fits)
_METRIC_CODE = {
    "ok": "ok", "rate_limited": str(ErrorCode.RATE_LIMITED), "unavailable": str(ErrorCode.UNAVAILABLE),
    "deadline_exceeded": str(ErrorCode.DEADLINE_EXCEEDED), "internal": str(ErrorCode.INTERNAL),
    "bad_header": str(ErrorCode.PROTOCOL), "bad_length": str(ErrorCode.PROTOCOL),
    "unknown_linear": str(ErrorCode.PROTOCOL), "not_open": str(ErrorCode.PROTOCOL),
    "version": str(ErrorCode.VERSION_MISMATCH), "unauthenticated": str(ErrorCode.UNAUTHENTICATED),
    "peer_refused": str(ErrorCode.PERMISSION_DENIED), "fallback": "lane_fallback",
    "busy": str(ErrorCode.UNAVAILABLE),
}


def _peer_uids() -> list[int]:
    """Allowed peer uids: ``PGW_SERVICE_LANE_PEER_UIDS`` (comma-separated), default this process's uid."""
    raw = os.environ.get("PGW_SERVICE_LANE_PEER_UIDS", "").strip()
    if not raw:
        return []
    try:
        return [int(v) for v in raw.split(",") if v.strip()]
    except ValueError as exc:
        raise ConfigError(f"PGW_SERVICE_LANE_PEER_UIDS must be comma-separated uids, got {raw!r}") from exc


def _spin_us() -> int:
    """Hybrid wait: microseconds a graph call spins on the stream before blocking
    (``PGW_SERVICE_LANE_SPIN_US``, default 50; 0 = block at once)."""
    raw = os.environ.get("PGW_SERVICE_LANE_SPIN_US", "").strip()
    try:
        return max(0, int(raw)) if raw else 50
    except ValueError as exc:
        raise ConfigError(f"PGW_SERVICE_LANE_SPIN_US must be an integer, got {raw!r}") from exc


def _read_spin_us() -> int:
    """Before blocking for a connection's next request, poll its socket this many microseconds
    (``PGW_SERVICE_LANE_READ_SPIN_US``, default 0 = block at once). A decode client's next trip comes a
    few milliseconds after the reply, and a blocking read then pays a scheduler wake-up (slow under gVisor);
    spinning costs one core per busy lane connection. Nothing on the wire changes."""
    raw = os.environ.get("PGW_SERVICE_LANE_READ_SPIN_US", "").strip()
    try:
        return min(100_000, max(0, int(raw))) if raw else 0
    except ValueError as exc:
        raise ConfigError(f"PGW_SERVICE_LANE_READ_SPIN_US must be an integer, got {raw!r}") from exc


def _graphs_wanted() -> bool:
    """``PGW_SERVICE_LANE_GRAPHS=0`` turns CUDA-graph replay off (the lane then uses the kernel path)."""
    return os.environ.get("PGW_SERVICE_LANE_GRAPHS", "1").strip().lower() not in ("0", "false", "no", "off")


@dataclass
class _Slot:
    fingerprint: str
    linears: list[str]
    rings: list[RingParams]


@dataclass
class GraphReport:
    """Outcome of the startup capture and equality self-check."""

    captured: int = 0
    checked: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    disabled: str | None = None
    seconds: float = 0.0


class LaneManager:
    """Owns the native lane of one service (see the module docstring).

    Args:
        settings: Service settings (``lane``, ``lane_socket``, ``lane_max_rows``, limits).
        state: The app's :class:`pgw.service.app.ServiceState`.
    """

    def __init__(self, settings: Any, state: Any):
        self.settings = settings
        self.state = state
        self.svc: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._slots: dict[str, int] = {}
        self._tables: dict[int, _Slot] = {}
        self._task: asyncio.Task[None] | None = None
        # tenant -> token id -> identity, for revocation: keyed by token so a tenant's identity set
        # is bounded by its tokens (no eviction window a revoked token could fall out of)
        self._identities: dict[str, dict[str, Any]] = {}
        self._graphs: list[Any] = []  # torch graphs and buffers kept alive while registered
        self.graph_report = GraphReport()
        self.unavailable_reason: str | None = None

    # ---- lifecycle ---------------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Any, state: Any) -> LaneManager | None:
        """A manager when ``settings.lane`` asks for one, else ``None``.

        Raises:
            ConfigError: ``lane = "tcp"`` with a password-protected TLS key (the native listener reads only
                unencrypted PEM keys).
        """
        if settings.lane == "off":
            return None
        if settings.lane == "tcp" and settings.tls.key_password is not None:
            raise ConfigError("lane = 'tcp' needs an unencrypted tls.key_file (the native TLS listener cannot "
                              "use tls.key_password); use lane = 'off' or an unencrypted key")
        return cls(settings, state)

    @property
    def transport(self) -> str:
        """``uds`` or ``tcp`` (TLS)."""
        return "tcp" if self.settings.lane == "tcp" else "uds"

    @property
    def running(self) -> bool:
        """True while the native listener runs."""
        return self.svc is not None

    async def start(self) -> bool:
        """Start the listener, publish the loaded models, attach shared admission to the handler,
        capture graphs (CUDA) and start the metrics drain. False (and a warning) when the native
        lane is unavailable: the service then runs without it."""
        loop = asyncio.get_running_loop()
        ok = await asyncio.to_thread(self._start_sync, loop)
        if ok:
            self._task = loop.create_task(self._drain_loop())
        return ok

    def _start_sync(self, loop: asyncio.AbstractEventLoop) -> bool:
        s = self.settings
        if not lane_native_available():
            self.unavailable_reason = "the pgw_fast extension with the lane is not installed"
            log.warning("lane = %r requested but %s; serving the stream channel only", s.lane, self.unavailable_reason)
            return False
        self.loop = loop
        cap_ms = int(min(s.max_deadline_s or 120.0, 3600.0) * 1000)
        use_graphs = _graphs_wanted() and self._any_cuda()
        listen: dict[str, Any] = {}
        if self.transport == "tcp":
            if not hasattr(fast.LaneService, "port"):
                self.unavailable_reason = "the pgw_fast extension has no TCP+TLS lane (rebuild it)"
                log.warning("lane = 'tcp' requested but %s; serving the stream channel only", self.unavailable_reason)
                return False
            tls = s.tls
            assert tls.cert_file is not None and tls.key_file is not None  # ServiceSettings checks lane = 'tcp'
            listen = {"tls_addr": _socket_addr(s.host, int(s.lane_port or 0)),
                      "tls_cert_pem": tls.cert_file.read_bytes(), "tls_key_pem": tls.key_file.read_bytes(),
                      "tls_client_ca_pem": tls.ca_file.read_bytes() if tls.ca_file is not None else None,
                      "require_client_cert": bool(tls.require_client_cert)}
            path = ""
        else:
            s.lane_socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = str(s.lane_socket)
        self.svc = fast.LaneService(
            path, peer_uids=_peer_uids(), max_rows=int(s.lane_max_rows), deadline_cap_ms=cap_ms,
            max_connections=int(s.max_inflight_requests), max_inflight=int(s.max_inflight_requests),
            tenant_max_inflight=s.tenant_max_inflight, tenant_rows_per_s=s.tenant_rows_per_s,
            tenant_burst_rows=float(s.tenant_burst_rows) if s.tenant_burst_rows else None,
            retry_after_ms=round(s.retry_after_s * 1000), executor=self._py_execute, use_graphs=use_graphs,
            spin_us=_spin_us(), **listen,
            **({"read_spin_us": rs} if (rs := _read_spin_us()) else {}),
        )
        for m in self.state.registry.models():
            self._slot_for(m.fingerprint)
        self.state.handler.lane_admission = self.svc
        if use_graphs:
            self._install_device_locks()
            self.graph_report = capture_lane_graphs(self)
        off = self.graph_report.disabled
        where = f"TLS port {self.svc.port}" if self.transport == "tcp" else str(s.lane_socket)
        log.info("trip lane listening on %s (max rows %d, graphs %d%s)", where, s.lane_max_rows,
                 self.svc.graph_count, f", graphs off: {off}" if off else "")
        return True

    def set_draining(self) -> None:
        """Refuse new lane calls (``unavailable``; clients fall back to the stream channel, which is
        draining too)."""
        if self.svc is not None:
            self.svc.set_draining(True)

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the listener, release graphs and detach from the handler."""
        if self.svc is None:
            return
        svc = self.svc
        svc.set_draining(True)
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: S110 - best effort on shutdown
                pass
        await asyncio.to_thread(svc.stop, timeout)
        await asyncio.to_thread(svc.clear_graphs)
        self.drain_metrics()
        self._graphs.clear()
        if getattr(self.state.handler, "lane_admission", None) is svc:
            self.state.handler.lane_admission = None
        self.svc = None

    # ---- tables and tickets ------------------------------------------------------------------
    def _slot_for(self, fingerprint: str) -> int:
        slot = self._slots.get(fingerprint)
        if slot is not None:
            return slot
        m = self.state.registry.get(fingerprint)
        rings = [RingParams.from_name(r) for r in m.rings]
        linears = [sp.id.key for sp in m.specs]
        slot = len(self._slots)
        self.svc.set_model(slot, [(int(sp.in_features), int(sp.out_features), str(sp.id.kind.value)) for sp in m.specs],
                           [(r.name, int(r.bytes_per_element)) for r in rings])
        self._slots[fingerprint] = slot
        self._tables[slot] = _Slot(fingerprint, linears, rings)
        return slot

    def open_ticket(self, identity: Any, fingerprint: str, *, client_cert_der: bytes | None = None,
                    client_cert_seen: bool = False) -> dict[str, Any]:
        """Issue a ticket for ``identity`` (a :class:`pgw.service.TenantIdentity` or ``None``) and model
        ``fingerprint``: the ``lane/open`` reply. On the TCP lane a ticket asked for with a client certificate
        (``client_cert_der``, the verified certificate of the ``lane/open`` request) is bound to it: only a TLS
        connection presenting the same certificate can spend it.

        Raises:
            ServiceUnavailableError: no lane, draining, or (TCP) a request authenticated with a client certificate
                the service cannot bind the ticket to (``client_cert_seen`` without ``client_cert_der``).
            PGWError: unknown model (``model_not_found``).
        """
        if self.svc is None:
            raise ServiceUnavailableError("the trip lane is not available on this service",
                                          details={"reason": self.unavailable_reason or "not running"})
        if self.svc.draining or self.state.handler.draining:
            raise ServiceUnavailableError("service is draining", details={"retry_after_s": self.settings.retry_after_s})
        tenant = getattr(identity, "tenant_id", None) or "anonymous"
        slot = self._slot_for(fingerprint)
        t = self._tables[slot]
        ticket = secrets.token_bytes(TICKET_LEN)
        bind: bytes | None = None
        if self.transport == "tcp":
            if client_cert_seen and client_cert_der is None:
                raise ServiceUnavailableError("the TCP trip lane cannot bind this client certificate (the "
                                              "certificate itself is not available to the service)",
                                              details={"reason": "client certificate not bindable"})
            bind = client_cert_der
        self.svc.add_ticket(ticket, tenant, slot, TICKET_TTL_S, bind)
        token_id = getattr(identity, "token_id", None)
        if token_id is not None and callable(getattr(identity, "still_valid", None)):
            self._identities.setdefault(tenant, {})[str(token_id)] = identity
        where: dict[str, Any] = ({"port": int(self.svc.port), "alpn": "pgw-lane/1"} if self.transport == "tcp"
                                 else {"socket": str(self.settings.lane_socket)})
        return {
            "version": LANE_VERSION,
            "transport": self.transport,
            **where,
            "ticket": ticket.hex(),
            "ttl_s": TICKET_TTL_S,
            "fingerprint": t.fingerprint,
            "linears": list(t.linears),
            "rings": [r.name for r in t.rings],
            "max_rows": int(self.settings.lane_max_rows),
            "deadline_cap_ms": int(min(self.settings.max_deadline_s or 120.0, 3600.0) * 1000),
        }

    # ---- Python executor (calls without a graph) ---------------------------------------------
    def _py_execute(self, slot: int, linear: int, ring: int, rows: int, body: bytes, tenant: str,
                    remaining_s: float) -> bytes | int:
        """Runs on a lane connection thread (GIL held): the regular kernel path on the event loop."""
        loop = self.loop
        t = self._tables.get(slot)
        if loop is None or t is None or loop.is_closed():
            return int(LaneStatus.FALLBACK)
        handler = self.state.handler
        fp, lin, rp = t.fingerprint, t.linears[linear], t.rings[ring]
        fut: concurrent.futures.Future[bytes] = concurrent.futures.Future()
        task: list[asyncio.Future[bytes]] = []

        def run() -> None:  # on the event loop: inline on an idle device (one hop), else the batcher's queue
            if not fut.set_running_or_notify_cancel():
                return
            try:
                y = handler.lane_compute_inline(fp, lin, rp, rows, body)
            except BaseException as exc:
                fut.set_exception(exc)
                return
            if y is not None:
                fut.set_result(y)
                return
            tk = asyncio.ensure_future(handler.lane_compute(fp, lin, rp, rows, body, tenant))
            task.append(tk)

            def done(f: asyncio.Future[bytes]) -> None:
                if f.cancelled():
                    fut.set_exception(TimeoutError())
                elif f.exception() is not None:
                    fut.set_exception(f.exception())  # type: ignore[arg-type]
                else:
                    fut.set_result(f.result())

            tk.add_done_callback(done)

        loop.call_soon_threadsafe(run)
        try:
            return bytes(fut.result(timeout=remaining_s))
        except TimeoutError:
            if not fut.cancel():  # running: cancel the queued kernel call, if any, on the loop
                loop.call_soon_threadsafe(lambda: [tk.cancel() for tk in task])
            return int(LaneStatus.DEADLINE_EXCEEDED)
        except RateLimitedError:
            return int(LaneStatus.RATE_LIMITED)
        except ServiceUnavailableError:
            return int(LaneStatus.UNAVAILABLE)
        except PGWError as exc:
            if exc.code == ErrorCode.DEADLINE_EXCEEDED:
                return int(LaneStatus.DEADLINE_EXCEEDED)
            log.warning("lane call failed on the kernel path: %s", exc.code)
            return int(LaneStatus.INTERNAL)

    # ---- metrics and revocation --------------------------------------------------------------
    async def _drain_loop(self) -> None:
        while True:
            await asyncio.sleep(DRAIN_INTERVAL_S)
            try:
                self.drain_metrics()
                await self._check_revocations()
            except Exception:  # metrics must never stop the lane
                log.exception("lane metrics drain failed")

    async def _check_revocations(self) -> None:
        """Close the lane connections of tenants whose token was revoked. ``still_valid`` may stat or
        reload the tokens file, so it runs off the event loop (as the stream channel's watcher does)."""
        snapshot = [(t, tid, i) for t, ids in self._identities.items() for tid, i in ids.items()]
        if not snapshot:
            return
        dead = await asyncio.to_thread(lambda: [(t, tid) for t, tid, i in snapshot if not i.still_valid()])
        if not dead or self.svc is None:
            return
        for tenant in sorted({t for t, _ in dead}):
            log.info("revoking lane connections of tenant %s (credentials revoked)", tenant)
            self.svc.revoke_tenant(tenant)
        for tenant, tid in dead:
            ids = self._identities.get(tenant)
            if ids is not None:
                ids.pop(tid, None)
                if not ids:
                    del self._identities[tenant]

    def drain_metrics(self) -> dict[str, Any]:
        """Fold the lane's counters into the service metrics; returns the raw drain (tests)."""
        if self.svc is None:
            return {}
        d: dict[str, Any] = self.svc.drain_stats()
        m = self.state.metrics
        if not m.enabled:
            return d
        for tenant, status, buckets, total, count in d["requests"]:
            code = _METRIC_CODE.get(status, status)
            m._child("pgw_service_requests_total", tenant, code).inc(count)
            _bulk_observe(m._child("pgw_service_request_seconds", tenant), buckets, total, count)
        for tenant, kind, rows in d["rows"]:
            m.rows(tenant, kind, rows)
        for tenant, bin_, bout in d["bytes"]:
            m.bytes_io(tenant, bin_, bout)
        return d

    def status(self) -> dict[str, Any]:
        """Readiness details."""
        if self.svc is None:
            return {"running": False, "reason": self.unavailable_reason}
        where = {"port": self.svc.port} if self.transport == "tcp" else {"socket": self.svc.path}
        return {"running": True, "transport": self.transport, **where, "connections": self.svc.connections(),
                "graphs": self.svc.graph_count, "graphs_disabled": self.svc.graphs_disabled_reason}

    # ---- CUDA --------------------------------------------------------------------------------
    def _any_cuda(self) -> bool:
        return any(str(k.device).startswith("cuda") for m in self.state.registry.models() for k in m.kernels.values())

    def _install_device_locks(self) -> None:  # pragma: no cover - GPU only
        """Make the lane and the Python path share one lock per device (one call per device)."""
        from pgw.service.gpu_exec import device_executor, trip_stream

        for dev in {str(k.device) for m in self.state.registry.models() for k in m.kernels.values()}:
            d = torch.device(dev)
            if d.type == "cuda":
                ex = device_executor(d)
                ex.share_lock(self.svc.device_lock(int(d.index or 0)))


def _socket_addr(host: str, port: int) -> str:
    """``ip:port`` for the native TCP listener (a host name is resolved once; IPv6 in brackets)."""
    import ipaddress
    import socket as _socket

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = ipaddress.ip_address(_socket.getaddrinfo(host, port, type=_socket.SOCK_STREAM)[0][4][0])
    return f"[{ip}]:{port}" if ip.version == 6 else f"{ip}:{port}"


def _bulk_observe(hist: Any, buckets: Sequence[int], total: float, count: int) -> None:
    """Add pre-bucketed observations to a ``prometheus_client`` histogram child (same bucket bounds).

    Uses the child's bucket counters when they are reachable, else observes each bucket's upper
    bound (the sum is then approximate).
    """
    b = getattr(hist, "_buckets", None)
    s = getattr(hist, "_sum", None)
    if isinstance(b, list) and len(b) == len(buckets) and s is not None:
        for v, c in zip(b, buckets, strict=True):
            if c:
                v.inc(c)
        s.inc(total)
        return
    from pgw.service.metrics import _LATENCY_BUCKETS

    bounds = (*_LATENCY_BUCKETS, _LATENCY_BUCKETS[-1])
    for bound, c in zip(bounds, buckets, strict=False):
        for _ in range(min(int(c), 10000)):
            hist.observe(bound)


# ---- CUDA graph capture and self-check (GPU only) -------------------------------------------
def _graph_api_ok() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA"
    if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph_exec"):
        return f"torch {torch.__version__} has no CUDAGraph.raw_cuda_graph_exec"
    why = fast.lane_cuda_driver_status() if fast is not None else "no native extension"
    return None if why is None else str(why)


def _as_ptr(handle: Any) -> int:
    """A CUDA handle as an integer address (torch returns an int or a cuda-python object)."""
    if isinstance(handle, int):
        return handle
    get = getattr(handle, "getPtr", None)
    return int(get()) if callable(get) else int(handle)


def capture_lane_graphs(mgr: LaneManager, *, inject_mismatch: bool = False,
                        rows_list: Sequence[int] | None = None) -> GraphReport:
    """Capture ``[H2D, memset, GEMV, D2H]`` graphs per (linear, K=32 ring, rows) on every CUDA
    device, register them with the lane, and replay each once **through the native serving path**
    against the digit-GEMM path on random and extreme inputs. Any mismatch (or ``inject_mismatch``)
    disables graph replay for the whole lane (calls then use the kernel path); so does any capture
    error. Returns what happened.
    """
    rep = GraphReport()
    t0 = time.perf_counter()
    why = _graph_api_ok()
    if why is not None:
        rep.disabled = why
        mgr.svc.disable_graphs(why)
        log.info("lane CUDA graphs off: %s", why)
        return rep
    rows_list = list(rows_list or range(1, int(mgr.settings.lane_max_rows) + 1))
    try:  # pragma: no cover - GPU only
        by_dev: dict[str, list[tuple[int, int, int, Any, RingParams]]] = {}
        for fp, slot in mgr._slots.items():
            t = mgr._tables[slot]
            model = mgr.state.registry.get(fp)
            for li, name in enumerate(t.linears):
                kern = model.kernel(name)
                for ri, ring in enumerate(t.rings):
                    if ring.k != 32:
                        rep.skipped["ring_not_k32"] = rep.skipped.get("ring_not_k32", 0) + 1
                    elif kern.device.type != "cuda" or not kern.triton_gemv:
                        rep.skipped["no_fused_gemv"] = rep.skipped.get("no_fused_gemv", 0) + 1
                    else:
                        by_dev.setdefault(str(kern.device), []).append((slot, li, ri, kern, ring))
        for dev, entries in by_dev.items():
            _capture_device(mgr, torch.device(dev), entries, rows_list, rep, inject_mismatch)
            if rep.disabled:
                break
    except Exception as exc:  # pragma: no cover - GPU only
        rep.disabled = f"capture failed: {type(exc).__name__}: {str(exc)[:200]}"
    if rep.disabled:
        mgr.svc.disable_graphs(rep.disabled)
        log.error("lane CUDA graphs disabled: %s; lane calls use the kernel path", rep.disabled)
    rep.seconds = time.perf_counter() - t0
    return rep


def _capture_device(mgr: LaneManager, device: torch.device, entries: list[tuple[int, int, int, Any, RingParams]],
                    rows_list: list[int], rep: GraphReport, inject_mismatch: bool) -> None:  # pragma: no cover - GPU
    from pgw.core.triton_gemv import max_rows, ring_gemv
    from pgw.service.gpu_exec import device_executor

    ex = device_executor(device)
    rmax = max(rows_list)
    if rmax > max_rows(device):
        rows_list = [r for r in rows_list if r <= max_rows(device)]
        rmax = max(rows_list, default=0)
        if not rows_list:
            rep.skipped["device_rows"] = len(entries)
            return
    n_max = max(int(k.in_features) for *_, k, _ in entries)
    m_max = max(int(k.out_features) for *_, k, _ in entries)
    with torch.cuda.device(device):
        stream = trip_stream(device)  # PGW_SERVICE_TRIP_STREAM_PRIORITY=high: above maskgen's streams
        pool = torch.cuda.graph_pool_handle()
        h_in = torch.zeros(rmax * n_max, dtype=torch.int32, pin_memory=True)
        h_out = torch.zeros(rmax * m_max, dtype=torch.int32, pin_memory=True)
        d_in = torch.zeros(rmax * n_max, dtype=torch.int32, device=device)
        d_out = torch.zeros(rmax * m_max, dtype=torch.int32, device=device)
        mgr._graphs.extend([h_in, h_out, d_in, d_out, stream])
        rng = np.random.default_rng(0x1A9E)
        for slot, li, ri, kern, ring in entries:
            n, m = int(kern.in_features), int(kern.out_features)
            w = kern._pw._w_dev
            for rows in rows_list:
                xin, hin = d_in[: rows * n].view(rows, n), h_in[: rows * n].view(rows, n)
                yout, hout = d_out[: rows * m].view(rows, m), h_out[: rows * m].view(rows, m)

                def body(xin: torch.Tensor = xin, hin: torch.Tensor = hin, yout: torch.Tensor = yout,
                         hout: torch.Tensor = hout, w: torch.Tensor = w, n: int = n, m: int = m) -> None:
                    xin.copy_(hin, non_blocking=True)
                    ring_gemv(xin, w, 32, n=n, m=m, out=yout)
                    hout.copy_(yout, non_blocking=True)

                with ex.lock:
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        body()  # warm-up: compiles the kernel outside the capture
                    stream.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool, stream=stream, capture_error_mode="thread_local"):
                        body()
                    exec_ptr = _as_ptr(g.raw_cuda_graph_exec())
                # the graph reads ``w`` by address: keep the weight tensor alive with the graph, so an
                # unload or hot reload cannot free it under a registered graph
                mgr._graphs.extend([g, w])
                mgr.svc.register_graph(slot, li, ri, rows, int(device.index or 0), exec_ptr, int(stream.cuda_stream),
                                       int(h_in.data_ptr()), rows * n * 4, int(h_out.data_ptr()), rows * m * 4)
                rep.captured += 1
                # equality self-check through the serving path: random, all-ones and zero inputs
                for case in range(3):
                    if case == 0:
                        x = torch.from_numpy(rng.integers(0, 1 << 32, size=(rows, n), dtype=np.int64))
                    else:
                        x = torch.full((rows, n), 0 if case == 2 else (1 << 32) - 1, dtype=torch.int64)
                    got = bytes(mgr.svc.run_graph(slot, li, ri, rows, pack_ring(x, ring)))
                    ref = ex.run(x, 32, w, kern._pw._rowsum(), m, graphs=None)
                    want = pack_ring(ref, ring)
                    if inject_mismatch:
                        want = bytes([want[0] ^ 1]) + want[1:]
                    if got != want:
                        rep.disabled = (f"graph replay differs from the digit-GEMM path ({kern.linear}, rows={rows}, "
                                        f"case {case})")
                        return
                rep.checked += 1


# ---- POST /pgw/v1/lane/open ------------------------------------------------------------------
def add_lane_routes(app: Any, state: Any) -> None:
    """Mount ``POST /pgw/v1/lane/open`` (JSON ``{"fingerprint": ...}`` -> ticket and tables).

    Authentication and admission prechecks are the compute path's (``PGW-Protocol`` header, bearer
    token / client certificate). Without a running lane it answers ``unavailable`` (503).
    """
    import json

    from starlette.requests import Request
    from starlette.responses import Response

    from pgw.wire.messages import PROTOCOL_HEADER
    from pgw.wire.server import (
        _client_cert_subject,
        _maybe_await,
        check_protocol_header,
        error_headers,
        error_response_bytes,
    )

    handler = state.handler

    async def lane_open(request: Request) -> Response:
        try:
            check_protocol_header(request.headers.get(PROTOCOL_HEADER))
            identity = await _maybe_await(handler.authenticate(request.headers.get("authorization"),
                                                               _client_cert_subject(request.scope)))
            handler.precheck()
            raw = await request.body()
            if len(raw) > 4096:
                raise ProtocolError("lane/open body too large")
            try:
                req = json.loads(raw or b"{}")
            except ValueError as exc:
                raise ProtocolError("lane/open body must be JSON") from exc
            fp = req.get("fingerprint") if isinstance(req, dict) else None
            if not isinstance(fp, str):
                raise ProtocolError("lane/open needs a model fingerprint")
            mgr: LaneManager | None = getattr(state, "lane", None)
            if mgr is None:
                raise ServiceUnavailableError("the trip lane is not enabled on this service",
                                              details={"reason": "lane = off"})
            from pgw.service.tls_peer import peer_cert_der

            subject = _client_cert_subject(request.scope)
            out = mgr.open_ticket(identity, fp, client_cert_der=peer_cert_der(request.scope),
                                  client_cert_seen=subject is not None)
            return Response(content=json.dumps(out), media_type="application/json",
                            headers={"Cache-Control": "no-store"})
        except Exception as exc:
            if not isinstance(exc, PGWError):
                log.error("lane/open internal error", exc_info=exc)
            status, body = error_response_bytes(exc, None, json_body=True)
            return Response(content=body, status_code=status, media_type="application/json",
                            headers=error_headers(exc))

    app.add_route(LANE_OPEN_PATH, lane_open, methods=["POST"], include_in_schema=False)

