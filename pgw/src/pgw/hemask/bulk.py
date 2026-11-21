"""Bulk maskgen channel: the same ``/pgw/v1/maskgen`` messages over threaded TLS sockets.

Why: one encrypted-mask slot moves about 2.5 MB through ``POST /pgw/v1/maskgen`` for Qwen2.5-0.5B and
about 13 MB for 7B (``n·bb`` bits up, ``m·yb`` bits down per linear, every layer). On the HTTP path
those bytes go through one asyncio event loop on each side (httpx/h11 on the customer, uvicorn on the
provider), where TLS, HTTP framing and Python chunk handling cap a connection at roughly 200-350 MB/s
(``benchmarks/maskgen_wire_bench.py``). At 512 slots per pass that is about 9 s of a 15 s factory pass
on a T4 container, while the provider's GPU work is well under a second. A blocking TLS socket on its
own thread moves the same bytes several times faster: OpenSSL runs with the GIL released, and no event
loop is involved.

What: a separate listener of the compute service (``maskgen_bulk_port``, off by default) that accepts
TLS connections (the service's certificate and client-certificate settings; plain TCP only where the
main listener is plain too), one thread per connection, and carries maskgen requests in a fixed binary
frame::

    request: "PGWM" | u16 version (1) | u16 flags (0) | u32 meta_len | u64 body_len | meta | body
    reply:   "PGWm" | u16 version (1) | u16 status    | u32 0        | u64 body_len | body

little-endian. ``meta`` is a msgpack map ``{"id": str, "auth": str?}`` (``auth`` is the HTTP
``Authorization`` value). ``body`` is byte for byte the msgpack maskgen request of the HTTP route
(:func:`pgw.hemask.wire.encode_maskgen_request`); the reply body is the HTTP reply body (status 200:
:func:`pgw.hemask.wire.encode_maskgen_response`; otherwise the msgpack error body with the HTTP status
of the error code).

Privacy: nothing new reaches the provider. The request bytes are the ones the HTTP route receives,
over TLS with the same server certificate; authentication (token, client certificate, the
``require_client_cert_for_compute`` rule), admission, draining, per-tenant limits, wire versions and the
GPU path are the handler's own (:meth:`pgw.service.handler.ServiceHandler.maskgen`). Request sizes and
counts are those of the HTTP path (when a client splits a multi-group request, it splits it into
fixed 256-column groups, a function of the configured batch only).
"""

from __future__ import annotations

import contextlib
import logging
import queue
import socket
import ssl
import struct
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

import msgpack

from pgw.core.errors import ErrorCode, PGWError, ProtocolError, ServiceUnavailableError

__all__ = [
    "BULK_VERSION",
    "MAX_META_BYTES",
    "REPLY_HEADER",
    "REPLY_MAGIC",
    "REQUEST_HEADER",
    "REQUEST_MAGIC",
    "BulkMaskgenClient",
    "recv_exact",
    "send_parts",
]

log = logging.getLogger(__name__)

T = TypeVar("T")

REQUEST_MAGIC = b"PGWM"
REPLY_MAGIC = b"PGWm"
BULK_VERSION = 1
#: ``magic | version | flags/status | meta_len/0 | body_len`` (20 bytes).
REQUEST_HEADER = struct.Struct("<4sHHIQ")
REPLY_HEADER = REQUEST_HEADER
MAX_META_BYTES = 16 << 10
#: Largest reply the client accepts (a 7B gate_up group of 16 x 256 columns at 64 bits is 77.6 MB).
MAX_REPLY_BYTES = 1 << 30
#: Socket writes at most this many bytes per call (keeps every SSL_write bounded).
_SEND_CHUNK = 4 << 20


def recv_exact(sock: Any, n: int, into: bytearray | None = None) -> bytearray:
    """Exactly ``n`` bytes from ``sock`` (raises :class:`ConnectionError` on EOF)."""
    buf = into if into is not None else bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], min(n - got, 1 << 22))
        if not k:
            raise ConnectionError("connection closed")
        got += k
    return buf


def send_parts(sock: Any, parts: list[Any]) -> None:
    """Write every buffer of ``parts`` in order (no concatenation)."""
    for p in parts:
        mv = memoryview(p).cast("B")
        for o in range(0, len(mv), _SEND_CHUNK):
            sock.sendall(mv[o:o + _SEND_CHUNK])


def _nodelay(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class BulkMaskgenClient:
    """Customer side of the bulk channel: a pool of blocking (TLS) connections, any number of threads.

    Each call takes an idle connection (or opens one, up to ``max_conns``), writes one request frame and
    reads one reply frame on the calling thread. A transport failure closes that connection and raises
    :class:`ServiceUnavailableError` (``details["bulk_transport"]``); callers fall back to HTTP with the
    same body. After a failure to connect, the channel is skipped for ``retry_s``.

    Args:
        host, port: The service's bulk listener.
        ssl_context: Client TLS context (``None``: plain TCP; only for loopback services without TLS).
        server_hostname: Name checked against the service certificate.
        authorization: ``Authorization`` value sent with every request (e.g. ``"Bearer <token>"``), or a
            callable returning it.
    """

    def __init__(self, host: str, port: int, *, ssl_context: ssl.SSLContext | None, server_hostname: str | None,
                 authorization: str | Callable[[], str | None] | None = None, timeout_s: float = 600.0,
                 connect_timeout_s: float = 10.0, max_conns: int = 8, retry_s: float = 30.0):
        self.host = host
        self.port = int(port)
        self.ssl_context = ssl_context
        self.server_hostname = server_hostname
        self._auth = authorization
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.max_conns = max(1, int(max_conns))
        self.retry_s = retry_s
        self._idle: queue.LifoQueue[Any] = queue.LifoQueue()
        self._open = 0
        self._lock = threading.Lock()
        self._down_until = 0.0
        self._closed = False
        self.calls = 0
        self.failures = 0

    def usable(self) -> bool:
        """False while the channel is closed or backing off after a failure to connect."""
        return not self._closed and time.monotonic() >= self._down_until

    def _connect(self) -> Any:
        try:
            raw = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        except OSError as exc:
            self._down_until = time.monotonic() + self.retry_s
            raise ServiceUnavailableError(f"maskgen bulk channel {self.host}:{self.port} unreachable "
                                          f"({type(exc).__name__})", details={"bulk_transport": True}) from exc
        _nodelay(raw)
        try:
            sock: Any = raw
            if self.ssl_context is not None:
                sock = self.ssl_context.wrap_socket(raw, server_hostname=self.server_hostname)
            sock.settimeout(self.timeout_s)
            return sock
        except (OSError, ssl.SSLError) as exc:
            raw.close()
            self._down_until = time.monotonic() + self.retry_s
            raise ServiceUnavailableError(f"maskgen bulk channel TLS handshake failed ({type(exc).__name__})",
                                          details={"bulk_transport": True}) from exc

    def _take(self) -> Any:
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._open >= self.max_conns:
                wait = True
            else:
                self._open += 1
                wait = False
        if wait:
            # poll: a connection dropped by another thread frees a slot without reaching the idle queue
            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    return self._idle.get(timeout=0.05)
                except queue.Empty:
                    pass
                with self._lock:
                    if self._open < self.max_conns:
                        self._open += 1
                        break
                if self._closed or time.monotonic() >= deadline:
                    raise ServiceUnavailableError("no maskgen bulk connection became free",
                                                  details={"bulk_transport": True})
        try:
            return self._connect()
        except BaseException:
            with self._lock:
                self._open -= 1
            raise

    def _drop(self, sock: Any) -> None:
        with contextlib.suppress(OSError):
            sock.close()
        with self._lock:
            self._open -= 1

    def _authorization(self) -> str | None:
        a = self._auth
        return a() if callable(a) else a

    def call(self, body: bytes | bytearray | memoryview, request_id: str, decode: Callable[[bytearray], T]) -> T:
        """Send one maskgen request body, return ``decode(reply_body)``.

        Raises:
            ServiceUnavailableError: transport failure (``details["bulk_transport"]``; resend over HTTP).
            PGWError: the service's error reply (the same error the HTTP route would return).
        """
        if self._closed:
            raise ServiceUnavailableError("maskgen bulk channel is closed", details={"bulk_transport": True})
        meta: dict[str, Any] = {"id": request_id}
        auth = self._authorization()
        if auth:
            meta["auth"] = auth
        mb = msgpack.packb(meta, use_bin_type=True)
        blen = len(memoryview(body).cast("B"))
        head = REQUEST_HEADER.pack(REQUEST_MAGIC, BULK_VERSION, 0, len(mb), blen)
        sock = self._take()
        ok = False
        try:
            try:
                try:
                    send_parts(sock, [head + mb, body])
                except (OSError, ssl.SSLError):
                    # the service may refuse (authentication, admission, size) before it reads the body and close
                    # the connection; its error reply is then already on the way: read it if it is there
                    sock.settimeout(min(self.timeout_s, 5.0))
                magic, ver, status, _r, n = REPLY_HEADER.unpack(recv_exact(sock, REPLY_HEADER.size))
                if magic != REPLY_MAGIC or ver != BULK_VERSION or n > MAX_REPLY_BYTES:
                    raise ProtocolError("maskgen bulk channel: malformed reply frame")
                data = recv_exact(sock, n)
            except (OSError, ssl.SSLError, ConnectionError) as exc:
                self.failures += 1
                raise ServiceUnavailableError(f"maskgen bulk channel failed ({type(exc).__name__}: {exc})",
                                              details={"bulk_transport": True}) from exc
            # after an error reply the service may close the connection (e.g. a refusal before it read the body)
            ok = status == 200
        finally:
            if ok:
                self._idle.put(sock)
            else:
                self._drop(sock)
        self.calls += 1
        if status != 200:
            raise _error_from(status, data)
        return decode(data)

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                sock = self._idle.get_nowait()
            except queue.Empty:
                break
            self._drop(sock)


def _error_from(status: int, data: bytes | bytearray) -> PGWError:
    from pgw.wire.codec import decode_message
    from pgw.wire.messages import ErrorResponse

    try:
        return decode_message(data, ErrorResponse, max_bytes=64 << 10).to_exception()
    except PGWError:
        code = ErrorCode.INTERNAL if status == 500 else (ErrorCode.UNAVAILABLE if status >= 500 else
                                                         ErrorCode.PROTOCOL)
        return PGWError(f"maskgen bulk channel: service returned status {status}", code=code,
                        details={"http_status": status})
