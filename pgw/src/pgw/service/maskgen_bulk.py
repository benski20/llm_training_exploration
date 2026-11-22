"""Service side of the bulk maskgen channel (:mod:`pgw.hemask.bulk`): a threaded TLS listener.

Started by :func:`pgw.service.server.serve_async` when ``maskgen_bulk_port`` is set (off by default).
One thread per connection reads a request frame, decodes the maskgen request, runs authentication,
admission and :meth:`ServiceHandler.maskgen` on the service's event loop (so tenant limits, draining,
in-flight accounting, the per-device maskgen slots and the decode yield are the HTTP route's), and
writes the reply frame straight from the result buffers (pinned device-copy buffers on CUDA; no
msgpack copy of ``y``). TLS reads and writes run on the connection's thread with the GIL released.

Limits: at most ``max_conns`` connections (others get one ``unavailable`` reply and are closed), a
handshake and first-byte timeout, an idle timeout between requests, ``max_request_bytes`` per body
(checked before the body is read), :data:`pgw.hemask.bulk.MAX_META_BYTES` per meta map, and the
service's shared request-body budget (``body_budget``, the compute and HTTP maskgen routes' own
:class:`pgw.wire.server.ByteBudget`): a body is reserved against it before it is read and released once
its reply is written; a refused body gets ``unavailable`` (Retry-After) and the connection is closed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import socket
import ssl
import threading
from typing import Any

import msgpack

from pgw.core.errors import ErrorCode, PayloadTooLargeError, PGWError, ProtocolError, ServiceUnavailableError
from pgw.hemask.bulk import (
    BULK_VERSION,
    MAX_META_BYTES,
    REPLY_HEADER,
    REPLY_MAGIC,
    REQUEST_HEADER,
    REQUEST_MAGIC,
    recv_exact,
    send_parts,
)
from pgw.hemask.server import DEFAULT_MAX_MASKGEN_BYTES
from pgw.hemask.wire import decode_maskgen_request, encode_maskgen_response_parts

__all__ = ["MaskgenBulkServer"]

log = logging.getLogger(__name__)

#: TLS handshake plus the first request header must arrive within this time.
HANDSHAKE_TIMEOUT_S = 10.0


def _served_versions(handler: Any) -> tuple[int, ...]:
    return tuple(sorted(set(getattr(handler, "maskgen_versions", (1,))) | {1}))


class MaskgenBulkServer:
    """Threaded listener for maskgen request frames (see module docstring).

    Args:
        handler: The service handler (``authenticate``, ``precheck``, ``maskgen``).
        loop: The service's running event loop (handler coroutines run there).
        ssl_context: Server TLS context (``None``: plain TCP, loopback only; enforced by the caller).
        host, port: Bind address (``port=0``: any free port, see :attr:`port`).
        verify_peer: Publish the verified client certificate subject to ``authenticate`` (mTLS).
    """

    def __init__(self, handler: Any, loop: asyncio.AbstractEventLoop, ssl_context: ssl.SSLContext | None, *,
                 host: str, port: int, verify_peer: bool = False, max_conns: int = 16,
                 max_request_bytes: int = DEFAULT_MAX_MASKGEN_BYTES, idle_timeout_s: float = 300.0,
                 io_timeout_s: float = 120.0, request_timeout_s: float = 600.0, body_budget: Any = None):
        self.handler = handler
        self.body_budget = body_budget
        self.loop = loop
        self.ssl_context = ssl_context
        self.verify_peer = verify_peer
        self.max_conns = max(1, int(max_conns))
        self.max_request_bytes = int(max_request_bytes)
        self.idle_timeout_s = idle_timeout_s
        self.io_timeout_s = io_timeout_s
        self.request_timeout_s = request_timeout_s
        fam = socket.AF_INET6 if ":" in host else socket.AF_INET
        self._sock = socket.socket(fam, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(64)
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._conns: set[Any] = set()
        self._lock = threading.Lock()
        self._refusers = threading.BoundedSemaphore(4)  # threads answering over-cap connections at most
        self._thread: threading.Thread | None = None
        self.requests = 0
        self.refused = 0
        self.refused_budget = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._accept_loop, name="pgw-maskgen-bulk-accept", daemon=True)
        self._thread.start()
        log.info("maskgen bulk channel listening on port %d (tls=%s, max %d connections)", self.port,
                 self.ssl_context is not None, self.max_conns)

    def close(self) -> None:
        """Stop accepting; open connections are shut down (a request in progress finishes on the handler)."""
        self._stop.set()
        with contextlib.suppress(OSError):
            self._sock.close()
        with self._lock:
            conns = list(self._conns)
        for c in conns:
            with contextlib.suppress(OSError):
                c.shutdown(socket.SHUT_RDWR)
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ---- connections -----------------------------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw, _addr = self._sock.accept()
            except OSError:
                if self._stop.is_set():
                    return
                continue
            with self._lock:
                busy = len(self._conns) >= self.max_conns
                if not busy:
                    self._conns.add(raw)
            if busy:
                self.refused += 1
                if self._refusers.acquire(blocking=False):  # bounded: excess connections are just closed
                    threading.Thread(target=self._refuse, args=(raw,), daemon=True).start()
                else:
                    with contextlib.suppress(OSError):
                        raw.close()
                continue
            threading.Thread(target=self._serve_conn, args=(raw,), name="pgw-maskgen-bulk", daemon=True).start()

    def _wrap(self, raw: socket.socket) -> Any:
        raw.settimeout(HANDSHAKE_TIMEOUT_S)
        with contextlib.suppress(OSError):
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.ssl_context is None:
            return raw
        return self.ssl_context.wrap_socket(raw, server_side=True)

    def _refuse(self, raw: socket.socket) -> None:
        try:
            sock = self._wrap(raw)
            exc = ServiceUnavailableError("too many maskgen bulk connections", details={"retry_after_s": 1.0})
            self._reply_error(sock, exc, None)
        except (OSError, ssl.SSLError) as e:  # best effort: the client may already be gone
            log.debug("maskgen bulk refusal not delivered: %s", e)
        finally:
            with contextlib.suppress(OSError):
                raw.close()
            self._refusers.release()

    def _subject(self, sock: Any) -> str | None:
        if not self.verify_peer or not isinstance(sock, ssl.SSLSocket):
            return None
        der = sock.getpeercert(binary_form=True)
        if not der:
            return None
        from pgw.service.tls_peer import subject_from_der

        return subject_from_der(der)

    def _serve_conn(self, raw: socket.socket) -> None:
        sock: Any = raw
        try:
            sock = self._wrap(raw)
            subject = self._subject(sock)
            first = True
            while not self._stop.is_set():
                sock.settimeout(HANDSHAKE_TIMEOUT_S if first else self.idle_timeout_s)
                try:
                    head = recv_exact(sock, REQUEST_HEADER.size)
                except (ConnectionError, TimeoutError, OSError):
                    return
                first = False
                sock.settimeout(self.io_timeout_s)
                if not self._one(sock, head, subject):
                    return
        except (OSError, ssl.SSLError, ConnectionError) as exc:
            log.debug("maskgen bulk connection closed: %s: %s", type(exc).__name__, exc)
        finally:
            with self._lock:
                self._conns.discard(raw)
            with contextlib.suppress(OSError):
                sock.close()

    def _one(self, sock: Any, head: bytearray, subject: str | None) -> bool:
        """Serve one request frame; False closes the connection (framing errors, oversized bodies)."""
        magic, ver, flags, meta_len, body_len = REQUEST_HEADER.unpack(head)
        if magic != REQUEST_MAGIC or ver != BULK_VERSION or flags != 0 or meta_len > MAX_META_BYTES:
            self._reply_error(sock, ProtocolError("maskgen bulk channel: malformed request frame"), None)
            return False
        try:
            meta = msgpack.unpackb(recv_exact(sock, meta_len), raw=False, strict_map_key=True, max_bin_len=0)
        except Exception:
            self._reply_error(sock, ProtocolError("maskgen bulk channel: malformed meta"), None)
            return False
        if not isinstance(meta, dict):
            self._reply_error(sock, ProtocolError("maskgen bulk channel: malformed meta"), None)
            return False
        rid = meta.get("id") if isinstance(meta.get("id"), str) and len(meta["id"]) <= 128 else None
        auth = meta.get("auth") if isinstance(meta.get("auth"), str) else None
        if body_len > self.max_request_bytes:
            self._reply_error(sock, PayloadTooLargeError(
                f"request body of {body_len} bytes exceeds the limit of {self.max_request_bytes}",
                details={"bytes": body_len, "max_bytes": self.max_request_bytes}), rid)
            return False
        try:
            identity = self._run(self._admit(auth, subject), self.io_timeout_s)
        except Exception as exc:
            self._reply_error(sock, exc, rid)
            return False  # the body was not read: the stream is out of sync
        budget = self.body_budget
        if budget is not None and not budget.try_acquire(body_len):
            from pgw.wire.server import _budget_exhausted

            self.refused_budget += 1
            self._reply_error(sock, _budget_exhausted(budget, body_len), rid)
            return False  # the body was not read: the stream is out of sync
        try:
            body = recv_exact(sock, body_len)
            try:
                versions = _served_versions(self.handler)
                req = decode_maskgen_request(body, versions)
                del body
                rid = req.request_id
                resp = self._run(self.handler.maskgen(req, identity), self.request_timeout_s)
                parts = encode_maskgen_response_parts(dataclasses.replace(resp, version=req.version,
                                                                          versions=versions))
            except Exception as exc:
                if not (isinstance(exc, PGWError) and exc.code != ErrorCode.INTERNAL):
                    log.error("pgw maskgen (bulk) internal error", extra={"request_id": rid}, exc_info=exc)
                self._reply_error(sock, exc, rid)
                return True
            total = sum(len(memoryview(p).cast("B")) for p in parts)
            send_parts(sock, [REPLY_HEADER.pack(REPLY_MAGIC, BULK_VERSION, 200, 0, total), *parts])
        finally:
            if budget is not None:
                budget.release(body_len)
        self.requests += 1
        return True

    async def _admit(self, auth: str | None, subject: str | None) -> Any:
        h = self.handler
        identity = None
        if callable(getattr(h, "authenticate", None)):
            identity = h.authenticate(auth, subject)
            if asyncio.iscoroutine(identity):
                identity = await identity
        pre = getattr(h, "precheck", None)
        if callable(pre):
            r = pre()
            if asyncio.iscoroutine(r):
                await r
        return identity

    def _run(self, coro: Any, timeout: float) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout)
        except TimeoutError:
            fut.cancel()
            raise PGWError("maskgen request timed out", code=ErrorCode.DEADLINE_EXCEEDED) from None

    def _reply_error(self, sock: Any, exc: BaseException, rid: str | None) -> None:
        from pgw.wire.server import error_response_bytes

        status, body = error_response_bytes(exc, rid)
        if isinstance(exc, PGWError) and exc.code != ErrorCode.INTERNAL:
            log.warning("pgw maskgen (bulk) rejected", extra={"request_id": rid, "code": str(exc.code)})
        with contextlib.suppress(OSError, ssl.SSLError):
            send_parts(sock, [REPLY_HEADER.pack(REPLY_MAGIC, BULK_VERSION, status, 0, len(body)) + body])
