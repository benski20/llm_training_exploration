"""A TCP proxy that adds a fixed one-way delay to every chunk, for per-trip latency checks on loopback.

``tc netem`` needs root and a kernel module that a shared box (or a container) may not have; this proxy
emulates a link with a fixed latency instead: every chunk read from one side is written to the other
``one_way_s`` seconds after it was read, in order, with ``TCP_NODELAY`` on all four sockets (so the
proxy itself never batches). A round trip through it therefore takes ``2 * one_way_s`` plus the proxy's
own hop (a few tens of microseconds; measure it with ``one_way_s = 0``). Threads, not asyncio: an
asyncio timer is rounded to the event loop's millisecond poll granularity.

Bytes pass through unchanged, so TLS runs end to end through it.

    proxy = DelayProxy(("127.0.0.1", service_port), one_way_s=0.0005)
    port = proxy.start()          # connect to 127.0.0.1:port
    ...
    proxy.stop()
"""

from __future__ import annotations

import collections
import socket
import threading
import time

__all__ = ["DelayProxy"]


def _nodelay(s: socket.socket) -> None:
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class DelayProxy:
    """Listens on ``127.0.0.1:<ephemeral>`` and forwards every connection to ``upstream``."""

    def __init__(self, upstream: tuple[str, int], one_way_s: float = 0.0):
        self.upstream = upstream
        self.one_way_s = max(0.0, float(one_way_s))
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self._socks: list[socket.socket] = []

    def start(self) -> int:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(16)
        srv.settimeout(0.2)
        self._srv = srv
        threading.Thread(target=self._accept, daemon=True).start()
        return int(srv.getsockname()[1])

    def stop(self) -> None:
        self._stop.set()
        for s in [*self._socks, *([self._srv] if self._srv else [])]:
            try:
                s.close()
            except OSError:
                pass

    def _accept(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                down, _ = self._srv.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                up = socket.create_connection(self.upstream)
            except OSError:
                down.close()
                continue
            down.settimeout(None)
            for s in (down, up):
                _nodelay(s)
                self._socks.append(s)
            self._pipe(down, up)
            self._pipe(up, down)

    def _pipe(self, src: socket.socket, dst: socket.socket) -> None:
        """Reader thread: timestamps chunks; writer thread: sends each once its delay has passed."""
        q: collections.deque[tuple[float, bytes]] = collections.deque()
        cv = threading.Condition()

        def reader() -> None:
            try:
                while True:
                    b = src.recv(1 << 20)
                    if not b:
                        break
                    with cv:
                        q.append((time.perf_counter() + self.one_way_s, b))
                        cv.notify()
            except OSError:
                pass
            with cv:
                q.append((0.0, b""))
                cv.notify()

        def writer() -> None:
            try:
                while True:
                    with cv:
                        while not q:
                            cv.wait()
                        due, b = q.popleft()
                    if not b:
                        break
                    wait = due - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    dst.sendall(b)
            except OSError:
                pass
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        threading.Thread(target=reader, daemon=True).start()
        threading.Thread(target=writer, daemon=True).start()
