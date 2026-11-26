"""A fault-injection TCP proxy for the client-library reliability tests.

:class:`ChaosProxy` forwards ``127.0.0.1:<port>`` to a target port (bytes untouched: TLS end to end, so the client
still verifies the provider's pinned CA through it) and can, on command:

* ``drop()``: reset every open connection (a network blip, a provider restart seen from the client);
* ``mode = "blackhole"``: keep connections open but stop forwarding in both directions (a hung network path);
* ``mode = "refuse"``: close new connections at once (the provider is down);
* ``delay_s``: hold every forwarded read this long (a slow but working path: requests take long, then complete).

It runs its own event loop in a daemon thread, so it works next to a synchronous client in the test process.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from typing import Literal

Mode = Literal["pass", "blackhole", "refuse"]


class ChaosProxy:
    """TCP proxy ``127.0.0.1:port -> 127.0.0.1:target`` with fault injection (see the module docstring)."""

    def __init__(self, target: int, port: int | None = None):
        self.target = target
        self.port = port or _free_port()
        self.mode: Mode = "pass"
        self.delay_s = 0.0
        self.connections = 0
        self._writers: set[asyncio.StreamWriter] = set()
        self._loop = asyncio.new_event_loop()
        self._server: asyncio.base_events.Server | None = None
        started = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            self._server = self._loop.run_until_complete(
                asyncio.start_server(self._handle, "127.0.0.1", self.port, reuse_address=True))
            started.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, name=f"chaos-proxy-{self.port}", daemon=True)
        self._thread.start()
        started.wait(10)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.mode == "refuse":
            _reset(writer)
            return
        try:
            ur, uw = await asyncio.open_connection("127.0.0.1", self.target)
        except OSError:
            _reset(writer)
            return
        self.connections += 1
        self._writers.update((writer, uw))

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    data = await src.read(65536)
                    if not data:
                        break
                    while self.mode == "blackhole":
                        await asyncio.sleep(0.05)
                    if self.delay_s > 0:
                        await asyncio.sleep(self.delay_s)
                    dst.write(data)
                    await dst.drain()
            except (OSError, asyncio.CancelledError):
                pass
            finally:
                _reset(dst)

        try:
            await asyncio.gather(pipe(reader, uw), pipe(ur, writer))
        finally:
            self._writers.discard(writer)
            self._writers.discard(uw)

    def _call(self, fn: object) -> None:
        fut = asyncio.run_coroutine_threadsafe(_run(fn), self._loop)
        fut.result(10)

    def drop(self) -> None:
        """Reset every open connection (both sides)."""

        def go() -> None:
            for w in list(self._writers):
                _reset(w)
            self._writers.clear()

        self._call(go)

    def close(self) -> None:
        """Stop listening and drop everything."""
        self.drop()

        def go() -> None:
            if self._server is not None:
                self._server.close()

        self._call(go)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


async def _run(fn: object) -> None:
    fn()  # type: ignore[operator]


def _reset(w: asyncio.StreamWriter) -> None:
    """Close with RST (SO_LINGER 0), like a dropped path or a killed peer."""
    with contextlib.suppress(Exception):
        sock = w.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    with contextlib.suppress(Exception):
        w.transport.abort()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
