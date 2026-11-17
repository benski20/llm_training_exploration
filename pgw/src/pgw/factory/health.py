"""Internal HTTP listener of the factory: ``/healthz`` (liveness) and ``/metrics`` (Prometheus).

The factory is a batch worker without an API. Without this listener a hung factory (stuck GPU,
store lock wait, configuration error loop) stays invisible until the gateway runs out of masks.
Serve it on a pod-internal port only; it exposes counters and timestamps, never secrets.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["FactoryHealthServer"]

log = logging.getLogger(__name__)


class FactoryHealthServer:
    """Background HTTP server for one :class:`pgw.factory.FactoryScheduler`.

    Args:
        scheduler: Object with ``health() -> (ok, details)`` and a ``metrics`` attribute that has
            ``render() -> (body, content_type)``.
        host: Bind address.
        port: Port (0 picks a free one; see :attr:`port`).
    """

    def __init__(self, scheduler: Any, host: str = "0.0.0.0", port: int = 9100):  # noqa: S104
        sched = scheduler

        class _Handler(BaseHTTPRequestHandler):
            server_version = "pgw-factory"
            sys_version = ""

            def log_message(self, fmt: str, *args: Any) -> None:  # quiet access log
                log.debug("factory health: " + fmt, *args)

            def _send(self, status: int, body: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                try:
                    if path == "/healthz":
                        ok, details = sched.health()
                        self._send(200 if ok else 503, json.dumps(details).encode(), "application/json")
                    elif path == "/metrics":
                        body, ctype = sched.metrics.render()
                        self._send(200, body, ctype)
                    else:
                        self._send(404, b"not found\n", "text/plain")
                except Exception:  # never let a probe kill the thread
                    log.exception("factory health endpoint failed")
                    self._send(500, b"error\n", "text/plain")

        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Bound port."""
        return int(self._httpd.server_address[1])

    def start(self) -> FactoryHealthServer:
        """Serve in a daemon thread."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="pgw-factory-health", daemon=True)
        self._thread.start()
        log.info("factory health/metrics listener on port %d", self.port)
        return self

    def stop(self) -> None:
        """Stop serving."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> FactoryHealthServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
