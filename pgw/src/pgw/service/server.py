"""Run the compute service with hypercorn (HTTP/2 over TLS, h2c or HTTP/1.1 without TLS).

* TLS from ``settings.tls``; with ``tls.require_client_cert`` the server demands a client
  certificate signed by ``tls.ca_file`` (mTLS). ALPN offers ``h2`` and ``http/1.1``.
* SIGTERM / SIGINT start a graceful drain: ``/readyz`` turns 503 at once (so the load balancer
  stops routing here), hypercorn stops accepting connections and waits for open requests
  (``drain_timeout_s``), then the app's lifespan drains the batch scheduler.
* ``metrics_port`` serves ``/metrics`` on a separate plain-HTTP listener (bind it to an internal
  interface) in the same event loop.
* Access logs are off: request paths are harmless, but the service keeps its logs minimal and
  never logs payloads anyway.

Entry points: :func:`serve` (blocking), :func:`serve_async`, ``python -m pgw.service``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import ssl
from typing import Any

from pgw.core.errors import ConfigError
from pgw.service.config import ServiceSettings

__all__ = ["build_hypercorn_config", "serve", "serve_async"]

log = logging.getLogger(__name__)

#: Hypercorn's own lifespan startup timeout; loading and hashing large packs takes a while.
STARTUP_TIMEOUT_S = 3600.0

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def build_hypercorn_config(settings: ServiceSettings, *, bind: str | None = None) -> Any:
    """Hypercorn ``Config`` for the main listener (TLS/mTLS, HTTP/2, timeouts).

    Raises:
        ConfigError: TLS files missing.
    """
    from hypercorn.config import Config

    s = settings
    cfg = Config()
    host = s.host if ":" not in s.host else f"[{s.host}]"
    cfg.bind = [bind or f"{host}:{s.port}"]
    cfg.accesslog = None
    cfg.errorlog = logging.getLogger("hypercorn.error")
    cfg.keep_alive_timeout = s.workers_keep_alive_s
    cfg.graceful_timeout = s.drain_timeout_s
    cfg.startup_timeout = STARTUP_TIMEOUT_S
    cfg.shutdown_timeout = s.drain_timeout_s + 30.0
    cfg.include_server_header = False
    cfg.h2_max_concurrent_streams = max(100, s.max_inflight_requests)
    cfg.alpn_protocols = ["h2", "http/1.1"]
    t = s.tls
    if t.enabled:
        for label, path in (("cert_file", t.cert_file), ("key_file", t.key_file), ("ca_file", t.ca_file)):
            if path is not None and not path.is_file():
                raise ConfigError(f"TLS {label} {path} does not exist", details={"path": str(path)})
        cfg.certfile = str(t.cert_file)
        cfg.keyfile = str(t.key_file)
        if t.key_password is not None:
            cfg.keyfile_password = t.key_password.get_secret_value()
        if t.ca_file is not None:
            cfg.ca_certs = str(t.ca_file)
            cfg.verify_mode = ssl.CERT_REQUIRED if t.require_client_cert else ssl.CERT_OPTIONAL
    elif t.require_client_cert:  # pragma: no cover - rejected by ServiceSettings already
        raise ConfigError("mTLS requires tls.cert_file and tls.key_file")
    else:
        host_only = (bind or s.host).rsplit(":", 1)[0] if bind else s.host
        if host_only.strip("[]") not in _LOOPBACK_HOSTS:
            log.warning(
                "compute service listens on %s WITHOUT TLS: bearer tokens travel in cleartext. "
                "Only use this behind a TLS-terminating proxy on a trusted network.",
                host_only,
            )
    return cfg


async def serve_async(settings: ServiceSettings, *, shutdown_event: asyncio.Event | None = None,
                      app: Any | None = None) -> None:
    """Serve until ``shutdown_event`` is set (or SIGTERM/SIGINT when ``shutdown_event`` is None).

    Args:
        settings: Service settings.
        shutdown_event: External stop signal (tests, embedding). When omitted, signal handlers
            are installed.
        app: Pre-built app (default: :func:`pgw.service.create_app(settings)`).
    """
    from hypercorn.asyncio import serve as hyper_serve
    from hypercorn.config import Config

    from pgw.service.app import create_app, create_metrics_app

    app = app if app is not None else create_app(settings)
    state = app.state.pgw
    stop = shutdown_event or asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[int] = []
    if shutdown_event is None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop.set)
                installed.append(sig)

    async def trigger() -> None:
        await stop.wait()
        state.begin_drain()  # /readyz -> 503 immediately; LB stops routing new requests here
        log.info("shutdown requested; draining")

    tasks = [asyncio.create_task(hyper_serve(app, build_hypercorn_config(settings), shutdown_trigger=trigger))]
    if settings.metrics_port is not None and settings.metrics_enabled:
        mcfg = Config()
        mcfg.bind = [f"{settings.metrics_host}:{settings.metrics_port}"]
        mcfg.accesslog = None
        mcfg.include_server_header = False

        async def mtrigger() -> None:
            await stop.wait()

        tasks.append(asyncio.create_task(hyper_serve(create_metrics_app(state), mcfg, shutdown_trigger=mtrigger)))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc is not None:
                stop.set()
                raise exc
        await asyncio.gather(*tasks)
    finally:
        stop.set()
        for t in tasks:
            if not t.done():
                t.cancel()
        for signum in installed:
            loop.remove_signal_handler(signum)


def serve(settings: ServiceSettings | None = None) -> None:
    """Blocking entry point used by ``pgw service serve`` and ``python -m pgw.service``."""
    from pgw.core.log import configure_logging
    from pgw.core.settings import CommonSettings

    common = CommonSettings()  # PGW_LOG_LEVEL / PGW_LOG_JSON
    configure_logging(level=common.log_level, json_lines=common.log_json)
    s = settings or ServiceSettings()
    log.info("starting %s on %s:%d (tls=%s, mtls=%s, packs=%d)", s.server_name, s.host, s.port, s.tls.enabled,
             s.tls.require_client_cert, len(s.packs))
    asyncio.run(serve_async(s))
