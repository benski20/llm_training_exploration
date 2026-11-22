"""Run the compute service with hypercorn (HTTP/2 over TLS, h2c or HTTP/1.1 without TLS) or uvicorn.

* ``settings.server`` picks the HTTP server of the main listener:

  - ``hypercorn``: HTTP/2 (and HTTP/1.1) for ``POST /pgw/v1/compute``; the WebSocket stream channel
    runs on hypercorn's pure-Python WebSocket framing (wsproto).
  - ``uvicorn``: HTTP/1.1 only (httptools) with the ``websockets`` framing for the stream channel,
    which costs about half the CPU per frame. Use it when clients use the stream channel (the
    default for :class:`pgw.wire.ComputeClient`); HTTP clients then keep a pool of HTTP/1.1
    connections instead of multiplexing HTTP/2 streams. Same routes, auth, limits, TLS policy
    (the SSL context is built by the same code) and verified client identities.

* TLS from ``settings.tls``; with ``tls.require_client_cert`` the server demands a client
  certificate signed by ``tls.ca_file`` (mTLS). ALPN offers ``h2`` and ``http/1.1`` (hypercorn) or
  ``http/1.1`` (uvicorn).
* SIGTERM / SIGINT start a graceful drain: ``/readyz`` turns 503 at once (so the load balancer
  stops routing here), hypercorn stops accepting connections and waits for open requests
  (``drain_timeout_s``), then the app's lifespan drains the batch scheduler.
* ``metrics_port`` serves ``/metrics`` on a separate plain-HTTP listener (bind it to an internal
  interface) in the same event loop.
* Verified client certificates are published to the app as
  ``scope["extensions"]["tls"]["client_cert_name"]`` (see :mod:`pgw.service.tls_peer`; neither
  server does itself), so the mTLS tenant map and the token-vs-certificate cross-check work natively.
* Access logs are off: request paths are harmless, but the service keeps its logs minimal and
  never logs payloads anyway.

Entry points: :func:`serve` (blocking), :func:`serve_async`, ``python -m pgw.service``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import signal
import ssl
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, cast

from pgw.core.errors import ConfigError
from pgw.service.config import ServiceSettings

__all__ = ["build_hypercorn_config", "build_uvicorn_config", "serve", "serve_async"]

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
    cfg.websocket_max_message_size = s.max_request_bytes  # the stream channel's frame limit
    cfg.websocket_ping_interval = None  # clients ping; the channel carries its own traffic
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


def _needs_peer_identity(settings: ServiceSettings) -> bool:
    t = settings.tls
    return t.enabled and t.ca_file is not None and not settings.trusted_client_cert_header


#: Messages a WebSocket connection may have queued for the app before reading pauses.
WS_QUEUE_PAUSE = 2
#: Strong references to frame-lane replies that had to wait for the transport (never in practice).
_LANE_SENDS: set[asyncio.Future[Any]] = set()


def stream_ws_protocol(base: type[Any]) -> type[Any]:
    """uvicorn's sans-I/O WebSocket protocol, pausing reads only when the app falls behind.

    uvicorn pauses the transport after *every* received message and resumes it when the app takes
    the message: two event-loop selector updates (``epoll_ctl``) per compute frame on the stream
    channel, whose reader takes every frame at once. Here reading pauses only once
    :data:`WS_QUEUE_PAUSE` messages are waiting (and uvicorn resumes it, as before, when the queue
    drains), so buffering stays bounded to a couple of frames. Classes without the expected
    ``read_paused`` / ``queue`` / ``send_receive_event_to_app`` hooks are returned unchanged.
    """
    import inspect

    hook = getattr(base, "send_receive_event_to_app", None)
    try:
        src = inspect.getsource(base) if hook is not None else ""
    except (OSError, TypeError):  # pragma: no cover - no source: cannot verify
        src = ""
    if "self.read_paused" not in src or "self.queue" not in src or "pause_reading" not in src:
        return base
    # The frame lane (pgw.wire.FrameLane) needs these sans-I/O protocol internals; without them the
    # protocol offers no lane and every frame takes the ASGI path, as before.
    lane_ok = all(h in src for h in (
        "def handle_bytes", "def handle_connect", "self.writable", "self.close_sent", "self.handshake_complete",
        "self.handshake_initiated", "self.conn", "self.scope", "async def send",
    ))
    from websockets.protocol import State

    from pgw.wire.server import FRAME_LANE_EXTENSION, FrameLane, ws_binary_header

    class StreamWebSocketProtocol(base):  # type: ignore[misc]
        _pgw_lane: FrameLane | None = None

        def handle_connect(self, event: Any) -> None:
            super().handle_connect(event)
            me = cast(Any, self)
            scope = getattr(me, "scope", None) if lane_ok else None
            if isinstance(scope, dict) and me.handshake_initiated and not me.close_sent:
                # The app task was created but has not run yet: it sees the lane in its scope.
                lane = FrameLane(self._pgw_ready, self._pgw_write)
                ext = scope.get("extensions")
                if not isinstance(ext, dict):
                    ext = scope["extensions"] = {}
                ext[FRAME_LANE_EXTENSION] = lane
                self._pgw_lane = lane

        def _pgw_ready(self) -> bool:
            me = cast(Any, self)
            return bool(me.handshake_complete and not me.close_sent and me.writable.is_set()
                        and not me.transport.is_closing() and me.conn.state is State.OPEN
                        and not me.conn.extensions)

        def _pgw_write(self, parts: Any) -> bool:
            """One unfragmented binary frame from ``parts``, written as a buffer list (the bytes
            ``websockets`` would write for that message, without assembling them first)."""
            if not self._pgw_ready():
                return False
            n = 0
            for p in parts:
                n += p.nbytes if isinstance(p, memoryview) else len(p)
            cast(Any, self).transport.writelines([ws_binary_header(n), *parts])
            return True

        def handle_bytes(self, event: Any) -> None:
            lane = self._pgw_lane
            handler = lane.handler if lane is not None else None
            if handler is not None and event.fin and self._pgw_ready():
                try:
                    parts = handler(event.data)
                except Exception:  # pragma: no cover - the lane maps every error to a reply
                    log.exception("compute stream frame lane failed; using the regular path")
                    parts = None
                if parts is not None:
                    if not self._pgw_write(parts):  # pragma: no cover - ready() held a moment ago
                        data = parts[0] if len(parts) == 1 else b"".join(parts)
                        t = asyncio.ensure_future(cast(Any, self).send({"type": "websocket.send", "bytes": data}))
                        _LANE_SENDS.add(t)
                        t.add_done_callback(_LANE_SENDS.discard)
                    return
            super().handle_bytes(event)

        def send_receive_event_to_app(self) -> None:
            me = cast(Any, self)
            paused = bool(me.read_paused)
            me.read_paused = True  # uvicorn pauses only a transport that is not paused already
            try:
                super().send_receive_event_to_app()
            finally:
                me.read_paused = paused
            if not paused and me.queue.qsize() >= WS_QUEUE_PAUSE and not me.transport.is_closing():
                me.read_paused = True
                me.transport.pause_reading()

    StreamWebSocketProtocol.__name__ = StreamWebSocketProtocol.__qualname__ = f"Stream{base.__name__}"
    return StreamWebSocketProtocol


def build_uvicorn_config(settings: ServiceSettings, app: Any, *, bind: str | None = None) -> Any:
    """uvicorn ``Config`` for the main listener: httptools + ``websockets`` framing, the same TLS
    policy as hypercorn (the SSL context comes from :func:`build_hypercorn_config`, with ALPN
    ``http/1.1``), no proxy-header rewriting, no access log, uvicorn's logging left alone.

    With client-certificate verification (and no ``trusted_client_cert_header``) the protocol
    classes publish the verified subject (:func:`pgw.service.tls_peer.uvicorn_peer_cert_protocol`).

    Raises:
        ConfigError: TLS files missing, or the identity hook does not fit this uvicorn version.
    """
    import uvicorn
    from uvicorn.config import HTTP_PROTOCOLS, WS_PROTOCOLS
    from uvicorn.importer import import_from_string

    from pgw.service.tls_peer import uvicorn_peer_cert_protocol, uvicorn_peer_cert_supported

    s = settings
    hcfg = build_hypercorn_config(s, bind=bind)  # validates TLS files, warns about plaintext
    host, _, port = hcfg.bind[0].rpartition(":")
    try:
        import httptools  # noqa: F401
        http_name = "httptools"
    except ImportError:  # pragma: no cover - part of uvicorn[standard]
        http_name = "h11"
    http_cls = import_from_string(HTTP_PROTOCOLS[http_name])
    ws_path = WS_PROTOCOLS.get("websockets-sansio")
    if ws_path is None:  # pragma: no cover - uvicorn < 0.35
        ws_path = WS_PROTOCOLS["websockets"]
    assert ws_path is not None
    ws_cls = stream_ws_protocol(import_from_string(ws_path))
    if _needs_peer_identity(s):
        if not uvicorn_peer_cert_supported(http_cls, ws_cls):
            raise ConfigError(
                "this uvicorn version does not let pgw publish verified client certificates; use "
                "server='hypercorn', or terminate TLS in a proxy and set trusted_client_cert_header"
            )
        http_cls, ws_cls = uvicorn_peer_cert_protocol(http_cls), uvicorn_peer_cert_protocol(ws_cls)
    kw: dict[str, Any] = {}
    if s.tls.enabled:
        import inspect

        if "ssl_context_factory" not in inspect.signature(uvicorn.Config).parameters:
            raise ConfigError("server='uvicorn' with TLS needs a uvicorn release with ssl_context_factory")
        ctx = hcfg.create_ssl_context()
        ctx.set_alpn_protocols(["http/1.1"])  # no HTTP/2 without uvicorn's optional h2 support

        def ssl_factory(config: Any, default: Callable[[], ssl.SSLContext]) -> ssl.SSLContext:
            return cast(ssl.SSLContext, ctx)

        kw["ssl_context_factory"] = ssl_factory
    return uvicorn.Config(
        app, host=host.strip("[]"), port=int(port), http=http_cls, ws=ws_cls, lifespan="on",
        log_config=None, access_log=False, proxy_headers=False, server_header=False,
        timeout_keep_alive=max(1, math.ceil(s.workers_keep_alive_s)),
        timeout_graceful_shutdown=max(1, math.ceil(s.drain_timeout_s)),
        ws_max_size=s.max_request_bytes, ws_ping_interval=None, ws_ping_timeout=None,
        ws_per_message_deflate=False, **kw,
    )


def _uvicorn_server(config: Any) -> Any:
    import uvicorn

    class _Server(uvicorn.Server):
        """Signals are handled by :func:`serve_async` (uvicorn's own handler re-raises them on exit)."""

        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:
            yield

    return _Server(config)


async def _serve_uvicorn(app: Any, settings: ServiceSettings, stop: asyncio.Event,
                         drained: Callable[[], Awaitable[None]]) -> None:
    server = _uvicorn_server(build_uvicorn_config(settings, app))
    task = asyncio.create_task(server.serve())

    async def trigger() -> None:
        await stop.wait()
        # In-flight requests (stream frames included) finish before uvicorn closes connections:
        # it closes WebSocket channels at once (1012) when told to exit.
        await drained()
        server.should_exit = True

    trig = asyncio.create_task(trigger())
    try:
        await asyncio.wait([task, trig], return_when=asyncio.FIRST_COMPLETED)
        if not task.done():
            await task
        task.result()
        if not server.started:
            raise ConfigError("compute service failed to start (see the log)")
    finally:
        trig.cancel()
        with contextlib.suppress(BaseException):
            await trig


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
    from pgw.service.tls_peer import PeerCertMiddleware, peer_cert_supported

    app = app if app is not None else create_app(settings)
    state = app.state.pgw
    if settings.server == "hypercorn" and _needs_peer_identity(settings):
        if not peer_cert_supported():
            # Fail closed: without the peer certificate the tenant map rejects everything and the
            # token-vs-certificate tenant cross-check would silently never run.
            raise ConfigError(
                "this hypercorn version does not expose verified client certificates to the app; "
                "terminate TLS in a proxy and set trusted_client_cert_header, or pin a supported hypercorn"
            )
        app = cast(Any, PeerCertMiddleware(app))
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

    async def drained() -> None:
        state.begin_drain()
        log.info("shutdown requested; draining")
        await state.handler.wait_idle(settings.drain_timeout_s)

    bulk = _start_maskgen_bulk(settings, state, loop)
    if settings.server == "uvicorn":
        main = _serve_uvicorn(app, settings, stop, drained)
    else:
        main = hyper_serve(app, build_hypercorn_config(settings), shutdown_trigger=trigger)
    tasks = [asyncio.create_task(main)]
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
        if bulk is not None:
            await asyncio.to_thread(bulk.close)
        for t in tasks:
            if not t.done():
                t.cancel()
        for signum in installed:
            loop.remove_signal_handler(signum)


def _start_maskgen_bulk(settings: ServiceSettings, state: Any, loop: asyncio.AbstractEventLoop) -> Any:
    """The bulk maskgen listener (:mod:`pgw.service.maskgen_bulk`) when ``maskgen_bulk_port`` is set, else None."""
    if settings.maskgen_bulk_port is None:
        return None
    from pgw.service.maskgen_bulk import MaskgenBulkServer

    ctx = None
    if settings.tls.enabled:
        ctx = build_hypercorn_config(settings).create_ssl_context()  # the main listener's TLS and client-cert policy
        ctx.set_alpn_protocols(["pgw-maskgen-bulk"])
    elif settings.host.strip("[]") not in _LOOPBACK_HOSTS:  # pragma: no cover - rejected by ServiceSettings
        raise ConfigError("maskgen_bulk_port without TLS is only allowed on a loopback host")
    srv = MaskgenBulkServer(state.handler, loop, ctx, host=settings.host, port=settings.maskgen_bulk_port,
                            verify_peer=_needs_peer_identity(settings), max_conns=settings.maskgen_bulk_max_conns,
                            body_budget=getattr(state, "body_budget", None))
    srv.start()
    return srv


def serve(settings: ServiceSettings | None = None) -> None:
    """Blocking entry point used by ``pgw service serve`` and ``python -m pgw.service``."""
    from pgw.core.log import configure_logging
    from pgw.core.proc_tuning import apply_malloc_tuning, loop_name, new_event_loop
    from pgw.core.settings import CommonSettings

    malloc_tuned = apply_malloc_tuning()
    common = CommonSettings()  # PGW_LOG_LEVEL / PGW_LOG_JSON
    configure_logging(level=common.log_level, json_lines=common.log_json)
    s = settings or ServiceSettings()
    log.info("starting %s on %s:%d (tls=%s, mtls=%s, packs=%d, loop=%s, malloc_tuning=%s)", s.server_name, s.host,
             s.port, s.tls.enabled, s.tls.require_client_cert, len(s.packs), loop_name(), malloc_tuned)
    asyncio.run(serve_async(s), loop_factory=new_event_loop)
