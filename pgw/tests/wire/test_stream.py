"""The persistent compute channel (``/pgw/v1/stream``) over a real hypercorn or uvicorn socket.

Same results, errors and limits as one HTTP request per call; the client falls back to HTTP
whenever the channel cannot be opened, and retries on a new connection when it breaks.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("hypercorn")
pytest.importorskip("websockets")

from pgw.core import RING_K40_A12, ErrorCode, PGWError, ServiceUnavailableError, unpack_ring
from pgw.wire import ClientConfig, ComputeClient, WireLimits
from tests.wire.conftest import DIMS, AuthHandler, FakeHandler, expected_y, make_app, make_request

RING = RING_K40_A12


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Server:
    def __init__(self, handler: Any, limits: WireLimits | None = None, kind: str = "hypercorn"):
        self.handler = handler
        self.port = _free_port()
        self.loop = asyncio.new_event_loop()
        self.stop = asyncio.Event()
        app = make_app(handler, limits)

        async def main() -> None:
            if kind == "uvicorn":
                import uvicorn

                server = uvicorn.Server(uvicorn.Config(
                    app, host="127.0.0.1", port=self.port, ws="websockets-sansio", log_config=None,
                    lifespan="off", ws_ping_interval=None, ws_per_message_deflate=False))
                task = asyncio.ensure_future(server.serve())
                await self.stop.wait()
                server.should_exit = True
                await task
                return
            from hypercorn.asyncio import serve
            from hypercorn.config import Config

            cfg = Config()
            cfg.bind = [f"127.0.0.1:{self.port}"]
            cfg.accesslog = None
            cfg.errorlog = None
            await serve(app, cfg, shutdown_trigger=self.stop.wait)  # type: ignore[arg-type]

        self.thread = threading.Thread(target=lambda: self.loop.run_until_complete(main()), daemon=True)
        self.thread.start()
        for _ in range(200):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(10)


@pytest.fixture(params=["hypercorn", "uvicorn"])
def serve(request: pytest.FixtureRequest) -> Iterator[Any]:
    servers: list[_Server] = []

    def start(handler: Any, limits: WireLimits | None = None) -> _Server:
        s = _Server(handler, limits, request.param)
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.close()


def _client(url: str, **kw: Any) -> ComputeClient:
    return ComputeClient(ClientConfig(base_url=url, request_timeout_s=20, backoff_base_s=0.001, **kw))


def _check(resp: Any, req: Any, xs: Any, handler: FakeHandler) -> None:
    for it, x, res in zip(req.items, xs, resp.items, strict=True):
        y = unpack_ring(res.y_hat, (it.rows, DIMS[it.linear][0]), RING)
        assert (y == expected_y(RING, it.linear, x, handler.weights)).all()


async def test_stream_exact_and_multiplexed(serve: Any) -> None:
    srv = serve(FakeHandler(delay_s=0.01))
    client = _client(srv.url)
    await client.connect(None, RING.name)
    reqs = [make_request(RING, [("0.qkv", 2), ("0.down", 1)], rid=f"r{i}", seed=i) for i in range(16)]
    resps = await asyncio.gather(*(client.compute(r) for r, _ in reqs))
    for (req, xs), resp in zip(reqs, resps, strict=True):
        assert resp.request_id == req.request_id
        _check(resp, req, xs, srv.handler)
    assert client.stats.stream_opens == 1 and client.stats.stream_requests == 16
    assert all(code is None for _, code in srv.handler.done)  # metrics hook ran for every frame
    # identical to the HTTP path
    http = _client(srv.url, stream=False)
    req, _ = reqs[0]
    assert await http.compute(req) == resps[0]
    assert http.stats.stream_requests == 0
    await client.aclose()
    await http.aclose()


async def test_stream_error_replies_are_typed_and_keep_the_channel(serve: Any) -> None:
    srv = serve(FakeHandler())
    client = _client(srv.url, max_retries=0)
    await client.connect(None, RING.name)
    bad, _ = make_request(RING, [("0.qkv", 1)], fp="f" * 64, rid="bad")
    with pytest.raises(PGWError) as ei:
        await client.compute(bad)
    http = _client(srv.url, stream=False, max_retries=0)
    with pytest.raises(PGWError) as eh:
        await http.compute(bad)
    assert ei.value.code == eh.value.code and ei.value.code != ErrorCode.UNAVAILABLE
    good, xs = make_request(RING, [("0.qkv", 1)], rid="good")
    _check(await client.compute(good), good, xs, srv.handler)
    assert client.stats.stream_opens == 1  # the error did not cost the connection
    await client.aclose()
    await http.aclose()


async def test_falls_back_to_http_when_the_service_has_no_stream(serve: Any) -> None:
    srv = serve(FakeHandler(), WireLimits(stream=False))
    client = _client(srv.url)
    req, xs = make_request(RING, [("0.qkv", 3)])
    _check(await client.compute(req), req, xs, srv.handler)
    _check(await client.compute(req), req, xs, srv.handler)
    assert client.stats.stream_open_failures == 1 and client.stats.stream_requests == 0  # not retried each call
    await client.aclose()


async def test_handshake_requires_authentication(serve: Any) -> None:
    from pydantic import SecretStr

    srv = serve(AuthHandler())
    bad = _client(srv.url, token=SecretStr("wrong"), max_retries=0)
    req, xs = make_request(RING, [("0.qkv", 1)])
    with pytest.raises(PGWError) as ei:  # refused at the handshake, then the HTTP reply is typed
        await bad.compute(req)
    assert ei.value.code == ErrorCode.UNAUTHENTICATED
    assert bad.stats.stream_open_failures == 1 and not srv.handler.calls
    good = _client(srv.url, token=SecretStr("good-token"))
    _check(await good.compute(req), req, xs, srv.handler)
    assert good.stats.stream_requests == 1 and srv.handler.contexts[-1].identity == {"tenant": "t1"}
    await bad.aclose()
    await good.aclose()


class _DrainOnce(FakeHandler):
    """Admission refuses one compute frame as 'draining' (precheck call 2; call 1 is the handshake)."""

    def __init__(self) -> None:
        super().__init__()
        self.prechecks = 0

    def precheck(self) -> None:
        self.prechecks += 1
        if self.prechecks == 2:
            raise ServiceUnavailableError("service is draining", details={"retry_after_s": 0.0})


async def test_unavailable_reply_retries_on_a_new_connection(serve: Any) -> None:
    srv = serve(_DrainOnce())
    client = _client(srv.url, max_retries=3, backoff_max_s=0.01)
    req, xs = make_request(RING, [("0.qkv", 2)])
    _check(await client.compute(req), req, xs, srv.handler)
    assert client.stats.retries == 1
    assert client.stats.stream_opens == 2  # the refusing connection was retired, the retry reconnected
    assert srv.handler.prechecks == 4
    await client.aclose()


async def test_broken_connection_is_retried(serve: Any) -> None:
    srv = serve(FakeHandler())
    client = _client(srv.url, max_retries=3, backoff_max_s=0.01)
    await client.connect(None, RING.name)
    req, xs = make_request(RING, [("0.qkv", 1)])
    await client.compute(req)
    ch = next(iter(client._channels.values()))
    await ch.ws.close()  # the connection drops between calls
    _check(await client.compute(req), req, xs, srv.handler)
    assert client.stats.stream_opens == 2
    await client.aclose()


def test_sync_api_uses_the_stream(serve: Any) -> None:
    srv = serve(FakeHandler())
    client = _client(srv.url)
    client.connect_sync(None, RING.name)
    req, xs = make_request(RING, [("0.gate_up", 1)])
    for _ in range(3):
        _check(client.compute_sync(req), req, xs, srv.handler)
    assert client.stats.stream_requests == 3 and client.stats.stream_opens == 1
    client.close()
