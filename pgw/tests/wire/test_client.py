"""ComputeClient: end-to-end over ASGI, retries, deadlines, splitting, validation, sync API."""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from pgw.core import (
    RING_K40_A12,
    RING_K48_A20,
    ConfigError,
    ErrorCode,
    FingerprintMismatchError,
    PayloadTooLargeError,
    PGWError,
    ProtocolError,
    ServiceUnavailableError,
    ShapeError,
    VersionMismatchError,
    unpack_ring,
)
from pgw.wire import (
    CONTENT_TYPE,
    ClientConfig,
    ComputeClient,
    ComputeItem,
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    ErrorResponse,
    ModelsResponse,
    WireLimits,
    decode_message,
    encode_error_json,
    encode_message,
)
from tests.wire.conftest import DIMS, FP, OTHER_FP, AuthHandler, FakeHandler, expected_y, make_app, make_request


def _cfg(**kw) -> ClientConfig:
    base = {"base_url": "http://svc", "backoff_base_s": 0.001, "backoff_max_s": 0.005}
    base.update(kw)
    return ClientConfig(**base)


def asgi_client(handler, limits=None, **cfg) -> ComputeClient:
    return ComputeClient(_cfg(**cfg), transport=httpx.ASGITransport(app=make_app(handler, limits)))


def _check_exact(req, resp, xs, ring, weights):
    assert resp.request_id == req.request_id
    assert len(resp.items) == len(req.items)
    for it, x, res in zip(req.items, xs, resp.items, strict=True):
        assert res.linear == it.linear and res.rows == it.rows
        y = unpack_ring(res.y_hat, (it.rows, DIMS[it.linear][0]), ring)
        assert (y == expected_y(ring, it.linear, x, weights)).all()


# ------------------------------------------------------------------------------ config
def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        ClientConfig(base_url="ftp://x")
    with pytest.raises(ValueError):
        ClientConfig(base_url="https://x", client_cert_file=tmp_path / "c.pem")
    c = ClientConfig(base_url="https://svc:8443/", token="t0k")
    assert c.base_url == "https://svc:8443" and c.uses_tls
    assert "t0k" not in repr(c)
    with pytest.raises(ConfigError):  # unreadable mTLS material
        ComputeClient(ClientConfig(base_url="https://x", client_cert_file=tmp_path / "a", client_key_file=tmp_path / "b"))
    with pytest.raises(ConfigError):
        ComputeClient(ClientConfig(base_url="https://x", ca_file=tmp_path / "missing.pem"))


def test_config_env_embedding(monkeypatch):
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class S(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="PGW_TEST_", env_nested_delimiter="__")
        service: ClientConfig

    monkeypatch.setenv("PGW_TEST_SERVICE__BASE_URL", "https://compute.example.net")
    monkeypatch.setenv("PGW_TEST_SERVICE__MAX_RETRIES", "7")
    monkeypatch.setenv("PGW_TEST_SERVICE__TOKEN", "secret")
    s = S()
    assert s.service.max_retries == 7 and s.service.token.get_secret_value() == "secret"


# ------------------------------------------------------------------------------ happy path
async def test_connect_and_compute_exact(handler, any_ring):
    client = asgi_client(handler)
    models = await client.connect(FP, any_ring.name)
    assert isinstance(models, ModelsResponse) and client.model is not None
    req, xs = make_request(any_ring, [("0.qkv", 4), ("0.o", 1), ("0.gate_up", 3), ("0.down", 2)])
    resp = await client.compute(req)
    _check_exact(req, resp, xs, any_ring, handler.weights)
    assert client.stats.requests == 1 and client.stats.attempts == 2 and client.stats.retries == 0
    assert client.stats.bytes_sent > req.payload_bytes
    await client.aclose()


async def test_connect_failures(handler):
    client = asgi_client(handler)
    with pytest.raises(FingerprintMismatchError):
        await client.connect(OTHER_FP)
    h2 = FakeHandler(rings=("K48A20",))
    with pytest.raises(ConfigError):
        await asgi_client(h2).connect(FP, "K40A12")
    await asgi_client(h2).connect(FP, "k48a20")
    await client.aclose()


async def test_connect_version_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        body = encode_message(ModelsResponse((), versions=(2,)))
        return httpx.Response(200, content=body, headers={"content-type": CONTENT_TYPE})

    c = ComputeClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(VersionMismatchError):
        await c.connect()


async def test_headers_and_auth():
    seen: list[httpx.Request] = []
    h = FakeHandler()

    async def mock(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        req = decode_message(request.content, ComputeRequest)
        resp = await h.compute(req, None)  # type: ignore[arg-type]
        return httpx.Response(200, content=encode_message(resp), headers={"content-type": CONTENT_TYPE})

    c = ComputeClient(_cfg(token="tok-123"), transport=httpx.MockTransport(mock))
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    await c.compute(req)
    r = seen[0]
    assert r.headers["authorization"] == "Bearer tok-123"
    assert r.headers["content-type"] == CONTENT_TYPE
    assert r.headers["pgw-protocol"] == "1"
    assert r.headers["pgw-request-id"] == req.request_id
    assert r.headers["pgw-attempt"] == "1"
    assert r.url.path == "/pgw/v1/compute"


async def test_auth_failure_not_retried():
    c = asgi_client(AuthHandler(), token="bad")
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    with pytest.raises(PGWError) as ei:
        await c.compute(req)
    assert ei.value.code == ErrorCode.UNAUTHENTICATED
    assert c.stats.attempts == 1
    ok = asgi_client(AuthHandler(), token="good-token")
    await ok.connect(FP)
    await ok.compute(req)


# ------------------------------------------------------------------------------ retries
class Scripted:
    """MockTransport handler failing according to a script, then computing for real."""

    def __init__(self, script: list):
        self.script = list(script)
        self.h = FakeHandler()
        self.bodies: list[bytes] = []
        self.attempt_headers: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        self.attempt_headers.append(request.headers.get("pgw-attempt", ""))
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(request)
            status, err, headers = step
            body = encode_message(err) if err is not None else b"oops"
            ct = CONTENT_TYPE if err is not None else "text/plain"
            return httpx.Response(status, content=body, headers={"content-type": ct, **headers})
        req = decode_message(request.content, ComputeRequest)
        resp = await self.h.compute(req, None)  # type: ignore[arg-type]
        return httpx.Response(200, content=encode_message(resp), headers={"content-type": CONTENT_TYPE})


async def test_retries_on_retryable_errors_resend_identical_body():
    unavailable = ErrorResponse("unavailable", "busy", True)
    script = Scripted([
        (503, unavailable, {}),
        httpx.ConnectError("refused"),
        (429, ErrorResponse("rate_limited", "slow", True), {"Retry-After": "0"}),
        (502, None, {}),
    ])
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    c = ComputeClient(_cfg(max_retries=5), transport=httpx.MockTransport(script), sleep=fake_sleep)
    req, xs = make_request(RING_K48_A20, [("0.qkv", 2)])
    resp = await c.compute(req)
    _check_exact(req, resp, xs, RING_K48_A20, script.h.weights)
    assert len(set(script.bodies)) == 1  # identical x_hat (same mask) on every attempt
    assert script.attempt_headers == ["1", "2", "3", "4", "5"]
    assert c.stats.retries == 4 and len(sleeps) == 4


async def test_retry_after_is_honoured_and_capped():
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    script = Scripted([(503, ErrorResponse("unavailable", "x", True), {"Retry-After": "3"}),
                       (503, ErrorResponse("unavailable", "x", True), {"Retry-After": "999"})])
    c = ComputeClient(_cfg(max_retry_after_s=5.0, request_timeout_s=60), transport=httpx.MockTransport(script),
                      sleep=fake_sleep)
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    await c.compute(req)
    assert sleeps[0] >= 3.0 and sleeps[1] == 5.0


@pytest.mark.parametrize(
    "status,err,exc_code",
    [
        (400, ErrorResponse("bad_shape", "s"), ErrorCode.BAD_SHAPE),
        (409, ErrorResponse("fingerprint_mismatch", "f"), ErrorCode.FINGERPRINT_MISMATCH),
        (413, ErrorResponse("payload_too_large", "p"), ErrorCode.PAYLOAD_TOO_LARGE),
        (500, ErrorResponse("internal", "i"), ErrorCode.INTERNAL),
        (500, None, ErrorCode.INTERNAL),
        (404, None, ErrorCode.MODEL_NOT_FOUND),
        (418, None, ErrorCode.PROTOCOL),
    ],
)
async def test_non_retryable_errors_fail_fast(status, err, exc_code):
    script = Scripted([(status, err, {})])
    c = ComputeClient(_cfg(), transport=httpx.MockTransport(script))
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    with pytest.raises(PGWError) as ei:
        await c.compute(req)
    assert ei.value.code == exc_code
    assert len(script.bodies) == 1 and c.stats.failures == 1


async def test_json_error_bodies_are_understood():
    def step(request):
        return httpx.Response(429, content=encode_error_json(ErrorResponse("rate_limited", "x", True)),
                              headers={"content-type": "application/json"})

    script = Scripted([step])
    c = ComputeClient(_cfg(), transport=httpx.MockTransport(script))
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    await c.compute(req)
    assert c.stats.retries == 1


async def test_retries_exhausted():
    script = Scripted([httpx.ConnectError("x")] * 10)
    c = ComputeClient(_cfg(max_retries=2), transport=httpx.MockTransport(script))
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    with pytest.raises(ServiceUnavailableError) as ei:
        await c.compute(req)
    assert len(script.bodies) == 3 and ei.value.details["attempts"] == 3


async def test_deadline_exceeded_against_slow_server():
    h = FakeHandler(delay_s=1.0)
    c = asgi_client(h, request_timeout_s=0.2, max_retries=0)
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    with pytest.raises(PGWError) as ei:
        await c.compute(req)
    assert ei.value.code in (ErrorCode.DEADLINE_EXCEEDED,)


async def test_deadline_ms_limits_retries():
    script = Scripted([httpx.ReadTimeout("t")] * 100)

    async def slow_sleep(s: float) -> None:
        await asyncio.sleep(0.03)

    c = ComputeClient(_cfg(max_retries=100, backoff_base_s=0.03, backoff_max_s=0.03),
                      transport=httpx.MockTransport(script), sleep=slow_sleep)
    req, _ = make_request(RING_K40_A12, [("0.o", 1)], deadline_ms=100)
    with pytest.raises(PGWError) as ei:
        await c.compute(req)
    assert ei.value.code == ErrorCode.DEADLINE_EXCEEDED
    assert 1 < len(script.bodies) < 20


# ------------------------------------------------------------------------------ response validation
@pytest.mark.parametrize("mutate", ["id", "count", "length", "linear", "garbage", "wrong_type"])
async def test_invalid_responses_are_rejected(mutate):
    h = FakeHandler()

    async def mock(request: httpx.Request) -> httpx.Response:
        req = decode_message(request.content, ComputeRequest)
        resp = await h.compute(req, None)  # type: ignore[arg-type]
        it = resp.items[0]
        if mutate == "id":
            resp = ComputeResponse("other", resp.items)
        elif mutate == "count":
            resp = ComputeResponse(resp.request_id, resp.items + resp.items)
        elif mutate == "length":
            resp = ComputeResponse(resp.request_id, (ComputeResult(it.linear, it.rows, it.y_hat + b"\x00" * 5),))
        elif mutate == "linear":
            resp = ComputeResponse(resp.request_id, (ComputeResult("0.qkv", it.rows, it.y_hat),))
        if mutate == "garbage":
            body = b"\x93garbage"
        elif mutate == "wrong_type":
            body = encode_message(ModelsResponse(()))
        else:
            body = encode_message(resp)
        return httpx.Response(200, content=body, headers={"content-type": CONTENT_TYPE})

    async def models_and_compute(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, content=encode_message(await h.models()), headers={"content-type": CONTENT_TYPE})
        return await mock(request)

    c = ComputeClient(_cfg(), transport=httpx.MockTransport(models_and_compute))
    await c.connect(FP)
    req, _ = make_request(RING_K40_A12, [("0.o", 2)])
    with pytest.raises(ProtocolError):
        await c.compute(req)
    assert c.stats.attempts == 2  # models + one compute: protocol errors are not retried


# ------------------------------------------------------------------------------ splitting
async def test_split_by_bytes_rows_and_items(any_ring):
    h = FakeHandler()
    # server advertises small limits; client must split transparently
    h.max_rows = 3
    h.max_bytes = 6000
    lim = WireLimits(max_request_bytes=6000, max_items=4, max_rows_per_item=3)
    c = asgi_client(h, lim, max_items=4)
    await c.connect(FP)
    spec = [("0.qkv", 10), ("0.o", 1), ("0.down", 7), ("0.gate_up", 2), ("0.qkv", 1), ("0.o", 5)]
    req, xs = make_request(any_ring, spec)
    resp = await c.compute(req)
    _check_exact(req, resp, xs, any_ring, h.weights)
    assert len(h.calls) > 1 and c.stats.splits == 1
    for sub in h.calls:
        assert len(sub.items) <= 4 and all(it.rows <= 3 for it in sub.items)
        assert len(encode_message(sub)) <= 6000
        assert sub.request_id.startswith(req.request_id + ".")
    assert sum(sub.total_rows for sub in h.calls) == req.total_rows


async def test_no_split_when_it_fits(handler):
    c = asgi_client(handler)
    await c.connect(FP)
    req, _ = make_request(RING_K40_A12, [("0.qkv", 5)])
    await c.compute(req)
    assert len(handler.calls) == 1 and handler.calls[0].request_id == req.request_id


async def test_split_row_too_large():
    c = ComputeClient(_cfg(max_request_bytes=4097 + 1024), transport=httpx.MockTransport(Scripted([])))
    item = ComputeItem("0.o", 1, b"\x00" * 10_000)
    with pytest.raises(PayloadTooLargeError):
        await c.compute(ComputeRequest("r", FP, "K40A12", (item,)))
    bad = ComputeItem("0.o", 3, b"\x00" * 10_000)
    with pytest.raises(ShapeError):
        await c.compute(ComputeRequest("r", FP, "K40A12", (bad, bad)))


async def test_split_failure_propagates_typed_error():
    h = FakeHandler()
    lim = WireLimits(max_rows_per_item=2)
    c = asgi_client(h, lim, max_rows_per_item=2)
    h.fail_with = ServiceUnavailableError("down")
    req, _ = make_request(RING_K40_A12, [("0.o", 6)])
    c2 = ComputeClient(_cfg(max_rows_per_item=2, max_retries=0),
                       transport=httpx.ASGITransport(app=make_app(h, lim)))
    with pytest.raises(ServiceUnavailableError):
        await c2.compute(req)
    await c.aclose()
    await c2.aclose()


# ------------------------------------------------------------------------------ sync API / lifecycle
def test_compute_sync_and_concurrent_threads(handler):
    c = asgi_client(handler)
    c.connect_sync(FP, "K40A12")
    results = []
    errors = []

    def worker(i: int) -> None:
        try:
            req, xs = make_request(RING_K40_A12, [("0.qkv", 2), ("0.down", 1)], rid=f"t{i}", seed=i)
            results.append((req, c.compute_sync(req), xs))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 6
    for req, resp, xs in results:
        _check_exact(req, resp, xs, RING_K40_A12, handler.weights)
    c.close()
    c.close()  # idempotent
    with pytest.raises(ServiceUnavailableError):
        c.compute_sync(make_request(RING_K40_A12, [("0.o", 1)])[0])


async def test_async_context_manager_and_closed(handler):
    async with asgi_client(handler) as c:
        await c.connect(FP)
    with pytest.raises(ServiceUnavailableError):
        await c.models()


async def test_mixed_sync_and_async_use(handler):
    c = asgi_client(handler)
    req, xs = make_request(RING_K40_A12, [("0.o", 1)])
    r1 = await asyncio.to_thread(c.compute_sync, req)
    r2 = await c.compute(req)
    assert r1 == r2
    await c.aclose()  # closes pools on both loops and stops the private thread
    assert c._loop is None


async def test_ready_endpoint():
    async def mock(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200 if request.url.path == "/readyz" else 404)

    c = ComputeClient(_cfg(), transport=httpx.MockTransport(mock))
    assert await c.ready()

    async def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("x")

    c2 = ComputeClient(_cfg(), transport=httpx.MockTransport(down))
    assert not await c2.ready()


async def test_on_attempt_hook():
    events = []
    script = Scripted([(503, ErrorResponse("unavailable", "x", True), {})])
    c = ComputeClient(_cfg(), transport=httpx.MockTransport(script),
                      on_attempt=lambda p, s, t, code: events.append((p, s, code)))
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    await c.compute(req)
    assert events == [("/pgw/v1/compute", 503, "unavailable"), ("/pgw/v1/compute", 200, None)]


async def test_engine_masked_backend_contract(handler):
    """The engine's MaskedBackend uses compute()/compute_sync() and expects request ids echoed."""
    c = asgi_client(handler)
    req = ComputeRequest("0123456789abcdef0123456789abcdef", FP, "K40A12",
                         (ComputeItem("0.o", 1, b"\x00" * 16 * 5),), deadline_ms=5000)
    resp = await c.compute(req)
    assert resp.request_id == req.request_id and resp.items[0].y_hat == b"\x00" * 16 * 5
