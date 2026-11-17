"""Server-side helpers and the FastAPI router (SPEC 6.2-6.6)."""

from __future__ import annotations

import json

import httpx
import msgpack
import pytest

from pgw.core import (
    RING_K40_A12,
    AuthError,
    ErrorCode,
    PayloadTooLargeError,
    PGWError,
    ProtocolError,
    RateLimitedError,
    ServiceUnavailableError,
    ShapeError,
    VersionMismatchError,
    unpack_ring,
)
from pgw.wire import (
    CONTENT_TYPE,
    PROTOCOL_HEADER,
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    ErrorResponse,
    ModelsResponse,
    WireLimits,
    check_content_type,
    check_protocol_header,
    decode_compute_request,
    decode_message,
    encode_message,
    error_headers,
    error_response_bytes,
    validate_compute_response,
)
from tests.wire.conftest import DIMS, FP, AuthHandler, FakeHandler, expected_y, make_app, make_request

HDRS = {"Content-Type": CONTENT_TYPE, PROTOCOL_HEADER: "1"}


# ------------------------------------------------------------------------------ pure helpers
def test_check_protocol_header():
    assert check_protocol_header(None) == 1
    assert check_protocol_header("") == 1
    assert check_protocol_header(" 1 ") == 1
    with pytest.raises(VersionMismatchError):
        check_protocol_header("2")
    with pytest.raises(ProtocolError):
        check_protocol_header("one")


def test_check_content_type():
    check_content_type(CONTENT_TYPE)
    check_content_type("Application/VND.pgw+msgpack; charset=binary")
    for bad in (None, "", "application/json", "application/msgpack"):
        with pytest.raises(ProtocolError):
            check_content_type(bad)


def test_decode_compute_request_ok_and_limits():
    req, _ = make_request(RING_K40_A12, [("0.qkv", 3), ("0.down", 2)])
    body = encode_message(req)
    assert decode_compute_request(body, WireLimits(), DIMS) == req
    # resolver form
    seen = []

    def resolver(fp, ring):
        seen.append((fp, ring))
        return DIMS

    decode_compute_request(body, WireLimits(), resolver)
    assert seen == [(FP, "K40A12")]
    with pytest.raises(PayloadTooLargeError):
        decode_compute_request(body, WireLimits(max_request_bytes=len(body) - 1), DIMS)
    with pytest.raises(PayloadTooLargeError):
        decode_compute_request(body, WireLimits(max_items=1), DIMS)
    with pytest.raises(PayloadTooLargeError):
        decode_compute_request(body, WireLimits(max_rows_per_item=2), DIMS)
    with pytest.raises(ProtocolError, match="unknown linear"):
        decode_compute_request(body, WireLimits(), {"0.qkv": DIMS["0.qkv"]})
    with pytest.raises(ShapeError):
        decode_compute_request(body, WireLimits(), {**DIMS, "0.down": (16, 21)})
    with pytest.raises(ValueError):
        WireLimits(max_items=0)


def test_validate_compute_response():
    req, _ = make_request(RING_K40_A12, [("0.o", 2)])
    good = ComputeResponse(req.request_id, (ComputeResult("0.o", 2, b"\x00" * (2 * 16 * 5)),))
    validate_compute_response(req, good, {"0.o": 16})
    validate_compute_response(req, good)
    bad_cases = [
        ComputeResponse("other", good.items),
        ComputeResponse(req.request_id, ()),
        ComputeResponse(req.request_id, (ComputeResult("0.qkv", 2, good.items[0].y_hat),)),
        ComputeResponse(req.request_id, (ComputeResult("0.o", 1, good.items[0].y_hat),)),
        ComputeResponse(req.request_id, (ComputeResult("0.o", 2, good.items[0].y_hat[:-5]),)),
    ]
    for bad in bad_cases:
        with pytest.raises(ProtocolError):
            validate_compute_response(req, bad, {"0.o": 16})
    with pytest.raises(ProtocolError):
        validate_compute_response(req, ComputeResponse(req.request_id, (ComputeResult("0.o", 2, b"\x00" * 7),)))


@pytest.mark.parametrize(
    "exc,status,code",
    [
        (ShapeError("s"), 400, "bad_shape"),
        (ProtocolError("p"), 400, "protocol"),
        (AuthError("a"), 401, "unauthenticated"),
        (PayloadTooLargeError("big"), 413, "payload_too_large"),
        (VersionMismatchError("v"), 426, "version_mismatch"),
        (RateLimitedError("r"), 429, "rate_limited"),
        (ServiceUnavailableError("u"), 503, "unavailable"),
        (PGWError("d", code=ErrorCode.DEADLINE_EXCEEDED), 504, "deadline_exceeded"),
        (RuntimeError("leaky secret"), 500, "internal"),
        (PGWError("internal leaky"), 500, "internal"),
    ],
)
def test_error_response_bytes(exc, status, code):
    st, body = error_response_bytes(exc, "rid")
    assert st == status
    err = decode_message(body, ErrorResponse)
    assert err.code == code and err.request_id == "rid"
    assert "leaky" not in err.message
    st2, jbody = error_response_bytes(exc, None, json_body=True)
    assert st2 == status and json.loads(jbody)["code"] == code


def test_error_headers_retry_after():
    assert error_headers(RateLimitedError("r", details={"retry_after_s": 2.2}))["Retry-After"] == "3"
    assert error_headers(ServiceUnavailableError("u"))["Retry-After"] == "1"
    assert "Retry-After" not in error_headers(ShapeError("s"))
    assert error_headers(ValueError())[PROTOCOL_HEADER] == "1"


# ------------------------------------------------------------------------------ router
def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc")


async def test_router_compute_exact(handler, any_ring):
    app = make_app(handler)
    req, xs = make_request(any_ring, [("0.qkv", 3), ("0.gate_up", 2), ("0.down", 1)])
    async with _client(app) as c:
        r = await c.post("/pgw/v1/compute", content=encode_message(req), headers={**HDRS, "PGW-Request-Id": "req-1"})
    assert r.status_code == 200, r.content
    assert r.headers["content-type"].startswith(CONTENT_TYPE)
    assert r.headers[PROTOCOL_HEADER] == "1" and r.headers["PGW-Request-Id"] == "req-1"
    resp = decode_message(r.content, ComputeResponse)
    for it, x, res in zip(req.items, xs, resp.items, strict=True):
        m = DIMS[it.linear][0]
        y = unpack_ring(res.y_hat, (it.rows, m), any_ring)
        assert (y == expected_y(any_ring, it.linear, x, handler.weights)).all()
    assert handler.done == [(200, None)]
    assert handler.contexts[0].body_bytes == len(encode_message(req))


async def test_router_models(handler):
    async with _client(make_app(handler)) as c:
        r = await c.get("/pgw/v1/models")
    assert r.status_code == 200
    m = decode_message(r.content, ModelsResponse)
    assert m.find(FP) is not None and m.models[0].linear_dims() == DIMS


async def _post(app, body: bytes, headers=None):
    async with _client(app) as c:
        return await c.post("/pgw/v1/compute", content=body, headers=headers or HDRS)


async def test_router_rejections(handler):
    app = make_app(handler, WireLimits(max_request_bytes=4096, max_items=4, max_rows_per_item=16))
    req, _ = make_request(RING_K40_A12, [("0.o", 2)])
    body = encode_message(req)

    async def code_of(r) -> str:
        return decode_message(r.content, ErrorResponse).code

    r = await _post(app, body, {**HDRS, PROTOCOL_HEADER: "7"})
    assert r.status_code == 426 and await code_of(r) == "version_mismatch"
    assert decode_message(r.content, ErrorResponse).details["versions"] == [1]

    r = await _post(app, body, {"Content-Type": "application/json"})
    assert r.status_code == 400 and await code_of(r) == "protocol"

    r = await _post(app, b"\x00" * 5000)
    assert r.status_code == 413 and await code_of(r) == "payload_too_large"

    r = await _post(app, b"garbage")
    assert r.status_code == 400 and await code_of(r) == "protocol"

    big, _ = make_request(RING_K40_A12, [("0.o", 17)])
    r = await _post(app, encode_message(big))
    assert r.status_code == 413

    wrong_fp, _ = make_request(RING_K40_A12, [("0.o", 1)], fp="cd" * 32)
    r = await _post(app, encode_message(wrong_fp))
    assert r.status_code == 409 and await code_of(r) == "fingerprint_mismatch"

    d = msgpack.unpackb(body, raw=False)
    d["items"][0]["x_hat"] = d["items"][0]["x_hat"][:-1]
    r = await _post(app, msgpack.packb(d, use_bin_type=True))
    assert r.status_code == 400 and await code_of(r) == "bad_shape"
    assert d["id"] == decode_message(r.content, ErrorResponse).request_id
    assert not handler.calls  # nothing reached compute


async def test_router_streamed_body_over_limit(handler):
    app = make_app(handler, WireLimits(max_request_bytes=4096))

    async def gen():
        for _ in range(10):
            yield b"\x00" * 1000

    async with _client(app) as c:
        r = await c.post("/pgw/v1/compute", content=gen(), headers=HDRS)
    assert r.status_code == 413


async def test_router_internal_errors_not_leaked(handler):
    handler.fail_with = RuntimeError("secret internals")
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    r = await _post(make_app(handler), encode_message(req))
    assert r.status_code == 500
    err = decode_message(r.content, ErrorResponse)
    assert err.code == "internal" and "secret" not in err.message and err.request_id == req.request_id
    assert handler.done == [(500, "internal")]


async def test_router_typed_errors_and_retry_after(handler):
    handler.fail_with = ServiceUnavailableError("draining", details={"retry_after_s": 4})
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    r = await _post(make_app(handler), encode_message(req))
    assert r.status_code == 503 and r.headers["Retry-After"] == "4"
    err = decode_message(r.content, ErrorResponse)
    assert err.retryable and err.code == "unavailable"


async def test_router_json_errors_for_json_clients(handler):
    r = await _post(make_app(handler), b"x", {**HDRS, "Accept": "application/json"})
    assert r.status_code == 400 and r.headers["content-type"].startswith("application/json")
    assert r.json()["code"] == "protocol"


async def test_router_self_check_catches_bad_handler(handler):
    handler.corrupt = True
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    r = await _post(make_app(handler), encode_message(req))
    assert r.status_code == 500 and decode_message(r.content, ErrorResponse).code == "internal"


async def test_router_deadline(handler):
    handler.delay_s = 0.5
    req, _ = make_request(RING_K40_A12, [("0.o", 1)], deadline_ms=50)
    r = await _post(make_app(handler), encode_message(req))
    assert r.status_code == 504 and decode_message(r.content, ErrorResponse).code == "deadline_exceeded"
    handler.delay_s = 0.3
    req2, _ = make_request(RING_K40_A12, [("0.o", 1)])
    r = await _post(make_app(handler, max_deadline_s=0.05), encode_message(req2))
    assert r.status_code == 504


async def test_router_auth():
    h = AuthHandler()
    app = make_app(h)
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    r = await _post(app, encode_message(req))
    assert r.status_code == 401 and decode_message(r.content, ErrorResponse).code == "unauthenticated"
    async with _client(app) as c:
        r = await c.get("/pgw/v1/models")
        assert r.status_code == 401
        r = await c.get("/pgw/v1/models", headers={"Authorization": "Bearer good-token"})
        assert r.status_code == 200
    r = await _post(app, encode_message(req), {**HDRS, "Authorization": "Bearer good-token", "PGW-Attempt": "3"})
    assert r.status_code == 200
    assert h.contexts[-1].identity == {"tenant": "t1"} and h.contexts[-1].attempt == 3


async def test_router_models_version_check(handler):
    async with _client(make_app(handler)) as c:
        r = await c.get("/pgw/v1/models", headers={PROTOCOL_HEADER: "2"})
    assert r.status_code == 426


async def test_router_done_hook_errors_are_swallowed():
    class H(FakeHandler):
        def on_request_done(self, ctx, status, code):
            raise RuntimeError("metrics broke")

    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    r = await _post(make_app(H()), encode_message(req))
    assert r.status_code == 200


def test_request_is_pure_function_of_body(handler):
    """Identical bodies give identical results (retries are safe)."""
    import asyncio

    req, _ = make_request(RING_K40_A12, [("0.qkv", 2)])

    async def go():
        app = make_app(handler)
        a = await _post(app, encode_message(req))
        b = await _post(app, encode_message(req))
        return a.content, b.content

    a, b = asyncio.run(go())
    assert a == b
    assert isinstance(decode_message(a, ComputeResponse), ComputeResponse)
    assert isinstance(req, ComputeRequest)
