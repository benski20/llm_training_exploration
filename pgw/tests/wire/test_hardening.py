"""Hardening regressions: hostile peers, oversized responses, cancellation, auth warnings."""

from __future__ import annotations

import asyncio
import logging

import httpx
import msgpack
import pytest

from pgw.core import RING_K40_A12, ErrorCode, PGWError, ProtocolError, ServiceUnavailableError
from pgw.wire import (
    CONTENT_TYPE,
    PROTOCOL_HEADER,
    ClientConfig,
    ComputeClient,
    ComputeResponse,
    ModelsResponse,
    decode_any,
    decode_error_json,
    decode_message,
    encode_message,
)
from tests.wire.conftest import FP, FakeHandler, make_app, make_request

HDRS = {"Content-Type": CONTENT_TYPE, PROTOCOL_HEADER: "1"}


def _cfg(**kw) -> ClientConfig:
    base = {"base_url": "http://svc", "backoff_base_s": 0.001, "backoff_max_s": 0.005}
    base.update(kw)
    return ClientConfig(**base)


# ------------------------------------------------------------------------------ codec
def test_json_error_deep_nesting_is_protocol_error():
    with pytest.raises(ProtocolError):
        decode_error_json(b"[" * 200_000 + b"]" * 200_000, max_bytes=10**7)


@pytest.mark.parametrize("const", [b"NaN", b"Infinity", b"-Infinity"])
def test_json_error_rejects_non_standard_constants(const):
    body = b'{"v":1,"type":"error","code":"x","message":"m","details":{"a":' + const + b"}}"
    with pytest.raises(ProtocolError):
        decode_error_json(body)


def _models(linears, fps=(FP,)):
    return msgpack.packb(
        {
            "v": 1,
            "type": "models",
            "models": [
                {
                    "name": "m",
                    "fp": fp,
                    "rings": ["K40A12"],
                    "linears": linears,
                    "max_rows_per_item": 8,
                    "max_request_bytes": 1 << 20,
                }
                for fp in fps
            ],
        },
        use_bin_type=True,
    )


def test_models_duplicate_linear_rejected():
    ok = _models([{"id": "0.o", "m": 2, "n": 2}, {"id": "0.qkv", "m": 6, "n": 2}])
    assert isinstance(decode_any(ok), ModelsResponse)
    with pytest.raises(ProtocolError, match="duplicates"):
        decode_any(_models([{"id": "0.o", "m": 2, "n": 2}, {"id": "0.o", "m": 4, "n": 4}]))


def test_models_duplicate_fingerprint_rejected():
    with pytest.raises(ProtocolError, match="more than once"):
        decode_any(_models([{"id": "0.o", "m": 2, "n": 2}], fps=(FP, FP)))


# ------------------------------------------------------------------------------ client
async def test_oversized_success_response_is_rejected_without_buffering():
    sent = {"n": 0}

    async def stream():
        for _ in range(64):
            sent["n"] += 1
            yield b"\x00" * 65536

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream(), headers={"content-type": CONTENT_TYPE})

    c = ComputeClient(_cfg(max_response_bytes=100_000), transport=httpx.MockTransport(handler))
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    with pytest.raises(ProtocolError, match="exceeds"):
        await c.compute(req)
    assert sent["n"] < 64  # stopped reading early
    assert c.stats.failures == 1


async def test_oversized_error_body_falls_back_to_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, content=b"x" * (2 << 20), headers={"content-type": CONTENT_TYPE})

    c = ComputeClient(_cfg(), transport=httpx.MockTransport(handler))
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    with pytest.raises(PGWError) as ei:
        await c.compute(req)
    assert ei.value.code == ErrorCode.FINGERPRINT_MISMATCH
    assert ei.value.details["http_status"] == 409


async def test_client_disables_compression():
    seen: list[str | None] = []
    h = FakeHandler()

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("accept-encoding"))
        from pgw.wire import ComputeRequest

        r = decode_message(request.content, ComputeRequest)
        resp = await h.compute(r, None)  # type: ignore[arg-type]
        return httpx.Response(200, content=encode_message(resp), headers={"content-type": CONTENT_TYPE})

    c = ComputeClient(_cfg(), transport=httpx.MockTransport(handler))
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    assert isinstance(await c.compute(req), ComputeResponse)
    assert seen == ["identity"]


def test_sync_api_after_close_does_not_restart_loop():
    c = ComputeClient(_cfg(), transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    c.close()
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    with pytest.raises(ServiceUnavailableError):
        c.compute_sync(req)
    assert c._loop is None


def test_plaintext_token_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="pgw.wire.client"):
        ComputeClient(_cfg(token="secret-token-value"))
    assert any("plaintext" in r.getMessage() for r in caplog.records)
    assert not any("secret-token-value" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------------------ server
def test_router_without_auth_hook_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="pgw.wire.server"):
        make_app(FakeHandler())
    assert any("authenticate hook" in r.getMessage() for r in caplog.records)


async def test_router_client_disconnect_is_protocol_error():
    h = FakeHandler()
    app = make_app(h)
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    body = encode_message(req)
    sent: list[dict] = []
    msgs = [{"type": "http.request", "body": body[:10], "more_body": True}, {"type": "http.disconnect"}]

    async def receive():
        return msgs.pop(0) if msgs else {"type": "http.disconnect"}

    async def send(m):
        sent.append(m)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/pgw/v1/compute",
        "raw_path": b"/pgw/v1/compute",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", CONTENT_TYPE.encode()), (b"pgw-protocol", b"1")],
        "client": ("127.0.0.1", 1),
        "server": ("svc", 80),
    }
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 400
    assert h.done[-1] == (400, "protocol")


async def test_router_done_hook_sees_cancellation():
    h = FakeHandler(delay_s=5.0)
    app = make_app(h)
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])

    async def post():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as c:
            await c.post("/pgw/v1/compute", content=encode_message(req), headers=HDRS)

    task = asyncio.create_task(post())
    for _ in range(200):
        await asyncio.sleep(0.01)
        if h.calls:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.done and h.done[-1][0] == 499


def test_from_exception_hides_internal_pgwerror():
    from pgw.wire import ErrorResponse

    err = ErrorResponse.from_exception(PGWError("secret path /etc/x", details={"k": "v"}), "rid-1")
    assert err.code == "internal" and "secret" not in err.message and err.details == {}
    assert err.request_id == "rid-1"
