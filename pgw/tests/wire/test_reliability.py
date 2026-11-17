"""Reliability review regressions: admission before body buffering, per-attempt deadlines, retry budget."""

from __future__ import annotations

import httpx
import pytest

from pgw.core import RING_K40_A12, ErrorCode, PGWError, ServiceUnavailableError
from pgw.wire import (
    CONTENT_TYPE,
    DEADLINE_HEADER,
    PROTOCOL_HEADER,
    ClientConfig,
    ComputeClient,
    ErrorResponse,
    encode_message,
)
from tests.wire.conftest import FakeHandler, make_app, make_request

HDRS = {"Content-Type": CONTENT_TYPE, PROTOCOL_HEADER: "1"}


async def _post(app, body: bytes, headers=None) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as c:
        return await c.post("/pgw/v1/compute", content=body, headers=headers or HDRS)


class _DrainingHandler(FakeHandler):
    def __init__(self) -> None:
        super().__init__()
        self.prechecks = 0
        self.draining = False

    def precheck(self) -> None:
        self.prechecks += 1
        if self.draining:
            raise ServiceUnavailableError("service is draining", details={"retry_after_s": 2.0})


async def test_precheck_runs_before_the_body_is_decoded() -> None:
    h = _DrainingHandler()
    app = make_app(h)
    req, _ = make_request(RING_K40_A12, [("0.o", 2)])
    body = encode_message(req)
    assert (await _post(app, body)).status_code == 200
    h.draining = True
    r = await _post(app, body)
    assert r.status_code == 503 and r.headers.get("retry-after") == "2"
    assert h.prechecks == 2 and len(h.calls) == 1


async def test_inflight_byte_budget_refuses_before_buffering() -> None:
    h = FakeHandler()
    req, _ = make_request(RING_K40_A12, [("0.o", 4)])
    body = encode_message(req)
    app = make_app(h, max_inflight_bytes=len(body) - 1)
    r = await _post(app, body)
    assert r.status_code == 503 and not h.calls
    app = make_app(h, max_inflight_bytes=len(body))
    for _ in range(3):  # the budget is released after every request
        assert (await _post(app, body)).status_code == 200


async def test_deadline_header_overrides_body_deadline() -> None:
    h = FakeHandler(delay_s=0.3)
    app = make_app(h)
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    req = type(req)(req.request_id, req.fingerprint, req.ring, req.items, deadline_ms=60_000)
    r = await _post(app, encode_message(req), {**HDRS, DEADLINE_HEADER: "50"})
    assert r.status_code == 504 or b"deadline" in r.content
    # malformed header is ignored (body deadline applies)
    h2 = FakeHandler()
    r = await _post(make_app(h2), encode_message(req), {**HDRS, DEADLINE_HEADER: "nan"})
    assert r.status_code == 200
    ctx = h2.contexts[0]
    assert ctx.deadline is not None and ctx.deadline - ctx.received_at == pytest.approx(60.0, abs=0.5)


class _Busy:
    def __init__(self) -> None:
        self.deadlines: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.deadlines.append(int(request.headers[DEADLINE_HEADER]))
        body = encode_message(ErrorResponse("unavailable", "busy", True))
        return httpx.Response(503, content=body, headers={"content-type": CONTENT_TYPE})


async def test_client_sends_remaining_budget_and_respects_retry_budget() -> None:
    busy = _Busy()

    async def no_sleep(s: float) -> None:
        return None

    cfg = ClientConfig(base_url="http://svc", max_retries=5, backoff_base_s=0.0, backoff_max_s=0.0,
                       request_timeout_s=30, retry_budget_burst=2, retry_budget_ratio=0.0)
    c = ComputeClient(cfg, transport=httpx.MockTransport(busy), sleep=no_sleep)
    req, _ = make_request(RING_K40_A12, [("0.o", 1)])
    with pytest.raises(PGWError) as ei:
        await c.compute(req)
    assert ei.value.code == ErrorCode.UNAVAILABLE
    assert len(busy.deadlines) == 3  # 1 attempt + 2 budgeted retries (not 1 + max_retries)
    assert all(0 < d <= 30_000 for d in busy.deadlines)
    assert busy.deadlines == sorted(busy.deadlines, reverse=True)
    with pytest.raises(PGWError):
        await c.compute(req)  # budget spent: no retries at all now
    assert len(busy.deadlines) == 4 and c.stats.retries_denied == 2
