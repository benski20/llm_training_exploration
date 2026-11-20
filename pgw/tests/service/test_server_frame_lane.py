"""The service's frame lane over a real uvicorn socket (``server="uvicorn"``, the stream protocol of
:func:`pgw.service.server.stream_ws_protocol`), against the same service with the lane switched off.

Every frame gets the same answer from both (results bit-identical to the reference, errors
byte-identical), tenant quotas and metrics move the same way, and draining refuses the same way.
"""

from __future__ import annotations

import asyncio
import re
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

from pgw.core import RingParams, unpack_ring
from pgw.service import ServiceSettings, create_app
from pgw.service.server import serve_async
from pgw.wire import ComputeItem, ComputeRequest, ComputeResponse, ErrorResponse
from pgw.wire.codec import decode_any, encode_message
from tests.service.conftest import TOKEN_A, expected, make_req, rand_x

RING = RingParams.from_name("K40A12")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Service:
    """The service app (tiny pack, CPU) served by uvicorn in a thread."""

    def __init__(self, pack_dir: Path, tokens_file: Path, *, lane: bool, monkeypatch: pytest.MonkeyPatch,
                 **settings: Any):
        monkeypatch.setenv("PGW_SERVICE_FRAME_LANE", "1" if lane else "0")
        self.port = _free_port()
        self.settings = ServiceSettings(host="127.0.0.1", port=self.port, server="uvicorn", packs=[pack_dir],
                                        device="cpu", tokens_file=tokens_file, **settings)
        self.app = create_app(self.settings)  # reads the lane switch now
        self.loop = asyncio.new_event_loop()
        self.stop_event: asyncio.Event | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/readyz", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise TimeoutError("service did not become ready")

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)

        async def main() -> None:
            self.stop_event = asyncio.Event()
            await serve_async(self.settings, shutdown_event=self.stop_event, app=self.app)

        try:
            self.loop.run_until_complete(main())
        except BaseException as exc:  # surfaced by the fixture
            self.error = exc

    @property
    def handler(self) -> Any:
        return self.app.state.pgw.handler

    def metric_lines(self, *names: str) -> list[str]:
        body, _ = self.app.state.pgw.metrics.render()
        return sorted(ln for ln in body.decode().splitlines() if ln.startswith(names) and not ln.startswith("#"))

    def stop(self) -> None:
        if self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(timeout=30)
        assert self.error is None, self.error


@pytest.fixture
def services(tiny_pack_dir: Path, tokens_file: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    started: list[_Service] = []

    def start(**settings: Any) -> tuple[_Service, _Service]:
        pair = (_Service(tiny_pack_dir, tokens_file, lane=True, monkeypatch=monkeypatch, **settings),
                _Service(tiny_pack_dir, tokens_file, lane=False, monkeypatch=monkeypatch, **settings))
        started.extend(pair)
        return pair

    yield start
    for s in started:
        s.stop()


async def _exchange(port: int, frames: list[bytes], token: str = TOKEN_A) -> list[bytes]:
    from websockets.asyncio.client import connect

    out = []
    async with connect(f"ws://127.0.0.1:{port}/pgw/v1/stream", additional_headers={"Authorization": f"Bearer {token}"},
                       compression=None, max_size=1 << 26) as ws:
        for f in frames:
            await ws.send(f)
            out.append(await ws.recv())
    return out


def _same(a: bytes, b: bytes) -> None:
    """Results equal (timings aside, they are measured); errors byte-identical."""
    da, db = decode_any(a), decode_any(b)
    if isinstance(da, ComputeResponse):
        assert isinstance(db, ComputeResponse)
        assert (da.request_id, da.items) == (db.request_id, db.items)
    else:
        assert isinstance(da, ErrorResponse) and a == b


def _frames(fp: str, dims: dict[str, tuple[int, int]]) -> tuple[list[bytes], list[Any]]:
    frames, reqs = [], []
    for i, (lid, (_m, n)) in enumerate(sorted(dims.items())):
        for rows in (1, 3, 20):  # 1 and 3 rows: the lane; 20 rows: above batch_inline_max_rows
            req = make_req(fp, RING, [(lid, rand_x(rows, n, RING, seed=i * 10 + rows))], rid=f"{lid}-{rows}")
            frames.append(encode_message(req))
            reqs.append(req)
    lids = sorted(dims)
    multi = make_req(fp, RING, [(lids[0], rand_x(1, dims[lids[0]][1], RING, 1)),
                                (lids[1], rand_x(1, dims[lids[1]][1], RING, 2))], rid="multi")
    frames.append(encode_message(multi))
    reqs.append(multi)
    lid0, (_, n0) = lids[0], dims[lids[0]]
    frames.append(encode_message(make_req("f" * 64, RING, [(lid0, rand_x(1, n0, RING))], rid="badfp")))
    frames.append(encode_message(make_req(fp, RING, [(lid0, rand_x(1, n0, RING))], rid="late", deadline_ms=0)))
    good = make_req(fp, RING, [(lid0, rand_x(1, n0, RING))], rid="short")
    it = good.items[0]
    frames.append(encode_message(ComputeRequest(good.request_id, fp, good.ring,
                                                (ComputeItem(it.linear, it.rows, it.x_hat[:-5]),))))
    return frames, reqs


async def test_lane_service_answers_like_the_regular_path(services: Any, pack_fp: str,
                                                          pack_weights: dict[str, Any]) -> None:
    on, off = services()
    dims = dict(on.app.state.pgw.registry.get(pack_fp).linear_dims)
    frames, reqs = _frames(pack_fp, dims)
    r_on = await _exchange(on.port, frames)
    r_off = await _exchange(off.port, frames)
    for a, b in zip(r_on, r_off, strict=True):
        _same(a, b)
    for req, raw in zip(reqs, r_on, strict=False):  # results are the exact ring product
        resp = decode_any(raw)
        assert isinstance(resp, ComputeResponse)
        for it, res in zip(req.items, resp.items, strict=True):
            m, n = dims[it.linear]
            x = unpack_ring(it.x_hat, (it.rows, n), RING)
            assert (unpack_ring(res.y_hat, (it.rows, m), RING) == expected(x, pack_weights[it.linear], RING)).all()
    n_lane = 2 * len(dims)  # the 1- and 3-row single-item requests
    assert on.handler.inline_requests == n_lane and off.handler.inline_requests == 0
    names = ("pgw_service_requests_total", "pgw_service_rows_total", "pgw_service_bytes_in_total",
             "pgw_service_bytes_out_total")
    assert on.metric_lines(*names) == off.metric_lines(*names)
    assert on.handler.inflight == 0 and on.app.state.pgw.batcher.queued_rows == 0


async def test_lane_service_quota_and_drain(services: Any, pack_fp: str) -> None:
    on, off = services(tenant_rows_per_s=0.001, tenant_burst_rows=3)
    dims = dict(on.app.state.pgw.registry.get(pack_fp).linear_dims)
    lid, (_m, n) = sorted(dims.items())[0]
    frames = [encode_message(make_req(pack_fp, RING, [(lid, rand_x(1, n, RING, i))], rid=f"q{i}")) for i in range(5)]

    def codes(raw: list[bytes]) -> list[str]:
        return ["ok" if isinstance(d := decode_any(r), ComputeResponse) else d.code for r in raw]

    assert codes(await _exchange(on.port, frames)) == codes(await _exchange(off.port, frames)) == [
        "ok", "ok", "ok", "rate_limited", "rate_limited"]
    assert on.handler.inline_requests == 3
    req_lines = [ln for ln in on.metric_lines("pgw_service_requests_total") if 'tenant="tenant-a"' in ln]
    assert any(re.search(r'code="rate_limited".* 2\.0$', ln) for ln in req_lines)
    # draining: the lane steps aside and the regular path refuses with 'unavailable'
    for s in (on, off):
        s.handler.draining = True
    frame = [encode_message(make_req(pack_fp, RING, [(lid, rand_x(1, n, RING, 9))], rid="d"))]
    from websockets.exceptions import InvalidStatus

    for s in (on, off):  # new channels are refused at the handshake while draining
        with pytest.raises(InvalidStatus):
            await _exchange(s.port, frame)
    for s in (on, off):
        s.handler.draining = False


async def test_lane_service_with_compute_client(services: Any, pack_fp: str, pack_weights: dict[str, Any]) -> None:
    from pydantic import SecretStr

    from pgw.wire import ClientConfig, ComputeClient

    on, _off = services()
    dims = dict(on.app.state.pgw.registry.get(pack_fp).linear_dims)
    client = ComputeClient(ClientConfig(base_url=f"http://127.0.0.1:{on.port}", token=SecretStr(TOKEN_A)))
    await client.connect(pack_fp, RING.name)
    lids = sorted(dims)
    for i in range(3 * len(lids)):
        lid = lids[i % len(lids)]
        m, n = dims[lid]
        rows = (1, 2, 300)[i % 3]
        x = rand_x(rows, n, RING, seed=i)
        resp = await client.compute(make_req(pack_fp, RING, [(lid, x)], rid=f"c{i}"))
        assert (unpack_ring(resp.items[0].y_hat, (rows, m), RING) == expected(x, pack_weights[lid], RING)).all()
    assert client.stats.lane_requests == 3 * len(lids) - 1  # the first call opened the channel
    assert on.handler.inline_requests > 0
    await client.aclose()
