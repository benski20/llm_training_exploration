"""Grouped kernel per linear: same-linear items of one request share one kernel call, bit-identically."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    LinearId,
    LinearKind,
    LinearSpec,
    PGWError,
    RingParams,
    ShapeError,
    unpack_ring,
)
from pgw.service import BatchScheduler, ModelRegistry, RegisteredModel, ServiceSettings
from pgw.service.batcher import _chunk_parts
from pgw.service.handler import ServiceHandler
from pgw.service.kernels import ComputeKernel
from pgw.service.metrics import ServiceMetrics
from pgw.wire.server import RequestContext
from tests.service.conftest import expected, make_req, rand_x

FP = "ab" * 32
QKV = "0.qkv"
O_ = "0.o"


class _RecKernel(ComputeKernel):
    """Records the row count of every kernel call; optionally fails, sleeps or claims a fused GEMV."""

    def __init__(self, *a: Any, fail: bool = False, delay: float = 0.0, gemv: bool = False, **kw: Any):
        super().__init__(*a, **kw)
        self.fail = fail
        self.delay = delay
        self.gemv = gemv
        self.calls: list[int] = []

    @property
    def triton_gemv(self) -> bool:
        return self.gemv

    def run(self, x_hat: torch.Tensor, ring: Any) -> torch.Tensor:
        self.calls.append(int(x_hat.shape[0]))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom (secret details)")
        return super().run(x_hat, ring)


def _registry(**kw: Any) -> tuple[ModelRegistry, dict[str, torch.Tensor], dict[str, _RecKernel]]:
    reg = ModelRegistry("cpu", verify="fingerprint")
    g = torch.Generator().manual_seed(1)
    specs = [LinearSpec(LinearId(0, LinearKind.QKV), 24, 16), LinearSpec(LinearId(0, LinearKind.O), 16, 8)]
    ws: dict[str, torch.Tensor] = {}
    kernels: dict[str, _RecKernel] = {}
    for s in specs:
        w = torch.randint(-127, 128, (s.out_features, s.in_features), dtype=torch.int8, generator=g)
        ws[s.id.key] = w
        kernels[s.id.key] = _RecKernel(s, w, device="cpu", **kw)
    reg.register(RegisteredModel("synthetic", FP, specs, kernels, rings=("K40A12", "K32A8")))  # type: ignore[arg-type]
    return reg, ws, kernels


def _handler(reg: ModelRegistry, *, inline_max_rows: int = 8, max_batch_rows: int = 4096,
             group_min_rows: int | None = None, cpu_workers: int = 2) -> tuple[ServiceHandler, BatchScheduler]:
    settings = ServiceSettings(allow_unauthenticated=True)
    metrics = ServiceMetrics(enabled=False)
    b = BatchScheduler(reg, max_batch_rows=max_batch_rows, inline_max_rows=inline_max_rows, metrics=metrics,
                       cpu_workers=cpu_workers)
    return ServiceHandler(settings, reg, b, None, metrics, group_min_rows=group_min_rows), b


def _y(res: Any, rows: int, m: int, ring: RingParams) -> torch.Tensor:
    return unpack_ring(res.y_hat, (rows, m), ring)


@pytest.mark.parametrize("ring", [RING_K40_A12, RING_K32_A8], ids=lambda r: r.name)
@pytest.mark.parametrize("inline_max_rows", [8, 0])
async def test_grouped_request_matches_single_item_requests(ring: RingParams, inline_max_rows: int) -> None:
    reg, ws, kernels = _registry()
    h, b = _handler(reg, inline_max_rows=inline_max_rows)
    await b.start()
    try:
        # 4 items on qkv (1, 3, 1, 2 rows) interleaved with 1 on o
        items = [(QKV, rand_x(1, 16, ring, 1)), (O_, rand_x(2, 8, ring, 2)), (QKV, rand_x(3, 16, ring, 3)),
                 (QKV, rand_x(1, 16, ring, 4)), (QKV, rand_x(2, 16, ring, 5))]
        batches0 = b.stats.batches
        resp = await h.compute(make_req(FP, ring, items, rid="grouped"), RequestContext("grouped"))
        assert b.stats.batches - batches0 == 2  # one kernel call per linear, not 5
        assert kernels[QKV].calls == [7] and kernels[O_].calls == [2]
        assert resp.request_id == "grouped"
        assert [r.linear for r in resp.items] == [lid for lid, _ in items]
        singles = []
        for k, (lid, x) in enumerate(items):
            one = await h.compute(make_req(FP, ring, [(lid, x)], rid=f"s{k}"), RequestContext(f"s{k}"))
            singles.append(one.items[0])
        for (lid, x), g, s in zip(items, resp.items, singles, strict=True):
            assert g.rows == s.rows == x.shape[0]
            assert g.y_hat == s.y_hat  # byte-identical
            m = ws[lid].shape[0]
            assert torch.equal(_y(g, int(x.shape[0]), m, ring), expected(x, ws[lid], ring))
        assert resp.timing.batch_rows == 7
    finally:
        await b.stop()


async def test_spec_sized_group_above_inline_max_rows_uses_queued_path_and_chunks() -> None:
    ring = RING_K40_A12
    reg, ws, kernels = _registry()
    h, b = _handler(reg, inline_max_rows=8, max_batch_rows=10)
    await b.start()
    try:
        # 5 items of 5 rows (25 rows > inline_max_rows and > max_batch_rows): chunked 10 + 10 + 5
        xs = [rand_x(5, 16, ring, 10 + i) for i in range(5)]
        resp = await h.compute(make_req(FP, ring, [(QKV, x) for x in xs]), RequestContext("r1"))
        assert kernels[QKV].calls == [10, 10, 5]
        for x, r in zip(xs, resp.items, strict=True):
            assert torch.equal(_y(r, 5, 24, ring), expected(x, ws[QKV], ring))
        assert b.queued_rows == 0 and b.queued_items == 0
    finally:
        await b.stop()


async def test_group_of_two_one_row_items_on_fused_gemv_kernel_stays_split() -> None:
    ring = RING_K40_A12
    reg, ws, kernels = _registry(gemv=True)
    h, b = _handler(reg)  # default group_min_rows = 3
    await b.start()
    try:
        xs = [rand_x(1, 16, ring, 20 + i) for i in range(2)]
        resp = await h.compute(make_req(FP, ring, [(QKV, x) for x in xs]), RequestContext("r"))
        assert kernels[QKV].calls == [1, 1]  # two fused one-row calls
        for x, r in zip(xs, resp.items, strict=True):
            assert torch.equal(_y(r, 1, 24, ring), expected(x, ws[QKV], ring))
        # three rows reach the threshold: one call
        xs3 = [rand_x(1, 16, ring, 30 + i) for i in range(3)]
        await h.compute(make_req(FP, ring, [(QKV, x) for x in xs3]), RequestContext("r3"))
        assert kernels[QKV].calls == [1, 1, 3]
    finally:
        await b.stop()
    # without a fused GEMV the two rows are grouped
    reg2, _, kernels2 = _registry(gemv=False)
    h2, b2 = _handler(reg2)
    await b2.start()
    try:
        await h2.compute(make_req(FP, ring, [(QKV, x) for x in xs]), RequestContext("r"))
        assert kernels2[QKV].calls == [2]
    finally:
        await b2.stop()


async def test_group_min_rows_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    reg, _, _ = _registry()
    monkeypatch.setenv("PGW_SERVICE_GROUP_MIN_ROWS", "5")
    assert _handler(reg)[0].group_min_rows == 5
    monkeypatch.setenv("PGW_SERVICE_GROUP_MIN_ROWS", "nope")
    assert _handler(reg)[0].group_min_rows == 3
    assert _handler(reg, group_min_rows=2)[0].group_min_rows == 2


async def test_group_min_rows_zero_turns_grouping_off() -> None:
    ring = RING_K40_A12
    reg, ws, kernels = _registry()
    h, b = _handler(reg, group_min_rows=0)
    await b.start()
    try:
        xs = [rand_x(2, 16, ring, 40 + i) for i in range(3)]
        resp = await h.compute(make_req(FP, ring, [(QKV, x) for x in xs]), RequestContext("r"))
        assert kernels[QKV].calls == [2, 2, 2]
        for x, r in zip(xs, resp.items, strict=True):
            assert torch.equal(_y(r, 2, 24, ring), expected(x, ws[QKV], ring))
    finally:
        await b.stop()


@pytest.mark.parametrize("inline_max_rows", [8, 0])
async def test_kernel_error_fails_the_group(inline_max_rows: int) -> None:
    ring = RING_K40_A12
    reg, _, kernels = _registry(fail=True)
    h, b = _handler(reg, inline_max_rows=inline_max_rows)
    await b.start()
    try:
        items = [(QKV, rand_x(2, 16, ring, 1)), (QKV, rand_x(2, 16, ring, 2)), (O_, rand_x(1, 8, ring, 3))]
        with pytest.raises(PGWError) as ei:
            await h.compute(make_req(FP, ring, items, rid="bad"), RequestContext("bad"))
        assert "secret" not in ei.value.message
        assert kernels[QKV].calls == [4]
        assert b.stats.failed_batches >= 1
        assert b.queued_rows == 0 and h.inflight == 0
    finally:
        await b.stop()


async def test_submit_parts_wrong_shape_fails_whole_group() -> None:
    ring = RING_K40_A12
    reg, _, kernels = _registry()
    _, b = _handler(reg)
    await b.start()
    try:
        with pytest.raises(ShapeError) as ei:
            await b.submit_parts(FP, ring, QKV, [rand_x(1, 16, ring), rand_x(1, 15, ring)])
        assert ei.value.details.get("part") == 1
        assert kernels[QKV].calls == [] and b.queued_rows == 0
    finally:
        await b.stop()


async def test_submit_parts_empty_blocks_and_compact_flag() -> None:
    ring = RING_K40_A12
    reg, ws, kernels = _registry()
    _, b = _handler(reg, inline_max_rows=0)
    await b.start()
    try:
        x1, x2 = rand_x(3, 16, ring, 7), rand_x(2, 16, ring, 8)
        empty = torch.zeros((0, 16), dtype=torch.int64)
        ys, t = await b.submit_parts(FP, ring, QKV, [x1, empty, x2])
        assert [tuple(y.shape) for y in ys] == [(3, 24), (0, 24), (2, 24)]
        assert torch.equal(ys[0], expected(x1, ws[QKV], ring)) and torch.equal(ys[2], expected(x2, ws[QKV], ring))
        assert t.batch_rows == 5 and kernels[QKV].calls == [5]
        ys0, _ = await b.submit_parts(FP, ring, QKV, [empty])
        assert ys0[0].shape == (0, 24) and kernels[QKV].calls == [5]
    finally:
        await b.stop()


async def test_submit_parts_joins_other_requests_batch() -> None:
    """A group may share a batch with another request's rows (queued path), results split correctly."""
    ring = RING_K40_A12
    reg, ws, kernels = _registry(delay=0.05)
    _, b = _handler(reg, inline_max_rows=0, cpu_workers=1)
    await b.start()
    try:
        blocker = asyncio.ensure_future(b.submit(FP, ring, QKV, rand_x(1, 16, ring, 0)))
        await asyncio.sleep(0.01)  # device busy now: the next submissions wait in one ready batch
        xa, xb, xc = rand_x(2, 16, ring, 1), rand_x(1, 16, ring, 2), rand_x(4, 16, ring, 3)
        single = asyncio.ensure_future(b.submit(FP, ring, QKV, xa))
        group = asyncio.ensure_future(b.submit_parts(FP, ring, QKV, [xb, xc]))
        await blocker
        y_single = await single
        ys, _ = await group
        assert kernels[QKV].calls == [1, 7]
        assert torch.equal(y_single, expected(xa, ws[QKV], ring))
        assert torch.equal(ys[0], expected(xb, ws[QKV], ring)) and torch.equal(ys[1], expected(xc, ws[QKV], ring))
    finally:
        await b.stop()


def test_chunk_parts() -> None:
    xs = [torch.zeros((n, 2), dtype=torch.int64) for n in (3, 5, 1)]
    chunks = _chunk_parts(xs, 4)
    assert [[int(p.shape[0]) for p in c] for c in chunks] == [[3, 1], [4], [1]]
    assert _chunk_parts(xs, 100) == [tuple(xs)]
    assert sum(int(p.shape[0]) for c in _chunk_parts(xs, 1) for p in c) == 9
