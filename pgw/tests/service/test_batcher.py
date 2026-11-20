"""Tests for the dynamic cross-request batch scheduler."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    LinearId,
    LinearKind,
    LinearSpec,
    ModelNotFoundError,
    PGWError,
    ProtocolError,
    ServiceUnavailableError,
    ShapeError,
)
from pgw.service import BatchScheduler, ModelRegistry, RegisteredModel
from pgw.service.kernels import ComputeKernel
from tests.service.conftest import expected, rand_x

FP = "ee" * 32


class _SlowKernel(ComputeKernel):
    """Kernel that records batch sizes and sleeps (to build up a queue)."""

    def __init__(self, *a: Any, delay: float = 0.0, fail: bool = False, **kw: Any):
        super().__init__(*a, **kw)
        self.delay = delay
        self.fail = fail
        self.calls: list[int] = []

    def run(self, x_hat: torch.Tensor, ring: Any) -> torch.Tensor:
        self.calls.append(int(x_hat.shape[0]))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom (secret details)")
        return super().run(x_hat, ring)


def _registry(delay: float = 0.0, fail: bool = False) -> tuple[ModelRegistry, dict[str, torch.Tensor]]:
    reg = ModelRegistry("cpu", verify="fingerprint")
    g = torch.Generator().manual_seed(0)
    specs = [LinearSpec(LinearId(0, LinearKind.QKV), 24, 16), LinearSpec(LinearId(0, LinearKind.O), 16, 8)]
    ws: dict[str, torch.Tensor] = {}
    kernels: dict[str, ComputeKernel] = {}
    for s in specs:
        w = torch.randint(-127, 128, (s.out_features, s.in_features), dtype=torch.int8, generator=g)
        ws[s.id.key] = w
        kernels[s.id.key] = _SlowKernel(s, w, device="cpu", delay=delay, fail=fail)
    reg.register(RegisteredModel("synthetic", FP, specs, kernels, rings=("K40A12", "K48A20", "K32A8")))  # type: ignore[arg-type]
    return reg, ws


async def test_concurrent_submissions_coalesce_exactly() -> None:
    reg, ws = _registry()
    b = BatchScheduler(reg, batch_window_ms=20, max_batch_rows=4096)
    await b.start()
    xs = [rand_x(1 + i % 3, 16, RING_K40_A12, seed=i) for i in range(20)]
    ys = await asyncio.gather(*(b.submit(FP, RING_K40_A12, "0.qkv", x) for x in xs))
    for x, y in zip(xs, ys, strict=True):
        assert torch.equal(y, expected(x, ws["0.qkv"], RING_K40_A12))
    k = reg.get(FP).kernel("0.qkv")
    assert sum(k.calls) == sum(int(x.shape[0]) for x in xs)  # type: ignore[attr-defined]
    assert len(k.calls) == 1, k.calls  # type: ignore[attr-defined]
    assert b.stats.batches == 1 and b.stats.max_batch_rows_seen == sum(k.calls)  # type: ignore[attr-defined]
    await b.stop()
    assert b.queued_rows == 0 and b.queued_items == 0


async def test_keys_are_separate_per_ring_and_linear() -> None:
    reg, ws = _registry()
    b = BatchScheduler(reg, batch_window_ms=10)
    await b.start()
    jobs = []
    for ring in (RING_K32_A8, RING_K40_A12, RING_K48_A20):
        for lid, n in (("0.qkv", 16), ("0.o", 8)):
            for s in range(3):
                x = rand_x(2, n, ring, seed=s)
                jobs.append((ring, lid, x, b.submit_timed(FP, ring, lid, x)))
    results = await asyncio.gather(*(j[3] for j in jobs))
    for (ring, lid, x, _), (y, timing) in zip(jobs, results, strict=True):
        assert torch.equal(y, expected(x, ws[lid], ring))
        assert timing.batch_rows == 6 and timing.compute_ms >= 0 and timing.queue_ms >= 0
    assert b.stats.batches == 6
    await b.stop()


async def test_window_zero_and_max_batch_rows_split() -> None:
    reg, ws = _registry()
    b = BatchScheduler(reg, batch_window_ms=0, max_batch_rows=4)
    await b.start()
    x = rand_x(11, 16, RING_K40_A12)
    y, timing = await b.submit_timed(FP, RING_K40_A12, "0.qkv", x)
    assert torch.equal(y, expected(x, ws["0.qkv"], RING_K40_A12))
    calls = reg.get(FP).kernel("0.qkv").calls  # type: ignore[attr-defined]
    assert max(calls) <= 4 and sum(calls) == 11
    assert timing.batch_rows <= 4
    await b.stop()


async def test_batches_grow_while_device_is_busy() -> None:
    reg, ws = _registry(delay=0.05)
    b = BatchScheduler(reg, batch_window_ms=0, cpu_workers=1)
    await b.start()
    first = asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12, seed=99)))
    await asyncio.sleep(0.01)  # worker is now busy with the first batch
    xs = [rand_x(1, 16, RING_K40_A12, seed=i) for i in range(10)]
    ys = await asyncio.gather(*(b.submit(FP, RING_K40_A12, "0.qkv", x) for x in xs))
    await first
    for x, y in zip(xs, ys, strict=True):
        assert torch.equal(y, expected(x, ws["0.qkv"], RING_K40_A12))
    calls = reg.get(FP).kernel("0.qkv").calls  # type: ignore[attr-defined]
    assert calls == [1, 10], calls  # queued batch kept absorbing rows while waiting
    await b.stop()


async def test_cancelled_rows_are_dropped() -> None:
    reg, _ = _registry(delay=0.05)
    b = BatchScheduler(reg, batch_window_ms=0, cpu_workers=1)
    await b.start()
    first = asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12)))
    await asyncio.sleep(0.01)
    doomed = asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.qkv", rand_x(5, 16, RING_K40_A12)))
    keep = asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.qkv", rand_x(2, 16, RING_K40_A12)))
    await asyncio.sleep(0)
    doomed.cancel()
    await first
    await keep
    calls = reg.get(FP).kernel("0.qkv").calls  # type: ignore[attr-defined]
    assert calls == [1, 2]
    assert b.stats.dropped_rows == 5
    await b.stop()
    assert b.queued_rows == 0


async def test_errors() -> None:
    reg, _ = _registry()
    b = BatchScheduler(reg)
    with pytest.raises(ServiceUnavailableError):
        await b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12))
    await b.start()
    with pytest.raises(ModelNotFoundError):
        await b.submit("00" * 32, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12))
    with pytest.raises(ProtocolError):
        await b.submit(FP, RING_K40_A12, "5.qkv", rand_x(1, 16, RING_K40_A12))
    with pytest.raises(ShapeError):
        await b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 15, RING_K40_A12))
    with pytest.raises(ShapeError):
        await b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12).to(torch.int32))
    y = await b.submit(FP, RING_K40_A12, "0.qkv", torch.zeros((0, 16), dtype=torch.int64))
    assert y.shape == (0, 24)
    await b.stop()
    with pytest.raises(ServiceUnavailableError):
        await b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12))
    with pytest.raises(ValueError):
        BatchScheduler(reg, batch_window_ms=-1)
    with pytest.raises(ValueError):
        BatchScheduler(reg, max_batch_rows=0)


async def test_queue_limit() -> None:
    reg, _ = _registry(delay=0.05)
    b = BatchScheduler(reg, batch_window_ms=0, cpu_workers=1, max_queued_rows=8)
    await b.start()
    first = asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.qkv", rand_x(6, 16, RING_K40_A12)))
    await asyncio.sleep(0)
    with pytest.raises(ServiceUnavailableError) as ei:
        await b.submit(FP, RING_K40_A12, "0.qkv", rand_x(3, 16, RING_K40_A12))
    assert ei.value.retryable and "retry_after_s" in ei.value.details
    await first
    await b.stop()


async def test_kernel_failure_is_generic_internal_error(caplog: pytest.LogCaptureFixture) -> None:
    reg, _ = _registry(fail=True)
    b = BatchScheduler(reg, batch_window_ms=5)
    await b.start()
    res = await asyncio.gather(
        *(b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12, seed=i)) for i in range(3)),
        return_exceptions=True,
    )
    assert all(isinstance(r, PGWError) and r.code == "internal" for r in res)
    assert all("secret" not in r.message for r in res if isinstance(r, PGWError))
    assert b.stats.failed_batches == 1 and b.queued_rows == 0
    # scheduler keeps working afterwards
    reg.get(FP).kernel("0.qkv").fail = False  # type: ignore[attr-defined]
    await b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12))
    await b.stop()


async def test_stop_drains_pending_work() -> None:
    reg, ws = _registry(delay=0.02)
    b = BatchScheduler(reg, batch_window_ms=50, cpu_workers=1)
    await b.start()
    xs = [rand_x(1, 16, RING_K40_A12, seed=i) for i in range(4)]
    futs = [asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.qkv", x)) for x in xs]
    await asyncio.sleep(0)
    await b.stop(drain=True)  # does not wait for the 50 ms window
    for x, f in zip(xs, futs, strict=True):
        assert torch.equal(f.result(), expected(x, ws["0.qkv"], RING_K40_A12))


async def test_stop_without_drain_fails_pending() -> None:
    reg, _ = _registry(delay=0.1)
    b = BatchScheduler(reg, batch_window_ms=0, cpu_workers=1)
    await b.start()
    running = asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.qkv", rand_x(1, 16, RING_K40_A12)))
    await asyncio.sleep(0.02)
    waiting = asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.o", rand_x(1, 8, RING_K40_A12)))
    await asyncio.sleep(0)
    await b.stop(drain=False)
    await asyncio.sleep(0)
    with pytest.raises(ServiceUnavailableError):
        waiting.result()
    assert running.done()


async def test_metrics_hooks() -> None:
    from pgw.service.metrics import ServiceMetrics

    reg, _ = _registry()
    m = ServiceMetrics()
    b = BatchScheduler(reg, batch_window_ms=1, metrics=m)
    await b.start()
    await asyncio.gather(*(b.submit(FP, RING_K40_A12, "0.o", rand_x(2, 8, RING_K40_A12, seed=i)) for i in range(3)))
    await b.stop()
    text = m.render()[0].decode()
    assert 'pgw_service_batch_rows_count{linear_kind="o"} 1.0' in text
    assert 'pgw_service_batch_rows_sum{linear_kind="o"} 6.0' in text
    assert "pgw_service_kernel_seconds_count" in text
    assert "pgw_service_queue_depth 0.0" in text


async def test_inline_small_batches_on_an_idle_device() -> None:
    import threading

    reg, ws = _registry(delay=0.05)
    b = BatchScheduler(reg, batch_window_ms=0, cpu_workers=1, inline_max_rows=2)
    await b.start()
    k = reg.get(FP).kernel("0.qkv")
    seen: list[str] = []
    orig = k.run

    def run(x: torch.Tensor, ring: Any) -> torch.Tensor:
        seen.append(threading.current_thread().name)
        return orig(x, ring)

    k.run = run  # type: ignore[method-assign]
    x = rand_x(2, 16, RING_K40_A12, seed=1)
    assert torch.equal(await b.submit(FP, RING_K40_A12, "0.qkv", x), expected(x, ws["0.qkv"], RING_K40_A12))
    assert seen == [threading.current_thread().name]  # ran on the event loop thread, no pool hand-off
    # larger batches use the pool; rows arriving while the device is busy still coalesce
    big = asyncio.ensure_future(b.submit(FP, RING_K40_A12, "0.qkv", rand_x(3, 16, RING_K40_A12, seed=2)))
    await asyncio.sleep(0.01)
    xs = [rand_x(1, 16, RING_K40_A12, seed=10 + i) for i in range(2)]
    ys = await asyncio.gather(*(b.submit(FP, RING_K40_A12, "0.qkv", xx) for xx in xs))
    await big
    for xx, y in zip(xs, ys, strict=True):
        assert torch.equal(y, expected(xx, ws["0.qkv"], RING_K40_A12))
    assert seen[1].startswith("pgw-kernel")
    assert k.calls[-2:] == [3, 2]  # type: ignore[attr-defined]
    await b.stop()
    assert b.queued_rows == 0 and b.queued_items == 0


async def test_inline_fast_path_accounting_and_failures() -> None:
    """A submission run inline on an idle device counts, times and fails like a batched one."""
    from pgw.service.metrics import ServiceMetrics

    reg, ws = _registry()
    m = ServiceMetrics()
    b = BatchScheduler(reg, batch_window_ms=0, inline_max_rows=8, metrics=m)
    await b.start()
    x = rand_x(3, 16, RING_K40_A12, seed=4)
    y, timing = await b.submit_timed(FP, RING_K40_A12, "0.qkv", x)
    assert torch.equal(y, expected(x, ws["0.qkv"], RING_K40_A12)) and y.dtype == torch.int64
    assert timing.batch_rows == 3 and timing.compute_ms > 0 and timing.queue_ms >= 0
    st = b.stats
    assert (st.submitted_rows, st.batches, st.batched_rows, st.max_batch_rows_seen) == (3, 1, 3, 3)
    assert b.queued_rows == 0 and b.queued_items == 0 and not b._open
    assert 'pgw_service_batch_rows_sum{linear_kind="qkv"} 3.0' in m.render()[0].decode()
    reg.get(FP).kernel("0.qkv").fail = True  # type: ignore[attr-defined]
    with pytest.raises(PGWError) as ei:
        await b.submit(FP, RING_K40_A12, "0.qkv", x)
    assert ei.value.code == "internal" and "secret" not in ei.value.message
    assert b.stats.failed_batches == 1 and b.queued_rows == 0
    assert all(d.busy == 0 for d in b._devices.values())
    reg.get(FP).kernel("0.qkv").fail = False  # type: ignore[attr-defined]
    assert torch.equal(await b.submit(FP, RING_K40_A12, "0.qkv", x), y)
    await b.stop()
