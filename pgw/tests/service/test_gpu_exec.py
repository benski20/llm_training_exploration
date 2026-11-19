"""CPU tests for :mod:`pgw.service.gpu_exec` (buckets, compact output, graph cache and fallbacks).

The executor runs on CPU tensors with a capturer that just calls the body, so everything except
the CUDA capture itself (tested in ``test_gpu_graphs.py``) is covered without a GPU.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import ConfigError, LinearId, LinearKind, LinearSpec, RingParams, ShapeError, pack_ring
from pgw.core.ringmm import cpu_int8_available, pad_weight_cuda, ringmm_reference
from pgw.service import ModelRegistry, ServiceSettings
from pgw.service.batcher import BatchScheduler
from pgw.service.gpu_exec import (
    GRAPH_BUCKETS,
    DeviceExecutor,
    KernelGraphs,
    bucket_for,
    eager_capturer,
    from_compact,
    to_compact,
    validate_buckets,
)
from pgw.service.kernels import ComputeKernel
from pgw.service.registry import RegisteredModel
from tests.service.conftest import expected, rand_x

needs_int8 = pytest.mark.skipif(not cpu_int8_available(), reason="CPU torch._int_mm unavailable or inexact")


# ---- buckets --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("k", "fits"),
    [(32, {32: range(1, 9), 64: range(9, 17), 128: range(17, 33)}),
     (40, {32: range(1, 7), 64: range(7, 13), 128: range(13, 26)}),
     (48, {32: range(1, 6), 64: range(6, 11), 128: range(11, 22)}),
     (16, {32: range(1, 17), 64: range(17, 33), 128: range(33, 65)})],
)
def test_bucket_for(k: int, fits: dict[int, range]) -> None:
    for p, rows in fits.items():
        for r in rows:
            assert bucket_for(r, k, GRAPH_BUCKETS) == p, (k, r)
    top = max(fits[128])
    assert bucket_for(top + 1, k, GRAPH_BUCKETS) is None
    assert bucket_for(0, k, GRAPH_BUCKETS) is None
    assert bucket_for(3, 32, (32,)) == 32 and bucket_for(9, 32, (32,)) is None
    assert bucket_for(1, 32, ()) is None


def test_validate_buckets() -> None:
    assert validate_buckets([128, 32, 64, 32]) == (32, 64, 128)
    assert validate_buckets([]) == ()
    for bad in ([0], [48], [-32], [8192]):
        with pytest.raises(ConfigError):
            validate_buckets(bad)
    with pytest.raises(ConfigError):
        ComputeKernel(LinearSpec(LinearId(0, LinearKind.O), 4, 4), torch.zeros((4, 4), dtype=torch.int8),
                      device="cpu", graph_buckets=(40,))


def test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    s = ServiceSettings()
    assert s.cuda_graphs is True and s.cuda_graph_buckets == [32, 64, 128]
    monkeypatch.setenv("PGW_SERVICE_CUDA_GRAPHS", "false")
    monkeypatch.setenv("PGW_SERVICE_CUDA_GRAPH_BUCKETS", "[64, 32]")
    s = ServiceSettings()
    assert s.cuda_graphs is False and s.cuda_graph_buckets == [32, 64]
    monkeypatch.setenv("PGW_SERVICE_CUDA_GRAPH_BUCKETS", "[33]")
    with pytest.raises(ValueError, match="multiples of 32"):
        ServiceSettings()


# ---- compact output -------------------------------------------------------------------------
def _edge_values(k: int, size: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    mod = 1 << k
    edges = [0, 1, mod - 1, mod // 2, mod // 2 - 1, mod // 2 + 1]
    if k > 31:
        edges += [(1 << 31) - 1, 1 << 31, (1 << 31) + 1]
    vals = np.concatenate([np.array(edges, dtype=np.int64), rng.integers(0, mod, size=size, dtype=np.int64)])
    return torch.from_numpy(vals).view(-1, 3)


@pytest.mark.parametrize("k", [16, 24, 32, 40, 48])
def test_compact_matches_pack_ring(k: int) -> None:
    y = _edge_values(k, 3 * 400, seed=k)
    c = to_compact(y, k)
    assert c.dtype == (torch.int32 if k <= 32 else torch.int64)
    assert pack_ring(c, k) == pack_ring(y, k)
    assert pack_ring(c.numpy(), k) == pack_ring(y, k)
    if k <= 32:
        assert pack_ring(c.numpy().view(np.uint32), k) == pack_ring(y, k)
        assert np.array_equal(c.numpy().view(np.uint32).astype(np.int64), y.numpy())
    assert torch.equal(from_compact(c, k), y)


def test_pack_ring_rejects_32bit_for_wide_rings() -> None:
    with pytest.raises(ShapeError):
        pack_ring(torch.zeros(4, dtype=torch.int32), 40)
    with pytest.raises(ShapeError):
        pack_ring(torch.zeros(4, dtype=torch.int16), 32)


def test_cpu_kernel_run_compact_is_int64() -> None:
    ring = RingParams.from_name("K32A12C27")
    w = torch.randint(-127, 128, (7, 20), dtype=torch.int8, generator=torch.Generator().manual_seed(1))
    k = ComputeKernel(LinearSpec(LinearId(0, LinearKind.O), 7, 20), w, device="cpu")
    x = rand_x(3, 20, ring)
    y = k.run_compact(x, ring)
    assert y.dtype == torch.int64 and torch.equal(y, expected(x, w, ring))
    assert k.cuda_graphs == 0
    assert k.run_compact(torch.zeros((0, 20), dtype=torch.int64), ring).shape == (0, 7)


# ---- executor on CPU ------------------------------------------------------------------------
class _Lin:
    def __init__(self, m: int, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
        self.w = pad_weight_cuda(self.w8, "cpu")
        self.rowsum = self.w.to(torch.int64).sum(1)
        self.m, self.n = m, n
        self.graphs = KernelGraphs()

    def run(self, ex: DeviceExecutor, x: torch.Tensor, k: int, graphs: bool = True) -> torch.Tensor:
        return ex.run(x, k, self.w, self.rowsum, self.m, graphs=self.graphs if graphs else None)

    def ref(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return torch.from_numpy(ringmm_reference(x, self.w8, k))


def _x(rows: int, n: int, k: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64))


@needs_int8
@pytest.mark.parametrize("k", [32, 40, 48])
def test_executor_graph_path_exact(k: int) -> None:
    ex = DeviceExecutor("cpu", capturer=eager_capturer)
    lin = _Lin(45, 70, seed=k)
    for rows in range(1, 9):
        for rep in range(2):  # first call captures, second replays
            x = _x(rows, lin.n, k, seed=rows * 10 + rep)
            if rep:
                x[0] -= 3 << k  # any int64 is taken mod 2^K
            y = lin.run(ex, x, k)
            assert y.dtype == (torch.int32 if k <= 32 else torch.int64)
            assert torch.equal(from_compact(y, k), lin.ref(x, k)), (k, rows, rep)
    assert set(lin.graphs.graphs) == {(k, bucket_for(r, k, GRAPH_BUCKETS)) for r in range(1, 9)}
    assert ex.stats.capture_failures == 0 and ex.stats.eager == 0
    assert ex.stats.captures == len(lin.graphs)
    assert ex.stats.replays == 16


@needs_int8
def test_executor_eager_when_no_bucket_or_no_graphs() -> None:
    ex = DeviceExecutor("cpu", capturer=eager_capturer)
    lin = _Lin(33, 40, seed=3)
    x = _x(40, lin.n, 32, seed=1)  # 40 rows * 4 planes > 128
    assert torch.equal(from_compact(lin.run(ex, x, 32), 32), lin.ref(x, 32))
    x = _x(2, lin.n, 32, seed=2)
    assert torch.equal(from_compact(lin.run(ex, x, 32, graphs=False), 32), lin.ref(x, 32))
    assert ex.stats.eager == 2 and ex.stats.captures == 0 and not lin.graphs.graphs


@needs_int8
def test_executor_interleaved_linears_share_buffers() -> None:
    """Several linears (different n, m) share the static buffers; replays in any order stay exact."""
    ex = DeviceExecutor("cpu", capturer=eager_capturer)
    lins = [_Lin(64, 96, 1), _Lin(200, 40, 2), _Lin(30, 260, 3)]
    for lin in lins:
        ex.reserve(lin.n, lin.m)
    rng = np.random.default_rng(0)
    for i in range(150):
        lin = lins[int(rng.integers(0, 3))]
        k = int(rng.choice([32, 40, 48]))
        rows = int(rng.integers(1, 12))
        x = _x(rows, lin.n, k, seed=i)
        assert torch.equal(from_compact(lin.run(ex, x, k), k), lin.ref(x, k)), (i, k, rows)
    assert ex.stats.replays > 100 and ex.stats.capture_failures == 0


@needs_int8
def test_capture_failure_falls_back_and_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    calls = []

    def failing(body: Callable[[], None]) -> Callable[[], None]:
        calls.append(1)
        raise RuntimeError("capture not supported")

    ex = DeviceExecutor("cpu", capturer=failing)
    lin = _Lin(20, 50, 4)
    with caplog.at_level(logging.DEBUG, logger="pgw.service.gpu_exec"):
        for rows in (1, 1, 12, 12, 3):
            x = _x(rows, lin.n, 32, seed=rows)
            assert torch.equal(from_compact(lin.run(ex, x, 32), 32), lin.ref(x, 32))
    assert len(calls) == 2  # one attempt per (K, bucket); then eager without retrying
    assert lin.graphs.graphs == {(32, 32): None, (32, 64): None}
    assert ex.stats.capture_failures == 2 and ex.stats.eager == 5 and ex.graphs_enabled
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "capture failed" in r.getMessage()]
    assert len(warnings) == 1


@needs_int8
def test_wrong_replay_disables_graphs(caplog: pytest.LogCaptureFixture) -> None:
    def noop(body: Callable[[], None]) -> Callable[[], None]:
        body()
        return lambda: None  # "replay" that does nothing

    ex = DeviceExecutor("cpu", capturer=noop)
    lin = _Lin(20, 50, 5)
    with caplog.at_level(logging.ERROR, logger="pgw.service.gpu_exec"):
        for i in range(3):
            x = _x(2, lin.n, 32, seed=i)
            assert torch.equal(from_compact(lin.run(ex, x, 32), 32), lin.ref(x, 32))
    assert not ex.graphs_enabled
    assert ex.stats.eager == 3 and ex.stats.replays == 0
    assert any("disagrees" in r.getMessage() for r in caplog.records)


def test_executor_without_capturer_on_cpu_runs_eagerly() -> None:
    ex = DeviceExecutor("cpu")
    if not cpu_int8_available():
        pytest.skip("CPU torch._int_mm unavailable")
    lin = _Lin(10, 12, 6)
    x = _x(1, lin.n, 40, seed=0)
    assert torch.equal(lin.run(ex, x, 40), lin.ref(x, 40))
    assert lin.graphs.graphs == {(40, 32): None} and ex.stats.eager == 1


# ---- batcher: compact vs int64 callers ------------------------------------------------------
class _CompactKernel(ComputeKernel):
    """CPU kernel that returns the compact form like the CUDA path does."""

    def run_compact(self, x_hat: torch.Tensor, ring: Any) -> torch.Tensor:
        return to_compact(self.run(x_hat, ring), ring.k)


def test_batcher_compact_and_int64_callers_share_a_batch() -> None:
    fp = "ab" * 32
    ring = RingParams.from_name("K32A12C27")
    spec = LinearSpec(LinearId(0, LinearKind.QKV), 24, 16)
    w = torch.randint(-127, 128, (24, 16), dtype=torch.int8, generator=torch.Generator().manual_seed(9))
    reg = ModelRegistry("cpu", verify="fingerprint")
    reg.register(RegisteredModel("synthetic", fp, (spec,), {"0.qkv": _CompactKernel(spec, w, device="cpu")},
                                 rings=("K32A12C27",)))

    async def go() -> None:
        b = BatchScheduler(reg, batch_window_ms=20.0)
        await b.start()
        xs = [rand_x(r, 16, ring, seed=r) for r in (1, 3, 2)]
        res = await asyncio.gather(
            b.submit(fp, ring, "0.qkv", xs[0]),
            b.submit_timed(fp, ring, "0.qkv", xs[1], compact=True),
            b.submit_timed(fp, ring, "0.qkv", xs[2]),
        )
        await b.stop()
        y0, (y1, t1), (y2, _) = res
        assert y0.dtype == torch.int64 and torch.equal(y0, expected(xs[0], w, ring))
        assert y1.dtype == torch.int32 and pack_ring(y1, ring) == pack_ring(expected(xs[1], w, ring), ring)
        assert y2.dtype == torch.int64 and torch.equal(y2, expected(xs[2], w, ring))
        assert t1.batch_rows == 6 and b.stats.batches == 1

    asyncio.run(go())
