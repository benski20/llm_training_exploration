"""GPU tests: CUDA-graph replay == eager == reference for the service's CUDA kernels.

All tests are marked ``gpu`` and skip without CUDA. Run on a GPU host with::

    pytest -q tests/service/test_gpu_graphs.py tests/service/test_gpu_exec.py
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import LinearId, LinearKind, LinearSpec, RingParams, pack_ring, ringmm_reference
from pgw.service.gpu_exec import DeviceExecutor, device_executor, from_compact
from pgw.service.kernels import ComputeKernel

pytestmark = pytest.mark.gpu

RINGS = {32: RingParams.from_name("K32A12C27"), 40: RingParams.from_name("K40A12"), 48: RingParams.from_name("K48A20")}
#: Qwen2.5-7B linears (m, n): qkv, o, gate_up, down
SHAPES_7B = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}
#: the same shapes scaled down 8x (plus odd sizes that need padding)
SHAPES_SMALL = [(m // 8, n // 8) for m, n in SHAPES_7B.values()] + [(77, 100), (5, 33)]


def _kernels(m: int, n: int, seed: int, extreme: bool = False) -> tuple[torch.Tensor, ComputeKernel, ComputeKernel]:
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    if extreme:
        w[::2] = -128
        w[1::2] = 127
    spec = LinearSpec(LinearId(0, LinearKind.O), m, n)
    # the fused Triton GEMV would take the K=32 small calls; these tests are about graphs vs eager
    graph = ComputeKernel(spec, w, device="cuda", cuda_graphs=True, triton_gemv=False)
    eager = ComputeKernel(spec, w, device="cuda", cuda_graphs=False, triton_gemv=False)
    return w, graph, eager


def _x(rows: int, n: int, k: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64))


def _cpu_exact(w: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    """Exact CPU result (cpu_f64 backend, itself checked against the Python-int reference)."""
    spec = LinearSpec(LinearId(0, LinearKind.O), int(w.shape[0]), int(w.shape[1]))
    return ComputeKernel(spec, w, device="cpu", backend="cpu_f64", cache_float=False).run(x, RINGS[k])


def _ref(w: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    """Python-int reference for small products, the exact CPU kernel for larger ones (speed)."""
    if w.numel() * x.shape[0] <= 2_000_000:
        return torch.from_numpy(ringmm_reference(x, w, k))
    return _cpu_exact(w, x, k)


@pytest.mark.parametrize("k", [32, 40, 48])
@pytest.mark.parametrize("shape", SHAPES_SMALL, ids=lambda s: f"{s[0]}x{s[1]}")
def test_graph_equals_eager_equals_reference(k: int, shape: tuple[int, int]) -> None:
    m, n = shape
    ring = RINGS[k]
    w, graph, eager = _kernels(m, n, seed=m + n + k)
    for rows in range(1, 9):
        for rep in range(2):  # capture, then replay
            x = _x(rows, n, k, seed=rows * 7 + rep)
            ref = _ref(w, x, k)
            yg = graph.run(x, ring)
            ye = eager.run(x, ring)
            assert yg.dtype == torch.int64 and yg.device.type == "cpu"
            assert torch.equal(ye, ref), ("eager", k, shape, rows)
            assert torch.equal(yg, ref), ("graph", k, shape, rows, rep)
            yc = graph.run_compact(x, ring)
            assert yc.dtype == (torch.int32 if k <= 32 else torch.int64)
            assert pack_ring(yc, ring) == pack_ring(ref, ring)
            assert torch.equal(from_compact(yc, k), ref)
    _require_capture(graph)
    assert graph.cuda_graphs >= 1 and eager.cuda_graphs == 0


@pytest.mark.parametrize("k", [32, 48])
def test_extreme_operands(k: int) -> None:
    ring = RINGS[k]
    w, graph, eager = _kernels(96, 160, seed=1, extreme=True)
    for rows in (1, 5, 8):
        x = torch.full((rows, 160), (1 << k) - 1, dtype=torch.int64)
        x[0] = 1 << (k - 1)
        x[-1, ::3] = 0
        ref = torch.from_numpy(ringmm_reference(x, w, k))
        for _ in range(2):
            assert torch.equal(graph.run(x, ring), ref)
        assert torch.equal(eager.run(x, ring), ref)


@pytest.mark.parametrize("name", ["o", "qkv"])
def test_full_size_7b_shape(name: str) -> None:
    m, n = SHAPES_7B[name]
    w, graph, eager = _kernels(m, n, seed=7)
    for k in (32, 40):
        for rows in (1, 5, 8):
            x = _x(rows, n, k, seed=rows + k)
            ref = _cpu_exact(w, x, k)
            for _ in range(2):
                assert torch.equal(graph.run(x, RINGS[k]), ref), (name, k, rows)
            assert torch.equal(eager.run(x, RINGS[k]), ref)


def _require_capture(graph: Any) -> None:
    if graph.cuda_graphs == 0:
        pytest.skip("CUDA graph capture unsupported in this torch build; eager results verified exact")


def test_warmup_captures_every_bucket_and_rows_beyond_run_eagerly() -> None:
    ring = RINGS[32]
    w, graph, _ = _kernels(64, 96, seed=2)
    graph.warmup(ring, all_buckets=True)
    _require_capture(graph)
    assert graph.cuda_graphs == 3  # 32, 64, 128
    ex = device_executor("cuda:0")
    eager_before = ex.stats.eager
    x = _x(40, 96, 32, seed=0)  # 160 digit rows > 128: eager
    assert torch.equal(graph.run(x, ring), torch.from_numpy(ringmm_reference(x, w, 32)))
    assert ex.stats.eager == eager_before + 1


def test_randomized_stress_interleaved_linears() -> None:
    rng = np.random.default_rng(123)
    lins = [_kernels(m, n, seed=i) for i, (m, n) in enumerate(SHAPES_SMALL[:4])]
    ex = device_executor("cuda:0")
    replays0 = ex.stats.replays
    for i in range(400):
        w, graph, _ = lins[int(rng.integers(0, len(lins)))]
        k = int(rng.choice([32, 40, 48]))
        rows = int(rng.choice([1, 1, 1, 2, 3, 5, 8, 12, 20, 40]))
        x = _x(rows, int(w.shape[1]), k, seed=i)
        assert torch.equal(graph.run(x, RINGS[k]), _cpu_exact(w, x, k)), (i, k, rows)
    if ex.stats.capture_failures:
        pytest.skip("CUDA graph capture unsupported in this torch build; eager results verified exact")
    assert ex.stats.replays - replays0 > 300
    assert ex.graphs_enabled


def test_concurrent_threads_are_serialized() -> None:
    lins = [_kernels(m, n, seed=10 + i) for i, (m, n) in enumerate(SHAPES_SMALL[:3])]
    errors: list[str] = []

    def worker(t: int) -> None:
        rng = np.random.default_rng(t)
        for i in range(60):
            w, graph, _ = lins[int(rng.integers(0, len(lins)))]
            k = int(rng.choice([32, 40, 48]))
            x = _x(int(rng.integers(1, 9)), int(w.shape[1]), k, seed=1000 * t + i)
            if not torch.equal(graph.run(x, RINGS[k]), _cpu_exact(w, x, k)):
                errors.append(f"thread {t} call {i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors


def test_private_executor_capture_and_pinned_staging() -> None:
    """A fresh executor (own pool, own buffers) produces the same results and uses pinned copies."""
    from pgw.core.ringmm import pad_weight_cuda
    from pgw.service.gpu_exec import KernelGraphs

    ex = DeviceExecutor("cuda")
    w8 = torch.randint(-128, 128, (300, 200), dtype=torch.int8, generator=torch.Generator().manual_seed(3))
    w = pad_weight_cuda(w8, "cuda")
    rowsum = w.to(torch.int64).sum(1)
    graphs = KernelGraphs()
    for rows in (1, 4, 8, 3):
        x = _x(rows, 200, 32, seed=rows)
        y = ex.run(x, 32, w, rowsum, 300, graphs=graphs)
        assert y.device.type == "cpu" and y.dtype == torch.int32
        assert torch.equal(from_compact(y, 32), torch.from_numpy(ringmm_reference(x, w8, 32)))
    assert ex.stats.captures == 1 and ex.stats.replays == 4 and ex.stats.pinned_copies >= 4
