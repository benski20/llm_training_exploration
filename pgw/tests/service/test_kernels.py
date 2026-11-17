"""Exactness and contract tests for :mod:`pgw.service.kernels`."""

from __future__ import annotations

import pytest
import torch

from pgw.core import (
    RING_K40_A12,
    BackendUnavailableError,
    ConfigError,
    LinearId,
    LinearKind,
    LinearSpec,
    RingParams,
    ShapeError,
)
from pgw.service.kernels import (
    ComputeKernel,
    cpu_int8_available,
    int8_ring_matmul,
    resolve_backend,
    resolve_device,
)
from tests.service.conftest import expected, rand_x

CPU_BACKENDS = ["cpu_f64", "cpu_f32"] + (["cpu_int8"] if cpu_int8_available() else [])


def _spec(m: int, n: int) -> LinearSpec:
    return LinearSpec(LinearId(0, LinearKind.QKV), m, n)


@pytest.mark.parametrize("backend", [*CPU_BACKENDS, "auto"])
@pytest.mark.parametrize("shape", [(1, 8, 8), (3, 37, 13), (33, 64, 96), (5, 1030, 7)])
def test_exact_against_reference(backend: str, shape: tuple[int, int, int], ring: RingParams) -> None:
    b, n, m = shape
    g = torch.Generator().manual_seed(b * 1000 + n + m)
    w = torch.randint(-127, 128, (m, n), dtype=torch.int8, generator=g)
    x = rand_x(b, n, ring, seed=n)
    k = ComputeKernel(_spec(m, n), w, device="cpu", backend=backend)
    y = k.run(x, ring)
    assert y.dtype == torch.int64 and y.shape == (b, m)
    assert torch.equal(y, expected(x, w, ring))
    assert int(y.min()) >= 0 and int(y.max()) < ring.modulus


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_extreme_values_including_minus_128(backend: str) -> None:
    ring = RingParams(48, 20)
    n, m = 300, 9
    w = torch.full((m, n), -128, dtype=torch.int8)
    w[::2] = 127
    x = torch.full((4, n), ring.modulus - 1, dtype=torch.int64)
    x[1] = ring.half
    x[2] = ring.half - 1
    x[3] = 0
    k = ComputeKernel(_spec(m, n), w, device="cpu", backend=backend)
    assert torch.equal(k.run(x, ring), expected(x, w, ring))


def test_any_int64_input_is_reduced_mod_2k() -> None:
    ring = RING_K40_A12
    w = torch.randint(-127, 128, (5, 16), dtype=torch.int8)
    x = rand_x(3, 16, ring)
    k = ComputeKernel(_spec(5, 16), w, device="cpu")
    shifted = x + (7 << 40)  # same ring element
    assert torch.equal(k.run(shifted, ring), k.run(x, ring))


def test_zero_rows_and_shape_errors() -> None:
    w = torch.zeros((4, 8), dtype=torch.int8)
    k = ComputeKernel(_spec(4, 8), w, device="cpu")
    assert k.run(torch.zeros((0, 8), dtype=torch.int64), RING_K40_A12).shape == (0, 4)
    with pytest.raises(ShapeError):
        k.run(torch.zeros((2, 7), dtype=torch.int64), RING_K40_A12)
    with pytest.raises(ShapeError):
        k.run(torch.zeros((2, 8), dtype=torch.int32), RING_K40_A12)
    with pytest.raises(ShapeError):
        ComputeKernel(_spec(4, 8), torch.zeros((4, 9), dtype=torch.int8), device="cpu")
    with pytest.raises(ShapeError):
        ComputeKernel(_spec(4, 8), torch.zeros((4, 8), dtype=torch.int16), device="cpu")


def test_metadata_and_device_bytes() -> None:
    w = torch.randint(-127, 128, (24, 16), dtype=torch.int8)
    spec = LinearSpec(LinearId(3, LinearKind.GATE_UP), 24, 16)
    k = ComputeKernel(spec, w, device="cpu", backend="cpu_f64")
    assert k.linear == "3.gate_up" and k.kind == "gate_up"
    assert (k.out_features, k.in_features) == (24, 16)
    assert k.device_bytes == 24 * 16 * (1 + 8)  # int8 copy + float64 cache
    k2 = ComputeKernel(spec, w, device="cpu", backend="cpu_f64", cache_float=False)
    assert k2.device_bytes == 24 * 16
    assert k.warmup(RING_K40_A12) >= 0
    assert "3.gate_up" in repr(k)


def test_kernel_owns_weight_memory() -> None:
    w = torch.randint(-127, 128, (4, 8), dtype=torch.int8)
    x = rand_x(2, 8, RING_K40_A12)
    for backend in CPU_BACKENDS:
        k = ComputeKernel(_spec(4, 8), w, device="cpu", backend=backend)
        before = k.run(x, RING_K40_A12)
        w.zero_()  # mutate the source after construction
        assert torch.equal(k.run(x, RING_K40_A12), before)
        w.copy_(torch.randint(-127, 128, (4, 8), dtype=torch.int8))


def test_resolve_backend_and_device_errors() -> None:
    cpu = torch.device("cpu")
    with pytest.raises(ConfigError):
        resolve_backend(cpu, "tpu_magic")
    with pytest.raises(BackendUnavailableError):
        resolve_backend(cpu, "cuda_int8")
    assert resolve_backend(cpu, "cpu_f32") == "cpu_f32"
    assert resolve_backend(cpu, "auto") in ("cpu_int8", "cpu_f64")
    assert resolve_device("cpu") == cpu
    assert resolve_device("auto").type in ("cpu", "cuda")
    with pytest.raises(ConfigError):
        resolve_device("not a device")
    with pytest.raises(ConfigError):
        resolve_device("meta")
    if not torch.cuda.is_available():
        with pytest.raises(BackendUnavailableError):
            resolve_device("cuda:0")


@pytest.mark.skipif(not cpu_int8_available(), reason="CPU torch._int_mm unavailable")
def test_int8_ring_matmul_long_inner_dimension_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.service.kernels as kmod

    monkeypatch.setattr(kmod, "_INT_MM_MAX_INNER", 16)  # force chunking on a small problem
    ring = RingParams(48, 20)
    w = torch.randint(-128, 128, (6, 50), dtype=torch.int8)
    x = rand_x(3, 50, ring)
    assert torch.equal(int8_ring_matmul(x, w, ring.k), expected(x, w, ring))


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 17, 33])
def test_cuda_int8_exact(rows: int, ring: RingParams) -> None:  # pragma: no cover - GPU only
    w = torch.randint(-127, 128, (40, 72), dtype=torch.int8)
    x = rand_x(rows, 72, ring)
    k = ComputeKernel(_spec(40, 72), w, device="cuda", backend="cuda_int8")
    assert k.backend == "cuda_int8"
    y = k.run(x, ring)
    assert y.device.type == "cpu" and torch.equal(y, expected(x, w, ring))
