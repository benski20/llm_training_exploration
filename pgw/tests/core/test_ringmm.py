"""Exactness tests for pgw.core.ringmm (every backend, every ring, vs arbitrary-precision references)."""

import numpy as np
import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    BackendUnavailableError,
    ConfigError,
    PreparedWeight,
    ShapeError,
    cuda_int8_available,
    mask_activations,
    ringmm,
    ringmm_reference,
    select_backend,
    unmask_outputs,
)
from pgw.core.ringmm import _f64_digit_bits

RINGS = [RING_K32_A8, RING_K40_A12, RING_K48_A20]
CPU_BACKENDS = ["cpu_f64", "cpu_f32"]
GPU = pytest.mark.skipif(not cuda_int8_available(), reason="CUDA int8 GEMM not available")


def _rand(shape, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.int64)


@pytest.mark.parametrize("p", RINGS)
@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("b,m,n", [(1, 8, 8), (3, 17, 29), (40, 64, 96), (5, 33, 1500)])
def test_exact_vs_bigint_reference(p, backend, b, m, n):
    x = _rand((b, n), 0, p.modulus, b * 1000 + n)
    w8 = _rand((m, n), -128, 128, m).to(torch.int8)  # includes -128 on purpose
    out = PreparedWeight(w8, backend=backend).matmul(x, p)
    assert out.dtype == torch.int64 and out.shape == (b, m)
    assert np.array_equal(out.numpy(), ringmm_reference(x, w8, p.k))


@pytest.mark.parametrize("p", RINGS)
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_extreme_values(p, backend):
    n = 64
    x = torch.full((4, n), p.modulus - 1, dtype=torch.int64)
    x[1] = p.half
    x[2] = p.half - 1
    x[3, ::2] = 0
    w8 = torch.full((9, n), -128, dtype=torch.int8)
    w8[1] = 127
    out = PreparedWeight(w8, backend=backend).matmul(x, p)
    assert np.array_equal(out.numpy(), ringmm_reference(x, w8, p.k))


@pytest.mark.parametrize("p", RINGS)
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_masked_product_matches_float64_reference(p, backend):
    """Decoded masked product equals the plain integer product computed in float64."""
    t, m, n = 6, 48, 896
    xq = _rand((t, n), -p.qmax, p.qmax + 1, 7)
    xq[0] = p.qmax  # worst-case magnitudes
    w8 = _rand((m, n), -127, 128, 8).to(torch.int8)
    w8[0] = 127
    r = _rand((t, n), 0, p.modulus, 9)
    pw = PreparedWeight(w8, backend=backend)
    yq = unmask_outputs(pw.matmul(mask_activations(xq, r, p), p), pw.matmul(r, p), p, in_features=n)
    ref = xq.double() @ w8.double().t()
    assert bool((yq.double() == ref).all())


def test_large_inner_dim_f64_digits():
    p = RING_K48_A20
    n = 20000
    assert _f64_digit_bits(n, 48) * 7 >= 48
    x = _rand((2, n), 0, p.modulus, 3)
    w8 = _rand((3, n), -127, 128, 4).to(torch.int8)
    out = PreparedWeight(w8, backend="cpu_f64").matmul(x, p)
    assert np.array_equal(out.numpy(), ringmm_reference(x, w8, p.k))


def test_oneshot_and_integer_k():
    x = _rand((2, 16), 0, 1 << 40, 1)
    w8 = _rand((4, 16), -127, 128, 2).to(torch.int8)
    assert np.array_equal(ringmm(x, w8, 40).numpy(), ringmm_reference(x, w8, 40))
    assert np.array_equal(ringmm(x, w8, RING_K40_A12, backend="cpu_f32").numpy(), ringmm_reference(x, w8, 40))


def test_empty_batch_and_shape_errors():
    pw = PreparedWeight(torch.zeros((4, 8), dtype=torch.int8))
    assert pw.matmul(torch.zeros((0, 8), dtype=torch.int64), 40).shape == (0, 4)
    with pytest.raises(ShapeError):
        pw.matmul(torch.zeros((2, 7), dtype=torch.int64), 40)
    with pytest.raises(ShapeError):
        pw.matmul(torch.zeros((2, 8), dtype=torch.int32), 40)
    with pytest.raises(ShapeError):
        PreparedWeight(torch.zeros((4, 8), dtype=torch.float32))
    assert pw.in_features == 8 and pw.out_features == 4 and pw.resolved_backend == "cpu_f64"


def test_backend_selection():
    assert select_backend("cpu") == "cpu_f64"
    assert select_backend("cpu", "cpu_f32") == "cpu_f32"
    with pytest.raises(ConfigError):
        select_backend("cpu", "nope")  # type: ignore[arg-type]
    with pytest.raises(BackendUnavailableError):
        select_backend("cpu", "cuda_int8")


def test_uncached_float_weight():
    p = RING_K40_A12
    x = _rand((3, 40), 0, p.modulus, 5)
    w8 = _rand((5, 40), -127, 128, 6).to(torch.int8)
    for be in CPU_BACKENDS:
        out = PreparedWeight(w8, backend=be, cache_float=False).matmul(x, p)
        assert np.array_equal(out.numpy(), ringmm_reference(x, w8, p.k))


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("p", RINGS)
@pytest.mark.parametrize("b,m,n", [(1, 8, 8), (3, 17, 29), (40, 64, 96)])
def test_cuda_exact(p, b, m, n):  # pragma: no cover - GPU only
    x = _rand((b, n), 0, p.modulus, 11)
    w8 = _rand((m, n), -128, 128, 12).to(torch.int8)
    out = PreparedWeight(w8, device="cuda").matmul(x.cuda(), p).cpu()
    assert np.array_equal(out.numpy(), ringmm_reference(x, w8, p.k))
