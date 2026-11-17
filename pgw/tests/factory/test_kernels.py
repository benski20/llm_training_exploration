"""Exactness of the factory ring kernels against arbitrary-precision references."""

from __future__ import annotations

import pytest
import torch

from pgw.core import (
    RING_K40_A12,
    BackendUnavailableError,
    ConfigError,
    RingParams,
    ShapeError,
    freivalds_precompute,
    ringmm_reference,
)
from pgw.core.ringmm import int8_ring_matmul as int8_ring_mm
from pgw.factory import kernels
from pgw.factory.kernels import (
    RingKernel,
    cpu_int8_available,
    resolve_backend,
    resolve_device,
    ring_vecmat,
)

CPU = torch.device("cpu")
CPU_BACKENDS = ["cpu_f64", "cpu_f32"] + (["cpu_int8"] if cpu_int8_available() else [])


def _rand(shape: tuple[int, int], k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1 << k, shape, dtype=torch.int64, generator=g)


def _w(m: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    w[0, :] = -128  # worst case magnitude row
    return w


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("shape", [(1, 3, 5), (7, 13, 21), (33, 64, 40), (5, 300, 9)])
def test_ring_kernel_exact(backend: str, shape: tuple[int, int, int], ring: RingParams) -> None:
    b, m, n = shape
    x = _rand((b, n), ring.k, b * 7 + n)
    w = _w(m, n, m)
    got = RingKernel(w, backend).matmul(x, ring)
    assert got.dtype == torch.int64 and got.shape == (b, m)
    assert (got >= 0).all() and (got < ring.modulus).all()
    assert got.numpy().tolist() == ringmm_reference(x, w, ring.k).tolist()


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_ring_kernel_extreme_values(backend: str) -> None:
    ring = RingParams(48, 20)
    x = torch.full((4, 64), (1 << 47), dtype=torch.int64)  # digits hit -128 everywhere
    x[1] = (1 << 48) - 1
    w = torch.full((8, 64), -128, dtype=torch.int8)
    w[1] = 127
    got = RingKernel(w, backend).matmul(x, ring)
    assert got.tolist() == ringmm_reference(x, w, 48).tolist()


def test_ring_kernel_accepts_noncanonical_int64() -> None:
    w = _w(6, 10, 1)
    x = _rand((3, 10), 40, 1) - (1 << 39)  # negative values = same ring elements
    k = RingKernel(w, resolve_backend(CPU))
    assert torch.equal(k(x, RING_K40_A12), k(x & RING_K40_A12.mask, RING_K40_A12))


def test_ring_kernel_empty_and_shape_errors() -> None:
    w = _w(6, 10, 2)
    k = RingKernel(w, "cpu_f64")
    assert k.matmul(torch.zeros((0, 10), dtype=torch.int64), RING_K40_A12).shape == (0, 6)
    with pytest.raises(ShapeError):
        k.matmul(torch.zeros((2, 9), dtype=torch.int64), RING_K40_A12)
    with pytest.raises(ShapeError):
        k.matmul(torch.zeros((2, 10), dtype=torch.int32), RING_K40_A12)
    with pytest.raises(ShapeError):
        RingKernel(w.to(torch.int16), "cpu_f64")
    with pytest.raises(ConfigError):
        RingKernel(w, "auto")
    assert (k.out_features, k.in_features) == (6, 10)


@pytest.mark.skipif(not cpu_int8_available(), reason="CPU torch._int_mm not exact here")
def test_int8_inner_dimension_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    core_ringmm = sys.modules["pgw.core.ringmm"]  # the package attribute is the ringmm() function

    monkeypatch.setattr(core_ringmm, "INT_MM_MAX_INNER_CPU", 16)
    x = _rand((5, 50), 40, 3)
    w = _w(7, 50, 3)
    assert int8_ring_mm(x, w, 40).tolist() == ringmm_reference(x, w, 40).tolist()


@pytest.mark.parametrize("method", ["f64"] + (["int8"] if cpu_int8_available() else []))
def test_ring_vecmat_matches_core(method: str, ring: RingParams) -> None:
    u = _rand((3, 37), ring.k, 5)
    w = _w(37, 19, 5)
    got = ring_vecmat(u, w, ring, method=method)  # type: ignore[arg-type]
    ref = freivalds_precompute(u, w, ring)
    assert torch.equal(got, ref)
    assert got.tolist() == ringmm_reference(u, w.t().contiguous(), ring.k).tolist()


def test_ring_vecmat_chunks_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "_VECMAT_CHUNK_ELEMS", 64)  # forces many row chunks
    u = _rand((2, 40), 40, 6)
    w = _w(40, 16, 6)
    assert ring_vecmat(u, w, RING_K40_A12, method="f64").tolist() == ringmm_reference(u, w.t().contiguous(), 40).tolist()


def test_ring_vecmat_edge_cases() -> None:
    w = _w(4, 3, 7)
    assert ring_vecmat(torch.zeros((0, 4), dtype=torch.int64), w, RING_K40_A12).shape == (0, 3)
    with pytest.raises(ShapeError):
        ring_vecmat(torch.zeros((2, 5), dtype=torch.int64), w, RING_K40_A12)
    with pytest.raises(ConfigError):
        ring_vecmat(torch.zeros((2, 4), dtype=torch.int64), w, RING_K40_A12, method="nope")  # type: ignore[arg-type]


def test_resolve_backend_and_device() -> None:
    assert resolve_backend(CPU, "auto") in ("cpu_int8", "cpu_f64")
    assert resolve_backend(CPU, "cpu_f32") == "cpu_f32"
    with pytest.raises(ConfigError):
        resolve_backend(CPU, "tpu")
    with pytest.raises(BackendUnavailableError):
        resolve_backend(CPU, "cuda_int8")
    with pytest.raises(BackendUnavailableError):
        resolve_backend(torch.device("cuda"), "cpu_f64")
    assert resolve_device("cpu") == CPU
    assert resolve_device("auto").type in ("cpu", "cuda")
    with pytest.raises(ConfigError):
        resolve_device("not a device")
    if not torch.cuda.is_available():
        with pytest.raises(BackendUnavailableError):
            resolve_device("cuda")


def test_cpu_int8_probe_rejects_inexact_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    real = torch._int_mm

    def saturating(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return real(a, b).clamp(-(1 << 20), 1 << 20)  # emulate an inexact/saturating kernel

    cpu_int8_available.cache_clear()
    monkeypatch.setattr(torch, "_int_mm", saturating)
    try:
        assert cpu_int8_available() is False
        assert resolve_backend(CPU, "auto") == "cpu_f64"
        with pytest.raises(BackendUnavailableError):
            resolve_backend(CPU, "cpu_int8")
    finally:
        monkeypatch.setattr(torch, "_int_mm", real)
        cpu_int8_available.cache_clear()


def test_cpu_int8_probe_handles_missing_op(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("not implemented for CPU")

    cpu_int8_available.cache_clear()
    monkeypatch.setattr(torch, "_int_mm", missing)
    try:
        assert cpu_int8_available() is False
    finally:
        monkeypatch.undo()
        cpu_int8_available.cache_clear()


@pytest.mark.gpu
def test_cuda_kernel_exact(ring: RingParams) -> None:  # pragma: no cover - GPU only
    x = _rand((5, 21), ring.k, 9)
    w = _w(13, 21, 9)
    got = RingKernel(w.cuda(), "cuda_int8").matmul(x, ring).cpu()
    assert got.tolist() == ringmm_reference(x, w, ring.k).tolist()
    u = _rand((2, 13), ring.k, 10)
    assert torch.equal(ring_vecmat(u, w.cuda(), ring), freivalds_precompute(u, w, ring))
