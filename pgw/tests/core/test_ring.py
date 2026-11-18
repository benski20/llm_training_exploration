"""Tests for pgw.core.ring."""

import pytest
import torch

from pgw.core import (
    DEFAULT_RING,
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    ConfigError,
    RingOverflowError,
    RingParams,
    ShapeError,
    balanced_digits,
    center,
    dequantize_outputs,
    quantize_activations,
    quantize_weight_rows,
    to_ring,
)

RINGS = [RING_K32_A8, RING_K40_A12, RING_K48_A20]


def test_presets_and_names():
    assert DEFAULT_RING == RING_K40_A12
    assert RING_K40_A12.name == "K40A12"
    assert RingParams.from_name("k48a20") == RING_K48_A20
    assert RING_K40_A12.limbs == 5 and RING_K40_A12.bytes_per_element == 5
    assert RING_K32_A8.qmax == 127
    assert RING_K48_A20.modulus == 1 << 48


@pytest.mark.parametrize("k,a", [(12, 4), (36, 12), (64, 20), (40, 1), (40, 33)])
def test_invalid_params(k, a):
    with pytest.raises(ConfigError):
        RingParams(k, a)


@pytest.mark.parametrize("bad", ["", "K40", "40A12", "KxxA1"])
def test_bad_names(bad):
    with pytest.raises(ConfigError):
        RingParams.from_name(bad)


@pytest.mark.parametrize("p", RINGS)
def test_headroom(p):
    n = p.max_in_features()
    assert p.output_bound(n) < p.half
    assert p.output_bound(n + 1) >= p.half or n + 1 > p.max_in_features()
    p.check_in_features(n)
    with pytest.raises(RingOverflowError):
        p.check_in_features(n + 1)
    assert p.headroom_bits(4096) > 0


def test_weight_quantization():
    w = torch.randn(33, 70, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    w[5] = 0.0
    w8, sw = quantize_weight_rows(w)
    assert w8.dtype == torch.int8 and sw.shape == (33,)
    assert int(w8.abs().max()) <= 127 and int(w8.min()) >= -127
    assert float(sw[5]) == 1.0 and int(w8[5].abs().max()) == 0
    err = (w8.double() * sw.double()[:, None] - w).abs()
    assert bool((err <= sw.double()[:, None] / 2 + 1e-12).all())
    assert bool((w8.abs().amax(1)[torch.arange(33) != 5] == 127).all())
    with pytest.raises(ShapeError):
        quantize_weight_rows(torch.zeros(3))


@pytest.mark.parametrize("p", RINGS)
def test_activation_quantization_roundtrip(p):
    x = torch.randn(7, 50) * 3
    x[2] = 0
    xq, sx = quantize_activations(x, p)
    assert xq.dtype == torch.int64 and int(xq.abs().max()) <= p.qmax
    assert float(sx[2]) == 1.0
    back = xq.double() * sx[:, None]
    assert bool(((back - x.double()).abs() <= sx[:, None] / 2 + 1e-9).all())
    with pytest.raises(RingOverflowError):
        quantize_activations(torch.tensor([[1.0, float("nan")]]), p)
    with pytest.raises(ShapeError):
        quantize_activations(torch.zeros(3), p)


def test_dequantize_with_bias():
    yq = torch.tensor([[2, -4]], dtype=torch.int64)
    y = dequantize_outputs(yq, torch.tensor([0.5]), torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
    assert torch.allclose(y, torch.tensor([[2.0, -3.0]]))


@pytest.mark.parametrize("p", RINGS)
def test_center_and_to_ring(p):
    v = torch.tensor([0, 1, -1, p.half - 1, -p.half, 5 * p.modulus + 3], dtype=torch.int64)
    r = to_ring(v, p)
    assert bool(((r >= 0) & (r < p.modulus)).all())
    c = center(r, p)
    assert c.tolist() == [0, 1, -1, p.half - 1, -p.half, 3]


@pytest.mark.parametrize("p", RINGS)
@pytest.mark.parametrize("bits", [8, 13, 20, 24])
def test_balanced_digits_reconstruct(p, bits):
    g = torch.Generator().manual_seed(1)
    v = torch.randint(0, p.modulus, (1000,), generator=g, dtype=torch.int64)
    v[:4] = torch.tensor([0, p.modulus - 1, p.half, p.half - 1])
    ds = balanced_digits(v, p.k, bits)
    assert len(ds) == -(-p.k // bits)
    acc = torch.zeros_like(v)
    for i, d in enumerate(ds):
        assert int(d.min()) >= -(1 << (bits - 1)) and int(d.max()) < (1 << (bits - 1))
        acc = (acc + ((d & ((1 << (p.k - bits * i)) - 1)) << (bits * i) if bits * i < p.k else 0)) & p.mask
    assert bool((acc == v).all())


def test_balanced_digits_int8_castable():
    v = torch.randint(0, 1 << 40, (500,))
    for d in balanced_digits(v, 40, 8):
        assert bool((d.to(torch.int8).to(torch.int64) == d).all())
    with pytest.raises(ConfigError):
        balanced_digits(v, 40, 1)
