"""Tests for Freivalds integrity checks."""

import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    IntegrityError,
    PreparedWeight,
    SeedExpander,
    ShapeError,
    freivalds_check,
    freivalds_precompute,
    freivalds_verify,
    ringmm_reference,
    soundness_bits,
)

RINGS = [RING_K32_A8, RING_K40_A12, RING_K48_A20]


def _setup(p, t=8, m=24, n=40, c=2, seed=0):
    sx = SeedExpander(bytes([seed]) * 32)
    w8 = sx.uniform_int((m, n), -127, 128, "w").to(torch.int8)
    x_hat = sx.ring_uniform((t, n), p.k, "x")
    y_hat = PreparedWeight(w8).matmul(x_hat, p)
    u = sx.ring_uniform((c, m), p.k, "u")
    uw = freivalds_precompute(u, w8, p)
    return w8, x_hat, y_hat, u, uw


@pytest.mark.parametrize("p", RINGS)
def test_precompute_exact(p):
    w8, _, _, u, uw = _setup(p)
    ref = ringmm_reference(u, w8.t().contiguous(), p.k)
    assert torch.equal(uw, torch.from_numpy(ref))


@pytest.mark.parametrize("p", RINGS)
def test_accepts_genuine(p):
    _, x_hat, y_hat, u, uw = _setup(p)
    assert bool(freivalds_check(x_hat, y_hat, u, uw, p).all())
    freivalds_verify(x_hat, y_hat, u, uw, p, linear_id="0.qkv")


@pytest.mark.parametrize("p", RINGS)
@pytest.mark.parametrize("delta", [1, 3, 255, 1 << 20])
def test_rejects_low_valuation_tamper(p, delta):
    _, x_hat, y_hat, u, uw = _setup(p, seed=delta % 251)
    bad = y_hat.clone()
    bad[3, 5] = (bad[3, 5] + delta) & p.mask
    ok = freivalds_check(x_hat, bad, u, uw, p)
    assert bool(ok[:3].all()) and bool(ok[4:].all())
    # an odd error is always detected; even ones with overwhelming probability for these seeds
    assert not bool(ok[3])
    with pytest.raises(IntegrityError):
        freivalds_verify(x_hat, bad, u, uw, p)


def test_rejection_rate_for_high_valuation_errors():
    """Errors of valuation v slip past one vector with probability 2^-(K-v)."""
    p = RING_K32_A8
    misses = 0
    trials = 400
    for s in range(trials):
        _, x_hat, y_hat, u, uw = _setup(p, t=1, m=8, n=8, c=1, seed=s % 256)
        sx = SeedExpander(s.to_bytes(32, "little"))
        u = sx.ring_uniform((1, 8), p.k, "u2")
        uw = freivalds_precompute(u, _setup(p, t=1, m=8, n=8, c=1, seed=s % 256)[0], p)
        bad = y_hat.clone()
        bad[0, 0] = (bad[0, 0] + (1 << 30)) & p.mask  # valuation 30 -> miss prob 1/4
        misses += int(bool(freivalds_check(x_hat, bad, u, uw, p).all()))
    assert 0.12 * trials < misses < 0.40 * trials


def test_shape_errors():
    p = RING_K40_A12
    w8, x_hat, y_hat, u, uw = _setup(p)
    with pytest.raises(ShapeError):
        freivalds_precompute(u[:, :3], w8, p)
    with pytest.raises(ShapeError):
        freivalds_check(x_hat[:2], y_hat, u, uw, p)
    with pytest.raises(ShapeError):
        freivalds_check(x_hat, y_hat, u[:, :3], uw, p)


def test_soundness_bits():
    assert soundness_bits(RING_K40_A12, 4096, 2) >= 12
    assert soundness_bits(RING_K48_A20, 4096, 1) >= 1
    assert soundness_bits(RING_K40_A12, 4096, 4) == 2 * soundness_bits(RING_K40_A12, 4096, 2)
