"""Tests for one-time-mask math, including the statistical privacy sanity check."""

import math

import numpy as np
import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    IntegrityError,
    PreparedWeight,
    RingOverflowError,
    SeedExpander,
    ShapeError,
    check_output_range,
    mask_activations,
    unmask_outputs,
)

RINGS = [RING_K32_A8, RING_K40_A12, RING_K48_A20]


@pytest.mark.parametrize("p", RINGS)
def test_roundtrip_exact(p):
    sx = SeedExpander(bytes(32))
    t, m, n = 9, 40, 72
    xq = sx.uniform_int((t, n), -p.qmax, p.qmax + 1, "x")
    w8 = sx.uniform_int((m, n), -127, 128, "w").to(torch.int8)
    r = sx.ring_uniform((t, n), p.k, "r")
    x_hat = mask_activations(xq, r, p)
    assert bool(((x_hat >= 0) & (x_hat < p.modulus)).all())
    pw = PreparedWeight(w8)
    yq = unmask_outputs(pw(x_hat, p), pw(r, p), p, in_features=n)
    assert torch.equal(yq, (xq.double() @ w8.double().t()).to(torch.int64))


def test_shape_checks():
    p = RING_K40_A12
    with pytest.raises(ShapeError):
        mask_activations(torch.zeros((2, 3), dtype=torch.int64), torch.zeros((2, 4), dtype=torch.int64), p)
    with pytest.raises(ShapeError):
        mask_activations(torch.zeros((2, 3), dtype=torch.int32), torch.zeros((2, 3), dtype=torch.int64), p)
    with pytest.raises(ShapeError):
        unmask_outputs(torch.zeros((2, 3), dtype=torch.int64), torch.zeros((1, 3), dtype=torch.int64), p)


def test_range_check_catches_top_bit_tamper():
    p = RING_K40_A12
    n = 64
    yq = torch.zeros((1, 4), dtype=torch.int64)
    check_output_range(yq, p, n)
    y_hat = torch.zeros((1, 4), dtype=torch.int64)
    wr = torch.zeros((1, 4), dtype=torch.int64)
    y_hat[0, 1] = p.half  # flip the top bit
    with pytest.raises(IntegrityError):
        unmask_outputs(y_hat, wr, p, in_features=n)


@pytest.mark.parametrize("p", [RING_K32_A8, RING_K40_A12])
def test_server_view_is_uniform_and_uncorrelated(p):
    """What the server sees (x_hat) must look uniform even for highly structured plaintext."""
    sx = SeedExpander(bytes(range(32)))
    t, n = 256, 256
    xq = torch.zeros((t, n), dtype=torch.int64)
    xq[:, ::3] = p.qmax  # very non-uniform plaintext
    r = sx.ring_uniform((t, n), p.k, "r")
    x_hat = mask_activations(xq, r, p)
    top = (x_hat >> (p.k - 8)).numpy().ravel()
    cnt = np.bincount(top, minlength=256)
    exp = top.size / 256
    chi2 = float(((cnt - exp) ** 2 / exp).sum())
    assert chi2 < 255 + 6 * math.sqrt(2 * 255)  # ~6 sigma
    corr = np.corrcoef(x_hat.double().numpy().ravel(), xq.double().numpy().ravel())[0, 1]
    assert abs(corr) < 0.02


@pytest.mark.parametrize("rows,n,m", [(1, 3584, 37888), (5, 70, 33), (40, 64, 16), (0, 8, 8)])
def test_u32_fast_path_is_identical_to_the_generic_path(rows: int, n: int, m: int) -> None:
    """K=32: mask_pack_u32 == mask_activations + pack_ring, and unmask_outputs_u32 == unpack + unmask_outputs_u64,
    including the range-check verdicts at +-bound, +-(bound + 1) and the int32 extremes."""
    from pgw.core import DEFAULT_RING as ring  # noqa: N811
    from pgw.core.masking import mask_pack_u32, unmask_outputs_u32, unmask_outputs_u64
    from pgw.core.packing import pack_ring, unpack_ring_u64

    assert ring.k == 32
    g = torch.Generator().manual_seed(rows * 7 + n)
    xq = torch.randint(-ring.qmax, ring.qmax + 1, (rows, n), generator=g)
    if rows:
        xq[0, : min(n, 2)] = torch.tensor([ring.qmax, -ring.qmax])[: min(n, 2)]
    r = torch.randint(0, 1 << 32, (rows, n), generator=g)
    x32, payload = mask_pack_u32(xq.numpy(), r.numpy(), ring)
    ref = mask_activations(xq, r, ring)
    assert payload == pack_ring(ref, ring) and np.array_equal(x32.astype(np.int64), ref.numpy())
    with pytest.raises(RingOverflowError):
        if rows:
            bad = xq.clone()
            bad[0, 0] = ring.qmax + 1
            mask_pack_u32(bad.numpy(), r.numpy(), ring)
        else:
            raise RingOverflowError("empty")
    bound = ring.output_bound(n)
    for yvals in ("rand", "edge"):
        if yvals == "rand":
            yq = torch.randint(-bound, bound + 1, (rows, m), generator=g)
        else:
            choices = torch.tensor([bound, -bound, bound + 1, -bound - 1, (1 << 31) - 1, -(1 << 31), 0])
            yq = choices[torch.randint(0, len(choices), (rows, m), generator=g)]
        wr = torch.randint(0, 1 << 32, (rows, m), generator=g)
        y_hat = pack_ring((yq + wr) & ring.mask, ring)
        try:
            want = unmask_outputs_u64(unpack_ring_u64(y_hat, (rows, m), ring), wr.numpy(), ring, n)
        except IntegrityError:
            with pytest.raises(IntegrityError):
                unmask_outputs_u32(y_hat, wr.numpy(), ring, n)
            continue
        got = unmask_outputs_u32(y_hat, wr.numpy(), ring, n)
        assert got.dtype == np.int32 and np.array_equal(got.astype(np.int64), want)
