"""``pgw_fast.quantize_mask_pack`` is bit-identical to quantize_activations + mask_activations + pack_ring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core import native
from pgw.core.errors import RingOverflowError, ShapeError
from pgw.core.masking import mask_activations, mask_pack_u32
from pgw.core.packing import pack_ring
from pgw.core.ring import RingParams, quantize_activations

pytestmark = pytest.mark.skipif(
    not native.native_available("quantize_mask_pack"), reason="pgw_fast.quantize_mask_pack not built"
)

RINGS = ["K32A12C27", "K40A12", "K48A20", "K32A8"]
ROWS = [1, 5, 40]
WIDTHS = [3584, 18944, 896, 7]


def _reference(
    x: np.ndarray, r: np.ndarray, ring: RingParams, wnorm: float | None
) -> tuple[bytes, np.ndarray, np.ndarray]:
    xq, sx = quantize_activations(torch.from_numpy(x), ring, wnorm if ring.cap else None)
    x_hat = mask_activations(xq, torch.from_numpy(r), ring)
    payload = pack_ring(x_hat, ring)
    if ring.k == 32:  # the gateway's K=32 fast path must agree too
        x32, p32 = mask_pack_u32(xq.numpy(), r, ring)
        assert p32 == payload and np.array_equal(x32.astype(np.int64), x_hat.numpy())
    return payload, x_hat.numpy(), sx.numpy()


def _native(
    x: np.ndarray, r: np.ndarray, ring: RingParams, wnorm: float | None
) -> tuple[bytes, np.ndarray, np.ndarray]:
    return native.fast.quantize_mask_pack(x, r, ring.k, ring.a, ring.cap, float(wnorm or 0.0))


def _assert_same(x: np.ndarray, r: np.ndarray, ring: RingParams, wnorm: float | None) -> None:
    try:
        want = _reference(x, r, ring, wnorm)
    except RingOverflowError:
        with pytest.raises(ValueError, match=r"^ring_overflow:"):
            _native(x, r, ring, wnorm)
        return
    got = _native(x, r, ring, wnorm)
    assert isinstance(got[0], bytes) and got[0] == want[0]
    assert got[1].dtype == np.int64 and got[1].shape == x.shape and np.array_equal(got[1], want[1])
    assert got[2].dtype == np.float64 and got[2].shape == (x.shape[0],)
    assert np.array_equal(got[2].view(np.uint64), want[2].view(np.uint64))  # bitwise, incl. -0.0


def _mask(rng: np.random.Generator, shape: tuple[int, int], ring: RingParams) -> np.ndarray:
    return rng.integers(0, 1 << ring.k, size=shape, dtype=np.int64)


def _realistic_wnorm(rng: np.random.Generator, n: int) -> float:
    """Largest L2 norm of a few int8 weight rows (typical |w8| ~ 40 on average)."""
    w8 = np.clip(np.rint(rng.normal(0, 40, size=(4, n))), -127, 127)
    return float(np.sqrt((w8 * w8).sum(axis=1)).max())


def _activations(rng: np.random.Generator, rows: int, n: int, kind: str) -> np.ndarray:
    x = rng.normal(0, 1, size=(rows, n)).astype(np.float32)
    if kind == "outliers":
        ch = rng.choice(n, size=max(1, n // 500), replace=False)
        x[:, ch] *= 300
    elif kind == "zero_rows":
        x[::2] = 0
        if rows > 1:
            x[1, : n // 2] = -0.0
    elif kind == "ties":
        # amax = qmax * 2^-3 -> sx = 2^-3, values k/16 give exact half-integers after scaling
        x = (rng.integers(-2 * 2047, 2 * 2047, size=(rows, n)) / 16.0).astype(np.float32)
        x[:, 0] = 2047 / 8
    elif kind == "tiny":
        x *= np.float32(1e-30)
    elif kind == "huge":
        x *= np.float32(1e30)
    elif kind == "subnormal":
        x = (rng.integers(-50, 50, size=(rows, n)) * np.float32(1e-45)).astype(np.float32)
    return x


KINDS = ["normal", "outliers", "zero_rows", "ties", "tiny", "huge", "subnormal"]


@pytest.mark.parametrize("ring_name", RINGS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("n", WIDTHS)
def test_matches_python(ring_name: str, rows: int, n: int) -> None:
    ring = RingParams.from_name(ring_name)
    rng = np.random.default_rng(hash((ring_name, rows, n)) & 0xFFFF_FFFF)
    for kind in KINDS:
        x = _activations(rng, rows, n, kind)
        wn = _realistic_wnorm(rng, n)
        _assert_same(x, _mask(rng, x.shape, ring), ring, wn)


@pytest.mark.parametrize("n", [7, 896, 3584, 18944])
def test_capped_fewer_bits(n: int) -> None:
    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(n)
    base = _realistic_wnorm(rng, n)
    for factor in (1.0, 3.0, 10.0, 40.0, 200.0, 1000.0, 2.0**14):
        for kind in ("normal", "outliers", "zero_rows", "ties"):
            x = _activations(rng, 5, n, kind)
            _assert_same(x, _mask(rng, x.shape, ring), ring, base * factor)


def test_capped_mixed_rows_pick_different_bits() -> None:
    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(7)
    n = 3584
    x = rng.normal(0, 1, size=(8, n)).astype(np.float32)
    x[1] = 0
    x[2, 5] = 1e4  # one outlier: tiny norm -> full 12 bits
    x[3] = np.sign(x[3]) + 0  # flat row: large norm -> fewer bits
    wn = (1 << 27) / (2047 * np.sqrt(n)) * 1.5
    _, _, sx = _native(x, _mask(rng, x.shape, ring), ring, wn)
    assert len(set(np.round(sx * np.abs(x).max(axis=1).clip(1e-30), 6).tolist())) > 1  # varying qmax
    _assert_same(x, _mask(rng, x.shape, ring), ring, wn)


def test_non_contiguous_inputs() -> None:
    ring = RingParams.from_name("K40A12")
    rng = np.random.default_rng(3)
    xb = rng.normal(size=(6, 40)).astype(np.float32)
    rb = _mask(rng, (6, 40), ring)
    x, r = xb[::2, ::3], rb[::2, ::3]
    want = _reference(np.ascontiguousarray(x), np.ascontiguousarray(r), ring, None)
    got = _native(x, r, ring, None)
    assert got[0] == want[0] and np.array_equal(got[1], want[1]) and np.array_equal(got[2], want[2])


def test_empty_rows() -> None:
    ring = RingParams.from_name("K32A12C27")
    payload, x_hat, sx = _native(np.zeros((0, 16), np.float32), np.zeros((0, 16), np.int64), ring, 100.0)
    assert payload == b"" and x_hat.shape == (0, 16) and sx.shape == (0,)


@pytest.mark.parametrize("ring_name", RINGS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_raises_like_python(ring_name: str, bad: float) -> None:
    ring = RingParams.from_name(ring_name)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 896)).astype(np.float32)
    x[3, 17] = bad
    r = _mask(rng, x.shape, ring)
    with pytest.raises(RingOverflowError):
        _reference(x, r, ring, 3000.0)
    with pytest.raises(ValueError, match=r"^ring_overflow:"):
        _native(x, r, ring, 3000.0)


@pytest.mark.parametrize("wnorm", [2.0**27 * 2, float("inf"), float("nan")])
def test_no_bits_fit_raises_like_python(wnorm: float) -> None:
    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 896)).astype(np.float32)
    r = _mask(rng, x.shape, ring)
    with pytest.raises(RingOverflowError):
        _reference(x, r, ring, wnorm)
    with pytest.raises(ValueError, match=r"^ring_overflow:"):
        _native(x, r, ring, wnorm)


def test_shape_and_dtype_errors() -> None:
    ring = RingParams.from_name("K32A12C27")
    x = np.ones((2, 8), np.float32)
    r = np.zeros((2, 8), np.int64)
    with pytest.raises(ShapeError):
        mask_activations(torch.zeros((2, 7), dtype=torch.int64), torch.from_numpy(r), ring)
    for xx, rr in [
        (x, r[:, :7]),
        (x, r[:1]),
        (x.astype(np.float64), r),
        (x, r.astype(np.uint64)),
        (x, r.astype(np.int32)),
        (x[0], r[0]),
        (x.tolist(), r),
    ]:
        with pytest.raises(ValueError) as ei:
            native.fast.quantize_mask_pack(xx, rr, 32, 12, 27, 100.0)
        assert not str(ei.value).startswith("ring_overflow:")
    for k, a in [(33, 12), (64, 12), (0, 12), (32, 1), (32, 32)]:
        with pytest.raises(ValueError):
            native.fast.quantize_mask_pack(x, r, k, a, 0, 0.0)


def test_inputs_not_modified() -> None:
    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 896)).astype(np.float32)
    r = _mask(rng, x.shape, ring)
    x0, r0 = x.copy(), r.copy()
    _native(x, r, ring, 3000.0)
    assert np.array_equal(x, x0) and np.array_equal(r, r0)
