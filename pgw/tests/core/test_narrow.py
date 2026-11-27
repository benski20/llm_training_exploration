"""Narrow wire rings (``pgw.core.narrow``, ``pgw.engine.narrow_backend``; opt-in prototype): exact outputs, sound
row bounds, uniform masked values, data-independent wire sizes and carry work, and unchanged integrity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import narrow as nw
from pgw.core.errors import ConfigError, IntegrityError, ProtocolError
from pgw.core.ring import RingParams, quantize_activations
from pgw.engine.backends import _row_norm_max

RING = RingParams(32, 12, 27)


def _w8(m: int, n: int, seed: int, spread: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((m, n))
    if spread:  # rows with very different norms (as real int8 weights: per-row absmax 127, kurtosis varies)
        w *= rng.lognormal(0, 1.0, (m, 1))
        w[:, : max(1, n // 50)] *= rng.uniform(1, 30, (m, 1))  # outlier columns
    w8 = np.rint(w / (np.abs(w).max(axis=1, keepdims=True) / 127)).clip(-127, 127).astype(np.int8)
    return w8


def _xq(t: int, n: int, w8: np.ndarray, seed: int, kind: str = "normal") -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "normal":
        x = rng.standard_normal((t, n))
    elif kind == "heavy":
        x = rng.standard_t(1.5, (t, n))
    elif kind == "flat":  # every element at the extremes: the cap binds, carries are densest
        x = np.sign(rng.standard_normal((t, n))) + 1e-3 * rng.standard_normal((t, n))
    else:  # aligned with the largest-norm row: Cauchy-Schwarz near equality for that row
        j = int(np.argmax((w8.astype(np.int64) ** 2).sum(1)))
        x = np.repeat(w8[j : j + 1].astype(np.float64), t, 0) + 1e-6 * rng.standard_normal((t, n))
    wnorm = _row_norm_max(torch.from_numpy(w8))
    xq, _ = quantize_activations(torch.from_numpy(x.astype(np.float32)), RING, wnorm)
    return xq.numpy()


def _exact(xq: np.ndarray, w8: np.ndarray) -> np.ndarray:
    return xq.astype(np.int64) @ w8.astype(np.int64).T


def _run(w8: np.ndarray, xq: np.ndarray, plan: nw.NarrowPlan, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w8t = torch.from_numpy(w8).t().contiguous()
    r = rng.integers(0, 1 << plan.in_bits, xq.shape, dtype=np.int64)
    xh, c = nw.mask_inputs(xq, r, plan)
    up = nw.pack_up(xh, plan)
    assert len(up) == plan.up_bytes(xq.shape[0])
    down = nw.provider_apply(w8t, nw.unpack_up(up, xq.shape[0], plan), plan)
    assert len(down) == plan.down_bytes(xq.shape[0])
    corr = nw.mask_correction(w8t, r, plan)
    budget = nw.carry_budget(plan.in_features, plan.in_bits, RING.qmax)
    ct, _ = nw.carry_term(w8t, c, plan, budget)
    return nw.decode_outputs(nw.unpack_down(down, xq.shape[0], plan), corr, ct, plan)


@pytest.mark.parametrize("kind", ["normal", "heavy", "flat", "aligned"])
def test_row_bounds_hold_for_capped_inputs(kind: str) -> None:
    """Every genuine output satisfies |y_j| <= bound_j < 2^(w_j - 1); widths never exceed C + 1."""
    w8 = _w8(96, 160, 1)
    plan = nw.plan_linear(w8, RING)
    y = _exact(_xq(64, 160, w8, 2, kind), w8)
    assert (np.abs(y) <= plan.bound[None, :]).all()
    assert (plan.bound < (1 << (plan.out_bits - 1))).all()
    assert plan.out_bits.max() <= RING.cap + 1 and plan.in_bits == plan.out_bits.max()
    assert plan.out_bits.min() < plan.out_bits.max()  # the spread rows really are narrower


def test_widths_are_minimal_per_row_bound() -> None:
    """w_j is the smallest power of two holding [-B_j, B_j] (the coloring floor of the note, section 3)."""
    w8 = _w8(64, 128, 3)
    plan = nw.plan_linear(w8, RING)
    for b, w in zip(plan.bound.tolist(), plan.out_bits.tolist(), strict=True):
        assert (1 << w) >= 2 * b + 1 and (w == 1 or (1 << (w - 1)) < 2 * b + 1)


@pytest.mark.parametrize("in_bits", [None, 12, 16, 20, 24])
@pytest.mark.parametrize("kind", ["normal", "heavy", "flat", "aligned"])
def test_exact_both_modes(in_bits: int | None, kind: str) -> None:
    w8 = _w8(80, 144, 4)
    xq = _xq(9, 144, w8, 5, kind)
    plan = nw.plan_linear(w8, RING, in_bits)
    assert np.array_equal(_run(w8, xq, plan, 6), _exact(xq, w8))


def test_uncapped_ring_exact() -> None:
    ring = RingParams(40, 12)
    w8 = _w8(48, 64, 7)
    plan = nw.plan_linear(w8, ring)
    assert plan.out_bits.max() <= ring.k
    rng = np.random.default_rng(8)
    xq = rng.integers(-ring.qmax, ring.qmax + 1, (5, 64))
    w8t = torch.from_numpy(w8).t().contiguous()
    r = rng.integers(0, 1 << plan.in_bits, xq.shape)
    xh, _ = nw.mask_inputs(xq, r, plan)
    down = nw.provider_apply(w8t, xh, plan)
    got = nw.decode_outputs(nw.unpack_down(down, 5, plan), nw.mask_correction(w8t, r, plan), None, plan)
    assert np.array_equal(got, _exact(xq, w8))


def test_weight_free_mode_reuses_k32_masks() -> None:
    """Weight-free mode needs no new masks: today's K32 slot (r uniform mod 2^32, W8 r mod 2^32) truncated."""
    w8 = _w8(64, 96, 9)
    xq = _xq(7, 96, w8, 10)
    plan = nw.plan_linear(w8, RING)
    assert not plan.carry_rows.size
    rng = np.random.default_rng(11)
    r32 = rng.integers(0, 1 << 32, xq.shape, dtype=np.int64)
    wr32 = (r32 @ w8.astype(np.int64).T) & 0xFFFFFFFF  # what the K32 mask store holds
    r = r32 & ((1 << plan.in_bits) - 1)
    xh, _ = nw.mask_inputs(xq, r, plan)
    w8t = torch.from_numpy(w8).t().contiguous()
    down = nw.provider_apply(w8t, xh, plan)
    corr = wr32.astype(np.uint64) & ((np.uint64(1) << plan.out_bits.astype(np.uint64)) - np.uint64(1))[None, :]
    assert np.array_equal(nw.decode_outputs(nw.unpack_down(down, 7, plan), corr, None, plan), _exact(xq, w8))


@pytest.mark.parametrize("x", [-2047, -1, 0, 1, 777, 2047])
def test_masked_input_is_a_permutation_of_the_ring(x: int) -> None:
    """Exact uniformity: for any input value, r -> x_hat is a bijection of Z_(2^m) (here m = 12, all r)."""
    w8 = _w8(8, 8, 12)
    plan = nw.plan_linear(w8, RING, 12)
    r = np.arange(1 << 12, dtype=np.int64).reshape(-1, 8)
    xh, c = nw.mask_inputs(np.full(r.shape, x, dtype=np.int64), r, plan)
    assert np.array_equal(np.sort(xh.reshape(-1)), np.arange(1 << 12, dtype=np.uint64))
    assert set(np.unique(c).tolist()) <= {-1, 0, 1}
    # the carry probability is exactly |x| / 2^m (client-local; it never leaves the client)
    assert int(np.count_nonzero(c)) == abs(x)


def test_provider_view_independent_of_data() -> None:
    """Statistical check on the wire bytes: extreme and zero inputs give the same x_hat distribution, and the
    payload length depends only on (rows, plan)."""
    w8 = _w8(16, 64, 13)
    plan = nw.plan_linear(w8, RING, 16)
    rng = np.random.default_rng(14)
    hist = []
    for xval in (0, 2047, -2047):
        xs = np.full((512, 64), xval, dtype=np.int64)
        r = rng.integers(0, 1 << 16, xs.shape)
        xh, _ = nw.mask_inputs(xs, r, plan)
        up = nw.pack_up(xh, plan)
        assert len(up) == plan.up_bytes(512)
        h = np.bincount((nw.unpack_up(up, 512, plan) >> np.uint64(12)).astype(np.int64).reshape(-1), minlength=16)
        hist.append(h)
    exp = 512 * 64 / 16
    for h in hist:  # chi-square, 15 dof: 99.99th percentile ~ 44
        assert float(((h - exp) ** 2 / exp).sum()) < 44.0


def test_carry_budget_is_data_independent_and_rarely_exceeded() -> None:
    n, m = 896, 16
    b = nw.carry_budget(n, m, RING.qmax)
    p = RING.qmax / 2 ** m
    assert n * p < b < n
    assert nw.carry_budget(n, m, RING.qmax, -10) < b
    rng = np.random.default_rng(15)  # worst case for the carry count: every |x_i| = qmax
    counts = (rng.random((20000, n)) < p).sum(1)
    assert int((counts > b).sum()) == 0
    w8 = _w8(32, n, 16)
    plan = nw.plan_linear(w8, RING, m)
    c = np.zeros((3, n), dtype=np.int8)
    c[1, :5] = 1
    c[2, 7] = -1
    _, over = nw.carry_term(torch.from_numpy(w8).t().contiguous(), c, plan, b)
    assert over == 0


def test_tampered_reply_is_detected() -> None:
    """Any change of the returned residues gives a wrong integer; range + prime-field Freivalds catch it."""
    from pgw.core.freivalds import P31, prime_check, prime_precompute

    w8 = _w8(64, 96, 17)
    xq = _xq(4, 96, w8, 18)
    plan = nw.plan_linear(w8, RING, 16)
    w8t = torch.from_numpy(w8).t().contiguous()
    rng = np.random.default_rng(19)
    r = rng.integers(0, 1 << 16, xq.shape)
    xh, c = nw.mask_inputs(xq, r, plan)
    down = bytearray(nw.provider_apply(w8t, xh, plan))
    corr = nw.mask_correction(w8t, r, plan)
    ct, _ = nw.carry_term(w8t, c, plan, nw.carry_budget(96, 16, RING.qmax))
    u = rng.integers(0, P31, (1, 64))
    uw = prime_precompute(torch.from_numpy(u), torch.from_numpy(w8)).numpy()
    good = nw.decode_outputs(nw.unpack_down(bytes(down), 4, plan), corr, ct, plan)
    assert prime_check(xq, good, u, uw).all()
    caught = 0
    for pos in rng.choice(len(down), 64, replace=False):
        bad = bytearray(down)
        bad[int(pos)] ^= 1 << int(rng.integers(0, 8))
        try:
            y = nw.decode_outputs(nw.unpack_down(bytes(bad), 4, plan), corr, ct, plan)
        except (IntegrityError, ProtocolError):  # range check, or a flipped padding bit (strict parsing)
            caught += 1
            continue
        if not prime_check(xq, y, u, uw).all():
            caught += 1
    assert caught == 64


def test_bad_inputs_refused() -> None:
    w8 = _w8(8, 16, 20)
    plan = nw.plan_linear(w8, RING, 16)
    with pytest.raises(ConfigError):
        nw.plan_linear(w8, RING, 11)  # below the activation width
    with pytest.raises(ConfigError):
        nw.mask_inputs(np.zeros((1, 16), np.int64), np.full((1, 16), 1 << 16), plan)
    with pytest.raises(ProtocolError):
        nw.unpack_up(b"\x00" * 3, 1, plan)
    with pytest.raises(ProtocolError):
        nw.unpack_down(b"\x00" * 3, 1, plan)


# ---- end to end on the tiny model: bit-identical to the plain reference --------------------------------------


def _models(hf_dir: Path, in_bits: int | None, provider: Any = None) -> tuple[Any, Any, Any]:
    from pgw.engine.backends import PlainBackend
    from pgw.engine.masked_head import MaskedHeadPack, enable_masked_head
    from pgw.engine.model import TransformerModel
    from pgw.engine.narrow_backend import NarrowMaskedBackend
    from pgw.engine.pack import InMemoryPack

    pack = MaskedHeadPack(InMemoryPack.from_hf(hf_dir))
    be = NarrowMaskedBackend(pack, RING, in_bits=in_bits, provider=provider)
    m = TransformerModel(pack, be)
    enable_masked_head(m)
    p = TransformerModel(pack, PlainBackend(pack, RING))
    enable_masked_head(p)
    return m, p, be


@pytest.mark.parametrize("in_bits", [None, 14, 16, 20])
def test_tiny_model_bit_identical(tiny_hf_dir: Path, in_bits: int | None) -> None:
    m, p, be = _models(tiny_hf_dir, in_bits)
    ids = torch.tensor([5, 17, 99, 3, 250, 41, 8, 77, 123, 9])
    with torch.no_grad():
        c1, c2 = m.new_cache(32), p.new_cache(32)
        a, b = m.forward(ids, c1), p.forward(ids, c2)
        assert torch.equal(a, b)
        for _ in range(4):
            x = torch.tensor([int(a[-1].argmax())])
            a, b = m.forward(x, c1), p.forward(x, c2)
            assert torch.equal(a, b)
    w = be.wire
    assert w.up < w.k_up and w.down < w.k_down
    assert w.carry_overflows == 0
    if in_bits is not None:
        assert w.up <= w.k_up * in_bits / 32 + 64 * len(w.per_linear) * 8


def test_tiny_model_tampering_provider_refused(tiny_hf_dir: Path) -> None:
    def evil(w8t: torch.Tensor, xh: np.ndarray, plan: nw.NarrowPlan) -> bytes:
        good = bytearray(nw.provider_apply(w8t, xh, plan))
        good[len(good) // 2] ^= 0x10
        return bytes(good)

    m, _, _be = _models(tiny_hf_dir, 16, evil)
    with torch.no_grad(), pytest.raises(IntegrityError):
        m.forward(torch.tensor([5, 17, 99]), m.new_cache(8))
