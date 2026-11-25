"""The 2-adic Freivalds check of provider hint operands (pgw.provision.hintcheck)."""

from __future__ import annotations

import numpy as np
import pytest

from pgw.core.prng import SeedExpander, new_seed
from pgw.hemask import pks
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.pks_backend import switch_hint_fast
from pgw.provision.hintcheck import (
    CHALLENGE_COLS,
    SOUNDNESS_BITS,
    LinearHCheck,
    _bits,
    a_times_x,
    decode_h56,
    encode_h56,
)

M56 = np.uint64((1 << 56) - 1)


def _setup(m: int = 96, n: int = 64, seed: bytes | None = None, key: str = "0.o"):
    seed = seed or new_seed()
    rng = np.random.default_rng(m * 1000 + n)
    w = rng.integers(-127, 128, (m, n), dtype=np.int8)
    a = pks.a_mlwe_rows(pks.public_ring_rows(seed, n, key))
    return seed, w, matmul_mod64(w, a) & M56


def _check(seed: bytes, w: np.ndarray, h: np.ndarray, key: str = "0.o", split: int | None = None) -> bool:
    ck = LinearHCheck(seed, key, *w.shape)
    ck.add_weights(0, w)
    if split:
        ck.add_h(split, h[split:])
        ck.add_h(0, h[:split])
    else:
        ck.add_h(0, h)
    return ck.passed()


def test_parameters() -> None:
    assert CHALLENGE_COLS == 42 and SOUNDNESS_BITS == 41 >= 40


@pytest.mark.parametrize("n", [16, 64, 896])
def test_a_times_x_matches_the_expanded_product(n: int) -> None:
    seed = new_seed()
    a = pks.a_mlwe_rows(pks.public_ring_rows(seed, n, "3.down"))
    x = _bits(SeedExpander(new_seed()), (pks.NU, 7), "x")
    want = matmul_mod64(x.astype(np.int8), a, small_left=False) & M56
    assert np.array_equal(a_times_x(seed, "3.down", n, x) & M56, want)


def test_h56_codec_roundtrip() -> None:
    rng = np.random.default_rng(1)
    h = rng.integers(0, 1 << 56, (5, pks.NU), dtype=np.uint64)
    h[0, 0], h[1, 1] = 0, (1 << 56) - 1
    data = encode_h56(h)
    assert len(data) == 5 * pks.NU * 7
    assert np.array_equal(decode_h56(data, 5), h)
    with pytest.raises(Exception):
        decode_h56(data[:-1], 5)


def test_switch_hint_fast_is_bit_identical() -> None:
    rng = np.random.default_rng(2)
    h = rng.integers(0, 1 << 56, (64, pks.NU), dtype=np.uint64)
    h[0, :4] = [0, 1, (1 << 56) - 1, 1 << 55]
    assert np.array_equal(switch_hint_fast(h), pks.switch_hint(h, pks.PKSParams()))


def test_honest_operand_accepted_in_any_block_order() -> None:
    seed, w, h = _setup()
    assert _check(seed, w, h)
    assert _check(seed, w, h, split=32)


def test_incomplete_or_repeated_rows_fail() -> None:
    seed, w, h = _setup()
    ck = LinearHCheck(seed, "0.o", *w.shape)
    ck.add_weights(0, w)
    ck.add_h(0, h[:64])
    assert not ck.passed()  # rows 64.. never fed
    ck = LinearHCheck(seed, "0.o", *w.shape)
    ck.add_h(0, h)
    with pytest.raises(Exception):
        ck.add_h(0, h[:32])


@pytest.mark.parametrize("bit", [0, 1, 27, 31, 48, 55])
def test_single_wrong_entry_rejected(bit: int) -> None:
    """One entry off by 2^bit (every 2-adic valuation, including the 2^55 error a single Z_2^56 vector misses half
    the time) is rejected in every trial."""
    seed, w, h = _setup()
    rng = np.random.default_rng(bit)
    for _ in range(25):
        bad = h.copy()
        i, j = int(rng.integers(0, h.shape[0])), int(rng.integers(0, h.shape[1]))
        bad[i, j] = (bad[i, j] + np.uint64(1 << bit)) & M56
        assert not _check(seed, w, bad)


def test_wrong_seed_rejected() -> None:
    """H built for another a_seed (or another linear's stream) is rejected."""
    seed, w, _ = _setup()
    _, _, h_other = _setup(seed=new_seed())
    assert not _check(seed, w, h_other)
    a_other_key = pks.a_mlwe_rows(pks.public_ring_rows(seed, w.shape[1], "1.o"))
    assert not _check(seed, w, matmul_mod64(w, a_other_key) & M56)


def test_wrong_weights_rejected() -> None:
    """H of a pack differing in one weight (by 1) is rejected against the pinned weights."""
    seed, w, _ = _setup()
    w2 = w.copy()
    w2[3, 5] = w2[3, 5] + (1 if w2[3, 5] < 127 else -1)
    a = pks.a_mlwe_rows(pks.public_ring_rows(seed, w.shape[1], "0.o"))
    assert not _check(seed, w, matmul_mod64(w2, a) & M56)


@pytest.mark.parametrize("pattern", ["high_bit_everywhere", "high_bit_rank1", "row", "column", "low_rank_2adic",
                                     "rounding_shift"])
def test_structured_errors_rejected(pattern: str) -> None:
    """Errors chosen to defeat weaker checks: 2^55 on many entries (parity structure), rank-1 / rank-2 patterns,
    a whole row / column, and a shift of every entry by one rounding step of the hint switch (2^8)."""
    seed, w, h = _setup()
    m, c = h.shape
    rng = np.random.default_rng(len(pattern))
    e = np.zeros((m, c), dtype=np.uint64)
    if pattern == "high_bit_everywhere":
        e[:] = np.uint64(1 << 55)
    elif pattern == "high_bit_rank1":
        r = rng.integers(0, 2, m, dtype=np.uint64)
        s = rng.integers(0, 2, c, dtype=np.uint64)
        r[0] = s[0] = 1
        e = (np.outer(r, s) & np.uint64(1)) << np.uint64(55)
    elif pattern == "row":
        e[7] = rng.integers(0, 1 << 56, c, dtype=np.uint64)
    elif pattern == "column":
        e[:, 100] = np.uint64(1 << 54)
    elif pattern == "low_rank_2adic":
        r1, s1 = rng.integers(0, 4, m, dtype=np.uint64), rng.integers(0, 4, c, dtype=np.uint64)
        r1[0] = s1[0] = 1
        e = (np.outer(r1, s1) << np.uint64(53)) & M56
    else:
        e[:] = np.uint64(256)
    assert e.any()
    for _ in range(10):
        assert not _check(seed, w, (h + e) & M56)


def test_soundness_is_not_one_vector_per_bit() -> None:
    """Sanity of the argument's bound ``2^(1-d)``: with ONE binary column per side a 2^55 single-entry error passes
    unless both challenge bits at its position are 1, i.e. 3/4 of the time; with ``d`` columns it passes only when
    the error's row of ``U`` or column of ``X`` is all zero (``<= 2 * 2^-d``)."""
    seed, w, h = _setup(m=32, n=16)
    bad = h.copy()
    bad[0, 0] = (bad[0, 0] + np.uint64(1 << 55)) & M56
    passes = 0
    for _ in range(200):
        ck = LinearHCheck(seed, "0.o", *w.shape, cols=1)
        ck.add_weights(0, w)
        ck.add_h(0, bad)
        passes += ck.passed()
    assert 120 < passes < 180  # ~ 3/4 of 200
