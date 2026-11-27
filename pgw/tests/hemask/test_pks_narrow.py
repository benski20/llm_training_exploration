"""pks-v3 with a narrow message ring (``pgw.hemask.pks_narrow``, research prototype): the reference round trip gives
``W8 r mod 2^k`` exactly with 30-bit replies at k = 28, and the noise margin is at least v3's."""

from __future__ import annotations

import numpy as np
import pytest

from pgw.core.errors import ConfigError
from pgw.hemask import pks
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.pks_narrow import PKSNarrowParams, margin_bits, smallest_reply_bits

M56 = np.uint64((1 << 56) - 1)
WORST = 127 * np.sqrt(29568)  # all-127 rows of the widest 72B input (SPEC 7.9's worst case)


def test_reply_bits_and_margin() -> None:
    assert smallest_reply_bits(28, WORST) == 30
    assert margin_bits(PKSNarrowParams(k=28, yb=30), WORST) >= margin_bits(pks.PKSParams(), WORST)
    assert PKSNarrowParams(k=28, yb=30).noise_ok(WORST, margin_bits=1.0)
    with pytest.raises(ConfigError):
        PKSNarrowParams(k=28, yb=27)
    with pytest.raises(ConfigError):  # the production class keeps its guard
        pks.PKSParams(k=28, yb=30)


@pytest.mark.parametrize("r_bits", [28, 16])
def test_reference_round_trip_exact_at_k28(r_bits: int) -> None:
    """Upload, provider reply (unchanged GEMM and key switch, 30-bit c0) and decryption: W8 r mod 2^28."""
    par = PKSNarrowParams(k=28, yb=30)
    m, n, rows = 64, 48, 256
    rng = np.random.default_rng(21)
    w8 = rng.integers(-127, 128, (m, n), dtype=np.int64)
    w8[0] = 127
    r = rng.integers(0, 1 << r_bits, (rows, n), dtype=np.int64)
    a_ring = pks.public_ring_rows(b"a" * 32, n, "0.o")
    ct = pks.encrypt_rlwe(r, a_ring, par, _seed=b"e" * 32)
    k_hat = pks.ks_a(b"k" * 32)
    rk = pks.make_reply_key(ct, k_hat, par, _seed=b"t" * 32)
    w8u = w8.astype(np.int8)
    hs = pks.switch_hint(matmul_mod64(w8u, pks.a_mlwe_rows(a_ring)) & M56, par)
    hint = pks.compact_hint(hs, k_hat, par)
    c0r = pks.provider_reply(matmul_mod64(w8u, ct.b) & M56, hs, rk.beta_hat, par)
    assert int(c0r.max()) < 1 << 30
    got = pks.decrypt_pks(c0r, hint, rk.t_reply, rows, par)
    assert np.array_equal(got, ((w8 @ r.T) & ((1 << 28) - 1)).T)
