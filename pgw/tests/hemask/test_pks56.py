"""``pks-v3-K56`` (``pgw.hemask.pks56``, research reference): one product gives ``W8 r mod 2^56`` exactly, the noise
budget is pks-v3's term for term, and the security numbers are the ones the note quotes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pgw.hemask import pks
from pgw.hemask import pks56 as k56
from pgw.hemask.pks import NR, PACK, PKSParams

PAR = k56.K56Params()


def test_parameters_and_noise_match_v3() -> None:
    assert len(set(k56.QP)) == 6 and all(p % (2 * NR) == 1 for p in k56.QP)
    assert all((1 << 23) < p < (1 << 30) for p in k56.QP)  # the provider's fused 32-bit kernels' range
    assert 71.9 < math.log2(PAR.q1) < 72.0 and 83.9 < math.log2(PAR.pmod) < 84.0
    assert math.prod(k56.EP) > 2 * (1 << 80) * 4096  # the pad's centred CRT holds |a t| < 2^92
    v3 = PKSParams()
    worst = 127 * math.sqrt(29568)
    a, b = PAR.noise_terms(worst), v3.noise_terms(worst)
    for key in a:  # every term within 0.05 bit of pks-v3's, relative to its own budget
        ra = math.log2(a[key] / PAR.budget())
        rb = math.log2(b[key] / v3.budget())
        assert abs(ra - rb) < 0.05, key
    assert PAR.noise_bound(worst) < PAR.budget(margin_bits=1.0)  # the factory's 1-bit refusal margin holds
    assert math.log2(PAR.noise_bound(worst)) - math.log2(PAR.budget()) < -1.5


def test_security_numbers() -> None:
    s = k56.security()
    assert s["v3 upload"]["core_svp_classical"] == pytest.approx(238.9, abs=0.2)
    assert s["K56 upload"]["log2_q"] == 80 and s["K56 upload"]["core_svp_classical"] > 128
    assert s["K56 upload"]["core_svp_quantum"] > 128
    assert s["K56 reply key"]["core_svp_classical"] > 128 and s["K56 reply key"]["core_svp_quantum"] > 128
    for name in ("K56 upload", "K56 reply key"):  # inside the HE-standard 128-bit tables
        assert s[name]["log2_q"] <= s[name]["he_std_128_limit"][1]
    assert s["K56 upload at n=8192"]["core_svp_classical"] > s["v3 upload"]["core_svp_classical"]


@pytest.mark.parametrize("rows", [256, 77])
def test_round_trip_exact_mod_2_56(rows: int) -> None:
    m, n = 64, 48
    rng = np.random.default_rng(rows)
    w8 = rng.integers(-128, 128, (m, n), dtype=np.int64)
    w8[0] = 127
    w8[1] = -128
    r = rng.integers(0, 1 << 56, (rows, n), dtype=np.int64)
    r[0] = (1 << 56) - 1
    r[1] = 0
    a_seed = b"a" * 32
    a_ring = k56.public_ring_rows(a_seed, n, "0.o")
    up = k56.encrypt(r, a_ring, PAR, seed=b"e" * 32)
    k_hat = k56.ks_a(b"k" * 32)
    beta, tr = k56.reply_key(up, k_hat, PAR, seed=b"t" * 32)
    w = w8.astype(np.int8)
    h = k56.wr_mod(w, k56._a_rows_cached(a_seed, n, "0.o"), 80)  # the provider's / customer's hint GEMM
    hs = k56.switch_hint(h, PAR)
    c1 = k56.hint(hs, k_hat, PAR)  # the customer's compact hint
    c0r = k56.provider_reply(k56.wr_mod(w, up.b, 80), hs, beta, PAR)
    assert c0r.shape == (m // PACK, NR) and int(np.max(c0r)) < 1 << 60
    got = k56.decrypt(c0r, c1, tr, rows, PAR)
    want = k56.wr_mod(w, np.asarray(r.T, dtype=object), 56).T
    assert got.shape == (rows, m) and np.array_equal(got, want)
    err = k56.decryption_error(c0r, c1, tr, w, r, PAR)
    assert err < PAR.budget()  # measured noise far inside the budget (q1 / 2^57)
    # the provider's view decrypts nothing without t': a wrong reply secret gives garbage
    bad = k56.decrypt(c0r, c1, -tr, rows, PAR)
    assert not np.array_equal(bad, want)
    _ = pks
