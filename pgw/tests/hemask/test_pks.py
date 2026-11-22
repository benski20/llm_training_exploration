"""Exactness of the pks-v3 research prototype (``pgw.hemask.pks``): ring-LWE upload in today's wire layout,
packed key-switched replies, compact hint, exact ``W8 r mod 2^32``."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pgw.hemask import pks
from pgw.hemask.lwe import HEParams, MaskCiphertext, decrypt_wr, matmul_mod64, secret_matrix

M56 = np.uint64((1 << 56) - 1)
PAR = pks.PKSParams()


def _naive_negacyclic(a: list[int], b: list[int], mod: int) -> list[int]:
    n = len(a)
    out = [0] * n
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                k = i + j
                if k < n:
                    out[k] += ai * bj
                else:
                    out[k - n] -= ai * bj
    return [v % mod for v in out]


def test_primes_and_security_envelope() -> None:
    for p in pks.Q1 + pks.PS + pks.EP:
        assert p % (2 * pks.NR) == 1
    assert 47.5 < np.log2(PAR.q1) < 48.0
    assert PAR.qks_bits <= 109.0  # HomomorphicEncryption.org 128-bit table, N = 4096, ternary secret


def test_ntt_product_matches_schoolbook() -> None:
    rng = np.random.default_rng(1)
    p = pks.Q1[0]
    a = rng.integers(0, p, pks.NR).astype(np.uint64)
    b = rng.integers(0, p, pks.NR).astype(np.uint64)
    nt = pks.Ntt.of(p)
    assert np.array_equal(nt.inv(nt.fwd(a)), a)
    got = nt.inv(nt.fwd(a) * nt.fwd(b) % np.uint64(p))
    # spot-check against a sparse schoolbook product (full schoolbook is 16M Python ops)
    bs = np.zeros(pks.NR, dtype=np.int64)
    idx = rng.choice(pks.NR, 40, replace=False)
    bs[idx] = rng.integers(-1, 2, 40)
    got_s = nt.inv(nt.fwd(a) * nt.fwd(pks._residues(bs, p)) % np.uint64(p))
    ref = _naive_negacyclic([int(v) for v in a], [int(v) for v in bs], p)
    assert [int(v) for v in got_s] == ref
    assert got.shape == (pks.NR,)


def test_exact_mod2_product() -> None:
    rng = np.random.default_rng(2)
    a = rng.integers(0, 1 << 56, (2, pks.NR), dtype=np.uint64)
    t = np.zeros(pks.NR, dtype=np.int64)
    idx = rng.choice(pks.NR, 30, replace=False)
    t[idx] = rng.choice([-1, 1], 30)
    got = pks.negacyclic_mul_mod2(a, t)
    for r in range(2):
        ref = _naive_negacyclic([int(v) for v in a[r]], [int(v) for v in t], 1 << 56)
        assert [int(v) for v in got[r]] == ref
    full = rng.integers(-1, 2, pks.NR)  # dense ternary: the CRT range must hold (|a t| < 2^69)
    worst = np.full((1, pks.NR), (1 << 56) - 1, dtype=np.uint64)
    v = pks.negacyclic_mul_mod2(worst, np.abs(full))
    assert int(v[0, -1]) == ((1 << 56) - 1) * int(np.abs(full).sum()) % (1 << 56)  # top coefficient: sum of t, no wrap


def test_rounding_helpers_exact() -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(0, 1 << 56, 2000, dtype=np.uint64)
    x[:3] = [0, (1 << 56) - 1, 1 << 55]
    for num, sh in ((PAR.q1, 56), (PAR.q1, 36), ((1 << 48) - 1, 48)):
        xs = x if sh == 56 else x >> np.uint64(56 - sh)
        got = pks.round_mul_shift(xs, num, sh)
        ref = [(int(v) * num + (1 << (sh - 1))) >> sh for v in xs]
        assert [int(g) for g in got] == ref
    v = rng.integers(0, PAR.q1, 2000, dtype=np.uint64)
    for bits in (32, 36):
        got = pks.round_pow2_div(v, bits, PAR.q1)
        assert [int(g) for g in got] == [((int(a) << bits) + PAR.q1 // 2) // PAR.q1 for a in v]


def test_noise_budget_real_rows() -> None:
    # Worst case of SPEC 7.9: all-127 rows of the widest 72B-class input (n = 29568) still pass.
    assert PAR.noise_ok(127 * np.sqrt(29568))
    assert PAR.noise_ok(127 * np.sqrt(29568), margin_bits=1.0)
    terms = PAR.noise_terms(0.37 * 127 * np.sqrt(29568))
    assert sum(terms.values()) < PAR.q1 / 2.0 ** 35


def _case(m: int, n: int, rows: int, seed: int, wmax: int = 127):
    rng = np.random.default_rng(seed)
    w = rng.integers(-wmax, wmax + 1, (m, n), dtype=np.int64).astype(np.int8)
    r = rng.integers(0, 1 << 32, (rows, n), dtype=np.int64)
    a_ring = pks.public_ring_rows(bytes([seed]) * 32, n)
    ct = pks.encrypt_rlwe(r, a_ring, PAR)
    y = matmul_mod64(w, ct.b) & M56  # the provider's unchanged kernel
    want = ((w.astype(np.int64) @ r.T) & 0xFFFFFFFF).T  # exact W8 r mod 2^32 (int64 is exact here)
    return w, r, a_ring, ct, y, want


def test_upload_is_old_format_mlwe() -> None:
    """The ring-LWE upload decrypts with today's hint path (provider and old customer code unchanged)."""
    w, r, a_ring, ct, y, want = _case(32, 64, 256, 4)
    a_m = pks.a_mlwe_rows(a_ring)
    # bodies equal A_mlwe S + e + Delta r for the module secret made of t's 16 components
    smat = secret_matrix(ct.t_components)
    pad = matmul_mod64(smat, a_m, small_left=False) & M56
    diff = (ct.b - pad - (r.T.astype(np.uint64) << np.uint64(24))) & M56
    centred = np.where(diff >= np.uint64(1 << 55), diff.astype(np.int64) - (1 << 56), diff.astype(np.int64))
    assert np.abs(centred).max() <= 21  # exactly the CBD(21) noise
    h = matmul_mod64(w, a_m) & M56
    got = decrypt_wr(y, h, MaskCiphertext(b=ct.b, secret=smat, rows=ct.rows), HEParams(k=32))
    assert np.array_equal(got, want)


@pytest.mark.parametrize("rows", [256, 77])
def test_pks_round_trip_exact(rows: int) -> None:
    w, _r, a_ring, ct, y, want = _case(48, 128, rows, 5)
    h = matmul_mod64(w, pks.a_mlwe_rows(a_ring)) & M56
    ks_seed = os.urandom(32)
    hint = pks.compact_hint(h, ks_seed, PAR)
    assert hint.shape == (2, 48 // 16, pks.NR)
    key = pks.make_reply_key(ct, ks_seed, PAR)
    reply = pks.provider_reply(y, pks._atilde_hat(h, PAR), key, PAR)
    assert reply.shape == (3, pks.NR) and int(reply.max()) < (1 << PAR.yb)
    got = pks.decrypt_pks(reply, hint, key, ct.rows, PAR)
    assert np.array_equal(got, want)
    assert not key.t_reply.any()  # wiped


def test_pks_tampered_hint_or_reply_changes_masks() -> None:
    w, _r, a_ring, ct, y, want = _case(16, 64, 256, 6)
    h = matmul_mod64(w, pks.a_mlwe_rows(a_ring)) & M56
    ks_seed = os.urandom(32)
    hint = pks.compact_hint(h, ks_seed, PAR)
    key = pks.make_reply_key(ct, ks_seed, PAR)
    reply = pks.provider_reply(y, pks._atilde_hat(h, PAR), key, PAR)
    bad = reply.copy()
    bad[0, 5] = (bad[0, 5] + np.uint64(1 << 20)) & np.uint64((1 << PAR.yb) - 1)
    got = pks.decrypt_pks(bad, hint, pks.ReplyKey(key.beta_hat, key.t_reply.copy()), ct.rows, PAR)
    assert not np.array_equal(got, want)  # caught downstream by the every-row Freivalds self-check


@pytest.mark.skipif(not os.environ.get("PGW_PKS_REAL_SHAPES"), reason="real 72B shapes: set PGW_PKS_REAL_SHAPES=1")
@pytest.mark.parametrize("n", [8192, 29568])
def test_pks_real_72b_input_width(n: int) -> None:
    """Exact at Qwen2.5-72B input widths (hidden 8192, down 29568) for two 16-row blocks, all-127 rows."""
    w, r, a_ring, ct, y, want = _case(32, n, 256, 7, wmax=127)
    w[:] = 127  # worst-case row norm
    y = matmul_mod64(w, ct.b) & M56
    want = ((w.astype(np.int64) @ r.T) & 0xFFFFFFFF).T
    h = matmul_mod64(w, pks.a_mlwe_rows(a_ring)) & M56
    ks_seed = os.urandom(32)
    hint = pks.compact_hint(h, ks_seed, PAR)
    key = pks.make_reply_key(ct, ks_seed, PAR)
    got = pks.decrypt_pks(pks.provider_reply(y, pks._atilde_hat(h, PAR), key, PAR), hint, key, 256, PAR)
    assert np.array_equal(got, want)
