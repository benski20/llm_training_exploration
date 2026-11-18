"""MLWE mask encryption: exactness, noise budget, secret structure and tamper behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from pgw.core.errors import ConfigError
from pgw.hemask import DEGREE, HEParams, decrypt_wr, encrypt_masks, matmul_mod64, public_matrix, secret_matrix

SEED = bytes(range(32))


def _ref_wr(w: np.ndarray, r: np.ndarray, k: int) -> np.ndarray:
    """Exact (r @ w.T) mod 2^k with Python integers."""
    return np.array((r.astype(object) @ w.T.astype(object)) % (1 << k), dtype=np.int64)


def _roundtrip(w: np.ndarray, rows: int, k: int, rank: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    _m, n = w.shape
    p = HEParams(k=k, rank=rank)
    rng = np.random.default_rng(seed)
    r = rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64)
    a = public_matrix(SEED, n, p)
    hint = matmul_mod64(w, a)  # gateway-side hint W8 A mod 2^64
    ct = encrypt_masks(r, a, p)
    y = matmul_mod64(w, ct.b)  # what the provider computes
    return decrypt_wr(y, hint, ct, p), _ref_wr(w, r, k)


def test_matmul_mod64_is_exact() -> None:
    rng = np.random.default_rng(1)
    w = rng.integers(-127, 128, size=(5, 300))
    x = rng.integers(0, 2**63, size=(300, 7), dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    got = matmul_mod64(w, x)
    ref = (w.astype(object) @ x.astype(object)) % (1 << 64)
    assert (got.astype(object) == ref).all()
    s = rng.integers(-1, 2, size=(300, 7))
    got2 = matmul_mod64(s.T, x)
    assert (got2.astype(object) == (s.T.astype(object) @ x.astype(object)) % (1 << 64)).all()


def test_secret_matrix_is_negacyclic_product() -> None:
    rng = np.random.default_rng(2)
    s = rng.integers(-1, 2, size=(2, DEGREE))
    a = rng.integers(0, 1000, size=(2, DEGREE))
    got = np.concatenate([a[0], a[1]]) @ secret_matrix(s).astype(np.int64)
    ref = np.zeros(DEGREE, dtype=np.int64)
    for li in range(2):
        full = np.convolve(a[li], s[li])
        ref += full[:DEGREE]
        ref[: DEGREE - 1] -= full[DEGREE:]
    assert (got == ref).all()


@pytest.mark.parametrize("k", [32, 40])
def test_roundtrip_exact_random_weights(k: int) -> None:
    w = np.random.default_rng(3).integers(-127, 128, size=(48, 512))
    got, ref = _roundtrip(w, rows=DEGREE, k=k)
    assert (got == ref).all()


def test_roundtrip_partial_group_and_worst_case_weights() -> None:
    # all-+-127 rows of a 7B down projection width: the largest |W e| the protocol must absorb
    rng = np.random.default_rng(4)
    w = np.where(rng.random((4, 18944)) < 0.5, -127, 127)
    got, ref = _roundtrip(w, rows=37, k=40)
    assert got.shape == (37, 4) and (got == ref).all()


def test_noise_budget_rejects_k48_on_wide_layers() -> None:
    p = HEParams(k=48)
    with pytest.raises(ConfigError):
        p.check_noise(127 * 18944**0.5, "down")
    HEParams(k=40).check_noise(127 * 18944**0.5, "down")


def test_rejects_small_rank_and_bad_k() -> None:
    with pytest.raises(ConfigError):
        HEParams(k=40, rank=4)
    with pytest.raises(ConfigError):
        HEParams(k=36)


def test_tampered_product_gives_wrong_masks() -> None:
    """The provider can corrupt W8 r (detected later by the factory self-check), never forge it silently."""
    w = np.random.default_rng(5).integers(-127, 128, size=(8, 64))
    p = HEParams(k=40, rank=8)
    r = np.random.default_rng(6).integers(0, 1 << 40, size=(16, 64), dtype=np.int64)
    a = public_matrix(SEED, 64, p)
    hint = matmul_mod64(w, a)
    ct = encrypt_masks(r, a, p)
    y = matmul_mod64(w, ct.b)
    y[3, 5] += np.uint64(1 << 30)
    got = decrypt_wr(y, hint, ct, p)
    ref = _ref_wr(w, r, 40)
    assert (got != ref).sum() == 1 and got[5, 3] != ref[5, 3]


def test_ciphertext_hides_masks() -> None:
    """Two very different mask batches give ciphertexts with the same (uniform) statistics."""
    p = HEParams(k=40, rank=8)
    a = public_matrix(SEED, 32, p)
    zeros = encrypt_masks(np.zeros((DEGREE, 32), dtype=np.int64), a, p).b
    ones = encrypt_masks(np.full((DEGREE, 32), (1 << 40) - 1, dtype=np.int64), a, p).b
    for b in (zeros, ones):
        top = (b >> np.uint64(56)).astype(np.int64)  # top byte must look uniform
        hist = np.bincount(top.ravel(), minlength=256) / top.size
        assert abs(hist.mean() - 1 / 256) < 1e-12 and hist.max() < 3 / 256
        assert abs(float((b >> np.uint64(63)).mean()) - 0.5) < 0.02


def test_secret_is_fresh_per_group() -> None:
    p = HEParams(k=40, rank=8)
    a = public_matrix(SEED, 16, p)
    r = np.zeros((4, 16), dtype=np.int64)
    c1, c2 = encrypt_masks(r, a, p), encrypt_masks(r, a, p)
    assert not np.array_equal(c1.secret, c2.secret) and not np.array_equal(c1.b, c2.b)
    c1.wipe()
    assert not c1.secret.any()


def test_public_matrix_depends_on_seed_and_prefix_is_stable() -> None:
    p = HEParams(k=40, rank=8)
    a = public_matrix(SEED, 10, p)
    assert np.array_equal(public_matrix(SEED, 4, p), a[:4])
    assert not np.array_equal(public_matrix(bytes(32), 4, p), a[:4])
