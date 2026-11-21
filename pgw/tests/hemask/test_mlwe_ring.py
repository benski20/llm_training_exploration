"""Exact fast encryption (public matrix in NTT form) and the HE parameter profiles."""

from __future__ import annotations

import numpy as np
import pytest

from pgw.core.errors import ConfigError
from pgw.hemask import ntt
from pgw.hemask.fastmm import to_planes
from pgw.hemask.lwe import DEGREE, HEParams, a_plane_count, decrypt_wr, encrypt_masks, matmul_mod64, public_matrix
from pgw.hemask.mlwe_ring import (
    R12_ACK,
    PretransformedEncryptor,
    check_rank,
    core_svp_beta,
    encrypt_cost_ops,
)

needs_ntt = pytest.mark.skipif(not ntt.available(), reason="pgw_ntt not built")


def _seeds(k: int, base: int = 0) -> list[bytes]:
    return [bytes([base + i]) * 32 for i in range(k)]


@needs_ntt
@pytest.mark.parametrize("k,rank", [(32, 16), (32, 12), (40, 16)])
@pytest.mark.parametrize("rows", [1, 255, 256, 700])
def test_bit_identical_to_encrypt_masks(k: int, rank: int, rows: int) -> None:
    """Same seed per group -> exactly the ciphertext (and secret) of lwe.encrypt_masks, both via the
    plane path (production) and the uint64 path."""
    params = HEParams(k=k, rank=rank)
    n = 37
    a = public_matrix(b"\x07" * 32, 50, params)
    rng = np.random.default_rng(rows + k)
    r = rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64)
    enc = PretransformedEncryptor(a, params, groups_per_pass=2)
    groups = -(-rows // DEGREE)
    seeds = _seeds(groups, 3)
    cts = enc.encrypt(r, _seeds=seeds)
    assert len(cts) == groups
    planes = to_planes(a[:n])[:, :a_plane_count(params)]
    for gi, ct in enumerate(cts):
        rg = r[gi * DEGREE:(gi + 1) * DEGREE]
        ref = encrypt_masks(rg, None, params, a_planes=planes, _seed=seeds[gi])
        ref64 = encrypt_masks(rg, a[:n], params, _seed=seeds[gi])
        assert ct.rows == rg.shape[0]
        assert ct.b.dtype == np.uint64 and ct.b.shape == (n, DEGREE)
        assert np.array_equal(ct.b, ref.b)
        assert np.array_equal(ct.b, ref64.b)
        assert np.array_equal(ct.secret, ref.secret)


@needs_ntt
@pytest.mark.parametrize("rank", [16, 12])
def test_round_trip_exact(rank: int) -> None:
    """encrypt -> provider W8 b mod 2^q -> decrypt with the hint W8 A == W8 r mod 2^K, for full-range
    int8 weights (worst-case row norms) and full-range masks."""
    params = HEParams(k=32, rank=rank)
    n, m, rows = 64, 48, 300
    rng = np.random.default_rng(rank)
    w = rng.integers(-127, 128, size=(m, n), dtype=np.int64)
    w[0] = 127  # largest row norm
    params.check_noise(float(np.sqrt((w.astype(np.float64) ** 2).sum(1)).max()))
    a = public_matrix(b"\x11" * 32, n, params)
    hint = matmul_mod64(w, a)  # gateway-side one-time hint H = W8 A mod 2^64
    r = rng.integers(0, 1 << 32, size=(rows, n), dtype=np.int64)
    enc = PretransformedEncryptor(a, params)
    got = []
    for ct in enc.encrypt(r):
        y = matmul_mod64(w, ct.b) & params.q_mask  # the provider's work: linear in the ciphertext
        got.append(decrypt_wr(y, hint, ct, params))
        assert not ct.secret.any()  # decrypt wipes the secret
    want = (r.astype(object) @ w.T.astype(object)) % (1 << 32)
    assert np.array_equal(np.concatenate(got).astype(object), want)


@needs_ntt
def test_fresh_randomness_per_group() -> None:
    params = HEParams(k=32, rank=16)
    a = public_matrix(b"\x01" * 32, 8, params)
    enc = PretransformedEncryptor(a, params)
    r = np.zeros((512, 8), dtype=np.int64)
    c1, c2 = enc.encrypt(r)
    c3, = enc.encrypt(r[:256])
    assert not np.array_equal(c1.b, c2.b) and not np.array_equal(c1.b, c3.b)
    assert not np.array_equal(c1.secret, c2.secret)


@needs_ntt
def test_shape_checks() -> None:
    params = HEParams(k=32, rank=16)
    a = public_matrix(b"\x01" * 32, 8, params)
    enc = PretransformedEncryptor(a, params)
    with pytest.raises(ConfigError):
        enc.encrypt(np.zeros((4, 9), dtype=np.int64))
    with pytest.raises(ConfigError):
        enc.encrypt(np.zeros((300, 8), dtype=np.int64), _seeds=_seeds(1))
    with pytest.raises(ConfigError):
        PretransformedEncryptor(a[:, :256], params)


def test_rank_gate() -> None:
    assert check_rank(16).name == "mlwe-r16"
    assert check_rank(24).reviewed
    with pytest.raises(ConfigError, match="he_params_ack"):
        check_rank(12)
    with pytest.raises(ConfigError, match="he_params_ack"):
        check_rank(12, "yes")
    prof = check_rank(12, R12_ACK)
    assert prof.dim == 3072 and not prof.reviewed
    for rank in (8, 10, 11, 13, 15):
        with pytest.raises(ConfigError, match="refused"):
            check_rank(rank, R12_ACK)


def test_factory_settings_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pydantic import ValidationError

    from pgw.factory.config import FactorySettings

    fs = FactorySettings(pack_path=tmp_path)
    assert fs.he_rank == 16 and fs.he_encrypt == "dense" and fs.he_params_ack == ""  # defaults unchanged
    assert FactorySettings(pack_path=tmp_path, he_encrypt="ntt_pre").he_encrypt == "ntt_pre"
    with pytest.raises(ValidationError, match="he_params_ack"):
        FactorySettings(pack_path=tmp_path, he_rank=12)
    assert FactorySettings(pack_path=tmp_path, he_rank=12, he_params_ack=R12_ACK).he_rank == 12
    with pytest.raises(ValidationError, match="refused"):
        FactorySettings(pack_path=tmp_path, he_rank=8, he_params_ack=R12_ACK)
    with pytest.raises(ValidationError):
        FactorySettings(pack_path=tmp_path, he_encrypt="fast")


def test_security_estimates_ordering() -> None:
    """The memo's numbers: core-SVP block sizes at q = 2^56, CBD(21) noise, ternary secret."""
    sig = float(np.sqrt(21 / 2))
    b16 = core_svp_beta(4096, 56, sig)
    b12 = core_svp_beta(3072, 56, sig)
    b8 = core_svp_beta(2048, 56, sig)
    assert 0.292 * b16 > 230 and 0.292 * b12 > 160 and 0.292 * b8 < 100
    # calibration point: the HE-standard 128-bit entry (n=2048, log q=54, sigma 3.2) sits at ~98 core-SVP
    assert 95 < 0.292 * core_svp_beta(2048, 54, 3.2) < 101


def test_cost_model() -> None:
    c = encrypt_cost_ops(896, 16)
    assert c["ratio"] > 300


@needs_ntt
@pytest.mark.parametrize("q_bits", [56, 64])
def test_backend_ntt_pre_exact_and_same_ciphertexts(q_bits: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """HEMaskBackend(encrypt='ntt_pre'): exact W8 r through the whole pipeline, and per-group
    ciphertexts equal to the dense backend's for the same seeds."""
    import torch

    import pgw.hemask.lwe as lwe
    import pgw.hemask.mlwe_ring as mr
    from pgw.hemask.backend import HEMaskBackend

    from .test_native_sampling import _Hints, _Pack, _Provider, _ref_wr, _ring_for

    rng = np.random.default_rng(q_bits)
    weights = {"a": rng.integers(-127, 128, size=(40, 96)).astype(np.int8),
               "b": np.where(rng.random((24, 160)) < 0.5, -127, 127).astype(np.int8)}
    ring = _ring_for(q_bits)
    provider = _Provider(weights, q_bits)
    be = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, groups_per_call=2,
                       max_inflight=2, groups_per_pass=3, encrypt="ntt_pre")
    assert be.encrypt_mode == "ntt_pre" and be._a.size == 0
    for key, rows in (("a", 5 * DEGREE + 3), ("b", 77)):
        w = weights[key]
        r = rng.integers(0, 1 << ring.k, size=(rows, w.shape[1]), dtype=np.int64)
        got = be.matmul(key, torch.from_numpy(r))
        assert np.array_equal(got.numpy(), _ref_wr(w.astype(np.int64), r, ring.k))
    # same seeds -> same ciphertexts as the dense backend (host path: the device encryptor draws its seeds through
    # its own import of new_seed, which the patch below would not reach on a CUDA host)
    dense = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, device=None)
    r = rng.integers(0, 1 << ring.k, size=(300, 96), dtype=np.int64)
    seeds = _seeds(2, 40)
    it = iter(seeds)
    monkeypatch.setattr(mr, "new_seed", lambda: next(it))
    fast = be._encrypt(r, 96)
    it2 = iter(seeds)
    monkeypatch.setattr(lwe, "new_seed", lambda: next(it2))
    ref = dense._encrypt(r, 96)
    assert [c.rows for c in fast] == [c.rows for c in ref] == [256, 44]
    for x, y in zip(fast, ref, strict=True):
        assert np.array_equal(x.b, y.b) and np.array_equal(x.secret, y.secret)
    with pytest.raises(ConfigError):
        HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, encrypt="bogus")


@needs_ntt
@pytest.mark.parametrize("q_bits", [56, 64])
def test_backend_ntt_pre_with_device_decryption_is_exact(q_bits: int) -> None:
    """``encrypt='ntt_pre'`` with a device: CPU NTT encryption, decryption as fused device passes (here the
    device code on CPU tensors). Exact ``W8 r``; the device holds no public matrix (ntt_pre keeps it)."""
    import torch

    from pgw.hemask.backend import HEMaskBackend

    from .test_native_sampling import _Hints, _Pack, _Provider, _ref_wr, _ring_for

    rng = np.random.default_rng(100 + q_bits)
    weights = {"a": rng.integers(-127, 128, size=(40, 96)).astype(np.int8),
               "b": np.where(rng.random((24, 160)) < 0.5, -127, 127).astype(np.int8)}
    ring = _ring_for(q_bits)
    be = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), _Provider(weights, q_bits), groups_per_call=2,
                       max_inflight=2, groups_per_pass=3, encrypt="ntt_pre", device="cpu")
    assert be._fast is not None and be._dev is not None and be.engine == "device:cpu"
    assert be._dev.a.numel() == 0
    for key, rows in (("a", 5 * DEGREE + 3), ("b", 77)):
        w = weights[key]
        r = rng.integers(0, 1 << ring.k, size=(rows, w.shape[1]), dtype=np.int64)
        got = be.matmul(key, torch.from_numpy(r))
        assert np.array_equal(got.numpy(), _ref_wr(w.astype(np.int64), r, ring.k))
    # without a device the decryption stays on the CPU engines (the behaviour before this option)
    cpu = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), _Provider(weights, q_bits), encrypt="ntt_pre",
                        device=None)
    assert cpu._dev is None
