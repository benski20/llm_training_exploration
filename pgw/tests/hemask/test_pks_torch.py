"""The customer's pks-v3 kernels in torch (``pgw.hemask.pks_torch``, client GPU prototype) equal the numpy reference
(``pgw.hemask.pks``) bit for bit on CPU torch; on CUDA with ``PGW_TEST_CUDA=1`` (the Kaggle job) as well."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from pgw.core.prng import SeedExpander
from pgw.hemask import pks
from pgw.hemask import pks_torch as pt
from pgw.hemask.lwe import _cbd, _ternary, matmul_mod64
from pgw.hemask.pks import EP, NR, NU, PACK, QP, RANK, D, PKSParams

PAR = PKSParams()
M56 = np.uint64((1 << 56) - 1)
DEVS = [torch.device("cpu")]
if os.environ.get("PGW_TEST_CUDA") == "1" and torch.cuda.is_available():  # pragma: no cover - GPU only
    DEVS.append(torch.device("cuda"))


def _u(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy().view(np.uint64)


@pytest.mark.parametrize("dev", DEVS, ids=str)
def test_ntt_equals_reference(dev: torch.device) -> None:
    rng = np.random.default_rng(1)
    for p, n in [(EP[0], NU), (EP[2], NU), (QP[0], NR), (QP[3], NR)]:
        x = rng.integers(0, p, (3, n), dtype=np.uint64)
        x[0, :4] = [0, 1, p - 1, p - 2]
        nt, ref = pt.TorchNtt(p, n, dev), pks.Ntt.of(p, n)
        xt = torch.from_numpy(x.view(np.int64)).to(dev)
        assert np.array_equal(_u(nt.fwd(xt)), ref.fwd(x))
        assert np.array_equal(_u(nt.inv(xt)), ref.inv(x))
        assert np.array_equal(_u(xt), x)  # input untouched


@pytest.mark.parametrize("dev", DEVS, ids=str)
def test_pad_equals_reference_at_range_extremes(dev: torch.device) -> None:
    """``a`` at ``2^56 - 1`` / ``0`` and ``t`` all ``+1`` / all ``-1`` (the CRT's largest positive and negative
    values) and random rows."""
    rng = np.random.default_rng(2)
    a = rng.integers(0, 1 << 56, (5, NU), dtype=np.uint64)
    a[0] = (1 << 56) - 1
    a[1] = 0
    a[2, ::2] = (1 << 56) - 1
    for t in (np.ones(NU, dtype=np.int64), -np.ones(NU, dtype=np.int64), rng.integers(-1, 2, NU)):
        want = pks.negacyclic_mul_mod2(a, t)
        assert np.array_equal(_u(pt.pad(a, t, dev)), want)
        assert np.array_equal(_u(pt.pad(a, t, dev, pt.pad_ahat(a, dev))), want)  # cached transforms of a


@pytest.mark.parametrize("dev", DEVS, ids=str)
def test_encrypt_body_and_reply_key_equal_reference(dev: torch.device) -> None:
    n, rows = 64, 200
    rng = np.random.default_rng(3)
    r = rng.integers(0, 1 << 32, (rows, n), dtype=np.int64)
    a_ring = pks.public_ring_rows(b"s" * 32, n, "0.o")
    seed = bytes(range(32))
    ct = pks.encrypt_rlwe(r, a_ring, PAR, _seed=seed)
    sx = SeedExpander(seed)  # the reference's own streams, replayed
    t = _ternary(sx, (NU,), "pks-t").astype(np.int64)
    e = _cbd(sx, (n, D), PAR.eta, "pks-e").astype(np.int64)
    assert np.array_equal(t, ct.t)
    assert np.array_equal(_u(pt.encrypt_body(r, a_ring, t, e, dev)), ct.b)
    k_hat = pks.ks_a(b"k" * 32)
    kseed = b"q" * 32
    rk = pks.make_reply_key(ct, k_hat, PAR, _seed=kseed)
    sk = SeedExpander(kseed)
    tr = _ternary(sk, (NR,), "pks-t-reply").astype(np.int64)
    ek = _cbd(sk, (RANK, NR), PAR.eta, "pks-e-ksk").astype(np.int64)
    assert np.array_equal(_u(pt.reply_key(ct.t_components, k_hat, tr, ek, dev)), rk.beta_hat)


@pytest.mark.parametrize("dev", DEVS, ids=str)
def test_decrypt_round_trip_equals_reference(dev: torch.device) -> None:
    """Full reference round trip (customer upload, provider reply, compact hint); the torch decryption returns the
    reference's masks, which are ``W8 r mod 2^32``."""
    m, n, rows = 64, 48, 256
    rng = np.random.default_rng(4)
    w8 = rng.integers(-128, 128, (m, n), dtype=np.int64)
    w8[0] = 127
    r = rng.integers(0, 1 << 32, (rows, n), dtype=np.int64)
    a_ring = pks.public_ring_rows(b"a" * 32, n, "0.o")
    ct = pks.encrypt_rlwe(r, a_ring, PAR, _seed=b"e" * 32)
    k_hat = pks.ks_a(b"k" * 32)
    rk = pks.make_reply_key(ct, k_hat, PAR, _seed=b"t" * 32)
    w8u = w8.astype(np.int8)
    hs = pks.switch_hint(matmul_mod64(w8u, pks.a_mlwe_rows(a_ring)) & M56, PAR)
    hint = pks.compact_hint(hs, k_hat, PAR)
    c0r = pks.provider_reply(matmul_mod64(w8u, ct.b) & M56, hs, rk.beta_hat, PAR)
    want = pks.decrypt_pks(c0r, hint, rk.t_reply, rows, PAR)
    assert np.array_equal(want, ((w8 @ r.T) & 0xFFFFFFFF).T)
    got = pt.decrypt(c0r, pt.hint_residues_t(hint, dev), rk.t_reply, rows, dev)
    assert got.shape == (rows, m)
    assert np.array_equal(got.cpu().numpy(), want)
    assert m % PACK == 0
