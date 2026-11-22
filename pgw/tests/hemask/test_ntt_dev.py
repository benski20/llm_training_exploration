"""Device NTT (pgw.hemask.ntt_dev): bit-identical to the CPU products, run on CPU tensors."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pgw.core.errors import ConfigError
from pgw.hemask import ntt, ntt_dev
from pgw.hemask.fastmm import to_planes
from pgw.hemask.lwe import DEGREE, HEParams, encrypt_masks, matmul_mod64, public_matrix, secret_matrix
from pgw.hemask.mlwe_ring import PretransformedEncryptor


@pytest.fixture(scope="module")
def dn() -> ntt_dev.DeviceNTT:
    return ntt_dev.DeviceNTT("cpu", use_triton=False)


def _secs(rng: np.random.Generator, g: int, rank: int) -> np.ndarray:
    s = rng.integers(-1, 2, size=(g, rank, DEGREE)).astype(np.int8)
    s[0] = -1  # largest L1 norm
    return s


def _smat(secs: np.ndarray) -> np.ndarray:
    return np.concatenate([secret_matrix(s) for s in secs], axis=1)


def test_primes_and_bound() -> None:
    for p in ntt_dev.PRIMES:
        assert (1 << 28) < p < (1 << 29) and p % 512 == 1 and ntt_dev._is_prime(p)
    assert ntt_dev.exact(64, 256 * 32) and ntt_dev.exact(56, 256 * 16)
    assert not ntt_dev.exact(64, 1 << 24)


def test_negacyclic_product(dn: ntt_dev.DeviceNTT) -> None:
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.integers(0, ntt_dev.PRIMES[2], size=(4, 3, DEGREE)))
    assert torch.equal(dn.inverse(dn.forward(x)), x)
    a = rng.integers(0, 1000, size=DEGREE)
    b = rng.integers(0, 1000, size=DEGREE)
    want = np.zeros(DEGREE, dtype=object)
    for i in range(DEGREE):
        for j in range(DEGREE):
            k = i + j
            want[k % DEGREE] += int(a[i]) * int(b[j]) * (-1 if k >= DEGREE else 1)
    ta = dn.forward(torch.from_numpy(np.stack([a] * 3))[None])
    tb = dn.forward(torch.from_numpy(np.stack([b] * 3))[None])
    got = dn.inverse(dn.mulmod(ta, tb))[0]
    for i, p in enumerate(ntt_dev.PRIMES):
        assert [int(v) for v in got[i]] == [int(v) % p for v in want]


@pytest.mark.parametrize("q_bits,rank,rows,g", [(56, 16, 37, 3), (64, 16, 5, 1), (56, 12, 20, 2), (64, 8, 260, 2)])
def test_transformed_a_matches_dense(dn: ntt_dev.DeviceNTT, q_bits: int, rank: int, rows: int, g: int) -> None:
    rng = np.random.default_rng(rows)
    a = rng.integers(0, 1 << 64, size=(rows + 3, rank * DEGREE), dtype=np.uint64)
    a[0] = np.uint64((1 << 64) - 1)
    secs = _secs(rng, g, rank)
    keep = secs.copy()
    got = ntt_dev.DeviceTransformedA(a, q_bits, dn).mm_secrets(secs, rows=rows)
    assert np.array_equal(secs, keep)  # the caller's secrets are not wiped
    aq = a[:rows] & np.uint64((1 << q_bits) - 1) if q_bits < 64 else a[:rows]
    want = matmul_mod64(_smat(secs), aq, small_left=False, fast=False)
    assert got.dtype == np.uint64 and np.array_equal(got, want)
    if ntt.available() and q_bits == 64:
        native = ntt.TransformedOperand.from_u64(a).mm_secrets(secs, rows=rows)
        assert np.array_equal(got, native)


@pytest.mark.parametrize("npl", [6, 7])
def test_hint_mm_matches_cpu(dn: ntt_dev.DeviceNTT, npl: int) -> None:
    rng = np.random.default_rng(npl)
    rank, m = 8, 17
    h = rng.integers(0, 1 << 64, size=(m, rank * DEGREE), dtype=np.uint64)
    planes = to_planes(h)[:, 1:1 + npl]
    smat = _smat(_secs(rng, 3, rank))
    got = dn.hint_mm(planes, smat, first_byte=1)
    want = ntt.big_planes_mm_small(planes, smat, first_byte=1)
    assert np.array_equal(got, want)
    assert np.array_equal(got, dn.hint_mm(torch.from_numpy(planes), smat, first_byte=1))


def test_secrets_from_matrix_checks_structure() -> None:
    rng = np.random.default_rng(1)
    secs = _secs(rng, 2, 4)
    smat = _smat(secs)
    assert np.array_equal(ntt_dev.secrets_from_matrix(smat), secs)
    bad = smat.copy()
    bad[5, 3] = np.int8(1 - int(bad[5, 3]))
    with pytest.raises(ConfigError):
        ntt_dev.secrets_from_matrix(bad)
    with pytest.raises(ConfigError):
        ntt_dev.secrets_from_matrix(smat[:100])


@pytest.mark.parametrize("k,rank", [(32, 16), (40, 16), (32, 12)])
def test_encryptor_bit_identical(dn: ntt_dev.DeviceNTT, k: int, rank: int) -> None:
    """PretransformedEncryptor on the device NTT == lwe.encrypt_masks for the same seeds."""
    params = HEParams(k=k, rank=rank)
    n, rows = 29, 600
    a = public_matrix(b"\x05" * 32, 40, params)
    rng = np.random.default_rng(k + rank)
    r = rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64)
    enc = PretransformedEncryptor(a, params, groups_per_pass=2, device_ntt=dn)
    seeds = [bytes([9 + i]) * 32 for i in range(3)]
    cts = enc.encrypt(r, _seeds=seeds)
    for gi, ct in enumerate(cts):
        ref = encrypt_masks(r[gi * DEGREE:(gi + 1) * DEGREE], a[:n], params, _seed=seeds[gi])
        assert ct.rows == ref.rows and np.array_equal(ct.b, ref.b) and np.array_equal(ct.secret, ref.secret)


def test_self_test_and_device_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    params = HEParams(k=32, rank=8)
    dn = ntt_dev.DeviceNTT("cpu", use_triton=False)
    assert ntt_dev.self_test(dn, params, rows=4, groups=1)
    assert ntt_dev.device_ntt_for(params, None) is None
    if not torch.cuda.is_available():
        assert ntt_dev.device_ntt_for(params, "auto") is None
    monkeypatch.setattr(ntt_dev, "_TESTED", {})
    monkeypatch.setattr(ntt_dev, "self_test", lambda *a, **k: False)
    assert ntt_dev.device_ntt_for(params, "cpu") is None  # failed self-test -> CPU path
    assert ntt_dev.status()["cpu"]["ok"] is False


@pytest.mark.parametrize("q_bits", [56, 64])
def test_backend_ntt_dev_exact(q_bits: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """HEMaskBackend(encrypt='ntt_dev', device='cpu'): exact W8 r end to end, device NTT for both the
    encryption pad and decryption's hint product; a failed self-test falls back to the CPU path."""
    from pgw.hemask.backend import HEMaskBackend

    from .test_native_sampling import _Hints, _Pack, _Provider, _ref_wr, _ring_for

    rng = np.random.default_rng(q_bits + 1)
    weights = {"a": rng.integers(-127, 128, size=(40, 96)).astype(np.int8),
               "b": np.where(rng.random((24, 160)) < 0.5, -127, 127).astype(np.int8)}
    ring = _ring_for(q_bits)
    provider = _Provider(weights, q_bits)
    be = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, groups_per_call=2,
                       max_inflight=2, groups_per_pass=3, encrypt="ntt_dev", device="cpu")
    assert be.encrypt_mode == "ntt_dev" and be._ntt_dev is not None and be.engine == "ntt_dev:cpu"
    calls = []
    orig = be._ntt_dev.hint_mm
    monkeypatch.setattr(be._ntt_dev, "hint_mm", lambda *a, **k: calls.append(1) or orig(*a, **k))
    for key, rows in (("a", 3 * DEGREE + 3), ("b", 77)):
        w = weights[key]
        r = rng.integers(0, 1 << ring.k, size=(rows, w.shape[1]), dtype=np.int64)
        got = be.matmul(key, torch.from_numpy(r))
        assert np.array_equal(got.numpy(), _ref_wr(w.astype(np.int64), r, ring.k))
    assert calls
    # a failing device hint product (e.g. out of device memory) falls back to the CPU product, same bits

    def _boom(*a: object, **k: object) -> None:
        raise RuntimeError("device out of memory")

    monkeypatch.setattr(be._ntt_dev, "hint_mm", _boom)
    r = rng.integers(0, 1 << ring.k, size=(77, 160), dtype=np.int64)
    got = be.matmul("b", torch.from_numpy(r))
    assert np.array_equal(got.numpy(), _ref_wr(weights["b"].astype(np.int64), r, ring.k))
    monkeypatch.setattr(ntt_dev, "_TESTED", {})
    monkeypatch.setattr(ntt_dev, "self_test", lambda *a, **k: False)
    fb = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, encrypt="ntt_dev", device="cpu")
    assert fb._ntt_dev is None and fb.encrypt_mode == ("ntt_pre" if ntt.available() else "dense")


def test_factory_setting_accepts_ntt_dev(tmp_path: Path) -> None:
    from pgw.factory.config import FactorySettings

    assert FactorySettings(pack_path=tmp_path).he_encrypt == "dense"  # off by default
    assert FactorySettings(pack_path=tmp_path, he_encrypt="ntt_dev").he_encrypt == "ntt_dev"


_TRITON_SCRIPT = r"""
import numpy as np, torch
from pgw.hemask import ntt_dev
from pgw.hemask.lwe import DEGREE, HEParams
ref = ntt_dev.DeviceNTT("cpu", use_triton=False)
tri = ntt_dev.DeviceNTT("cpu", use_triton=True)
assert tri.triton is not None
rng = np.random.default_rng(3)
for rank, rows, g in ((2, 5, 2), (17, 3, 1)):
    a = torch.from_numpy(rng.integers(0, 1 << 64, size=(rows, rank * 256), dtype=np.uint64).view(np.int64))
    a_hat = ref.forward(ref.residues_u64(a).view(3, -1, rank, DEGREE).permute(1, 2, 0, 3)).contiguous()
    s_hat = ref.secrets_hat(rng.integers(-1, 2, size=(g, rank, DEGREE)).astype(np.int8))
    assert torch.equal(ref.mac_inverse_crt(a_hat, s_hat), tri.mac_inverse_crt(a_hat.to(torch.int32), s_hat))
assert ntt_dev.self_test(tri, HEParams(k=32, rank=8), rows=3, groups=1)
print("ok")
"""


def test_triton_kernel_in_interpreter() -> None:
    """The fused Triton kernel (GPU path) gives the torch reference's bits, run in Triton's CPU interpreter."""
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", PGW_NTT_DEV_TRITON="1")
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run([sys.executable, "-c", _TRITON_SCRIPT], env=env,  # noqa: S603 - fixed script
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-3000:]
