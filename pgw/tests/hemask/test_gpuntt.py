"""GPU NTT engine of the encrypted factory (pgw.hemask.gpuntt): bit-identical to the dense references.

The eager torch implementation runs here on CPU tensors; the Triton kernels run under the Triton
interpreter (CPU) in a subprocess, and on CUDA when present.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from pgw.hemask import gpuntt as G
from pgw.hemask.devmm import _reference_encrypt
from pgw.hemask.fastmm import to_planes
from pgw.hemask.lwe import DEGREE, HEParams, matmul_mod64, public_matrix, sample_secret

CPU = torch.device("cpu")


def _naive_negacyclic(a: np.ndarray, s: np.ndarray, p: int) -> np.ndarray:
    out = [0] * DEGREE
    for i in range(DEGREE):
        for j in range(DEGREE):
            k = i + j
            v = int(a[i]) * int(s[j])
            if k >= DEGREE:
                out[k - DEGREE] -= v
            else:
                out[k] += v
    return np.array([x % p for x in out], dtype=np.int64)


def test_primes_and_bound():
    for p in G.PRIMES:
        assert p < 1 << 29 and (p - 1) % 512 == 0
        assert all(p % d for d in range(2, 2000))
    assert G.HALF > (1 << 63) * 256 * 32  # 64-bit operands, rank 32 ternary secrets


def test_ntt_roundtrip_and_negacyclic_product():
    tb = G._tables(CPU)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1 << 20, size=DEGREE)
    s = rng.integers(-1, 2, size=DEGREE)
    x = torch.from_numpy(np.stack([a] * 3).reshape(3, DEGREE, 1) % np.array(G.PRIMES).reshape(3, 1, 1))
    y = torch.from_numpy(np.stack([s] * 3).reshape(3, DEGREE, 1) % np.array(G.PRIMES).reshape(3, 1, 1))
    x0 = x.clone()
    G.fwd_ntt(x, tb)
    G.fwd_ntt(y, tb)
    prod = (x * y) % tb.p.view(3, 1, 1)
    back = G.inv_ntt(x.clone(), tb)
    back = (back * tb.ninv.view(3, 1, 1)) % tb.p.view(3, 1, 1)
    assert torch.equal(back, x0)
    G.inv_ntt(prod, tb)
    prod = (prod * tb.ninv.view(3, 1, 1)) % tb.p.view(3, 1, 1)
    for i, p in enumerate(G.PRIMES):
        assert np.array_equal(prod[i, :, 0].numpy(), _naive_negacyclic(a, s, p))


@pytest.mark.parametrize("groups", [1, 3])
def test_mac_equals_dense_mod64(groups):
    """Full 64-bit operands (signed view) and an offset row window."""
    tb = G._tables(CPU)
    rng = np.random.default_rng(groups)
    a = rng.integers(-(1 << 63), (1 << 63) - 1, size=(30, 16 * DEGREE), dtype=np.int64)
    sm = [sample_secret(bytes([i + 1]) * 32, 16) for i in range(groups)]
    ta = G.transform_rows(torch.from_numpy(a), 16, tb)
    sh = G.transform_secrets(torch.from_numpy(G._secret_polys(sm, 16)), tb)
    z = G.mac_inv_crt_eager(ta, 4, 20, sh, tb).numpy().view(np.uint64)
    ref = matmul_mod64(np.concatenate(sm, axis=1), a[4:24].view(np.uint64), small_left=False, fast=False)
    assert np.array_equal(z, ref)


def test_hint_planes_transform():
    tb = G._tables(CPU)
    rng = np.random.default_rng(5)
    h = rng.integers(0, 1 << 63, size=(9, 16 * DEGREE), dtype=np.int64).view(np.uint64)
    pl = to_planes(h)[:, 1:7]
    sm = [sample_secret(bytes(32), 16)]
    th = G.fwd_planes_eager(torch.from_numpy(np.ascontiguousarray(pl)), 1, 16, tb)
    sh = G.transform_secrets(torch.from_numpy(G._secret_polys(sm, 16)), tb)
    z = G.mac_inv_crt_eager(th, 0, 9, sh, tb).numpy().view(np.uint64)
    hm = h & np.uint64(((1 << 56) - 1) ^ 255)  # bytes 1..6
    assert np.array_equal(z, matmul_mod64(sm[0], hm, small_left=False, fast=False))


def test_engine_encrypt_decrypt_bit_identical():
    p = HEParams(k=32, rank=16)
    rng = np.random.default_rng(9)
    n, m, rows = 40, 21, DEGREE + 5
    a = public_matrix(bytes(range(32)), 64, p)
    eng = G.GpuNTTHE(a, p, CPU, hint_budget=1 << 20)
    r = rng.integers(0, 1 << 32, size=(rows, n), dtype=np.int64)
    r_in = r.copy()
    seeds = [bytes([3 + i]) * 32 for i in range(2)]
    cts = eng.encrypt(r_in, _seeds=seeds)
    assert np.array_equal(r_in, r)
    assert [ct.rows for ct in cts] == [DEGREE, 5]
    for gi, (ct, seed) in enumerate(zip(cts, seeds, strict=True)):
        assert np.array_equal(ct.b, _reference_encrypt(r[gi * DEGREE:(gi + 1) * DEGREE], a[:n], p, seed))
    b = np.concatenate([ct.b for ct in cts], axis=1)
    w = rng.integers(-127, 128, size=(m, n)).astype(np.int64)
    hint = to_planes(matmul_mod64(w, a[:n], fast=False))[:, 1:p.q // 8]
    y = matmul_mod64(w, b, fast=False) & p.q_mask
    want = np.array((r.astype(object) @ w.T.astype(object)) % (1 << 32), dtype=np.int64)
    out = np.zeros((rows, m), dtype=np.int64)
    eng.decrypt_into("x", hint, [(y[:, DEGREE:], cts[1:], DEGREE), (y[:, :DEGREE], cts[:1], 0)], out,
                     p.delta_bits, 32)
    assert np.array_equal(out, want)
    assert eng.stats["hint_uploads"] == 1


def test_engine_matches_cpu_ntt_pre():
    from pgw.hemask import ntt
    from pgw.hemask.mlwe_ring import PretransformedEncryptor

    if not ntt.available():
        pytest.skip("pgw_ntt not built")
    p = HEParams(k=32, rank=16)
    a = public_matrix(bytes(32), 50, p)
    r = np.random.default_rng(2).integers(0, 1 << 32, size=(300, 50), dtype=np.int64)
    seeds = [b"\x01" * 32, b"\x02" * 32]
    want = PretransformedEncryptor(a, p).encrypt(r, _seeds=seeds)
    got = G.GpuNTTHE(a, p, CPU, hint_budget=0).encrypt(r, _seeds=seeds)
    for x, y in zip(got, want, strict=True):
        assert np.array_equal(x.b, y.b) and np.array_equal(x.secret, y.secret)


def test_self_test_cpu_and_selection():
    assert G.self_test(CPU, kernel="eager")
    assert G.status()[f"{CPU}/k32/r16/eager"]["ok"]
    if not torch.cuda.is_available():
        assert G.gpu_ntt_for(public_matrix(bytes(32), 8, HEParams(k=32, rank=16)), HEParams(k=32, rank=16)) is None


def test_kernel_choice_errors():
    p = HEParams(k=32, rank=16)
    a = public_matrix(bytes(32), 4, p)
    with pytest.raises(Exception, match="kernel"):
        G.GpuNTTHE(a, p, CPU, kernel="bogus")
    with pytest.raises(Exception, match="CUDA"):
        G.GpuNTTHE(a, p, CPU, kernel="triton")


_INTERP = textwrap.dedent("""
    import numpy as np, torch
    from pgw.hemask import gpuntt as G
    from pgw.hemask.lwe import sample_secret, matmul_mod64
    from pgw.hemask.fastmm import to_planes
    assert G.triton_available(), G._TRITON_ERR
    tb = G._tables(torch.device("cpu"))
    rng = np.random.default_rng(0)
    h = rng.integers(0, 1 << 63, size=(37, 4096), dtype=np.int64).view(np.uint64) * np.uint64(2)
    pl = np.ascontiguousarray(to_planes(h)[:, 1:7])
    hm = h & np.uint64(((1 << 56) - 1) ^ 255)
    th = G.fwd_planes_triton(torch.from_numpy(pl), 1, 16, tb.zetas)
    assert torch.equal(th, G.fwd_planes_eager(torch.from_numpy(pl), 1, 16, tb))
    for g in (1, 3):
        sm = [sample_secret(bytes([i]) * 32, 16) for i in range(g)]
        sh = G.transform_secrets(torch.from_numpy(G._secret_polys(sm, 16)), tb)
        z = G.mac_inv_crt_triton(th, 3, 30, sh, tb.zetas).numpy().view(np.uint64)
        assert np.array_equal(z, matmul_mod64(np.concatenate(sm, axis=1), hm[3:33], small_left=False, fast=False))
    print("OK")
""")


def test_triton_kernels_in_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1")
    src = os.path.join(os.path.dirname(G.__file__), "..", "..")
    env["PYTHONPATH"] = os.path.abspath(src) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run([sys.executable, "-c", _INTERP], env=env, capture_output=True, text=True, timeout=900)
    if "triton_available" in res.stderr or "AssertionError" in res.stderr and "TRITON" in res.stderr:
        pytest.skip(f"triton interpreter unavailable: {res.stderr[-300:]}")
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-2000:]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("kernel", ["triton", "eager"])
def test_cuda_self_test(kernel):  # pragma: no cover - GPU only
    if kernel == "triton" and not G.triton_available():
        pytest.skip("no triton")
    assert G.self_test("cuda", kernel=kernel), G.status()
