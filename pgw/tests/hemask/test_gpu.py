"""CUDA path of the provider's encrypted multiply (ComputeKernel.mulmod64): exact vs the CPU reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core.types import LinearId, LinearSpec
from pgw.hemask import DEGREE, HEParams, decrypt_wr, encrypt_masks, matmul_mod64, public_matrix
from pgw.service.kernels import ComputeKernel


@pytest.mark.gpu
@pytest.mark.parametrize("m,n,cols", [(96, 64, DEGREE), (4608, 3584, DEGREE), (64, 18944, 4 * DEGREE), (37, 70, DEGREE)])
def test_cuda_mulmod64_exact(m: int, n: int, cols: int) -> None:
    rng = np.random.default_rng(m + n)
    w = rng.integers(-127, 128, size=(m, n)).astype(np.int8)
    if n == 18944:
        w = np.where(rng.random((m, n)) < 0.5, -127, 127).astype(np.int8)  # worst case
    b = rng.integers(0, 2**63, size=(n, cols), dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, size=(n, cols),
                                                                                             dtype=np.uint64)
    spec = LinearSpec(LinearId(0, "qkv"), m, n)
    kern = ComputeKernel(spec, torch.from_numpy(w), device="cuda")
    assert np.array_equal(kern.mulmod64(b), matmul_mod64(w.astype(np.int64), b))


@pytest.mark.gpu
def test_cuda_provider_roundtrip_decrypts_exactly() -> None:
    rng = np.random.default_rng(9)
    m, n = 512, 3584
    w = rng.integers(-127, 128, size=(m, n)).astype(np.int8)
    p = HEParams(k=40, rank=16)
    a = public_matrix(bytes(32), n, p)
    hint = matmul_mod64(w.astype(np.int64), a)
    r = rng.integers(0, 1 << 40, size=(DEGREE, n), dtype=np.int64)
    ct = encrypt_masks(r, a, p)
    kern = ComputeKernel(LinearSpec(LinearId(0, "o"), m, n), torch.from_numpy(w), device="cuda")
    got = decrypt_wr(kern.mulmod64(ct.b), hint, ct, p)
    ref = np.array((r.astype(object) @ w.T.astype(object)) % (1 << 40), dtype=np.int64)
    assert np.array_equal(got, ref)
