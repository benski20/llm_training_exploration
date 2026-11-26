"""The pks-v3 hint build's native kernels (``pgw_pks.switch_hint`` / ``a_planes`` / ``combine_planes`` /
``hint_keyswitch``) and :mod:`pgw.hemask.hint_gemm`: every value equal to the numpy reference, for every ISA copy
and thread count, and a whole store bit-identical to the reference compact hint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pgw.core.ringmm import cpu_int8_available
from pgw.hemask import pks
from pgw.hemask import pks_native as pn
from pgw.hemask.hint_gemm import MAX_N, PLANES, HintGemm, a_planes_ref, hint_rows
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.pks_backend import PKSHintStore, switch_hint_fast

M56 = np.uint64((1 << 56) - 1)
PAR = pks.PKSParams()
NATIVE = pytest.mark.skipif(not (pn.available() and hasattr(pn.native(), "hint_keyswitch")),
                            reason="pgw_pks (with the hint-build kernels) not built")
ISAS = ("scalar", "avx2", "avx512", "avx512ifma")


def _edge_h(rows: int, seed: int = 0) -> np.ndarray:
    """Random ``H`` plus values at the rounding boundaries of the switch and the extremes of 2^56."""
    rng = np.random.default_rng(seed)
    h = rng.integers(0, 1 << 56, (rows, pks.NU), dtype=np.uint64)
    q1 = PAR.q1
    edge = []
    for k in (0, 1, 2, q1 // 2, q1 - 1, q1):  # h with h q1 / 2^56 = k +- 1/2
        base = ((2 * k + 1) << 55) // q1
        edge += [base - 1, base, base + 1]
    edge += [0, 1, (1 << 55) - 1, 1 << 55, (1 << 56) - 2, (1 << 56) - 1]
    edge_a = np.array([e % (1 << 56) for e in edge], dtype=np.uint64)
    h.reshape(-1)[:edge_a.size] = edge_a
    return h


@NATIVE
@pytest.mark.parametrize("isa", ISAS)
def test_native_switch_equals_reference(isa: str) -> None:
    h = _edge_h(64)
    want = pks.switch_hint(h, PAR)
    for thr in (1, 3):
        assert np.array_equal(pn.native().switch_hint(h, thr, isa), want)
    hi = h | np.uint64(0xFF << 56)  # bits above 56 are ignored, as in the reference
    assert np.array_equal(pn.native().switch_hint(hi, 2, isa), pks.switch_hint(hi, PAR))
    assert np.array_equal(switch_hint_fast(h), want)


@NATIVE
@pytest.mark.parametrize("n", [16, 896, 4864])
def test_a_planes_and_gemm_exact(n: int) -> None:
    seed = bytes(range(32))
    ring = pks.public_ring_rows(seed, n, "3.down")
    for cb in (4096, 512, 64):
        planes = pn.native().a_planes(ring, cb, 3)
        assert planes.shape == (PLANES * pks.NU, n) and np.array_equal(planes, a_planes_ref(ring, cb))
    p0 = a_planes_ref(ring).reshape(PLANES, pks.NU, n)  # plane-major: row d * 4096 + col
    a = pks.a_mlwe_rows(ring)
    for d in range(PLANES):
        assert np.array_equal(p0[d].view(np.uint8) ^ np.uint8(0x80), ((a.T >> np.uint64(8 * d)) & np.uint64(255)))
    rng = np.random.default_rng(n)
    w = rng.integers(-127, 128, (96, n), dtype=np.int8)
    w[0], w[1], w[2] = 127, -127, 0  # extreme rows (largest int32 partial sums) and a zero row
    want = matmul_mod64(w, a, fast=False) & M56
    g = HintGemm(seed, n, "3.down", fast=True)  # the int8 path where CPU int8 GEMMs are exact
    assert g.fast == cpu_int8_available()
    assert np.array_equal(g(w), want)
    assert np.array_equal(g(w[5:37]), want[5:37])  # any row block
    assert np.array_equal(HintGemm(seed, n, "3.down", fast=False)(w), want)


@NATIVE
def test_combine_planes_exact() -> None:
    rng = np.random.default_rng(3)
    part = rng.integers(-(1 << 31), 1 << 31, (5, PLANES, 64), dtype=np.int64).astype(np.int32)
    off = rng.integers(-(1 << 31), 1 << 31, 5, dtype=np.int64).astype(np.int32)
    want = np.zeros((5, 64), dtype=np.uint64)
    for d in range(PLANES):
        want += (part[:, d].astype(np.int64) + off[:, None].astype(np.int64)).astype(np.uint64) << np.uint64(8 * d)
    for isa in ISAS:
        assert np.array_equal(pn.native().combine_planes(part, off, 2, isa), want & M56)


def test_max_inner_dimension_is_exact() -> None:
    assert 2 * 127 * 128 * MAX_N < 1 << 31


@NATIVE
@pytest.mark.parametrize("isa", ISAS)
def test_hint_keyswitch_equals_reference(isa: str) -> None:
    """The fused switch + key switch (and the old two-step native path) against the numpy compact hint."""
    h = _edge_h(64, 1)
    k_hat = pks.ks_a(b"k" * 32)
    hs = pks.switch_hint(h, PAR)
    want = pks.compact_hint(hs, k_hat.astype(np.uint64), PAR)
    nat = pn.native()
    ops = k_hat[None].astype(np.uint32)
    for thr in (1, 3):
        assert np.array_equal(nat.hint_keyswitch(h, ops, True, thr, isa), want)
        assert np.array_equal(nat.hint_keyswitch(hs, ops, False, thr, isa), want)
        v = nat.keyswitch(hs, ops, None, 36, thr, isa).astype("<u4")
        assert np.array_equal(v.view(np.uint8).reshape(*v.shape, 4)[..., :3], want)
    assert np.array_equal(hint_rows(h, ops), want)


@NATIVE
@pytest.mark.parametrize("isa", ISAS)
def test_provider_reply_keyswitch_equals_reference(isa: str) -> None:
    """The provider's reply key switch (same kernel, ``y`` given) is unchanged on every ISA copy."""
    rng = np.random.default_rng(7)
    m = 64
    hs = pks.switch_hint(_edge_h(m, 2), PAR)
    beta = np.stack([np.stack([rng.integers(0, p, (pks.RANK, pks.NR), dtype=np.uint64) for p in pks.QP])
                     for _ in range(2)]).astype(np.uint32)
    y = rng.integers(0, 1 << 56, (m, 2 * pks.D), dtype=np.uint64)
    want = pn.keyswitch(hs, beta, y, force_reference=True)
    for thr in (1, 2):
        assert np.array_equal(pn.native().keyswitch(hs, beta, y, 36, thr, isa), want)


@NATIVE
def test_store_build_equals_reference_compact_hint(tmp_path: Path) -> None:
    """A whole :meth:`PKSHintStore.build` (fast GEMM, fused tail) equals the numpy reference hint file for file."""
    rng = np.random.default_rng(11)
    shapes = {"0.qkv": (96, 64), "0.down": (64, 160)}
    weights = {k: rng.integers(-127, 128, s, dtype=np.int8) for k, s in shapes.items()}
    lins = [SimpleNamespace(id=SimpleNamespace(key=k), w8=torch.from_numpy(w)) for k, w in weights.items()]
    pack = SimpleNamespace(fingerprint="ab" * 32, iter_linears=lambda: iter(lins))
    seeds = (b"a" * 32, b"k" * 32)
    st = PKSHintStore.build(tmp_path, pack, seeds=seeds)
    k_hat = pks.ks_a(seeds[1]).astype(np.uint64)
    for k, w in weights.items():
        a = pks.a_mlwe_rows(pks.public_ring_rows(seeds[0], w.shape[1], k))
        want = pks.compact_hint(pks.switch_hint(matmul_mod64(w, a, fast=False) & M56, PAR), k_hat, PAR)
        assert np.array_equal(np.asarray(st.hint(k)), want)
    st.close()
