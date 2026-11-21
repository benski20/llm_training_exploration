"""Fused NTT decryption of the encrypted mask factory (``HEMaskBackend(decrypt=...)``): same bits as the
dense path in every mode, pre-transformed hints within a budget, calibration and fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.ring import RingParams
from pgw.factory import MaskGenerator
from pgw.hemask import DEGREE, HEParams, matmul_mod64, ntt
from pgw.hemask.backend import DECRYPT_MODES, HEMaskBackend, HintStore
from pgw.hemask.lwe import MaskCiphertext, secret_matrix

pytestmark = pytest.mark.skipif(not ntt.decrypt_available(), reason="pgw_ntt with decrypt_planes not built")

SEED = bytes(range(7, 39))


class _Provider:
    def __init__(self, pack: Any):
        self.w = {lw.id.key: lw.w8.numpy() for lw in pack.iter_linears()}

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        return matmul_mod64(self.w[linear], b)


def _items(rng: np.random.Generator, rank: int, m: int, rows: list[list[int]]) -> list[Any]:
    items = []
    s0 = 3
    for group_rows in rows:
        cts = [MaskCiphertext(b=np.empty((0, DEGREE), np.uint64),
                              secret=secret_matrix(rng.integers(-1, 2, size=(rank, DEGREE))), rows=r)
               for r in group_rows]
        y = rng.integers(0, 1 << 63, size=(m, DEGREE * len(cts)), dtype=np.uint64)
        items.append((y, cts, s0))
        s0 += sum(group_rows)
    return items


@pytest.mark.parametrize("q_bits", [56, 64])
def test_decrypt_into_matches_dense(q_bits: int) -> None:
    """Native fused decryption (planes, row blocks, pre-transformed) == the backend's dense path."""
    rng = np.random.default_rng(q_bits)
    params = HEParams(k=32 if q_bits == 56 else 40, rank=8, q_bits=q_bits)
    m, npl = 53, q_bits // 8 - 1
    hint = rng.integers(-128, 128, size=(m, npl, params.dim), dtype=np.int8)
    items = _items(rng, params.rank, m, [[256, 100], [7]])
    total = 3 + 256 + 100 + 7
    be = object.__new__(HEMaskBackend)  # only the attributes the decrypt helpers read
    be.params = params
    be.ring = RingParams.from_name("K32A12C27" if q_bits == 56 else "K40A12")
    want = np.full((total, m), -1, dtype=np.int64)
    be._decrypt_dense(hint, items, want)
    got = np.full((total, m), -1, dtype=np.int64)
    be._decrypt_ntt(hint, items, got)
    assert np.array_equal(got, want)
    pre = ntt.TransformedOperand.from_planes(hint, first_byte=1)
    got = np.full((total, m), -1, dtype=np.int64)
    be._decrypt_ntt(pre, items, got)
    assert np.array_equal(got, want)
    # rows in two blocks with a row offset
    secs = np.stack([ntt.secrets_of(ct.secret)[0] for _, cts, _ in items for ct in cts])
    slots = np.concatenate([np.r_[np.arange(3, 359), np.full(156, -1)], np.r_[np.arange(359, 366), np.full(249, -1)]])
    got = np.full((total, m), -1, dtype=np.int64)
    for r0, r1 in ((0, 20), (20, m)):
        ntt.decrypt_into(hint[r0:r1], secs, [y for y, _, _ in items], slots, params.delta_bits, be.ring.k, got,
                         row_base=r0)
    assert np.array_equal(got, want)
    with pytest.raises(Exception, match="out of range or repeated"):
        ntt.decrypt_into(hint, secs, [y for y, _, _ in items], np.zeros_like(slots), params.delta_bits, be.ring.k,
                         got)


@pytest.mark.parametrize("mode,budget", [("dense", 0), ("ntt", 0), ("ntt_pre", 1 << 40), ("ntt_pre", 1),
                                         ("auto", 0), ("auto", 1 << 40)])
def test_factory_output_identical_in_every_mode(tiny_pack: Any, tmp_path: Path, mode: str, budget: int) -> None:
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    local = {(g.id.key, g.slot_start): g.wr.clone()
             for g in MaskGenerator(tiny_pack, ring, device="cpu").generate(SEED, 300, checks=2, chunk_slots=300)}
    remote = HEMaskBackend(tiny_pack, ring, hints, _Provider(tiny_pack), groups_per_call=1, decrypt=mode,
                           hint_ntt_bytes=budget, device=None)
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    he = {(g.id.key, g.slot_start): g.wr.clone() for g in gen.generate(SEED, 300, checks=2, chunk_slots=300)}
    assert local.keys() == he.keys()
    for k in local:
        assert torch.equal(local[k], he[k]), k
    paths = remote.decrypt_paths
    if mode == "dense":
        assert set(paths) == {"dense"}
    elif mode == "ntt" or (mode == "ntt_pre" and budget == 1):
        assert set(paths) == {"ntt"}  # a budget too small for any hint: the planes path
    elif budget:
        assert set(paths) == {"ntt_pre"} and remote._pre
    else:  # auto, no budget: whatever the calibration chose, recorded with an identity check
        assert set(paths) <= {"dense", "ntt"} and remote.decrypt_calibration
        assert all(c.get("identical") == "True" for c in remote.decrypt_calibration.values())


def test_pre_budget_is_respected(tiny_pack: Any, tmp_path: Path) -> None:
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    specs = list(tiny_pack.linear_specs())
    one = 12 * (-(-specs[0].out_features // 16) * 16) * 8 * DEGREE
    remote = HEMaskBackend(tiny_pack, ring, hints, _Provider(tiny_pack), decrypt="ntt_pre", hint_ntt_bytes=one,
                           device=None)
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    list(gen.generate(SEED, 256, checks=2, chunk_slots=256))
    assert remote._pre_used <= one and len(remote._pre) >= 1
    assert remote.decrypt_paths.get("ntt", 0) > 0 and remote.decrypt_paths.get("ntt_pre", 0) > 0


def test_native_fault_falls_back_to_dense(tiny_pack: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    local = {(g.id.key, g.slot_start): g.wr.clone()
             for g in MaskGenerator(tiny_pack, ring, device="cpu").generate(SEED, 256, checks=2, chunk_slots=256)}

    def boom(*a: Any, **k: Any) -> None:
        raise ValueError("simulated native fault")

    monkeypatch.setattr(ntt, "decrypt_into", boom)
    remote = HEMaskBackend(tiny_pack, ring, hints, _Provider(tiny_pack), decrypt="ntt", device=None)
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    he = {(g.id.key, g.slot_start): g.wr.clone() for g in gen.generate(SEED, 256, checks=2, chunk_slots=256)}
    assert all(torch.equal(local[k], he[k]) for k in local)
    assert remote._ntt_broken and set(remote.decrypt_paths) == {"dense"}


def test_calibration_refuses_differing_bits(tiny_pack: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    remote = HEMaskBackend(tiny_pack, ring, hints, _Provider(tiny_pack), decrypt="auto", device=None)
    real = HEMaskBackend._decrypt_ntt

    def wrong(self: Any, big: Any, items: Any, out: np.ndarray) -> None:
        real(self, big, items, out)
        out[0, 0] ^= 1

    monkeypatch.setattr(HEMaskBackend, "_decrypt_ntt", wrong)
    assert remote._calibrated(6, 2) == "dense"
    assert remote.decrypt_calibration[2]["identical"] == "False"


def test_mode_validation(tiny_pack: Any, tmp_path: Path) -> None:
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    assert DECRYPT_MODES == ("auto", "dense", "ntt", "ntt_pre")
    with pytest.raises(Exception, match="decrypt must be one of"):
        HEMaskBackend(tiny_pack, ring, hints, _Provider(tiny_pack), decrypt="fast")
    with pytest.raises(Exception, match="hint_ntt_bytes"):
        HEMaskBackend(tiny_pack, ring, hints, _Provider(tiny_pack), hint_ntt_bytes=-1)
