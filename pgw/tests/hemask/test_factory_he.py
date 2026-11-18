"""Encrypted mask generation inside the real factory: bit-identical to local generation, tamper-safe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import IntegrityError
from pgw.core.ring import RingParams
from pgw.factory import MaskGenerator
from pgw.hemask import matmul_mod64
from pgw.hemask.backend import HEMaskBackend, HintStore

SEED = bytes(range(1, 33))


class FakeProvider:
    """Stands in for the compute service: W8 b mod 2^64 (optionally corrupting one element)."""

    def __init__(self, pack: Any, tamper: bool = False):
        self.w = {lw.id.key: lw.w8.numpy() for lw in pack.iter_linears()}
        self.tamper = tamper
        self.calls = 0
        self.seen: list[np.ndarray] = []

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.seen.append(b.copy())
        y = matmul_mod64(self.w[linear], b)
        if self.tamper and self.calls == 3:
            y[1, 2] += np.uint64(1 << 30)
        return y


def _collect(gen: MaskGenerator, slots: int, chunk: int) -> dict[tuple[str, int], torch.Tensor]:
    return {(g.id.key, g.slot_start): g.wr.clone() for g in gen.generate(SEED, slots, checks=2, chunk_slots=chunk)}


def test_he_factory_matches_local_factory(tiny_pack: Any, tmp_path: Path) -> None:
    ring = RingParams.from_name("K40A12")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8)
    assert HintStore.open(tmp_path / "hints", tiny_pack.fingerprint, 8) is not None  # reused, not rebuilt
    provider = FakeProvider(tiny_pack)
    remote = HEMaskBackend(tiny_pack, ring, hints, provider, groups_per_call=2)
    local = _collect(MaskGenerator(tiny_pack, ring, device="cpu"), 300, 300)
    he_gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    assert he_gen.self_check_rows == -1  # every row is verified in encrypted mode
    he = _collect(he_gen, 300, 300)
    assert local.keys() == he.keys()
    for k in local:
        assert torch.equal(local[k], he[k]), k
    assert he_gen.last_stats.self_checked_rows == 300 * len(list(tiny_pack.linear_specs()))
    # the provider only ever saw uniform-looking 64-bit words
    top = np.concatenate([(b >> np.uint64(63)).ravel() for b in provider.seen]).astype(np.float64)
    assert abs(top.mean() - 0.5) < 0.05


def test_he_factory_rejects_tampered_provider(tiny_pack: Any, tmp_path: Path) -> None:
    ring = RingParams.from_name("K40A12")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8)
    remote = HEMaskBackend(tiny_pack, ring, hints, FakeProvider(tiny_pack, tamper=True))
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    with pytest.raises(IntegrityError):
        _collect(gen, 256, 256)


def test_hint_seed_is_fresh_per_build(tiny_pack: Any, tmp_path: Path) -> None:
    a = HintStore.build(tmp_path / "a", tiny_pack, rank=8)
    b = HintStore.build(tmp_path / "b", tiny_pack, rank=8)
    assert a.a_seed != b.a_seed and len(a.a_seed) == 32
