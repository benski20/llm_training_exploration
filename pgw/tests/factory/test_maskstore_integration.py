"""Factory -> real :class:`pgw.maskstore.MaskStore` round trip (skipped while the store is a stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, LinearId, center, freivalds_verify, mask_activations, ringmm_reference
from pgw.factory import FactoryScheduler, FactorySettings, MaskGenerator

from .conftest import FakePack


@pytest.fixture
def real_store(tmp_path: Path) -> Any:
    try:
        from pgw.maskstore import FileKeyProvider, MaskStore

        kp = FileKeyProvider(tmp_path / "kek", create=True)
        store = MaskStore.open(tmp_path / "store", kp)
    except NotImplementedError:
        pytest.skip("pgw.maskstore not implemented yet")
    yield store
    store.close()


def test_store_round_trip(real_store: Any) -> None:
    pack = FakePack()
    gen = MaskGenerator(pack, RING_K40_A12)
    seg = gen.generate_into_store(real_store, 9, checks=2, chunk_slots=4, factory_id="it")
    assert isinstance(seg, str) and seg
    inv = real_store.inventory(pack.fingerprint, RING_K40_A12)
    assert sum(i.slots_available for i in inv) == 9
    with real_store.reserve(pack.fingerprint, RING_K40_A12, 5) as res:
        for spec in pack.linear_specs():
            lid: LinearId = spec.id
            r, wr = res.take(lid, 5)
            w = pack.weights[lid.key]
            assert wr.tolist() == ringmm_reference(r, w, 40).tolist()
            u, uw = res.checks(lid)
            xq = torch.randint(-100, 100, (5, spec.in_features), dtype=torch.int64)
            x_hat = mask_activations(xq, r, RING_K40_A12)
            y_hat = torch.from_numpy(ringmm_reference(x_hat, w, 40))
            freivalds_verify(x_hat, y_hat, u, uw, RING_K40_A12)
            assert torch.equal(center(y_hat - wr, RING_K40_A12), xq @ w.to(torch.int64).t())
    assert sum(i.slots_available for i in real_store.inventory(pack.fingerprint, RING_K40_A12)) == 4


def test_scheduler_with_real_store(real_store: Any, tmp_path: Path) -> None:
    pack = FakePack()
    g = MaskGenerator(pack, RING_K40_A12)
    s = FactorySettings(pack_path=tmp_path, segment_slots=16, chunk_slots=8, low_water_slots=20, target_slots=40,
                        state_dir=tmp_path / "state")
    with FactoryScheduler(s, real_store, {g.key: g}, metrics=False) as sch:
        ids = sch.run_once()
        assert len(ids) == 3
        assert sch.plan() == []
