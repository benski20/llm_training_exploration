"""Factory into the in-memory mask ring: bit-identical masks, ring-first scheduling, wake-ups, overflow."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A12_C27, SeedExpander
from pgw.core.errors import ConfigError
from pgw.factory import FactoryScheduler, MaskGenerator
from pgw.factory.config import FactorySettings
from pgw.factory.embedded import EmbeddedFactory
from pgw.factory.sinks import RingSink, StoreSink
from pgw.maskstore import MaskStore, StaticKeyProvider
from pgw.maskstore.ringbuf import MaskRing, RingFull, SeedRegistry, ring_geometry, seed_digest

from .conftest import FakePack, reference_segment

RING = RING_K32_A12_C27


def _ring(pack: FakePack, slabs: int = 6, segment_slots: int = 16, seeds: SeedRegistry | None = None) -> MaskRing:
    g = ring_geometry(pack.linear_specs(), RING, 1 << 30, segment_slots=segment_slots)
    return MaskRing(pack.fingerprint, RING, pack.linear_specs(), capacity_bytes=slabs * g.slab_bytes,
                    segment_slots=segment_slots, min_soundness_bits=0,
                    seeds=seeds if seeds is not None else SeedRegistry())


class _Recording:
    """Wraps a sink; keeps a copy of every W r chunk the generator wrote (in order)."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.rows: dict[str, list[torch.Tensor]] = {}

    def writer(self, *a: Any, **kw: Any) -> Any:
        w = self.inner.writer(*a, **kw)
        rec = self

        class W:
            def write_linear(self, lid: Any, wr: torch.Tensor, uw: torch.Tensor) -> None:
                rec.rows.setdefault(str(lid), []).append(wr.clone())
                w.write_linear(lid, wr, uw)

            def commit(self) -> str:
                return str(w.commit())

            def abort(self) -> None:
                w.abort()

        return W()


@pytest.mark.parametrize("pipeline", [False, True])
def test_ring_masks_equal_factory_output(fake_pack: FakePack, pipeline: bool) -> None:
    """Masks taken from the ring are bit-identical to what the factory produced: the W r rows it wrote,
    the independent reference W r, and r regenerated from the seed."""
    ring = _ring(fake_pack)
    try:
        gen = MaskGenerator(fake_pack, RING, pipeline=pipeline)
        rec = _Recording(RingSink({ring.key: ring}))
        seed = os.urandom(32)
        gen.generate_into_store(rec, 16, checks=2, chunk_slots=5, seed=seed, store_chunk_slots=4)
        ref = reference_segment(fake_pack, RING, seed, 16, 2)
        res = ring.reserve(16)
        exp = SeedExpander(seed)
        for s in fake_pack.linear_specs():
            key = s.id.key
            produced = torch.cat(rec.rows[key])
            rs, wrs = [], []
            while res.remaining(key):
                _, r, wr, _ = res.take_checked(key, 7, checks=True)
                rs.append(r)
                wrs.append(wr.to(torch.int64))
            r_all, wr_all = torch.cat(rs), torch.cat(wrs)
            assert torch.equal(wr_all, produced)
            assert torch.equal(wr_all, ref[key][0])
            assert torch.equal(r_all, exp.ring_uniform_rows(s.in_features, RING.k, 0, 16, "r", key))
            assert torch.equal(res.checks(key)[1], ref[key][1])
        res.close()
    finally:
        ring.close()


def _settings(tmp_path: Path, **kw: Any) -> FactorySettings:
    base: dict[str, Any] = {"pack_path": tmp_path / "pack", "segment_slots": 32, "chunk_slots": 16,
                            "low_water_slots": 64, "target_slots": 128, "poll_interval_s": 30.0,
                            "state_dir": tmp_path / "state", "error_backoff_s": 0.05, "checks": 2}
    base.update(kw)
    return FactorySettings(**base)


def _store(tmp_path: Path) -> MaskStore:
    return MaskStore.open(tmp_path / "store", StaticKeyProvider(os.urandom(32)), low_water_slots=0, fsync=False,
                          min_free_bytes=0, allow_no_external_counter=True, min_soundness_bits=0)


def test_scheduler_ring_first_then_store(tmp_path: Path, fake_pack: FakePack) -> None:
    store = _store(tmp_path)
    seeds = SeedRegistry()
    ring = _ring(fake_pack, slabs=4, seeds=seeds)
    try:
        gen = MaskGenerator(fake_pack, RING)
        sch = FactoryScheduler(_settings(tmp_path), StoreSink(store, seeds=seeds), {gen.key: gen},
                               ring_sink=RingSink({ring.key: ring}, store=store), metrics=False)
        snapshots: list[str] = []  # every factory-state.json the run writes, not only the last one
        orig_save = sch._save_state

        def save() -> None:
            orig_save()
            snapshots.append((sch.state_dir / "factory-state.json").read_text())

        sch._save_state = save  # type: ignore[method-assign]
        plans = sch.plan()
        assert [(p.tier, p.priority) for p in plans] == [("ring", 0), ("store", 1)]
        assert plans[0].segments == 4 and plans[0].slots_per_segment == ring.segment_slots
        produced = sch.run_once()
        assert len(produced) == 4 + plans[1].segments
        assert ring.available == ring.capacity_slots
        state_text = "\n".join(snapshots)
        assert len(snapshots) >= 2  # the store plan saved state (in_progress) right after the ring plans
        ring_ids = produced[:4]
        assert ring_ids and not any(i in state_text for i in ring_ids)  # ring ids never on disk ...
        assert sch.state.last_segment_id == produced[-1]  # ... even when a store plan saves state after them
        assert sch.status()["last_segment_id"] not in ring_ids
        assert sch.ring_segments == 4
        # the next plan: ring full (no ring plan), store refilling to target
        assert all(p.tier == "store" for p in sch.plan())
        # overflow segments have fresh seeds: store seed hashes and ring seeds never meet
        with store._lock.hold():
            store_hashes = {r[0] for r in store.conn.execute("SELECT seed_hash FROM segments")}
        assert store_hashes and all(h in seeds for h in store_hashes)
        assert len(seeds) == 4 + len(store_hashes)
    finally:
        ring.close()
        store.close()


def test_scheduler_ring_full_is_not_an_error(tmp_path: Path, fake_pack: FakePack, monkeypatch: Any) -> None:
    ring = _ring(fake_pack, slabs=4)
    try:
        gen = MaskGenerator(fake_pack, RING)
        sch = FactoryScheduler(_settings(tmp_path), None, {gen.key: gen}, ring_sink=RingSink({ring.key: ring}),
                               metrics=False)
        plans = sch.plan()
        assert [p.tier for p in plans] == ["ring"]  # no store tier at all
        w = ring.writer(fake_pack.fingerprint, RING, fake_pack.linear_specs(), slots=4, seed=os.urandom(32))
        w2 = ring.writer(fake_pack.fingerprint, RING, fake_pack.linear_specs(), slots=4, seed=os.urandom(32))
        produced = sch.run_once()  # planned 4 slabs, only 2 are free: RingFull ends the plan quietly
        assert len(produced) == 2
        assert sch.consecutive_errors == 0
        w.abort()
        w2.abort()
        assert sch._fit_to_disk([]) == []
        assert not (sch.state_dir / "factory-state.json").exists()  # ring plans never write state
    finally:
        ring.close()


def test_free_slab_wakes_run_forever(tmp_path: Path, fake_pack: FakePack) -> None:
    ring = _ring(fake_pack, slabs=4)
    gen = MaskGenerator(fake_pack, RING)
    sch = FactoryScheduler(_settings(tmp_path, poll_interval_s=30.0), None, {gen.key: gen},
                           ring_sink=RingSink({ring.key: ring}), metrics=False)
    t = threading.Thread(target=sch.run_forever, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 20
        while ring.available < ring.capacity_slots and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ring.available == ring.capacity_slots
        before = ring.counts()["published_segments"]
        ring.reserve(ring.segment_slots).close()  # drains one slab -> wiper -> free listener -> wake
        t0 = time.monotonic()
        while ring.counts()["published_segments"] == before and time.monotonic() - t0 < 10:
            time.sleep(0.02)
        assert ring.counts()["published_segments"] == before + 1
        assert time.monotonic() - t0 < 5  # far below poll_interval_s (30 s)
    finally:
        sch.stop()
        t.join(10)
        ring.close()


def test_embedded_factory_ring_only(tmp_path: Path, fake_pack: FakePack) -> None:
    from pgw.maskstore.tiered import TieredMaskSource

    class Cfg:
        capacity_bytes = 6 * ring_geometry(fake_pack.linear_specs(), RING, 1 << 30, segment_slots=16).slab_bytes
        segment_slots = 16
        allow_unlocked = False

    src = TieredMaskSource(None, Cfg(), min_soundness_bits=0)
    src.attach_model("fake", fake_pack.fingerprint, RING, fake_pack.linear_specs())
    gen = MaskGenerator(fake_pack, RING)
    fac = EmbeddedFactory(src, settings=_settings(tmp_path, state_dir=None), run_dir=tmp_path / "run",
                          lock_dir=tmp_path / "locks", generators={gen.key: gen})
    other = EmbeddedFactory(src, settings=_settings(tmp_path, state_dir=None), run_dir=tmp_path / "run2",
                            lock_dir=tmp_path / "locks", generators={gen.key: gen})
    try:
        assert fac.start()
        assert not other.start()  # one device, one factory
        assert "already running" in (other.error or "")
        mr = src.ring_for(fake_pack.fingerprint, RING)
        assert mr is not None
        deadline = time.monotonic() + 20
        while mr.available < mr.capacity_slots and time.monotonic() < deadline:
            time.sleep(0.02)
        assert mr.available == mr.capacity_slots
        res = src.reserve(fake_pack.fingerprint, RING, 20)
        res.take(fake_pack.linear_specs()[0].id, 3)
        res.close()
    finally:
        fac.stop()
        src.close()
    assert mr.closed and mr.available == 0


def test_ring_sink_refuses(fake_pack: FakePack) -> None:
    ring = _ring(fake_pack)
    try:
        sink = RingSink({ring.key: ring})
        specs = fake_pack.linear_specs()
        with pytest.raises(ConfigError):
            sink.writer(fake_pack.fingerprint, RING, specs, slots=4, seed=os.urandom(32), destination="ab" * 16)
        with pytest.raises(ConfigError):
            sink.writer("cd" * 32, RING, specs, slots=4, seed=os.urandom(32))
        assert not hasattr(sink, "export_segment") and not hasattr(sink, "recover") and not hasattr(sink, "state_dir")
        for _ in range(ring.n_slabs):
            sink.writer(fake_pack.fingerprint, RING, specs, slots=4, seed=os.urandom(32))
        with pytest.raises(RingFull):
            sink.writer(fake_pack.fingerprint, RING, specs, slots=4, seed=os.urandom(32))
        assert seed_digest(b"\x01" * 32) == __import__("pgw.maskstore.format", fromlist=["seed_hash"]).seed_hash(
            b"\x01" * 32)
    finally:
        ring.close()


def test_store_knows_seed_prefers_public_api_and_fails_closed(tmp_path: Path) -> None:
    from pgw.core.errors import MaskStoreError
    from pgw.factory.sinks import store_knows_seed

    class Public:
        def has_seed(self, digest: str) -> bool:
            return digest == "known"

    assert store_knows_seed(Public(), "known") and not store_knows_seed(Public(), "other")
    with pytest.raises(MaskStoreError, match="seed table"):
        store_knows_seed(object(), "x")  # internals changed: refuse, never "unknown seed"

    class Broken:
        class _lock:  # noqa: N801
            @staticmethod
            def hold() -> Any:
                import contextlib

                return contextlib.nullcontext()

        class conn:  # noqa: N801
            @staticmethod
            def execute(*a: Any) -> Any:
                raise RuntimeError("no such column: seed_hash")

    with pytest.raises(MaskStoreError, match="seed table"):
        store_knows_seed(Broken(), "x")
    store = _store(tmp_path)
    try:
        seed = os.urandom(32)
        assert not store_knows_seed(store, seed_digest(seed))
        MaskGenerator(FakePack(), RING).generate_into_store(StoreSink(store), 8, checks=2, chunk_slots=8, seed=seed)
        assert store_knows_seed(store, seed_digest(seed))
    finally:
        store.close()


def test_device_lock_shared_by_v1_and_v2(tmp_path: Path) -> None:
    from pgw.factory.embedded import acquire_device_lock

    lock = acquire_device_lock("cuda", tmp_path)
    try:
        with pytest.raises(ConfigError):
            acquire_device_lock("cuda:0", tmp_path)  # same GPU, other spelling
        other = acquire_device_lock("cuda:1", tmp_path)
        other.release()
    finally:
        lock.release()
    acquire_device_lock("CUDA:0", tmp_path).release()
