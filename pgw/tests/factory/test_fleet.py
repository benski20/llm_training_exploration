"""Horizontal scale-out of mask generation (pgw.factory.fleet) and store safety under concurrent writers."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import RING_K40_A12
from pgw.core.errors import ConfigError
from pgw.factory import FactoryScheduler, FactorySettings, MaskGenerator
from pgw.factory.fleet import (
    FactoryFleet,
    FleetScheduler,
    ProductionLedger,
    SegmentShardedRemote,
    worker_settings,
)
from pgw.maskstore import FileKeyProvider, MaskStore, write_key_file
from pgw.maskstore.segment_io import partial_in_use

from . import fleet_workers
from .conftest import FakePack

CTX = mp.get_context("spawn")


def _assert_clean(rep: Any) -> None:
    assert not (rep.missing_segments or rep.quarantined_segments or rep.orphan_files), rep


# ---- ledger -----------------------------------------------------------------------------------------
def test_ledger_claims_visible_to_live_peers_only(tmp_path: Path) -> None:
    a = ProductionLedger(tmp_path, "w0").open()
    b = ProductionLedger(tmp_path, "w1").open()
    with pytest.raises(ConfigError):
        ProductionLedger(tmp_path, "w0").open()  # same id twice
    with a.plan_lock():
        a.claim("ab" * 32, "K40A12", 16, 1000)
    peers = b.peer_claims()
    assert [(c["worker"], c["slots"], c["bytes"]) for c in peers] == [("w0", 16, 1000)]
    assert a.peer_claims() == []
    assert b.live_workers() == ["w0", "w1"]
    # a dead worker's claim is ignored even though its file is still there
    claim_file = tmp_path / "inflight" / "w0.json"
    data = claim_file.read_bytes()
    a.close()
    claim_file.write_bytes(data)
    assert b.peer_claims() == []
    assert b.live_workers() == ["w1"]
    # a restarted w0 voids its old claim
    a2 = ProductionLedger(tmp_path, "w0").open()
    assert not claim_file.exists()
    a2.close()
    b.close()


def test_plan_lock_is_exclusive_across_ledgers(tmp_path: Path) -> None:
    a = ProductionLedger(tmp_path, "w0").open()
    b = ProductionLedger(tmp_path, "w1").open()
    inside: list[str] = []
    overlap = []

    def run(led: ProductionLedger, tag: str) -> None:
        for _ in range(20):
            with led.plan_lock():
                inside.append(tag)
                if len(inside) > 1:
                    overlap.append(tuple(inside))
                time.sleep(0.001)
                inside.remove(tag)

    ts = [threading.Thread(target=run, args=(a, "a")), threading.Thread(target=run, args=(b, "b"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert overlap == []
    a.close()
    b.close()


# ---- fleet scheduler (two workers in threads, one real store) --------------------------------------
def _real_store(tmp_path: Path) -> MaskStore:
    kp = FileKeyProvider(tmp_path / "kek", create=True)
    return MaskStore.open(tmp_path / "store", kp)


@pytest.mark.parametrize("workers", [2, 3])
def test_fleet_workers_stop_at_target_together(tmp_path: Path, workers: int) -> None:
    pack = FakePack()
    root = tmp_path / "fleet"
    kp = FileKeyProvider(tmp_path / "kek", create=True)
    scheds = []
    stores = []
    for k in range(workers):
        store = MaskStore.open(tmp_path / "store", kp)
        stores.append(store)  # one store handle per worker, as in separate processes
        s = FactorySettings(pack_path=tmp_path, segment_slots=16, chunk_slots=8, low_water_slots=60,
                            target_slots=100, state_dir=root / f"w{k}", factory_id=f"t-w{k}")
        g = MaskGenerator(pack, RING_K40_A12)
        scheds.append(FleetScheduler(s, store, {g.key: g}, metrics=False,
                                     ledger=ProductionLedger(root, f"w{k}")))
    results: dict[int, list[str]] = {}
    errors: list[BaseException] = []

    def run(k: int) -> None:
        try:
            results[k] = scheds[k].run_once()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    ts = [threading.Thread(target=run, args=(k,)) for k in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    assert not errors, errors
    ids = [i for v in results.values() for i in v]
    assert len(ids) == len(set(ids)) == 7  # ceil(100 / 16): no overshoot with several workers
    store = stores[0]
    assert store.available_slots(pack.fingerprint, RING_K40_A12) == 112
    _assert_clean(store.verify_integrity())
    for sc, st in zip(scheds, stores, strict=True):
        assert sc.plan() == []
        sc.close()
        st.close()


def test_fleet_scheduler_counts_peer_inflight_and_disk(tmp_path: Path) -> None:
    pack = FakePack()
    root = tmp_path / "fleet"
    store = _real_store(tmp_path)
    s = FactorySettings(pack_path=tmp_path, segment_slots=16, low_water_slots=40, target_slots=60,
                        state_dir=root / "w0")
    g = MaskGenerator(pack, RING_K40_A12)
    sch = FleetScheduler(s, store, {g.key: g}, metrics=False, ledger=ProductionLedger(root, "w0"))
    peer = ProductionLedger(root, "w1").open()
    with peer.plan_lock():
        peer.claim(pack.fingerprint, RING_K40_A12.name, 32, 5 << 20)
    sch._start()
    sch._peers = sch.ledger.peer_claims()
    plans = sch.plan()
    assert len(plans) == 1 and plans[0].available == 32 and plans[0].segments == 2  # 60 - 32 -> two segments
    base = FactoryScheduler.disk_budget(sch)
    if base is not None:
        assert sch.disk_budget() == max(0, base - (5 << 20))
    peer.close()
    sch._peers = sch.ledger.peer_claims()
    assert (sch.plan()[0].available, sch.plan()[0].segments) == (0, 4)
    sch.close()
    store.close()


def test_fleet_scheduler_skips_failing_model(tmp_path: Path) -> None:
    pack = FakePack()
    store = _real_store(tmp_path)
    s = FactorySettings(pack_path=tmp_path, segment_slots=16, low_water_slots=10, target_slots=20,
                        state_dir=tmp_path / "fleet" / "w0")
    g = MaskGenerator(pack, RING_K40_A12)
    sch = FleetScheduler(s, store, {g.key: g}, metrics=False, ledger=ProductionLedger(tmp_path / "fleet", "w0"))

    def boom(*a: Any, **kw: Any) -> str:
        raise RuntimeError("provider down")

    g.generate_into_store = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        sch.run_once()
    assert not (tmp_path / "fleet" / "inflight" / "w0.json").exists()  # claim cleared on failure
    sch.close()
    store.close()


# ---- provider sharding ----------------------------------------------------------------------------
class _Rec:
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail
        self.calls = 0

    def __call__(self, linear: str, b: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise OSError(f"{self.name} down")
        return self.name


def test_sharded_remote_pins_rotates_and_fails_over() -> None:
    a, b, c = _Rec("a"), _Rec("b"), _Rec("c")
    pinned = SegmentShardedRemote([a, b, c], start=1)
    for _ in range(3):
        pinned.begin_segment()
        assert {pinned("x", None) for _ in range(4)} == {"b"}  # one segment -> one service
    rot = SegmentShardedRemote([a, b, c], start=0, rotate=True)
    seen = []
    for _ in range(4):
        rot.begin_segment()
        seen.append(rot("x", None))
    assert seen == ["a", "b", "c", "a"]
    b.fail = True
    fo = SegmentShardedRemote([a, b, c], start=1)
    fo.begin_segment()
    with pytest.raises(OSError):
        fo("x", None)
    fo.begin_segment()  # the failed segment is aborted; the next one moves on
    assert fo("x", None) == "c"
    assert fo.segments == [0, 1, 1]


def test_he_segments_sharded_over_two_providers_bit_identical(tmp_path: Path, tiny_pack: Any) -> None:
    """Two provider stubs, one factory worker rotating per segment: every request of a segment goes to one
    provider, and the stored masks equal the plain factory's for the same seeds."""
    from pgw.hemask.backend import HEMaskBackend, HintStore
    from tests.hemask.test_factory_he import FakeProvider

    ring = RING_K40_A12
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8)
    p0, p1 = FakeProvider(tiny_pack), FakeProvider(tiny_pack)
    sharded = SegmentShardedRemote([p0, p1], rotate=True)
    he = MaskGenerator(tiny_pack, ring, device="cpu", remote=HEMaskBackend(tiny_pack, ring, hints, sharded,
                                                                           groups_per_call=2))
    local = MaskGenerator(tiny_pack, ring, device="cpu")
    seeds = [bytes([i]) * 32 for i in (3, 4)]
    calls = []
    for seed in seeds:
        sharded.begin_segment()
        before = (p0.calls, p1.calls)
        got = {(x.id.key, x.slot_start): x.wr.clone() for x in he.generate(seed, 300, checks=2, chunk_slots=300)}
        calls.append((p0.calls - before[0], p1.calls - before[1]))
        ref = {(x.id.key, x.slot_start): x.wr.clone() for x in local.generate(seed, 300, checks=2, chunk_slots=300)}
        assert got.keys() == ref.keys()
        assert all(torch.equal(got[k], ref[k]) for k in ref)
    # segment 0 only on provider 0, segment 1 only on provider 1, the same number of requests each
    assert calls[0][1] == 0 and calls[1][0] == 0 and calls[0][0] == calls[1][1] > 0
    # request sizes identical across providers (public shape only)
    assert [b.shape for b in p0.seen] == [b.shape for b in p1.seen]
    assert all(np.asarray(b).dtype == np.uint64 for b in p0.seen + p1.seen)


# ---- store safety: concurrent writer processes and readers ----------------------------------------
def test_concurrent_writer_and_reader_processes_never_share_slots(tmp_path: Path) -> None:
    key = tmp_path / "kek"
    write_key_file(key)
    root = tmp_path / "store"
    MaskStore.open(root, FileKeyProvider(key), low_water_slots=0).close()
    writers = [CTX.Process(target=fleet_workers.write_loop,
                           args=(str(root), str(key), 4, 20, i, str(tmp_path / f"w{i}.txt"))) for i in range(3)]
    readers = [CTX.Process(target=fleet_workers.reserve_until,
                           args=(str(root), str(key), 70, 100 + i, str(tmp_path / f"r{i}.txt"), 90.0))
               for i in range(3)]
    for p in writers + readers:
        p.start()
    for p in writers + readers:
        p.join(180)
        assert p.exitcode == 0, p
    seg_ids = [ln for i in range(3) for ln in (tmp_path / f"w{i}.txt").read_text().split()]
    assert len(seg_ids) == len(set(seg_ids)) == 12
    slots: Counter[tuple[str, int]] = Counter()
    for i in range(3):
        for ln in (tmp_path / f"r{i}.txt").read_text().splitlines():
            sid, start, count = ln.split()
            assert sid in seg_ids
            for s in range(int(start), int(start) + int(count)):
                slots[(sid, s)] += 1
    assert slots and max(slots.values()) == 1  # no slot ever handed out twice
    assert all(0 <= s < 20 for _, s in slots)
    st = MaskStore.open(root, FileKeyProvider(key), low_water_slots=0)
    try:
        _assert_clean(st.verify_integrity())
        assert st.available_slots("ab" * 32, RING_K40_A12) == 240 - sum(slots.values())
    finally:
        st.close()


def test_partial_cleanup_spares_live_writers_and_removes_dead_ones(tmp_path: Path) -> None:
    key = tmp_path / "kek"
    write_key_file(key)
    root = tmp_path / "store"
    st = MaskStore.open(root, FileKeyProvider(key), low_water_slots=0)
    ready, go = tmp_path / "ready", tmp_path / "go"
    p = CTX.Process(target=fleet_workers.hold_partial, args=(str(root), str(key), str(ready), str(go)))
    p.start()
    deadline = time.monotonic() + 120
    while not ready.exists():
        assert time.monotonic() < deadline and p.is_alive()
        time.sleep(0.05)
    parts = list(root.glob("*/*/segments/.*.part"))
    assert len(parts) == 1 and partial_in_use(parts[0])
    # a restarting peer's crash clean-up and gc must leave the live writer's file alone
    assert st.cleanup_partials() == []
    st.gc()
    assert parts[0].exists()
    go.write_text("1")
    p.join(60)
    assert p.exitcode == 0
    seg = Path(str(go) + ".done").read_text()
    assert st.available_slots("ab" * 32, RING_K40_A12) == 12 and seg

    # a writer killed mid-segment: its partial file is unlocked and removed; nothing was registered
    ready.unlink()
    go.unlink()
    p = CTX.Process(target=fleet_workers.hold_partial, args=(str(root), str(key), str(ready), str(go)))
    p.start()
    while not ready.exists():
        assert time.monotonic() < deadline + 120 and p.is_alive()
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGKILL)
    p.join(30)
    parts = list(root.glob("*/*/segments/.*.part"))
    assert len(parts) == 1 and not partial_in_use(parts[0])
    assert len(st.cleanup_partials()) == 1
    assert st.available_slots("ab" * 32, RING_K40_A12) == 12
    _assert_clean(st.verify_integrity())
    st.close()


# ---- supervisor ------------------------------------------------------------------------------------
def test_worker_settings_assign_devices_services_and_state(tmp_path: Path) -> None:
    s = FactorySettings(pack_path=tmp_path, workers=3, devices=("cuda:0", "cuda:1"),
                        he_service_urls=("https://a", "https://b"), metrics_port=9100, state_dir=tmp_path)
    ws = [worker_settings(s, k, tmp_path / "fleet") for k in range(3)]
    assert [w.device for w in ws] == ["cuda:0", "cuda:1", "cuda:0"]
    assert [w.state_dir for w in ws] == [tmp_path / "fleet" / f"w{k}" for k in range(3)]
    assert [w.metrics_port for w in ws] == [9100, 9101, 9102]
    assert len({w.factory_id for w in ws}) == 3 and all(w.workers == 1 for w in ws)
    assert s.he_urls == ("https://a", "https://b")
    with pytest.raises(ValueError):
        FactorySettings(pack_path=tmp_path, workers=2, output="export", export_dir=tmp_path)


def _dummy_worker(settings_json: str, k: int, n: int, root: str, store_overrides: dict[str, Any]) -> None:
    if k == 1 and not (Path(root) / "w1-crashed").exists():
        (Path(root) / "w1-crashed").write_text("1")
        os._exit(3)
    (Path(root) / f"w{k}-alive").write_text(str(os.getpid()))
    time.sleep(60)


def test_fleet_supervisor_restarts_dead_workers(tmp_path: Path) -> None:
    s = FactorySettings(pack_path=tmp_path, workers=2, state_dir=tmp_path, error_backoff_s=0.1)
    fleet = FactoryFleet(s, target=_dummy_worker).start()
    try:
        deadline = time.monotonic() + 120
        while not (fleet.root / "w1-alive").exists():
            fleet.supervise_once()
            assert time.monotonic() < deadline
            time.sleep(0.1)
        st = fleet.status()
        assert st["processes"][1]["restarts"] == 1 and all(p["alive"] for p in st["processes"])
    finally:
        fleet.stop(timeout_s=10)
    assert all(not p["alive"] for p in fleet.status()["processes"])


def test_fleet_processes_fill_a_real_store_to_target(tmp_path: Path, tiny_pack_dir: Path) -> None:
    """``pgw factory run --workers 2`` end to end: two spawned workers on one store stop at target together."""
    import json

    from pgw.modelpack import ModelPack

    key = tmp_path / "kek"
    write_key_file(key)
    root = tmp_path / "store"
    st = MaskStore.open(root, FileKeyProvider(key), low_water_slots=0)
    s = FactorySettings(pack_path=tiny_pack_dir, rings=("K40A12",), device="cpu", segment_slots=64, chunk_slots=64,
                        low_water_slots=200, target_slots=256, poll_interval_s=0.5, workers=2, cpu_threads=1)
    fleet = FactoryFleet(s, store_overrides={"root": root, "key_file": key, "key_provider": "file"}).start()
    pfp = ModelPack.open(tiny_pack_dir, verify="none", components=("linears",)).fingerprint
    try:
        deadline = time.monotonic() + 240
        while st.available_slots(pfp, RING_K40_A12) < 256:
            fleet.supervise_once()
            assert time.monotonic() < deadline, fleet.status()
            time.sleep(0.2)
        time.sleep(1.5)  # a further poll round must not overshoot
    finally:
        fleet.stop(timeout_s=60)
    status = fleet.status()
    assert st.available_slots(pfp, RING_K40_A12) == 256, status
    done = [json.loads(p.read_text()) for p in sorted(fleet.root.glob("w*/factory-state.json"))]
    # exactly ceil(256 / 64) segments in total, whichever worker made them (a worker restarted under heavy
    # load is fine: its claims die with it)
    assert sum(int(t["segments"]) for d in done for t in d["totals"].values()) == 4, (status, done)
    _assert_clean(st.verify_integrity())
    st.close()
