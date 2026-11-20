"""FactoryScheduler: hysteresis planning, prioritisation, crash recovery, locking and the run loop."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from pgw.core import RING_K40_A12, ConfigError
from pgw.factory import FactoryScheduler, FactorySettings, FactoryState, MaskGenerator, RefillPlan
from pgw.factory.scheduler import STATE_FILE, FactoryLock

from .conftest import FakeMaskStore, FakePack


def _settings(tmp_path: Path, **kw: Any) -> FactorySettings:
    base: dict[str, Any] = {"pack_path": tmp_path / "pack", "segment_slots": 50, "chunk_slots": 16,
                            "low_water_slots": 100, "target_slots": 300, "poll_interval_s": 0.05,
                            "state_dir": tmp_path / "state", "error_backoff_s": 0.05}
    if kw.get("target_slots") is not None and "low_water_slots" not in kw:
        base["low_water_slots"] = min(100, kw["target_slots"])
    base.update(kw)
    return FactorySettings(**base)


def _sched(tmp_path: Path, store: Any, packs: list[FakePack], **kw: Any) -> FactoryScheduler:
    gens = {}
    for p in packs:
        g = MaskGenerator(p, RING_K40_A12)
        gens[g.key] = g
    return FactoryScheduler(_settings(tmp_path, **kw), store, gens, metrics=False)


def test_plan_hysteresis(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    sch = _sched(tmp_path, fake_store, [fake_pack])
    plans = sch.plan()
    assert len(plans) == 1
    p = plans[0]
    assert isinstance(p, RefillPlan)
    assert (p.segments, p.slots_per_segment, p.available, p.slots) == (6, 50, 0, 300)
    assert "low water" in p.reason
    ids = sch.run_once()
    assert len(ids) == 6 and sch.plan() == []  # at target
    fake_store.consume(fake_pack.fingerprint, 150)  # 150 left: above low water -> no refill
    assert sch.plan() == []
    fake_store.consume(fake_pack.fingerprint, 60)  # 90 < 100 -> refill to target
    p = sch.plan()[0]
    assert p.available == 90 and p.segments == 5
    sch.run_once()
    fake_store.consume(fake_pack.fingerprint, 0)
    assert sch.plan() == []
    sch.close()


def test_first_segment_of_a_new_store_is_small(tmp_path: Path, fake_store: FakeMaskStore,
                                               fake_pack: FakePack) -> None:
    sch = _sched(tmp_path, fake_store, [fake_pack], first_segment_slots=20)
    p = sch.plan()[0]
    assert (p.segments, p.slots_per_segment, p.available) == (1, 20, 0) and "first segment" in p.reason
    assert len(sch.run_once()) == 1 and [s.slots for s in fake_store.segments.values()] == [20]
    p = sch.plan()[0]  # then full segments up to the target
    assert (p.segments, p.slots_per_segment, p.available) == (6, 50, 20)
    sch.run_once()
    assert sch.plan() == [] and len(fake_store.segments) == 7
    fake_store.consume(fake_pack.fingerprint, 290)  # 30 left: below low water, but not empty -> full segments
    p = sch.plan()[0]
    assert p.slots_per_segment == 50
    sch.close()


def test_first_segment_slots_not_larger_than_segment_slots(tmp_path: Path, fake_store: FakeMaskStore,
                                                          fake_pack: FakePack) -> None:
    sch = _sched(tmp_path, fake_store, [fake_pack], first_segment_slots=500)
    assert sch.plan()[0].slots_per_segment == 50
    sch.close()


def test_refill_continues_until_target(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    sch = _sched(tmp_path, fake_store, [fake_pack], max_segments_per_run=2)
    assert sch.run_once() and len(fake_store.segments) == 2  # 100 available, still refilling
    p = sch.plan()[0]
    assert p.available == 100 and "refilling" in p.reason
    sch.run_once()
    sch.run_once()
    assert sch.plan() == [] and len(fake_store.segments) == 6
    sch.close()


def test_most_urgent_first(tmp_path: Path, fake_store: FakeMaskStore) -> None:
    a, b = FakePack(seed=1, name="a"), FakePack(seed=2, name="b")
    sch = _sched(tmp_path, fake_store, [a, b], low_water_slots=100, target_slots=100, segment_slots=50)
    g = sch.generators[(a.fingerprint, "K40A12")]
    g.generate_into_store(fake_store, 50)  # a has 50, b has 0
    plans = sch.plan()
    assert [p.fingerprint for p in plans] == [b.fingerprint, a.fingerprint]
    sch.run_once()
    order = [s.fingerprint for s in fake_store.segments.values()][1:]
    assert order[0] == b.fingerprint  # b served first, then interleaved by fill ratio
    assert sum(1 for f in order if f == b.fingerprint) == 2 and sum(1 for f in order if f == a.fingerprint) == 1
    sch.close()


def test_state_persisted_and_status(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    sch = _sched(tmp_path, fake_store, [fake_pack], target_slots=100)
    ids = sch.run_once()
    sch.close()
    data = json.loads((tmp_path / "state" / STATE_FILE).read_text())
    key = f"{fake_pack.fingerprint}/K40A12"
    assert data["totals"][key]["segments"] == len(ids) == 2
    assert data["totals"][key]["slots"] == 100 and data["in_progress"] is None
    assert data["last_segment_id"] == ids[-1]
    sch2 = _sched(tmp_path, fake_store, [fake_pack], target_slots=100)
    st = sch2.status()
    m = st["models"][0]
    assert m["available_slots"] == 100 and m["segments_generated"] == 2 and m["model"] == "fake"
    assert st["plan"] == [] and st["running_elsewhere"] is False
    assert "seed" not in json.dumps(st)
    sch2.close()


def test_crash_recovery(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    st = FactoryState(in_progress={"fingerprint": fake_pack.fingerprint, "ring": "K40A12", "pid": 1})
    st.save(tmp_path / "state" / STATE_FILE)
    sch = _sched(tmp_path, fake_store, [fake_pack], target_slots=50)
    assert sch.run_once()
    assert sch.state.crash_recoveries == 1 and sch.state.in_progress is None
    sch.close()


def test_corrupt_state_file_is_preserved(tmp_path: Path) -> None:
    p = tmp_path / STATE_FILE
    p.write_text("{not json")
    st = FactoryState.load(p)
    assert st.totals == {} and not p.exists()
    assert list(tmp_path.glob("factory-state.corrupt-*"))


def test_lock_excludes_second_factory(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    a = _sched(tmp_path, fake_store, [fake_pack], target_slots=50)
    b = _sched(tmp_path, fake_store, [fake_pack], target_slots=50)
    a.run_once()
    assert b.status()["running_elsewhere"] is True
    with pytest.raises(ConfigError, match="already running"):
        b.run_once()
    a.close()
    b.run_once()
    b.close()


def test_failing_model_skipped_others_continue(tmp_path: Path, fake_store: FakeMaskStore) -> None:
    good, bad = FakePack(seed=1), FakePack(seed=2)
    sch = _sched(tmp_path, fake_store, [good, bad], target_slots=100)

    def boom(*a: Any, **k: Any) -> str:
        raise RuntimeError("gpu fell off the bus")

    sch.generators[(bad.fingerprint, "K40A12")].generate_into_store = boom  # type: ignore[method-assign]
    ids = sch.run_once()
    assert len(ids) == 2
    assert "gpu fell off" in (sch.state.last_error or "") or sch.state.last_error is None
    sch.close()


def test_all_failing_raises(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    fake_store.fail_after = 0
    sch = _sched(tmp_path, fake_store, [fake_pack])
    with pytest.raises(OSError):
        sch.run_once()
    assert sch.state.in_progress is None and "disk full" in (sch.state.last_error or "")
    sch.close()


def test_run_forever_stops_and_wakes(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    sch = _sched(tmp_path, fake_store, [fake_pack], poll_interval_s=30.0, target_slots=100)
    stop = threading.Event()
    th = threading.Thread(target=sch.run_forever, args=(stop,), daemon=True)
    th.start()
    deadline = time.time() + 20
    while len(fake_store.segments) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert len(fake_store.segments) == 2
    fake_store.consume(fake_pack.fingerprint, 100)
    assert fake_store.listeners, "scheduler must subscribe to low-water notifications"
    fake_store.listeners[0](None)  # store notifies -> scheduler wakes long before the 30 s poll
    while len(fake_store.segments) < 4 and time.time() < deadline:
        time.sleep(0.01)
    assert len(fake_store.segments) == 4
    sch.stop()
    th.join(10)
    assert not th.is_alive() and not sch.lock.held


def test_run_forever_backs_off_on_errors(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    fake_store.fail_after = 0
    sch = _sched(tmp_path, fake_store, [fake_pack])
    stop = threading.Event()
    th = threading.Thread(target=sch.run_forever, args=(stop,), daemon=True)
    th.start()
    time.sleep(0.4)
    stop.set()
    sch._wake.set()
    th.join(10)
    assert not th.is_alive()
    assert fake_store.aborted >= 2  # retried after back-off


def test_stop_during_generation(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    from pgw.factory import FactoryStopped

    sch = _sched(tmp_path, fake_store, [fake_pack])
    sch.stop()
    with pytest.raises(FactoryStopped):
        sch.run_once()
    sch.close()


def test_levels_default_from_store(tmp_path: Path, fake_pack: FakePack) -> None:
    store = FakeMaskStore(low=10, target=20)
    s = FactorySettings(pack_path=tmp_path, state_dir=tmp_path / "st")
    sch = FactoryScheduler(s, store, {MaskGenerator(fake_pack, RING_K40_A12).key: MaskGenerator(fake_pack, RING_K40_A12)},
                           metrics=False)
    assert (sch.low_water_slots, sch.target_slots) == (10, 20)


def test_constructor_validation(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    s = _settings(tmp_path)
    with pytest.raises(ConfigError):
        FactoryScheduler(s, fake_store, {}, metrics=False)
    g = MaskGenerator(fake_pack, RING_K40_A12)
    with pytest.raises(ConfigError):
        FactoryScheduler(s, fake_store, {("ab" * 32, "K40A12"): g}, metrics=False)
    with pytest.raises(ConfigError):
        FactoryScheduler(_settings(tmp_path, low_water_slots=None, target_slots=None),
                         FakeMaskStore(low=50, target=10), {g.key: g}, metrics=False)


def test_settings_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FactorySettings(pack_path=tmp_path, output="export")
    with pytest.raises(ValueError):
        FactorySettings(pack_path=tmp_path, low_water_slots=10, target_slots=5)
    s = FactorySettings(pack_path=tmp_path, rings=("k48a20", "K48A20", "K40A12"), ring="k32a8")
    assert s.ring_names == ("K48A20", "K40A12") and s.ring == "K32A8"
    assert FactorySettings(pack_path=tmp_path).ring_names == ("K32A12C27",)


def test_settings_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_FACTORY_PACK_PATH", str(tmp_path))
    monkeypatch.setenv("PGW_FACTORY_MAX_SLOTS_PER_HOUR", "1000")
    monkeypatch.setenv("PGW_FACTORY_RINGS", '["K48A20"]')
    s = FactorySettings()
    assert s.max_slots_per_hour == 1000 and s.ring_names == ("K48A20",) and s.pack_path == tmp_path


def test_lock_release_idempotent(tmp_path: Path) -> None:
    lk = FactoryLock(tmp_path / "x.lock")
    lk.acquire()
    lk.acquire()
    assert lk.held
    lk.release()
    lk.release()
    assert not lk.held


def test_metrics_emitted(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    calls: list[tuple[str, dict[str, str], str, float]] = []

    class Coll:
        def __init__(self, name: str) -> None:
            self.name = name

        def labels(self, **kw: str) -> Any:
            outer = self

            class L:
                def inc(self, v: float = 1) -> None:
                    calls.append((outer.name, kw, "inc", v))

                def set(self, v: float) -> None:
                    calls.append((outer.name, kw, "set", v))

            return L()

    class M:
        def get(self, name: str) -> Coll:
            return Coll(name)

    g = MaskGenerator(fake_pack, RING_K40_A12)
    sch = FactoryScheduler(_settings(tmp_path, target_slots=50), fake_store, {g.key: g}, metrics=M())
    sch.run_once()
    sch.close()
    names = {c[0] for c in calls}
    assert {"pgw_factory_slots_generated_total", "pgw_factory_slots_per_second", "pgw_masks_available_slots"} <= names
    gen_total = [c for c in calls if c[0] == "pgw_factory_slots_generated_total"]
    assert gen_total[0][3] == 50 and len(gen_total[0][1]["fingerprint"]) == 12


@pytest.mark.parametrize("content", ["[1, 2]", '{"totals": [1]}', '{"in_progress": "x"}'])
def test_wrongly_typed_state_file_is_reset(tmp_path: Path, content: str) -> None:
    p = tmp_path / STATE_FILE
    p.write_text(content)
    st = FactoryState.load(p)
    assert st.totals == {} and st.in_progress is None and not p.exists()


def test_crash_recovery_cleans_partials_once(tmp_path: Path, fake_store: FakeMaskStore, fake_pack: FakePack) -> None:
    calls: list[int] = []
    fake_store.cleanup_partials = lambda: calls.append(1) or ["x.part"]  # type: ignore[attr-defined]
    sch = _sched(tmp_path, fake_store, [fake_pack], target_slots=50)
    sch.run_once()
    sch.close()
    assert calls == []  # clean start: a shared store's temp files may belong to a live writer
    FactoryState(in_progress={"fingerprint": fake_pack.fingerprint, "ring": "K40A12", "pid": 1}).save(
        tmp_path / "state" / STATE_FILE)
    sch = _sched(tmp_path, fake_store, [fake_pack], target_slots=50)
    sch.run_once()
    sch.run_once()
    sch.close()
    assert calls == [1]
