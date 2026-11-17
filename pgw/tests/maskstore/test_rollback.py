"""Crash safety, rollback detection and state tampering (SPEC 3.3, 7.5, 7.6)."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from pgw.core import RING_K40_A12, ConfigError, MaskReuseError, MaskStoreError
from pgw.maskstore import (
    FileKeyProvider,
    FileMonotonicCounter,
    GenerationJournal,
    MaskStore,
    ProviderCounter,
    StaticKeyProvider,
    TpmNvCounter,
    write_key_file,
)

from . import workers
from .conftest import FP, ToyModel, write_segment

CTX = mp.get_context("spawn")


@pytest.fixture
def key_file(tmp_path: Path) -> Path:
    return write_key_file(tmp_path / "kek")


@pytest.fixture
def fkp(key_file: Path) -> FileKeyProvider:
    return FileKeyProvider(key_file)


def _snapshot(root: Path, dest: Path, names: tuple[str, ...] = ("state.sqlite",)) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in root.iterdir():
        if any(f.name.startswith(n) for n in names):
            shutil.copy2(f, dest / f.name)


def _restore(snap: Path, root: Path) -> None:
    for f in root.iterdir():
        if f.name.startswith("state.sqlite"):
            f.unlink()
    for f in snap.iterdir():
        shutil.copy2(f, root / f.name)


def _sql(root: Path, q: str, args: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(root / "state.sqlite", isolation_level=None)
    try:
        return conn.execute(q, args).fetchall()
    finally:
        conn.close()


def test_restoring_old_database_is_detected(tmp_path: Path, fkp: FileKeyProvider, model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, fkp, low_water_slots=0)
    write_segment(s, model, 20)
    s.reserve(FP, RING_K40_A12, 5).close()
    s.close()
    _snapshot(root, tmp_path / "snap")
    s = MaskStore.open(root, fkp, low_water_slots=0)
    s.reserve(FP, RING_K40_A12, 5).close()  # slots 5..9 handed out after the snapshot
    s.close()
    _restore(tmp_path / "snap", root)
    with pytest.raises(MaskReuseError, match="rollback"):
        MaskStore.open(root, fkp, low_water_slots=0)


def test_restoring_database_and_journal_needs_a_counter(tmp_path: Path, fkp: FileKeyProvider, model: ToyModel) -> None:
    """A full snapshot of the store dir defeats the journal; an external counter catches it."""
    ctr = tmp_path / "other-volume" / "pgw.counter"
    for use_counter in (False, True):
        root = tmp_path / f"s{use_counter}"
        kw: dict[str, Any] = {"low_water_slots": 0}
        if use_counter:
            kw.update(counter="file", counter_file=ctr)
        s = MaskStore.open(root, fkp, **kw)
        write_segment(s, model, 20)
        s.close()
        _snapshot(root, tmp_path / f"snap{use_counter}", ("state.sqlite", "generation.journal"))
        s = MaskStore.open(root, fkp, **kw)
        s.reserve(FP, RING_K40_A12, 5).close()
        s.close()
        for f in (tmp_path / f"snap{use_counter}").iterdir():
            shutil.copy2(f, root / f.name)
        for f in root.iterdir():
            if f.name.startswith("state.sqlite-") and not (tmp_path / f"snap{use_counter}" / f.name).exists():
                f.unlink()
        if use_counter:
            with pytest.raises(MaskReuseError, match="monotonic counter"):
                MaskStore.open(root, fkp, **kw)
        else:  # documented limitation of the journal-only mode (SPEC 7.6)
            MaskStore.open(root, fkp, **kw).close()


def test_deleted_journal_is_detected(tmp_path: Path, fkp: FileKeyProvider, model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, fkp, low_water_slots=0)
    write_segment(s, model, 4)
    s.close()
    (root / "generation.journal").unlink()
    with pytest.raises(MaskReuseError, match="journal is missing"):
        MaskStore.open(root, fkp)


def test_forged_or_foreign_journal_is_rejected(tmp_path: Path, fkp: FileKeyProvider, model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, fkp, low_water_slots=0)
    write_segment(s, model, 4)
    s.reserve(FP, RING_K40_A12, 1).close()
    s.reserve(FP, RING_K40_A12, 1).close()
    s.close()
    j = root / "generation.journal"
    good = j.read_text()
    lines = good.splitlines()
    parts = lines[-1].split()
    parts[1] = "0"  # claim an older generation
    j.write_text("\n".join([*lines[:-1], " ".join(parts)]) + "\n")
    with pytest.raises(MaskStoreError, match=r"authentication|backwards"):
        MaskStore.open(root, fkp)
    assert len(lines) == 3
    j.write_text("\n".join(lines[:1] + lines[2:]) + "\n")  # drop a record from the middle
    with pytest.raises(MaskStoreError, match="chain broken"):
        MaskStore.open(root, fkp)
    # a journal from another store (same KEK) does not verify: records are bound to the store id
    other = MaskStore.open(tmp_path / "o", fkp, low_water_slots=0)
    other.close()
    shutil.copy2(tmp_path / "o" / "generation.journal", j)
    with pytest.raises(MaskStoreError, match="authentication"):
        MaskStore.open(root, fkp)
    j.write_text(good)
    MaskStore.open(root, fkp).close()


def test_torn_journal_write_is_tolerated(tmp_path: Path, fkp: FileKeyProvider, model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, fkp, low_water_slots=0)
    write_segment(s, model, 6)
    s.reserve(FP, RING_K40_A12, 2).close()
    s.close()
    with (root / "generation.journal").open("a") as f:
        f.write("g 99 99 deadbe")  # crash in the middle of an append
    s = MaskStore.open(root, fkp, low_water_slots=0)
    s.reserve(FP, RING_K40_A12, 2).close()
    s.close()
    s = MaskStore.open(root, fkp, low_water_slots=0)  # chain still verifies after repair
    assert s.available_slots(FP, RING_K40_A12) == 2
    s.close()


def test_row_tampering_is_detected(tmp_path: Path, fkp: FileKeyProvider, model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, fkp, low_water_slots=0)
    sid = write_segment(s, model, 10)
    s.reserve(FP, RING_K40_A12, 4).close()
    s.close()
    old_row = _sql(root, "SELECT next_slot, status, updated_gen, mac FROM segments WHERE segment_id=?", (sid,))[0]
    # 1) rewind the cursor directly: MAC fails
    _sql(root, "UPDATE segments SET next_slot=0 WHERE segment_id=?", (sid,))
    with pytest.raises(MaskReuseError, match="authentication"):
        MaskStore.open(root, fkp)
    _sql(root, "UPDATE segments SET next_slot=? WHERE segment_id=?", (old_row[0], sid))
    MaskStore.open(root, fkp, low_water_slots=0).close()
    # 2) replay an older, validly MAC'd row after more consumption: the cursor-sum check fails
    s = MaskStore.open(root, fkp, low_water_slots=0)
    s.reserve(FP, RING_K40_A12, 3).close()
    s.close()
    _sql(root, "UPDATE segments SET next_slot=?, status=?, updated_gen=?, mac=? WHERE segment_id=?", (*old_row, sid))
    with pytest.raises(MaskReuseError, match="cursors were modified"):
        MaskStore.open(root, fkp)


def test_deleting_a_row_or_editing_meta_is_detected(tmp_path: Path, fkp: FileKeyProvider, model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, fkp, low_water_slots=0)
    sid = write_segment(s, model, 10)
    write_segment(s, model, 10)
    s.reserve(FP, RING_K40_A12, 4).close()
    s.close()
    snap = tmp_path / "snap"
    _snapshot(root, snap)
    _sql(root, "DELETE FROM segments WHERE segment_id=?", (sid,))
    with pytest.raises(MaskReuseError):
        MaskStore.open(root, fkp)
    _restore(snap, root)
    _sql(root, "UPDATE meta SET value='0' WHERE key='consumed_total'")
    with pytest.raises(MaskReuseError, match="meta"):
        MaskStore.open(root, fkp)


def test_tampering_while_running_is_caught_on_next_reserve(store: MaskStore, model: ToyModel) -> None:
    sid = write_segment(store, model, 10)
    store.reserve(FP, RING_K40_A12, 4).close()
    _sql(store.root, "UPDATE segments SET next_slot=1 WHERE segment_id=?", (sid,))
    with pytest.raises(MaskReuseError):
        store.reserve(FP, RING_K40_A12, 1)
    with pytest.raises(MaskReuseError):  # stays refused (tainted)
        store.reserve(FP, RING_K40_A12, 1)


def test_rebind_witnesses_is_explicit(tmp_path: Path, fkp: FileKeyProvider, model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, fkp, low_water_slots=0)
    write_segment(s, model, 8)
    s.close()
    (root / "generation.journal").unlink()
    s = MaskStore.open(root, fkp, low_water_slots=0, verify_on_open=False)
    with pytest.raises(MaskStoreError, match="verify_integrity"):
        s.reserve(FP, RING_K40_A12, 1)
    with pytest.raises(MaskReuseError):
        s.verify_integrity()
    with pytest.raises(ConfigError):
        s.rebind_witnesses()
    s.rebind_witnesses(i_understand_this_can_hide_a_rollback=True)
    s.verify_integrity()
    s.reserve(FP, RING_K40_A12, 1).close()
    s.close()


@pytest.mark.parametrize("point", ["after_commit", "after_witness"])
def test_crash_mid_reserve_never_reuses(tmp_path: Path, key_file: Path, model: ToyModel, point: str) -> None:
    root = tmp_path / "s"
    ctr = tmp_path / "ctr"
    kw = {"low_water_slots": 0, "counter": "file", "counter_file": str(ctr)}
    s = MaskStore.open(root, FileKeyProvider(key_file), **kw)
    write_segment(s, model, 10)
    s.close()
    p = CTX.Process(target=workers.crash_during_reserve, args=(str(root), str(key_file), 4, point, kw))
    p.start()
    p.join(60)
    assert p.exitcode == 9
    s = MaskStore.open(root, FileKeyProvider(key_file), verify_on_open=False, **kw)
    rep = s.verify_integrity()
    assert rep.generation == 1 and rep.consumed_total == 4
    if point == "after_commit":  # witnesses were one step behind: resynchronised, not an alarm
        assert set(rep.resynced_witnesses) == {"journal", "counter:file"}
    else:
        assert rep.resynced_witnesses == []
    # the crashed reservation's slots are gone for good
    assert s.available_slots(FP, RING_K40_A12) == 6
    with s.reserve(FP, RING_K40_A12, 6) as res:
        assert res.extents[0].start == 4
    s.close()
    head = GenerationJournal(root / "generation.journal", "x", lambda b: b).path.read_text().splitlines()[-1]
    assert head.split()[1] == "2"
    assert FileMonotonicCounter(ctr).read() == 2


def test_dead_process_lease_is_released(tmp_path: Path, key_file: Path, model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0, shred_consumed=False)
    sid = write_segment(s, model, 4)
    ready = tmp_path / "ready"
    p = CTX.Process(target=workers.crash_holding_reservation, args=(str(root), str(key_file), 4, str(ready)))
    p.start()
    p.join(60)
    assert p.exitcode == 7 and ready.exists()
    seg = next(root.rglob(f"{sid}.pgwm"))
    size = seg.stat().st_size
    assert s.gc() == size  # the dead worker's lease does not block shredding
    assert not seg.exists()
    assert _sql(root, "SELECT COUNT(*) FROM reservations WHERE closed_at IS NULL") == [(0,)]
    assert s.available_slots(FP, RING_K40_A12) == 0
    s.close()


def test_missing_and_corrupt_segment_files_are_written_off(tmp_path: Path, fkp: FileKeyProvider,
                                                           model: ToyModel) -> None:
    root = tmp_path / "s"
    s = MaskStore.open(root, fkp, low_water_slots=0)
    a = write_segment(s, model, 5)
    b = write_segment(s, model, 6)
    c = write_segment(s, model, 7)
    s.close()
    next(root.rglob(f"{a}.pgwm")).unlink()
    pb = next(root.rglob(f"{b}.pgwm"))
    pb.write_bytes(pb.read_bytes()[:-10])  # truncated
    orphan = root / FP / "K40A12" / "segments" / ("f" * 32 + ".pgwm")
    orphan.write_bytes(b"junk")
    part = root / FP / "K40A12" / "segments" / ".old.pgwm.part"
    part.write_bytes(b"partial")
    os.utime(part, (time.time() - 7200, time.time() - 7200))
    s = MaskStore.open(root, fkp, low_water_slots=0, verify_on_open=False)
    rep = s.verify_integrity()
    assert rep.missing_segments == [a] and rep.quarantined_segments == [b]
    assert rep.orphan_files == [str(orphan.relative_to(root))]
    assert not part.exists()
    assert s.available_slots(FP, RING_K40_A12) == 7
    with s.reserve(FP, RING_K40_A12, 7) as res:
        assert res.extents[0].segment_id == c
    s.close()
    MaskStore.open(root, fkp).close()  # consistent afterwards


def test_counter_implementations(tmp_path: Path) -> None:
    c = FileMonotonicCounter(tmp_path / "c")
    assert c.read() == 0 and c.increment() == 1 and c.increment() == 2 and c.read() == 2
    (tmp_path / "c").write_bytes(b"garbage")
    with pytest.raises(MaskStoreError, match="corrupt"):
        c.read()

    nv = {"v": 41}

    def runner(cmd: list[str]) -> bytes:
        if cmd[0] == "tpm2_nvincrement":
            nv["v"] += 1
            return b""
        assert cmd[0] == "tpm2_nvread" and cmd[1] == hex(0x1500016)
        return nv["v"].to_bytes(8, "big")

    t = TpmNvCounter(0x1500016, runner=runner)
    assert t.hardware and t.read() == 41 and t.increment() == 42
    with pytest.raises(ConfigError):
        ProviderCounter(StaticKeyProvider(os.urandom(32)))


class _HsmLike(StaticKeyProvider):
    """Provider with a hardware counter (HSM / KMS-backed counter)."""

    def __init__(self, kek: bytes, box: dict[str, int]):
        super().__init__(kek)
        self.box = box

    def monotonic_counter(self) -> int:
        return self.box["v"]

    def increment_counter(self) -> int:
        self.box["v"] += 1
        return self.box["v"]


def test_provider_counter_detects_full_restore(tmp_path: Path, model: ToyModel) -> None:
    kek = os.urandom(32)
    box = {"v": 1000}
    root = tmp_path / "s"
    s = MaskStore.open(root, _HsmLike(kek, box), counter="provider", low_water_slots=0)
    write_segment(s, model, 10)
    s.close()
    snap = tmp_path / "snap"
    shutil.copytree(root, snap)
    s = MaskStore.open(root, _HsmLike(kek, box), counter="provider", low_water_slots=0)
    s.reserve(FP, RING_K40_A12, 3).close()
    assert s.status()["rollback_protection"]["hardware_counter"]
    s.close()
    shutil.rmtree(root)
    shutil.copytree(snap, root)  # full disk restore, journal included
    with pytest.raises(MaskReuseError, match="monotonic counter"):
        MaskStore.open(root, _HsmLike(kek, box), counter="provider", low_water_slots=0)


def test_counter_config_errors(tmp_path: Path, kp: StaticKeyProvider) -> None:
    with pytest.raises(ConfigError, match="counter_file"):
        MaskStore.open(tmp_path / "a", kp, counter="file")
    with pytest.raises(ConfigError, match="tpm_nv_index"):
        MaskStore.open(tmp_path / "b", kp, counter="tpm")
