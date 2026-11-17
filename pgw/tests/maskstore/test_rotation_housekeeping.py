"""KEK rotation, interrupted rotations, partial-file cleanup and witness failures after commit."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

import pytest
import torch

from pgw.core import RING_K40_A12, ConfigError, MaskReuseError, MaskStoreError
from pgw.maskstore import (
    FileKeyProvider,
    GenerationJournal,
    MaskStore,
    SegmentFile,
    StaticKeyProvider,
    write_key_file,
)

from .conftest import FP, ToyModel, write_segment


def _drain_check(store: MaskStore, model: ToyModel, n: int) -> list[tuple[str, int]]:
    """Reserve ``n`` slots, check every row of every linear against the model, return the slots."""
    out: list[tuple[str, int]] = []
    with store.reserve(FP, RING_K40_A12, n) as res:
        out = [(e.segment_id, e.start + i) for e in res.extents for i in range(e.count)]
        for spec in model.specs:
            key = spec.id.key
            while res.remaining(key):
                k = res.rows_until_boundary(key)
                u, uw = res.checks(key)
                r, wr = res.take(key, k)
                assert torch.equal(wr, model.wr(key, r, RING_K40_A12))
                # Freivalds vectors consistent with this extent: u.(W r) == (u W).r
                assert torch.equal((u @ wr.t()) & RING_K40_A12.mask, (uw @ r.t()) & RING_K40_A12.mask)
    return out


def _kek_ids(root: Path) -> set[str]:
    ids = set()
    for p in root.rglob("*.pgwm"):
        with SegmentFile(p) as f:
            ids.add(f.header.kek_id)
    return ids


@pytest.mark.parametrize("reencrypt", [True, False])
def test_rotate_kek_preserves_masks_and_state(tmp_path: Path, model: ToyModel, reencrypt: bool) -> None:
    root = tmp_path / "s"
    old_file, new_file = write_key_file(tmp_path / "old"), write_key_file(tmp_path / "new")
    old = FileKeyProvider(old_file)
    s = MaskStore.open(root, old, low_water_slots=0)
    write_segment(s, model, 6)
    write_segment(s, model, 5)
    first = _drain_check(s, model, 3)  # one partly consumed segment
    gen = s.generation
    new = FileKeyProvider(new_file, previous=[] if reencrypt else [old_file])
    summary = s.rotate_kek(new, reencrypt=reencrypt)
    assert summary["new_kek_id"] == new.kek_id and summary["old_kek_id"] == old.kek_id
    assert summary["segments_reencrypted"] == (2 if reencrypt else 0)
    assert s.generation == gen
    # usable immediately in this process with the new key
    second = _drain_check(s, model, 4)
    s.close()
    assert _kek_ids(root) == ({new.kek_id} if reencrypt else {old.kek_id})
    # reopen with the new provider only: state authenticates, remaining masks are intact
    s = MaskStore.open(root, new, low_water_slots=0)
    assert s.available_slots(FP, RING_K40_A12) == 4
    third = _drain_check(s, model, 4)
    allslots = first + second + third
    assert len(allslots) == len(set(allslots)) == 11
    s.close()
    # the old key alone no longer authenticates the state
    with pytest.raises(MaskReuseError):
        MaskStore.open(root, old, low_water_slots=0)


def test_rotate_kek_refuses_open_reservations_and_weaker_providers(tmp_path: Path, model: ToyModel) -> None:
    s = MaskStore.open(tmp_path / "s", StaticKeyProvider(os.urandom(32)), low_water_slots=0)
    write_segment(s, model, 4)
    res = s.reserve(FP, RING_K40_A12, 1)
    with pytest.raises(MaskStoreError, match="reservations are open"):
        s.rotate_kek(StaticKeyProvider(os.urandom(32)))
    res.close()

    class NoMac(StaticKeyProvider):
        def mac(self, data: bytes) -> bytes | None:
            return None

    with pytest.raises(ConfigError):
        s.rotate_kek(NoMac(os.urandom(32)))
    s.rotate_kek(StaticKeyProvider(os.urandom(32)))
    s.close()


def test_interrupted_rotation_before_db_commit_is_discarded(tmp_path: Path, model: ToyModel,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "s"
    old = StaticKeyProvider(os.urandom(32))
    new = StaticKeyProvider(os.urandom(32))
    s = MaskStore.open(root, old, low_water_slots=0)
    write_segment(s, model, 4)
    real_write_meta = MaskStore._write_meta

    def boom(self: MaskStore, conn: sqlite3.Connection, m: object) -> None:
        real_write_meta(self, conn, m)  # type: ignore[arg-type]
        if self.key_provider is new:
            raise RuntimeError("power loss")

    monkeypatch.setattr(MaskStore, "_write_meta", boom)
    with pytest.raises(RuntimeError):
        s.rotate_kek(new)
    monkeypatch.undo()
    store_id, gen = s.store_id, s.generation
    s.close()
    # A real crash would have left the staged journal (valid under the NEW key) behind.
    GenerationJournal(root / "generation.journal", store_id, new.mac).stage_checkpoint(gen, 0, bytes(32))
    assert (root / "generation.journal.next").exists()
    # Segment files were already re-encrypted, but the DB still authenticates under the old key.
    assert _kek_ids(root) == {new.kek_id}
    s = MaskStore.open(root, StaticKeyProvider(old._kek, previous=[new._kek]), low_water_slots=0)
    assert not (root / "generation.journal.next").exists()  # stale staged journal discarded
    # resume the rotation: already re-encrypted files are skipped, then everything works
    summary = s.rotate_kek(new)
    assert summary["segments_reencrypted"] == 0
    s.close()
    s = MaskStore.open(root, new, low_water_slots=0)
    _drain_check(s, model, 4)
    s.close()


def test_interrupted_rotation_after_db_commit_is_completed(tmp_path: Path, model: ToyModel,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "s"
    old = StaticKeyProvider(os.urandom(32))
    new = StaticKeyProvider(os.urandom(32))
    s = MaskStore.open(root, old, low_water_slots=0)
    write_segment(s, model, 4)
    _drain_check(s, model, 1)
    def crash(self: GenerationJournal) -> None:
        raise RuntimeError("power loss before promotion")

    monkeypatch.setattr(GenerationJournal, "promote_pending", crash)
    with pytest.raises(RuntimeError):
        s.rotate_kek(new)
    monkeypatch.undo()
    s.close()
    assert (root / "generation.journal.next").exists()
    with pytest.raises(MaskReuseError):  # DB is under the new key now
        MaskStore.open(root, old, low_water_slots=0)
    assert (root / "generation.journal.next").exists()  # not discarded by the failed open
    s = MaskStore.open(root, new, low_water_slots=0)
    assert not (root / "generation.journal.next").exists()
    assert s.available_slots(FP, RING_K40_A12) == 3
    _drain_check(s, model, 3)
    s.close()


def test_cleanup_partials_and_gc_do_not_flag_live_segments(tmp_path: Path, model: ToyModel,
                                                           caplog: pytest.LogCaptureFixture) -> None:
    s = MaskStore.open(tmp_path / "s", StaticKeyProvider(os.urandom(32)), low_water_slots=0)
    sid = write_segment(s, model, 4)
    seg = next((tmp_path / "s").rglob(f"{sid}.pgwm"))
    part = seg.with_name(".deadbeef.pgwm.part")
    part.write_bytes(b"partial")
    fresh = seg.with_name(".cafe.pgwm.part")
    fresh.write_bytes(b"partial")
    old_t = part.stat().st_mtime - 7200
    os.utime(part, (old_t, old_t))
    with caplog.at_level(logging.WARNING, logger="pgw.maskstore"):
        assert s.gc() == 0
    assert "not registered" not in caplog.text  # live segments are not reported as orphans
    assert not part.exists() and fresh.exists()  # gc only removes old partials
    assert s.cleanup_partials() == [fresh.name]
    assert not fresh.exists() and seg.exists()
    _drain_check(s, model, 4)
    s.close()


def test_witness_failure_after_commit_loses_slots_but_never_reuses(tmp_path: Path, model: ToyModel) -> None:
    s = MaskStore.open(tmp_path / "s", StaticKeyProvider(os.urandom(32)), low_water_slots=0)
    write_segment(s, model, 6)

    def fail(point: str) -> None:
        if point == "after_commit":
            raise MaskStoreError("counter device unavailable")

    s._fault_hook = fail
    with pytest.raises(MaskStoreError, match="counter device"):
        s.reserve(FP, RING_K40_A12, 2)
    s._fault_hook = None
    assert not s._open_reservations
    st = s.status()
    assert st["open_reservations"] == 0 and st["consumed_total"] == 2
    assert s.metrics()["pgw_masks_burned_slots_total"][(FP, "K40A12")] == 2
    slots = _drain_check(s, model, 4)
    assert [x[1] for x in slots] == [2, 3, 4, 5]
    rep = s.verify_integrity()
    assert rep.consumed_total == 6
    s.close()


def test_admin_rotate_store_kek(tmp_path: Path, model: ToyModel) -> None:
    from pgw.maskstore import MaskStoreSettings
    from pgw.maskstore.admin import init_store, open_store, rotate_store_kek

    kf, nk = tmp_path / "kek", tmp_path / "kek.new"
    init_store(tmp_path / "s", kf, low_water_slots=0)
    settings = MaskStoreSettings(root=tmp_path / "s", key_file=kf, low_water_slots=0)
    with open_store(settings) as s:
        write_segment(s, model, 3)
    out = rotate_store_kek(nk, settings)
    assert nk.exists() and out["segments_reencrypted"] == 1
    with open_store(MaskStoreSettings(root=tmp_path / "s", key_file=nk, low_water_slots=0)) as s:
        _drain_check(s, model, 3)
    with pytest.raises(ConfigError):
        rotate_store_kek(nk, MaskStoreSettings(root=tmp_path / "s", key_provider="env"))
