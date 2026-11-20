"""Mask store: write/reserve/take round trips, extents, exhaustion, burning, shredding, inventory."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    ConfigError,
    MaskExhaustedError,
    MaskReuseError,
    MaskStoreError,
    RingParams,
    ShapeError,
    center,
    freivalds_check,
    mask_activations,
)
from pgw.maskstore import Inventory, MaskStore, SegmentFile

from .conftest import FP, FP2, ToyModel, lid, make_model, write_segment


def _verify_rows(model: ToyModel, ring: RingParams, key: str, r: torch.Tensor, wr: torch.Tensor,
                 u: torch.Tensor, uw: torch.Tensor) -> None:
    assert r.dtype == torch.int64 and wr.dtype == torch.int64
    assert torch.equal(wr, model.wr(key, r, ring))
    # protocol round trip: mask -> "service" -> unmask is exact, and Freivalds accepts
    w = model.weights[key]
    xq = torch.randint(-ring.qmax, ring.qmax + 1, (r.shape[0], r.shape[1]), dtype=torch.int64)
    x_hat = mask_activations(xq, r, ring)
    y_hat = (x_hat @ w.t()) & ring.mask
    assert bool(freivalds_check(x_hat, y_hat, u, uw, ring).all())
    assert torch.equal(center((y_hat - wr) & ring.mask, ring), xq @ w.t())
    bad = y_hat.clone()
    bad[0, 0] = (bad[0, 0] + 1) & ring.mask
    assert not bool(freivalds_check(x_hat, bad, u, uw, ring)[0])


@pytest.mark.parametrize("ring", [RING_K32_A8, RING_K40_A12, RING_K48_A20], ids=lambda r: r.name)
@pytest.mark.parametrize("order", ["linear", "slot"])
def test_round_trip_exact(store: MaskStore, model: ToyModel, ring: RingParams, order: str) -> None:
    write_segment(store, model, 11, ring=ring, chunk_slots=4, write_rows=3, order=order)
    with store.reserve(FP, ring, 11) as res:
        assert res.slots == 11 and len(res.extents) == 1
        for s in model.specs:
            key = s.id.key
            u, uw = res.checks(s.id)
            assert u.shape == (2, s.out_features) and uw.shape == (2, s.in_features)
            got_r, got_wr = [], []
            for n in (1, 4, 6):  # crosses chunk boundaries of size 4
                r, wr = res.take(s.id, n)
                assert r.shape == (n, s.in_features) and wr.shape == (n, s.out_features)
                got_r.append(r)
                got_wr.append(wr)
            _verify_rows(model, ring, key, torch.cat(got_r), torch.cat(got_wr), u, uw)
            assert res.remaining(s.id) == 0


def test_take_accepts_string_ids_and_rows_are_distinct(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 8)
    with store.reserve(FP, RING_K40_A12, 8) as res:
        r_all, _ = res.take("0.qkv", 8)
        assert res.taken(lid("0.qkv")) == 8 and res.remaining("0.o") == 8
        assert res.linear_keys == tuple(s.id.key for s in model.specs)
    # r rows are distinct (uniform 40-bit vectors never collide)
    assert len({tuple(row.tolist()) for row in r_all}) == 8


def test_reservation_spans_segments_and_extents(store: MaskStore, model: ToyModel) -> None:
    s1 = write_segment(store, model, 5)
    s2 = write_segment(store, model, 7)
    s3 = write_segment(store, model, 3)
    with store.reserve(FP, RING_K40_A12, 3) as a:
        assert [(e.segment_id, e.start, e.count) for e in a.extents] == [(s1, 0, 3)]
    with store.reserve(FP, RING_K40_A12, 10) as b:
        assert [(e.segment_id, e.start, e.count) for e in b.extents] == [(s1, 3, 2), (s2, 0, 7), (s3, 0, 1)]
        q = lid("1.down")
        assert b.rows_until_boundary(q) == 2
        u1, uw1 = b.checks(q)
        with pytest.raises(ConfigError, match="boundary"):
            b.take(q, 3)
        r, wr = b.take(q, 2)
        _verify_rows(model, RING_K40_A12, q.key, r, wr, u1, uw1)
        assert b.rows_until_boundary(q) == 7
        u2, uw2 = b.checks(q)
        assert not torch.equal(u1, u2)  # different segment, different check vectors
        r, wr = b.take(q, 7)
        _verify_rows(model, RING_K40_A12, q.key, r, wr, u2, uw2)
        u3, uw3 = b.checks(q)
        r, wr = b.take(q, 1)
        _verify_rows(model, RING_K40_A12, q.key, r, wr, u3, uw3)
        assert b.rows_until_boundary(q) == 0
        with pytest.raises(MaskExhaustedError):
            b.take(q, 1)
        # after exhaustion, checks() refers to the last extent
        assert torch.equal(b.checks(q)[0], u3)
    assert store.available_slots(FP, RING_K40_A12) == 2


def test_exhaustion_is_atomic(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 4)
    with pytest.raises(MaskExhaustedError) as ei:
        store.reserve(FP, RING_K40_A12, 5)
    assert ei.value.details["available"] == 4 and ei.value.details["needed"] == 5
    assert store.available_slots(FP, RING_K40_A12) == 4  # nothing consumed
    gen = store.generation
    with pytest.raises(MaskExhaustedError):
        store.reserve(FP2, RING_K40_A12, 1)  # unknown model
    with pytest.raises(MaskExhaustedError):
        store.reserve(FP, RING_K48_A20, 1)  # other ring
    assert store.generation == gen
    store.reserve(FP, RING_K40_A12, 4).close()
    with pytest.raises(MaskExhaustedError):
        store.reserve(FP, RING_K40_A12, 1)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_reserve_rejects_bad_counts(store: MaskStore, bad: Any) -> None:
    with pytest.raises(ConfigError):
        store.reserve(FP, RING_K40_A12, bad)


def test_take_validation_and_closed_reservation(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 6)
    res = store.reserve(FP, RING_K40_A12, 4)
    with pytest.raises(ConfigError):
        res.take("0.qkv", 0)
    with pytest.raises(ConfigError):
        res.take("7.qkv", 1)
    with pytest.raises(MaskExhaustedError):
        res.take("0.qkv", 5)
    assert res.remaining("0.qkv") == 4  # failed calls consume nothing
    res.take("0.qkv", 2)
    res.close()
    res.close()  # idempotent
    assert res.closed
    with pytest.raises(MaskReuseError):
        res.take("0.qkv", 1)
    with pytest.raises(MaskReuseError):
        res.checks("0.qkv")


def test_close_burns_unused_rows_and_counts_them(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 10)
    with store.reserve(FP, RING_K40_A12, 6) as res:
        for s in model.specs:
            res.take(s.id, 2)
    met = store.metrics()
    assert met["pgw_masks_consumed_slots_total"][(FP, "K40A12")] == 6
    assert met["pgw_masks_burned_slots_total"][(FP, "K40A12")] == 4
    assert met["pgw_masks_available_slots"][(FP, "K40A12")] == 4
    # burned slots never come back
    with store.reserve(FP, RING_K40_A12, 4) as res2:
        assert res2.extents[0].start == 6


def test_checks_zero_segments(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 3, checks=0)
    with store.reserve(FP, RING_K40_A12, 3) as res:
        with pytest.raises(ConfigError, match="checks=0"):
            res.checks("0.qkv")
        r, wr = res.take("0.o", 3)
        assert torch.equal(wr, model.wr("0.o", r, RING_K40_A12))


def test_inventory_and_status(store: MaskStore, model: ToyModel) -> None:
    other = make_model(FP2, layers=1, seed=3)
    write_segment(store, model, 5)
    write_segment(store, model, 6)
    write_segment(store, other, 4, ring=RING_K48_A20)
    inv = store.inventory()
    assert [(i.fingerprint, i.ring, i.segments, i.slots_total, i.slots_available) for i in inv] == [
        (FP, "K40A12", 2, 11, 11), (FP2, "K48A20", 1, 4, 4),
    ]
    assert all(i.bytes_on_disk > 0 for i in inv)
    assert store.inventory(FP2, "K48A20")[0].slots_available == 4
    assert store.inventory(FP2, RING_K40_A12) == [Inventory(FP2, "K40A12", 0, 0, 0, 0)]
    st = store.status()
    assert st["segments_by_status"] == {"active": 3}
    assert st["rollback_protection"]["journal"] and st["rollback_protection"]["row_macs"]
    assert "kek" not in str(st).lower().replace("kek_id", "")  # no key material


def test_low_water_listener_fires_once_per_crossing(open_store: Any, model: ToyModel) -> None:
    store = open_store(low_water_slots=5)
    seen: list[Inventory] = []
    store.add_low_water_listener(seen.append)
    store.add_low_water_listener(lambda inv: 1 / 0)  # a failing listener must not break the store
    write_segment(store, model, 10)
    assert seen == []
    store.reserve(FP, RING_K40_A12, 4).close()  # 6 left
    assert seen == []
    store.reserve(FP, RING_K40_A12, 2).close()  # 4 left -> crossing
    assert len(seen) == 1 and seen[0].slots_available == 4
    store.reserve(FP, RING_K40_A12, 1).close()  # still below: no new event
    assert len(seen) == 1
    write_segment(store, model, 10)  # refill above
    store.reserve(FP, RING_K40_A12, 9).close()  # 13 - 9 = 4 -> second crossing
    assert len(seen) == 2
    assert store.metrics()["pgw_masks_low_water"][(FP, "K40A12")] == 1.0
    with pytest.raises(MaskExhaustedError):
        store.reserve(FP2, RING_K40_A12, 1)
    assert seen[-1].fingerprint == FP2  # exhaustion of an unknown model also alerts


def test_gc_shreds_consumed_segments_but_not_leased_ones(open_store: Any, model: ToyModel) -> None:
    store = open_store(shred_consumed=False)
    s1 = write_segment(store, model, 4)
    write_segment(store, model, 4)
    p1 = next(store.root.rglob(f"{s1}.pgwm"))
    res = store.reserve(FP, RING_K40_A12, 4)  # consumes s1 fully, still open
    assert store.gc() == 0 and p1.exists()  # leased by an open reservation
    res.take("0.qkv", 4)
    res.close()
    size = p1.stat().st_size
    assert store.gc() == size
    assert not p1.exists()
    assert store.status()["segments_by_status"] == {"active": 1, "shredded": 1}
    assert store.gc() == 0


def test_auto_shred_on_close_and_open_reader_survives(store: MaskStore, model: ToyModel) -> None:
    s1 = write_segment(store, model, 4)
    p1 = next(store.root.rglob(f"{s1}.pgwm"))
    a = store.reserve(FP, RING_K40_A12, 2)
    b = store.reserve(FP, RING_K40_A12, 2)  # segment now fully consumed, two leases
    a.close()
    assert p1.exists()  # b still holds it
    r, wr = b.take("1.gate_up", 2)
    assert torch.equal(wr, ToyModel.wr(make_model(), "1.gate_up", r, RING_K40_A12))
    b.close()
    assert not p1.exists()


def test_writer_validation(store: MaskStore, model: ToyModel) -> None:
    ring = RING_K40_A12
    spec = model.specs[0]
    good_uw = torch.zeros((2, spec.in_features), dtype=torch.int64)
    with pytest.raises(ConfigError):
        store.writer("nothex", ring, model.specs, slots=4)
    with pytest.raises(ConfigError):
        store.writer(FP, ring, model.specs, slots=0)
    with pytest.raises(ConfigError):
        store.writer(FP, ring, model.specs, slots=4, seed=b"short")
    with pytest.raises(ConfigError):
        store.writer(FP, ring, [], slots=4)
    w = store.writer(FP, ring, model.specs, slots=4, chunk_slots=2)
    with pytest.raises(ShapeError):
        w.write_linear(spec.id, torch.zeros((2, spec.out_features + 1), dtype=torch.int64), good_uw)
    with pytest.raises(ShapeError):
        w.write_linear(spec.id, torch.zeros((2, spec.out_features), dtype=torch.int32), good_uw)
    with pytest.raises(ShapeError):
        w.write_linear(spec.id, torch.full((2, spec.out_features), 1 << 41, dtype=torch.int64), good_uw)
    with pytest.raises(ShapeError):
        w.write_linear(spec.id, torch.zeros((2, spec.out_features), dtype=torch.int64), good_uw[:1])
    with pytest.raises(ConfigError):
        w.write_linear("9.o", torch.zeros((2, 16), dtype=torch.int64), good_uw)
    w.write_linear(spec.id, torch.zeros((3, spec.out_features), dtype=torch.int64), good_uw)
    with pytest.raises(MaskStoreError, match="changed"):
        w.write_linear(spec.id, torch.zeros((1, spec.out_features), dtype=torch.int64), good_uw + 1)
    with pytest.raises(ShapeError, match="exceed"):
        w.write_linear(spec.id, torch.zeros((2, spec.out_features), dtype=torch.int64), good_uw)
    with pytest.raises(MaskStoreError, match="incomplete"):
        w.commit()
    with pytest.raises(MaskStoreError):
        w.write_linear(spec.id, torch.zeros((1, spec.out_features), dtype=torch.int64), good_uw)
    seg_dir = store.root / FP / "K40A12" / "segments"
    assert not any(seg_dir.iterdir())  # aborted: temp file removed
    assert store.inventory() == []


def test_writer_abort_via_context_manager(store: MaskStore, model: ToyModel) -> None:
    with pytest.raises(RuntimeError), store.writer(FP, RING_K40_A12, model.specs, slots=4) as w:
        spec = model.specs[0]
        w.write_linear(spec.id, torch.zeros((4, spec.out_features), dtype=torch.int64),
                       torch.zeros((2, spec.in_features), dtype=torch.int64))
        raise RuntimeError("factory crashed")
    assert store.inventory() == []
    assert not list(store.root.rglob("*.pgwm*"))


def test_duplicate_seed_is_refused_forever(store: MaskStore, model: ToyModel) -> None:
    seed = os.urandom(32)
    write_segment(store, model, 2, seed=seed)
    with pytest.raises(MaskReuseError):
        write_segment(store, model, 2, seed=seed)
    store.reserve(FP, RING_K40_A12, 2).close()
    store.gc()  # shredded, but the seed stays blacklisted
    with pytest.raises(MaskReuseError):
        write_segment(store, model, 2, seed=seed)


def test_segment_files_are_encrypted(store: MaskStore, model: ToyModel) -> None:
    seed = os.urandom(32)
    sid = write_segment(store, model, 4, seed=seed)
    path = next(store.root.rglob(f"{sid}.pgwm"))
    data = path.read_bytes()
    assert seed not in data
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    with SegmentFile(path) as seg:
        assert seg.header.segment_id == sid and seg.header.factory_id == "test"
        with pytest.raises(MaskStoreError, match="data key"):
            seg.decrypt(seg.layout.secrets_chunk())


def test_tampered_chunk_is_detected_on_take(store: MaskStore, model: ToyModel) -> None:
    sid = write_segment(store, model, 8, chunk_slots=4)
    path = next(store.root.rglob(f"{sid}.pgwm"))
    with SegmentFile(path) as seg:
        ref = seg.layout.wr_chunk(1, 1)  # linear 0.o, slots 4..7
    raw = bytearray(path.read_bytes())
    raw[ref.offset + 3] ^= 0x01
    path.write_bytes(bytes(raw))
    with store.reserve(FP, RING_K40_A12, 8) as res:
        res.take("0.o", 4)  # first chunk is fine
        with pytest.raises(MaskStoreError, match="authentication"):
            res.take("0.o", 1)
        assert res.remaining("0.o") == 3  # the failed row stays consumed


def test_writes_with_large_chunk_and_single_slot(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 1, chunk_slots=256, write_rows=1)
    with store.reserve(FP, RING_K40_A12, 1) as res:
        u, uw = res.checks("1.qkv")
        r, wr = res.take("1.qkv", 1)
        _verify_rows(model, RING_K40_A12, "1.qkv", r, wr, u, uw)


def test_reopen_preserves_state(open_store: Any, tmp_path: Path, model: ToyModel) -> None:
    s = open_store()
    write_segment(s, model, 10)
    s.reserve(FP, RING_K40_A12, 3).close()
    gen = s.generation
    s.close()
    s2 = open_store()
    assert s2.generation == gen
    assert s2.available_slots(FP, RING_K40_A12) == 7
    with s2.reserve(FP, RING_K40_A12, 2) as res:
        assert res.extents[0].start == 3


def test_closed_store_rejects_use(open_store: Any, model: ToyModel) -> None:
    s = open_store()
    write_segment(s, model, 4)
    res = s.reserve(FP, RING_K40_A12, 2)
    s.close()
    assert res.closed  # store close burns open reservations
    with pytest.raises(MaskStoreError):
        s.reserve(FP, RING_K40_A12, 1)


def test_state_db_records_reservations(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 6)
    with store.reserve(FP, RING_K40_A12, 4, session_id="sess-1") as res:
        rid = res.id
    conn = sqlite3.connect(store.root / "state.sqlite")
    row = conn.execute("SELECT session_id, slots, closed_at IS NOT NULL, burned FROM reservations WHERE id=?",
                       (rid,)).fetchone()
    conn.close()
    assert row == ("sess-1", 4, 1, 4)


@pytest.mark.parametrize("ring", [RING_K32_A8, RING_K40_A12], ids=lambda r: r.name)
def test_take_checked_and_compact_checks(store: MaskStore, model: ToyModel, ring: RingParams) -> None:
    s1 = write_segment(store, model, 5, ring=ring)
    s2 = write_segment(store, model, 7, ring=ring)
    q = lid("1.down")
    a = store.reserve(FP, ring, 3)  # s1[0:3]
    b = store.reserve(FP, ring, 6)  # s1[3:5] + s2[0:4]
    try:
        ta, ua, uwa = a.checks_compact(q)
        tb, ub, uwb = b.checks_compact(q)
        assert ta == tb == (s1, q.key)  # same segment and linear: the same vectors, for any reservation
        width = np.uint32 if ring.k <= 32 else np.uint64
        assert ua.dtype == width and not ua.flags.writeable and not uwa.flags.writeable
        assert np.array_equal(ua, ub) and np.array_equal(uwa, uwb)
        u1, uw1 = b.checks(q)
        assert np.array_equal(ub.astype(np.int64), u1.numpy()) and np.array_equal(uwb.astype(np.int64), uw1.numpy())

        rows, r, wr, chk = b.take_checked(q, 5, True)  # stops at the extent boundary
        assert rows == 2 and chk is not None and chk[0] == tb
        _verify_rows(model, ring, q.key, r, wr.to(torch.int64), u1, uw1)
        u2, uw2 = b.checks(q)
        rows, r, wr, chk = b.take_checked(q, 5, True)
        assert rows == 4 and chk is not None and chk[0] == (s2, q.key)
        assert np.array_equal(chk[1].astype(np.int64), u2.numpy())
        _verify_rows(model, ring, q.key, r, wr.to(torch.int64), u2, uw2)
        assert b.take_checked(lid("0.qkv"), 1, False)[3] is None
        with pytest.raises(MaskExhaustedError):
            b.take_checked(q, 1, True)
        with pytest.raises(ConfigError):
            b.take_checked(q, 0, True)
        # rows taken through take_checked are gone for take() too, and vice versa
        assert b.remaining(q) == 0 and b.remaining(lid("0.qkv")) == 5
        b.take(lid("0.qkv"), 1)
        assert b.take_checked(lid("0.qkv"), 9, False)[0] == 4
    finally:
        a.close()
        b.close()
    assert not ub.any() and not uwb.any()  # wiped on close (views handed out earlier see zeros)
    with pytest.raises(MaskReuseError):
        b.take_checked(q, 1, True)
    with pytest.raises(MaskReuseError):
        b.checks_compact(q)
