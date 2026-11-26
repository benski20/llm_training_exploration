"""In-memory mask ring (``pgw.maskstore.ringbuf``): single use, wiping, crash, and the design-review fixes.

Invariants (design ``research/plans/mask-ringbuffer-design.md`` §10): I1 exclusive slots, I2 single use
within a reservation, I3 unique seeds, I4 no persistence, I5 wipe, I6 no recycling under a reader,
I7 same masks as disk, I8 fail closed on locking, I9 accounting. Arenas stay far below the default
8 MiB ``RLIMIT_MEMLOCK``.
"""

from __future__ import annotations

import gc
import hashlib
import os
import random
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import RING_K40_A12, SeedExpander
from pgw.core.errors import ConfigError, MaskExhaustedError, MaskReuseError, MaskStoreError, PGWError
from pgw.core.ring import RingParams
from pgw.core.settings import ring_from_setting
from pgw.maskstore import ringbuf
from pgw.maskstore.ringbuf import (
    LockedArena,
    MaskRing,
    RamSegment,
    RingFull,
    RingShort,
    SeedRegistry,
    SlabState,
    swap_status,
)

from .conftest import ToyModel, make_model, write_segment

K32 = ring_from_setting("K32A12C27")
SLOTS = 16


@pytest.fixture
def rings() -> Iterator[Callable[..., MaskRing]]:
    made: list[MaskRing] = []

    def _make(model: ToyModel, ring: RingParams = RING_K40_A12, slabs: int = 8, segment_slots: int = SLOTS,
              **kw: Any) -> MaskRing:
        kw.setdefault("min_soundness_bits", 0.0)
        kw.setdefault("seeds", SeedRegistry())
        probe = ringbuf.ring_geometry(model.specs, ring, 1 << 30, segment_slots=segment_slots)
        r = MaskRing(model.fingerprint, ring, model.specs, capacity_bytes=slabs * probe.slab_bytes,
                     segment_slots=segment_slots, **kw)
        made.append(r)
        return r

    yield _make
    for r in made:
        r.close()


def publish(ring: MaskRing, model: ToyModel, slots: int = SLOTS, seed: bytes | None = None,
            **kw: Any) -> tuple[str, bytes]:
    seed = seed if seed is not None else os.urandom(32)
    sid = write_segment(ring, model, slots, ring=ring.ring, seed=seed, chunk_slots=kw.pop("chunk_slots", 4), **kw)
    return sid, seed


def expected_r(seed: bytes, model: ToyModel, key: str, ring: RingParams, slot: int, rows: int) -> torch.Tensor:
    n = next(s.in_features for s in model.specs if s.id.key == key)
    return SeedExpander(seed).ring_uniform_rows(n, ring.k, slot, rows, "r", key)


def take_all(res: Any, key: str, step: int = 3) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    while res.remaining(key):
        n = min(step, res.rows_until_boundary(key))
        out.append(res.take(key, n))
    return out


def slot_list(res: Any) -> list[tuple[str, int]]:
    return [(e.segment_id, e.start + i) for e in res.extents for i in range(e.count)]


def check_rows(res: Any, seeds: dict[str, bytes], model: ToyModel, ring: RingParams) -> None:
    """Every row of every linear equals the factory's derivation for exactly its own slot."""
    slots = slot_list(res)
    for s in model.specs:
        key = s.id.key
        pos = 0
        while res.remaining(key):
            n = min(3, res.rows_until_boundary(key))
            r, wr = res.take(key, n)
            for i in range(n):
                sid, slot = slots[pos + i]
                assert torch.equal(r[i : i + 1], expected_r(seeds[sid], model, key, ring, slot, 1))
            assert torch.equal(wr.to(torch.int64), model.wr(key, r, ring))
            pos += n


def invariant(ring: MaskRing) -> None:
    """I9: published = reserved + available + lost; reserved = taken + burned + leaked + open."""
    c = ring.counts()
    assert c["published_slots"] == c["reserved_slots"] + c["available_slots"] + c["lost_slots"], c
    assert c["reserved_slots"] == c["taken_slots"] + c["burned_slots"] + c["leaked_slots"] + c["open_slots"], c


def slab_bytes(ring: MaskRing, index: int) -> bytes:
    g = ring.geometry
    return bytes(memoryview(ring._arena._cbuf)[index * g.slab_bytes : (index + 1) * g.slab_bytes])


# ---- I1 / I2 / I9 ---------------------------------------------------------------------------------
def test_concurrent_reserve_disjoint(rings: Any, model: ToyModel) -> None:
    ring = rings(model, slabs=6)
    rng = random.Random(1234)
    sizes = [rng.randint(1, 40) for _ in range(400)]
    lock = threading.Lock()
    seen: list[tuple[str, int]] = []
    published: dict[str, bytes] = {}
    done = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for _ in range(30):
                while True:
                    try:
                        sid, seed = publish(ring, model)
                        with lock:
                            published[sid] = seed
                        break
                    except RingFull:
                        time.sleep(0.002)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            done.set()

    def reserver(k: int) -> None:
        try:
            while True:
                with lock:
                    if not sizes:
                        return
                    n = sizes.pop()
                try:
                    res = ring.reserve(n, session_id=f"t{k}")
                except RingShort:
                    with lock:
                        sizes.append(n)
                    if done.is_set() and ring.available < n:
                        return
                    time.sleep(0.001)
                    continue
                with lock:
                    seen.extend(slot_list(res))
                if n % 3 == 0:  # take a few rows of one linear before closing
                    res.take(model.specs[0].id, min(2, res.rows_until_boundary(model.specs[0].id)))
                res.close()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=writer)] + [threading.Thread(target=reserver, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert not errors
    assert len(seen) == len(set(seen))  # I1: pairwise disjoint
    assert {sid for sid, _ in seen} <= set(published)
    assert len(seen) == ring.counts()["reserved_slots"]
    ring.wait_idle()
    invariant(ring)


def test_rows_single_use(rings: Any, model: ToyModel) -> None:
    ring = rings(model)
    sid, seed = publish(ring, model)
    a = ring.reserve(6)
    b = ring.reserve(10)
    assert set(slot_list(a)).isdisjoint(slot_list(b))
    assert sorted(slot_list(a) + slot_list(b)) == [(sid, i) for i in range(SLOTS)]
    for res in (a, b):
        check_rows(res, {sid: seed}, model, ring.ring)
        with pytest.raises(MaskExhaustedError):
            res.take(model.specs[0].id, 1)
        res.close()
    invariant(ring)


@pytest.mark.parametrize("rname", ["K40A12", "K32A12C27"])
def test_ring_matches_store(rings: Any, store: Any, model: ToyModel, rname: str) -> None:
    """I7: the same segment through the ring and through the encrypted store gives bit-identical r, W r
    and check vectors."""
    rp = ring_from_setting(rname)
    ring = rings(model, ring=rp)
    seed = os.urandom(32)
    publish(ring, model, seed=seed)
    write_segment(store, model, SLOTS, ring=rp, seed=seed, chunk_slots=4)
    a = ring.reserve(SLOTS)
    b = store.reserve(model.fingerprint, rp, SLOTS)
    for s in model.specs:
        key = s.id.key
        assert torch.equal(a.checks(key)[0], b.checks(key)[0]) and torch.equal(a.checks(key)[1], b.checks(key)[1])
        ta, tb = a.checks_compact(key), b.checks_compact(key)
        assert np.array_equal(ta[1], tb[1]) and np.array_equal(ta[2], tb[2])
        while a.remaining(key):
            ra = a.take_checked(key, 5, checks=False)
            rb = b.take_checked(key, 5, checks=False)
            assert ra[0] == rb[0]
            assert torch.equal(ra[1], rb[1]) and torch.equal(ra[2], rb[2])
    a.close()
    b.close()


@pytest.mark.parametrize("rname", ["K32A12C27", "K40A12", "K48A20"])
def test_direct_row_reads_match_chunk_copies(rings: Any, model: ToyModel, rname: str) -> None:
    """Zero-copy reads (``RamSegment.read_wr_rows``) decode exactly the rows of the copied chunks, for every
    slot range (including ranges that span chunks), in the int64 and the compact form."""
    from pgw.core.packing import unpack_ring

    rp = ring_from_setting(rname)
    ring = rings(model, ring=rp)
    publish(ring, model, chunk_slots=4)
    res = ring.reserve(SLOTS)
    seg = res._extents[0].seg
    lay = seg.layout
    for li, s in enumerate(model.specs):
        full = torch.cat([unpack_ring(seg.decrypt_into(lay.wr_chunk(li, j)), (lay.chunk_rows(j), s.out_features), rp)
                          for j in range(lay.chunks_per_linear)])
        for slot0, rows in [(0, 1), (3, 2), (2, 7), (0, SLOTS), (SLOTS - 1, 1)]:
            for compact in (False, True):
                got = seg.read_wr_rows(li, slot0, rows, compact=compact)
                assert got.flags.writeable and got.flags.owndata
                assert got.dtype == (np.uint32 if compact and rp.k <= 32 else np.int64)
                assert np.array_equal(got.astype(np.int64), full[slot0:slot0 + rows].numpy())
        with pytest.raises(MaskStoreError):
            seg.read_wr_rows(li, SLOTS - 1, 2, compact=False)
    # the take path uses the direct reads: no per-reservation chunk copies are cached
    take_all(res, model.specs[0].id.key)
    assert not res._chunk_cache
    res.close()
    with pytest.raises(MaskStoreError):
        seg.read_wr_rows(0, 0, 1, compact=False)
    ring2 = rings(model, ring=rp)
    publish(ring2, model)
    seg2 = ring2.reserve(4)._extents[0].seg
    ring2.close()
    with pytest.raises(MaskStoreError):
        seg2.read_wr_rows(0, 0, 1, compact=False)


@pytest.mark.parametrize("threads", [1, 3])
def test_batched_claim_on_ring_matches_sequential(rings: Any, model: ToyModel, threads: int,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """``take_checked_many`` over ring reservations (native, rows read from the slabs under the slab
    locks) hands out exactly the rows the sequential ``take_checked`` loop does: same r, same W r, same
    check vectors, same cursors, across extent boundaries and with prefetched rows."""
    from pgw.maskstore import store as store_mod
    from pgw.maskstore import take_checked_many

    if not store_mod._native_claim:
        pytest.skip("pgw_fast.claim_rows_many not built")
    monkeypatch.setattr(ringbuf.RingReservation, "batched_claim", True)
    calls = []
    real = store_mod.native.fast.claim_rows_many
    monkeypatch.setattr(store_mod.native.fast, "claim_rows_many", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    a, b = rings(model, ring=K32), rings(model, ring=K32)
    for _ in range(3):
        seed = os.urandom(32)
        publish(a, model, seed=seed, chunk_slots=4)
        publish(b, model, seed=seed, chunk_slots=4)
    sizes = [5, 11, 7, 20]  # 20 spans two slabs
    ra = [a.reserve(n) for n in sizes]
    rb = [b.reserve(n) for n in sizes]
    keys = [s.id.key for s in model.specs]
    ra[1].prefetch(keys[0], 3)
    rb[1].prefetch(keys[0], 3)
    rng = random.Random(7)
    while any(x.remaining(k) for x in ra for k in keys):
        reqs = [(i, k, rng.randint(1, 4)) for i in range(len(sizes)) for k in keys if ra[i].remaining(k)]
        got = take_checked_many([(ra[i], k, n, True) for i, k, n in reqs], threads=threads)
        for (i, k, n), g in zip(reqs, got, strict=True):
            w = rb[i].take_checked(k, n, True)
            assert g[0] == w[0]
            assert torch.equal(g[1], w[1]) and torch.equal(g[2].to(torch.int64), w[2].to(torch.int64))
            assert g[3][0][1] == w[3][0][1]  # (segment id, linear): segment ids differ between the two rings
            assert np.array_equal(g[3][1], w[3][1]) and np.array_equal(g[3][2], w[3][2])
    assert calls  # the native batched path ran (no fallback to the sequential loop)
    for x in ra + rb:
        x.close()


def test_batched_claim_on_closed_ring_raises(rings: Any, model: ToyModel, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.maskstore import store as store_mod
    from pgw.maskstore import take_checked_many

    if not store_mod._native_claim:
        pytest.skip("pgw_fast.claim_rows_many not built")
    monkeypatch.setattr(ringbuf.RingReservation, "batched_claim", True)
    ring = rings(model, ring=K32)
    publish(ring, model)
    res = ring.reserve(4)
    seg = res._extents[0].seg
    with seg.slab_lock:
        seg._slab.gen += 1  # a recycled slab: the re-check under the lock refuses it
        try:
            with pytest.raises(MaskStoreError):
                seg.wr_view_locked(0)
        finally:
            seg._slab.gen -= 1
    ring.close()
    with pytest.raises(PGWError):
        take_checked_many([(res, model.specs[0].id.key, 1, True)])


def test_close_burns_never_rewinds(rings: Any, model: ToyModel) -> None:
    ring = rings(model)
    sid, _ = publish(ring, model)
    a = ring.reserve(10)
    a.close()
    b = ring.reserve(6)
    assert b.extents[0].segment_id == sid and b.extents[0].start == 10
    with pytest.raises(RingShort):
        ring.reserve(1)
    b.close()
    assert ring.counts()["burned_slots"] == 16
    invariant(ring)


# ---- I5 wipe ----------------------------------------------------------------------------------------
def test_abort_wipes(rings: Any, model: ToyModel) -> None:
    ring = rings(model)
    w = ring.writer(model.fingerprint, ring.ring, model.specs, slots=SLOTS, seed=os.urandom(32), checks=2,
                    chunk_slots=4)
    exp = SeedExpander(os.urandom(32))
    for s in model.specs[:3]:
        r = exp.ring_uniform_rows(s.in_features, ring.ring.k, 0, SLOTS, "r", s.id.key)
        w.write_linear(s.id, model.wr(s.id.key, r, ring.ring), torch.zeros((2, s.in_features), dtype=torch.int64))
    idx = w._slab.index
    assert any(slab_bytes(ring, idx))
    w.abort()
    assert ring.wait_idle()
    assert not any(slab_bytes(ring, idx))
    assert ring.slab_states()[idx] == SlabState.FREE
    c = ring.counts()
    assert c["available_slots"] == 0 and c["aborted_slots"] == SLOTS and c["published_slots"] == 0
    with pytest.raises(MaskStoreError):
        w.write_linear(model.specs[3].id, torch.zeros((1, model.specs[3].out_features), dtype=torch.int64),
                       torch.zeros((2, model.specs[3].in_features), dtype=torch.int64))


def test_drain_wipes_and_recycles(rings: Any, model: ToyModel) -> None:
    ring = rings(model)
    fired: list[int] = []
    ring.add_free_listener(lambda: fired.append(1))
    publish(ring, model)
    idx = next(i for i, s in enumerate(ring.slab_states()) if s == SlabState.READY)
    res = ring.reserve(SLOTS)
    res.take(model.specs[0].id, 4)
    assert ring.slab_states()[idx] == SlabState.DRAINING
    res.close()
    assert ring.wait_idle()
    deadline = time.monotonic() + 2
    while not fired and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ring.slab_states()[idx] == SlabState.FREE
    assert not any(slab_bytes(ring, idx))
    assert fired == [1]
    invariant(ring)


def test_no_recycle_while_open(rings: Any, model: ToyModel) -> None:
    """I6: a fully reserved slab with an open reservation is never wiped or granted; close() force-closes."""
    ring = rings(model, slabs=4)
    publish(ring, model)
    res = ring.reserve(SLOTS)
    pinned = next(i for i, s in enumerate(ring.slab_states()) if s == SlabState.DRAINING)
    granted = []
    for _ in range(3):
        w = ring.writer(model.fingerprint, ring.ring, model.specs, slots=SLOTS, seed=os.urandom(32), chunk_slots=4)
        granted.append(w._slab.index)
    with pytest.raises(RingFull):
        ring.writer(model.fingerprint, ring.ring, model.specs, slots=SLOTS, seed=os.urandom(32), chunk_slots=4)
    assert pinned not in granted
    assert ring.slab_states()[pinned] == SlabState.DRAINING
    res.take(model.specs[0].id, 2)  # still readable
    ring.close()
    assert res.closed
    with pytest.raises(MaskReuseError):
        res.take(model.specs[0].id, 1)


def test_new_ring_is_empty_and_close_wipes_all(rings: Any, model: ToyModel) -> None:
    ring = rings(model)
    assert ring.available == 0 and ring._arena.is_zero()
    with pytest.raises(RingShort):
        ring.reserve(1)
    publish(ring, model)
    publish(ring, model)
    res = ring.reserve(20)
    res.take(model.specs[0].id, 3)
    arena = ring._arena
    zero_at_unmap: list[bool] = []
    orig = arena._unmap

    def unmap() -> None:
        zero_at_unmap.append(ringbuf._all_zero(memoryview(arena._cbuf)))
        orig()

    arena._unmap = unmap  # type: ignore[method-assign]
    ring.close()
    assert zero_at_unmap == [True]
    assert res.closed
    with pytest.raises(MaskStoreError):
        ring.reserve(1)
    c = ring.counts()
    assert c["lost_slots"] == 12  # published, never reserved, wiped
    invariant(ring)


# ---- I3 seeds -------------------------------------------------------------------------------------
def test_duplicate_seed_refused(rings: Any, model: ToyModel) -> None:
    seeds = SeedRegistry()
    ring = rings(model, seeds=seeds)
    seed = os.urandom(32)
    publish(ring, model, seed=seed)
    with pytest.raises(MaskReuseError):  # at writer() (grant)
        ring.writer(model.fingerprint, ring.ring, model.specs, slots=SLOTS, seed=seed)
    other = rings(make_model(seed=1), seeds=seeds)  # another ring sharing the registry
    with pytest.raises(MaskReuseError):
        other.writer(other.fingerprint, other.ring, other.specs, slots=SLOTS, seed=seed)
    with pytest.raises(MaskReuseError):
        ring.writer(model.fingerprint, ring.ring, model.specs, slots=SLOTS, seed=bytes(32))
    # at commit: the writer must still own its seed claim
    w = ring.writer(model.fingerprint, ring.ring, model.specs, slots=1, seed=os.urandom(32), chunk_slots=1)
    for s in model.specs:
        w.write_linear(s.id, torch.zeros((1, s.out_features), dtype=torch.int64),
                       torch.zeros((2, s.in_features), dtype=torch.int64))
    w._token = "someone-else"  # noqa: S105 - an owner token, not a secret
    with pytest.raises(MaskReuseError):
        w.commit()
    assert ring.wait_idle() and ring.counts()["published_segments"] == 1


def test_store_and_ring_share_seed_registry(rings: Any, store: Any, model: ToyModel) -> None:
    from pgw.factory.sinks import RingSink, StoreSink

    seeds = SeedRegistry()
    ring = rings(model, seeds=seeds)
    ssink = StoreSink(store, seeds=seeds)
    rsink = RingSink({ring.key: ring}, store=store)
    seed = os.urandom(32)
    write_segment(ssink, model, 4, ring=ring.ring, seed=seed)
    with pytest.raises(MaskReuseError):  # the registry knows it
        rsink.writer(model.fingerprint, ring.ring, model.specs, slots=4, seed=seed)
    seed2 = os.urandom(32)
    write_segment(store, model, 4, ring=ring.ring, seed=seed2)  # written past the registry
    with pytest.raises(MaskReuseError):  # the store's seed table knows it
        rsink.writer(model.fingerprint, ring.ring, model.specs, slots=4, seed=seed2)
    seed3 = os.urandom(32)
    publish(ring, model, seed=seed3)
    with pytest.raises(MaskReuseError):
        ssink.writer(model.fingerprint, ring.ring, model.specs, slots=4, seed=seed3, checks=2)


# ---- geometry and soundness -----------------------------------------------------------------------
def test_soundness_floor(rings: Any, model: ToyModel) -> None:
    ring = rings(model, min_soundness_bits=10_000.0)
    with pytest.raises(MaskStoreError, match="soundness"):
        publish(ring, model)
    assert ring.wait_idle()
    assert ring.counts()["published_slots"] == 0 and ring.slab_states().count(SlabState.FREE) == ring.n_slabs


def test_geometry_refused(rings: Any, model: ToyModel) -> None:
    ring = rings(model)
    seed = os.urandom
    with pytest.raises(ConfigError):
        ring.writer("cd" * 32, ring.ring, model.specs, slots=4, seed=seed(32))
    with pytest.raises(ConfigError):
        ring.writer(model.fingerprint, K32, model.specs, slots=4, seed=seed(32))
    with pytest.raises(ConfigError):
        ring.writer(model.fingerprint, ring.ring, list(reversed(model.specs)), slots=4, seed=seed(32))
    with pytest.raises(ConfigError):
        ring.writer(model.fingerprint, ring.ring, model.specs, slots=4, seed=seed(32), destination="ab" * 16)
    with pytest.raises(ConfigError):
        ring.writer(model.fingerprint, ring.ring, model.specs, slots=SLOTS + 1, seed=seed(32))
    with pytest.raises(ConfigError):
        ring.writer(model.fingerprint, ring.ring, model.specs, slots=4, seed=seed(32), checks=9)
    with pytest.raises(ConfigError, match="slab"):
        rings(model, slabs=3)


def test_range_check_and_compact_input(rings: Any, model: ToyModel) -> None:
    ring = rings(model, ring=K32)
    w = ring.writer(model.fingerprint, K32, model.specs, slots=2, seed=os.urandom(32), chunk_slots=2)
    s = model.specs[0]
    uw = torch.zeros((2, s.in_features), dtype=torch.int64)
    with pytest.raises(PGWError):
        w.write_linear(s.id, torch.full((2, s.out_features), 1 << 32, dtype=torch.int64), uw)
    w.write_linear(s.id, torch.full((2, s.out_features), -1, dtype=torch.int32), uw)  # compact uint32 form
    w.abort()


# ---- I4 / I8 arena --------------------------------------------------------------------------------
def _smaps_entry(addr: int) -> dict[str, str]:
    lines = Path("/proc/self/smaps").read_text().splitlines()
    out: dict[str, str] = {}
    hit = False
    for line in lines:
        head = line.split()[0] if line else ""
        if "-" in head and ":" not in head:
            if hit:
                break
            hit = int(head.split("-")[0], 16) == addr
            continue
        if hit:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def test_arena_locked_dontdump_dontfork() -> None:
    arena = LockedArena(1 << 20)
    try:
        e = _smaps_entry(arena.addr)
        assert e, "arena VMA not found"
        assert e["Locked"] == e["Size"]
        flags = e["VmFlags"].split()
        assert {"lo", "dd", "dc"} <= set(flags)
        assert arena.locked and arena.dontdump
    finally:
        arena.close()


def test_mlock_failure_fails_closed(caplog: pytest.LogCaptureFixture, model: ToyModel) -> None:
    def failing(addr: int, n: int) -> int:
        return -1

    with pytest.raises(ConfigError, match="mlock"):
        LockedArena(1 << 16, mlock=failing)
    with pytest.raises(ConfigError, match="mlock"):
        MaskRing(model.fingerprint, RING_K40_A12, model.specs, capacity_bytes=1 << 20, segment_slots=SLOTS,
                 _mlock=failing)
    ring = MaskRing(model.fingerprint, RING_K40_A12, model.specs, capacity_bytes=1 << 20, segment_slots=SLOTS,
                    _mlock=failing, allow_unlocked=True, min_soundness_bits=0, seeds=SeedRegistry())
    try:
        assert ring.locked_bytes == 0
        assert "UNLOCKED" in caplog.text or "allow_unlocked" in caplog.text
    finally:
        ring.close()


def test_swap_status_parsing(tmp_path: Path) -> None:
    p = tmp_path / "swaps"
    p.write_text("Filename\tType\tSize\tUsed\tPriority\n/dev/zram0 partition 1 0 100\n/swapfile file 1 0 -2\n")
    st = swap_status(str(p), str(tmp_path))
    assert st["active"] == ["/dev/zram0", "/swapfile"] and st["unprotected"] == ["/swapfile"]


# ---- crash ----------------------------------------------------------------------------------------
_CRASH_CHILD = textwrap.dedent('''
    import os, sys, time
    _tests, *_rest = sys.argv[3].split(os.pathsep)
    sys.path[:0] = _rest
    sys.path.append(_tests)  # last: tests/platform must not shadow the standard library's platform
    from maskstore.conftest import make_model, write_segment
    from pgw.core import RING_K40_A12
    from pgw.factory.sinks import RingSink, StoreSink
    from pgw.maskstore import MaskStore, StaticKeyProvider
    from pgw.maskstore.ringbuf import MaskRing, SeedRegistry
    root, kek = sys.argv[1], bytes.fromhex(sys.argv[2])
    model = make_model()
    store = MaskStore.open(root, StaticKeyProvider(kek), low_water_slots=0, allow_no_external_counter=True,
                           fsync=False, min_free_bytes=0)
    seeds = SeedRegistry()
    write_segment(StoreSink(store, seeds=seeds), model, 8, ring=RING_K40_A12)
    print("STORE_READY", flush=True)
    sys.stdin.readline()
    ring = MaskRing(model.fingerprint, RING_K40_A12, model.specs, capacity_bytes=1 << 20, segment_slots=16,
                    seeds=seeds, min_soundness_bits=0)
    sink = RingSink({ring.key: ring}, store=store)
    ids = [write_segment(sink, model, 16, ring=RING_K40_A12) for _ in range(3)]
    res = ring.reserve(20)
    res.take(model.specs[0].id, 5)
    print("RING " + " ".join(ids), flush=True)
    time.sleep(120)
''')


def _store_snapshot(root: Path) -> dict[str, str]:
    snap: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith(("-shm", "-wal", ".lock")):
            snap[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    db = root / "state.sqlite"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        snap["__db__"] = hashlib.sha256("\n".join(con.iterdump()).encode()).hexdigest()
    finally:
        con.close()
    return snap


def test_crash_leaves_disk_untouched(tmp_path: Path) -> None:
    """I3/I4: a SIGKILLed process holding ring segments and a reservation leaves no trace on disk."""
    root = tmp_path / "store"
    script = tmp_path / "child.py"
    script.write_text(_CRASH_CHILD)
    tests_dir = str(Path(__file__).resolve().parents[1])
    src = str(Path(__file__).resolve().parents[2] / "src")
    proc = subprocess.Popen(  # noqa: S603 - our own interpreter and script
        [sys.executable, str(script), str(root), os.urandom(32).hex(), os.pathsep.join([tests_dir, src])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=str(tmp_path),
    )
    try:
        assert proc.stdout is not None and proc.stdin is not None
        line = proc.stdout.readline()
        assert line.strip() == "STORE_READY", line
        before = _store_snapshot(root)
        proc.stdin.write("go\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        assert line.startswith("RING "), line
        ids = line.split()[1:]
        assert len(ids) == 3
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    after = _store_snapshot(root)
    assert before == after
    for p in tmp_path.rglob("*"):
        if p.is_file() and p != script:
            data = p.read_bytes()
            assert not any(i.encode() in data for i in ids), p


# ---- I9 property ----------------------------------------------------------------------------------
def test_accounting_property(rings: Any, model: ToyModel) -> None:
    rng = random.Random(7)
    ring = rings(model, slabs=5)
    open_res: list[Any] = []
    for _ in range(300):
        op = rng.choice(["publish", "reserve", "take", "close", "abort"])
        if op == "publish":
            try:
                publish(ring, model, slots=rng.randint(1, SLOTS))
            except RingFull:
                pass
        elif op == "reserve":
            try:
                open_res.append(ring.reserve(rng.randint(1, 24)))
            except RingShort:
                pass
        elif op == "take" and open_res:
            res = rng.choice(open_res)
            key = rng.choice(model.specs).id.key
            if res.remaining(key):
                res.take(key, min(rng.randint(1, 4), res.rows_until_boundary(key)))
        elif op == "close" and open_res:
            open_res.pop(rng.randrange(len(open_res))).close()
        elif op == "abort":
            try:
                w = ring.writer(model.fingerprint, ring.ring, model.specs, slots=4, seed=os.urandom(32))
                w.abort()
            except RingFull:
                pass
        invariant(ring)
    for res in open_res:
        res.close()
    ring.wait_idle()
    invariant(ring)
    assert ring.counts()["open_reservations"] == 0


# ---- design-review fixes --------------------------------------------------------------------------
def test_reserve_interleaved_with_close_never_serves_zero_rows(rings: Any, model: ToyModel,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocking issue 1: close() between the cursor move (step 2) and the reservation's registration."""
    ring = rings(model)
    publish(ring, model)
    entered, release = threading.Event(), threading.Event()
    orig_init = ringbuf.RingReservation.__init__

    def slow_init(self: Any, *a: Any, **kw: Any) -> None:
        entered.set()
        release.wait(5)
        orig_init(self, *a, **kw)

    monkeypatch.setattr(ringbuf.RingReservation, "__init__", slow_init)
    out: list[Any] = []
    t = threading.Thread(target=lambda: out.append(_try(lambda: ring.reserve(8))))
    t.start()
    assert entered.wait(5)
    closer = threading.Thread(target=ring.close)
    closer.start()
    time.sleep(0.1)
    release.set()
    t.join(5)
    closer.join(5)
    assert isinstance(out[0], MaskStoreError)  # never a reservation over a wiped slab


def _try(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except BaseException as exc:
        return exc


def test_concurrent_takes_during_close_raise_or_are_exact(rings: Any, model: ToyModel) -> None:
    """Reads racing close(): reservations stay open across close() and every take after it either
    raises or returns exactly the factory's rows (never zeros, never another slot's masks)."""
    ring = rings(model, slabs=6)
    seeds = dict(publish(ring, model, chunk_slots=1) for _ in range(5))  # one chunk per row: every take reads
    key = model.specs[1].id.key
    n_readers, per = 6, 13
    held = [ring.reserve(per) for _ in range(n_readers)]  # open before close() and held across it
    handles = [e.seg for res in held for e in res._extents]
    ready = threading.Barrier(n_readers + 1)
    go = threading.Event()
    exact = [0] * n_readers
    raised = [0] * n_readers
    bad: list[str] = []

    def reader(i: int) -> None:
        res = held[i]
        slots = slot_list(res)
        pos = 0

        def take_one() -> None:
            nonlocal pos
            r, wr = res.take(key, 1)
            sid, slot = slots[pos]
            pos += 1
            if not torch.equal(r, expected_r(seeds[sid], model, key, ring.ring, slot, 1)):
                bad.append("r")
            elif not wr.any() or not torch.equal(wr, model.wr(key, r, ring.ring)):
                bad.append("wr")
            else:
                exact[i] += 1

        take_one()
        take_one()
        ready.wait(10)
        go.wait(10)
        try:
            while pos < per:
                take_one()
                time.sleep(0.001)  # 11 more rows take >= 11 ms: close() always lands mid-take
        except (MaskStoreError, MaskReuseError):
            raised[i] += 1

    ts = [threading.Thread(target=reader, args=(i,)) for i in range(n_readers)]
    for t in ts:
        t.start()
    ready.wait(10)
    go.set()
    ring.close()
    for t in ts:
        t.join(10)
    assert not bad
    assert sum(raised) >= 1, (exact, raised)  # close() really raced the takes
    assert all(exact[i] == per or raised[i] == 1 for i in range(n_readers)), (exact, raised)
    assert sum(exact) >= 2 * n_readers
    for h in handles:  # every handle is revoked: a late read raises instead of reading wiped memory
        with pytest.raises(MaskStoreError):
            h.decrypt_into(h.layout.wr_chunk(1, 0))
    assert ring._arena.closed


def test_seed_check_on_secrets_read(rings: Any, model: ToyModel) -> None:
    ring = rings(model)
    publish(ring, model)
    res = ring.reserve(4)
    seg = res._extents[0].seg
    assert isinstance(seg, RamSegment)
    slab = seg._slab
    ring._arena.zero(slab.off, 32)  # seed region zeroed (e.g. a wipe raced the read)
    with pytest.raises(MaskStoreError, match="seed"):
        res.take(model.specs[0].id, 1)
    res.close()


def test_finalizer_close_while_ring_lock_held(rings: Any, model: ToyModel) -> None:
    """Blocking issue 2: a GC finalizer of a ring reservation must never take the ring lock."""
    ring = rings(model)
    publish(ring, model)
    res = ring.reserve(SLOTS)
    res.take(model.specs[0].id, 2)
    res._cycle = res  # type: ignore[attr-defined]  # only the cyclic GC can free it
    del res
    done = threading.Event()

    def collect_under_lock() -> None:
        with ring._lock:
            gc.collect()
        done.set()

    gc_was = gc.isenabled()
    gc.disable()
    try:
        t = threading.Thread(target=collect_under_lock, daemon=True)
        t.start()
        assert done.wait(10), "finalizer deadlocked on the ring lock"
    finally:
        if gc_was:
            gc.enable()
    assert ring.wait_idle()
    deadline = time.monotonic() + 3
    while ring.counts()["open_reservations"] and time.monotonic() < deadline:
        time.sleep(0.05)
    c = ring.counts()
    assert c["leaked_slots"] == SLOTS and c["open_reservations"] == 0
    assert c["free_slabs"] == ring.n_slabs  # the slab was recycled
    invariant(ring)


def test_close_joins_wiper_before_unmap(rings: Any, model: ToyModel) -> None:
    """Blocking issue 3: the wiper never writes into unmapped memory."""
    ring = rings(model)
    publish(ring, model)
    arena = ring._arena
    order: list[str] = []
    started = threading.Event()
    orig_zero, orig_unmap = arena.zero, arena._unmap

    def slow_zero(off: int, n: int) -> None:
        if threading.current_thread().name == "pgw-maskring-wiper":
            started.set()
            order.append("wipe-start")
            time.sleep(0.3)
            orig_zero(off, n)
            order.append("wipe-end")
        else:
            orig_zero(off, n)

    def unmap() -> None:
        order.append("unmap")
        orig_unmap()

    arena.zero = slow_zero  # type: ignore[method-assign]
    arena._unmap = unmap  # type: ignore[method-assign]
    ring.reserve(SLOTS).close()  # drained: the wiper starts a (slow) wipe
    assert started.wait(5)
    ring.close()
    assert order == ["wipe-start", "wipe-end", "unmap"]


def test_copy_out_makes_no_immutable_copy(rings: Any, model: ToyModel) -> None:
    """Blocking issue 4: reads copy arena -> bytearray directly (never by slicing the mmap)."""
    ring = rings(model)
    publish(ring, model)
    arena = ring._arena
    real_mm = arena._mm

    class NoSlice:
        def __getitem__(self, k: Any) -> Any:
            raise AssertionError("mmap sliced: immutable bytes copy")

    arena._mm = NoSlice()  # type: ignore[assignment]
    try:
        res = ring.reserve(4)
        seg = res._extents[0].seg
        assert type(seg.decrypt_into(seg.layout.secrets_chunk())) is bytearray
        assert type(seg.decrypt_into(seg.layout.wr_chunk(0, 0))) is bytearray
        take_all(res, model.specs[0].id.key)
        res.close()
    finally:
        arena._mm = real_mm




# ---- implementation review, round 2 ---------------------------------------------------------------
def test_seed_registry_is_bounded_and_pins_live_slabs(rings: Any, model: ToyModel) -> None:
    reg = SeedRegistry(max_recent=4)
    for i in range(10):
        reg.claim(f"store-{i}", "store")
    assert len(reg) == 4 and reg.evicted == 6  # a fixed window, not one entry per segment forever
    with pytest.raises(MaskReuseError):
        reg.claim("store-9", "store")
    ring = rings(model, slabs=4, seeds=reg)
    seeds = [publish(ring, model)[1] for _ in range(3)]
    digests = [ringbuf.seed_digest(s) for s in seeds]
    assert reg.live == 3
    for i in range(20):  # flood the recent window: live ring seeds stay pinned
        reg.claim(f"more-{i}", "store")
    assert all(d in reg for d in digests)
    ring.reserve(SLOTS).close()  # drain one slab -> wipe -> its seed is unpinned
    assert ring.wait_idle()
    assert reg.live == 2
    assert digests[0] in reg  # still remembered in the recent window ...
    for seed in seeds[:2]:  # a free slab exists now: duplicates are refused by the registry
        with pytest.raises(MaskReuseError):
            publish(ring, model, seed=seed)
    for i in range(4):
        reg.claim(f"later-{i}", "store")
    assert digests[0] not in reg and digests[1] in reg  # ... until it ages out; live ones never do
    ring.close()
    assert reg.live == 0


def test_init_on_free_status(tmp_path: Path) -> None:
    from pgw.maskstore.tiered import check_init_on_free

    cmd = tmp_path / "cmdline"
    cfg = tmp_path / "config"
    cfg.write_text("CONFIG_INIT_ON_FREE_DEFAULT_ON=y\n")
    cmd.write_text("ro quiet init_on_free=1\n")
    assert ringbuf.init_on_free_status(str(cmd), [str(cfg)]) is True
    cmd.write_text("init_on_free=1 init_on_free=0\n")  # the last one wins
    assert ringbuf.init_on_free_status(str(cmd), [str(cfg)]) is False
    cmd.write_text("ro quiet\n")
    assert ringbuf.init_on_free_status(str(cmd), [str(cfg)]) is True  # kernel default
    cfg.write_text("# CONFIG_INIT_ON_FREE_DEFAULT_ON is not set\n")
    assert ringbuf.init_on_free_status(str(cmd), [str(cfg)]) is False
    assert ringbuf.init_on_free_status(str(cmd), [str(tmp_path / "missing")]) is None
    assert check_init_on_free("refuse", status_fn=lambda: True) is True
    assert check_init_on_free("warn", status_fn=lambda: None) is None
    with pytest.raises(ConfigError, match="init_on_free"):
        check_init_on_free("refuse", status_fn=lambda: False)


def test_memlock_raised_for_arena_and_memfd_slabs(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1 arenas and v2 slab mappings both raise RLIMIT_MEMLOCK soft-to-hard before mlock."""
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(ringbuf.resource, "getrlimit", lambda _r: (1 << 16, 1 << 30))
    monkeypatch.setattr(ringbuf.resource, "setrlimit", lambda _r, lim: calls.append(lim))
    LockedArena(1 << 20, allow_unlocked=True, mlock=lambda a, n: 0).close()
    assert calls == [(1 << 30, 1 << 30)]
    calls.clear()
    slab = ringbuf.MemfdSlab(1 << 16, allow_unlocked=True)
    slab.close()
    assert calls == [(1 << 30, 1 << 30)]


def test_unsealed_memfd_slab_is_zeroed_on_drop() -> None:
    slab = ringbuf.MemfdSlab(1 << 16, allow_unlocked=True)
    os.pwrite(slab.fd, b"\xa5" * 4096, 100)
    keep = os.dup(slab.fd)
    try:
        slab.close()
        assert os.pread(keep, 1 << 16, 0) == bytes(1 << 16)
    finally:
        os.close(keep)
    sealed = ringbuf.MemfdSlab(1 << 16, allow_unlocked=True)
    os.pwrite(sealed.fd, b"\x5a", 0)
    sealed.seal()
    sealed.close()  # sealed: cannot be written, only dropped (SPEC 7.8)
