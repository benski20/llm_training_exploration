"""Reliability review regressions for the mask store (shredding off the lock, chunk sizing, PRNG limits)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from pgw.core import RING_K40_A12, ConfigError
from pgw.maskstore import MaskStore
from pgw.maskstore import store as store_mod
from pgw.maskstore.format import SEGMENT_MAGIC

from .conftest import FP, ToyModel, write_segment


def test_close_crypto_erases_under_lock_and_wipes_in_background(
    store: MaskStore, model: ToyModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a reservation that finished a segment must not hold the store lock for the bulk wipe."""
    seg = write_segment(store, model, 4)
    path = next(store.root.rglob(f"{seg}.pgwm"))
    started, release = threading.Event(), threading.Event()
    real = store_mod.wipe_and_unlink

    def slow_wipe(p: Path, **kw: Any) -> int:
        started.set()
        assert release.wait(10)
        return real(p, **kw)

    monkeypatch.setattr(store_mod, "wipe_and_unlink", slow_wipe)
    res = store.reserve(FP, RING_K40_A12, 4)
    for s in model.specs:
        res.take(s.id, 4)
    t0 = time.perf_counter()
    res.close()  # returns although the wipe is blocked
    assert time.perf_counter() - t0 < 5
    assert started.wait(5)
    # the original file is gone (renamed) and the detached copy no longer carries the header
    assert not path.exists()
    leftovers = list(path.parent.glob(".*.shred"))
    assert len(leftovers) == 1
    assert not leftovers[0].read_bytes()[: len(SEGMENT_MAGIC)] == SEGMENT_MAGIC
    assert store.status()["segments_by_status"].get("shredded") == 1
    # the lock is free while the wipe runs: another writer/reserver proceeds immediately
    t0 = time.perf_counter()
    with store._lock.hold():
        pass
    assert time.perf_counter() - t0 < 1
    release.set()
    assert store.wait_for_wipes(10)
    assert not leftovers[0].exists()


def test_gc_finishes_interrupted_wipes(store: MaskStore, model: ToyModel) -> None:
    seg = write_segment(store, model, 2)
    seg_dir = next(store.root.rglob(f"{seg}.pgwm")).parent
    stray = seg_dir / ".deadbeef.shred"
    stray.write_bytes(b"\0" * 1024)
    store.gc()
    assert not stray.exists()


def test_zero_fill_can_be_disabled(open_store: Any, model: ToyModel, monkeypatch: pytest.MonkeyPatch) -> None:
    s = open_store(shred_zero_fill=False)
    calls: list[bool] = []
    real = store_mod.wipe_and_unlink

    def spy(p: Path, **kw: Any) -> int:
        calls.append(kw["zero_fill"])
        return real(p, **kw)

    monkeypatch.setattr(store_mod, "wipe_and_unlink", spy)
    write_segment(s, model, 2)
    with s.reserve(FP, RING_K40_A12, 2) as res:
        for sp in model.specs:
            res.take(sp.id, 2)
    assert s.wait_for_wipes(10)
    assert calls == [False]
    assert not list(s.root.rglob("*.pgwm")) and not list(s.root.rglob(".*.shred"))


# ---- chunk sizing / PRNG range / disk space -------------------------------------------------------
def _wide_specs(out: int, n: int = 64) -> list[Any]:
    from pgw.core.types import LinearId, LinearKind, LinearSpec

    return [LinearSpec(LinearId(0, LinearKind.QKV), out, n), LinearSpec(LinearId(0, LinearKind.GATE_UP), 2 * out, n)]


def test_default_chunk_is_byte_bounded(store: MaskStore) -> None:
    from pgw.maskstore.format import DEFAULT_CHUNK_BYTES, auto_chunk_slots, chunk_cache_bytes

    # 8B-like widths at K40: the old fixed 256-slot chunk cached ~1.8 GB per session
    outs = [6144, 4096, 28672, 4096] * 32
    cs = auto_chunk_slots(outs, RING_K40_A12)
    assert cs * 28672 * 5 <= DEFAULT_CHUNK_BYTES and cs >= 1
    assert chunk_cache_bytes(cs, outs, RING_K40_A12) < 64 << 20
    assert chunk_cache_bytes(256, outs, RING_K40_A12) > 1 << 30
    w = store.writer(FP, RING_K40_A12, _wide_specs(40_000), slots=64)
    try:
        assert w.header.chunk_slots == auto_chunk_slots([40_000, 80_000], RING_K40_A12)
        assert w.header.chunk_slots * 80_000 * 5 <= DEFAULT_CHUNK_BYTES
    finally:
        w.abort()


def test_writer_refuses_huge_chunks(store: MaskStore) -> None:
    with pytest.raises(ConfigError, match="chunk"):
        store.writer(FP, RING_K40_A12, _wide_specs(40_000), slots=4096, chunk_slots=256)


def test_segment_slots_beyond_prng_range_refused(store: MaskStore) -> None:
    from pgw.core.prng import max_stream_rows
    from pgw.maskstore.store import check_segment_slots

    assert 2_000_000 < max_stream_rows(14336) < 2_500_000
    check_segment_slots(max_stream_rows(14336), [4096, 14336])
    with pytest.raises(ConfigError, match="PRNG"):
        check_segment_slots(max_stream_rows(14336) + 1, [4096, 14336])
    with pytest.raises(ConfigError, match="PRNG"):
        store.writer(FP, RING_K40_A12, _wide_specs(8, n=1 << 20), slots=40_000)


def test_writer_refuses_when_volume_would_fill(open_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.core import MaskStoreError

    s = open_store(min_free_bytes=10 << 20)
    monkeypatch.setattr(MaskStore, "free_bytes", lambda self: 11 << 20)
    with pytest.raises(MaskStoreError, match="free space"):
        s.writer(FP, RING_K40_A12, _wide_specs(4096), slots=64)
    assert not list(s.root.rglob("*.part"))
    monkeypatch.setattr(MaskStore, "free_bytes", lambda self: 1 << 40)
    s.writer(FP, RING_K40_A12, _wide_specs(4096), slots=64).abort()


def test_reservation_cache_estimate(store: MaskStore, model: ToyModel) -> None:
    from pgw.maskstore.format import chunk_cache_bytes

    write_segment(store, model, 16, chunk_slots=8)
    outs = [s.out_features for s in model.specs]
    assert store.reservation_cache_bytes(FP, RING_K40_A12) == chunk_cache_bytes(8, outs, RING_K40_A12)
    assert store.reservation_cache_bytes(FP, "K48A20") == 0
