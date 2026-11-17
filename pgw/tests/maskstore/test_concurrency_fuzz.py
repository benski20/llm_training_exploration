"""Never-reuse under concurrency (threads, processes) and randomized operation sequences.

The property checked everywhere: every ``(segment_id, slot)`` is handed out at most once, across
threads, processes, store re-opens, crashes, gc, imports and closes; and the consumed counters
always equal what was handed out (plus nothing else).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import random
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, ConfigError, MaskExhaustedError, MaskReuseError
from pgw.maskstore import FileKeyProvider, MaskReservation, MaskStore, write_key_file

from . import workers
from .conftest import FP, ToyModel, make_model, write_segment

CTX = mp.get_context("spawn")


def _slots_of(res: MaskReservation) -> list[tuple[str, int]]:
    return [(e.segment_id, e.start + i) for e in res.extents for i in range(e.count)]


def test_threads_never_share_slots(store: MaskStore, model: ToyModel) -> None:
    for _ in range(6):
        write_segment(store, model, 25, chunk_slots=8)
    handed: list[tuple[str, int]] = []
    r_hashes: list[str] = []
    lock = threading.Lock()
    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        try:
            while True:
                try:
                    res = store.reserve(FP, RING_K40_A12, rng.randint(1, 9))
                except MaskExhaustedError:
                    return
                with res:
                    mine = _slots_of(res)
                    left = res.slots
                    hs = []
                    while left:
                        n = min(left, res.rows_until_boundary("1.down"))
                        r, wr = res.take("1.down", n)
                        assert torch.equal(wr, model.wr("1.down", r, RING_K40_A12))
                        hs += [hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in r]
                        left -= n
                with lock:
                    handed.extend(mine)
                    r_hashes.extend(hs)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    left = store.available_slots(FP, RING_K40_A12)
    assert left < 9  # every worker stopped only when its request no longer fitted
    assert len(handed) == 150 - left == len(set(handed))
    assert len(r_hashes) == len(handed) == len(set(r_hashes))
    assert store.verify_integrity().consumed_total == len(handed)


@pytest.mark.slow
def test_processes_never_share_slots(tmp_path: Path, model: ToyModel) -> None:
    key_file = write_key_file(tmp_path / "kek")
    root = tmp_path / "store"
    s = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0)
    for _ in range(8):
        write_segment(s, model, 30, chunk_slots=16)
    s.close()
    procs = []
    for i in range(4):
        out = tmp_path / f"out{i}"
        p = CTX.Process(target=workers.reserve_loop, args=(str(root), str(key_file), 1000, 11, i, str(out)))
        procs.append((p, out))
        p.start()
    for p, _ in procs:
        p.join(240)
        assert p.exitcode == 0
    slots: list[tuple[str, int]] = []
    rows: list[str] = []
    for _, out in procs:
        for line in out.read_text().splitlines():
            if not line.strip():
                continue  # a worker that found the store already drained
            kind, *rest = line.split()
            if kind == "E":
                seg, start, count = rest[0], int(rest[1]), int(rest[2])
                slots += [(seg, start + k) for k in range(count)]
            elif kind == "R":
                rows.append(rest[0])
    dup = [k for k, v in Counter(slots).items() if v > 1]
    assert not dup, f"slots handed out twice: {dup[:5]}"
    s = MaskStore.open(root, FileKeyProvider(key_file))
    left = s.available_slots(FP, RING_K40_A12)
    assert left < 11
    assert len(slots) == 240 - left and len(rows) == len(slots) == len(set(rows))
    assert s.verify_integrity().consumed_total == len(slots)
    s.close()


@pytest.mark.parametrize("seed", range(20))
def test_random_operation_sequences_never_reuse(tmp_path: Path, seed: int) -> None:
    """Model-based fuzzing: random interleavings of every store operation."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    key = os.urandom(32)
    from pgw.maskstore import StaticKeyProvider

    kp = StaticKeyProvider(key)
    root = tmp_path / "store"
    model = make_model(layers=1, seed=seed)
    overrides: dict[str, Any] = {
        "low_water_slots": rng.choice([0, 5, 50]),
        "shred_consumed": rng.random() < 0.5,
        "dek_cache_size": rng.choice([0, 2, 64]),
    }
    store = MaskStore.open(root, kp, **overrides)
    handed: set[tuple[str, int]] = set()
    open_res: list[MaskReservation] = []
    expected_available = 0
    consumed = 0
    for _step in range(200):
        op = rng.random()
        if op < 0.18:
            n = rng.randint(1, 12)
            write_segment(store, model, n, chunk_slots=rng.choice([1, 3, 5, 64]), checks=rng.choice([0, 1, 2]),
                          write_rows=rng.randint(1, 6), order=rng.choice(["linear", "slot"]))
            expected_available += n
        elif op < 0.50:
            n = rng.randint(1, 10)
            try:
                res = store.reserve(FP, RING_K40_A12, n, session_id=f"s{_step}")
            except MaskExhaustedError:
                assert expected_available < n
                continue
            assert n <= expected_available
            mine = _slots_of(res)
            assert len(mine) == n
            assert not (set(mine) & handed), "slot handed out twice"
            handed.update(mine)
            expected_available -= n
            consumed += n
            open_res.append(res)
        elif op < 0.72 and open_res:
            res = rng.choice(open_res)
            key_l = rng.choice(res.linear_keys)
            left = res.remaining(key_l)
            if left:
                n = rng.randint(1, max(1, min(left, res.rows_until_boundary(key_l))))
                r, wr = res.take(key_l, n)
                assert torch.equal(wr, model.wr(key_l, r, RING_K40_A12))
        elif op < 0.85 and open_res:
            res = open_res.pop(rng.randrange(len(open_res)))
            res.close()
            with pytest.raises(MaskReuseError):
                res.take(res.linear_keys[0], 1)
        elif op < 0.92:
            store.gc()
        else:  # restart (sometimes without closing reservations: simulated crash of the worker)
            if rng.random() < 0.5:
                for res in open_res:
                    res.close()
            open_res = []
            store.close()
            store = MaskStore.open(root, kp, **overrides)
        assert store.available_slots(FP, RING_K40_A12) == expected_available
    for res in open_res:
        res.close()
    rep = store.verify_integrity()
    assert rep.consumed_total == consumed == len(handed)
    store.gc()
    # every fully consumed segment is shredded (no open reservations remain)
    st = store.status()["segments_by_status"]
    assert "consumed" not in st
    store.close()


def test_reservation_is_thread_safe(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 64, chunk_slots=7)
    res = store.reserve(FP, RING_K40_A12, 64)
    got: list[torch.Tensor] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                r, wr = res.take("0.gate_up", 1)
            except MaskExhaustedError:
                return
            assert torch.equal(wr, model.wr("0.gate_up", r, RING_K40_A12))
            with lock:
                got.append(r)

    ts = [threading.Thread(target=worker) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert len(got) == 64
    assert len({tuple(r[0].tolist()) for r in got}) == 64
    res.close()
    with pytest.raises(ConfigError):
        res.take("0.gate_up", -3)
