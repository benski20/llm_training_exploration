"""Many processes x many threads of gateway-style incremental sessions over one mask store (the client library's
multi-worker layout: gunicorn workers with the same key and cache directory).

Checked: no ``(segment, slot)`` and no mask row is ever handed out twice, under any interleaving of admissions,
first parts, on-demand and background (ahead) extensions, background retirement, session closes, store gc and a
concurrent factory committing segments; the store's consumed total equals what was handed out; and with the
cross-process entitlement ledger no running session's extension ever fails for lack of stock.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import pytest

from pgw.gateway.masks import EntitlementLedger, SharedLedger
from pgw.maskstore import FileKeyProvider, MaskStore, write_key_file

from . import scale_workers

CTX = mp.get_context("spawn")
FP = "ab" * 32


def _run(tmp_path: Path, *, procs: int, threads: int, per_thread: int, initial: int, later: int,
         shared: bool) -> tuple[list[tuple[str, int]], list[str], Counter[str], int]:
    from .conftest import make_model, write_segment

    key_file = write_key_file(tmp_path / "kek")
    root = tmp_path / "store"
    s = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0)
    model = make_model()
    for _ in range(initial):
        write_segment(s, model, 24, chunk_slots=8)
    s.close()
    ps = [CTX.Process(target=scale_workers.writer, args=(str(root), str(key_file), later, 24, 0.05))]
    outs = []
    for i in range(procs):
        out = tmp_path / f"out{i}"
        outs.append(out)
        ps.append(CTX.Process(target=scale_workers.sessions,
                              args=(str(root), str(key_file), str(tmp_path / "ledger"), threads, per_thread, i,
                                    str(out), shared, 60.0)))
    for p in ps:
        p.start()
    for p in ps:
        p.join(300)
        assert p.exitcode == 0
    slots: list[tuple[str, int]] = []
    rows: list[str] = []
    kinds: Counter[str] = Counter()
    for out in outs:
        for line in out.read_text().splitlines():
            if not line.strip():
                continue
            kind, *rest = line.split()
            kinds[kind] += 1
            if kind == "E":
                slots += [(rest[0], int(rest[1]) + k) for k in range(int(rest[2]))]
            elif kind == "R":
                rows.append(rest[0])
    s = MaskStore.open(root, FileKeyProvider(key_file))
    consumed = s.verify_integrity().consumed_total
    s.close()
    return slots, rows, kinds, consumed


@pytest.mark.timeout(600)
def test_many_processes_and_threads_never_reuse_a_mask(tmp_path: Path) -> None:
    slots, rows, kinds, consumed = _run(tmp_path, procs=4, threads=4, per_thread=12, initial=12, later=40,
                                        shared=True)
    dup = [k for k, v in Counter(slots).items() if v > 1]
    assert not dup, f"slots handed out twice: {dup[:5]}"
    assert len(rows) == len(set(rows)), "a mask row was handed out twice"
    assert consumed == len(slots), "the store consumed slots that were not handed out (or the reverse)"
    assert kinds["A"] > 0 and len(rows) > 500  # real contention happened
    assert kinds["F"] == 0, "a running session's extension failed: admission promised the same stock twice"


def test_shared_ledger_counts_other_processes(tmp_path: Path) -> None:
    """Two ledgers over one directory (two processes' view): stock promised by one is not admitted by the other;
    a closed session's entitlement is released; a dead process's file is ignored and removed."""

    class _W:  # a WindowedReservation stand-in (outstanding entitlement, closed flag)
        def __init__(self, n: int) -> None:
            self.outstanding, self.closed, self._on_close = n, False, None

        def close(self) -> None:
            self.closed, self.outstanding = True, 0
            if self._on_close is not None:
                self._on_close()

    a = EntitlementLedger(SharedLedger(tmp_path / "l"))
    b = EntitlementLedger(SharedLedger(tmp_path / "l"))
    b.shared.own = tmp_path / "l" / f"{os.getppid()}-x.json"  # another live process's file
    w = _W(80)
    a.shared.acquire()
    assert a.admit(100, 80)
    a.add(w)  # type: ignore[arg-type]
    a.shared.release()
    assert b.admit(100, 20) and not b.admit(100, 30)
    assert EntitlementLedger().admit(100, 30)  # per-process ledger: would promise the same stock twice
    w.close()
    assert b.admit(100, 100)
    dead = tmp_path / "l" / "999999999-1.json"
    dead.write_text('{"outstanding": 1000}')
    assert b.admit(100, 100) and not dead.exists()

