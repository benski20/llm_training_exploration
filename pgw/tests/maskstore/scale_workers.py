"""Subprocess entry points of the many-process stress test (``test_scale_stress.py``; importable under spawn).

``sessions``: several threads, each running gateway-style incremental sessions (``EntitlementLedger`` admission over a
cross-process ``SharedLedger``, ``WindowedReservation`` with whole chunks, background ahead reservations and
background retirement), taking rows step by step like a decoder, with ``MaskStore.gc`` running concurrently.
``writer``: a factory stand-in committing segments while the sessions consume.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from pathlib import Path

from pgw.core import RING_K40_A12, MaskExhaustedError
from pgw.maskstore import FileKeyProvider, MaskStore

LID = "0.qkv"
CHUNK = 8


def writer(root: str, key_file: str, segments: int, slots: int, pause_s: float) -> None:
    """Commit ``segments`` segments of ``slots`` slots, ``pause_s`` apart."""
    from .conftest import make_model, write_segment

    store = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0)
    model = make_model()
    for _ in range(segments):
        write_segment(store, model, slots, chunk_slots=8)
        time.sleep(pause_s)
    store.close()


def sessions(root: str, key_file: str, ledger_dir: str, threads: int, per_thread: int, seed: int, out: str,
             shared: bool = True, budget_s: float = 60.0) -> None:
    """Run ``threads x per_thread`` sessions; write ``E seg start count`` / ``R hash`` / ``F`` (a session whose
    extension failed mid-answer) / ``A`` (admitted) / ``N`` (refused) lines to ``out``."""
    from pgw.gateway.masks import EntitlementLedger, SharedLedger, WindowedReservation, chunk_entitlement

    store = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0)
    ledger = EntitlementLedger(SharedLedger(ledger_dir) if shared else None)
    admit_mu = threading.Lock()  # the runtime's per-process admission lock
    lines: list[str] = []
    out_mu = threading.Lock()
    stop = threading.Event()
    deadline = time.monotonic() + budget_s

    def gc_loop() -> None:
        while not stop.is_set():
            try:
                store.gc()
            except Exception:  # noqa: S110 - the stress test checks reuse; a failed gc only delays shredding
                pass
            stop.wait(0.05)

    def run(tid: int) -> None:
        rng = random.Random(seed * 1000 + tid)
        mine: list[str] = []
        done = 0
        while done < per_thread and time.monotonic() < deadline:
            prompt, max_tokens = rng.randint(1, 12), rng.randint(1, 40)
            first = prompt + CHUNK
            entitled = chunk_entitlement(first, prompt + max_tokens, CHUNK) + CHUNK
            with admit_mu:
                if ledger.shared is not None:
                    ledger.shared.acquire()
                try:
                    if not ledger.admit(store.available_slots("ab" * 32, RING_K40_A12), entitled):
                        mine.append("N")
                        w = None
                    else:
                        def extend(n: int, _sid: str = f"s{seed}.{tid}.{done}") -> object:
                            # every reservation is recorded when made (an ahead part burned after close included)
                            part = store.reserve("ab" * 32, RING_K40_A12, n, session_id=_sid)
                            mine.extend(f"E {e.segment_id} {e.start} {e.count}" for e in part.extents)
                            return part

                        res = extend(first)

                        w = WindowedReservation(res, extend, cap=entitled, window=CHUNK, ahead=True, chunked=True)
                        ledger.add(w)
                        mine.append("A")
                except MaskExhaustedError:
                    mine.append("N")
                    w = None
                finally:
                    if ledger.shared is not None:
                        ledger.shared.release()
            if w is None:
                time.sleep(rng.uniform(0.001, 0.02))
                continue
            done += 1
            try:
                steps = [prompt] + [1] * max_tokens  # a prompt pass, then one row per decode step
                for rows in steps:
                    w.ensure(LID, rows)
                    left = rows
                    while left:
                        n = min(left, w.rows_until_boundary(LID))
                        r, _ = w.take(LID, n)
                        mine.extend("R " + hashlib.sha256(row.numpy().tobytes()).hexdigest()[:24] for row in r)
                        left -= n
            except MaskExhaustedError:
                mine.append("F")
            finally:
                w.close()
        with out_mu:
            lines.extend(mine)

    gct = threading.Thread(target=gc_loop, daemon=True)
    gct.start()
    ts = [threading.Thread(target=run, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    stop.set()
    gct.join()
    store.close()
    Path(out).write_text("\n".join(lines) + "\n")
