"""Subprocess entry points for crash and multi-process tests (must be importable under spawn)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from pgw.core import RING_K40_A12, MaskExhaustedError
from pgw.maskstore import FileKeyProvider, MaskStore

FP = "ab" * 32


def crash_during_reserve(root: str, key_file: str, slots: int, point: str, overrides: dict[str, Any]) -> None:
    """Reserve, then die abruptly (``os._exit``) at fault point ``point``."""
    store = MaskStore.open(root, FileKeyProvider(key_file), **overrides)

    def hook(p: str) -> None:
        if p == point:
            os._exit(9)

    store._fault_hook = hook
    store.reserve(FP, RING_K40_A12, slots)
    os._exit(0)  # pragma: no cover - only reached if the fault point never fired


def crash_holding_reservation(root: str, key_file: str, slots: int, ready: str) -> None:
    """Reserve, signal the parent, then die without closing (lease left behind)."""
    store = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0)
    res = store.reserve(FP, RING_K40_A12, slots)
    res.take("0.qkv", 1)
    Path(ready).write_text(res.id)
    os._exit(7)


def reserve_loop(root: str, key_file: str, n: int, max_slots: int, seed: int, out: str) -> None:
    """Reserve random sizes until exhausted or ``n`` done; record extents and a hash of each r row."""
    import random

    rng = random.Random(seed)
    store = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0, verify_on_open=True)
    lines: list[str] = []
    for _ in range(n):
        k = rng.randint(1, max_slots)
        try:
            res = store.reserve(FP, RING_K40_A12, k, session_id=f"w{seed}")
        except MaskExhaustedError:
            break
        with res:
            for e in res.extents:
                lines.append(f"E {e.segment_id} {e.start} {e.count}")
            left = res.slots
            while left:
                take = min(left, res.rows_until_boundary("0.qkv"))
                r, _ = res.take("0.qkv", take)
                for row in r:
                    lines.append("R " + hashlib.sha256(row.numpy().tobytes()).hexdigest()[:24])
                left -= take
    store.close()
    Path(out).write_text("\n".join(lines) + "\n")
