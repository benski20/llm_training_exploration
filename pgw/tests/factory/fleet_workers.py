"""Subprocess entry points for the factory fleet tests (importable under ``spawn``)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from pgw.core import RING_K40_A12, MaskExhaustedError
from pgw.maskstore import FileKeyProvider, MaskStore


def write_loop(root: str, key_file: str, segments: int, slots: int, seed: int, out: str) -> None:
    """Write ``segments`` correct segments (the toy model of tests/maskstore) as one factory worker would."""
    from tests.maskstore.conftest import make_model, write_segment

    model = make_model()
    store = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0)
    ids = [write_segment(store, model, slots, chunk_slots=4, write_rows=5) for _ in range(segments)]
    store.close()
    Path(out).write_text("\n".join(ids) + "\n")


def reserve_until(root: str, key_file: str, total: int, seed: int, out: str, deadline_s: float) -> None:
    """Reserve random sizes while writers are still producing; record every extent handed out."""
    import random

    from tests.maskstore.conftest import FP

    rng = random.Random(seed)
    store = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0)
    lines: list[str] = []
    got = 0
    end = time.monotonic() + deadline_s
    while got < total and time.monotonic() < end:
        try:
            res = store.reserve(FP, RING_K40_A12, rng.randint(1, 7), session_id=f"r{seed}")
        except MaskExhaustedError:
            time.sleep(0.01)
            continue
        with res:
            for e in res.extents:
                lines.append(f"{e.segment_id} {e.start} {e.count}")
            got += res.slots
            r, wr = res.take("0.qkv", min(res.slots, res.rows_until_boundary("0.qkv")))
            assert r.shape[0] == wr.shape[0]
    store.close()
    Path(out).write_text("\n".join(lines) + ("\n" if lines else ""))


def hold_partial(root: str, key_file: str, ready: str, go: str) -> None:
    """Start a segment, write part of it, signal, wait for ``go`` to appear, then finish and commit."""
    from tests.maskstore.conftest import make_model, write_segment

    model = make_model()
    store = MaskStore.open(root, FileKeyProvider(key_file), low_water_slots=0)
    orig = store.writer

    class _Slow:
        def __init__(self, w: Any):
            self.w = w
            self.n = 0

        def __getattr__(self, k: str) -> Any:
            return getattr(self.w, k)

        def write_linear(self, *a: Any) -> None:
            self.w.write_linear(*a)
            self.n += 1
            if self.n == 2:
                Path(ready).write_text(str(os.getpid()))
                while not Path(go).exists():
                    time.sleep(0.02)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, et: Any, ev: Any, tb: Any) -> None:
            if et is None:
                self.w.commit()
            else:
                self.w.abort()

    store.writer = lambda *a, **kw: _Slow(orig(*a, **kw))  # type: ignore[method-assign]
    seg = write_segment(store, model, 12, chunk_slots=4, write_rows=4)
    store.close()
    Path(go + ".done").write_text(seg)
