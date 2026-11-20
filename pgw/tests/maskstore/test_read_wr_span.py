"""``W r`` reads whose rows cross chunk boundaries: one output, native unpack per piece, pieces wiped."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import RING_K32_A8, RING_K40_A12, RING_K48_A20, RingParams
from pgw.core import native as core_native
from pgw.maskstore import MaskStore
from pgw.maskstore import store as store_mod
from tests.maskstore.conftest import FP, ToyModel, write_segment

pytestmark = pytest.mark.skipif(not core_native.native_available("unpack_rows_u32"), reason="pgw_fast not built")


def _takes(res: Any, model: ToyModel, ring: RingParams, compact: bool) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    out = []
    for s in model.specs:
        for n in (1, 6, 4):  # 1..7 crosses the chunk boundary at slot 4 and 8 (chunk_slots=4)
            if compact:
                rows, r, wr, _ = res.take_checked(s.id, n, False)
                assert rows == n
            else:
                r, wr = res.take(s.id, n)
            out.append((s.id.key, r.clone(), wr.clone()))
    return out


@pytest.mark.parametrize("ring", [RING_K32_A8, RING_K40_A12, RING_K48_A20], ids=lambda r: r.name)
@pytest.mark.parametrize("compact", [False, True])
def test_spanning_reads_exact(store: MaskStore, model: ToyModel, ring: RingParams, compact: bool) -> None:
    write_segment(store, model, 11, ring=ring, chunk_slots=4, write_rows=3)
    with store.reserve(FP, ring, 11) as res:
        got = _takes(res, model, ring, compact)
    for key, r, wr in got:
        want_dtype = torch.uint32 if compact and ring.k <= 32 else torch.int64
        assert wr.dtype == want_dtype
        assert torch.equal(wr.to(torch.int64), model.wr(key, r, ring))


@pytest.mark.parametrize("ring", [RING_K32_A8, RING_K40_A12], ids=lambda r: r.name)
def test_native_matches_python_fallback(open_store: Any, tmp_path: Path, model: ToyModel, ring: RingParams,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    seed = bytes(range(32))
    store = open_store(tmp_path / "a")
    write_segment(store, model, 11, ring=ring, chunk_slots=4, write_rows=3, seed=seed)
    with store.reserve(FP, ring, 11) as res:
        native_rows = _takes(res, model, ring, True)
    store = open_store(tmp_path / "b")  # same seed in a second store (one store refuses a reused seed)
    write_segment(store, model, 11, ring=ring, chunk_slots=4, write_rows=3, seed=seed)
    monkeypatch.setattr(store_mod, "_native_unpack", False)
    with store.reserve(FP, ring, 11) as res:
        python_rows = _takes(res, model, ring, True)
    for (k1, r1, w1), (k2, r2, w2) in zip(native_rows, python_rows, strict=True):
        assert k1 == k2 and torch.equal(r1, r2)
        assert torch.equal(w1.to(torch.int64), w2.to(torch.int64))


def test_spanning_pieces_are_wiped(store: MaskStore, model: ToyModel, monkeypatch: pytest.MonkeyPatch) -> None:
    ring = RING_K32_A8
    write_segment(store, model, 11, ring=ring, chunk_slots=4, write_rows=3)
    fast = core_native.fast
    pieces: list[np.ndarray] = []
    real = fast.unpack_rows_u32

    def spy(*a: Any) -> np.ndarray:
        arr = real(*a)
        pieces.append(arr)
        return arr

    monkeypatch.setattr(fast, "unpack_rows_u32", spy)
    with store.reserve(FP, ring, 11) as res:
        res.take_checked(model.specs[0].id, 2, False)  # slots 0-1
        pieces.clear()
        rows, r, wr, _ = res.take_checked(model.specs[0].id, 5, False)  # slots 2-6: pieces 2-3 and 4-6
        assert rows == 5
        assert len(pieces) == 2
        assert all(not p.any() for p in pieces)  # temporaries zeroed after the copy
        assert wr.numpy().any()
        assert torch.equal(wr.to(torch.int64), model.wr(model.specs[0].id.key, r, ring))
        pieces.clear()
        _, _r1, wr1, _ = res.take_checked(model.specs[0].id, 1, False)  # one chunk: the piece is the result
        assert len(pieces) == 1 and np.shares_memory(pieces[0], wr1.numpy())


def test_spanning_read_failure_wipes_partial_output(store: MaskStore, model: ToyModel,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    ring = RING_K32_A8
    write_segment(store, model, 11, ring=ring, chunk_slots=4, write_rows=3)
    fast = core_native.fast
    real = fast.unpack_rows_u32
    calls: list[int] = []
    dests: list[np.ndarray] = []
    real_empty = np.empty

    def flaky(*a: Any) -> np.ndarray:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*a)

    def spy_empty(*a: Any, **k: Any) -> np.ndarray:
        arr = real_empty(*a, **k)
        dests.append(arr)
        return arr

    monkeypatch.setattr(fast, "unpack_rows_u32", flaky)
    monkeypatch.setattr(store_mod.np, "empty", spy_empty)
    with store.reserve(FP, ring, 11) as res:
        res.take_checked(model.specs[0].id, 2, False)
        calls.clear()
        dests.clear()
        with pytest.raises(RuntimeError):
            res.take_checked(model.specs[0].id, 5, False)
    out_m = model.specs[0].out_features
    mine = [d for d in dests if d.shape == (5, out_m) and d.dtype == np.uint32]
    assert mine and all(not d.any() for d in mine)
