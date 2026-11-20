"""Single-use mask prefetch (``MaskReservation.prefetch``): staged rows are exactly the rows the next
takes would derive, handed out once, never across an extent, zeroed and burned on close."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pgw.core import RING_K40_A12, ConfigError, MaskExhaustedError, RingParams
from pgw.gateway.masks import WindowedReservation
from pgw.maskstore import MaskStore

from .conftest import FP, ToyModel, lid, write_segment

K32C = RingParams.from_name("K32A12C27")
SEEDS = [bytes([i + 1]) * 32 for i in range(3)]
SIZES = [5, 7, 4]  # three segments: a 16-slot reservation has three extents


def _twin_stores(open_store: Any, tmp_path: Path, model: ToyModel, ring: RingParams) -> tuple[MaskStore, MaskStore]:
    """Two stores holding identical segments (same seeds): the same reservation in each."""
    stores = []
    for name in ("a", "b"):
        s = open_store(tmp_path / name)
        for seed, n in zip(SEEDS, SIZES, strict=True):
            write_segment(s, model, n, ring=ring, chunk_slots=3, seed=seed)
        stores.append(s)
    return stores[0], stores[1]


def _same_take(a: Any, b: Any) -> None:
    rows_a, ra, wra, ca = a
    rows_b, rb, wrb, cb = b
    assert rows_a == rows_b
    assert ra.dtype == rb.dtype and wra.dtype == wrb.dtype
    assert np.array_equal(ra.numpy(), rb.numpy()) and np.array_equal(wra.numpy(), wrb.numpy())
    assert (ca is None) == (cb is None)
    if ca is not None:
        # tokens name (segment id, linear); segment ids differ between the twin stores
        assert ca[0][1] == cb[0][1] and np.array_equal(ca[1], cb[1]) and np.array_equal(ca[2], cb[2])


@pytest.mark.parametrize("ring", [RING_K40_A12, K32C], ids=lambda r: r.name)
def test_prefetched_takes_equal_plain_takes(open_store: Any, tmp_path: Path, model: ToyModel,
                                            ring: RingParams) -> None:
    sa, sb = _twin_stores(open_store, tmp_path, model, ring)
    ra, rb = sa.reserve(FP, ring, 16), sb.reserve(FP, ring, 16)
    assert [e.count for e in ra.extents] == SIZES
    staged = 0
    # call sizes: decode (1), speculative verification (3), odd sizes that split stages
    plan = [1, 1, 3, 2, 1, 3, 1, 1, 2, 1]
    for step, rows in enumerate(plan):
        for spec in model.specs:
            q = spec.id
            if rb.remaining(q) == 0:
                continue
            ahead = (rows * 2) if step % 3 else rows - 1 or 1
            staged += rb.prefetch(q, ahead)
            # a stage never reaches past the extent the cursor is in
            assert rb.staged_rows(q) <= rb.rows_until_boundary(q)
            want = ra.take_checked(q, rows, step % 2 == 0)
            got = rb.take_checked(q, rows, step % 2 == 0)
            _same_take(got, want)
            assert ra.taken(q) == rb.taken(q) and ra.remaining(q) == rb.remaining(q)
    assert staged > 0
    # take() (full-width W r) served from a stage
    q = model.specs[0].id
    if rb.remaining(q):
        rb.prefetch(q, 1)
        r1, w1 = ra.take(q, 1)
        r2, w2 = rb.take(q, 1)
        assert w2.dtype == w1.dtype and np.array_equal(w1.numpy(), w2.numpy())
        assert np.array_equal(r1.numpy(), r2.numpy())
    ra.close()
    rb.close()


def test_stage_is_single_use_and_zeroed(open_store: Any, tmp_path: Path, model: ToyModel) -> None:
    sa, sb = _twin_stores(open_store, tmp_path, model, K32C)
    burned: dict[str, int] = {}
    for s, name in ((sa, "a"), (sb, "b")):
        orig = s._on_reservation_closed

        def spy(res: Any, n: int, orig: Any = orig, name: str = name) -> None:
            burned[name] = n
            orig(res, n)

        s._on_reservation_closed = spy  # type: ignore[method-assign]
    ra, rb = sa.reserve(FP, K32C, 16), sb.reserve(FP, K32C, 16)
    q = lid("0.qkv")
    assert rb.prefetch(q, 3) == 3
    assert rb.prefetch(q, 3) == 0  # one stage per linear at a time
    st = rb._stage[q.key]
    rows, r, _wr, _ = rb.take_checked(q, 1, False)
    assert rows == 1 and rb.staged_rows(q) == 2
    assert not st.r[:1].any() and not st.wr[:1].any()  # the handed-out rows' staged copies are gone
    assert st.r[1:].any()
    r[:] = 0  # the caller wipes its copy: the stage is not affected
    assert st.r[1:].any()
    for s in model.specs[1:]:  # other linears advance as usual
        ra.take_checked(s.id, 1, False)
        rb.take_checked(s.id, 1, False)
    ra.take_checked(q, 1, False)
    rb.prefetch(model.specs[1].id, 2)
    st2 = rb._stage[model.specs[1].id.key]
    left = rb.staged_rows(q) + rb.staged_rows(model.specs[1].id)
    assert left == 2 + 2
    ra.close()
    rb.close()
    assert rb.staged_burned == left
    assert not st.r.any() and not st.wr.any() and not st2.r.any() and not st2.wr.any()
    assert burned["a"] == burned["b"] == 16 - 1  # staged rows count as burned, like every unused row
    assert rb.prefetch(q, 1) == 0  # closed: nothing staged, no error
    assert rb.staged_rows(q) == 0


def test_stage_never_crosses_an_extent(open_store: Any, tmp_path: Path, model: ToyModel) -> None:
    _, sb = _twin_stores(open_store, tmp_path, model, RING_K40_A12)
    with sb.reserve(FP, RING_K40_A12, 16) as res:
        q = lid("1.down")
        res.take_checked(q, 3, False)
        assert res.prefetch(q, 10) == 2  # the first extent has 5 rows
        rows, *_ = res.take_checked(q, 4, True)
        assert rows == 2  # served from the stage; the call still stops at the boundary
        assert res.prefetch(q, 100) == 7  # whole second extent, nothing of the third
        assert res.rows_until_boundary(q) == 7
        for _ in range(7):
            res.take_checked(q, 1, False)
        assert res.prefetch(q, 100) == 4
        rows, *_ = res.take_checked(q, 100, False)
        assert rows == 4 and res.remaining(q) == 0
        assert res.prefetch(q, 1) == 0
        with pytest.raises(MaskExhaustedError):
            res.take_checked(q, 1, False)


def test_stage_budget_and_validation(open_store: Any, tmp_path: Path, model: ToyModel) -> None:
    _, sb = _twin_stores(open_store, tmp_path, model, RING_K40_A12)
    with sb.reserve(FP, RING_K40_A12, 16) as res:
        q = lid("0.gate_up")  # 16 in, 32 out: 384 bytes per staged row
        res.stage_max_bytes = 2 * 384  # type: ignore[misc]
        assert res.prefetch(q, 5) == 2
        assert res.prefetch(lid("0.o"), 1) == 0  # budget used up
        res.take_checked(q, 2, False)
        assert res.prefetch(lid("0.o"), 1) == 1  # handed out: budget back
        assert res.prefetch(q, 0) == 0 and res.prefetch(q, -1) == 0
        with pytest.raises(ConfigError):
            res.prefetch("9.qkv", 1)


def test_windowed_reservation_forwards_to_the_current_part(open_store: Any, tmp_path: Path,
                                                           model: ToyModel) -> None:
    sa, sb = _twin_stores(open_store, tmp_path, model, RING_K40_A12)
    wa = WindowedReservation(sa.reserve(FP, RING_K40_A12, 4), lambda n: sa.reserve(FP, RING_K40_A12, n),
                             cap=12, window=4)
    wb = WindowedReservation(sb.reserve(FP, RING_K40_A12, 4), lambda n: sb.reserve(FP, RING_K40_A12, n),
                             cap=12, window=4)
    q = lid("0.o")
    for _ in range(10):
        wa.ensure(q, 1)
        wb.ensure(q, 1)
        wb.prefetch(q, 3)
        _same_take(wb.take_checked(q, 1, True), wa.take_checked(q, 1, True))
    wa.close()
    wb.close()
    assert wb.prefetch(q, 1) == 0
