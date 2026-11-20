"""``pgw_fast.quantize_mask_pack_many`` is bit-identical to one ``quantize_mask_pack`` per item.

Items are the wire items of one round trip (every session's call of a linear, row blocks of a
prefill); large batches run on the persistent native pool, so these tests cover the pooled split
(rows of one item spread over tasks) as well as the inline path.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from pgw.core import native
from pgw.core.ring import RingParams

pytestmark = pytest.mark.skipif(
    not native.native_available("quantize_mask_pack", "quantize_mask_pack_many"),
    reason="pgw_fast.quantize_mask_pack_many not built",
)

RINGS = ["K32A12C27", "K40A12", "K48A20", "K32A8"]


def _mask(rng: np.random.Generator, shape: tuple[int, int], ring: RingParams) -> np.ndarray:
    return rng.integers(0, 1 << ring.k, size=shape, dtype=np.int64)


def _items(rng: np.random.Generator, ring: RingParams, shapes: list[tuple[int, int]]):  # type: ignore[no-untyped-def]
    xs, rs, wns = [], [], []
    for rows, n in shapes:
        x = rng.normal(0, 1, size=(rows, n)).astype(np.float32)
        if rows > 2:
            x[1] = 0
            x[2, : max(1, n // 50)] *= 300
        xs.append(x)
        rs.append(_mask(rng, (rows, n), ring))
        w8 = np.clip(np.rint(rng.normal(0, 40, size=(4, n))), -127, 127)
        wns.append(float(np.sqrt((w8 * w8).sum(axis=1)).max()))
    return xs, rs, wns


def _per_item(xs, rs, ring, wns, want):  # type: ignore[no-untyped-def]
    out = []
    for x, r, wn, w in zip(xs, rs, wns, want, strict=True):
        out.append(native.fast.quantize_mask_pack(x, r, ring.k, ring.a, ring.cap, wn, w))
    return out


def _assert_same(got, want) -> None:  # type: ignore[no-untyped-def]
    assert len(got) == len(want)
    for (p1, x1, s1), (p2, x2, s2) in zip(got, want, strict=True):
        assert isinstance(p1, bytes) and p1 == p2
        assert (x1 is None) == (x2 is None)
        if x1 is not None:
            assert x1.dtype == np.int64 and x1.shape == x2.shape and np.array_equal(x1, x2)
        assert s1.dtype == np.float64 and np.array_equal(s1.view(np.uint64), s2.view(np.uint64))


SHAPES = [
    [(1, 3584)] * 4,  # decode, 4 sessions (inline)
    [(1, 3584)] * 16,  # decode, 16 sessions (pooled)
    [(1, 18944)] * 6,  # wide down-projection inputs (pooled)
    [(40, 896), (1, 896), (3, 896)],  # prefill next to decode and a speculative verification
    [(64, 1024)],  # one prefill call split over the pool
    [(5, 7), (0, 7), (2, 7)],  # tiny and empty items
]


@pytest.mark.parametrize("ring_name", RINGS)
@pytest.mark.parametrize("shapes", SHAPES, ids=lambda s: f"{len(s)}x{s[0]}")
def test_matches_per_item(ring_name: str, shapes: list[tuple[int, int]]) -> None:
    ring = RingParams.from_name(ring_name)
    rng = np.random.default_rng(len(shapes) * 1000 + shapes[0][1])
    xs, rs, wns = _items(rng, ring, shapes)
    for want in ([True] * len(xs), [False] * len(xs), [i % 2 == 0 for i in range(len(xs))]):
        got = native.fast.quantize_mask_pack_many(xs, rs, ring.k, ring.a, ring.cap, wns, want)
        _assert_same(got, _per_item(xs, rs, ring, wns, want))


def test_non_contiguous_and_inputs_untouched() -> None:
    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(5)
    xb = rng.normal(size=(8, 600)).astype(np.float32)
    rb = _mask(rng, (8, 600), ring)
    xs, rs = [xb[::2, ::3], xb[1::2]], [rb[::2, ::3], rb[1::2]]
    copies = [a.copy() for a in xs + rs]
    got = native.fast.quantize_mask_pack_many(xs, rs, ring.k, ring.a, ring.cap, [900.0, 900.0], [True, False])
    want = _per_item([np.ascontiguousarray(a) for a in xs], [np.ascontiguousarray(a) for a in rs], ring,
                     [900.0, 900.0], [True, False])
    _assert_same(got, want)
    assert all(np.array_equal(a, c) for a, c in zip(xs + rs, copies, strict=True))


def test_errors_are_the_first_failing_items() -> None:
    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(9)
    xs, rs, _ = _items(rng, ring, [(1, 3584)] * 16)
    wns = [300.0] * 16
    xs[9] = xs[9].copy()
    xs[9][0, 3] = np.nan  # non-finite in item 9
    wns[5] = 2.0**40  # no bit width fits in item 5: the first failing item wins
    with pytest.raises(ValueError, match=r"^ring_overflow: K32A12C27"):
        native.fast.quantize_mask_pack_many(xs, rs, ring.k, ring.a, ring.cap, wns, [False] * 16)
    wns[5] = 300.0
    with pytest.raises(ValueError, match=r"^ring_overflow: non-finite"):
        native.fast.quantize_mask_pack_many(xs, rs, ring.k, ring.a, ring.cap, wns, [False] * 16)
    # inside one (pooled) item, a non-finite row outranks a cap overflow in an earlier row
    x = rng.normal(size=(64, 1024)).astype(np.float32)
    x[60, 0] = np.inf
    r = _mask(rng, x.shape, ring)
    with pytest.raises(ValueError, match=r"^ring_overflow: non-finite"):
        native.fast.quantize_mask_pack_many([x], [r], ring.k, ring.a, ring.cap, [2.0**40], [True])
    with pytest.raises(ValueError, match=r"^ring_overflow: non-finite"):
        native.fast.quantize_mask_pack(x, r, ring.k, ring.a, ring.cap, 2.0**40)


def test_argument_errors() -> None:
    x = np.ones((2, 8), np.float32)
    r = np.zeros((2, 8), np.int64)
    f = native.fast.quantize_mask_pack_many
    assert f([], [], 32, 12, 27, [], []) == []
    for args in [
        ([x], [r, r], [1.0], [True]),
        ([x], [r], [1.0, 2.0], [True]),
        ([x], [r[:, :7]], [1.0], [True]),
        ([x.astype(np.float64)], [r], [1.0], [True]),
        ([x], [r.astype(np.uint64)], [1.0], [True]),
    ]:
        with pytest.raises(ValueError) as ei:
            f(args[0], args[1], 32, 12, 27, args[2], args[3])
        assert not str(ei.value).startswith("ring_overflow:")
    with pytest.raises(ValueError):
        f([x], [r], 33, 12, 0, [0.0], [True])


def test_concurrent_callers_share_the_pool() -> None:
    """Python threads calling at once (the GIL is released): every result stays exact."""
    ring = RingParams.from_name("K32A12C27")
    assert native.fast.prepare_pool_workers() >= 0
    rng = np.random.default_rng(11)
    batches = [_items(rng, ring, [(1, 3584)] * 16 + [(20, 896)]) for _ in range(4)]
    wants = [_per_item(xs, rs, ring, wns, [False] * len(xs)) for xs, rs, wns in batches]
    errors: list[BaseException] = []

    def run(i: int) -> None:
        xs, rs, wns = batches[i]
        try:
            for _ in range(20):
                _assert_same(native.fast.quantize_mask_pack_many(xs, rs, ring.k, ring.a, ring.cap, wns,
                                                                 [False] * len(xs)), wants[i])
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
