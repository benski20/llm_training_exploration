"""Chat-speed WP-A4b: speculation padding rows in :class:`MaskedBackend` (fixed-R verify passes).

``LinearCall.meta["pad_rows"] = p`` appends ``p`` padding rows to the call's wire item: uniform random
ring elements (indistinguishable from ``xq + r mod 2^K``), no mask row taken, outputs dropped after a
dummy finish that still verifies them against the item's check vectors. ``meta["dummy"] = True`` makes
every row padding (a mask-free dummy pass on the real code path; result all zeros).

* the item row count is real + padding (== R), padding takes zero masks, real rows == plain bit for bit;
* on the TripEngine, today's native kernels and the pure-Python path; sync and async; across an extent
  boundary; next to unpadded calls of other sessions;
* a wrong result on a padding row fails the Freivalds check like a real row.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import IntegrityError, ShapeError, native
from pgw.core.ring import DEFAULT_RING
from pgw.core.types import LinearId, LinearKind
from pgw.engine import (
    InMemoryPack,
    LinearCall,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
)
from pgw.engine.backends import DUMMY, PAD_ROWS
from tests.engine.conftest import TwoExtentReservation

QKV = LinearId(0, LinearKind.QKV)
DOWN = LinearId(1, LinearKind.DOWN)
PATHS = ["trip", "native", "python"]


def _backend(pack: Any, client: Any, path: str, **kw: Any) -> MaskedBackend:
    if path == "trip" and not native.native_available("trip_prepare_into"):
        pytest.skip("pgw_fast trip kernels not built")
    be = MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint, trip_engine=path == "trip", **kw)
    if path == "python":
        be._native_prepare = be._native_finish = be._native_group = False
        be.prepare_many = False
    return be


def _x(rows: int, n: int, seed: int) -> torch.Tensor:
    return torch.randn(rows, n, generator=torch.Generator().manual_seed(seed)) * 2


def _n(pack: Any, lid: LinearId) -> int:
    return int(pack.load_linear(lid).w8.shape[1])


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("lid", [QKV, DOWN], ids=str)
def test_padding_rows_take_no_masks(qwen_pack: InMemoryPack, path: str, lid: LinearId) -> None:
    client = LocalComputeClient(qwen_pack, record=True)
    be = _backend(qwen_pack, client, path)
    plain = PlainBackend(qwen_pack, DEFAULT_RING)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 16)
    n = _n(qwen_pack, lid)
    for real, pad in ((1, 4), (3, 2), (5, 0)):
        before = res.remaining(lid)
        x = _x(real, n, real)
        y = be.forward(LinearCall(lid, x, res, {PAD_ROWS: pad}))
        assert torch.equal(y, plain.forward(LinearCall(lid, x)))  # real rows: exact
        assert res.remaining(lid) == before - real  # padding took no mask row
        item_lid, x_hat = client.received[-1]
        assert item_lid == lid.key and x_hat.shape == (real + pad, n)  # one item of R = real + pad rows
        assert int(x_hat.min()) >= 0 and int(x_hat.max()) < DEFAULT_RING.modulus
    assert be.stats.pad_rows == 6 and be.stats.integrity_failures == 0


def test_padding_is_uniform_and_fresh(qwen_pack: InMemoryPack) -> None:
    client = LocalComputeClient(qwen_pack, record=True)
    be = _backend(qwen_pack, client, "trip")
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 16)
    n = _n(qwen_pack, QKV)
    pads = []
    for _ in range(2):
        be.forward(LinearCall(QKV, _x(1, n, 0), res, {PAD_ROWS: 7}))
        pads.append(client.received[-1][1][1:].numpy())
    a, b = pads
    assert not np.array_equal(a, b)  # fresh randomness per pass
    vals = np.concatenate([a, b]).ravel().astype(np.float64) / DEFAULT_RING.modulus
    assert abs(vals.mean() - 0.5) < 0.05 and abs(vals.var() - 1 / 12) < 0.02  # uniform on [0, 2^K)
    # the top bit is set about half the time, like a masked row's
    top = (np.concatenate([a, b]).ravel() >> (DEFAULT_RING.k - 1)) & 1
    assert 0.4 < top.mean() < 0.6


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("session", [True, False], ids=["session", "no-session"])
def test_dummy_call(qwen_pack: InMemoryPack, path: str, session: bool) -> None:
    client = LocalComputeClient(qwen_pack, record=True)
    be = _backend(qwen_pack, client, path)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8) if session else None
    n = _n(qwen_pack, QKV)
    m = qwen_pack.load_linear(QKV).w8.shape[0]
    y = be.forward(LinearCall(QKV, _x(4, n, 1), res, {DUMMY: True}))
    assert torch.equal(y, torch.zeros(4, m)) and y.dtype == torch.float32
    assert client.received[-1][1].shape == (4, n)
    if res is not None:
        # fixed-structure path (L3): a dummy call takes one mask row like a real call's take; else mask-free
        assert res.remaining(QKV) == 8 - (1 if be.fixed_pad else 0)
    y = be.forward(LinearCall(QKV, _x(2, n, 2), res, {DUMMY: True, PAD_ROWS: 3}))  # R = 5, result: own rows
    assert y.shape == (2, m) and client.received[-1][1].shape == (5, n)
    assert be.stats.dummy_calls == 2


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_mixed_batch(qwen_pack: InMemoryPack, path: str, mode: str) -> None:
    """Padded, unpadded and dummy calls of three sessions in one round trip: item order = call order."""
    client = LocalComputeClient(qwen_pack, record=True)
    be = _backend(qwen_pack, client, path)
    plain = PlainBackend(qwen_pack, DEFAULT_RING)
    r1, r2, r3 = (LocalMaskReservation(qwen_pack, DEFAULT_RING, 8) for _ in range(3))
    n = _n(qwen_pack, QKV)
    calls = [LinearCall(QKV, _x(2, n, 1), r1, {PAD_ROWS: 2}), LinearCall(QKV, _x(1, n, 2), r2),
             LinearCall(QKV, _x(3, n, 3), r3, {DUMMY: True}), LinearCall(QKV, _x(0, n, 4), r1, {PAD_ROWS: 3})]
    out = be.forward_many(calls) if mode == "sync" else asyncio.run(be.aforward_many(calls))
    assert torch.equal(out[0], plain.forward(calls[0])) and torch.equal(out[1], plain.forward(calls[1]))
    assert not out[2].any() and out[2].shape[0] == 3 and out[3].shape[0] == 0
    assert [x.shape[0] for _, x in client.received[-4:]] == [4, 1, 3, 3]
    assert (r1.remaining(QKV), r2.remaining(QKV), r3.remaining(QKV)) == (6, 7, 8 - (1 if be.fixed_pad else 0))


@pytest.mark.parametrize("path", PATHS)
def test_padding_across_an_extent_boundary(qwen_pack: InMemoryPack, path: str) -> None:
    client = LocalComputeClient(qwen_pack, record=True)
    be = _backend(qwen_pack, client, path)
    res = TwoExtentReservation(qwen_pack, DEFAULT_RING, 2, 8)
    n = _n(qwen_pack, QKV)
    x = _x(3, n, 5)
    y = be.forward(LinearCall(QKV, x, res, {PAD_ROWS: 4}))
    assert torch.equal(y, PlainBackend(qwen_pack, DEFAULT_RING).forward(LinearCall(QKV, x)))
    # L3: one item of real + padding rows; where the real rows cross the extent (it depends on how many masks
    # the session used, which with padding depends on the drafts) is not visible on the wire
    assert [x.shape[0] for _, x in client.received[-1:]] == [3 + 4]
    assert res.boundaries_seen[-2:] == [2, 1]  # the masks still came from both extents
    assert res.remaining(QKV) == 10 - 3


@pytest.mark.parametrize("path", PATHS)
def test_wrong_result_on_a_padding_row_fails(qwen_pack: InMemoryPack, path: str) -> None:
    hit: dict[str, int] = {"row": -1}

    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        if hit["row"] >= 0:
            y = y.clone()
            y[hit["row"], 0] += 1
        return y

    client = LocalComputeClient(qwen_pack, tamper=tamper)
    be = _backend(qwen_pack, client, path)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8)
    n = _n(qwen_pack, QKV)
    hit["row"] = 3  # rows 0..1 real, 2..4 padding
    with pytest.raises(IntegrityError) as ei:
        be.forward(LinearCall(QKV, _x(2, n, 1), res, {PAD_ROWS: 3}))
        be.join_checks()
    # L3 fixed-structure path: one Freivalds pass over all rows; the error does not say which rows were padding
    assert ei.value.details.get("linear") == QKV.key
    assert be.stats.integrity_failures == 1


def test_bad_meta(qwen_pack: InMemoryPack) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8)
    n = _n(qwen_pack, QKV)
    from pgw.core import ConfigError

    with pytest.raises(ConfigError):
        be.forward(LinearCall(QKV, _x(1, n, 1), res, {PAD_ROWS: -1}))
    with pytest.raises(ShapeError):
        be.forward(LinearCall(QKV, torch.zeros(1, n + 1), res, {PAD_ROWS: 1}))
    assert res.remaining(QKV) == 8  # nothing taken
