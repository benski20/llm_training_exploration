"""Mask stock across endpoints: the stock is bound to (pack fingerprint, ring, scheme), not to a host, and switching
endpoints changes nothing about single use or integrity (docs/operations/placement.md, "Mask stock").

The store API has no notion of an endpoint at all (``reserve(fingerprint, ring, slots)``): a reservation made for
trips to endpoint A and one made after a switch to endpoint B draw disjoint slots from one ledger, a reopened
store (a restart onto another endpoint) never hands out a consumed slot, and another fingerprint or ring gets
nothing. The end-to-end proof with two real endpoints is ``tests/sdk/test_routing.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pgw.core import RING_K40_A12
from pgw.core.errors import MaskExhaustedError
from pgw.sdk.regions import stock_compatible
from tests.maskstore.conftest import FP, FP2, kp, make_model, open_store, write_segment  # noqa: F401 (fixtures)


def _slots(res: Any) -> set[tuple[str, int]]:
    return {(e.segment_id, e.start + i) for e in res.extents for i in range(e.count)}


def test_reservations_across_endpoints_are_disjoint_and_survive_restart(open_store: Callable[..., Any],
                                                                        tmp_path: Any) -> None:
    store = open_store()
    write_segment(store, make_model(FP), 24, ring=RING_K40_A12)
    a = store.reserve(FP, RING_K40_A12, 8, session_id="trips-to-endpoint-a")
    b = store.reserve(FP, RING_K40_A12, 8, session_id="trips-to-endpoint-b")  # after a switch
    assert not _slots(a) & _slots(b) and len(_slots(a) | _slots(b)) == 16
    store.close()
    store2 = open_store()  # the client restarts the model on a third endpoint
    c = store2.reserve(FP, RING_K40_A12, 8, session_id="trips-to-endpoint-c")
    assert not _slots(c) & (_slots(a) | _slots(b))
    with pytest.raises(MaskExhaustedError):  # 24 slots, 24 consumed: never one twice
        store2.reserve(FP, RING_K40_A12, 1, session_id="trips-to-endpoint-a")


def test_other_fingerprint_or_ring_gets_nothing(open_store: Callable[..., Any]) -> None:
    from pgw.core import RING_K32_A12_C27

    store = open_store()
    write_segment(store, make_model(FP), 8, ring=RING_K40_A12)
    with pytest.raises(MaskExhaustedError):  # an endpoint serving another pack revision
        store.reserve(FP2, RING_K40_A12, 1)
    with pytest.raises(MaskExhaustedError):  # another ring
        store.reserve(FP, RING_K32_A12_C27, 1)
    assert store.reserve(FP, RING_K40_A12, 8).slots == 8


def test_routing_only_to_endpoints_serving_the_stock() -> None:
    stock = {"fingerprint": FP, "ring": "K40A12"}
    assert stock_compatible(stock, [FP, FP2], "K40A12")
    assert not stock_compatible(stock, [FP2], "K40A12")
