"""Trips of several answers on the native per-trip path (``PGW_TRIP_NATIVE_MULTI``): one lease per call, prepared
and finished in one native call each (``trip_prepare_many`` / ``trip_finish_many``, on the native pool for large
trips) or per lease. Against the batch path (``_prepare_trip``) on real store reservations: masked == plain bit for
bit, the same provider payloads in the same order, the same noise statistics; a wrong result in any item fails
the trip; nothing is prepared before every call's lease exists. Default off (the SDK turns it on).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import native
from pgw.core.errors import IntegrityError, RingOverflowError
from pgw.core.ring import DEFAULT_RING
from pgw.engine import InMemoryPack, LocalComputeClient, MaskedBackend
from pgw.engine.backends import LinearCall
from tests.engine import test_trip_native as tn
from tests.engine.test_trip_engine import _plain_ref, _same

pytestmark = pytest.mark.skipif(not native.native_available("TripLease", "trip_prepare_into"),
                                reason="pgw_fast native trip lease not built")

store_template = tn.store_template


def _run(pack: Any, template: Path, monkeypatch: pytest.MonkeyPatch, multi: bool, mode: str = "sync",
         noise_async: bool = False) -> tuple[Any, Any, Any, int, dict[str, Any]]:
    monkeypatch.setenv("PGW_TRIP_NATIVE_MULTI", "1" if multi else "0")
    served: list[int] = []
    real = MaskedBackend._prepare_trip_leases

    def spy(self: Any, calls: Any) -> Any:
        p = real(self, calls)
        served.append(0 if p is None else 1)
        return p

    monkeypatch.setattr(MaskedBackend, "_prepare_trip_leases", spy)
    out, sent, stats, (_, _, snap) = tn._scenario(pack, template, monkeypatch, True, True, noise_async, mode=mode)
    monkeypatch.setattr(MaskedBackend, "_prepare_trip_leases", real)
    return out, sent, stats, sum(served), snap


def test_default_off(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_TRIP_NATIVE_MULTI", raising=False)
    monkeypatch.setenv("PGW_TRIP_FAST", "1")
    monkeypatch.setenv("PGW_TRIP_NATIVE", "1")
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       trip_engine=True)
    assert be.trip_native and not be.trip_native_multi


@pytest.mark.parametrize(("mode", "noise_async", "many"), [("sync", False, "native"), ("sync", True, "native"),
                                                          ("async", False, "native"), ("sync", False, "parallel"),
                                                          ("sync", False, "python")])
def test_multi_equals_plain_and_bytes_identical(qwen_pack: InMemoryPack, store_template: Path,
                                                monkeypatch: pytest.MonkeyPatch, mode: str, noise_async: bool,
                                                many: str) -> None:
    ref = _plain_ref(qwen_pack)
    base_out, base_sent, base_stats, base_served, _ = _run(qwen_pack, store_template, monkeypatch, False, mode,
                                                           noise_async)
    _same(ref, base_out)
    assert base_served == 0
    if many == "python":  # the per-lease loop (an extension without the batched calls)
        monkeypatch.delattr(native.fast, "trip_prepare_many")
        monkeypatch.delattr(native.fast, "trip_finish_many")
    elif many == "parallel":  # every several-answer trip on the native pool
        monkeypatch.setattr(MaskedBackend, "trip_many_parallel_elems", 1)
    out, sent, stats, served, snap = _run(qwen_pack, store_template, monkeypatch, True, mode, noise_async)
    _same(ref, out)  # masked == plain, bit for bit
    assert served > 0  # the two-answer decode steps took one lease per answer
    if mode == "async":  # concurrent pipelined blocks reach the recording client in either order (also run to run)
        assert sorted(sent) == sorted(base_sent)
    else:
        assert sent == base_sent  # byte-identical provider payloads, same order
    assert stats == base_stats  # same noise draws and statistics
    assert snap["integrity_failures"] == 0


@pytest.fixture
def two(qwen_pack: InMemoryPack, store_template: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    tn._env(monkeypatch, True, True)
    monkeypatch.setenv("PGW_TRIP_NATIVE_MULTI", "1")
    st = tn._Store(store_template, qwen_pack)
    client = LocalComputeClient(qwen_pack, record=True)
    model = tn._masked(qwen_pack, client=client, trip_engine=True, sync_trips=True)
    model.backend.freivalds_rate = 1.0
    try:
        yield st, model.backend, client
    finally:
        model.close()
        st.close()


def _calls(be: Any, st: Any, n: int = 3, seed: int = 0) -> list[LinearCall]:
    spec = be.pack.linear_specs()[0]
    out = []
    for i in range(n):
        res = st.windowed(40)
        res.ensure(spec.id, 1)
        x = torch.randn(1, spec.in_features, generator=torch.Generator().manual_seed(seed + i))
        out.append(LinearCall(spec.id, x, res))
    return out


@pytest.mark.parametrize("parallel", [False, True])
def test_tampered_item_fails_the_trip(two: Any, monkeypatch: pytest.MonkeyPatch, parallel: bool) -> None:
    st, be, client = two
    if parallel:
        monkeypatch.setattr(MaskedBackend, "trip_many_parallel_elems", 1)
    calls = _calls(be, st)
    be.forward_many(calls)  # leases attached, every row correct
    calls = _calls(be, st, seed=10)
    prep = be._prepare(calls)
    assert prep.tickets is not None and len(prep.tickets) == 3
    resp = client.compute_sync(prep.request) if hasattr(client, "compute_sync") else None
    if resp is None:
        from pgw.engine.backends import _run_coro_blocking

        resp = _run_coro_blocking(client.compute(prep.request))
    items = list(resp.items)
    y = bytearray(items[1].y_hat)
    y[0] ^= 1
    items[1] = type(items[1])(items[1].linear, items[1].rows, bytes(y))
    resp = type(resp)(resp.request_id, tuple(items), resp.timing)
    with pytest.raises(IntegrityError):
        be._finish(prep, resp, 0.0)
        be.join_checks()  # a deferred check reports at the join
    assert be.stats.snapshot()["integrity_failures"] >= 1


def test_nonfinite_takes_no_row(two: Any) -> None:
    st, be, _ = two
    calls = _calls(be, st)
    be.forward_many(calls)
    pos = [c.session.trip_leases[c.id.key].pos for c in calls]
    calls2 = [LinearCall(c.id, c.x.clone(), c.session) for c in calls]
    calls2[2].x[0, 0] = float("nan")
    with pytest.raises(RingOverflowError):
        be._prepare(calls2)
    assert [c.session.trip_leases[c.id.key].pos for c in calls] == pos  # no session's row taken


def test_prepare_many_validates_before_taking(two: Any) -> None:
    st, be, _ = two
    calls = _calls(be, st)
    be.forward_many(calls)
    leases = [c.session.trip_leases[c.id.key] for c in calls]
    pos = [ls.pos for ls in leases]
    xs = [np.ascontiguousarray(c.x.numpy()) for c in calls]
    with pytest.raises(ValueError, match="twice"):
        native.fast.trip_prepare_many([leases[0], leases[0]], xs[:2], bytes(2), [None, None])
    with pytest.raises(ValueError):
        native.fast.trip_prepare_many(leases, [xs[0], xs[1], xs[2][:, :3].copy()], bytes(3), [None] * 3)
    assert [ls.pos for ls in leases] == pos
