"""Trip leases prepared ahead (``PGW_TRIP_LEASE_AHEAD``; ``MaskReservation.lease_ahead``).

The next native trip lease of a linear is built on the lease-ahead thread (the ``W r`` chunks decrypted ahead, the
extent's soundness check, the native ``TripLease``) instead of on the plaintext thread when the attached lease runs
out. Proven here:

Prepared only for several-answer trips (the per-trip path of several answers and the native decode driver's
several-answer passes), where the plaintext thread is the limit; these tests prepare ahead on every trip.

* exact: payloads (every byte, in order), outputs and check statistics equal the path without it, for ring and p31
  extents, plain and incremental (chunked) windowed reservations, across chunk, extent, part and segment boundaries;
  the prepared leases are actually used;
* single use: a lease prepared ahead hands out no row before it is attached, is attached only at its first row and
  under the same soundness policy, and is closed (zeroed) when another path takes the rows, when the policy
  changes, and at close; every ``W r`` chunk it held is zero after the run;
* admission: preparing ahead never reserves (no window ``ensure``, no extension): the session's reserved slots,
  extensions and taken rows are those of the path without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import native
from pgw.core.ring import DEFAULT_RING
from pgw.engine import InMemoryPack, LocalComputeClient, MaskedBackend
from pgw.engine.backends import LinearCall
from tests.engine.test_trip_native import SLOTS, _env, _Store, _x, p31_template, store_template  # noqa: F401

pytestmark = pytest.mark.skipif(not native.native_available("TripLease", "trip_prepare_into", "chunk_open_into"),
                                reason="pgw_fast native trip lease not built")


def _drain(be: MaskedBackend) -> None:
    """Wait until the lease-ahead thread has run every job queued so far (one worker, in order)."""
    pool = be._ahead_pool
    if pool is not None:
        pool.submit(lambda: None).result()


def _flags(monkeypatch: pytest.MonkeyPatch, ahead: bool) -> None:
    from pgw.maskstore import store as store_mod

    monkeypatch.setattr(store_mod, "TRIP_DECRYPT_AHEAD", True)
    monkeypatch.setattr(store_mod, "TRIP_LEASE_AHEAD", ahead)
    # the backend prepares ahead for several-answer trips only; here every trip of these one-call runs does
    real = MaskedBackend._trip_lease_for

    def every(self: Any, res: Any, leases: Any, lid: Any, meta: Any, t: int, ahead: bool = False) -> Any:
        return real(self, res, leases, lid, meta, t, True)

    monkeypatch.setattr(MaskedBackend, "_trip_lease_for", every)


class _Chunks:
    """Records every ``W r`` chunk handed to a native ``TripLease`` (to check they are zero afterwards)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bufs: list[bytearray] = []
        self.leases: list[Any] = []
        real = native.fast.TripLease

        def make(*a: Any) -> Any:
            for b, _, _ in a[8]:
                self.bufs.append(b)
            lease = real(*a)
            self.leases.append(lease)
            return lease

        monkeypatch.setattr(native.fast, "TripLease", make)


def _run(pack: InMemoryPack, template: Path, monkeypatch: pytest.MonkeyPatch, ahead: bool, *, chunked: bool,
         p31: bool, tokens: int, wait: bool = True) -> tuple[list[torch.Tensor], list[bytes], dict[str, Any]]:
    _env(monkeypatch, True, True)
    _flags(monkeypatch, ahead)
    chunks = _Chunks(monkeypatch)
    kw: dict[str, Any] = {"prime_check": True, "min_soundness_bits": 20} if p31 else {}
    st = _Store(template, pack)
    client = LocalComputeClient(pack, record=True)
    be = MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint, trip_engine=True, **kw)
    be.freivalds_rate = 1.0
    specs = pack.linear_specs()[:3]
    info: dict[str, Any] = {}
    try:
        if chunked:
            res = st.windowed(tokens, first=8, window=8, chunked=True)
        else:
            res = st.windowed(tokens, first=70, window=50)  # parts cross the 300-slot segment boundary
        outs = []
        for i in range(tokens):
            for spec in specs:
                outs.append(be.forward_many([LinearCall(spec.id, _x(spec, i % 7), res)])[0])
            if wait:
                _drain(be)  # deterministic: the leases prepared ahead exist before the next token
                if res._ahead_thread is not None:
                    res._ahead_thread.join()
        be.join_checks()
        _drain(be)
        info["taken"] = [int(res.taken(s.id)) for s in specs]
        info["slots"] = int(res.slots)
        info["extensions"] = int(res.extensions)
        parts = list(res._parts)
        info["ahead"] = sum(int(getattr(p, "leases_ahead", 0)) for p in parts)
        info["used"] = sum(int(getattr(p, "leases_ahead_used", 0)) for p in parts)
        info["failures"] = be.stats.integrity_failures
        snap = be.stats.snapshot()
        info["checked"] = (snap["freivalds_checked"], snap["deferred_checks"])
        res.close()
        info["zero"] = all(not any(b) for b in chunks.bufs)
        info["closed"] = all(lease.closed or lease.left == 0 for lease in chunks.leases)
        info["leases"] = len(chunks.leases)
    finally:
        be.close()
        st.close()
    return outs, [x.numpy().tobytes() for _, x in client.received], info


@pytest.mark.parametrize(("chunked", "p31"), [(False, False), (False, True), (True, False)])
def test_lease_ahead_exact_and_used(qwen_pack: InMemoryPack, store_template: Path, p31_template: Path,
                                    monkeypatch: pytest.MonkeyPatch, chunked: bool, p31: bool) -> None:
    tpl = p31_template if p31 else store_template
    tokens = 120 if chunked else 330
    o0, b0, i0 = _run(qwen_pack, tpl, monkeypatch, False, chunked=chunked, p31=p31, tokens=tokens)
    o1, b1, i1 = _run(qwen_pack, tpl, monkeypatch, True, chunked=chunked, p31=p31, tokens=tokens)
    assert b0 == b1  # every payload byte, in order
    assert all(torch.equal(a, b) for a, b in zip(o0, o1, strict=True))
    assert i0["failures"] == i1["failures"] == 0
    assert i0["checked"] == i1["checked"]
    # admission: the same slots reserved, extensions made and rows taken (preparing ahead reserves nothing)
    assert (i0["taken"], i0["slots"], i0["extensions"]) == (i1["taken"], i1["slots"], i1["extensions"])
    assert i0["ahead"] == i0["used"] == 0
    assert i1["used"] > 0 and i1["ahead"] >= i1["used"], i1
    assert i1["zero"] and i1["closed"]  # every chunk zero, every lease closed or used up after close


def test_lease_ahead_racing_the_plaintext_thread_is_exact(qwen_pack: InMemoryPack, store_template: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Without waiting for the lease-ahead thread (leases attached from ahead or made inline, by timing)."""
    o0, b0, _ = _run(qwen_pack, store_template, monkeypatch, False, chunked=True, p31=False, tokens=96)
    o1, b1, i1 = _run(qwen_pack, store_template, monkeypatch, True, chunked=True, p31=False, tokens=96, wait=False)
    assert b0 == b1 and all(torch.equal(a, b) for a, b in zip(o0, o1, strict=True))
    assert i1["failures"] == 0 and i1["zero"]


def _one(qwen_pack: InMemoryPack, template: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any, Any]:
    _env(monkeypatch, True, True)
    _flags(monkeypatch, True)
    st = _Store(template, qwen_pack)
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack, record=True), DEFAULT_RING,
                       qwen_pack.fingerprint, trip_engine=True)
    be.freivalds_rate = 1.0
    return st, be, qwen_pack.linear_specs()[0], st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 60)


def test_prepared_lease_hands_out_nothing_until_attached(qwen_pack: InMemoryPack, store_template: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.maskstore import store as store_mod

    st, be, spec, res = _one(qwen_pack, store_template, monkeypatch)
    try:
        key = spec.id.key
        be.forward_many([LinearCall(spec.id, _x(spec), res)])
        _drain(be)
        first = res._lease[key]
        nxt = res._next[key][0]
        rows = store_mod.TRIP_LEASE_ROWS
        assert int(nxt.start_row) == int(first.start_row) + int(first.rows) == rows
        assert res.taken(spec.id) == 1 and int(nxt.pos) == 0  # the cursor is the attached lease's
        for i in range(1, rows):
            be.forward_many([LinearCall(spec.id, _x(spec, i), res)])
            assert int(nxt.pos) == 0 and res._lease[key] is first  # nothing from the next lease yet
        be.forward_many([LinearCall(spec.id, _x(spec, rows), res)])
        assert res._lease[key] is nxt and int(nxt.pos) == 1 and res.leases_ahead_used == 1
        assert res.taken(spec.id) == rows + 1
        res.close()
        assert nxt.closed and not nxt.ready(1)
    finally:
        be.close()
        st.close()


def test_prepared_lease_closed_when_another_path_takes_the_rows(qwen_pack: InMemoryPack, store_template: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    st, be, spec, res = _one(qwen_pack, store_template, monkeypatch)
    try:
        key = spec.id.key
        be.forward_many([LinearCall(spec.id, _x(spec), res)])
        _drain(be)
        nxt = res._next[key][0]
        r, _wr = res.take(spec.id, 1)  # the ordinary path: settles the cursor, closes the prepared lease
        assert key not in res._next and nxt.closed and not nxt.ready(1)
        assert res.taken(spec.id) == 2
        # the trip path continues from the cursor with a new lease (rows in order, each once)
        be.forward_many([LinearCall(spec.id, _x(spec, 2), res)])
        assert int(res._lease[key].start_row) == 2 and res.taken(spec.id) == 3
        res.close()
    finally:
        be.close()
        st.close()


def test_prepared_lease_closed_on_other_row_or_policy(qwen_pack: InMemoryPack, store_template: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.maskstore import store as store_mod

    st, be, spec, res = _one(qwen_pack, store_template, monkeypatch)
    try:
        key = spec.id.key
        tag = ((0.0, 0), False)
        lease = res.trip_lease(spec.id, 1, None, 0.0, tag=tag)
        assert res.lease_ahead(spec.id, 1, None, 0.0, tag=tag) == 1
        nxt = res._next[key][0]
        # a different policy tag: not attached, closed
        lease.detach()
        got = res.trip_lease(spec.id, 1, None, 0.0, tag=((1.0, 0), False))
        assert got is not nxt and nxt.closed and int(got.start_row) == 0
        # a lease prepared for a later row than the cursor (the attached lease stopped early): closed
        assert res.lease_ahead(spec.id, 1, None, 0.0, tag=tag) == 1
        nxt = res._next[key][0]
        assert int(nxt.start_row) == store_mod.TRIP_LEASE_ROWS
        got.detach()  # cursor 0: the attached lease handed out nothing
        got2 = res.trip_lease(spec.id, 1, None, 0.0, tag=tag)
        assert got2 is not nxt and nxt.closed and int(got2.start_row) == 0
        # without a tag (the switch off at the caller) nothing is prepared
        assert res.lease_ahead(spec.id, 1, None, 0.0, tag=None) == 0
        monkeypatch.setattr(store_mod, "TRIP_LEASE_AHEAD", False)
        assert res.lease_ahead(spec.id, 1, None, 0.0, tag=tag) == 0
        res.close()
        assert res.lease_ahead(spec.id, 1, None, 0.0, tag=tag) == 0
    finally:
        be.close()
        st.close()


def test_soundness_failure_ahead_is_raised_by_the_lease_path(qwen_pack: InMemoryPack, store_template: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A soundness check that fails on the lease-ahead thread prepares nothing; the trip path meets it before
    any row of that extent is handed out."""
    from pgw.core.errors import MaskStoreError

    st, be, spec, res = _one(qwen_pack, store_template, monkeypatch)
    try:
        key = spec.id.key
        be.forward_many([LinearCall(spec.id, _x(spec), res)])
        _drain(be)
        assert key in res._next
        res._next.pop(key)[0].close()
        calls: list[int] = []

        def bad(chk: Any) -> None:
            calls.append(1)
            raise MaskStoreError("too few checks")

        tag = ((0.0, 0), False)
        assert res.lease_ahead(spec.id, 1, bad, 0.0, tag=tag) == 0 and calls and key not in res._next
        res._lease[key].detach()
        taken = res.taken(spec.id)
        with pytest.raises(MaskStoreError):
            res.trip_lease(spec.id, 1, bad, 0.0, tag=tag)
        assert res.taken(spec.id) == taken
        res.close()
    finally:
        be.close()
        st.close()


def test_sdk_turns_it_on() -> None:
    from pgw.sdk.engine import NATIVE_ENV

    assert NATIVE_ENV.get("PGW_TRIP_LEASE_AHEAD") == "1"


def test_default_off() -> None:
    """Off unless set (read at import, as ``PGW_TRIP_DECRYPT_AHEAD``); the SDK sets it."""
    from pgw.maskstore import store as store_mod

    assert 'os.environ.get("PGW_TRIP_LEASE_AHEAD", "0")' in Path(store_mod.__file__).read_text()


def test_single_answer_trips_prepare_nothing_ahead(qwen_pack: InMemoryPack, store_template: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """One answer per trip (the single-answer paths): nothing is prepared ahead (no extra CPU where the plaintext
    thread is not the limit); several answers in one trip: the next leases are prepared."""
    from pgw.maskstore import store as store_mod

    _env(monkeypatch, True, True)
    monkeypatch.setenv("PGW_TRIP_NATIVE_MULTI", "1")
    monkeypatch.setattr(store_mod, "TRIP_DECRYPT_AHEAD", True)
    monkeypatch.setattr(store_mod, "TRIP_LEASE_AHEAD", True)
    st = _Store(store_template, qwen_pack)
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       trip_engine=True)
    try:
        spec = qwen_pack.linear_specs()[0]
        a = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 40)
        b = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 40)
        be.forward_many([LinearCall(spec.id, _x(spec), a)])
        _drain(be)
        assert not a._next and be._ahead_pool is None
        be.forward_many([LinearCall(spec.id, _x(spec, 1), a), LinearCall(spec.id, _x(spec, 2), b)])
        _drain(be)
        assert spec.id.key in b._next  # b's first lease came from the several-answer trip
        a.close()
        b.close()
    finally:
        be.close()
        st.close()
