"""The native per-trip path (``PGW_TRIP_NATIVE``): one native call per trip on each side of the lane call,
fed by a lease the mask reservation hands over. Exact and byte-identical to the ``PGW_TRIP_FAST`` path on
real store reservations (windowed, several parts, chunk and extent boundaries): same logits as the plain
reference, the same provider payloads and noise statistics, single-use rows under faults, the same
soundness/exhaustion checks before any row is taken, wiping, burn-on-close. Default off.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import native
from pgw.core.errors import IntegrityError, MaskExhaustedError, MaskReuseError, MaskStoreError, RingOverflowError
from pgw.core.ring import DEFAULT_RING
from pgw.engine import InMemoryPack, LocalComputeClient, MaskedBackend
from pgw.engine.backends import LinearCall
from pgw.gateway.masks import WindowedReservation
from pgw.observability.noise import OutboundNoiseMonitor
from pgw.wire.messages import ComputeResponse, ComputeResult
from tests.engine import test_model_hooks as hooks
from tests.engine.test_trip_engine import _masked, _plain_ref, _same

pytestmark = pytest.mark.skipif(not native.native_available("TripLease", "trip_prepare_into"),
                                reason="pgw_fast native trip lease not built")

SLOTS = 600


def _env(monkeypatch: pytest.MonkeyPatch, fast: bool, nat: bool, noise_async: bool = False) -> None:
    monkeypatch.setenv("PGW_TRIP_FAST", "1" if fast else "0")
    monkeypatch.setenv("PGW_TRIP_NATIVE", "1" if nat else "0")
    monkeypatch.setenv("PGW_NOISE_ASYNC_SAMPLED", "1" if noise_async else "0")


@pytest.fixture(scope="module")
def store_template(qwen_pack: InMemoryPack) -> Iterator[Path]:
    """A mask store with small chunks (8 slots) and two segments, copied per run (identical masks)."""
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter

    d = Path(tempfile.mkdtemp(prefix="pgw-tn-"))
    store = MaskStore.open(d / "store", FileKeyProvider(d / "kek", create=True), fsync=False,
                           counter=FileMonotonicCounter(d / "ctr", fsync=False))
    gen = MaskGenerator(qwen_pack, DEFAULT_RING)
    gen.generate_into_store(store, SLOTS // 2, chunk_slots=8, store_chunk_slots=8, factory_id="t")
    gen.generate_into_store(store, SLOTS // 2, chunk_slots=8, store_chunk_slots=8, factory_id="t")
    store.close()
    yield d
    shutil.rmtree(d, ignore_errors=True)


class _Store:
    def __init__(self, template: Path, pack: Any):
        from pgw.maskstore import FileKeyProvider, MaskStore
        from pgw.maskstore.counters import FileMonotonicCounter

        self.dir = Path(tempfile.mkdtemp(prefix="pgw-tn-run-"))
        shutil.copytree(template, self.dir, dirs_exist_ok=True)
        self.store = MaskStore.open(self.dir / "store", FileKeyProvider(self.dir / "kek"), fsync=False,
                                    counter=FileMonotonicCounter(self.dir / "ctr", fsync=False))
        self.fp = pack.fingerprint
        self.made: list[Any] = []

    def windowed(self, slots: int, first: int = 5, window: int = 3, chunked: bool = False) -> WindowedReservation:
        def reserve(n: int) -> Any:
            return self.store.reserve(self.fp, DEFAULT_RING, n)

        if chunked:  # mask_reservation="incremental": chunk-aligned entitlement, ahead, background retirement
            from pgw.gateway.masks import chunk_entitlement

            w = WindowedReservation(reserve(first), reserve, cap=chunk_entitlement(first, slots, window),
                                    window=window, ahead=True, chunked=True)
        else:
            w = WindowedReservation(reserve(min(first, slots)), reserve, cap=slots, window=window)
        self.made.append(w)
        return w

    def close(self) -> None:
        for w in self.made:
            w.close()
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class _Seeded(OutboundNoiseMonitor):
    def __init__(self) -> None:
        super().__init__(DEFAULT_RING, sample_rate=0.5, fail_closed=True)
        import random

        self._rng = random.Random(7)


def _scenario(pack: Any, template: Path, monkeypatch: pytest.MonkeyPatch, fast: bool, nat: bool,
              noise_async: bool = False, mode: str = "sync", windows: tuple[int, int] = (5, 3),
              chunked: bool = False, **kw: Any) -> tuple[dict[str, torch.Tensor], list[Any], Any, Any]:
    _env(monkeypatch, fast, nat, noise_async)
    st = _Store(template, pack)
    monkeypatch.setattr(hooks, "_res", lambda p, n, seed: st.windowed(n, *windows, chunked=chunked))
    client = kw.pop("client", None) or LocalComputeClient(pack, record=True)
    noise = _Seeded()
    model = _masked(pack, client=client, trip_engine=True, sync_trips=True, noise_monitor=noise, **kw)
    model.backend.freivalds_rate = 1.0
    be = model.backend
    calls: list[int] = []
    real = be._prepare_trip_lease

    def spy(call: Any) -> Any:
        p = real(call)
        calls.append(0 if p is None else 1)
        return p

    be._prepare_trip_lease = spy  # type: ignore[method-assign]
    try:
        assert be.trip_native == (fast and nat)
        out = hooks.scenario_run(model, pack, mode, True)
        be._join_noise()
        be.join_checks()
    finally:
        model.close()
        st.close()
    rep = noise.report()
    stats = (rep.calls, rep.sampled_calls, rep.samples, rep.rows_checked, rep.duplicate_rows, rep.windows)
    sent = [(lid, tuple(x.shape), x.numpy().tobytes()) for lid, x in client.received]
    return out, sent, stats, (sum(calls), len(calls), be.stats.snapshot())


def test_default_off(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_TRIP_NATIVE", raising=False)
    monkeypatch.setenv("PGW_TRIP_FAST", "1")
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       trip_engine=True)
    assert be.trip_fast and not be.trip_native


@pytest.mark.parametrize("noise_async", [False, True])
def test_native_equals_plain_and_bytes_identical(qwen_pack: InMemoryPack, store_template: Path,
                                                 monkeypatch: pytest.MonkeyPatch, noise_async: bool) -> None:
    ref = _plain_ref(qwen_pack)
    base_out, base_sent, base_stats, _ = _scenario(qwen_pack, store_template, monkeypatch, True, False, noise_async)
    _same(ref, base_out)
    out, sent, stats, (served, tried, snap) = _scenario(qwen_pack, store_template, monkeypatch, True, True,
                                                        noise_async)
    _same(ref, out)  # masked == plain, bit for bit
    assert served > 0 and tried > served  # the lease served the decode trips; others fell back
    assert sent == base_sent  # byte-identical provider payloads, same order
    assert stats == base_stats  # same noise draws and statistics
    assert snap["integrity_failures"] == 0


@pytest.mark.parametrize("nat", [False, True])
def test_incremental_chunks_equal_plain_and_bytes_identical(qwen_pack: InMemoryPack, store_template: Path,
                                                            monkeypatch: pytest.MonkeyPatch, nat: bool) -> None:
    """``mask_reservation = "incremental"`` (chunked windows: whole-chunk extensions, ahead reservations in the
    background, retired parts closed in the background), with and without native trip leases: masked == plain
    bit for bit, no integrity failure. (Payload bytes differ from run to run here: the ahead reservations of the
    scenario's concurrent sessions interleave in the store in a thread-dependent order, so a session's masks
    differ; what the provider sees as row counts is checked at the gateway level.)"""
    ref = _plain_ref(qwen_pack)
    out, _, _, (served, _, snap) = _scenario(qwen_pack, store_template, monkeypatch, True, nat,
                                             windows=(4, 2), chunked=True)
    _same(ref, out)
    assert (served > 0) == nat and snap["integrity_failures"] == 0


@pytest.mark.parametrize("split", [2, 4])
def test_native_split_equals_plain_and_bytes_identical(qwen_pack: InMemoryPack, store_template: Path,
                                                       monkeypatch: pytest.MonkeyPatch, split: int) -> None:
    """``PGW_TRIP_SPLIT`` (multi-core per-trip kernels; the kernels themselves fall back to one thread below
    their column-block size, the bits are covered by ``tests/native/test_trip_split.py``): same logits, same
    payloads and noise statistics as the native path without it."""
    ref = _plain_ref(qwen_pack)
    _, base_sent, base_stats, _ = _scenario(qwen_pack, store_template, monkeypatch, True, True)
    monkeypatch.setenv("PGW_TRIP_SPLIT", str(split))
    monkeypatch.setenv("PGW_TRIP_SPLIT_MIN_ELEMS", "1")
    out, sent, stats, (served, _, snap) = _scenario(qwen_pack, store_template, monkeypatch, True, True)
    _same(ref, out)
    assert served > 0 and sent == base_sent and stats == base_stats
    assert snap["integrity_failures"] == 0


@pytest.mark.parametrize("split", [0, 2])
def test_native_driver_over_the_lane(qwen_pack: InMemoryPack, store_template: Path, monkeypatch: pytest.MonkeyPatch,
                                     split: int) -> None:
    """``PGW_TRIP_DRIVER``: the lane call and the finish in one native call (``TripTicket.lane_finish``) on a native
    lane channel. Same logits as plain, the same provider payloads in the same order as without the driver, and
    lane refusals / broken connections fall back to the wrapped client with the same request."""
    from pgw.wire.lane import LaneComputeClient, LaneConfig, native_channel_class
    from tests.engine.test_trip_engine import _LaneLocal

    if native_channel_class() is None or not hasattr(native.fast.TripTicket, "lane_finish"):
        pytest.skip("pgw_fast without the native lane channel / driver")
    ref = _plain_ref(qwen_pack)
    monkeypatch.setenv("PGW_TRIP_SPLIT", str(split))
    monkeypatch.setenv("PGW_TRIP_SPLIT_MIN_ELEMS", "1")
    runs = []
    for driver, faults in ((False, ""), (True, ""), (True, "refuse"), (True, "drop")):
        monkeypatch.setenv("PGW_TRIP_DRIVER", "1" if driver else "0")
        with tempfile.TemporaryDirectory(prefix="pgwl") as d:
            local = _LaneLocal(qwen_pack, d)
            if faults == "refuse":  # refused trips (the provider saw the payload): sent again on the wrapped client
                local.lane.refuse_next = [8, 8]
            elif faults == "drop":  # a dropped connection: the lane cools down, everything on the wrapped client
                local.lane.drop_next = 1
            lane = LaneComputeClient(local, LaneConfig(native=None))
            try:
                out, sent, stats, (served, _, snap) = _scenario(qwen_pack, store_template, monkeypatch, True, True,
                                                                client=lane)
            finally:
                lane.close()
            _same(ref, out)
            assert served > 0 and snap["integrity_failures"] == 0
            reasons = lane.lane_stats.fallback_reasons
            if faults == "drop":
                assert reasons.get("error:OSError") == 1
            else:
                assert lane.lane_stats.trips > 0
            if faults == "refuse":
                assert reasons.get("status:BUSY") == 2
            runs.append((sent, stats))
    assert runs[1] == runs[0]  # byte-identical payloads and noise statistics with and without the driver
    assert runs[2][1] == runs[0][1] and runs[3][1] == runs[0][1]


def test_native_async_mode(qwen_pack: InMemoryPack, store_template: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = _plain_ref(qwen_pack)
    out, _, _, (served, _, _) = _scenario(qwen_pack, store_template, monkeypatch, True, True, mode="async")
    _same(ref, out)
    assert served > 0


# ---- unit tests on one reservation ---------------------------------------------------------------------
@pytest.fixture
def one(qwen_pack: InMemoryPack, store_template: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    _env(monkeypatch, True, True)
    st = _Store(store_template, qwen_pack)
    client = LocalComputeClient(qwen_pack, record=True)
    model = _masked(qwen_pack, client=client, trip_engine=True, sync_trips=True)
    model.backend.freivalds_rate = 1.0
    try:
        yield st, model.backend, client
    finally:
        model.close()
        st.close()


def _x(spec: Any, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, spec.in_features, generator=g)


def test_rows_single_use_in_order(one: Any, qwen_pack: InMemoryPack) -> None:
    """Every trip takes the next row exactly once; the lease's rows are the store's rows in order."""
    st, be, client = one
    spec = qwen_pack.linear_specs()[0]
    res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 20)
    for i in range(20):
        be.forward_many([LinearCall(spec.id, _x(spec, i), res)])
        assert res.taken(spec.id) == i + 1
        assert res.remaining(spec.id) == 19 - i
    assert len({x.numpy().tobytes() for _, x in client.received}) == 20
    with pytest.raises(MaskExhaustedError):
        be.forward_many([LinearCall(spec.id, _x(spec), res)])
    assert res.taken(spec.id) == 20
    res.close()


def test_fault_after_prepare_burns_the_row(one: Any, qwen_pack: InMemoryPack) -> None:
    """A transport failure, a tampered result and a malformed reply all leave the row consumed; the
    next trip uses the next row (never the failed one again), and its result is exact."""
    st, be, client = one
    spec = qwen_pack.linear_specs()[0]
    res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 12)
    ref_be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                           trip_engine=False)
    real = client.compute_sync

    def broken(req: Any) -> Any:
        raise ConnectionError("down")

    def tampered(req: Any) -> Any:
        r = real(req)
        it = r.items[0]
        y = bytearray(it.y_hat)
        y[0] ^= 1
        return ComputeResponse(r.request_id, (ComputeResult(it.linear, it.rows, bytes(y)),))

    def short(req: Any) -> Any:
        r = real(req)
        it = r.items[0]
        return ComputeResponse(r.request_id, (ComputeResult(it.linear, it.rows, bytes(it.y_hat)[:-1]),))

    x = _x(spec, 3)
    for fault, exc in ((broken, Exception), (tampered, IntegrityError), (short, Exception)):
        client.compute_sync = fault  # type: ignore[method-assign]
        before = res.taken(spec.id)
        with pytest.raises(exc):
            be.forward_many([LinearCall(spec.id, x, res)])
            be.join_checks()  # the tampered trip's check may be deferred
        assert res.taken(spec.id) == before + 1  # consumed, never handed out again
        client.compute_sync = real  # type: ignore[method-assign]
    out = be.forward_many([LinearCall(spec.id, x, res)])[0]
    be.join_checks()
    res2 = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 1)
    assert torch.equal(out, ref_be.forward_many([LinearCall(spec.id, x, res2)])[0])
    res2.close()
    payloads = [xs.numpy().tobytes() for _, xs in client.received]
    assert len(set(payloads)) == len(payloads)  # every payload under a fresh row
    res.close()


def test_soundness_and_exhaustion_before_any_row(one: Any, qwen_pack: InMemoryPack) -> None:
    st, be, _client = one
    spec = qwen_pack.linear_specs()[0]
    res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 3)
    be.forward_many([LinearCall(spec.id, _x(spec), res)])
    lease = res.trip_leases[spec.id.key]
    assert lease.cursor == 1
    lease.detach()  # the next trip makes a new lease: its soundness check fails before any row
    be.min_soundness_bits = 10_000.0
    be._sound_ok.clear()
    with pytest.raises(MaskStoreError):
        be.forward_many([LinearCall(spec.id, _x(spec), res)])
    assert res.taken(spec.id) == 1
    be.min_soundness_bits = 20.0
    with pytest.raises(RingOverflowError):  # non-finite: refused before the row is taken
        be.forward_many([LinearCall(spec.id, torch.full((1, spec.in_features), float("nan")), res)])
    assert res.taken(spec.id) == 1
    res.close()


def test_close_burns_and_fails_inflight(one: Any, qwen_pack: InMemoryPack) -> None:
    """Closing the reservation mid-trip: the in-flight trip fails closed (no output), the lease hands
    out nothing more, and the burned count covers every row past the lease's cursor."""
    st, be, client = one
    spec = qwen_pack.linear_specs()[0]
    res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 6)
    be.forward_many([LinearCall(spec.id, _x(spec), res)])
    prep = be.prepare_calls([LinearCall(spec.id, _x(spec, 1), res)])
    assert prep.ticket is not None
    resp = client.compute_sync(prep.request)
    lease = res.trip_leases[spec.id.key]
    res.close()
    assert lease.closed and not lease.ready(1) and lease.left == 0
    with pytest.raises(MaskReuseError):
        be.finish_prepared(prep, resp, 0.0)
    with pytest.raises(MaskReuseError):
        be.forward_many([LinearCall(spec.id, _x(spec), res)])


def test_detach_hands_over_to_ordinary_path(one: Any, qwen_pack: InMemoryPack) -> None:
    """A multi-row call (ordinary path) after lease trips continues at the lease's cursor."""
    st, be, _client = one
    spec = qwen_pack.linear_specs()[0]
    res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 40)
    for i in range(3):
        be.forward_many([LinearCall(spec.id, _x(spec, i), res)])
    lease = res.trip_leases[spec.id.key]
    r, wr = res.take(spec.id, 2)  # another path: the lease is detached first
    assert not lease.ready(1) and res.taken(spec.id) == 5 and spec.id.key not in res.trip_leases
    r.zero_()
    wr.zero_()
    be.forward_many([LinearCall(spec.id, _x(spec, 9), res)])
    assert res.taken(spec.id) == 6
    res.close()


def test_ticket_single_use_and_wipe(one: Any, qwen_pack: InMemoryPack) -> None:
    st, be, client = one
    spec = qwen_pack.linear_specs()[0]
    res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 8)
    prep = be.prepare_calls([LinearCall(spec.id, _x(spec), res)])
    resp = client.compute_sync(prep.request)
    tk = prep.ticket
    be.discard(prep)  # wiped: W r rows zeroed, the ticket refuses to finish
    scale, bias, bound = be._consts(spec.id, be._meta.get(spec.id))
    with pytest.raises(ValueError):
        tk.finish(resp.items[0].y_hat, scale, bias, bound)
    assert res.taken(spec.id) == 1
    res.close()


@pytest.mark.parametrize("ring_name", ["K40A12", "K48A20", "K24A12C21"])
def test_other_rings_bytes_identical(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch, ring_name: str) -> None:
    """Rings whose W r is not read in place (K != 32: widened per trip): same payloads and outputs."""
    from pgw.core.ring import RingParams
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter

    ring = RingParams.from_name(ring_name)
    d = Path(tempfile.mkdtemp(prefix="pgw-tn-ring-"))
    try:
        store = MaskStore.open(d / "store", FileKeyProvider(d / "kek", create=True), fsync=False,
                               counter=FileMonotonicCounter(d / "ctr", fsync=False))
        MaskGenerator(qwen_pack, ring).generate_into_store(store, 24, chunk_slots=4, factory_id="t")
        store.close()
        got = []
        for nat in (False, True):
            run = d / f"run{int(nat)}"
            shutil.copytree(d, run, ignore=shutil.ignore_patterns("run*"))
            _env(monkeypatch, True, nat)
            st = MaskStore.open(run / "store", FileKeyProvider(run / "kek"), fsync=False,
                                counter=FileMonotonicCounter(run / "ctr", fsync=False))
            client = LocalComputeClient(qwen_pack, record=True, rings=(ring_name,))
            be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint, trip_engine=True)
            be.defer_min_out = 0
            assert be.trip_native == nat
            res = st.reserve(qwen_pack.fingerprint, ring, 10)
            outs = []
            for i in range(10):
                for spec in qwen_pack.linear_specs()[:2]:
                    outs.append(be.forward_many([LinearCall(spec.id, _x(spec, i), res)])[0])
            be.join_checks()
            res.close()
            st.close()
            be.close()
            got.append((outs, [x.numpy().tobytes() for _, x in client.received]))
        (o0, b0), (o1, b1) = got
        assert b0 == b1
        assert all(torch.equal(a, b) for a, b in zip(o0, o1, strict=True))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_lease_rejects_bad_buffers() -> None:
    k = bytes(32)
    u = np.zeros((2, 4), dtype=np.uint32)
    uw = np.zeros((2, 3), dtype=np.uint32)
    with pytest.raises(ValueError):  # chunk too small for the lease's rows
        native.fast.TripLease(k, 3, 4, 32, 12, 27, 1.0, 0, [(bytearray(10), 0, 4)], 0, u, uw)
    with pytest.raises(ValueError):  # check vectors of another linear
        native.fast.TripLease(k, 5, 4, 32, 12, 27, 1.0, 0, [(bytearray(64), 0, 1)], 0, u, uw)
    with pytest.raises(ValueError):
        native.fast.TripLease(bytes(31), 3, 4, 32, 12, 27, 1.0, 0, [(bytearray(64), 0, 1)], 0, u, uw)
    with pytest.raises(ValueError):  # no rows
        native.fast.TripLease(k, 3, 4, 32, 12, 27, 1.0, 0, [], 0, u, uw)
    lease = native.fast.TripLease(k, 3, 4, 32, 12, 27, 1.0, 0, [(bytearray(16), 0, 1), (bytearray(32), 1, 1)], 7,
                                  u, uw)
    assert lease.rows == 2 and lease.cursor == 7 and lease.ready(2) and not lease.ready(3)
    assert lease.detach() == 7 and not lease.ready(1) and lease.left == 0


# ---- prime-field ("p31") extents through leases ----------------------------------------------------------
@pytest.fixture(scope="module")
def p31_template(qwen_pack: InMemoryPack) -> Iterator[Path]:
    """:func:`store_template` in the prime-field check format (``check_field = "p31"``, one vector)."""
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter

    d = Path(tempfile.mkdtemp(prefix="pgw-tn-p31-"))
    store = MaskStore.open(d / "store", FileKeyProvider(d / "kek", create=True), fsync=False,
                           counter=FileMonotonicCounter(d / "ctr", fsync=False))
    gen = MaskGenerator(qwen_pack, DEFAULT_RING, check_field="p31")
    gen.generate_into_store(store, SLOTS // 2, chunk_slots=8, store_chunk_slots=8, factory_id="t")
    gen.generate_into_store(store, SLOTS // 2, chunk_slots=8, store_chunk_slots=8, factory_id="t")
    store.close()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_p31_leases_equal_plain_and_bytes_identical(qwen_pack: InMemoryPack, p31_template: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Prime-field extents are leased (one native call per side) with the prime check synchronous inside
    ``ticket.finish``: masked == plain bit for bit, byte-identical payloads and the same noise statistics
    as the PGW_TRIP_FAST path, every check counted synchronously (never deferred)."""
    ref = _plain_ref(qwen_pack)
    base_out, base_sent, base_stats, (_, _, base_snap) = _scenario(
        qwen_pack, p31_template, monkeypatch, True, False, prime_check=True, min_soundness_bits=20)
    _same(ref, base_out)
    out, sent, stats, (served, _tried, snap) = _scenario(
        qwen_pack, p31_template, monkeypatch, True, True, prime_check=True, min_soundness_bits=20)
    _same(ref, out)
    assert served > 0
    assert sent == base_sent and stats == base_stats
    assert snap["integrity_failures"] == 0 and snap["deferred_checks"] == 0
    assert snap["freivalds_checked"] == base_snap["freivalds_checked"] > 0


@pytest.fixture
def one_p31(qwen_pack: InMemoryPack, p31_template: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    _env(monkeypatch, True, True)
    st = _Store(p31_template, qwen_pack)
    client = LocalComputeClient(qwen_pack, record=True)
    model = _masked(qwen_pack, client=client, trip_engine=True, sync_trips=True, prime_check=True,
                    min_soundness_bits=20)
    model.backend.freivalds_rate = 1.0
    try:
        yield st, model.backend, client
    finally:
        model.close()
        st.close()


def test_p31_lease_tamper_detected_and_single_use(one_p31: Any, qwen_pack: InMemoryPack) -> None:
    """Through a prime lease: a tampered entry (any position) or an out-of-range reply fails closed before
    any output is used, the row stays consumed, and the next trip (next row) is exact."""
    from pgw.core.freivalds import P31

    st, be, client = one_p31
    spec = qwen_pack.linear_specs()[0]
    res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 16)
    real = client.compute_sync
    x = _x(spec, 5)
    out0 = be.forward_many([LinearCall(spec.id, x, res)])[0]
    lease = res.trip_leases[spec.id.key]
    assert lease.field == P31
    m = spec.out_features

    def tamper(pos: int, delta: int) -> Any:
        def f(req: Any) -> Any:
            r = real(req)
            it = r.items[0]
            y = np.frombuffer(bytes(it.y_hat), dtype=np.uint32).copy()
            y[pos] = (int(y[pos]) + delta) & 0xFFFFFFFF
            return ComputeResponse(r.request_id, (ComputeResult(it.linear, it.rows, y.tobytes()),))
        return f

    for fault in (tamper(0, 1), tamper(m - 1, -3), tamper(m // 2, 1 << 20), tamper(1, 1 << 31)):
        client.compute_sync = fault  # type: ignore[method-assign]
        before = res.taken(spec.id)
        with pytest.raises(IntegrityError):
            be.forward_many([LinearCall(spec.id, x, res)])
        assert res.taken(spec.id) == before + 1
        assert res.trip_leases.get(spec.id.key) is lease  # served by the lease
        client.compute_sync = real  # type: ignore[method-assign]
    out = be.forward_many([LinearCall(spec.id, x, res)])[0]
    assert torch.equal(out, out0)  # exact under a fresh row (masked == plain, see the scenario test)
    assert be.stats.deferred_checks == 0 and be.pending_checks == 0
    res.close()


def test_p31_not_leased_without_prime_check(qwen_pack: InMemoryPack, p31_template: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway without prime_check gets no lease for a prime extent and refuses it before any row."""
    _env(monkeypatch, True, True)
    st = _Store(p31_template, qwen_pack)
    try:
        be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                           trip_engine=True, min_soundness_bits=20)
        spec = qwen_pack.linear_specs()[0]
        res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 4)
        assert res.trip_lease(spec.id, 1) is None
        with pytest.raises(MaskStoreError):
            be.forward_many([LinearCall(spec.id, _x(spec), res)])
        assert res.taken(spec.id) == 0 and not res.trip_leases
        res.close()
        be.close()
    finally:
        st.close()


def test_lease_field_validation() -> None:
    k = bytes(32)
    u64 = np.zeros((1, 4), dtype=np.uint64)
    uw64 = np.zeros((1, 3), dtype=np.uint64)
    with pytest.raises(ValueError):  # prime vectors must be uint32 words
        native.fast.TripLease(k, 3, 4, 32, 12, 27, 1.0, 0, [(bytearray(16), 0, 1)], 0, u64, uw64, False, 2**31 - 1)
    u = np.zeros((1, 4), dtype=np.uint32)
    uw = np.zeros((1, 3), dtype=np.uint32)
    with pytest.raises(ValueError):  # unknown field
        native.fast.TripLease(k, 3, 4, 32, 12, 27, 1.0, 0, [(bytearray(16), 0, 1)], 0, u, uw, False, 7)
    lease = native.fast.TripLease(k, 3, 4, 32, 12, 27, 1.0, 0, [(bytearray(16), 0, 1)], 0, u, uw, False, 2**31 - 1)
    assert lease.field == 2**31 - 1


# ---- W r chunks decrypted ahead (PGW_TRIP_DECRYPT_AHEAD) ---------------------------------------------------
def _flip(path: Any, off: int, bit: int) -> None:
    import os

    fd = os.open(path, os.O_RDWR)
    try:
        b = os.pread(fd, 1, off)
        os.pwrite(fd, bytes([b[0] ^ bit]), off)
    finally:
        os.close(fd)


def test_native_chunk_open_matches_decrypt_into(store_template: Path, qwen_pack: InMemoryPack) -> None:
    from pgw.maskstore import store as store_mod

    if not native.native_available("chunk_open_into"):
        pytest.skip("pgw_fast.chunk_open_into not built")
    st = _Store(store_template, qwen_pack)
    try:
        res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 4)
        seg = res._extents[0].seg
        for ref in seg.layout.chunks()[:6]:
            assert store_mod._open_chunk(seg, ref) == seg.decrypt_into(ref)
        ref = seg.layout.chunks()[2]
        _flip(seg.path, ref.offset + 5, 1)  # one ciphertext bit: authentication fails
        with pytest.raises(MaskStoreError):
            store_mod._open_chunk(seg, ref)
        _flip(seg.path, ref.offset + 5, 1)
        _flip(seg.path, ref.offset + ref.plain_len + 2, 4)  # one tag bit
        with pytest.raises(MaskStoreError):
            store_mod._open_chunk(seg, ref)
        res.close()
    finally:
        st.close()


@pytest.mark.parametrize("p31", [False, True])
def test_decrypt_ahead_exact_and_identical(qwen_pack: InMemoryPack, store_template: Path, p31_template: Path,
                                           monkeypatch: pytest.MonkeyPatch, p31: bool) -> None:
    """Decrypting the next chunk ahead changes nothing: same payloads, same outputs, same statistics; the
    ahead buffers are used by the leases and are zero after the run."""
    from pgw.maskstore import store as store_mod

    if not native.native_available("chunk_open_into"):
        pytest.skip("pgw_fast.chunk_open_into not built")
    tpl = p31_template if p31 else store_template
    kw: dict[str, Any] = {"prime_check": True, "min_soundness_bits": 20} if p31 else {}
    ref = _plain_ref(qwen_pack)
    base_out, base_sent, base_stats, _ = _scenario(qwen_pack, tpl, monkeypatch, True, True, windows=(40, 40),
                                                   **kw)
    monkeypatch.setattr(store_mod, "TRIP_DECRYPT_AHEAD", True)
    used: list[int] = []
    bufs: list[dict[int, bytearray]] = []
    real_take = store_mod.MaskReservation._ahead_take_locked

    def spy(self: Any, ei: int, key: str, j: int) -> Any:
        b = real_take(self, ei, key, j)
        used.append(len(b))
        bufs.append(b)
        return b

    monkeypatch.setattr(store_mod.MaskReservation, "_ahead_take_locked", spy)
    out, sent, stats, (served, _, snap) = _scenario(qwen_pack, tpl, monkeypatch, True, True, windows=(40, 40),
                                                    **kw)
    _same(ref, out)
    _same(base_out, out)
    assert sent == base_sent and stats == base_stats and served > 0
    assert snap["integrity_failures"] == 0
    assert all(not any(b) for d in bufs for b in d.values())  # every buffer decrypted ahead is zero after the run


def test_decrypt_ahead_zeroed_on_close_and_tamper_fails_closed(qwen_pack: InMemoryPack, store_template: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.maskstore import store as store_mod

    if not native.native_available("chunk_open_into"):
        pytest.skip("pgw_fast.chunk_open_into not built")
    monkeypatch.setattr(store_mod, "TRIP_DECRYPT_AHEAD", True)
    _env(monkeypatch, True, True)
    st = _Store(store_template, qwen_pack)
    try:
        be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                           trip_engine=True)
        spec = qwen_pack.linear_specs()[0]
        # close with a chunk decrypted ahead and never used: zeroed
        res = st.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 20)
        be.forward_many([LinearCall(spec.id, _x(spec), res)])
        hits = [v for k, v in res._ahead.items() if k[1] == spec.id.key]
        assert len(hits) == 1
        bufs = hits[0][1].result()
        assert bufs and all(any(b) for b in bufs.values())
        res.close()
        assert not any(any(b) for b in bufs.values()) and not res._ahead
        # a tampered chunk decrypted ahead: the lease that needs it fails closed, no row of it is handed out
        st2 = _Store(store_template, qwen_pack)
        try:
            res = st2.store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 40)
            ex = res._extents[0]
            seg = ex.seg
            cs = seg.header.chunk_slots
            assert ex.ext.start == 0 and cs == 8
            lease_rows = store_mod.TRIP_LEASE_ROWS  # the first lease: rows [0, 16), chunks 0 and 1
            ref = seg.layout.wr_chunk(ex.li[spec.id.key], lease_rows // cs + 1)  # in the next lease (ahead)
            _flip(seg.path, ref.offset + 3, 0x80)
            for i in range(lease_rows):
                be.forward_many([LinearCall(spec.id, _x(spec, i), res)])
            assert res._ahead  # the next lease's chunks are being decrypted ahead
            with pytest.raises(MaskStoreError):
                be.forward_many([LinearCall(spec.id, _x(spec), res)])
            assert res.taken(spec.id) == lease_rows
            res.close()
        finally:
            st2.close()
        res.close()
        be.close()
    finally:
        st.close()


@pytest.mark.parametrize("p31", [False, True])
def test_decrypt_ahead_feeds_leases_byte_identical(qwen_pack: InMemoryPack, store_template: Path,
                                                   p31_template: Path, monkeypatch: pytest.MonkeyPatch,
                                                   p31: bool) -> None:
    """Long decode runs (leases refilled many times, chunk and extent boundaries): with the chunks decrypted
    ahead the leases take them (counted), outputs and payloads are identical to inline decryption."""
    from pgw.maskstore import store as store_mod

    if not native.native_available("chunk_open_into"):
        pytest.skip("pgw_fast.chunk_open_into not built")
    tpl = p31_template if p31 else store_template
    kw: dict[str, Any] = {"prime_check": True, "min_soundness_bits": 20} if p31 else {}
    _env(monkeypatch, True, True)
    used: list[int] = []
    real_take = store_mod.MaskReservation._ahead_take_locked

    def spy(self: Any, ei: int, key: str, j: int) -> Any:
        b = real_take(self, ei, key, j)
        used.append(len(b))
        return b

    monkeypatch.setattr(store_mod.MaskReservation, "_ahead_take_locked", spy)
    got = []
    for ahead in (False, True):
        monkeypatch.setattr(store_mod, "TRIP_DECRYPT_AHEAD", ahead)
        st = _Store(tpl, qwen_pack)
        client = LocalComputeClient(qwen_pack, record=True)
        be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint, trip_engine=True, **kw)
        be.freivalds_rate = 1.0
        try:
            res = st.windowed(330, first=70, window=50)  # parts cross the 300-slot segment boundary
            outs = []
            for i in range(330):
                for spec in qwen_pack.linear_specs()[:3]:
                    outs.append(be.forward_many([LinearCall(spec.id, _x(spec, i % 7), res)])[0])
            be.join_checks()
            assert be.stats.integrity_failures == 0
        finally:
            be.close()
            st.close()
        got.append((outs, [x.numpy().tobytes() for _, x in client.received]))
    (o0, b0), (o1, b1) = got
    assert b0 == b1
    assert all(torch.equal(a, b) for a, b in zip(o0, o1, strict=True))
    assert sum(used) > 0  # leases were fed from chunks decrypted ahead


def test_decrypt_ahead_reaches_background_parts(qwen_pack: InMemoryPack, store_template: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Incremental reservations (small parts reserved ahead in the background, as the client library runs them):
    a part the background extension adds starts decrypting every linear's first lease at once, so the first
    lease of each linear in each part is fed from chunks decrypted ahead instead of decrypting on the token's
    critical path. Outputs and payloads identical to inline decryption; every buffer zero after the run."""
    from pgw.gateway import masks as masks_mod
    from pgw.maskstore import store as store_mod

    if not native.native_available("chunk_open_into"):
        pytest.skip("pgw_fast.chunk_open_into not built")
    _env(monkeypatch, True, True)
    real_take = store_mod.MaskReservation._ahead_take_locked
    real_first = masks_mod._decrypt_first_leases
    real_open = store_mod._open_chunk
    got = []
    inline_opens = []
    for ahead, kick_on in ((False, False), (True, False), (True, True)):
        monkeypatch.setattr(store_mod, "TRIP_DECRYPT_AHEAD", ahead)
        kicked: list[int] = []
        bufs: list[dict[int, bytearray]] = []
        opens: list[int] = []

        def spy(self: Any, ei: int, key: str, j: int, _bufs: list[dict[int, bytearray]] = bufs) -> Any:
            b = real_take(self, ei, key, j)
            _bufs.append(b)
            return b

        def kick(part: Any, _k: list[int] = kicked, _on: bool = kick_on) -> None:
            _k.append(1)
            if _on:
                real_first(part)

        def opened(seg: Any, ref: Any, _o: list[int] = opens) -> Any:
            if threading.current_thread() is threading.main_thread():
                _o.append(1)  # decrypted by the lease itself, on the token's path
            return real_open(seg, ref)

        monkeypatch.setattr(store_mod.MaskReservation, "_ahead_take_locked", spy)
        monkeypatch.setattr(masks_mod, "_decrypt_first_leases", kick)
        monkeypatch.setattr(store_mod, "_open_chunk", opened)
        st = _Store(store_template, qwen_pack)
        client = LocalComputeClient(qwen_pack, record=True)
        be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint, trip_engine=True)
        be.freivalds_rate = 1.0
        specs = qwen_pack.linear_specs()[:3]
        try:
            res = st.windowed(120, first=8, window=8, chunked=True)
            outs = []
            for i in range(96):
                for spec in specs:
                    outs.append(be.forward_many([LinearCall(spec.id, _x(spec, i % 7), res)])[0])
                if res._ahead_thread is not None:
                    res._ahead_thread.join()  # deterministic: a part reserved ahead exists before the next token
            be.join_checks()
            assert be.stats.integrity_failures == 0
            res.close()
        finally:
            be.close()
            st.close()
        if ahead:
            assert kicked  # parts were reserved in the background
            assert all(not any(b) for d in bufs for b in d.values())  # zero after the run
        inline_opens.append(len(opens))
        got.append((outs, [x.numpy().tobytes() for _, x in client.received]))
    (o0, b0), (o1, b1), (o2, b2) = got
    assert b0 == b1 == b2
    assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(o0, o1, o2, strict=True))
    # the first lease of each linear in each background part no longer decrypts on the token's path
    assert inline_opens[2] < inline_opens[1] // 2, inline_opens


def test_driver_on_a_non_native_lane_uses_the_ordinary_lane_path() -> None:
    """PGW_TRIP_DRIVER with a lane whose channel is not native (e.g. the Python UDS channel): once known, the
    driver stays off and calls take compute_sync (the ordinary lane path), never the wrapped client directly."""
    from pgw.wire.lane import LaneComputeClient

    c = LaneComputeClient(object())
    assert c.lane_eligible_for_driver is True
    c.lane_eligible_for_driver = False
    assert getattr(c, "lane_eligible_for_driver", True) is False
