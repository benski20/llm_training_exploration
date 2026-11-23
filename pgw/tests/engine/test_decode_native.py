"""The native decode driver (``PGW_DECODE_NATIVE``): a decode pass's decoder layers in one native call per pass
(``pgw_fast.DecodeDriver``) on the native float path and the native per-trip path. Exact: the same logits as
the plain reference with the same native float ops, the same provider payloads (bytes, order) and the same
noise statistics as without the driver; over a native lane channel, refusals and a broken connection fall back
with the same request; tampering raises ``IntegrityError``; rows stay single use; default off.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import native
from pgw.core.errors import IntegrityError
from pgw.core.ring import DEFAULT_RING
from pgw.engine import InMemoryPack, LocalComputeClient, MaskedBackend, PlainBackend, TransformerModel
from pgw.engine import decode_native as dn
from pgw.wire.messages import ComputeResponse, ComputeResult
from tests.engine import test_model_hooks as hooks
from tests.engine.test_trip_engine import _masked, _same
from tests.engine.test_trip_native import _Seeded, _Store, p31_template, store_template  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(not native.native_available("DecodeDriver", "TripLease", "float_rope_attend"),
                                reason="pgw_fast without the native decode driver")


def _plain_native(pack: Any) -> dict[str, torch.Tensor]:
    model = TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))
    model.set_float_ops("native")
    try:
        return hooks.scenario_run(model, pack, "sync", False)
    finally:
        model.close()


@pytest.fixture(scope="module")
def ref_native(qwen_pack: InMemoryPack) -> dict[str, torch.Tensor]:
    return _plain_native(qwen_pack)


def _run(pack: Any, template: Path, monkeypatch: pytest.MonkeyPatch, driver: bool, client: Any = None,
         noise_async: bool = False, layers_per_call: int = 0,
         on_failure: Any = None, **kw: Any) -> tuple[dict[str, torch.Tensor], list[Any], Any, Any]:
    monkeypatch.setenv("PGW_TRIP_FAST", "1")
    monkeypatch.setenv("PGW_TRIP_NATIVE", "1")
    monkeypatch.setenv("PGW_NOISE_ASYNC_SAMPLED", "1" if noise_async else "0")
    monkeypatch.setenv("PGW_DECODE_NATIVE", "1" if driver else "0")
    monkeypatch.setenv("PGW_DECODE_NATIVE_LAYERS", str(layers_per_call))
    st = _Store(template, pack)
    monkeypatch.setattr(hooks, "_res", lambda p, n, seed: st.windowed(n, 5, 3))
    client = client or LocalComputeClient(pack, record=True)
    noise = kw.pop("noise", None) or _Seeded()
    model = _masked(pack, client=client, trip_engine=True, sync_trips=True, noise_monitor=noise, **kw)
    model.set_float_ops("native")
    be = model.backend
    be.freivalds_rate = 1.0
    be.on_integrity_failure = on_failure
    passes = []
    real = dn.NativeDecode.run

    def spy(self: Any, s: Any) -> None:
        passes.append(int(s.hs[0].shape[0]))
        real(self, s)

    monkeypatch.setattr(dn.NativeDecode, "run", spy)
    try:
        assert be.decode_native == driver
        out = hooks.scenario_run(model, pack, "sync", True)
        be._join_noise()
        be.join_checks()
    finally:
        model.close()
        st.close()
    rep = noise.report()
    stats = (rep.calls, rep.sampled_calls, rep.samples, rep.rows_checked, rep.duplicate_rows, rep.windows)
    sent = [(lid, tuple(x.shape), x.numpy().tobytes()) for lid, x in getattr(client, "received", [])]
    return out, sent, stats, (passes, be.stats.snapshot())


def test_default_off(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_DECODE_NATIVE", raising=False)
    monkeypatch.setenv("PGW_TRIP_FAST", "1")
    monkeypatch.setenv("PGW_TRIP_NATIVE", "1")
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       trip_engine=True)
    assert be.trip_native and not be.decode_native


@pytest.mark.parametrize(("noise_async", "per_call"), [(False, 0), (True, 1)])
def test_driver_equals_plain_and_bytes_identical(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                                 monkeypatch: pytest.MonkeyPatch, ref_native: dict[str, torch.Tensor],
                                                 noise_async: bool, per_call: int) -> None:
    """No native channel (a local client): the driver hands each trip back after its native prepare and
    continues with its output. Same logits as plain, same payloads and noise statistics as without it."""
    base, base_sent, base_stats, _ = _run(qwen_pack, store_template, monkeypatch, False, noise_async=noise_async)
    _same(ref_native, base)
    out, sent, stats, (passes, snap) = _run(qwen_pack, store_template, monkeypatch, True, noise_async=noise_async,
                                            layers_per_call=per_call)
    _same(ref_native, out)  # masked == plain, bit for bit
    assert passes and max(passes) <= dn.MAX_ROWS
    assert sent == base_sent  # byte-identical provider payloads, same order
    assert stats == base_stats  # same noise draws and statistics
    assert snap["integrity_failures"] == 0


class _Lane:
    """A native-lane client over the reference lane server (tests.engine.test_trip_engine._LaneLocal)."""

    def __init__(self, pack: Any, d: str, faults: str = ""):
        from pgw.wire.lane import LaneComputeClient, LaneConfig
        from tests.engine.test_trip_engine import _LaneLocal

        self.local = _LaneLocal(pack, d)
        if faults == "refuse":
            self.local.lane.refuse_next = [8, 8]
        elif faults == "drop":
            self.local.lane.drop_next = 1
        self.client = LaneComputeClient(self.local, LaneConfig(native=False if faults == "pychan" else None))

    def payloads(self) -> list[Any]:
        return [(h["linear"], h["rows"], body) for h, body in self.local.lane.trips]


@pytest.mark.parametrize("faults", ["", "refuse", "drop", "pychan"])
def test_driver_over_the_native_lane(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                     monkeypatch: pytest.MonkeyPatch, ref_native: dict[str, torch.Tensor],
                                     faults: str) -> None:
    from pgw.wire.lane import native_channel_class

    if native_channel_class() is None:
        pytest.skip("pgw_fast without the native lane channel")
    runs = []
    for driver in (False, True):
        with tempfile.TemporaryDirectory(prefix="pgwd") as d:
            lane = _Lane(qwen_pack, d, faults)
            try:
                out, _, stats, (passes, snap) = _run(qwen_pack, store_template, monkeypatch, driver, client=lane.client)
                lane_payloads = lane.payloads()
                regular = [x.numpy().tobytes() for _, x in lane.local.received]
                reasons = dict(lane.client.lane_stats.fallback_reasons)
                trips = lane.client.lane_stats.trips
            finally:
                lane.client.close()
            _same(ref_native, out)
            assert snap["integrity_failures"] == 0
            assert trips > 0 or faults == "drop"
            if driver:
                assert passes
                if faults == "pychan":  # a Python channel: every trip handed back, sent on the ordinary lane path
                    assert not lane.client.lane_eligible_for_driver
            if faults == "refuse":
                assert reasons.get("status:BUSY") == 2
            if faults == "drop":
                assert reasons.get("error:OSError") == 1
            runs.append((lane_payloads, regular, stats, trips))
    # the same bytes reached the provider in the same order (lane and regular path), same noise statistics
    assert runs[1][0] == runs[0][0]
    assert runs[1][1] == runs[0][1]
    assert runs[1][2] == runs[0][2]
    assert runs[1][3] == runs[0][3]


class _Tamper(LocalComputeClient):
    def __init__(self, pack: Any, after: int):
        super().__init__(pack, record=True)
        self.after = after
        self.n = 0

    def compute_sync(self, request: Any) -> Any:
        r = super().compute_sync(request)
        self.n += 1
        if self.n != self.after:
            return r
        it = r.items[0]
        y = bytearray(it.y_hat)
        y[0] ^= 1
        return ComputeResponse(r.request_id, (ComputeResult(it.linear, it.rows, bytes(y)),))


@pytest.mark.parametrize("defer", [False, True])
def test_tamper_raises_and_rows_single_use(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                           monkeypatch: pytest.MonkeyPatch, defer: bool) -> None:
    """A tampered result in a driver pass (deferred and synchronous checks) raises IntegrityError; every
    payload sent was built from a fresh row (no row reused under the fault)."""
    monkeypatch.setenv("PGW_DEFERRED_CHECK", "1" if defer else "0")
    for after in (10, 13, 20):  # trips of 1-row decode passes (the driver's; pass 1 is the prompt)
        client = _Tamper(qwen_pack, after)
        with pytest.raises(IntegrityError):
            _run(qwen_pack, store_template, monkeypatch, True, client=client)
        payloads = [x.numpy().tobytes() for _, x in client.received]
        assert len(set(payloads)) == len(payloads)


@pytest.mark.parametrize("defer", [False, True])
def test_tamper_on_the_native_lane(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                   monkeypatch: pytest.MonkeyPatch, defer: bool) -> None:
    """A tampered lane reply finished inside the driver (synchronous or native deferred check) raises
    IntegrityError, counted in integrity_failures; payloads stay single use."""
    from pgw.wire.lane import native_channel_class

    if native_channel_class() is None:
        pytest.skip("pgw_fast without the native lane channel")
    monkeypatch.setenv("PGW_DEFERRED_CHECK", "1" if defer else "0")
    for after in (3, 6):
        with tempfile.TemporaryDirectory(prefix="pgwd") as d:
            lane = _Lane(qwen_pack, d)
            real = lane.local._lane_compute
            n = [0]

            def bad(li: Any, rows: int, ring: str, x: bytes, real: Any = real, n: list[int] = n,
                    after: int = after) -> bytes:
                y = real(li, rows, ring, x)
                n[0] += 1
                if n[0] == after:
                    b = bytearray(y)
                    b[1] ^= 4
                    return bytes(b)
                return y

            lane.local.lane.compute = bad
            failures: list[Any] = []
            try:
                with pytest.raises(IntegrityError):
                    _run(qwen_pack, store_template, monkeypatch, True, client=lane.client, on_failure=failures.append)
                payloads = lane.payloads()
            finally:
                lane.client.close()
            assert failures  # reported through on_integrity_failure
            bodies = [b for _, _, b in payloads]
            assert len(set(bodies)) == len(bodies)


def test_driver_wipes_and_single_use_unit(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver's pass buffers are zeroed at the end of a pass and on abort; a lease row handed out by the
    driver is never handed out again (the lease cursor moved once per trip)."""
    monkeypatch.setenv("PGW_TRIP_FAST", "1")
    monkeypatch.setenv("PGW_TRIP_NATIVE", "1")
    monkeypatch.setenv("PGW_DECODE_NATIVE", "1")
    st = _Store(store_template, qwen_pack)
    model = _masked(qwen_pack, client=LocalComputeClient(qwen_pack, record=True), trip_engine=True, sync_trips=True)
    model.set_float_ops("native")
    try:
        from pgw.engine.model import SequenceInput

        res = st.windowed(40, 5, 3)
        cache = model.new_cache(64)
        model.forward_batch([SequenceInput([1, 2, 3], cache, res)])  # prompt: ordinary path
        nd = dn.driver_for(model)
        assert nd is not None
        before = {k: int(v.cursor) for k, v in res.trip_leases.items()}
        model.forward_batch([SequenceInput([4], cache, res)])
        model.backend.join_checks()
        after = {k: int(v.cursor) for k, v in res.trip_leases.items()}
        assert all(after[k] == before.get(k, after[k] - 1) + 1 for k in after)
        with pytest.raises(ValueError):
            nd.drv.current_input()  # no pass in progress: buffers gone
    finally:
        model.close()
        st.close()


@pytest.mark.parametrize("lane", [False, True])
def test_p31_driver_equals_plain_bytes_identical_and_tamper(qwen_pack: InMemoryPack, p31_template: Path,  # noqa: F811
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            ref_native: dict[str, torch.Tensor], lane: bool) -> None:
    """Prime-field ("p31") leases: the prime check runs synchronously inside the driver's finish (before the
    unmask). Same logits as plain, the same payloads as without the driver; a tampered reply raises."""
    from pgw.wire.lane import native_channel_class

    if lane and native_channel_class() is None:
        pytest.skip("pgw_fast without the native lane channel")
    kw = {"prime_check": True, "min_soundness_bits": 20}
    runs = []
    for driver in (False, True):
        with tempfile.TemporaryDirectory(prefix="pgwd") as d:
            ln = _Lane(qwen_pack, d) if lane else None
            client = ln.client if ln is not None else LocalComputeClient(qwen_pack, record=True)
            try:
                out, sent, stats, (passes, snap) = _run(qwen_pack, p31_template, monkeypatch, driver, client=client,
                                                        **kw)
                if ln is not None:
                    sent = (ln.payloads(), [x.numpy().tobytes() for _, x in ln.local.received])
            finally:
                if ln is not None:
                    ln.client.close()
        _same(ref_native, out)
        assert snap["integrity_failures"] == 0 and (passes or not driver)
        runs.append((sent, stats))
    assert runs[0] == runs[1]
    if lane:
        with tempfile.TemporaryDirectory(prefix="pgwd") as d:
            ln = _Lane(qwen_pack, d)
            real = ln.local._lane_compute
            n = [0]

            def bad(li: Any, rows: int, ring: str, x: bytes) -> bytes:
                y = real(li, rows, ring, x)
                n[0] += 1
                return y if n[0] != 5 else bytes([y[0] ^ 1]) + y[1:]

            ln.local.lane.compute = bad
            try:
                with pytest.raises(IntegrityError):
                    _run(qwen_pack, p31_template, monkeypatch, True, client=ln.client, **kw)
            finally:
                ln.client.close()
    else:
        client = _Tamper(qwen_pack, 12)
        with pytest.raises(IntegrityError):
            _run(qwen_pack, p31_template, monkeypatch, True, client=client, **kw)


class _AlarmAt(_Seeded):
    """Alarms in the statistics of its ``at``-th sampled call and records how many payloads had left by then."""

    def __init__(self, at: int, sent: Any):
        super().__init__()
        self.at = at
        self.sent = sent
        self.seen = 0
        self.sent_at_alarm: int | None = None

    def _inspect(self, x: torch.Tensor, xq: torch.Tensor | None, reasons: list[str]) -> None:
        super()._inspect(x, xq, reasons)  # every path's statistics (guard, inspect_planned), under the lock
        self.seen += 1
        if self.seen == self.at:
            reasons.append("test: statistics alarm")
            self.sent_at_alarm = self.sent()


@pytest.mark.parametrize("at", [18, 21])
@pytest.mark.parametrize("noise_async", [False, True])
@pytest.mark.parametrize("lane", [False, True])
def test_sampled_alarm_stops_the_next_payload(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                              monkeypatch: pytest.MonkeyPatch, noise_async: bool,
                                              lane: bool, at: int) -> None:
    """An alarm raised by a sampled trip's statistics blocks the very next payload (fail closed), with and
    without the driver: the driver ends its call at each sampled trip, so the statistics run (or are
    joined) before the next trip's sampling draw and payload, as on the per-trip path; with inline statistics
    the sampled trip's own payload never leaves (it is handed back before its send). The alarm comes in a
    decode pass (past the first 16 always-sampled calls: the prompt pass and the first decode pass)."""
    from pgw.wire.lane import native_channel_class

    if lane and native_channel_class() is None:
        pytest.skip("pgw_fast without the native lane channel")
    got = []
    for driver in (False, True):
        with tempfile.TemporaryDirectory(prefix="pgwd") as d:
            ln = _Lane(qwen_pack, d) if lane else None
            client = ln.client if ln is not None else LocalComputeClient(qwen_pack, record=True)
            def count(ln: Any = ln, client: Any = client) -> int:
                return len(ln.local.lane.trips) + len(ln.local.received) if ln is not None else len(client.received)

            noise = _AlarmAt(at, count)
            try:
                with pytest.raises(IntegrityError, match="outbound-noise monitor alarm"):
                    _run(qwen_pack, store_template, monkeypatch, driver, client=client, noise_async=noise_async,
                         noise=noise)
            finally:
                if ln is not None:
                    ln.client.close()
            assert noise.sent_at_alarm is not None
            if not noise_async:  # inline statistics: nothing left after the alarm
                assert count() == noise.sent_at_alarm, (driver, count(), noise.sent_at_alarm)
            got.append(count())
    # the same trip carried the alarm and the same payloads had left when it stopped traffic (with
    # noise_async_sampled the statistics run beside the sampled trip's own send and are joined before the next)
    assert got[0] == got[1]
