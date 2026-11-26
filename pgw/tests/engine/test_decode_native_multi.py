"""Several answers per pass in the native decode driver (``PGW_DECODE_NATIVE_MULTI``; ``DecodeDriver.run_multi``).

A decode pass of several answers runs its decoder layers in the driver (one native call per stretch of trips,
GIL released), with one lease per answer and trip, the answers' items in one request per trip. Against the
per-trip path of several answers (``PGW_TRIP_NATIVE_MULTI``: ``_prepare_trip_leases`` + ``_finish_leases``) on
real store reservations (windowed, across chunk, part and segment boundaries), ring and p31 extents, with the noise
monitor sampling half the calls:

* the same logits as the plain reference with the same native float ops, bit for bit;
* the same provider requests: items, rows, payload bytes and order; the same noise draws and statistics, the same
  checks counted;
* a sampled trip's statistics run before its payloads leave (an alarm stops it); a wrong result in any answer's
  item fails the pass (synchronous and deferred checks); payloads stay single use; buffers wiped;
* the lane path (one native multi call per trip) is covered against the real provider (tests/sdk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import native
from pgw.core.errors import IntegrityError
from pgw.core.ring import DEFAULT_RING
from pgw.engine import InMemoryPack, LocalComputeClient, PlainBackend, TransformerModel
from pgw.engine import decode_native as dn
from pgw.engine.model import SequenceInput
from pgw.wire.messages import ComputeResponse, ComputeResult
from tests.engine.test_model_hooks import _ids, _rows
from tests.engine.test_trip_engine import _masked, _same
from tests.engine.test_trip_native import _Seeded, _Store, p31_template, store_template  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(not native.native_available("DecodeDriver", "TripLease", "float_rope_attend")
                                or not hasattr(native.fast.DecodeDriver, "run_multi"),
                                reason="pgw_fast without the native several-answer decode driver")

K = 3
STEPS = 12


class _Multi(LocalComputeClient):
    """A local client that says one multi call carries 64 rows (the driver's several-answer eligibility); with no
    native lane channel the driver hands every trip back after its native prepare (same request, finished by
    ``_finish_leases``). Optionally flips a byte of answer ``item``'s reply in the ``after``-th request of ``K``
    items."""

    def __init__(self, pack: Any, after: int = 0, item: int = 0):
        super().__init__(pack, record=True)
        self.after, self.item, self.n = after, item, 0
        self.reqs: list[tuple[int, tuple[Any, ...]]] = []

    def multi_rows(self) -> int:
        return 64

    def compute_sync(self, request: Any) -> Any:
        self.reqs.append((len(request.request_id), tuple((it.linear, it.rows) for it in request.items)))
        r = super().compute_sync(request)
        if len(r.items) < K:  # count the lockstep passes' trips only (one item per answer)
            return r
        self.n += 1
        if self.n != self.after:
            return r
        items = list(r.items)
        it = items[self.item]
        y = bytearray(it.y_hat)
        y[2] ^= 8
        items[self.item] = ComputeResult(it.linear, it.rows, bytes(y))
        return ComputeResponse(r.request_id, tuple(items))


def _scenario(model: TransformerModel, sess: Any) -> dict[str, torch.Tensor]:
    """Three answers: prompts alone, then ``STEPS`` decode steps in lockstep (one with a 3-row pass for answer 1)."""
    out: dict[str, torch.Tensor] = {}
    caches = [model.new_cache(256) for _ in range(K)]
    res = [sess(200, 11 + a) for a in range(K)]
    for a in range(K):
        out[f"p{a}"] = _rows(model.forward_batch([SequenceInput(_ids(5 + 3 * a, a), caches[a], res[a])])[0])
    for s in range(STEPS):
        seqs = []
        for a in range(K):
            ids = [7 + s, 8 + s, 9 + s] if (a == 1 and s == 5) else [20 + 3 * s + a]
            seqs.append(SequenceInput(ids, caches[a], res[a], all_logits=True))
        for a, lg in enumerate(model.forward_batch(seqs)):
            out[f"s{s}_{a}"] = _rows(lg)
    for r in res:
        if r is not None:
            r.close()
    return out


@pytest.fixture(scope="module")
def ref_native(qwen_pack: InMemoryPack) -> dict[str, torch.Tensor]:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    model.set_float_ops("native")
    try:
        return _scenario(model, lambda n, s: None)
    finally:
        model.close()


def _run(pack: Any, template: Path, monkeypatch: pytest.MonkeyPatch, multi: bool, client: Any = None,
         noise: Any = None, per_call: int = 0, **kw: Any) -> tuple[dict[str, torch.Tensor], list[Any], Any, Any]:
    for k, v in (("PGW_TRIP_FAST", "1"), ("PGW_TRIP_NATIVE", "1"), ("PGW_TRIP_NATIVE_MULTI", "1"),
                 ("PGW_DECODE_NATIVE", "1"), ("PGW_NOISE_ASYNC_SAMPLED", "0"),
                 ("PGW_DECODE_NATIVE_MULTI", "1" if multi else "0"), ("PGW_DECODE_NATIVE_LAYERS", str(per_call))):
        monkeypatch.setenv(k, v)
    st = _Store(template, pack)
    client = client or _Multi(pack)
    noise = noise or _Seeded()
    model = _masked(pack, client=client, trip_engine=True, sync_trips=True, noise_monitor=noise, **kw)
    model.set_float_ops("native")
    be = model.backend
    be.freivalds_rate = 1.0
    ran = {"multi": 0}
    real = dn.NativeDecode.run_multi

    def spy(self: Any, s: Any) -> None:
        ran["multi"] += 1
        real(self, s)

    monkeypatch.setattr(dn.NativeDecode, "run_multi", spy)
    try:
        assert be.decode_native and be.trip_native_multi
        out = _scenario(model, lambda n, s: st.windowed(n, 6, 4))
        be._join_noise()
        be.join_checks()
    finally:
        model.close()
        st.close()
    rep = noise.report()
    stats = (rep.calls, rep.sampled_calls, rep.samples, rep.rows_checked, rep.duplicate_rows, rep.windows)
    sent = [(lid, tuple(x.shape), x.numpy().tobytes()) for lid, x in client.received]
    snap = be.stats.snapshot()
    return out, sent, stats, (ran["multi"], snap, list(client.reqs))


def test_default_off(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_DECODE_NATIVE_MULTI", raising=False)
    monkeypatch.setenv("PGW_TRIP_FAST", "1")
    monkeypatch.setenv("PGW_TRIP_NATIVE", "1")
    monkeypatch.setenv("PGW_DECODE_NATIVE", "1")
    model = _masked(qwen_pack, client=_Multi(qwen_pack), trip_engine=True, sync_trips=True)
    model.set_float_ops("native")
    try:
        nd = dn.driver_for(model)
        assert nd is not None and not nd.multi
    finally:
        model.close()


@pytest.mark.parametrize("per_call", [0, 1])
def test_multi_equals_plain_and_per_trip_path(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                              monkeypatch: pytest.MonkeyPatch, ref_native: dict[str, torch.Tensor],
                                              per_call: int) -> None:
    base, base_sent, base_stats, (base_ran, base_snap, base_req) = _run(qwen_pack, store_template, monkeypatch,
                                                                        False)
    _same(ref_native, base)
    assert base_ran == 0
    out, sent, stats, (ran, snap, req) = _run(qwen_pack, store_template, monkeypatch, True, per_call=per_call)
    _same(ref_native, out)  # masked == plain, bit for bit
    assert ran == STEPS  # every lockstep decode pass ran in the driver
    assert sent == base_sent  # byte-identical provider payloads, same order
    assert req == base_req  # the same requests: id format, items (linear, rows) in answer order
    assert stats == base_stats  # same noise draws and statistics
    for k in ("calls", "rows", "round_trips", "bytes_up", "bytes_down", "freivalds_checked", "freivalds_skipped"):
        assert snap[k] == base_snap[k], k
    assert snap["integrity_failures"] == 0


def test_p31_multi_equals_plain_and_per_trip_path(qwen_pack: InMemoryPack, p31_template: Path,  # noqa: F811
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  ref_native: dict[str, torch.Tensor]) -> None:
    kw = {"prime_check": True, "min_soundness_bits": 20}
    base, base_sent, base_stats, (_, base_snap, base_req) = _run(qwen_pack, p31_template, monkeypatch, False, **kw)
    out, sent, stats, (ran, snap, req) = _run(qwen_pack, p31_template, monkeypatch, True, **kw)
    _same(ref_native, base)
    _same(ref_native, out)
    assert ran == STEPS and sent == base_sent and req == base_req and stats == base_stats
    assert snap["freivalds_checked"] == base_snap["freivalds_checked"] > 0
    assert snap["deferred_checks"] == base_snap["deferred_checks"] == 0  # p31: always synchronous
    assert snap["integrity_failures"] == 0


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("item", [0, 2])
def test_wrong_result_in_any_answer_fails(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                          monkeypatch: pytest.MonkeyPatch, defer: bool, item: int) -> None:
    monkeypatch.setenv("PGW_DEFERRED_CHECK", "1" if defer else "0")
    for after in (3, 17, 40):  # trips of the lockstep decode passes
        client = _Multi(qwen_pack, after=after, item=item)
        with pytest.raises(IntegrityError):
            _run(qwen_pack, store_template, monkeypatch, True, client=client)
        payloads = [x.numpy().tobytes() for _, x in client.received]
        assert len(set(payloads)) == len(payloads)  # every payload from fresh rows


class _AlarmAt(_Seeded):
    def __init__(self, at: int, sent: Any):
        super().__init__()
        self.at, self.sent, self.seen = at, sent, 0
        self.sent_at_alarm: int | None = None

    def _inspect(self, x: torch.Tensor, xq: torch.Tensor | None, reasons: list[str]) -> None:
        super()._inspect(x, xq, reasons)
        self.seen += 1
        if self.seen == self.at:
            reasons.append("test: statistics alarm")
            self.sent_at_alarm = self.sent()


@pytest.mark.parametrize("at", [40, 57])
def test_sampled_alarm_stops_the_payload(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                         monkeypatch: pytest.MonkeyPatch, at: int) -> None:
    """A sampled answer's statistics alarm before its trip's payloads leave (the driver holds the trip), exactly
    where the per-trip path stops: the same payloads had left."""
    got = []
    for multi in (False, True):
        client = _Multi(qwen_pack)
        noise = _AlarmAt(at, lambda c=client: len(c.received))
        with pytest.raises(IntegrityError, match="outbound-noise monitor alarm"):
            _run(qwen_pack, store_template, monkeypatch, multi, client=client, noise=noise)
        assert noise.sent_at_alarm is not None and len(client.received) == noise.sent_at_alarm
        got.append([x.numpy().tobytes() for _, x in client.received])
    assert got[0] == got[1]


def test_driver_multi_unit_wipes_and_validation(qwen_pack: InMemoryPack) -> None:
    """``begin_multi`` / ``abort`` / argument checks of the native side."""
    import numpy as np

    from pgw.engine.model import _np32

    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    model.set_float_ops("native")
    try:
        ops = model.float_ops
        drv = native.fast.DecodeDriver(int(model.arch.hidden_size), int(model.num_heads), int(model.num_kv_heads),
                                       int(model.head_dim), int(model.split_gate_up[0]), float(model.scale),
                                       float(model.eps), int(ops.attn_threads), int(ops.attn_block),
                                       bool(ops.kv_bf16), int(model.num_layers))
        for layer in range(model.num_layers):
            drv.set_layer(layer, _np32(model.input_norms[layer]), _np32(model.post_norms[layer]))
        h = [np.ones((1, int(model.arch.hidden_size)), np.float32), np.ones((2, int(model.arch.hidden_size)),
                                                                             np.float32)]
        with pytest.raises(ValueError):
            drv.begin_multi(h, [0])  # one start per answer
        drv.begin_multi(h, [0, 3])
        xs = drv.current_inputs()
        assert [x.shape for x in xs] == [(1, int(model.arch.hidden_size)), (2, int(model.arch.hidden_size))]
        with pytest.raises(ValueError):
            drv.end_multi([np.zeros_like(a) for a in h])  # not finished
        with pytest.raises(ValueError):
            drv.current_inputs()  # wiped: no pass
        drv.begin_multi(h, [0, 3])
        drv.abort()
        with pytest.raises(ValueError):
            drv.run_multi(4, [], b"", b"", [], [])
    finally:
        model.close()


def test_nonfinite_answer_takes_no_row_and_sends_nothing(qwen_pack: InMemoryPack, store_template: Path,  # noqa: F811
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-finite activation in one answer of a lockstep pass fails it before any row of any answer is taken
    for that trip and before anything of it is sent, as on the per-trip path."""
    from pgw.core.errors import RingOverflowError

    real = TransformerModel._embed
    bad_id = 20 + 3 * 4 + 1  # answer 1 at step 4

    def embed(self: Any, t: torch.Tensor) -> torch.Tensor:
        h = real(self, t)
        if t.numel() == 1 and int(t.reshape(-1)[0]) == bad_id:
            h = h.clone()
            h[0, 0] = float("nan")
        return h

    monkeypatch.setattr(TransformerModel, "_embed", embed)
    got = []
    for multi in (False, True):
        client = _Multi(qwen_pack)
        with pytest.raises(RingOverflowError):
            _run(qwen_pack, store_template, monkeypatch, multi, client=client)
        got.append([x.numpy().tobytes() for _, x in client.received])
    assert got[0] == got[1]  # the same payloads had left; none of the failing trip
