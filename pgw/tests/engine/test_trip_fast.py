"""T3: the one-call trip path (``PGW_TRIP_FAST``) and the off-critical-path noise statistics
(``PGW_NOISE_ASYNC_SAMPLED``) are exact: same logits as the plain reference, byte-identical provider
payloads, the same checks (a vacuous / unsound extent takes no row), the same noise statistics, and an
alarm still blocks the next payload. Both default off.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import native
from pgw.core.errors import IntegrityError, MaskStoreError
from pgw.core.ring import DEFAULT_RING
from pgw.engine import InMemoryPack, LocalComputeClient, MaskedBackend, TransformerModel
from pgw.engine import backends as B
from pgw.engine.backends import LinearCall
from pgw.observability.noise import OutboundNoiseMonitor
from tests.engine.test_model_hooks import scenario_run
from tests.engine.test_trip_engine import _masked, _plain_ref, _same

pytestmark = pytest.mark.skipif(not native.native_available("trip_prepare_into"),
                                reason="pgw_fast trip kernels not built")


def _env(monkeypatch: pytest.MonkeyPatch, fast: bool, noise_async: bool = False) -> None:
    monkeypatch.setenv("PGW_TRIP_FAST", "1" if fast else "0")
    monkeypatch.setenv("PGW_NOISE_ASYNC_SAMPLED", "1" if noise_async else "0")


def test_defaults_off(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_TRIP_FAST", raising=False)
    monkeypatch.delenv("PGW_NOISE_ASYNC_SAMPLED", raising=False)
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       trip_engine=True)
    assert not be.trip_fast and not be.noise_async_sampled


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("noise_async", [False, True])
def test_fast_equals_plain(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch, mode: str,
                           noise_async: bool) -> None:
    ref = _plain_ref(qwen_pack)
    _env(monkeypatch, True, noise_async)
    noise = OutboundNoiseMonitor(DEFAULT_RING, sample_rate=0.5, fail_closed=True)
    model = _masked(qwen_pack, trip_engine=True, sync_trips=True, noise_monitor=noise)
    try:
        assert model.backend.trip_fast and model.backend.noise_async_sampled == noise_async
        _same(ref, scenario_run(model, qwen_pack, mode, True))
        assert model.backend.stats.integrity_failures == 0
    finally:
        model.close()


class _Seeded(OutboundNoiseMonitor):
    """A monitor whose sampling draws are reproducible (same draws on both paths)."""

    def __init__(self) -> None:
        super().__init__(DEFAULT_RING, sample_rate=0.5, fail_closed=True)
        import random

        self._rng = random.Random(7)


def _run_bytes(pack: Any, monkeypatch: pytest.MonkeyPatch, fast: bool, noise_async: bool) -> tuple[Any, Any]:
    _env(monkeypatch, fast, noise_async)
    client = LocalComputeClient(pack, record=True)
    noise = _Seeded()
    model = _masked(pack, client=client, trip_engine=True, noise_monitor=noise)
    model.backend.freivalds_rate = 1.0
    try:
        scenario_run(model, pack, "sync", True)
        model.backend._join_noise()
    finally:
        model.close()
    rep = noise.report()
    stats = (rep.calls, rep.sampled_calls, rep.samples, rep.rows_checked, rep.duplicate_rows, rep.windows)
    return [(lid, tuple(x.shape), x.numpy().tobytes()) for lid, x in client.received], stats


def test_provider_bytes_and_noise_stats_identical(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    base_bytes, base_stats = _run_bytes(qwen_pack, monkeypatch, False, False)
    assert base_stats[1] > 0  # some calls were sampled
    for noise_async in (False, True):
        got_bytes, got_stats = _run_bytes(qwen_pack, monkeypatch, True, noise_async)
        assert got_bytes == base_bytes
        assert got_stats == base_stats


def test_request_id_format() -> None:
    for _ in range(200):
        h = B._uuid4_hex()
        assert re.fullmatch(r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}", h)
        import uuid

        assert uuid.UUID(hex=h).version == 4 and uuid.UUID(hex=h).hex == h


def test_async_alarm_blocks_next_payload(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    """An alarm raised by the off-thread statistics blocks the very next prepare (nothing more is sent)."""
    _env(monkeypatch, True, True)
    noise = OutboundNoiseMonitor(DEFAULT_RING, sample_rate=1.0, fail_closed=True)
    client = LocalComputeClient(qwen_pack, record=True)
    model = _masked(qwen_pack, client=client, trip_engine=True, noise_monitor=noise)
    be = model.backend
    real = noise.inspect_planned

    def alarming(x_hat: torch.Tensor, xq: Any) -> None:
        real(x_hat, xq)
        with noise._lock:
            noise._set_alarm(["test alarm"])

    noise.inspect_planned = alarming  # type: ignore[method-assign]
    try:
        from tests.engine.test_model_hooks import _res

        res = _res(qwen_pack, 8, 1)
        spec = qwen_pack.linear_specs()[0]
        x = torch.randn(1, spec.in_features)
        be.forward_many([LinearCall(spec.id, x, res)])
        sent = len(client.received)
        with pytest.raises(IntegrityError):
            be.forward_many([LinearCall(spec.id, x, res)])
        assert len(client.received) == sent  # the next payload was never sent
        res.close()
    finally:
        model.close()


def test_store_reservation_soundness_before_take(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a store reservation the extent's soundness check runs inside the take: a failing one takes no row."""
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter

    _env(monkeypatch, True, False)
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        store = MaskStore.open(work / "store", FileKeyProvider(work / "kek", create=True), fsync=False,
                               counter=FileMonotonicCounter(work / "ctr", fsync=False))
        MaskGenerator(qwen_pack, DEFAULT_RING).generate_into_store(store, 16, chunk_slots=8, factory_id="t")
        res = store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 8)
        model = _masked(qwen_pack, trip_engine=True)
        be = model.backend
        try:
            assert getattr(res, "take_checked_verify", False) is True
            spec = qwen_pack.linear_specs()[0]
            x = torch.randn(1, spec.in_features)
            out = be.forward_many([LinearCall(spec.id, x, res)])[0]
            ref = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING,
                                qwen_pack.fingerprint, trip_engine=False)
            res2 = store.reserve(qwen_pack.fingerprint, DEFAULT_RING, 1)
            assert torch.equal(out, ref.forward_many([LinearCall(spec.id, x, res2)])[0])
            res2.close()
            assert res.taken(spec.id) == 1
            be.min_soundness_bits = 10_000.0  # no segment is that sound
            be._sound_ok.clear()
            with pytest.raises(MaskStoreError):
                be.forward_many([LinearCall(spec.id, x, res)])
            assert res.taken(spec.id) == 1  # no row consumed by the failed call
        finally:
            res.close()
            model.close()
            store.close()


def test_fast_path_used(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, True, False)
    model = _masked(qwen_pack, trip_engine=True, sync_trips=True)
    seen: list[int] = []
    real = model.backend._prepare_trip_one

    def spy(call: Any) -> Any:
        seen.append(int(call.x.shape[0]))
        return real(call)

    model.backend._prepare_trip_one = spy  # type: ignore[method-assign]
    try:
        scenario_run(model, qwen_pack, "sync", True)
    finally:
        model.close()
    assert 1 in seen


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_local_lean_equals_plain(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Local linears on the lean one-call path (cached constants, same native kernel): same logits."""
    ref = _plain_ref(qwen_pack)
    _env(monkeypatch, True, False)
    model = _masked(qwen_pack, trip_engine=True, sync_trips=True)
    be = model.backend
    be.configure_local_linears("attention_down")
    hits: list[bool] = []
    real = be._local_lean_one

    def spy(call: Any) -> Any:
        y = real(call)
        hits.append(y is not None)
        return y

    be._local_lean_one = spy  # type: ignore[method-assign]
    try:
        _same(ref, scenario_run(model, qwen_pack, mode, True))
    finally:
        model.close()
    assert any(hits) and be._local_lean


def _unused(m: TransformerModel) -> None:  # pragma: no cover - keeps the import used for type checkers
    del m
