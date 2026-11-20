"""Chat-speed WP-A4b: deferred Freivalds check (TripEngine).

The output half of a trip's check runs on the native helper thread while the gateway continues. It is
joined before the same session's next payload is built, and the model's last linear is checked
synchronously, so a wrong service result

* raises :class:`IntegrityError` before the session sends anything more (the recording service sees no
  further request), and
* never lets logits out (the forward pass raises).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from pgw.core import IntegrityError, native
from pgw.core.ring import DEFAULT_RING
from pgw.core.types import LinearId, LinearKind
from pgw.engine import (
    InMemoryPack,
    LinearCall,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SequenceInput,
    TransformerModel,
)

pytestmark = pytest.mark.skipif(not native.native_available("trip_freivalds_submit"),
                                reason="pgw_fast trip kernels not built")


class Recording(LocalComputeClient):
    """In-process service that records the linear of every item it receives, in order, and can return
    a wrong result for one linear (``y_hat + delta`` on one element: still in range, so only the
    Freivalds check can catch it)."""

    def __init__(self, pack: Any):
        self.armed: tuple[str, int] | None = None  # (linear, delta)
        super().__init__(pack, tamper=self._tamper)
        self.seen: list[str] = []

    def _tamper(self, lid: str, y: torch.Tensor) -> torch.Tensor:
        if self.armed is not None and lid == self.armed[0]:
            y = y.clone()
            y[0, 0] += self.armed[1]
        return y

    def compute_sync(self, request: Any) -> Any:
        self.seen.extend(it.linear for it in request.items)
        return super().compute_sync(request)


def _model(pack: InMemoryPack, client: Any, **kw: Any) -> TransformerModel:
    be = MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint, trip_engine=True, **kw)
    be.defer_min_out = 0  # tiny pack: every linear is small; defer them all
    return TransformerModel(pack, be)


def _run(model: TransformerModel, seqs: list[SequenceInput], mode: str) -> list[Any]:
    if mode == "sync":
        return model.forward_batch(seqs)
    return asyncio.run(model.aforward_batch(seqs))


def _setup(pack: InMemoryPack, mode: str, **kw: Any) -> tuple[TransformerModel, Recording, Any, Any, list[Any]]:
    client = Recording(pack)
    failures: list[Any] = []
    model = _model(pack, client, on_integrity_failure=failures.append, sync_trips=True, **kw)
    model.sync_decode = mode == "sync_decode"
    res = LocalMaskReservation(pack, DEFAULT_RING, 32, seed=bytes([5]) * 32)
    cache = model.new_cache(32)
    _run(model, [SequenceInput([1, 2, 3, 4], cache, res)], "sync" if mode == "sync" else "async")  # prefill: fine
    return model, client, res, cache, failures


MODES = ["sync", "async", "sync_decode"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("linear", ["0.qkv", "0.gate_up", "1.o"])
def test_wrong_result_stops_the_session_before_its_next_send(qwen_pack: InMemoryPack, mode: str, linear: str) -> None:
    model, client, res, cache, failures = _setup(qwen_pack, mode)
    be = model.backend
    assert be.deferred_check and be.stats.deferred_checks > 0
    client.armed = (linear, 1)
    client.seen.clear()
    with pytest.raises(IntegrityError) as ei:
        _run(model, [SequenceInput([7], cache, res)], "sync" if mode == "sync" else "async")
    # the wrong result was the last thing the provider computed: nothing was sent after it
    assert client.seen[-1] == linear, client.seen
    assert ei.value.details.get("deferred") is True and ei.value.details["linear"] == linear
    assert len(failures) == 1 and be.stats.integrity_failures == 1
    model.close()
    res.close()


@pytest.mark.parametrize("mode", MODES)
def test_last_linear_is_checked_before_logits_leave(qwen_pack: InMemoryPack, mode: str) -> None:
    model, client, res, cache, failures = _setup(qwen_pack, mode)
    last = f"{qwen_pack.arch.num_layers - 1}.down"
    assert not model.backend._defer_ok(LinearId.parse(last))
    assert model.backend._defer_ok(LinearId.parse("0.down"))
    client.armed = (last, 1)
    with pytest.raises(IntegrityError) as ei:
        _run(model, [SequenceInput([7], cache, res)], "sync" if mode == "sync" else "async")
    assert client.seen[-1] == last
    assert "deferred" not in ei.value.details and len(failures) == 1
    model.close()
    res.close()


@pytest.mark.parametrize("mode", MODES)
def test_out_of_range_result_still_fails_at_once(qwen_pack: InMemoryPack, mode: str) -> None:
    model, client, res, cache, _ = _setup(qwen_pack, mode)
    client.armed = ("0.qkv", 1 << 30)  # far outside the output bound: the range check in finish refuses it
    with pytest.raises(IntegrityError) as ei:
        _run(model, [SequenceInput([7], cache, res)], "sync" if mode == "sync" else "async")
    assert client.seen[-1] == "0.qkv" and "deferred" not in ei.value.details
    model.close()
    res.close()


def test_without_deferral_the_same_error_at_the_same_point(qwen_pack: InMemoryPack) -> None:
    model, client, res, cache, _ = _setup(qwen_pack, "sync", deferred_check=False)
    assert not model.backend.deferred_check
    client.armed = ("0.gate_up", 1)
    client.seen.clear()
    with pytest.raises(IntegrityError) as ei:
        model.forward_batch([SequenceInput([7], cache, res)])
    assert client.seen[-1] == "0.gate_up" and "deferred" not in ei.value.details
    model.close()
    res.close()


def test_pending_checks_are_per_session(qwen_pack: InMemoryPack) -> None:
    """A session's pending check is joined by its own next call, not by another session's."""
    client = Recording(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint, trip_engine=True)
    be.defer_min_out = 0
    plain = PlainBackend(qwen_pack, DEFAULT_RING)
    ra = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8, seed=bytes([1]) * 32)
    rb = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8, seed=bytes([2]) * 32)
    n = qwen_pack.arch.hidden_size
    qkv, gu = LinearId(0, LinearKind.QKV), LinearId(0, LinearKind.GATE_UP)
    x = torch.randn(1, n, generator=torch.Generator().manual_seed(0))
    assert torch.equal(be.forward(LinearCall(qkv, x, ra)), plain.forward(LinearCall(qkv, x)))
    assert be.pending_checks == 1
    client.armed = ("0.qkv", 1)
    be.forward(LinearCall(qkv, x, rb))  # B's wrong result: its check is pending
    client.armed = None
    # A's check may already be harvested by B's call (finished checks of other sessions are dropped)
    assert 1 <= be.pending_checks <= 2
    assert torch.equal(be.forward(LinearCall(gu, x, ra)), plain.forward(LinearCall(gu, x)))  # A's joined: fine
    assert be.stats.integrity_failures <= 1  # B's failure: reported if harvested, raised only for B
    seen = len(client.seen)
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(gu, x, rb))  # B's next call joins B's check first
    assert len(client.seen) == seen  # and sends nothing
    be.join_checks()  # A's last one is fine
    assert be.pending_checks == 0
    be.close()


def test_close_and_join_checks(qwen_pack: InMemoryPack) -> None:
    client = Recording(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint, trip_engine=True)
    be.defer_min_out = 0
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8)
    x = torch.randn(2, qwen_pack.arch.hidden_size)
    client.armed = ("0.qkv", 1)
    be.forward(LinearCall(LinearId(0, LinearKind.QKV), x, res))
    with pytest.raises(IntegrityError):
        be.join_checks()
    be.forward(LinearCall(LinearId(0, LinearKind.QKV), x, res))
    assert be.pending_checks == 1
    be.close()  # waits for the helper, never raises
    assert be.pending_checks == 0


def test_freivalds_rate_zero_defers_nothing(qwen_pack: InMemoryPack) -> None:
    client = Recording(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint, trip_engine=True,
                       freivalds_rate=0.0)
    be.defer_min_out = 0
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8)
    be.forward(LinearCall(LinearId(0, LinearKind.QKV), torch.randn(1, qwen_pack.arch.hidden_size), res))
    assert be.pending_checks == 0 and be.stats.freivalds_skipped == 1


def _two_sessions(pack: InMemoryPack, **kw: Any) -> tuple[MaskedBackend, Recording, Any, Any, list[Any]]:
    client = Recording(pack)
    failures: list[Any] = []
    be = MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint, trip_engine=True,
                       on_integrity_failure=failures.append, **kw)
    be.defer_min_out = 0
    ra = LocalMaskReservation(pack, DEFAULT_RING, 8, seed=bytes([1]) * 32)
    rb = LocalMaskReservation(pack, DEFAULT_RING, 8, seed=bytes([2]) * 32)
    return be, client, ra, rb, failures


def test_close_reports_failed_checks(qwen_pack: InMemoryPack) -> None:
    be, client, ra, _, failures = _two_sessions(qwen_pack)
    client.armed = ("0.qkv", 1)
    be.forward(LinearCall(LinearId(0, LinearKind.QKV), torch.randn(1, qwen_pack.arch.hidden_size), ra))
    assert be.pending_checks == 1 and be.stats.integrity_failures == 0
    be.close()  # never raises, but the verdict is not lost
    assert be.pending_checks == 0 and be.stats.integrity_failures == 1
    assert len(failures) == 1 and failures[0].details.get("deferred") is True


def test_pending_limit_never_raises_another_sessions_error(qwen_pack: InMemoryPack) -> None:
    """Above ``max_pending_checks`` other sessions' checks are waited for, but a failure is reported
    at once and raised in its own session's next call, never inside an unrelated request."""
    be, client, ra, rb, failures = _two_sessions(qwen_pack)
    be.max_pending_checks = 0
    plain = PlainBackend(qwen_pack, DEFAULT_RING)
    qkv, gu = LinearId(0, LinearKind.QKV), LinearId(0, LinearKind.GATE_UP)
    x = torch.randn(1, qwen_pack.arch.hidden_size, generator=torch.Generator().manual_seed(3))
    client.armed = ("0.qkv", 1)
    be.forward(LinearCall(qkv, x, rb))  # B's wrong result: pending
    client.armed = None
    assert torch.equal(be.forward(LinearCall(qkv, x, ra)), plain.forward(LinearCall(qkv, x)))  # A: fine
    assert len(failures) == 1 and be.stats.integrity_failures == 1  # B's failure reported meanwhile
    seen = len(client.seen)
    with pytest.raises(IntegrityError) as ei:
        be.forward(LinearCall(gu, x, rb))  # B's next call raises it and sends nothing
    assert len(client.seen) == seen and ei.value.details.get("deferred") is True
    assert len(failures) == 1  # reported once
    be.join_checks()
    assert be.pending_checks == 0
    be.close()


def test_finished_checks_of_other_sessions_are_harvested(qwen_pack: InMemoryPack) -> None:
    """A session aborted with a check pending (e.g. between gate_up and down) is not held: another
    session's call drops its finished check (and its reservation references)."""
    be, _, ra, rb, failures = _two_sessions(qwen_pack)
    qkv = LinearId(0, LinearKind.QKV)
    x = torch.randn(1, qwen_pack.arch.hidden_size)
    be.forward(LinearCall(qkv, x, rb))
    with be._pending_lock:
        b_check = be._pending[0].check
    b_check.wait()  # finished (the helper is fast; make it certain)
    be.forward(LinearCall(qkv, x, ra))
    with be._pending_lock:
        assert all(not any(r is rb for r in p.reservations) for p in be._pending)
    assert be.stats.freivalds_checked >= 1 and not failures
    be.close()


def test_release_session_checks(qwen_pack: InMemoryPack) -> None:
    be, client, ra, rb, failures = _two_sessions(qwen_pack)
    qkv = LinearId(0, LinearKind.QKV)
    x = torch.randn(1, qwen_pack.arch.hidden_size)
    client.armed = ("0.qkv", 1)
    be.forward(LinearCall(qkv, x, rb))
    client.armed = None
    be.forward(LinearCall(qkv, x, ra))
    be.release_session_checks(rb)  # reports B's failure, never raises
    assert be.pending_checks == 1 and len(failures) == 1 and be.stats.integrity_failures == 1
    be.forward(LinearCall(qkv, x, ra))  # A unaffected
    be.close()
    assert len(failures) == 1


def test_deferred_chunk_counts_as_checked_only_after_its_verdict(qwen_pack: InMemoryPack) -> None:
    be, _, ra, _, _ = _two_sessions(qwen_pack)
    be.forward(LinearCall(LinearId(0, LinearKind.QKV), torch.randn(1, qwen_pack.arch.hidden_size), ra))
    assert be.pending_checks == 1 and be.stats.freivalds_checked == 0 and be.stats.deferred_checks == 1
    be.join_checks()
    assert be.stats.freivalds_checked == 1
    be.defer_min_out = 1 << 30  # not deferred: counted at once
    be.forward(LinearCall(LinearId(0, LinearKind.QKV), torch.randn(1, qwen_pack.arch.hidden_size), ra))
    assert be.pending_checks == 0 and be.stats.freivalds_checked == 2
    be.close()
