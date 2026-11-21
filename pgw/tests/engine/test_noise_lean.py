"""Noise-monitor lean path: non-sampled calls are checked on the packed payload (no int64 x_hat / xq)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A8, RING_K32_A12_C27, RING_K40_A12, RING_K48_A20, IntegrityError, LinearId, RingParams
from pgw.core import native as core_native
from pgw.core.packing import pack_ring
from pgw.engine import InMemoryPack, LinearCall, LocalComputeClient, LocalMaskReservation, MaskedBackend, PlainBackend
from pgw.observability.noise import OutboundNoiseMonitor

QKV0 = LinearId.parse("0.qkv")
RINGS = [RING_K32_A8, RING_K32_A12_C27, RING_K40_A12, RING_K48_A20]
native_only = pytest.mark.skipif(not core_native.native_available("finish_group"), reason="pgw_fast not built")


def _x(rows: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, n, generator=g) * 3


def _uniform(rows: int, n: int, ring: RingParams, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    hi = torch.randint(0, 1 << (ring.k - 24), (rows, n), generator=g, dtype=torch.int64)
    lo = torch.randint(0, 1 << 24, (rows, n), generator=g, dtype=torch.int64)
    return (hi << 24) | lo


@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
@pytest.mark.parametrize("n", [64, 896, 4864])
def test_packed_fingerprints_equal_unpacked(ring: RingParams, n: int) -> None:
    """guard_packed computes exactly the duplicate-row fingerprints guard computes (same history)."""
    a = OutboundNoiseMonitor(ring, seed=11, always_sample_first=0, sample_rate=1e-9)
    b = OutboundNoiseMonitor(ring, seed=11, always_sample_first=0, sample_rate=1e-9)
    for i, rows in enumerate((1, 3, 17)):
        x = _uniform(rows, n, ring, 100 + i)
        a.guard(x)
        b.guard_packed(pack_ring(x, ring), rows, n)
    assert a._dup_seen == b._dup_seen and list(a._dup_order) == list(b._dup_order)
    assert a.report().rows_checked == b.report().rows_checked == 21
    # and a repeated row alarms on both forms
    x = _uniform(1, n, ring, 100)
    with pytest.raises(IntegrityError):
        a.guard(x)
    with pytest.raises(IntegrityError):
        b.guard_packed(pack_ring(x, ring), 1, n)


def test_plan_call_draws_like_guard() -> None:
    """plan_call + guard(planned) / guard_packed samples the same calls as guard alone."""
    ring = RING_K32_A12_C27
    a = OutboundNoiseMonitor(ring, seed=5, always_sample_first=3, sample_rate=0.3, max_values_per_call=256)
    b = OutboundNoiseMonitor(ring, seed=5, always_sample_first=3, sample_rate=0.3, max_values_per_call=256)
    lean = 0
    for i in range(60):
        x = _uniform(4, 128, ring, i)
        a.guard(x)
        if b.plan_call():
            b.guard(x, planned=True)
        else:
            lean += 1
            b.guard_packed(pack_ring(x, ring), 4, 128)
    ra, rb = a.report(), b.report()
    assert lean > 20
    assert (ra.calls, ra.sampled_calls, ra.samples, ra.rows_checked) == (rb.calls, rb.sampled_calls, rb.samples,
                                                                          rb.rows_checked)
    assert ra.calls == 60 and not ra.alarm and not rb.alarm


@pytest.mark.skipif(not core_native.native_available("noise_dup_keys"), reason="pgw_fast without noise_dup_keys")
@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
@pytest.mark.parametrize("n", [64, 896, 18944])
def test_native_fingerprints_equal_numpy(ring: RingParams, n: int) -> None:
    """L2: guard_packed's native fingerprints (pgw_fast.noise_dup_keys) equal the numpy ones bit for bit."""
    a = OutboundNoiseMonitor(ring, seed=3, always_sample_first=0, sample_rate=1e-9)
    b = OutboundNoiseMonitor(ring, seed=3, always_sample_first=0, sample_rate=1e-9)
    assert b._native_keys is not None
    a._native_keys = None  # the numpy form
    for i, rows in enumerate((1, 2, 9)):
        payload = pack_ring(_uniform(rows, n, ring, 200 + i), ring)
        a.guard_packed(payload, rows, n)
        b.guard_packed(bytearray(payload) if i == 1 else payload, rows, n)
    assert a._dup_seen == b._dup_seen and list(a._dup_order) == list(b._dup_order)
    assert a.report().rows_checked == b.report().rows_checked == 12
    with pytest.raises(IntegrityError):
        b.guard_packed(pack_ring(_uniform(1, n, ring, 200), ring), 1, n)


def test_guard_packed_rejects_bad_size() -> None:
    from pgw.core import ShapeError

    mon = OutboundNoiseMonitor(RING_K32_A8)
    with pytest.raises(ShapeError):
        mon.guard_packed(b"\0" * 10, 1, 4)


class _SpyMonitor(OutboundNoiseMonitor):
    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.full = 0
        self.packed = 0

    def guard(self, x_hat: torch.Tensor, xq: torch.Tensor | None = None, *, planned: bool | None = None) -> None:
        self.full += 1
        assert xq is not None
        super().guard(x_hat, xq, planned=planned)

    def guard_packed(self, payload: Any, rows: int, n: int) -> None:
        self.packed += 1
        super().guard_packed(payload, rows, n)


@native_only
@pytest.mark.parametrize("ring", [RING_K32_A12_C27, RING_K40_A12], ids=lambda r: r.name)
def test_lean_and_sampled_calls_stay_exact(qwen_pack: InMemoryPack, ring: RingParams) -> None:
    mon = _SpyMonitor(ring, seed=3, always_sample_first=2, sample_rate=0.25)
    client = LocalComputeClient(qwen_pack, record=True)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint, noise_monitor=mon)
    plain = PlainBackend(qwen_pack, ring)
    n = qwen_pack.arch.hidden_size
    for i in range(24):
        res = LocalMaskReservation(qwen_pack, ring, 4)
        x = _x(1 + i % 4, n, i)
        assert torch.equal(be.forward(LinearCall(QKV0, x, session=res)), plain.forward(LinearCall(QKV0, x)))
    assert mon.full >= 2 and mon.packed >= 10 and mon.full + mon.packed == 24
    rep = mon.report()
    assert rep.calls == 24 and rep.sampled_calls == mon.full and not rep.alarm
    assert rep.rows_checked == sum(1 + i % 4 for i in range(24))


class _ReplayReservation(LocalMaskReservation):
    """Injected bug: hands out the same one-time pad again (mask reuse)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self._first: tuple[torch.Tensor, torch.Tensor] | None = None

    def take(self, lid: LinearId, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, wr = super().take(lid, rows)
        if self._first is None:
            self._first = (r.clone(), wr.clone())
        return self._first[0].clone(), self._first[1].clone()


class _UnmaskedReservation(LocalMaskReservation):
    """Injected bug: a zero pad (the payload is the plaintext xq)."""

    def take(self, lid: LinearId, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, wr = super().take(lid, rows)
        return torch.zeros_like(r), torch.zeros_like(wr)


@native_only
def test_lean_path_alarms_on_reused_mask(qwen_pack: InMemoryPack) -> None:
    ring = RING_K32_A12_C27
    mon = _SpyMonitor(ring, always_sample_first=0, sample_rate=1e-9)
    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint, noise_monitor=mon)
    n = qwen_pack.arch.hidden_size
    res = _ReplayReservation(qwen_pack, ring, 8)
    be.forward(LinearCall(QKV0, _x(1, n, 1), session=res))
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, _x(1, n, 2), session=res))
    assert client.requests == 1  # blocked before sending
    assert mon.full == 0 and mon.packed == 2  # caught on the lean path
    assert mon.report().duplicate_rows == 1


@native_only
def test_lean_path_alarms_on_unmasked_send(qwen_pack: InMemoryPack) -> None:
    ring = RING_K32_A12_C27
    mon = _SpyMonitor(ring, always_sample_first=0, sample_rate=1e-9)
    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint, noise_monitor=mon)
    n = qwen_pack.arch.hidden_size
    x = _x(6, n, 4).abs() + 0.5  # plaintext rows with the same top bits everywhere
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, x, session=_UnmaskedReservation(qwen_pack, ring, 8)))
    assert client.requests == 0 and mon.full == 0 and mon.packed == 1
    # an alarmed monitor stops the next call before any mask row is taken
    res = LocalMaskReservation(qwen_pack, ring, 8)
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, _x(1, n, 5), session=res))
    assert res.remaining(QKV0) == 8


@native_only
def test_sampled_call_alarms_on_unmasked_send(qwen_pack: InMemoryPack) -> None:
    """Sampled calls keep the full statistical path (here: the first call is always sampled)."""
    ring = RING_K32_A12_C27
    mon = _SpyMonitor(ring, always_sample_first=1, sample_rate=1e-9, window_values=4096)
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), ring, qwen_pack.fingerprint,
                       noise_monitor=mon)
    n = qwen_pack.arch.hidden_size
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, _x(8, n, 6).abs() + 0.5, session=_UnmaskedReservation(qwen_pack, ring, 8)))
    assert mon.full == 1 and mon.packed == 0
