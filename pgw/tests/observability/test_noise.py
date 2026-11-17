"""Tests for the outbound-noise monitor (SPEC 3.3, 11.3)."""

from __future__ import annotations

import json
import math
import threading

import pytest
import torch

from pgw.core import IntegrityError, SeedExpander, mask_activations
from pgw.core.errors import ConfigError, ShapeError
from pgw.core.ring import RING_K32_A8, RING_K40_A12, RING_K48_A20, RingParams, center
from pgw.observability import PGWMetrics
from pgw.observability.noise import (
    NoiseMonitorConfig,
    NoiseReport,
    OutboundNoiseMonitor,
    chi2_sf,
    hoeffding_deviation,
)

RINGS = [RING_K32_A8, RING_K40_A12, RING_K48_A20]


def _xq(shape: tuple[int, int], ring: RingParams, g: torch.Generator) -> torch.Tensor:
    """Realistic quantized activations: Gaussian-ish, saturating at +-qmax."""
    qmax = ring.qmax
    return (torch.randn(shape, generator=g, dtype=torch.float64) * qmax / 4).round().clamp(-qmax, qmax).to(torch.int64)


def _mask(shape: tuple[int, int], ring: RingParams, g: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 1 << ring.k, shape, generator=g, dtype=torch.int64)


def _monitor(ring: RingParams = RING_K40_A12, **kw: object) -> OutboundNoiseMonitor:
    kw.setdefault("sample_rate", 1.0)
    kw.setdefault("seed", 1234)
    return OutboundNoiseMonitor(ring, **kw)  # type: ignore[arg-type]


# ---- helpers -----------------------------------------------------------------------------------
def test_chi2_sf_matches_closed_forms() -> None:
    for x in (0.5, 2.0, 7.3):
        assert chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-12)
    assert chi2_sf(0.0, 255) == 1.0
    assert chi2_sf(255.0, 255) == pytest.approx(0.49, abs=0.03)  # median ~ dof
    assert chi2_sf(415.0, 255) < 2e-9  # ~ the 1e-9 critical value


def test_hoeffding_deviation() -> None:
    assert hoeffding_deviation(0, 1e-9) == math.inf
    t = hoeffding_deviation(10_000, 1e-9)
    assert 2 * math.exp(-2 * 10_000 * t * t) == pytest.approx(1e-9, rel=1e-9)


def test_config_validation() -> None:
    with pytest.raises(ConfigError):
        OutboundNoiseMonitor(RING_K40_A12, sample_rate=0.0)
    with pytest.raises(ConfigError):
        OutboundNoiseMonitor(RING_K40_A12, window_values=100)
    with pytest.raises(ConfigError):
        OutboundNoiseMonitor(RING_K40_A12, alpha=0.5)
    with pytest.raises(ConfigError):
        OutboundNoiseMonitor("K40A12")  # type: ignore[arg-type]
    cfg = NoiseMonitorConfig(sample_rate=0.5, window_values=8192)
    mon = OutboundNoiseMonitor.from_config(RING_K40_A12, cfg, model="m", seed=1)
    assert mon.config == cfg


# ---- genuine traffic: no false alarms ---------------------------------------------------------
@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
def test_masked_traffic_does_not_alarm(ring: RingParams) -> None:
    g = torch.Generator().manual_seed(ring.k)
    mon = _monitor(ring, window_values=1 << 14, feature_window_rows=256)
    for i in range(160):
        shape = (1 + (i % 7), (256, 704, 512)[i % 3])
        xq = _xq(shape, ring, g)
        x_hat = mask_activations(xq, _mask(shape, ring, g), ring)
        mon.guard(x_hat, xq)
    rep = mon.report()
    assert not rep.alarm, rep.reasons
    assert rep.windows >= 10
    assert rep.calls == rep.sampled_calls == 160
    assert rep.duplicate_rows == 0
    assert rep.last_p_top is not None and rep.last_p_top > 1e-6
    assert rep.last_corr is not None and abs(rep.last_corr) < 0.05
    assert rep.last_top_bit == pytest.approx(0.5, abs=0.03)
    assert rep.last_runs == pytest.approx(0.5, abs=0.03)
    assert rep.rows_checked > 0


def test_many_small_windows_no_false_alarm() -> None:
    """Hundreds of minimum-size windows: alpha=1e-9 must not fire on genuine uniform data."""
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(7)
    mon = _monitor(ring, window_values=4096, feature_window_rows=64)
    for _ in range(300):
        xq = _xq((4, 1024), ring, g)
        mon.guard(mask_activations(xq, _mask((4, 1024), ring, g), ring), xq)
    rep = mon.report()
    assert rep.windows == 300
    assert not rep.alarm, rep.reasons


def test_real_csprng_masks_pass() -> None:
    ring = RING_K40_A12
    exp = SeedExpander(b"\x01" * 32)
    g = torch.Generator().manual_seed(3)
    mon = _monitor(ring, window_values=1 << 13)
    for slot in range(40):
        r = exp.ring_uniform_rows(896, ring.k, slot * 2, 2, "r", 0)
        xq = _xq((2, 896), ring, g)
        mon.guard(mask_activations(xq, r, ring), xq)
    assert not mon.report().alarm


# ---- catastrophic failures are caught ---------------------------------------------------------
@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
def test_unmasked_send_alarms_and_blocks(ring: RingParams) -> None:
    g = torch.Generator().manual_seed(1)
    mon = _monitor(ring, duplicate_check=False, window_values=4096)
    xq = _xq((8, 512), ring, g)
    with pytest.raises(IntegrityError) as ei:
        mon.guard(xq & ring.mask, xq)  # forgot to add the mask
    assert "blocked" in ei.value.message
    rep = mon.report()
    assert rep.alarm and rep.alarm_since is not None
    joined = " ".join(rep.reasons)
    assert "top-byte" in joined  # wide activations make the low byte look uniform; the top never does
    assert "correlate" in joined
    # Fail closed: even genuine traffic is blocked now, before inspection.
    good = mask_activations(xq, _mask((8, 512), ring, g), ring)
    calls = rep.calls
    with pytest.raises(IntegrityError):
        mon.guard(good, xq)
    assert mon.report().calls == calls  # rejected on the fast path
    mon.reset()
    assert not mon.report().alarm and mon.report().reasons == []
    mon.guard(good, xq)


@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
def test_mask_reuse_detected_without_sampling(ring: RingParams) -> None:
    """Reusing a slot with a *different* activation is caught by duplicate-row detection alone."""
    g = torch.Generator().manual_seed(2)
    mon = _monitor(ring, sample_rate=1e-6, always_sample_first=0)
    n = 896
    masks = [_mask((1, n), ring, g) for _ in range(50)]
    for r in masks:  # 50 decode steps with fresh masks
        mon.guard(mask_activations(_xq((1, n), ring, g), r, ring))
    assert mon.report().sampled_calls == 0
    with pytest.raises(IntegrityError):
        mon.guard(mask_activations(_xq((1, n), ring, g), masks[17].clone(), ring))
    rep = mon.report()
    assert rep.duplicate_rows == 1
    assert any("mask reuse" in r for r in rep.reasons)


def test_mask_reuse_within_one_call() -> None:
    ring = RING_K32_A8
    g = torch.Generator().manual_seed(5)
    mon = _monitor(ring, sample_rate=1e-6, always_sample_first=0, fail_closed=False)
    r = _mask((1, 1024), ring, g).expand(16, 1024).clone()
    mon.guard(mask_activations(_xq((16, 1024), ring, g), r, ring))
    rep = mon.report()
    assert rep.alarm
    assert rep.duplicate_rows >= 14  # carries can rarely change the fingerprint of a row


def test_duplicate_history_is_bounded() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(9)
    mon = _monitor(ring, sample_rate=1e-6, always_sample_first=0, duplicate_history_rows=64)
    first = _mask((1, 512), ring, g)
    mon.guard(first)
    for _ in range(200):
        mon.guard(_mask((2, 512), ring, g))
    assert len(mon._dup_seen) <= 2 * 64
    mon.guard(first)  # forgotten after eviction: accepted (history is a bounded window)
    assert not mon.report().alarm


def test_narrow_rows_skip_duplicate_check() -> None:
    mon = _monitor(RING_K40_A12, sample_rate=1e-6, always_sample_first=0)
    x = torch.tensor([[5], [5], [5]], dtype=torch.int64)
    mon.guard(x)  # 1 column: too few bits for a safe fingerprint; must not false-alarm
    assert mon.report().duplicate_rows == 0


def test_partial_mask_detected_by_chi_square() -> None:
    """A mask that only randomizes the low 20 bits leaves the top byte non-uniform."""
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(11)
    mon = _monitor(ring, duplicate_check=False, fail_closed=False, window_values=4096)
    for _ in range(4):
        xq = _xq((4, 1024), ring, g)
        weak = torch.randint(0, 1 << 20, (4, 1024), generator=g)
        mon.guard(mask_activations(xq, weak, ring))
    rep = mon.report()
    assert rep.alarm
    assert any("top-byte" in r for r in rep.reasons)
    assert rep.last_p_top is not None and rep.last_p_top < 1e-12


def test_correlation_only_leak_detected() -> None:
    """Payload marginally uniform but carrying xq: only the correlation test can see it."""
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(13)
    mon = _monitor(ring, duplicate_check=False, fail_closed=False, window_values=1 << 14, feature_window_rows=1 << 20)
    for _ in range(8):
        x_hat = _mask((4, 1024), ring, g)
        xq = center(x_hat, ring) >> (ring.k - ring.a)  # A-bit signed, a function of x_hat
        mon.guard(x_hat, xq)
    rep = mon.report()
    assert rep.alarm
    assert all("correlate" in r for r in rep.reasons), rep.reasons
    assert rep.last_p_top is not None and rep.last_p_top > 1e-9
    assert rep.last_corr is not None and abs(rep.last_corr) > 0.1


def test_runs_test_detects_alternating_top_bits() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(17)
    mon = _monitor(ring, duplicate_check=False, fail_closed=False, window_values=4096, feature_window_rows=1 << 20)
    low = _mask((8, 1024), ring, g) & ((1 << (ring.k - 1)) - 1)
    parity = (torch.arange(1024) % 2).expand(8, 1024)
    mon.guard(low | (parity << (ring.k - 1)))
    rep = mon.report()
    assert rep.alarm
    assert any("runs test" in r for r in rep.reasons), rep.reasons
    assert not any("top-byte" in r for r in rep.reasons)
    assert rep.last_runs == pytest.approx(1.0)


def test_per_feature_imbalance_detected() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(19)
    mon = _monitor(ring, duplicate_check=False, fail_closed=False, window_values=1 << 20, feature_window_rows=256)
    for _ in range(32):
        x = _mask((8, 512), ring, g)
        x[:, 100] &= (1 << (ring.k - 1)) - 1  # column 100 never has the top bit set
        mon.guard(x)
    rep = mon.report()
    assert rep.alarm
    assert any("per-feature" in r and "column 100" in r for r in rep.reasons), rep.reasons
    assert rep.windows == 0  # the window-level tests never ran


def test_out_of_ring_values_alarm() -> None:
    ring = RING_K32_A8
    mon = _monitor(ring, duplicate_check=False, fail_closed=False)
    x = torch.full((2, 64), 1 << 33, dtype=torch.int64)
    mon.guard(x)
    assert any("outside the ring" in r for r in mon.report().reasons)
    mon.reset()
    mon.guard(torch.full((2, 64), -1, dtype=torch.int64))
    assert mon.report().alarm


def test_fail_open_reports_but_does_not_raise() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(23)
    mon = _monitor(ring, fail_closed=False, window_values=4096)
    xq = _xq((8, 512), ring, g)
    mon.guard(xq & ring.mask, xq)
    mon.guard(xq & ring.mask, xq)
    assert mon.report().alarm and mon.alarmed


# ---- mechanics ---------------------------------------------------------------------------------
def test_sampling_rate_and_first_calls() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(29)
    mon = OutboundNoiseMonitor(ring, sample_rate=0.1, always_sample_first=5, seed=99)
    for _ in range(1000):
        mon.guard(_mask((1, 64), ring, g))
    rep = mon.report()
    assert rep.calls == 1000
    assert 5 + 50 < rep.sampled_calls < 5 + 160
    assert rep.samples == rep.sampled_calls * 64


def test_values_per_call_are_capped() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(31)
    mon = _monitor(ring, max_values_per_call=1024, window_values=1 << 20)
    mon.guard(_mask((64, 256), ring, g))
    assert mon.report().samples == 1024  # 4 whole rows of 256


def test_flush_and_report_dict() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(37)
    mon = _monitor(ring, window_values=1 << 20)
    assert mon.report().to_dict()["status"] == "warming_up"
    mon.guard(_mask((1, 1000), ring, g))
    mon.flush()  # below the 4096 minimum: nothing evaluated
    assert mon.report().windows == 0
    xq = _xq((8, 1000), ring, g)
    mon.guard(mask_activations(xq, _mask((8, 1000), ring, g), ring), xq)
    mon.flush()
    rep = mon.report()
    assert rep.windows == 1
    d = rep.to_dict()
    assert d["status"] == "ok"
    json.dumps(d)
    assert isinstance(rep, NoiseReport)


def test_input_validation() -> None:
    mon = _monitor()
    with pytest.raises(ShapeError):
        mon.guard(torch.zeros(4, 8))  # float
    with pytest.raises(ShapeError):
        mon.guard(torch.zeros(2, 2, 2, dtype=torch.int64))
    with pytest.raises(ShapeError):
        mon.guard(torch.zeros(2, 8, dtype=torch.int64), torch.zeros(2, 9, dtype=torch.int64))
    with pytest.raises(ShapeError):
        mon.guard([1, 2, 3])  # type: ignore[arg-type]
    # 1-D rows and other integer dtypes are accepted.
    g = torch.Generator().manual_seed(41)
    mon.guard(_mask((1, 128), RING_K40_A12, g).reshape(-1))
    lax = _monitor(fail_closed=False)
    lax.guard(torch.randint(0, 255, (2, 128), generator=g, dtype=torch.int32))  # int32 converted
    assert lax.report().calls == 1
    mon.guard(torch.zeros(0, 16, dtype=torch.int64))  # empty call is a no-op


def test_listener_and_metrics_gauge() -> None:
    from prometheus_client import CollectorRegistry

    reg = CollectorRegistry()
    metrics = PGWMetrics(reg)
    ring = RING_K40_A12
    seen: list[NoiseReport] = []
    mon = _monitor(ring, metrics=metrics, model="qwen", fail_closed=False, window_values=4096)
    mon.add_alarm_listener(seen.append)
    mon.add_alarm_listener(lambda rep: (_ for _ in ()).throw(RuntimeError("listener bug")))
    g = torch.Generator().manual_seed(43)
    xq = _xq((8, 512), ring, g)
    mon.guard(xq & ring.mask, xq)
    assert len(seen) == 1 and seen[0].alarm
    assert metrics.value("pgw_gateway_noise_alarm", model="qwen") == 1.0
    mon.reset()
    assert metrics.value("pgw_gateway_noise_alarm", model="qwen") == 0.0


def test_reset_keeps_duplicate_history() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(47)
    mon = _monitor(ring, sample_rate=1e-6, always_sample_first=0)
    r = _mask((1, 512), ring, g)
    mon.guard(r)
    with pytest.raises(IntegrityError):
        mon.guard(r.clone())
    mon.reset()
    with pytest.raises(IntegrityError):
        mon.guard(r.clone())
    assert mon.report().duplicate_rows == 2


def test_thread_safety() -> None:
    ring = RING_K40_A12
    mon = _monitor(ring, sample_rate=0.5, window_values=4096)
    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        try:
            for _ in range(60):
                xq = _xq((2, 256), ring, g)
                mon.guard(mask_activations(xq, _mask((2, 256), ring, g), ring), xq)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    rep = mon.report()
    assert rep.calls == 240 and rep.rows_checked == 480
    assert not rep.alarm, rep.reasons


def test_masked_backend_style_usage_with_gpu_tensor_shapes() -> None:
    """The backend calls guard(x_hat, xq[row:row+rows]) on views; non-contiguous views work."""
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(53)
    mon = _monitor(ring)
    xq_full = _xq((10, 300), ring, g)
    x_hat = mask_activations(xq_full, _mask((10, 300), ring, g), ring)
    mon.guard(x_hat[2:6], xq_full[2:6])
    mon.guard(x_hat.t().contiguous().t()[6:], xq_full[6:])
    assert not mon.report().alarm


# ---- hardening ---------------------------------------------------------------------------------
@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
def test_unmasked_signed_send_caught_on_unsampled_call(ring: RingParams) -> None:
    """The ring range check runs on every call, not only on sampled ones."""
    g = torch.Generator().manual_seed(11)
    mon = _monitor(ring, sample_rate=1e-6, always_sample_first=0, duplicate_check=False)
    xq = _xq((2, 256), ring, g)
    xq[0, 0] = -1  # signed plaintext: negative values cannot be a ring element
    with pytest.raises(IntegrityError):
        mon.guard(xq, xq)
    rep = mon.report()
    assert rep.alarm and rep.sampled_calls == 0
    assert any("outside the ring" in r for r in rep.reasons)


def test_xq_shape_checked_on_every_call() -> None:
    ring = RING_K40_A12
    g = torch.Generator().manual_seed(12)
    mon = _monitor(ring, sample_rate=1e-6, always_sample_first=0)
    x_hat = _mask((2, 64), ring, g)
    with pytest.raises(ShapeError):
        mon.guard(x_hat, _xq((3, 64), ring, g))
    with pytest.raises(ShapeError):
        mon.guard(x_hat, [1, 2])  # type: ignore[arg-type]
    mon.guard(x_hat[0], _xq((64,), ring, g).reshape(64))  # 1-D pair is fine
    assert not mon.alarmed


def test_gauge_consistent_with_alarm_after_reset_race() -> None:
    """The alarm gauge is updated under the monitor lock, so it always ends equal to the state."""
    from prometheus_client import CollectorRegistry

    ring = RING_K40_A12
    metrics = PGWMetrics(CollectorRegistry())
    mon = _monitor(ring, metrics=metrics, model="m", fail_closed=False, sample_rate=1e-6, always_sample_first=0)
    bad = torch.full((1, 512), -5, dtype=torch.int64)
    stop = threading.Event()

    def alarm_loop() -> None:
        while not stop.is_set():
            mon.guard(bad)

    th = threading.Thread(target=alarm_loop)
    th.start()
    for _ in range(200):
        mon.reset()
    stop.set()
    th.join()
    mon.reset()
    assert not mon.alarmed
    assert metrics.value("pgw_gateway_noise_alarm", model="m") == 0.0
    mon.guard(bad)
    assert metrics.value("pgw_gateway_noise_alarm", model="m") == 1.0


def test_listener_must_be_callable() -> None:
    with pytest.raises(ConfigError):
        _monitor().add_alarm_listener(42)  # type: ignore[arg-type]


def test_reasons_and_logs_carry_no_payload_values(caplog: pytest.LogCaptureFixture) -> None:
    """SPEC 11.2: alarms log statistics only, never activation or payload values."""
    ring = RING_K48_A20
    mon = _monitor(ring, fail_closed=False, window_values=4096)
    marker = 987654321  # a distinctive plaintext value
    xq = torch.full((8, 512), marker, dtype=torch.int64)
    with caplog.at_level("DEBUG", logger="pgw.observability.noise"):
        mon.guard(xq, xq)
        mon.flush()
    assert mon.alarmed
    text = " ".join(caplog.messages) + " ".join(str(r.__dict__) for r in caplog.records)
    text += json.dumps(mon.report().to_dict())
    assert str(marker) not in text
