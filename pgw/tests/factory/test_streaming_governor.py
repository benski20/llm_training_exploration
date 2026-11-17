"""WeightStreamer residency/streaming and the Throttle / priority governor."""

from __future__ import annotations

import threading

import pytest
import torch

from pgw.core import ConfigError
from pgw.factory import FactoryStopped, Throttle, WeightStreamer, apply_process_priority
from pgw.factory.streaming import auto_resident_bytes

from .conftest import FakePack


def test_streamer_canonical_order_and_values(fake_pack: FakePack) -> None:
    s = WeightStreamer(fake_pack, "cpu", 0)
    got = list(s.iter_device_weights())
    assert [lid.key for lid, _ in got] == [sp.id.key for sp in fake_pack.linear_specs()]
    for lid, w in got:
        assert torch.equal(w, fake_pack.weights[lid.key])
    assert s.streaming and s.bytes_streamed == s.total_bytes


def test_streamer_partial_residency(fake_pack: FakePack) -> None:
    specs = fake_pack.linear_specs()
    first_two = sum(sp.out_features * sp.in_features for sp in specs[:2])
    s = WeightStreamer(fake_pack, "cpu", first_two)
    assert s.resident_ids == {specs[0].id.key, specs[1].id.key}
    for _ in range(3):  # three passes (= three segments)
        list(s.iter_device_weights())
    loads = [k for k in fake_pack.loads]
    assert loads.count(specs[0].id.key) == 1  # resident: loaded once
    assert loads.count(specs[2].id.key) == 3  # streamed: every pass
    assert s.bytes_streamed == 3 * (s.total_bytes - first_two)
    s.close()


def test_streamer_all_resident(fake_pack: FakePack) -> None:
    s = WeightStreamer(fake_pack, "cpu", -1)
    assert not s.streaming and s.resident_bytes_used == s.total_bytes
    a = dict((lid.key, w) for lid, w in s.iter_device_weights())
    a["0.qkv"].zero_()  # resident copies are private clones, the pack is untouched
    assert fake_pack.weights["0.qkv"].abs().sum() > 0


def test_streamer_empty_pack_rejected() -> None:
    p = FakePack(layers=0)
    with pytest.raises(ConfigError):
        WeightStreamer(p, "cpu", 0)


def test_auto_resident_bytes_cpu() -> None:
    assert auto_resident_bytes(torch.device("cpu"), 0.5) == 0
    with pytest.raises(ConfigError):
        auto_resident_bytes(torch.device("cpu"), 0.0)


# ---- throttle --------------------------------------------------------------------------------------
class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def test_throttle_rate_limit() -> None:
    clk = FakeClock()
    th = Throttle(max_slots_per_hour=3600, clock=clk, sleep=clk.sleep)  # 1 slot/s
    clk.t += 0.5
    th.account(0.5, 10)  # 10 slots in 0.5 s -> must wait until t0 + 10 s
    assert clk.sleeps == [pytest.approx(9.5)]
    clk.t += 1.0
    th.account(1.0, 1)
    assert clk.sleeps[-1] == pytest.approx(0.0, abs=1e-9) or len(clk.sleeps) == 1
    assert th.active


def test_throttle_duty_cycle() -> None:
    clk = FakeClock()
    th = Throttle(duty_cycle=0.25, clock=clk, sleep=clk.sleep)
    th.account(1.0, 1)
    assert clk.sleeps == [pytest.approx(3.0)]
    assert th.slept_s == pytest.approx(3.0)


def test_throttle_unlimited_never_sleeps() -> None:
    clk = FakeClock()
    th = Throttle(clock=clk, sleep=clk.sleep)
    for _ in range(5):
        th.account(0.1, 100)
    assert clk.sleeps == [] and not th.active


def test_throttle_stop() -> None:
    ev = threading.Event()
    th = Throttle(stop_event=ev)
    th.check()
    ev.set()
    with pytest.raises(FactoryStopped):
        th.account(0.0, 1)


def test_throttle_stop_interrupts_sleep() -> None:
    ev = threading.Event()
    th = Throttle(max_slots_per_hour=1, stop_event=ev)  # would sleep for an hour
    threading.Timer(0.05, ev.set).start()
    with pytest.raises(FactoryStopped):
        th.account(0.0, 1)


def test_throttle_validation() -> None:
    with pytest.raises(ValueError):
        Throttle(max_slots_per_hour=-1)
    with pytest.raises(ValueError):
        Throttle(duty_cycle=0)


def test_apply_process_priority_threads() -> None:
    before = torch.get_num_threads()
    try:
        apply_process_priority(nice=0, cpu_threads=1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
