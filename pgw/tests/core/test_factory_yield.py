"""Factory yield (``pgw.core.factory_yield``): the gateway refreshes the file only while passes run with ample
stock; the factory waits only while the file is fresh, and at most ``max_s``."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from pgw.core import factory_yield as fy


def test_writer_needs_activity_and_stock(tmp_path: Path) -> None:
    p = tmp_path / "yield"
    stock = [5000]
    w = fy.YieldWriter(p, lambda: stock[0], min_slots=2000, period_s=0.05, stock_period_s=0.0)
    fy._activity[0] = 0.0
    assert not w.tick(now=1e6) and not p.exists()  # no pass recently
    fy.note_activity()
    assert w.tick() and p.exists()
    stock[0] = 1999  # below the threshold: no refresh, the factory runs
    old = p.stat().st_mtime - 10
    os.utime(p, (old, old))
    fy.note_activity()
    assert not w.tick() and p.stat().st_mtime == pytest.approx(old)

    def boom() -> int:
        raise RuntimeError("store down")

    w2 = fy.YieldWriter(p, boom, min_slots=0, stock_period_s=0.0)
    fy.note_activity()
    assert not w2.tick()  # unknown stock: never yield


def test_reader_waits_while_fresh_and_is_bounded(tmp_path: Path) -> None:
    p = tmp_path / "yield"
    r = fy._Reader(p, hold_s=0.2, max_s=0.3)
    assert not r.active() and r.wait() == 0.0  # no file: no wait
    p.touch()
    assert r.active()
    t0 = time.monotonic()
    waited = r.wait(poll_s=0.01)
    assert 0.15 <= waited <= 0.5 and time.monotonic() - t0 < 0.6  # until stale (hold) or max_s
    stop = threading.Event()
    p.touch()
    stop.set()
    assert r.wait(stop) < 0.1  # a stop request ends the wait


def test_reader_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fy, "_reader", [])
    monkeypatch.delenv(fy.ENV_FILE, raising=False)
    assert fy.reader() is None and fy.wait_if_yielding() == 0.0
    monkeypatch.setattr(fy, "_reader", [])
    monkeypatch.setenv(fy.ENV_FILE, str(tmp_path / "y"))
    monkeypatch.setenv(fy.ENV_HOLD_S, "0.5")
    r = fy.reader()
    assert r is not None and r.hold_s == 0.5 and r.max_s == 30.0


def test_partial_yield_duty_follows_stock(tmp_path: Path) -> None:
    p = tmp_path / "yield"
    stock = [6000]
    w = fy.YieldWriter(p, lambda: stock[0], min_slots=2000, full_slots=6000, duty_period_s=1.0,
                       stock_period_s=0.0, horizon_s=0.0)
    fy._activity[0] = 1e6
    assert w.tick(now=1e6 + 0.01) and w.duty == 1.0  # at FULL: always
    stock[0] = 3000  # a quarter of the band: yield in the first quarter of each 1 s cycle only
    fy._activity[0] = 2e6
    got = []
    for i in range(100):
        t = 2e6 + i / 100
        fy._activity[0] = t
        got.append(w.tick(now=t))
    assert w.duty == pytest.approx(0.25) and 20 <= sum(got) <= 30
    stock[0] = 1999
    fy._activity[0] = 3e6
    assert not w.tick(now=3e6) and w.duty == 0.0  # below the low mark: never


def test_default_is_all_or_nothing(tmp_path: Path) -> None:
    w = fy.YieldWriter(tmp_path / "y", lambda: 2000, min_slots=2000, stock_period_s=0.0)
    assert w.full_slots == 2000
    got = []
    for i in range(10):
        fy._activity[0] = 5e6 + i / 10
        got.append(w.tick(now=5e6 + i / 10))
    assert all(got)


def test_horizon_raises_low_mark_when_draining(tmp_path: Path) -> None:
    p = tmp_path / "yield"
    stock = [5000]
    w = fy.YieldWriter(p, lambda: stock[0], min_slots=1000, horizon_s=60.0, rate_window_s=30.0,
                       stock_period_s=1.0)
    t = 7e6
    for _ in range(20):  # 50 slots/s consumed while decoding, the factory yielding
        fy._activity[0] = t
        w.tick(now=t)
        stock[0] -= 50
        t += 1.0
    # 50 slots/s x 60 s = 3000 > stock 4050 - ... : low mark 3000; stock 4000 still above
    assert w.rate == pytest.approx(50, rel=0.1) and w.low == pytest.approx(3000, rel=0.1)
    for _ in range(25):
        fy._activity[0] = t
        w.tick(now=t)
        stock[0] -= 50
        t += 1.0
    fy._activity[0] = t
    assert stock[0] < w.low and not w.tick(now=t)  # drained below rate x horizon: the factory runs
