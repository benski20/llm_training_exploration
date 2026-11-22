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
