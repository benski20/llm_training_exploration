"""The encrypted mask factory steps aside while the gateway decodes (opt-in, same host).

On a host where the gateway and its mask factory share CPUs, memory bandwidth and the provider's GPU (the
co-located demo), the factory's encrypted maskgen traffic slows each decoded token by about 10% (measured:
36 vs 32 ms per token on Modal with the factory paused). With ``PGW_FACTORY_YIELD_FILE`` set (in the gateway's
environment, which the factory child inherits):

* the **gateway** (:class:`YieldWriter`, a daemon thread) refreshes that file's modification time every
  ``period`` while masked passes run (:func:`note_activity`, one store of a float per pass) **and** the mask
  stock is at least ``PGW_FACTORY_YIELD_MIN_SLOTS`` (default 2000, re-read every second off the decode thread);
* the **factory** (:func:`wait_if_yielding`, called before each encrypted batch is encrypted and before each
  provider request) waits while the file is fresher than ``PGW_FACTORY_YIELD_HOLD_S`` (default 0.25 s), at most
  ``PGW_FACTORY_YIELD_MAX_S`` (default 30 s) at a time.

The factory therefore never yields below the stock threshold, and above it runs at full speed between chats, so
its capacity is unchanged (it still makes several times the decode rate whenever the stock runs low). Same masks,
same requests, same bytes: only when the factory's batches start moves. The provider learns nothing new: it
already sees every decode trip, so the pauses of the maskgen requests tell it when the customer decodes, which it
knows anyway. Off by default (unset path).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

__all__ = ["ENV_FILE", "YieldWriter", "note_activity", "reader", "wait_if_yielding"]

log = logging.getLogger(__name__)

ENV_FILE = "PGW_FACTORY_YIELD_FILE"
ENV_MIN_SLOTS = "PGW_FACTORY_YIELD_MIN_SLOTS"
ENV_HOLD_S = "PGW_FACTORY_YIELD_HOLD_S"
ENV_MAX_S = "PGW_FACTORY_YIELD_MAX_S"

_activity = [0.0]  # monotonic time of the last masked pass in this process (gateway side)


def note_activity() -> None:
    """Record a masked pass (called by the gateway's backend; one float store)."""
    _activity[0] = time.monotonic()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, os.environ.get(name), default)
        return default


class YieldWriter:
    """Gateway side: keeps ``path`` fresh while passes run and the stock is at least ``min_slots``."""

    def __init__(self, path: str | Path, stock: Callable[[], int], *, min_slots: int | None = None,
                 period_s: float = 0.05, stock_period_s: float = 1.0):
        self.path = Path(path)
        self.stock = stock
        self.min_slots = int(min_slots if min_slots is not None else _env_float(ENV_MIN_SLOTS, 2000))
        self.period_s = float(period_s)
        self.stock_period_s = float(stock_period_s)
        self.touches = 0
        self.last_stock: int | None = None
        self._stock_at = -1e9
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> YieldWriter:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="pgw-factory-yield", daemon=True)
            self._thread.start()
            log.info("factory yield: %s refreshed while decoding with at least %d mask slots in stock",
                     self.path, self.min_slots)
        return self

    def stop(self) -> None:
        self._stop.set()

    def tick(self, now: float | None = None) -> bool:
        """One step of the loop (tests): ``True`` if the file was refreshed."""
        now = time.monotonic() if now is None else now
        if now - _activity[0] > 2 * self.period_s:
            return False
        if now - self._stock_at >= self.stock_period_s:
            try:
                self.last_stock = int(self.stock())
            except Exception:  # unknown stock: do not yield
                self.last_stock = None
            self._stock_at = now
        if self.last_stock is None or self.last_stock < self.min_slots:
            return False
        try:
            if self.path.exists():
                os.utime(self.path)
            else:
                self.path.touch()
        except OSError:
            return False
        self.touches += 1
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.tick()
            except Exception:  # pragma: no cover - never let the helper die loudly
                log.debug("factory yield tick failed", exc_info=True)


class _Reader:
    """Factory side: is the gateway asking to yield (``path`` modified within ``hold_s``)?"""

    def __init__(self, path: str | Path, hold_s: float, max_s: float):
        self.path = str(path)
        self.hold_s = float(hold_s)
        self.max_s = float(max_s)
        self.waited_s = 0.0
        self.waits = 0
        self._lock = threading.Lock()

    def active(self) -> bool:
        try:
            return time.time() - os.stat(self.path).st_mtime < self.hold_s
        except OSError:
            return False

    def wait(self, stop: threading.Event | None = None, poll_s: float = 0.01) -> float:
        """Wait while :meth:`active` (at most ``max_s``); returns the seconds waited."""
        if not self.active():
            return 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.max_s and self.active():
            if stop is not None:
                if stop.wait(poll_s):
                    break
            else:
                time.sleep(poll_s)
        dt = time.monotonic() - t0
        with self._lock:
            self.waited_s += dt
            self.waits += 1
        return dt


_reader: list[_Reader | None] = []
_reader_lock = threading.Lock()


def reader() -> _Reader | None:
    """The factory process's reader from its environment (``None``: ``PGW_FACTORY_YIELD_FILE`` unset)."""
    if not _reader:
        with _reader_lock:
            if not _reader:
                p = os.environ.get(ENV_FILE)
                _reader.append(_Reader(p, _env_float(ENV_HOLD_S, 0.25), _env_float(ENV_MAX_S, 30.0)) if p else None)
    return _reader[0]


def wait_if_yielding(stop: threading.Event | None = None) -> float:
    """Factory side: wait while the gateway decodes with ample stock (no-op when not configured)."""
    r = reader()
    return r.wait(stop) if r is not None else 0.0
