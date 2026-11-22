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

**Keep-up guard (opt-in, S5).** With all-or-nothing yielding the factory makes nothing while back-to-back chats
decode above the threshold, so a whole segment must be built after the stock fell under it (measured on Modal:
mask exhaustion after 130 s of back-to-back chats). Two settings bound that:

* ``PGW_FACTORY_YIELD_FULL_SLOTS`` (default = ``MIN_SLOTS``: all or nothing, as before): between ``MIN_SLOTS`` and
  ``FULL_SLOTS`` the writer yields only a fraction ``(stock - low) / (FULL - MIN)`` of each
  ``PGW_FACTORY_YIELD_PERIOD_S`` (default 3 s) cycle, so the factory keeps a duty cycle that grows as the stock
  falls; at or above ``FULL_SLOTS`` it yields fully, below the low mark never.
* ``PGW_FACTORY_YIELD_HORIZON_S`` (default 0 = off): the low mark is raised to at least the observed consumption
  rate (stock decreases over the last ``PGW_FACTORY_YIELD_RATE_WINDOW_S``, default 30 s) times this horizon, so
  a store draining fast stops yielding early enough for a segment to land.

Both are functions of the stock and of the decode activity only; the stock is what the provider can compute from
what it already sees (the maskgen requests it serves and the decode trips that consume the slots), so the pause
pattern still tells it nothing new.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

__all__ = ["ENV_FILE", "YieldWriter", "note_activity", "reader", "wait_if_yielding"]

log = logging.getLogger(__name__)

ENV_FILE = "PGW_FACTORY_YIELD_FILE"
ENV_MIN_SLOTS = "PGW_FACTORY_YIELD_MIN_SLOTS"
ENV_HOLD_S = "PGW_FACTORY_YIELD_HOLD_S"
ENV_MAX_S = "PGW_FACTORY_YIELD_MAX_S"
ENV_FULL_SLOTS = "PGW_FACTORY_YIELD_FULL_SLOTS"
ENV_PERIOD_S = "PGW_FACTORY_YIELD_PERIOD_S"
ENV_HORIZON_S = "PGW_FACTORY_YIELD_HORIZON_S"
ENV_RATE_WINDOW_S = "PGW_FACTORY_YIELD_RATE_WINDOW_S"

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
    """Gateway side: keeps ``path`` fresh while passes run and the stock is at least ``min_slots``.

    ``full_slots`` (> ``min_slots``) turns on the partial yield, ``horizon_s`` the consumption-rate guard (module
    docstring); both default to the environment, and to the all-or-nothing behaviour when unset."""

    def __init__(self, path: str | Path, stock: Callable[[], int], *, min_slots: int | None = None,
                 period_s: float = 0.05, stock_period_s: float = 1.0, full_slots: int | None = None,
                 duty_period_s: float | None = None, horizon_s: float | None = None,
                 rate_window_s: float | None = None):
        self.path = Path(path)
        self.stock = stock
        self.min_slots = int(min_slots if min_slots is not None else _env_float(ENV_MIN_SLOTS, 2000))
        self.full_slots = int(full_slots if full_slots is not None else _env_float(ENV_FULL_SLOTS, self.min_slots))
        self.duty_period_s = max(0.2, float(duty_period_s if duty_period_s is not None
                                            else _env_float(ENV_PERIOD_S, 3.0)))
        self.horizon_s = max(0.0, float(horizon_s if horizon_s is not None else _env_float(ENV_HORIZON_S, 0.0)))
        self.rate_window_s = max(2.0, float(rate_window_s if rate_window_s is not None
                                            else _env_float(ENV_RATE_WINDOW_S, 30.0)))
        self.period_s = float(period_s)
        self.stock_period_s = float(stock_period_s)
        self.touches = 0
        self.last_stock: int | None = None
        #: last decision's inputs (diagnostics): consumption rate (slots/s), effective low mark, yield fraction
        self.rate = 0.0
        self.low = self.min_slots
        self.duty = 1.0
        self._samples: deque[tuple[float, int]] = deque()
        self._stock_at = -1e9
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> YieldWriter:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="pgw-factory-yield", daemon=True)
            self._thread.start()
            log.info("factory yield: %s refreshed while decoding with at least %d mask slots in stock "
                     "(fully from %d, duty period %.1fs, horizon %.0fs)", self.path, self.min_slots,
                     max(self.full_slots, self.min_slots), self.duty_period_s, self.horizon_s)
        return self

    def stop(self) -> None:
        self._stop.set()

    def _sample(self, now: float) -> None:
        try:
            self.last_stock = int(self.stock())
        except Exception:  # unknown stock: do not yield
            self.last_stock = None
        self._stock_at = now
        if self.last_stock is None:
            return
        self._samples.append((now, self.last_stock))
        while len(self._samples) > 2 and now - self._samples[0][0] > self.rate_window_s:
            self._samples.popleft()
        span = self._samples[-1][0] - self._samples[0][0]
        if span >= min(5.0, self.rate_window_s / 2):
            down = 0
            prev = self._samples[0][1]
            for _, s in list(self._samples)[1:]:
                down += max(0, prev - s)  # decreases only: a landing segment is production, not consumption
                prev = s
            self.rate = down / span

    def decision(self, now: float) -> bool:
        """Whether to yield now, given the last stock sample (the caller checked the decode activity)."""
        stock = self.last_stock
        if stock is None:
            return False
        low = max(self.min_slots, int(self.rate * self.horizon_s)) if self.horizon_s > 0 else self.min_slots
        self.low = low
        if stock < low:
            self.duty = 0.0
            return False
        width = self.full_slots - self.min_slots
        if width <= 0:
            self.duty = 1.0
            return True
        self.duty = min(1.0, (stock - low) / width)
        if self.duty >= 1.0:
            return True
        return (now % self.duty_period_s) < self.duty * self.duty_period_s

    def tick(self, now: float | None = None) -> bool:
        """One step of the loop (tests): ``True`` if the file was refreshed."""
        now = time.monotonic() if now is None else now
        active = now - _activity[0] <= 2 * self.period_s
        # the rate guard samples while the gateway decoded within the last rate window (to see the drain);
        # otherwise the stock is read only while passes run, as before
        if (active or (self.horizon_s > 0 and now - _activity[0] <= self.rate_window_s)) \
                and now - self._stock_at >= self.stock_period_s:
            self._sample(now)
        if not active:
            return False
        if not self.decision(now):
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
