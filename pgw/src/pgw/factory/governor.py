"""Resource governance for the factory: rate limits, duty cycle, low priority, cooperative stop.

The factory is throughput-oriented but often shares a GPU or CPU with other work. A
:class:`Throttle` is consulted after every GEMM chunk:

* ``max_slots_per_hour`` caps the production rate (no credit accumulates while idle, so there is
  no burst beyond the current chunk),
* ``duty_cycle`` idles ``work * (1 - d) / d`` seconds after ``work`` seconds of compute,
* a ``stop_event`` aborts generation promptly (the partial segment is discarded by the writer).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

import torch

from pgw.core.errors import PGWError

__all__ = ["FactoryStopped", "Throttle", "apply_process_priority"]

log = logging.getLogger(__name__)


class FactoryStopped(PGWError):  # noqa: N818 - a control-flow signal, not an error condition
    """Generation was interrupted by a stop request (the partial segment is discarded)."""


class Throttle:
    """Rate limit / duty cycle / stop control, shared across segments of one factory process.

    Args:
        max_slots_per_hour: Average slot rate cap; ``0`` disables.
        duty_cycle: Fraction of wall time spent computing, in ``(0, 1]``.
        stop_event: When set, :meth:`account` and :meth:`check` raise :class:`FactoryStopped`.
        clock: Monotonic clock (injectable for tests).
        sleep: ``sleep(seconds)``; defaults to waiting on ``stop_event`` so stops are prompt.
    """

    def __init__(
        self,
        max_slots_per_hour: float = 0.0,
        duty_cycle: float = 1.0,
        stop_event: threading.Event | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
    ):
        if max_slots_per_hour < 0:
            raise ValueError("max_slots_per_hour must be >= 0")
        if not 0 < duty_cycle <= 1:
            raise ValueError("duty_cycle must be in (0, 1]")
        self.max_slots_per_hour = float(max_slots_per_hour)
        self.duty_cycle = float(duty_cycle)
        self.stop_event = stop_event or threading.Event()
        self._clock = clock
        self._sleep = sleep or self._wait
        self._next_free: float | None = None
        self.slept_s = 0.0

    def _wait(self, seconds: float) -> None:
        self.stop_event.wait(seconds)

    @property
    def active(self) -> bool:
        """True if any limit is configured."""
        return self.max_slots_per_hour > 0 or self.duty_cycle < 1

    def check(self) -> None:
        """Raise :class:`FactoryStopped` if a stop was requested."""
        if self.stop_event.is_set():
            raise FactoryStopped("factory stop requested")

    def account(self, work_seconds: float, slots: float) -> None:
        """Record ``work_seconds`` of compute that produced ``slots`` slot-equivalents, then idle as needed.

        The rate limit is a budget with no carried-over credit: each chunk "costs"
        ``slots * 3600 / max_slots_per_hour`` seconds starting when it began (or when the previous
        chunk's budget ended, if later). Idle time (e.g. while inventories are full) therefore
        never allows a burst above the configured rate afterwards.
        """
        self.check()
        now = self._clock()
        delay = 0.0
        if self.duty_cycle < 1 and work_seconds > 0:
            delay = work_seconds * (1 - self.duty_cycle) / self.duty_cycle
        if self.max_slots_per_hour > 0:
            began = now - max(0.0, work_seconds)
            start = began if self._next_free is None else max(self._next_free, began)
            self._next_free = start + max(0.0, slots) * 3600.0 / self.max_slots_per_hour
            delay = max(delay, self._next_free - now)
        if delay > 0:
            self.slept_s += delay
            self._sleep(delay)
        self.check()


def apply_process_priority(nice: int = 0, cpu_threads: int = 0) -> None:
    """Lower the process CPU priority and/or cap torch threads (call once at start-up).

    ``nice`` is an *increment* (POSIX ``nice(2)``); it cannot be undone without privileges.
    I/O priority follows CPU niceness on Linux CFQ/BFQ for best-effort processes.
    """
    if nice > 0 and hasattr(os, "nice"):
        try:
            new = os.nice(nice)
            log.info("factory process niceness now %d", new)
        except OSError as exc:  # pragma: no cover - platform dependent
            log.warning("could not change niceness: %s", exc)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        log.info("torch intra-op threads set to %d", cpu_threads)
