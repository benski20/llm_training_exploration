"""Fail-closed guards of the gateway: the integrity circuit breaker and request admission.

* :class:`CircuitBreaker` opens on any integrity failure (Freivalds or range check) of a compute
  service endpoint. While open, all masked traffic to that endpoint is refused. It closes after the
  configured cool-down or by an admin reset (SPEC 9.5).
* :class:`AdmissionController` bounds concurrently generating sessions per model and the number of
  requests waiting for a slot (request queue). Requests beyond the queue, or waiting longer than
  the queue timeout, fail fast with ``unavailable`` and a ``Retry-After`` hint.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from pgw.core.errors import ErrorCode, PGWError, ServiceUnavailableError

__all__ = ["AdmissionController", "AdmissionLimitError", "BreakerOpenError", "CircuitBreaker"]

log = logging.getLogger(__name__)


class BreakerOpenError(ServiceUnavailableError):
    """Masked traffic to a compute service is blocked after an integrity failure."""


@dataclass
class _Trip:
    at: float
    reason: str
    code: str


class CircuitBreaker:
    """Integrity circuit breaker for one compute-service endpoint.

    Args:
        name: Endpoint label (model name) for logs.
        cooldown_s: Seconds until the breaker half-closes by itself; ``None`` = admin reset only.
        max_auto_resets: With a cool-down, how many times the breaker may re-open by itself; after
            that it latches (admin reset only). The counter restarts with an admin reset.
        clock: Injectable monotonic clock (tests).
        on_change: Optional ``fn(open: bool)`` hook (metrics, alerts).
    """

    def __init__(
        self,
        name: str,
        cooldown_s: float | None,
        *,
        max_auto_resets: int | None = 2,
        clock: Callable[[], float] | None = None,
        on_change: Callable[[bool], None] | None = None,
    ):
        self.name = name
        self.cooldown_s = cooldown_s
        self.max_auto_resets = max_auto_resets
        self.auto_resets = 0
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._trip: _Trip | None = None
        self.trips = 0
        self._on_change = on_change

    def trip(self, exc: BaseException | None = None, reason: str = "integrity_check_failed") -> None:
        """Open the breaker (called on every integrity failure)."""
        code = str(getattr(exc, "code", reason))
        with self._lock:
            self._trip = _Trip(self._clock(), reason, code)
            self.trips += 1
        log.critical(
            "integrity circuit breaker OPEN: masked traffic blocked (operator alert)",
            extra={"model": self.name, "code": code, "trips": self.trips},
        )
        if self._on_change is not None:
            self._on_change(True)

    @property
    def is_open(self) -> bool:
        """Whether traffic is currently blocked (auto-closes after the cool-down)."""
        with self._lock:
            t = self._trip
            if t is None:
                return False
            latched = self.max_auto_resets is not None and self.auto_resets >= self.max_auto_resets
            if self.cooldown_s is not None and not latched and self._clock() - t.at >= self.cooldown_s:
                self._trip = None
                self.auto_resets += 1
                closed = True
            else:
                return True
        if closed:
            log.warning("integrity circuit breaker cool-down elapsed; traffic resumes", extra={"model": self.name})
            if self._on_change is not None:
                self._on_change(False)
        return False

    def check(self) -> None:
        """Raise :class:`BreakerOpenError` while open."""
        if self.is_open:
            with self._lock:
                t = self._trip
            retry = None
            latched = self.max_auto_resets is not None and self.auto_resets >= self.max_auto_resets
            if t is not None and self.cooldown_s is not None and not latched:
                retry = max(1.0, self.cooldown_s - (self._clock() - t.at))
            details: dict[str, Any] = {"breaker": "open"}
            if retry is not None:
                details["retry_after_s"] = round(retry, 1)
            raise BreakerOpenError(
                "masked compute is suspended after an integrity check failure (operator action required)",
                details=details,
            )

    def reset(self) -> bool:
        """Admin reset; returns whether it was open."""
        with self._lock:
            was = self._trip is not None
            self._trip = None
            self.auto_resets = 0
        if was:
            log.warning("integrity circuit breaker reset by admin", extra={"model": self.name})
            if self._on_change is not None:
                self._on_change(False)
        return was

    def status(self) -> dict[str, Any]:
        """Admin view."""
        open_ = self.is_open
        with self._lock:
            t = self._trip
        return {
            "open": open_,
            "trips": self.trips,
            "cooldown_s": self.cooldown_s,
            "open_for_s": round(self._clock() - t.at, 3) if t is not None else None,
            "last_code": t.code if t is not None else None,
        }


class AdmissionLimitError(PGWError):
    """A request needs more concurrent sessions than the model allows at all (HTTP 400)."""

    code = ErrorCode.PROTOCOL


class AdmissionController:
    """Bounded concurrency with a bounded FIFO wait queue.

    Args:
        max_active: Sessions allowed to generate at once.
        max_queue: Requests allowed to wait for a slot.
        queue_timeout_s: Max time a request waits.
    """

    def __init__(self, max_active: int, max_queue: int, queue_timeout_s: float):
        if max_active < 1 or max_queue < 0 or queue_timeout_s <= 0:
            raise ValueError("invalid admission limits")
        self.max_active = max_active
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self.active = 0
        self.waiting = 0
        self.rejected = 0
        self.rejected_by_reason: dict[str, int] = {}
        #: Optional metric hooks: ``on_change(active, waiting)``, ``on_reject(reason)``, ``on_wait(s)``.
        self.on_change: Callable[[int, int], None] | None = None
        self.on_reject: Callable[[str], None] | None = None
        self.on_wait: Callable[[float], None] | None = None
        self._cond: asyncio.Condition | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._draining = False

    def _condition(self) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if self._cond is None or self._loop is not loop:
            self._cond = asyncio.Condition()
            self._loop = loop
        return self._cond

    async def acquire(self, n: int = 1) -> float:
        """Wait for ``n`` slots, taken atomically; returns seconds waited.

        All ``n`` slots are granted together or none is held, so multi-prompt requests can never
        deadlock each other by each holding part of the slots they need (hold-and-wait).

        Raises:
            ServiceUnavailableError: draining, queue full or queue timeout.
            AdmissionLimitError: ``n`` exceeds ``max_active`` (can never be admitted).
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self.max_active:
            self._reject("too_large")
            raise AdmissionLimitError(
                f"request needs {n} concurrent sessions; this model allows {self.max_active}",
                details={"sessions": n, "max_active": self.max_active},
            )
        if self._draining:
            self._reject("draining")
            raise ServiceUnavailableError("gateway is draining", details={"retry_after_s": 5.0})
        cond = self._condition()
        t0 = time.monotonic()
        async with cond:
            if self.active + n <= self.max_active and self.waiting == 0:
                self.active += n
                self._observe_wait(0.0)
                return 0.0
            if self.waiting >= self.max_queue:
                self._reject("queue_full")
                raise ServiceUnavailableError(
                    "request queue is full", details={"retry_after_s": 1.0, "queue": self.waiting}
                )
            self.waiting += 1
            self._changed()
            try:
                await asyncio.wait_for(
                    cond.wait_for(lambda: self.active + n <= self.max_active or self._draining), self.queue_timeout_s
                )
            except TimeoutError as exc:
                self._reject("timeout")
                self._pass_wakeup(cond)
                raise ServiceUnavailableError(
                    "timed out waiting in the request queue", details={"retry_after_s": 1.0}
                ) from exc
            except BaseException:
                # Cancelled (client went away): a notify() meant for this waiter must not be lost.
                self._pass_wakeup(cond)
                raise
            finally:
                self.waiting -= 1
                self._changed()
            if self._draining:
                cond.notify_all()
                self._reject("draining")
                raise ServiceUnavailableError("gateway is draining", details={"retry_after_s": 5.0})
            self.active += n
            waited = time.monotonic() - t0
            self._observe_wait(waited)
            return waited

    def _reject(self, reason: str) -> None:
        self.rejected += 1
        self.rejected_by_reason[reason] = self.rejected_by_reason.get(reason, 0) + 1
        if self.on_reject is not None:
            with contextlib.suppress(Exception):
                self.on_reject(reason)

    def _observe_wait(self, waited: float) -> None:
        self._changed()
        if self.on_wait is not None:
            with contextlib.suppress(Exception):
                self.on_wait(waited)

    def _changed(self) -> None:
        if self.on_change is not None:
            with contextlib.suppress(Exception):
                self.on_change(self.active, self.waiting)

    def _pass_wakeup(self, cond: asyncio.Condition) -> None:
        """Hand a possibly consumed wake-up to the other waiters (caller holds ``cond``'s lock)."""
        if self.active < self.max_active:
            with contextlib.suppress(RuntimeError):  # lock not held (cancelled while re-acquiring)
                cond.notify_all()

    async def release(self, n: int = 1) -> None:
        """Give ``n`` slots back."""
        cond = self._condition()
        async with cond:
            self.active = max(0, self.active - n)
            # Waiters need different numbers of slots: let each re-check its own predicate.
            cond.notify_all()
        self._changed()

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[float]:
        """``async with admission.slot() as waited_s: ...``."""
        waited = await self.acquire()
        try:
            yield waited
        finally:
            await self.release()

    async def drain(self, timeout_s: float) -> bool:
        """Refuse new work, wake waiters, wait up to ``timeout_s`` for active sessions; True if idle."""
        self._draining = True
        cond = self._condition()
        async with cond:
            cond.notify_all()
        deadline = time.monotonic() + timeout_s
        while self.active > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        return self.active == 0

    def stop_accepting(self) -> None:
        """Refuse new requests (synchronous; queued waiters time out or are woken by :meth:`drain`)."""
        self._draining = True

    def reopen(self) -> None:
        """Accept work again after :meth:`drain` (model reload)."""
        self._draining = False

    def status(self) -> dict[str, Any]:
        """Admin view."""
        return {
            "active": self.active, "waiting": self.waiting, "max_active": self.max_active,
            "max_queue": self.max_queue, "rejected": self.rejected, "draining": self._draining,
            "rejected_by_reason": dict(self.rejected_by_reason),
        }
