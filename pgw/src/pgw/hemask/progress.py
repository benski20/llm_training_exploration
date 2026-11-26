"""Liveness of the provider's encrypted mask generation (``POST /pgw/v1/maskgen`` and the bulk channel).

A provider whose device wedges (a hung kernel, a GPU that stopped answering) still answers ``/readyz``: the HTTP
server, the batcher thread and the device probe's cached result look healthy while every maskgen item waits
forever. Before this module a client noticed only when its maskgen request ran out of its budget (600 s).

:class:`MaskgenProgress` counts every maskgen item from admission to the end of its worker thread: admitted
(waiting for a device slot), running (its thread started), finished. From that it derives

* a **coarse, public bit** (:meth:`public_state`: ``"ok"`` or ``"stalled"``) for ``/readyz``: stalled when an item
  has been running longer than the bound, or items wait and none started or finished within the bound. Nothing
  else is public: no counts, no ages, no tenant (``/readyz`` is reachable by every tenant; security review S5);
* the **caller's own numbers** (:meth:`tenant_view`, ``GET /pgw/v1/maskgen/status``, authenticated): its items
  waiting and running, the age of its oldest running item, its finished and failed items and its slowest recent
  item. Never another tenant's;
* the **operator's numbers** (:meth:`operator_view`): the same over all tenants, on the non-public metrics port's
  ``/readyz`` only.

The bound adapts to progress, not to wall time alone: ``max(stall_s, 4 x the slowest recent item)``, and the first
item of a (device, shape) after start-up (kernel compilation, the device self-tests) gets ``warmup_s`` at least.
A slow but healthy provider (a 72B linear on a slow device) finishes items, which raises the bound; a wedged one
finishes none.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DEFAULT_STALL_S", "DEFAULT_WARMUP_S", "MaskgenProgress", "stall_bound"]

#: ``PGW_SERVICE_MASKGEN_STALL_S``: an item running (or a queue not moving) this long counts as a stall
DEFAULT_STALL_S = 30.0
#: the first item of a (device, shape) since start-up may take this long (kernel compilation, device self-tests)
DEFAULT_WARMUP_S = 300.0
#: the bound is at least this multiple of the slowest recent item
SLOW_ITEM_FACTOR = 4.0


def stall_bound(stall_s: float, slowest_item_s: float | None, cold: bool, warmup_s: float = DEFAULT_WARMUP_S) -> float:
    """How long an item may run (or a queue stand still) before it counts as stalled: ``max(stall_s, 4 x the
    slowest recent item)``, and at least ``warmup_s`` for the first item of a shape (``cold``)."""
    b = max(float(stall_s), SLOW_ITEM_FACTOR * float(slowest_item_s or 0.0))
    return max(b, float(warmup_s)) if cold else b


@dataclass
class _Item:
    tenant: str
    shape: Hashable
    admitted: float
    started: float | None = None
    cold: bool = False


@dataclass
class _Tenant:
    done: int = 0
    failed: int = 0
    last_done: float | None = None
    durations: deque[float] = field(default_factory=lambda: deque(maxlen=64))


class MaskgenProgress:
    """Item-level progress of maskgen (see the module docstring). Thread-safe; every call is O(items in flight)
    at most.

    Args:
        stall_s: Base bound (``PGW_SERVICE_MASKGEN_STALL_S``, default 30 s).
        warmup_s: Bound of the first item of a (device, shape) since start-up (``PGW_SERVICE_MASKGEN_WARMUP_S``).
        clock: Monotonic clock (tests).
    """

    def __init__(self, stall_s: float | None = None, warmup_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.stall_s = float(stall_s if stall_s is not None else _env_float("PGW_SERVICE_MASKGEN_STALL_S",
                                                                            DEFAULT_STALL_S))
        self.warmup_s = float(warmup_s if warmup_s is not None else _env_float("PGW_SERVICE_MASKGEN_WARMUP_S",
                                                                               DEFAULT_WARMUP_S))
        self._clock = clock
        self._mu = threading.Lock()
        self._items: dict[int, _Item] = {}
        self._seq = 0
        self._tenants: dict[str, _Tenant] = {}
        self._durations: deque[float] = deque(maxlen=256)
        self._warm: set[Hashable] = set()
        self._moved = clock()  # last admission into an empty queue, start or finish
        self.done = 0
        self.failed = 0

    # ---- events -----------------------------------------------------------------------------------------------
    def admit(self, tenant: str, shape: Hashable = None) -> int:
        """An item is admitted (waits for a device slot); returns its token."""
        with self._mu:
            now = self._clock()
            if not self._items:
                self._moved = now
            self._seq += 1
            self._items[self._seq] = _Item(str(tenant), shape, now)
            return self._seq

    def start(self, token: int) -> None:
        """The item's worker thread starts."""
        with self._mu:
            it = self._items.get(token)
            if it is None or it.started is not None:
                return
            it.started = self._moved = self._clock()
            it.cold = it.shape not in self._warm

    def finish(self, token: int, ok: bool = True) -> None:
        """The item's worker thread ended (``ok``: it returned a result), or the item was abandoned before it
        started (a cancelled request: not counted as done or failed)."""
        with self._mu:
            it = self._items.pop(token, None)
            if it is None:
                return
            now = self._moved = self._clock()
            if it.started is None:
                return
            t = self._tenants.setdefault(it.tenant, _Tenant())
            dur = now - it.started
            if ok:
                t.done += 1
                t.last_done = now
                t.durations.append(dur)
                self._durations.append(dur)
                self._warm.add(it.shape)
                self.done += 1
            else:
                t.failed += 1
                self.failed += 1

    # ---- views ------------------------------------------------------------------------------------------------
    def _verdict(self, items: list[_Item], durations: deque[float] | list[float], now: float,
                 queue_moved: float | None) -> tuple[bool, dict[str, Any]]:
        running = [it for it in items if it.started is not None]
        waiting = [it for it in items if it.started is None]
        slowest = max(durations, default=None)
        oldest = min(running, key=lambda i: i.started or now, default=None)
        age = now - float(oldest.started or now) if oldest is not None else 0.0
        cold = bool(oldest is not None and oldest.cold)
        bound = stall_bound(self.stall_s, slowest, cold, self.warmup_s)
        stalled = oldest is not None and age >= bound
        if not stalled and waiting and queue_moved is not None:
            stalled = now - queue_moved >= stall_bound(self.stall_s, slowest, False, self.warmup_s) and (
                oldest is None or age >= self.stall_s)
        return stalled, {
            "waiting": len(waiting), "running": len(running),
            "oldest_running_s": round(age, 1) if oldest is not None else None,
            "oldest_waiting_s": round(now - min(i.admitted for i in waiting), 1) if waiting else None,
            "slowest_item_s": None if slowest is None else round(slowest, 3),
            "cold": cold, "bound_s": round(bound, 1),
        }

    def stalled(self) -> bool:
        """Whether maskgen as a whole is stalled (the coarse bit)."""
        with self._mu:
            return self._verdict(list(self._items.values()), self._durations, self._clock(), self._moved)[0]

    def public_state(self) -> str:
        """``"stalled"`` or ``"ok"``: the only maskgen value on the public ``/readyz``."""
        return "stalled" if self.stalled() else "ok"

    def tenant_view(self, tenant: str) -> dict[str, Any]:
        """The caller's own maskgen progress (authenticated status endpoint): nothing about other tenants but the
        coarse public bit."""
        with self._mu:
            now = self._clock()
            mine = [it for it in self._items.values() if it.tenant == tenant]
            t = self._tenants.get(tenant) or _Tenant()
            stalled_all = self._verdict(list(self._items.values()), self._durations, now, self._moved)[0]
            mine_stalled, v = self._verdict(mine, t.durations, now, None)
            v.update({"done": t.done, "failed": t.failed,
                      "last_done_s": None if t.last_done is None else round(now - t.last_done, 1),
                      "stalled": mine_stalled})
            return {"maskgen": "stalled" if stalled_all else "ok", "stall_s": self.stall_s,
                    "warmup_s": self.warmup_s, "tenant": v}

    def operator_view(self) -> dict[str, Any]:
        """All tenants together (no tenant names): for the operator's metrics port only."""
        with self._mu:
            now = self._clock()
            stalled, v = self._verdict(list(self._items.values()), self._durations, now, self._moved)
            v.update({"done": self.done, "failed": self.failed, "stalled": stalled,
                      "last_progress_s": round(now - self._moved, 1)})
            return v


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return max(1.0, float(raw)) if raw else default
    except ValueError:
        return default
