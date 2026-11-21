"""Routing-independent refill of per-expert mask pools (opt-in; for MoE serving).

With routing-oblivious dispatch (:mod:`pgw.moe.oblivious`) a token consumes one mask slot of each of its
``k`` routed experts. Per-expert consumption therefore follows the routing. With the *encrypted* mask
factory the provider performs every refill (it multiplies the ciphertexts by ``W_e``, so it learns which
expert a refill is for). A refill policy that tops up whichever pool runs low would hand the provider the
routing histogram over time. That would be a data-dependent leak.

:class:`LockstepRefill` decides refills from **public** quantities only: the number of rows the gateway has
submitted (visible to the provider anyway, as trip shapes) and fixed configuration. Every expert of a
layer is refilled by the same number of slots at the same moments, so the refill stream is a function of
public information and reveals nothing about routing.

The price is waste and possible stalls: popular experts drain faster than ``k / E`` of the rows, cold ones
accumulate unused masks. ``overprovision`` (rho) buys slack; :func:`simulate_refill` measures stalls and
waste for a routing trace. While a routed expert's pool is empty the gateway must not pause visibly
(a pause would correlate with routing). It sends a constant-shape trip whose rows are all uniform dummies
(``plan_dispatch(np.empty((0, k), int), E, rows=B)`` is such an all-dummy plan) and retries the real rows later.
The plain (trusted, local) factory has no such constraint: the provider never sees its refills.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from pgw.core.errors import ConfigError

__all__ = ["LockstepRefill", "RefillSimulation", "simulate_refill"]


@dataclass(frozen=True)
class LockstepRefill:
    """Refill every expert of a layer by the same amount, from the public row counter only.

    Attributes:
        num_experts: ``E``.
        top_k: ``k``.
        overprovision: ``rho >= 1``: slots produced per expert per submitted row are ``rho * k / E``.
        reserve_slots: Slots per expert produced up front (covers start-up and routing bursts).
        group_slots: Refill granularity (the encrypted factory works in 256-slot groups).
    """

    num_experts: int
    top_k: int
    overprovision: float = 1.5
    reserve_slots: int = 1024
    group_slots: int = 256

    def __post_init__(self) -> None:
        if self.num_experts < 1 or not 1 <= self.top_k <= self.num_experts:
            raise ConfigError(f"need 1 <= k <= E, got k={self.top_k} E={self.num_experts}")
        if not (math.isfinite(self.overprovision) and self.overprovision >= 1.0):
            raise ConfigError("overprovision must be a finite number >= 1")
        if self.reserve_slots < 0 or self.group_slots < 1:
            raise ConfigError("reserve_slots must be >= 0 and group_slots >= 1")

    def target_per_expert(self, rows_submitted: int) -> int:
        """Slots every expert should have been given after ``rows_submitted`` public rows (rounded up to
        whole groups). Depends on public inputs only."""
        if rows_submitted < 0:
            raise ConfigError("rows_submitted must be >= 0")
        want = self.reserve_slots + self.overprovision * self.top_k * rows_submitted / self.num_experts
        g = self.group_slots
        return math.ceil(want / g) * g

    def due(self, rows_submitted: int, produced_per_expert: int) -> int:
        """Slots to request now **for every expert** (the same number for all of them)."""
        return max(0, self.target_per_expert(rows_submitted) - produced_per_expert)


@dataclass
class RefillSimulation:
    """Outcome of :func:`simulate_refill`."""

    trips: int
    stall_trips: int
    rows_served: int
    slots_produced: int
    slots_consumed: int

    @property
    def waste(self) -> float:
        """Fraction of produced slots not consumed by the end of the trace (still in the pools)."""
        return 0.0 if self.slots_produced == 0 else 1.0 - self.slots_consumed / self.slots_produced

    @property
    def stall_fraction(self) -> float:
        """Fraction of trips that had to be all-dummy because a routed expert's pool was empty."""
        return 0.0 if self.trips == 0 else self.stall_trips / self.trips


def simulate_refill(policy: LockstepRefill, trace: Iterable[np.ndarray], *, lag_trips: int = 0) -> RefillSimulation:
    """Replay a routing trace (one ``(B, k)`` top-k array per trip) against ``policy``.

    A trip whose routed experts all have slots is served; otherwise the gateway sends an all-dummy trip of
    the same shape and retries the same rows next trip (counted in ``stall_trips``). Refills requested
    after trip ``t`` arrive ``lag_trips`` trips later. Every trip, served or dummy, counts as public rows.
    """
    e_count = policy.num_experts
    produced = policy.target_per_expert(0)
    pending: list[tuple[int, int]] = []  # (arrival trip, slots per expert)
    avail = np.full(e_count, produced, dtype=np.int64)
    requested = produced
    rows_public = trips = stalls = served = 0
    for topk in trace:
        t = np.asarray(topk)
        while True:
            for _arrival, n in [p for p in pending if p[0] <= trips]:
                avail += n
                produced += n
            pending = [p for p in pending if p[0] > trips]
            need = np.bincount(t.ravel(), minlength=e_count)
            trips += 1
            rows_public += int(t.shape[0])
            ok = bool((need <= avail).all())
            if ok:
                avail -= need
                served += int(t.shape[0])
            else:
                stalls += 1
            due = policy.due(rows_public, requested)
            if due:
                requested += due
                pending.append((trips + lag_trips, due))
            if ok:
                break
            if stalls > 1_000_000:  # pragma: no cover - misconfigured policy
                raise ConfigError("refill policy cannot keep up with the trace")
    consumed = int(produced * e_count - int(avail.sum()))
    return RefillSimulation(trips, stalls, served, produced * e_count, consumed)
