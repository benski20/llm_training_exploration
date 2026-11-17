"""Keeps mask store inventories between low-water and target levels."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FactoryScheduler", "RefillPlan"]


@dataclass(frozen=True)
class RefillPlan:
    """What the scheduler intends to produce."""

    fingerprint: str
    ring: str
    segments: int
    slots_per_segment: int
    reason: str


class FactoryScheduler:
    """Polls inventory and runs the generator until each ``(fingerprint, ring)`` reaches target.

    Resumable and crash-safe (partial segments are aborted by the store writer). Exposes progress
    metrics (``pgw_factory_*``). Runs as ``pgw factory run``.
    """

    def __init__(self, settings: object, store: object, generators: dict[tuple[str, str], object]):
        raise NotImplementedError("factory team: FactoryScheduler")

    def plan(self) -> list[RefillPlan]:
        """Compute refill work from the current inventory."""
        raise NotImplementedError

    def run_once(self) -> list[str]:
        """Execute one plan; return produced segment ids."""
        raise NotImplementedError

    def run_forever(self, stop_event: object | None = None) -> None:
        """Loop ``plan``/``run_once`` every ``poll_interval_s`` until stopped."""
        raise NotImplementedError
