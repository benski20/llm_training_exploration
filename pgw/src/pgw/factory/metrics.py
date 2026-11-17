"""Factory metrics (``pgw_factory_*`` from :data:`pgw.observability.metrics.METRIC_NAMES`).

Degrades to no-ops when ``prometheus_client`` or :class:`pgw.observability.PGWMetrics` is not
available, so the factory never fails because of monitoring.
"""

from __future__ import annotations

import logging
from typing import Any

from pgw.core.fingerprint import short_fingerprint

__all__ = ["FactoryMetrics"]

log = logging.getLogger(__name__)


class FactoryMetrics:
    """Thin wrapper emitting the factory's contract metrics.

    Args:
        metrics: A :class:`pgw.observability.PGWMetrics` (or compatible ``get(name)``), or ``None``
            to try creating one on the default registry; ``False`` disables metrics.
    """

    def __init__(self, metrics: Any = None):
        self._m: Any = None
        if metrics is False:
            return
        if metrics is not None:
            self._m = metrics
            return
        try:
            from pgw.observability.metrics import PGWMetrics

            self._m = PGWMetrics()
        except Exception as exc:  # NotImplementedError, ImportError, duplicate registration ...
            log.debug("factory metrics disabled: %r", exc)
            self._m = None

    @property
    def enabled(self) -> bool:
        """True if metrics are emitted."""
        return self._m is not None

    def _get(self, name: str) -> Any:
        if self._m is None:
            return None
        try:
            return self._m.get(name)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("metric %s unavailable: %r", name, exc)
            return None

    def segment_done(self, fingerprint: str, ring: str, device: str, slots: int, slots_per_second: float) -> None:
        """Record a committed segment."""
        labels = {"fingerprint": short_fingerprint(fingerprint), "ring": ring, "device": device}
        c = self._get("pgw_factory_slots_generated_total")
        if c is not None:
            c.labels(**labels).inc(slots)
        g = self._get("pgw_factory_slots_per_second")
        if g is not None:
            g.labels(**labels).set(slots_per_second)

    def inventory(self, fingerprint: str, ring: str, available: int, low_water: bool) -> None:
        """Mirror the inventory the scheduler observed (useful for partner factories)."""
        labels = {"fingerprint": short_fingerprint(fingerprint), "ring": ring}
        g = self._get("pgw_masks_available_slots")
        if g is not None:
            g.labels(**labels).set(available)
        lw = self._get("pgw_masks_low_water")
        if lw is not None:
            lw.labels(**labels).set(1 if low_water else 0)
