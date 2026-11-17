"""Gateway Prometheus metrics.

Only names from :data:`pgw.observability.metrics.METRIC_NAMES` are emitted (the contract,
SPEC 11.1): the ``pgw_gateway_*`` family plus the ``pgw_masks_*`` gauges/counters, which are
refreshed from :meth:`pgw.maskstore.MaskStore.metrics` at scrape time. Collectors live on a private
``CollectorRegistry`` so several gateways (tests) can coexist in one process. Labels never carry
prompt data. If ``prometheus_client`` is missing, every collector is a no-op.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pgw.observability.metrics import METRIC_NAMES

__all__ = ["GatewayMetrics"]

log = logging.getLogger(__name__)

_PREFIXES = ("pgw_gateway_", "pgw_masks_")
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)


class _Noop:
    def labels(self, *a: Any, **k: Any) -> _Noop:
        return self

    def inc(self, *a: Any, **k: Any) -> None:
        pass

    def dec(self, *a: Any, **k: Any) -> None:
        pass

    def set(self, *a: Any, **k: Any) -> None:
        pass

    def observe(self, *a: Any, **k: Any) -> None:
        pass


class GatewayMetrics:
    """Collectors for the gateway and mask-store metric families.

    Args:
        enabled: ``False`` makes every collector a no-op.
    """

    def __init__(self, *, enabled: bool = True):
        self._collectors: dict[str, Any] = {}
        self.registry: Any = None
        self._collect_hooks: list[Callable[[], None]] = []
        self.enabled = False
        if not enabled:
            return
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        except ImportError:  # pragma: no cover - dependency
            log.warning("prometheus_client not installed; metrics disabled")
            return
        try:  # only the contract's series: no *_created samples
            from prometheus_client import disable_created_metrics

            disable_created_metrics()  # type: ignore[no-untyped-call]
        except ImportError:  # pragma: no cover - old prometheus_client
            pass
        self.registry = CollectorRegistry(auto_describe=True)
        for name, (kind, labels, help_) in METRIC_NAMES.items():
            if not name.startswith(_PREFIXES):
                continue
            if kind == "counter":
                # prometheus_client appends _total itself.
                c: Any = Counter(name.removesuffix("_total"), help_, labels, registry=self.registry)
            elif kind == "gauge":
                c = Gauge(name, help_, labels, registry=self.registry)
            elif kind == "histogram":
                c = Histogram(name, help_, labels, registry=self.registry, buckets=_LATENCY_BUCKETS)
            else:  # pragma: no cover - contract only has these kinds
                continue
            self._collectors[name] = c
        self.enabled = True

    def get(self, name: str) -> Any:
        """Collector for ``name`` (``KeyError`` if not a gateway/masks metric in the contract)."""
        if name not in METRIC_NAMES or not name.startswith(_PREFIXES):
            raise KeyError(name)
        return self._collectors.get(name, _Noop()) if self.enabled else _Noop()

    def add_collect_hook(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` before every scrape (refresh gauges)."""
        self._collect_hooks.append(fn)

    def update_masks(self, values: dict[str, dict[tuple[str, str], float]]) -> None:
        """Apply :meth:`MaskStore.metrics` output (counters are set to the store's absolute values)."""
        if not self.enabled:
            return
        for name, series in values.items():
            if name not in self._collectors:
                continue
            c = self._collectors[name]
            for key, v in series.items():
                if not key:  # store-wide series without labels
                    c.set(v)
                    continue
                fp, ring = key
                child = c.labels(fingerprint=fp[:12], ring=ring)
                if name.endswith("_total"):
                    # Counters mirror durable store counters: advance to the store's value.
                    cur = child._value.get()
                    if v > cur:
                        child.inc(v - cur)
                else:
                    child.set(v)

    def render(self) -> tuple[bytes, str]:
        """Prometheus exposition ``(body, content_type)``."""
        for fn in self._collect_hooks:
            try:
                fn()
            except Exception:
                log.exception("metrics collect hook failed")
        if not self.enabled:
            return b"", "text/plain; version=0.0.4; charset=utf-8"
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST
