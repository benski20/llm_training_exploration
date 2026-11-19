"""Service metrics: thin typed wrapper over the ``pgw_service_*`` contract metrics.

Only names listed in :data:`pgw.observability.METRIC_NAMES` are emitted (SPEC 11.1). Collectors
come from :class:`pgw.observability.PGWMetrics` when it is available; otherwise they are created
directly with ``prometheus_client`` from the same table on a private registry. Labels carry tenant
ids, linear kinds, backends and error codes only, never payload data.
"""

from __future__ import annotations

import logging
from typing import Any

from pgw.observability.metrics import METRIC_NAMES

__all__ = ["ServiceMetrics"]

log = logging.getLogger(__name__)

_LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


class _Noop:
    """Stand-in collector when metrics are disabled."""

    def labels(self, *_a: Any, **_kw: Any) -> _Noop:
        return self

    def inc(self, *_a: Any, **_kw: Any) -> None:
        return None

    def dec(self, *_a: Any, **_kw: Any) -> None:
        return None

    def set(self, *_a: Any, **_kw: Any) -> None:
        return None

    def observe(self, *_a: Any, **_kw: Any) -> None:
        return None


class ServiceMetrics:
    """Collectors used by the compute service.

    Args:
        enabled: When false every collector is a no-op and :meth:`render` returns an empty page.
        registry: Optional ``prometheus_client.CollectorRegistry`` (default: a private one, so
            several services in one process, e.g. tests, do not clash).
    """

    def __init__(self, *, enabled: bool = True, registry: Any | None = None):
        self.enabled = enabled
        self.registry: Any = None
        self._c: dict[str, Any] = {}
        # labelled children by (metric, label values): ``Collector.labels(**kw)`` validates and
        # sorts on every call, which is measurable on the per-request hot path
        self._children: dict[tuple[str, ...], Any] = {}
        if not enabled:
            return
        try:
            import prometheus_client as pc
        except ImportError:  # pragma: no cover - prometheus is a service dependency
            log.warning("prometheus_client is not installed; service metrics disabled")
            self.enabled = False
            return
        self.registry = registry if registry is not None else pc.CollectorRegistry(auto_describe=True)
        backend = self._try_observability()
        for name, (typ, labels, help_) in METRIC_NAMES.items():
            if not name.startswith("pgw_service_"):
                continue
            col = backend(name) if backend is not None else None
            if col is None:
                col = self._make(pc, name, typ, labels, help_)
            self._c[name] = col

    def _try_observability(self) -> Any:
        """``PGWMetrics(registry).get`` if the observability package implements it, else ``None``."""
        try:
            from pgw.observability.metrics import PGWMetrics

            pm = PGWMetrics(self.registry, enabled=True)

            def get(name: str) -> Any:
                try:
                    return pm.get(name)
                except Exception:
                    return None

            return get
        except NotImplementedError:
            return None
        except Exception as exc:  # pragma: no cover - defensive: never fail startup over metrics
            log.warning("PGWMetrics unavailable (%s); using local collectors", type(exc).__name__)
            return None

    def _make(self, pc: Any, name: str, typ: str, labels: tuple[str, ...], help_: str) -> Any:
        if typ == "counter":
            return pc.Counter(name.removesuffix("_total"), help_, labels, registry=self.registry)
        if typ == "gauge":
            return pc.Gauge(name, help_, labels, registry=self.registry)
        if typ == "histogram":
            buckets = _ROW_BUCKETS if name.endswith("_rows") else _LATENCY_BUCKETS
            return pc.Histogram(name, help_, labels, registry=self.registry, buckets=buckets)
        raise ValueError(f"unknown metric type {typ!r} for {name}")  # pragma: no cover

    def get(self, name: str) -> Any:
        """Collector by contract name (no-op when disabled).

        Raises:
            KeyError: ``name`` is not a ``pgw_service_*`` contract metric.
        """
        if name not in METRIC_NAMES or not name.startswith("pgw_service_"):
            raise KeyError(name)
        if not self.enabled:
            return _Noop()
        return self._c[name]

    def _child(self, name: str, *values: str) -> Any:
        """``get(name).labels(*values)``, cached (label sets are bounded: tenants x kinds x codes)."""
        key = (name, *values)
        c = self._children.get(key)
        if c is None:
            c = self.get(name).labels(*values)
            if len(self._children) < 65536:
                self._children[key] = c
        return c

    # ---- convenience ---------------------------------------------------------------------
    def request_done(self, tenant: str, code: str, seconds: float) -> None:
        """Count one finished compute request and its latency."""
        if not self.enabled:
            return
        self._child("pgw_service_requests_total", tenant, code).inc()
        self._child("pgw_service_request_seconds", tenant).observe(max(0.0, seconds))

    def rows(self, tenant: str, kind: str, rows: int) -> None:
        """Count rows computed for a tenant."""
        if self.enabled:
            self._child("pgw_service_rows_total", tenant, kind).inc(rows)

    def bytes_io(self, tenant: str, bytes_in: int, bytes_out: int) -> None:
        """Count payload bytes."""
        if not self.enabled:
            return
        self._child("pgw_service_bytes_in_total", tenant).inc(bytes_in)
        self._child("pgw_service_bytes_out_total", tenant).inc(bytes_out)

    def batch(self, kind: str, backend: str, rows: int, seconds: float) -> None:
        """Record one kernel batch."""
        if not self.enabled:
            return
        self._child("pgw_service_batch_rows", kind).observe(rows)
        self._child("pgw_service_kernel_seconds", kind, backend).observe(seconds)

    def queue_depth(self, items: int) -> None:
        """Set the number of queued compute items."""
        if self.enabled:
            self._c["pgw_service_queue_depth"].set(items)

    def inflight(self, n: int) -> None:
        """Set the number of in-flight requests."""
        if self.enabled:
            self._c["pgw_service_inflight"].set(n)

    def render(self) -> tuple[bytes, str]:
        """``(body, content_type)`` in Prometheus text format."""
        if not self.enabled or self.registry is None:
            return b"", "text/plain; version=0.0.4; charset=utf-8"
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST
