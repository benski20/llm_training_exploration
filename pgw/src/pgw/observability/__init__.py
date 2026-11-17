"""Metrics, the outbound-noise monitor, and the ops dashboard.

Owner: observability team. Contract: SPEC.md section 11.

* :mod:`pgw.observability.metrics`: the metric-name contract (:data:`METRIC_NAMES`) and
  :class:`PGWMetrics`, a lazily-populated Prometheus registry shared by gateway, service and factory.
* :mod:`pgw.observability.noise`: :class:`OutboundNoiseMonitor`, a statistical fail-closed guard
  that outbound payloads look uniformly random and are never repeated (mask reuse).
* :mod:`pgw.observability.dashboard`: a dependency-free HTML ops page served at ``/dashboard``.
"""

from pgw.observability.dashboard import DASHBOARD_CSP, dashboard_html, mount_dashboard
from pgw.observability.metrics import METRIC_NAMES, PGWMetrics, default_registry, safe_label
from pgw.observability.noise import NoiseMonitorConfig, NoiseReport, OutboundNoiseMonitor

__all__ = [
    "DASHBOARD_CSP",
    "METRIC_NAMES",
    "NoiseMonitorConfig",
    "NoiseReport",
    "OutboundNoiseMonitor",
    "PGWMetrics",
    "dashboard_html",
    "default_registry",
    "mount_dashboard",
    "safe_label",
]
