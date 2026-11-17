"""Metrics, the outbound-noise monitor, and the ops dashboard.

Owner: observability team. Contract: SPEC.md section 11.
"""

from pgw.observability.dashboard import dashboard_html, mount_dashboard
from pgw.observability.metrics import METRIC_NAMES, PGWMetrics
from pgw.observability.noise import NoiseReport, OutboundNoiseMonitor

__all__ = ["METRIC_NAMES", "NoiseReport", "OutboundNoiseMonitor", "PGWMetrics", "dashboard_html", "mount_dashboard"]
