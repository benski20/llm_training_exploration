"""Self-contained ops dashboard page (HTML + JS polling ``/metrics`` and ``/admin/status``)."""

from __future__ import annotations

from typing import Any

__all__ = ["dashboard_html", "mount_dashboard"]


def dashboard_html(title: str = "PGW") -> str:
    """Return the dashboard page (no external assets; works air-gapped)."""
    raise NotImplementedError("observability team: dashboard_html")


def mount_dashboard(app: Any, path: str = "/dashboard") -> None:
    """Add the dashboard route to a FastAPI app."""
    raise NotImplementedError("observability team: mount_dashboard")
