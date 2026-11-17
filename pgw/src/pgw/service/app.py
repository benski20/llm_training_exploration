"""ASGI application of the compute service.

Routes (SPEC.md section 6.2):
    POST /pgw/v1/compute   msgpack ComputeRequest -> ComputeResponse
    GET  /pgw/v1/models    ModelsResponse
    GET  /healthz          liveness;  GET /readyz  readiness (packs loaded, device OK)
    GET  /metrics          Prometheus
Run with hypercorn for HTTP/2 (``pgw service serve``).
"""

from __future__ import annotations

from typing import Any

from pgw.service.config import ServiceSettings

__all__ = ["create_app"]


def create_app(settings: ServiceSettings | None = None) -> Any:
    """Build the FastAPI app (loads packs on startup, starts the batcher)."""
    raise NotImplementedError("service team: create_app")
