"""ASGI application of the gateway."""

from __future__ import annotations

from typing import Any

from pgw.gateway.config import GatewaySettings

__all__ = ["create_app"]


def create_app(settings: GatewaySettings | None = None) -> Any:
    """Build the FastAPI app.

    Public (API key): ``POST /v1/chat/completions``, ``POST /v1/completions``, ``GET /v1/models``.
    Admin (admin token): ``GET /admin/status``, ``GET /admin/masks``, ``POST /admin/keys``,
    ``DELETE /admin/keys/{key_id}``, ``POST /admin/models/{name}/reload``.
    Ops: ``/healthz``, ``/readyz``, ``/metrics``, ``/dashboard``.
    Errors use the OpenAI error shape with ``code`` = :class:`pgw.core.ErrorCode` value.
    """
    raise NotImplementedError("gateway team: create_app")
