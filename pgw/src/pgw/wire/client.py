"""Async HTTP/2 client for the compute service (pooled, mTLS, retries, deadlines)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, SecretStr

from pgw.wire.messages import ComputeRequest, ComputeResponse, ModelsResponse

__all__ = ["ClientConfig", "ComputeClient"]


class ClientConfig(BaseModel):
    """Connection settings for one compute service endpoint."""

    base_url: str = Field(description="e.g. https://compute.example.net:8443")
    token: SecretStr | None = Field(None, description="Bearer token (tenant credential)")
    ca_file: Path | None = None
    client_cert_file: Path | None = None
    client_key_file: Path | None = None
    http2: bool = True
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 60.0
    max_connections: int = 16
    max_retries: int = 3
    backoff_base_s: float = 0.05
    backoff_max_s: float = 2.0
    max_request_bytes: int = 256 << 20


class ComputeClient:
    """Client used by :class:`pgw.engine.MaskedBackend`.

    * Retries only retryable errors (``unavailable``, ``rate_limited``, ``deadline_exceeded``,
      transport errors) with jittered exponential backoff. Retrying resends the same ``x_hat``
      (the server already saw it, so no extra leakage, and no extra mask is consumed).
    * Splits requests above ``max_request_bytes`` into several (by rows) and reassembles.
    * Verifies on :meth:`connect` that the server serves the expected fingerprint and ring.
    * Never logs payload bytes.
    """

    def __init__(self, config: ClientConfig):
        raise NotImplementedError("wire team: ComputeClient")

    async def connect(self, expected_fingerprint: str | None = None, ring: str | None = None) -> ModelsResponse:
        """Open the pool, fetch ``/pgw/v1/models`` and check fingerprint/ring/version."""
        raise NotImplementedError

    async def compute(self, request: ComputeRequest) -> ComputeResponse:
        """Send one compute request (with retries); raises typed :class:`pgw.core.PGWError` on failure."""
        raise NotImplementedError

    def compute_sync(self, request: ComputeRequest) -> ComputeResponse:
        """Blocking wrapper (runs on a private event loop thread)."""
        raise NotImplementedError

    async def models(self) -> ModelsResponse:
        """``GET /pgw/v1/models``."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Close connections."""
        raise NotImplementedError
