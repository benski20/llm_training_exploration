"""Compute service settings (env prefix ``PGW_SERVICE_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from pgw.core.settings import PGWSettings, TLSConfig

__all__ = ["ServiceSettings"]


class ServiceSettings(PGWSettings):
    """Compute service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PGW_SERVICE_", env_nested_delimiter="__", extra="ignore", frozen=True
    )

    host: str = "0.0.0.0"  # noqa: S104 - service is meant to listen on all interfaces
    port: int = 8443
    packs: list[Path] = Field(default_factory=list, description="Model pack directories to serve")
    rings: list[str] = Field(default_factory=lambda: ["K40A12", "K48A20", "K32A8"])
    device: str = "auto"
    backend: Literal["auto", "cuda_int8", "cpu_f64", "cpu_f32"] = "auto"
    batch_window_ms: float = Field(2.0, ge=0, description="Max wait to coalesce rows for one linear")
    max_batch_rows: int = Field(4096, gt=0)
    max_request_bytes: int = 256 << 20
    max_rows_per_item: int = 8192
    max_inflight_requests: int = Field(256, gt=0, description="Above this, reply 'unavailable'")
    tls: TLSConfig = TLSConfig()
    tokens_file: Path | None = Field(None, description="Tenant tokens: lines 'tenant_id:sha256(token)'")
    allow_unauthenticated: bool = Field(False, description="Dev only")
    metrics_enabled: bool = True
    server_name: str = "pgw-service"
