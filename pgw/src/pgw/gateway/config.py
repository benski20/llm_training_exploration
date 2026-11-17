"""Gateway settings (env prefix ``PGW_GATEWAY_``; models usually from a YAML/JSON file)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import SettingsConfigDict

from pgw.core.settings import PGWSettings, TLSConfig

__all__ = ["GatewayModelConfig", "GatewaySettings"]


class GatewayModelConfig(BaseModel):
    """One served model."""

    name: str = Field(description="Name clients use in 'model'")
    pack_path: Path
    mode: Literal["masked", "plain", "float"] = Field(
        "masked", description="'masked' = production; 'plain'/'float' = local dev/testing only (no service)"
    )
    ring: str = "K40A12"
    service_url: str | None = None
    service_token: SecretStr | None = None
    service_ca_file: Path | None = None
    client_cert_file: Path | None = None
    client_key_file: Path | None = None
    freivalds_rate: float = Field(1.0, ge=0.0, le=1.0)
    max_context: int | None = None
    max_concurrent_sessions: int = 32


class GatewaySettings(PGWSettings):
    """Gateway configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PGW_GATEWAY_", env_nested_delimiter="__", extra="ignore", frozen=True
    )

    host: str = "127.0.0.1"
    port: int = 8080
    models: list[GatewayModelConfig] = Field(default_factory=list)
    models_file: Path | None = Field(None, description="JSON/YAML file with a list of GatewayModelConfig")
    maskstore_root: Path = Path("/var/lib/pgw/masks")
    api_keys_file: Path | None = Field(None, description="Hashed API keys (managed with 'pgw gateway keys')")
    admin_token: SecretStr | None = None
    allow_anonymous: bool = Field(False, description="Dev only")
    requests_per_minute: int = 600
    tokens_per_minute: int = 200_000
    max_prompt_tokens: int = 32_768
    tls: TLSConfig = TLSConfig()
    noise_monitor: bool = Field(True, description="Statistically check outbound payloads look uniform")
    noise_monitor_fail_closed: bool = Field(True, description="Block traffic if the monitor alarms")
    cors_origins: list[str] = Field(default_factory=list)
    dashboard_enabled: bool = True
