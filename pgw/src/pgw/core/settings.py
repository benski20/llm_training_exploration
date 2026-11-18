"""Configuration base classes.

Every component defines a settings class deriving from :class:`PGWSettings` with its own env
prefix (``PGW_GATEWAY_``, ``PGW_SERVICE_``, ``PGW_FACTORY_``, ``PGW_MASKSTORE_`` ...). Values come
from (highest priority first): explicit constructor arguments, environment variables, a ``.env``
file, defaults. Nested fields use ``__`` as the delimiter (``PGW_GATEWAY_TLS__CERT_FILE``).
Secrets are typed :class:`pydantic.SecretStr` so they never appear in ``repr`` or logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pgw.core.ring import DEFAULT_RING, RingParams

__all__ = ["CommonSettings", "PGWSettings", "TLSConfig", "ring_from_setting"]


class PGWSettings(BaseSettings):
    """Base for all PGW settings classes (env prefix ``PGW_``)."""

    model_config = SettingsConfigDict(
        env_prefix="PGW_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )


class TLSConfig(BaseModel):
    """TLS / mTLS material. Paths are PEM files."""

    cert_file: Path | None = None
    key_file: Path | None = None
    key_password: SecretStr | None = None
    ca_file: Path | None = Field(None, description="CA bundle used to verify the peer (enables mTLS on servers)")
    require_client_cert: bool = False

    @property
    def enabled(self) -> bool:
        """True when a certificate and key are configured."""
        return self.cert_file is not None and self.key_file is not None


class CommonSettings(PGWSettings):
    """Settings shared by all processes."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = Field(False, description="Emit JSON log lines (recommended in production)")
    ring: str = Field(DEFAULT_RING.name, description="Ring preset, e.g. K40A12, K48A20, K32A8, K32A12C27")
    hemask_engine: Literal["auto", "cuda", "cpu_int8", "cpu_f64"] = Field(
        "auto", description="Encrypted mask generation: where the gateway-side GEMMs run (auto: a gateway "
        "GPU, else CPU int8 with AMX/VNNI, else exact float64). Read from PGW_HEMASK_ENGINE.",
    )

    @field_validator("ring")
    @classmethod
    def _valid_ring(cls, v: str) -> str:
        RingParams.from_name(v)
        return v.upper()

    @property
    def ring_params(self) -> RingParams:
        """Parsed :class:`RingParams`."""
        return RingParams.from_name(self.ring)


def ring_from_setting(name: str | RingParams | None) -> RingParams:
    """Accept a preset name, a :class:`RingParams` or ``None`` (default ring)."""
    if name is None:
        return DEFAULT_RING
    if isinstance(name, RingParams):
        return name
    return RingParams.from_name(name)
