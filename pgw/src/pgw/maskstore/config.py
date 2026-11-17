"""Mask store settings (env prefix ``PGW_MASKSTORE_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from pgw.core.settings import PGWSettings

__all__ = ["MaskStoreSettings"]


class MaskStoreSettings(PGWSettings):
    """Where the store lives and how it is protected."""

    model_config = SettingsConfigDict(
        env_prefix="PGW_MASKSTORE_", env_nested_delimiter="__", extra="ignore", frozen=True
    )

    root: Path = Field(Path("/var/lib/pgw/masks"), description="Store root directory")
    key_provider: Literal["file", "env", "pkcs11", "kms"] = "file"
    key_file: Path | None = Field(None, description="KEK file for the 'file' provider (0600, 32 bytes)")
    key_env_var: str = Field("PGW_MASKSTORE_KEK", description="Env var holding a base64 KEK for 'env'")
    pkcs11_module: Path | None = None
    pkcs11_token_label: str | None = None
    pkcs11_key_label: str | None = None
    kms_key_uri: str | None = Field(None, description="e.g. aws-kms://arn:..., gcp-kms://projects/...")
    low_water_slots: int = Field(4096, ge=0, description="Alert when available slots fall below this")
    target_slots: int = Field(65536, ge=0, description="Factory refill target")
    shred_consumed: bool = Field(True, description="Delete segment files once fully consumed")
    fsync: bool = Field(True, description="fsync state changes (never disable in production)")
