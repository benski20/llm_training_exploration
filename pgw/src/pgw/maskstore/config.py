"""Mask store settings (env prefix ``PGW_MASKSTORE_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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
    key_previous_files: tuple[Path, ...] = Field((), description="Retired KEK files still accepted for unwrapping")
    key_env_var: str = Field("PGW_MASKSTORE_KEK", description="Env var holding a base64 KEK for 'env'")
    pkcs11_module: Path | None = None
    pkcs11_token_label: str | None = None
    pkcs11_key_label: str | None = None
    pkcs11_pin: SecretStr | None = None
    kms_key_uri: str | None = Field(None, description="e.g. aws-kms://arn:..., gcp-kms://projects/...")
    kms_mac_key: SecretStr | None = Field(None, description="base64 HMAC key enabling the journal with 'kms'")
    recipient_key_file: Path | None = Field(
        None, description="X25519 private key (0600) for importing partner segments (SPEC 7.7)"
    )
    low_water_slots: int = Field(4096, ge=0, description="Alert when available slots fall below this")
    target_slots: int = Field(65536, ge=0, description="Factory refill target")
    shred_consumed: bool = Field(True, description="Delete segment files once fully consumed")
    fsync: bool = Field(True, description="fsync state changes (never disable in production)")

    # ---- rollback protection (SPEC 7.6) ----------------------------------------------------------
    journal: bool = Field(True, description="Keep the HMAC-chained generation journal (needs a MAC-capable provider)")
    journal_file: Path | None = Field(None, description="Journal path (default <root>/generation.journal)")
    counter: Literal["none", "file", "tpm", "provider"] = Field(
        "none", description="External monotonic counter mirrored on every reservation"
    )
    counter_file: Path | None = Field(None, description="Counter file for counter='file' (use a separate volume)")
    tpm_nv_index: int | None = Field(None, description="TPM NV counter index for counter='tpm'")
    tpm_tcti: str | None = Field(None, description="TPM TCTI, e.g. device:/dev/tpmrm0")
    verify_on_open: bool = Field(True, description="Run verify_integrity() when the store is opened")

    # ---- housekeeping ----------------------------------------------------------------------------
    lease_timeout_s: float = Field(
        86400.0, gt=0, description="Open reservations older than this no longer block shredding"
    )
    audit_retention_days: float = Field(30.0, ge=0, description="Keep closed reservation records this long")
    lock_timeout_s: float = Field(60.0, gt=0, description="Max wait for the inter-process store lock")
    dek_cache_size: int = Field(64, ge=0, description="Unwrapped segment keys cached in memory (KMS round trips)")
