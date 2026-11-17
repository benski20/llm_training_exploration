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
    trusted_partner_keys: tuple[str, ...] = Field(
        (),
        description="Ed25519 public keys (hex or base64) of partner factories whose signed segments may be imported",
    )
    require_partner_signature: bool = Field(
        False,
        description="Legacy: signatures are required by default now; when set, it also overrides "
        "allow_unsigned_partner_segments",
    )
    allow_unsigned_partner_segments: bool = Field(
        False,
        description="DEV ONLY: import HPKE partner segments without a trusted signature. Anyone holding the "
        "public recipient key (e.g. the compute operator) could then inject masks it knows",
    )
    allow_unbound_partner_segments: bool = Field(
        False,
        description="DEV ONLY: import HPKE partner segments whose signed header names no destination store "
        "(such a file could be imported into a second store and its masks used twice)",
    )
    min_soundness_bits: float = Field(
        20.0,
        ge=0,
        description="Refuse to import segments whose Freivalds checks give fewer bits of soundness per call "
        "for any linear (see pgw.core.freivalds.soundness_bits)",
    )
    low_water_slots: int = Field(4096, ge=0, description="Alert when available slots fall below this")
    target_slots: int = Field(65536, ge=0, description="Factory refill target")
    shred_consumed: bool = Field(True, description="Delete segment files once fully consumed")
    min_free_bytes: int = Field(
        1 << 30,
        ge=0,
        description="New segments are refused unless the store volume keeps at least this many bytes free after "
        "writing them (headroom for state.sqlite, the journal and gateway state on the same volume)",
    )
    shred_zero_fill: bool = Field(
        True,
        description="After the crypto-erase of a consumed segment (random overwrite of its key header), also "
        "overwrite the whole body with zeros in a background thread. Useless on SSDs and copy-on-write "
        "filesystems (disable there to save the write bandwidth)",
    )
    fsync: bool = Field(True, description="fsync state changes (never disable in production)")

    # ---- rollback protection (SPEC 7.6) ----------------------------------------------------------
    journal: bool = Field(True, description="Keep the HMAC-chained generation journal (needs a MAC-capable provider)")
    journal_file: Path | None = Field(None, description="Journal path (default <root>/generation.journal)")
    counter: Literal["none", "file", "tpm", "provider"] = Field(
        "none",
        description="External monotonic counter mirrored on every reservation. Required for reserving masks "
        "unless allow_no_external_counter is set (dev/staging only)",
    )
    counter_file: Path | None = Field(None, description="Counter file for counter='file' (use a separate volume)")
    tpm_nv_index: int | None = Field(None, description="TPM NV counter index for counter='tpm'")
    tpm_tcti: str | None = Field(None, description="TPM TCTI, e.g. device:/dev/tpmrm0")
    allow_no_external_counter: bool = Field(
        False,
        description="DEV/STAGING ONLY: hand out masks without an external monotonic counter. The HMAC journal "
        "lives next to the database, so restoring a snapshot of the whole volume would go undetected and "
        "re-serve consumed masks",
    )
    allow_network_filesystem: bool = Field(
        False,
        description="Open a store on NFS/CIFS/CephFS/GlusterFS/FUSE. SQLite WAL and flock are not reliable "
        "there; a lost cursor update re-serves masks. Never share one store between hosts",
    )
    verify_on_open: bool = Field(True, description="Run verify_integrity() when the store is opened")
    allow_no_rollback_witness: bool = Field(
        False,
        description="Dev/staging only: open a store with neither a MAC journal nor a monotonic counter "
        "(rollback of the state database would be undetectable)",
    )

    # ---- housekeeping ----------------------------------------------------------------------------
    lease_timeout_s: float = Field(
        86400.0, gt=0, description="Open reservations older than this no longer block shredding"
    )
    audit_retention_days: float = Field(30.0, ge=0, description="Keep closed reservation records this long")
    lock_timeout_s: float = Field(60.0, gt=0, description="Max wait for the inter-process store lock")
    dek_cache_size: int = Field(64, ge=0, description="Unwrapped segment keys cached in memory (KMS round trips)")
