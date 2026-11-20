"""Factory settings (env prefix ``PGW_FACTORY_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from pgw.core.ring import RingParams
from pgw.core.settings import PGWSettings
from pgw.maskstore.format import DEFAULT_CHUNK_BYTES, MAX_CHUNK_BYTES

__all__ = ["FactorySettings"]


class FactorySettings(PGWSettings):
    """Mask factory configuration.

    Example (env)::

        PGW_FACTORY_PACK_PATH=/packs/llama3-8b
        PGW_FACTORY_RINGS='["K32A12C27"]'
        PGW_FACTORY_LOW_WATER_SLOTS=50000
        PGW_FACTORY_TARGET_SLOTS=500000
        PGW_FACTORY_MAX_SLOTS_PER_HOUR=200000   # leave GPU headroom for other jobs
        PGW_FACTORY_NICE=10
    """

    model_config = SettingsConfigDict(
        env_prefix="PGW_FACTORY_", env_nested_delimiter="__", extra="ignore", frozen=True
    )

    pack_path: Path = Field(description="Model pack directory")
    extra_pack_paths: tuple[Path, ...] = Field((), description="Further packs kept topped up by the same factory")
    ring: str = "K32A12C27"
    rings: tuple[str, ...] = Field((), description="Rings to produce (default: just `ring`)")
    device: str = Field("auto", description="'auto', 'cpu', 'cuda', 'cuda:1'")
    backend: Literal["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"] = "auto"
    segment_slots: int = Field(4096, gt=0, description="Token positions per segment")
    first_segment_slots: int | None = Field(
        None, gt=0,
        description="When a (model, ring) has no masks at all (a new store), produce one segment of this many slots "
        "first, then continue with segment_slots: a new gateway can serve after a small segment instead of waiting "
        "for a full one (minutes with encrypted generation over a slow link). Unset = segment_slots",
    )
    chunk_slots: int = Field(
        256, gt=0, description="Rows per GEMM batch (factory throughput; not the stored chunk size)"
    )
    segment_chunk_slots: int | None = Field(
        None, gt=0,
        description="Slots per encrypted W r chunk in the segment file. Unset = sized so that the widest "
        "linear's chunk is about segment_chunk_bytes. Every open gateway session caches one decrypted chunk "
        "per linear, so large chunks multiply gateway memory by max_concurrent_sessions",
    )
    segment_chunk_bytes: int = Field(
        DEFAULT_CHUNK_BYTES, gt=0, le=MAX_CHUNK_BYTES,
        description="Target plaintext bytes per encrypted chunk of the widest linear (auto chunk sizing)",
    )
    min_free_bytes: int = Field(
        2 << 30, ge=0,
        description="Stop producing when a further segment would leave less than this free on the target volume",
    )
    max_store_bytes: int = Field(
        0, ge=0, description="Cap on bytes of segment files held in the store / outbox (0 = only the free-space guard)"
    )
    checks: int | None = Field(
        None, ge=0, le=8,
        description="Freivalds vectors per linear per segment; unset = smallest number meeting 20 bits of "
        "soundness for the ring and model (gateways and stores refuse weaker segments by default)",
    )
    gpu_memory_fraction: float = Field(0.85, gt=0, le=1, description="Above this, stream weights from host RAM")
    factory_id: str = Field("local", description="Recorded in segment headers")
    output: Literal["store", "export"] = Field("store", description="Write into a local MaskStore or export files")
    export_dir: Path | None = None
    recipient_public_key_file: Path | None = Field(
        None, description="export: customer's raw 32-byte X25519 public key (or hex/base64 text) for HPKE wrapping"
    )
    destination_store_id: str | None = Field(
        None,
        description="export: store id of the customer mask store the segments are for ('pgw maskstore status'); "
        "recorded in the signed header so one file can never be imported into two stores",
    )
    partner_signing_key_file: Path | None = Field(
        None, description="export: partner Ed25519 signing key; each exported segment gets a .sig sidecar (SPEC 7.7)"
    )
    poll_interval_s: float = Field(30.0, gt=0, description="Scheduler inventory poll period")
    max_segments_per_run: int = Field(0, ge=0, description="0 = unlimited")
    low_water_slots: int | None = Field(None, ge=0, description="Start refilling below this (default: store setting)")
    target_slots: int | None = Field(None, ge=0, description="Refill up to this (default: store setting)")
    # ---- resource governance ----------------------------------------------------------------
    max_slots_per_hour: float = Field(0.0, ge=0, description="Rate limit on produced slots; 0 = unlimited")
    duty_cycle: float = Field(1.0, gt=0, le=1, description="Fraction of wall time spent computing (idle in between)")
    nice: int = Field(0, ge=0, le=19, description="Process niceness increment (low CPU priority)")
    cpu_threads: int = Field(0, ge=0, description="torch intra-op threads; 0 = torch default")
    pipeline: bool = Field(True, description="Overlap r generation, GEMMs and encryption in threads")
    self_check_rows: int = Field(
        8, ge=-1, description="Rows per chunk whose W8.r output is verified before writing (-1 = all, 0 = off)"
    )
    gpu_prng: Literal["auto"] | bool = Field(
        "auto",
        description="Expand mask rows r (ChaCha20 + digit split) on the factory GPU instead of the CPU: 'auto' = on "
        "for CUDA factories once the kernel's self-test against the CPU expander passed, false = always CPU, "
        "true = on (still self-tested; never used when the test fails). Needs self_check_rows != 0 (the output "
        "self-check re-derives sampled r rows on the CPU). Same derivation and output",
    )
    # ---- encrypted mask generation (pgw.hemask) -----------------------------------------------
    he_service_url: str | None = Field(
        None, description="Compute service URL: get W8.r from the provider on MLWE-encrypted masks instead of "
        "computing it locally (no local GEMMs; the provider learns nothing about the masks)",
    )
    he_token_file: Path | None = Field(None, description="File with the service bearer token (encrypted mode)")
    he_ca_file: Path | None = Field(None, description="CA bundle for the service (encrypted mode)")
    he_client_cert_file: Path | None = Field(
        None, description="mTLS client certificate (PEM) presented to the service (encrypted mode): the gateway's "
        "enrolled client.crt, needed when the service requires client certificates for compute. Renewals of the "
        "file are picked up by new connections",
    )
    he_client_key_file: Path | None = Field(None, description="Private key of he_client_cert_file (PEM, 0600)")
    he_hint_dir: Path | None = Field(None, description="Where the one-time hint W8.A is kept (default: <state>/hemask)")
    he_rank: int = Field(16, ge=8, le=32, description="MLWE module rank (LWE dimension = 256 x rank)")
    he_groups_per_call: int = Field(4, ge=1, le=16, description="256-slot groups per provider request")
    he_timeout_s: float = Field(600.0, gt=0, description="Budget per provider request")
    # ---- crash safety -----------------------------------------------------------------------
    state_dir: Path | None = Field(None, description="Factory state/lock directory (default: store root)")
    error_backoff_s: float = Field(10.0, gt=0, description="Initial back-off after a failed run (doubles to 600 s)")
    # ---- observability ----------------------------------------------------------------------
    metrics_port: int | None = Field(
        None, ge=1, le=65535, description="Serve /metrics and /healthz on this internal port (unset = off)"
    )
    metrics_host: str = Field("0.0.0.0", description="Bind address of the metrics/health listener")  # noqa: S104
    health_max_stale_s: float = Field(
        3600.0, gt=0,
        description="/healthz fails when no scheduler plan/segment completed for this long (hung GPU, lock wait)",
    )
    health_max_consecutive_errors: int = Field(
        5, ge=1, description="/healthz fails after this many consecutive failed runs"
    )

    @field_validator("ring")
    @classmethod
    def _valid_ring(cls, v: str) -> str:
        return RingParams.from_name(v).name

    @field_validator("rings")
    @classmethod
    def _valid_rings(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        out: list[str] = []
        for r in v:
            name = RingParams.from_name(r).name
            if name not in out:
                out.append(name)
        return tuple(out)

    @model_validator(mode="after")
    def _check(self) -> FactorySettings:
        if self.output == "export" and self.export_dir is None:
            raise ValueError("output='export' requires export_dir")
        lw, tg = self.low_water_slots, self.target_slots
        if lw is not None and tg is not None and tg < lw:
            raise ValueError("target_slots must be >= low_water_slots")
        if (self.he_client_cert_file is None) != (self.he_client_key_file is None):
            raise ValueError("he_client_cert_file and he_client_key_file must be set together")
        return self

    @property
    def ring_names(self) -> tuple[str, ...]:
        """Rings to produce, de-duplicated."""
        return self.rings or (self.ring,)

    @property
    def all_pack_paths(self) -> tuple[Path, ...]:
        """Primary plus extra packs."""
        return (self.pack_path, *[p for p in self.extra_pack_paths if p != self.pack_path])
