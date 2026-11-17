"""Factory settings (env prefix ``PGW_FACTORY_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from pgw.core.ring import RingParams
from pgw.core.settings import PGWSettings

__all__ = ["FactorySettings"]


class FactorySettings(PGWSettings):
    """Mask factory configuration.

    Example (env)::

        PGW_FACTORY_PACK_PATH=/packs/llama3-8b
        PGW_FACTORY_RINGS='["K40A12"]'
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
    ring: str = "K40A12"
    rings: tuple[str, ...] = Field((), description="Rings to produce (default: just `ring`)")
    device: str = Field("auto", description="'auto', 'cpu', 'cuda', 'cuda:1'")
    backend: Literal["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"] = "auto"
    segment_slots: int = Field(4096, gt=0, description="Token positions per segment")
    chunk_slots: int = Field(256, gt=0, description="Rows per GEMM batch / encrypted chunk")
    checks: int = Field(2, ge=0, le=8, description="Freivalds vectors per linear per segment")
    gpu_memory_fraction: float = Field(0.85, gt=0, le=1, description="Above this, stream weights from host RAM")
    factory_id: str = Field("local", description="Recorded in segment headers")
    output: Literal["store", "export"] = Field("store", description="Write into a local MaskStore or export files")
    export_dir: Path | None = None
    recipient_public_key_file: Path | None = Field(
        None, description="export: customer's raw 32-byte X25519 public key (or hex/base64 text) for HPKE wrapping"
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
    # ---- crash safety -----------------------------------------------------------------------
    state_dir: Path | None = Field(None, description="Factory state/lock directory (default: store root)")
    error_backoff_s: float = Field(10.0, gt=0, description="Initial back-off after a failed run (doubles to 600 s)")

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
        return self

    @property
    def ring_names(self) -> tuple[str, ...]:
        """Rings to produce, de-duplicated."""
        return self.rings or (self.ring,)

    @property
    def all_pack_paths(self) -> tuple[Path, ...]:
        """Primary plus extra packs."""
        return (self.pack_path, *[p for p in self.extra_pack_paths if p != self.pack_path])
