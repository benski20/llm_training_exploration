"""Factory settings (env prefix ``PGW_FACTORY_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from pgw.core.settings import PGWSettings

__all__ = ["FactorySettings"]


class FactorySettings(PGWSettings):
    """Mask factory configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PGW_FACTORY_", env_nested_delimiter="__", extra="ignore", frozen=True
    )

    pack_path: Path = Field(description="Model pack directory")
    ring: str = "K40A12"
    device: str = Field("auto", description="'auto', 'cpu', 'cuda', 'cuda:1'")
    backend: Literal["auto", "cuda_int8", "cpu_f64", "cpu_f32"] = "auto"
    segment_slots: int = Field(4096, gt=0, description="Token positions per segment")
    chunk_slots: int = Field(256, gt=0, description="Rows per GEMM batch / encrypted chunk")
    checks: int = Field(2, ge=0, le=8, description="Freivalds vectors per linear per segment")
    gpu_memory_fraction: float = Field(0.85, gt=0, le=1, description="Above this, stream weights from host RAM")
    factory_id: str = Field("local", description="Recorded in segment headers")
    output: Literal["store", "export"] = Field("store", description="Write into a local MaskStore or export files")
    export_dir: Path | None = None
    poll_interval_s: float = Field(30.0, gt=0, description="Scheduler inventory poll period")
    max_segments_per_run: int = Field(0, ge=0, description="0 = unlimited")
