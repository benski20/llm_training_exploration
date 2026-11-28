"""Pre-staged model packs on per-data-centre network volumes (fast cold start; verified at every boot).

A RunPod network volume lives in one data centre and can be mounted by any pod there. The controller keeps one
volume per data centre it may place pods in (``pgw-packs-<dc>``), filled once by a short CPU **packer pod** in that
data centre with the shard-wise packer (:mod:`pgw.modelpack.shardwise`: one checkpoint shard on disk at a time;
byte-identical to ``pgw pack``) from the pinned Hugging Face revision. A GPU pod then mounts the volume read-only
and never downloads or packs anything at boot.

Nothing on the volume is trusted: at boot the pod opens the pack with ``verify="full"`` (every tensor and file
hashed and compared with the manifest, the manifest's fingerprint recomputed) and requires **the pinned
fingerprint** (:data:`pgw.sdk.provisioning.PINNED_MODELS`); a mismatch stops the boot before the pod registers. A
tampered or half-written volume therefore costs a failed boot, never a wrong answer (and the clients check the
fingerprint and every result anyway).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["PACK_GB", "StageItem", "pinned_fingerprint", "stage_plan", "verify_staged", "volume_name"]

#: Pack sizes in GB (int8 linears + bf16 embeddings and head; ``scaling_model`` shapes; format 2). Estimates within
#: a few % (tokenizer, manifest and the masked-head entry are small).
PACK_GB: dict[str, float] = {"qwen2.5-0.5b-instruct": 0.9, "qwen2.5-1.5b-instruct": 2.2,
                             "qwen2.5-3b-instruct": 3.9, "qwen2.5-7b-instruct": 8.7,
                             "qwen2.5-14b-instruct": 16.5, "qwen2.5-32b-instruct": 34.3,
                             "qwen2.5-72b-instruct": 75.2}
#: Shard-wise packer scratch (two checkpoint shards of ~4 GB) while staging; freed afterwards.
SCRATCH_GB = 9.0


def volume_name(dc: str) -> str:
    return f"pgw-packs-{dc.lower()}"


def pinned_fingerprint(model: str) -> str:
    from pgw.sdk.provisioning import PINNED_MODELS

    pin = PINNED_MODELS.get(model)
    if pin is None or not pin.fingerprint:
        raise KeyError(f"{model!r} has no pinned fingerprint: it cannot be pre-staged")
    return pin.fingerprint


@dataclass(frozen=True)
class StageItem:
    dc: str
    volume: str
    size_gb: int
    models: tuple[str, ...]
    fingerprints: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {"dc": self.dc, "volume": self.volume, "size_gb": self.size_gb, "models": list(self.models),
                "fingerprints": [f[:16] for f in self.fingerprints]}


def stage_plan(dcs: Iterable[str], models: Iterable[str], *, headroom: float = 1.15) -> list[StageItem]:
    """One volume per data centre holding every model (size: packs x ``headroom`` + packer scratch, rounded up)."""
    ms = tuple(models)
    fps = tuple(pinned_fingerprint(m) for m in ms)
    size = math.ceil(sum(PACK_GB[m] for m in ms) * headroom + SCRATCH_GB)
    return [StageItem(dc, volume_name(dc), size, ms, fps) for dc in dict.fromkeys(dcs)]


def verify_staged(root: Path, model: str, *, expected: str | None = None, workers: int | None = None) -> Any:
    """Open ``root/model`` with full verification against the pinned (or ``expected``) fingerprint.

    Returns:
        The open :class:`pgw.modelpack.loader.ModelPack` (linears only).

    Raises:
        FingerprintMismatchError / ConfigError: the staged pack is not the pinned one, or incomplete.
    """
    from pgw.modelpack.loader import ModelPack

    fp = expected or pinned_fingerprint(model)
    return ModelPack.open(Path(root) / model, verify="full", expected_fingerprint=fp, components=("linears",),
                          workers=workers)


def cost_per_month(items: Iterable[StageItem], usd_gb_month: float = 0.07) -> float:
    """Network-volume storage, USD per month (RunPod's table price, ``launch.NETWORK_VOLUME_GB_MONTH``)."""
    return sum(i.size_gb for i in items) * usd_gb_month


def summary(items: Iterable[StageItem], prices: Mapping[str, float] | None = None) -> dict[str, Any]:
    items = list(items)
    return {"volumes": [i.public() for i in items], "total_gb": sum(i.size_gb for i in items),
            "usd_per_month": round(cost_per_month(items, (prices or {}).get("network_volume_gb_month", 0.07)), 2)}
