"""Model registry: which packs are served, their prepared weights, and device placement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pgw.core.types import LinearSpec

__all__ = ["ModelRegistry", "RegisteredModel"]


@dataclass
class RegisteredModel:
    """A loaded pack: name, fingerprint, linear specs and a :class:`ComputeKernel` per linear."""

    name: str
    fingerprint: str
    specs: Sequence[LinearSpec]
    kernels: dict[str, object]


class ModelRegistry:
    """Loads model packs (linears only, ``verify="full"`` at startup) onto the configured device.

    Lookup is by fingerprint (the wire protocol never trusts model names). Packs that do not fit
    in device memory use weight streaming (see :class:`pgw.factory.WeightStreamer`).
    """

    def __init__(self, device: str = "auto", backend: str = "auto"):
        raise NotImplementedError("service team: ModelRegistry")

    def load(self, pack_path: str | Path) -> RegisteredModel:
        """Load and register a pack; returns the entry."""
        raise NotImplementedError

    def get(self, fingerprint: str) -> RegisteredModel:
        """Entry by fingerprint; raises :class:`pgw.core.ModelNotFoundError`."""
        raise NotImplementedError

    def models(self) -> list[RegisteredModel]:
        """All registered models."""
        raise NotImplementedError
