"""Core mask generation: for each linear, ``r`` from the segment seed and ``W r mod 2^K``."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from pgw.core.ring import RingParams
from pgw.core.types import LinearId

__all__ = ["GeneratedLinear", "MaskGenerator"]


@dataclass
class GeneratedLinear:
    """Output for one linear and one slot chunk.

    Attributes:
        id: Linear id.
        slot_start: First slot of this chunk.
        wr: ``(rows, m)`` int64 ring elements ``W8 r``.
        uw: ``(checks, n)`` int64 ``u^T W8`` (only on the first chunk of each linear, else ``None``).
    """

    id: LinearId
    slot_start: int
    wr: torch.Tensor
    uw: torch.Tensor | None


class MaskGenerator:
    """Computes masks for segments of a model pack.

    Derivation is normative (see :mod:`pgw.maskstore.format`): ``r`` rows come from
    ``SeedExpander(seed).ring_uniform_rows(n, K, slot, rows, "r", linear_id)`` and the check vectors
    from ``ring_uniform((checks, m), K, "u", linear_id)``. Uses :class:`pgw.core.PreparedWeight`
    (``cuda_int8`` on GPU: ~0.0066 s/token for 8B at K=48 on a T4).

    Args:
        pack: Open :class:`pgw.modelpack.ModelPack`.
        ring: Ring parameters.
        device: Torch device for GEMMs.
        backend: ringmm backend.
        streamer: Optional :class:`WeightStreamer` when the model does not fit on the device.
    """

    def __init__(self, pack: object, ring: RingParams, device: str = "cpu", backend: str = "auto",
                 streamer: object | None = None):
        raise NotImplementedError("factory team: MaskGenerator")

    def generate(
        self, seed: bytes, slots: int, *, checks: int = 2, chunk_slots: int = 256
    ) -> Iterator[GeneratedLinear]:
        """Yield chunks for every linear (canonical order, then slot order) of one segment."""
        raise NotImplementedError

    def generate_into_store(self, store: object, slots: int, *, checks: int = 2, chunk_slots: int = 256,
                            factory_id: str = "local") -> str:
        """Generate a fresh-seeded segment straight into a :class:`pgw.maskstore.MaskStore`; return its id."""
        raise NotImplementedError

    def tokens_per_second(self) -> float:
        """Measured throughput of the last run (slots/s)."""
        raise NotImplementedError
