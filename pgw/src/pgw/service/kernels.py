"""Compute kernels: exact ``W8 x_hat mod 2^K`` for one linear on the service device."""

from __future__ import annotations

import torch

from pgw.core.ring import RingParams
from pgw.core.types import LinearSpec

__all__ = ["ComputeKernel"]


class ComputeKernel:
    """Wraps :class:`pgw.core.PreparedWeight` for one linear (GPU ``cuda_int8`` or CPU float).

    Measured: Llama-3-8B, T4, K=48: 0.45 s/token at batch 1, 0.014 s/token amortized at batch 32.
    Future: fused CUDA/Triton kernel doing digit split + K/8 GEMMs + shift-accumulate in one pass.
    """

    def __init__(self, spec: LinearSpec, w8: torch.Tensor, device: str = "auto", backend: str = "auto"):
        raise NotImplementedError("service team: ComputeKernel")

    def run(self, x_hat: torch.Tensor, ring: RingParams) -> torch.Tensor:
        """``(B, n)`` int64 -> ``(B, m)`` int64 in ``[0, 2^K)``; input/output on CPU."""
        raise NotImplementedError

    @property
    def device_bytes(self) -> int:
        """Device memory held."""
        raise NotImplementedError
