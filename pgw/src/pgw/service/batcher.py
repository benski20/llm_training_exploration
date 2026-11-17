"""Cross-request batching: coalesce rows for the same (fingerprint, ring, linear) into one GEMM."""

from __future__ import annotations

import torch

from pgw.core.ring import RingParams

__all__ = ["BatchScheduler"]


class BatchScheduler:
    """Async micro-batcher.

    Requests for the same ``(fingerprint, ring, linear)`` arriving within ``batch_window_ms`` (or
    until ``max_batch_rows``) are concatenated, computed with one kernel call and split back.
    Batching is the main throughput lever (batch 32 is ~30x cheaper per token than batch 1).
    A single worker per device serializes GPU work; CPU work runs in a thread pool.
    """

    def __init__(self, registry: object, *, batch_window_ms: float = 2.0, max_batch_rows: int = 4096):
        raise NotImplementedError("service team: BatchScheduler")

    async def submit(self, fingerprint: str, ring: RingParams, linear: str, x_hat: torch.Tensor) -> torch.Tensor:
        """Queue rows and await their results."""
        raise NotImplementedError

    async def start(self) -> None:
        """Start worker tasks."""
        raise NotImplementedError

    async def stop(self, drain: bool = True) -> None:
        """Stop workers, optionally finishing queued work."""
        raise NotImplementedError
