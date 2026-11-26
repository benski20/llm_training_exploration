"""Per-device byte budget for encrypted maskgen items (opt-in, ``PGW_SERVICE_MASKGEN_DEVICE_MB``; off by default).

``ServiceHandler`` bounds maskgen by count (``maskgen_concurrency`` items per CUDA device). Items differ in size by
orders of magnitude (a pks-v3 item of 7B's masked head vs one of an ``o`` projection; ``max_request_bytes`` allows
~11k columns on 7B's head), so a count can still exhaust a small GPU. With a budget, an item additionally waits until
its modelled device bytes (:func:`item_device_bytes`, the memory plan's model of the code) fit next to the items
running on the same device. Queueing only, never a refusal: an item larger than the whole budget runs alone (it is
charged the whole budget). Waiters are served first in, first out, so a large item is not starved by small ones.
The budget changes when items start, never what they compute.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from typing import Any

__all__ = ["ByteBudget", "budget_bytes", "item_device_bytes", "maskgen_budget_for"]


class ByteBudget:
    """An asyncio byte budget with FIFO waiters (one event loop; :meth:`release` from that loop's thread)."""

    def __init__(self, budget: int):
        self.budget = max(1, int(budget))
        self.used = 0
        self._waiters: deque[tuple[int, asyncio.Future[None]]] = deque()

    @property
    def waiting(self) -> int:
        return sum(1 for _, f in self._waiters if not f.done())

    async def acquire(self, nbytes: int) -> int:
        """Wait until ``nbytes`` (at most the whole budget) fit; returns the bytes charged (pass to :meth:`release`)."""
        n = max(0, min(int(nbytes), self.budget))
        if not self._waiters and self.used + n <= self.budget:
            self.used += n
            return n
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (n, fut)
        self._waiters.append(entry)
        try:
            await fut
        except BaseException:
            if fut.done() and not fut.cancelled():  # granted, then cancelled before running: give the bytes back
                self.release(n)
            else:
                try:
                    self._waiters.remove(entry)
                except ValueError:
                    pass
                self._wake()
            raise
        return n

    def release(self, charged: int) -> None:
        self.used -= int(charged)
        self._wake()

    def _wake(self) -> None:
        while self._waiters:
            n, fut = self._waiters[0]
            if fut.done():
                self._waiters.popleft()
                continue
            if self.used + n > self.budget:
                return
            self._waiters.popleft()
            self.used += n
            fut.set_result(None)


def budget_bytes() -> int:
    """``PGW_SERVICE_MASKGEN_DEVICE_MB`` in bytes (0: no budget)."""
    raw = (os.environ.get("PGW_SERVICE_MASKGEN_DEVICE_MB") or "").strip()
    return max(0, int(raw)) << 20 if raw else 0


def item_device_bytes(kern: Any, it: Any) -> int:
    """Modelled peak device bytes of one maskgen item on ``kern`` (:mod:`pgw.service.memplan`'s models of the code;
    a pks-v3 item counts the larger of its two hint paths; tensor-parallel kernels count their largest shard)."""
    from pgw.service import memplan as mp

    m, n, cols = int(kern.out_features), int(kern.in_features), int(it.cols)
    bounds = getattr(kern, "bounds", None)
    if bounds:
        m = max(int(r1) - int(r0) for r0, r1 in bounds)
    if getattr(it, "ksk", None) is not None:
        cfg = mp.MemConfig(maskgen_wire="v3", maskgen_cols=cols)
        return max(mp._maskgen_v3_bytes(m, n, cfg, False, False), mp._maskgen_v3_bytes(m, n, cfg, False, True))
    b_bits = int(getattr(it, "b_bits", 0) or 0)
    if b_bits:
        cfg = mp.MemConfig(maskgen_wire="v2", maskgen_cols=cols, maskgen_b_bits=b_bits,
                           maskgen_y_bits=int(getattr(it, "y_bits", 0) or b_bits))
        return mp._maskgen_item_bytes(m, n, cfg, False)
    nb = max(1, int(getattr(it, "q_bits", 64) or 64) // 8)
    return mp._mulmod64_bytes(m, n, cols, nb, mp.MASKGEN_PART_BYTES_CUDA, False)


def maskgen_budget_for(owner: Any, kern: Any) -> ByteBudget | None:
    """The byte budget of ``kern``'s CUDA device, kept on ``owner`` (the service handler); ``None`` when no budget
    is configured or the kernel is not on a CUDA device."""
    dev = getattr(kern, "device", None)
    if getattr(dev, "type", "cpu") != "cuda":
        return None
    per_dev: dict[str, ByteBudget | None] | None = owner.__dict__.get("_maskgen_byte_budgets")
    if per_dev is None:
        per_dev = owner.__dict__["_maskgen_byte_budgets"] = {}
    key = str(dev)
    if key not in per_dev:
        total = budget_bytes()
        per_dev[key] = ByteBudget(total) if total > 0 else None
    return per_dev[key]
