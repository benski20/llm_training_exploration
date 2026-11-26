"""Per-device maskgen byte budget (``pgw.service.maskgen_budget``, opt-in): queueing only, FIFO, never a refusal."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pgw.service.maskgen_budget import ByteBudget, budget_bytes, item_device_bytes, maskgen_budget_for

CUDA0 = torch.device("cuda", 0)


async def test_fifo_and_oversize_items_run_alone() -> None:
    b = ByteBudget(100)
    assert await b.acquire(60) == 60
    order: list[str] = []

    async def take(name: str, n: int) -> int:
        got = await b.acquire(n)
        order.append(name)
        return got

    big = asyncio.ensure_future(take("big", 500))  # larger than the budget: charged the whole budget, runs alone
    await asyncio.sleep(0)
    small = asyncio.ensure_future(take("small", 10))  # would fit now, but waits behind the earlier waiter (FIFO)
    await asyncio.sleep(0.01)
    assert not big.done() and not small.done() and b.waiting == 2
    b.release(60)
    assert await big == 100
    await asyncio.sleep(0.01)
    assert not small.done()
    b.release(100)
    assert await small == 10 and order == ["big", "small"]
    b.release(10)
    assert b.used == 0


async def test_cancelled_waiters_release_nothing_and_wake_the_next() -> None:
    b = ByteBudget(100)
    await b.acquire(100)
    first = asyncio.ensure_future(b.acquire(80))
    second = asyncio.ensure_future(b.acquire(20))
    await asyncio.sleep(0.01)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert b.used == 100 and not second.done()
    b.release(100)
    assert await second == 20 and b.used == 20
    # granted, then cancelled before the waiter ran: the bytes go back
    third = asyncio.ensure_future(b.acquire(90))
    await asyncio.sleep(0.01)
    b.release(20)  # grants third (sets its future) ...
    third.cancel()  # ... which is cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await third
    assert b.used == 0


def test_budget_off_by_default_and_per_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = SimpleNamespace()
    monkeypatch.delenv("PGW_SERVICE_MASKGEN_DEVICE_MB", raising=False)
    assert budget_bytes() == 0
    assert maskgen_budget_for(owner, SimpleNamespace(device=CUDA0)) is None
    owner2 = SimpleNamespace()
    monkeypatch.setenv("PGW_SERVICE_MASKGEN_DEVICE_MB", "4096")
    b0 = maskgen_budget_for(owner2, SimpleNamespace(device=CUDA0))
    assert isinstance(b0, ByteBudget) and b0.budget == 4096 << 20
    assert maskgen_budget_for(owner2, SimpleNamespace(device=CUDA0)) is b0
    assert maskgen_budget_for(owner2, SimpleNamespace(device=torch.device("cuda", 1))) is not b0
    assert maskgen_budget_for(owner2, SimpleNamespace(device=torch.device("cpu"))) is None
    assert maskgen_budget_for(owner2, None) is None


def test_item_bytes_follow_the_memory_plan() -> None:
    from pgw.hemask.wire import MaskgenItem

    gib = 1 << 30
    head = SimpleNamespace(out_features=152064, in_features=3584)
    o = SimpleNamespace(out_features=3584, in_features=3584)
    v3 = SimpleNamespace(cols=1024, ksk=b"k", b_bits=56)
    for k in (head, o):  # pks-v3 items are bounded by row and column blocks whatever the linear
        assert 0.3 * gib < item_device_bytes(k, v3) < 3 * gib
    # wire v2 items grow with their columns (max_request_bytes allows ~11k on 7B's head): what the budget is for
    v2 = MaskgenItem("28.head", 1024, b"", b_bits=53, y_bits=35)
    v2_big = MaskgenItem("28.head", 11264, b"", b_bits=53, y_bits=35)
    assert item_device_bytes(head, v2_big) > 5 * item_device_bytes(head, v2) > 0
    v1 = MaskgenItem("0.o", 1024, b"", q_bits=56)
    assert item_device_bytes(o, v1) > 0
    sharded = SimpleNamespace(out_features=152064, in_features=3584, bounds=[(0, 76032), (76032, 152064)])
    assert item_device_bytes(sharded, v2_big) < item_device_bytes(head, v2_big)


async def test_handler_items_queue_on_the_byte_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two items whose modelled bytes do not fit together run one after the other although two slots are free;
    nothing is refused and the budget is empty afterwards."""
    from pgw.service import handler as handler_mod
    from pgw.service import maskgen_budget as mb

    active = peak = 0
    lock = threading.Lock()

    def item(kern: Any, it: Any) -> bytes:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return b"y"

    monkeypatch.setenv("PGW_SERVICE_MASKGEN_DEVICE_MB", "100")
    monkeypatch.setattr(handler_mod, "_maskgen_item", item)
    monkeypatch.setattr(mb, "item_device_bytes", lambda kern, it: 60 << 20)
    h = handler_mod.ServiceHandler.__new__(handler_mod.ServiceHandler)
    h.maskgen_concurrency = 2
    h._maskgen_slots = asyncio.Semaphore(2)
    h._maskgen_dev_slots = {}
    kern = SimpleNamespace(device=CUDA0)
    out = await asyncio.gather(*(h._maskgen_in_slot(kern, i) for i in range(3)))
    assert out == [b"y"] * 3 and peak == 1
    assert maskgen_budget_for(h, kern).used == 0  # type: ignore[union-attr]
    # 40 MB items fit two at a time
    monkeypatch.setattr(mb, "item_device_bytes", lambda kern, it: 40 << 20)
    peak = 0
    await asyncio.gather(*(h._maskgen_in_slot(kern, i) for i in range(4)))
    assert peak == 2
