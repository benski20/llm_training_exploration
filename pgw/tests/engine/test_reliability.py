"""Reliability review regressions: the engine never blocks the event loop."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import torch

from pgw.core import RING_K40_A12
from pgw.engine import (
    ChatTokenizer,
    Generator,
    InMemoryPack,
    LocalMaskReservation,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from tests.engine.conftest import plain_and_masked

PROMPT = [256, 72, 105, 33, 10]


class _LagProbe:
    """Measures the worst event-loop scheduling delay while active."""

    def __init__(self, period: float = 0.01):
        self.period = period
        self.worst = 0.0
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            await asyncio.sleep(self.period)
            self.worst = max(self.worst, loop.time() - t0 - self.period)

    async def __aenter__(self) -> _LagProbe:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass


class _SlowCloseReservation(LocalMaskReservation):
    """Reservation whose close is slow (store lock + fsync + segment erase in production)."""

    close_threads: list[int]

    def close(self) -> None:
        self.close_threads.append(threading.get_ident())
        time.sleep(0.4)
        super().close()


async def test_owned_reservation_closed_off_the_event_loop(qwen_pack: InMemoryPack,
                                                           tiny_tokenizer: ChatTokenizer) -> None:
    _, masked, _ = plain_and_masked(qwen_pack, RING_K40_A12)
    gen = Generator(masked, tiny_tokenizer)
    p = SamplingParams(max_tokens=3, temperature=0, ignore_eos=True)
    sessions = []
    for _ in range(2):
        s = _SlowCloseReservation(qwen_pack, RING_K40_A12, len(PROMPT) + 3)
        s.close_threads = []
        sessions.append(s)
    loop_thread = threading.get_ident()
    async with _LagProbe() as probe:
        # streamed + cancelled early (closed by the batching loop) and run to completion
        agen = gen.astream(PROMPT, p, session=sessions[0], own_session=True)
        await agen.__anext__()
        await agen.aclose()
        await gen.agenerate(PROMPT, p, session=sessions[1], own_session=True)
        await gen.aclose()  # waits for outstanding closes
    for s in sessions:
        assert s.close_threads and loop_thread not in s.close_threads
    assert probe.worst < 0.3, f"event loop stalled for {probe.worst:.3f}s"


async def test_plaintext_forward_and_sampling_run_off_the_event_loop(qwen_pack: InMemoryPack,
                                                                     tiny_tokenizer: ChatTokenizer) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    threads: set[int] = set()
    real_attention = model._attention

    def slow_attention(*a: Any, **kw: Any) -> torch.Tensor:
        threads.add(threading.get_ident())
        time.sleep(0.15)  # stands in for attention over a long KV cache / a big LM head
        return real_attention(*a, **kw)

    model._attention = slow_attention  # type: ignore[method-assign]
    loop_thread = threading.get_ident()
    for batching in (True, False):
        gen = Generator(model, tiny_tokenizer, batching=batching)
        sample_threads: set[int] = set()
        real_consume = gen._consume

        def consume(st: Any, logits: torch.Tensor, _rc: Any = real_consume, _t: set[int] = sample_threads) -> Any:
            _t.add(threading.get_ident())
            return _rc(st, logits)

        gen._consume = consume  # type: ignore[method-assign]
        p = SamplingParams(max_tokens=3, temperature=0, ignore_eos=True)
        async with _LagProbe() as probe:
            res = await asyncio.gather(*(gen.agenerate(PROMPT, p) for _ in range(3)))
        await gen.aclose()
        assert all(r.completion_tokens == 3 for r in res)
        assert threads and loop_thread not in threads
        assert sample_threads and loop_thread not in sample_threads
        # each forward pass sleeps >= 0.15 s per layer in attention; the loop itself stays responsive
        assert probe.worst < 0.1, f"event loop stalled for {probe.worst:.3f}s (batching={batching})"
    model.close()
