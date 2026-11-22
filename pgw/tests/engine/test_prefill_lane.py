"""Prefill lane (chat-speed WP-A7): prompt prefills run concurrently with the decode steps of running streams.

A long prompt used to join the decode step: every running stream then waited for its whole prefill pass.
With ``prefill_lane=True`` the prefill pass runs in a lane of its own (shortest first with aging); each
sequence's tokens are unchanged because its logits do not depend on what shares a pass.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pgw.core import DEFAULT_RING
from pgw.engine import (
    ChatTokenizer,
    Generator,
    InMemoryPack,
    LinearCall,
    LocalMaskReservation,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from tests.engine.conftest import plain_and_masked, save_tiny_variant

GREEDY = SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True, logprobs=1)


@pytest.fixture(scope="module")
def long_ctx_pack(tmp_path_factory: pytest.TempPathFactory) -> InMemoryPack:
    """Tiny Qwen2 with a 1024-token context (room for a 500-token prompt)."""
    d = save_tiny_variant(tmp_path_factory.mktemp("tiny_long"), "qwen2", seed=3, max_position_embeddings=1024)
    return InMemoryPack.from_hf(Path(d), name="tiny-long")


class _SlowBackend(PlainBackend):
    """Plain integer product behind a simulated round trip: ``per_call + per_row * rows`` seconds."""

    per_call = 0.002
    per_row = 0.0004

    async def aforward_many(self, calls: Sequence[LinearCall]) -> list[Any]:
        rows = sum(int(c.x.shape[0]) for c in calls)
        await asyncio.sleep(self.per_call + self.per_row * rows)
        return self.forward_many(calls)


class _Session:
    """A distinct (dummy) reservation per request: the plain backend takes no masks."""


def _prompt(n: int, salt: int = 0) -> list[int]:
    return [256] + [32 + (i * 7 + salt) % 90 for i in range(n - 1)]


def test_lane_output_identical(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    prompts = [_prompt(40, 1), _prompt(5, 2), _prompt(90, 3), _prompt(20, 4), _prompt(60, 5)]
    params = [GREEDY, SamplingParams(max_tokens=10, temperature=0.8, seed=4, ignore_eos=True, logprobs=0)] * 3
    base = [Generator(model, tiny_tokenizer).generate(p, sp) for p, sp in zip(prompts, params, strict=False)]
    gen = Generator(model, tiny_tokenizer, prefill_lane=True, prefill_lane_min_tokens=16, max_batch=3)

    async def one(i: int) -> Any:
        await asyncio.sleep(0.002 * i)  # staggered arrivals: prompts join while others decode
        return await gen.agenerate(prompts[i], params[i], session=_Session())

    async def main() -> list[Any]:
        try:
            return list(await asyncio.gather(*(one(i) for i in range(len(prompts)))))
        finally:
            await gen.aclose()

    got = asyncio.run(main())
    for g, b in zip(got, base, strict=True):
        assert g.token_ids == b.token_ids and g.logprobs == b.logprobs and g.finish_reason == b.finish_reason
    assert gen.sched_stats["lane_sequences"] == 4  # the 5-token prompt prefilled in a decode step


def test_lane_masked_identical(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Through MaskedBackend with one reservation per request: prefill passes concurrent with decode."""
    plain, masked, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    prompts = [_prompt(30, 1), _prompt(4, 2), _prompt(50, 3)]
    base = [Generator(plain, tiny_tokenizer).generate(p, GREEDY) for p in prompts]
    gen = Generator(masked, tiny_tokenizer, prefill_lane=True, prefill_lane_min_tokens=16)

    async def one(i: int) -> Any:
        await asyncio.sleep(0.01 * i)
        res = LocalMaskReservation(qwen_pack, DEFAULT_RING, len(prompts[i]) + GREEDY.max_tokens)
        return await gen.agenerate(prompts[i], GREEDY, session=res, own_session=True)

    async def main() -> list[Any]:
        try:
            return list(await asyncio.gather(*(one(i) for i in range(len(prompts)))))
        finally:
            await gen.aclose()

    got = asyncio.run(main())
    for g, b in zip(got, base, strict=True):
        assert g.token_ids == b.token_ids and g.logprobs == b.logprobs
        assert g.masks_consumed == g.prompt_tokens + g.completion_tokens - 1
    assert gen.sched_stats["lane_sequences"] == 2


def _gap_run(pack: InMemoryPack, tok: ChatTokenizer, lane: bool) -> tuple[float, list[int], list[int]]:
    """Largest inter-token gap of a running stream while a 500-token prompt is admitted."""
    model = TransformerModel(pack, _SlowBackend(pack, DEFAULT_RING))
    gen = Generator(model, tok, prefill_lane=lane, prefill_lane_min_tokens=16)
    running = SamplingParams(max_tokens=40, temperature=0.0, ignore_eos=True)
    stamps: list[float] = []
    tokens: list[int] = []

    async def stream() -> None:
        async for ev in gen.astream(_prompt(8), running, session=_Session()):
            stamps.append(time.perf_counter())
            tokens.append(ev.token_id)

    async def main() -> list[int]:
        try:
            task = asyncio.ensure_future(stream())
            while len(stamps) < 4:
                await asyncio.sleep(0.001)
            long = await gen.agenerate(_prompt(500, 9), SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True),
                                       session=_Session())
            await task
            return long.token_ids
        finally:
            await gen.aclose()

    long_tokens = asyncio.run(main())
    gaps = [b - a for a, b in itertools.pairwise(stamps)]
    return max(gaps), tokens, long_tokens


def test_long_admission_does_not_freeze_running_stream(long_ctx_pack: InMemoryPack,
                                                       tiny_tokenizer: ChatTokenizer) -> None:
    """A 500-token prompt (simulated prefill pass ~1.6 s) admitted during a stream: with the lane the
    stream's largest inter-token gap stays near one decode step; without it the stream waits for the
    whole prefill. Tokens are the same either way."""
    gap_lane, toks_lane, long_lane = _gap_run(long_ctx_pack, tiny_tokenizer, lane=True)
    gap_old, toks_old, long_old = _gap_run(long_ctx_pack, tiny_tokenizer, lane=False)
    assert toks_lane == toks_old and long_lane == long_old
    prefill_s = 8 * (_SlowBackend.per_call + _SlowBackend.per_row * 500)  # 2 layers x 4 linears
    assert gap_old > 0.8 * prefill_s
    assert gap_lane < 0.35 * prefill_s


class _FakeState:
    def __init__(self, n: int, t: float, session: object | None = None):
        self.prompt = [1] * n
        self.pending = self.prompt  # prompt-pass rows (a reused conversation prefix would shorten them)
        self.t_lane = t
        self.session = session if session is not None else object()
        self.cancelled = False
        self.done = False


def test_pick_prefill_shortest_first_with_aging(qwen_pack: InMemoryPack) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    gen: Any = Generator(model, None, prefill_lane=True, prefill_lane_rows=100, prefill_aging=100.0)
    now = time.perf_counter()
    a, b, c = _FakeState(90, now), _FakeState(30, now), _FakeState(50, now)
    gen._lane_wait = [a, b, c]
    assert gen._pick_prefill() == [b, c]  # shortest first, up to 100 rows
    assert gen._lane_wait == [a]
    # aging: a 400-token prompt that waited 4 s outranks a fresh 20-token prompt
    old, fresh = _FakeState(400, now - 4.0), _FakeState(20, now)
    gen._lane_wait = [fresh, old]
    assert gen._pick_prefill() == [old]  # alone: it fills the pass budget
    assert gen._pick_prefill() == [fresh]
    # a prompt sharing a reservation with a running sequence prefills in the decode step instead
    shared = object()
    running = _FakeState(3, now, shared)
    gen._active = [running]
    clash = _FakeState(40, now, shared)
    gen._lane_wait = [clash]
    assert gen._pick_prefill() == []
    assert clash in gen._active and not gen._lane_wait


def test_lane_with_draft_model_identical(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Speculation (variable-row and fixed-row) with the lane: the draft's prompt prefill runs in the lane
    while decode steps plan drafts; the draft model is never used by both at once."""
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    draft = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    prompts = [_prompt(30, 1), _prompt(70, 2), _prompt(45, 3)]
    base = [Generator(model, tiny_tokenizer).generate(p, GREEDY).token_ids for p in prompts]
    for kw in ({"spec_tokens": 3}, {"spec_rows": 3, "spec_nonstrict_for_tests": True},
               {"spec_rows": 3, "spec_strict": True}):
        gen = Generator(model, tiny_tokenizer, draft=draft, prefill_lane=True, spec_max_batch=4, **kw)  # type: ignore[arg-type]

        async def one(i: int, g: Generator = gen) -> list[int]:
            await asyncio.sleep(0.003 * i)
            return [ev.token_id async for ev in g.astream(prompts[i], GREEDY, session=_Session())]

        async def main(g: Generator = gen) -> list[list[int]]:
            try:
                out = list(await asyncio.gather(*(one(i) for i in range(len(prompts)))))
                await g.wait_drained()
                return out
            finally:
                await g.aclose()

        assert asyncio.run(main()) == base, kw


def test_lane_holds_back_decode_sharing_its_reservation(qwen_pack: InMemoryPack,
                                                        tiny_tokenizer: ChatTokenizer) -> None:
    """A short prompt (not lane-eligible) that shares a reservation with a prompt in a lane pass waits for
    that pass: one reservation is never used by two passes at once. Outputs unchanged."""
    model = TransformerModel(qwen_pack, _SlowBackend(qwen_pack, DEFAULT_RING))
    prompts = [_prompt(90, 1), _prompt(5, 2)]
    base = [Generator(model, tiny_tokenizer).generate(p, GREEDY).token_ids for p in prompts]
    gen = Generator(model, tiny_tokenizer, prefill_lane=True, prefill_lane_min_tokens=16)
    shared = _Session()
    inflight: dict[int, int] = {}
    overlaps: list[int] = []
    orig = model.aforward_batch

    async def spy(inputs: Any) -> Any:
        ids = {id(si.session) for si in inputs if si.session is not None}
        overlaps.extend(i for i in ids if inflight.get(i, 0))
        for i in ids:
            inflight[i] = inflight.get(i, 0) + 1
        try:
            return await orig(inputs)
        finally:
            for i in ids:
                inflight[i] -= 1

    model.aforward_batch = spy  # type: ignore[method-assign]

    async def one(i: int) -> list[int]:
        await asyncio.sleep(0.004 * i)  # the short prompt arrives during the long prompt's lane pass
        return [ev.token_id async for ev in gen.astream(prompts[i], GREEDY, session=shared)]

    async def main() -> list[list[int]]:
        try:
            return list(await asyncio.gather(*(one(i) for i in range(len(prompts)))))
        finally:
            await gen.aclose()

    assert asyncio.run(main()) == base
    assert gen.sched_stats["lane_steps"] >= 1
    assert not overlaps
