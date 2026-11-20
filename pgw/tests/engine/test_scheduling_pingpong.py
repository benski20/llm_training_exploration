"""Two-cohort ping-pong in the engine batching loop: byte-identical outputs and identical per-session mask
use vs the plain loop, the > 4 sessions threshold, MaskExhaustedError isolation, round-robin, cancellation."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from typing import Any

import pytest

from pgw.core import RING_K32_A12_C27, RING_K40_A12, ConfigError, MaskExhaustedError, RingParams
from pgw.core.types import LinearId, LinearKind
from pgw.engine import (
    ChatTokenizer,
    Generator,
    InMemoryPack,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)

PROMPTS = [[256, 72, 105, 33, 10], [256, 65, 66], [256, 80, 81, 82], [256, 90], [256, 72, 105, 33, 10, 40],
           [256, 100, 101, 102, 103], [256, 7, 8], [256, 50, 51, 52]]
FIRST = LinearId(0, LinearKind.QKV)


def _params(i: int, n: int = 6) -> SamplingParams:
    # mixed greedy and seeded sampling, with logprobs (bit-identical logits => identical logprobs)
    if i % 2:
        return SamplingParams(max_tokens=n, temperature=0.8, seed=100 + i, logprobs=1, ignore_eos=True)
    return SamplingParams(max_tokens=n, temperature=0, logprobs=1, ignore_eos=True)


class _CountingClient(LocalComputeClient):
    """In-process service that records how many decode (1-row items only) requests are in flight at once,
    and can add a fixed latency per request (a stand-in for the network and the GPU)."""

    def __init__(self, pack: Any, delay_s: float = 0.0, **kw: Any):
        super().__init__(pack, **kw)
        self.delay_s = delay_s
        self._mu = threading.Lock()
        self.inflight = 0
        self.max_decode_inflight = 0
        self.items_per_request: list[int] = []

    def compute_sync(self, request: Any) -> Any:
        decode = all(it.rows == 1 for it in request.items)
        with self._mu:
            self.inflight += 1
            if decode:
                self.max_decode_inflight = max(self.max_decode_inflight, self.inflight)
            self.items_per_request.append(len(request.items))
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            return super().compute_sync(request)
        finally:
            with self._mu:
                self.inflight -= 1


def _masked(pack: InMemoryPack, ring: RingParams, **client_kw: Any) -> tuple[TransformerModel, _CountingClient]:
    client = _CountingClient(pack, **client_kw)
    return TransformerModel(pack, MaskedBackend(pack, None, client, ring, pack.fingerprint)), client


async def _run_all(gen: Generator, pack: InMemoryPack, ring: RingParams, prompts: list[list[int]],
                   params: list[SamplingParams], slots: list[int] | None = None) -> tuple[list[Any], list[Any]]:
    sessions = [LocalMaskReservation(pack, ring, slots[i] if slots else len(p) + sp.max_tokens)
                for i, (p, sp) in enumerate(zip(prompts, params, strict=True))]

    async def one(i: int) -> Any:
        try:
            return await gen.agenerate(prompts[i], params[i], session=sessions[i])
        except MaskExhaustedError as exc:
            return exc

    out = await asyncio.gather(*(one(i) for i in range(len(prompts))))
    return list(out), sessions


@pytest.fixture(scope="module")
def reference(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> list[Any]:
    plain = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    g = Generator(plain, tiny_tokenizer)
    return [g.generate(p, _params(i)) for i, p in enumerate(PROMPTS)]


@pytest.mark.parametrize("ring", [RING_K40_A12, RING_K32_A12_C27], ids=lambda r: r.name)
async def test_pingpong_byte_identical_and_same_mask_use(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                                        reference: list[Any], ring: RingParams) -> None:
    n = 6
    prompts, params = PROMPTS[:n], [_params(i) for i in range(n)]
    results: dict[bool, list[Any]] = {}
    received: dict[bool, Counter[str]] = {}
    for pp in (False, True):
        model, client = _masked(qwen_pack, ring, record=True)
        gen = Generator(model, tiny_tokenizer, pingpong=pp)
        res, sessions = await _run_all(gen, qwen_pack, ring, prompts, params)
        await gen.aclose()
        results[pp] = res
        # rows the service saw per linear (prefill items may be grouped differently: the prefill pipelining
        # splits round trips of >= 16 rows in halves, and a cohort's prefill has fewer rows than the whole step's)
        received[pp] = Counter()
        for lid, x in client.received:
            received[pp][lid] += int(x.shape[0])
        for r, s, p in zip(res, sessions, prompts, strict=True):
            assert r.masks_consumed == len(p) + 5
            assert s.remaining(FIRST) == 1  # rows taken == rows sent; the last token is never fed
            assert s._closed is False  # caller-owned reservations stay open
        if pp:
            assert gen.sched_stats["pingpong_steps"] > 0
            assert client.max_decode_inflight == 2  # the two cohorts' decode round trips overlap
            assert max(client.items_per_request) <= n
        else:
            assert gen.sched_stats["pingpong_steps"] == 0
            assert client.max_decode_inflight == 1
    plain = TransformerModel(qwen_pack, PlainBackend(qwen_pack, ring))
    ref_ring = [Generator(plain, tiny_tokenizer).generate(p, sp) for p, sp in zip(prompts, params, strict=True)]
    if ring == RING_K40_A12:
        assert [r.token_ids for r in ref_ring] == [r.token_ids for r in reference[:n]]
    for a, b, ref in zip(results[False], results[True], ref_ring, strict=True):
        assert a.token_ids == b.token_ids == ref.token_ids
        assert a.text == b.text == ref.text
        assert a.logprobs == b.logprobs == ref.logprobs
        assert a.finish_reason == b.finish_reason
    assert received[False] == received[True]


async def test_pingpong_threshold_and_flag(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    ring = RING_K40_A12
    params = [_params(i, 3) for i in range(5)]
    # 4 live sessions: never split (enabled only with more than 4)
    model, client = _masked(qwen_pack, ring)
    gen = Generator(model, tiny_tokenizer, pingpong=True)
    await _run_all(gen, qwen_pack, ring, PROMPTS[:4], params[:4])
    assert gen.sched_stats["steps"] > 0 and gen.sched_stats["pingpong_steps"] == 0
    assert client.max_decode_inflight == 1
    await gen.aclose()
    # 5 sessions: split (3 + 2)
    model, client = _masked(qwen_pack, ring)
    gen = Generator(model, tiny_tokenizer, pingpong=True)
    await _run_all(gen, qwen_pack, ring, PROMPTS[:5], params)
    assert gen.sched_stats["pingpong_steps"] > 0
    assert set(client.items_per_request) <= {2, 3, 5}  # cohorts of 3 and 2 (5: the shared prefill step)
    await gen.aclose()
    # flag off (the default): never split
    model, _ = _masked(qwen_pack, ring)
    gen = Generator(model, tiny_tokenizer)
    assert gen.pingpong is False
    await _run_all(gen, qwen_pack, ring, PROMPTS[:5], params)
    assert gen.sched_stats["pingpong_steps"] == 0
    await gen.aclose()
    with pytest.raises(ConfigError):
        Generator(model, tiny_tokenizer, pingpong=True, pingpong_min_sessions=1)


async def test_pingpong_needs_own_reservations(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                               reference: list[Any]) -> None:
    """No per-request reservation (a shared default source): no split, same outputs."""
    plain = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    gen = Generator(plain, tiny_tokenizer, pingpong=True)
    res = await asyncio.gather(*(gen.agenerate(p, _params(i)) for i, p in enumerate(PROMPTS[:6])))
    assert gen.sched_stats["pingpong_steps"] == 0
    assert [r.token_ids for r in res] == [r.token_ids for r in reference[:6]]
    await gen.aclose()


async def test_pingpong_isolates_mask_exhaustion(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                                 reference: list[Any]) -> None:
    ring = RING_K40_A12
    n = 6
    prompts, params = PROMPTS[:n], [_params(i) for i in range(n)]
    slots = [len(p) + 6 for p in prompts]
    slots[1] = len(prompts[1]) + 2  # cohort A: runs out after 3 tokens
    slots[4] = len(prompts[4]) + 3  # cohort B: runs out after 4 tokens
    model, _ = _masked(qwen_pack, ring)
    gen = Generator(model, tiny_tokenizer, pingpong=True)
    res, sessions = await _run_all(gen, qwen_pack, ring, prompts, params, slots)
    assert gen.sched_stats["pingpong_steps"] > 0
    for i, r in enumerate(res):
        if i in (1, 4):
            assert isinstance(r, MaskExhaustedError)
            assert sessions[i].remaining(FIRST) == 0  # nothing taken beyond the rows actually sent
        else:
            assert r.token_ids == reference[i].token_ids and r.logprobs == reference[i].logprobs
            assert r.masks_consumed == len(prompts[i]) + 5
    await gen.aclose()


async def test_pingpong_round_robin_over_max_batch(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                                   reference: list[Any]) -> None:
    """8 sequences, max_batch 6: steps of 6 picked round-robin (every step split 3 + 3), outputs unchanged."""
    ring = RING_K40_A12
    params = [_params(i) for i in range(8)]
    model, client = _masked(qwen_pack, ring)
    gen = Generator(model, tiny_tokenizer, max_batch=6, pingpong=True)
    res, _ = await _run_all(gen, qwen_pack, ring, PROMPTS, params)
    assert [r.token_ids for r in res] == [r.token_ids for r in reference]
    assert [r.logprobs for r in res] == [r.logprobs for r in reference]
    assert max(client.items_per_request) <= 6
    st = gen.sched_stats
    assert st["pingpong_steps"] > 0 and st["step_sequences"] <= 6 * st["steps"]
    await gen.aclose()


async def test_pingpong_with_lookup_speculation(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Prompt-lookup drafts are planned once per step (same passes and draft rows as unsplit); outputs unchanged."""
    ring = RING_K40_A12
    rep = [256, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]
    prompts = [rep, rep[1:], rep[2:], PROMPTS[0], rep[:-1], PROMPTS[2]]
    params = [SamplingParams(max_tokens=6, temperature=0, logprobs=1, ignore_eos=True) for _ in prompts]
    plain = TransformerModel(qwen_pack, PlainBackend(qwen_pack, ring))
    ref = [Generator(plain, tiny_tokenizer).generate(p, sp) for p, sp in zip(prompts, params, strict=True)]
    slots = [len(p) + 3 * 6 for p in prompts]
    out: dict[bool, list[Any]] = {}
    rows: dict[bool, int] = {}
    spec: dict[bool, dict[str, float]] = {}
    for pp in (False, True):
        model, client = _masked(qwen_pack, ring, record=True)
        gen = Generator(model, tiny_tokenizer, spec_tokens=2, lookup=3, spec_max_batch=32, pingpong=pp)
        res, _ = await _run_all(gen, qwen_pack, ring, prompts, params, slots)
        out[pp] = res
        rows[pp] = sum(int(x.shape[0]) for lid, x in client.received if lid == "0.qkv")
        spec[pp] = dict(gen.spec_stats)
        assert gen.spec_stats["spec_passes"] > 0
        if pp:
            assert gen.sched_stats["pingpong_steps"] > 0
        await gen.aclose()
    for a, b, r in zip(out[False], out[True], ref, strict=True):
        assert a.token_ids == b.token_ids == r.token_ids and a.logprobs == b.logprobs == r.logprobs
        assert a.masks_consumed == b.masks_consumed
    assert rows[False] == rows[True]
    assert spec[False] == spec[True]  # same verify passes, draft rows and acceptances (one plan per step)


async def test_pingpong_cancel_one_and_aclose(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                              reference: list[Any]) -> None:
    ring = RING_K40_A12
    model, _ = _masked(qwen_pack, ring, delay_s=0.002)
    made: list[LocalMaskReservation] = []

    def factory(n: int) -> LocalMaskReservation:
        r = LocalMaskReservation(qwen_pack, ring, n)
        made.append(r)
        return r

    gen = Generator(model, tiny_tokenizer, pingpong=True, session_factory=factory)
    params = [_params(i) for i in range(6)]

    async def full(i: int) -> list[int]:
        return [e.token_id async for e in gen.astream(PROMPTS[i], params[i])]

    async def early(i: int) -> int:
        agen = gen.astream(PROMPTS[i], SamplingParams(max_tokens=50, temperature=0, ignore_eos=True))
        first = await agen.__anext__()
        await agen.aclose()
        return first.token_id

    got = await asyncio.gather(*(full(i) for i in range(5)), early(5))
    assert got[:5] == [r.token_ids for r in reference[:5]]
    assert gen.sched_stats["pingpong_steps"] > 0
    await gen.wait_closed()
    for _ in range(100):
        if not gen._active:
            break
        await asyncio.sleep(0.01)
    await gen.wait_closed()
    assert all(r._closed for r in made)  # every owned reservation closed (unused rows burned)

    # aclose while a split step is in flight: every stream fails, nothing is left running
    streams = [asyncio.ensure_future(full(i)) for i in range(6)]
    for _ in range(2000):  # until the streams are decoding (split steps in flight)
        if any(len(s.generated) for s in gen._active):
            break
        await asyncio.sleep(0.005)
    await gen.aclose()
    outcomes = await asyncio.gather(*streams, return_exceptions=True)
    assert all(isinstance(o, BaseException) for o in outcomes)
    assert gen._task is None and not gen._active
    await gen.wait_closed()
    assert all(r._closed for r in made)
