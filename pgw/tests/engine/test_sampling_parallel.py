"""Sampling at many concurrent answers, exactly as before.

* ``pgw_fast.torch_multinomial1`` (used by ``Sampler`` for ``temperature > 0``) returns the index
  ``torch.multinomial(probs, 1, generator=g)`` returns and leaves the generator in the same state, byte for byte;
  inputs torch rejects are left to torch (the state is untouched).
* ``PGW_SAMPLE_THREADS`` (the SDK sets ``auto``): the answers of one step sample on a thread pool; tokens, logprobs,
  top alternatives and every sampler's generator state equal the sequential loop's.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, native
from pgw.engine import ChatTokenizer, Generator, InMemoryPack, PlainBackend, SamplingParams, TransformerModel
from pgw.engine import sampling as smp
from pgw.engine.sampling import Sampler

needs_native = pytest.mark.skipif(not native.native_available("torch_multinomial1"),
                                  reason="pgw_fast without torch_multinomial1")


@needs_native
@pytest.mark.parametrize("n", [1, 2, 7, 300, 4096, 151936])
def test_native_multinomial_equals_torch(n: int) -> None:
    for case in range(25):
        g = torch.Generator().manual_seed(case * 7919 + n)
        for _ in range(case % 5):  # any position of the generator, including right before a regeneration
            torch.rand(case * 37 % 700 + 1, generator=g)
        logits = torch.randn(n, generator=torch.Generator().manual_seed(case)) * (1 + case % 6) * 2
        if case % 4 == 0 and n > 2:
            logits[::3] = float("-inf")  # zero probabilities
        if case % 7 == 0 and n > 3:
            logits[1] = logits[2]  # ties
        probs = torch.softmax(logits / (0.2 + (case % 5) * 0.4), dim=-1)
        g2 = torch.Generator()
        g2.set_state(g.get_state())
        ref = int(torch.multinomial(probs, 1, generator=g))
        st = g2.get_state()
        got = native.fast.torch_multinomial1(st.numpy(), probs.numpy())
        g2.set_state(st)
        assert got == ref
        assert torch.equal(g.get_state(), g2.get_state())
        assert torch.equal(torch.rand(5, generator=g), torch.rand(5, generator=g2))


@needs_native
def test_native_multinomial_fresh_generator_and_rejects() -> None:
    probs = torch.softmax(torch.randn(1000), dim=-1)
    g, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)  # left == 1: regenerate first
    st = g2.get_state()
    assert native.fast.torch_multinomial1(st.numpy(), probs.numpy()) == int(torch.multinomial(probs, 1, generator=g))
    g2.set_state(st)
    assert torch.equal(g.get_state(), g2.get_state())
    for bad in (torch.zeros(5), torch.tensor([0.5, -0.1]), torch.tensor([0.5, float("nan")]),
                torch.tensor([0.5, float("inf")]), torch.zeros(0)):
        st = g2.get_state()
        before = st.clone()
        assert native.fast.torch_multinomial1(st.numpy(), bad.numpy()) is None
        assert torch.equal(st, before)  # untouched: the caller lets torch raise its own error
    st = g2.get_state()[:-8].clone()
    assert native.fast.torch_multinomial1(st.numpy(), probs.numpy()) is None  # a state layout it does not know


@needs_native
def test_sampler_same_with_and_without_native(monkeypatch: pytest.MonkeyPatch) -> None:
    variants = [SamplingParams(temperature=0.8, seed=3, logprobs=3),
                SamplingParams(temperature=1.3, top_p=0.9, seed=4, logprobs=0),
                SamplingParams(temperature=0.7, top_k=40, min_p=0.01, seed=5),
                SamplingParams(temperature=1.0, repetition_penalty=1.2, presence_penalty=0.5, seed=6)]
    steps = [torch.randn(151936, generator=torch.Generator().manual_seed(i)) * 3 for i in range(12)]

    def run() -> list[Any]:
        out = []
        for p in variants:
            s = Sampler(p, prompt_ids=[1, 2, 3])
            gen: list[int] = []
            for lg in steps:
                tok, lp = s.sample(lg, gen)
                gen.append(tok)
                out.append((tok, lp, s.last_top_logprobs, s.propose(lg, gen)))
            out.append(s._gen.get_state().tolist())
        return out

    assert smp._native_multinomial_ok()
    with_native = run()
    monkeypatch.setattr(smp, "_MN_OK", False)
    assert run() == with_native


@pytest.fixture(scope="module")
def gen_model(qwen_pack: InMemoryPack) -> TransformerModel:
    return TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))


def _requests() -> list[tuple[list[int], SamplingParams]]:
    out = []
    for i in range(6):
        p = SamplingParams(max_tokens=8, temperature=[0.0, 0.7, 1.2][i % 3], top_p=[1.0, 0.9][i % 2], seed=100 + i,
                           logprobs=2, ignore_eos=True)
        out.append(([256, 70 + i, 71, 72][: 2 + i % 3], p))
    return out


async def _concurrent(gen: Generator) -> list[Any]:
    async def one(pr: list[int], p: SamplingParams) -> list[Any]:
        return [(e.token_id, e.logprob, e.top_logprobs) async for e in gen.astream(pr, p)]

    return list(await asyncio.gather(*(one(pr, p) for pr, p in _requests())))


@pytest.mark.parametrize("threads", ["4", "auto"])
def test_parallel_sampling_equals_sequential(gen_model: TransformerModel, tiny_tokenizer: ChatTokenizer,
                                             monkeypatch: pytest.MonkeyPatch, threads: str) -> None:
    monkeypatch.setenv("PGW_SAMPLE_THREADS", "0")
    seq = Generator(gen_model, tiny_tokenizer)
    ref = asyncio.run(_concurrent(seq))
    alone = [[(e.token_id, e.logprob, e.top_logprobs) for e in Generator(gen_model, tiny_tokenizer).stream(pr, p)]
             for pr, p in _requests()]
    assert ref == alone  # batching itself is exact
    monkeypatch.setenv("PGW_SAMPLE_THREADS", threads)
    par = Generator(gen_model, tiny_tokenizer)
    par.sample_parallel_min_vocab = 0  # the tiny model's vocabulary is below the production threshold
    calls: list[int] = []
    real = par._sample_parallel

    def spy(batch: Any, logits: Any) -> Any:
        got = real(batch, logits)
        calls.append(0 if got is None else len(got))
        return got

    par._sample_parallel = spy  # type: ignore[method-assign]
    assert asyncio.run(_concurrent(par)) == ref
    assert max(calls) >= 2  # steps of several answers sampled on the pool
    # the blocking API (one pass of several answers, generate_many) takes the same path
    many = Generator(gen_model, tiny_tokenizer, max_batch=6)
    many.sample_parallel_min_vocab = 0
    reqs = _requests()
    got = many.generate_many([pr for pr, _ in reqs], [p for _, p in reqs])
    assert [r.token_ids for r in got] == [[t for t, _, _ in a] for a in alone]


def test_parallel_sampling_default_off(gen_model: TransformerModel, tiny_tokenizer: ChatTokenizer,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_SAMPLE_THREADS", raising=False)
    g = Generator(gen_model, tiny_tokenizer)
    assert g._sample_threads() == 0
