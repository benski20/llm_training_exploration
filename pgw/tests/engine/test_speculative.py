"""Speculative decoding: a local draft model proposes tokens, the target checks them in one pass.

Acceptance samples every position with the request's own sampler, so the output must be the same as
without speculation: same tokens, text, finish reason and logprobs, for greedy and seeded sampling,
with stop strings and penalties, through the masked path too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from pgw.core import DEFAULT_RING
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

PROMPT = [256, 72, 105, 33, 10, 72, 105]

PARAMS = [
    SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True, logprobs=2),
    SamplingParams(max_tokens=24, temperature=0.9, top_p=0.95, seed=7, ignore_eos=True, logprobs=0),
    SamplingParams(max_tokens=20, temperature=0.0, repetition_penalty=1.3, frequency_penalty=0.4, ignore_eos=True),
    SamplingParams(max_tokens=30, temperature=0.0, stop=("e",)),
]


@pytest.fixture(scope="module")
def other_draft_pack(tmp_path_factory: pytest.TempPathFactory) -> InMemoryPack:
    """A different tiny model with the same tokenizer: realistic partial acceptance."""
    from tests.conftest import _save_tiny

    d = _save_tiny(tmp_path_factory.mktemp("tiny_draft"), "qwen2", seed=5)
    return InMemoryPack.from_hf(Path(d), name="tiny-draft")


def _plain(pack: Any) -> TransformerModel:
    return TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))


def _run(gen: Generator, params: SamplingParams) -> Any:
    return gen.generate(PROMPT, params)


@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded", "penalties", "stop"])
@pytest.mark.parametrize("which", ["same", "other"])
def test_speculative_output_identical(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                      tiny_tokenizer: ChatTokenizer, params: SamplingParams, which: str) -> None:
    target = _plain(qwen_pack)
    base = _run(Generator(target, tiny_tokenizer), params)
    draft = _plain(qwen_pack if which == "same" else other_draft_pack)
    spec = _run(Generator(target, tiny_tokenizer, draft=draft, spec_tokens=4), params)
    assert spec.token_ids == base.token_ids
    assert spec.text == base.text
    assert spec.finish_reason == base.finish_reason
    assert spec.completion_tokens == base.completion_tokens
    if params.logprobs is not None:
        assert spec.logprobs is not None and base.logprobs is not None
        assert spec.logprobs == pytest.approx(base.logprobs, abs=1e-4)
    passes = spec.timings_ms.get("spec_passes", 0.0)
    if which == "same" and params.temperature == 0.0 and params.repetition_penalty == 1.0 and not params.stop:
        # a perfect draft: about spec_tokens + 1 tokens per target pass
        assert passes <= base.completion_tokens / 3
        assert spec.timings_ms["spec_accept_rate"] > 0.9


def test_speculative_masked_path_identical(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                           tiny_tokenizer: ChatTokenizer) -> None:
    """Through MaskedBackend: same tokens; each verify row consumes one mask slot, never more."""
    params = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    plain, masked, _client = plain_and_masked(qwen_pack, DEFAULT_RING)
    base = Generator(plain, tiny_tokenizer).generate(PROMPT, params)
    k = 3
    slots = len(PROMPT) + params.max_tokens * (k + 1)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, slots)
    gen = Generator(masked, tiny_tokenizer, draft=_plain(other_draft_pack), spec_tokens=k)
    spec = gen.generate(PROMPT, params, session=res)
    assert spec.token_ids == base.token_ids
    assert spec.masks_consumed >= len(PROMPT) + params.max_tokens - 1
    assert spec.masks_consumed <= slots


def test_speculation_never_takes_masks_needed_to_finish(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """With a reservation sized exactly for the request, speculation only uses spare rows: it never fails
    the request and never exceeds the reservation (accepted drafts are rows that were needed anyway)."""
    params = SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True)
    plain, masked, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    base = Generator(plain, tiny_tokenizer).generate(PROMPT, params)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, len(PROMPT) + params.max_tokens)
    gen = Generator(masked, tiny_tokenizer, draft=_plain(qwen_pack), spec_tokens=4)
    out = gen.generate(PROMPT, params, session=res)
    assert out.token_ids == base.token_ids
    assert out.masks_consumed <= len(PROMPT) + params.max_tokens


def test_speculative_async_batched_streams(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                           tiny_tokenizer: ChatTokenizer) -> None:
    """Concurrent streams through the batching loop emit the same tokens, one event per token."""
    target = _plain(qwen_pack)
    prompts = [PROMPT, [256, 65, 66, 67], [256, 120, 121]]
    params = SamplingParams(max_tokens=18, temperature=0.0, ignore_eos=True)
    base = [Generator(target, tiny_tokenizer).generate(p, params).token_ids for p in prompts]
    gen = Generator(target, tiny_tokenizer, draft=_plain(other_draft_pack), spec_tokens=4)

    async def one(p: list[int]) -> list[int]:
        return [ev.token_id async for ev in gen.astream(p, params)]

    async def main() -> list[list[int]]:
        try:
            return list(await asyncio.gather(*(one(p) for p in prompts)))
        finally:
            await gen.aclose()

    assert asyncio.run(main()) == base


def test_draft_vocab_must_match(qwen_pack: InMemoryPack, llama_pack: InMemoryPack,
                                tiny_tokenizer: ChatTokenizer) -> None:
    from pgw.core import ConfigError

    t, d = _plain(qwen_pack), _plain(llama_pack)
    if getattr(t.arch, "vocab_size", None) == getattr(d.arch, "vocab_size", None):
        pytest.skip("tiny test models share a vocabulary size")
    with pytest.raises(ConfigError):
        Generator(t, tiny_tokenizer, draft=d)
