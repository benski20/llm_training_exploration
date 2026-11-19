"""Speculative decoding: a local draft model proposes tokens, the target checks them in one pass.

Acceptance samples every position with the request's own sampler, and verify rows are computed as plain
decoding would compute them (serial attention, 1-row logits), so the output is bit-identical to decoding
without speculation: same tokens, text, finish reason and logprobs, for greedy and seeded sampling, with
stop strings and penalties, through the masked path too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import torch

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
        assert spec.logprobs == base.logprobs  # bit-identical, not just close
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


class _OddVocabDraft:
    """Wraps a draft: reports a smaller table (ids fed to it must be clamped) and, every other call,
    proposes an id the target does not have (that draft must end there, never reach the target)."""

    def __init__(self, inner: TransformerModel, vocab: int, target_vocab: int):
        self.inner = inner
        self.arch = type("A", (), {"vocab_size": vocab})()
        self.target_vocab = target_vocab
        self.calls = 0
        self.max_fed = -1

    def new_cache(self, n: int) -> Any:
        return self.inner.new_cache(n)

    def forward_batch(self, inputs: list[Any]) -> list[torch.Tensor]:
        self.calls += 1
        self.max_fed = max(self.max_fed, *(max(i.input_ids) for i in inputs))
        outs = self.inner.forward_batch(inputs)
        if self.calls % 2 == 0:
            outs = [torch.cat([o, torch.full((3,), 1e9)]) for o in outs]  # argmax = target_vocab + k
        return outs


def test_draft_with_another_vocab_size(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                       tiny_tokenizer: ChatTokenizer) -> None:
    """Real draft/target pairs pad their embedding tables differently (Qwen2.5-0.5B vs 7B): allowed,
    and the output stays identical."""
    target = _plain(qwen_pack)
    tv = int(target.arch.vocab_size)
    params = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    small = tv - 40
    draft = _OddVocabDraft(_plain(other_draft_pack), small, tv)
    spec = Generator(target, tiny_tokenizer, draft=draft, spec_tokens=4).generate(PROMPT, params)  # type: ignore[arg-type]
    assert spec.token_ids == base.token_ids
    assert draft.calls > 0 and draft.max_fed < small


def test_int8_draft_backend_keeps_output_identical(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                                   tiny_tokenizer: ChatTokenizer) -> None:
    """The gateway's CPU draft (int8 weights, dynamic activations, int8 lm_head) is approximate by design;
    the output is still bit-identical because the masked target decides every token."""
    from pgw.engine.backends import Int8DraftBackend, int8_lm_head

    if not Int8DraftBackend.available():
        pytest.skip("no quantized CPU engine in this torch build")
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=20, temperature=0.8, seed=3, ignore_eos=True, logprobs=1)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    for pack in (qwen_pack, other_draft_pack):
        draft = TransformerModel(pack, Int8DraftBackend(pack, DEFAULT_RING))
        draft.lm_head_fn = int8_lm_head(draft.lm_head)
        assert draft.lm_head_fn is not None
        spec = Generator(target, tiny_tokenizer, draft=draft, spec_tokens=4).generate(PROMPT, params)
        assert spec.token_ids == base.token_ids and spec.logprobs == base.logprobs


def test_int8_draft_backend_is_accurate_with_activation_outliers(qwen_pack: InMemoryPack) -> None:
    """Regression: per-tensor dynamic activation quantization lost ~4% on outlier-heavy activations
    (real Qwen drafts were then almost never accepted). Weight-only int8 must track float closely."""
    from pgw.core.types import LinearId
    from pgw.engine import FloatBackend, LinearCall
    from pgw.engine.backends import Int8DraftBackend

    if not Int8DraftBackend.available():
        pytest.skip("no int8 weight-only CPU kernel in this torch build")
    lid = LinearId(0, "gate_up")
    n = qwen_pack.load_linear(lid).w8.shape[1]
    x = torch.randn(3, n, generator=torch.Generator().manual_seed(0))
    x[:, 1] = 300.0  # massive-activation outlier channel
    ref = FloatBackend(qwen_pack, DEFAULT_RING).forward(LinearCall(lid, x))
    got = Int8DraftBackend(qwen_pack, DEFAULT_RING).forward(LinearCall(lid, x))
    assert float((got - ref).abs().max() / ref.abs().max()) < 1e-4


def test_lookup_drafts_unit() -> None:
    from pgw.engine.generate import _lookup_drafts

    seq = [1, 2, 3, 4, 5, 9, 2, 3]
    assert _lookup_drafts(seq, 3, 3) == [4, 5, 9]  # "2 3" matched earlier -> what followed
    assert _lookup_drafts([7, 8, 9], 2, 3) == []  # no repeat
    assert _lookup_drafts([5, 1, 5, 2, 5], 2, 3) == [2, 5]  # latest earlier match of "5" wins
    assert _lookup_drafts([1, 2, 1, 2], 4, 3) == [1, 2]  # stops at the end of the context
    assert _lookup_drafts([4], 2, 3) == []


@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded", "penalties", "stop"])
@pytest.mark.parametrize("with_draft", [False, True])
def test_prompt_lookup_output_identical(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                        tiny_tokenizer: ChatTokenizer, params: SamplingParams,
                                        with_draft: bool) -> None:
    target = _plain(qwen_pack)
    prompt = PROMPT * 4  # repetitive context: lookup has something to propose
    base = Generator(target, tiny_tokenizer).generate(prompt, params)
    draft = _plain(other_draft_pack) if with_draft else None
    spec = Generator(target, tiny_tokenizer, draft=draft, spec_tokens=4, lookup=3).generate(prompt, params)
    assert spec.token_ids == base.token_ids and spec.text == base.text
    assert spec.finish_reason == base.finish_reason and spec.logprobs == base.logprobs


def test_prompt_lookup_cuts_passes_on_repetitive_output(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Greedy decoding of a tiny random model quickly loops; lookup then proposes the loop."""
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=48, temperature=0.0, ignore_eos=True)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    spec = Generator(target, tiny_tokenizer, spec_tokens=4, lookup=3).generate(PROMPT, params)
    assert spec.token_ids == base.token_ids
    assert spec.timings_ms["spec_passes"] < len(base.token_ids)
