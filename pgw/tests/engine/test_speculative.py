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
    gen = Generator(target, tiny_tokenizer, draft=_plain(other_draft_pack), spec_tokens=4, spec_max_batch=8)

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

    seq = [1, 2, 3, 4, 5, 9, 1, 2, 3]
    assert _lookup_drafts(seq, 3, 3) == [4, 5, 9]  # "1 2 3" matched earlier -> what followed
    assert _lookup_drafts([7, 8, 9], 2, 3) == []  # no repeat
    # 1- and 2-gram coincidences are not matches any more (default min_n=3) ...
    assert _lookup_drafts([5, 1, 5, 2, 5], 2, 3) == []
    assert _lookup_drafts([1, 2, 3, 4, 5, 9, 2, 3], 3, 3) == []
    # ... unless explicitly allowed
    assert _lookup_drafts([5, 1, 5, 2, 5], 2, 3, min_n=1) == [2, 5]  # latest earlier match of "5" wins
    assert _lookup_drafts([1, 2, 1, 2], 4, 3, min_n=1) == [1, 2]  # short: stops at the end of the context
    assert _lookup_drafts([4], 2, 3) == []
    # a match followed by a full k tokens wins over a later one cut short by the end of the context
    assert _lookup_drafts([4, 4, 4, 4, 4, 4], 3, 3) == [4, 4, 4]
    assert _lookup_drafts([8, 1, 2, 3, 7, 7, 1, 2, 3, 6, 1, 2, 3], 4, 3) == [6, 1, 2, 3]


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


def test_sampler_rng_advance_does_not_depend_on_logits() -> None:
    """Coupled drafting relies on this: after one draw the RNG state is the same whatever the logits."""
    from pgw.engine.sampling import Sampler

    params = SamplingParams(max_tokens=4, temperature=0.8, top_p=0.9, seed=11)
    a, b = Sampler(params), Sampler(params)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        a.propose(torch.randn(300, generator=g) * 3, [])
        b.propose(torch.randn(300, generator=g), [])
        assert torch.equal(a._gen.get_state(), b._gen.get_state())
    f = a.fork()
    assert torch.equal(f._gen.get_state(), a._gen.get_state())
    lg = torch.randn(300, generator=g)
    assert f.propose(lg.clone(), [1, 2]) == a.sample(lg.clone(), [1, 2])[0]  # a fork draws what the target draws


@pytest.mark.parametrize("params", PARAMS[1:2], ids=["seeded"])
def test_coupled_drafting_accepts_sampled_tokens(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                                 params: SamplingParams) -> None:
    """With the target itself as draft, coupled proposals at temperature 0.9 are (almost) always accepted."""
    target = _plain(qwen_pack)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    spec = Generator(target, tiny_tokenizer, draft=_plain(qwen_pack), spec_tokens=4).generate(PROMPT, params)
    assert spec.token_ids == base.token_ids and spec.logprobs == base.logprobs
    assert spec.timings_ms["spec_accept_rate"] > 0.9


@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded", "penalties", "stop"])
@pytest.mark.parametrize("tau", [0.0, 0.3, 0.99])
def test_adaptive_draft_length_identical_and_two_sizes(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                                       tiny_tokenizer: ChatTokenizer, params: SamplingParams,
                                                       tau: float) -> None:
    target = _plain(qwen_pack)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    gen = Generator(target, tiny_tokenizer, draft=_plain(other_draft_pack), spec_tokens=8, spec_min_tokens=3,
                    draft_confidence=tau)
    sizes: list[int] = []
    orig = gen._inputs

    def spy(batch: Any) -> Any:
        sizes.extend(len(st.drafts) for st in batch)
        return orig(batch)

    gen._inputs = spy  # type: ignore[method-assign]
    spec = gen.generate(PROMPT, params)
    assert spec.token_ids == base.token_ids and spec.logprobs == base.logprobs
    for k in sizes:  # two fixed sizes (or none), also when the budget runs out
        assert k in (0, 3, 8)
    if tau == 0.0:
        assert 8 in sizes  # never unsure: always the long draft when the budget allows


# ---- privacy of speculation: fixed draft sizes, retune, metrics -----------------------------------------


def _pass_rows(gen: Generator) -> list[list[int]]:
    """Spy: rows of every target pass, per sequence of the step."""
    rows: list[list[int]] = []
    orig = gen._inputs

    def spy(batch: Any) -> Any:
        out = orig(batch)
        rows.append([len(si.input_ids) for si in out])
        return out

    gen._inputs = spy  # type: ignore[method-assign]
    return rows


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded", "penalties", "stop"])
def test_fixed_k_output_identical(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                  tiny_tokenizer: ChatTokenizer, params: SamplingParams, k: int) -> None:
    target = _plain(qwen_pack)
    base = _run(Generator(target, tiny_tokenizer), params)
    for draft_pack in (qwen_pack, other_draft_pack):
        gen = Generator(target, tiny_tokenizer, draft=_plain(draft_pack), spec_tokens=k)
        rows = _pass_rows(gen)
        spec = _run(gen, params)
        assert spec.token_ids == base.token_ids and spec.text == base.text
        assert spec.finish_reason == base.finish_reason and spec.logprobs == base.logprobs
        assert all(r[0] in (1, 1 + k) for r in rows[1:])


def test_fixed_k_masked_identical(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                  tiny_tokenizer: ChatTokenizer) -> None:
    """k=2 through MaskedBackend with the default row budget (2.0): same tokens, masks == rows sent."""
    params = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True, logprobs=1)
    plain, masked, _client = plain_and_masked(qwen_pack, DEFAULT_RING)
    base = Generator(plain, tiny_tokenizer).generate(PROMPT, params)
    slots = len(PROMPT) + 2 * params.max_tokens
    gen = Generator(masked, tiny_tokenizer, draft=_plain(other_draft_pack), spec_tokens=2)
    rows = _pass_rows(gen)
    spec = gen.generate(PROMPT, params, session=LocalMaskReservation(qwen_pack, DEFAULT_RING, slots))
    assert spec.token_ids == base.token_ids and spec.logprobs == base.logprobs
    assert spec.masks_consumed == sum(r[0] for r in rows) <= slots
    assert all(r[0] in (1, 3) for r in rows[1:])


@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded", "penalties", "stop"])
@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize("k", [2, 3])
def test_lookup_rows_per_pass_fixed(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                    tiny_tokenizer: ChatTokenizer, params: SamplingParams, with_draft: bool,
                                    k: int) -> None:
    """With prompt lookup on, every decode pass carries 1 or 1 + k rows: a short match is completed by the
    draft model (or dropped without one), never sent at its own length."""
    target = _plain(qwen_pack)
    prompt = PROMPT * 4
    base = Generator(target, tiny_tokenizer).generate(prompt, params)
    draft = _plain(other_draft_pack) if with_draft else None
    gen = Generator(target, tiny_tokenizer, draft=draft, spec_tokens=k, lookup=3)
    rows = _pass_rows(gen)
    spec = gen.generate(prompt, params)
    assert spec.token_ids == base.token_ids and spec.text == base.text
    assert spec.finish_reason == base.finish_reason and spec.logprobs == base.logprobs
    assert rows[0][0] in (len(prompt), len(prompt) + k)  # prefill (lookup may speculate on the prompt)
    assert all(r[0] in (1, 1 + k) for r in rows[1:])
    assert rows[0][0] == len(prompt) + k or any(r[0] == 1 + k for r in rows)  # lookup did propose


def test_lookup_pads_short_match_with_draft(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                            tiny_tokenizer: ChatTokenizer) -> None:
    """A match cut short by the end of the context: completed by the draft (k tokens), else no drafts."""
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    tail = [40, 41, 42]
    prompt = [256, 50, *tail, 60, *tail]  # the earlier "40 41 42" is followed by only 4 tokens
    k = 6
    lookup_only = Generator(target, tiny_tokenizer, spec_tokens=k, lookup=3)
    st = lookup_only._new_state(prompt, params, None)
    lookup_only._plan_drafts([st])
    assert st.drafts == []  # [60, 40, 41, 42] is shorter than k: dropped, not sent with 4 rows

    gen = Generator(target, tiny_tokenizer, draft=_plain(other_draft_pack), spec_tokens=k, lookup=3)
    st = gen._new_state(prompt, params, None)
    gen._step_sync([st])  # prefill: the draft reads the prompt meanwhile
    assert st.draft_cache is not None and st.draft_cache.length == len(prompt)
    st.pending = [prompt[-1]]  # rewind one position: the context ends in the repeated tail again
    st.ctx = list(prompt[:-1])
    st.cache.set_length(len(st.ctx))
    st.draft_cache.truncate(len(st.ctx))
    from pgw.engine.generate import _lookup_drafts

    found = _lookup_drafts(st.ctx + st.pending, k, 3)
    assert 0 < len(found) < k
    gen._plan_drafts([st])
    assert len(st.drafts) == k and st.drafts[: len(found)] == found


def test_one_gram_match_does_not_preempt_draft(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                               tiny_tokenizer: ChatTokenizer) -> None:
    from pgw.engine.generate import _lookup_drafts

    target = _plain(qwen_pack)
    draft = _plain(other_draft_pack)
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    with_lookup = Generator(target, tiny_tokenizer, draft=draft, spec_tokens=2, lookup=3)
    draft_only = Generator(target, tiny_tokenizer, draft=draft, spec_tokens=2)
    st = with_lookup._new_state(PROMPT, params, None)
    with_lookup._step_sync([st])  # prefill
    st.drafts = []
    st.pending = [PROMPT[1]]  # 72 occurred before: a 1-gram match only
    seq = st.ctx + st.pending
    assert _lookup_drafts(seq, 2, 3, min_n=1) and not _lookup_drafts(seq, 2, 3)
    with_lookup._plan_drafts([st])
    got = list(st.drafts)
    draft_only._plan_drafts([st])
    assert len(got) == 2 and got == st.drafts  # the draft model's proposal, not the 1-gram lookup


def test_no_drafts_when_step_exceeds_spec_max_batch(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    prompt = PROMPT * 4
    for limit in (1, 2):
        gen = Generator(target, tiny_tokenizer, spec_tokens=2, lookup=3, spec_max_batch=limit)
        a, b, c = (gen._new_state(prompt, params, None) for _ in range(3))
        gen._plan_drafts([a])
        assert len(a.drafts) == 2
        gen._plan_drafts([a, b])
        assert [len(a.drafts), len(b.drafts)] == ([0, 0] if limit == 1 else [2, 2])
        gen._plan_drafts([a, b, c])
        assert a.drafts == b.drafts == c.drafts == []
    assert Generator(target, tiny_tokenizer, spec_tokens=2, lookup=3).spec_max_batch == 1  # default


def test_budget_drops_to_zero_not_to_remainder(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Near the end of max_tokens or the spare mask rows, k is spec_tokens or 0, never in between."""
    target = _plain(qwen_pack)
    gen = Generator(target, tiny_tokenizer, spec_tokens=4, lookup=3)
    for max_tokens in range(1, 8):
        st = gen._new_state(PROMPT, SamplingParams(max_tokens=max_tokens, temperature=0.0), None)
        assert gen._spec_budget(st) == (4 if max_tokens - 1 >= 4 else 0)
    params = SamplingParams(max_tokens=10, temperature=0.0)
    for spare in range(0, 7):
        res = LocalMaskReservation(qwen_pack, DEFAULT_RING, len(PROMPT) + params.max_tokens - 1 + spare)
        st = gen._new_state(PROMPT, params, res)
        assert gen._spec_budget(st) == (4 if spare >= 4 else 0)
    adaptive = Generator(target, tiny_tokenizer, spec_tokens=4, spec_min_tokens=2, lookup=3, draft_confidence=0.5)
    for spare in range(0, 7):
        res = LocalMaskReservation(qwen_pack, DEFAULT_RING, len(PROMPT) + params.max_tokens - 1 + spare)
        st = adaptive._new_state(PROMPT, params, res)
        assert adaptive._spec_budget(st) == (4 if spare >= 4 else 2 if spare >= 2 else 0)


@pytest.mark.parametrize("which", ["same", "other"])
def test_k2_with_row_budget_2_does_not_starve(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                              tiny_tokenizer: ChatTokenizer, which: str) -> None:
    """The default retune: k=2 drafts with 2.0 mask rows per requested token. A perfect draft speculates on
    every pass; a poor one at least until half the spare rows (one per requested token) are used up."""
    params = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
    plain, masked, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    base = Generator(plain, tiny_tokenizer).generate(PROMPT, params)
    slots = len(PROMPT) + 2 * params.max_tokens
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, slots)
    draft = _plain(qwen_pack if which == "same" else other_draft_pack)
    gen = Generator(masked, tiny_tokenizer, draft=draft, spec_tokens=2)
    rows = _pass_rows(gen)
    out = gen.generate(PROMPT, params, session=res)
    assert out.token_ids == base.token_ids
    assert out.masks_consumed <= slots
    decode = [r[0] for r in rows[1:]]
    spec_passes = sum(1 for r in decode if r == 3)
    if which == "same":
        # every pass but the last couple (max_tokens budget) carries 2 drafts, all accepted
        assert spec_passes >= len(decode) - 2
        assert len(rows) <= params.max_tokens // 3 + 3
    else:
        # each rejected row burns one spare row: at least max_tokens / 2 tokens' worth of speculative passes
        assert spec_passes >= min(len(decode) - 2, params.max_tokens // 2 - 1)


def test_speculation_metrics(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                             tiny_tokenizer: ChatTokenizer) -> None:
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True)
    gen = Generator(target, tiny_tokenizer, draft=_plain(other_draft_pack), spec_tokens=2)
    rows = _pass_rows(gen)
    out = gen.generate(PROMPT, params)
    t = out.timings_ms
    spec_passes = sum(1 for r in rows if r[0] == 3)
    assert t["spec_passes"] == spec_passes > 0
    assert t["spec_rows"] == 2 * spec_passes
    assert 0 <= t["spec_accepted"] <= t["spec_rows"]
    assert t["draft_ms"] > 0
    st = gen.spec_stats
    assert st["spec_passes"] == t["spec_passes"] and st["spec_rows"] == t["spec_rows"]
    assert st["spec_accepted"] == t["spec_accepted"] and st["draft_ms"] >= t["draft_ms"] > 0
    assert st["draft_calls"] >= 1
    plain_gen = Generator(target, tiny_tokenizer)
    res = plain_gen.generate(PROMPT, params)
    assert "draft_ms" not in res.timings_ms and "spec_rows" not in res.timings_ms


def test_fixed_size_skips_confidence_softmax(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                             tiny_tokenizer: ChatTokenizer, monkeypatch: pytest.MonkeyPatch) -> None:
    """One fixed draft size (spec_min_tokens >= spec_tokens): no per-token softmax over the draft logits."""
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    gen = Generator(target, tiny_tokenizer, draft=_plain(other_draft_pack), spec_tokens=2, spec_min_tokens=2,
                    draft_confidence=0.5)
    calls = 0
    real = torch.softmax

    def counting(*a: Any, **kw: Any) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return real(*a, **kw)

    import pgw.engine.generate as G

    monkeypatch.setattr(G.torch, "softmax", counting)
    out = gen.generate(PROMPT, params)
    monkeypatch.setattr(G.torch, "softmax", real)
    assert out.token_ids == base.token_ids
    assert calls == 0


# ---- fixed-row speculation, strict-pass mode, the Drafter interface (chat-speed WP-A7) ------------------
# spec_rows=R: every pass after the prefill pass carries exactly R rows (never 1 because of the text, the
# drafts, the budget or the masks); spec_strict: the provider sees exactly as many passes as tokens generated.

LONG = [256, 72, 105, 33, 10, 72, 105, 32, 116, 104, 101, 114, 101]


def _gen_rows(gen: Generator) -> list[list[tuple[int, int, bool]]]:
    """Spy: per target pass, ``(rows, passes so far, dummy)`` per sequence."""
    seen: list[list[tuple[int, int, bool]]] = []
    orig = gen._inputs

    def spy(batch: Any) -> Any:
        out = orig(batch)
        seen.append([(len(si.input_ids), st.passes, st.dummy) for si, st in zip(out, batch, strict=True)])
        return out

    gen._inputs = spy  # type: ignore[method-assign]
    return seen


def _sources(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack) -> dict[str, dict[str, Any]]:
    return {
        "same": {"draft": _plain(qwen_pack)},
        "other": {"draft": _plain(other_draft_pack)},
        "lookup": {"lookup": 3},
        "lookup+other": {"draft": _plain(other_draft_pack), "lookup": 3},
    }


def test_lookup_fixed_work_matches_reference() -> None:
    """The constant-work lookup returns exactly what the early-exit reference returns."""
    from pgw.engine.generate import _lookup_drafts, _lookup_drafts_fixed

    g = torch.Generator().manual_seed(1)
    cases = 0
    for vocab in (2, 3, 5, 40):
        for size in (0, 1, 2, 3, 5, 9, 20, 64, 300):
            for _ in range(6):
                seq = torch.randint(0, vocab, (size,), generator=g).tolist()
                for k in (1, 2, 4):
                    for max_n, min_n in ((3, 3), (4, 2), (2, 1), (8, 3)):
                        assert _lookup_drafts_fixed(seq, k, max_n, min_n) == _lookup_drafts(seq, k, max_n, min_n)
                        cases += 1
    long_seq = (torch.randint(0, 7, (5000,), generator=g)).tolist()  # past the window
    assert _lookup_drafts_fixed(long_seq, 3, 4, 3) == _lookup_drafts(long_seq, 3, 4, 3)
    assert cases > 1000


def test_fixed_rows_config_validation(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    from pgw.core.errors import ConfigError

    target = _plain(qwen_pack)
    with pytest.raises(ConfigError, match="spec_strict needs spec_rows"):
        Generator(target, tiny_tokenizer, lookup=3, spec_strict=True)
    with pytest.raises(ConfigError, match=r"spec_rows must be in \[2, 8\]"):
        Generator(target, tiny_tokenizer, lookup=3, spec_rows=1)
    with pytest.raises(ConfigError, match="draft source"):
        Generator(target, tiny_tokenizer, spec_rows=3)
    with pytest.raises(ConfigError, match="set spec_rows"):
        Generator(target, tiny_tokenizer, drafter=_RandomDrafter(10))
    with pytest.raises(ConfigError, match="spec_rows needs spec_strict"):
        Generator(target, tiny_tokenizer, lookup=3, spec_rows=4)
    gen = Generator(target, tiny_tokenizer, lookup=3, spec_rows=4, spec_nonstrict_for_tests=True)
    assert gen.spec_rows == 4 and not gen.spec_strict and gen.spec_tokens == 3
    # masked filler rows (no padding support here): the reservation covers R rows per decode pass
    assert gen.mask_rows_needed(10, 20) == 10 + 20 + 19 * 3
    assert Generator(target, tiny_tokenizer).mask_rows_needed(10, 20) == 30


@pytest.mark.parametrize("strict", [False, True], ids=["fixed", "strict"])
@pytest.mark.parametrize("rows", [2, 4])
@pytest.mark.parametrize("source", ["same", "other", "lookup", "lookup+other"])
@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded", "penalties", "stop"])
def test_fixed_rows_output_identical(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                     tiny_tokenizer: ChatTokenizer, params: SamplingParams, source: str, rows: int,
                                     strict: bool) -> None:
    """Tokens, text, finish reason and logprobs are bit-identical to decoding without speculation; every
    pass after the prefill pass has exactly R rows (the first decode pass, the budget's end and stop
    strings included); in strict mode the pass count equals the number of generated tokens."""
    target = _plain(qwen_pack)
    prompt = LONG * 2
    base = Generator(target, tiny_tokenizer).generate(prompt, params)
    gen = Generator(target, tiny_tokenizer, spec_rows=rows, spec_strict=strict, spec_nonstrict_for_tests=True,
                    **_sources(qwen_pack, other_draft_pack)[source])
    seen = _gen_rows(gen)
    spec = gen.generate(prompt, params)
    assert spec.token_ids == base.token_ids and spec.text == base.text
    assert spec.finish_reason == base.finish_reason and spec.completion_tokens == base.completion_tokens
    assert spec.logprobs == base.logprobs
    assert seen[0] == [(len(prompt), 0, False)]
    assert all(p[0][0] == rows for p in seen[1:])
    assert spec.timings_ms["passes"] == len(seen)
    if strict:
        assert len(seen) == base.completion_tokens
    else:
        assert len(seen) <= base.completion_tokens
        assert not any(p[0][2] for p in seen)


def test_fixed_rows_perfect_draft_saves_passes_and_strict_restores_count(
        qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=30, temperature=0.0, ignore_eos=True)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    fast = Generator(target, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_nonstrict_for_tests=True)
    out = fast.generate(PROMPT, params)
    assert out.token_ids == base.token_ids
    assert out.timings_ms["passes"] <= 1 + -(-(params.max_tokens - 1) // 4) + 1  # ~4 tokens per pass
    strict = Generator(target, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True)
    seen = _gen_rows(strict)
    out2 = strict.generate(PROMPT, params)
    assert out2.token_ids == base.token_ids
    assert out2.timings_ms["passes"] == params.max_tokens == len(seen)
    dummies = [p for p in seen if p[0][2]]
    assert len(dummies) == params.max_tokens - out.timings_ms["passes"]
    assert strict.spec_stats["dummy_passes"] == len(dummies)
    assert all(p[0][0] == 4 for p in seen[1:])


def _provider_passes(client: Any) -> list[int]:
    """Rows of every pass the provider saw (one ``0.qkv`` item per pass for a single sequence)."""
    return [int(x.shape[0]) for lid, x in client.received if lid == "0.qkv"]


@pytest.mark.parametrize("case", ["predictable", "unpredictable"])
def test_strict_provider_pass_count_equals_length(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                                  tiny_tokenizer: ChatTokenizer, case: str) -> None:
    """Through MaskedBackend: the provider sees the prompt pass then exactly R rows per pass, and in strict
    mode exactly L passes, whether the text is highly predictable (the draft is the target) or not."""
    rows = 3
    if case == "predictable":
        params = SamplingParams(max_tokens=18, temperature=0.0, ignore_eos=True, logprobs=1)
        draft = qwen_pack
    else:
        params = SamplingParams(max_tokens=18, temperature=1.0, seed=5, ignore_eos=True, logprobs=1)
        draft = other_draft_pack
    plain, _, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    base = Generator(plain, tiny_tokenizer).generate(PROMPT, params)
    counts = {}
    for strict in (False, True):
        _, masked, client = plain_and_masked(qwen_pack, DEFAULT_RING, record=True)
        gen = Generator(masked, tiny_tokenizer, draft=_plain(draft), spec_rows=rows, spec_strict=strict,
                        spec_nonstrict_for_tests=True)
        need = gen.mask_rows_needed(len(PROMPT), params.max_tokens)
        res = LocalMaskReservation(qwen_pack, DEFAULT_RING, need)
        out = gen.generate(PROMPT, params, session=res)
        assert out.token_ids == base.token_ids and out.logprobs == base.logprobs
        seen = _provider_passes(client)
        assert seen[0] == len(PROMPT) and all(r == rows for r in seen[1:])
        assert out.masks_consumed == sum(seen) <= need
        counts[strict] = len(seen)
        if strict:
            assert len(seen) == out.completion_tokens == params.max_tokens
    if case == "predictable":
        assert counts[False] < counts[True]  # what non-strict mode reveals and strict mode hides


def test_fixed_rows_reservation_too_small_refused_before_any_pass(qwen_pack: InMemoryPack,
                                                                   tiny_tokenizer: ChatTokenizer) -> None:
    """Masked filler rows: a reservation that could run out mid-request is refused up front, so a pass
    with fewer than R rows (which would reveal the mask budget) is never sent."""
    from pgw.core.errors import MaskExhaustedError

    params = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True)
    _, masked, client = plain_and_masked(qwen_pack, DEFAULT_RING, record=True)
    gen = Generator(masked, tiny_tokenizer, lookup=3, spec_rows=4, spec_nonstrict_for_tests=True)
    need = gen.mask_rows_needed(len(PROMPT), params.max_tokens)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, need - 1)
    with pytest.raises(MaskExhaustedError):
        gen.generate(PROMPT, params, session=res, own_session=True)
    assert client.received == [] and client.requests == 0


class _RandomDrafter:
    """A custom :class:`Drafter`: random proposals of a fixed cost (outputs must not change)."""

    needs_prefill = False

    def __init__(self, vocab: int, seed: int = 0):
        self.vocab = vocab
        self.g = torch.Generator().manual_seed(seed)
        self.opened = 0
        self.closed = 0
        self.ks: list[int] = []

    def open(self, prompt: Any, max_len: int) -> Any:
        self.opened += 1
        return {"n": len(prompt)}

    def prefill(self, states: Any, prompts: Any) -> None:  # pragma: no cover - needs_prefill is False
        raise AssertionError("not called")

    def propose(self, jobs: Any) -> None:
        for job in jobs:
            self.ks.append(job.k)
            job.out = torch.randint(0, self.vocab, (job.k,), generator=self.g).tolist()

    def close(self, state: Any) -> None:
        self.closed += 1


def test_custom_drafter_output_identical(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    from pgw.engine.generate import Drafter

    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=14, temperature=0.7, seed=9, ignore_eos=True, logprobs=0)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    d = _RandomDrafter(int(target.arch.vocab_size))
    assert isinstance(d, Drafter)
    gen = Generator(target, tiny_tokenizer, drafter=d, spec_rows=5, spec_strict=True)
    seen = _gen_rows(gen)
    out = gen.generate(PROMPT, params)
    assert out.token_ids == base.token_ids and out.logprobs == base.logprobs
    assert len(seen) == params.max_tokens and all(p[0][0] == 5 for p in seen[1:])
    assert set(d.ks) == {4} and len(d.ks) == params.max_tokens - 1  # dummy passes run the drafter too
    assert d.opened == d.closed == 1


class _CountingDraft:
    """Draft model wrapper counting forward steps (fixed per pass whatever the text)."""

    def __init__(self, inner: TransformerModel):
        self.inner = inner
        self.arch = inner.arch
        self.steps = 0

    def new_cache(self, n: int) -> Any:
        return self.inner.new_cache(n)

    def forward_batch(self, inputs: list[Any]) -> list[torch.Tensor]:
        self.steps += 1
        return self.inner.forward_batch(inputs)


@pytest.mark.parametrize("which", ["same", "other"])
@pytest.mark.parametrize("lookup", [0, 3])
def test_fixed_draft_steps_per_pass(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                    tiny_tokenizer: ChatTokenizer, which: str, lookup: int) -> None:
    """Exactly k draft-model steps per decode pass (plus one prompt prefill), for a perfect and a poor draft,
    with and without lookup hits, and in dummy passes: no early exit when a proposal is found or ends."""
    target = _plain(qwen_pack)
    rows = 4
    params = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    draft = _CountingDraft(_plain(qwen_pack if which == "same" else other_draft_pack))
    gen = Generator(target, tiny_tokenizer, draft=draft, lookup=lookup, spec_rows=rows, spec_strict=True)  # type: ignore[arg-type]
    out = gen.generate(LONG * 3, params)
    assert out.timings_ms["passes"] == params.max_tokens
    assert draft.steps == 1 + (params.max_tokens - 1) * (rows - 1)


def test_draft_with_another_vocab_size_fixed_rows(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                                  tiny_tokenizer: ChatTokenizer) -> None:
    """Out-of-vocabulary draft ids end a proposal (filler rows complete it); ids fed to the draft are clamped."""
    target = _plain(qwen_pack)
    tv = int(target.arch.vocab_size)
    params = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    small = tv - 40
    draft = _OddVocabDraft(_plain(other_draft_pack), small, tv)
    gen = Generator(target, tiny_tokenizer, draft=draft, spec_rows=3, spec_nonstrict_for_tests=True)  # type: ignore[arg-type]
    seen = _gen_rows(gen)
    out = gen.generate(PROMPT, params)
    assert out.token_ids == base.token_ids
    assert all(p[0][0] == 3 for p in seen[1:])
    assert draft.calls > 0 and draft.max_fed < small


def test_strict_stream_closed_early_runs_due_passes(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """A client that goes away mid-stream: the passes still due up to the tokens generated so far run."""
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=30, temperature=0.0, ignore_eos=True)
    gen = Generator(target, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True)
    states: list[Any] = []
    orig = gen._inputs

    def spy(batch: Any) -> Any:
        states.extend(b for b in batch if b not in states)
        return orig(batch)

    gen._inputs = spy  # type: ignore[method-assign]
    it: Any = gen.stream(PROMPT, params)
    got = [next(it) for _ in range(6)]
    it.close()  # the client went away
    st = states[0]
    assert len(got) == 6 and st.finish == "cancelled"
    assert len(st.generated) < params.max_tokens
    assert st.passes == len(st.generated)
    assert gen.spec_stats["dummy_passes"] > 0


def _pad_capable(monkeypatch: pytest.MonkeyPatch, inner: TransformerModel) -> Any:
    """A model and SequenceInput with the padding-row hook (``pad_rows``) that WP-A4b/A5 provide: padding
    rows are computed like any row (outputs of real rows unchanged) but take no masks."""
    import dataclasses

    import pgw.engine.generate as G
    from pgw.engine.model import SequenceInput

    @dataclasses.dataclass
    class PadInput(SequenceInput):
        pad_rows: int = 0

    monkeypatch.setattr(G, "SequenceInput", PadInput)

    class PadModel:
        supports_pad_rows = True

        def __init__(self) -> None:
            self.inner = inner
            self.arch = inner.arch
            self.vocab_size = inner.vocab_size
            self.seen: list[tuple[int, int]] = []

        def new_cache(self, n: int) -> Any:
            return inner.new_cache(n)

        def forward_batch(self, seqs: list[Any]) -> list[Any]:
            plain = []
            for s in seqs:
                rows, pad = len(s.input_ids), int(getattr(s, "pad_rows", 0))
                self.seen.append((rows, pad))
                if s.session is not None:
                    s.session.take_rows(rows - pad)
                plain.append(SequenceInput(s.input_ids, s.cache, None, s.all_logits, s.last_logits, s.serial_tail))
            return inner.forward_batch(plain)

    return PadModel()


class _CountingSession:
    def __init__(self, rows: int):
        self.left = rows

    def remaining(self, lid: Any) -> int:
        return self.left

    def reserved_rows(self, lid: Any) -> int:
        return self.left

    def take_rows(self, n: int) -> None:
        assert n <= self.left, "took more mask rows than the reservation holds"
        self.left -= n


@pytest.mark.parametrize("strict", [False, True])
def test_padding_rows_keep_r_rows_at_mask_exhaustion(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                                     monkeypatch: pytest.MonkeyPatch, strict: bool) -> None:
    """With ``spec_mask_rows`` = C on a pad-capable model the reservation needs prompt + C * (max_tokens - 1)
    rows: drafts use the charged credit, the rest of the R rows are padding. Every pass has R rows whatever
    the credit (padding, never fewer rows); dummy passes are all padding (mask-free)."""
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True, logprobs=1)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    model = _pad_capable(monkeypatch, target)
    assert not Generator(model, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=strict,
                         spec_nonstrict_for_tests=True)._pad_ok  # padding only with an explicit mask charge
    gen = Generator(model, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=strict,
                    spec_nonstrict_for_tests=True, spec_mask_rows=2)
    assert gen._pad_ok
    need = gen.mask_rows_needed(len(PROMPT), params.max_tokens)
    assert need == len(PROMPT) + 2 * (params.max_tokens - 1)
    sess = _CountingSession(need)
    out = gen.generate(PROMPT, params, session=sess)
    assert out.token_ids == base.token_ids and out.logprobs == base.logprobs
    assert model.seen[0] == (len(PROMPT), 0)
    assert all(rows == 4 for rows, _ in model.seen[1:])
    assert out.masks_consumed == need - sess.left
    assert all(pad >= 2 for _, pad in model.seen[1:])  # at most C = 2 real rows per pass here (no spare credit)
    if strict:
        assert len(model.seen) == params.max_tokens
        assert any(pad == 4 for _, pad in model.seen)  # a dummy pass: mask-free


def test_fixed_rows_async_streams_identical(qwen_pack: InMemoryPack, other_draft_pack: InMemoryPack,
                                            tiny_tokenizer: ChatTokenizer) -> None:
    """Concurrent streams through the batching loop (steps of up to spec_max_batch sequences carry R rows,
    larger ones 1 row): the same tokens; strict mode drains every stream to L passes."""
    target = _plain(qwen_pack)
    prompts = [PROMPT, [256, 65, 66, 67], [256, 120, 121]]
    params = SamplingParams(max_tokens=14, temperature=0.0, ignore_eos=True)
    base = [Generator(target, tiny_tokenizer).generate(p, params).token_ids for p in prompts]
    for max_batch in (1, 8):
        gen = Generator(target, tiny_tokenizer, draft=_plain(other_draft_pack), lookup=3, spec_rows=3,
                        spec_strict=True, spec_max_batch=max_batch)
        seen = _gen_rows(gen)

        async def one(p: list[int], g: Generator = gen) -> list[int]:
            return [ev.token_id async for ev in g.astream(p, params)]

        async def main(g: Generator = gen) -> list[list[int]]:
            try:
                out = list(await asyncio.gather(*(one(p) for p in prompts)))
                await g.wait_drained()
                return out
            finally:
                await g.aclose()

        assert asyncio.run(main()) == base
        for step in seen:
            rows = {r for r, passes, _ in step if passes > 0}
            assert rows <= ({3} if len(step) <= max_batch else {1})
        per_seq = sum(len(step) for step in seen)
        assert per_seq == len(prompts) * params.max_tokens  # strict: L passes per sequence


def test_strict_answer_streams_before_dummy_passes_and_conversation_queues(
        qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """The answer arrives before the dummy passes run; a follow-up turn of the same conversation starts only
    after them (the provider never sees the new turn overlap the old one's passes)."""
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
    gen = Generator(target, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True)
    order: list[tuple[int, bool]] = []
    orig = gen._inputs

    def spy(batch: Any) -> Any:
        order.extend((len(st.prompt), st.dummy) for st in batch)
        return orig(batch)

    gen._inputs = spy  # type: ignore[method-assign]
    second = [256, 65, 66]

    async def main() -> tuple[int, list[int]]:
        try:
            first = [ev async for ev in gen.astream(PROMPT, params, conversation="c1")]
            passes_at_answer = len([o for o in order if o[0] == len(PROMPT)])
            nxt = [ev.token_id async for ev in gen.astream(second, params, conversation="c1")]
            assert first[-1].finish_reason is not None
            await gen.wait_drained()
            return passes_at_answer, nxt
        finally:
            await gen.aclose()

    passes_at_answer, nxt = asyncio.run(main())
    assert passes_at_answer < params.max_tokens  # the answer did not wait for the dummy passes
    first_passes = [i for i, o in enumerate(order) if o[0] == len(PROMPT)]
    second_passes = [i for i, o in enumerate(order) if o[0] == len(second)]
    assert len(first_passes) == len(second_passes) == params.max_tokens
    assert max(first_passes) < min(second_passes)  # the follow-up queued behind every dummy pass
    assert nxt == Generator(target, tiny_tokenizer).generate(second, params).token_ids


def test_strict_non_batching_drains_in_background(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True)
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    gen = Generator(target, tiny_tokenizer, draft=_plain(qwen_pack), spec_rows=4, spec_strict=True, batching=False)
    seen = _gen_rows(gen)

    async def main() -> Any:
        try:
            res = await gen.agenerate(PROMPT, params)
            await gen.wait_drained()
            return res
        finally:
            await gen.aclose()

    res = asyncio.run(main())
    assert res.token_ids == base.token_ids
    assert len(seen) == params.max_tokens
    # returned before the background dummy passes: the final provider-visible count, not a partial one
    assert res.timings_ms["passes"] == params.max_tokens
    assert 0 <= res.timings_ms.get("passes_pending", 0) < params.max_tokens


def test_fixed_rows_mask_exhaustion_retry_keeps_plan(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """A MaskExhaustedError in a multi-sequence step retries each sequence alone with the step's plan: its
    rows stay those of the step (a function of the public session count, never R because of the retry)
    and strict-mode dummy passes are retried too. Outputs unchanged, L passes per sequence."""
    from pgw.core.errors import MaskExhaustedError

    target = _plain(qwen_pack)
    prompts = [PROMPT, [256, 65, 66, 67], [256, 120, 121]]
    params = [SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True) for n in (16, 6, 11)]
    base = [Generator(target, tiny_tokenizer).generate(p, sp).token_ids
            for p, sp in zip(prompts, params, strict=True)]
    gen = Generator(target, tiny_tokenizer, draft=_plain(qwen_pack), lookup=3, spec_rows=3, spec_strict=True,
                    spec_max_batch=2)
    seen = _gen_rows(gen)  # one entry per forward call, failed ones included
    sessions = [type("_S", (), {})() for _ in prompts]
    who = {id(s): i for i, s in enumerate(sessions)}
    calls: list[tuple[list[int], bool]] = []  # (sequence index per input, failed)
    multi = 0
    orig = target.aforward_batch

    async def spy(inputs: Any) -> Any:
        nonlocal multi
        idx = [who[id(si.session)] for si in inputs]
        if len(inputs) > 1:
            multi += 1
            if multi % 3 == 1:  # transient exhaustion: nothing consumed
                calls.append((idx, True))
                raise MaskExhaustedError("injected")
        calls.append((idx, False))
        return await orig(inputs)

    monkeypatch.setattr(target, "aforward_batch", spy)

    async def one(i: int) -> list[int]:
        return [ev.token_id async for ev in gen.astream(prompts[i], params[i], session=sessions[i])]

    async def main() -> list[list[int]]:
        try:
            out = list(await asyncio.gather(*(one(i) for i in range(len(prompts)))))
            await gen.wait_drained()
            return out
        finally:
            await gen.aclose()

    assert asyncio.run(main()) == base
    assert len(calls) == len(seen)
    failures = [j for j, (_, failed) in enumerate(calls) if failed]
    assert failures
    retried_dummy = retried_r = False
    for j in failures:
        idx, _ = calls[j]
        for n, i in enumerate(idx):  # each sequence of the failed step, alone, with the step's rows
            assert calls[j + 1 + n] == ([i], False)
            assert seen[j + 1 + n] == [seen[j][n]]
            retried_dummy |= seen[j][n][2]
            retried_r |= seen[j][n][0] == 3 and seen[j][n][1] > 0
        assert all(r == (3 if len(idx) <= 2 else 1) for r, p, _ in seen[j] if p > 0)
    assert retried_dummy and retried_r
    for i, sp in enumerate(params):  # strict: the provider saw L passes per sequence
        assert sum(idx.count(i) for idx, failed in calls if not failed) == sp.max_tokens


def test_builtin_drafter_catch_up_compares_only_new_tokens(qwen_pack: InMemoryPack,
                                                           tiny_tokenizer: ChatTokenizer) -> None:
    """The drafter's catch-up starts at the prefix already confirmed (O(accepted drafts), not O(context)),
    and its draft cache always ends up holding the context exactly as a full comparison would."""
    from pgw.engine.generate import BuiltinDrafter

    target = _plain(qwen_pack)
    params = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
    base = Generator(target, tiny_tokenizer).generate(LONG * 3, params)
    d = BuiltinDrafter(_plain(qwen_pack), lookup=3, target_vocab=qwen_pack.arch.vocab_size)
    checked: list[tuple[int, int]] = []
    orig = d.propose

    def propose(jobs: Any) -> None:
        for job in jobs:
            st = job.state
            if st.cache is not None:
                ctx = job.context
                assert st.fed[: st.ok] == list(ctx[: st.ok])  # the invariant the shortcut relies on
                checked.append((st.ok, len(ctx)))
        orig(jobs)
        for job in jobs:
            st = job.state
            if st.cache is not None:
                assert st.cache.length == len(st.fed) and st.ok == len(job.context) - 1
                assert st.fed[: len(job.context)] == [int(t) for t in job.context]

    d.propose = propose  # type: ignore[method-assign]
    gen = Generator(target, tiny_tokenizer, drafter=d, spec_rows=4, spec_strict=True)
    out = gen.generate(LONG * 3, params)
    assert out.token_ids == base.token_ids
    assert checked and all(len_ctx - ok <= 4 + 1 for ok, len_ctx in checked)  # at most the last pass's rows
