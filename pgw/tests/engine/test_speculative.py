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
