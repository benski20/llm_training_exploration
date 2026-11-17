"""Generator: decode loop, stop handling, streaming, batching, masked == plain end to end."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A8, RING_K40_A12, RING_K48_A20, ConfigError, MaskExhaustedError, RingParams
from pgw.engine import (
    ChatTokenizer,
    FinishReason,
    Generator,
    InMemoryPack,
    LocalMaskReservation,
    LocalMaskSource,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from pgw.engine.kv_cache import ContextOverflowError
from tests.engine.conftest import plain_and_masked

PROMPT = [256, 72, 105, 33, 10]


@pytest.fixture(scope="module")
def plain_gen(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> Generator:
    return Generator(TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12)), tiny_tokenizer)


def manual_greedy(model: TransformerModel, prompt: list[int], n: int) -> list[int]:
    cache = model.new_cache(len(prompt) + n)
    logits = model.forward(torch.tensor(prompt), cache)
    out = []
    for _ in range(n):
        tok = int(torch.argmax(logits))
        out.append(tok)
        logits = model.forward(torch.tensor([tok]), cache)
    return out


def test_greedy_matches_manual_loop(plain_gen: Generator) -> None:
    p = SamplingParams(max_tokens=12, temperature=0, ignore_eos=True)
    r = plain_gen.generate(PROMPT, p)
    assert r.token_ids == manual_greedy(plain_gen.model, PROMPT, 12)
    assert r.finish_reason == FinishReason.LENGTH
    assert r.prompt_tokens == len(PROMPT) and r.completion_tokens == 12
    assert r.masks_consumed == len(PROMPT) + 11  # the last sampled token is never fed back
    assert r.text == plain_gen.tokenizer.decode(r.token_ids)  # type: ignore[union-attr]
    assert {"ttft_ms", "decode_ms", "total_ms", "tpot_ms"} <= set(r.timings_ms)


def test_stream_concatenates_to_generate(plain_gen: Generator) -> None:
    p = SamplingParams(max_tokens=10, temperature=0.8, seed=5, ignore_eos=True)
    events = list(plain_gen.stream(PROMPT, p))
    r = plain_gen.generate(PROMPT, p)
    assert [e.token_id for e in events] == r.token_ids
    assert "".join(e.text for e in events) == r.text
    assert events[-1].finish_reason == FinishReason.LENGTH
    assert all(e.finish_reason is None for e in events[:-1])


def test_stop_strings_truncate(plain_gen: Generator) -> None:
    base = plain_gen.generate(PROMPT, SamplingParams(max_tokens=24, temperature=0, ignore_eos=True))
    text = base.text
    candidates = [text[i : i + 2] for i in range(3, len(text) - 2) if "�" not in text[i : i + 2]]
    assert candidates, "need some decodable text"
    stop = candidates[0]
    cut = text.find(stop)
    r = plain_gen.generate(PROMPT, SamplingParams(max_tokens=24, temperature=0, ignore_eos=True, stop=(stop,)))
    assert r.finish_reason == FinishReason.STOP
    assert r.text == text[:cut]
    events = list(plain_gen.stream(PROMPT, SamplingParams(max_tokens=24, temperature=0, ignore_eos=True,
                                                          stop=(stop,))))
    assert "".join(e.text for e in events) == text[:cut]
    assert stop not in "".join(e.text for e in events)
    r2 = plain_gen.generate(PROMPT, SamplingParams(max_tokens=24, temperature=0, ignore_eos=True, stop=(stop,),
                                                   include_stop_str_in_output=True))
    assert r2.text == text[: cut + len(stop)]


def test_stop_token_ids_and_eos(plain_gen: Generator) -> None:
    base = plain_gen.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
    stop_tok = base.token_ids[3]
    first = base.token_ids.index(stop_tok)
    r = plain_gen.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0, stop_token_ids=(stop_tok,)))
    assert r.finish_reason == FinishReason.STOP
    assert r.token_ids == base.token_ids[:first]  # stop token excluded from output
    assert r.completion_tokens == first + 1
    events = list(plain_gen.stream(PROMPT, SamplingParams(max_tokens=8, temperature=0,
                                                          stop_token_ids=(stop_tok,))))
    assert events[-1].token_id == stop_tok and events[-1].finish_reason == FinishReason.STOP


def test_eos_from_tokenizer_stops(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Force EOS by biasing the model: a backend wrapper is not needed, use a fake model."""

    class EosModel:
        arch = qwen_pack.arch

        def new_cache(self, max_len: int) -> Any:
            from pgw.engine import KVCache

            return KVCache(1, 1, 1, max_len)

        def forward_batch(self, seqs: Any) -> list[torch.Tensor]:
            out = []
            for s in seqs:
                s.cache.append(0, torch.zeros(len(s.input_ids), 1, 1), torch.zeros(len(s.input_ids), 1, 1))
                s.cache.advance(len(s.input_ids))
                lg = torch.zeros(qwen_pack.arch.vocab_size)
                lg[65 if s.cache.length < len(PROMPT) + 2 else 258] = 10.0
                out.append(lg)
            return out

    gen = Generator(EosModel(), tiny_tokenizer)  # type: ignore[arg-type]
    r = gen.generate(PROMPT, SamplingParams(max_tokens=20, temperature=0))
    assert r.finish_reason == FinishReason.STOP and r.text == "AA" and r.token_ids == [65, 65]
    r = gen.generate(PROMPT, SamplingParams(max_tokens=5, temperature=0, ignore_eos=True))
    assert r.finish_reason == FinishReason.LENGTH and len(r.token_ids) == 5


def test_context_limit(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    gen = Generator(model, tiny_tokenizer)
    ctx = qwen_pack.arch.max_position_embeddings
    with pytest.raises(ContextOverflowError):
        gen.generate([65] * ctx, SamplingParams(max_tokens=1))
    r = gen.generate([65] * (ctx - 3), SamplingParams(max_tokens=50, temperature=0, ignore_eos=True))
    assert r.finish_reason == FinishReason.LENGTH and r.completion_tokens == 3
    with pytest.raises(ConfigError):
        gen.generate([], SamplingParams())
    with pytest.raises(ConfigError):
        Generator(model, None).generate(PROMPT, SamplingParams(stop=("x",)))
    ids_only = Generator(model, None).generate(PROMPT, SamplingParams(max_tokens=3, temperature=0))
    assert ids_only.text == "" and len(ids_only.token_ids) <= 3


@pytest.mark.parametrize("ring", [RING_K32_A8, RING_K40_A12, RING_K48_A20], ids=lambda r: r.name)
def test_masked_generation_bit_identical_to_plain(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                                  ring: RingParams) -> None:
    plain, masked, client = plain_and_masked(qwen_pack, ring)
    src = LocalMaskSource(qwen_pack)
    p = SamplingParams(max_tokens=8, temperature=0.9, seed=42, logprobs=0, ignore_eos=True)
    rp = Generator(plain, tiny_tokenizer).generate(PROMPT, p)
    sess = src.reserve(qwen_pack.fingerprint, ring, len(PROMPT) + p.max_tokens)
    rm = Generator(masked, tiny_tokenizer).generate(PROMPT, p, session=sess)
    assert rm.token_ids == rp.token_ids and rm.text == rp.text
    assert rm.logprobs == rp.logprobs  # bit-identical logits => identical logprobs
    # one round trip per linear group per forward pass
    assert client.requests == 4 * qwen_pack.arch.num_layers * p.max_tokens
    remaining = sess.remaining(qwen_pack.linear_specs()[0].id)
    assert remaining == 1  # prompt + max_tokens reserved, last token never fed


def test_masked_logits_bit_identical(llama_pack: InMemoryPack) -> None:
    for ring in (RING_K32_A8, RING_K40_A12, RING_K48_A20):
        plain, masked, _ = plain_and_masked(llama_pack, ring)
        res = LocalMaskReservation(llama_pack, ring, 16)
        ids = torch.tensor(PROMPT * 2)
        a = plain.forward(ids, plain.new_cache(16), all_logits=True)
        b = masked.forward(ids, masked.new_cache(16), session=res, all_logits=True)
        assert torch.equal(a, b), ring.name


def test_session_factory_reserves_and_burns(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    ring = RING_K40_A12
    _, masked, _ = plain_and_masked(qwen_pack, ring)
    made: list[LocalMaskReservation] = []

    def factory(slots: int) -> LocalMaskReservation:
        r = LocalMaskReservation(qwen_pack, ring, slots)
        made.append(r)
        return r

    gen = Generator(masked, tiny_tokenizer, session_factory=factory)
    r = gen.generate(PROMPT, SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
    assert len(made) == 1 and made[0].slots == len(PROMPT) + 4
    assert made[0]._closed  # unused rows burned
    assert r.masks_consumed == len(PROMPT) + 3
    # early-closed stream also releases
    it = gen.stream(PROMPT, SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
    next(it)
    it.close()  # type: ignore[attr-defined]
    assert made[1]._closed


def test_generate_many_matches_individual(plain_gen: Generator) -> None:
    prompts = [PROMPT, PROMPT[:2], [256, 65, 66, 67, 68, 69, 70]]
    params = [SamplingParams(max_tokens=n, temperature=0, ignore_eos=True) for n in (5, 7, 3)]
    singles = [plain_gen.generate(p, sp) for p, sp in zip(prompts, params, strict=True)]
    gen2 = Generator(plain_gen.model, plain_gen.tokenizer, max_batch=2)
    many = gen2.generate_many(prompts, params)
    for a, b in zip(singles, many, strict=True):
        assert a.token_ids == b.token_ids and a.text == b.text and a.finish_reason == b.finish_reason
    with pytest.raises(ConfigError):
        plain_gen.generate_many(prompts, params[:1])


async def test_astream_batches_concurrent_sessions(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    ring = RING_K40_A12
    plain, masked, client = plain_and_masked(qwen_pack, ring)
    src = LocalMaskSource(qwen_pack)
    prompts = [PROMPT, PROMPT[:3], [256, 80, 81]]
    p = SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)
    expected = [Generator(plain, tiny_tokenizer).generate(pr, p).token_ids for pr in prompts]
    gen = Generator(masked, tiny_tokenizer, session_factory=lambda n: src.reserve(qwen_pack.fingerprint, ring, n))

    async def collect(pr: list[int]) -> list[int]:
        return [ev.token_id async for ev in gen.astream(pr, p)]

    got = await asyncio.gather(*(collect(pr) for pr in prompts))
    assert got == expected
    per_pass = 4 * qwen_pack.arch.num_layers
    # 3 sequences x 6 forward passes each would be 18 passes unbatched; batching shares round trips
    assert client.requests < 3 * 6 * per_pass
    assert client.requests % per_pass == 0
    res = await gen.agenerate(PROMPT, p)
    assert res.token_ids == expected[0] and res.finish_reason == FinishReason.LENGTH
    assert res.masks_consumed == len(PROMPT) + 5
    await gen.aclose()


async def test_astream_unbatched_and_cancel(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    p = SamplingParams(max_tokens=5, temperature=0, ignore_eos=True)
    ref = Generator(model, tiny_tokenizer).generate(PROMPT, p).token_ids
    gen = Generator(model, tiny_tokenizer, batching=False)
    assert [e.token_id async for e in gen.astream(PROMPT, p)] == ref
    for batching in (True, False):
        g = Generator(model, tiny_tokenizer, batching=batching)
        agen = g.astream(PROMPT, SamplingParams(max_tokens=50, temperature=0, ignore_eos=True))
        first = await agen.__anext__()
        assert first.token_id == ref[0]
        await agen.aclose()
        await asyncio.sleep(0.05)
        assert not g._active  # cancelled session dropped by the loop
        await g.aclose()


async def test_astream_isolates_mask_exhaustion(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    ring = RING_K40_A12
    plain, masked, _ = plain_and_masked(qwen_pack, ring)
    p = SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)
    expected = Generator(plain, tiny_tokenizer).generate(PROMPT, p).token_ids
    gen = Generator(masked, tiny_tokenizer)
    good = LocalMaskReservation(qwen_pack, ring, len(PROMPT) + 6)
    short = LocalMaskReservation(qwen_pack, ring, len(PROMPT) + 2)

    async def run(sess: Any) -> list[int] | BaseException:
        try:
            return [e.token_id async for e in gen.astream(PROMPT, p, session=sess)]
        except MaskExhaustedError as exc:
            return exc

    a, b = await asyncio.gather(run(good), run(short))
    assert a == expected
    assert isinstance(b, MaskExhaustedError)
    await gen.aclose()


async def test_astream_propagates_errors(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    from pgw.core import IntegrityError
    from pgw.engine import LocalComputeClient, MaskedBackend

    ring = RING_K40_A12

    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        return y + 1 if lid == "1.o" else y

    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack, tamper=tamper), ring, qwen_pack.fingerprint)
    gen = Generator(TransformerModel(qwen_pack, be), tiny_tokenizer)
    with pytest.raises(IntegrityError):
        async for _ in gen.astream(PROMPT, SamplingParams(max_tokens=3),
                                   session=LocalMaskReservation(qwen_pack, ring, 8)):
            pass
    with pytest.raises(IntegrityError):
        gen.generate(PROMPT, SamplingParams(max_tokens=3), session=LocalMaskReservation(qwen_pack, ring, 8))
    await gen.aclose()


def test_chat_prompt_end_to_end(plain_gen: Generator, tiny_tokenizer: ChatTokenizer) -> None:
    ids = tiny_tokenizer.apply_chat_template([{"role": "system", "content": "be brief"},
                                              {"role": "user", "content": "hello"}])
    r = plain_gen.generate(ids, SamplingParams(max_tokens=5, temperature=0.7, top_p=0.9, top_k=50, seed=3))
    assert r.prompt_tokens == len(ids) and 1 <= r.completion_tokens <= 5
