"""The masked LM head with strict speculation (opt-in): verify rows as one head call, filler rows (R rows per
pass, unpadded) and padded passes (``spec_mask_rows``) equal plain decoding with the same int8 head, bit for
bit; the provider sees the head as one item of a fixed row count per pass, whatever the drafts."""

from __future__ import annotations

from typing import Any

import pytest

from pgw.core import DEFAULT_RING
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
from pgw.engine.masked_head import MaskedHeadPack, enable_masked_head, head_linear_id
from tests.engine.test_spec_no_compromise import LONG, PROMPT, _NoiseDrafter

PARAMS = [
    SamplingParams(max_tokens=18, temperature=0.0, ignore_eos=True, logprobs=2),
    SamplingParams(max_tokens=18, temperature=0.9, top_p=0.95, seed=7, ignore_eos=True, logprobs=0),
]


@pytest.fixture(scope="module")
def head_pack(qwen_pack: InMemoryPack) -> MaskedHeadPack:
    return MaskedHeadPack(qwen_pack)


def _plain_head(pack: MaskedHeadPack) -> TransformerModel:
    m = TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))
    enable_masked_head(m)
    return m


def _masked_head(pack: MaskedHeadPack) -> tuple[TransformerModel, LocalComputeClient]:
    client = LocalComputeClient(pack, record=True)
    m = TransformerModel(pack, MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint))
    enable_masked_head(m)
    return m, client


def _source(name: str, qwen_pack: Any) -> dict[str, Any]:
    if name == "noise":
        return {"drafter": _NoiseDrafter(int(qwen_pack.arch.vocab_size))}
    if name == "same":  # the base model with its float head drafts (it only proposes)
        return {"draft": TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))}
    return {"lookup": 3}


@pytest.mark.parametrize("pad", [None, 2], ids=["fillers", "padded"])
@pytest.mark.parametrize("rows", [2, 4])
@pytest.mark.parametrize("source", ["same", "lookup"])
@pytest.mark.parametrize("params", PARAMS, ids=["greedy", "seeded"])
def test_strict_speculation_exact(qwen_pack: InMemoryPack, head_pack: MaskedHeadPack, tiny_tokenizer: ChatTokenizer,
                                  params: SamplingParams, source: str, rows: int, pad: int | None) -> None:
    prompt = LONG * 2
    base = Generator(_plain_head(head_pack), tiny_tokenizer).generate(prompt, params)
    model, client = _masked_head(head_pack)
    assert model.supports_pad_rows
    kw: dict[str, Any] = {"spec_mask_rows": pad} if pad else {}
    gen = Generator(model, tiny_tokenizer, spec_rows=rows, spec_strict=True, **kw, **_source(source, qwen_pack))
    need = gen.mask_rows_needed(len(prompt), params.max_tokens)
    res = LocalMaskReservation(head_pack, DEFAULT_RING, need)
    out = gen.generate(prompt, params, session=res)
    assert out.token_ids == base.token_ids and out.text == base.text
    assert out.finish_reason == base.finish_reason and out.logprobs == base.logprobs
    assert model.backend.stats.integrity_failures == 0
    key = head_linear_id(head_pack.arch.num_layers).key
    heads = [int(x.shape[0]) for lid, x in client.received if lid == key]
    layer0 = [int(x.shape[0]) for lid, x in client.received if lid == "0.qkv"]
    # the head: one item per pass (prompt pass: its last row), then R rows on every decode pass, as the layers
    assert len(heads) == len(layer0) == out.completion_tokens
    assert heads[0] == 1 and all(n == rows for n in heads[1:])
    if source == "same" and params.temperature == 0.0:
        assert gen.spec_stats["spec_accepted"] > 0  # verify rows were really used


@pytest.mark.parametrize("pad", [None, 2], ids=["fillers", "padded"])
def test_provider_view_independent_of_drafts(qwen_pack: InMemoryPack, head_pack: MaskedHeadPack,
                                             tiny_tokenizer: ChatTokenizer, pad: int | None) -> None:
    views, accepted = [], []
    for source in ("same", "noise", "lookup"):
        model, client = _masked_head(head_pack)
        kw: dict[str, Any] = {"spec_mask_rows": pad} if pad else {}
        gen = Generator(model, tiny_tokenizer, spec_rows=4, spec_strict=True, **kw, **_source(source, qwen_pack))
        res = LocalMaskReservation(head_pack, DEFAULT_RING, gen.mask_rows_needed(len(PROMPT), 20))
        gen.generate(PROMPT, PARAMS[0], session=res)
        views.append([(lid, int(x.shape[0])) for lid, x in client.received])
        accepted.append(gen.spec_stats["spec_accepted"])
    assert views[0] == views[1] == views[2]
    assert accepted[0] > accepted[1]


def test_plain_decode_with_verify_rows_equals_one_row_steps(head_pack: MaskedHeadPack) -> None:
    """``last_logits`` rows through the masked head equal 1-row decode steps' logits bit for bit."""
    import torch

    from pgw.engine.model import SequenceInput

    ids = [int(t) for t in torch.randint(0, 250, (12,), generator=torch.Generator().manual_seed(4))]
    m = _plain_head(head_pack)
    try:
        c1 = m.new_cache(32)
        m.forward_batch([SequenceInput(ids[:7], c1, None)])
        ref = [m.forward_batch([SequenceInput([t], c1, None)])[0] for t in ids[7:]]  # after ids[7], ..., ids[11]
        c2 = m.new_cache(32)
        m.forward_batch([SequenceInput(ids[:7], c2, None)])
        # a verify pass as the generator builds it: the pending row as a 1-row block, the drafts serial
        got = m.forward_batch([SequenceInput(ids[7:], c2, None, last_logits=5, serial_tail=4)])[0]
        assert len(got) == 5
        for i in range(5):
            assert torch.equal(got[i], ref[i]), i
    finally:
        m.close()
