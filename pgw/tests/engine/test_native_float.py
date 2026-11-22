"""The pinned native float path (``float_ops = "native"``): close to HF and to torch, row-invariant,
masked == plain bit for bit, speculation unchanged, independent of torch's thread count."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pgw.core import DEFAULT_RING, RING_K40_A12, ConfigError
from pgw.engine import (
    ChatTokenizer,
    FloatBackend,
    Generator,
    InMemoryPack,
    LocalMaskReservation,
    PlainBackend,
    SamplingParams,
    SequenceInput,
    TransformerModel,
)
from pgw.engine.model import NativeFloatOps, native_float_available
from tests.engine.conftest import plain_and_masked
from tests.engine.test_model import IDS, hf_logits

pytestmark = pytest.mark.skipif(not native_float_available(), reason="pgw_fast without the native float path")

PROMPT = [256, 72, 105, 33, 10, 72, 105]


def _native(model: TransformerModel) -> TransformerModel:
    model.set_float_ops("native")
    assert isinstance(model.float_ops, NativeFloatOps)
    return model


@pytest.mark.parametrize("which", ["qwen", "llama"])
def test_native_close_to_hf_and_torch(which: str, request: pytest.FixtureRequest) -> None:
    d: Path = request.getfixturevalue("tiny_hf_dir" if which == "qwen" else "tiny_llama_hf_dir")
    pack = InMemoryPack.from_hf(d)
    ids = IDS * 3
    ref = hf_logits(d, pack, ids)
    torch_out = TransformerModel(pack, FloatBackend(pack, RING_K40_A12)).forward(
        torch.tensor(ids), TransformerModel(pack, FloatBackend(pack, RING_K40_A12)).new_cache(64), all_logits=True)
    model = _native(TransformerModel(pack, FloatBackend(pack, RING_K40_A12)))
    out = model.forward(torch.tensor(ids), model.new_cache(64), all_logits=True)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, torch_out, rtol=1e-5, atol=1e-5)


def test_native_rows_invariant(qwen_pack: InMemoryPack) -> None:
    """Every hidden row is the same bits whether the prompt runs as one block or token by token."""
    model = _native(TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12)))
    last = qwen_pack.arch.num_layers - 1
    block = SequenceInput(IDS, model.new_cache(32), capture_hidden=(last,))
    model.forward_batch([block])
    assert block.hidden is not None
    cache = model.new_cache(32)
    rows = []
    for split in ([*IDS[:5]], *[[t] for t in IDS[5:9]], IDS[9:]):
        s = SequenceInput(split, cache, capture_hidden=(last,))
        model.forward_batch([s])
        assert s.hidden is not None
        rows.append(s.hidden[last])
    assert torch.equal(torch.cat(rows), block.hidden[last])


def test_native_masked_equals_plain(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    params = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True, logprobs=2)
    plain, masked, _client = plain_and_masked(qwen_pack, DEFAULT_RING)
    _native(plain)
    _native(masked)
    base = Generator(plain, tiny_tokenizer).generate(PROMPT, params)
    got = Generator(masked, tiny_tokenizer).generate(
        PROMPT, params, session=LocalMaskReservation(qwen_pack, DEFAULT_RING, len(PROMPT) + 16))
    assert got.token_ids == base.token_ids and got.logprobs == base.logprobs
    torch_ref = Generator(TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING)),
                          tiny_tokenizer).generate(PROMPT, params)
    assert torch_ref.token_ids == base.token_ids  # a new reference, same greedy answer on this model


def test_native_speculation_identical(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    params = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True, logprobs=2)
    target = _native(TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING)))
    base = Generator(target, tiny_tokenizer).generate(PROMPT, params)
    draft = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))  # torch-path draft: proposals only
    spec = Generator(target, tiny_tokenizer, draft=draft, spec_tokens=4).generate(PROMPT, params)
    assert spec.token_ids == base.token_ids and spec.logprobs == base.logprobs


def test_native_independent_of_torch_threads(qwen_pack: InMemoryPack) -> None:
    model = _native(TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12)))
    prev = torch.get_num_threads()
    outs = []
    try:
        for n in (1, max(2, prev)):
            torch.set_num_threads(n)
            outs.append(model.forward(torch.tensor(IDS * 2), model.new_cache(64), all_logits=True))
    finally:
        torch.set_num_threads(prev)
    assert torch.equal(outs[0], outs[1])


def test_set_float_ops_validation(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    with pytest.raises(ConfigError):
        model.set_float_ops("fast")
    model.set_float_ops("native")
    model.set_float_ops("torch")
    assert model.float_ops is None
    with pytest.raises(ConfigError):
        TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12), dtype=torch.float64).set_float_ops("native")
    monkeypatch.setenv("PGW_FLOAT_OPS", "native")
    assert isinstance(TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12)).float_ops, NativeFloatOps)
