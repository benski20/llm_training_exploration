"""Real-model checks on Qwen/Qwen2.5-0.5B-Instruct (network download; skipped with PGW_OFFLINE=1)."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12
from pgw.engine import (
    ChatTokenizer,
    FloatBackend,
    Generator,
    InMemoryPack,
    LocalComputeClient,
    LocalMaskSource,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from tests.engine.conftest import dequantize_into_hf

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def q_pack(qwen05b_dir: Path) -> InMemoryPack:
    return InMemoryPack.from_hf(qwen05b_dir, name="qwen2.5-0.5b-instruct")


@pytest.fixture(scope="module")
def q_tok(qwen05b_dir: Path) -> ChatTokenizer:
    return ChatTokenizer(qwen05b_dir)


@pytest.fixture(scope="module")
def prompt_ids(q_tok: ChatTokenizer) -> list[int]:
    return q_tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}])


def test_float_backend_matches_hf_logits(qwen05b_dir: Path, q_pack: InMemoryPack, prompt_ids: list[int]) -> None:
    from transformers import AutoModelForCausalLM

    hf: Any = AutoModelForCausalLM.from_pretrained(qwen05b_dir, dtype=torch.float32).eval()
    dequantize_into_hf(hf, q_pack)
    with torch.no_grad():
        ref = hf(torch.tensor([prompt_ids])).logits[0]
    del hf
    gc.collect()
    model = TransformerModel(q_pack, FloatBackend(q_pack, RING_K40_A12))
    out = model.forward(torch.tensor(prompt_ids), model.new_cache(64), all_logits=True)
    assert float((out - ref).abs().max()) < 5e-3
    assert torch.equal(out.argmax(-1), ref.argmax(-1))


def test_plain_greedy_answer_and_masked_bit_exact(
    q_pack: InMemoryPack, q_tok: ChatTokenizer, prompt_ids: list[int]
) -> None:
    ring = RING_K40_A12
    p = SamplingParams(max_tokens=8, temperature=0)
    plain = Generator(TransformerModel(q_pack, PlainBackend(q_pack, ring)), q_tok)
    rp = plain.generate(prompt_ids, p)
    assert "Paris" in rp.text

    client = LocalComputeClient(q_pack, backend="cpu_f32")
    src = LocalMaskSource(q_pack, backend="cpu_f32")
    masked_model = TransformerModel(q_pack, MaskedBackend(q_pack, None, client, ring, q_pack.fingerprint))
    gen = Generator(masked_model, q_tok, session_factory=lambda n: src.reserve(q_pack.fingerprint, ring, n))
    rm = gen.generate(prompt_ids, p)
    assert rm.token_ids == rp.token_ids and rm.text == rp.text

    # logits of the first forward pass are bit-identical
    res = src.reserve(q_pack.fingerprint, ring, len(prompt_ids))
    a = plain.model.forward(torch.tensor(prompt_ids), plain.model.new_cache(64))
    b = masked_model.forward(torch.tensor(prompt_ids), masked_model.new_cache(64), session=res)
    assert torch.equal(a, b)
    res.close()
    client.close()
