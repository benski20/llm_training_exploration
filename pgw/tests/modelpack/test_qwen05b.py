"""Real-model test: pack Qwen/Qwen2.5-0.5B-Instruct and run one masked linear exactly (network, slow)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from pgw.core import RING_K40_A12, PreparedWeight, quantize_activations, quantize_weight_rows
from pgw.core.masking import mask_activations, unmask_outputs
from pgw.modelpack import ModelPack, pack_checkpoint

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def qwen_pack(qwen05b_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> ModelPack:
    out = tmp_path_factory.mktemp("q05") / "pack"
    m = pack_checkpoint(qwen05b_dir, out, shard_bytes=128 << 20)
    assert m.source.repo == "Qwen/Qwen2.5-0.5B-Instruct"
    return ModelPack.open(out, verify="full")


def test_manifest(qwen_pack):
    a = qwen_pack.arch
    assert (a.arch.value, a.hidden_size, a.num_layers, a.num_heads, a.num_kv_heads, a.head_dim) == (
        "qwen2", 896, 24, 14, 2, 64)
    assert a.tie_word_embeddings and 151645 in a.eos_token_ids
    assert qwen_pack.name == "qwen2.5-0.5b-instruct"
    assert len(qwen_pack.manifest.linear_files()) > 1
    assert qwen_pack.supported_rings() == ["K32A8", "K40A12", "K48A20"]
    assert 300e6 < qwen_pack.size_bytes("linears") < 400e6  # ~358M int8 params + scales


def test_linear_matches_checkpoint_and_masked_matmul_is_exact(qwen_pack, qwen05b_dir):
    with safe_open(str(qwen05b_dir / "model.safetensors"), framework="pt") as f:
        w = torch.cat([f.get_tensor(f"model.layers.5.self_attn.{p}.weight") for p in ("q_proj", "k_proj", "v_proj")])
        b = torch.cat([f.get_tensor(f"model.layers.5.self_attn.{p}.bias") for p in ("q_proj", "k_proj", "v_proj")])
    lw = qwen_pack.load_linear("5.qkv")
    w8, sw = quantize_weight_rows(w)
    assert torch.equal(lw.w8, w8) and torch.equal(lw.scale, sw) and torch.equal(lw.bias, b.float())

    p = RING_K40_A12
    x = torch.randn(3, 896)
    xq, _sx = quantize_activations(x, p)
    r = torch.randint(0, p.modulus, xq.shape)
    pw = PreparedWeight(lw.w8)
    yq = unmask_outputs(pw(mask_activations(xq, r, p), p), pw(r, p), p, in_features=896)
    assert torch.equal(yq, xq @ lw.w8.to(torch.int64).T)


def test_tokenizer_from_pack(qwen_pack):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qwen_pack.tokenizer_dir)
    text = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    assert "<|im_start|>assistant" in text
