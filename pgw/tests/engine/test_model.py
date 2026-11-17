"""TransformerModel vs Hugging Face transformers (Qwen2 and Llama, incl. Llama-3 rope scaling)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, RING_K48_A20, ConfigError, ShapeError
from pgw.engine import (
    ContextOverflowError,
    FloatBackend,
    InMemoryPack,
    PlainBackend,
    SequenceInput,
    TransformerModel,
    arch_config_from_hf,
)
from pgw.engine.backends import LinearBackend, LinearCall
from tests.engine.conftest import dequantize_into_hf, save_tiny_variant

IDS = [256, 72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]


def hf_logits(model_dir: Path, pack: InMemoryPack, ids: list[int]) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    dequantize_into_hf(hf, pack)
    with torch.no_grad():
        return hf(torch.tensor([ids])).logits[0]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def llama3_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tiny_variant(
        tmp_path_factory.mktemp("llama3"), "llama", seed=3, tie_word_embeddings=False,
        rope_parameters={"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
                         "high_freq_factor": 4.0, "original_max_position_embeddings": 32},
    )


@pytest.fixture(scope="module")
def qwen_untied_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tiny_variant(tmp_path_factory.mktemp("qwen_untied"), "qwen2", seed=4, tie_word_embeddings=False,
                             num_attention_heads=8, num_key_value_heads=2)


@pytest.mark.parametrize("which", ["qwen", "llama", "llama3", "qwen_untied"])
def test_float_backend_matches_hf(which: str, request: pytest.FixtureRequest) -> None:
    d: Path = {
        "qwen": lambda: request.getfixturevalue("tiny_hf_dir"),
        "llama": lambda: request.getfixturevalue("tiny_llama_hf_dir"),
        "llama3": lambda: request.getfixturevalue("llama3_dir"),
        "qwen_untied": lambda: request.getfixturevalue("qwen_untied_dir"),
    }[which]()
    pack = InMemoryPack.from_hf(d)
    if which == "llama3":
        assert pack.arch.rope_scaling is not None and pack.arch.rope_scaling["rope_type"] == "llama3"
        assert not pack.arch.tie_word_embeddings
    ids = IDS * 3  # 42 tokens: crosses the llama3 original context (32)
    ref = hf_logits(d, pack, ids)
    model = TransformerModel(pack, FloatBackend(pack, RING_K40_A12))
    cache = model.new_cache(64)
    out = model.forward(torch.tensor(ids), cache, all_logits=True)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    assert cache.length == len(ids)
    last = TransformerModel(pack, FloatBackend(pack, RING_K40_A12)).forward(torch.tensor(ids), model.new_cache(64))
    torch.testing.assert_close(last, ref[-1], rtol=1e-4, atol=1e-4)


def test_incremental_decode_equals_full_prefill(qwen_pack: InMemoryPack) -> None:
    model = TransformerModel(qwen_pack, FloatBackend(qwen_pack, RING_K40_A12))
    full = model.forward(torch.tensor(IDS), model.new_cache(32), all_logits=True)
    cache = model.new_cache(32)
    parts = [model.forward(torch.tensor(IDS[:5]), cache, all_logits=True)]
    for t in IDS[5:9]:
        parts.append(model.forward(torch.tensor([t]), cache)[None])
    parts.append(model.forward(torch.tensor(IDS[9:]), cache, all_logits=True))
    torch.testing.assert_close(torch.cat(parts), full, rtol=1e-5, atol=1e-5)


def test_plain_backend_close_to_hf(tiny_hf_dir: Path, qwen_pack: InMemoryPack) -> None:
    ref = hf_logits(tiny_hf_dir, qwen_pack, IDS)
    for ring, tol in ((RING_K40_A12, 2e-2), (RING_K48_A20, 1e-4)):
        model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, ring))
        out = model.forward(torch.tensor(IDS), model.new_cache(32), all_logits=True)
        assert float((out - ref).abs().max()) < tol, ring.name


def test_plain_incremental_is_bit_exact(qwen_pack: InMemoryPack) -> None:
    """Exact integer linears: per-token quantization makes prefill/decode splits bit-identical
    except for attention reductions, which are deterministic for identical shapes."""
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    c1 = model.new_cache(32)
    a = model.forward(torch.tensor(IDS[:6]), c1)
    c2 = model.new_cache(32)
    b = model.forward(torch.tensor(IDS[:6]), c2)
    assert torch.equal(a, b)


def test_forward_batch_equals_individual(qwen_pack: InMemoryPack) -> None:
    calls_seen: list[int] = []

    class Counting(PlainBackend):
        def forward_many(self, calls):  # type: ignore[no-untyped-def]
            calls_seen.append(len(calls))
            return super().forward_many(calls)

    model = TransformerModel(qwen_pack, Counting(qwen_pack, RING_K40_A12))
    seqs = [IDS[:7], IDS[3:5], IDS]
    singles = [model.forward(torch.tensor(s), model.new_cache(32)) for s in seqs]
    calls_seen.clear()
    caches = [model.new_cache(32) for _ in seqs]
    batched = model.forward_batch([SequenceInput(s, c) for s, c in zip(seqs, caches, strict=True)])
    assert calls_seen == [3] * (4 * qwen_pack.arch.num_layers)  # one backend round per linear group
    for s, b in zip(singles, batched, strict=True):
        torch.testing.assert_close(b, s, rtol=1e-5, atol=1e-5)
    assert [c.length for c in caches] == [len(s) for s in seqs]


async def test_aforward_matches_forward(qwen_pack: InMemoryPack) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    a = model.forward(torch.tensor(IDS), model.new_cache(32))
    b = await model.aforward(torch.tensor(IDS), model.new_cache(32))
    assert torch.equal(a, b)


def test_model_errors(qwen_pack: InMemoryPack) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    with pytest.raises(ShapeError):
        model.forward(torch.tensor([], dtype=torch.long), model.new_cache(8))
    with pytest.raises(ShapeError):
        model.forward(torch.tensor([999999]), model.new_cache(8))
    cache = model.new_cache(4)
    with pytest.raises(ContextOverflowError):
        model.forward(torch.tensor(IDS[:5]), cache)
    assert cache.length == 0
    c = model.new_cache(8)
    with pytest.raises(ConfigError):
        model.forward_batch([SequenceInput([1], c), SequenceInput([2], c)])


def test_failed_backend_leaves_cache_untouched(qwen_pack: InMemoryPack) -> None:
    class Flaky(LinearBackend):
        ring = RING_K40_A12

        def __init__(self) -> None:
            self.inner = PlainBackend(qwen_pack, RING_K40_A12)
            self.fail_at: str | None = None

        def forward(self, call: LinearCall) -> torch.Tensor:
            if call.id.key == self.fail_at:
                raise RuntimeError("boom")
            return self.inner.forward(call)

    be = Flaky()
    model = TransformerModel(qwen_pack, be)
    cache = model.new_cache(32)
    model.forward(torch.tensor(IDS[:4]), cache)
    be.fail_at = "1.down"
    with pytest.raises(RuntimeError):
        model.forward(torch.tensor(IDS[4:6]), cache)
    assert cache.length == 4
    be.fail_at = None
    out = model.forward(torch.tensor(IDS[4:6]), cache)
    ref = model.forward(torch.tensor(IDS[:6]), model.new_cache(32))
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_arch_config_from_hf_variants(tiny_hf_dir: Path) -> None:
    cfg = json.loads((tiny_hf_dir / "config.json").read_text())
    a = arch_config_from_hf(cfg)
    assert a.arch == "qwen2" and a.head_dim == 16 and a.rope_scaling is None and a.eos_token_ids == (258,)
    old_style: dict[str, Any] = {k: v for k, v in cfg.items() if k != "rope_parameters"}
    old_style.update(rope_theta=1234.0, rope_scaling={"type": "linear", "factor": 2.0}, eos_token_id=[1, 2])
    b = arch_config_from_hf(old_style, {"eos_token_id": [2, 3]})
    assert b.rope_theta == 1234.0 and b.rope_scaling == {"type": "linear", "factor": 2.0}
    assert b.eos_token_ids == (1, 2, 3)
    with pytest.raises(ConfigError):
        arch_config_from_hf({**cfg, "model_type": "gpt2"})
    with pytest.raises(ConfigError):
        arch_config_from_hf({**cfg, "hidden_act": "gelu"})


def test_in_memory_pack_contract(qwen_pack: InMemoryPack, tiny_linear_weights: dict[str, Any]) -> None:
    assert len(qwen_pack.fingerprint) == 64
    specs = qwen_pack.linear_specs()
    assert [s.id.key for s in specs[:4]] == ["0.qkv", "0.o", "0.gate_up", "0.down"]
    for key, ref in tiny_linear_weights.items():
        lw = qwen_pack.load_linear(key)
        assert torch.equal(lw.w8, ref.w8) and torch.equal(lw.scale, ref.scale)
        assert (lw.bias is None) == (ref.bias is None)
    assert torch.equal(qwen_pack.load_plain("lm_head"), qwen_pack.load_plain("embed_tokens"))
    with pytest.raises(ConfigError):
        qwen_pack.load_plain("nope")
    with pytest.raises(ConfigError):
        qwen_pack.load_linear("9.qkv")
    with pytest.raises(ConfigError):
        InMemoryPack.from_hf(Path("/nonexistent"))
