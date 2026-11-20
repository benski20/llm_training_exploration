"""Shared pytest fixtures for all PGW test suites.

Fixtures (session-scoped unless noted):

``tiny_hf_dir`` / ``tiny_llama_hf_dir``
    A tiny random Qwen2 (resp. Llama) checkpoint saved with ``save_pretrained`` plus a byte-level
    tokenizer with a Qwen-style chat template. 2 layers, hidden 64, GQA 4/2 heads, vocab 260.
``tiny_hf_model``
    The same Qwen2 model loaded in float32 eval mode (reference logits).
``tiny_linear_weights``
    ``{linear_id: pgw.modelpack.LinearWeights}`` computed directly from the HF state dict with
    :func:`pgw.core.quantize_weight_rows` (does not depend on ``pgw.modelpack`` being implemented).
``tiny_pack_dir`` / ``tiny_pack``
    The tiny checkpoint packed with :func:`pgw.modelpack.pack_checkpoint` / opened with
    :meth:`pgw.modelpack.ModelPack.open`. Tests using them are skipped until modelpack is implemented.
``qwen05b_dir``
    Real ``Qwen/Qwen2.5-0.5B-Instruct`` snapshot (network; skipped with ``PGW_OFFLINE=1``).
``ring`` (function)
    Parametrized over the three ring presets.
``seed_expander`` (function)
    Deterministic :class:`pgw.core.SeedExpander`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

# Dev/staging escape hatch for the whole test suite: most fixtures open throw-away stores without an
# external monotonic counter. Tests of the production default (refusal) pass the setting explicitly.
os.environ.setdefault("PGW_MASKSTORE_ALLOW_NO_EXTERNAL_COUNTER", "true")
# Without CUDA, Triton kernels (pgw.core.triton_gemv) run in the Triton interpreter on CPU tensors. The mode
# must be chosen before triton is first imported (its own library functions are jitted at import time).
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    LinearId,
    RingParams,
    SeedExpander,
    linear_ids,
    quantize_weight_rows,
)

TINY_VOCAB = 260  # 256 byte tokens + 4 specials
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
TINY_ARCH: dict[str, Any] = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": TINY_VOCAB,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "bos_token_id": 256,
    "eos_token_id": 258,
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``gpu`` tests without CUDA and ``network`` tests when ``PGW_OFFLINE=1``."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_net = pytest.mark.skip(reason="PGW_OFFLINE=1")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
        if "network" in item.keywords and os.environ.get("PGW_OFFLINE") == "1":
            item.add_marker(skip_net)


# ---- tokenizer -------------------------------------------------------------------------------
def _save_byte_tokenizer(out: Path) -> None:
    """Byte-level tokenizer: ids 0..255 are raw bytes, 256..259 are special tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], ignore_merges=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
    )
    fast.chat_template = QWEN_CHAT_TEMPLATE
    fast.save_pretrained(out)


def _save_tiny(out: Path, arch: str, seed: int) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    if arch == "qwen2":
        cfg = Qwen2Config(**TINY_ARCH)
        model = Qwen2ForCausalLM(cfg)
        with torch.no_grad():  # make q/k/v biases non-trivial
            for layer in model.model.layers:
                for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                    proj.bias.normal_(0, 0.02)
    else:
        cfg = LlamaConfig(**TINY_ARCH)
        model = LlamaForCausalLM(cfg)
    with torch.no_grad():  # random (non-unit) norm weights so norms are actually exercised
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    model.eval().save_pretrained(out, safe_serialization=True)
    _save_byte_tokenizer(out)
    return out


@pytest.fixture(scope="session")
def tiny_hf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny random Qwen2 checkpoint directory (HF format + tokenizer)."""
    return _save_tiny(tmp_path_factory.mktemp("tiny_qwen2"), "qwen2", seed=0)


@pytest.fixture(scope="session")
def tiny_llama_hf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny random Llama checkpoint directory (HF format + tokenizer)."""
    return _save_tiny(tmp_path_factory.mktemp("tiny_llama"), "llama", seed=1)


@pytest.fixture(scope="session")
def tiny_hf_model(tiny_hf_dir: Path) -> Any:
    """The tiny Qwen2 model loaded in float32 eval mode."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(tiny_hf_dir, dtype=torch.float32).eval()


@pytest.fixture(scope="session")
def tiny_linear_weights(tiny_hf_model: Any) -> dict[str, Any]:
    """``{"0.qkv": LinearWeights, ...}`` built straight from the HF state dict."""
    from pgw.modelpack.arch import hf_linear_parts
    from pgw.modelpack.loader import LinearWeights

    sd = tiny_hf_model.state_dict()
    arch = tiny_hf_model.config.model_type
    out: dict[str, Any] = {}
    for lid in linear_ids(tiny_hf_model.config.num_hidden_layers):
        parts = hf_linear_parts(arch, lid.layer, lid.kind)
        w = torch.cat([sd[wn] for _, wn, _ in parts], dim=0)
        biases = [sd[bn] for _, _, bn in parts if bn is not None]
        w8, sw = quantize_weight_rows(w)
        bias = torch.cat(biases).float() if biases else None
        out[lid.key] = LinearWeights(lid, w8, sw, bias)
    return out


@pytest.fixture(scope="session")
def tiny_pack_dir(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny Qwen2 checkpoint packed with ``pgw.modelpack.pack_checkpoint`` (skips if unimplemented)."""
    from pgw.modelpack import pack_checkpoint

    out = tmp_path_factory.mktemp("tiny_pack") / "pack"
    try:
        pack_checkpoint(tiny_hf_dir, out, name="tiny-qwen2")
    except NotImplementedError:
        pytest.skip("pgw.modelpack.pack_checkpoint not implemented yet")
    return out


@pytest.fixture(scope="session")
def tiny_pack(tiny_pack_dir: Path) -> Any:
    """Opened :class:`pgw.modelpack.ModelPack` for the tiny checkpoint (skips if unimplemented)."""
    from pgw.modelpack import ModelPack

    try:
        return ModelPack.open(tiny_pack_dir, verify="full")
    except NotImplementedError:
        pytest.skip("pgw.modelpack.ModelPack.open not implemented yet")


@pytest.fixture(scope="session")
def qwen05b_dir() -> Path:
    """Local snapshot of Qwen/Qwen2.5-0.5B-Instruct (network)."""
    if os.environ.get("PGW_OFFLINE") == "1":
        pytest.skip("PGW_OFFLINE=1")
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct",
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"]))
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"cannot download Qwen2.5-0.5B-Instruct: {exc!r}")


# ---- small helpers -------------------------------------------------------------------------------
@pytest.fixture(params=[RING_K32_A8, RING_K40_A12, RING_K48_A20], ids=lambda p: p.name)
def ring(request: pytest.FixtureRequest) -> RingParams:
    """Each ring preset."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def seed_expander() -> SeedExpander:
    """Deterministic seed expander (seed = bytes 0..31)."""
    return SeedExpander(bytes(range(32)))


def tiny_config_json(path: Path) -> dict[str, Any]:
    """Read ``config.json`` of a checkpoint directory (helper for tests)."""
    return json.loads((path / "config.json").read_text())


def all_linear_ids(num_layers: int = TINY_ARCH["num_hidden_layers"]) -> list[LinearId]:
    """Linear ids of the tiny model."""
    return linear_ids(num_layers)
