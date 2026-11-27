"""Qwen3 dense (``model_type = "qwen3"``): per-head q/k RMSNorm before RoPE, no linear biases.

The q/k norm is gateway-side float work between the qkv trip and RoPE, so a Qwen3 pass has the same trips, rows
and provider view as a Llama pass of the same shape. Checked here at a tiny config: the float reference matches
Hugging Face's ``Qwen3ForCausalLM``, the masked path is bit-identical to the plaintext int8 reference (logits and
sampled generation), the packer writes and verifies the norms, and paths without the norm (the native float ops
and the native decode drivers) refuse or step aside instead of computing a different model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A8, RING_K40_A12, RING_K48_A20, ConfigError, RingParams
from pgw.core.types import ModelArch
from pgw.engine import (
    ChatTokenizer,
    FloatBackend,
    Generator,
    InMemoryPack,
    LocalMaskReservation,
    LocalMaskSource,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from pgw.modelpack import ModelPack
from pgw.modelpack.arch import arch_config_from_hf, has_qk_norm, hf_linear_parts, hf_plain_tensor_names
from pgw.modelpack.packer import pack_checkpoint
from tests.engine.conftest import dequantize_into_hf, plain_and_masked

IDS = [256, 72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
PROMPT = [256, 72, 105, 33, 10]


def _save_qwen3(out: Path, seed: int, **overrides: Any) -> Path:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from tests.conftest import TINY_ARCH, _save_byte_tokenizer

    torch.manual_seed(seed)
    # head_dim 24 != hidden / heads (16): Qwen3 sets head_dim on its own, as the real 4B / 32B do
    kw = {**TINY_ARCH, "head_dim": 24, "use_sliding_window": False, "attention_bias": False, **overrides}
    model = Qwen3ForCausalLM(Qwen3Config(**kw))
    with torch.no_grad():  # non-unit norms (q/k norms included) so every norm is exercised
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    model.eval().save_pretrained(out, safe_serialization=True)
    _save_byte_tokenizer(out)
    return out


@pytest.fixture(scope="module")
def qwen3_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _save_qwen3(tmp_path_factory.mktemp("tiny_qwen3"), seed=11)


@pytest.fixture(scope="module")
def qwen3_untied_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _save_qwen3(tmp_path_factory.mktemp("tiny_qwen3_untied"), seed=12, tie_word_embeddings=False,
                       num_attention_heads=8, num_key_value_heads=2, head_dim=16)


@pytest.fixture(scope="module")
def qwen3_pack(qwen3_dir: Path) -> InMemoryPack:
    return InMemoryPack.from_hf(qwen3_dir, name="tiny-qwen3")


def test_arch_tables() -> None:
    assert has_qk_norm("qwen3") and not has_qk_norm("qwen2") and not has_qk_norm("llama")
    # the qwen2 / llama tables are unchanged (with or without the arch argument)
    assert hf_plain_tensor_names(2, True) == hf_plain_tensor_names(2, True, "qwen2") == \
        hf_plain_tensor_names(2, True, "llama")
    names = hf_plain_tensor_names(2, False, "qwen3")
    assert names["layers.1.q_norm"] == "model.layers.1.self_attn.q_norm.weight"
    assert names["layers.0.k_norm"] == "model.layers.0.self_attn.k_norm.weight"
    assert len(names) == len(hf_plain_tensor_names(2, False)) + 4
    assert all(b is None for _, _, b in hf_linear_parts("qwen3", 0, "qkv"))


def test_arch_config_rejects_unsupported(qwen3_dir: Path) -> None:
    import json

    cfg = json.loads((qwen3_dir / "config.json").read_text())
    a = arch_config_from_hf(cfg)
    assert a.arch == ModelArch.QWEN3 and a.head_dim == 24
    for bad in ({"attention_bias": True}, {"use_sliding_window": True}):
        with pytest.raises(ConfigError):
            arch_config_from_hf({**cfg, **bad})


@pytest.mark.parametrize("which", ["tied", "untied"])
def test_float_backend_matches_hf(which: str, request: pytest.FixtureRequest) -> None:
    from transformers import AutoModelForCausalLM

    d: Path = request.getfixturevalue("qwen3_dir" if which == "tied" else "qwen3_untied_dir")
    pack = InMemoryPack.from_hf(d)
    assert pack.arch.arch == ModelArch.QWEN3
    ids = IDS * 3
    hf = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    dequantize_into_hf(hf, pack)
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0]
    model = TransformerModel(pack, FloatBackend(pack, RING_K40_A12))
    assert model.qk_norms is not None and len(model.qk_norms) == pack.arch.num_layers
    out = model.forward(torch.tensor(ids), model.new_cache(64), all_logits=True)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    # the norm matters: without it the same weights give different logits
    model.qk_norms = None
    off = model.forward(torch.tensor(ids), model.new_cache(64), all_logits=True)
    assert not torch.allclose(off, ref, rtol=1e-3, atol=1e-3)


def test_incremental_decode_equals_full_prefill(qwen3_pack: InMemoryPack) -> None:
    """Prefill / decode splits agree (to float rounding: attention reductions differ by shape, as for qwen2), and
    a repeated pass is bit-identical."""
    model = TransformerModel(qwen3_pack, PlainBackend(qwen3_pack, RING_K40_A12))
    ids = torch.tensor(IDS)
    full = model.forward(ids, model.new_cache(32), all_logits=True)
    assert torch.equal(full, model.forward(ids, model.new_cache(32), all_logits=True))
    cache = model.new_cache(32)
    model.forward(ids[:5], cache)
    rows = torch.stack([model.forward(ids[i : i + 1], cache) for i in range(5, len(IDS))])
    torch.testing.assert_close(rows, full[5:], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("ring", [RING_K32_A8, RING_K40_A12, RING_K48_A20], ids=lambda r: r.name)
def test_masked_logits_bit_identical(qwen3_pack: InMemoryPack, ring: RingParams) -> None:
    plain, masked, client = plain_and_masked(qwen3_pack, ring)
    res = LocalMaskReservation(qwen3_pack, ring, 32)
    ids = torch.tensor(IDS * 2)
    a = plain.forward(ids, plain.new_cache(32), all_logits=True)
    b = masked.forward(ids, masked.new_cache(32), session=res, all_logits=True)
    assert torch.equal(a, b)
    # the q/k norm adds no trip: one per linear group, as for qwen2 / llama
    assert client.requests == 4 * qwen3_pack.arch.num_layers


def test_masked_generation_bit_identical(qwen3_pack: InMemoryPack, qwen3_dir: Path) -> None:
    tok = ChatTokenizer(qwen3_dir)
    ring = RING_K40_A12
    plain, masked, client = plain_and_masked(qwen3_pack, ring)
    p = SamplingParams(max_tokens=8, temperature=0.9, seed=7, logprobs=0, ignore_eos=True)
    rp = Generator(plain, tok).generate(PROMPT, p)
    sess = LocalMaskSource(qwen3_pack).reserve(qwen3_pack.fingerprint, ring, 64)
    rm = Generator(masked, tok).generate(PROMPT, p, session=sess)
    assert rm.token_ids == rp.token_ids and rm.logprobs == rp.logprobs
    assert client.requests == 4 * qwen3_pack.arch.num_layers * p.max_tokens


def test_packer_roundtrip(qwen3_dir: Path, qwen3_pack: InMemoryPack, tmp_path: Path) -> None:
    man = pack_checkpoint(qwen3_dir, tmp_path / "pack", name="tiny-qwen3")
    assert man.arch.arch == ModelArch.QWEN3
    names = man.plain_tensor_names()
    assert "layers.0.q_norm" in names and "layers.1.k_norm" in names
    assert tuple(man.tensors["layers.0.q_norm"].shape) == (24,)
    pack = ModelPack.open(tmp_path / "pack", verify="full")
    try:
        a = TransformerModel(pack, PlainBackend(pack, RING_K40_A12))
        b = TransformerModel(qwen3_pack, PlainBackend(qwen3_pack, RING_K40_A12))
        ids = torch.tensor(IDS)
        assert torch.equal(a.forward(ids, a.new_cache(32), all_logits=True),
                           b.forward(ids, b.new_cache(32), all_logits=True))
    finally:
        pack.close()


def test_manifest_requires_qk_norms(qwen3_dir: Path, tmp_path: Path) -> None:
    man = pack_checkpoint(qwen3_dir, tmp_path / "pack")
    man.check_consistency()
    broken = man.model_copy(update={"tensors": {k: v for k, v in man.tensors.items() if k != "layers.1.k_norm"}})
    with pytest.raises(ConfigError, match="k_norm"):
        broken.check_consistency()


def test_native_paths_refuse(qwen3_pack: InMemoryPack) -> None:
    model = TransformerModel(qwen3_pack, PlainBackend(qwen3_pack, RING_K40_A12))
    for name in ("native", "native-blocked"):
        with pytest.raises(ConfigError, match="q/k norm"):
            model.set_float_ops(name)
    assert model.float_ops is None


def test_fused2_masked_equals_plain(qwen3_pack: InMemoryPack) -> None:
    """fused2 (2L + 1 trips): the q/k norm sits in the gateway's attention step between the pair trips; masked ==
    plain bit for bit, and the norm is applied there (the fused2 reference with and without it differ)."""
    from pgw.engine import SequenceInput
    from pgw.engine.fused2 import (
        Fused2MaskedBackend,
        Fused2MaskSource,
        Fused2PlainBackend,
        Fused2Provider,
        Fused2Weights,
        enable_fused2,
    )

    ring = RingParams(48, 12, 45)
    w = Fused2Weights(qwen3_pack)

    def run(backend: Any, norms: bool = True) -> list[torch.Tensor]:
        m = TransformerModel(qwen3_pack, backend)
        if not norms:
            m.qk_norms = None
        enable_fused2(m, w, ring)
        cache = m.new_cache(64)
        out = m.forward_batch([SequenceInput(torch.tensor(IDS[:6]), cache, all_logits=True)])
        for t in IDS[6:9]:
            out += m.forward_batch([SequenceInput(torch.tensor([t]), cache)])
        return out

    ref = run(Fused2PlainBackend(ring))
    be = Fused2MaskedBackend(ring, Fused2Provider(w, ring.k), Fused2MaskSource(w, ring))
    got = run(be)
    assert all(torch.equal(a, b) for a, b in zip(ref, got, strict=True))
    assert be.f2.trips == 4 * (2 * qwen3_pack.arch.num_layers + 1)
    off = run(Fused2PlainBackend(ring), norms=False)
    assert not all(torch.equal(a, b) for a, b in zip(ref, off, strict=True))
