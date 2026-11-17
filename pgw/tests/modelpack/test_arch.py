"""HF config -> ArchConfig mapping and the normative linear-part tables."""

from __future__ import annotations

import pytest

from pgw.core.errors import ConfigError
from pgw.core.types import LinearId, LinearKind, ModelArch
from pgw.modelpack import ArchConfig, arch_config_from_hf, expected_linear_shape, hf_linear_parts, hf_plain_tensor_names

QWEN25_05B = {
    "architectures": ["Qwen2ForCausalLM"], "bos_token_id": 151643, "eos_token_id": 151645, "hidden_act": "silu",
    "hidden_size": 896, "intermediate_size": 4864, "max_position_embeddings": 32768, "model_type": "qwen2",
    "num_attention_heads": 14, "num_hidden_layers": 24, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0, "sliding_window": 32768, "tie_word_embeddings": True, "use_sliding_window": False,
    "vocab_size": 151936,
}
LLAMA31_8B = {
    "architectures": ["LlamaForCausalLM"], "attention_bias": False, "bos_token_id": 128000,
    "eos_token_id": [128001, 128008, 128009], "hidden_act": "silu", "hidden_size": 4096,
    "intermediate_size": 14336, "max_position_embeddings": 131072, "mlp_bias": False, "model_type": "llama",
    "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8, "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
    "rope_theta": 500000.0, "tie_word_embeddings": False, "vocab_size": 128256,
}


def test_qwen25_config():
    a = arch_config_from_hf(QWEN25_05B, {"eos_token_id": [151645, 151643], "bos_token_id": 151643})
    assert a.arch == ModelArch.QWEN2
    assert (a.hidden_size, a.num_heads, a.num_kv_heads, a.head_dim) == (896, 14, 2, 64)
    assert a.rope_theta == 1e6 and a.rope_scaling is None
    assert a.tie_word_embeddings and a.eos_token_ids == (151645, 151643) and a.bos_token_id == 151643


def test_llama31_config_classic_rope_layout():
    a = arch_config_from_hf(LLAMA31_8B)
    assert a.arch == ModelArch.LLAMA and a.head_dim == 128 and not a.tie_word_embeddings
    assert a.rope_theta == 500000.0
    assert a.rope_scaling == LLAMA31_8B["rope_scaling"]
    assert a.eos_token_ids == (128001, 128008, 128009)


def test_transformers5_rope_parameters_layout():
    cfg = {k: v for k, v in LLAMA31_8B.items() if k not in ("rope_scaling", "rope_theta")}
    cfg["rope_parameters"] = {**LLAMA31_8B["rope_scaling"], "rope_theta": 500000.0}
    a = arch_config_from_hf(cfg)
    assert a.rope_theta == 500000.0 and a.rope_scaling == LLAMA31_8B["rope_scaling"]
    cfg["rope_parameters"] = {"rope_theta": 1234.0, "rope_type": "default"}
    a = arch_config_from_hf(cfg)
    assert a.rope_theta == 1234.0 and a.rope_scaling is None


def test_legacy_rope_scaling_type_key_and_default_theta():
    cfg = dict(QWEN25_05B)
    del cfg["rope_theta"]
    cfg["rope_scaling"] = {"type": "linear", "factor": 2.0}
    a = arch_config_from_hf(cfg)
    assert a.rope_theta == 10000.0 and a.rope_scaling == {"rope_type": "linear", "factor": 2.0}


def test_unknown_rope_type_is_recorded_with_warning(caplog):
    cfg = dict(QWEN25_05B, rope_scaling={"rope_type": "yarn", "factor": 4.0})
    a = arch_config_from_hf(cfg)
    assert a.rope_scaling == {"rope_type": "yarn", "factor": 4.0}
    assert "yarn" in caplog.text


def test_explicit_head_dim_and_kv_default():
    cfg = dict(LLAMA31_8B, head_dim=64)
    del cfg["num_key_value_heads"]
    a = arch_config_from_hf(cfg)
    assert a.head_dim == 64 and a.num_kv_heads == 32


def test_eos_union_dedup_and_missing():
    a = arch_config_from_hf(dict(QWEN25_05B, eos_token_id=None), {"eos_token_id": 7})
    assert a.eos_token_ids == (7,)
    a = arch_config_from_hf(dict(QWEN25_05B, eos_token_id=[1, 2]), {"eos_token_id": [2, 3]})
    assert a.eos_token_ids == (1, 2, 3)
    with pytest.raises(ConfigError):
        arch_config_from_hf(dict(QWEN25_05B, eos_token_id="x"))


@pytest.mark.parametrize(
    "patch, match",
    [
        ({"model_type": "mistral"}, "unsupported model_type"),
        ({"model_type": None}, "unsupported model_type"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"use_sliding_window": True}, "sliding"),
        ({"num_key_value_heads": 3}, "multiple"),
        ({"hidden_size": 897}, "divisible"),
    ],
)
def test_rejects_unsupported_qwen(patch, match):
    with pytest.raises(ConfigError, match=match):
        arch_config_from_hf({**QWEN25_05B, **patch})


@pytest.mark.parametrize("patch", [{"attention_bias": True}, {"mlp_bias": True}])
def test_rejects_llama_biases(patch):
    with pytest.raises(ConfigError, match="bias"):
        arch_config_from_hf({**LLAMA31_8B, **patch})


def test_missing_required_key():
    cfg = dict(QWEN25_05B)
    del cfg["hidden_size"]
    with pytest.raises(ConfigError, match="hidden_size"):
        arch_config_from_hf(cfg)


def test_expected_shapes_llama8b():
    a = arch_config_from_hf(LLAMA31_8B)
    assert expected_linear_shape(a, LinearId(0, LinearKind.QKV)) == (
        4096 + 2 * 1024, 4096, [("q_proj", 4096), ("k_proj", 1024), ("v_proj", 1024)])
    assert expected_linear_shape(a, LinearId(5, LinearKind.O))[:2] == (4096, 4096)
    assert expected_linear_shape(a, LinearId(5, LinearKind.GATE_UP))[:2] == (2 * 14336, 4096)
    assert expected_linear_shape(a, LinearId(5, LinearKind.DOWN))[:2] == (4096, 14336)


def test_hf_linear_parts_bias_rules():
    q = hf_linear_parts("qwen2", 3, LinearKind.QKV)
    assert [p[0] for p in q] == ["q_proj", "k_proj", "v_proj"]
    assert q[1] == ("k_proj", "model.layers.3.self_attn.k_proj.weight", "model.layers.3.self_attn.k_proj.bias")
    assert all(b is None for _, _, b in hf_linear_parts("qwen2", 0, LinearKind.O))
    assert all(b is None for _, _, b in hf_linear_parts("llama", 0, LinearKind.QKV))
    assert [p[0] for p in hf_linear_parts("llama", 0, LinearKind.GATE_UP)] == ["gate_proj", "up_proj"]


def test_plain_names_tied_and_untied():
    tied = hf_plain_tensor_names(2, True)
    assert "lm_head" not in tied and tied["layers.1.post_attn_norm"] == "model.layers.1.post_attention_layernorm.weight"
    assert len(tied) == 2 + 2 * 2
    assert hf_plain_tensor_names(2, False)["lm_head"] == "lm_head.weight"


def test_arch_config_is_frozen():
    a = arch_config_from_hf(QWEN25_05B)
    assert isinstance(a, ArchConfig)
    with pytest.raises(Exception):
        a.hidden_size = 1  # type: ignore[misc]
