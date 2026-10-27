import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from vt.qwen_glassbox import convert, explain, layer_contributions


def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      attn_implementation="eager")
    return Qwen2ForCausalLM(cfg).eval()


def test_dense_explain_is_exact():
    m = tiny()
    ids = torch.randint(64, (1, 10))
    parts, logit = explain(m, ids, 7, 5)
    assert abs(sum(parts.values()) - logit) < 1e-4
    assert len(parts) == 10 + 3


def test_grouped_explain_is_exact():
    m = tiny()
    ids = torch.randint(64, (1, 10))
    groups = ["prompt"] * 4 + [f"q@{j}" for j in range(4, 10)]
    parts, logit = explain(m, ids, 9, 3, groups)
    full, _ = explain(m, ids, 9, 3)
    assert abs(sum(parts.values()) - logit) < 1e-4
    assert abs(parts["prompt"] - sum(full[f"token@{j}"] for j in range(4))) < 1e-4


def test_layer_contributions_are_exact():
    m = tiny()
    ids = torch.randint(64, (1, 10))
    parts, logit = layer_contributions(m, ids, 6, 2)
    assert abs(sum(parts.values()) - logit) < 1e-4
    assert len(parts) == 1 + 2 * 3


def test_converted_explain_is_exact():
    m = tiny()
    convert(m, n_features=128, k=8)
    ids = torch.randint(64, (1, 10))
    parts, logit = explain(m, ids, 8, 4)
    assert abs(sum(parts.values()) - logit) < 1e-4
