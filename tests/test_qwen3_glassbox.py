"""Exact attribution on the Qwen3 architecture (per-head q/k RMS norm, head_dim * heads != hidden).
Needs transformers >= 4.51; skipped otherwise."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3ForCausalLM"):
    pytest.skip("transformers without Qwen3", allow_module_level=True)

from vt.qwen_glassbox import explain, explain_all  # noqa: E402


def tiny(attn="eager"):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,   # 4*16 != 32
                                   max_position_embeddings=64, attn_implementation=attn)
    return transformers.Qwen3ForCausalLM(cfg).eval()


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_qwen3_explain_is_exact(attn):
    m = tiny(attn)
    ids = torch.randint(64, (1, 10))
    parts, logit = explain(m, ids, 7, 5)
    assert abs(sum(parts.values()) - logit) < 1e-4


def test_qwen3_explain_all_is_exact():
    m = tiny("sdpa")
    ids = torch.randint(64, (1, 12))
    groups = ["doc"] * 5 + ["q"] * 3 + ["reply"] * 4
    names, c, logits = explain_all(m, ids, groups, start=7)
    assert torch.allclose(c.sum(0), logits, atol=1e-4)
