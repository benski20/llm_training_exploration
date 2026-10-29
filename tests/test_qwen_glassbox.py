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


def test_explain_all_is_exact_for_every_position():
    from vt.qwen_glassbox import explain_all
    m = tiny()
    ids = torch.randint(64, (1, 12))
    groups = ["doc"] * 5 + ["question"] * 3 + ["answer"] * 4
    names, contrib, logits = explain_all(m, ids, groups, start=7)
    assert names == ["doc", "question", "answer", "bias"] and contrib.shape == (4, 4)
    assert torch.allclose(contrib.sum(0), logits, atol=1e-4)
    one, logit = explain(m, ids, 9, int(ids[0, 10]), groups)
    assert abs(one["doc"] - float(contrib[0, 2])) < 1e-4


def test_lookback_ratio_is_a_share():
    from vt.qwen_glassbox import lookback_ratio
    m = tiny()
    ids = torch.randint(64, (1, 12))
    r = lookback_ratio(m, ids, ["document"] * 6 + ["instruction"] * 2 + ["response"] * 4, start=7)
    assert r.shape == (4,) and bool(((r >= 0) & (r <= 1)).all())


def test_dense_dequantizes_4bit_weights(monkeypatch):
    import sys
    import types

    import torch
    from vt.qwen_glassbox import _dense
    full = torch.randn(6, 4)
    lin = torch.nn.Linear(4, 6, bias=False)
    assert torch.equal(_dense(lin), lin.weight.float())            # an ordinary layer: its own weight
    packed = torch.zeros(12, 1, dtype=torch.uint8)                  # what a bitsandbytes 4-bit layer stores
    q = types.SimpleNamespace(weight=types.SimpleNamespace(data=packed, quant_state="state"))
    fake = types.ModuleType("bitsandbytes.functional")
    fake.dequantize_4bit = lambda data, state: full if data is packed and state == "state" else None
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", fake)
    assert torch.equal(_dense(q), full)
