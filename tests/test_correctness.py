import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from vt.correctness import (CorrectnessHead, Probe, auroc, ece, entity_index, features, grounding,
                            is_right, same_answer, stack_features)


def test_metrics():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5           # ties count half
    assert ece([0.0, 1.0], [0, 1]) == 0.0
    assert is_right("It is Paris, France.", ["Paris"]) and not is_right("Lyon", ["Paris"])
    assert same_answer("John Ford", "The director was John Ford") and not same_answer("Ford", "Hawks")


def test_probe_learns_a_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    p = Probe().fit(X[:300], y[:300])
    assert auroc(p.logit(X[300:]), y[300:]) > 0.95


def tiny():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager")
    return tok, Qwen2ForCausalLM(cfg).eval()


def test_entity_index():
    tok, _ = tiny()
    ids = tok("The Chipmunks were created by Jim Henson.").input_ids
    assert tok.decode(ids[entity_index(tok, ids, "Who was the man behind The Chipmunks?")]).strip() == "Jim"


def test_features_match_unbatched_and_grounding_is_exact():
    tok, m = tiny()
    from vt.correctness import chat
    qs = ["Who wrote Hamlet?", "What is the largest planet in the solar system?"]
    ps = [tok(chat(tok, q)).input_ids for q in qs]
    ans = [tok(" William Shakespeare").input_ids, tok(" Jupiter is largest").input_ids]
    both = features(m, tok, ps, ans, qs, [0, 1], batch=2)
    one = features(m, tok, ps[1:], ans[1:], qs[1:], [0, 1], batch=1)
    assert torch.allclose(both[1]["h"].float(), one[0]["h"].float(), atol=1e-2)   # left padding is harmless
    assert abs(both[1]["min_lp"] - one[0]["min_lp"]) < 1e-3
    g = grounding(m, tok, chat(tok, qs[0]), qs[0], ans[0])
    assert g["exact_err"] < 1e-3


def test_head_answers_with_probability():
    tok, m = tiny()
    rng = np.random.default_rng(0)
    probe = Probe().fit(rng.normal(size=(50, 32)).astype(np.float32), rng.integers(0, 2, 50).astype(np.float32))
    row = {"first_p": 0.5, "entity_p": 0.5, "min_lp": -1.0, "mean_lp": -0.5, "consistency": 1.0,
           "question_signed": 1.0, "question_share": 0.2, "template_signed": 0.1}
    Xs = np.stack([stack_features(float(rng.normal()), row, True, True) for _ in range(40)])
    st = Probe().fit(Xs + rng.normal(size=Xs.shape).astype(np.float32), rng.integers(0, 2, 40).astype(np.float32))
    head = CorrectnessHead([1], probe, st, True, True, k_samples=2)
    out = head.answer(m, tok, "Who wrote Hamlet?")
    assert 0.0 <= out["p_correct"] <= 1.0 and "consistency" in out["evidence"]
