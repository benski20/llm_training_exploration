import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from vt.correctness import Probe
from vt.guided import STOP_NOTE, DoubtAwareGenerator


def setup(bias):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    p.w, p.b = p.w * 0, p.b * 0 + bias                       # a probe that always says sigmoid(bias)
    return tok, m, p


def test_flag_records_claims_and_stop_withholds_on_doubt():
    tok, m, p = setup(-5.0)                                  # P(correct) ~ 0.007 everywhere
    g = DoubtAwareGenerator(m, tok, p, layer=1, threshold=0.5, combine=False)
    msgs = [{"role": "user", "content": "hi"}]
    flag = g.generate(msgs, 30, "flag")
    stop = g.generate(msgs, 30, "stop")
    if flag["claims"]:                                       # random weights: claims are not guaranteed
        assert all(c["doubted"] for c in flag["claims"])
        assert stop["stopped_on_doubt"] and stop["text"].endswith(STOP_NOTE)
        assert len(stop["text"]) <= len(flag["text"]) + len(STOP_NOTE) + 1
    sure = DoubtAwareGenerator(m, tok, setup(5.0)[2], layer=1, threshold=0.5, combine=False).generate(msgs, 30, "stop")
    assert not sure["stopped_on_doubt"]
    assert sure["text"] == flag["text"]                      # same greedy text when nothing is doubted


def test_claim_detection_skips_context_words():
    tok, m, p = setup(0.0)
    g = DoubtAwareGenerator(m, tok, p, layer=1)
    ctx = {"chipmunks", "the"}
    assert g._is_claim_start(" Jim", True, ctx)
    assert not g._is_claim_start(" Chipmunks", True, ctx)
    assert not g._is_claim_start(" the", True, ctx) and not g._is_claim_start("son", False, ctx)


def test_risk_probe_scores_every_token_and_groups_sentences():
    tok, m, p = setup(5.0)                                   # claims always look right
    risky = setup(-5.0)[2]
    risky.w, risky.b = risky.w * 0, risky.b * 0 + 5.0        # risk probe: P(hallucinated) ~ 0.993 everywhere
    safe = setup(5.0)[2]
    safe.w, safe.b = safe.w * 0, safe.b * 0 - 5.0
    msgs = [{"role": "user", "content": "Tell me a story."}]
    flag = DoubtAwareGenerator(m, tok, p, layer=1, combine=False, risk_probe=safe, risk_layer=0,
                               risk_threshold=0.5).generate(msgs, 30, "flag")
    assert flag["sentences"] and all(not s_["doubted"] for s_ in flag["sentences"])
    for s_ in flag["sentences"]:                             # spans point into the text
        assert flag["text"][s_["start"]:s_["end"]] == s_["text"]
    stop = DoubtAwareGenerator(m, tok, p, layer=1, combine=False, risk_probe=risky, risk_layer=0,
                               risk_threshold=0.5).generate(msgs, 30, "stop")
    assert stop["stopped_on_doubt"] and stop["text"].endswith(STOP_NOTE) and not stop["sentences"]
    batch = DoubtAwareGenerator(m, tok, p, layer=1, combine=False, risk_probe=safe, risk_layer=0).generate_batch(
        [{"messages": msgs, "max_new_tokens": 30, "mode": "flag"}, {"messages": msgs, "max_new_tokens": 12, "mode": "flag"}])
    assert batch[0]["text"] == flag["text"]
