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
