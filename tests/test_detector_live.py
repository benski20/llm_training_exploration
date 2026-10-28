import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from vt.correctness import Probe
from vt.detector import Detector


def _det():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(20), np.ones(20)].astype(np.float32)
    probe = Probe().fit(rng.normal(size=(40, 32)).astype(np.float32), y).state_dict()
    stack = Probe().fit(rng.normal(size=(40, 6)).astype(np.float32), y).state_dict()
    return Detector(m, tok, {"response_layer": 1, "token_layer": 0, "response_probe": probe, "token_probe": probe,
                             "stacker": stack, "threshold": 0.0})


def test_judge_scores_every_response_token_with_offsets():
    det = _det()
    resp = "The Eiffel Tower was finished in 1925 by Napoleon."
    out = det.judge([{"role": "user", "content": "When was it finished?"}], resp,
                    documents=["The Eiffel Tower was completed in 1889."], token_threshold=0.0)
    assert 0.0 <= out["p_hallucinated"] <= 1.0 and isinstance(out["flagged"], bool)
    assert "".join(t["text"] for t in out["tokens"]) == resp       # tokens tile the response exactly
    assert all(resp[s["start"]:s["end"]] == s["text"] for s in out["spans"])
    assert out["spans"] and out["spans"][0]["start"] == 0         # threshold 0: everything flagged, one span


def test_judge_without_documents_and_long_prompt_trimmed():
    det = _det()
    out = det.judge([{"role": "user", "content": "word " * 300}], "Paris.", max_tokens=64)
    assert "".join(t["text"] for t in out["tokens"]) == "Paris."
