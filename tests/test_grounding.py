"""vt/grounding.py: a probe on the model's own neurons scores each sentence of a reply against the evidence."""
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from vt.grounding import Grounding, exact_lines, sentence_spans


def tiny():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    return m, tok


def probe(tmp_path, coef):
    p = tmp_path / "probe.npz"
    np.savez(p, layers=np.array([1, 3]), mean=np.zeros(128, np.float32), scale=np.ones(128, np.float32),
             coef=coef.astype(np.float32), intercept=np.float32(0), score_mean=np.float32(0), score_std=np.float32(1),
             model=np.array("tiny"))
    return p


def test_sentences_and_exact_lines():
    text = "Paris is the capital. It has 2 million people!\nAnd 17 × 23 = 401."
    assert [text[a:b].strip() for a, b in sentence_spans(text)] == ["Paris is the capital.", "It has 2 million people!",
                                                                    "And 17 × 23 = 401."]
    assert exact_lines(text) == ["17 × 23 = 391"]


def test_every_sentence_scored_from_one_pass(tmp_path):
    m, tok = tiny()
    calls = []
    m.register_forward_pre_hook(lambda mod, args, kw=None: calls.append(1))
    g = Grounding(m, tok, probe(tmp_path, np.ones(128)), flag_z=1e9)
    out = g.check([{"role": "user", "content": "Summarize."}], "It opened in 1889. It is tall.", documents=["A doc."])
    assert len(out["sentences"]) == 2 and len(calls) == 1                 # one forward pass for the whole reply
    assert out["verdict"] == "consistent" and all(s["verdict"] == "consistent" for s in out["sentences"])
    assert not m.model.layers[1].mlp.down_proj._forward_pre_hooks           # hooks removed after reading


def test_flags_follow_the_probe_and_evidence_reaches_the_context(tmp_path):
    m, tok = tiny()
    seen = []
    g = Grounding(m, tok, probe(tmp_path, np.ones(128)), retriever=lambda q: seen.append(q) or ["Found (Wikipedia): x"],
                  flag_z=-1e9)
    assert g.check([{"role": "user", "content": "When?"}], "In 1889.")["verdict"] == "not grounded"
    assert seen == ["When?", "When? In 1889."]                            # the question, then with the reply's words
    ev = g.evidence([{"role": "user", "content": "When?"}], "17 × 23 = 401.", documents=["Doc."])
    assert ev.startswith("When?") and "Doc." in ev and "Exact arithmetic: 17 × 23 = 391" in ev
    assert "does not parse" in g.evidence([{"role": "user", "content": "Code?"}], "```python\ndef f(x)\n  pass\n```")


def test_with_natural_scales_the_evidence_signals_join(tmp_path):
    m, tok = tiny()
    p = tmp_path / "probe.npz"
    np.savez(p, layers=np.array([1, 3]), mean=np.zeros(128, np.float32), scale=np.ones(128, np.float32),
             coef=np.ones(128, np.float32), intercept=np.float32(0), score_mean=np.float32(0), score_std=np.float32(1),
             model=np.array("tiny"), **{f"nat_{k}_{s}": np.float32(v) for k in ("probe", "mean_shift", "max_disagreement")
                                        for s, v in (("mean", 0.0), ("std", 1.0))})
    out = Grounding(m, tok, p, flag_z=1e9).check([{"role": "user", "content": "Summarize."}], "It opened in 1889.",
                                                 documents=["The tower opened in 1889."])
    s = out["sentences"][0]
    assert {"z_probe", "z_shift", "z_disagreement"} <= set(s) and out["verdict"] == "consistent"
