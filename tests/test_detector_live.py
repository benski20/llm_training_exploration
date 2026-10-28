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
    assert out["spans"] and out["spans"][0]["start"] == 0         # threshold 0: every word flagged
    assert all(any(ch.isalnum() for ch in s["text"]) for s in out["spans"])


def test_judge_without_documents_and_long_prompt_trimmed():
    det = _det()
    out = det.judge([{"role": "user", "content": "word " * 300}], "Paris.", max_tokens=64)
    assert "".join(t["text"] for t in out["tokens"]) == "Paris."


def test_server_judge_endpoint():
    import json
    import threading
    import urllib.request
    from vt.verichat import VeriChat
    from vt.verichat_server import make_server
    det = _det()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    srv = make_server(VeriChat(det.model, det.tok, p, layer=1), port=0, detector=det)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    assert json.load(urllib.request.urlopen(base + "/health"))["detector"] is True
    body = json.dumps({"messages": [{"role": "user", "content": "Capital of France?"}], "response": "Lyon."}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/judge", body, {"Content-Type": "application/json"})))
    assert set(r) == {"p_hallucinated", "flagged", "spans"}
    two = json.dumps({"items": [json.loads(body), json.loads(body)]}).encode()
    rb = json.load(urllib.request.urlopen(urllib.request.Request(base + "/judge", two, {"Content-Type": "application/json"})))
    assert len(rb["items"]) == 2 and rb["items"][0]["p_hallucinated"] == r["p_hallucinated"]
    bad = urllib.request.Request(base + "/judge", json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(bad)
        assert False
    except urllib.error.HTTPError as e:
        assert e.code == 400
    srv.shutdown()


def test_judge_batch_matches_one_at_a_time():
    det = _det()
    items = [{"messages": [{"role": "user", "content": "Capital of France?"}], "response": "Paris."},
             {"messages": [{"role": "user", "content": "Who wrote Hamlet, and when?"}],
              "response": "William Shakespeare wrote it around 1600.", "documents": ["Hamlet is a tragedy by Shakespeare."]},
             {"messages": [{"role": "user", "content": "x"}], "response": ""}]
    batch = det.judge_batch(items, batch_tokens=10 ** 6)
    assert "error" in batch[2]
    for it, b in zip(items[:2], batch[:2]):
        one = det.judge(it["messages"], it["response"], documents=it.get("documents"))
        assert abs(one["logit"] - b["logit"]) < 1e-3
        assert [t["text"] for t in one["tokens"]] == [t["text"] for t in b["tokens"]]


def test_batched_features_match_training_features():
    """judge_batch hooks the probe layers; training used response_states (output_hidden_states)."""
    from vt.detector import encode, response_states
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(1)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(20), np.ones(20)].astype(np.float32)
    probe = Probe().fit(rng.normal(size=(40, 32)).astype(np.float32), y).state_dict()
    stack = Probe().fit(rng.normal(size=(40, 6)).astype(np.float32), y).state_dict()
    det = Detector(m, tok, {"response_layer": 0, "token_layer": 1, "response_probe": probe, "token_probe": probe,
                            "stacker": stack, "threshold": 0.0})
    msgs, resp = [{"role": "user", "content": "Who wrote Hamlet?"}], "Shakespeare wrote it."
    e = encode(tok, det.item(msgs, resp), 4096)
    h, lp = response_states(m, e["ids"], e["r0"], [0, 1])
    ts = det.tok_probe.logit(h[1].float().numpy())
    resp_z = float(det.resp_probe.logit(h[0].float().mean(0, keepdim=True).numpy())[0])
    f = np.array([[resp_z, ts.max(), ts.mean(), np.percentile(ts, 95), float(lp.mean()), float(lp.min())]])
    assert abs(float(det.stacker.logit(f)[0]) - det.judge(msgs, resp)["logit"]) < 1e-2   # float16 copy in training
