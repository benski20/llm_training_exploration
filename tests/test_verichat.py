import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from vt.correctness import Probe
from vt.verichat import VeriChat


def test_reply_reports_claims_with_provenance_for_documents():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    bot = VeriChat(m, tok, p, layer=1, threshold=0.0)
    doc = "The Eiffel Tower was completed in 1889. It is 330 metres tall."
    out = bot.reply([{"role": "user", "content": "When was it completed?"}], documents=[doc], mode="flag", max_new_tokens=24)
    assert set(out) >= {"text", "claims", "withheld", "reply_confidence"}
    for c in out["claims"]:                                 # random weights: claims are not guaranteed
        assert 0.0 <= c["doc_share"] <= 1.0 and c["source"] in ("your documents", "the model's memory")
        assert c["verdict"] in ("doubtful", "uncertain", "likely right")
    ids = tok("Paris is lovely.", add_special_tokens=False).input_ids
    msgs = bot._messages([{"role": "user", "content": "Where?"}], [doc])
    prov = bot.provenance(msgs, ids, [0], [doc], "Where?")
    assert 0.0 <= prov[0]["doc_share"] <= 1.0


def test_server_page_health_and_chat():
    import json
    import threading
    import urllib.request
    from vt.verichat_server import make_server
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    srv = make_server(VeriChat(m, tok, p, layer=1), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    assert b"VeriChat" in urllib.request.urlopen(base + "/").read()
    assert json.load(urllib.request.urlopen(base + "/health"))["status"] == "ok"
    body = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "mode": "flag"}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/chat", body, {"Content-Type": "application/json"})))
    assert "text" in r and "claims" in r
    bad = urllib.request.Request(base + "/chat", json.dumps({"messages": []}).encode(), {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(bad)
        assert False
    except urllib.error.HTTPError as e:
        assert e.code == 400
    srv.shutdown()


def test_doubt_triggered_retrieval_regenerates_from_passages():
    from vt.retrieval import Corpus
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    corpus = Corpus.from_texts(["The Eiffel Tower was designed by Gustave Eiffel's company and completed in 1889.",
                                "Bananas are rich in potassium."])
    q = [{"role": "user", "content": "Who designed the Eiffel Tower?"}]
    always = VeriChat(m, tok, p, layer=1, threshold=0.0, retriever=corpus, retrieve_below=2.0)   # every reply doubted
    out = always.reply(q, mode="flag", max_new_tokens=8)
    assert out["retrieval"]["retrieved"] and "Eiffel" in out["retrieval"]["passages"][0]
    assert "closed_book_text" in out["retrieval"] and "_trigger" not in out
    never = VeriChat(m, tok, p, layer=1, threshold=0.0, retriever=corpus, retrieve_below=-1.0)
    out = never.reply(q, mode="flag", max_new_tokens=8)
    assert out["retrieval"]["retrieved"] is False
    assert "retrieval" not in VeriChat(m, tok, p, layer=1).reply(q, mode="flag", max_new_tokens=4)


def test_bm25_corpus_ranks_the_matching_passage_first():
    from vt.retrieval import Corpus
    c = Corpus.from_texts(["Paris is the capital of France.", "Mount Everest is the highest mountain on Earth.",
                           "The Nile flows through Egypt."])
    assert c("What is the highest mountain?", k=1) == ["Mount Everest is the highest mountain on Earth."]
    assert c("zzzz qqqq") == []


def test_rate_limiter_window():
    from vt.verichat_server import RateLimiter
    r = RateLimiter(2)
    assert r.allow("a", 0) and r.allow("a", 1) and not r.allow("a", 2) and r.allow("b", 2)
    assert r.allow("a", 61) and RateLimiter(0).allow("a")


def test_server_api_key_and_rate_limit():
    import json
    import threading
    import urllib.request
    from vt.verichat_server import make_server
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    srv = make_server(VeriChat(m, tok, p, layer=1), port=0, api_key="s3cret", rate_per_minute=2)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/chat"
    body = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "mode": "flag"}).encode()

    def post(key):
        h = {"Content-Type": "application/json"} | ({"Authorization": f"Bearer {key}"} if key else {})
        try:
            return urllib.request.urlopen(urllib.request.Request(url, body, h)).status
        except urllib.error.HTTPError as e:
            return e.code
    assert post(None) == 401 and post("wrong") == 401
    assert post("s3cret") == 200 and post("s3cret") == 200 and post("s3cret") == 429
    srv.shutdown()


def test_batcher_groups_concurrent_requests_by_kind():
    import threading
    import time
    from vt.verichat_server import Batcher
    calls = []

    def chat(items):
        calls.append(("chat", len(items)))
        time.sleep(0.05)
        return [x * 2 for x in items]

    def bad(items):
        raise ValueError("boom")
    b = Batcher({"chat": chat, "judge": bad}, max_batch=8, window=0.1)
    out = [None] * 5
    ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, b.submit("chat", i, 5))) for i in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == [0, 2, 4, 6, 8] and calls[0] == ("chat", 5)       # one batched call
    assert b.submit_many("chat", [1, 2], 5) == [2, 4]
    try:
        b.submit("judge", 1, 5)
        assert False
    except ValueError:
        pass
    assert b.submit("chat", 3, 5) == 6                              # the worker survives an error


def test_batched_replies_match_single_replies():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    bot = VeriChat(m, tok, p, layer=1, threshold=0.5)
    reqs = [{"messages": [{"role": "user", "content": q}], "mode": md, "max_new_tokens": 12,
             "documents": ["The Eiffel Tower was completed in 1889."] if d else None}
            for q, md, d in (("Hi", "flag", False), ("Who designed the Eiffel Tower and when?", "hedge", True),
                             ("Name three rivers.", "stop", False))]
    batch = bot.reply_batch(reqs)
    for r, b in zip(reqs, batch):
        one = bot.reply(r["messages"], documents=r["documents"], mode=r["mode"], max_new_tokens=12)
        assert one["text"] == b["text"] and [c["text"] for c in one["claims"]] == [c["text"] for c in b["claims"]]
