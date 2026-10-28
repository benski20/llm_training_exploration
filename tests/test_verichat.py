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
