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
    m = urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/metrics").read().decode()
    assert 'verichat_requests_total{path="/chat",code="200"} 2' in m
    assert 'path="/chat",code="401"} 2' in m and 'code="429"} 1' in m
    assert 'verichat_batches_total{kind="chat",size="1"} 2' in m and "verichat_latency_seconds_total" in m
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


def test_moderation_refuses_flagged_input_and_withholds_flagged_output():
    from vt.moderation import REFUSAL, WITHHELD, Moderator
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    mod = Moderator(lambda t: {"toxic": 0.99 if "idiot" in t else 0.0})
    bot = VeriChat(m, tok, p, layer=1, moderator=mod)
    out = bot.reply_batch([{"messages": [{"role": "user", "content": "you idiot"}], "max_new_tokens": 4},
                           {"messages": [{"role": "user", "content": "hello"}], "max_new_tokens": 4}])
    assert out[0]["text"] == REFUSAL and out[0]["moderation"]["input"]["flagged"]
    assert out[1]["moderation"]["output"]["flagged"] is False
    strict = VeriChat(m, tok, p, layer=1, moderator=Moderator(lambda t: {"toxic": 0.0 if t == "hello" else 0.99}))
    o = strict.reply([{"role": "user", "content": "hello"}], max_new_tokens=4)
    assert o["text"] == WITHHELD and o["withheld"] and o["claims"] == []


def test_risk_head_file_drives_claims_and_sentences(tmp_path):
    from vt.verichat import load_head, load_risk
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    path = tmp_path / "x_risk_head.pt"
    torch.save({"model": "m", "risk_layer": 0, "risk_probe": p.state_dict(), "risk_threshold": 0.4,
                "claim_layer": 1, "claim_probe": p.state_dict()}, path)
    probe, layer = load_head(str(path))
    risk, rl, th = load_risk(str(path))
    assert layer == 1 and rl == 0 and th == 0.4
    torch.save({"model": "m", "risk_layer": 0, "risk_probe": p.state_dict(), "claim_layer": 0, "claim_probe": None},
               tmp_path / "no_claims.pt")
    neg, nl = load_head(str(tmp_path / "no_claims.pt"))       # falls back to 1 - risk
    x = np.random.default_rng(1).normal(size=(3, 32)).astype(np.float32)
    assert nl == 0 and np.allclose(neg.logit(x), -p.logit(x))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    bot = VeriChat(m, tok, probe, layer)
    bot.gen.risk_probe, bot.gen.risk_layer, bot.gen.risk_threshold = risk, rl, th
    out = bot.reply([{"role": "user", "content": "Tell me about Paris."}], mode="flag", max_new_tokens=24)
    assert "sentences" in out and all(0 <= s["risk"] <= 1 for s in out["sentences"])


def test_server_streams_server_sent_events():
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
    body = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "mode": "flag", "stream": True}).encode()
    resp = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/chat", body,
                                                         {"Content-Type": "application/json"}))
    assert resp.headers["Content-Type"] == "text/event-stream"
    events = [json.loads(e[6:]) for e in resp.read().decode().split("\n\n") if e.startswith("data: ")]
    assert events[-1]["done"] and "reply" in events[-1]
    streamed = "".join(e.get("delta", "") for e in events[:-1])
    assert streamed.strip().replace(" [unsure:]", "") == events[-1]["reply"]["text"].replace(" [unsure:]", "")
    srv.shutdown()


def test_server_validates_sampling_overrides():
    import json
    import threading
    import urllib.error
    import urllib.request
    from vt.verichat_server import make_server
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    bot = VeriChat(m, tok, p, layer=1)
    bot.gen.sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    srv = make_server(bot, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def post(body):
        return json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{srv.server_port}/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})).read())
    msgs = [{"role": "user", "content": "Hi"}]
    try:
        for bad in ({"temperature": 3}, {"temperature": "hot"}, {"seed": 1.5}, {"seed": True}):
            try:
                post(dict(bad, messages=msgs))
                raise AssertionError(bad)
            except urllib.error.HTTPError as e:
                assert e.code == 400
        a = post({"messages": msgs, "mode": "flag", "temperature": 1.0, "seed": 5})
        assert a["text"] == post({"messages": msgs, "mode": "flag", "temperature": 1.0, "seed": 5})["text"]
    finally:
        srv.shutdown()


def test_long_chats_and_documents_fit_the_prompt_budget():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    bot = VeriChat(m, tok, p, layer=1, max_prompt_tokens=300)
    from vt.claims import prompt_of
    chat = []
    for k in range(12):
        chat += [{"role": "user", "content": f"Question {k}: tell me about topic {k} " * 3},
                 {"role": "assistant", "content": f"Answer {k} about topic {k}. " * 4}]
    chat.append({"role": "user", "content": "And the last question?"})
    msgs = bot._messages(chat, [])
    assert len(tok(prompt_of(tok, msgs)).input_ids) <= 300
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and msgs[-1] == chat[-1]
    assert len(msgs) < len(chat)                                    # the oldest turns went
    docs = bot._fit_documents(["word " * 2000, "short doc"])
    assert docs[1] == "short doc" and docs[0].endswith("[...]")
    assert len(tok(docs[0], add_special_tokens=False).input_ids) <= 300 // 4 + 4
    out = bot.reply(chat, documents=["word " * 2000], mode="flag", max_new_tokens=8)
    assert isinstance(out["text"], str)


def test_out_of_memory_batches_are_retried_in_halves():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    from vt.guided import DoubtAwareGenerator
    g = DoubtAwareGenerator(m, tok, p, layer=1)
    reqs = [{"messages": [{"role": "user", "content": q}], "max_new_tokens": 8, "mode": "flag"}
            for q in ("Hi", "Who wrote Hamlet?", "Name a colour.")]
    want = [g.generate_batch([r])[0]["text"] for r in reqs]
    real, sizes = g._generate_batch, []

    def tight(requests, *a):                                        # a GPU that fits one sequence
        sizes.append(len(requests))
        if len(requests) > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(requests, *a)
    g._generate_batch = tight
    got = {}
    rs = g.generate_batch(reqs, on_text=lambda i, d: got.__setitem__(i, got.get(i, "") + d))
    assert [r["text"] for r in rs] == want and sizes[0] == 3 and sorted(sizes)[:3] == [1, 1, 1]
    assert all(got.get(i, "").strip() == want[i] for i in range(3))


def test_serving_settings_judge_sentences_by_their_claims(tmp_path):
    import json
    from vt.verichat import load_serving
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    risk = Probe().fit(np.random.default_rng(1).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    head = tmp_path / "h_risk_head.pt"
    cfg = {"claim_calib": [1.0, 0.0], "threshold": 0.99, "likely_right": 0.995, "combine": False, "sentence_doubt": "claims"}
    (tmp_path / "h_risk_head.serving.json").write_text(json.dumps(cfg))
    assert load_serving(str(head)) == cfg and load_serving(str(tmp_path / "other.pt")) == {}
    bot = VeriChat(m, tok, p, layer=1)
    bot.gen.risk_probe, bot.gen.risk_layer, bot.gen.risk_threshold = risk, 1, 0.0   # risk alone would doubt everything
    bot.configure(cfg)
    assert bot.threshold == bot.gen.threshold == 0.99 and bot.gen.combine is False
    out = bot.reply([{"role": "user", "content": "Tell me about Paris and Berlin."}], mode="flag", max_new_tokens=40)
    claims = out["claims"]
    for s in out["sentences"]:                                # doubted exactly when a doubted claim lies inside
        inside = [c for c in claims if s["start"] <= c["start"] < s["end"]]
        if not any(c["doubted"] for c in inside):
            assert not s["doubted"]
    assert all(c["verdict"] != "likely right" or c["score"] >= 0.995 for c in claims)
    bot.configure({}, threshold=0.3)                          # an explicit threshold wins
    assert bot.threshold == 0.3


def test_wikipedia_retriever_parses_caches_and_fails_quietly():
    from vt.retrieval import Wikipedia
    calls = []

    def fake(url):
        calls.append(url)
        if "search/page" in url:
            return {"pages": [{"key": "Gustave_Eiffel", "title": "Gustave Eiffel"}, {"key": "Eiffel_Tower", "title": "Eiffel Tower"}]}
        title = url.rsplit("/", 1)[-1]
        return {"title": title.replace("_", " "), "extract": f"{title} lead section."}
    w = Wikipedia(k=2, fetch=fake)
    out = w("Who designed the Eiffel Tower?")
    assert out == ["Gustave Eiffel (Wikipedia): Gustave_Eiffel lead section.", "Eiffel Tower (Wikipedia): Eiffel_Tower lead section."]
    n = len(calls)
    assert w("who designed the eiffel tower") == out and len(calls) == n          # cached (normalised query)

    def broken(url):
        raise OSError("429 Too Many Requests")
    assert Wikipedia(fetch=broken)("anything") == []
