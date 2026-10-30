import numpy as np
from types import SimpleNamespace
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


def _tiny_bot():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    return VeriChat(m, tok, p, layer=1)


def test_batcher_isolates_a_failing_request():
    from vt.verichat_server import Batcher

    def handler(items):
        if any(it == "bad" for it in items):
            raise RuntimeError("boom")
        return [it.upper() for it in items]
    b = Batcher({"x": handler}, max_batch=8, window=0.2)
    import threading
    out = {}

    def go(v):
        try:
            out[v] = b.submit("x", v, 5)
        except RuntimeError as e:
            out[v] = e
    ts = [threading.Thread(target=go, args=(v,)) for v in ("a", "bad", "c")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out["a"] == "A" and out["c"] == "C" and isinstance(out["bad"], RuntimeError)


def test_doubt_trigger_is_the_weakest_claim_and_lookup_uses_it():
    bot = _tiny_bot()
    r = {"claims": [{"score": 0.9, "text": "Paris"}, {"score": 0.2, "text": "Depardieu"}], "first_token": None}
    assert bot.doubt_score(r) == (0.2, "Depardieu")
    assert bot.doubt_score({"claims": [], "first_token": None}) == (None, None)
    queries = []
    bot.retriever = lambda q: (queries.append(q), [f"passage for {q}"])[1]
    bot.retrieve_below = 1.01                                       # look everything up
    bot.reply([{"role": "user", "content": "Who took Russian citizenship?"}], mode="flag", max_new_tokens=8)
    assert queries and queries[0] == "Who took Russian citizenship?"
    assert all(q.startswith("Who took Russian citizenship?") for q in queries) and len(queries) <= 2


def test_reply_survives_a_provenance_failure():
    bot = _tiny_bot()

    def broken(*a, **k):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")
    bot.provenance = broken
    out = bot.reply([{"role": "user", "content": "Tell me about Paris."}], documents=["Paris is the capital of France."],
                    mode="flag", max_new_tokens=12)
    assert isinstance(out["text"], str) and all("source" not in c for c in out["claims"])
    assert "_claim" not in out and "_trigger" not in out


def test_server_answers_500_cleanly_when_the_model_fails():
    import json
    import threading
    import urllib.error
    import urllib.request
    from vt.verichat_server import make_server
    bot = _tiny_bot()

    def broken(reqs, **kw):
        raise RuntimeError("model failure")
    bot.reply_batch = broken
    srv = make_server(bot, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "Hi"}]}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/chat", body,
                                                          {"Content-Type": "application/json"}))
            raise AssertionError("expected a 500")
        except urllib.error.HTTPError as e:
            assert e.code == 500 and "internal error" in json.loads(e.read())["error"]
    finally:
        srv.shutdown()


def test_replies_without_lookup_are_ready_before_the_batch_lookups():
    bot = _tiny_bot()
    events = []
    bot.retriever = lambda q: (events.append("lookup"), ["Some passage."])[1]
    bot.retrieve_below = 1.01                                       # every retrieving request is looked up
    reqs = [{"messages": [{"role": "user", "content": "Hi"}], "mode": "flag", "max_new_tokens": 6, "retrieve": False},
            {"messages": [{"role": "user", "content": "Who wrote Hamlet?"}], "mode": "flag", "max_new_tokens": 6}]
    outs = bot.reply_batch(reqs, on_ready=lambda i, o: events.append(("ready", i)))
    assert events.index(("ready", 0)) < events.index("lookup") < events.index(("ready", 1))
    assert len(outs) == 2 and outs[1]["retrieval"]["retrieved"]


def test_batcher_releases_early_results():
    import threading
    import time as _t
    from vt.verichat_server import Batcher

    def handler(items, ready=None):
        ready(0, "first")
        _t.sleep(1.0)
        return ["first", "second"]
    handler.early = True
    b = Batcher({"x": handler}, max_batch=8, window=0.2)
    got = {}

    def go(v):
        t = _t.time()
        got[v] = (b.submit("x", v, 5), _t.time() - t)
    ts = [threading.Thread(target=go, args=(v,)) for v in ("a", "b")]
    for t in ts:
        t.start()
        _t.sleep(0.01)
    for t in ts:
        t.join()
    (r0, t0), (r1, t1) = got["a"], got["b"]
    assert {r0, r1} == {"first", "second"} and min(t0, t1) < 0.8 < max(t0, t1)


def test_stream_sends_heartbeats_while_the_model_is_slow():
    """A slow reply keeps the stream alive with SSE comments, so the page can tell it from a stuck one."""
    import json
    import threading
    import time as _time
    import urllib.request
    from types import SimpleNamespace
    from vt.verichat_server import make_server

    class Slow:
        gen = SimpleNamespace(layer=1, risk_probe=None, thinking=False, think_open=None, sampling=None, think_sampling=None)
        threshold, retriever = 0.65, None

        def reply_batch(self, reqs, on_text=None, on_reasoning=None, on_ready=None):
            _time.sleep(11)
            return [{"text": "ok", "claims": []} for _ in reqs]

    srv = make_server(Slow(), port=0, rate_per_minute=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()
        r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/chat", body,
                                                          {"Content-Type": "application/json"}), timeout=30)
        text = r.read().decode()
        assert text.startswith(": ping\n\n") and '"done": true' in text
    finally:
        srv.shutdown()


def test_wikipedia_full_text_finds_the_passage_with_the_answer():
    import urllib.parse
    from vt.retrieval import Wikipedia
    film = ("MASH is a 1970 American black comedy war film directed by Robert Altman. " + "plot words " * 120 +
            "Cast: Sally Kellerman as Major Margaret Houlihan, Robert Duvall as Major Frank Burns. " + "other " * 80)
    pages = {"MASH_(film)": film, "Houlihan": "Margaret Houlihan is a character in the M*A*S*H franchise."}

    def fake(url):
        if "search" in url:
            return {"pages": [{"key": "MASH_(film)", "title": "MASH (film)"}, {"key": "Houlihan", "title": "Houlihan"}]}
        if "extracts" in url:
            k = urllib.parse.unquote(url.split("titles=")[1])
            return {"query": {"pages": [{"title": k, "extract": pages[k]}]}}
        raise OSError("no lead endpoint in this test")
    got = Wikipedia(k=2, fetch=fake, full_text=True)("Who played Margaret Houlihan in the film MASH cast?")
    assert any("Sally Kellerman" in p for p in got) and len(got) == 2
    lead_only = Wikipedia(k=2, fetch=fake)("Who played Margaret Houlihan in the film MASH cast?")
    assert lead_only == []                       # the lead endpoint failed: no lookup, quietly


def test_wikipedia_lookup_stops_fetching_when_its_time_budget_is_spent():
    import time as _time
    from vt.retrieval import Wikipedia
    calls = []

    def slow(url):
        calls.append(url)
        _time.sleep(0.3)
        if "search" in url:
            return {"pages": [{"key": "A", "title": "A"}, {"key": "B", "title": "B"}, {"key": "C", "title": "C"}]}
        return {"query": {"pages": [{"title": "A", "extract": "Alpha answer text here."}]}}
    w = Wikipedia(k=2, fetch=slow, full_text=True, budget_s=0.5)
    got = w("alpha answer")
    assert len(calls) <= 3 and got and "Alpha" in got[0]       # stopped early, kept what it had found


def test_wikipedia_rerank_reorders_bm25s_best_passages():
    import urllib.parse
    from vt.retrieval import Wikipedia
    text = "Alpha alpha alpha words. " * 30 + "The answer is Zeta. " + "alpha filler " * 60

    def fake(url):
        if "search" in url:
            return {"pages": [{"key": "P", "title": "P"}]}
        return {"query": {"pages": [{"title": urllib.parse.unquote(url.split("titles=")[1]), "extract": text}]}}
    plain = Wikipedia(k=1, fetch=fake, full_text=True, passage_words=20)("alpha words")
    ranked = Wikipedia(k=1, fetch=fake, full_text=True, passage_words=20,
                       rerank=lambda q, ps: [("Zeta" in p) * 1.0 for p in ps])("alpha words")
    assert "Zeta" not in plain[0] and "Zeta" in ranked[0]
    broken = Wikipedia(k=1, fetch=fake, full_text=True, passage_words=20, rerank=lambda q, ps: 1 / 0)
    assert broken("alpha words") == plain and broken.errors      # a failing reranker keeps BM25's order


def test_final_replies_carry_claim_verification_and_survive_its_failure():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    bot = VeriChat(m, tok, p, layer=1, threshold=0.0)
    seen = []

    class Fake:
        def verify_batch(self, pairs):
            seen.extend(q for q, _ in pairs)
            return [[{"claim": "x", "verdict": "supported", "p_supported": 0.9, "sources": ["X"]}] for _ in pairs]
    bot.verifier = Fake()
    out = bot.reply([{"role": "user", "content": "Who built it?"}], max_new_tokens=8)
    assert out["verification"][0]["verdict"] == "supported" and seen == ["Who built it?"]

    class Broken:
        def verify_batch(self, pairs):
            raise RuntimeError("lookup down")
    bot.verifier = Broken()
    out = bot.reply([{"role": "user", "content": "Who built it?"}], max_new_tokens=8)
    assert "text" in out and "verification" not in out


def test_assessment_mixes_the_source_check_with_the_models_own_doubt():
    a = VeriChat.assess
    bot = SimpleNamespace(JUDGE_WEIGHT=0.8, LIKELY_WRONG=0.6, UNCERTAIN=0.3, doubt_score=VeriChat.doubt_score)
    claims = [{"score": 0.9, "text": "Paris"}]
    sourced_wrong = a(bot, {"claims": claims, "verification": [{"p_supported": 0.05}]})
    assert sourced_wrong["verdict"] == "likely wrong" and sourced_wrong["basis"] == "sources"
    assert a(bot, {"claims": claims, "verification": [{"p_supported": 0.99}]})["verdict"] == "likely right"
    own = a(bot, {"claims": [{"score": 0.2, "text": "x"}], "verification": [{"p_supported": None}]})
    assert own["verdict"] == "likely wrong" and own["basis"] == "the model's own confidence"
    assert a(bot, {"claims": []}) is None                       # small talk: no verdict


def test_routing_grounds_information_and_rechecks_math():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    p = Probe().fit(np.random.default_rng(0).normal(size=(40, 32)).astype(np.float32), np.r_[np.zeros(20), np.ones(20)].astype(np.float32))
    bot = VeriChat(m, tok, p, layer=1, threshold=0.0)
    looked = []

    class FakeRouter:
        def route_batch(self, convs):
            kinds = {"Who built it?": "information", "What is 17 * 23?": "math", "Write a poem.": "chat"}
            return [(kinds[c[-1]["content"]], {}) for c in convs]
    bot.router = FakeRouter()
    bot.retriever = lambda q: looked.append(q) or ["Eiffel Tower (Wikipedia): Gustave Eiffel's company built it in 1889."]
    outs = bot.reply_batch([{"messages": [{"role": "user", "content": q}], "max_new_tokens": 8}
                            for q in ("Who built it?", "What is 17 * 23?", "Write a poem.")])
    info, math, chat = outs
    assert looked == ["Who built it?"]                           # one lookup, before answering, information only
    assert info["route"] == "information" and info["grounded"] and info["sources"][0].startswith("Eiffel")
    assert info["mode"] == "flag"                                # judged by its sources, no hedge markers
    assert math["route"] == "math" and isinstance(math["math_check"], list) and not math.get("grounded")
    assert chat["route"] == "chat" and chat.get("assessment") is None and chat["mode"] == "flag"   # no markers in a poem
    wrong = {"claims": [], "math_check": [{"expression": "17 * 23", "stated": 401.0, "computed": 391.0, "ok": False}]}
    assert VeriChat.assess(bot, wrong)["verdict"] == "likely wrong"
    right = {"claims": [], "route": "math", "math_check": [{"expression": "17 * 23", "stated": 391.0, "computed": 391.0, "ok": True}]}
    assert VeriChat.assess(bot, right)["verdict"] == "likely right"
    sourced = {"grounded": True, "claims": [{"text": "Gustave Eiffel", "source": "the sources found", "score": 0.4}]}
    assert VeriChat.assess(bot, sourced)["basis"] == "every claim is in the sources"
    memory = {"grounded": True, "claims": [{"text": "Le Corbusier", "source": "the model's memory", "score": 0.9}]}
    assert VeriChat.assess(bot, memory)["verdict"] == "uncertain"
    unsourced = {"route": "information", "claims": [{"text": "x", "score": 0.1}]}
    assert VeriChat.assess(bot, unsourced)["verdict"] == "unverified"       # never a guess from the fact probe
    yours = {"grounded": True, "source_label": "your text",
             "claims": [{"text": "1889", "source": "your text", "score": 0.4}]}
    assert VeriChat.assess(bot, yours)["basis"] == "every claim is in your text"
    broken = {"route": "code", "claims": [], "code_check": [{"language": "python", "ok": False, "error": "x (line 2)"}]}
    assert VeriChat.assess(bot, broken)["verdict"] == "likely wrong"
    assert VeriChat.assess(bot, dict(broken, code_check=[{"language": "python", "ok": True}])) is None
    plain = {"grounded": True, "source_label": "your text", "claims": [], "sentences": [{"text": "It closed.", "doubted": False}]}
    assert VeriChat.assess(bot, plain)["verdict"] == "likely right"         # no claim spans: its sentences decide
    off = dict(plain, sentences=[{"text": "It closes in October.", "doubted": True}])
    assert VeriChat.assess(bot, off)["basis"] == "1 sentence(s) doubted against your text"

    class TextRouter:
        def route_batch(self, convs):
            return [("text", {}) for _ in convs]
    bot.router, looked[:] = TextRouter(), []
    note = "Summarize the text below.\n\nThe Eiffel Tower was built in 1889 by Gustave Eiffel's company."
    out = bot.reply_batch([{"messages": [{"role": "user", "content": note}], "max_new_tokens": 8}])[0]
    assert looked == [] and out["route"] == "text" and out["grounded"]   # no search: the pasted text is the source
    assert out["source_label"] == "your text" and "sources" not in out      # not shown back to the user
