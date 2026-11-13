"""The guard proxy (vt/guard_proxy.py): a fake OpenAI-compatible upstream, the verification API with a fake
scorer, and the proxy between a client and both."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from vt.guard_proxy import Guard, evidence, make_server
from vt.verify_client import VerifyClient
from vt.verify_engine import Engine, Service
from vt.verify_service import make_server as make_verify_server


class Words:
    """Unsupported when the reply has a word missing from every document."""
    name, default_threshold = "words", 0.5

    def score(self, item):
        known = {w for d in item["documents"] for w in d.lower().split()}
        missing = [w for w in item["reply"].lower().split() if w not in known]
        return (1.0 if missing else 0.0), {"missing": missing}

    def detail(self, item, extra, threshold):
        return {"spans": [{"text": w} for w in extra["missing"]]}

    def tokens(self, text):
        return len(text.split())


class Upstream(BaseHTTPRequestHandler):
    """Answers with the request's "_say" field, or the last user message's text after "say:"; records what
    it was sent."""
    seen: ClassVar[list] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.seen.append((body, self.headers.get("Authorization")))
        if body.get("model") == "missing":
            data = json.dumps({"error": {"message": "no such model"}}).encode()
            self.send_response(404)
        else:
            say = body.get("_say") or body["messages"][-1]["content"].split("say:")[-1].strip()
            data = json.dumps({"id": "x", "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": say}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(srv):
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture(scope="module")
def stack():
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    vf = make_verify_server(Service(Engine([Words()])), port=0, keys={"vk": "guard"}, rate_per_minute=1000)
    up_url, vf_url = serve(up), serve(vf)
    out = {}
    for mode in ("annotate", "block"):
        g = make_server(Guard(up_url, VerifyClient(vf_url, "vk", retries=0), on_unsupported=mode), port=0)
        out[mode] = serve(g)
    dead = make_server(Guard(up_url, VerifyClient("http://127.0.0.1:9", "vk", retries=0)), port=0)
    out["dead"] = serve(dead)
    yield out
    for s in (up, vf):
        s.shutdown()


def chat(url, body, key="sk-provider"):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r), r.headers.get("X-Verification")
    except urllib.error.HTTPError as e:
        return e.code, json.load(e), e.headers.get("X-Verification")


DOCS = ["the bridge reopened in march"]


def test_evidence_from_field_then_messages():
    q, d, src = evidence({"messages": [{"role": "user", "content": "when?"}], "documents": DOCS})
    assert (q, d, src) == ("when?", DOCS, "documents")
    msgs = [{"role": "system", "content": "Context: x"}, {"role": "tool", "content": [{"type": "text", "text": "y"}]},
            {"role": "user", "content": "q"}]
    assert evidence({"messages": msgs})[1:] == (["Context: x", "y"], "messages")
    assert evidence({"messages": [{"role": "user", "content": "q"}]})[2] is None


def test_supported_reply_passes_and_documents_stay_out_of_the_upstream_request(stack):
    Upstream.seen.clear()
    code, r, h = chat(stack["annotate"], {"model": "m", "documents": DOCS,
                                          "messages": [{"role": "user", "content": "say: the bridge reopened in march"}]})
    assert code == 200 and h == "supported" and r["verification"][0]["verdict"] == "supported"
    sent, auth = Upstream.seen[-1]
    assert "documents" not in sent and auth == "Bearer sk-provider"


def test_unsupported_reply_is_annotated_with_spans(stack):
    code, r, h = chat(stack["annotate"], {"model": "m", "documents": DOCS,
                                          "messages": [{"role": "user", "content": "say: the bridge reopened in june"}]})
    v = r["verification"][0]
    assert code == 200 and h == "unsupported" and [s["text"] for s in v["spans"]] == ["june"]
    assert r["choices"][0]["message"]["content"] == "the bridge reopened in june"


def test_block_mode_replaces_the_reply(stack):
    code, r, h = chat(stack["block"], {"model": "m", "documents": DOCS,
                                       "messages": [{"role": "user", "content": "say: the bridge reopened in june"}]})
    assert code == 200 and h == "unsupported" and r["choices"][0]["finish_reason"] == "content_filter"
    assert "not supported" in r["choices"][0]["message"]["content"] and r["verification"][0]["blocked"]


def test_no_documents_is_skipped_streaming_refused_upstream_errors_pass_through(stack):
    code, r, h = chat(stack["annotate"], {"model": "m", "messages": [{"role": "user", "content": "say: hi"}]})
    assert code == 200 and h == "skipped" and r["verification"][0]["verdict"] == "skipped"
    code, r, _ = chat(stack["annotate"], {"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]})
    assert code == 400 and "stream" in r["error"]["message"]
    code, r, _ = chat(stack["annotate"], {"model": "missing", "documents": DOCS,
                                          "messages": [{"role": "user", "content": "say: x"}]})
    assert code == 404 and r["error"]["message"] == "no such model"


def test_verification_down_passes_with_error_verdict(stack):
    code, r, h = chat(stack["dead"], {"model": "m", "documents": DOCS,
                                      "messages": [{"role": "user", "content": "say: the bridge reopened in june"}]})
    assert code == 200 and h == "error" and r["verification"][0]["verdict"] == "error"
