"""The verification API: any chatbot sends a reply and the evidence it should rest on, and gets back each
sentence's verdict, read from a reader model's own neurons (vt/grounding.py). Model-agnostic: the reply can come
from any model, open or not; the reader only reads it.

    POST /v1/verify        {"question": "...", "reply": "...", "documents": ["..."], "lookup": false}
                           -> {"verdict": "consistent" | "not grounded" | "contradicted", "p_hallucinated"?,
                               "total", "sentences": [{"text", "start", "end", "verdict", "z", ...}], "usage", "ms"}
    POST /v1/verify/batch  {"items": [ ...up to 16 of the above... ]} -> {"results": [...]}
    GET  /health           reader, probe, whether it is calibrated
    GET  /metrics          Prometheus: requests, latency, tokens verified per key (the basis for billing)

API keys: VERIFY_API_KEYS="name1:key1,name2:key2" (or a file of such lines in VERIFY_API_KEYS_FILE); with none
set the API is open (local use). Each key has its own rate limit. "lookup": true lets the server search for
evidence (when started with --wikipedia) for replies sent without documents.

    python -m vt.verify_service --reader Qwen/Qwen3-4B --probe heads/grounding_probe_qwen3_4b.npz --port 8000
"""
import argparse
import collections
import hmac
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer

from vt.serve import HardenedHandler, json_depth_ok
from vt.verichat_server import RateLimiter

MAX_REPLY, MAX_DOCS_CHARS, MAX_DOCS, MAX_BATCH = 8000, 60000, 20, 16


def keys_from_env():
    """{key: name} from VERIFY_API_KEYS ("name:key,...") or VERIFY_API_KEYS_FILE (one name:key a line)."""
    raw = os.environ.get("VERIFY_API_KEYS", "")
    path = os.environ.get("VERIFY_API_KEYS_FILE")
    if path:
        with open(path) as f:
            raw += "," + ",".join(line.strip() for line in f if line.strip() and not line.startswith("#"))
    out = {}
    for part in filter(None, (p.strip() for p in raw.split(","))):
        name, _, key = part.partition(":")
        if key:
            out[key] = name
    return out


def validate(item):
    """The item's fields, checked; or a reason it is refused."""
    if not isinstance(item, dict):
        return None, "each item must be an object"
    reply, question = item.get("reply"), item.get("question", "")
    docs = item.get("documents") or []
    if not isinstance(reply, str) or not reply.strip():
        return None, "reply: a non-empty string"
    if not isinstance(question, str):
        return None, "question: a string"
    if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
        return None, "documents: a list of strings"
    if len(reply) > MAX_REPLY:
        return None, f"reply: at most {MAX_REPLY} characters"
    if len(docs) > MAX_DOCS or sum(map(len, docs)) > MAX_DOCS_CHARS:
        return None, f"documents: at most {MAX_DOCS}, {MAX_DOCS_CHARS} characters in all"
    return {"question": question, "reply": reply, "documents": docs, "lookup": bool(item.get("lookup"))}, None


class Usage:
    """Requests and tokens verified, per key: what /metrics exposes and billing reads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = collections.Counter()
        self.tokens = collections.Counter()
        self.latency = collections.defaultdict(float)
        self.codes = collections.Counter()

    def add(self, name, tokens, seconds):
        with self.lock:
            self.requests[name] += 1
            self.tokens[name] += tokens
            self.latency[name] += seconds

    def code(self, path, code):
        with self.lock:
            self.codes[(path, code)] += 1

    def text(self):
        with self.lock:
            lines = ["# TYPE verify_requests_total counter"]
            lines += [f'verify_requests_total{{key="{k}"}} {n}' for k, n in sorted(self.requests.items())]
            lines += ["# TYPE verify_tokens_total counter"]
            lines += [f'verify_tokens_total{{key="{k}"}} {n}' for k, n in sorted(self.tokens.items())]
            lines += ["# TYPE verify_latency_seconds_total counter"]
            lines += [f'verify_latency_seconds_total{{key="{k}"}} {v:.3f}' for k, v in sorted(self.latency.items())]
            lines += ["# TYPE verify_responses_total counter"]
            lines += [f'verify_responses_total{{path="{p}",code="{c}"}} {n}' for (p, c), n in sorted(self.codes.items())]
        return "\n".join(lines) + "\n"


class Verifier:
    """One model-owning lock around the grounding check (the reader's hooks must not see another request)."""

    def __init__(self, grounding, info):
        self.g, self.info, self.lock = grounding, info, threading.Lock()
        self.retriever = grounding.retriever
        grounding.retriever = None                              # used only when an item asks for a lookup

    def verify(self, item):
        t = time.time()
        msgs = [{"role": "user", "content": item["question"]}]
        with self.lock:
            self.g.retriever = self.retriever if item["lookup"] and not item["documents"] else None
            try:
                res = self.g.check(msgs, item["reply"], documents=item["documents"])
            finally:
                self.g.retriever = None
        tok = self.g.tok
        n = len(tok(item["reply"], add_special_tokens=False).input_ids) + \
            sum(len(tok(d, add_special_tokens=False).input_ids) for d in item["documents"])
        res["usage"] = {"tokens": n}
        res["ms"] = round(1000 * (time.time() - t), 1)
        return res


def make_server(verifier, host="127.0.0.1", port=8000, keys=None, rate_per_minute=120, max_body=1 << 20,
                deadline=300.0):
    keys = keys or {}
    usage, limiter = Usage(), RateLimiter(rate_per_minute)

    class Handler(HardenedHandler):
        timeout_s, deadline_s = 30.0, deadline

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                usage.code("/health", 200)
                return self.send_json(200, dict(status="ok", **verifier.info))
            if self.path == "/metrics":
                body = usage.text().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return usage.code("/metrics", 200)
            usage.code("other", 404)
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path
            if path not in ("/v1/verify", "/v1/verify/batch"):
                usage.code("other", 404)
                return self.send_json(404, {"error": "not found"})
            name = "open"
            if keys:
                auth = self.headers.get("Authorization", "")
                key = auth[7:] if auth.startswith("Bearer ") else ""
                name = next((n for k, n in keys.items() if hmac.compare_digest(k, key)), None)
                if name is None:
                    usage.code(path, 401)
                    return self.send_json(401, {"error": "a valid API key: Authorization: Bearer <key>"})
            if not limiter.allow(name):
                usage.code(path, 429)
                return self.send_json(429, {"error": f"rate limit: {rate_per_minute} requests a minute"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if n <= 0 or n > max_body:
                usage.code(path, 413)
                return self.send_json(413, {"error": f"a JSON body of at most {max_body} bytes"})
            body = self.rfile.read(n)
            if not json_depth_ok(body):
                usage.code(path, 400)
                return self.send_json(400, {"error": "JSON nested too deeply"})
            try:
                req = json.loads(body)
            except ValueError:
                usage.code(path, 400)
                return self.send_json(400, {"error": "invalid JSON"})
            items = req.get("items") if path.endswith("/batch") and isinstance(req, dict) else [req]
            if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH:
                usage.code(path, 400)
                return self.send_json(400, {"error": f"items: a list of 1 to {MAX_BATCH}"})
            checked = []
            for it in items:
                ok, why = validate(it)
                if why:
                    usage.code(path, 400)
                    return self.send_json(400, {"error": why})
                checked.append(ok)
            t = time.time()
            try:
                results = [verifier.verify(it) for it in checked]
            except Exception as e:                              # the reader failed: say so, keep serving
                print(f"verify failed: {e!r}", flush=True)
                usage.code(path, 500)
                return self.send_json(500, {"error": "verification failed"})
            usage.add(name, sum(r["usage"]["tokens"] for r in results), time.time() - t)
            usage.code(path, 200)
            self.send_json(200, {"results": results} if path.endswith("/batch") else results[0])

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    srv.usage = usage
    return srv


def build(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vt.verify_service")
    ap.add_argument("--reader", required=True, help="the model whose neurons the probe reads")
    ap.add_argument("--probe", required=True, help="a probe from vt.verifier_pipeline (calibrated, ideally)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rate-per-minute", type=int, default=120)
    ap.add_argument("--wikipedia", action="store_true", help="allow \"lookup\": true (Wikipedia search for evidence)")
    a = ap.parse_args(argv)
    from vt.grounding import Grounding
    from vt.verifier_pipeline import load_reader, sha256
    model, tok = load_reader(a.reader)
    retriever = None
    if a.wikipedia:
        from vt.retrieval import Wikipedia
        retriever = Wikipedia(full_text=True, pages=3)
    g = Grounding(model, tok, a.probe, retriever=retriever)
    info = {"reader": a.reader, "probe": os.path.basename(a.probe), "probe_sha256": sha256(a.probe)[:16],
            "calibrated": g.calibration is not None, "evidence_signals": len(g.natural) == 3,
            "lookup": retriever is not None}
    return make_server(Verifier(g, info), a.host, a.port, keys_from_env(), a.rate_per_minute), a


def main(argv=None):
    srv, a = build(argv)
    print(f"verification API on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
