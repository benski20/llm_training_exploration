"""A drop-in guard for any OpenAI-compatible chat API: point the client's base URL here instead of at the model
provider, and every reply comes back checked against the documents it should rest on (the verification API,
vt/verify_service.py --engine ...).

    python -m vt.guard_proxy --upstream https://api.openai.com --verify-url http://verify:8000 --port 8080
    # VERIFY_API_KEY: this proxy's key for the verification API (its calibration is that key's promise)

    client = OpenAI(base_url="http://guard:8080/v1", api_key=OPENAI_KEY)      # the provider key passes through
    r = client.chat.completions.create(model=..., messages=..., extra_body={"documents": retrieved_passages})
    r.model_extra["verification"][0]["verdict"]          # "supported" | "unsupported" | "skipped" | "error"

Documents: the request's "documents" field (removed before the request goes upstream); without it, the contents
of the request's system and tool messages (where RAG apps usually put retrieved text). With neither, the reply is
not checked ("skipped"): there is nothing to check it against.

Each choice gets an entry in a top-level "verification" list (verdict, unsupported spans, scores, the promise),
and the response carries X-Verification (the first choice's verdict). --on-unsupported block replaces an
unsupported reply's content with --block-message and sets finish_reason "content_filter". If the verification
API fails, the reply passes with verdict "error" (--on-verify-error pass, the default) or the request fails with
502 (--on-verify-error fail). Streaming requests are refused: a reply cannot be checked before it is complete.
The provider key is forwarded, never stored or logged; the upstream URL is fixed by the operator.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from vt.serve import HardenedHandler, json_depth_ok
from vt.verify_client import VerifyClient, VerifyError

MAX_BODY = 8 << 20
PATH = "/v1/chat/completions"


def text_of(content):
    """A message's text: a string, or the text parts of a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def evidence(req):
    """(question, documents, source) for a chat request; documents from "documents", else system/tool messages."""
    msgs = [m for m in req.get("messages", []) if isinstance(m, dict)]
    users = [text_of(m.get("content")) for m in msgs if m.get("role") == "user"]
    question = users[-1] if users else ""
    docs = req.get("documents")
    if isinstance(docs, list) and all(isinstance(d, str) for d in docs) and docs:
        return question, docs, "documents"
    found = [text_of(m.get("content")) for m in msgs if m.get("role") in ("system", "developer", "tool")]
    found = [d for d in found if d.strip()]
    return question, found, "messages" if found else None


class Guard:
    def __init__(self, upstream, verifier, on_unsupported="annotate", on_verify_error="pass",
                 block_message="I can't give that answer: parts of it are not supported by the provided documents.",
                 timeout=300.0):
        self.upstream, self.v = upstream.rstrip("/"), verifier
        self.on_unsupported, self.on_verify_error, self.block_message = on_unsupported, on_verify_error, block_message
        self.timeout = timeout

    def forward(self, body, auth):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = auth
        req = urllib.request.Request(self.upstream + PATH, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosec B310: the operator's upstream URL
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def check(self, question, docs, reply):
        try:
            r = self.v.verify(reply=reply, question=question, documents=docs)
        except VerifyError as e:
            if self.on_verify_error == "fail":
                raise
            return {"verdict": "error", "error": str(e)}
        keep = ("verdict", "spans", "sentences", "scores", "fired", "promise", "note")
        return {k: r[k] for k in keep if k in r}

    def handle(self, req, auth):
        """(status, response body dict or bytes, X-Verification value)."""
        if req.get("stream"):
            return 400, {"error": {"message": "stream: not supported through the guard (a reply is checked when it "
                                              "is complete); send stream=false", "type": "invalid_request_error"}}, ""
        question, docs, source = evidence(req)
        body = {k: v for k, v in req.items() if k != "documents"}
        status, raw = self.forward(body, auth)
        if status != 200:
            return status, raw, ""
        try:
            resp = json.loads(raw)
        except ValueError:
            return 502, {"error": {"message": "upstream returned invalid JSON", "type": "upstream_error"}}, ""
        results = []
        for choice in resp.get("choices", []):
            msg = choice.get("message") or {}
            reply = text_of(msg.get("content"))
            if not source or not reply.strip():
                results.append({"verdict": "skipped", "reason": "no documents in the request" if not source
                                else "empty reply"})
                continue
            res = self.check(question, docs, reply)
            res["documents_from"] = source
            if res["verdict"] == "unsupported" and self.on_unsupported == "block":
                msg["content"] = self.block_message
                choice["finish_reason"] = "content_filter"
                res["blocked"] = True
            results.append(res)
        resp["verification"] = results
        return 200, resp, results[0]["verdict"] if results else "skipped"


def make_server(guard, host="127.0.0.1", port=8080):
    class Handler(HardenedHandler):
        timeout_s, deadline_s, write_timeout_s = 60.0, guard.timeout + 330.0, 120.0

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, {"status": "ok", "upstream": guard.upstream,
                                            "on_unsupported": guard.on_unsupported})
            self.send_json(404, {"error": {"message": "not found"}})

        def reply(self, status, body, verdict):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if verdict:
                self.send_header("X-Verification", verdict)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != PATH:
                return self.send_json(404, {"error": {"message": f"only POST {PATH}"}})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if n <= 0 or n > MAX_BODY:
                return self.send_json(413, {"error": {"message": f"a JSON body of at most {MAX_BODY} bytes"}})
            raw = self.rfile.read(n)
            if not json_depth_ok(raw):
                return self.send_json(400, {"error": {"message": "JSON nested too deeply"}})
            try:
                req = json.loads(raw)
            except ValueError:
                return self.send_json(400, {"error": {"message": "invalid JSON"}})
            if not isinstance(req, dict) or not isinstance(req.get("messages"), list):
                return self.send_json(400, {"error": {"message": "messages: a list"}})
            t = time.time()
            try:
                status, body, verdict = guard.handle(req, self.headers.get("Authorization"))
            except VerifyError as e:
                return self.send_json(502, {"error": {"message": f"verification failed: {e}"}})
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                return self.send_json(502, {"error": {"message": f"upstream unreachable: {e}"}})
            if isinstance(body, dict) and status == 200:
                body.setdefault("verification_ms", round(1000 * (time.time() - t), 1))
            self.reply(status, body, verdict)

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vt.guard_proxy")
    ap.add_argument("--upstream", required=True, help="the OpenAI-compatible API to forward to")
    ap.add_argument("--verify-url", required=True, help="the verification API (VERIFY_API_KEY: its key)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--on-unsupported", choices=["annotate", "block"], default="annotate")
    ap.add_argument("--on-verify-error", choices=["pass", "fail"], default="pass")
    ap.add_argument("--block-message", default=None)
    a = ap.parse_args(argv)
    kw = {"block_message": a.block_message} if a.block_message else {}
    guard = Guard(a.upstream, VerifyClient(a.verify_url, os.environ.get("VERIFY_API_KEY"), retries=1),
                  a.on_unsupported, a.on_verify_error, **kw)
    srv = make_server(guard, a.host, a.port)
    print(f"guard on http://{a.host}:{a.port}{PATH} -> {a.upstream}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
