"""HTTP inference server with a verification gate.

    vt serve model.npz certificate.json --spec dyck --port 8000
    curl localhost:8000/health
    curl -X POST localhost:8000/predict -d '{"inputs": ["(())()"]}'

Trust model: the certificate is NOT trusted on its own word (anyone can write
JSON). Before serving, the gate
  1. reads the artifact once and hashes exactly those bytes (no TOCTOU),
  2. requires the certificate to name a consistent spec, matching the spec the
     operator pinned (--spec / --spec-file) when one is given,
  3. re-proves the artifact against that spec with vt's own verifier,
unless the certificate carries a valid Ed25519 signature from a key the
operator trusts (--trusted-key), which is the fast path for large models.
It then serves exactly the forward pass that was verified (numpy float64).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .artifact import ArtifactError, read_artifact, signature_valid
from .verify import np_forward_batch


class GateError(RuntimeError):
    pass


def resolve_cert_spec(cert, pinned=None):
    """Return (spec, name, alphabet) for a certificate, refusing inconsistent ones.

    pinned: optional (spec, name, alphabet) chosen by the operator. Built-in spec
    names are self-describing (re-verification uses vt's own implementation of
    that spec); an embedded DFA must be pinned by the operator, because otherwise
    the certificate itself would be deciding what "correct" means."""
    from .cli import TOKENS
    from .counterspec import spec_from_def
    from .global_verify import SPECS
    name = cert.get("spec")
    dfa = cert.get("spec_dfa")
    if dfa is not None:
        if name != "dfa:" + dfa.get("name", "dfa"):
            raise GateError(f"certificate names spec {name!r} but embeds DFA {dfa.get('name')!r}")
        if pinned is None or not hasattr(pinned[0], "canonical"):
            raise GateError("certificate uses a spec file: pin it with --spec-file")
        try:
            same = pinned[0].canonical() == spec_from_def(dfa).canonical()
        except (ValueError, KeyError, TypeError):
            same = False
        if not same:
            raise GateError("certificate spec differs from the pinned spec")
        return pinned
    if name not in SPECS:
        raise GateError(f"unknown spec {name!r}")
    if pinned is not None and pinned[1] != name:
        raise GateError(f"certificate is for spec {name!r}, not the pinned {pinned[1]!r}")
    return SPECS[name](), name, TOKENS[name]


def load_gated(artifact, certificate=None, allow_unverified=False, pinned=None,
               trusted_key=None, verify_time=600.0, workers=1):
    """Returns (P, digest, cert, info). Raises GateError unless the artifact is
    proved (re-verified here, or signed by trusted_key)."""
    try:
        P, digest = read_artifact(artifact)
    except (ArtifactError, OSError) as e:
        raise GateError(f"cannot load artifact: {e}") from None
    if not certificate:
        if not allow_unverified:
            raise GateError("no certificate given (use --allow-unverified to serve anyway)")
        return P, digest, None, {"verified": False, "how": "unverified"}
    with open(certificate) as f:
        cert = json.load(f)
    if cert.get("artifact_sha256") != digest:
        raise GateError("artifact hash does not match certificate")
    if cert.get("status") != "proved":
        if allow_unverified:
            return P, digest, cert, {"verified": False, "how": "unverified"}
        raise GateError(f"certificate status is {cert.get('status')!r}, not 'proved'")
    spec, name, alphabet = resolve_cert_spec(cert, pinned)
    if trusted_key and signature_valid(cert, trusted_key):
        return P, digest, cert, {"verified": True, "how": "signature", "spec": name,
                                 "alphabet": alphabet}
    if trusted_key:
        raise GateError("certificate signature missing or not from the trusted key")
    from .global_verify import verify_auto
    method = "exact" if cert.get("method") == "exact-dp" else "auto"
    out = verify_auto(P, spec, method=method, tau=float(cert.get("tau", 1e-4)),
                      time_limit=verify_time, workers=workers)
    if out["status"] != "proved":
        raise GateError(f"re-verification failed: {out['status']}")
    return P, digest, cert, {"verified": True, "how": f"re-verified ({out['method']})",
                             "spec": name, "alphabet": alphabet}


def json_depth_ok(body, limit=8):
    """Cheap pre-parse check that JSON nesting stays shallow (json.loads recursion
    and memory amplification guard). Strings are skipped."""
    depth, in_str, esc = 0, False, False
    for b in body:
        if in_str:
            if esc:
                esc = False
            elif b == 0x5C:
                esc = True
            elif b == 0x22:
                in_str = False
        elif b == 0x22:
            in_str = True
        elif b in (0x5B, 0x7B):
            depth += 1
            if depth > limit:
                return False
        elif b in (0x5D, 0x7D):
            depth -= 1
    return True


class HardenedHandler(BaseHTTPRequestHandler):
    """Per-socket timeout, a hard deadline per connection (so trickling clients are
    cut off), quiet logs and JSON responses. Subclasses set `timeout_s`, `deadline_s`
    and may override `on_response` for metrics."""
    server_version = "vt-serve"
    sys_version = ""
    timeout_s = 10.0
    deadline_s = 30.0
    write_timeout_s = 120.0     # once a response starts: a slow reader (e.g. a busy proxy) must not truncate it

    def setup(self):
        super().setup()
        self.request.settimeout(self.timeout_s)
        sock = self.request

        def cut():
            try:
                sock.shutdown(2)
            except OSError:
                pass
        self._deadline = threading.Timer(self.deadline_s, cut)
        self._deadline.daemon = True
        self._deadline.start()

    def finish(self):
        self._deadline.cancel()
        try:
            super().finish()
        except OSError:
            pass

    def on_response(self, code):
        pass

    def send_response(self, code, message=None):
        """Responses get a longer socket timeout than requests: the short one cuts off clients trickling a
        request in, but a reader slow to drain the reply would otherwise leave it half-written."""
        self.request.settimeout(self.write_timeout_s)
        super().send_response(code, message)
        self.send_header("Connection", "close")          # one request per connection, said out loud: a pooling
                                                         # proxy must not reuse a socket this server closes

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.on_response(code)

    def read_json_object(self, max_body):
        """The request body as a JSON object, or (code, message) for the error response."""
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return None, (411, "Content-Length required")
        if n < 0 or n > max_body:
            self.close_connection = True
            return None, (413, f"body must be at most {max_body} bytes")
        try:
            body = self.rfile.read(n)
            if not json_depth_ok(body):
                raise ValueError("JSON nested too deeply")
            req = json.loads(body.decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, UnicodeDecodeError, RecursionError) as e:
            return None, (400, str(e)[:200])
        return req, None

    def log_message(self, *args):  # quiet
        pass


def bounded_server(host, port, handler, max_concurrency):
    """A threading HTTP server that handles at most `max_concurrency` connections at
    once and answers the rest with 503 immediately, instead of spawning unbounded threads."""
    slots = threading.BoundedSemaphore(max_concurrency)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 512   # the default (5) drops bursts into 1 s SYN retries

        def process_request(self, request, client_address):
            if not slots.acquire(blocking=False):
                try:
                    request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n"
                                    b"Retry-After: 1\r\n\r\n")
                finally:
                    self.shutdown_request(request)
                return
            super().process_request(request, client_address)

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                slots.release()

    return Server((host, port), handler)


def make_server(P, digest, cert, alphabet, host="127.0.0.1", port=8000, max_batch=1024,
                max_body=1 << 20, timeout=10.0, deadline=30.0, max_concurrency=64, gate=None):
    """max_body: request size cap (413). timeout: per socket operation. deadline:
    hard cap on a whole connection, so trickling clients are cut off.
    max_concurrency: connections handled at once; beyond it clients get 503
    immediately instead of spawning unbounded threads."""
    L, V = P["L"], P["V"]
    stats = {"requests": 0, "predictions": 0, "errors_4xx": 0, "latency_s_total": 0.0}
    lock = threading.Lock()
    info = {"status": "ok", "artifact_sha256": digest, "length": L, "alphabet": alphabet,
            "verified": bool(gate and gate.get("verified")),
            "how": gate.get("how") if gate else None,
            "spec": (gate or {}).get("spec") or (cert.get("spec") if cert else None),
            "method": cert.get("method") if cert else None}

    class Handler(HardenedHandler):
        timeout_s, deadline_s = timeout, deadline

        def on_response(self, code):
            with lock:
                stats["requests"] += 1
                if 400 <= code < 500:
                    stats["errors_4xx"] += 1

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, info)
            if self.path == "/metrics":
                with lock:
                    snap = dict(stats)
                return self.send_json(200, snap)
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self.send_json(404, {"error": "not found"})
            t0 = time.perf_counter()
            req, err = self.read_json_object(max_body)
            if err:
                return self.send_json(err[0], {"error": err[1]})
            try:
                inputs = req["inputs"]
                if not isinstance(inputs, list) or not 0 < len(inputs) <= max_batch:
                    raise ValueError(f"'inputs' must be a list of 1..{max_batch} strings")
                seqs = []
                for s in inputs:
                    if not isinstance(s, str) or len(s) != L or any(ch not in alphabet for ch in s):
                        raise ValueError(f"each input must be a string of length {L} over {alphabet!r}")
                    seqs.append([alphabet.index(ch) for ch in s])
            except (ValueError, KeyError, TypeError) as e:
                return self.send_json(400, {"error": str(e)[:200]})
            logits = np_forward_batch(P, np.eye(V)[np.array(seqs)])
            preds = logits.argmax(1)
            self.send_json(200, {"artifact_sha256": digest, "predictions": [
                {"input": s, "label": int(p)} for s, p in zip(inputs, preds)]})
            with lock:
                stats["predictions"] += len(inputs)
                stats["latency_s_total"] += time.perf_counter() - t0

    return bounded_server(host, port, Handler, max_concurrency)
