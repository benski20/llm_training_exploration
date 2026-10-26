"""Minimal HTTP inference server with a verification gate.

    vt serve model.npz certificate.json --port 8000
    curl localhost:8000/health
    curl -X POST localhost:8000/predict -d '{"inputs": ["(())()"]}'

The server refuses to start unless the artifact's SHA-256 matches a
certificate whose status is "proved" (or --allow-unverified is given), and it
serves exactly the forward pass that was verified (numpy float64).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .artifact import load_artifact, sha256_file
from .verify import np_forward_batch


class GateError(RuntimeError):
    pass


def load_gated(artifact, certificate=None, allow_unverified=False):
    digest = sha256_file(artifact)
    cert = None
    if certificate:
        with open(certificate) as f:
            cert = json.load(f)
        if cert.get("artifact_sha256") != digest:
            raise GateError("artifact hash does not match certificate")
        if cert.get("status") != "proved" and not allow_unverified:
            raise GateError(f"certificate status is {cert.get('status')!r}, not 'proved'")
    elif not allow_unverified:
        raise GateError("no certificate given (use --allow-unverified to serve anyway)")
    return load_artifact(artifact), digest, cert


def make_server(P, digest, cert, alphabet, host="127.0.0.1", port=8000, max_batch=1024,
                max_body=1 << 20, timeout=10.0):
    """max_body: request size cap in bytes (413 above it); timeout: per-connection
    socket timeout, so slow or idle clients cannot hold a worker thread."""
    L, V = P["L"], P["V"]
    stats = {"requests": 0, "predictions": 0, "errors_4xx": 0, "latency_s_total": 0.0}
    lock = threading.Lock()
    info = {"status": "ok", "artifact_sha256": digest, "length": L, "alphabet": alphabet,
            "verified": bool(cert and cert.get("status") == "proved"),
            "spec": cert.get("spec") if cert else None,
            "method": cert.get("method") if cert else None}

    class Handler(BaseHTTPRequestHandler):
        server_version = "vt-serve"
        sys_version = ""

        def setup(self):
            super().setup()
            self.request.settimeout(timeout)

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            with lock:
                stats["requests"] += 1
                if 400 <= code < 500:
                    stats["errors_4xx"] += 1

        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, info)
            if self.path == "/metrics":
                with lock:
                    snap = dict(stats)
                return self._send(200, snap)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, {"error": "not found"})
            t0 = time.perf_counter()
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._send(411, {"error": "Content-Length required"})
            if n < 0 or n > max_body:
                self.close_connection = True
                return self._send(413, {"error": f"body must be at most {max_body} bytes"})
            try:
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                if not isinstance(req, dict):
                    raise ValueError("body must be a JSON object")
                inputs = req["inputs"]
                if not isinstance(inputs, list) or not 0 < len(inputs) <= max_batch:
                    raise ValueError(f"'inputs' must be a list of 1..{max_batch} strings")
                seqs = []
                for s in inputs:
                    if not isinstance(s, str) or len(s) != L or any(ch not in alphabet for ch in s):
                        raise ValueError(f"each input must be a string of length {L} over {alphabet!r}")
                    seqs.append([alphabet.index(ch) for ch in s])
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
                return self._send(400, {"error": str(e)[:200]})
            logits = np_forward_batch(P, np.eye(V)[np.array(seqs)])
            preds = logits.argmax(1)
            self._send(200, {"artifact_sha256": digest, "predictions": [
                {"input": s, "label": int(p)} for s, p in zip(inputs, preds)]})
            with lock:
                stats["predictions"] += len(inputs)
                stats["latency_s_total"] += time.perf_counter() - t0

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 512   # the default (5) drops bursts into 1 s SYN retries

    return Server((host, port), Handler)
