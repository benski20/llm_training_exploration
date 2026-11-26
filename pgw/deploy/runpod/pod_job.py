"""In-pod boot of a measurement job pod (``jobs.py``): TTL self-stop, a token-gated control port, the code bundle.

For sessions that reach RunPod only through its HTTPS API and its HTTPS pod proxy (``https://<pod id>-8888.proxy.
runpod.net``; no raw TCP, no SSH, no custom image registry). ``jobs.py`` creates the pod from a public base image and
passes this file (zlib + base64) in ``PGW_JOB_BOOT``; the pod's start command runs it with the base image's Python
(stdlib only). Then, in order:

1. the TTL timer is armed first (``PGW_JOB_STOP_AT``, absolute): at the deadline the job's process group is killed
   and the pod stops itself through RunPod's REST API with the pod-scoped ``RUNPOD_API_KEY`` RunPod injects (kept
   out of the job's environment);
2. the control server listens on 8888 (RunPod's HTTP proxy maps it to HTTPS). Every route needs a token whose
   SHA-256 is in the pod's environment; the raw tokens never are, except the pair token of a client pod;
3. ``PUT /bundle`` (job token): the code bundle (a tar.gz of ``pgw/``), accepted only if its SHA-256 equals
   ``PGW_JOB_BUNDLE_SHA256`` (fixed by the session when it created the pod), extracted with tarfile's ``data``
   filter, then ``deploy/runpod/job_setup.sh <role>`` and ``jobs.py pod`` run in the background;
4. ``GET /status``, ``GET /log``, ``GET /results.tar.gz`` (job token): progress, the log tail, the results;
5. ``GET|PUT /peer/<name>`` (pair token): small files exchanged between the GPU pod and its client pod (the
   provider's address and test SDK key, the lane bench's throwaway tickets, "done" markers).

Environment: ``PGW_JOB_TOKEN_SHA256``, ``PGW_JOB_PAIR_TOKEN_SHA256`` (or, on a client pod, ``PGW_JOB_PAIR_TOKEN``),
``PGW_JOB_BUNDLE_SHA256``, ``PGW_JOB_STOP_AT``, ``PGW_JOB_ROLE`` (``gpu`` | ``cpu``), ``PGW_JOB_STEPS``; RunPod's
``RUNPOD_POD_ID`` and ``RUNPOD_API_KEY``. Local test: ``python pod_job.py --root DIR --port 0`` (no TTL stop call
without ``RUNPOD_POD_ID``).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import http.server
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from typing import ClassVar

PORT = 8888
MAX_BUNDLE = 96 << 20
MAX_PEER = 1 << 20
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
STATE: dict = {"state": "waiting-bundle", "since": time.time(), "rc": None}
LOCK = threading.Lock()
PROC: list = []


def log(root: str, msg: str) -> None:
    line = f"[pod-job {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(root, "boot.log"), "a") as f:
        f.write(line + "\n")


def _ok(token: str | None, want_hex: str) -> bool:
    if not token or not want_hex:
        return False
    return hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), want_hex.strip().lower())


def stop_pod(pod_id: str | None, key: str | None) -> bool:
    if not pod_id or not key:
        return False
    req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod_id}/stop", data=b"{}", method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed https RunPod URL
            return 200 <= r.status < 300
    except Exception:
        return False


def ttl(root: str, stop_at: float | None, pod_id: str | None, key: str | None) -> None:
    if stop_at is None:
        return
    while (left := stop_at - time.time()) > 0:
        time.sleep(min(left, 20.0))
    log(root, "TTL reached: killing the job and stopping this pod")
    for p in PROC:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
    for attempt in range(6):
        if stop_pod(pod_id, key):
            log(root, "stop requested through the RunPod API")
            return
        time.sleep(min(60.0, 5.0 * 2 ** attempt))
    log(root, "WARNING: could not stop the pod through the API; the session's watchdog must stop it")


def run_job(root: str, env: dict[str, str]) -> None:
    src = os.path.join(root, "src")
    res = os.path.join(root, "results")
    os.makedirs(res, exist_ok=True)
    jenv = {k: v for k, v in env.items() if k not in ("RUNPOD_API_KEY", "PGW_JOB_BOOT")}  # least privilege
    jenv.update(PGW_JOB_ROOT=root, PGW_JOB_RESULTS=res, PYTHONUNBUFFERED="1")
    role = env.get("PGW_JOB_ROLE", "cpu")
    cmd = ["bash", "-c", f"bash deploy/runpod/job_setup.sh {role} && exec /opt/pgw/bin/python deploy/runpod/jobs.py "
           "pod"]
    if env.get("PGW_JOB_SETUP") == "none":  # local tests: the interpreter running this file, no setup
        cmd = [sys.executable, "deploy/runpod/jobs.py", "pod"]
    with LOCK:
        STATE.update(state="running", since=time.time())
    with open(os.path.join(res, "job.log"), "ab") as out:
        p = subprocess.Popen(cmd, cwd=src, env=jenv, stdout=out, stderr=subprocess.STDOUT,  # noqa: S603 - pinned bundle
                             stdin=subprocess.DEVNULL,
                             start_new_session=True)
        PROC.append(p)
        rc = p.wait()
    with LOCK:
        STATE.update(state="done" if rc == 0 else "failed", rc=rc, since=time.time())
    log(root, f"job exited with {rc}")


class Handler(http.server.BaseHTTPRequestHandler):
    root = "/workspace/job"
    env: ClassVar[dict[str, str]] = {}

    def log_message(self, fmt: str, *a: object) -> None:  # no request logging (tokens are in headers only)
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if code >= 400:
            self.send_header("Connection", "close")  # an unread request body must not be parsed as a request
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj).encode())

    def _job_ok(self) -> bool:
        return _ok(self.headers.get("X-PGW-Job-Token"), self.env.get("PGW_JOB_TOKEN_SHA256", ""))

    def _pair_ok(self) -> bool:
        tok = self.headers.get("X-PGW-Pair-Token")
        want = self.env.get("PGW_JOB_PAIR_TOKEN_SHA256") or (
            hashlib.sha256(self.env["PGW_JOB_PAIR_TOKEN"].encode()).hexdigest() if self.env.get("PGW_JOB_PAIR_TOKEN")
            else "")
        return _ok(tok, want)

    def _body(self, limit: int) -> bytes | None:
        n = int(self.headers.get("Content-Length") or -1)
        if n < 0 or n > limit:
            self._json(413, {"error": f"body must have a Content-Length of at most {limit} bytes"})
            return None
        return self.rfile.read(n)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/peer/"):
            return self._peer_get(path[6:])
        if not self._job_ok():
            return self._json(403, {"error": "forbidden"})
        if path == "/status":
            with LOCK:
                st = dict(STATE)
            st.update(role=self.env.get("PGW_JOB_ROLE"), steps=self.env.get("PGW_JOB_STEPS"),
                      stop_at=self.env.get("PGW_JOB_STOP_AT"), now=time.time())
            prog = os.path.join(self.root, "results", "progress.json")
            if os.path.exists(prog):
                try:
                    st["progress"] = json.load(open(prog))
                except ValueError:
                    pass
            return self._json(200, st)
        if path == "/log":
            data = b""
            for name in ("boot.log", "results/job.log"):
                p = os.path.join(self.root, name)
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        f.seek(max(0, os.path.getsize(p) - 64000))
                        data += f"==> {name}\n".encode() + f.read() + b"\n"
            return self._send(200, data, "text/plain; charset=utf-8")
        if path == "/results.tar.gz":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.add(os.path.join(self.root, "results"), arcname="results")
                if os.path.exists(os.path.join(self.root, "boot.log")):
                    tf.add(os.path.join(self.root, "boot.log"), arcname="results/boot.log")
            return self._send(200, buf.getvalue(), "application/gzip")
        return self._json(404, {"error": "not found"})

    def do_PUT(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/peer/"):
            return self._peer_put(path[6:])
        if path != "/bundle":
            return self._json(404, {"error": "not found"})
        if not self._job_ok():
            return self._json(403, {"error": "forbidden"})
        with LOCK:
            if STATE["state"] != "waiting-bundle":
                return self._json(409, {"error": f"already {STATE['state']}"})
            STATE["state"] = "receiving"
        body = self._body(MAX_BUNDLE)
        want = self.env.get("PGW_JOB_BUNDLE_SHA256", "").strip().lower()
        if body is None or not want or not hmac.compare_digest(hashlib.sha256(body).hexdigest(), want):
            with LOCK:
                STATE["state"] = "waiting-bundle"
            if body is not None:
                self._json(400, {"error": "bundle sha256 does not match PGW_JOB_BUNDLE_SHA256"})
            return None
        src = os.path.join(self.root, "src")
        os.makedirs(src, exist_ok=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
                tf.extractall(src, filter="data")  # no absolute paths, no .., no links out, no devices
        except (tarfile.TarError, OSError) as exc:
            with LOCK:
                STATE["state"] = "bad-bundle"
            log(self.root, f"bundle refused: {type(exc).__name__}: {exc}")
            return self._json(400, {"error": f"bundle refused: {type(exc).__name__}"})
        log(self.root, f"bundle accepted ({len(body)} bytes); starting the job")
        threading.Thread(target=run_job, args=(self.root, self.env), daemon=True).start()
        return self._json(200, {"accepted": len(body)})

    def _peer_get(self, name: str) -> None:
        if not self._pair_ok():
            return self._json(403, {"error": "forbidden"})
        p = os.path.join(self.root, "peer", name)
        if not NAME.match(name) or not os.path.isfile(p):
            return self._json(404, {"error": "not found"})
        with open(p, "rb") as f:
            return self._send(200, f.read(), "application/octet-stream")

    def _peer_put(self, name: str) -> None:
        if not self._pair_ok():
            return self._json(403, {"error": "forbidden"})
        if not NAME.match(name):
            return self._json(400, {"error": "bad name"})
        body = self._body(MAX_PEER)
        if body is None:
            return None
        d = os.path.join(self.root, "peer")
        os.makedirs(d, mode=0o700, exist_ok=True)
        tmp = os.path.join(d, f".{name}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        os.replace(tmp, os.path.join(d, name))
        return self._json(200, {"stored": len(body)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PGW measurement job pod boot")
    ap.add_argument("--root", default="/workspace/job")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args(argv)
    env = dict(os.environ)
    os.makedirs(a.root, mode=0o700, exist_ok=True)
    os.makedirs(os.path.join(a.root, "peer"), mode=0o700, exist_ok=True)
    stop_at = float(env["PGW_JOB_STOP_AT"]) if env.get("PGW_JOB_STOP_AT") else None
    threading.Thread(target=ttl, args=(a.root, stop_at, env.get("RUNPOD_POD_ID"), env.get("RUNPOD_API_KEY")),
                     daemon=True).start()
    Handler.root, Handler.env = a.root, env
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", a.port), Handler)  # noqa: S104 - RunPod's proxy connects
    log(a.root, f"control port {srv.server_address[1]}; role {env.get('PGW_JOB_ROLE')}; steps "
        f"{env.get('PGW_JOB_STEPS')}; TTL {time.strftime('%H:%M:%S UTC', time.gmtime(stop_at)) if stop_at else 'NONE'}")
    with open(os.path.join(a.root, "port"), "w") as f:
        f.write(str(srv.server_address[1]))
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
