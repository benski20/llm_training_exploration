"""Does Modal pass a streamed reply through as it is written? A CPU-only app (no GPU, cents) serving the real
VeriChat handler around a fake bot that takes 50 s, two ways: in-process (asgi_app, as the demo) and
web_server. A client then records when each chunk arrives: heartbeats every 10 s mean chunks every 10 s.

    modal deploy deploy/stream_probe.py
    python deploy/stream_probe.py https://<workspace>--verichat-stream-probe-asgi.modal.run
    modal app stop verichat-stream-probe
"""
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FAKE = '''
import time
from types import SimpleNamespace


class SlowBot:
    gen = SimpleNamespace(layer=1, risk_probe=None, thinking=False, think_open=None, sampling=None, think_sampling=None)
    threshold, retriever = 0.65, None

    def reply_batch(self, reqs, on_text=None, on_reasoning=None, on_ready=None):
        time.sleep(50)
        out = [{"text": "Paris.", "claims": []} for _ in reqs]
        for i, o in enumerate(out):
            if on_text:
                on_text(i, o["text"])
        return out
'''


def measure(url, seconds=90):
    """Prints when each chunk of a streamed /chat reply arrives."""
    import json
    import time
    import urllib.request
    body = json.dumps({"messages": [{"role": "user", "content": "q"}], "stream": True}).encode()
    t0 = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=seconds) as r:  # nosec B310: our own probe URL
        print(f"headers after {time.time() - t0:.1f} s", flush=True)
        while True:
            chunk = r.read1(1 << 16)
            if not chunk:
                break
            print(f"{time.time() - t0:6.1f} s  {chunk[:60]!r}", flush=True)
    print(f"done after {time.time() - t0:.1f} s", flush=True)


try:
    import modal
except ImportError:                                           # measuring needs no Modal
    modal = None

if __name__ == "__main__":
    measure(sys.argv[1])
elif modal is not None:
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("numpy==2.1.3", "scipy==1.14.1", "fastapi==0.115.6")
             .add_local_dir(os.path.join(ROOT, "vt"), "/app/vt"))
    app = modal.App("verichat-stream-probe", image=image)

    def server():
        sys.path.insert(0, "/app")
        ns = {}
        exec(FAKE, ns)  # nosec B102: the fixed fake bot above
        from vt.verichat_server import make_server
        return make_server(ns["SlowBot"](), port=0, rate_per_minute=0, deadline=300)

    @app.function(cpu=0.5, timeout=600, scaledown_window=60)
    @modal.concurrent(max_inputs=8)
    @modal.asgi_app()
    def asgi():
        from vt.asgi import asgi_from
        return asgi_from(server())

    @app.function(cpu=0.5, timeout=600, scaledown_window=60)
    @modal.concurrent(max_inputs=8)
    @modal.web_server(8000, startup_timeout=120)
    def web():
        code = FAKE + ("\nimport sys\nsys.path.insert(0, '/app')\nfrom vt.verichat_server import make_server\n"
                       "make_server(SlowBot(), host='0.0.0.0', port=8000, rate_per_minute=0, deadline=300).serve_forever()\n")
        subprocess.Popen(["python", "-c", code])
