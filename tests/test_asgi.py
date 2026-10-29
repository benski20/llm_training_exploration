"""The VeriChat handler served in-process as an ASGI app (vt/asgi.py), behind uvicorn."""
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from vt.asgi import asgi_from
from vt.verichat_server import make_server

uvicorn = pytest.importorskip("uvicorn")


class Bot:
    gen = SimpleNamespace(layer=1, risk_probe=None, thinking=False, think_open=None, sampling=None, think_sampling=None)
    threshold, retriever = 0.65, None

    def __init__(self, delay=0.0):
        self.delay = delay

    def reply_batch(self, reqs, on_text=None, on_reasoning=None, on_ready=None):
        time.sleep(self.delay)
        out = [{"text": f"reply to {r['messages'][-1]['content']}", "claims": []} for r in reqs]
        for i, o in enumerate(out):
            if on_text:
                on_text(i, o["text"])
        return out


@pytest.fixture
def served():
    started = []

    def start(bot, **kw):
        app = asgi_from(make_server(bot, port=0, **kw))
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        started.append(server)
        return f"http://127.0.0.1:{port}"
    yield start
    for s in started:
        s.should_exit = True


def post(url, obj, headers=None, raw=None):
    req = urllib.request.Request(url, raw if raw is not None else json.dumps(obj).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), dict(e.headers)


def test_chat_health_page_and_errors_through_asgi(served):
    base = served(Bot(), rate_per_minute=0)
    code, body, headers = post(base + "/chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert code == 200 and json.loads(body)["text"] == "reply to hi"
    assert json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())["status"] == "ok"
    assert b"VeriChat" in urllib.request.urlopen(base + "/", timeout=10).read()
    assert post(base + "/chat", None, raw=b"{not json")[0] == 400
    assert post(base + "/nowhere", {})[0] == 404


def test_streams_and_heartbeats_arrive_as_they_are_written(served):
    base = served(Bot(delay=11), rate_per_minute=0)
    req = urllib.request.Request(base + "/chat", json.dumps({"messages": [{"role": "user", "content": "q"}],
                                                             "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    t0, first = time.time(), None
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        text = b""
        while True:
            chunk = r.read1(1 << 16)
            if not chunk:
                break
            first = first or time.time() - t0
            text += chunk
    text = text.decode()
    assert first < 10.9 and text.startswith(": ping") and '"done": true' in text      # the ping came first


def test_chunked_request_bodies_and_concurrent_clients(served):
    base = served(Bot(delay=0.2), rate_per_minute=0)

    def chunked():
        yield json.dumps({"messages": [{"role": "user", "content": "chunked"}]}).encode()
    code, body, _ = post(base + "/chat", None, raw=chunked(), headers={"Transfer-Encoding": "chunked"})
    assert code == 200 and "chunked" in body
    results = []

    def one(i):
        results.append(post(base + "/chat", {"messages": [{"role": "user", "content": f"q{i}"}]})[0])
    ts = [threading.Thread(target=one, args=(i,)) for i in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count(200) == 12
