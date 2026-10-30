"""vt.client: a streamed request that goes silent (lost on the way) is abandoned and sent again."""
import threading
import time
from types import SimpleNamespace

import pytest

from vt.client import Lost, chat
from vt.verichat_server import make_server


class Bot:
    gen = SimpleNamespace(layer=1, risk_probe=None, thinking=False, think_open=None, sampling=None, think_sampling=None)
    threshold, retriever = 0.65, None

    def reply_batch(self, reqs, on_text=None, on_reasoning=None, on_ready=None):
        out = [{"text": "Paris.", "claims": []} for _ in reqs]
        for i, o in enumerate(out):
            if on_text:
                on_text(i, o["text"])
        return out


def serve(drop_first):
    srv = make_server(Bot(), port=0, rate_per_minute=0)
    base, posts = srv.RequestHandlerClass, []

    class Dropping(base):                                    # takes the request, never answers (a lost one)
        def do_POST(self):
            posts.append(1)
            if len(posts) <= drop_first:
                time.sleep(5)
                return
            return base.do_POST(self)
    srv.RequestHandlerClass = Dropping
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", posts


def test_a_lost_request_is_retried_and_answered():
    srv, url, posts = serve(drop_first=1)
    try:
        r = chat(url, [{"role": "user", "content": "Capital of France?"}], silence=1.0)
        assert r["text"] == "Paris." and r["_attempts"] == 2 and len(posts) == 2
    finally:
        srv.shutdown()


def test_gives_up_after_its_retries():
    srv, url, posts = serve(drop_first=5)
    try:
        with pytest.raises(Lost):
            chat(url, [{"role": "user", "content": "q"}], silence=0.5, retries=1)
        assert len(posts) == 2
    finally:
        srv.shutdown()


def test_a_reply_cut_off_midway_is_retried():
    import socket as _s
    srv, url, posts = serve(drop_first=0)
    base = srv.RequestHandlerClass
    cuts = []

    class Cut(base):                                         # first reply: headers promise more, then the line drops
        def do_POST(self):
            cuts.append(1)
            if len(cuts) == 1:
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b"data: ")
                self.request.shutdown(_s.SHUT_RDWR)
                return
            return base.do_POST(self)
    srv.RequestHandlerClass = Cut
    try:
        r = chat(url, [{"role": "user", "content": "q"}], silence=2.0)
        assert r["text"] == "Paris." and r["_attempts"] == 2
    finally:
        srv.shutdown()
