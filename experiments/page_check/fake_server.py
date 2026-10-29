"""A fake VeriChat for checking the page in a browser (experiments/page_check/check.js drives it):
the first POST is taken at the socket and never answered (a stuck proxy), the next reply is slow (25 s).
The page should abort the stuck one after 45 s of silence and retry it, and keep the slow one alive on
heartbeats. Run: python experiments/page_check/fake_server.py & node experiments/page_check/check.js"""
import sys, threading, time
from types import SimpleNamespace
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from vt.verichat_server import make_server

calls = []
class Bot:
    gen = SimpleNamespace(layer=1, risk_probe=None, thinking=False, think_open=None, sampling=None, think_sampling=None)
    threshold, retriever = 0.65, None
    def reply_batch(self, reqs, on_text=None, on_reasoning=None, on_ready=None):
        calls.append(time.time())
        n = len(calls)
        if n == 2:
            time.sleep(25)          # slow but alive: heartbeats keep it
        out = [{"text": f"Reply {n}.", "claims": [], "reply_confidence": 0.9, "threshold": 0.65, "mode": "hedge"} for _ in reqs]
        if on_text:
            for i, o in enumerate(out):
                on_text(i, o["text"])
        return out

srv = make_server(Bot(), port=8098, rate_per_minute=0, deadline=300)
H = srv.RequestHandlerClass
posts = []
class Stall(H):                     # the first POST is taken and never answered (a stuck proxy or socket)
    def do_POST(self):
        posts.append(1)
        if len(posts) == 1:
            time.sleep(120)
            return
        return H.do_POST(self)
srv.RequestHandlerClass = Stall
threading.Thread(target=srv.serve_forever, daemon=True).start()
print("PORT", srv.server_port, flush=True)
time.sleep(200)
print("model calls:", len(calls), "posts:", len(posts))
