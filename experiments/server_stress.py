"""Can a request reach the VeriChat server and never be answered? The real server (batcher, early replies,
isolated retries, streaming, heartbeats) around a fake model that behaves like the live one: a batched
generation, replies that need no lookup handed over early, a lookup pass for the doubted ones, and now and
then a failure. Many concurrent clients; every request must end with a reply or a clean error.

    python experiments/server_stress.py --clients 8 --per-client 25
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vt.verichat_server import make_server  # noqa: E402


class FakeBot:
    gen = SimpleNamespace(layer=1, risk_probe=None, thinking=False, think_open=None, sampling=None, think_sampling=None)
    threshold, retriever = 0.65, None

    def __init__(self, rng, scale, fail_rate):
        self.rng, self.scale, self.fail_rate, self.lock = rng, scale, fail_rate, threading.Lock()
        self.calls = 0

    def _r(self):
        with self.lock:
            return self.rng.random()

    def reply_batch(self, reqs, on_text=None, on_reasoning=None, on_ready=None):
        self.calls += 1
        time.sleep(self.scale * (0.5 + 2 * self._r()))                  # the batched generation
        if self._r() < self.fail_rate:
            raise RuntimeError("simulated model failure")
        outs = [{"text": f"reply {i}", "claims": [], "reply_confidence": 0.9, "mode": "hedge"} for i in range(len(reqs))]
        doubted = [i for i in range(len(reqs)) if self._r() < 0.3]
        for i in range(len(reqs)):
            if on_text:
                on_text(i, outs[i]["text"])
            if i not in doubted and on_ready:
                on_ready(i, outs[i])                                  # final now: no lookup
        if doubted:
            time.sleep(self.scale * (0.5 + 2 * self._r()))              # the lookup pass
            if self._r() < self.fail_rate:
                raise RuntimeError("simulated lookup failure")
            for i in doubted:
                outs[i] = dict(outs[i], retrieval={"retrieved": True})
                if on_ready:
                    on_ready(i, outs[i])
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--per-client", type=int, default=25)
    ap.add_argument("--scale", type=float, default=0.3, help="seconds per unit of fake model time")
    ap.add_argument("--fail-rate", type=float, default=0.05)
    ap.add_argument("--deadline", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/server_stress.json")
    ap.add_argument("--asgi", action="store_true", help="serve the handler in-process through vt.asgi and uvicorn")
    a = ap.parse_args()
    bot = FakeBot(random.Random(a.seed), a.scale, a.fail_rate)
    srv = make_server(bot, port=0, rate_per_minute=0, deadline=a.deadline, log_requests=False)
    if a.asgi:
        import socket
        import uvicorn
        from vt.asgi import asgi_from
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        us = uvicorn.Server(uvicorn.Config(asgi_from(srv), host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=us.run, daemon=True).start()
        while not us.started:
            time.sleep(0.05)
        url = f"http://127.0.0.1:{port}/chat"
    else:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_port}/chat"
    rows, lock = [], threading.Lock()

    def client(c):
        rng = random.Random(1000 + c)
        for k in range(a.per_client):
            stream = rng.random() < 0.5
            body = json.dumps({"messages": [{"role": "user", "content": f"q {c}-{k}"}], "stream": stream}).encode()
            t, outcome = time.time(), None
            try:
                r = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}),
                                           timeout=a.deadline + 60)
                text = r.read().decode()
                if stream:
                    ev = [json.loads(e[6:]) for e in text.split("\n\n") if e.startswith("data: ")]
                    end = ev[-1] if ev else {}
                    outcome = "ok" if end.get("reply") else ("stream error" if end.get("error") else "stream cut")
                else:
                    outcome = "ok" if "text" in json.loads(text) else "bad body"
            except urllib.error.HTTPError as e:
                outcome = f"HTTP {e.code}"
            except Exception as e:                                    # dropped, timed out: the losses we hunt
                outcome = "LOST " + type(e).__name__
            with lock:
                rows.append({"client": c, "k": k, "stream": stream, "outcome": outcome,
                             "seconds": round(time.time() - t, 2)})
            time.sleep(rng.random() * a.scale)

    t0 = time.time()
    ts = [threading.Thread(target=client, args=(c,)) for c in range(a.clients)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if not a.asgi:
        srv.shutdown()
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    res = {"requests": len(rows), "outcomes": counts, "model_calls": bot.calls, "wall_s": round(time.time() - t0, 1),
           "max_seconds": max(r["seconds"] for r in rows), "lost": [r for r in rows if r["outcome"].startswith("LOST")]}
    print(json.dumps({k: v for k, v in res.items() if k != "lost"}), "lost:", len(res["lost"]))
    json.dump(dict(res, settings=vars(a)), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
