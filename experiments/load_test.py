"""End-to-end load test of the VeriChat server: N concurrent clients, each sending chat requests.
Reports throughput and latency percentiles, 429/503 counts, and whether requests were batched.

    python experiments/load_test.py --model Qwen/Qwen2.5-0.5B-Instruct --head <head> --clients 1,8 --per-client 4
    python experiments/load_test.py --url https://<...>.modal.run --clients 1,8,16 --per-client 4   # a running server

A remote server keeps its own reply length and rate limit (deploy with VERICHAT_RATE=0 for a load test;
VERICHAT_API_KEY, if set, is sent as the Bearer token).
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

QS = ["Who wrote Pride and Prejudice?", "What is the capital of Australia?", "In which year did the Berlin Wall fall?",
      "Who painted the Mona Lisa?", "What is the largest planet in the Solar System?", "Who discovered penicillin?",
      "When did the first Moon landing happen?", "Who wrote One Hundred Years of Solitude?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--head", default="results/kaggle/qwen-correctness/correctness/Qwen2.5-0.5B-Instruct_heads.pt")
    ap.add_argument("--clients", default="1,8")
    ap.add_argument("--per-client", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--url", help="load-test a running server instead of starting one")
    ap.add_argument("--out", default="results/load_test.json")
    a = ap.parse_args()
    calls, srv, headers = [], None, {"Content-Type": "application/json"}
    if a.url:
        url = a.url.rstrip("/") + "/chat"
        if os.environ.get("VERICHAT_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["VERICHAT_API_KEY"]
        res = {"url": a.url, "runs": []}
    else:                                                # a local server (needs torch and the model)
        import torch
        from vt.verichat import VeriChat
        from vt.verichat_server import make_server
        torch.set_num_threads(a.threads)
        bot = VeriChat.load(a.model, a.head)
        inner = bot.reply_batch
        bot.reply_batch = lambda reqs, **kw: (calls.append(len(reqs)),
                                              inner([dict(r, max_new_tokens=a.tokens) for r in reqs], **kw))[1]
        srv = make_server(bot, port=0, rate_per_minute=0, max_batch=8, deadline=600)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_port}/chat"
        res = {"model": a.model, "threads": a.threads, "max_new_tokens": a.tokens, "runs": []}
    for n in [int(x) for x in a.clients.split(",")]:
        calls.clear()
        lat, codes = [], []

        def client(c):
            for k in range(a.per_client):
                q = QS[(c * a.per_client + k) % len(QS)]
                body = json.dumps({"messages": [{"role": "user", "content": q}], "mode": "hedge"}).encode()
                t = time.perf_counter()
                try:
                    code = urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=600).status
                except urllib.error.HTTPError as e:
                    code = e.code
                lat.append(time.perf_counter() - t)
                codes.append(code)
        t0 = time.perf_counter()
        ts = [threading.Thread(target=client, args=(c,)) for c in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.perf_counter() - t0
        run = {"clients": n, "requests": len(codes), "ok": codes.count(200), "other_codes": sorted(set(codes) - {200}),
               "wall_s": round(wall, 1), "replies_per_s": round(codes.count(200) / wall, 3),
               "latency_p50_s": round(float(np.percentile(lat, 50)), 2), "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
               "batch_sizes": calls[:] if not a.url else None}
        print(json.dumps(run), flush=True)
        res["runs"].append(run)
    if srv is not None:
        srv.shutdown()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
