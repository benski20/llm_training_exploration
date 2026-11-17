"""Measure a running gateway + service stack: wall time per token and CPU seconds per token of each process.

Usage: python benchmarks/stack_bench.py --gateway http://127.0.0.1:PORT --key API_KEY --gateway-pid PID --service-pid PID
       [--tokens 24] [--concurrency 1] [--model demo]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request


def cpu_s(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    tick = os.sysconf("SC_CLK_TCK")
    return (int(parts[11]) + int(parts[12])) / tick


def one(url: str, key: str, model: str, tokens: int, out: list[dict]) -> None:
    body = json.dumps({"model": model, "prompt": "hi", "max_tokens": tokens, "temperature": 0,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(f"{url}/v1/completions", data=body, headers={
        "Authorization": f"Bearer {key}", "content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.load(r)
    out.append({"s": time.perf_counter() - t, "usage": res["usage"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--gateway-pid", type=int, required=True)
    ap.add_argument("--service-pid", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--model", default="demo")
    a = ap.parse_args()
    g0, s0, t0 = cpu_s(a.gateway_pid), cpu_s(a.service_pid), time.perf_counter()
    out: list[dict] = []
    th = [threading.Thread(target=one, args=(a.gateway, a.key, a.model, a.tokens, out)) for _ in range(a.concurrency)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    toks = sum(o["usage"]["total_tokens"] for o in out)
    gen = sum(o["usage"]["completion_tokens"] for o in out)
    res = {
        "concurrency": a.concurrency, "tokens_total": toks, "generated": gen, "wall_s": round(wall, 3),
        "wall_s_per_generated_token_per_stream": round(wall / max(1, gen / a.concurrency), 4),
        "gateway_cpu_s_per_token": round((cpu_s(a.gateway_pid) - g0) / max(1, toks), 4),
        "service_cpu_s_per_token": round((cpu_s(a.service_pid) - s0) / max(1, toks), 4),
    }
    print(json.dumps(res))


if __name__ == "__main__":
    main()
