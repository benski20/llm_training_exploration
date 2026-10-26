"""Load-test vt serve: throughput and latency percentiles under concurrency.

    python experiments/bench_serve.py results/cegis/exact_dyck_L2-64/model.npz \
        results/cegis/exact_dyck_L2-64/certificate.json --requests 2000 --concurrency 16 --batch 32
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.cli import TOKENS  # noqa: E402
from vt.serve import load_gated, make_server  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact"); ap.add_argument("certificate")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    P, digest, cert = load_gated(a.artifact, a.certificate)
    alphabet = cert["spec_dfa"]["alphabet"] if "spec_dfa" in cert else TOKENS[cert["spec"]]
    srv = make_server(P, digest, cert, alphabet, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/predict"
    rng = np.random.default_rng(0)
    body = json.dumps({"inputs": ["".join(rng.choice(list(alphabet), P["L"]))
                                  for _ in range(a.batch)]}).encode()

    def one(_):
        t0 = time.perf_counter()
        with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=30) as r:
            assert r.status == 200
            r.read()
        return time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        lat = np.array(list(ex.map(one, range(a.requests))))
    wall = time.perf_counter() - t0
    srv.shutdown()
    out = {"L": P["L"], "batch": a.batch, "concurrency": a.concurrency, "requests": a.requests,
           "requests_per_s": a.requests / wall, "sequences_per_s": a.requests * a.batch / wall,
           "p50_ms": 1000 * float(np.percentile(lat, 50)), "p99_ms": 1000 * float(np.percentile(lat, 99))}
    print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in out.items()}))


if __name__ == "__main__":
    main()
