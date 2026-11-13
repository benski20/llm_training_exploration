"""Measured hosted cost of the AI-text API (vt/aitext.py): the real server, the real models, over HTTP.

Starts `vt-aitext serve` in-process with one API key and a calibration directory, calibrates through
POST /v1/aitext/calibrate with 150 human texts, then sends texts one request at a time and in batches of 8
through POST /v1/aitext and /v1/aitext/batch. Reports wall time per text, the server process's peak memory,
and the cost per 1,000 texts at a stated hourly machine price (the price is an input, not a measurement).

    AI_TEXT_SCRATCH=<dir> python experiments/aitext_service_bench.py --price-per-hour 0.17
"""
import argparse
import json
import os
import resource
import sys
import threading
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SCRATCH = os.environ.get("AI_TEXT_SCRATCH", "results/ai_text")


def post(url, body, key="bench-key"):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Authorization": f"Bearer {key}",
                                                                   "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--singles", type=int, default=40)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--price-per-hour", type=float, default=0.17, help="USD for the machine the bench runs on")
    ap.add_argument("--report", default="results/ai_text/service_bench.json")
    a = ap.parse_args()
    import tempfile

    from vt.aitext import Detector, Service, validate
    from vt.verify_service import make_server
    with open(f"{SCRATCH}/hape_sample.jsonl") as f:
        rows = [json.loads(line) for line in f]
    rng = np.random.default_rng(1)
    human = [r["text"] for r in rows if r["y"] == 0 and r["genre"] == "blog"]
    test = [rows[i]["text"] for i in rng.permutation(len(rows))[:a.singles + 8 * a.batches]]

    t = time.time()
    det = Detector(threads=a.threads)
    load_s = time.time() - t
    srv = make_server(Service(det, calibration_dir=tempfile.mkdtemp()), port=0, keys={"bench-key": "bench"},
                      rate_per_minute=10000, max_body=8 << 20, deadline=3600.0, path="/v1/aitext", validator=validate)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/aitext"

    t = time.time()
    cal = post(url + "/calibrate", {"human_texts": human[:150]})
    cal_s = time.time() - t
    post(url, {"text": test[0]})                                       # warm-up
    single = []
    for x in test[:a.singles]:
        t = time.time()
        r = post(url, {"text": x})
        single.append(time.time() - t)
        assert r["verdict"] in ("ai", "not flagged")
    batch = []
    rest = test[a.singles:]
    for b in range(a.batches):
        items = [{"text": x} for x in rest[8 * b:8 * b + 8]]
        t = time.time()
        post(url + "/batch", {"items": items})
        batch.append((time.time() - t) / len(items))
    srv.shutdown()
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    per_text = float(np.median(single))
    res = {"machine": f"{os.cpu_count()} vCPU, " + open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0]
           .strip(": "), "threads": a.threads, "model_load_s": round(load_s, 1),
           "calibrate_150_texts_s": round(cal_s, 1), "calibration_thresholds": cal["thresholds"],
           "single_request_s": {"median": round(per_text, 3), "p90": round(float(np.percentile(single, 90)), 3)},
           "batch_of_8_s_per_text": round(float(np.median(batch)), 3), "peak_memory_gb": round(peak_gb, 2),
           "texts_per_hour": int(3600 / min(per_text, float(np.median(batch)))),
           "price_per_hour_usd_assumed": a.price_per_hour}
    res["usd_per_1000_texts"] = round(1000 * a.price_per_hour / res["texts_per_hour"], 4)
    with open(a.report, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
