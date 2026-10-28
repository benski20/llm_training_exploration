"""Throughput of the general detector: replies judged per second one at a time vs in padded batches.

    python experiments/judge_throughput.py --head <detector head> --n 32
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.detector import HALUBENCH, Detector, load_halubench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/judge_throughput.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    head = torch.load(a.head, map_location="cpu", weights_only=True)
    det = Detector.load(head["model"], a.head)
    items = [{"messages": it["messages"], "response": it["response"]}
             for it in load_halubench("test", {s: a.n // len(HALUBENCH) for s in HALUBENCH.values()})]
    det.judge_batch(items[:2])                                   # warm-up
    t = time.perf_counter()
    one = [det.judge_batch([it])[0] for it in items]
    t_one = time.perf_counter() - t
    res = {"model": head["model"], "device": str(det.model.device), "threads": a.threads, "n": len(items),
           "mean_tokens": sum(len(det.tok.apply_chat_template(det.item(i["messages"], i["response"])["messages"],
                                                               tokenize=True)) for i in items) / len(items),
           "one_at_a_time_per_s": round(len(items) / t_one, 3)}
    for bt in (2048, 8192):
        t = time.perf_counter()
        b = det.judge_batch(items, batch_tokens=bt)
        res[f"batched_{bt}_per_s"] = round(len(items) / (time.perf_counter() - t), 3)
        res[f"batched_{bt}_max_logit_diff"] = max(abs(x["logit"] - y["logit"]) for x, y in zip(one, b))
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
