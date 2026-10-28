"""Few-shot domain adaptation of a trained detector (Detector.adapt): how many labelled replies from a
new domain recover what zero-shot loses? HaluBench domains the head never saw (DROP, PubMedQA,
FinanceBench, CovidQA): adapt on k = 0 / 25 / 50 / 100 / 200 training replies per domain (balanced),
score the same test replies, report AUROC with CIs and accuracy at the adapted threshold.

    python kaggle/detector/adapt_eval.py --model Qwen/Qwen2.5-3B-Instruct --head <head> --out <dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vtpkg"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch  # noqa: E402

from vt.correctness import auroc, auroc_ci  # noqa: E402
from vt.detector import HALUBENCH, Detector, load_halubench  # noqa: E402


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--ks", default="0,25,50,100,200")
    ap.add_argument("--test", type=int, default=200, help="test replies per domain")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="/kaggle/working/adapt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hp = a.head if os.path.exists(a.head) else os.path.join(HERE, "files", os.path.basename(a.head))
    det = Detector.load(a.model, hp)
    ks = [int(k) for k in a.ks.split(",")]
    t0 = time.time()
    res = {"model": a.model, "head": os.path.basename(hp), "ks": ks, "per_domain": {}}
    for src in HALUBENCH.values():
        train = load_halubench("train", {src: 10 ** 6})
        test = load_halubench("test", {src: a.test})
        rng = np.random.default_rng(0)
        pos = [x for x in train if x["label"] == 1]
        neg = [x for x in train if x["label"] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        items = lambda xs: [{"messages": x["messages"], "response": x["response"], "label": x["label"]} for x in xs]  # noqa: E731
        y = [x["label"] for x in test]
        d = {"n_test": len(test), "positives": int(sum(y))}
        for k in ks:
            m = det if k == 0 else det.adapt(items(pos[:k // 2] + neg[:k - k // 2]), a.max_tokens)
            rs = m.judge_batch(items(test), a.max_tokens)
            ok = [(r["logit"], yy) for r, yy in zip(rs, y) if "error" not in r]
            z, yy = [o[0] for o in ok], [o[1] for o in ok]
            d[str(k)] = {"auroc": auroc(z, yy), "auroc_ci": auroc_ci(z, yy),
                         "accuracy": float(np.mean((np.array(z) >= m.threshold) == np.array(yy)))}
            log(phase="adapted", domain=src, k=k, auroc=d[str(k)]["auroc"], accuracy=d[str(k)]["accuracy"],
                s=round(time.time() - t0))
        res["per_domain"][src] = d
        json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    for k in ks:
        res.setdefault("mean_auroc", {})[str(k)] = float(np.mean([d[str(k)]["auroc"] for d in res["per_domain"].values()]))
    res["complete"] = True
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    log(phase="done", mean_auroc=res["mean_auroc"], minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
