"""Which scorers should the verification engine (vt/verify_engine.py) ship with?

Two questions, from scores already computed:
1. Out of domain: LettuceDetect (trained on RAGTruth), the frozen-reader internals check (no human labels) and
   MiniCheck on the same 300 HaluBench answers (results/halubench_compare_qwen3_4b_seed1.json; 50 per source, half
   hallucinated). AUROC overall and per source with bootstrap intervals for the differences; hallucinations caught
   at 5% and 10% false alarms for each scorer alone, the union rule (either passes its threshold at half the
   budget) and one combined score (logit of LettuceDetect plus the internals total, each divided by its spread
   over the clean answers).
2. The promise on LettuceDetect's own scores, task by task (RAGTruth test answers: data-to-text, QA, summaries):
   calibrate "at most 10% false alarms with 95% confidence" on half of a task's clean answers, measure false
   alarms on the other half and hallucinations caught, 200 random halves; against the published 0.5 cut.

    python experiments/verify_engine_choice.py --lettuce-ragtruth <dir>/lettuce_test.jsonl --shards <rt_proj dir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vt.correctness import auroc
from vt.promise import guaranteed_threshold


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def caught_at(s, y, fpr):
    neg = np.sort(s[y == 0])
    t = neg[int(np.ceil((1 - fpr) * len(neg))) - 1]
    return float((s[y == 1] > t).mean()), t


def out_of_domain(path, boots=2000):
    with open(path) as f:
        rows = json.load(f)["rows"]
    y = np.array([o["y"] for o in rows])
    src = np.array([o["source"] for o in rows])
    lettuce = np.array([o["lettuce"] for o in rows])
    internals = np.array([o["grounding_total"] for o in rows])
    minicheck = np.array([o["minicheck"] for o in rows])
    lg = logit(lettuce)
    combined = lg / lg[y == 0].std() + internals / internals[y == 0].std()
    scores = {"lettuce": lettuce, "internals": internals, "minicheck": minicheck, "combined": combined}
    res = {"answers": len(y), "hallucinated": int(y.sum()), "auroc": {}, "per_source": {}, "caught": {}}
    for name, s in scores.items():
        res["auroc"][name] = round(float(auroc(s, y.astype(float))), 3)
        res["per_source"][name] = {k: round(float(auroc(s[src == k], y[src == k].astype(float))), 3)
                                   for k in sorted(set(src))}
        res["caught"][name] = {f"{f:.0%}": round(caught_at(s, y, f)[0], 3) for f in (0.05, 0.10)}
    for f in (0.05, 0.10):                                 # union: each at half the budget
        _, tl = caught_at(lettuce, y, f / 2)
        _, tg = caught_at(internals, y, f / 2)
        flag = (lettuce > tl) | (internals > tg)
        res["caught"].setdefault("union", {})[f"{f:.0%}"] = round(float(flag[y == 1].mean()), 3)
        res["caught"]["union"][f"false_alarms_at_{f:.0%}"] = round(float(flag[y == 0].mean()), 3)
    rng = np.random.default_rng(0)
    diffs = {"lettuce - internals": (lettuce, internals), "combined - lettuce": (combined, lettuce),
             "combined - internals": (combined, internals)}
    res["auroc_difference_ci95"] = {}
    for name, (a, b) in diffs.items():
        d = []
        for _ in range(boots):
            i = rng.integers(0, len(y), len(y))
            d.append(auroc(a[i], y[i].astype(float)) - auroc(b[i], y[i].astype(float)))
        res["auroc_difference_ci95"][name] = [round(float(x), 3) for x in np.percentile(d, [2.5, 50, 97.5])]
    return res


def promise_by_task(lettuce_path, shards, splits=200, fpr=0.10):
    from ragtruth_sota import load
    with open(lettuce_path) as f:
        s = np.array([json.loads(line)["max_prob"] for line in f])
    te = load(shards, "test")
    y, task = te["resp_y"][:len(s)], te["resp_task"][:len(s)]
    rng = np.random.default_rng(0)
    out = {}
    for t in sorted(set(task.tolist())):
        clean, bad = np.where((task == t) & (y == 0))[0], np.where((task == t) & (y == 1))[0]
        fa, ca = [], []
        for _ in range(splits):
            c = rng.permutation(clean)
            cal, test = c[:len(c) // 2], c[len(c) // 2:]
            thr = guaranteed_threshold(s[cal], fpr, 0.95)
            fa.append(float((s[test] > thr).mean()))
            ca.append(float((s[bad] > thr).mean()))
        out[t] = {"clean": len(clean), "hallucinated": len(bad), "calibrated_on": len(clean) // 2,
                  "false_alarms_mean": round(float(np.mean(fa)), 3),
                  "test_halves_over_promise": round(float(np.mean(np.array(fa) > fpr)), 3),
                  "caught": round(float(np.mean(ca)), 3),
                  "published_cut_0.5": {"false_alarms": round(float((s[clean] > 0.5).mean()), 3),
                                        "caught": round(float((s[bad] > 0.5).mean()), 3)}}
    return {"promise": f"at most {fpr:.0%} false alarms, 95% confidence", "splits": splits, "by_task": out,
            "note": "test_halves_over_promise counts halves whose measured rate exceeds the promise; with 20-80 "
                    "clean test answers a half can exceed it by chance even when the true rate is within it"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halubench", default="results/halubench_compare_qwen3_4b_seed1.json")
    ap.add_argument("--lettuce-ragtruth", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default="results/verify_engine_choice.json")
    a = ap.parse_args()
    res = {"out_of_domain_halubench": out_of_domain(a.halubench),
           "lettuce_promise_ragtruth": promise_by_task(a.lettuce_ragtruth, a.shards)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
