"""Does a bigger frozen reader make the internals probe a better verifier out of its training domain?

A linear probe on the reader's neurons (experiments/ragtruth_sota.py features: MLP neurons of every 4th layer,
averaged per answer sentence, a fixed random projection to 4,096 numbers) is trained on the first N RAGTruth
training answers' sentence labels and scored on 300 HaluBench answers (50 per source, half hallucinated; the
sample of results/halubench_compare_qwen3_4b_seed1.json): an answer's score is its highest sentence probability.
Reported: AUROC overall and per source, hallucinations caught at 5% and 10% false alarms, and the same for the
probe combined with LettuceDetect (the engine's combined score: each on its spread over clean answers, summed),
next to LettuceDetect, the label-free grounding check and MiniCheck on the same answers.

    python experiments/bigreader_halubench.py --shards <dir with train_*.npz and halu_*.npz> --train 1600 \
        --reader Qwen/Qwen3-8B --report results/bigreader_halubench_qwen3_8b.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragtruth_sota import load, response_scores

from vt.correctness import auroc


def caught_at(s, y, fpr):
    neg = np.sort(s[y == 0])
    t = neg[int(np.ceil((1 - fpr) * len(neg))) - 1]
    return round(float((s[y == 1] > t).mean()), 3)


def summary(s, y, src):
    return {"auroc": round(float(auroc(s, y.astype(float))), 3),
            "per_source": {k: round(float(auroc(s[src == k], y[src == k].astype(float))), 3) for k in sorted(set(src))
                           if len(set(y[src == k].tolist())) == 2},
            "caught_at_5pct": caught_at(s, y, 0.05), "caught_at_10pct": caught_at(s, y, 0.10)}


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--train", type=int, default=1600)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--reader", default="")
    ap.add_argument("--others", default="results/halubench_compare_qwen3_4b_seed1.json")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    tr, ha = load(a.shards, "train"), load(a.shards, "halu")
    keep = tr["resp"] < a.train
    sc = StandardScaler().fit(tr["X"][keep].astype(np.float32))
    clf = LogisticRegression(C=a.C, max_iter=3000, class_weight="balanced").fit(
        sc.transform(tr["X"][keep].astype(np.float32)), tr["y"][keep])
    p = clf.predict_proba(sc.transform(ha["X"].astype(np.float32)))[:, 1]
    probe = response_scores(p, ha["resp"], ha["resp_ids"])
    y, src = ha["resp_y"], ha["resp_task"]
    res = {"reader": a.reader, "train_answers": len(np.unique(tr["resp"][keep])),
           "train_sentences": int(keep.sum()), "halubench_answers": len(y), "C": a.C,
           "probe": summary(probe, y, src)}
    with open(a.others) as f:
        rows = json.load(f)["rows"][:len(y)]
    assert [o["source"] for o in rows] == src.tolist() and [o["y"] for o in rows] == y.tolist(), "not the same sample"
    lg = np.array([o["lettuce"] for o in rows])
    lg = np.log(np.clip(lg, 1e-6, 1 - 1e-6) / (1 - np.clip(lg, 1e-6, 1 - 1e-6)))
    pl = np.log(np.clip(probe, 1e-6, 1 - 1e-6) / (1 - np.clip(probe, 1e-6, 1 - 1e-6)))
    res["probe+lettuce"] = summary(lg / lg[y == 0].std() + pl / pl[y == 0].std(), y, src)
    for name, key in (("lettuce", "lettuce"), ("grounding_check_4b", "grounding_total"), ("minicheck", "minicheck")):
        res[name] = summary(np.array([o[key] for o in rows]), y, src)
    with open(a.report, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
