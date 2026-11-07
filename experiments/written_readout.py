"""A readout written without labels for a new domain (RAGTruth QA), against the one set with 300 labels.

Context: on RAGTruth QA the shipped check ranks test replies at AUROC 0.765 with no labels at all; 300 labelled
replies only set its threshold (docs/product.md: 10.7% false alarms against a 10% target, 42% caught). Two
questions, both without a single label from the domain:

1. Ranking: can a readout whose weights are written, not fitted, rank better than 0.765? The weights are written
   from reasoning and from unlabeled replies of the domain (the features' spread and how they move together), in
   the spirit of "Sparks of Weight-Space Programming" (MIT CSAIL, 2026), then frozen before any test label is read.
2. Threshold: can the false-alarm threshold be set from unlabeled replies alone (the clean replies' score
   distribution estimated by a two-component mixture), and does 10% hold on the labelled test replies?

    python experiments/written_readout.py read --set unlabeled --n 300     # natural RAGTruth QA training replies
    python experiments/written_readout.py read --set test                  # results/pipeline_demo/ragtruth_qa_test300
    python experiments/written_readout.py explore                          # unlabeled set only: no labels read
    python experiments/written_readout.py evaluate                         # the written weights on the test set
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "results/written_readout"
PROBE = "heads/grounding_probe_qwen3_4b_flow.npz"
SIGNALS = ("probe", "shift", "disagreement", "copy_conflict", "uncopied", "self_reliance")


def unlabeled_rows(n, seed=1):
    """RAGTruth QA training replies sampled at random (their natural mix), none of the 300 calibration replies;
    labels are kept out of the record so nothing downstream can read them."""
    from datasets import load_dataset
    cal = {json.loads(l)["reply"] for l in open("results/pipeline_demo/ragtruth_qa_cal300.jsonl")}
    rows = [r for r in load_dataset("wandb/RAGTruth-processed", split="train")
            if r["task_type"] == "QA" and r["output"].strip() and r["output"] not in cal]
    random.Random(seed).shuffle(rows)
    return [{"question": r["query"], "reply": r["output"], "documents": [r["context"]]} for r in rows[:n]]


def read(a):
    from vt.grounding import Grounding
    from vt.verifier_pipeline import load_reader, read_jsonl
    rows = unlabeled_rows(a.n) if a.set == "unlabeled" else read_jsonl("results/pipeline_demo/ragtruth_qa_test300.jsonl")
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{a.set}.jsonl"
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    model, tok = load_reader("Qwen/Qwen3-4B")
    g = Grounding(model, tok, PROBE)
    t0 = time.time()
    with open(path, "a") as f:
        for i, r in enumerate(rows[done:], done):
            sents = g.raw([{"role": "user", "content": r["question"]}], r["reply"], documents=r["documents"])
            rec = {"i": i, "sentences": [{k: s[k] for k in ("score", "mean_shift", "max_disagreement", "copy_conflict",
                                                           "uncopied", "self_reliance") if k in s} for s in sents]}
            if "hallucinated" in r:
                rec["y"] = int(r["hallucinated"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1 - done) % 25 == 0:
                print(f"{a.set} {i + 1}/{len(rows)} {(time.time() - t0) / (i + 1 - done):.1f}s a reply", flush=True)


def sentence_matrix(recs):
    """Per sentence: the six signals, oriented so that higher means less grounded (signs fixed in advance, as in
    vt/grounding.py: low evidence shift is bad, the rest high is bad)."""
    rows, owner = [], []
    for j, r in enumerate(recs):
        for s in r["sentences"]:
            rows.append([s["score"], -s["mean_shift"], s["max_disagreement"], s["copy_conflict"], s["uncopied"],
                         s["self_reliance"]])
            owner.append(j)
    return np.array(rows), np.array(owner)


def load(name):
    return [json.loads(l) for l in open(f"{OUT}/{name}.jsonl")]


def explore(a):
    """Unlabeled replies only: each signal's spread, and how the signals move together."""
    recs = load("unlabeled")
    assert not any("y" in r for r in recs), "the unlabeled set carries no labels"
    X, owner = sentence_matrix(recs)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    C = np.corrcoef(Z.T)
    w, v = np.linalg.eigh(C)
    pc = v[:, -1] * np.sign(v[:, -1].sum())
    info = {"replies": len(recs), "sentences": len(X), "sentences_per_reply": round(len(X) / len(recs), 2),
            "mean": dict(zip(SIGNALS, mu.round(4).tolist())), "std": dict(zip(SIGNALS, sd.round(4).tolist())),
            "correlation": {s: dict(zip(SIGNALS, C[i].round(2).tolist())) for i, s in enumerate(SIGNALS)},
            "first_component": dict(zip(SIGNALS, pc.round(3).tolist())), "explained": round(float(w[-1] / w.sum()), 3)}
    print(json.dumps(info, indent=1))
    json.dump(info, open(f"{OUT}/explore.json", "w"), indent=1)


def reply_scores(recs, weights, mu, sd, agg="max"):
    X, owner = sentence_matrix(recs)
    s = ((X - mu) / sd) @ weights
    out = np.full(len(recs), -np.inf)
    for j in range(len(recs)):
        m = owner == j
        if m.any():
            out[j] = s[m].max() if agg == "max" else np.clip(s[m], 0, None).sum()
    out[np.isinf(out)] = np.nanmin(out[~np.isinf(out)]) if (~np.isinf(out)).any() else 0
    return out


def mixture_threshold(scores, fpr=0.1, seed=0):
    """Two Gaussian components on unlabeled reply scores; the lower one stands for the clean replies, and the
    threshold is its (1 - fpr) quantile."""
    from scipy.stats import norm
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(2, random_state=seed).fit(scores[:, None])
    lo = int(np.argmin(gm.means_[:, 0]))
    m, s = float(gm.means_[lo, 0]), float(np.sqrt(gm.covariances_[lo].ravel()[0]))
    return float(norm.ppf(1 - fpr, m, s)), {"clean_share": round(float(gm.weights_[lo]), 3), "clean_mean": round(m, 3),
                                             "clean_std": round(s, 3)}


def evaluate(a):
    from vt.correctness import auroc
    from vt.verifier_pipeline import rates
    spec = json.load(open(f"{OUT}/written_weights.json"))              # written and frozen before this runs
    unl, test = load("unlabeled"), load("test")
    Xu, _ = sentence_matrix(unl)
    mu, sd = Xu.mean(0), Xu.std(0)                                      # the domain's own scales, no labels
    y = np.array([r["y"] for r in test])
    shipped = json.load(open("results/pipeline_demo/model_card_rtqa.json"))
    res = {"test_replies": len(test), "written": spec, "readouts": {}}
    res["readouts"]["shipped check (no labels for ranking)"] = {"auroc": shipped["auroc"]}
    for name, w in spec["readouts"].items():
        w = np.array([w[s] for s in SIGNALS], float)
        su, st = reply_scores(unl, w, mu, sd, spec["aggregate"]), reply_scores(test, w, mu, sd, spec["aggregate"])
        thr, mix = mixture_threshold(su)
        res["readouts"][name] = {"auroc": round(float(auroc(st, y.astype(float))), 3),
                                 "label_free_threshold": round(thr, 3), "mixture": mix, "at_threshold": rates(st, y, thr)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{OUT}/evaluate.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--set", choices=["unlabeled", "test"], required=True)
    r.add_argument("--n", type=int, default=300)
    sub.add_parser("explore")
    sub.add_parser("evaluate")
    a = ap.parse_args()
    {"read": read, "explore": explore, "evaluate": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
