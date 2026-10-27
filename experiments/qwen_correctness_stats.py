"""Paired bootstrap on the rows pickled by qwen_correctness.py: are the unfitted attribution
signals (one exact number each, no training) better correctness predictors than softmax?
All 1,000 questions are used for unfitted signals; fitted combinations use the same halves.

    python experiments/qwen_correctness_stats.py <rows.pkl>
"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "kaggle", "qwen_glassbox"))
import run as R  # noqa: E402
from qwen_correctness import ATTR_NAMES  # noqa: E402


def boot(a, b, y, n=2000, seed=0):
    """AUROC(a) - AUROC(b), 95% interval, by resampling questions."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(len(y), size=len(y))
        if y[i].all() or (~y[i].astype(bool)).all():
            continue
        d.append(R.auroc(a[i], y[i]) - R.auroc(b[i], y[i]))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    rows = pickle.load(open(sys.argv[1], "rb"))
    y = np.array([r["right"] for r in rows])
    msp = np.array([r["msp"] for r in rows])
    attr = np.stack([r["attr"] for r in rows])
    out = {"n": len(rows), "right": int(y.sum()), "softmax_auroc": R.auroc(msp, y)}
    for j, name in enumerate(ATTR_NAMES):
        s = attr[:, j]
        out[name] = {"auroc": R.auroc(s, y), "minus_softmax": boot(s, msp, y)}
    logit = attr[:, ATTR_NAMES.index("logit")]
    q = attr[:, ATTR_NAMES.index("question_signed")]
    out["question_signed_over_logit"] = {"auroc": R.auroc(q / np.abs(logit).clip(1e-3), y)}
    idx = np.random.default_rng(0).permutation(len(rows))       # same halves as qwen_correctness.py
    fit, test = idx[: len(idx) // 2], idx[len(idx) // 2:]

    def two(f, name):
        X = np.stack([f(r) for r in rows])
        pr, w, l2 = R.fit_probe_cv(X[fit], y[fit], "cpu")
        s = pr(X[test])
        out[name] = {"auroc_test": R.auroc(s, y[test]), "minus_softmax_test": boot(s, msp[test], y[test]),
                     "l2": l2}
    two(lambda r: [r["msp"], r["neg_nll"], r["attr"][5]], "softmax_nll_plus_question_signed")
    L = 19
    hs = np.stack([r["h"][L] for r in rows])
    pr, _, _ = R.fit_probe_cv(hs[fit], y[fit], "cpu")
    probe = pr(hs[test])
    out["question_signed_minus_probe_test"] = boot(q[test], probe, y[test])
    two(lambda r: np.concatenate([r["h"][L], [r["attr"][5], r["attr"][0]]]), "probe_plus_question_attr")
    out["probe_plus_question_attr"]["minus_probe_test"] = boot(
        R.fit_probe_cv(np.stack([np.concatenate([r["h"][L], [r["attr"][5], r["attr"][0]]]) for r in rows])[fit],
                       y[fit], "cpu")[0](np.stack([np.concatenate([r["h"][L], [r["attr"][5], r["attr"][0]]])
                                                   for r in rows])[test]), probe, y[test])
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "qwen_correctness_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
