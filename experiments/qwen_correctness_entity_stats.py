"""Analyse the rows of qwen_correctness_entity.py: unfitted signals on all questions (with a paired
bootstrap against the best output-probability baseline), fitted combinations on the usual halves.

    python experiments/qwen_correctness_entity_stats.py <rows.pkl>
"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "qwen_glassbox"))
import run as R  # noqa: E402
from qwen_correctness_stats import boot  # noqa: E402

UNFITTED = ["first_p", "entity_p", "mean_lp", "min_lp", "question_signed", "question_share",
            "answer_signed", "answer_share", "template_signed", "logit"]


def main():
    rows = pickle.load(open(sys.argv[1], "rb"))
    y = np.array([r["right"] for r in rows])
    col = {k: np.array([r[k] for r in rows], float) for k in UNFITTED}
    out = {"n": len(rows), "right": int(y.sum()), "entity_is_first_token": float(np.mean([r["e"] == 0 for r in rows])),
           "max_exact_err": float(max(r["exact_err"] for r in rows)), "unfitted_auroc": {}}
    for k in UNFITTED:
        out["unfitted_auroc"][k] = R.auroc(col[k], y)
    base = max(["first_p", "entity_p", "mean_lp", "min_lp"], key=lambda k: out["unfitted_auroc"][k])
    out["best_output_baseline"] = base
    out["question_signed_minus_best_output"] = boot(col["question_signed"], col[base], y)
    idx = np.random.default_rng(0).permutation(len(rows))
    fit, test = idx[: len(idx) // 2], idx[len(idx) // 2:]
    out["fitted_test_auroc"] = {}

    def fitted(name, X):
        pr, _, l2 = R.fit_probe_cv(X[fit], y[fit], "cpu")
        s = pr(X[test])
        out["fitted_test_auroc"][name] = {"auroc": R.auroc(s, y[test]), "l2": l2}
        return s
    outp = np.stack([col[k] for k in ("first_p", "entity_p", "mean_lp", "min_lp")], 1)
    attr = np.stack([col[k] for k in ("question_signed", "question_share", "answer_signed", "answer_share",
                                      "template_signed", "logit")], 1)
    s_out = fitted("output_probabilities", outp)
    s_oa = fitted("output_plus_attribution", np.concatenate([outp, attr], 1))
    out["attribution_gain_over_output_test"] = boot(s_oa, s_out, y[test])
    best, h = None, len(fit) // 2                   # probe layer chosen inside the fitting half
    for L in range(0, len(rows[0]["h_entity"]), 2):
        X = np.stack([rows[i]["h_entity"][L] for i in fit])
        pr, _ = R.fit_probe(X[:h], y[fit][:h], "cpu")
        a = R.auroc(pr(X[h:]), y[fit][h:])
        if a is not None and (best is None or a > best[0]):
            best = (a, L)
    L = best[1]
    H = np.stack([r["h_entity"][L] for r in rows])
    s_p = fitted("probe_at_entity", H)
    out["fitted_test_auroc"]["probe_at_entity"]["layer"] = L
    s_pa = fitted("probe_plus_output_plus_attribution", np.concatenate([H, outp, attr], 1))
    out["attribution_and_output_gain_over_probe_test"] = boot(s_pa, s_p, y[test])
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "qwen_correctness_entity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
