"""Can a general verifier be trained without human labels? A cheap student (Qwen3-1.7B features + linear head) is trained
on the TEACHER's scores (the two one-pass judges, Qwen3-14B + phi-4; no human labels anywhere) pooled over every other
domain, and evaluated with human labels on the held-out domain (LODO). Compared with the same student trained on human
labels, and with the teacher itself. If the judge-trained student matches the human-trained one, a general verifier
scales with unlabelled traffic rather than with human labelling.

    python experiments/general_student.py --feats /data/omni/q17 --out results/omni/general_student.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_corpus import load_features  # noqa: E402
from omni_eval import LogReg, auroc, balanced_weights, pca_fit, standardise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--judges", default="results/omni/judge_probs.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = load_features(a.feats)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.corpus)}
    ids = F["ids"]
    dom = np.array([rows[int(i)]["domain"] for i in ids])
    fmt = np.array([rows[int(i)]["format"] for i in ids])
    y = np.array([rows[int(i)]["label"] for i in ids]).astype(float)
    n, mid = len(y), [2, 3, 4]
    H = np.concatenate([F["r_mean"][:, mid].reshape(n, -1), F["j_state"][:, mid].reshape(n, -1)], 1).astype(np.float32)
    H = standardise(H)
    mu, V = pca_fit(H, 256)
    X = standardise(np.concatenate([(H - mu) @ V.T, F["r_sc"], F["j_logit"][:, None]], 1), dom)
    J = np.load(a.judges)
    jp = {int(i): k for k, i in enumerate(J["ids"])}
    o = [jp[int(i)] for i in ids]
    t = -(J["logit_qwen14"][o] + J["logit_phi4"][o]) / 2          # teacher: higher = wrong
    tz = np.zeros(n)
    for d in np.unique(dom):                                    # per-domain z (unlabelled), then soft labels
        m = dom == d
        tz[m] = (t[m] - t[m].mean()) / (t[m].std() + 1e-9)
    soft = 1 / (1 + np.exp(-1.5 * tz))
    hard = (tz > 0).astype(float)
    res = {k: {} for k in ("student_human_labels", "student_teacher_soft", "student_teacher_hard", "teacher",
                           "student_teacher_soft+teacher")}
    for d in np.unique(dom):
        te = dom == d
        res["teacher"][d] = auroc(t[te], y[te])
        h = LogReg(0.05).fit(X[~te], y[~te], balanced_weights(y[~te]))
        res["student_human_labels"][d] = auroc(h.score(X[te]), y[te])
        s = LogReg(0.05).fit(X[~te], soft[~te])
        res["student_teacher_soft"][d] = auroc(s.score(X[te]), y[te])
        hh = LogReg(0.05).fit(X[~te], hard[~te], balanced_weights(hard[~te]))
        res["student_teacher_hard"][d] = auroc(hh.score(X[te]), y[te])
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        res["student_teacher_soft+teacher"][d] = auroc(z(s.score(X[te])) + z(t[te]), y[te])
    F_ = {d: fmt[dom == d][0] for d in np.unique(dom)}
    summ = {k: {g: float(np.mean([x for d, x in v.items() if g == "all" or F_[d] == g]))
                for g in ("all", "reasoning", "longform", "rag", "chat")} for k, v in res.items()}
    for k, v in summ.items():
        print(f"{k:30s}", {g: round(x, 3) for g, x in v.items()})
    json.dump({"per_domain": res, "summary": summ}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
