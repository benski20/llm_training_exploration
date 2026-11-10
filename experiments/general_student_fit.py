"""Fit and test the general student (data from kaggle/omni/general_student.py): a linear head on the student reader's
features (reply mean + judge state at three middle layers, output scalars, own judge), trained ONLY on the teacher
judge's scores of diverse UltraFeedback replies (no human labels), tested with human labels on all 21 corpus domains
(none seen in training), against the same reader's head trained on human labels leave-one-domain-out.

    python experiments/general_student_fit.py --dir results/kaggle --out results/omni/general_student_scaled.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_eval import LogReg, auroc, balanced_weights  # noqa: E402


def vec(z):
    n = len(z["ids"])
    return np.concatenate([z["r_mean"].reshape(n, -1), z["j_state"].reshape(n, -1)], 1).astype(np.float32), \
        np.concatenate([z["r_sc"], z["j_logit"][:, None]], 1).astype(np.float32)


def fit_eval(Htr, Str, t, Hte, Ste, dom_te, y_te, fmt_te, pca=256, l2=0.05, sizes=(None,)):
    """train on teacher scores t (z-scored, soft labels); per-domain standardised test features (unlabelled)."""
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-4
    Z = (Htr - mu) / sd
    pm = Z.mean(0)
    _, _, Vt = np.linalg.svd((Z - pm)[np.random.default_rng(0).permutation(len(Z))[:20000]], full_matrices=False)
    V = Vt[:pca]
    Xtr = np.concatenate([(Z - pm) @ V.T, Str], 1)
    Xtr = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-6)
    Xte = np.concatenate([(((Hte - mu) / sd) - pm) @ V.T, Ste], 1)
    for d in np.unique(dom_te):                                      # each test domain by its own statistics
        m = dom_te == d
        Xte[m] = (Xte[m] - Xte[m].mean(0)) / (Xte[m].std(0) + 1e-6)
    tz = (t - np.nanmean(t)) / (np.nanstd(t) + 1e-9)
    ok = ~np.isnan(tz)
    soft = 1 / (1 + np.exp(-1.5 * tz))
    out = {}
    rng = np.random.default_rng(1)
    for n in sizes:
        idx = np.where(ok)[0] if n is None or n >= ok.sum() else rng.choice(np.where(ok)[0], n, replace=False)
        m = LogReg(l2).fit(Xtr[idx], soft[idx])
        s = m.score(Xte)
        per = {d: auroc(s[dom_te == d], y_te[dom_te == d]) for d in np.unique(dom_te)}
        fm = {d: fmt_te[dom_te == d][0] for d in per}
        out[str(n or int(ok.sum()))] = {"per_domain": per, "mean": float(np.mean(list(per.values()))),
                                        "by_format": {f: float(np.mean([v for d, v in per.items() if fm[d] == f]))
                                                      for f in sorted(set(fm.values()))}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/kaggle", help="where the omni-gs-* kernel outputs were fetched")
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = {}
    for p in glob.glob(os.path.join(a.dir, "**", "gs_judge_*.npz"), recursive=True):
        z = np.load(p, allow_pickle=True)
        J.update(dict(zip(z["ids"].tolist(), z["logodds_wrong"].tolist())))
    H, S, ids = [], [], []
    for p in sorted(glob.glob(os.path.join(a.dir, "**", "gs_feats_*.npz"), recursive=True)):
        z = np.load(p, allow_pickle=True)
        h, s = vec(z)
        keep = [k for k, i in enumerate(z["ids"].tolist()) if i in J]
        H.append(h[keep])
        S.append(s[keep])
        ids += [z["ids"][k] for k in keep]
    H, S = np.concatenate(H), np.concatenate(S)
    t = np.array([J[i] for i in ids], float)
    cp = glob.glob(os.path.join(a.dir, "**", "gs_corpus_feats.npz"), recursive=True)[0]
    C = np.load(cp, allow_pickle=True)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.corpus)}
    cid = C["ids"].tolist()
    dom = np.array([rows[int(i)]["domain"] for i in cid])
    fmt = np.array([rows[int(i)]["format"] for i in cid])
    y = np.array([rows[int(i)]["label"] for i in cid])
    Hc, Sc = vec(C)
    n = len(t)
    sizes = tuple(k for k in (2000, 8000, 32000) if k < n) + (None,)
    res = {"n_train": int(n), "student_on_teacher": fit_eval(H, S, t, Hc, Sc, dom, y, fmt, sizes=sizes)}
    # the same reader, trained on human labels, leave one domain out (the comparison)
    from omni_eval import lodo, pca_fit, standardise
    Z = standardise(Hc)
    mu, V = pca_fit(Z, 256)
    X = np.concatenate([(Z - mu) @ V.T, Sc], 1)
    per, _ = lodo(X, y.astype(float), dom, 0.05, True)
    res["human_labels_lodo"] = {"per_domain": per, "mean": float(np.mean(list(per.values())))}
    json.dump(res, open(a.out, "w"), indent=1)
    print("trained on", n, "teacher-labelled replies")
    for k, v in res["student_on_teacher"].items():
        print(f"  student on teacher, {k:>6s} replies: {v['mean']:.3f}", {f: round(x, 3) for f, x in v["by_format"].items()})
    print(f"  same reader, human labels LODO: {res['human_labels_lodo']['mean']:.3f}")


if __name__ == "__main__":
    main()
