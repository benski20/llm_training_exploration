"""With a domain's own labels, does an internals-only probe overtake the LLM judges?

Stored Qwen3-4B reader features (reply mean + judge-question state, middle layers, output scalars; experiments/
omni_corpus.py) and stored one-pass judge outputs (results/omni/judge_probs.npz: Qwen3-14B + phi-4). For each of the 21
domains and k labelled replies from that domain (half wrong, half right; 20 random draws), compare on the domain's
remaining replies:
    probe_prior   the cross-domain probe (trained on the other 20 domains) refit on the k labels, pulled to its prior
    probe_fresh   a probe trained on the k labels only
    judges        z(-logit_qwen14) + z(-logit_phi4) (no labels used; the same test replies)
    both          z(probe_prior) + z(judges)
No model is run. Mean of per-domain AUROC, by format.

    python experiments/novel_label_curve.py --feats /home/user/omni/q4b --out results/omni/novel_label_curve.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_corpus import load_features  # noqa: E402
from omni_eval import LogReg, auroc, balanced_weights, pca_fit, reply_matrix, standardise  # noqa: E402


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/home/user/omni/q4b")
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--judges", default="results/omni/judge_probs.npz")
    ap.add_argument("--ks", default="25,50,100,200")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = load_features(a.feats)
    rows = {}
    for l in open(a.corpus):
        r = json.loads(l)
        rows[r["id"]] = r
    ids = [int(i) for i in F["ids"]]
    dom = np.array([rows[i]["domain"] for i in ids])
    fmt = {rows[i]["domain"]: rows[i]["format"] for i in ids}
    y = np.array([rows[i]["label"] for i in ids], float)
    J = np.load(a.judges, allow_pickle=True)
    jmap = {int(i): (q, p) for i, q, p in zip(J["ids"], J["logit_qwen14"], J["logit_phi4"])}
    jq = np.array([-jmap[i][0] for i in ids])
    jp = np.array([-jmap[i][1] for i in ids])
    nl = len(F["meta"]["layers"])
    mid = [nl // 2 - 1, nl // 2, nl // 2 + 1]
    H, S = reply_matrix(F, "mean+judge+scalar", mid)
    Z = standardise(H)
    mu, V = pca_fit(Z, 256)
    X = standardise(np.concatenate([(Z - mu) @ V.T, S], 1), dom)       # per-domain standardisation, label-free
    ks = [int(k) for k in a.ks.split(",")]
    res = {}
    rng = np.random.default_rng(0)
    for g in sorted(set(dom)):
        te = dom == g
        prior = LogReg(a.l2).fit(X[~te], y[~te], balanced_weights(y[~te]))
        Xg, yg = X[te], y[te]
        jg = z(jq[te]) + z(jp[te])
        res[g] = {"format": fmt[g], "n": int(te.sum()), "judges_all": auroc(jg, yg),
                  "probe_zero_label": auroc(prior.score(Xg), yg)}
        pos, neg = np.where(yg == 1)[0], np.where(yg == 0)[0]
        for k in ks:
            if k // 2 >= min(len(pos), len(neg)) - 10:
                continue
            acc = {m: [] for m in ("probe_prior", "probe_fresh", "judges", "both")}
            for _ in range(a.reps):
                lab = np.r_[rng.choice(pos, k // 2, replace=False), rng.choice(neg, k // 2, replace=False)]
                test = np.setdiff1d(np.arange(len(yg)), lab)
                mp = LogReg(a.l2, w0=prior.w, b0=prior.b).fit(Xg[lab], yg[lab])
                mf = LogReg(a.l2).fit(Xg[lab], yg[lab])
                sp, sf = mp.score(Xg[test]), mf.score(Xg[test])
                acc["probe_prior"].append(auroc(sp, yg[test]))
                acc["probe_fresh"].append(auroc(sf, yg[test]))
                acc["judges"].append(auroc(jg[test], yg[test]))
                acc["both"].append(auroc(z(sp) + z(jg[test]), yg[test]))
            res[g][str(k)] = {m: float(np.mean(v)) for m, v in acc.items()}
    summ = {}
    for k in ["zero"] + [str(k) for k in ks]:
        for f in sorted(set(fmt.values())) + ["all"]:
            ds = [d for d in res if (f == "all" or res[d]["format"] == f) and (k == "zero" or k in res[d])]
            if not ds:
                continue
            if k == "zero":
                summ.setdefault(k, {})[f] = {"probe": float(np.mean([res[d]["probe_zero_label"] for d in ds])),
                                             "judges": float(np.mean([res[d]["judges_all"] for d in ds])), "domains": len(ds)}
            else:
                summ.setdefault(k, {})[f] = {m: float(np.mean([res[d][k][m] for d in ds])) for m in res[ds[0]][k]}
                summ[k][f]["domains"] = len(ds)
    json.dump({"summary": summ, "per_domain": res}, open(a.out, "w"), indent=1)
    for k, v in summ.items():
        print(k, {f: {m: round(x, 3) for m, x in d.items()} for f, d in v.items()})


if __name__ == "__main__":
    main()
