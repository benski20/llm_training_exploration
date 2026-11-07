"""The omni verifier, end to end, on held-out domains: every expert, their combination, adaptation and the guarantee.

Experts (each scored on a held-out domain by a head that never saw it: leave-one-domain-out, LODO):
    reply       logistic head on [middle-layer mean + judge states (PCA), output signals, judge log-odds]
    worst_unit  unit head (sentences / steps / segments) -> the reply's worst unit
    judge       the reader's own log-odds of "No" to "is the reply above correct?" (no training)
    evidence    head on evidence-shift signals (document-grounded replies only; experiments/omni_evidence.py)
    exact       arithmetic re-computed exactly (vt/mathcheck.py): a wrong stated result is a flag
Combination:
    zero labels   sum of the experts' z-scores, each standardised on the domain's own (unlabelled) replies,
                  exact flags adding a fixed +2
    k labels      expert weights fitted on k labelled replies (pulled toward equal weights)
Guarantee: k uniformly sampled replies at 20% prevalence; Clopper-Pearson threshold on their clean ones (cross-fitted
when the same labels fit the weights); false alarms and recall on the rest.

    python experiments/omni_final.py --feats /data/omni/q17 --corpus corpus.jsonl --evidence /data/omni/q17_ev.npz \
        --report results/omni/final_q17.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omni_corpus import load_features  # noqa: E402
from omni_eval import LogReg, auroc, balanced_weights, cp_threshold, lodo, pca_fit, standardise  # noqa: E402


def zscore(v):
    v = v.astype(float)
    m = np.isnan(v)
    if m.all():
        return np.zeros_like(v)
    mu, sd = np.nanmean(v), np.nanstd(v) + 1e-9
    return np.where(m, 0.0, (v - mu) / sd)


def mean(d):
    v = [x for x in d.values() if x == x]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--evidence")
    ap.add_argument("--nli", help="experiments/omni_nli.py output: MiniCheck + HHEM on document-grounded replies")
    ap.add_argument("--no-evidence-expert", action="store_true")
    ap.add_argument("--report", required=True)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--exact-weight", type=float, default=2.0)
    ap.add_argument("--same-ids-as", help="another features dir: keep only the replies it has (fair comparisons)")
    a = ap.parse_args()
    t0 = time.time()
    F = load_features(a.feats)
    if a.same_ids_as:
        from vt.omni import subset
        keep = set(load_features(a.same_ids_as)["ids"].tolist())
        F = subset(F, np.array([int(i) in keep for i in F["ids"]]))
    rows = {}
    for line in open(a.corpus):
        r = json.loads(line)
        rows[r["id"]] = r
    ids = F["ids"]
    dom = np.array([rows[i]["domain"] for i in ids])
    fmt = np.array([rows[i]["format"] for i in ids])
    y = np.array([rows[i]["label"] for i in ids])
    n, nl = len(y), F["r_mean"].shape[1]
    mid = [nl // 2 - 1, nl // 2, nl // 2 + 1]
    pos = {int(i): k for k, i in enumerate(ids)}
    experts = {}
    rep_nli = None

    # reply head
    H = np.concatenate([F["r_mean"][:, mid].reshape(n, -1), F["j_state"][:, mid].reshape(n, -1)], 1).astype(np.float32)
    H = standardise(H)
    mu, V = pca_fit(H, a.pca)
    X = np.concatenate([(H - mu) @ V.T, F["r_sc"], F["j_logit"][:, None]], 1)
    _, experts["reply"] = lodo(X, y, dom, a.l2, True)
    print(f"reply head done ({time.time() - t0:.0f}s)", flush=True)

    # unit head -> worst unit
    U = F["u_mean"][:, mid].reshape(len(F["u_row"]), -1).astype(np.float32)
    U = standardise(U)
    umu, UV = pca_fit(U[np.random.default_rng(0).permutation(len(U))[:20000]], a.pca)
    UX = np.concatenate([(U - umu) @ UV.T, F["u_sc"]], 1)
    udom = dom[[pos[int(i)] for i in F["u_row"]]]
    UXs = standardise(UX, udom)
    worst = np.full(n, np.nan)
    unit_loc = {}
    for d in np.unique(dom):
        tr, te = (udom != d) & (F["u_lab"] >= 0), udom == d
        m = LogReg(a.l2).fit(UXs[tr], F["u_lab"][tr].astype(float), balanced_weights(F["u_lab"][tr].astype(float)))
        su = m.score(UXs[te])
        lab = F["u_lab"][te]
        if (lab >= 0).sum() > 10 and len(set(lab[lab >= 0].tolist())) == 2:
            unit_loc[d] = auroc(su[lab >= 0], lab[lab >= 0])
        for r_, v in zip(F["u_row"][te], su):
            k = pos[int(r_)]
            worst[k] = v if np.isnan(worst[k]) else max(worst[k], v)
    experts["worst_unit"] = worst
    print(f"unit head done ({time.time() - t0:.0f}s)", flush=True)

    experts["judge"] = -F["j_logit"].astype(float)

    # evidence
    if a.evidence and os.path.exists(a.evidence):
        E = np.load(a.evidence)
        ev = np.full((n, E["ev"].shape[1]), np.nan)
        for i, row in zip(E["ids"], E["ev"]):
            if int(i) in pos:
                ev[pos[int(i)]] = row
        has = ~np.isnan(ev).any(1)
        evs = np.full(n, np.nan)
        if has.sum() > 50:
            _, s_ev = lodo(ev[has], y[has], dom[has], a.l2, True)
            evs[has] = s_ev
        experts["evidence"] = evs
        print(f"evidence: {has.sum()} replies", flush=True)

    # NLI grounding experts (MiniCheck, HHEM): a LODO head on their four scores, document-grounded replies only
    if a.nli and os.path.exists(a.nli):
        Z_ = np.load(a.nli)
        nl = np.full((n, Z_["nli"].shape[1]), np.nan)
        for i, row in zip(Z_["ids"], Z_["nli"]):
            if int(i) in pos:
                nl[pos[int(i)]] = row
        has = ~np.isnan(nl).any(1)
        s_nl = np.full(n, np.nan)
        if has.sum() > 50 and len(np.unique(dom[has])) > 1:
            _, s_ = lodo(nl[has], y[has], dom[has], a.l2, True)
            s_nl[has] = s_
        experts["nli"] = s_nl
        rep_nli = {}
        for d in np.unique(dom[has]):
            m = has & (dom == d)
            rep_nli[d] = {"n": int(m.sum()), "nli_head": auroc(s_nl[m], y[m]),
                          **{nm: auroc(nl[m, j], y[m]) for j, nm in enumerate(("minicheck_reply", "minicheck_worst",
                                                                                   "hhem_reply", "hhem_worst"))}}
        print("NLI experts on their replies:", json.dumps({d: {k: round(v, 2) for k, v in r_.items()}
                                                           for d, r_ in rep_nli.items()}), flush=True)
    if a.no_evidence_expert:
        experts.pop("evidence", None)

    # exact arithmetic
    from vt.mathcheck import check_arithmetic
    exact = np.zeros(n)
    for k, i in enumerate(ids):
        r = rows[int(i)]
        explicit = r["format"] != "reasoning" and "math" not in r["domain"]
        exact[k] = sum(not e["ok"] for e in check_arithmetic(r["response"], explicit=explicit))
    exact_flag = (exact > 0).astype(float)

    names = list(experts)
    Z = np.zeros((n, len(names)))
    for d in np.unique(dom):
        idx = dom == d
        for j, nm in enumerate(names):
            Z[idx, j] = zscore(experts[nm][idx])
    final0 = Z.sum(1) + a.exact_weight * exact_flag

    rep = {"reader": F["meta"]["reader"], "n": int(n), "experts": names, "per_domain": {}, "exact": {}, "nli": rep_nli}
    for d in np.unique(dom):
        idx = dom == d
        row = {nm: auroc(Z[idx, j], y[idx]) for j, nm in enumerate(names)}
        row["zero_label"] = auroc(final0[idx], y[idx])
        row["zero_label_no_exact"] = auroc(Z[idx].sum(1), y[idx])
        row["unit_localisation"] = unit_loc.get(d)
        row["format"] = rows[int(ids[idx][0])]["format"]
        row["n"] = int(idx.sum())
        rep["per_domain"][d] = row
        if exact_flag[idx].sum():
            rep["exact"][d] = {"fires_wrong": int(exact_flag[idx & (y == 1)].sum()),
                               "fires_correct": int(exact_flag[idx & (y == 0)].sum())}

    # adaptation: expert weights from k balanced labels
    rng = np.random.default_rng(1)
    Zx = np.concatenate([Z, exact_flag[:, None]], 1)
    w_eq = np.concatenate([np.ones(len(names)), [a.exact_weight]])
    for k in (25, 50, 100):
        for d in np.unique(dom):
            idx = np.where(dom == d)[0]
            yd, Zd = y[idx], Zx[idx]
            p_, n_ = np.where(yd == 1)[0], np.where(yd == 0)[0]
            if k // 2 >= len(p_) - 10 or k - k // 2 >= len(n_) - 10:
                continue
            res = []
            for _ in range(a.reps):
                cal = np.concatenate([rng.choice(p_, k // 2, replace=False), rng.choice(n_, k - k // 2, replace=False)])
                rest = np.setdiff1d(np.arange(len(yd)), cal)
                m = LogReg(2.0 / k, w0=w_eq * 0.5).fit(Zd[cal], yd[cal].astype(float), balanced_weights(yd[cal].astype(float)))
                res.append(auroc(m.score(Zd[rest]), yd[rest]))
            rep["per_domain"][d][f"adapt_{k}"] = float(np.nanmean(res))

    # realistic guarantee on the zero-label score (calibration only) at 20% prevalence
    for k in (50, 100, 150):
        for d in np.unique(dom):
            idx = np.where(dom == d)[0]
            yd, sd_ = y[idx], final0[idx]
            neg, pos_ = np.where(yd == 0)[0], np.where(yd == 1)[0]
            npos = min(len(pos_), int(round(len(neg) * 0.25)))
            if k >= len(neg) + npos - 40:
                continue
            fa, rc = [], []
            for _ in range(a.reps):
                pool = np.concatenate([neg, rng.choice(pos_, npos, replace=False)])
                cal = rng.choice(pool, k, replace=False)
                rest = np.setdiff1d(pool, cal)
                t = cp_threshold(sd_[cal][yd[cal] == 0], 0.10, 0.95)
                fa.append(float((sd_[rest][yd[rest] == 0] >= t).mean()))
                rc.append(float((sd_[rest][yd[rest] == 1] >= t).mean()))
            rep["per_domain"][d][f"guarantee_{k}"] = {"fpr": float(np.mean(fa)), "fpr_over_12pct": float(np.mean(np.array(fa) > 0.12)),
                                                     "recall": float(np.mean(rc))}

    # summaries
    pdm = rep["per_domain"]
    summ = {}
    for group, members in [("all", list(pdm))] + [(f, [d for d in pdm if pdm[d]["format"] == f]) for f in
                                                   sorted(set(fmt.tolist()))]:
        s = {key: float(np.nanmean([pdm[d][key] for d in members if pdm[d].get(key) is not None]))
             for key in names + ["zero_label", "zero_label_no_exact", "adapt_25", "adapt_50", "adapt_100"]
             if any(pdm[d].get(key) is not None for d in members)}
        for k in (50, 100, 150):
            g = [pdm[d][f"guarantee_{k}"] for d in members if f"guarantee_{k}" in pdm[d]]
            if g:
                s[f"guarantee_{k}"] = {kk: float(np.mean([x[kk] for x in g])) for kk in g[0]} | {"domains": len(g)}
        summ[group] = s
    rep["summary"] = summ
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(f"\n{'domain':20s} " + " ".join(f"{x[:8]:>8s}" for x in names + ["zero", "k=50", "k=100"]))
    for d in sorted(pdm):
        r = pdm[d]
        print(f"{d:20s} " + " ".join(f"{(r.get(x) if r.get(x) is not None else float('nan')):8.2f}"
                                     for x in names + ["zero_label", "adapt_50", "adapt_100"]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
