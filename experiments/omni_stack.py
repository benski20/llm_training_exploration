"""The full shipped stack on held-out domains, from exported per-reply expert scores (each already leave-one-domain-out):

    reader experts   reply head, worst unit, the reader's own judge  (expert_scores_<reader>.npz)
    nli              MiniCheck + HHEM on document-grounded replies     (experiments/omni_nli.py)
    checkers         written domain checkers (vt/checkers), exact arithmetic (vt/mathcheck.py)

Combination per domain:
    zero labels   z-sum of available experts (standardised on the domain's own unlabelled replies), the judge weighted
                  by --judge-weight, written-checker / arithmetic flags adding a fixed bump
    k labels      expert weights refitted on k labelled replies (pulled toward the zero-label weights)
    guarantee     k uniform labels at 20% prevalence -> Clopper-Pearson threshold on clean ones; false alarms / caught

    python experiments/omni_stack.py --scores results/omni/expert_scores_q8b.npz --nli /data/omni/nli.npz \
        --corpus corpus.jsonl --report results/omni/stack_q8b.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omni_eval import LogReg, auroc, balanced_weights, cp_threshold  # noqa: E402


def z(v):
    v = np.asarray(v, float)
    m = np.isnan(v)
    if m.all():
        return np.zeros_like(v)
    return np.where(m, 0.0, (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--nli")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--judge-weight", type=float, default=1.0)
    ap.add_argument("--flag-weight", type=float, default=2.0)
    ap.add_argument("--nli-only-covered", action="store_true",
                    help="evaluate grounded domains only on replies that have NLI scores")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    S = np.load(a.scores, allow_pickle=True)
    ids = S["ids"].astype(int)
    rows = {}
    for line in open(a.corpus):
        r = json.loads(line)
        rows[r["id"]] = r
    dom = np.array([rows[i]["domain"] for i in ids])
    fmt = np.array([rows[i]["format"] for i in ids])
    y = np.array([rows[i]["label"] for i in ids])
    n = len(ids)
    ex = {"reply": S["reply"], "worst_unit": S["worst_unit"], "judge": S["judge"]}
    pos = {int(i): k for k, i in enumerate(ids)}
    has_nli = np.zeros(n, bool)
    if a.nli and os.path.exists(a.nli):
        Z = np.load(a.nli)
        nli = np.full(n, np.nan)
        for i, r in zip(Z["ids"], Z["nli"]):
            if int(i) in pos:
                nli[pos[int(i)]] = (r[2] + r[1]) / 2          # HHEM reply + MiniCheck worst sentence
        has_nli = ~np.isnan(nli)
        ex["nli"] = nli
    from vt.checkers.business_record import check as check_record
    from vt.mathcheck import check_arithmetic
    flag = np.zeros(n)
    for k, i in enumerate(ids):
        r = rows[i]
        if r["domain"] == "ragtruth_data2text":
            flag[k] += min(len(check_record(r["response"], r["doc"])), 2)
        explicit = r["format"] != "reasoning" and "math" not in r["domain"]
        flag[k] += float(any(not e["ok"] for e in check_arithmetic(r["response"], explicit=explicit)))
    names = list(ex)
    w0 = np.array([a.judge_weight if nm == "judge" else 1.0 for nm in names] + [a.flag_weight])
    rep = {"scores": a.scores, "experts": names, "judge_weight": a.judge_weight, "per_domain": {}}
    rng = np.random.default_rng(0)
    for d in np.unique(dom):
        idx = np.where(dom == d)[0]
        if a.nli_only_covered and has_nli[idx].any():
            idx = idx[has_nli[idx]]
        Zd = np.stack([z(ex[nm][idx]) for nm in names] + [flag[idx]], 1)
        yd = y[idx]
        s0 = Zd @ w0
        row = {"format": fmt[idx[0]], "n": int(len(idx)), "zero_label": auroc(s0, yd),
               "reader_only": auroc(Zd[:, :3] @ w0[:3], yd)}
        for j, nm in enumerate(names):
            row[nm] = auroc(Zd[:, j], yd) if not np.all(Zd[:, j] == 0) else None
        for k in (25, 50, 100):
            p_, n_ = np.where(yd == 1)[0], np.where(yd == 0)[0]
            if k // 2 >= len(p_) - 10 or k - k // 2 >= len(n_) - 10:
                continue
            res = []
            for _ in range(a.reps):
                cal = np.concatenate([rng.choice(p_, k // 2, replace=False), rng.choice(n_, k - k // 2, replace=False)])
                rest = np.setdiff1d(np.arange(len(yd)), cal)
                m = LogReg(2.0 / k, w0=w0 * 0.5).fit(Zd[cal], yd[cal].astype(float), balanced_weights(yd[cal].astype(float)))
                res.append(auroc(m.score(Zd[rest]), yd[rest]))
            row[f"adapt_{k}"] = float(np.nanmean(res))
        neg, pos_ = np.where(yd == 0)[0], np.where(yd == 1)[0]
        npos = min(len(pos_), int(round(len(neg) * 0.25)))
        if 100 < len(neg) + npos - 40:
            fa, rc = [], []
            for _ in range(a.reps):
                pool = np.concatenate([neg, rng.choice(pos_, npos, replace=False)])
                cal = rng.choice(pool, 100, replace=False)
                rest = np.setdiff1d(pool, cal)
                t = cp_threshold(s0[cal][yd[cal] == 0], 0.10, 0.95)
                fa.append(float((s0[rest][yd[rest] == 0] >= t).mean()))
                rc.append(float((s0[rest][yd[rest] == 1] >= t).mean()))
            row["guarantee_100"] = {"fpr": float(np.mean(fa)), "fpr_over_12pct": float(np.mean(np.array(fa) > 0.12)),
                                    "recall": float(np.mean(rc))}
        rep["per_domain"][d] = row
    pdm = rep["per_domain"]
    summ = {}
    for g in ["all"] + sorted(set(fmt.tolist())):
        mem = [d for d in pdm if g == "all" or pdm[d]["format"] == g]
        summ[g] = {k: float(np.nanmean([pdm[d][k] for d in mem if pdm[d].get(k) is not None]))
                   for k in ("reader_only", "zero_label", "adapt_25", "adapt_50", "adapt_100")
                   if any(pdm[d].get(k) is not None for d in mem)}
        gs = [pdm[d]["guarantee_100"] for d in mem if "guarantee_100" in pdm[d]]
        if gs:
            summ[g]["guarantee_100"] = {k: float(np.mean([x[k] for x in gs])) for k in gs[0]} | {"domains": len(gs)}
    rep["summary"] = summ
    json.dump(rep, open(a.report, "w"), indent=1)
    for g, s in summ.items():
        print(g, {k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in s.items()})


if __name__ == "__main__":
    main()
