"""Can an offline LLM judge's labels replace human labels when calibrating the false-alarm guarantee?

Inputs: the shipped zero-label score per reply (results/omni/expert_scores_q8b.npz: per-domain z-sum of reply +
worst_unit + 2*judge, the reader's own judge; every expert held out leave-one-domain-out, as experiments/omni_stack.py)
and an external judge's P(correct) per reply (results/omni/judge_probs.npz, from kaggle/omni/judge_calibration.py).

Protocol per domain (as omni_stack.py): a pool of all clean replies + wrong ones at 20% prevalence; draw N replies
uniformly ("onboarding sample"), set a threshold by each method on the sample, measure the TRUE false-alarm rate on the
clean replies of the rest of the pool and recall on its wrong ones (human labels). Target: <= 10% false alarms with
95% confidence. Methods:

    human        Clopper-Pearson (CP) threshold on the human-clean replies of the sample (the shipped protocol)
    naive        CP threshold on the judge-clean replies (P(correct) >= 0.5), judge labels taken as truth
    conf>=p      CP threshold on replies the judge is very confident are clean (P(correct) >= p)
    conf-top30   CP threshold on the 30% of the sample with the highest P(correct) (rank-based, bias-free)
    ppi-m        prediction-powered risk control (Angelopoulos et al. 2023; Einbinder et al. 2024): control
                 E[(1-y)(1{s>=t} - alpha)] <= 0, imputed with the judge's soft P(clean) on all N and rectified by m
                 uniform human labels from the sample; normal upper bound at 95%, thresholds scanned from the top
                 (fixed-sequence). m = 0 is the soft judge alone.
    bound-m      a finite-sample conservative bound: with A = judge-clean, B = judge-wrong,
                   FAR(t) <= [P(s>=t, A) + P(B) P(clean, s>=t | B)] / [P(A) (1 - P(wrong | A))],
                 each term bounded by exact CP at delta/5 (Bonferroni), m/2 human labels drawn in A and m/2 in B;
                 fixed-sequence over t. Valid whatever the judge's errors; infeasible draws flag nothing.

    python experiments/judge_calibration_sim.py --judge results/omni/judge_probs.npz --draws 200 \
        --report results/omni/judge_calibration.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_eval import auroc, cp_threshold  # noqa: E402

ALPHA, CONF = 0.10, 0.95
MIN_CLEAN = 29            # (1 - 0.1)^29 < 0.05: fewer clean replies cannot carry the guarantee


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def cp_up(k, n, d):
    return 1.0 if n == 0 else (1.0 if k >= n else float(beta_dist.ppf(1 - d, k + 1, n - k)))


def cp_lo(k, n, d):
    return 0.0 if (n == 0 or k == 0) else float(beta_dist.ppf(d, k, n - k + 1))


def cands(s):
    """candidate thresholds, highest first: just above each sample score (flag s >= t)."""
    return np.concatenate([[np.inf], np.sort(s)[::-1] + 1e-9])


def fixed_sequence(ts, ok):
    """lowest t reached scanning from the top while ok(t) holds (Learn-then-Test fixed-sequence testing)."""
    best = np.inf
    for t in ts:
        if not ok(t):
            break
        best = t
    return best


def ppi_threshold(s, qc, lab, y_lab):
    """PPI risk control. s, qc: score and judge P(clean) for the N sample; lab: indices of the m human-labelled
    replies inside the sample, y_lab their labels (1 = wrong)."""
    zq = norm.ppf(CONF)
    N, m = len(s), len(lab)

    def ok(t):
        f = (s >= t) - ALPHA
        lhat = qc * f
        est, var = lhat.mean(), lhat.var(ddof=1) / N
        if m:
            r = (1 - y_lab) * f[lab] - lhat[lab]
            est += r.mean()
            var += r.var(ddof=1) / m
        return est + zq * np.sqrt(var) <= 0
    return fixed_sequence(cands(s), ok)


def bound_threshold(s, A, labA_wrong, labB_clean_scores, mB):
    """conservative stratified bound. A: judge-clean mask over the sample; labA_wrong: #wrong among the human
    labels drawn in A (of mA); labB_clean_scores: scores of the human-labelled clean replies drawn in B (of mB)."""
    N, nA = len(s), int(A.sum())
    nB = N - nA
    mA = labA_wrong[1]
    d = (1 - CONF) / 5
    pA_lo = cp_lo(nA, N, d)
    pB_up = cp_up(nB, N, d)
    epsA = cp_up(labA_wrong[0], mA, d) if mA else 1.0
    den = pA_lo * (1 - epsA)
    if den <= 0:
        return np.inf

    def ok(t):
        k = int(((s >= t) & A).sum())
        num = cp_up(k, N, d) + (pB_up * cp_up(int((labB_clean_scores >= t).sum()), mB, d) if nB else 0.0)
        return num / den <= ALPHA
    return fixed_sequence(cands(s), ok)


def cp_or_inf(sc):
    return cp_threshold(sc, ALPHA, CONF) if len(sc) else np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/omni/expert_scores_q8b.npz")
    ap.add_argument("--judge", default="results/omni/judge_probs.npz")
    ap.add_argument("--judges", nargs="+", default=None, help="judge keys in the npz (default: all + ensemble)")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 60])
    ap.add_argument("--report", default="results/omni/judge_calibration.json")
    a = ap.parse_args()
    S = np.load(a.scores, allow_pickle=True)
    J = np.load(a.judge, allow_pickle=True)
    ids, dom, fmt, y = S["ids"].astype(int), S["domain"], S["format"], S["label"].astype(int)
    keys = a.judges or [k[len("logit_"):] for k in J.files if k.startswith("logit_")]
    pos = {int(i): k for k, i in enumerate(J["ids"])}
    have = np.array([int(i) in pos for i in ids])
    jl = {}
    for k in keys:
        v = J["logit_" + k]
        jl[k] = np.array([v[pos[int(i)]] if int(i) in pos else np.nan for i in ids])
    if len(keys) > 1 and "ens" not in keys:
        jl["ens"] = np.nanmean(np.stack([jl[k] for k in keys]), 0)
        keys = keys + ["ens"]
    rep = {"alpha": ALPHA, "confidence": CONF, "draws": a.draws, "judges": keys,
           "judge_models": {k: str(J["model_" + k]) for k in keys if "model_" + k in J.files},
           "n_with_judge": int(have.sum()), "judge_accuracy": {}, "per_domain": {}, "summary": {}}

    # ---- judge accuracy per domain (all replies of the domain that have a judge score)
    for d in np.unique(dom):
        idx = np.where((dom == d) & have)[0]
        row = {"format": str(fmt[idx[0]]), "n": int(len(idx))}
        s0 = z(S["reply"][dom == d]) + z(S["worst_unit"][dom == d]) + 2 * z(S["judge"][dom == d])
        row["auroc_shipped_score"] = auroc(s0[have[dom == d]], y[idx])
        for k in keys:
            q = 1 / (1 + np.exp(-jl[k][idx]))
            jc = q >= 0.5
            row[k] = {"auroc": auroc(-jl[k][idx], y[idx]), "agreement": float(np.mean(jc == (y[idx] == 0))),
                      "judge_clean_share": float(jc.mean()),
                      "clean_precision": float(np.mean(y[idx][jc] == 0)) if jc.any() else None,
                      "clean_recall": float(np.mean(jc[y[idx] == 0])),
                      "corr_with_shipped_score": float(np.corrcoef(-jl[k][idx], s0[have[dom == d]])[0, 1]),
                      "auroc_shipped_plus_judge": auroc(s0[have[dom == d]] + z(-jl[k][idx]), y[idx])}
        rep["judge_accuracy"][d] = row

    # ---- calibration simulation
    rng = np.random.default_rng(0)
    for d in np.unique(dom):
        idx = np.where((dom == d) & have)[0]
        full = dom == d
        s_all = z(S["reply"][full]) + z(S["worst_unit"][full]) + 2 * z(S["judge"][full])
        s_dom = s_all[have[full]]
        yd = y[idx]
        neg, wr = np.where(yd == 0)[0], np.where(yd == 1)[0]
        npos = min(len(wr), int(round(len(neg) * 0.25)))
        out = {"format": str(fmt[idx[0]])}
        for N in a.ns:
            if len(neg) + npos - N < 40:
                continue
            res = {}

            def add(name, t, rest_c, rest_w, n_clean=None):
                """n_clean: size of the set a CP threshold was set on (feasible = a guarantee is possible: >= 29)."""
                r = res.setdefault(name, {"fa": [], "rc": [], "feasible": []})
                r["fa"].append(float((rest_c >= t).mean()))
                r["rc"].append(float((rest_w >= t).mean()))
                r["feasible"].append(bool(np.isfinite(t)) and (n_clean is None or n_clean >= MIN_CLEAN))

            for _ in range(a.draws):
                pool = np.concatenate([neg, rng.choice(wr, npos, replace=False)])
                cal = rng.choice(pool, N, replace=False)
                rest = np.setdiff1d(pool, cal)
                rc_, rw_ = s_dom[rest][yd[rest] == 0], s_dom[rest][yd[rest] == 1]
                s, yc = s_dom[cal], yd[cal]
                add("human", cp_or_inf(s[yc == 0]), rc_, rw_, int((yc == 0).sum()))
                for k in keys:
                    lg = jl[k][idx][cal]
                    qc = 1 / (1 + np.exp(-lg))
                    A = qc >= 0.5
                    add(f"{k}:naive", cp_or_inf(s[A]), rc_, rw_, int(A.sum()))
                    for p in (0.9, 0.99):
                        add(f"{k}:conf>={p}", cp_or_inf(s[qc >= p]), rc_, rw_, int((qc >= p).sum()))
                    top = np.argsort(-lg)[: int(round(0.3 * N))]
                    add(f"{k}:conf-top30", cp_or_inf(s[top]), rc_, rw_, len(top))
                    perm = rng.permutation(N)
                    for m in (0, 10, 20, 40):
                        lab = perm[:m]
                        add(f"{k}:ppi-{m}", ppi_threshold(s, qc, lab, yc[lab]), rc_, rw_)
                    for m in (10, 20, 40):
                        iA, iB = rng.permutation(np.where(A)[0]), rng.permutation(np.where(~A)[0])
                        la, lb = iA[: m // 2], iB[: m - min(m // 2, len(iA))]
                        add(f"{k}:bound-{m}", bound_threshold(s, A, (int(yc[la].sum()), len(la)),
                                                              s[lb][yc[lb] == 0], len(lb)), rc_, rw_)
            out[f"N{N}"] = {nm: {"fpr": float(np.mean(r["fa"])), "fpr_over_10pct": float(np.mean(np.array(r["fa"]) > 0.10)),
                                 "fpr_over_12pct": float(np.mean(np.array(r["fa"]) > 0.12)),
                                 "recall": float(np.mean(r["rc"])), "feasible": float(np.mean(r["feasible"]))}
                            for nm, r in res.items()}
        rep["per_domain"][d] = out

    pdm = rep["per_domain"]
    for N in a.ns:
        sk = f"N{N}"
        for g in ["all"] + sorted(set(fmt.tolist())):
            mem = [d for d in pdm if sk in pdm[d] and (g == "all" or pdm[d]["format"] == g)]
            if not mem:
                continue
            names = list(pdm[mem[0]][sk])
            rep["summary"].setdefault(sk, {})[g] = {
                "domains": len(mem),
                **{nm: {m_: float(np.mean([pdm[d][sk][nm][m_] for d in mem])) for m_ in pdm[mem[0]][sk][nm]}
                   for nm in names}}
    acc = rep["judge_accuracy"]
    rep["judge_accuracy_summary"] = {
        g: {k: {m_: float(np.nanmean([acc[d][k][m_] for d in acc if g == "all" or acc[d]["format"] == g
                                      if acc[d][k][m_] is not None]))
                for m_ in ("auroc", "agreement", "clean_precision", "clean_recall", "judge_clean_share",
                           "corr_with_shipped_score", "auroc_shipped_plus_judge")} for k in keys}
        | {"auroc_shipped_score": float(np.nanmean([acc[d]["auroc_shipped_score"] for d in acc
                                                     if g == "all" or acc[d]["format"] == g]))}
        for g in ["all"] + sorted(set(fmt.tolist()))}
    json.dump(rep, open(a.report, "w"), indent=1)
    for g, v in rep["judge_accuracy_summary"].items():
        print("judge", g, {k: ({m_: round(x, 3) for m_, x in vv.items()} if isinstance(vv, dict) else round(vv, 3))
                           for k, vv in v.items()})
    for sk, gs in rep["summary"].items():
        for g, v in gs.items():
            print(f"\n== {sk} {g} ({v['domains']} domains)")
            for nm, r in v.items():
                if nm != "domains":
                    print(f"  {nm:22s} FA {r['fpr']:.3f}  >10% {r['fpr_over_10pct']:.2f}  >12% {r['fpr_over_12pct']:.2f}"
                          f"  recall {r['recall']:.3f}  feasible {r['feasible']:.2f}")


if __name__ == "__main__":
    main()
