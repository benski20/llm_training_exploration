"""Zero-label expert weighting from AGREEMENT: unsupervised ensemble learning per domain (stored scores only).

Idea (crowdsourcing / unsupervised ensemble learning): on a new domain, how strongly the experts co-vary on the
domain's UNLABELLED replies tells how accurate each one is, if their errors are (conditionally) independent.
Estimate each expert's accuracy from agreement alone, weight the experts accordingly, never read a label.

Experts per reply, z-scored within the domain (all held out leave-one-domain-out upstream):
    reply, worst_unit, judge (the Qwen3-8B reader's three)   qwen14, phi4 (external judges, -logit P(correct))
NLI is left out: it covers only ~half the replies of 9 domains (the 0.842 baseline does not use it either).

DEPENDENCE. The reader's three experts read the same hidden states, so they are NOT conditionally independent:
their within-class errors correlate and a plain agreement model would treat them as three confirming witnesses.
Handling (fixed before the run): three conditionally independent GROUPS, R = {reply, worst_unit, judge}, Q14, PHI.
Only cross-group agreement is used to identify accuracies; the within-reader block is modelled as free dependence.

METHODS (each returns a 5-vector of weights on the z-scores, a prevalence estimate, and per-expert accuracy estimates)
  sml_g   continuous Spectral Meta-Learner, grouped (one-factor / rank-one model, Parisi et al. 2014 in its second-
          moment form for scores, the triad idea of Jaffe et al. 2015). For z-scores of a binary latent class with
          conditionally independent groups, Cov(z_i, z_j) = a_i a_j for i, j in different groups, with
          a_i = sqrt(pi(1-pi)) * Delta_i (Delta_i = class mean gap). a is fitted by least squares to the 7 CROSS-group
          covariances only (the within-reader block is masked). The within-class covariance is then
          Sigma_w = C - a a^T on the reader block and on the diagonal, 0 across groups (the assumption);
          weights = Sigma_w^-1 a (LDA: this de-duplicates the correlated reader experts).
          Prevalence from third moments (Jaffe et al. 2015, continuous form): for i, j, k in different groups
          E[z_i z_j z_k] / (a_i a_j a_k) = (1 - 2 pi) / sqrt(pi (1 - pi)), solved for pi (needs within-class symmetry).
          Expert AUROC estimate: binormal, Phi(Delta_i / sqrt(2 (1 - a_i^2))).
  ds_g    Dawid-Skene EM (two-coin) on votes binarised at the domain median; 3 voters: the reader group's mean z
          (one vote), qwen14, phi4. Weight per voter = log(sens*spec / ((1-sens)(1-spec))); the reader weight is split
          equally over its three z-scores (no hand prior). Prevalence = the class prior. Expert balanced accuracy of
          the median vote = (sens + spec)/2; for reader experts DS is re-run on the triad (expert, qwen14, phi4).
  gmm     latent-class model on the joint 5-d z-scores: 2-component Gaussian mixture with a TIED full covariance
          (sklearn, 8 inits). No independence assumption at all (identification comes from the bimodal shape).
          The "wrong" component is the one with the larger mean of summed z. Weights = Sigma^-1 (mu1 - mu0)
          (the posterior log-odds is linear); prevalence = its mixing weight; AUROC_i = Phi(dmu_i / sqrt(2 S_ii)).
  combo   PRIMARY: the mean of the three weight vectors, each scaled to unit L1 norm; prevalence = median of three.
  Naive controls (dependence ignored, to show what the grouping buys): sml_naive (leading eigenvector of the
  off-diagonal covariance of all 5, diagonal imputed iteratively, weights a_i/(1-a_i^2)), ds_naive (5 median votes).
  hybrid  (secondary) the mean of the unit-L1 hand weights (1,1,2,3,3) and the unit-L1 combo weights.

PRE-STATED RULES (written before the first run)
  R1  Primary comparison: combo vs the fixed hand weights (1,1,2,3,3) z-sum, mean AUROC over the 21 domains on the
      natural pools. "Better" only if the 95% CI of the paired difference is entirely > 0; "worse" if entirely < 0;
      otherwise "no detectable difference". CI: hierarchical paired bootstrap, B = 300: domains resampled with
      replacement (within format for the per-format rows) and replies resampled within each domain, the
      unsupervised weights RE-ESTIMATED on every resample (estimation noise included).
  R2  All other methods/variants are secondary (no multiplicity correction; read them as descriptive).
  R3  Secondary pool: 20% prevalence (all clean replies + a random 20% share of wrong ones; 10 draws) - real traffic
      is mostly clean and a median split / mixture behaves differently there.
  R4  Calibration: estimated vs true per-expert accuracy over (domain, expert) pairs: MAE, Pearson r, and the mean
      within-domain Spearman of the expert ranking. True = AUROC (sml_g, gmm) or the median-vote balanced accuracy (ds_g).
  R5  Prevalence: pools at prevalence 10/20/30/50% (or the natural rate if lower) by subsampling, 5 draws; MAE and
      bias of each estimate, against two zero-label classify-and-count baselines from the judges' raw outputs
      (share with mean judge P(wrong) > 0.5, and mean judge P(wrong)). "Useful for reporting" only if the MAE of
      the combo estimate is <= 0.05 and below both judge baselines.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/novel_unsup_ensemble.py \
        --report results/omni/novel_unsup_ensemble.json
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_eval import auroc  # noqa: E402

warnings.filterwarnings("ignore")
NAMES = ["reply", "worst_unit", "judge", "qwen14", "phi4"]
GROUP = np.array([0, 0, 0, 1, 2])
HAND = np.array([1, 1, 2, 3, 3.])
R_IDX = [0, 1, 2]
CROSS = [(i, j) for i in range(5) for j in range(i + 1, 5) if GROUP[i] != GROUP[j]]
TRIPLES = [(i, 3, 4) for i in R_IDX]


def zc(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def load(a):
    S = np.load(a.scores, allow_pickle=True)
    J = np.load(a.judge, allow_pickle=True)
    ids = S["ids"].astype(int)
    pj = {int(i): k for k, i in enumerate(J["ids"])}
    o = np.array([pj[int(i)] for i in ids])
    raw = np.stack([S["reply"].astype(float), S["worst_unit"].astype(float), S["judge"].astype(float),
                    -J["logit_qwen14"][o].astype(float), -J["logit_phi4"][o].astype(float)], 1)
    return dict(dom=S["domain"].astype(str), fmt=S["format"].astype(str), y=S["label"].astype(int), raw=raw)


def zmat(raw):
    return np.stack([zc(raw[:, k]) for k in range(raw.shape[1])], 1)


def l1(w):
    return w / (np.abs(w).sum() + 1e-12)


def pi_from_ratio(r):
    s2 = r * r
    disc = max(0.0, 1 - 4 / (4 + s2))
    p = (1 - np.sqrt(disc)) / 2 if r > 0 else (1 + np.sqrt(disc)) / 2
    return float(np.clip(p, 0.01, 0.99))


def binormal_auc(delta, var_w):
    return float(norm.cdf(delta / np.sqrt(2 * max(var_w, 1e-3))))


# ------------------------------------------------------------------ methods

def sml_grouped(Z):
    C = np.cov(Z.T, bias=True)
    # closed-form init from the triad (reader mean, q14, phi), then least squares on the 7 cross-group covariances
    a0 = np.full(5, 0.5)
    c34 = C[3, 4]
    rbar = Z[:, R_IDX].mean(1)
    c_r3, c_r4 = np.mean(rbar * Z[:, 3]), np.mean(rbar * Z[:, 4])
    if c34 * c_r3 * c_r4 > 0:
        a0[3] = np.sqrt(abs(c34 * c_r3 / c_r4)); a0[4] = np.sqrt(abs(c34 * c_r4 / c_r3))
        for i in R_IDX:
            a0[i] = 0.5 * (C[i, 3] / a0[3] + C[i, 4] / a0[4])
    a0 = np.clip(a0, -0.95, 0.95)
    res = least_squares(lambda a: np.array([C[i, j] - a[i] * a[j] for i, j in CROSS]), a0,
                        bounds=(-0.99, 0.99))
    a = res.x
    if a.sum() < 0:                       # orientation: experts on average point at "wrong"
        a = -a
    Sw = np.zeros((5, 5))
    Sw[np.ix_(R_IDX, R_IDX)] = C[np.ix_(R_IDX, R_IDX)] - np.outer(a[R_IDX], a[R_IDX])
    for k in (3, 4):
        Sw[k, k] = C[k, k] - a[k] ** 2
    ev, U = np.linalg.eigh(Sw)
    Sw = U @ np.diag(np.maximum(ev, 0.05)) @ U.T
    w = np.linalg.solve(Sw, a)
    ratios = [np.mean(Z[:, i] * Z[:, j] * Z[:, k]) / (a[i] * a[j] * a[k])
              for i, j, k in TRIPLES if abs(a[i] * a[j] * a[k]) > 1e-3]
    pi = pi_from_ratio(float(np.median(ratios))) if ratios else 0.5
    d = a / np.sqrt(pi * (1 - pi))
    acc = np.array([binormal_auc(d[k], Sw[k, k]) for k in range(5)])
    return w, pi, acc


def sml_naive(Z):
    C = np.cov(Z.T, bias=True)
    M = C.copy()
    for _ in range(50):                   # impute the diagonal with the rank-one fit (Parisi-style)
        ev, U = np.linalg.eigh(M)
        a = U[:, -1] * np.sqrt(max(ev[-1], 1e-9))
        np.fill_diagonal(M, a ** 2)
    if a.sum() < 0:
        a = -a
    a = np.clip(a, -0.95, 0.95)
    return a / (1 - a ** 2)


def dawid_skene(V, iters=200):
    """V: n x m binary votes (1 = 'wrong'). Two-coin DS. Returns sens, spec, prior, posterior."""
    post = V.mean(1)
    post = np.clip(post, 0.01, 0.99)
    for _ in range(iters):
        pi = np.clip(post.mean(), 0.01, 0.99)
        sens = np.clip((post[:, None] * V).sum(0) / post.sum(), 0.01, 0.99)
        spec = np.clip(((1 - post)[:, None] * (1 - V)).sum(0) / (1 - post).sum(), 0.01, 0.99)
        l1_ = np.log(pi) + (V * np.log(sens) + (1 - V) * np.log(1 - sens)).sum(1)
        l0_ = np.log(1 - pi) + (V * np.log(1 - spec) + (1 - V) * np.log(spec)).sum(1)
        new = 1 / (1 + np.exp(l0_ - l1_))
        if np.max(np.abs(new - post)) < 1e-7:
            post = new
            break
        post = new
    return sens, spec, float(post.mean()), post


def votes(Z):
    return (Z > np.median(Z, 0)).astype(float)


def ds_grouped(Z):
    rbar = Z[:, R_IDX].mean(1)
    V = votes(np.stack([rbar, Z[:, 3], Z[:, 4]], 1))
    sens, spec, pi, _ = dawid_skene(V)
    lw = np.log(sens * spec / ((1 - sens) * (1 - spec)))
    w = np.array([lw[0] / 3] * 3 + [lw[1], lw[2]])
    acc = np.zeros(5)
    acc[3:] = (sens[1:] + spec[1:]) / 2
    for i in R_IDX:
        s2, p2, _, _ = dawid_skene(votes(Z[:, [i, 3, 4]]))
        acc[i] = (s2[0] + p2[0]) / 2
    return w, pi, acc


def ds_naive(Z):
    sens, spec, _, _ = dawid_skene(votes(Z))
    return np.log(sens * spec / ((1 - sens) * (1 - spec)))


def gmm_tied(Z, seed=0):
    g = GaussianMixture(2, covariance_type="tied", n_init=8, random_state=seed, reg_covar=1e-3).fit(Z)
    hi = int(np.argmax(g.means_.sum(1)))
    dmu = g.means_[hi] - g.means_[1 - hi]
    S = g.covariances_
    w = np.linalg.solve(S, dmu)
    acc = np.array([binormal_auc(dmu[k], S[k, k]) for k in range(5)])
    return w, float(g.weights_[hi]), acc


def all_methods(Z, seed=0):
    out = {}
    ws, ps, accs = sml_grouped(Z)
    wd, pd_, accd = ds_grouped(Z)
    wg, pg, accg = gmm_tied(Z, seed)
    out["sml_g"] = (ws, ps, accs)
    out["ds_g"] = (wd, pd_, accd)
    out["gmm"] = (wg, pg, accg)
    wc = (l1(ws) + l1(wd) + l1(wg)) / 3
    out["combo"] = (wc, float(np.median([ps, pd_, pg])), None)
    out["hybrid"] = ((l1(HAND) + l1(wc)) / 2, None, None)
    out["sml_naive"] = (sml_naive(Z), None, None)
    out["ds_naive"] = (ds_naive(Z), None, None)
    out["hand"] = (HAND, None, None)
    out["equal"] = (np.ones(5), None, None)
    return out


METHODS = ["hand", "equal", "combo", "sml_g", "ds_g", "gmm", "hybrid", "sml_naive", "ds_naive"]


def eval_pool(raw, y, seed=0):
    Z = zmat(raw)
    M = all_methods(Z, seed)
    return {k: auroc(Z @ M[k][0], y) for k in METHODS}, M, Z


def judge_cc(raw):
    p = (1 / (1 + np.exp(-raw[:, 3])) + 1 / (1 + np.exp(-raw[:, 4]))) / 2   # raw[:,3:] = -logit P(correct)
    return float((p > 0.5).mean()), float(p.mean())


def subsample(y, prev, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = int(round(prev * len(neg) / (1 - prev)))
    if n_pos > len(pos):
        n_neg = int(round(len(pos) * (1 - prev) / prev))
        return np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    return np.concatenate([rng.choice(pos, n_pos, replace=False), neg])


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/omni/expert_scores_q8b.npz")
    ap.add_argument("--judge", default="results/omni/judge_probs.npz")
    ap.add_argument("--report", default="results/omni/novel_unsup_ensemble.json")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    D = load(a)
    doms = sorted(set(D["dom"]))
    fmt = {d: str(D["fmt"][D["dom"] == d][0]) for d in doms}
    fmts = sorted(set(fmt.values()))
    rep = {"rules": __doc__.split("PRE-STATED RULES")[1].split("OMP_NUM")[0].strip(), "experts": NAMES}

    # ---------- natural pools: AUROC, weights, calibration, true dependence diagnostics
    per = {}
    calib = {m: {"est": [], "true": [], "rho": []} for m in ["sml_g", "ds_g", "gmm"]}
    for d in doms:
        m = D["dom"] == d
        raw, y = D["raw"][m], D["y"][m]
        A, M, Z = eval_pool(raw, y)
        true_auc = np.array([auroc(Z[:, k], y) for k in range(5)])
        V = votes(Z)
        true_ba = np.array([(V[y == 1, k].mean() + (1 - V[y == 0, k]).mean()) / 2 for k in range(5)])
        for meth, tru in (("sml_g", true_auc), ("ds_g", true_ba), ("gmm", true_auc)):
            est = M[meth][2]
            calib[meth]["est"] += est.tolist(); calib[meth]["true"] += tru.tolist()
            calib[meth]["rho"].append(float(spearmanr(est, tru)[0]))
        # true within-class correlations (diagnostic of the independence assumption; labels used ONLY here)
        wc = {}
        for cls in (0, 1):
            R = np.corrcoef(Z[y == cls].T)
            wc[cls] = {"reader_block_mean": float(np.mean([R[i, j] for i in R_IDX for j in R_IDX if i < j])),
                       "q14_phi4": float(R[3, 4]),
                       "reader_vs_judges_mean": float(np.mean([R[i, j] for i in R_IDX for j in (3, 4)]))}
        per[d] = {"format": fmt[d], "n": int(m.sum()), "prevalence": float(y.mean()),
                  "auroc": {k: round(v, 4) for k, v in A.items()},
                  "weights_l1": {k: [round(x, 3) for x in l1(M[k][0])] for k in ["combo", "sml_g", "ds_g", "gmm"]},
                  "prev_est": {k: round(M[k][1], 3) for k in ["sml_g", "ds_g", "gmm", "combo"]},
                  "true_auroc": [round(x, 3) for x in true_auc], "true_median_ba": [round(x, 3) for x in true_ba],
                  "est_acc": {k: [round(x, 3) for x in M[k][2]] for k in ["sml_g", "ds_g", "gmm"]},
                  "within_class_corr": {"clean": wc[0], "wrong": wc[1]}}
        print(d, {k: round(v, 3) for k, v in A.items()}, flush=True)

    def mean_auc(k, ds=doms):
        return float(np.mean([per[d]["auroc"][k] for d in ds]))
    rep["natural"] = {"mean_auroc": {k: round(mean_auc(k), 4) for k in METHODS},
                      "by_format": {f: {k: round(mean_auc(k, [d for d in doms if fmt[d] == f]), 4) for k in METHODS}
                                    for f in fmts},
                      "wins_vs_hand": {k: int(sum(per[d]["auroc"][k] > per[d]["auroc"]["hand"] for d in doms))
                                       for k in METHODS if k != "hand"},
                      "reference_50_labels": 0.859}
    cal = {}
    for k, c in calib.items():
        e, t = np.array(c["est"]), np.array(c["true"])
        cal[k] = {"target": "median-vote balanced accuracy" if k == "ds_g" else "AUROC",
                  "mae": round(float(np.mean(np.abs(e - t))), 4), "bias": round(float(np.mean(e - t)), 4),
                  "pearson_r": round(float(pearsonr(e, t)[0]), 3),
                  "mean_within_domain_spearman": round(float(np.nanmean(c["rho"])), 3),
                  "by_expert_mae": {NAMES[j]: round(float(np.mean(np.abs(e[j::5] - t[j::5]))), 4) for j in range(5)},
                  "by_expert_bias": {NAMES[j]: round(float(np.mean(e[j::5] - t[j::5])), 4) for j in range(5)}}
    rep["calibration"] = cal
    rep["dependence_diagnostic_mean"] = {
        cls: {kk: round(float(np.mean([per[d]["within_class_corr"][cls][kk] for d in doms])), 3)
              for kk in ["reader_block_mean", "q14_phi4", "reader_vs_judges_mean"]} for cls in ("clean", "wrong")}
    print("natural", rep["natural"]["mean_auroc"], flush=True)
    print("calibration", json.dumps(cal), flush=True)

    # ---------- hierarchical paired bootstrap with re-estimation
    rng = np.random.default_rng(0)
    comp = ["combo", "sml_g", "ds_g", "gmm", "hybrid", "equal", "sml_naive", "ds_naive"]
    diffs = {k: np.zeros((a.boot, len(doms))) for k in comp}
    for b in range(a.boot):
        for j, d in enumerate(doms):
            m = np.where(D["dom"] == d)[0]
            y = D["y"][m]
            idx = np.concatenate([rng.choice(np.where(y == c)[0], (y == c).sum()) for c in (0, 1)])
            A, _, _ = eval_pool(D["raw"][m][idx], y[idx], seed=b)
            for k in comp:
                diffs[k][b, j] = A[k] - A["hand"]
        if b % 25 == 0:
            print("boot", b, round(time.time() - t0), "s", flush=True)
    boot = {}
    for k in comp:
        row = {}
        for f in ["all"] + fmts:
            cols = np.arange(len(doms)) if f == "all" else np.array([j for j, d in enumerate(doms) if fmt[d] == f])
            point = float(np.mean([per[doms[j]]["auroc"][k] - per[doms[j]]["auroc"]["hand"] for j in cols]))
            stats = [diffs[k][b, rng.choice(cols, len(cols))].mean() for b in range(a.boot)]
            lo, hi = np.percentile(stats, [2.5, 97.5])
            row[f] = {"diff": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                      "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "no detectable difference")}
        boot[k] = row
    rep["paired_bootstrap_vs_hand"] = boot
    print("boot", json.dumps(boot["combo"]), flush=True)

    # ---------- 20% prevalence pools (R3)
    rng = np.random.default_rng(1)
    p20 = {k: [] for k in METHODS}
    for d in doms:
        m = np.where(D["dom"] == d)[0]
        y = D["y"][m]
        acc = {k: [] for k in METHODS}
        for r in range(10):
            idx = subsample(y, 0.2, rng)
            A, _, _ = eval_pool(D["raw"][m][idx], y[idx], seed=r)
            for k in METHODS:
                acc[k].append(A[k])
        for k in METHODS:
            p20[k].append(float(np.mean(acc[k])))
    rep["prev20"] = {"mean_auroc": {k: round(float(np.mean(v)), 4) for k, v in p20.items()},
                     "diff_vs_hand": {k: {"mean": round(float(np.mean(np.array(v) - np.array(p20["hand"]))), 4),
                                          "se_domains": round(float(np.std(np.array(v) - np.array(p20["hand"]), ddof=1)
                                                                    / np.sqrt(len(doms))), 4),
                                          "wins": int(np.sum(np.array(v) > np.array(p20["hand"])))}
                                      for k, v in p20.items() if k != "hand"},
                     "by_format": {f: {k: round(float(np.mean([p20[k][j] for j, d in enumerate(doms) if fmt[d] == f])), 4)
                                       for k in METHODS} for f in fmts}}
    print("prev20", rep["prev20"]["mean_auroc"], flush=True)

    # ---------- prevalence estimation (R5)
    rng = np.random.default_rng(2)
    rows = []
    for d in doms:
        m = np.where(D["dom"] == d)[0]
        y = D["y"][m]
        for target in (0.1, 0.2, 0.3, 0.5):
            if target > y.mean() + 1e-9 and target != 0.1:
                if target != 0.5:
                    continue
                target = float(y.mean())          # natural rate as the top level
            for r in range(5):
                idx = subsample(y, target, rng) if target < y.mean() - 1e-9 else np.arange(len(y))
                raw, yy = D["raw"][m][idx], y[idx]
                M = all_methods(zmat(raw), seed=r)
                cc, pa = judge_cc(raw)
                rows.append({"domain": d, "true": float(yy.mean()), "sml_g": M["sml_g"][1], "ds_g": M["ds_g"][1],
                             "gmm": M["gmm"][1], "combo": M["combo"][1], "judge_cc": cc, "judge_prob": pa})
                if target >= y.mean() - 1e-9:
                    break
    t = np.array([r["true"] for r in rows])
    prev = {}
    for k in ["sml_g", "ds_g", "gmm", "combo", "judge_cc", "judge_prob"]:
        e = np.array([r[k] for r in rows])
        bands = {}
        for lo, hi in ((0, 0.15), (0.15, 0.25), (0.25, 0.4), (0.4, 0.6)):
            s = (t >= lo) & (t < hi)
            bands[f"{lo}-{hi}"] = {"n": int(s.sum()), "mean_true": round(float(t[s].mean()), 3),
                                   "mean_est": round(float(e[s].mean()), 3), "mae": round(float(np.abs(e - t)[s].mean()), 3)}
        prev[k] = {"mae": round(float(np.mean(np.abs(e - t))), 4), "bias": round(float(np.mean(e - t)), 4),
                   "pearson_r": round(float(pearsonr(e, t)[0]), 3), "by_true_band": bands}
    rep["prevalence"] = {"n_pools": len(rows), "estimators": prev}
    best_judge = min(prev["judge_cc"]["mae"], prev["judge_prob"]["mae"])
    rep["prevalence"]["useful_rule_R5"] = bool(prev["combo"]["mae"] <= 0.05 and prev["combo"]["mae"] < best_judge)
    print("prevalence", json.dumps({k: (v["mae"], v["bias"], v["pearson_r"]) for k, v in prev.items()}), flush=True)

    rep["per_domain"] = per
    rep["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)
    print("wrote", a.report)


if __name__ == "__main__":
    main()
