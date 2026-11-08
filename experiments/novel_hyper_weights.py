"""Hypernetwork-lite: WRITE a new domain's expert-combination weights from its UNLABELLED statistics (zero labels).

Stored scores only (no model is loaded). Every expert score is already held out leave-one-domain-out.
    results/omni/expert_scores_q8b.npz   reply head, worst unit, the 8B reader's own judge
    results/omni/judge_probs.npz         external judges' logit P(correct): Qwen3-14B, phi-4 (we use -logit = "wrong")
    /home/user/omni/corpus.jsonl         reply text, units, document (for label-free text statistics)

Experts per reply, z-scored on the domain's own replies (unlabelled): reply, worst_unit, judge, qwen14, phi4.

(1) ORACLE weights per domain: balanced L2 logistic regression on ALL of the domain's labels (l2 = 0.05, toward 0),
    rescaled to sum |w| = 10 (the fixed rule's scale: 1+1+2+3+3). Used only as regression TARGETS for other domains,
    and (in-sample, cheating) as a headroom number.
(2) DESCRIPTORS per domain, computed WITHOUT labels (36): per expert mean / std / skew of the RAW score (15), the 10
    pairwise Spearman correlations of the experts, share of replies with a document, format one-hot (4), log mean reply
    length (chars), mean number of units, digit density, share of replies with math markup / equations, log mean
    document length, share of replies that are short (< 200 chars), number of replies (log).
(3) Leave-one-domain-out (LODO) over 21 domains: everything fitted on the other 20 domains, applied to the held-out
    domain, scored by AUROC on ALL its replies (it used none of its labels).

PRE-REGISTERED RULES (written before any result was computed)
    * Primary method: `hyper_ridge` = ridge regression from the standardised descriptors to the oracle weight vector
      (5 outputs), alpha chosen by an inner LODO over the 20 training domains minimising weight MSE (not AUROC), grid
      {0.1, 1, 10, 100, 1000}. Everything else is secondary / exploratory.
    * Primary comparison: mean AUROC over the 21 domains vs the fixed zero-label weights (1,1,2,3,3); paired bootstrap
      over domains (10,000 resamples), 95% percentile CI of the mean difference. "Better" needs the CI to exclude 0.
    * Also reported vs the 50-label refit (omni_stack adapt_50 protocol: 25+25 labels, LogReg(2/k) pulled toward
      0.5*w_fixed, 20 draws, scored on the remaining replies) -- a different evaluation set, so indicative only.
    * Secondary variants: global mean of the other domains' oracle weights (`mean_others`); format-level mean
      (`format_mean`, a-priori rule; falls back to mean_others); kNN-average (k=3) of nearest domains' oracle weights in
      descriptor space (`knn3`); kernel ridge / GP posterior mean, RBF with median-heuristic length scale, noise 1
      (`gp`); unsupervised proxies on the domain itself: `pc1` (weights = loadings of the first principal component
      of the z-scored experts, sign-aligned, negatives clipped to 0), `one_factor` (one-factor model optimal weights
      lambda_j / psi_j, lambda from leave-one-out consensus correlation, clipped to [0, 0.95]); `consensus_corr`
      (weights proportional to each expert's correlation with the mean of the other four, clipped at 0); and
      `shrink_hyper` = average of hyper_ridge weights and the fixed weights (each normalised to sum 10).
    * POST HOC (added after seeing that the inner LODO always picked alpha = 1000, i.e. the mean): `hyper_ridge_a10`,
      ridge forced to alpha = 10, to show what the descriptors do when the regressor is made to use them.
    * Weights from any method are clipped at >= 0 (all experts are oriented as "wrong" scores) before scoring.
    * With 21 domains and ~10 variants, a single variant beating the baseline at p~0.05 is weak evidence; we say so.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/novel_hyper_weights.py \
        --report results/omni/novel_hyper_weights.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.stats import skew, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_eval import LogReg, auroc, balanced_weights  # noqa: E402

NAMES = ["reply", "worst_unit", "judge", "qwen14", "phi4"]
W_FIXED = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
FORMATS = ["chat", "longform", "rag", "reasoning"]
MATH = re.compile(r"(\\frac|\\\(|\\\[|\$[^$]+\$|\\boxed|\d\s*[\+\-\*/=×÷]\s*\d)")


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def norm10(w):
    w = np.clip(np.asarray(w, float), 0, None)
    return W_FIXED.copy() if w.sum() <= 1e-9 else 10 * w / w.sum()


def load(a):
    S = np.load(a.scores, allow_pickle=True)
    J = np.load(a.judge, allow_pickle=True)
    ids = S["ids"].astype(int)
    pj = {int(i): k for k, i in enumerate(J["ids"])}
    o = np.array([pj[int(i)] for i in ids])
    rows = {}
    for line in open(a.corpus):
        r = json.loads(line)
        rows[r["id"]] = r
    raw = np.stack([S["reply"], S["worst_unit"], S["judge"], -J["logit_qwen14"][o], -J["logit_phi4"][o]], 1)
    return dict(ids=ids, dom=S["domain"].astype(str), fmt=S["format"].astype(str), y=S["label"].astype(int),
                raw=raw, rows=[rows[int(i)] for i in ids])


def descriptors(D, d):
    m = np.where(D["dom"] == d)[0]
    R = D["raw"][m]
    f, names = [], []
    for j, nm in enumerate(NAMES):
        f += [R[:, j].mean(), R[:, j].std(), skew(R[:, j])]
        names += [f"{nm}_mean", f"{nm}_std", f"{nm}_skew"]
    C = spearmanr(R).correlation
    for i in range(5):
        for j in range(i + 1, 5):
            f.append(C[i, j])
            names.append(f"corr_{NAMES[i]}_{NAMES[j]}")
    rr = [D["rows"][k] for k in m]
    resp = [r["response"] for r in rr]
    docs = [r.get("doc") or "" for r in rr]
    fmt = D["fmt"][m[0]]
    f += [np.mean([len(x.strip()) > 0 for x in docs])] + [float(fmt == g) for g in FORMATS]
    names += ["has_doc"] + [f"fmt_{g}" for g in FORMATS]
    f += [np.log(np.mean([len(x) for x in resp]) + 1), np.mean([len(r.get("units") or []) for r in rr]),
          np.mean([sum(c.isdigit() for c in x) / max(len(x), 1) for x in resp]),
          np.mean([bool(MATH.search(x)) for x in resp]),
          np.log(np.mean([len(x) for x in docs]) + 1), np.mean([len(x) < 200 for x in resp]), np.log(len(m))]
    names += ["log_len", "n_units", "digit_density", "math_share", "log_doc_len", "short_share", "log_n"]
    return np.array(f, float), names


def domain_Z(D, d):
    m = D["dom"] == d
    return np.stack([z(D["raw"][m][:, j]) for j in range(5)], 1), D["y"][m]


def oracle(Z, y, l2):
    return LogReg(l2).fit(Z, y.astype(float), balanced_weights(y.astype(float))).w


# ---------------------------------------------------------------- regressors (fit on <= 20 domains)

def ridge_fit_predict(X, Y, x, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs, xs = (X - mu) / sd, (x - mu) / sd
    ym = Y.mean(0)
    B = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(X.shape[1]), Xs.T @ (Y - ym))
    return ym + xs @ B


def ridge_inner_alpha(X, Y, grid):
    best, ba = np.inf, grid[0]
    for al in grid:
        err = np.mean([((ridge_fit_predict(np.delete(X, i, 0), np.delete(Y, i, 0), X[i], al) - Y[i]) ** 2).sum()
                       for i in range(len(X))])
        if err < best:
            best, ba = err, al
    return ba


def knn_predict(X, Y, x, k=3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    dist = (((X - mu) / sd - (x - mu) / sd) ** 2).sum(1)
    return Y[np.argsort(dist)[:k]].mean(0)


def gp_predict(X, Y, x, noise=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs, xs = (X - mu) / sd, (x - mu) / sd
    D2 = ((Xs[:, None] - Xs[None]) ** 2).sum(-1)
    ell2 = np.median(D2[np.triu_indices(len(X), 1)])
    K = np.exp(-D2 / (2 * ell2))
    k = np.exp(-((Xs - xs) ** 2).sum(1) / (2 * ell2))
    ym = Y.mean(0)
    return ym + k @ np.linalg.solve(K + noise * np.eye(len(X)), Y - ym)


# ---------------------------------------------------------------- unsupervised proxies (the domain alone)

def loo_consensus_corr(Z):
    return np.array([np.corrcoef(Z[:, j], np.delete(Z, j, 1).mean(1))[0, 1] for j in range(Z.shape[1])])


def pc1(Z):
    C = np.corrcoef(Z.T)
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, -1]
    return v * np.sign(v.sum())


def one_factor(Z):
    lam = np.clip(loo_consensus_corr(Z), 0, 0.95)
    return lam / (1 - lam ** 2)


# ---------------------------------------------------------------- 50-label reference (omni_stack adapt_50)

def adapt_k(Z, y, k, reps, rng):
    p_, n_ = np.where(y == 1)[0], np.where(y == 0)[0]
    if k // 2 >= len(p_) - 10 or k - k // 2 >= len(n_) - 10:
        return None
    res = []
    for _ in range(reps):
        cal = np.concatenate([rng.choice(p_, k // 2, replace=False), rng.choice(n_, k - k // 2, replace=False)])
        rest = np.setdiff1d(np.arange(len(y)), cal)
        m = LogReg(2.0 / k, w0=W_FIXED * 0.5).fit(Z[cal], y[cal].astype(float), balanced_weights(y[cal].astype(float)))
        res.append(auroc(m.score(Z[rest]), y[rest]))
    return float(np.mean(res))


def boot_ci(diff, rng, B=10000):
    diff = np.asarray(diff, float)
    bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
    return [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
            float((bs <= 0).mean())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/omni/expert_scores_q8b.npz")
    ap.add_argument("--judge", default="results/omni/judge_probs.npz")
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--report", default="results/omni/novel_hyper_weights.json")
    ap.add_argument("--oracle-l2", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    D = load(a)
    doms = sorted(set(D["dom"].tolist()))
    fmt = {d: str(D["fmt"][D["dom"] == d][0]) for d in doms}
    Zs, ys, X, W = {}, {}, [], []
    for d in doms:
        Zs[d], ys[d] = domain_Z(D, d)
        x, dnames = descriptors(D, d)
        X.append(x)
        W.append(norm10(oracle(Zs[d], ys[d], a.oracle_l2)))      # oracle targets (clipped >= 0, sum 10)
    X, W = np.array(X), np.array(W)
    X = np.nan_to_num(X)
    grid = [0.1, 1.0, 10.0, 100.0, 1000.0]

    per, wts, alphas = {}, {}, {}
    for i, d in enumerate(doms):
        tr = [j for j in range(len(doms)) if j != i]
        Xt, Wt = X[tr], W[tr]
        al = ridge_inner_alpha(Xt, Wt, grid)
        alphas[d] = al
        same = [j for j in tr if fmt[doms[j]] == fmt[d]]
        hyp = norm10(ridge_fit_predict(Xt, Wt, X[i], al))
        cand = {
            "fixed": W_FIXED,
            "equal": np.ones(5),
            "mean_others": Wt.mean(0),
            "format_mean": W[same].mean(0) if same else Wt.mean(0),
            "hyper_ridge": hyp,
            "shrink_hyper": (hyp + norm10(W_FIXED)) / 2,
            "hyper_ridge_a10(posthoc)": ridge_fit_predict(Xt, Wt, X[i], 10.0),
            "knn3": knn_predict(Xt, Wt, X[i], 3),
            "gp": gp_predict(Xt, Wt, X[i]),
            "pc1": pc1(Zs[d]),
            "one_factor": one_factor(Zs[d]),
            "consensus_corr": loo_consensus_corr(Zs[d]),
            "oracle_insample(cheat)": W[i],
        }
        per[d] = {"format": fmt[d], "n": int(len(ys[d])), "ridge_alpha": al}
        wts[d] = {}
        for k, w in cand.items():
            w = norm10(w)
            wts[d][k] = [round(float(v), 3) for v in w]
            per[d][k] = auroc(Zs[d] @ w, ys[d])
        per[d]["adapt_50"] = adapt_k(Zs[d], ys[d], 50, a.reps, rng)

    methods = [k for k in per[doms[0]] if k not in ("format", "n", "ridge_alpha")]
    summ = {}
    for g in ["all"] + FORMATS:
        mem = [d for d in doms if g == "all" or fmt[d] == g]
        summ[g] = {"domains": len(mem)}
        for k in methods:
            v = [per[d][k] for d in mem if per[d][k] is not None]
            summ[g][k] = float(np.mean(v)) if v else None
    brng = np.random.default_rng(1)
    paired = {}
    for k in methods:
        if k in ("fixed",):
            continue
        dv = [per[d][k] - per[d]["fixed"] for d in doms if per[d][k] is not None]
        paired[f"{k} - fixed"] = boot_ci(dv, brng)
    a50 = [d for d in doms if per[d]["adapt_50"] is not None]
    for k in ["fixed", "mean_others", "format_mean", "hyper_ridge", "shrink_hyper", "knn3", "gp", "one_factor"]:
        paired[f"{k} - adapt_50 ({len(a50)} domains)"] = boot_ci([per[d][k] - per[d]["adapt_50"] for d in a50], brng)
    # how well does the regressor predict the weights at all? LODO weight MSE vs predicting the mean
    mse = {k: float(np.mean([((np.array(wts[d][k]) - W[i]) ** 2).sum() for i, d in enumerate(doms)]))
           for k in ["fixed", "mean_others", "format_mean", "hyper_ridge", "hyper_ridge_a10(posthoc)", "knn3", "gp",
                     "one_factor"]}
    rep = {"experts": NAMES, "w_fixed": W_FIXED.tolist(), "oracle_l2": a.oracle_l2, "descriptors": dnames,
           "oracle_weights": {d: [round(float(v), 3) for v in W[i]] for i, d in enumerate(doms)},
           "summary": summ, "paired_bootstrap_mean_ci_pleq0": paired, "lodo_weight_sq_error": mse,
           "per_domain": per, "weights_used": wts,
           "adapt_50_domains": a50}
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)
    print(f"{'method':24s}" + "".join(f"{g:>10s}" for g in summ))
    for k in methods:
        print(f"{k:24s}" + "".join(f"{(summ[g][k] if summ[g][k] is not None else float('nan')):10.3f}" for g in summ))
    for k, v in paired.items():
        print(f"{k:40s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] P(<=0)={v[3]:.3f}")
    print("weight sq err", {k: round(v, 2) for k, v in mse.items()})
    print("alphas", alphas)


if __name__ == "__main__":
    main()
