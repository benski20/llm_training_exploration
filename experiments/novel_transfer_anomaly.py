"""Novel zero-label domain adaptation on stored Qwen3-1.7B reader features (no model is loaded).

Three questions, each answered leave-one-domain-out (LODO) over the 21 corpus domains, AUROC within each domain:

  (1) similarity-weighted transfer: weight the training domains (or training replies) by their similarity to the
      held-out domain's UNLABELLED replies when fitting the head; does it beat uniform training?
  (2) label-free anomaly scores: is a wrong reply atypical for its own domain? Mahalanobis / kNN / isolation
      forest fitted on the domain's unlabelled replies, and "directional" anomaly; alone and z-summed with the head.
  (3) label-free re-estimation of the head direction on the new domain (1-D GMM EM, a shared-covariance GMM in a
      16-dim subspace, Sinkhorn / optimal-transport pseudo-labels), each followed by prior-pulled self-training;
      compared with judge pseudo-label self-training (results/omni/judge_distillation_q17.json).

Features (the 0.712 baseline of results/omni/q17.json, "mean+judge_mid3"): reply mean state + judgement state at
layers 15/20/24, globally standardised, PCA 256; then per-domain standardisation from each domain's own (unlabelled)
replies; L2 logistic head (l2 = 0.05), class-balanced weights.

PRE-STATED RULES (written before any result was seen):
  R1 primary metric: mean AUROC over the 21 held-out domains; per-format means are reported too.
  R2 a method "beats" uniform training only if the paired, domain-stratified bootstrap 95% CI (1,000 resamples of
     replies within each domain) of (method - baseline) mean AUROC lies above 0 AND it improves >= 12 of 21 domains.
  R3 hyperparameters are fixed in advance (constants below). Grids (k neighbours, prior strength) are reported in full,
     but only the designated PRIMARY variant of each family counts for R2; the best-of-grid chosen with the target's
     labels is reported as an "oracle" upper bound and is not a valid result.
  R4 anomaly scores are used with their natural sign (larger = more atypical = more suspect), never flipped.
  R5 the label-free rule for the number of neighbour domains: nested LODO inside the training domains (each training
     domain in turn is a pseudo-target, its k nearest *other* training domains by MMD train the head; pick the k with
     the best mean pseudo-target AUROC). Uses only training-domain labels.
  PRIMARY variants: (1) knn_mmd_auto, mmd_soft, kmm_mix, iw_tempered; (2) head + maha_probe; (3) gmm1d_self (l2p=0.5).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/novel_transfer_anomaly.py \
        --feats /home/user/omni/q17 --corpus /home/user/omni/corpus.jsonl --out results/omni/novel_transfer_anomaly.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_corpus import load_features  # noqa: E402
from omni_eval import LogReg, auroc, pca_fit, standardise  # noqa: E402

L2 = 0.05
PCA_K = 256
MID = [2, 3, 4]                         # layers 15, 20, 24 of the stored (7, 11, 15, 20, 24, 28)
K_GRID = [1, 2, 4, 8, 12, 16]           # neighbour domains ("all" = uniform)
MMD_SUB = 150                           # replies per domain for kernel means
IW_C = 0.01                             # domain-classifier L2 (sklearn C)
IW_CLIP_Q = 0.99
KMM_MIX = 0.5
L2P_GRID = [0.05, 0.5, 5.0]             # self-training prior strength; 0.5 primary
SUB_DIM = 16                            # GMM subspace: head direction + 15 domain PCs
OT_EPS = 0.05
B = 1000


def wbal(y, w):
    """class-balanced weights times w (each class gets half the total weight)."""
    out = np.empty(len(y))
    for c in (0, 1):
        m = y == c
        out[m] = w[m] * 0.5 / max(w[m].sum(), 1e-12)
    return out * len(y)


def fit(X, y, w=None, l2=L2, w0=None, b0=0.0):
    w = np.ones(len(y)) if w is None else w
    return LogReg(l2, w0=w0, b0=b0).fit(X, y.astype(float), wbal(y, w))


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-9)


# ----------------------------------------------------------------- domain similarity (unlabelled)

def kernel_means(Z, dom, doms, rng):
    sub = {d: rng.choice(np.where(dom == d)[0], min(MMD_SUB, (dom == d).sum()), replace=False) for d in doms}
    allx = np.concatenate([Z[sub[d]] for d in doms])
    sq = (allx ** 2).sum(1)
    D2 = sq[:, None] + sq[None] - 2 * allx @ allx.T
    gamma = 1.0 / np.median(D2[D2 > 0])
    Kall = np.exp(-gamma * np.maximum(D2, 0))
    off = np.cumsum([0] + [len(sub[d]) for d in doms])
    nd = len(doms)
    Kb = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):
            Kb[i, j] = Kall[off[i]:off[i + 1], off[j]:off[j + 1]].mean()
    mmd2 = np.diag(Kb)[:, None] + np.diag(Kb)[None] - 2 * Kb
    return Kb, np.maximum(mmd2, 0)


def kmm_weights(Kb, tr, t):
    """simplex weights over training domains minimising MMD(sum a_j P_j, P_t)."""
    K, k = Kb[np.ix_(tr, tr)], Kb[tr, t]
    n = len(tr)
    r = minimize(lambda a: (a @ K @ a - 2 * a @ k, 2 * K @ a - 2 * k), np.ones(n) / n, jac=True, method="SLSQP",
                 bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1}])
    a = np.clip(r.x, 0, None)
    return a / a.sum()


# ----------------------------------------------------------------- anomaly scores (fitted on one domain, unlabelled)

def maha_pcs(X, k):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = (X - mu) @ Vt[:k].T
    return ((P / (P.std(0) + 1e-9)) ** 2).sum(1)


def maha_sub(X, basis, shrink=0.1):
    P = (X - X.mean(0)) @ basis.T
    C = np.cov(P.T)
    C = (1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C))
    return np.einsum("ij,jk,ik->i", P, np.linalg.inv(C), P)


def knn_dist(X, k=10):
    P = X - X.mean(0)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    P = P @ Vt[:32].T
    P = P / (P.std(0) + 1e-9)
    sq = (P ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * P @ P.T, 0))
    np.fill_diagonal(D, np.inf)
    return np.sort(D, 1)[:, :k].mean(1)


def iforest(X):
    from sklearn.ensemble import IsolationForest
    P = X - X.mean(0)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    P = P @ Vt[:32].T
    return -IsolationForest(n_estimators=300, random_state=0).fit(P).score_samples(P)


# ----------------------------------------------------------------- label-free direction re-estimation

def gmm1d_post(s):
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, n_init=5, random_state=0).fit(s[:, None])
    hi = int(np.argmax(g.means_.ravel()))
    return g.predict_proba(s[:, None])[:, hi], float(g.weights_[hi])


def gmm_sub(X, s, w_dir, iters=50):
    """2-component, shared-covariance GMM in span(head direction, top domain PCs); init from the head-score median."""
    u = w_dir / np.linalg.norm(w_dir)
    R = X - X.mean(0)
    R2 = R - np.outer(R @ u, u)
    _, _, Vt = np.linalg.svd(R2, full_matrices=False)
    basis = np.vstack([u, Vt[:SUB_DIM - 1]])
    P = R @ basis.T
    r = (s > np.median(s)).astype(float)
    for _ in range(iters):
        pi = np.clip(r.mean(), 0.05, 0.95)
        m1 = (r[:, None] * P).sum(0) / r.sum()
        m0 = ((1 - r)[:, None] * P).sum(0) / (1 - r).sum()
        C = ((r[:, None] * (P - m1)).T @ (P - m1) + ((1 - r)[:, None] * (P - m0)).T @ (P - m0)) / len(P)
        C += 1e-3 * np.trace(C) / len(C) * np.eye(len(C))
        Ci = np.linalg.inv(C)
        a = Ci @ (m1 - m0)
        lo = P @ a - 0.5 * (m1 @ Ci @ m1 - m0 @ Ci @ m0) + np.log(pi / (1 - pi))
        r = 1 / (1 + np.exp(-np.clip(lo, -30, 30)))
    if np.corrcoef(lo, s)[0, 1] < 0:      # keep the component the head calls "wrong" as the positive one
        lo, r = -lo, 1 - r
    return r, lo


def ot_pseudo(X, s, pi, iters=20):
    """TSV-style: two prototypes, Sinkhorn assignment with class marginals (pi, 1-pi)."""
    P = X - X.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    P = P @ Vt[:32].T
    P = P / (P.std(0) + 1e-9)
    q = np.quantile(s, [0.3, 0.7])
    c1, c0 = P[s >= q[1]].mean(0), P[s <= q[0]].mean(0)
    n = len(P)
    for _ in range(iters):
        C = np.stack([((P - c0) ** 2).sum(1), ((P - c1) ** 2).sum(1)], 1)
        C = C / C.mean()
        Kk = np.exp(-C / OT_EPS)
        a, bm = np.ones(n) / n, np.array([1 - pi, pi])
        v = np.ones(2)
        for _ in range(200):
            uu = a / (Kk @ v + 1e-300)
            v = bm / (Kk.T @ uu + 1e-300)
        T = uu[:, None] * Kk * v[None]
        r = T[:, 1] / T.sum(1)
        c1 = (r[:, None] * P).sum(0) / r.sum()
        c0 = ((1 - r)[:, None] * P).sum(0) / (1 - r).sum()
    return r


def self_train(X, r, uni, l2p, rounds=1):
    m = LogReg(l2p, w0=uni.w, b0=uni.b).fit(X, r, wbal_soft(r))
    return m.score(X)


def wbal_soft(r):
    p = np.clip(r.mean(), 1e-3, 1 - 1e-3)
    # soft-label balanced weights: positives' expected mass and negatives' each get half
    return np.where(r >= 0.5, 0.5 / p, 0.5 / (1 - p))


# ----------------------------------------------------------------- bootstrap

def boot(scores, y, dom, doms, base, rng):
    idx = {d: np.where(dom == d)[0] for d in doms}
    res = {}
    draws = [{d: rng.choice(idx[d], len(idx[d])) for d in doms} for _ in range(B)]
    base_b = np.array([[auroc(base[dr[d]], y[dr[d]]) for d in doms] for dr in draws])
    for name, s in scores.items():
        mb = np.array([[auroc(s[dr[d]], y[dr[d]]) for d in doms] for dr in draws])
        m = np.nanmean(mb, 1)
        diff = m - np.nanmean(base_b, 1)
        res[name] = {"ci": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                     "diff_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/home/user/omni/q17")
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--out", default="results/omni/novel_transfer_anomaly.json")
    ap.add_argument("--cache", default="")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)

    rows = {}
    for l in open(a.corpus):
        r = json.loads(l)
        rows[r["id"]] = (r["domain"], r["format"], r["label"])
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache)
        ids, Xg = z["ids"], z["Xg"]
    else:
        F = load_features(a.feats)
        ids = F["ids"]
        n = len(ids)
        H = np.concatenate([F["r_mean"][:, MID].reshape(n, -1), F["j_state"][:, MID].reshape(n, -1)], 1).astype(np.float32)
        del F
        H = (H - H.mean(0)) / (H.std(0) + 1e-4)
        mu, V = pca_fit(H, PCA_K)
        Xg = ((H - mu) @ V.T).astype(np.float64)
        del H
        if a.cache:
            np.savez(a.cache, ids=ids, Xg=Xg)
    log("features", Xg.shape)
    dom = np.array([rows[i][0] for i in ids])
    fmt = np.array([rows[i][1] for i in ids])
    y = np.array([rows[i][2] for i in ids])
    doms = list(np.unique(dom))
    nd = len(doms)
    di = {d: k for k, d in enumerate(doms)}
    dfmt = {d: fmt[dom == d][0] for d in doms}
    Xs = standardise(Xg, dom)                               # per-domain standardisation (unlabelled)
    Zg = Xg / Xg.std(0)                                     # global coordinates for similarity
    rng = np.random.default_rng(0)
    Kb, MMD = kernel_means(Zg, dom, doms, rng)
    log("kernel means done")

    S = {}                                                  # method -> LODO scores (n,)
    info = {"knn_auto_k": {}, "kmm_top": {}, "iw_ess_frac": {}, "gmm_pi": {}}

    def put(name, te, s):
        S.setdefault(name, np.zeros(len(y)))[te] = s

    # per-training-domain heads (for the probe subspace) and pooled diff-of-means directions
    Xc = Xg.copy()
    for d in doms:
        Xc[dom == d] -= Xc[dom == d].mean(0)
    dom_heads = {d: fit(Xs[dom == d], y[dom == d]).w for d in doms}

    # ---------------- (1) similarity-weighted transfer
    # nested pseudo-target curves for the k rule: curve[t][k] = mean AUROC over pseudo-targets d != t
    curves = {}
    for t in doms:
        ti = di[t]
        tr_doms = [d for d in doms if d != t]
        acc = {k: [] for k in K_GRID + ["all"]}
        for d in tr_doms:
            others = [o for o in tr_doms if o != d]
            order = sorted(others, key=lambda o: MMD[di[d], di[o]])
            ted = dom == d
            for k in K_GRID + ["all"]:
                pick = order if k == "all" else order[:k]
                trm = np.isin(dom, pick)
                if len(set(y[trm])) < 2:
                    continue
                acc[k].append(auroc(fit(Xs[trm], y[trm]).score(Xs[ted]), y[ted]))
        curves[t] = {str(k): float(np.mean(v)) for k, v in acc.items()}
        log("nested curve", t, {k: round(v, 3) for k, v in curves[t].items()})

    uni_models = {}
    for t in doms:
        te = dom == t
        ti = di[t]
        tr = ~te
        tr_idx = [di[d] for d in doms if d != t]
        uni = fit(Xs[tr], y[tr])
        uni_models[t] = uni
        put("uniform", te, uni.score(Xs[te]))
        order = sorted([d for d in doms if d != t], key=lambda o: MMD[ti, di[o]])
        order_mean = sorted([d for d in doms if d != t],
                            key=lambda o: np.linalg.norm(Zg[dom == o].mean(0) - Zg[te].mean(0)))
        for k in K_GRID:
            m = np.isin(dom, order[:k])
            put(f"knn_mmd_{k}", te, fit(Xs[m], y[m]).score(Xs[te]))
            m = np.isin(dom, order_mean[:k])
            put(f"knn_mean_{k}", te, fit(Xs[m], y[m]).score(Xs[te]))
        c = curves[t]
        kbest = max(c, key=lambda k: c[k])
        info["knn_auto_k"][t] = kbest
        S.setdefault("knn_mmd_auto", np.zeros(len(y)))[te] = (S["uniform"][te] if kbest == "all"
                                                              else S[f"knn_mmd_{kbest}"][te])
        # soft MMD weights, temperature = median MMD^2 to the training domains
        md = np.array([MMD[ti, j] for j in tr_idx])
        wd = np.exp(-md / np.median(md))
        wmap = {d: wd[k] for k, d in enumerate([d for d in doms if d != t])}
        w = np.array([wmap.get(d, 0.0) for d in dom[tr]])
        put("mmd_soft", te, fit(Xs[tr], y[tr], w).score(Xs[te]))
        # KMM domain weights (mixture of domain distributions matched to the target), per-reply weight a_j / n_j
        al = kmm_weights(Kb, tr_idx, ti)
        amap = {d: al[k] / (dom == d).sum() for k, d in enumerate([d for d in doms if d != t])}
        wk = np.array([amap[d] for d in dom[tr]])
        wk = wk / wk.mean()
        info["kmm_top"][t] = {d: round(float(al[k]), 3) for k, d in enumerate([d for d in doms if d != t]) if al[k] > 0.05}
        put("kmm", te, fit(Xs[tr], y[tr], wk + 1e-6).score(Xs[te]))
        put("kmm_mix", te, fit(Xs[tr], y[tr], KMM_MIX * wk + (1 - KMM_MIX)).score(Xs[te]))
        # reply-level covariate-shift weights: domain classifier target-vs-pool on global coordinates
        from sklearn.linear_model import LogisticRegression
        lab = te.astype(int)
        clf = LogisticRegression(C=IW_C, max_iter=2000).fit(Zg, lab)
        p = clf.predict_proba(Zg[tr])[:, 1]
        iw = p / (1 - p)
        iw = np.minimum(iw, np.quantile(iw, IW_CLIP_Q))
        iw = iw / iw.mean()
        info["iw_ess_frac"][t] = float(iw.sum() ** 2 / (iw ** 2).sum() / len(iw))
        put("iw", te, fit(Xs[tr], y[tr], iw).score(Xs[te]))
        put("iw_tempered", te, fit(Xs[tr], y[tr], np.sqrt(iw)).score(Xs[te]))

        # ---------------- (2) anomaly scores on the target's own unlabelled replies
        Xt = Xs[te]
        put("an_maha_pc10", te, maha_pcs(Xt, 10))
        put("an_maha_pc32", te, maha_pcs(Xt, 32))
        Wd = np.stack([dom_heads[d] / np.linalg.norm(dom_heads[d]) for d in doms if d != t])
        _, _, Vb = np.linalg.svd(Wd, full_matrices=False)
        put("an_maha_probe", te, maha_sub(Xt, Vb[:5]))
        put("an_knn", te, knn_dist(Xt))
        put("an_iforest", te, iforest(Xt))
        dm = Xs[tr][y[tr] == 1].mean(0) - Xs[tr][y[tr] == 0].mean(0)
        put("an_dir_diffmeans", te, (Xt - Xt.mean(0)) @ dm)
        # centering only (no per-domain scaling): universal head fitted on centred-only features
        sdc = Xc[tr].std(0)
        put("an_dir_centred_head", te, fit(Xc[tr] / sdc, y[tr]).score(Xc[te] / sdc))

        # ---------------- (3) label-free direction re-estimation
        s0 = uni.score(Xt)
        r1, pi = gmm1d_post(s0)
        info["gmm_pi"][t] = pi
        for l2p in L2P_GRID:
            put(f"gmm1d_self_{l2p}", te, self_train(Xt, r1, uni, l2p))
        rs, lo = gmm_sub(Xt, s0, uni.w)
        put("gmm_sub16_post", te, lo)
        put("gmm_sub16_self", te, self_train(Xt, rs, uni, 0.5))
        ro = ot_pseudo(Xt, s0, float(np.clip(pi, 0.2, 0.8)))
        put("ot_post", te, ro)
        put("ot_self", te, self_train(Xt, ro, uni, 0.5))
        # iterate the 1-D GMM self-training 3 rounds
        s_ = s0
        for _ in range(3):
            rr, _ = gmm1d_post(s_)
            s_ = self_train(Xt, rr, uni, 0.5)
        put("gmm1d_self_3rounds", te, s_)
        log("done", t, "auroc uniform %.3f knn_auto(k=%s) %.3f iw %.3f maha_probe %.3f gmm1d %.3f" % (
            auroc(S["uniform"][te], y[te]), kbest, auroc(S["knn_mmd_auto"][te], y[te]), auroc(S["iw_tempered"][te], y[te]),
            auroc(S["an_maha_probe"][te], y[te]), auroc(S["gmm1d_self_0.5"][te], y[te])))

    # z-sum combinations with the head (per-domain z, zero labels)
    def zsum(*names):
        out = np.zeros(len(y))
        for d in doms:
            m = dom == d
            out[m] = sum(zs(S[n_][m]) for n_ in names)
        return out
    for an in [k for k in list(S) if k.startswith("an_")]:
        S[f"head+{an}"] = zsum("uniform", an)
    S["head+gmm1d_self"] = zsum("uniform", "gmm1d_self_0.5")
    # oracle: best k per target chosen with the target's labels (NOT valid, upper bound)
    S["knn_mmd_oracle"] = np.zeros(len(y))
    for t in doms:
        te = dom == t
        cand = ["uniform"] + [f"knn_mmd_{k}" for k in K_GRID]
        S["knn_mmd_oracle"][te] = S[max(cand, key=lambda c: auroc(S[c][te], y[te]))][te]

    # the 8B stack's reply score + the best label-free anomaly (does it add at the premium tier?)
    q8 = np.load("results/omni/expert_scores_q8b.npz")
    pos8 = {int(i): k for k, i in enumerate(q8["ids"])}
    o8 = np.array([pos8[int(i)] for i in ids])
    S["q8b_reply"] = q8["reply"][o8]
    S["q8b_reply+an_maha_probe"] = zsum("q8b_reply", "an_maha_probe")
    S["q8b_reply+an_knn"] = zsum("q8b_reply", "an_knn")

    # ---------------- summarise
    per = {n_: {d: auroc(s[dom == d], y[dom == d]) for d in doms} for n_, s in S.items()}
    fmts = sorted(set(dfmt.values()))
    summ = {}
    for n_, pd in per.items():
        summ[n_] = {"mean": float(np.mean(list(pd.values()))),
                    **{f: float(np.mean([v for d, v in pd.items() if dfmt[d] == f])) for f in fmts},
                    "wins_vs_uniform": int(sum(pd[d] > per["uniform"][d] for d in doms))}
    log("bootstrap")
    bs = boot(S, y, dom, doms, S["uniform"], np.random.default_rng(1))
    q8base = bs_q8 = None
    bs_q8 = boot({k: S[k] for k in ("q8b_reply+an_maha_probe", "q8b_reply+an_knn")}, y, dom, doms, S["q8b_reply"],
                 np.random.default_rng(2))
    for n_ in summ:
        summ[n_].update(bs[n_])
        if n_ in bs_q8:
            summ[n_]["diff_ci_vs_q8b_reply"] = bs_q8[n_]["diff_ci"]
        summ[n_]["beats_baseline_R2"] = bool(summ[n_]["diff_ci"][0] > 0 and summ[n_]["wins_vs_uniform"] >= 12)
    jd = json.load(open("results/omni/judge_distillation_q17.json"))
    judge_ref = {k: float(np.mean(list(v.values()))) for k, v in jd.items()}
    for k, v in jd.items():
        judge_ref[k + "_by_format"] = {f: float(np.mean([v[d] for d in doms if dfmt[d] == f])) for f in fmts}
    for n_, v in sorted(summ.items(), key=lambda kv: -kv[1]["mean"]):
        log(f"{n_:28s} {v['mean']:.3f} diff CI [{v['diff_ci'][0]:+.3f},{v['diff_ci'][1]:+.3f}] wins {v['wins_vs_uniform']:2d} "
            + " ".join(f"{f} {v[f]:.3f}" for f in fmts))
    log("judge distillation reference", {k: round(v, 3) for k, v in judge_ref.items() if not k.endswith("format")})
    rep = {"rules": __doc__.split("PRE-STATED RULES")[1].split("    OMP_")[0].strip(),
           "config": {"l2": L2, "pca": PCA_K, "layers": [15, 20, 24], "k_grid": K_GRID, "iw_C": IW_C,
                      "kmm_mix": KMM_MIX, "l2p_grid": L2P_GRID, "bootstrap": B},
           "summary": summ, "per_domain": per, "nested_k_curves": curves, "info": info,
           "domain_mmd2": {doms[i]: {doms[j]: float(MMD[i, j]) for j in range(nd)} for i in range(nd)},
           "judge_distillation_reference": judge_ref, "formats": dfmt}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
