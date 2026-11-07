"""Transfer and adaptation of a verifier head on stored reader features (experiments/omni_corpus.py).

The commercial questions, each measured per held-out domain (AUROC for "wrong / hallucinated"):

  ceiling     trained on the domain itself (5-fold within the domain)
  universal   trained on every other domain, zero labels from this one (leave-one-domain-out, LODO)
  +dstd       the same, with features standardised per domain from that domain's *unlabelled* replies
              (train domains each with their own statistics, the new domain with its own): a covariate-shift
              correction that costs nothing but a few hundred unlabelled replies
  LOFO        leave the whole *format* out (rag / chat / longform / reasoning): a new kind of output
  adapt@k     k labelled replies from the new domain: a fresh probe, and the universal probe refitted with
              a pull toward its own weights (the prior), compared on the rest of the domain
  guarantee   with k labels, a Clopper-Pearson threshold for <= 10% false alarms at 95% confidence: the
              false-alarm rate and recall actually reached on the rest

    python experiments/omni_eval.py --feats /home/user/omni/q17 --corpus /home/user/omni/corpus.jsonl \
        --report results/omni/q17.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omni_corpus import load_features  # noqa: E402


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)       # average ties
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = (y == 1).sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


# ----------------------------------------------------------------- a logistic probe with an optional prior

class LogReg:
    """L2 logistic regression on given (already standardised) features; with w0, the penalty pulls toward w0."""

    def __init__(self, l2=1.0, w0=None, b0=0.0):
        self.l2, self.w0, self.b0 = l2, w0, b0

    def fit(self, X, y, sample_weight=None):
        n, d = X.shape
        w0 = np.zeros(d) if self.w0 is None else self.w0
        sw = np.ones(n) if sample_weight is None else sample_weight
        sw = sw / sw.mean()

        def f(p):
            w, b = p[:d], p[d]
            z = X @ w + b
            ll = np.logaddexp(0, z) - y * z
            g = (1 / (1 + np.exp(-z)) - y) * sw
            loss = (sw * ll).mean() + 0.5 * self.l2 * ((w - w0) ** 2).sum()
            gw = X.T @ g / n + self.l2 * (w - w0)
            return loss, np.concatenate([gw, [g.mean()]])

        p0 = np.concatenate([w0, [self.b0]])
        r = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.w, self.b = r.x[:d], r.x[d]
        return self

    def score(self, X):
        return X @ self.w + self.b


def balanced_weights(y):
    p = y.mean()
    return np.where(y == 1, 0.5 / max(p, 1e-6), 0.5 / max(1 - p, 1e-6))


# ----------------------------------------------------------------- features

def reply_matrix(F, kind, layer_idx):
    """kind: '+'-joined parts of mean (reply mean state), last (reply last token), judge (state after the judge
    question), scalar (output signals + judge log-odds). Returns (hidden block, scalar block)."""
    n = len(F["ids"])
    hid, sc = [], []
    for part in kind.split("+"):
        if part == "mean":
            hid.append(F["r_mean"][:, layer_idx].reshape(n, -1).astype(np.float32))
        elif part == "last":
            hid.append(F["r_last"][:, layer_idx].reshape(n, -1).astype(np.float32))
        elif part == "judge":
            hid.append(F["j_state"][:, layer_idx].reshape(n, -1).astype(np.float32))
        elif part == "scalar":
            sc.append(np.concatenate([F["r_sc"], F["j_logit"][:, None]], 1).astype(np.float32))
    return (np.concatenate(hid, 1) if hid else None), (np.concatenate(sc, 1) if sc else None)


def standardise(X, groups=None, ref=None):
    """z-score; per group if groups is given (each group by its own mean / sd)."""
    X = X.copy()
    if groups is None:
        mu, sd = X.mean(0), X.std(0) + 1e-4
        return (X - mu) / sd
    for g in np.unique(groups):
        m = groups == g
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-4
        X[m] = (X[m] - mu) / sd
    return X


def pca_fit(X, k):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]


# ----------------------------------------------------------------- protocols

def fit_score(Xtr, ytr, Xte, l2):
    m = LogReg(l2).fit(Xtr, ytr, balanced_weights(ytr))
    return m.score(Xte), m


def lodo(X, y, dom, l2, per_domain_std, hold_key=None):
    """{held-out group: AUROC} training on all other groups. hold_key: array of groups to leave out (default dom)."""
    key = dom if hold_key is None else hold_key
    Xs = standardise(X, dom) if per_domain_std else None
    out, scores = {}, np.zeros(len(y))
    for g in np.unique(key):
        te = key == g
        if per_domain_std:
            Xtr, Xte = Xs[~te], Xs[te]
        else:
            mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-4
            Xtr, Xte = (X[~te] - mu) / sd, (X[te] - mu) / sd
        s, _ = fit_score(Xtr, y[~te], Xte, l2)
        scores[te] = s
        for d in np.unique(dom[te]):
            out[d] = auroc(s[dom[te] == d], y[te][dom[te] == d])
    return out, scores


def in_domain(X, y, dom, l2, folds=5, seed=0):
    out = {}
    rng = np.random.default_rng(seed)
    for d in np.unique(dom):
        idx = np.where(dom == d)[0]
        rng.shuffle(idx)
        s = np.zeros(len(idx))
        Xd = standardise(X[idx])
        for f in range(folds):
            te = np.zeros(len(idx), bool)
            te[f::folds] = True
            if len(set(y[idx][~te])) < 2:
                continue
            s[te], _ = fit_score(Xd[~te], y[idx][~te], Xd[te], l2)
        out[d] = auroc(s, y[idx])
    return out


def cp_threshold(scores_clean, fpr, conf):
    """smallest threshold whose false-alarm rate is <= fpr with confidence conf (one-sided Clopper-Pearson)."""
    s = np.sort(scores_clean)[::-1]
    n = len(s)
    for k in range(n + 1):                       # k clean scores at or above the threshold
        ub = 1.0 if k == n else beta_dist.ppf(conf, k + 1, n - k)
        if ub > fpr:
            return s[k - 1] + 1e-9 if k > 0 else s[0] + 1e-9
    return s[-1]


def adapt(X, y, dom, l2, ks, reps, per_domain_std, l2_prior_grid=(0.01, 0.1, 1.0, 10.0), seed=0):
    """per held-out domain and k: AUROC on the rest for (fresh probe, universal prior refit, universal as is),
    and the guarantee for the refit."""
    Xs = standardise(X, dom) if per_domain_std else None
    res = {}
    rng = np.random.default_rng(seed)
    for g in np.unique(dom):
        te = dom == g
        if per_domain_std:
            Xtr, Xg = Xs[~te], Xs[te]
        else:
            mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-4
            Xtr, Xg = (X[~te] - mu) / sd, (X[te] - mu) / sd
        yg = y[te]
        uni = LogReg(l2).fit(Xtr, y[~te], balanced_weights(y[~te]))
        res[g] = {}
        for k in ks:
            rows = {"fresh": [], "prior": [], "universal": [], "g_fpr": [], "g_recall": []}
            for r in range(reps):
                pos, neg = np.where(yg == 1)[0], np.where(yg == 0)[0]
                kp = k // 2
                if kp >= len(pos) - 5 or k - kp >= len(neg) - 5:
                    continue
                cal = np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, k - kp, replace=False)])
                rest = np.setdiff1d(np.arange(len(yg)), cal)
                rows["universal"].append(auroc(uni.score(Xg[rest]), yg[rest]))
                if k == 0:
                    continue
                Xc, yc = Xg[cal], yg[cal]
                fresh = LogReg(l2).fit(Xc, yc, balanced_weights(yc))
                rows["fresh"].append(auroc(fresh.score(Xg[rest]), yg[rest]))
                # the prior's strength by leave-half-out on the k labels themselves
                best, best_s = None, -1
                for lp in l2_prior_grid:
                    sc = []
                    for rr in range(3):
                        perm = rng.permutation(len(cal))
                        a, b = perm[: len(cal) // 2], perm[len(cal) // 2:]
                        if len(set(yc[a])) < 2 or len(set(yc[b])) < 2:
                            continue
                        m = LogReg(lp, w0=uni.w, b0=uni.b).fit(Xc[a], yc[a], balanced_weights(yc[a]))
                        sc.append(auroc(m.score(Xc[b]), yc[b]))
                    v = np.mean(sc) if sc else -1
                    if v > best_s:
                        best, best_s = lp, v
                m = LogReg(best, w0=uni.w, b0=uni.b).fit(Xc, yc, balanced_weights(yc))
                rows["prior"].append(auroc(m.score(Xg[rest]), yg[rest]))
                # guarantee: threshold from the clean calibration replies' *out-of-fold* scores (cross-fitting:
                # each is scored by a refit that never saw it), so the head's fit does not bias the threshold
                oof = np.zeros(len(cal))
                folds = np.arange(len(cal)) % 5
                rng.shuffle(folds)
                for f_ in range(5):
                    tr_, te_ = folds != f_, folds == f_
                    if len(set(yc[tr_])) < 2:
                        oof[te_] = Xc[te_] @ uni.w + uni.b
                        continue
                    mf = LogReg(best, w0=uni.w, b0=uni.b).fit(Xc[tr_], yc[tr_], balanced_weights(yc[tr_]))
                    oof[te_] = mf.score(Xc[te_])
                clean = oof[yc == 0]
                t = cp_threshold(clean, 0.10, 0.95)
                sr = m.score(Xg[rest])
                rows["g_fpr"].append(float((sr[yg[rest] == 0] >= t).mean()))
                rows["g_recall"].append(float((sr[yg[rest] == 1] >= t).mean()))
            res[g][k] = {kk: (float(np.nanmean(v)) if v else None) for kk, v in rows.items()}
            res[g][k]["g_fpr_over_12pct"] = float(np.mean(np.array(rows["g_fpr"]) > 0.12)) if rows["g_fpr"] else None
    return res


def guarantee_curve(X, y, dom, l2, ks, reps, prevalence=0.2, fpr=0.10, conf=0.95, seed=0):
    """Realistic onboarding: the domain's traffic has `prevalence` wrong replies; k replies are drawn uniformly and
    labelled; only their clean ones set the threshold. Universal head on the domain's own standardisation, threshold
    by Clopper-Pearson; false alarms and recall measured on the rest of the domain's traffic."""
    Xs = standardise(X, dom)
    rng = np.random.default_rng(seed)
    out = {}
    for g in np.unique(dom):
        te = dom == g
        uni = LogReg(l2).fit(Xs[~te], y[~te], balanced_weights(y[~te]))
        sg, yg = uni.score(Xs[te]), y[te]
        neg, pos = np.where(yg == 0)[0], np.where(yg == 1)[0]
        n_pos = min(len(pos), int(round(len(neg) * prevalence / (1 - prevalence))))
        out[g] = {}
        for k in ks:
            fa, rec, nclean = [], [], []
            for _ in range(reps):
                pool = np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])
                if k >= len(pool) - 30:
                    break
                cal = rng.choice(pool, k, replace=False)
                rest = np.setdiff1d(pool, cal)
                clean = sg[cal][yg[cal] == 0]
                t = cp_threshold(clean, fpr, conf)
                fa.append(float((sg[rest][yg[rest] == 0] >= t).mean()))
                rec.append(float((sg[rest][yg[rest] == 1] >= t).mean()))
                nclean.append(len(clean))
            if fa:
                out[g][k] = {"fpr_mean": float(np.mean(fa)), "fpr_over_12pct": float(np.mean(np.array(fa) > 0.12)),
                             "recall": float(np.mean(rec)), "clean_labels": float(np.mean(nclean))}
    return out


def mean(d):
    v = [x for x in d.values() if x == x]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--pca", type=int, default=256, help="PCA dims of the hidden-state block (0 = none)")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 10, 25, 50, 100])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    F = load_features(a.feats)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.corpus)}
    dom = np.array([rows[i]["domain"] for i in F["ids"]])
    fmt = np.array([rows[i]["format"] for i in F["ids"]])
    y = np.array([rows[i]["label"] for i in F["ids"]])
    nlay = F["r_mean"].shape[1]
    print(f"{len(y)} replies, {len(np.unique(dom))} domains, layers {F['meta']['layers']}", flush=True)
    report = {"meta": F["meta"], "n": int(len(y)), "domains": {d: int((dom == d).sum()) for d in np.unique(dom)}}

    proj = {}

    def build(kind, layers):
        H, S = reply_matrix(F, kind, layers)
        if H is not None and a.pca:
            hmu, hsd = H.mean(0), H.std(0) + 1e-4
            H = (H - hmu) / hsd
            mu, V = pca_fit(H, a.pca)
            H = (H - mu) @ V.T
            proj[kind, tuple(layers)] = (hmu, hsd, mu, V)
        return np.concatenate([x for x in (H, S) if x is not None], 1)

    # 0. training-free baselines (per domain, no fitting): judge log-odds (Yes = correct), token confidence
    tf = {"judge_no_logodds": -F["j_logit"], "low_mean_logprob": -F["r_sc"][:, 0], "high_mean_entropy": F["r_sc"][:, 3]}
    report["training_free"] = {}
    for name, s_ in tf.items():
        per = {d: auroc(s_[dom == d], y[dom == d]) for d in np.unique(dom)}
        report["training_free"][name] = {"per_domain": per, "mean": mean(per)}
        print(f"training-free {name:18s} mean AUROC {mean(per):.3f}", flush=True)

    # 1. which features: per layer, zero-label LODO and in-domain
    feats = {"scalar": ("scalar", [0])}
    for li in range(nlay):
        feats[f"mean_L{F['meta']['layers'][li]}"] = ("mean", [li])
    for li in range(nlay):
        feats[f"judge_L{F['meta']['layers'][li]}"] = ("judge", [li])
    mid = [nlay // 2 - 1, nlay // 2, nlay // 2 + 1]
    feats["mean_mid3"] = ("mean", mid)
    feats["mean+judge_mid3"] = ("mean+judge", mid)
    feats["mean+judge+scalar_mid3"] = ("mean+judge+scalar", mid)
    report["features"] = {}
    for name, (kind, layers) in feats.items():
        X = build(kind, layers)
        ind = in_domain(X, y, dom, a.l2)
        lo, _ = lodo(X, y, dom, a.l2, False)
        lod, _ = lodo(X, y, dom, a.l2, True)
        report["features"][name] = {"in_domain": ind, "lodo": lo, "lodo_dstd": lod,
                                    "mean": {"in_domain": mean(ind), "lodo": mean(lo), "lodo_dstd": mean(lod)}}
        print(f"{name:12s} in-domain {mean(ind):.3f}  LODO {mean(lo):.3f}  LODO+dstd {mean(lod):.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        if a.quick and name.startswith("mean_L") and len(report["features"]) > 3:
            break
    best = max((k for k in report["features"] if k != "scalar"),
               key=lambda k: report["features"][k]["mean"]["lodo_dstd"])
    report["best_features"] = best
    kind, layers = feats[best]
    X = build(kind, layers)
    # 2. leave-one-format-out
    _, reply_lodo = lodo(X, y, dom, a.l2, True)
    lf, _ = lodo(X, y, dom, a.l2, False, hold_key=fmt)
    lfd, _ = lodo(X, y, dom, a.l2, True, hold_key=fmt)
    report["lofo"] = {"plain": lf, "dstd": lfd, "mean": {"plain": mean(lf), "dstd": mean(lfd)}}
    print(f"LOFO ({best}): {mean(lf):.3f}  +dstd {mean(lfd):.3f}", flush=True)
    # 2b. diversity: zero-label AUROC on a held-out domain vs the number of training domains (random subsets)
    Xs = standardise(X, dom)
    doms = np.unique(dom)
    rng = np.random.default_rng(0)
    curve = {}
    for m in [1, 2, 4, 8, 12, len(doms) - 1]:
        vals = []
        for g in doms:
            for rep in range(3 if m < len(doms) - 1 else 1):
                others = [d for d in doms if d != g]
                pick = rng.choice(others, m, replace=False)
                tr = np.isin(dom, pick)
                if len(set(y[tr])) < 2:
                    continue
                s_, _ = fit_score(Xs[tr], y[tr], Xs[dom == g], a.l2)
                vals.append(auroc(s_, y[dom == g]))
        curve[m] = float(np.nanmean(vals))
        print(f"diversity: {m:2d} training domains -> held-out AUROC {curve[m]:.3f}", flush=True)
    report["diversity_curve"] = curve

    # 3. adaptation + guarantee
    report["adapt"] = adapt(X, y, dom, a.l2, a.ks, a.reps, True)
    for k in a.ks:
        def avg(key):
            v = [report["adapt"][g][k][key] for g in report["adapt"] if report["adapt"][g][k][key] is not None]
            return float(np.mean(v)) if v else float("nan")
        print(f"k={k:3d}: universal {avg('universal'):.3f} fresh {avg('fresh'):.3f} prior {avg('prior'):.3f}"
              f"  guarantee fpr {avg('g_fpr'):.3f} recall {avg('g_recall'):.3f}", flush=True)
    # 3b. realistic guarantee: uniform labelling at 20% prevalence, calibration only (no adaptation)
    gc = guarantee_curve(X, y, dom, a.l2, [50, 100, 150], max(a.reps, 20))
    report["guarantee_curve"] = gc
    for k in (50, 100, 150):
        v = [gc[g][k] for g in gc if k in gc[g]]
        if v:
            print(f"guarantee @ {k} uniform labels (~{np.mean([x['clean_labels'] for x in v]):.0f} clean): false alarms "
                  f"{np.mean([x['fpr_mean'] for x in v]):.3f} (draws over 12%: {np.mean([x['fpr_over_12pct'] for x in v]):.3f})"
                  f", recall {np.mean([x['recall'] for x in v]):.3f}  [{len(v)} domains]", flush=True)

    # 4. units (sentence / segment / step): localisation and worst-unit reply score, zero labels from the domain
    unit_layers = feats["mean_mid3"][1]
    Xr = build("mean+scalar", unit_layers)
    hmu, hsd, pmu, V = proj["mean+scalar", tuple(unit_layers)]
    U = F["u_mean"][:, unit_layers].reshape(len(F["u_row"]), -1).astype(np.float32)
    U = (((U - hmu) / hsd) - pmu) @ V.T
    usc = np.concatenate([F["u_sc"], np.zeros((len(U), 1), np.float32)], 1)   # no judge log-odds per unit
    U = np.concatenate([U, usc], 1)
    pos = {i: k for k, i in enumerate(F["ids"])}
    udom = dom[[pos[i] for i in F["u_row"]]]
    ulab = F["u_lab"]
    Us = standardise(U, udom)
    rep_units = {}
    for d in np.unique(dom):
        te_u, tr_u = udom == d, (udom != d) & (ulab >= 0)
        m = LogReg(a.l2).fit(Us[tr_u], ulab[tr_u].astype(float), balanced_weights(ulab[tr_u].astype(float)))
        su = m.score(Us[te_u])
        lab_te = ulab[te_u]
        loc = auroc(su[lab_te >= 0], lab_te[lab_te >= 0]) if (lab_te >= 0).sum() > 10 else float("nan")
        rows_te = F["u_row"][te_u]
        worst = {}
        for r_, v in zip(rows_te, su):
            worst[r_] = max(worst.get(r_, -1e9), v)
        idx = np.where(dom == d)[0]
        wv = np.array([worst.get(F["ids"][i], np.nan) for i in idx])
        ok = ~np.isnan(wv)
        z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)        # per-domain, unlabelled
        comb = z(reply_lodo[idx]) + np.where(ok, z(np.where(ok, wv, np.nanmean(wv))), 0)
        rep_units[d] = {"unit_auroc": loc, "worst_unit_reply_auroc": auroc(wv[ok], y[idx][ok]),
                        "reply_auroc": auroc(reply_lodo[idx], y[idx]), "combined_auroc": auroc(comb, y[idx])}
    report["units"] = rep_units
    print("units: localisation AUROC (labelled units, LODO) mean %.3f; worst-unit reply AUROC mean %.3f; "
          "reply %.3f; reply+worst-unit %.3f" % tuple(mean({d: v[k] for d, v in rep_units.items()}) for k in
                                                       ("unit_auroc", "worst_unit_reply_auroc", "reply_auroc",
                                                        "combined_auroc")), flush=True)
    for d, v in rep_units.items():
        if v["unit_auroc"] == v["unit_auroc"]:
            print(f"   {d:22s} unit {v['unit_auroc']:.3f}  worst-unit reply {v['worst_unit_reply_auroc']:.3f}"
                  f"  combined {v['combined_auroc']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(report, open(a.report, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {a.report}")


if __name__ == "__main__":
    main()
