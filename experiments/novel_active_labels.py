"""Which replies should a customer label first? Active / smart labelling for onboarding, from STORED scores only.

Inputs (every expert already held out leave-one-domain-out, as experiments/omni_stack.py):
    results/omni/expert_scores_q8b.npz   reply head, worst unit, the 8B reader's own judge
    results/omni/judge_probs.npz         external judges' logit P(correct): Qwen3-14B, phi-4
    /home/user/omni/nli.npz              NLI (HHEM reply + MiniCheck worst sentence) on 1,200 grounded replies

Labels are revealed only for the replies a scheme chooses.

(a) ADAPTATION (expert weights refitted on k labels, pulled toward the zero-label weights, as omni_stack adapt_k).
    Per domain and draw: a pool (20% prevalence, or the corpus's natural 50%), split at random into candidates (60%)
    and a test set (40%) the selection never touches; AUROC on the test set vs labels k in (10, 25, 50, 100).
    Strategies (an ordering of the candidates; budgets are prefixes, so they are nested):
      uniform           random order (today's protocol)
      uncert-static     closest to the 80th percentile of the zero-label score (the boundary at an assumed 20% rate)
      uncert-seq        10 uniform seeds, then batches of 5 closest to the refitted model's decision boundary
      uncert+uniform    10 uniform seeds, then batches of 10: 5 uniform + 5 closest to the boundary
      disagree          largest |z(external judges) - z(reader internals)|
      diversity         k-center greedy (farthest first) in the standardised expert-score space

(b) GUARANTEE (threshold for <= 10% false alarms with 95% confidence). Per domain and draw: a pool with all clean
    replies and wrong ones at 20% prevalence (as judge_calibration_sim.py; the same 14 domains). The pool is the
    traffic the promise is about; every scheme is scored by its realised false-alarm rate on ALL clean replies of the
    pool and recall on all wrong ones. Two frames:
      bootstrap   labels drawn WITH replacement (traffic = the pool's empirical distribution: a large population,
                  as a customer with thousands of unlabelled replies) -- the primary frame
      finite      labels drawn without replacement from the pool
    Schemes, B labels:
      uniform           B uniform labels, Clopper-Pearson (CP) threshold on the clean ones (shipped protocol)
      uniform-rest      the same at B = 100, scored on the unlabelled rest (the shipped baseline's frame; finite only)
      lowest-first      the B replies with the lowest score (the stack's "most likely clean"), CP on the clean ones
      judge-clean-first the B replies the external judges most believe correct, CP on the clean ones
      strat(q, f)       two strata from the UNLABELLED scores: H = top q of the pool, L = the rest; f*B labels drawn
                        uniformly within H, the rest within L. A conservative stratified bound (exact, finite sample):
                          FAR(t) = P(clean, s >= t) / P(clean),   t inside H
                                 <= P(H) * CPup(#clean labelled in H with s >= t ; m_H, d1)
                                    / [P(L) * CPlo(#clean in L sample ; m_L, d2) + P(H) * CPlo(#clean in H ; m_H, d3)]
                        d1 = 0.03, d2 = d3 = 0.01 (Bonferroni, total 0.05); thresholds scanned from the top
                        (fixed-sequence), so the bound holds simultaneously. P(H), P(L) are the stratum shares of the
                        unlabelled traffic (known: scoring unlabelled replies is free). Infeasible -> flag nothing.
      strat-cond(q, f)  the conditional form FAR = P(H | clean) * P(s >= t | clean, H), same strata
      ipw-kish(q, f)    the same stratified sample, inverse-probability-weighted FAR estimate, CP on the Kish effective
                        sample size (approximate, no finite-sample proof: tested for validity)
    Reported per scheme and B: mean realised false alarms, share of draws over 10% / 12%, recall, feasible share.
    Scores: 'stack8b' (z reply + z worst unit + 2 z own judge: the shipped baseline score) and 'stack+judges'
    (+ z of both external judges).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/novel_active_labels.py \
        --report results/omni/novel_active_labels.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_eval import LogReg, auroc, balanced_weights, cp_threshold  # noqa: E402

ALPHA, CONF = 0.10, 0.95
D_TAIL, D_CL, D_CH = 0.03, 0.01, 0.01


def z(v):
    v = np.asarray(v, float)
    m = np.isnan(v)
    if m.all():
        return np.zeros_like(v)
    return np.where(m, 0.0, (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9))


# ------------------------------------------------------------------ data

def load(a):
    S = np.load(a.scores, allow_pickle=True)
    J = np.load(a.judge, allow_pickle=True)
    ids = S["ids"].astype(int)
    pj = {int(i): k for k, i in enumerate(J["ids"])}
    o = np.array([pj[int(i)] for i in ids])
    nli = np.full(len(ids), np.nan)
    if a.nli and os.path.exists(a.nli):
        Z = np.load(a.nli)
        pos = {int(i): k for k, i in enumerate(ids)}
        for i, r in zip(Z["ids"], Z["nli"]):
            if int(i) in pos:
                nli[pos[int(i)]] = (r[2] + r[1]) / 2
    return dict(ids=ids, dom=S["domain"].astype(str), fmt=S["format"].astype(str), y=S["label"].astype(int),
                reply=S["reply"], wu=S["worst_unit"], judge=S["judge"],
                j14=-J["logit_qwen14"][o], phi=-J["logit_phi4"][o], nli=nli)


def domain_features(D, d):
    m = D["dom"] == d
    cols = [z(D["reply"][m]), z(D["wu"][m]), z(D["judge"][m]), z(D["j14"][m]), z(D["phi"][m])]
    names = ["reply", "worst_unit", "judge", "qwen14", "phi4"]
    if not np.isnan(D["nli"][m]).all():
        cols.append(z(D["nli"][m]))
        names.append("nli")
    return np.stack(cols, 1), names, D["y"][m], str(D["fmt"][m][0])


# ------------------------------------------------------------------ (a) adaptation

def fit_score(X, y, Xt, w0, s0t):
    if len(set(y.tolist())) < 2:
        return s0t
    k = len(y)
    m = LogReg(2.0 / k, w0=w0 * 0.5).fit(X, y.astype(float), balanced_weights(y.astype(float)))
    return m.score(Xt)


def kcenter_order(X, n):
    c = np.argmin(((X - X.mean(0)) ** 2).sum(1))
    order = [c]
    dmin = ((X - X[c]) ** 2).sum(1)
    for _ in range(n - 1):
        j = int(np.argmax(dmin))
        order.append(j)
        dmin = np.minimum(dmin, ((X - X[j]) ** 2).sum(1))
    return np.array(order)


def seq_uncertainty(X, y, w0, s0, nmax, rng, n_uniform_per_batch=0, batch=5, seed=10):
    """returns an ordering of indices: `seed` uniform, then batches of `batch` (of which n_uniform_per_batch uniform,
    the rest closest to the current model's decision boundary). Labels are read only for chosen indices."""
    n = len(y)
    order = list(rng.permutation(n)[:seed])
    while len(order) < nmax:
        chosen = np.array(order)
        left = np.setdiff1d(np.arange(n), chosen)
        if not len(left):
            break
        yl = y[chosen]
        n_unc = batch - n_uniform_per_batch
        if len(set(yl.tolist())) == 2:
            k = len(chosen)
            m = LogReg(2.0 / k, w0=w0 * 0.5).fit(X[chosen], yl.astype(float), balanced_weights(yl.astype(float)))
            dist = np.abs(m.score(X[left]))
        else:                      # one class seen so far: look where the other class should be
            dist = s0[left] * (-1 if yl[0] == 0 else 1)
        pick = list(left[np.argsort(dist)[:n_unc]])
        rest = np.setdiff1d(left, pick)
        if n_uniform_per_batch and len(rest):
            pick += list(rng.choice(rest, min(n_uniform_per_batch, len(rest)), replace=False))
        order += pick
    return np.array(order[:nmax])


STRATS = ["uniform", "uncert-static", "uncert-seq", "uncert+uniform", "disagree", "diversity"]


def part_a(D, a, rng):
    out = {}
    for d in np.unique(D["dom"]):
        X, names, y, fmt = domain_features(D, d)
        w0 = np.ones(X.shape[1])
        s0 = X @ w0
        zJ = z(X[:, 3] + X[:, 4])
        zR = z(X[:, 0] + X[:, 1] + X[:, 2])
        dis = np.abs(zJ - zR)
        row = {"format": fmt, "experts": names}
        for prev in a.prevalences:
            neg, wr = np.where(y == 0)[0], np.where(y == 1)[0]
            res = {s: {b: [] for b in a.budgets} for s in STRATS}
            res["zero_label"] = {b: [] for b in a.budgets}
            for _ in range(a.draws_a):
                if prev == "natural":
                    pool = np.arange(len(y))
                else:
                    npos = min(len(wr), int(round(len(neg) * prev / (1 - prev))))
                    pool = np.concatenate([neg, rng.choice(wr, npos, replace=False)])
                pool = rng.permutation(pool)
                nt = int(round(0.4 * len(pool)))
                test, cand = pool[:nt], pool[nt:]
                Xc, yc, Xt, yt = X[cand], y[cand], X[test], y[test]
                s0c, s0t = s0[cand], s0[test]
                bud = [b for b in a.budgets if b <= len(cand) - 5]
                if not bud:
                    continue
                nmax = max(bud)
                q80 = np.quantile(s0c, 0.8)
                orders = {
                    "uniform": rng.permutation(len(cand))[:nmax],
                    "uncert-static": np.argsort(np.abs(s0c - q80))[:nmax],
                    "uncert-seq": seq_uncertainty(Xc, yc, w0, s0c, nmax, rng),
                    "uncert+uniform": seq_uncertainty(Xc, yc, w0, s0c, nmax, rng, n_uniform_per_batch=5, batch=10),
                    "disagree": np.argsort(-dis[cand])[:nmax],
                    "diversity": kcenter_order(Xc, nmax),
                }
                za = auroc(s0t, yt)
                for b in bud:
                    res["zero_label"][b].append(za)
                    for s, o in orders.items():
                        sel = o[:b]
                        res[s][b].append(auroc(fit_score(Xc[sel], yc[sel], Xt, w0, s0t), yt))
            row[str(prev)] = {s: {str(b): (float(np.nanmean(v)) if v else None) for b, v in r.items()}
                              for s, r in res.items()}
        out[d] = row
        print("a", d, {str(p): {s: row[str(p)][s].get("50") for s in ["zero_label"] + STRATS} for p in a.prevalences},
              flush=True)
    return out


# ------------------------------------------------------------------ (b) guarantee

class CPTab:
    """Clopper-Pearson one-sided bounds tabulated for n <= nmax."""

    def __init__(self, nmax, d):
        self.up = np.ones((nmax + 1, nmax + 1))
        self.lo = np.zeros((nmax + 1, nmax + 1))
        for n in range(1, nmax + 1):
            k = np.arange(n)
            self.up[n, :n] = beta_dist.ppf(1 - d, k + 1, n - k)
            k1 = np.arange(1, n + 1)
            self.lo[n, 1:n + 1] = beta_dist.ppf(d, k1, n - k1 + 1)


def cp_thresh_tab(sc, tab):
    """= omni_eval.cp_threshold(sc, ALPHA, CONF): just above the k-th highest clean score, k the most exceedances
    whose CP upper bound stays <= ALPHA. inf if no threshold can carry the promise (n < 29)."""
    n = len(sc)
    if n == 0:
        return np.inf
    s = np.sort(sc)[::-1]
    ok = tab.up[n, :n + 1] <= ALPHA
    if not ok[0]:
        return np.inf
    k = int(np.argmin(ok)) if not ok.all() else n       # first failing k
    k -= 1                                                # most exceedances allowed
    return float(s[k] + 1e-9) if k < n else float(s[-1])


def strat_thresh(s, H, labH, labL, yH, yL, tabT, tabL, tabC, M):
    """conservative stratified bound (see module doc). s: pool scores; H: mask of the top stratum; labH/labL: labelled
    indices in each stratum; yH/yL their labels."""
    nH = int(H.sum())
    pH, pL = nH / M, 1 - nH / M
    mH, mL = len(labH), len(labL)
    cL = tabL.lo[mL, int((yL == 0).sum())] if mL else 0.0
    cH = tabC.lo[mH, int((yH == 0).sum())] if mH else 0.0
    den = pL * cL + pH * cH
    if den <= 0 or mH == 0:
        return np.inf
    sc = np.sort(s[labH][yH == 0])[::-1]
    mc = len(sc)
    bound = pH * tabT.up[mH, :mc + 1] / den               # k = 0 .. mc exceedances
    ok = bound <= ALPHA
    if not ok[0]:
        return np.inf
    k = (int(np.argmin(ok)) if not ok.all() else mc + 1) - 1
    return float(sc[k] + 1e-9) if k < mc else float(s[H].min())


def strat_cond_thresh(s, H, labH, labL, yH, yL, tabT, tabL, tabC, M):
    """conditional form: FAR(t) = P(H | clean) * P(s >= t | clean, H) for t inside H, with
    P(H | clean) <= pH cH_up / (pH cH_up + pL cL_lo) and the tail by CP on the clean labelled replies of H
    (uniform within clean & H given their number)."""
    nH = int(H.sum())
    pH, pL = nH / M, 1 - nH / M
    mH, mL = len(labH), len(labL)
    cHn, cLn = int((yH == 0).sum()), int((yL == 0).sum())
    cL = tabL.lo[mL, cLn] if mL else 0.0
    cH = tabC.up[mH, cHn] if mH else 1.0
    r = pH * cH / (pH * cH + pL * cL) if pH * cH + pL * cL > 0 else 1.0
    if cHn == 0:
        return np.inf
    sc = np.sort(s[labH][yH == 0])[::-1]
    bound = r * tabT.up[cHn, :cHn + 1]
    ok = bound <= ALPHA
    if not ok[0]:
        return np.inf
    k = (int(np.argmin(ok)) if not ok.all() else cHn + 1) - 1
    return float(sc[k] + 1e-9) if k < cHn else float(s[H].min())


def ipw_thresh(s, lab, w, ylab):
    """inverse-probability-weighted FAR estimate with a CP bound on the Kish effective size (approximate)."""
    c = ylab == 0
    sc, wc = s[lab][c], w[c]
    if not len(sc):
        return np.inf
    o = np.argsort(-sc)
    sc, wc = sc[o], wc[o]
    neff = wc.sum() ** 2 / (wc ** 2).sum()
    far = np.concatenate([[0.0], np.cumsum(wc)]) / wc.sum()      # k = 0..n exceedances
    kk = far * neff
    up = np.where(kk < neff, beta_dist.ppf(CONF, kk + 1, np.maximum(neff - kk, 1e-9)), 1.0)
    ok = up <= ALPHA
    if not ok[0]:
        return np.inf
    k = (int(np.argmin(ok)) if not ok.all() else len(sc) + 1) - 1
    return float(sc[k] + 1e-9) if k < len(sc) else float(sc[-1])


def part_b(D, a, rng):
    nmax = 260
    tab95 = CPTab(nmax, 1 - CONF)
    tabT, tabL, tabC = CPTab(nmax, D_TAIL), CPTab(nmax, D_CL), CPTab(nmax, D_CH)
    # sanity: the tabulated CP threshold equals the shipped one
    rr = np.random.default_rng(1)
    for n in (30, 50, 80, 120):
        x = rr.normal(size=n)
        assert abs(cp_thresh_tab(x, tab95) - cp_threshold(x, ALPHA, CONF)) < 1e-6, n
    out = {}
    for d in np.unique(D["dom"]):
        X, _, y, fmt = domain_features(D, d)
        m = D["dom"] == d
        scores = {"stack8b": z(D["reply"][m]) + z(D["wu"][m]) + 2 * z(D["judge"][m]),
                  "stack+judges": X[:, :5].sum(1)}
        jclean = -(z(D["j14"][m]) + z(D["phi"][m]))          # external judges' belief that the reply is correct
        neg, wr = np.where(y == 0)[0], np.where(y == 1)[0]
        npos = min(len(wr), int(round(len(neg) * 0.25)))
        if len(neg) + npos - 100 < 40:                        # the shipped baseline's domain rule (14 domains)
            continue
        row = {"format": fmt, "pool": int(len(neg) + npos)}
        for frame in a.frames:
          row[frame] = {}
          for sname, sfull in scores.items():
            res = {}

            def add(name, B, t, sp, yp, rest=None):
                r = res.setdefault(name, {}).setdefault(B, {"fa": [], "rc": [], "feas": []})
                ev = np.ones(len(sp), bool) if rest is None else rest
                r["fa"].append(float((sp[ev & (yp == 0)] >= t).mean()))
                r["rc"].append(float((sp[ev & (yp == 1)] >= t).mean()))
                r["feas"].append(bool(np.isfinite(t)))

            boot = frame == "bootstrap"

            def draw(idx, m):
                """m labels uniformly from idx: with replacement in the bootstrap frame (traffic = the pool's
                empirical distribution, i.e. a large population), without in the finite frame."""
                if m <= 0 or not len(idx):
                    return idx[:0]
                return rng.choice(idx, m, replace=True) if boot else rng.permutation(idx)[:m]

            for _ in range(a.draws_b):
                pool = np.concatenate([neg, rng.choice(wr, npos, replace=False)])
                sp, yp, jp = sfull[pool], y[pool], jclean[pool]
                M = len(pool)
                rank = np.argsort(np.argsort(-sp))               # 0 = highest score
                allidx = np.arange(M)
                for B in a.budgets_b:
                    lab = draw(allidx, B)
                    add("uniform", B, cp_thresh_tab(sp[lab][yp[lab] == 0], tab95), sp, yp)
                    if B == 100 and not boot:
                        rest = np.ones(M, bool)
                        rest[lab] = False
                        add("uniform-rest", B, cp_thresh_tab(sp[lab][yp[lab] == 0], tab95), sp, yp, rest)
                    low = np.argsort(sp)[:B]
                    add("lowest-first", B, cp_thresh_tab(sp[low][yp[low] == 0], tab95), sp, yp)
                    jc = np.argsort(-jp)[:B]
                    add("judge-clean-first", B, cp_thresh_tab(sp[jc][yp[jc] == 0], tab95), sp, yp)
                    for q in a.qs:
                        H = rank < int(round(q * M))
                        iH, iL = np.where(H)[0], np.where(~H)[0]
                        for f in a.fs:
                            mH = int(round(f * B)) if boot else min(int(round(f * B)), len(iH))
                            mL = B - mH if boot else min(B - mH, len(iL))
                            lH, lL = draw(iH, mH), draw(iL, mL)
                            t = strat_thresh(sp, H, lH, lL, yp[lH], yp[lL], tabT, tabL, tabC, M)
                            add(f"strat(q={q},f={f})", B, t, sp, yp)
                            t = strat_cond_thresh(sp, H, lH, lL, yp[lH], yp[lL], tabT, tabL, tabC, M)
                            add(f"strat-cond(q={q},f={f})", B, t, sp, yp)
                            if q == 0.2 and f in (0.5, 0.67):
                                labs = np.concatenate([lH, lL])
                                w = np.concatenate([np.full(len(lH), H.sum() / max(len(lH), 1)),
                                                    np.full(len(lL), (~H).sum() / max(len(lL), 1))])
                                add(f"ipw-kish(q={q},f={f})", B, ipw_thresh(sp, labs, w, yp[labs]), sp, yp)
            row[frame][sname] = {nm: {str(B): {"fpr": float(np.mean(r["fa"])),
                                        "fpr_over_10pct": float(np.mean(np.array(r["fa"]) > 0.10)),
                                        "fpr_over_12pct": float(np.mean(np.array(r["fa"]) > 0.12)),
                                        "recall": float(np.mean(r["rc"])), "feasible": float(np.mean(r["feas"]))}
                               for B, r in byB.items()} for nm, byB in res.items()}
        out[d] = row
        fr = a.frames[0]
        u, st = row[fr]["stack8b"]["uniform"]["100"], row[fr]["stack8b"]["strat(q=0.15,f=0.5)"]["50"]
        print("b", d, "uniform100", {k: round(v, 3) for k, v in u.items()},
              "strat50", {k: round(v, 3) for k, v in st.items()}, flush=True)
    return out


def summarise(per, keys_fn):
    groups = {"all": list(per)}
    for d, r in per.items():
        groups.setdefault(r["format"], []).append(d)
    return {g: keys_fn([per[d] for d in mem]) | {"domains": len(mem)} for g, mem in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/omni/expert_scores_q8b.npz")
    ap.add_argument("--judge", default="results/omni/judge_probs.npz")
    ap.add_argument("--nli", default="/home/user/omni/nli.npz")
    ap.add_argument("--draws-a", type=int, default=40)
    ap.add_argument("--draws-b", type=int, default=200)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100])
    ap.add_argument("--budgets-b", type=int, nargs="+", default=[30, 40, 50, 60, 80, 100])
    ap.add_argument("--prevalences", nargs="+", default=["0.2", "natural"])
    ap.add_argument("--qs", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--fs", type=float, nargs="+", default=[0.5, 0.67, 0.8, 1.0])
    ap.add_argument("--frames", nargs="+", default=["bootstrap", "finite"])
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--report", default="results/omni/novel_active_labels.json")
    a = ap.parse_args()
    a.prevalences = [p if p == "natural" else float(p) for p in a.prevalences]
    D = load(a)
    t0 = time.time()
    rep = {"alpha": ALPHA, "confidence": CONF, "draws_a": a.draws_a, "draws_b": a.draws_b,
           "deltas": {"tail": D_TAIL, "clean_L": D_CL, "clean_H": D_CH}}
    rep["guarantee"] = part_b(D, a, np.random.default_rng(0))
    print(f"b done {time.time() - t0:.0f}s", flush=True)

    def sum_b(rows):
        o = {}
        for fr in a.frames:
            for sname in ("stack8b", "stack+judges"):
                for nm, byB in rows[0][fr][sname].items():
                    for B in byB:
                        o.setdefault(fr, {}).setdefault(sname, {}).setdefault(nm, {})[B] = {
                            k: float(np.mean([r[fr][sname][nm][B][k] for r in rows])) for k in byB[B]}
        return o
    rep["guarantee_summary"] = summarise(rep["guarantee"], sum_b)
    if not a.skip_a:
        rep["adaptation"] = part_a(D, a, np.random.default_rng(0))
        print(f"a done {time.time() - t0:.0f}s", flush=True)

        def sum_a(rows):
            o = {}
            for p in a.prevalences:
                for s in ["zero_label"] + STRATS:
                    for b in map(str, a.budgets):
                        v = [r[str(p)][s][b] for r in rows if r[str(p)][s][b] is not None]
                        o.setdefault(str(p), {}).setdefault(s, {})[b] = float(np.mean(v)) if v else None
                    o[str(p)].setdefault("domains_per_budget", {b: len([r for r in rows if r[str(p)]["uniform"][b]
                                                                        is not None]) for b in map(str, a.budgets)})
            return o
        rep["adaptation_summary"] = summarise(rep["adaptation"], sum_a)
    json.dump(rep, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
