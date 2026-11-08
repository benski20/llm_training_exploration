"""Two novel signals for the omni verifier, on stored features only (no model is loaded).

(1) MULTI-READER CONSENSUS. Two different frozen readers (Qwen3-1.7B, Qwen3-0.6B) read the same reply. Does their
    *disagreement* carry signal about the reply being wrong?
      d_reply    |z(reply head 0.6B) - z(reply head 1.7B)|   (both heads LODO; z within the domain, unlabelled)
      d_worst    the same for the worst unit
      d_judge    the same for the readers' own judge log-odds
      d_units    mean |z(unit score 0.6B) - z(unit score 1.7B)| over the reply's units
      cca        canonical correlation analysis between the readers' reply representations (PCA 64 each),
                 fitted without labels on the other domains; per reply the mean over 32 components of
                 (u_k - v_k)^2 / 2(1 - rho_k): how far the two readers' views sit from their usual agreement
      ridge      residual of predicting the 0.6B representation from the 1.7B one (ridge, other domains)
      cca_lr     a LODO logistic on the 32 per-component discrepancies
      concat     one reply head on both readers' features vs the 1.7B head alone
(2) STEP / SENTENCE DYNAMICS. Features of the unit-score sequence (unit head trained LODO as in
    experiments/omni_final.py): max jump between consecutive units, relative position of the max, variance, slope,
    max - median, count and fraction above 0 (the balanced probe's decision boundary), mean, log #units; each alone
    (a-priori sign) and a LODO logistic on them. Localisation on ProcessBench: argmax vs "first unit where the score
    jumps" vs "first unit above a threshold", the threshold chosen leave-one-PB-domain-out.

PRE-STATED RULES (fixed before the first run; the bootstrap is 1,000 resamples of replies within each domain,
stratified by label, paired across methods):
  R1 signal   a disagreement / dynamics score carries signal if its mean within-domain AUROC (a-priori sign: more
              disagreement / bigger jump = wrong) has a 95% CI lower bound > 0.50.
  R2 useful   an idea is adopted only if it raises the mean AUROC over 21 domains over its matched baseline by
              >= +0.010 with the 95% CI of the paired difference above 0. Matched baselines:
                zero labels  C0 = z(reply 1.7B) + z(worst unit 1.7B) + z(judge 1.7B)
                stacker      S0 = LODO logistic on those three z-scores (nested: training-domain scores come from
                             heads that never saw the held-out domain nor the row's own domain)
              A format-level claim needs the same within that format.
  R3 local.   "first jump" beats argmax if the hit rate on the first wrong ProcessBench step is higher with the 95%
              CI of the paired difference (bootstrap over wrong replies within PB domains) above 0.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/novel_consensus_dynamics.py \
        --q17 /home/user/omni/q17 --q06 /home/user/omni/q06 --corpus /home/user/omni/corpus.jsonl \
        --report results/omni/novel_consensus_dynamics.json
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
from omni_eval import LogReg, auroc, balanced_weights, pca_fit, standardise  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def zdom(v, dom):
    """z-score within each domain (unlabelled statistics); nan -> 0."""
    v = np.asarray(v, float)
    out = np.zeros_like(v)
    for d in np.unique(dom):
        m = dom == d
        x = v[m]
        mu, sd = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-9
        out[m] = np.nan_to_num((x - mu) / sd)
    return out


# ----------------------------------------------------------------- nested leave-domain-out heads

def nested_heads(X, y, dom, doms, l2, train_mask=None):
    """S[k, i]: score of row i from a head trained on rows whose domain is neither doms[k] nor dom[i]
    (rows with train_mask False never train). S[k, dom == doms[k]] is the plain LODO score.
    One fit per distinct training set (cached)."""
    tm = np.ones(len(y), bool) if train_mask is None else train_mask
    labelled = set(np.unique(dom[tm]).tolist())
    cache = {}

    def model(excl):
        key = frozenset(e for e in excl if e in labelled)
        if key not in cache:
            tr = tm & ~np.isin(dom, list(key))
            cache[key] = LogReg(l2).fit(X[tr], y[tr].astype(float), balanced_weights(y[tr].astype(float)))
        return cache[key]

    S = np.zeros((len(doms), len(y)))
    for k, d in enumerate(doms):
        for e in doms:
            rows = dom == e
            S[k, rows] = model({d, e}).score(X[rows])
    log(f"  nested heads: {len(cache)} fits")
    return S


def reply_X(F, pca):
    n, nl = len(F["ids"]), F["r_mean"].shape[1]
    mid = [nl // 2 - 1, nl // 2, nl // 2 + 1]
    H = np.concatenate([F["r_mean"][:, mid].reshape(n, -1), F["j_state"][:, mid].reshape(n, -1)], 1).astype(np.float32)
    H = standardise(H)
    mu, V = pca_fit(H, pca)
    P = (H - mu) @ V.T
    return P, np.concatenate([P, F["r_sc"], F["j_logit"][:, None]], 1)


def unit_X(F, pca):
    nl = F["r_mean"].shape[1]
    mid = [nl // 2 - 1, nl // 2, nl // 2 + 1]
    U = F["u_mean"][:, mid].reshape(len(F["u_row"]), -1).astype(np.float32)
    U = standardise(U)
    umu, UV = pca_fit(U[np.random.default_rng(0).permutation(len(U))[:20000]], pca)
    return np.concatenate([(U - umu) @ UV.T, F["u_sc"]], 1)


# ----------------------------------------------------------------- unit-sequence dynamics

DYN = ["worst", "max_jump", "pos_max", "var", "slope", "max_minus_median", "count_above0", "frac_above0", "mean",
       "log_units"]
DYN_SIGN = {"worst": 1, "max_jump": 1, "pos_max": 1, "var": 1, "slope": 1, "max_minus_median": 1, "count_above0": 1,
            "frac_above0": 1, "mean": 1, "log_units": 1}


class Seq:
    """reply -> its units in order."""

    def __init__(self, u_rowpos, u_idx, n):
        order = np.lexsort((u_idx, u_rowpos))
        self.order = order
        rp = u_rowpos[order]
        self.starts = np.searchsorted(rp, np.arange(n))
        self.ends = np.searchsorted(rp, np.arange(n), side="right")

    def seqs(self, su):
        s = su[self.order]
        return [s[a:b] for a, b in zip(self.starts, self.ends)]


def dyn_features(seqs):
    out = np.full((len(seqs), len(DYN)), np.nan)
    for i, s in enumerate(seqs):
        m = len(s)
        if m == 0:
            continue
        j = int(np.argmax(s))
        if m > 1:
            pos = np.arange(m) / (m - 1)
            slope = np.polyfit(pos, s, 1)[0]
            jump = np.max(np.diff(s))
            pmax = j / (m - 1)
        else:
            slope, jump, pmax = 0.0, 0.0, 1.0
        out[i] = [s.max(), jump, pmax, s.var(), slope, s.max() - np.median(s), (s > 0).sum(), (s > 0).mean(), s.mean(),
                  np.log(m)]
    return out


# ----------------------------------------------------------------- CCA / ridge between readers

def cca_disc(A, B, tr, te, k, reg=1e-3):
    ma, mb = A[tr].mean(0), B[tr].mean(0)
    Xa, Xb = A[tr] - ma, B[tr] - mb
    n = len(Xa)
    Caa, Cbb, Cab = Xa.T @ Xa / n, Xb.T @ Xb / n, Xa.T @ Xb / n
    Caa += reg * np.trace(Caa) / len(Caa) * np.eye(len(Caa))
    Cbb += reg * np.trace(Cbb) / len(Cbb) * np.eye(len(Cbb))

    def isqrt(C):
        w, V = np.linalg.eigh(C)
        return V @ np.diag(w ** -0.5) @ V.T
    Wa, Wb = isqrt(Caa), isqrt(Cbb)
    U, S, Vt = np.linalg.svd(Wa @ Cab @ Wb)
    Pa, Pb, rho = Wa @ U[:, :k], Wb @ Vt[:k].T, np.clip(S[:k], 0, 0.999)
    u, v = (A[te] - ma) @ Pa, (B[te] - mb) @ Pb
    return (u - v) ** 2 / (2 * (1 - rho)), rho


def ridge_resid(A, B, tr, te, lam):
    ma, mb = A[tr].mean(0), B[tr].mean(0)
    Xa, Xb = A[tr] - ma, B[tr] - mb
    W = np.linalg.solve(Xa.T @ Xa + lam * len(Xa) * np.eye(Xa.shape[1]), Xa.T @ Xb)
    sd = (Xb - Xa @ W).std(0) + 1e-9
    R = ((B[te] - mb) - (A[te] - ma) @ W) / sd
    return (R ** 2).mean(1)


# ----------------------------------------------------------------- stacking (nested LODO)

def stack(cols, y, dom, doms, l2):
    """cols: list of arrays, each (n,) static, (n_dom, n) nested scores, or (n_dom, n, f) nested features.
    Held-out domain k: features from slice k, z-scored per domain, logistic on the other domains."""
    out = np.zeros(len(y))
    for k, d in enumerate(doms):
        parts = []
        for c in cols:
            c = c if c.ndim == 1 or c.shape[0] != len(doms) else c[k]
            parts.append(c[:, None] if c.ndim == 1 else c)
        Z = zdom(np.concatenate(parts, 1), dom)
        te = dom == d
        m = LogReg(l2).fit(Z[~te], y[~te].astype(float), balanced_weights(y[~te].astype(float)))
        out[te] = m.score(Z[te])
    return out


# ----------------------------------------------------------------- bootstrap

def boot_index(y, dom, doms, B, seed=0):
    rng = np.random.default_rng(seed)
    idx = {d: (np.where((dom == d) & (y == 1))[0], np.where((dom == d) & (y == 0))[0]) for d in doms}
    for _ in range(B):
        yield {d: np.concatenate([rng.choice(p, len(p)), rng.choice(q, len(q))]) for d, (p, q) in idx.items()}


def evaluate(methods, y, dom, fmt_of, doms, B, comparisons):
    """methods: {name: score (n,)}. Returns per-domain AUROC, means per format with CI, and paired differences."""
    groups = {"all": list(doms)}
    for f in sorted(set(fmt_of.values())):
        groups[f] = [d for d in doms if fmt_of[d] == f]
    names = list(methods)
    M = np.stack([methods[k] for k in names])
    pt = np.array([[auroc(M[j, dom == d], y[dom == d]) for d in doms] for j in range(len(names))])
    bs = []
    for samp in boot_index(y, dom, doms, B):
        bs.append([[auroc(M[j, samp[d]], y[samp[d]]) for d in doms] for j in range(len(names))])
    bs = np.array(bs)                                            # B x methods x domains
    di = {d: i for i, d in enumerate(doms)}
    res = {"per_domain": {nm: {d: round(float(pt[j, di[d]]), 4) for d in doms} for j, nm in enumerate(names)},
           "mean": {}, "diff": {}}
    for g, mem in groups.items():
        cols = [di[d] for d in mem]
        res["mean"][g] = {}
        for j, nm in enumerate(names):
            b = bs[:, j, cols].mean(1)
            res["mean"][g][nm] = [round(float(pt[j, cols].mean()), 4), round(float(np.percentile(b, 2.5)), 4),
                                  round(float(np.percentile(b, 97.5)), 4)]
        res["diff"][g] = {}
        for a, base in comparisons:
            ja, jb = names.index(a), names.index(base)
            b = (bs[:, ja, cols] - bs[:, jb, cols]).mean(1)
            res["diff"][g][f"{a} - {base}"] = [round(float((pt[ja, cols] - pt[jb, cols]).mean()), 4),
                                              round(float(np.percentile(b, 2.5)), 4),
                                              round(float(np.percentile(b, 97.5)), 4)]
    return res


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q17", required=True)
    ap.add_argument("--q06", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--cca-pca", type=int, default=64)
    ap.add_argument("--cca-k", type=int, default=32)
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    rows = {}
    for line in open(a.corpus):
        r = json.loads(line)
        rows[r["id"]] = r

    R = {}
    for name, path in (("17", a.q17), ("06", a.q06)):
        F = load_features(path)
        if name == "17":
            ids = F["ids"]
            u_row, u_idx, u_lab = F["u_row"], F["u_idx"], F["u_lab"]
        else:
            assert (F["ids"] == ids).all() and (F["u_row"] == u_row).all() and (F["u_idx"] == u_idx).all()
        P, X = reply_X(F, a.pca)
        UX = unit_X(F, a.pca)
        R[name] = {"P": P, "X": X, "UX": UX, "judge": -F["j_logit"].astype(float)}
        del F
        log(f"reader {name} features built")

    n = len(ids)
    dom = np.array([rows[int(i)]["domain"] for i in ids])
    fmt = np.array([rows[int(i)]["format"] for i in ids])
    y = np.array([rows[int(i)]["label"] for i in ids])
    doms = sorted(np.unique(dom).tolist())
    fmt_of = {d: fmt[dom == d][0] for d in doms}
    di = {d: k for k, d in enumerate(doms)}
    pos = {int(i): k for k, i in enumerate(ids)}
    u_rowpos = np.array([pos[int(r_)] for r_ in u_row])
    udom = dom[u_rowpos]
    seq = Seq(u_rowpos, u_idx, n)
    k_own = np.array([di[d] for d in dom])                       # each row's own-domain slice
    own = lambda S: S[k_own, np.arange(S.shape[1])]            # noqa: E731  plain LODO score per row

    # ---- heads (nested LODO) for both readers
    for nm in ("17", "06"):
        r = R[nm]
        r["Sreply"] = nested_heads(standardise(r["X"], dom), y, dom, doms, a.l2)
        r["Sunit"] = nested_heads(standardise(r["UX"], udom), u_lab.astype(float), udom, doms, a.l2,
                                  train_mask=u_lab >= 0)
        r["Sunit_own"] = r["Sunit"][np.array([di[d] for d in udom]), np.arange(len(udom))]
        r["Sworst"] = np.stack([dyn_features(seq.seqs(r["Sunit"][k]))[:, 0] for k in range(len(doms))])
        log(f"reader {nm} heads done")
    Xcat = np.concatenate([R["17"]["X"], R["06"]["X"]], 1)
    S_cat = nested_heads(standardise(Xcat, dom), y, dom, doms, a.l2)
    log("concat head done")

    # ---- dynamics features (1.7B unit head), nested
    DynS = np.stack([dyn_features(seq.seqs(R["17"]["Sunit"][k])) for k in range(len(doms))])   # n_dom x n x f
    dyn_own = DynS[k_own, np.arange(n)]
    log("dynamics done")

    # ---- disagreement
    z = {f"{c}{nm}": zdom(v, dom) for nm in ("17", "06") for c, v in
         (("reply", own(R[nm]["Sreply"])), ("worst", own(R[nm]["Sworst"])), ("judge", R[nm]["judge"]))}
    d_reply, d_worst, d_judge = (np.abs(z[f"{c}06"] - z[f"{c}17"]) for c in ("reply", "worst", "judge"))
    zu = {nm: zdom(R[nm]["Sunit_own"], udom) for nm in ("17", "06")}
    du = np.abs(zu["06"] - zu["17"])
    d_units = np.array([s.mean() if len(s) else np.nan for s in seq.seqs(du)])
    # nested disagreement (for the stacker): slice k from heads that never saw domain k
    D_nested = np.stack([np.stack([np.abs(zdom(R["06"][S][k], dom) - zdom(R["17"][S][k], dom))
                                   for S in ("Sreply", "Sworst")] + [d_judge], 1) for k in range(len(doms))])

    # CCA / ridge, each row's discrepancy from a fit on the other domains (no labels used)
    A = standardise(R["17"]["P"][:, :a.cca_pca], dom)
    Bm = standardise(R["06"]["P"][:, :a.cca_pca], dom)
    cca_comp = np.zeros((n, a.cca_k))
    rid = np.zeros(n)
    rhos = {}
    for d in doms:
        te = dom == d
        cca_comp[te], rho = cca_disc(A, Bm, ~te, te, a.cca_k)
        rid[te] = ridge_resid(A, Bm, ~te, te, 1e-2)
        rhos[d] = [round(float(x), 3) for x in rho[:5]]
    cca = cca_comp.mean(1)
    log("CCA done")

    # ---- scores
    m = {}
    m["reply17"], m["worst17"], m["judge17"] = own(R["17"]["Sreply"]), own(R["17"]["Sworst"]), R["17"]["judge"]
    m["reply06"], m["worst06"], m["judge06"] = own(R["06"]["Sreply"]), own(R["06"]["Sworst"]), R["06"]["judge"]
    m["C0"] = z["reply17"] + z["worst17"] + z["judge17"]
    # (1) consensus
    m["d_reply"], m["d_worst"], m["d_judge"], m["d_units"] = d_reply, d_worst, d_judge, d_units
    m["cca"], m["ridge"] = cca, rid
    m["cca_lr"] = stack([cca_comp], y, dom, doms, a.l2)
    m["reply_cat"] = own(S_cat)
    m["C0+d_reply"] = m["C0"] + zdom(d_reply, dom)
    m["C0+cca"] = m["C0"] + zdom(cca, dom)
    m["C0+d_all"] = m["C0"] + zdom(d_reply + d_worst + d_judge, dom)
    m["C0_cat"] = zdom(m["reply_cat"], dom) + z["worst17"] + z["judge17"]
    m["C_both"] = m["C0"] + z["reply06"] + z["worst06"] + z["judge06"]
    s17 = [R["17"]["Sreply"], R["17"]["Sworst"], m["judge17"]]
    s06 = [R["06"]["Sreply"], R["06"]["Sworst"], m["judge06"]]
    m["S0"] = stack(s17, y, dom, doms, a.l2)
    m["S0+06"] = stack(s17 + s06, y, dom, doms, a.l2)
    m["S0+06+disagree"] = stack(s17 + s06 + [D_nested, cca], y, dom, doms, a.l2)
    m["S0+disagree"] = stack(s17 + [D_nested, cca], y, dom, doms, a.l2)
    # (2) dynamics
    for j, f in enumerate(DYN):
        m[f"dyn_{f}"] = DYN_SIGN[f] * dyn_own[:, j]
    m["dyn_lr"] = stack([DynS], y, dom, doms, a.l2)
    m["C0+dyn_lr"] = m["C0"] + zdom(m["dyn_lr"], dom)
    m["C0_dyn_for_worst"] = z["reply17"] + zdom(m["dyn_lr"], dom) + z["judge17"]
    m["C0+max_jump"] = m["C0"] + zdom(dyn_own[:, 1], dom)
    m["S0+dyn"] = stack(s17 + [DynS], y, dom, doms, a.l2)
    log("scores done; bootstrapping")

    comps = [("d_reply", "reply17"), ("reply_cat", "reply17"), ("C0+d_reply", "C0"), ("C0+cca", "C0"),
             ("C0+d_all", "C0"), ("C0_cat", "C0"), ("C_both", "C0"), ("S0", "C0"), ("S0+06", "S0"),
             ("S0+06+disagree", "S0+06"), ("S0+disagree", "S0"),
             ("dyn_lr", "worst17"), ("dyn_max_jump", "worst17"), ("dyn_max_minus_median", "worst17"),
             ("C0+dyn_lr", "C0"), ("C0_dyn_for_worst", "C0"), ("C0+max_jump", "C0"), ("S0+dyn", "S0")]
    ev = evaluate(m, y, dom, fmt_of, doms, a.boot, comps)
    log("bootstrap done")

    # ---- (2b) localisation of the first wrong ProcessBench step
    su = zdom(R["17"]["Sunit_own"], udom)                       # unit scores z-scored per domain (unlabelled)
    labs = seq.seqs(u_lab.astype(float))
    sseq = seq.seqs(su)
    pb = [d for d in doms if d.startswith("pb_")]
    cases = {d: [] for d in pb}
    for i in range(n):
        if dom[i] in cases and y[i] == 1 and (labs[i] == 1).sum() == 1:
            cases[dom[i]].append((sseq[i], int(np.argmax(labs[i] == 1))))
    grid = np.round(np.arange(0.0, 3.01, 0.25), 2)

    def first_jump(s, t):
        prev = np.concatenate([[0.0], s[:-1]])                  # before step 1: the domain's average unit (z = 0)
        hit = np.where(s - prev > t)[0]
        return int(hit[0]) if len(hit) else int(np.argmax(s))

    def first_cross(s, t):
        hit = np.where(s > t)[0]
        return int(hit[0]) if len(hit) else int(np.argmax(s))

    rules = {"argmax": lambda s, t: int(np.argmax(s)), "first_jump": first_jump, "first_cross": first_cross}

    def hits(d, rule, t, tol=0):
        return np.array([abs(rules[rule](s, t) - g) <= tol for s, g in cases[d]], float)

    loc = {"n_wrong": {d: len(cases[d]) for d in pb},
           "chance_top1": {d: float(np.mean([1 / len(s) for s, _ in cases[d]])) for d in pb},
           "mean_steps": {d: float(np.mean([len(s) for s, _ in cases[d]])) for d in pb},
           "first_wrong_rel_pos": {d: float(np.mean([g / max(len(s) - 1, 1) for s, g in cases[d]])) for d in pb},
           "tau": {}, "hit": {}, "hit_pm1": {}}
    pred = {}
    for rule in ("first_jump", "first_cross"):
        loc["tau"][rule] = {}
        for d in pb:
            others = [e for e in pb if e != d]
            acc = [np.mean([hits(e, rule, t).mean() for e in others]) for t in grid]
            loc["tau"][rule][d] = float(grid[int(np.argmax(acc))])
    for rule in rules:
        pred[rule] = {d: hits(d, rule, loc["tau"].get(rule, {}).get(d, 0.0)) for d in pb}
        pred[rule + "_pm1"] = {d: hits(d, rule, loc["tau"].get(rule, {}).get(d, 0.0), tol=1) for d in pb}
    for key, v in pred.items():
        loc["hit"][key] = {d: round(float(v[d].mean()), 4) for d in pb} | {"mean": round(float(np.mean([v[d].mean() for d in pb])), 4)}
    rng = np.random.default_rng(0)
    bsl = {k: [] for k in pred}
    for _ in range(a.boot):
        samp = {d: rng.integers(0, len(cases[d]), len(cases[d])) for d in pb}
        for k in pred:
            bsl[k].append(np.mean([pred[k][d][samp[d]].mean() for d in pb]))
    bsl = {k: np.array(v) for k, v in bsl.items()}
    loc["ci"] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in bsl.items()}
    loc["diff"] = {}
    for r_ in ("first_jump", "first_cross"):
        for suf in ("", "_pm1"):
            dd = bsl[r_ + suf] - bsl["argmax" + suf]
            loc["diff"][f"{r_}{suf} - argmax{suf}"] = [round(float(loc["hit"][r_ + suf]["mean"] - loc["hit"]["argmax" + suf]["mean"]), 4),
                                                      round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]

    # ---- verdicts by the pre-stated rules
    verdict = {"R1_signal": {}, "R2_useful": {}}
    for k in ["d_reply", "d_worst", "d_judge", "d_units", "cca", "ridge", "cca_lr"] + [f"dyn_{f}" for f in DYN] + ["dyn_lr"]:
        verdict["R1_signal"][k] = ev["mean"]["all"][k][1] > 0.5
    for c, v in ev["diff"]["all"].items():
        verdict["R2_useful"][c] = (v[0] >= 0.01) and (v[1] > 0)
    verdict["R2_useful_by_format"] = {g: {c: (v[0] >= 0.01 and v[1] > 0) for c, v in ev["diff"][g].items()}
                                      for g in ev["diff"] if g != "all"}
    verdict["R3_localisation"] = {c: v[1] > 0 for c, v in loc["diff"].items()}

    rep = {"readers": ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"], "n": int(n), "domains": doms,
           "formats": fmt_of, "args": vars(a), "cca_top_rho": rhos, "eval": ev, "localisation": loc,
           "verdict": verdict, "seconds": round(time.time() - T0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)

    groups = list(ev["mean"])
    print(f"\n{'method':22s}" + "".join(f"{g:>22s}" for g in groups))
    for k in m:
        print(f"{k:22s}" + "".join(f"  {ev['mean'][g][k][0]:.3f} [{ev['mean'][g][k][1]:.3f},{ev['mean'][g][k][2]:.3f}]"
                                   for g in groups))
    print("\ndifferences")
    for c in comps:
        key = f"{c[0]} - {c[1]}"
        print(f"{key:30s}" + "".join(f"  {ev['diff'][g][key][0]:+.3f} [{ev['diff'][g][key][1]:+.3f},{ev['diff'][g][key][2]:+.3f}]"
                                     for g in groups))
    print("\nlocalisation", json.dumps({k: loc[k] for k in ("hit", "ci", "diff", "tau", "chance_top1")}, indent=1))
    print(json.dumps(verdict, indent=1))
    log("done")


if __name__ == "__main__":
    main()
