"""Verdict fragility: use the 1.7B reader's internals CAUSALLY.

The reader (Qwen3-1.7B, vt/omni.py Reader) appends the judge question after the reply and reads
j = log P(Yes) - log P(No) at the last position. Here we intervene on the residual stream at that position.

DIRECTIONS (all "wrong"-pointing, unit norm, in the hidden space after decoder block L-1 = hidden_states[L]):
  dm20  difference of class means of the stored judge state (j_state, /home/user/omni/q17) at layer 20,
        LEAVE-ONE-DOMAIN-OUT: for a reply of domain d the direction is mean over the 20 OTHER domains of
        (mean wrong - mean correct). The held-out domain's labels and states are never used.
  dm15  the same at layer 15.
  head  (reference only, LEAKY: the shipped head saw every domain) the shipped universal head's weights mapped
        back through its standardisation and PCA to the judge-state block at layer 20.
  Scale: alpha is in units of the sd of the projection of the judge state on the direction (within-domain,
  training domains only).

INTERVENTIONS (forward hook on the decoder block, last position only; the prefix is KV-cached so each
intervention reruns one token):
  steer   j(alpha) for alpha in {+-1, +-2, +-4, +-8} (dm20) and {+-2, +-4} (dm15, head)
  patch   counterfactual: the same reply read WITHOUT the documents / question (user turn emptied, same system
          prompt); its judge-position state at layer L replaces the real one; j_patchL. Also the counterfactual
          run's own verdict j_cf.

SIGNALS (a-priori sign: larger = wrong; stated before the first run)
  judge0      -j(0)                                    the reader's own verdict (baseline, not new)
  flip        -alpha*   alpha* = the dm20 push at which the verdict crosses 0 (interpolated on the grid,
                         extrapolated with the outer slope, clipped to +-20)
  fragility   -|alpha*|  a verdict that flips easily = wrong (H1)
  sens20      (j(-2) - j(+2)) / 4                      how strongly the verdict follows the direction (H1')
  asym20 / asym20_4  [j(0) - j(+a)] - [j(-a) - j(0)], a = 2 / 4: easier to push toward "wrong" than toward
                     "right" = wrong (H2)
  sens15, asym15     the same with dm15
  cf_patch20, cf_patch15  j_patchL - j(0): removing the evidence makes the verdict MORE favourable = the context
                     argued against the reply = wrong (H3)
  cf_full     j_cf - j(0)
  head_sens, head_asym  reference only (leaky direction)

PRE-STATED RULES (bootstrap: 1,000 resamples of replies within each domain, stratified by label, paired):
  R1 signal   a signal carries signal if its mean within-domain AUROC (a-priori sign) has 95% CI lower bound > 0.50.
  R2 new      a causal signal is NEW information only if adding it to the reader's own verdict,
              z(judge0) + z(signal), beats z(judge0) by >= +0.010 mean AUROC with the paired 95% CI above 0
              (flip / fragility are largely functions of j(0); this separates them from it).
  R3 adopt    adopted only if it raises the existing 1.7B stack C0 = z(reply head, LODO) + z(judge0) by >= +0.010
              mean AUROC over 21 domains, paired 95% CI above 0, both for the zero-label sum (C0 + z(signal)) and for
              a LODO logistic stacker on the z-scores. A format-level claim needs the same within that format.
  z-scores are within domain from unlabelled statistics.

    python experiments/novel_verdict_fragility.py run --n-per-domain 40 --threads 2 --out <cache.jsonl>
    python experiments/novel_verdict_fragility.py analyze --cache <cache.jsonl> --report results/omni/novel_verdict_fragility.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

T0 = time.time()
CORPUS = "/home/user/omni/corpus.jsonl"
Q17 = "/home/user/omni/q17"
HEAD = os.path.join(ROOT, "heads/omni_qwen3_1p7b.npz")
EXPERT = os.path.join(ROOT, "results/omni/expert_scores_q17.npz")
STORED_LAYERS = [7, 11, 15, 20, 24, 28]
A20 = [-8, -4, -2, -1, 1, 2, 4, 8]
A_OTHER = [-4, -2, 2, 4]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ----------------------------------------------------------------- sample and directions

def sample(rows, n_per_domain, seed):
    rng = np.random.default_rng(seed)
    out = []
    for d in sorted({r["domain"] for r in rows}):
        for lab in (0, 1):
            pool = [r for r in rows if r["domain"] == d and r["label"] == lab]
            k = min(n_per_domain // 2, len(pool))
            out += [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]
    return out


def stored_jstate(layers):
    ids, J = [], []
    for p in sorted(glob.glob(os.path.join(Q17, "shard_*.npz"))):
        z = np.load(p)
        ids.append(z["ids"])
        J.append(z["j_state"][:, [STORED_LAYERS.index(L) for L in layers]].astype(np.float32))
    return np.concatenate(ids), np.concatenate(J)


def lodo_directions(rows_by_id, layers):
    """{(heldout_domain, L): (unit dir, sd of projection)} from training domains only."""
    ids, J = stored_jstate(layers)
    dom = np.array([rows_by_id[i]["domain"] for i in ids])
    y = np.array([rows_by_id[i]["label"] for i in ids])
    doms = sorted(set(dom.tolist()))
    diffs, cent = {}, {}
    for k, L in enumerate(layers):
        for d in doms:
            m = dom == d
            diffs[(d, L)] = J[m & (y == 1), k].mean(0) - J[m & (y == 0), k].mean(0)
            cent[(d, L)] = J[m, k] - J[m, k].mean(0)
    out = {}
    for k, L in enumerate(layers):
        for h in doms:
            tr = [d for d in doms if d != h]
            v = np.mean([diffs[(d, L)] for d in tr], 0)
            v /= np.linalg.norm(v)
            sd = float(np.concatenate([cent[(d, L)] @ v for d in tr]).std())
            out[(h, L)] = (v.astype(np.float32), sd)
    return out


def head_direction(L=20):
    """LEAKY reference: the shipped head's weights mapped back to the judge-state block at layer L."""
    z = np.load(HEAD, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    k = z["pca_V"].shape[0]
    w = z["w"][:k] / z["x_sd"][:k]
    g = (z["pca_V"].astype(np.float32).T @ w) / z["h_sd"]
    nl = len(meta["layer_idx"])
    j = meta["parts"].index("judge") * nl + meta["layer_idx"].index(STORED_LAYERS.index(L))
    v = g[j * 2048:(j + 1) * 2048].astype(np.float32)
    v /= np.linalg.norm(v)
    ids, J = stored_jstate([L])
    return v, float((J[:, 0] @ v).std())


# ----------------------------------------------------------------- the causal runs

class Causal:
    def __init__(self, threads, max_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vt.omni import judge_suffix
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).eval()
        self.max_tokens = max_tokens
        self.suffix = self.tok(judge_suffix(self.tok), add_special_tokens=False)["input_ids"]
        tid = lambda w: self.tok(w, add_special_tokens=False)["input_ids"][0]
        self.yes = [tid(w) for w in ("Yes", "yes", " Yes")]
        self.no = [tid(w) for w in ("No", "no", " No")]
        self.blocks = self.model.model.layers
        self.Wyn = self.model.lm_head.weight[self.yes + self.no].float()
        self.edit = {}                                           # L -> fn(last-position state) -> new state

        def mk(L):
            def hook(mod, inp, out):
                if L not in self.edit:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[:, -1] = self.edit[L](h[:, -1])
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return hook
        for L in (15, 20):
            self.blocks[L - 1].register_forward_hook(mk(L))   # output of block L-1 = hidden_states[L]

    def jl(self, o):
        """log P(Yes) - log P(No) at the last position, read in fp32 from the final normed state (bf16 logits are
        quantised to ~0.1-0.25 at these magnitudes; the softmax normaliser cancels)."""
        h = o.hidden_states[-1][0, -1].float()
        lg = self.Wyn @ h
        k = len(self.yes)
        return float(self.torch.logsumexp(lg[:k], 0) - self.torch.logsumexp(lg[k:], 0))

    def ids(self, item):
        from vt.omni import encode
        return encode(self.tok, item, self.max_tokens, self.suffix)[0]

    def full(self, ids, layers=(15, 20)):
        t = self.torch
        with t.no_grad():
            o = self.model(input_ids=t.tensor([ids]), output_hidden_states=True, logits_to_keep=1)
        return self.jl(o), {L: o.hidden_states[L][0, -1].float().clone() for L in layers}

    def run(self, row, dirs, alphas):
        """dirs: {name: (L, unit vector, sd)}; alphas: {name: [alpha...]}."""
        t = self.torch
        from transformers import DynamicCache
        ids = self.ids(row)
        n = len(ids)
        cache = DynamicCache()
        with t.no_grad():
            self.model(input_ids=t.tensor([ids[:-1]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        last = t.tensor([[ids[-1]]])

        def one(edit=None):
            self.edit = edit or {}
            with t.no_grad():
                o = self.model(input_ids=last, past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache.crop(-1)
            self.edit = {}
            return self.jl(o), o

        j0, o0 = one()
        res = {"id": row["id"], "n_tokens": n, "j0": j0, "proj": {}, "steer": {}}
        for name, (L, v, sd) in dirs.items():
            vt_ = t.tensor(v)
            res["proj"][name] = float(o0.hidden_states[L][0, -1].float() @ vt_) / sd
            res["steer"][name] = {}
            for a in alphas[name]:
                delta = (a * sd * vt_).to(t.bfloat16)
                res["steer"][name][str(a)] = one({L: lambda h, d=delta: h + d})[0]
        # counterfactual: no documents / question
        cf = {"messages": [dict(row["messages"][0]), {"role": "user", "content": ""}], "response": row["response"]}
        res["j_cf"], hcf = self.full(self.ids(cf))
        for L in (15, 20):
            s = hcf[L].to(t.bfloat16)
            res[f"j_patch{L}"] = one({L: lambda h, s=s: s.expand_as(h)})[0]
        return res


def cmd_run(a):
    rows = [json.loads(l) for l in open(CORPUS)]
    by_id = {r["id"]: r for r in rows}
    smp = sample(rows, a.n_per_domain, a.seed)
    if a.limit:
        smp = [smp[i] for i in np.random.default_rng(1).permutation(len(smp))[:a.limit]]
    log(f"{len(smp)} replies")
    D = lodo_directions(by_id, [15, 20])
    hv, hsd = head_direction(20)
    log("directions ready; cos(head, dm20 of first domain) =", float(hv @ D[(smp[0]["domain"], 20)][0]))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    C = Causal(a.threads, a.max_tokens)
    alphas = {"dm20": A20, "dm15": A_OTHER, "head": A_OTHER}
    with open(a.out, "a") as f:
        for k, r in enumerate(smp):
            if r["id"] in done:
                continue
            d = r["domain"]
            dirs = {"dm20": (20,) + D[(d, 20)], "dm15": (15,) + D[(d, 15)], "head": (20, hv, hsd)}
            t1 = time.time()
            res = C.run(r, dirs, alphas)
            res.update(domain=d, format=r["format"], label=r["label"])
            f.write(json.dumps(res) + "\n")
            f.flush()
            log(f"{k + 1}/{len(smp)} {d} y={r['label']} n={res['n_tokens']} j0={res['j0']:+.2f} "
                f"j(+4)={res['steer']['dm20']['4']:+.2f} j(-4)={res['steer']['dm20']['-4']:+.2f} "
                f"cf={res['j_cf']:+.2f} p20={res['j_patch20']:+.2f} ({time.time() - t1:.1f}s)")


# ----------------------------------------------------------------- analysis

def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def flip_alpha(j0, curve):
    """alpha* where the verdict crosses 0 along the dm20 grid (piecewise linear; extrapolated; clipped)."""
    al = np.array(sorted([0.0] + [float(k) for k in curve]))
    jv = np.array([j0 if x == 0 else curve[str(int(x))] for x in al])
    for i in range(len(al) - 1):
        if (jv[i] >= 0) != (jv[i + 1] >= 0):
            return al[i] + (al[i + 1] - al[i]) * jv[i] / (jv[i] - jv[i + 1] + 1e-12)
    if jv[-1] > 0:      # still Yes at the strongest wrong push: extrapolate to the right
        s = (jv[-1] - jv[-2]) / (al[-1] - al[-2])
        x = al[-1] - jv[-1] / s if s < -1e-6 else 20.0
    else:               # still No at the strongest right push
        s = (jv[1] - jv[0]) / (al[1] - al[0])
        x = al[0] - jv[0] / s if s < -1e-6 else -20.0
    return float(np.clip(x, -20, 20))


SIGNALS = ["judge0", "flip", "fragility", "sens20", "asym20", "asym20_4", "sens15", "asym15", "cf_patch20",
           "cf_patch15", "cf_full", "head_sens", "head_asym"]
CAUSAL = ["flip", "fragility", "sens20", "asym20", "asym20_4", "sens15", "asym15", "cf_patch20", "cf_patch15",
          "cf_full"]


def signals(R):
    s = {k: [] for k in SIGNALS}
    for r in R:
        j0, c, c15, ch = r["j0"], r["steer"]["dm20"], r["steer"]["dm15"], r["steer"]["head"]
        af = flip_alpha(j0, c)
        s["judge0"].append(-j0)
        s["flip"].append(-af)
        s["fragility"].append(-abs(af))
        s["sens20"].append((c["-2"] - c["2"]) / 4)
        s["asym20"].append((j0 - c["2"]) - (c["-2"] - j0))
        s["asym20_4"].append((j0 - c["4"]) - (c["-4"] - j0))
        s["sens15"].append((c15["-2"] - c15["2"]) / 4)
        s["asym15"].append((j0 - c15["2"]) - (c15["-2"] - j0))
        s["cf_patch20"].append(r["j_patch20"] - j0)
        s["cf_patch15"].append(r["j_patch15"] - j0)
        s["cf_full"].append(r["j_cf"] - j0)
        s["head_sens"].append((ch["-2"] - ch["2"]) / 4)
        s["head_asym"].append((j0 - ch["2"]) - (ch["-2"] - j0))
    return {k: np.array(v, float) for k, v in s.items()}


def zdom(v, dom):
    out = np.zeros(len(v))
    for d in np.unique(dom):
        m = dom == d
        out[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-9)
    return out


def lodo_stack(X, y, dom):
    from sklearn.linear_model import LogisticRegression
    out = np.zeros(len(y))
    for d in np.unique(dom):
        tr = dom != d
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X[tr], y[tr])
        out[~tr] = m.decision_function(X[~tr])
    return out


def per_domain(scores, y, dom, doms):
    return np.array([auroc(scores[dom == d], y[dom == d]) for d in doms])


def boot(score_dict, y, dom, doms, B, seed=0):
    """{name: (B, n_domains)} per-domain AUROCs on paired within-domain, label-stratified resamples."""
    rng = np.random.default_rng(seed)
    idx = {d: (np.where((dom == d) & (y == 0))[0], np.where((dom == d) & (y == 1))[0]) for d in doms}
    out = {k: np.zeros((B, len(doms))) for k in score_dict}
    for b in range(B):
        for j, d in enumerate(doms):
            i0, i1 = idx[d]
            ii = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
            yy = y[ii]
            for k, s in score_dict.items():
                out[k][b, j] = auroc(s[ii], yy)
    return out


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def cmd_analyze(a):
    from scipy.stats import spearmanr
    R = [json.loads(l) for l in open(a.cache)]
    log(f"{len(R)} replies")
    y = np.array([r["label"] for r in R])
    dom = np.array([r["domain"] for r in R])
    fmt = np.array([r["format"] for r in R])
    doms = sorted(set(dom.tolist()))
    fmts = sorted(set(fmt.tolist()))
    S = signals(R)
    ex = np.load(EXPERT, allow_pickle=True)
    pos = {int(i): k for k, i in enumerate(ex["ids"])}
    rep = np.array([ex["reply"][pos[r["id"]]] for r in R])
    zj = zdom(S["judge0"], dom)
    zr = zdom(rep, dom)
    Z = {k: zdom(v, dom) for k, v in S.items()}

    scores = {k: S[k] for k in SIGNALS}
    scores["reply_head"] = rep
    scores["C0"] = zr + zj
    for k in CAUSAL + ["head_sens", "head_asym"]:
        scores[f"judge0+{k}"] = zj + Z[k]
        scores[f"C0+{k}"] = zr + zj + Z[k]
    scores["stack_C0"] = lodo_stack(np.c_[zr, zj], y, dom)
    for k in CAUSAL:
        scores[f"stack_C0+{k}"] = lodo_stack(np.c_[zr, zj, Z[k]], y, dom)
    scores["stack_C0+all_causal"] = lodo_stack(np.column_stack([zr, zj] + [Z[k] for k in CAUSAL]), y, dom)

    point = {k: per_domain(v, y, dom, doms) for k, v in scores.items()}
    BS = boot(scores, y, dom, doms, a.boot)
    fmt_of = {d: fmt[dom == d][0] for d in doms}

    def summ(k, base=None):
        o = {"mean": float(np.nanmean(point[k])), "ci": ci(np.nanmean(BS[k], 1)),
             "by_format": {f: float(np.nanmean(point[k][[fmt_of[d] == f for d in doms]])) for f in fmts}}
        if base:
            dlt = np.nanmean(BS[k] - BS[base], 1)
            o["delta_vs"] = base
            o["delta"] = float(np.nanmean(point[k] - point[base]))
            o["delta_ci"] = ci(dlt)
            o["delta_by_format"] = {}
            for f in fmts:
                m = np.array([fmt_of[d] == f for d in doms])
                o["delta_by_format"][f] = {"delta": float(np.nanmean((point[k] - point[base])[m])),
                                           "ci": ci(np.nanmean((BS[k] - BS[base])[:, m], 1))}
        return o

    rep_ = {"n": len(R), "domains": doms, "per_domain": {k: dict(zip(doms, map(float, v))) for k, v in point.items()}}
    rep_["signals"] = {k: summ(k) for k in SIGNALS + ["reply_head", "C0"]}
    rep_["R1"] = {k: rep_["signals"][k]["ci"][0] > 0.5 for k in CAUSAL + ["head_sens", "head_asym"]}
    rep_["R2"] = {}
    for k in CAUSAL + ["head_sens", "head_asym"]:
        s = summ(f"judge0+{k}", "judge0")
        rep_["R2"][k] = dict(s, new=bool(s["delta"] >= 0.010 and s["delta_ci"][0] > 0))
    rep_["R3"] = {}
    for k in CAUSAL:
        s1 = summ(f"C0+{k}", "C0")
        s2 = summ(f"stack_C0+{k}", "stack_C0")
        rep_["R3"][k] = {"sum": s1, "stack": s2, "adopt": bool(s1["delta"] >= 0.010 and s1["delta_ci"][0] > 0 and
                                                                s2["delta"] >= 0.010 and s2["delta_ci"][0] > 0)}
    rep_["stack_all_causal"] = summ("stack_C0+all_causal", "stack_C0")
    rep_["stack_C0"] = summ("stack_C0")
    # mechanics: does steering work at all, and how much is the signal just j0?
    c = np.array([[r["steer"]["dm20"][str(al)] for al in A20] for r in R])
    j0 = np.array([r["j0"] for r in R])
    rep_["mechanics"] = {
        "mean_j_by_alpha_dm20": {str(al): float(c[:, i].mean()) for i, al in enumerate(A20)} | {"0": float(j0.mean())},
        "frac_monotone_decreasing_dm20": float(np.mean(np.all(np.diff(np.c_[c[:, :4], j0, c[:, 4:]], 1) <= 1e-3, 1))),
        "frac_flipped_within_4sd": float(np.mean(np.abs(-S["flip"]) <= 4)),
        "mean_proj_sd_dm20_by_label": {str(l): float(np.mean([r["proj"]["dm20"] for r, yy in zip(R, y) if yy == l]))
                                      for l in (0, 1)},
        "spearman_with_judge0": {k: float(spearmanr(S[k], S["judge0"])[0]) for k in CAUSAL},
        "mean_j_cf": float(np.mean([r["j_cf"] for r in R])), "mean_j0": float(j0.mean()),
    }
    rep_["notes"] = ("judge0 here is the reader's own verdict at max_tokens 1024 (stored expert 'judge' used 2048). "
                     "reply_head = LODO 1.7B reply head from results/omni/expert_scores_q17.npz. "
                     "head_* use the shipped head (trained on all 21 domains): leaky, reference only.")
    json.dump(rep_, open(a.report, "w"), indent=1)
    print(f"{'signal':22s} mean  [95% CI]       " + " ".join(f"{f:>9s}" for f in fmts))
    for k in SIGNALS + ["reply_head", "C0"]:
        s = rep_["signals"][k]
        print(f"{k:22s} {s['mean']:.3f} [{s['ci'][0]:.3f},{s['ci'][1]:.3f}] " +
              " ".join(f"{s['by_format'][f]:9.3f}" for f in fmts))
    print("R2 judge0 + signal vs judge0 / R3 C0 + signal vs C0 (sum; stack)")
    for k in CAUSAL:
        r2, r3 = rep_["R2"][k], rep_["R3"][k]
        print(f"{k:12s} R2 {r2['delta']:+.3f} [{r2['delta_ci'][0]:+.3f},{r2['delta_ci'][1]:+.3f}]  "
              f"R3 sum {r3['sum']['delta']:+.3f} [{r3['sum']['delta_ci'][0]:+.3f},{r3['sum']['delta_ci'][1]:+.3f}] "
              f"stack {r3['stack']['delta']:+.3f} [{r3['stack']['delta_ci'][0]:+.3f},{r3['stack']['delta_ci'][1]:+.3f}]")
    s = rep_["stack_all_causal"]
    print(f"stack all causal vs stack_C0 {rep_['stack_C0']['mean']:.3f}: {s['delta']:+.3f} {s['delta_ci']}")
    print(json.dumps(rep_["mechanics"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n-per-domain", type=int, default=40)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--cache", required=True)
    z.add_argument("--report", required=True)
    z.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
