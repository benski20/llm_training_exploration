"""Layer-wise "commitment" (logit lens) as a signal that a reply is wrong.

IDEA. For every reply token, apply the reader's final norm + unembedding to the residual stream after every layer
(logit lens, layers 1..28 of Qwen3-1.7B; layer 28 = the real output) and ask when the token the reply actually wrote
becomes the model's prediction. Hypothesis: correct content is committed early and stably; fabricated or wrong
content is committed late or erratically.

Per token (L = layer, nL = 28, depths are L / nL; "never" = 29 / 28):
    settle      first layer from which the written token is top-1 at every later layer (never if final top-1 differs)
    first_top1  first layer at which the written token is top-1
    p50         first layer at which p(written token) > 0.5
    flips       number of top-1 changes across layers 7..28 (from 25% depth; earlier lens layers are noise)
    kl_mid      mean KL(final || lens) over layers 14..21 (50-75% depth)
    logrank_mid mean log(1 + rank of the written token) over layers 14..27
    mid_disagree top-1 at layer 21 (75% depth) differs from the final top-1
    nlogp_late  -mean log p(written token) over layers 14..28 (area under the commitment curve, late half)
Per reply (20 features, all signed a priori so that larger = "wrong"):
    mean over tokens of the 8 token measures; top-decile mean of settle, p50, flips, kl_mid, logrank_mid, nlogp_late;
    worst sentence (max over the reply's units of the unit mean) of settle, p50, kl_mid, logrank_mid;
    frac_never (final top-1 is not the written token); frac_late (settle depth > 0.75, never included).
The lens is exact (full 151,936-token vocabulary), computed layer by layer, 256 reply positions at a time.

SUBSET. 25 wrong + 25 correct replies per domain (seed 0) among replies with stored 1.7B features: 1,050 replies.
One forward pass each (bf16, CPU, 2 threads, max_tokens 1024, no judge suffix: the reader is causal).

ANALYSES (AUROC within a domain, then averaged over domains; CIs: 1,000 bootstrap resamples of replies within each
domain, stratified by label, paired across methods):
  (a) each feature alone, zero training, a-priori sign;
  (b) LODO logistic on the 20 commitment features (per-domain standardisation from unlabelled replies);
      control: the reader's 10 output scalars (log-prob, entropy, margin: already in the reply head) alone, and with
      the commitment features;
  (c) the 1.7B reply head (omni_final.py recipe, LODO on all 5,107 stored replies; nested heads for stacking: the
      score of a training row in domain e comes from a head that saw neither the held-out domain nor e), plus
        sum   z(reply) + z(commitment LODO score), z within the domain (unlabelled)
        stack LODO logistic on [z(reply), z(20 commitment features)]

PRE-STATED RULES (fixed before the first full run):
  R1 signal   a feature (or a head) carries signal if its mean AUROC over the 21 domains has a 95% CI lower bound > 0.50
              (a-priori sign). The hypothesis is supported only if the timing features (settle, first_top1, p50,
              flips) pass R1 in the stated direction.
  R2 beyond   commitment carries information beyond the output log-prob if (scalars + commitment) - scalars >= +0.010
              with the CI above 0.
  R3 adds     adopted only if stack (primary) or sum raises the mean AUROC over the reply head by >= +0.010 with the
              95% CI of the paired difference above 0. A format-level claim needs the same within that format.

    OMP_NUM_THREADS=2 python experiments/novel_logit_lens.py --stage all --q17 /home/user/omni/q17 \
        --corpus /home/user/omni/corpus.jsonl --cache <scratch>/lens.npz --report results/omni/novel_logit_lens.json
    (smoke test: --limit 20)
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
from omni_eval import LogReg, auroc, balanced_weights, lodo, pca_fit, standardise  # noqa: E402
from novel_consensus_dynamics import evaluate, nested_heads, stack, zdom  # noqa: E402

T0 = time.time()
TOKEN = ["settle", "first_top1", "p50", "flips", "kl_mid", "logrank_mid", "mid_disagree", "nlogp_late"]
DECILE = ["settle", "p50", "flips", "kl_mid", "logrank_mid", "nlogp_late"]
WORST = ["settle", "p50", "kl_mid", "logrank_mid"]
FEATS = [f"mean_{t}" for t in TOKEN] + [f"top10_{t}" for t in DECILE] + [f"worstsent_{t}" for t in WORST] + \
    ["frac_never", "frac_late"]
TIMING = [f for f in FEATS if any(k in f for k in ("settle", "first_top1", "p50", "flips"))]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def pick_subset(ids_avail, rows, per_class, seed):
    rng = np.random.default_rng(seed)
    out = []
    avail = set(int(i) for i in ids_avail)
    for d in sorted({r["domain"] for r in rows.values()}):
        for lab in (0, 1):
            pool = sorted(i for i, r in rows.items() if r["domain"] == d and r["label"] == lab and i in avail)
            out += rng.choice(pool, min(per_class, len(pool)), replace=False).tolist()
    return sorted(out)


# ----------------------------------------------------------------- the lens

class Lens:
    def __init__(self, name, max_tokens, threads):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
        self.nL = self.model.config.num_hidden_layers
        self.max_tokens = max_tokens
        self.W = self.model.lm_head.weight                              # V x d (bf16)
        self.norm = self.model.model.norm

    def token_stats(self, r, chunk=256):
        """per reply token: per-layer log p(written), rank(written), top-1 id, KL(final || lens); layers 1..nL."""
        from vt.omni import encode, token_units
        torch = self.torch
        ids, r0, roff = encode(self.tok, r, self.max_tokens)
        n = len(ids) - r0
        if n <= 1:
            return None
        nL = self.nL
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([ids]), output_hidden_states=True, logits_to_keep=n + 1)
            tgt = torch.tensor(ids[r0:r0 + n])
            lp_t = np.zeros((nL, n), np.float32)
            rank = np.zeros((nL, n), np.float32)
            top1 = np.zeros((nL, n), np.int64)
            kl = np.zeros((nL, n), np.float32)
            for a in range(0, n, chunk):
                b = min(n, a + chunk)
                fin = o.logits[0, a:b].float().log_softmax(-1)              # logits at r0-1+a .. predict r0+a ..
                pf = fin.exp()
                t = tgt[a:b, None]
                for L in range(1, nL + 1):
                    if L == nL:
                        lp = fin
                    else:
                        h = self.norm(o.hidden_states[L][0, r0 - 1 + a:r0 - 1 + b])
                        lp = (h @ self.W.T).float().log_softmax(-1)
                    v = lp.gather(-1, t)
                    lp_t[L - 1, a:b] = v[:, 0].numpy()
                    rank[L - 1, a:b] = (lp > v).sum(-1).float().numpy()
                    top1[L - 1, a:b] = lp.argmax(-1).numpy()
                    kl[L - 1, a:b] = (pf * (fin - lp)).sum(-1).clamp_min(0).numpy() if L < nL else 0.0
                    del lp
        tu = token_units(roff[:n], r["units"])
        return {"lp": lp_t, "rank": rank, "top1": top1, "kl": kl, "tgt": tgt.numpy(), "tu": tu, "n": n}


def reply_features(s, nL):
    lp, rank, top1, kl, tgt, tu = s["lp"], s["rank"], s["top1"], s["kl"], s["tgt"], s["tu"]
    n = s["n"]
    is1 = top1 == tgt[None]                                              # nL x n
    never = nL + 1
    # settle: first layer L such that is1 holds for all layers >= L
    tail_all = np.flip(np.cumprod(np.flip(is1, 0), 0), 0).astype(bool)   # tail_all[L] = all(is1[L:])
    settle = np.where(tail_all.any(0), tail_all.argmax(0) + 1, never)
    first = np.where(is1.any(0), is1.argmax(0) + 1, never)
    p_ok = lp > np.log(0.5)
    p50 = np.where(p_ok.any(0), p_ok.argmax(0) + 1, never)
    l7, l14, l21 = nL // 4, nL // 2, (3 * nL) // 4                     # 7, 14, 21 (1-based layer numbers)
    tt = top1[l7 - 1:]
    flips = (tt[1:] != tt[:-1]).sum(0).astype(float)
    kl_mid = kl[l14 - 1:l21].mean(0)
    logrank_mid = np.log1p(rank[l14 - 1:nL - 1]).mean(0)
    mid_dis = (top1[l21 - 1] != top1[nL - 1]).astype(float)
    nlogp = -lp[l14 - 1:].mean(0)
    tok = {"settle": settle / nL, "first_top1": first / nL, "p50": p50 / nL, "flips": flips, "kl_mid": kl_mid,
           "logrank_mid": logrank_mid, "mid_disagree": mid_dis, "nlogp_late": nlogp}
    k = max(1, int(np.ceil(0.1 * n)))
    top10 = lambda v: float(np.sort(v)[-k:].mean())
    f = {f"mean_{t}": float(tok[t].mean()) for t in TOKEN}
    f.update({f"top10_{t}": top10(tok[t]) for t in DECILE})
    units = [u for u in np.unique(tu) if u >= 0]
    for t in WORST:
        f[f"worstsent_{t}"] = float(max(tok[t][tu == u].mean() for u in units)) if units else float(tok[t].mean())
    f["frac_never"] = float((settle == never).mean())
    f["frac_late"] = float((settle > l21).mean())
    # depth trajectories (reply means per layer): for the report
    traj = np.stack([is1.mean(1), np.log1p(rank).mean(1), kl.mean(1), lp.mean(1)]).astype(np.float32)
    return np.array([f[k_] for k_ in FEATS], np.float32), traj


def extract(a, rows, sub):
    done = {}
    if os.path.exists(a.cache):
        C = np.load(a.cache)
        done = {int(i): (x, t) for i, x, t in zip(C["ids"], C["X"], C["traj"])}
        log(f"cache: {len(done)} replies")
    todo = [i for i in sub if i not in done]
    if not todo:
        return done
    lens = Lens(a.reader, a.max_tokens, a.threads)
    t0, k = time.time(), 0

    def save():
        ks = sorted(done)
        np.savez(a.cache, ids=np.array(ks), X=np.stack([done[i][0] for i in ks]),
                 traj=np.stack([done[i][1] for i in ks]))
    for i in todo:
        s = lens.token_stats(rows[i])
        if s is None:
            continue
        done[i] = reply_features(s, lens.nL)
        k += 1
        if k % 25 == 0 or k == len(todo):
            save()
            el = time.time() - t0
            log(f"{len(done)}/{len(sub)} replies ({el / k:.1f} s/reply, eta {el / k * (len(todo) - k) / 60:.0f} min)")
    save()
    return done


# ----------------------------------------------------------------- analysis

def reply_head(F, l2, pca):
    """omni_final.py reply head features (all stored replies)."""
    n, nl = len(F["ids"]), F["r_mean"].shape[1]
    mid = [nl // 2 - 1, nl // 2, nl // 2 + 1]
    H = np.concatenate([F["r_mean"][:, mid].reshape(n, -1), F["j_state"][:, mid].reshape(n, -1)], 1).astype(np.float32)
    H = standardise(H)
    mu, V = pca_fit(H, pca)
    return np.concatenate([(H - mu) @ V.T, F["r_sc"], F["j_logit"][:, None]], 1)


def analyse(a, rows, done, F):
    ids = np.array(sorted(done))
    X = np.stack([done[i][0] for i in ids]).astype(float)
    traj = np.stack([done[i][1] for i in ids])
    dom = np.array([rows[i]["domain"] for i in ids])
    y = np.array([rows[i]["label"] for i in ids])
    doms = sorted(set(dom.tolist()))
    fmt_of = {d: rows[int(ids[dom == d][0])]["format"] for d in doms}
    log(f"analysis on {len(ids)} replies, {len(doms)} domains")

    # reply head on all stored replies (plain LODO and nested), restricted to the subset
    pos = {int(i): k for k, i in enumerate(F["ids"])}
    sel = np.array([pos[int(i)] for i in ids])
    domF = np.array([rows[int(i)]["domain"] for i in F["ids"]])
    yF = np.array([rows[int(i)]["label"] for i in F["ids"]])
    XF = reply_head(F, a.l2, a.pca)
    _, sF = lodo(XF, yF, domF, a.l2, True)
    reply = sF[sel]
    XFs = standardise(XF, domF)
    S = nested_heads(XFs, yF, domF, doms, a.l2)[:, sel]                 # (n_dom, n_sub)
    log("reply head done")
    rsc = F["r_sc"][sel].astype(float)

    Xs = standardise(X, dom)
    _, commit = lodo(X, y, dom, a.l2, True)
    _, scal = lodo(rsc, y, dom, a.l2, True)
    _, scal_commit = lodo(np.concatenate([rsc, X], 1), y, dom, a.l2, True)
    plus_sum = zdom(reply, dom) + zdom(commit, dom)
    plus_stack = stack([S, Xs], y, dom, doms, a.l2)
    methods = {f: X[:, j] for j, f in enumerate(FEATS)}
    methods.update({"commit_lodo": commit, "scalars_lodo": scal, "scalars+commit_lodo": scal_commit,
                    "reply": reply, "reply+commit_sum": plus_sum, "reply+commit_stack": plus_stack,
                    "neg_lp_mean (output only)": -rsc[:, 0]})
    comps = [("scalars+commit_lodo", "scalars_lodo"), ("reply+commit_sum", "reply"), ("reply+commit_stack", "reply"),
             ("commit_lodo", "scalars_lodo")]
    res = evaluate(methods, y, dom, fmt_of, doms, a.boot, comps)

    # correlations with the output log-prob mean (is commitment just log-prob?)
    corr = {f: float(np.mean([np.corrcoef(X[dom == d, j], -rsc[dom == d, 0])[0, 1] for d in doms]))
            for j, f in enumerate(FEATS)}
    # depth trajectories: mean over replies of the per-layer reply means, correct vs wrong, per format
    tr = {}
    for g in ["all"] + sorted(set(fmt_of.values())):
        m = np.ones(len(y), bool) if g == "all" else np.array([fmt_of[d] == g for d in dom])
        tr[g] = {nm: {"correct": np.round(traj[m & (y == 0), k].mean(0), 4).tolist(),
                      "wrong": np.round(traj[m & (y == 1), k].mean(0), 4).tolist()}
                 for k, nm in enumerate(["top1_is_written", "log1p_rank", "kl_to_final", "logp_written"])}
    mean_all = res["mean"]["all"]
    rules = {
        "R1_features_passing": [f for f in FEATS if mean_all[f][1] > 0.5],
        "R1_features_reversed": [f for f in FEATS if mean_all[f][2] < 0.5],
        "R1_timing_features_passing": [f for f in TIMING if mean_all[f][1] > 0.5],
        "R1_commit_lodo": mean_all["commit_lodo"][1] > 0.5,
        "R2_beyond_logprob": {g: (v["scalars+commit_lodo - scalars_lodo"][0] >= 0.010 and
                                  v["scalars+commit_lodo - scalars_lodo"][1] > 0) for g, v in res["diff"].items()},
        "R3_adds_stack": {g: (v["reply+commit_stack - reply"][0] >= 0.010 and v["reply+commit_stack - reply"][1] > 0)
                          for g, v in res["diff"].items()},
        "R3_adds_sum": {g: (v["reply+commit_sum - reply"][0] >= 0.010 and v["reply+commit_sum - reply"][1] > 0)
                        for g, v in res["diff"].items()},
    }
    rep = {"reader": a.reader, "n": int(len(ids)), "domains": len(doms), "max_tokens": a.max_tokens,
           "features": FEATS, "per_class_per_domain": a.per_class, "bootstrap": a.boot, "results": res,
           "corr_with_neg_logprob_mean": corr, "trajectories_layers_1_to_28": tr, "rules": rules,
           "ids": [int(i) for i in ids]}
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)
    groups = list(res["mean"])
    print(f"{'method':32s} " + " ".join(f"{g[:9]:>20s}" for g in groups))
    for nm in methods:
        print(f"{nm:32s} " + " ".join("{:.3f} [{:.3f},{:.3f}]".format(*res['mean'][g][nm]) for g in groups))
    for g in groups:
        print(g, json.dumps(res["diff"][g]))
    print(json.dumps(rules, indent=1))
    print("corr with -lp_mean:", json.dumps({k: round(v, 2) for k, v in corr.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "analyse", "all"], default="all")
    ap.add_argument("--q17", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--reader", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first N replies of the subset only")
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rows = {}
    for line in open(a.corpus):
        r = json.loads(line)
        rows[r["id"]] = r
    F = load_features(a.q17)
    sub = pick_subset(F["ids"], rows, a.per_class, a.seed)
    if a.limit:
        sub = list(np.random.default_rng(a.seed).choice(sub, a.limit, replace=False))
    log(f"subset: {len(sub)} replies")
    done = {}
    if a.stage in ("extract", "all"):
        done = extract(a, rows, sub)
    if a.stage in ("analyse", "all"):
        if not done:
            C = np.load(a.cache)
            done = {int(i): (x, t) for i, x, t in zip(C["ids"], C["X"], C["traj"])}
        done = {i: v for i, v in done.items() if i in set(sub)}
        if a.limit:
            ids = sorted(done)
            X = np.stack([done[i][0] for i in ids])
            y = np.array([rows[i]["label"] for i in ids])
            for j, f in enumerate(FEATS):
                print(f"{f:24s} correct {X[y == 0, j].mean():8.3f}  wrong {X[y == 1, j].mean():8.3f}  "
                      f"pooled AUROC {auroc(X[:, j], y):.2f}")
            return
        analyse(a, rows, done, F)
    log("done")


if __name__ == "__main__":
    main()
