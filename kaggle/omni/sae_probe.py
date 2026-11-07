"""Kaggle kernel: can a probe *written* from interpretable SAE latents (chosen without labels, hand weights)
transfer across domains better than a fitted probe?  Qwen3-1.7B reader + Qwen-Scope residual SAE.

    python kaggle/kgrun.py --slug omni-sae-probe --script kaggle/omni/sae_probe.py \
        --bundle vt/omni.py experiments/omni_corpus.py experiments/recall_neurons.py \
        --out results/kaggle/omni-sae-probe [--args "--per-domain 12 --boot 50 --draws 2"] --wait

Steps (all in one kernel):
 0. build the omni corpus (experiments/omni_corpus.py build), optionally a per-domain subset (smoke test);
 1. SAE fit check: FVU of the layer-L Qwen-Scope SAE (trained on Qwen3-1.7B-Base) on reply-token residuals of
    the instruct reader and of the Base model. Pre-set rule: the instruct reader is used if its FVU <= 0.5 and
    <= 1.5 x the Base FVU; otherwise the Base model is the reader for the whole experiment;
 2. one pass per reply (vt.omni.encode, max_tokens 1024, no judge suffix): mean-pooled raw residual (resid_post of
    block L) over reply tokens, mean-pooled TopK-SAE latents, and label-free token statistics for selection;
 3. label-free latent selection (rules fixed before any label is read; see docs/omni_experiments/sae_probe.md):
      ENT  10 latents whose token activation correlates most with the reader's next-token entropy (all reply
           tokens of the corpus, unlabelled), firing on 0.5%-50% of tokens;
      UNK  10 latents with the largest standardised difference (invented - real entity) at the entity's last
           token in the real/invented entity contrast of experiments/recall_neurons.py, firing on >= 20% of the
           invented-entity prompts;
      NOV  10 latents whose token activation correlates most with "this reply token is NOT in a trigram copied
           from the prompt", over the replies that have a document (unlabelled), firing on 0.5%-50% of tokens;
    written score = mean over the 3 groups of the mean z-score of the group's latents (all weights +1/30);
 4. evaluation, leave-one-domain-out over the 21 domains, AUROC within each held-out domain:
      (a) raw-residual L2 logistic probe; (b) SAE-latent L1 logistic probe (lambda by inner leave-domains-out CV);
      (c) written probe (no labels); (c_fit) LODO logistic on the written probe's latents (diagnostic);
      (d) written probe refitted on k = 25 / 50 in-domain labels (prior-pulled to the hand weights), against
          (a)+k (the LODO raw probe refitted the same way);
    each in two variants: global standardisation and per-domain standardisation (z-scores from each domain's own
    unlabelled replies). Bootstrap CIs (replies within domain, stratified) on differences against (a).
"""
import argparse
import json
import os
import random
import sys
import time
from argparse import Namespace

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
INSTRUCT, BASE = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-Base"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_transformers():
    try:
        import transformers
        from transformers import Qwen3ForCausalLM  # noqa: F401
        log("transformers", transformers.__version__)
    except Exception as e:                                     # noqa: BLE001
        log("upgrading transformers:", e)
        os.system(f"{sys.executable} -m pip install -q -U 'transformers>=4.51' accelerate")


# ------------------------------------------------------------------ metrics

def auroc(s, y):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_domain(score, y, dom, doms):
    return np.array([auroc(score[dom == d], y[dom == d]) for d in doms])


# ------------------------------------------------------------------ model / SAE

def load_model(name, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, attn_implementation="sdpa")
    return m.to(device).eval()


class SAE:
    def __init__(self, repo, layer, device):
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        sd = torch.load(hf_hub_download(repo, f"layer{layer}.sae.pt"), map_location="cpu")
        self.k, self.cfg = int(cfg["k"]), cfg
        self.W_enc = sd["W_enc"].float().to(device)           # (d_sae, d_model)
        self.b_enc = sd["b_enc"].float().to(device)
        self.W_dec = sd["W_dec"].float().to(device)           # (d_model, d_sae)
        self.b_dec = sd["b_dec"].float().to(device)
        self.d_sae = self.W_enc.shape[0]

    def encode(self, h):                                        # as in the Qwen-Scope model card
        pre = h @ self.W_enc.T + self.b_enc
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, v)

    def decode(self, a):
        return a @ self.W_dec.T + self.b_dec


def run_layer(model, ids, layer, keep_logits=0):
    """resid_post of block `layer` (= hidden_states[layer + 1]) for every position, and the last keep_logits
    logits."""
    cap = {}

    def hook(mod, inp, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out)[0]
    hd = model.model.layers[layer].register_forward_hook(hook)
    dev = model.get_input_embeddings().weight.device
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids], device=dev), logits_to_keep=max(keep_logits, 1))
    hd.remove()
    return cap["h"], (o.logits[0] if keep_logits else None)


def entropy_rows(logits, chunk=256):
    out = []
    for s in range(0, len(logits), chunk):
        lp = logits[s:s + chunk].float().log_softmax(-1)
        out.append(-(lp.exp() * lp).sum(-1))
    return torch.cat(out)


# ------------------------------------------------------------------ logistic fits

def logreg_np(X, y, l2=0.05, w0=None, b0=0.0):
    from vt.omni import LogReg, balanced_weights
    return LogReg(l2, w0, b0).fit(X, y, balanced_weights(y))


def fit_l1_gpu(X, y, lam, iters=300):
    """L1 logistic regression (balanced sample weights, mean loss + lam * |w|_1) by FISTA on the GPU."""
    n, d = X.shape
    p = y.mean()
    sw = torch.where(y > 0, 0.5 / p, 0.5 / (1 - p))
    sw = sw / sw.mean()
    Xa = torch.cat([X, torch.ones(n, 1, device=X.device)], 1)
    v = torch.randn(d + 1, device=X.device)
    for _ in range(25):                                        # power iteration for the Lipschitz constant
        v = Xa.T @ (Xa @ v)
        v = v / v.norm()
    L = 0.25 * float(sw.max()) * float((Xa @ v).norm() ** 2) / n + 1e-6
    w = torch.zeros(d + 1, device=X.device)
    z, t = w.clone(), 1.0
    for _ in range(iters):
        g = Xa.T @ ((torch.sigmoid(Xa @ z) - y) * sw) / n
        wn = z - g / L
        wn[:d] = torch.sign(wn[:d]) * torch.clamp(wn[:d].abs() - lam / L, min=0)
        tn = (1 + (1 + 4 * t * t) ** 0.5) / 2
        z = wn + (t - 1) / tn * (wn - w)
        w, t = wn, tn
    return w


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14, help="SAE layer (resid_post of block L = hidden_states[L+1])")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--per-domain", type=int, default=0, help="smoke test: at most this many replies per domain")
    ap.add_argument("--fvu-n", type=int, default=200)
    ap.add_argument("--n-select", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--force-reader", default="")
    ap.add_argument("--ks", type=int, nargs="+", default=[25, 50])
    a = ap.parse_args()
    t0 = time.time()
    ensure_transformers()
    from transformers import AutoTokenizer

    import omni_corpus
    from recall_neurons import FIRST, LAST, PLACES, REAL, TEMPLATES, WORKS  # noqa: F401
    from recall_neurons import fake_entities
    from vt.omni import encode

    # ---- 0. corpus
    cpath = os.path.join(OUT, "corpus.jsonl")
    if not os.path.exists(cpath):
        omni_corpus.cmd_build(Namespace(out=cpath, ragtruth=300, halueval_pairs=150, truthfulqa=200, halubench=200,
                                        processbench=240))
    rows = [json.loads(l) for l in open(cpath)]
    if a.per_domain:
        cnt, keep = {}, []
        for r in rows:
            if cnt.get(r["domain"], 0) < a.per_domain:
                keep.append(r)
                cnt[r["domain"]] = cnt.get(r["domain"], 0) + 1
        rows = keep
    log("rows", len(rows))
    tok = AutoTokenizer.from_pretrained(INSTRUCT)              # same vocabulary as Base; chat template from instruct
    enc = []
    for r in rows:
        ids, r0, _ = encode(tok, r, a.max_tokens)
        enc.append((ids, r0))
    L = a.layer
    dev0 = "cuda:0"
    dev1 = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    sae = SAE(a.sae, L, dev0)
    res = {"setup": {"reader_instruct": INSTRUCT, "base": BASE, "sae": a.sae, "sae_cfg": sae.cfg, "layer": L,
                     "hidden_states_index": L + 1, "max_tokens": a.max_tokens, "n_rows": len(rows),
                     "per_domain_cap": a.per_domain, "dtype": "float16", "attn": "sdpa"}}

    # ---- 1. which reader the SAE fits
    rng = random.Random(0)
    fidx = rng.sample(range(len(rows)), min(a.fvu_n, len(rows)))

    def fvu(model):
        num = den = 0.0
        nonfinite = 0
        X, R = [], []
        for i in fidx:
            ids, r0 = enc[i]
            if len(ids) - r0 <= 0:
                continue
            h, _ = run_layer(model, ids, L)
            h = h[r0:].float().to(dev0)
            nonfinite += int((~torch.isfinite(h)).any(1).sum())
            h = torch.nan_to_num(h)
            xh = sae.decode(sae.encode(h))
            X.append(h.cpu())
            R.append((h - xh).cpu())
        X, R = torch.cat(X), torch.cat(R)
        num = float((R ** 2).sum())
        den = float(((X - X.mean(0)) ** 2).sum())
        cos = float(torch.nn.functional.cosine_similarity(X, X - R, dim=1).mean())
        return {"fvu": num / den, "mean_cos": cos, "tokens": int(len(X)), "nonfinite_tokens": nonfinite}

    m_inst = load_model(INSTRUCT, dev0)
    fit_inst = fvu(m_inst)
    m_base = load_model(BASE, dev1)
    fit_base = fvu(m_base)
    use_inst = fit_inst["fvu"] <= 0.5 and fit_inst["fvu"] <= 1.5 * fit_base["fvu"]
    reader_name = INSTRUCT if use_inst else BASE
    if a.force_reader:
        reader_name = a.force_reader
    res["sae_fit"] = {"instruct": fit_inst, "base": fit_base, "rule": "instruct if FVU<=0.5 and <=1.5x Base FVU",
                      "reader_used": reader_name}
    log("SAE fit", res["sae_fit"])
    if reader_name == INSTRUCT:
        del m_base
        model = m_inst
    else:
        del m_inst
        model = m_base
    torch.cuda.empty_cache()

    # ---- 2. one pass per reply
    D = sae.d_sae
    N = len(rows)
    RAW = np.zeros((N, model.config.hidden_size), np.float32)
    LAT = np.zeros((N, D), np.float16)
    ok = np.ones(N, bool)
    f64 = dict(device=dev0, dtype=torch.float64)
    st = {k: torch.zeros(D, **f64) for k in ("e_sa", "e_saa", "e_sae", "e_fire", "c_sa", "c_saa", "c_sac", "c_fire")}
    sc = {"e_n": 0, "e_se": 0.0, "e_see": 0.0, "c_n": 0, "c_sc": 0.0, "c_scc": 0.0}
    nonfinite = 0
    for i, r in enumerate(rows):
        ids, r0 = enc[i]
        n = len(ids) - r0
        if n <= 0:
            ok[i] = False
            continue
        h, lg = run_layer(model, ids, L, keep_logits=n)
        h = h[r0:].float().to(dev0)
        bad = ~torch.isfinite(h)
        if bad.any():
            nonfinite += 1
            h = torch.nan_to_num(h, posinf=0.0, neginf=0.0)
        act = sae.encode(h)                                    # (n, D)
        RAW[i] = h.mean(0).cpu().numpy()
        LAT[i] = act.mean(0).cpu().numpy().astype(np.float16)
        ad = act.double()
        e = entropy_rows(lg).to(dev0).double()                 # entropy of the next-token prediction at each reply token
        e = torch.nan_to_num(e)
        st["e_sa"] += ad.sum(0)
        st["e_saa"] += (ad ** 2).sum(0)
        st["e_sae"] += e @ ad
        st["e_fire"] += (ad > 0).double().sum(0)
        sc["e_n"] += n
        sc["e_se"] += float(e.sum())
        sc["e_see"] += float((e ** 2).sum())
        if r.get("doc"):                                       # novelty: reply token not in a trigram seen in the prompt
            tri = {tuple(ids[j - 2:j + 1]) for j in range(2, r0)}
            c = torch.tensor([0.0 if (j >= 2 and tuple(ids[j - 2:j + 1]) in tri) else 1.0
                              for j in range(r0, r0 + n)], **f64)
            st["c_sa"] += ad.sum(0)
            st["c_saa"] += (ad ** 2).sum(0)
            st["c_sac"] += c @ ad
            st["c_fire"] += (ad > 0).double().sum(0)
            sc["c_n"] += n
            sc["c_sc"] += float(c.sum())
            sc["c_scc"] += float((c ** 2).sum())
        if i % 250 == 0:
            log(f"read {i}/{N}  {(time.time() - t0) / 60:.1f} min")
    res["setup"]["rows_nonfinite_hidden"] = nonfinite
    res["setup"]["rows_empty"] = int((~ok).sum())

    def corr(prefix, sk, n, s, ss):
        sa, saa, sab = st[prefix + "_sa"], st[prefix + "_saa"], st[prefix + "_sa" + sk]
        cov = sab / n - (sa / n) * (s / n)
        va = saa / n - (sa / n) ** 2
        vb = ss / n - (s / n) ** 2
        return (cov / (va.clamp(min=1e-12) * vb) ** 0.5).cpu().numpy(), (st[prefix + "_fire"] / n).cpu().numpy()

    rho_e, fire_e = corr("e", "e", sc["e_n"], sc["e_se"], sc["e_see"])
    rho_c, fire_c = corr("c", "c", sc["c_n"], sc["c_sc"], sc["c_scc"])

    # ---- 3. entity contrast (no correctness labels): invented vs real entities, last entity token
    erng = random.Random(0)
    fakes = fake_entities(len(REAL), erng)
    prompts = [(t, e, 0) for e in REAL for t in erng.sample(TEMPLATES, 2)] + \
              [(t, e, 1) for e in fakes for t in erng.sample(TEMPLATES, 2)]
    EA, EY = [], []
    for t, ent, lab in prompts:
        q = t.format(ent)
        msgs = [{"role": "user", "content": q}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        end = text.index(q) + q.index(ent) + len(ent)
        pos = max(j for j, (s0, e0) in enumerate(x["offset_mapping"]) if s0 < end)
        h, _ = run_layer(model, x["input_ids"][:pos + 1], L)
        EA.append(sae.encode(h[pos:pos + 1].float().to(dev0))[0].cpu().numpy())
        EY.append(lab)
    EA, EY = np.stack(EA), np.array(EY)
    mu1, mu0 = EA[EY == 1].mean(0), EA[EY == 0].mean(0)
    sdp = np.sqrt(0.5 * (EA[EY == 1].var(0) + EA[EY == 0].var(0))) + 1e-6
    d_unk = (mu1 - mu0) / sdp
    fire_unk = (EA[EY == 1] > 0).mean(0)

    K = a.n_select
    win_e = (fire_e >= 0.005) & (fire_e <= 0.5)
    win_c = (fire_c >= 0.005) & (fire_c <= 0.5)
    sel = {"ENT": [int(j) for j in np.argsort(-np.where(win_e, rho_e, -np.inf))[:K]],
           "UNK": [int(j) for j in np.argsort(-np.where(fire_unk >= 0.2, d_unk, -np.inf))[:K]],
           "NOV": [int(j) for j in np.argsort(-np.where(win_c, rho_c, -np.inf))[:K]]}
    res["selection"] = {g: [{"latent": j, "rho_entropy": round(float(rho_e[j]), 4),
                             "d_unknown": round(float(d_unk[j]), 3), "rho_novel": round(float(rho_c[j]), 4),
                             "token_fire_rate": round(float(fire_e[j]), 4)} for j in v] for g, v in sel.items()}
    log("selected", {g: v for g, v in sel.items()})
    del model
    torch.cuda.empty_cache()

    # ---- 4. evaluation
    rows_ok = [r for r, o in zip(rows, ok) if o]
    RAW, LAT = RAW[ok], LAT[ok]
    y = np.array([r["label"] for r in rows_ok], float)
    dom = np.array([r["domain"] for r in rows_ok])
    fmt_of = {r["domain"]: r["format"] for r in rows_ok}
    doms = sorted(set(dom.tolist()))
    fmts = sorted(set(fmt_of.values()))
    ids_ok = [r["id"] for r in rows_ok]

    def zglobal(X, tr):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
        return (X - mu) / sd

    def zdom(X):
        X = X.astype(np.float32).copy()
        for d in doms:
            m = dom == d
            X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-4)
        return X

    # latents kept for the fitted SAE probe: active (non-zero mean) in >= 1% of replies (label-free)
    act_cols = np.where((LAT > 0).mean(0) >= 0.01)[0]
    LATf = LAT[:, act_cols].astype(np.float32)
    res["setup"]["sae_probe_columns"] = int(len(act_cols))
    allsel = sorted(set(sum(sel.values(), [])))
    w_hand = np.zeros(len(allsel))
    for g, v in sel.items():
        for j in v:
            w_hand[allsel.index(j)] += 1.0 / (len(sel) * len(v))
    SEL = LAT[:, allsel].astype(np.float32)
    clip = lambda X: np.clip(X, -10, 10)
    scores = {}                                                # method -> variant -> per-reply held-out score
    nnz = {}
    gpu = torch.device(dev0)
    for var in ("global", "dstd"):
        sa, sb, sb2, scf = (np.zeros(len(y)) for _ in range(4))
        Rd = zdom(RAW) if var == "dstd" else None
        Ld = clip(zdom(LATf)) if var == "dstd" else None
        Sd = clip(zdom(SEL)) if var == "dstd" else None
        nnz[var] = []
        for d in doms:
            te, tr = dom == d, dom != d
            Xr = Rd if var == "dstd" else zglobal(RAW, tr)
            sa[te] = logreg_np(Xr[tr], y[tr], 0.05).score(Xr[te])
            Xl = Ld if var == "dstd" else clip(zglobal(LATf, tr))
            XL = torch.tensor(Xl, device=gpu)
            yt = torch.tensor(y, device=gpu, dtype=torch.float32)
            # lambda by inner leave-domains-out CV (5 group folds of the training domains)
            trd = [x for x in doms if x != d]
            random.Random(1).shuffle(trd)
            folds = [trd[k::5] for k in range(5)]
            grid = (1e-3, 3e-3, 1e-2, 3e-2)
            cv = {lam: [] for lam in grid}
            for f in folds:
                itr = torch.tensor(np.isin(dom, f, invert=True) & tr, device=gpu)
                ite = np.isin(dom, f)
                for lam in grid:
                    w = fit_l1_gpu(XL[itr], yt[itr], lam)
                    s = (XL[torch.tensor(ite, device=gpu)] @ w[:-1]).cpu().numpy()
                    cv[lam] += [auroc(s[dom[ite] == x], y[ite][dom[ite] == x]) for x in f]
            best = max(grid, key=lambda lam: np.nanmean(cv[lam]))
            w = fit_l1_gpu(XL[torch.tensor(tr, device=gpu)], yt[torch.tensor(tr, device=gpu)], best)
            sb[te] = (XL[torch.tensor(te, device=gpu)] @ w[:-1]).cpu().numpy()
            nnz[var].append({"domain": d, "lambda": best, "nonzero": int((w[:-1] != 0).sum())})
            # dense L2 variant on the same latents (diagnostic)
            m2 = logreg_np(Xl[tr], y[tr], 0.05)
            sb2[te] = m2.score(Xl[te])
            Xs = Sd if var == "dstd" else clip(zglobal(SEL, tr))
            scf[te] = logreg_np(Xs[tr], y[tr], 0.05).score(Xs[te])
            log(var, d, "l1 lambda", best)
        Xs = Sd if var == "dstd" else clip(zglobal(SEL, np.ones(len(y), bool)))   # label-free stats, all replies
        scores[var] = {"a_raw": sa, "b_sae_l1": sb, "b_sae_l2": sb2, "c_written": Xs @ w_hand, "c_fit": scf}
        for g, v in sel.items():
            cols = [allsel.index(j) for j in v]
            scores[var][f"c_{g}"] = Xs[:, cols].mean(1)

    # zero-label tables + bootstrap
    rngb = np.random.default_rng(0)
    boot_idx = {}
    for d in doms:
        ii = np.where(dom == d)[0]
        pos, neg = ii[y[ii] == 1], ii[y[ii] == 0]
        boot_idx[d] = [np.concatenate([rngb.choice(pos, len(pos)), rngb.choice(neg, len(neg))]) for _ in range(a.boot)]

    def boot_aurocs(s):
        return np.array([[auroc(s[b], y[b]) for b in boot_idx[d]] for d in doms])   # (n_dom, B)

    fmt_mask = {f: np.array([fmt_of[d] == f for d in doms]) for f in fmts}
    tables = {}
    for var in scores:
        pdm = {m: per_domain(s, y, dom, doms) for m, s in scores[var].items()}
        bt = {m: boot_aurocs(s) for m, s in scores[var].items()}
        tab = {}
        for m in pdm:
            ent = {"all": round(float(np.nanmean(pdm[m])), 4)}
            ent.update({f: round(float(np.nanmean(pdm[m][fmt_mask[f]])), 4) for f in fmts})
            diffs = {}
            for key, msk in [("all", np.ones(len(doms), bool))] + list(fmt_mask.items()):
                dist = np.nanmean(bt[m][msk] - bt["a_raw"][msk], 0)
                diffs[key] = [round(float(np.nanmean(pdm[m][msk] - pdm["a_raw"][msk])), 4),
                              round(float(np.percentile(dist, 2.5)), 4), round(float(np.percentile(dist, 97.5)), 4)]
            ent["diff_vs_a_ci95"] = diffs
            ent["per_domain"] = {d: round(float(v), 4) for d, v in zip(doms, pdm[m])}
            tab[m] = ent
        tables[var] = tab
        log(var, {m: tab[m]["all"] for m in tab})
    res["zero_label"] = tables
    res["l1_fits"] = nnz

    # ---- (d) few labels: refit only the written probe's weights (prior-pulled), against (a)+k
    def adapt_eval(X, yy, w0, b0, k, draws, seed):
        rng2 = np.random.default_rng(seed)
        pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
        out, out_keep = [], []
        for _ in range(draws):
            lab = np.concatenate([rng2.choice(pos, k // 2, replace=False), rng2.choice(neg, k - k // 2, replace=False)])
            test = np.setdiff1d(np.arange(len(yy)), lab)
            Xl, yl = X[lab], yy[lab]
            grid = (0.01, 0.1, 1.0, 10.0)
            cv = {None: []}
            cv.update({g: [] for g in grid})
            for _ in range(5):
                p = rng2.permutation(len(yl))
                h1, h2 = p[:len(yl) // 2], p[len(yl) // 2:]
                if len(set(yl[h1])) < 2 or len(set(yl[h2])) < 2:
                    continue
                cv[None].append(auroc(Xl[h2] @ w0, yl[h2]))
                for g in grid:
                    cv[g].append(auroc(logreg_np(Xl[h1], yl[h1], g, w0, b0).score(Xl[h2]), yl[h2]))
            best, base = None, np.mean(cv[None]) if cv[None] else 0
            for g in grid:
                if cv[g] and np.mean(cv[g]) > base + 0.005:
                    best, base = g, np.mean(cv[g])
            s = X[test] @ w0 if best is None else logreg_np(Xl, yl, best, w0, b0).score(X[test])
            out.append(auroc(s, yy[test]))
            out_keep.append(auroc(X[test] @ w0, yy[test]))
        return float(np.mean(out)), float(np.mean(out_keep))

    Sd = clip(zdom(SEL))
    Rd = zdom(RAW)
    sw = Sd @ w_hand
    wsc = w_hand / (sw.std() + 1e-6)                            # hand weights on a unit-sd logit scale (label-free)
    few = {}
    for k in a.ks:
        rd, ra, rc = [], [], []
        for di, d in enumerate(doms):
            m = dom == d
            yy = y[m]
            if min((yy == 1).sum(), (yy == 0).sum()) < k // 2 + 5:
                rd.append(np.nan); ra.append(np.nan); rc.append(np.nan)
                continue
            dw, keep = adapt_eval(Sd[m], yy, wsc, 0.0, k, a.draws, 100 + di)
            tr = dom != d
            ma = logreg_np(Rd[tr], y[tr], 0.05)
            da, _ = adapt_eval(Rd[m], yy, ma.w, ma.b, k, a.draws, 100 + di)
            rd.append(dw); ra.append(da); rc.append(keep)
        rd, ra, rc = map(np.array, (rd, ra, rc))
        pa = np.array([tables["dstd"]["a_raw"]["per_domain"][d] for d in doms])
        ent = {}
        for name, arr in (("d_written_refit", rd), ("a_raw_refit", ra), ("c_written_same_test", rc)):
            e = {"all": round(float(np.nanmean(arr)), 4)}
            e.update({f: round(float(np.nanmean(arr[fmt_mask[f]])), 4) for f in fmts})
            cis = {}
            for ref_name, ref in (("vs_a_zero_label", pa), ("vs_a_refit", ra)):
                dd = arr - ref
                bs = [np.nanmean(dd[rngb.integers(0, len(dd), len(dd))]) for _ in range(2000)]
                cis[ref_name] = [round(float(np.nanmean(dd)), 4), round(float(np.nanpercentile(bs, 2.5)), 4),
                                 round(float(np.nanpercentile(bs, 97.5)), 4)]
            e["diff_ci95_domain_bootstrap"] = cis
            e["per_domain"] = {d: (None if np.isnan(v) else round(float(v), 4)) for d, v in zip(doms, arr)}
            ent[name] = e
        few[f"k{k}"] = ent
        log("few", k, {n_: v["all"] for n_, v in ent.items()})
    res["few_label"] = few
    res["setup"]["formats"] = {f: [d for d in doms if fmt_of[d] == f] for f in fmts}
    res["setup"]["domain_sizes"] = {d: int((dom == d).sum()) for d in doms}
    res["setup"]["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(OUT, "sae_probe.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "sae_probe_feats.npz"), ids=np.array(ids_ok), raw=RAW.astype(np.float16),
                        sel=SEL, sel_latents=np.array(allsel), y=y, dom=dom,
                        **{f"{v}_{m}": s for v in scores for m, s in scores[v].items()})
    log("done", res["setup"]["minutes"], "min")


if __name__ == "__main__":
    main()
