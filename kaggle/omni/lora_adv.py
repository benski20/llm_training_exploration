"""Kaggle kernel: does editing the reader (LoRA) make a verification probe more accurate and more domain-invariant?

Reader Qwen/Qwen3-1.7B (fp16, sdpa, T4). Feature: mean residual state over the reply tokens after decoder layer 15
(hidden_states[15], "layer 15 of 28" in docs/omni_verifier.md). Inputs are tokenised exactly as the product does
(vt/omni.encode with the judge suffix, max_tokens 1024); the suffix is cut before the forward because the reader is
causal and it cannot change the reply's states. Only layers 0..14 are run: nothing above can change the feature.

    frozen   frozen reader + logistic probe (PCA 256 + L2 logistic, as vt/omni.train_artifact), global or
             per-domain standardisation
    lora     LoRA (rank 16, alpha 32, q/k/v/o/gate/up/down of layers 7..14, i.e. the layers just below the probe)
             + a linear head initialised AT the frozen probe, BCE (balanced) on reply labels, 2 epochs
    adv      the same + a gradient-reversal domain classifier (MLP 2048-256-n_domains) on the same pooled,
             standardised state, lambda ramped 0 -> 0.3 (DANN schedule)

Folds: leave-one-format-out (4), leave-one-domain-out for 6 domains, in-distribution 80/20 (stratified).
Metric: AUROC within each held-out domain, averaged; paired bootstrap CIs (replies resampled within domain).
Two GPUs: the lora and adv variants run in parallel worker processes (one per GPU), the same folds and seeds.

    args: --limit-per-domain N (smoke test)  --folds a,b  --epochs E  --max-tokens 1024
Outputs: /kaggle/working/lora_adv.json
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("la_out")
TMP = "/tmp/la"
READER = "Qwen/Qwen3-1.7B"
PROBE_LAYER = 15
LODO = ["halubench_finance", "ragtruth_summary", "truthfulqa", "felm_wk", "pb_math", "halueval_dialogue"]
FORMATS = ["rag", "chat", "longform", "reasoning"]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-domain", type=int, default=0)
    ap.add_argument("--folds", default="")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lora-from", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--tok-batch", type=int, default=4096)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--smoke-first", action="store_true",
                    help="run a tiny smoke configuration in a subprocess first; abort if it fails")
    ap.add_argument("--tmp", default="/tmp/la")
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--deadline-h", type=float, default=10.8, help="workers stop starting folds after this")
    return ap.parse_args()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ data and folds

def build_corpus(a):
    path = "/tmp/la_corpus/corpus.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        import omni_corpus
        subprocess.run([sys.executable, omni_corpus.__file__, "build", "--out", path], check=True)
    rows = [json.loads(l) for l in open(path)]
    if a.limit_per_domain:
        rng = random.Random(0)
        keep = []
        for d in sorted({r["domain"] for r in rows}):
            for lab in (0, 1):
                g = [r for r in rows if r["domain"] == d and r["label"] == lab]
                rng.shuffle(g)
                keep += g[:a.limit_per_domain // 2]
        rows = sorted(keep, key=lambda r: r["id"])
    return rows


def make_folds(rows, only=""):
    dom = np.array([r["domain"] for r in rows])
    fmt = np.array([r["format"] for r in rows])
    y = np.array([r["label"] for r in rows])
    folds = {}
    for f in FORMATS:
        folds[f"lofo_{f}"] = fmt != f
    for d in LODO:
        folds[f"lodo_{d}"] = dom != d
    rng = np.random.default_rng(0)
    tr = np.zeros(len(rows), bool)
    for d in np.unique(dom):
        for lab in (0, 1):
            idx = np.where((dom == d) & (y == lab))[0]
            idx = rng.permutation(idx)
            tr[idx[:int(round(0.8 * len(idx)))]] = True
    folds["id"] = tr
    if only:
        folds = {k: v for k, v in folds.items() if k in only.split(",")}
    return {k: v.tolist() for k, v in folds.items()}


# ------------------------------------------------------------------------------------------------ worker (one GPU)

def worker(variant, a):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vt.omni import LogReg, _pca, balanced_weights, encode, judge_suffix

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [json.loads(l) for l in open(os.path.join(TMP, "rows.jsonl"))]
    folds = {k: np.array(v) for k, v in json.load(open(os.path.join(TMP, "folds.json"))).items()}
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(READER)
    model = AutoModelForCausalLM.from_pretrained(READER, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    suffix = tok(judge_suffix(tok), add_special_tokens=False)["input_ids"]
    enc = []
    for r in rows:
        ids, r0, _ = encode(tok, r, a.max_tokens, suffix)
        ids = ids[:len(ids) - len(suffix)]                    # causal: the suffix cannot change the reply's states
        enc.append((ids, r0, len(ids) - r0))
    N = len(rows)
    lens = np.array([len(e[0]) for e in enc])
    log(variant, "rows", N, "tokens", int(lens.sum()), "mean len", float(lens.mean()), "max", int(lens.max()))
    y = np.array([r["label"] for r in rows], float)
    dom = np.array([r["domain"] for r in rows])
    timing = {}

    def time_pass(fn, sample):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for i in sample:
                fn(i)
        torch.cuda.synchronize()
        return (time.time() - t0) / len(sample)

    sample = list(range(0, N, max(1, N // 64)))[:64]
    full_layers = model.model.layers
    if variant == "lora":
        # equivalence check: layer-15 mean over the reply from the full model's hidden_states == truncated + hook
        ids, r0, n = enc[sample[0]]
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, logits_to_keep=n + 1)
            ref = o.hidden_states[PROBE_LAYER][0, r0:r0 + n].float().mean(0)
        timing["full_28_layers_s_per_reply"] = time_pass(
            lambda i: model(input_ids=torch.tensor([enc[i][0]], device=dev), output_hidden_states=True,
                            logits_to_keep=enc[i][2] + 1), sample)
    model.model.layers = nn.ModuleList(list(full_layers)[:PROBE_LAYER])
    cap = {}
    model.model.layers[PROBE_LAYER - 1].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    def run(batch_idx):
        L = max(len(enc[i][0]) for i in batch_idx)
        x = torch.full((len(batch_idx), L), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(batch_idx), L), dtype=torch.long)
        rm = torch.zeros((len(batch_idx), L))
        for b, i in enumerate(batch_idx):
            ids, r0, n = enc[i]
            x[b, :len(ids)] = torch.tensor(ids)
            am[b, :len(ids)] = 1
            rm[b, r0:r0 + n] = 1
        model.model(input_ids=x.to(dev), attention_mask=am.to(dev), use_cache=False)
        h = cap.pop("h")
        rm = rm.to(dev)
        return (h.float() * rm[..., None]).sum(1) / rm.sum(1, keepdim=True)

    def batches(idx, budget, shuffle, rng=None):
        idx = sorted(idx, key=lambda i: len(enc[i][0]))
        out, cur = [], []
        for i in idx:
            if cur and (len(cur) + 1) * len(enc[i][0]) > budget:
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        if shuffle:
            rng.shuffle(out)
        return out

    def pooled_all():
        F = np.zeros((N, model.config.hidden_size), np.float32)
        with torch.no_grad():
            for b in batches(range(N), 2 * a.tok_batch, False):
                F[b] = run(b).cpu().numpy()
        return F

    if variant == "lora":
        with torch.no_grad():
            got = run([sample[0]])[0]
        timing["hook_vs_hidden_states_max_abs_diff"] = float((got - ref).abs().max())
        timing["hook_vs_hidden_states_rel_diff"] = float((got - ref).norm() / ref.norm())
        timing["layers_0_14_s_per_reply"] = time_pass(lambda i: run([i]), sample)
        log("equivalence", timing)

    # ---- LoRA modules
    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.zeros(a.rank, base.in_features, device=dev))
            self.B = nn.Parameter(torch.zeros(base.out_features, a.rank, device=dev))
            self.scale = a.alpha / a.rank
            self.drop = nn.Dropout(0.05)

        def reset(self, g):
            with torch.no_grad():
                bound = 1 / np.sqrt(self.base.in_features)
                self.A.copy_((torch.rand(self.A.shape, generator=g) * 2 - 1) * bound)
                self.B.zero_()

        def forward(self, x):
            return self.base(x) + (self.drop(x) @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale

    loras = []
    for li in range(a.lora_from, PROBE_LAYER):
        layer = model.model.layers[li]
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for nm in names:
                m = LoRALinear(getattr(parent, nm))
                setattr(parent, nm, m)
                loras.append(m)
    lora_params = [p for m in loras for p in (m.A, m.B)]
    log("LoRA params", sum(p.numel() for p in lora_params))

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    # ---- frozen features (LoRA B = 0 -> exactly the frozen reader)
    g = torch.Generator().manual_seed(0)
    for m in loras:
        m.reset(g)
    model.eval()
    t0 = time.time()
    F0 = pooled_all()
    timing["frozen_full_corpus_pass_s"] = time.time() - t0
    log("frozen features", F0.shape, "finite", bool(np.isfinite(F0).all()), f"{time.time() - t0:.0f}s")
    if variant == "lora":
        np.save(os.path.join(TMP, "F_frozen.npy"), F0.astype(np.float16))
        timing["layers_0_14_lora_unmerged_s_per_reply"] = time_pass(lambda i: run([i]), sample)

    report = {"timing": timing, "folds": {}}
    t_start = float(os.environ.get("LA_T0", time.time()))
    for fname, trmask in folds.items():
        if time.time() - t_start > a.deadline_h * 3600:
            log(variant, "deadline reached; skipping", fname)
            continue
        t0 = time.time()
        seed = 1000 + list(folds).index(fname)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        g = torch.Generator().manual_seed(seed)
        for m in loras:
            m.reset(g)
        tr = np.where(trmask)[0]
        # head initialised at the frozen probe (PCA 256 + L2 logistic, balanced), composed into one linear map
        mu, sd, pmu, V = _pca(F0[tr], min(256, len(tr) - 1))     # mu, sd: global standardisation of the state
        Xp = ((F0[tr] - mu) / sd - pmu) @ V.T
        lr0 = LogReg(0.05).fit(Xp, y[tr], balanced_weights(y[tr]))
        w_full = V.T @ lr0.w
        b_full = lr0.b - float(pmu @ w_full)
        head = nn.Linear(F0.shape[1], 1).to(dev)
        with torch.no_grad():
            head.weight.copy_(torch.tensor(w_full, dtype=torch.float32)[None])
            head.bias.fill_(b_full)
        mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)
        doms = sorted(set(dom[tr].tolist()))
        dix = {d: k for k, d in enumerate(doms)}
        disc = nn.Sequential(nn.Linear(F0.shape[1], 256), nn.ReLU(), nn.Linear(256, len(doms))).to(dev)
        groups = [{"params": lora_params, "lr": a.lr, "weight_decay": 0.0},
                  {"params": list(head.parameters()), "lr": a.head_lr, "weight_decay": 0.01}]
        if variant == "adv":
            groups.append({"params": list(disc.parameters()), "lr": 1e-3, "weight_decay": 0.01})
        opt = torch.optim.AdamW(groups)
        scaler = torch.amp.GradScaler("cuda")
        sw = balanced_weights(y[tr])
        swd = dict(zip(tr.tolist(), sw.tolist()))
        n_ep = int(np.ceil(a.epochs))
        ep_batches = [batches(tr.tolist(), a.tok_batch, True, rng) for _ in range(n_ep)]
        steps = [b for eb in ep_batches for b in eb][:int(round(a.epochs * len(ep_batches[0])))]
        total = len(steps)
        warm = max(1, total // 20)
        model.train()
        hist = []
        for s, b in enumerate(steps):
            f = min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * max(0, s - warm) / max(1, total - warm)))
            for gi, gr in enumerate(opt.param_groups):
                gr["lr"] = [a.lr, a.head_lr, 1e-3][gi] * f
            z = (run(b) - mu_t) / sd_t
            yt = torch.tensor(y[b], device=dev, dtype=torch.float32)
            wt = torch.tensor([swd[i] for i in b], device=dev, dtype=torch.float32)
            loss_y = (nn.functional.binary_cross_entropy_with_logits(head(z)[:, 0], yt, reduction="none") * wt).mean()
            loss = loss_y
            loss_d = torch.zeros(())
            if variant == "adv":
                p = s / total
                lam = a.lam * (2 / (1 + np.exp(-10 * p)) - 1)
                dt = torch.tensor([dix[dom[i]] for i in b], device=dev)
                loss_d = nn.functional.cross_entropy(disc(GRL.apply(z, lam)), dt)
                loss = loss + loss_d
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append((float(loss_y), float(loss_d)))
            if s % 50 == 0 or s == total - 1:
                h = np.array(hist[-50:])
                log(variant, fname, f"step {s}/{total} loss_y {h[:, 0].mean():.4f} loss_d {h[:, 1].mean():.4f}",
                    f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB {time.time() - t0:.0f}s")
        t_train = time.time() - t0
        model.eval()
        F = pooled_all()
        with torch.no_grad():
            Ft = torch.tensor(F, device=dev)
            e2e = head((Ft - mu_t) / sd_t)[:, 0].cpu().numpy()
        np.save(os.path.join(TMP, f"F_{variant}_{fname}.npy"), F.astype(np.float16))
        np.save(os.path.join(TMP, f"S_{variant}_{fname}.npy"), e2e)
        report["folds"][fname] = {"train_s": t_train, "total_s": time.time() - t0, "steps": total,
                                  "final_loss_y": float(np.mean([h[0] for h in hist[-50:]])),
                                  "final_loss_d": float(np.mean([h[1] for h in hist[-50:]])),
                                  "finite": bool(np.isfinite(F).all())}
        log(variant, fname, "done", report["folds"][fname])
        json.dump(report, open(os.path.join(TMP, f"report_{variant}.json"), "w"))
    log(variant, "all folds done")


# ------------------------------------------------------------------------------------------------ analysis (CPU)

def auroc(s, y):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe_scores(F, y, dom, tr, dstd):
    """vt/omni.train_artifact recipe on one hidden block: standardise, PCA 256, (per-domain standardisation),
    balanced L2 logistic. Per-domain stats use the domain's replies without labels (the product's profile step)."""
    from vt.omni import LogReg, _dstd, _pca, _project, balanced_weights
    F = F.astype(np.float32)
    hmu, hsd, pmu, V = _pca(F[tr], min(256, int(tr.sum()) - 1))
    X = _project(F, hmu, hsd, pmu, V)
    if dstd:
        X = _dstd(X, dom)
    m = LogReg(0.05).fit(X[tr], y[tr], balanced_weights(y[tr]))
    return m.score(X)


def domain_info(F, dom, tr_rows, rng_seed=0):
    """how much the pooled state encodes the domain: multinomial logistic on PCA-64, 80/20 within tr_rows."""
    from sklearn.linear_model import LogisticRegression
    from vt.omni import _pca, _project
    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(np.where(tr_rows)[0])
    k = int(0.8 * len(idx))
    a, b = idx[:k], idx[k:]
    hmu, hsd, pmu, V = _pca(F[a].astype(np.float32), 64)
    X = _project(F.astype(np.float32), hmu, hsd, pmu, V)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[a], dom[a])
    return float((clf.predict(X[b]) == dom[b]).mean())


def analyse(a, rows, folds, reports, wall):
    y = np.array([r["label"] for r in rows], float)
    dom = np.array([r["domain"] for r in rows])
    fmt_of = {r["domain"]: r["format"] for r in rows}
    F0 = np.load(os.path.join(TMP, "F_frozen.npy")).astype(np.float32)
    methods = ["frozen_probe", "frozen_probe_dstd"]
    for v in ("lora", "adv"):
        methods += [f"{v}_e2e_head", f"{v}_refit_probe", f"{v}_refit_probe_dstd"]
    scores = {}          # fold -> method -> scores (N)
    dinfo = {}
    for fname, trl in folds.items():
        tr = np.array(trl)
        s = {"frozen_probe": probe_scores(F0, y, dom, tr, False), "frozen_probe_dstd": probe_scores(F0, y, dom, tr, True)}
        di = {"frozen": domain_info(F0, dom, tr)}
        for v in ("lora", "adv"):
            fp = os.path.join(TMP, f"F_{v}_{fname}.npy")
            if not os.path.exists(fp):
                continue
            Fv = np.load(fp).astype(np.float32)
            s[f"{v}_e2e_head"] = np.load(os.path.join(TMP, f"S_{v}_{fname}.npy"))
            s[f"{v}_refit_probe"] = probe_scores(Fv, y, dom, tr, False)
            s[f"{v}_refit_probe_dstd"] = probe_scores(Fv, y, dom, tr, True)
            di[v] = domain_info(Fv, dom, tr)
        scores[fname] = s
        dinfo[fname] = di
        log("analysed", fname)

    def protocol_sets(proto):
        """[(fold, test indices of one held-out domain)] for a protocol."""
        out = []
        for fname, trl in folds.items():
            tr = np.array(trl)
            if not fname.startswith(proto):
                continue
            for d in sorted(set(dom[~tr].tolist())):
                out.append((fname, d, np.where((~tr) & (dom == d))[0]))
        return out

    res = {"per_domain": {}, "means": {}, "diff_vs_frozen": {}}
    rng = np.random.default_rng(0)
    for proto in ("lofo", "lodo", "id"):
        sets = protocol_sets(proto)
        if not sets:
            continue
        per = {}
        for fname, d, idx in sets:
            per[d] = {m: auroc(scores[fname][m][idx], y[idx]) for m in methods if m in scores[fname]}
            per[d]["fold"] = fname
            per[d]["format"] = fmt_of[d]
            per[d]["n"] = int(len(idx))
        res["per_domain"][proto] = per
        avail = [m for m in methods if all(m in scores[f] for f, _, _ in sets)]
        res["means"][proto] = {m: float(np.nanmean([per[d][m] for _, d, _ in sets])) for m in avail}
        if proto == "lofo":
            res["means"]["lofo_by_format"] = {f: {m: float(np.nanmean([per[d][m] for _, d, _ in sets if fmt_of[d] == f]))
                                                   for m in avail} for f in FORMATS}
        # paired bootstrap: replies resampled within each held-out domain (stratified by label), domain mean diff
        boots = {m: [] for m in avail}
        pos = [(fname, idx[y[idx] == 1], idx[y[idx] == 0]) for fname, _, idx in sets]
        for _ in range(a.boot):
            acc = {m: [] for m in avail}
            for fname, p1, p0 in pos:
                bi = np.concatenate([rng.choice(p1, len(p1)), rng.choice(p0, len(p0))])
                for m in avail:
                    acc[m].append(auroc(scores[fname][m][bi], y[bi]))
            for m in avail:
                boots[m].append(np.nanmean(acc[m]))
        boots = {m: np.array(v) for m, v in boots.items()}
        dd = {}
        for m in avail:
            if m.startswith("frozen"):
                continue
            ref = "frozen_probe_dstd" if m.endswith("dstd") else "frozen_probe"
            diff = boots[m] - boots[ref]
            dd[m] = {"vs": ref, "diff": res["means"][proto][m] - res["means"][proto][ref],
                     "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
        dd["frozen_probe_dstd"] = {"vs": "frozen_probe",
                                   "diff": res["means"][proto]["frozen_probe_dstd"] - res["means"][proto]["frozen_probe"],
                                   "ci95": [float(np.percentile(boots["frozen_probe_dstd"] - boots["frozen_probe"], 2.5)),
                                            float(np.percentile(boots["frozen_probe_dstd"] - boots["frozen_probe"], 97.5))]}
        res["diff_vs_frozen"][proto] = dd
    res["domain_id_accuracy"] = dinfo
    res["domain_id_accuracy_mean"] = {k: float(np.mean([dinfo[f][k] for f in dinfo if k in dinfo[f]]))
                                      for k in ("frozen", "lora", "adv") if any(k in dinfo[f] for f in dinfo)}
    res["domain_id_chance"] = float(max(np.unique(dom, return_counts=True)[1]) / len(dom))
    out = {"setup": {"reader": READER, "probe_layer": PROBE_LAYER, "feature": "mean residual over reply tokens, "
                     "hidden_states[15] (after 15 of 28 decoder layers)", "max_tokens": a.max_tokens,
                     "dtype": "float16, sdpa, 2x T4", "lora": {"rank": a.rank, "alpha": a.alpha,
                     "layers": list(range(a.lora_from, PROBE_LAYER)), "modules": "q,k,v,o,gate,up,down",
                     "lr": a.lr, "dropout": 0.05}, "head": "linear on globally standardised pooled state, initialised "
                     "at the frozen probe, lr %g" % a.head_lr, "adversary": "MLP 2048-256-n_train_domains via "
                     "gradient reversal, lambda ramp 0->%g (DANN)" % a.lam, "epochs": a.epochs,
                     "tok_batch": a.tok_batch, "probe": "PCA 256 + L2 logistic (0.05), balanced weights (vt/omni)",
                     "n_replies": len(rows), "n_domains": len(set(dom.tolist())), "lodo_domains": LODO,
                     "limit_per_domain": a.limit_per_domain, "bootstrap": a.boot, "seeds": "fold seed 1000+k, "
                     "id split seed 0; one training seed per fold (training variance not in the CIs)"},
           "results": res, "worker_reports": reports, "wall_s": wall}
    np.savez_compressed(os.path.join(WORK, "scores.npz"), ids=np.array([r["id"] for r in rows]), y=y,
                        **{f"{f}__{m}": v for f, s in scores.items() for m, v in s.items()})
    return out


# ------------------------------------------------------------------------------------------------ main

def main():
    global TMP, WORK
    a = args()
    TMP, WORK = a.tmp, a.work
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    variant = os.environ.get("LA_WORKER")
    if variant:
        return worker(variant, a)
    t0 = time.time()
    from importlib.metadata import version
    log("transformers", version("transformers"), "torch", version("torch"))
    tv = tuple(int(x) for x in version("transformers").split(".")[:2])
    if tv < (4, 51):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51,<4.58"], check=False)
    os.environ.setdefault("LA_T0", str(t0))
    rows = build_corpus(a)
    bundled = "/kaggle/working/src/kaggle/omni/lora_adv.py"           # bundle this file too (kgrun --bundle)
    script = bundled if os.path.exists(bundled) else os.path.abspath(sys.argv[0])
    if a.smoke_first:
        # one GPU session for both: a tiny end-to-end run first (2 folds, 24 replies/domain), abort on any failure
        sm = ["--limit-per-domain", "24", "--folds", "lofo_chat,lodo_truthfulqa", "--epochs", "0.5", "--boot", "20",
              "--tmp", "/tmp/la_smoke", "--work", os.path.join(WORK, "smoke")]
        r = subprocess.run([sys.executable, script] + sm, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
        sj = os.path.join(WORK, "smoke", "lora_adv.json")
        ok = r.returncode == 0 and os.path.exists(sj) and json.load(open(sj))["worker_exit_codes"] == [0, 0]
        log("smoke", "OK" if ok else "FAILED", f"{time.time() - t0:.0f}s")
        if not ok:
            sys.exit(1)
    folds = make_folds(rows, a.folds)
    with open(os.path.join(TMP, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(folds, open(os.path.join(TMP, "folds.json"), "w"))
    log("corpus", len(rows), "folds", list(folds), f"{time.time() - t0:.0f}s")
    import torch
    ngpu = torch.cuda.device_count()
    log("gpus", ngpu)
    procs = []
    for k, v in enumerate(("lora", "adv")):
        env = dict(os.environ, LA_WORKER=v, CUDA_VISIBLE_DEVICES=str(k % max(ngpu, 1)),
                   PYTHONPATH=os.pathsep.join(sys.path))
        wargs = [x for x in sys.argv[1:] if x != "--smoke-first"]
        p = subprocess.Popen([sys.executable, script] + wargs + ["--tmp", TMP, "--work", WORK], env=env)
        if ngpu < 2:
            p.wait()
        procs.append(p)
    codes = [p.wait() for p in procs]
    log("workers exited", codes)
    reports = {}
    for v in ("lora", "adv"):
        fp = os.path.join(TMP, f"report_{v}.json")
        reports[v] = json.load(open(fp)) if os.path.exists(fp) else None
    out = analyse(a, rows, folds, reports, time.time() - t0)
    out["worker_exit_codes"] = codes
    json.dump(out, open(os.path.join(WORK, "lora_adv.json"), "w"), indent=1)
    log("means", json.dumps(out["results"]["means"], indent=1))
    log("diffs", json.dumps(out["results"]["diff_vs_frozen"], indent=1))
    log("domain id", out["results"]["domain_id_accuracy_mean"])
    log("total", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
