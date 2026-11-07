"""Planted channel: write the probe first, then edit the reader so the written probe is true.

Instead of fitting a probe to a frozen reader, the probe is fixed in advance:
    h_mean = mean residual state over the reply's tokens at layer L (output of block L of Qwen3-1.7B, L = 15 of 28)
    score  = <h_mean, u>,  u a fixed random unit vector (seed 12345, drawn before any data is seen)
    p_wrong = sigmoid(score), flag if score > 0
and a LoRA on the reader's first L blocks is trained (BCE on sigmoid(score) vs label, 1 = wrong) so that this channel
carries P(reply is wrong). Nothing about the readout is fitted.

Baselines in the same kernel, folds and validation splits:
    probe_global  frozen reader, same pooled feature, L2 logistic probe on train-standardised features
    probe_pds     the same, each domain standardised by its own (label-free) statistics (the repo's best practice)
    written_frozen  the written probe <h_mean, u> on the frozen reader (sanity: should be chance)

Protocol: leave-one-format-out (4 folds) and leave-one-domain-out on 6 domains. In each fold 15% of the training
domains' replies (stratified by domain) are held back as a validation split: never trained on, used only to set the
"10% false alarms on training domains" threshold for every method. Metrics per held-out domain: AUROC, false-alarm
rate / recall at the fixed threshold (0: planted score, probe logit) and at the transferred 10% threshold, ECE.

    python kaggle/kgrun.py --slug omni-planted --script kaggle/omni/planted_channel.py \
        --bundle vt/omni.py experiments/omni_corpus.py --out results/kaggle/omni-planted [--args "--smoke"] --wait
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/planted"
LODO = ["halubench_finance", "ragtruth_summary", "truthfulqa", "felm_wk", "pb_math", "halueval_dialogue"]
FORMATS = ["rag", "chat", "longform", "reasoning"]
LOCK = threading.Lock()
LOAD_LOCK = threading.Lock()      # from_pretrained flips the process-wide default dtype: never load in two threads


def log(**kw):
    with LOCK:
        print(json.dumps(kw), flush=True)


# ---------------------------------------------------------------- data

def build_rows(a):
    path = os.path.join(OUT, "corpus.jsonl")
    if not os.path.exists(path):
        import omni_corpus
        ns = argparse.Namespace(out=path, ragtruth=300, halueval_pairs=150, truthfulqa=200, halubench=200,
                                processbench=240)
        omni_corpus.cmd_build(ns)
    rows = [json.loads(l) for l in open(path)]
    if a.smoke:
        rng = random.Random(0)
        by = {}
        for r in rows:
            by.setdefault((r["domain"], r["label"]), []).append(r)
        rows = []
        for k in sorted(by):
            rng.shuffle(by[k])
            rows += by[k][:a.smoke_per]
    return rows


def tokenise(rows, tok, max_tokens):
    from vt.omni import encode
    out = []
    for r in rows:
        ids, r0, _ = encode(tok, r, max_tokens)
        out.append((ids, r0))
    return out


# ---------------------------------------------------------------- model

def load_backbone(name, layer, device):
    import torch
    from transformers import AutoModelForCausalLM
    with LOAD_LOCK:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, attn_implementation="sdpa")
    bb = m.model
    bb.layers = bb.layers[:layer]
    bb.config.num_hidden_layers = layer
    bb.norm = torch.nn.Identity()                  # last_hidden_state = output of block `layer` = hidden_states[layer]
    del m
    return bb.to(device).eval()


def pooled(bb, ids, r0, device):
    import torch
    h = bb(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state[0]
    return h[r0:].float().mean(0)


def extract(bb, toks, idx, device):
    import torch
    out = np.zeros((len(idx), bb.config.hidden_size), np.float32)
    with torch.no_grad():
        for k, i in enumerate(idx):
            out[k] = pooled(bb, toks[i][0], toks[i][1], device).cpu().numpy()
    return out


def train_fold(a, fold, toks, y, u_np, device):
    import torch
    import peft.import_utils
    from peft import LoraConfig, get_peft_model
    peft.import_utils.is_torchao_available = lambda: False      # Kaggle's torchao 0.10 trips peft's version check
    try:
        import peft.tuners.lora.torchao as _tao
        _tao.is_torchao_available = lambda: False
    except ImportError:
        pass
    t0 = time.time()
    bb = load_backbone(a.model, a.layer, device)
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    with LOAD_LOCK:
        pm = get_peft_model(bb, cfg)
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    u = torch.tensor(u_np, device=device)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    tr = list(fold["train"])
    n_steps = a.epochs * len(tr) // a.accum
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, n_steps)))))
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(fold["seed"])
    pm.train()
    curve, nan = [], 0
    step = 0
    for ep in range(a.epochs):
        order = rng.permutation(tr)
        run, cnt = 0.0, 0
        for k, i in enumerate(order):
            s = pooled(pm, toks[i][0], toks[i][1], device) @ u
            loss = torch.nn.functional.binary_cross_entropy_with_logits(s, torch.tensor(float(y[i]), device=device))
            if not torch.isfinite(loss):
                nan += 1
                continue
            scaler.scale(loss / a.accum).backward()
            run += float(loss)
            cnt += 1
            if (k + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
            if (k + 1) % 500 == 0:
                log(fold=fold["name"], epoch=ep, seen=k + 1, loss=run / max(cnt, 1), min=(time.time() - t0) / 60)
        curve.append(run / max(cnt, 1))
        log(fold=fold["name"], epoch_done=ep, train_loss=curve[-1], nan=nan, min=(time.time() - t0) / 60)
    pm.eval()
    ev = list(fold["val"]) + list(fold["test"])
    H = extract(pm, toks, ev, device)
    del pm, bb, opt
    torch.cuda.empty_cache()
    return {"ev_idx": np.array(ev), "H_tuned": H, "curve": curve, "nan": nan, "minutes": (time.time() - t0) / 60}


# ---------------------------------------------------------------- metrics

def auroc(s, y):
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(s, y, bins=10):
    p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
    e, n = 0.0, len(p)
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins) if b < bins - 1 else (p >= b / bins)
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def dom_metrics(s, y, thr0, thr10):
    c, w = s[y == 0], s[y == 1]
    return {"auroc": auroc(s, y), "fa0": float((c > thr0).mean()), "rec0": float((w > thr0).mean()),
            "fa10": float((c > thr10).mean()), "rec10": float((w > thr10).mean()), "ece": ece(s, y),
            "n": int(len(y)), "n_clean": int(len(c))}


def fit_probe(X, y, l2):
    from vt.omni import LogReg
    return LogReg(l2=l2).fit(X, y)


def dstd(X, groups):
    X = X.copy()
    for g in np.unique(groups):
        m = groups == g
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-4)
    return X


def fold_scores(fold, res, Hf, y, dom, u, l2):
    """per method: scores on val and test (aligned with fold['val'], fold['test'])."""
    tr, va, te = np.array(fold["train"]), np.array(fold["val"]), np.array(fold["test"])
    nv = len(va)
    out = {"planted": (res["H_tuned"][:nv] @ u, res["H_tuned"][nv:] @ u),
           "written_frozen": (Hf[va] @ u, Hf[te] @ u)}
    mu, sd = Hf[tr].mean(0), Hf[tr].std(0) + 1e-4
    pr = fit_probe((Hf[tr] - mu) / sd, y[tr], l2)
    out["probe_global"] = (pr.score((Hf[va] - mu) / sd), pr.score((Hf[te] - mu) / sd))
    Z = dstd(Hf, dom)                              # every domain by its own unlabelled statistics
    pr2 = fit_probe(Z[tr], y[tr], l2)
    out["probe_pds"] = (pr2.score(Z[va]), pr2.score(Z[te]))
    return out


def evaluate(folds, scores, y, dom, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    methods = list(next(iter(scores.values())).keys())
    per = {}       # protocol -> method -> domain -> metrics
    boot = {}      # protocol -> method -> metric -> (B,) mean over domains
    for f in folds:
        te, va = np.array(f["test"]), np.array(f["val"])
        yv0 = y[va] == 0
        tdoms = sorted(set(dom[te]))
        # bootstrap indices shared across methods (paired)
        n0v = int(yv0.sum())
        bidx_val = [rng.integers(0, n0v, n0v) for _ in range(n_boot)]
        bidx_dom = {}
        for d in tdoms:
            m = np.where(dom[te] == d)[0]
            p, n = m[y[te][m] == 1], m[y[te][m] == 0]
            bidx_dom[d] = [np.concatenate([rng.choice(p, len(p)), rng.choice(n, len(n))]) for _ in range(n_boot)]
        for meth in methods:
            sv, st = scores[f["name"]][meth]
            thr0 = 0.0
            thr10 = float(np.quantile(sv[yv0], 0.9))
            P = per.setdefault(f["protocol"], {}).setdefault(meth, {})
            P["_val_" + f["name"]] = {"auroc_val": auroc(sv, y[va]), "thr10": thr10,
                                     "val_fa0": float((sv[yv0] > 0).mean())}
            for d in tdoms:
                m = dom[te] == d
                P[d] = dom_metrics(st[m], y[te][m], thr0, thr10)
            B = boot.setdefault(f["protocol"], {}).setdefault(meth, {})
            for b in range(n_boot):
                t10 = float(np.quantile(sv[yv0][bidx_val[b]], 0.9))
                for d in tdoms:
                    ii = bidx_dom[d][b]
                    s_, y_ = st[ii], y[te][ii]
                    c, w = s_[y_ == 0], s_[y_ == 1]
                    for k, v in (("auroc", auroc(s_, y_)), ("fa0", (c > 0).mean()), ("fa10", (c > t10).mean()),
                                 ("rec10", (w > t10).mean()), ("absdev10", abs((c > t10).mean() - 0.1))):
                        B.setdefault(k, {}).setdefault(d, np.zeros(n_boot))[b] = v
    summary = {}
    for prot, P in per.items():
        for meth, D in P.items():
            doms = [d for d in D if not d.startswith("_")]
            S = {k: float(np.mean([D[d][k] for d in doms])) for k in ("auroc", "fa0", "rec0", "fa10", "rec10", "ece")}
            S["absdev10"] = float(np.mean([abs(D[d]["fa10"] - 0.1) for d in doms]))
            S["max_fa10"] = float(np.max([D[d]["fa10"] for d in doms]))
            S["n_fa10_over_15"] = int(sum(D[d]["fa10"] > 0.15 for d in doms))
            S["n_domains"] = len(doms)
            summary.setdefault(prot, {})[meth] = S
    diffs = {}
    for prot, Bm in boot.items():
        for base in [m for m in methods if m != "planted"]:
            for k in ("auroc", "fa0", "fa10", "rec10", "absdev10"):
                pl = np.mean([v for v in Bm["planted"][k].values()], 0)
                bs = np.mean([v for v in Bm[base][k].values()], 0)
                dd = pl - bs
                point = summary[prot]["planted"].get(k, None)
                pb = summary[prot][base].get(k, None)
                diffs.setdefault(prot, {}).setdefault(f"planted_minus_{base}", {})[k] = {
                    "diff": None if point is None else point - pb,
                    "ci95": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))]}
    return per, summary, diffs


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-per", type=int, default=10)
    ap.add_argument("--folds", default="", help="comma list of fold names to run (default all)")
    a = ap.parse_args()
    try:
        import peft  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)
    import torch
    from transformers import AutoTokenizer
    if a.smoke:
        a.epochs, a.boot = 1, 50
    t0 = time.time()
    rows = build_rows(a)
    y = np.array([r["label"] for r in rows])
    dom = np.array([r["domain"] for r in rows])
    fmt = np.array([r["format"] for r in rows])
    log(stage="corpus", n=len(rows), domains=len(set(dom)), wrong=int(y.sum()), min=(time.time() - t0) / 60)
    tok = AutoTokenizer.from_pretrained(a.model)
    toks = tokenise(rows, tok, a.max_tokens)
    lens = np.array([len(t[0]) for t in toks])
    log(stage="tokenised", max_tokens=a.max_tokens, median_len=float(np.median(lens)), p90=float(np.percentile(lens, 90)),
        frac_at_cap=float((lens >= a.max_tokens).mean()))

    u = np.random.default_rng(12345).standard_normal(2048).astype(np.float32)
    u /= np.linalg.norm(u)

    # folds, with a domain-stratified validation split of the training domains
    folds = []
    for name, prot, test_mask in [(f"lofo_{f}", "lofo", fmt == f) for f in FORMATS] + \
                                 [(f"lodo_{d}", "lodo", dom == d) for d in LODO]:
        seed = sum(map(ord, name))
        rng = np.random.default_rng(seed)
        tr, va = [], []
        for d in sorted(set(dom[~test_mask])):
            ii = rng.permutation(np.where((dom == d) & ~test_mask)[0])
            k = int(round(a.val_frac * len(ii)))
            va += ii[:k].tolist()
            tr += ii[k:].tolist()
        folds.append({"name": name, "protocol": prot, "train": sorted(tr), "val": sorted(va),
                      "test": np.where(test_mask)[0].tolist(), "seed": seed})
    if a.smoke:
        folds = [f for f in folds if f["name"] in ("lofo_chat", "lodo_pb_math")]
    if a.folds:
        folds = [f for f in folds if f["name"] in a.folds.split(",")]

    ngpu = max(1, torch.cuda.device_count())
    devs = [f"cuda:{i}" for i in range(ngpu)]
    log(stage="setup", gpus=ngpu, folds=[(f["name"], len(f["train"]), len(f["val"]), len(f["test"])) for f in folds])

    # phase 1: frozen features for every reply (split across GPUs)
    hf_path = os.path.join(OUT, "frozen_hmean.npy")
    if os.path.exists(hf_path):
        Hf = np.load(hf_path)
    else:
        Hf = np.zeros((len(rows), 2048), np.float32)
        parts = np.array_split(np.arange(len(rows)), ngpu)

        def ext(i):
            bb = load_backbone(a.model, a.layer, devs[i])
            Hf[parts[i]] = extract(bb, toks, parts[i].tolist(), devs[i])
            del bb
            torch.cuda.empty_cache()
        th = [threading.Thread(target=ext, args=(i,)) for i in range(ngpu)]
        [t.start() for t in th]
        [t.join() for t in th]
        np.save(hf_path, Hf)
    s0 = Hf @ u
    log(stage="frozen", min=(time.time() - t0) / 60, written_score_mean=float(s0.mean()), written_score_sd=float(s0.std()),
        hmean_norm=float(np.linalg.norm(Hf, axis=1).mean()))

    # phase 2: one LoRA per fold, folds spread over the GPUs (one thread per GPU)
    os.makedirs(os.path.join(OUT, "folds"), exist_ok=True)
    queue = sorted(folds, key=lambda f: -len(f["train"]))
    loads = [[] for _ in devs]
    tot = [0] * len(devs)
    for f in queue:
        g = int(np.argmin(tot))
        loads[g].append(f)
        tot[g] += len(f["train"])
    results, errors = {}, []

    def work(g):
        for f in loads[g]:
            p = os.path.join(OUT, "folds", f["name"] + ".npz")
            try:
                if os.path.exists(p):
                    z = np.load(p, allow_pickle=True)
                    results[f["name"]] = {k: z[k] for k in z.files}
                    continue
                r = train_fold(a, f, toks, y, u, devs[g])
                np.savez(p, ev_idx=r["ev_idx"], H_tuned=r["H_tuned"].astype(np.float32), curve=np.array(r["curve"]),
                         nan=r["nan"], minutes=r["minutes"])
                results[f["name"]] = r
                log(stage="fold_done", fold=f["name"], minutes=r["minutes"], curve=r["curve"])
            except Exception as e:  # keep the other folds going
                import traceback
                errors.append((f["name"], repr(e)))
                log(stage="fold_error", fold=f["name"], err=traceback.format_exc()[-2000:])
    th = [threading.Thread(target=work, args=(g,)) for g in range(len(devs))]
    [t.start() for t in th]
    [t.join() for t in th]

    # phase 3: baselines, metrics, bootstrap
    done = [f for f in folds if f["name"] in results]
    if not done:
        raise SystemExit(f"no fold finished: {errors}")
    scores = {f["name"]: fold_scores(f, results[f["name"]], Hf, y, dom, u, a.l2) for f in done}
    np.savez(os.path.join(OUT, "scores.npz"), **{f"{fn}__{m}__{part}": v[j] for fn, S in scores.items()
                                                 for m, v in S.items() for j, part in enumerate(("val", "test"))},
             y=y, dom=dom, fmt=fmt)
    per, summary, diffs = evaluate(done, scores, y, dom, a.boot)
    out = {"config": vars(a), "written_probe": {"layer": a.layer, "pool": "mean over reply tokens",
                                                "w": "random unit vector, numpy default_rng(12345), d=2048",
                                                "readout": "score=<h_mean,w>, p=sigmoid(score), threshold 0"},
           "n_rows": len(rows), "token_lengths": {"median": float(np.median(lens)), "frac_at_cap": float((lens >= a.max_tokens).mean())},
           "folds": {f["name"]: {"n_train": len(f["train"]), "n_val": len(f["val"]), "n_test": len(f["test"]),
                                 "train_curve": np.asarray(results[f["name"]]["curve"]).tolist(),
                                 "minutes": float(results[f["name"]]["minutes"])} for f in done},
           "errors": errors, "summary": summary, "diffs_bootstrap": diffs, "per_domain": per,
           "minutes_total": (time.time() - t0) / 60}
    json.dump(out, open(os.path.join(OUT, "planted_channel.json"), "w"), indent=1)
    log(stage="done", summary=summary, min=(time.time() - t0) / 60)


if __name__ == "__main__":
    main()
