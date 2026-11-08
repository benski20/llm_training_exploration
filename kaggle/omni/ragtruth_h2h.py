"""Kaggle kernel: the omni verifier head-to-head on the official RAGTruth benchmark (2x T4, fp16).

Protocol (as the published detectors are scored):
    train   wandb/RAGTruth-processed train split (15,090 answers; a random order, as many as the time budget reads)
    test    the full official test split (2,700 answers: 900 QA, 900 Summary, 900 Data2txt), always read first
    metric  response-level precision / recall / F1 (an answer is hallucinated if it has any labelled span),
            overall and by task; AUROC; 95% bootstrap intervals (test answers resampled)
    threshold  chosen on held-out TRAINING answers only (5-fold out-of-fold scores), never on test

Readers: Qwen/Qwen3-8B (fp16, layers split over both T4s), then Qwen/Qwen3-4B (one worker per T4) if time allows.
One forward pass per answer through the product's reader (vt.omni.Reader.features, unchanged: documents + question
+ reply + the judge question appended after the reply); in the same pass forward hooks read the MLP neurons
(down_proj inputs) of three middle layers, averaged per sentence (experiments/ragtruth_sota.py's feature, without
its second, question-only pass: its evidence signals added nothing there). The NLI expert (vt.omni.NLIScorer:
HHEM-2.1-open at its pinned revision + MiniCheck-RoBERTa-L) runs on GPU 1 alongside.

Experts (each fitted on training answers only):
    reply     omni reply head: PCA-256 of [mean residual, judge state] at the 3 middle layers + output signals +
              judge log-odds -> class-balanced logistic regression (per response)
    unit      omni unit head: PCA-256 of a sentence's mean residual + its output signals -> logistic on sentence
              labels; response score = worst sentence
    neurons   MLP-neuron sentence probe (ragtruth_sota.py): logistic on sentence labels; response = worst sentence
    hhem, minicheck, judge (-log-odds of "Yes", no training)
    stacks    class-balanced logistic on out-of-fold expert scores + task one-hot

    python kaggle/kgrun.py --slug ragtruth-h2h --script kaggle/omni/ragtruth_h2h.py --bundle <files> \
        --out results/kaggle/ragtruth-h2h --wait
    (--args "--smoke-only" runs the gated smoke stage alone)

Outputs in /kaggle/working: results_<tag>.json, scores_<tag>.npz (per-answer expert scores: train out-of-fold and
test), timing, logs. Features stay in /tmp.
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

SRC = "/kaggle/working/src"
WORK = "/kaggle/working"
TMP = "/tmp/h2h"
SELF = os.path.join(SRC, "kaggle", "omni", "ragtruth_h2h.py")
READERS = {"q8b": ("Qwen/Qwen3-8B", 2), "q4b": ("Qwen/Qwen3-4B", 1)}   # tag: (model, GPUs per worker)
PUBLISHED = {"RAG-HAT (fine-tuned, reported)": 83.9, "LettuceDetect-large (fine-tuned ModernBERT)": 79.2,
             "Llama-2-13B fine-tuned (RAGTruth paper)": 78.7, "GPT-4-turbo prompted (RAGTruth paper)": 63.4}
MID = [2, 3, 4]                # the omni reply/unit head's middle layers (indices into Reader.layers)
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def jdump(o, path):
    json.dump(o, open(path, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


# ===================================================================== items

def build_items(path):
    from datasets import load_dataset
    out = []
    for split in ("test", "train"):
        rows = list(load_dataset("wandb/RAGTruth-processed", split=split))  # nosec B615: benchmark data
        if split == "train":
            random.Random(0).shuffle(rows)   # nosec B311: a random order, so a time-cut prefix keeps the mix
        for x in rows:
            sp = [(int(h["start"]), int(h["end"])) for h in json.loads(x["hallucination_labels"] or "[]")]
            out.append({"g": len(out), "split": split, "id": str(x["id"]), "task": x["task_type"], "gen": x["model"],
                        "user": x["input_str"], "doc": x["context"], "query": x["query"], "response": x["output"],
                        "spans": sp, "y": int(bool(sp))})
    with open(path, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    return out


def read_items(path):
    return [json.loads(l) for l in open(path)]


# ===================================================================== worker: the reader

def reader_worker(cfg):
    import numpy as np
    import torch
    sys.path[:0] = [SRC]
    from vt import omni
    items = read_items(cfg["items"])
    idx = cfg["indices"]
    R = omni.Reader(cfg["model"], max_tokens=cfg["max_tokens"])
    model, tok = R.model, R.tok
    log("loaded", cfg["model"], "layers", R.layers, "devices", sorted({str(p.device) for p in model.parameters()}),
        "dtype", next(model.parameters()).dtype)
    neu_layers = [R.layers[i] - 1 for i in (1, 2, 3)]     # decoder layers whose outputs are hidden_states 14/20/25
    st = {"sl": None, "buf": {}}

    def mk(li):
        def pre(mod, args):
            if st["sl"] is not None:
                a, b = st["sl"]
                st["buf"][li] = args[0][0, a:b]
        return pre

    for li in neu_layers:
        model.model.layers[li].mlp.down_proj.register_forward_pre_hook(mk(li))
    os.makedirs(cfg["out"], exist_ok=True)
    jdump({"model": cfg["model"], "layers": R.layers, "mid_layers": [R.layers[i] for i in MID],
           "neuron_layers": neu_layers, "max_tokens": cfg["max_tokens"],
           "d": model.config.hidden_size, "inter": model.config.intermediate_size}, os.path.join(cfg["out"], "meta.json"))
    stats = {"n": 0, "skipped": [], "oom_retries": 0, "nonfinite": 0}
    t_start = time.time()
    S = cfg["shard"]
    for s0 in range(0, len(idx), S):
        if (s0 // S) % cfg["nw"] != cfg["w"]:
            continue
        path = os.path.join(cfg["out"], f"shard_{s0:06d}.npz")
        if os.path.exists(path):
            continue
        if time.time() > cfg["deadline"]:
            log("deadline reached at position", s0)
            break
        A = {k: [] for k in ("g", "r3", "j3", "rsc", "jl", "t", "ntok")}
        U = {k: [] for k in ("g", "k", "y", "u3", "sc", "neu")}
        for g in idx[s0:s0 + S]:
            x = items[g]
            it = omni.make_item(x["response"], messages=[{"role": "system", "content": omni.RAG_SYSTEM},
                                                        {"role": "user", "content": x["user"]}])
            f = None
            for mt in (cfg["max_tokens"], 2048, 1536):
                R.max_tokens = mt
                ids, r0, roff = omni.encode(tok, it, mt, R.suffix)
                n = len(ids) - r0 - len(R.suffix)
                st["sl"], st["buf"] = (r0, r0 + n), {}
                try:
                    t = time.time()
                    f = R.features(it)
                    neu = []
                    if f is not None:
                        tu = omni.token_units(roff[:n], it["units"])
                        for u in f["units"]:
                            pos = torch.from_numpy(np.nonzero(tu == u["index"])[0])
                            neu.append(torch.cat([st["buf"][li][pos.to(st["buf"][li].device)].float().mean(0).cpu()
                                                  for li in neu_layers]).half().numpy())
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dt = time.time() - t
                    break
                except torch.cuda.OutOfMemoryError:
                    f = None
                    st["buf"] = {}
                    torch.cuda.empty_cache()
                    stats["oom_retries"] += 1
                    log("OOM at", mt, "tokens, item", g)
                except Exception as e:                       # noqa: BLE001: one bad answer must not stop the run
                    f = None
                    log("error on item", g, repr(e)[:300])
                    break
            R.max_tokens = cfg["max_tokens"]
            st["sl"], st["buf"] = None, {}
            if f is None:
                stats["skipped"].append(g)
                continue
            if not (np.isfinite(f["r_mean"]).all() and np.isfinite(f["j_state"]).all()):
                stats["nonfinite"] += 1
            A["g"].append(g)
            A["r3"].append(f["r_mean"][MID].astype(np.float16))
            A["j3"].append(f["j_state"][MID].astype(np.float16))
            A["rsc"].append(f["r_sc"])
            A["jl"].append(f["j_logit"])
            A["t"].append(dt)
            A["ntok"].append(len(ids))
            for u, v in zip(f["units"], neu):
                a, b = it["units"][u["index"]]
                U["g"].append(g)
                U["k"].append(u["index"])
                U["y"].append(int(any(s < b and e > a for s, e in x["spans"])))
                U["u3"].append(u["mean"][MID].reshape(-1).astype(np.float16))
                U["sc"].append(u["sc"])
                U["neu"].append(v)
        if not A["g"]:
            continue
        np.savez(path, g=np.array(A["g"]), r3=np.stack(A["r3"]), j3=np.stack(A["j3"]),
                 rsc=np.stack(A["rsc"]).astype(np.float32), jl=np.array(A["jl"], np.float32),
                 t=np.array(A["t"], np.float32), ntok=np.array(A["ntok"]),
                 s_g=np.array(U["g"]), s_k=np.array(U["k"]), s_y=np.array(U["y"]),
                 s_u3=np.stack(U["u3"]), s_sc=np.stack(U["sc"]).astype(np.float32), s_neu=np.stack(U["neu"]))
        stats["n"] += len(A["g"])
        el = time.time() - t_start
        log(f"w{cfg['w']} shard {s0}: {stats['n']} answers in {el / 60:.1f} min ({el / stats['n']:.2f} s/answer)")
    stats["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    stats["peak_mem_gb"] = [torch.cuda.max_memory_allocated(d) / 1e9 for d in range(torch.cuda.device_count())]
    jdump(stats, os.path.join(cfg["out"], f"stats_w{cfg['w']}.json"))
    log("reader worker done", json.dumps({k: v for k, v in stats.items() if k != "skipped"}), "skipped", len(stats["skipped"]))


# ===================================================================== worker: NLI

def nli_worker(cfg):
    import numpy as np
    import torch
    sys.path[:0] = [SRC]
    from vt.omni import NLIScorer, sentence_spans

    class GPUNLI(NLIScorer):
        """vt.omni.NLIScorer (HHEM revision pinned there), on the GPU, returning both experts."""

        def __init__(self, dev):
            super().__init__()
            self.dev = dev
            self.hhem.to(dev)
            self.mc.to(dev)

        def _mc(self, doc, claim):
            w = doc.split()
            chunks = [" ".join(w[i:i + self.chunk_words]) for i in range(0, max(1, len(w)), self.chunk_words)] or [""]
            x = self.mtok(chunks, [claim] * len(chunks), return_tensors="pt", padding=True, truncation="only_first",
                          max_length=512).to(self.dev)
            with torch.no_grad():
                p = torch.softmax(self.mc(**x).logits, -1)[:, 1]
            return float(1 - p.max())

        def both(self, doc, question, reply):          # = NLIScorer.__call__, the two halves kept apart
            doc = doc[:12000]
            claim = f"{question[-300:]} {reply}" if len(reply.split()) < 8 else reply
            sents = [reply[a:b].strip() for a, b in sentence_spans(reply)][:self.max_sentences] or [reply]
            m = max(self._mc(doc, s_) for s_ in sents) if len(sents) > 1 else self._mc(doc, claim)
            with torch.no_grad():
                h = 1 - float(self.hhem.predict([(doc, claim)])[0])
            return h, m

    items = read_items(cfg["items"])
    idx = cfg["indices"]
    nli = GPUNLI("cuda" if torch.cuda.is_available() else "cpu")
    log("NLI loaded on", nli.dev)
    os.makedirs(cfg["out"], exist_ok=True)
    S, t_start, n = 200, time.time(), 0
    for s0 in range(0, len(idx), S):
        path = os.path.join(cfg["out"], f"nli_{s0:06d}.npz")
        if os.path.exists(path):
            continue
        if time.time() > cfg["deadline"]:
            break
        G, H, M, T = [], [], [], []
        for g in idx[s0:s0 + S]:
            x = items[g]
            t = time.time()
            try:
                h, m = nli.both(x["doc"], x["query"], x["response"])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                h, m = float("nan"), float("nan")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            G.append(g)
            H.append(h)
            M.append(m)
            T.append(time.time() - t)
        np.savez(path, g=np.array(G), hhem=np.array(H, np.float32), mc=np.array(M, np.float32), t=np.array(T, np.float32))
        n += len(G)
        if (s0 // S) % 10 == 0:
            log(f"NLI {n} answers in {(time.time() - t_start) / 60:.1f} min")
    log("NLI worker done", n)


# ===================================================================== fit and score

def fit(cfg):
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    t_fit = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = read_items(cfg["items"])
    meta = json.load(open(os.path.join(cfg["feats"], "meta.json")))
    files = sorted(glob.glob(os.path.join(cfg["feats"], "shard_*.npz")))
    # ---- load (preallocated: the neuron block is large)
    sizes = []
    for p in files:
        z = np.load(p)
        sizes.append((len(z["g"]), len(z["s_g"]), z["r3"].shape[1:], z["s_u3"].shape[1], z["s_neu"].shape[1]))
    NA, NS = sum(s[0] for s in sizes), sum(s[1] for s in sizes)
    dU, dN = sizes[0][3], sizes[0][4]
    log(f"fit {cfg['tag']}: {NA} answers, {NS} sentences, unit dim {dU}, neuron dim {dN} "
        f"({NS * dN * 2 / 1e9:.1f} GB)")
    A = {"g": np.zeros(NA, int), "r3": np.zeros((NA,) + sizes[0][2], np.float16), "j3": np.zeros((NA,) + sizes[0][2], np.float16),
         "rsc": np.zeros((NA, 10), np.float32), "jl": np.zeros(NA, np.float32), "t": np.zeros(NA, np.float32),
         "ntok": np.zeros(NA, int)}
    Sg, Sy = np.zeros(NS, int), np.zeros(NS, int)
    Su = np.zeros((NS, dU), np.float16)
    Ssc = np.zeros((NS, 10), np.float32)
    Sn = np.zeros((NS, dN), np.float16)
    ia = isn = 0
    for p in files:
        z = np.load(p)
        a, s = len(z["g"]), len(z["s_g"])
        for k in A:
            A[k][ia:ia + a] = z[k]
        Sg[isn:isn + s], Sy[isn:isn + s], Su[isn:isn + s], Ssc[isn:isn + s], Sn[isn:isn + s] = \
            z["s_g"], z["s_y"], z["s_u3"], z["s_sc"], z["s_neu"]
        ia, isn = ia + a, isn + s
    order = np.argsort(A["g"])
    A = {k: v[order] for k, v in A.items()}
    pos_of = {g: i for i, g in enumerate(A["g"])}
    split = np.array([items[g]["split"] for g in A["g"]])
    task = np.array([items[g]["task"] for g in A["g"]])
    y = np.array([items[g]["y"] for g in A["g"]])
    tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
    test_all = [it["g"] for it in items if it["split"] == "test"]
    missing_test = sorted(set(test_all) - set(A["g"][te].tolist()))
    s_ans = np.array([pos_of[g] for g in Sg])               # sentence -> answer position
    # NLI
    nli = {}
    for p in glob.glob(os.path.join(cfg["nli"], "nli_*.npz")):
        z = np.load(p)
        for g, h, m, t in zip(z["g"], z["hhem"], z["mc"], z["t"]):
            nli[int(g)] = (float(h), float(m), float(t))
    hh = np.array([nli.get(int(g), (np.nan,) * 3)[0] for g in A["g"]])
    mc = np.array([nli.get(int(g), (np.nan,) * 3)[1] for g in A["g"]])
    nli_t = np.array([nli.get(int(g), (np.nan,) * 3)[2] for g in A["g"]])
    # folds over training answers
    K = cfg.get("folds", 5)
    fold = np.full(NA, -1)
    fold[tr[np.random.default_rng(0).permutation(len(tr))]] = np.arange(len(tr)) % K
    s_tr = fold[s_ans] >= 0
    log(f"train {len(tr)} (hallucinated {y[tr].mean():.3f}), test {len(te)} (missing {len(missing_test)}), "
        f"NLI on {np.isfinite(hh).sum()} answers")

    # ---- metrics
    def prf(flag, yy):
        tp, fp, fn = int((flag & (yy == 1)).sum()), int((flag & (yy == 0)).sum()), int((~flag & (yy == 1)).sum())
        p_, r_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
        return {"precision": round(100 * p_, 1), "recall": round(100 * r_, 1),
                "f1": round(100 * 2 * p_ * r_ / max(1e-9, p_ + r_), 1), "n": int(len(yy)), "positives": int(yy.sum())}

    def f1v(flag, yy):
        tp, fp, fn = (flag & (yy == 1)).sum(), (flag & (yy == 0)).sum(), (~flag & (yy == 1)).sum()
        return 2 * tp / max(1, 2 * tp + fp + fn)

    def best_thr(sc, yy):
        o = np.argsort(-sc)
        s_, y_ = sc[o], yy[o]
        tp = np.cumsum(y_)
        k = np.arange(1, len(y_) + 1)
        f = 2 * tp / (k + y_.sum())
        last = np.r_[s_[1:] != s_[:-1], True]               # only cut between distinct scores
        f = np.where(last, f, -1)
        i = int(np.argmax(f))
        return float(s_[i]), float(f[i])

    def auc(sc, yy):
        return float(roc_auc_score(yy, sc)) if 0 < yy.sum() < len(yy) else None

    rng_b = np.random.default_rng(1)
    B = cfg.get("boot", 2000)
    boot_idx = [rng_b.integers(0, len(te), len(te)) for _ in range(B)]

    def evaluate(name, oof, test, info=None):
        """oof: out-of-fold scores on the training answers (positions tr); test: scores on test answers (te)."""
        thr, f_tr = best_thr(oof, y[tr])
        yt = y[te]
        flag = test >= thr
        row = {"train_oof": {"f1": round(100 * f_tr, 1), "auroc": round(auc(oof, y[tr]), 4)},
               "threshold": thr, "response": dict(prf(flag, yt), auroc=round(auc(test, yt), 4))}
        bf = np.array([f1v(flag[i], yt[i]) for i in boot_idx]) * 100
        ba = np.array([auc(test[i], yt[i]) for i in boot_idx[:1000]])
        row["response"]["f1_ci95"] = [round(float(np.percentile(bf, 2.5)), 1), round(float(np.percentile(bf, 97.5)), 1)]
        row["response"]["auroc_ci95"] = [round(float(np.percentile(ba, 2.5)), 4), round(float(np.percentile(ba, 97.5)), 4)]
        row["by_task"] = {}
        flag_pt = np.zeros(len(te), bool)
        for t in ("QA", "Summary", "Data2txt"):
            mt, mtr = task[te] == t, task[tr] == t
            row["by_task"][t] = dict(prf(flag[mt], yt[mt]), auroc=round(auc(test[mt], yt[mt]) or float("nan"), 4))
            tt, _ = best_thr(oof[mtr], y[tr][mtr])
            flag_pt[mt] = test[mt] >= tt
            row["by_task"][t]["per_task_threshold"] = prf(flag_pt[mt], yt[mt])
        row["per_task_threshold"] = prf(flag_pt, yt)
        bpt = np.array([f1v(flag_pt[i], yt[i]) for i in boot_idx]) * 100
        row["per_task_threshold"]["f1_ci95"] = [round(float(np.percentile(bpt, 2.5)), 1),
                                                round(float(np.percentile(bpt, 97.5)), 1)]
        if info:
            row["info"] = info
        res["experts"][name] = row
        scores[name] = (oof, test)
        r = row["response"]
        log(f"{name:28s} test F1 {r['f1']} {r['f1_ci95']} P {r['precision']} R {r['recall']} AUROC {r['auroc']} | "
            f"train-OOF F1 {row['train_oof']['f1']} AUROC {row['train_oof']['auroc']} | by task "
            + " ".join(f"{t} {v['f1']}" for t, v in row["by_task"].items()))
        jdump(res, cfg["report"])

    res = {"reader": meta["model"], "meta": meta, "train_answers": int(len(tr)), "train_available": 15090,
           "test_answers": int(len(te)), "test_missing": missing_test, "test_hallucinated": int(y[te].sum()),
           "train_hallucinated_frac": round(float(y[tr].mean()), 4), "train_sentences": int(s_tr.sum()),
           "test_sentences": int((~s_tr).sum()), "folds": K, "published_response_f1": PUBLISHED, "experts": {}}
    scores = {}

    # ---- helpers
    def resp_max(ps):
        out = np.full(NA, -np.inf)
        np.maximum.at(out, s_ans, ps)
        fin = np.isfinite(out)
        out[~fin] = out[fin].min() if fin.any() else 0
        return out

    def pca_fit(Xtr_rows, k):
        """standardise + top-k principal directions, on the GPU; Xtr_rows: float16 numpy (n, d)."""
        Xt = torch.from_numpy(Xtr_rows).to(dev).float()
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-4
        Z = (Xt - mu) / sd
        pm = Z.mean(0)
        _, _, V = torch.pca_lowrank(Z - pm, q=min(k, Z.shape[1], Z.shape[0]), center=False, niter=4)
        del Xt, Z
        return mu, sd, pm, V[:, :k]

    def pca_apply(X, P, chunk=8192):
        mu, sd, pm, V = P
        out = []
        for i in range(0, len(X), chunk):
            Z = (torch.from_numpy(X[i:i + chunk]).to(dev).float() - mu) / sd
            out.append(((Z - pm) @ V).cpu().numpy())
        return np.concatenate(out)

    def sk_lr(X, yy, C):
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X), yy)
        return lambda Z: m.decision_function(sc.transform(Z))

    def crossfit(train_fn, score_fn, rows_of_fold, grid, select_on="auroc"):
        """grid search on fold 0 (fit on folds 1..K-1), then out-of-fold + full fit with the chosen setting.
        train_fn(rows, C) -> model; score_fn(model) -> response scores for all answers."""
        sel = {}
        fold0 = {}
        for C in grid:
            m = train_fn(rows_of_fold(lambda f: f > 0), C)
            sc = score_fn(m)
            v = tr[fold[tr] == 0]
            sel[C] = {"auroc": auc(sc[v], y[v]), "f1": best_thr(sc[v], y[v])[1]}
            fold0[C] = sc
        C = max(grid, key=lambda c: (sel[c][select_on] or 0))
        oof = np.zeros(NA)
        v0 = fold[tr] == 0
        oof[tr[v0]] = fold0[C][tr[v0]]
        for k in range(1, K):
            m = train_fn(rows_of_fold(lambda f, k=k: f != k), C)
            vk = tr[fold[tr] == k]
            oof[vk] = score_fn(m)[vk]
        full = score_fn(train_fn(rows_of_fold(lambda f: f >= 0), C))
        return oof[tr], full[te], {"C": C, "fold0_selection": {str(c): v for c, v in sel.items()}}

    ans_rows = lambda cond: tr[cond(fold[tr])]                              # noqa: E731
    sen_rows = lambda cond: np.where(s_tr & cond(fold[s_ans]))[0]          # noqa: E731

    # ---- zero-training experts
    evaluate("judge (reader's own, no training)", -A["jl"][tr], -A["jl"][te])
    if np.isfinite(hh).any():
        nm = lambda v: np.where(np.isfinite(v), v, np.nanmean(v[tr]))     # noqa: E731
        hh_, mc_ = nm(hh), nm(mc)
        evaluate("HHEM-2.1-open", hh_[tr], hh_[te])
        evaluate("MiniCheck-RoBERTa-L", mc_[tr], mc_[te])
        evaluate("NLI mean (product)", ((hh_ + mc_) / 2)[tr], ((hh_ + mc_) / 2)[te])

    # ---- omni reply head
    t1 = time.time()
    H = np.concatenate([A["r3"].reshape(NA, -1), A["j3"].reshape(NA, -1)], 1)
    P = pca_fit(H[tr], 256)
    XR = np.concatenate([pca_apply(H, P), A["rsc"], A["jl"][:, None]], 1)
    del H, P
    o, t, info = crossfit(lambda rows, C: sk_lr(XR[rows], y[rows], C), lambda m: m(XR), ans_rows,
                          cfg.get("grid_lr", (0.001, 0.01, 0.1, 1.0)))
    info["seconds"] = round(time.time() - t1)
    evaluate("omni reply head", o, t, info)

    # ---- omni unit head -> worst sentence
    t1 = time.time()
    sub = np.where(s_tr)[0]
    sub = np.random.default_rng(0).permutation(sub)[:20000]
    P = pca_fit(Su[np.sort(sub)], 256)
    XU = np.concatenate([pca_apply(Su, P), Ssc], 1)
    del P
    o, t, info = crossfit(lambda rows, C: sk_lr(XU[rows], Sy[rows], C), lambda m: resp_max(m(XU)), sen_rows,
                          cfg.get("grid_lr", (0.001, 0.01, 0.1, 1.0)))
    info["seconds"] = round(time.time() - t1)
    evaluate("omni worst unit", o, t, info)
    del XU

    # ---- MLP-neuron sentence probe (ragtruth_sota.py) -> worst sentence
    t1 = time.time()
    on_gpu = dev == "cuda" and Sn.nbytes < 11.5e9
    Xn = torch.from_numpy(Sn).to(dev) if on_gpu else torch.from_numpy(Sn)
    trs = np.where(s_tr)[0]
    mu = torch.zeros(dN, dtype=torch.float64, device=dev)
    sq = torch.zeros(dN, dtype=torch.float64, device=dev)
    for i in range(0, len(trs), 4096):
        c = Xn[torch.from_numpy(trs[i:i + 4096]).to(Xn.device)].to(dev).double()
        mu += c.sum(0)
        sq += (c ** 2).sum(0)
    mu /= len(trs)
    sd = torch.sqrt(torch.clamp(sq / len(trs) - mu ** 2, min=1e-8))
    mu, sd = mu.float(), sd.float()

    def batch(rows):
        return (Xn[torch.from_numpy(rows).to(Xn.device)].to(dev, non_blocking=True).float() - mu) / sd

    def neu_train(rows, C, epochs=cfg.get("epochs", 6), lr=1e-3, bs=512):
        g = torch.Generator().manual_seed(0)
        w = torch.zeros(dN, device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        yy = torch.tensor(Sy[rows], dtype=torch.float32)
        p = float(yy.mean())
        wt = torch.tensor([0.5 / max(1 - p, 1e-6), 0.5 / max(p, 1e-6)])
        l2 = 1.0 / (C * len(rows))
        for _ in range(epochs):
            for ix in torch.randperm(len(rows), generator=g).split(bs):
                xb = batch(rows[ix.numpy()])
                yb, wb = yy[ix].to(dev), wt[yy[ix].long()].to(dev)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(xb @ w + b, yb, reduction="none") * wb).mean() \
                    + l2 * (w ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return w.detach(), b.detach()

    def neu_score(m):
        w, b = m
        out = []
        with torch.no_grad():
            for i in range(0, NS, 8192):
                out.append((batch(np.arange(i, min(NS, i + 8192))) @ w + b).cpu().numpy())
        return np.concatenate(out)

    sent_neu = {}

    def neu_resp(m):
        s = neu_score(m)
        sent_neu["last"] = s
        return resp_max(s)

    o, t, info = crossfit(neu_train, neu_resp, sen_rows, cfg.get("grid_neu", (0.001, 0.01, 0.1)))
    info.update(seconds=round(time.time() - t1), on_gpu=bool(on_gpu), dim=int(dN))
    te_s = ~s_tr
    if te_s.any():
        info["sentence_auroc_test"] = round(auc(sent_neu["last"][te_s], Sy[te_s]) or float("nan"), 4)
    evaluate("MLP-neuron sentence probe", o, t, info)
    del Xn
    torch.cuda.empty_cache() if dev == "cuda" else None

    # ---- stacks: class-balanced logistic on out-of-fold expert scores + task one-hot
    onehot = np.stack([(task == t_).astype(float) for t_ in ("QA", "Summary", "Data2txt")], 1)

    def stack(name, parts):
        if not all(p in scores for p in parts):
            return
        F = np.zeros((NA, len(parts)))
        for j, p in enumerate(parts):
            F[tr, j], F[te, j] = scores[p]
        F = np.concatenate([F, onehot], 1)
        oof = np.zeros(NA)
        for k in range(K):
            r_, v_ = tr[fold[tr] != k], tr[fold[tr] == k]
            oof[v_] = sk_lr(F[r_], y[r_], 1.0)(F[v_])
        full = sk_lr(F[tr], y[tr], 1.0)(F[te])
        evaluate(name, oof[tr], full, {"parts": parts})

    nli_parts = ["HHEM-2.1-open", "MiniCheck-RoBERTa-L"]
    om = ["omni reply head", "omni worst unit", "judge (reader's own, no training)"]
    stack("stack: omni (reply + unit + judge)", om)
    stack("stack: omni + NLI", om + nli_parts)
    stack("stack: neurons + NLI", ["MLP-neuron sentence probe"] + nli_parts)
    stack("stack: omni + neurons", om + ["MLP-neuron sentence probe"])
    stack("stack: all (omni + neurons + NLI)", om + ["MLP-neuron sentence probe"] + nli_parts)
    # feature-family choice and the headline, both by training answers only
    fam = {k: res["experts"][k]["train_oof"] for k in ("stack: omni (reply + unit + judge)", "MLP-neuron sentence probe",
                                                        "stack: omni + NLI", "stack: neurons + NLI") if k in res["experts"]}
    res["feature_choice_on_train_oof"] = fam
    head = max(res["experts"], key=lambda k: res["experts"][k]["train_oof"]["f1"])
    res["headline"] = {"chosen_by": "highest training out-of-fold F1 (test never used to choose)", "expert": head,
                       "test": res["experts"][head]["response"], "by_task": res["experts"][head]["by_task"]}
    # timing (test answers, one at a time, unbatched)
    tt = A["t"][te]
    gpw = cfg["gpus_per_worker"]
    nt = nli_t[te][np.isfinite(nli_t[te])]
    res["timing"] = {"gpu": "Tesla T4", "reader_wall_s_per_answer": {"mean": round(float(tt.mean()), 3),
                     "median": round(float(np.median(tt)), 3), "p95": round(float(np.percentile(tt, 95)), 3)},
                     "reader_gpus": gpw, "reader_t4_s_per_answer": round(float(tt.mean()) * gpw, 3),
                     "nli_t4_s_per_answer": round(float(nt.mean()), 3) if len(nt) else None,
                     "tokens_per_answer": {"mean": round(float(A["ntok"][te].mean())), "p95": int(np.percentile(A["ntok"][te], 95))},
                     "note": "batch 1; reader = one forward with hidden states, judge suffix and neuron hooks; "
                             "the NLI ran concurrently on GPU 1 (contention included)"}
    res["timing"]["total_t4_s_per_answer"] = round(res["timing"]["reader_t4_s_per_answer"] + (res["timing"]["nli_t4_s_per_answer"] or 0), 3)
    res["fit_seconds"] = round(time.time() - t_fit)
    jdump(res, cfg["report"])
    np.savez_compressed(cfg["scores"], g=A["g"], split=split, task=task, y=y, fold=fold, hhem=hh, mc=mc,
                        names=np.array(list(scores)), tr_pos=tr, te_pos=te,
                        oof=np.stack([scores[k][0] for k in scores]), test=np.stack([scores[k][1] for k in scores]))
    log("fit done", cfg["tag"], "headline", head, json.dumps(res["headline"]["test"]))


# ===================================================================== driver

def sh(cmd, logfile=None, env=None):
    log("$", " ".join(cmd))
    out = open(logfile, "a") if logfile else None
    r = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT if out else None, env=env)
    if out:
        out.close()
    return r.returncode


def spawn(role, cfg, gpus, logname):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               TOKENIZERS_PARALLELISM="false", H2H_ROLE=role, H2H_CFG=json.dumps(cfg))
    lf = open(os.path.join(WORK, logname), "a")
    return subprocess.Popen([sys.executable, SELF], env=env, stdout=lf, stderr=subprocess.STDOUT), lf


def extract(tag, items_path, indices, out, nli_out, deadline, max_tokens, with_nli):
    name, gpw = READERS[tag]
    ngpu = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
                if l.startswith("GPU")]) or 1
    nw = max(1, ngpu // gpw)
    procs = []
    for w in range(nw):
        cfg = {"items": items_path, "indices": indices, "model": name, "out": out, "deadline": deadline,
               "max_tokens": max_tokens, "shard": 50, "w": w, "nw": nw}
        gpus = ",".join(str(g) for g in range(w * gpw, (w + 1) * gpw))
        procs.append(spawn("reader", cfg, gpus, f"reader_{tag}_{os.path.basename(out)}_w{w}.log"))
    if with_nli:
        cfg = {"items": items_path, "indices": indices, "out": nli_out, "deadline": deadline}
        procs.append(spawn("nli", cfg, str(ngpu - 1), f"nli_{os.path.basename(nli_out)}.log"))
    rcs = [p.wait() for p, _ in procs]
    for _, lf in procs:
        lf.close()
    return rcs


def run_fit(tag, items_path, feats, nli_out, report, scores, gpw, extra=None, timeout=None):
    cfg = dict({"items": items_path, "feats": feats, "nli": nli_out, "report": report, "scores": scores, "tag": tag,
                "gpus_per_worker": gpw}, **(extra or {}))
    p, lf = spawn("fit", cfg, "0,1", f"fit_{tag}.log")
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = -9
    lf.close()
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--readers", default="q8b,q4b")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--budget-min", type=float, default=700, help="the kernel's own time budget (Kaggle stops at 720)")
    ap.add_argument("--q8b-extract-until", type=float, default=425, help="minutes after start")
    ap.add_argument("--q4b-min-left", type=float, default=150, help="start the 4B only with this many minutes left")
    a = ap.parse_args()
    import transformers
    log("transformers", transformers.__version__)
    sh(["nvidia-smi", "-L"])                                       # no CUDA context in the driver
    for mod, pkg in (("datasets", "datasets"), ("sklearn", "scikit-learn")):
        try:
            __import__(mod)
        except ImportError:
            sh([sys.executable, "-m", "pip", "install", "-q", pkg])
    os.makedirs(TMP, exist_ok=True)
    du = shutil.disk_usage("/tmp")
    log(f"disk /tmp free {du.free / 1e9:.0f} GB; /kaggle/working free {shutil.disk_usage(WORK).free / 1e9:.0f} GB")
    items_path = os.path.join(TMP, "items.jsonl")
    items = build_items(items_path)
    test_idx = [it["g"] for it in items if it["split"] == "test"]
    train_idx = [it["g"] for it in items if it["split"] == "train"]
    log("items: test", len(test_idx), "train", len(train_idx))
    end = T0 + a.budget_min * 60
    status = {"stages": {}}

    # ---- stage 0 (gate): smoke test, the whole pipeline on 80 test + 120 train answers with the 8B
    sm_idx = test_idx[:80] + train_idx[:120]
    rcs = extract("q8b", items_path, sm_idx, os.path.join(TMP, "smoke_q8b"), os.path.join(TMP, "smoke_nli"),
                  time.time() + 40 * 60, a.max_tokens, True)
    rc = run_fit("smoke", items_path, os.path.join(TMP, "smoke_q8b"), os.path.join(TMP, "smoke_nli"),
                 os.path.join(WORK, "results_smoke.json"), os.path.join(WORK, "scores_smoke.npz"), 2,
                 {"folds": 3, "boot": 200, "epochs": 2, "grid_lr": (0.1,), "grid_neu": (0.01,)}, timeout=20 * 60)
    ok = all(r == 0 for r in rcs) and rc == 0 and os.path.exists(os.path.join(WORK, "scores_smoke.npz"))
    status["stages"]["smoke"] = {"extract_rc": rcs, "fit_rc": rc, "ok": ok, "minutes": round((time.time() - T0) / 60, 1)}
    jdump(status, os.path.join(WORK, "status.json"))
    log("smoke", "OK" if ok else "FAILED", status["stages"]["smoke"])
    if not ok or a.smoke_only:
        return
    shutil.rmtree(os.path.join(TMP, "smoke_q8b"), ignore_errors=True)

    # ---- the readers
    nli_out = os.path.join(TMP, "nli")
    for tag in a.readers.split(","):
        left = (end - time.time()) / 60
        if tag != "q8b" and left < a.q4b_min_left:
            log(f"skip {tag}: {left:.0f} min left")
            status["stages"][tag] = {"skipped": f"{left:.0f} min left"}
            break
        feats = os.path.join(TMP, f"x_{tag}")
        if tag == "q8b":
            deadline = T0 + a.q8b_extract_until * 60
        else:
            deadline = end - 55 * 60                               # leave the fit its time
        t = time.time()
        rcs = extract(tag, items_path, test_idx + train_idx, feats, nli_out, deadline, a.max_tokens,
                      with_nli=not glob.glob(os.path.join(nli_out, "nli_*.npz")))
        log(tag, "extracted, rc", rcs, f"{(time.time() - t) / 60:.0f} min")
        timeout = max(10 * 60, end - time.time() - 5 * 60)
        rc = run_fit(tag, items_path, feats, nli_out, os.path.join(WORK, f"results_{tag}.json"),
                     os.path.join(WORK, f"scores_{tag}.npz"), READERS[tag][1], timeout=timeout)
        for p in glob.glob(os.path.join(feats, "stats_w*.json")) + glob.glob(os.path.join(feats, "meta.json")):
            shutil.copy(p, os.path.join(WORK, f"{tag}_{os.path.basename(p)}"))
        status["stages"][tag] = {"extract_rc": rcs, "fit_rc": rc, "minutes": round((time.time() - T0) / 60, 1)}
        jdump(status, os.path.join(WORK, "status.json"))
        shutil.rmtree(feats, ignore_errors=True)                   # free /tmp for the next reader
        hub = os.path.expanduser("~/.cache/huggingface/hub/models--" + READERS[tag][0].replace("/", "--"))
        shutil.rmtree(hub, ignore_errors=True)
    log("all done", json.dumps(status))


if __name__ == "__main__":
    role = os.environ.get("H2H_ROLE")
    if role:
        sys.path[:0] = [SRC]
        c = json.loads(os.environ["H2H_CFG"])
        {"reader": reader_worker, "nli": nli_worker, "fit": fit}[role](c)
    else:
        main()
