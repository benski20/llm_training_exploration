"""Toward state of the art on RAGTruth, from a frozen reader's internals: no fine-tuned detector, no LLM judge.

Published response-level F1 on RAGTruth's test set: a fine-tuned Llama-2-13B 78.7 (RAGTruth paper), LettuceDetect
large 79.2 (a fine-tuned ModernBERT), prompted GPT-4-turbo 63.4. All of them train on RAGTruth's human labels.
Here the reader (Qwen3-4B) stays frozen: each answer sentence is read once with its documents in context and once
with the question alone, and a linear probe learns from

    neurons      the MLP neurons (down_proj inputs) of every 4th layer, averaged over the sentence's tokens
    evidence     how much the documents raise the words written (mean shift) and where they make the reader
                 prefer another word (largest disagreement)

trained on RAGTruth's training answers (a sentence is hallucinated if it overlaps a labeled span) and scored on its
test answers: an answer is flagged when any sentence's probability passes a threshold chosen on held-out training
answers. Precision, recall and F1 overall and by task (QA, Summary, Data2txt), AUROC, as the published numbers are.

Features are extracted once, in shards of 50 answers that a restart resumes from:

    python experiments/ragtruth_sota.py extract --split train --n 1200 --out /tmp/rt
    python experiments/ragtruth_sota.py extract --split test  --n 900  --out /tmp/rt
    python experiments/ragtruth_sota.py fit --out /tmp/rt --report results/ragtruth_sota_qwen3_4b.json
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evidence_shift import disagreement, prefix, sentences, spans, token_logprobs  # noqa: E402

PUBLISHED = {"Llama-2-13B fine-tuned (RAGTruth paper)": 78.7, "LettuceDetect large (ModernBERT fine-tuned)": 79.22,
             "GPT-4-turbo prompted (RAGTruth paper)": 63.4}


def halubench_rows(n, seed=1, per_source=50):
    """HaluBench answers (experiments/halubench_compare.py's sample, same order) as RAGTruth-shaped rows, for scoring
    a probe trained on RAGTruth out of its domain: a hallucinated answer is labelled as one span covering it
    (HaluBench labels answers, not spans), its source as the task."""
    from halubench_compare import sample
    rows = []
    for r in sample(per_source, seed)[:n]:
        bad = r["label"] == "FAIL"
        rows.append({"query": r["question"], "context": r["passage"], "output": r["answer"],
                     "task_type": r["source_ds"],
                     "hallucination_labels": json.dumps([{"start": 0, "end": len(r["answer"])}] if bad else [])})
    return rows


def rows_for(split, n, seed=0):
    if split == "halu":                                         # out of domain: HaluBench (seed 1 by default)
        return halubench_rows(n, seed or 1)
    from datasets import load_dataset
    rows = list(load_dataset("wandb/RAGTruth-processed", split=split))
    random.Random(seed).shuffle(rows)                          # a random subset keeps the natural mix
    return rows[:n]


def projection(n_layers, width, d, seed=1):
    """A fixed Gaussian projection of each read layer's neurons, summed: d numbers per sentence instead of
    n_layers * width. Its first k columns are themselves a k-dimensional projection. On 800 / 600 answers a
    2,048-dimensional one kept response F1 (70.0 against 70.4 for all 87,552 neurons)."""
    return [(np.random.default_rng((seed, l)).standard_normal((width, d), dtype=np.float32) / np.sqrt(d)).astype(np.float16)
            for l in range(n_layers)]


def project(x, P):
    """x: n_layers * width (one sentence, or rows of them) -> d."""
    x = np.atleast_2d(np.asarray(x, np.float32))
    w = P[0].shape[0]
    return sum(x[:, i * w:(i + 1) * w] @ Pl.astype(np.float32) for i, Pl in enumerate(P)).astype(np.float16)


def convert(a):
    """Project already extracted shards (all neurons) into --out."""
    import shutil
    os.makedirs(a.out, exist_ok=True)
    P = None
    for src in sorted(glob.glob(os.path.join(a.source, "*_*.npz"))):
        z = dict(np.load(src))
        if P is None:
            n_layers = z["X"].shape[1] // a.width
            P = projection(n_layers, a.width, a.project)
        z["X"] = np.concatenate([project(z["X"][i:i + 256], P) for i in range(0, len(z["X"]), 256)])
        np.savez(os.path.join(a.out, os.path.basename(src)), **z)
        print("converted", os.path.basename(src), z["X"].shape, flush=True)


def _trim():
    """Hand freed heap memory back to the system. Activation tensors vary in size per answer, and glibc keeps the
    freed blocks: over ~1,000 answers on CPU the heap reached 7 GB and pushed the memory-mapped weights out of RAM, so
    every pass re-read them from disk (4 s an answer became 25 s)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def extract(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.probe_train import Reader
    os.makedirs(a.out, exist_ok=True)
    rows = rows_for(a.split, a.n, a.seed)
    tok = AutoTokenizer.from_pretrained(a.reader)
    if torch.cuda.is_available():                               # bfloat16 where supported (L4, A100); a T4: float16
        dt = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(a.reader, torch_dtype=dt, device_map="auto").eval()
        if torch.cuda.get_device_capability(0)[0] < 8:
            for mod in model.modules():
                if getattr(mod, "compute_dtype", None) == torch.bfloat16:
                    mod.compute_dtype = torch.float16
    else:
        model = AutoModelForCausalLM.from_pretrained(a.reader, torch_dtype=torch.bfloat16).eval()
    reader = Reader(model, every=a.every)
    P = projection(len(reader.layers), model.config.intermediate_size, a.project) if a.project else None
    t0 = time.time()
    for start in range(0, len(rows), a.shard):
        path = os.path.join(a.out, f"{a.split}_{start:05d}.npz")
        if os.path.exists(path):                               # done before a restart
            continue
        X, ev, y, resp, kind, task, t_sec, t_tok = [], [], [], [], [], [], [], []
        for i, r in enumerate(rows[start:start + a.shard], start):
            t_r = time.time()
            out = r["output"]
            pre = prefix(tok, f"{r['query']}\n\n{r['context'][:a.max_context_chars]}")
            enc = tok(out, add_special_tokens=False, return_offsets_mapping=True)
            if not enc.input_ids:
                continue
            lw, dw = token_logprobs(model, tok, pre, enc.input_ids)          # the reader's hooks fire here too
            acts = dict(reader.acts)
            lo, do = token_logprobs(model, tok, prefix(tok, r["query"]), enc.input_ids)
            shift, dis = lw - lo, disagreement(dw, do, enc.input_ids)
            del dw, do
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_sec.append(time.time() - t_r)                  # both passes, as a served check reads them
            t_tok.append(len(pre) + len(enc.input_ids))
            sp, xs = spans(r), []
            for s, e in sentences(out):
                js = [j for j, (ts, te) in enumerate(enc.offset_mapping) if ts >= s and te <= e]
                if not js:
                    continue
                a_, b_ = len(pre) + js[0], len(pre) + js[-1] + 1
                x = np.concatenate([acts[li][a_:b_].float().mean(0).numpy() for li in reader.layers])
                xs.append(x)
                ev.append((float(shift[js].mean()), float(dis[js].max())))
                kinds = [k for x_, y_, k in sp if s < y_ and x_ < e]
                y.append(int(bool(kinds)))
                kind.append(kinds[0] if kinds else "")
                resp.append(i)
                task.append(r["task_type"])
            if xs:                                          # one projection per answer, not per sentence
                X.extend(project(np.stack(xs), P) if P is not None else np.stack(xs).astype(np.float16))
            del acts, lw, lo, shift, dis, xs
            reader.acts.clear()
            _trim()                                         # glibc keeps freed blocks; without this the heap grows
        np.savez(path, X=np.stack(X), ev=np.array(ev, np.float32), y=np.array(y), resp=np.array(resp),
                 kind=np.array(kind), task=np.array(task),
                 resp_y=np.array([int(bool(spans(r))) for r in rows[start:start + a.shard]]),
                 resp_task=np.array([r["task_type"] for r in rows[start:start + a.shard]]),
                 resp_ids=np.arange(start, start + len(rows[start:start + a.shard])),
                 t_sec=np.array(t_sec, np.float32), t_tok=np.array(t_tok),
                 device=np.array(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
        print(f"{a.split} {min(start + a.shard, len(rows))}/{len(rows)} {time.time() - t0:.0f}s", flush=True)


def load(out, split):
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(out, f"{split}_*.npz")))]
    cat = lambda k: np.concatenate([p[k] for p in parts])      # noqa: E731
    return {k: cat(k) for k in ("X", "ev", "y", "resp", "kind", "task", "resp_y", "resp_task", "resp_ids")}


def timing(out, split="test"):
    """Measured seconds per answer (both reader passes, as served) and tokens read, from the extraction shards."""
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(out, f"{split}_*.npz")))]
    parts = [p for p in parts if "t_sec" in p.files]
    if not parts:
        return None
    sec, tok = np.concatenate([p["t_sec"] for p in parts]), np.concatenate([p["t_tok"] for p in parts])
    return {"device": str(parts[0]["device"]), "answers": int(len(sec)),
            "seconds_per_answer": {"mean": round(float(sec.mean()), 3), "median": round(float(np.median(sec)), 3),
                                   "p95": round(float(np.percentile(sec, 95)), 3)},
            "tokens_per_answer": {"mean": round(float(tok.mean())), "median": int(np.median(tok)),
                                  "p95": int(np.percentile(tok, 95))},
            "answers_per_hour": round(3600 / float(sec.mean())),
            "note": "one answer at a time, unbatched, hooks on every 6th layer kept on the CPU: an upper bound"}


def response_scores(p, resp, ids):
    return np.array([p[resp == i].max() if (resp == i).any() else 0.0 for i in ids])


def prf(flag, y):
    tp, fp, fn = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": round(100 * pr, 1), "recall": round(100 * rc, 1),
            "f1": round(100 * 2 * pr * rc / max(1e-9, pr + rc), 1), "n": int(len(y))}


def torch_logreg(X, y, l2, epochs=8, lr=1e-3, batch=512, seed=0):
    """Class-balanced logistic regression on standardized features, minibatch Adam; X stays on the CPU (float16),
    batches go to the GPU when there is one. Returns (mean, std, w, b) as numpy."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    n, d = X.shape
    mean = np.zeros(d, np.float64)
    sq = np.zeros(d, np.float64)
    for i in range(0, n, 4096):                                 # streamed mean and variance
        c = X[i:i + 4096].astype(np.float64)
        mean += c.sum(0)
        sq += (c ** 2).sum(0)
    mean /= n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 1e-8))
    m_t, s_t = torch.tensor(mean, dtype=torch.float32, device=dev), torch.tensor(std, dtype=torch.float32, device=dev)
    w = torch.zeros(d, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    yt = torch.tensor(y, dtype=torch.float32)
    pos = float(y.mean())
    weight = torch.tensor([0.5 / max(1 - pos, 1e-6), 0.5 / max(pos, 1e-6)])
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=g).split(batch):
            xb = (torch.from_numpy(X[idx.numpy()].astype(np.float32)).to(dev) - m_t) / s_t
            yb = yt[idx].to(dev)
            wb = weight[yt[idx].long()].to(dev)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(xb @ w + b, yb, reduction="none") * wb).mean() \
                + l2 * (w ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return mean.astype(np.float32), std.astype(np.float32), w.detach().cpu().numpy(), float(b.detach().cpu())


def torch_predict(model, X):
    mean, std, w, b = model
    out = []
    for i in range(0, len(X), 4096):
        z = ((X[i:i + 4096].astype(np.float32) - mean) / std) @ w + b
        out.append(1 / (1 + np.exp(-z)))
    return np.concatenate(out)


def fit(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from vt.correctness import auroc
    tr, te = load(a.out, "train"), load(a.out, "test")
    ids = np.unique(tr["resp_ids"])
    rng = np.random.default_rng(0)
    val_ids = set(rng.choice(ids, size=len(ids) // 5, replace=False).tolist())
    is_val = np.array([r in val_ids for r in tr["resp"]])
    res = {"reader": a.reader, "train_answers": int(len(ids)), "test_answers": int(len(te["resp_ids"])),
           "train_sentences": int(len(tr["y"])), "test_sentences": int(len(te["y"])),
           "test_hallucinated_answers": int(te["resp_y"].sum()), "published_response_f1": PUBLISHED}

    big = a.fitter == "torch"                                   # float16 until each batch reaches the GPU

    def feats(d, which):
        dt = np.float16 if big else np.float32
        z = [d["X"].astype(dt, copy=False)] if which in ("neurons", "both") else []
        if which in ("evidence", "both"):
            z.append(np.c_[d["ev"], d["ev"] ** 2].astype(dt))
        return np.concatenate(z, 1) if len(z) > 1 else z[0]

    def train(X, y, C):
        if big:
            m = torch_logreg(X, y, l2=1.0 / (C * len(y)))
            return lambda Z: torch_predict(m, Z)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
        return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]

    rv = {i: y for i, y in zip(tr["resp_ids"], tr["resp_y"])}
    for which in ("evidence", "neurons", "both"):
        Xtr, Xte = feats(tr, which), feats(te, which)
        best = None
        for C in (0.001, 0.01, 0.1):
            pv = train(Xtr[~is_val], tr["y"][~is_val], C)(Xtr[is_val])
            vids = np.array(sorted(val_ids))
            rs, ry = response_scores(pv, tr["resp"][is_val], vids), np.array([rv[i] for i in vids])
            ts = np.unique(rs)
            f1s = [prf(rs >= t, ry)["f1"] for t in ts]
            k = int(np.argmax(f1s))
            if best is None or f1s[k] > best[0]:
                best = (f1s[k], C, float(ts[k]))
        _, C, thr = best
        p = train(Xtr, tr["y"], C)(Xte)
        rs = response_scores(p, te["resp"], te["resp_ids"])
        flag = rs >= thr
        row = {"C": C, "threshold": round(thr, 4), "validation_f1": best[0],
               "response": dict(prf(flag, te["resp_y"]), auroc=round(float(auroc(rs, te["resp_y"].astype(float))), 3)),
               "sentence_auroc": round(float(auroc(p, te["y"].astype(float))), 3),
               "by_task": {t: prf(flag[te["resp_task"] == t], te["resp_y"][te["resp_task"] == t])
                           for t in sorted(set(te["resp_task"].tolist()))}}
        boot = []
        for _ in range(1000):                                   # an interval for the F1: answers resampled
            i = rng.integers(0, len(rs), len(rs))
            boot.append(prf(flag[i], te["resp_y"][i])["f1"])
        row["response"]["f1_ci95"] = [round(float(np.percentile(boot, 2.5)), 1), round(float(np.percentile(boot, 97.5)), 1)]
        res[which] = row
        print(which, json.dumps(row["response"]), flush=True)
    res["timing"] = timing(a.out)
    json.dump(res, open(a.report, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("published_response_f1",)}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit", "convert"])
    ap.add_argument("--reader", default="Qwen/Qwen3-4B")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=50)
    ap.add_argument("--max-context-chars", type=int, default=6000)
    ap.add_argument("--every", type=int, default=4, help="read every n-th layer's neurons")
    ap.add_argument("--fitter", choices=["sklearn", "torch"], default="sklearn",
                    help="torch: streaming minibatch logistic regression (GPU when present) for large feature sets")
    ap.add_argument("--project", type=int, default=0, help="store a fixed random projection of the neurons to this "
                    "many numbers per sentence (0: all neurons)")
    ap.add_argument("--source", help="convert: the shards to project")
    ap.add_argument("--width", type=int, default=9728, help="convert: neurons per read layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="results/ragtruth_sota.json")
    a = ap.parse_args()
    {"extract": extract, "fit": fit, "convert": convert}[a.cmd](a)


if __name__ == "__main__":
    main()
