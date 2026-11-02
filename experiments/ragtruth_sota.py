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


def rows_for(split, n, seed=0):
    from datasets import load_dataset
    rows = list(load_dataset("wandb/RAGTruth-processed", split=split))
    random.Random(seed).shuffle(rows)                          # a random subset keeps the natural mix
    return rows[:n]


def extract(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.probe_train import Reader
    os.makedirs(a.out, exist_ok=True)
    rows = rows_for(a.split, a.n, a.seed)
    tok = AutoTokenizer.from_pretrained(a.reader)
    model = AutoModelForCausalLM.from_pretrained(a.reader, torch_dtype=torch.bfloat16).eval()
    reader = Reader(model)
    t0 = time.time()
    for start in range(0, len(rows), a.shard):
        path = os.path.join(a.out, f"{a.split}_{start:05d}.npz")
        if os.path.exists(path):                               # done before a restart
            continue
        X, ev, y, resp, kind, task = [], [], [], [], [], []
        for i, r in enumerate(rows[start:start + a.shard], start):
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
            sp = spans(r)
            for s, e in sentences(out):
                js = [j for j, (ts, te) in enumerate(enc.offset_mapping) if ts >= s and te <= e]
                if not js:
                    continue
                a_, b_ = len(pre) + js[0], len(pre) + js[-1] + 1
                X.append(np.concatenate([acts[li][a_:b_].float().mean(0).numpy() for li in reader.layers]).astype(np.float16))
                ev.append((float(shift[js].mean()), float(dis[js].max())))
                kinds = [k for x_, y_, k in sp if s < y_ and x_ < e]
                y.append(int(bool(kinds)))
                kind.append(kinds[0] if kinds else "")
                resp.append(i)
                task.append(r["task_type"])
        np.savez(path, X=np.stack(X), ev=np.array(ev, np.float32), y=np.array(y), resp=np.array(resp),
                 kind=np.array(kind), task=np.array(task),
                 resp_y=np.array([int(bool(spans(r))) for r in rows[start:start + a.shard]]),
                 resp_task=np.array([r["task_type"] for r in rows[start:start + a.shard]]),
                 resp_ids=np.arange(start, start + len(rows[start:start + a.shard])))
        print(f"{a.split} {min(start + a.shard, len(rows))}/{len(rows)} {time.time() - t0:.0f}s", flush=True)


def load(out, split):
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(out, f"{split}_*.npz")))]
    cat = lambda k: np.concatenate([p[k] for p in parts])      # noqa: E731
    return {k: cat(k) for k in ("X", "ev", "y", "resp", "kind", "task", "resp_y", "resp_task", "resp_ids")}


def response_scores(p, resp, ids):
    return np.array([p[resp == i].max() if (resp == i).any() else 0.0 for i in ids])


def prf(flag, y):
    tp, fp, fn = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": round(100 * pr, 1), "recall": round(100 * rc, 1),
            "f1": round(100 * 2 * pr * rc / max(1e-9, pr + rc), 1), "n": int(len(y))}


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

    def feats(d, which):
        z = [d["X"].astype(np.float32)] if which in ("neurons", "both") else []
        if which in ("evidence", "both"):
            z.append(np.c_[d["ev"], d["ev"] ** 2])
        return np.concatenate(z, 1)

    rv = {i: y for i, y in zip(tr["resp_ids"], tr["resp_y"])}
    for which in ("evidence", "neurons", "both"):
        Xtr, Xte = feats(tr, which), feats(te, which)
        best = None
        for C in (0.001, 0.01, 0.1):
            sc = StandardScaler().fit(Xtr[~is_val])
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr[~is_val]),
                                                                                     tr["y"][~is_val])
            pv = clf.predict_proba(sc.transform(Xtr[is_val]))[:, 1]
            vids = np.array(sorted(val_ids))
            rs, ry = response_scores(pv, tr["resp"][is_val], vids), np.array([rv[i] for i in vids])
            ts = np.unique(rs)
            f1s = [prf(rs >= t, ry)["f1"] for t in ts]
            k = int(np.argmax(f1s))
            if best is None or f1s[k] > best[0]:
                best = (f1s[k], C, float(ts[k]))
        _, C, thr = best
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), tr["y"])
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
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
    json.dump(res, open(a.report, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("published_response_f1",)}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit"])
    ap.add_argument("--reader", default="Qwen/Qwen3-4B")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=50)
    ap.add_argument("--max-context-chars", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="results/ragtruth_sota.json")
    a = ap.parse_args()
    (extract if a.cmd == "extract" else fit)(a)


if __name__ == "__main__":
    main()
