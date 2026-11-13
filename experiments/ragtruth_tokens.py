"""Token-level reading of RAGTruth: does scoring every token beat scoring sentence means?

The sentence-mean readout of experiments/ragtruth_sota.py plateaus near response F1 72 (800 / 1,600 / 2,400
training answers: 70.6 / 71.4 / 72.1), and neither context, an MLP nor a learned reply aggregate moves it
(results/ragtruth_supervised/variants_train2400.json). RAGTruth's labels are character spans, often a few words
(a wrong figure, an added detail); a sentence mean dilutes them, and the published 79-F1 systems label tokens.

One reader pass with the evidence (no second pass: the evidence shift adds nothing to the neurons, see the fits),
the reply's tokens only, each layer's MLP input projected to --dims numbers by a fixed Gaussian projection (seed 2)
and summed over the nine read layers. Because the projection is linear, a sentence's mean of token vectors is the
projection of its mean, so the sentence readout trained on these same vectors is a matched baseline.

    python experiments/ragtruth_tokens.py extract --split train --n 1200 --out <dir>
    python experiments/ragtruth_tokens.py extract --split test --n 600 --out <dir>
    python experiments/ragtruth_tokens.py fit --out <dir> --report results/ragtruth_supervised/tokens_train1200.json

Rows are the same shuffled RAGTruth rows as ragtruth_sota.py (seed 0), so the first n match its shards.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragtruth_sota import _trim, prefix, prf, projection, project, rows_for, sentences, spans  # noqa: E402


def extract(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.probe_train import Reader
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    rows = rows_for(a.split, a.n)
    tok = AutoTokenizer.from_pretrained(a.reader)
    model = AutoModelForCausalLM.from_pretrained(a.reader, dtype=torch.bfloat16).eval()
    reader = Reader(model, every=a.every)
    P = projection(len(reader.layers), model.config.intermediate_size, a.dims, seed=2)
    t0 = time.time()
    for start in range(0, len(rows), a.shard):
        path = os.path.join(a.out, f"{a.split}_{start:05d}.npz")
        if os.path.exists(path):
            continue
        X, y, sent, resp, resp_y, resp_task = [], [], [], [], [], []
        for i, r in enumerate(rows[start:start + a.shard], start):
            out = r["output"]
            pre = prefix(tok, f"{r['query']}\n\n{r['context'][:a.max_context_chars]}")
            enc = tok(out, add_special_tokens=False, return_offsets_mapping=True)
            sp = spans(r)
            resp_y.append(int(bool(sp)))
            resp_task.append(r["task_type"])
            if not enc.input_ids:
                continue
            with torch.no_grad():
                model.model(input_ids=torch.tensor([pre + enc.input_ids]))      # hooks keep the MLP inputs
            T = len(enc.input_ids)
            x = np.concatenate([reader.acts[li][len(pre):len(pre) + T].float().numpy() for li in reader.layers], 1)
            X.append(project(x, P))
            sid = np.full(T, -1)
            for k, (s, e) in enumerate(sentences(out)):
                for j, (ts, te) in enumerate(enc.offset_mapping):
                    if ts >= s and te <= e:
                        sid[j] = k
            y.extend(int(any(ts < e_ and s_ < te for s_, e_, _ in sp)) for ts, te in enc.offset_mapping)
            sent.extend(sid.tolist())
            resp.extend([i] * T)
            reader.acts.clear()
            del x
            _trim()
        np.savez(path, X=np.concatenate(X), y=np.array(y, np.int8), sent=np.array(sent), resp=np.array(resp),
                 resp_y=np.array(resp_y), resp_task=np.array(resp_task),
                 resp_ids=np.arange(start, start + len(resp_y)))
        print(f"{a.split} {start + len(resp_y)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)


def load(out, split):
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(out, f"{split}_*.npz")))]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "sent", "resp", "resp_y", "resp_task",
                                                                 "resp_ids")}


def sentence_view(d):
    """Mean token vector per (reply, sentence); a sentence is hallucinated when any of its tokens is."""
    key = d["resp"].astype(np.int64) * 10000 + d["sent"]
    keep = d["sent"] >= 0
    uk, inv = np.unique(key[keep], return_inverse=True)
    X = np.zeros((len(uk), d["X"].shape[1]), np.float32)
    np.add.at(X, inv, d["X"][keep].astype(np.float32))
    n = np.bincount(inv, minlength=len(uk))[:, None]
    y = np.zeros(len(uk), int)
    np.maximum.at(y, inv, d["y"][keep].astype(int))
    return X / n, y, uk // 10000


def reply_scores(p, resp, ids, how):
    out = []
    for i in ids:
        s = p[resp == i]
        if not len(s):
            out.append(0.0)
        elif how == "max":
            out.append(float(s.max()))
        else:                                                   # mean of the top k
            k = int(how.split("top")[1])
            out.append(float(np.sort(s)[-k:].mean()))
    return np.array(out)


def fit(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from vt.correctness import auroc
    tr, te = load(a.out, "train"), load(a.out, "test")
    ids = tr["resp_ids"]
    rng = np.random.default_rng(0)
    val_ids = set(rng.choice(ids, size=len(ids) // 5, replace=False).tolist())
    vids = np.array(sorted(val_ids))
    rv = dict(zip(tr["resp_ids"], tr["resp_y"]))
    ry_val = np.array([rv[i] for i in vids])
    res = {"train_answers": int(len(ids)), "test_answers": int(len(te["resp_ids"])), "dims": int(tr["X"].shape[1]),
           "train_tokens": int(len(tr["y"])), "hallucinated_token_share": round(float(tr["y"].mean()), 4),
           "readouts": {}}

    views = {"sentence mean": (*sentence_view(tr), *sentence_view(te))}
    views["token"] = (tr["X"].astype(np.float32), tr["y"], tr["resp"], te["X"].astype(np.float32), te["y"], te["resp"])
    for name, (Xtr, ytr, rtr, Xte, yte, rte) in views.items():
        is_val = np.isin(rtr, vids)
        sub = np.ones(len(ytr), bool)
        if name == "token" and (~is_val).sum() > a.max_tokens:      # subsample clean tokens of training replies
            neg = np.where(~is_val & (ytr == 0))[0]
            drop = rng.choice(neg, size=(~is_val).sum() - a.max_tokens, replace=False)
            sub[drop] = False
        best = None
        for C in (0.0003, 0.001, 0.003):
            sc = StandardScaler().fit(Xtr[~is_val & sub])
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr[~is_val & sub]),
                                                                                    ytr[~is_val & sub])
            pv = clf.predict_proba(sc.transform(Xtr[is_val]))[:, 1]
            for how in (("max",) if name != "token" else ("max", "top3", "top8")):
                rs = reply_scores(pv, rtr[is_val], vids, how)
                ts = np.unique(rs)
                f1s = [prf(rs >= t, ry_val)["f1"] for t in ts]
                k = int(np.argmax(f1s))
                if best is None or f1s[k] > best[0]:
                    best = (f1s[k], C, how, float(ts[k]))
            print(name, C, best, flush=True)
        vf1, C, how, thr = best
        sc = StandardScaler().fit(Xtr[sub])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr[sub]), ytr[sub])
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        rs = reply_scores(p, rte, te["resp_ids"], how)
        flag = rs >= thr
        row = dict(prf(flag, te["resp_y"]), auroc=round(float(auroc(rs, te["resp_y"].astype(float))), 3),
                   unit_auroc=round(float(auroc(p, yte.astype(float))), 3), C=C, aggregate=how, validation_f1=vf1,
                   by_task={t: prf(flag[te["resp_task"] == t], te["resp_y"][te["resp_task"] == t])
                            for t in sorted(set(te["resp_task"].tolist()))})
        boot = [prf(flag[i], te["resp_y"][i])["f1"] for i in (rng.integers(0, len(rs), len(rs)) for _ in range(1000))]
        row["f1_ci95"] = [round(float(np.percentile(boot, 2.5)), 1), round(float(np.percentile(boot, 97.5)), 1)]
        res["readouts"][name] = row
        print(name, json.dumps(row), flush=True)
        json.dump(res, open(a.report, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit"])
    ap.add_argument("--reader", default="Qwen/Qwen3-4B")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--dims", type=int, default=1024)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--shard", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-context-chars", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=150000, help="training tokens kept (clean ones dropped)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    {"extract": extract, "fit": fit}[a.cmd](a)


if __name__ == "__main__":
    main()
