"""Cheap AI-text detection from a small frozen model's internals, with a false-accusation rate that is guaranteed,
not hoped for.

A detector's costly error is calling a person's writing AI. The test is whether that rate can be promised on a
domain the detector never saw, from that domain's human writing alone (no AI samples needed from the customer):

    data     RAID (Dugan et al., 2024), train split, streamed: 8 domains (abstracts, books, news, poetry, recipes,
             reddit, reviews, wiki), human text and 11 generators (GPT-2/3/4, ChatGPT, Llama-2-chat, Mistral(-chat),
             MPT(-chat), Cohere(-chat)), mixed decoding; per domain 300 human, 300 AI (no attack) and 60 AI rewritten
             by RAID's paraphrase attack. First 256 tokens of each text.
    reader   Qwen3-0.6B-Base (observer) and Qwen3-0.6B (performer): one pass of each, CPU.
    scores   log-perplexity under the observer (low = AI);
             Binoculars (Hans et al., 2024): performer log-perplexity / observer-performer cross-entropy;
             probe: logistic regression on the observer's mean hidden state at four depths plus the scalar
             scores, trained on the other seven domains (leave one domain out), never on the domain it is tested on.
    promises (a) a threshold set at 1% false alarms on the other domains, carried to the new one (transfer);
             (b) a threshold from 150 of the new domain's human texts, at 5% false alarms with 95% confidence
                 (one-sided Clopper-Pearson, vt/verifier_pipeline.guaranteed_threshold), tested on its other human
                 texts and its AI texts, over 50 random splits: the promise holds if the measured false-alarm rate
                 exceeds 5% in at most ~5% of splits.

    python experiments/ai_text_detect.py sample           # stream RAID, keep the per-domain sample
    python experiments/ai_text_detect.py read             # both readers over the sample (resumable)
    python experiments/ai_text_detect.py evaluate
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUT = "results/ai_text"
SCRATCH = os.environ.get("AI_TEXT_SCRATCH", OUT)
DOMAINS = ("abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki")
OBSERVER, PERFORMER = "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"
LAYERS = (7, 14, 21, 28)


def sample(a):
    """One pass over the streamed CSV with a reservoir per bucket, so the sample is not the file's first rows."""
    from datasets import load_dataset
    caps = {}
    for d in DOMAINS:
        caps[(d, "human", "none")] = a.per_domain
        caps[(d, "ai", "none")] = a.per_domain
        caps[(d, "ai", "paraphrase")] = a.paraphrased
    res = {k: [] for k in caps}
    seen = Counter()
    rng = random.Random(0)
    t0 = time.time()
    ds = load_dataset("liamdugan/raid", "raid", split="train", streaming=True)
    for n, r in enumerate(ds):
        if a.max_rows and n >= a.max_rows:
            break
        if r["domain"] not in DOMAINS or r["attack"] not in ("none", "paraphrase"):
            continue
        who = "human" if r["model"] == "human" else "ai"
        key = (r["domain"], who, r["attack"])
        if key not in caps or len(r["generation"].split()) < 40:
            continue
        rec = {"text": r["generation"], "domain": r["domain"], "model": r["model"], "attack": r["attack"],
               "decoding": r["decoding"], "y": int(who == "ai")}
        # per-generator reservoirs would be fairer still; one reservoir per bucket keeps the generators' mix
        seen[key] += 1
        if len(res[key]) < caps[key]:
            res[key].append(rec)
        else:
            j = rng.randrange(seen[key])
            if j < caps[key]:
                res[key][j] = rec
        if n % 500000 == 0 and n:                                  # survive a restart: the domains read so far
            _save(res, done=False)
        if n % 200000 == 0:
            print(f"{n} rows {time.time() - t0:.0f}s", {k[0][:4] + k[1][0] + k[2][0]: v for k, v in seen.items()},
                  flush=True)
    rows = _save(res, done=True)
    print(json.dumps({"rows": len(rows), "by_bucket": {"/".join(k): len(v) for k, v in res.items()},
                      "models": Counter(r["model"] for r in rows)}, indent=1))


def _save(res, done):
    rows = [r for k in sorted(res) for r in res[k]]
    os.makedirs(SCRATCH, exist_ok=True)
    path = f"{SCRATCH}/sample.jsonl" if done else f"{SCRATCH}/sample.partial.jsonl"
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    return rows


def read(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)
    rows = [json.loads(l) for l in open(f"{SCRATCH}/sample.jsonl")]
    tok = AutoTokenizer.from_pretrained(OBSERVER)
    obs = AutoModelForCausalLM.from_pretrained(OBSERVER, dtype=torch.float32).eval()
    perf = AutoModelForCausalLM.from_pretrained(PERFORMER, dtype=torch.float32).eval()
    path = f"{SCRATCH}/features.npz"
    have = dict(np.load(path)) if os.path.exists(path) else None
    start = len(have["scalars"]) if have else 0
    S, H, secs = ([have["scalars"]], [have["hidden"]], [have["sec"]]) if have else ([], [], [])
    t0 = time.time()
    for b in range(start, len(rows), a.batch):
        t = time.time()
        enc = [tok(r["text"], add_special_tokens=False).input_ids[:a.max_tokens] for r in rows[b:b + a.batch]]
        L = max(map(len, enc))
        ids = torch.full((len(enc), L), tok.pad_token_id or 0)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)], att[i, :len(e)] = torch.tensor(e), 1
        with torch.no_grad():
            o = obs(input_ids=ids, attention_mask=att, output_hidden_states=True)
            p = perf(input_ids=ids, attention_mask=att)
        lo, lp = torch.log_softmax(o.logits[:, :-1].float(), -1), torch.log_softmax(p.logits[:, :-1].float(), -1)
        tgt, m = ids[:, 1:], att[:, 1:].float()
        nll_o = -lo.gather(-1, tgt[..., None])[..., 0]
        nll_p = -lp.gather(-1, tgt[..., None])[..., 0]
        x_op = -(lo.exp() * lp).sum(-1)                           # observer's distribution, performer's log-probs
        ent_o = -(lo.exp() * lo).sum(-1)
        rank1 = (lo.argmax(-1) == tgt).float()
        mean = lambda v: ((v * m).sum(1) / m.sum(1)).numpy()       # noqa: E731
        sd = lambda v: np.sqrt(np.maximum(mean(v ** 2) - mean(v) ** 2, 0))   # noqa: E731
        sc = np.c_[mean(nll_o), mean(nll_p), mean(x_op), mean(ent_o), sd(nll_o), mean(rank1), m.sum(1).numpy()]
        S.append(sc.astype(np.float32))
        am = att[..., None].float()
        H.append(np.concatenate([((o.hidden_states[l] * am).sum(1) / am.sum(1)).numpy() for l in LAYERS], 1)
                 .astype(np.float16))
        secs.append(np.full(len(enc), (time.time() - t) / len(enc), np.float32))
        if (b // a.batch) % 25 == 0 or b + a.batch >= len(rows):
            np.savez(path, scalars=np.concatenate(S), hidden=np.concatenate(H), sec=np.concatenate(secs))
            print(f"{min(b + a.batch, len(rows))}/{len(rows)} {(time.time() - t0) / (b + a.batch - start):.3f}s a text",
                  flush=True)
    np.savez(path, scalars=np.concatenate(S), hidden=np.concatenate(H), sec=np.concatenate(secs))


SCALARS = ("nll_observer", "nll_performer", "xent_obs_perf", "entropy_observer", "nll_sd", "top1_share", "tokens")


def evaluate(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from vt.correctness import auroc
    from vt.verifier_pipeline import guaranteed_threshold
    rows = [json.loads(l) for l in open(f"{SCRATCH}/sample.jsonl")]
    z = np.load(f"{SCRATCH}/features.npz")
    n = len(z["scalars"])
    rows = rows[:n]
    S, Hd = z["scalars"].astype(np.float64), z["hidden"].astype(np.float32)
    dom = np.array([r["domain"] for r in rows])
    y = np.array([r["y"] for r in rows])
    para = np.array([r["attack"] == "paraphrase" for r in rows])
    model = np.array([r["model"] for r in rows])
    # every score: higher = more AI-like
    scores = {"log-perplexity": -S[:, 0], "binoculars": -(S[:, 1] / S[:, 2])}
    probe = np.zeros(n)
    X = np.c_[Hd, S[:, :6], S[:, 1] / S[:, 2]]
    for d in DOMAINS:
        tr = (dom != d) & ~para
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=a.C, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        probe[dom == d] = clf.decision_function(sc.transform(X[dom == d]))
    scores["probe (other domains)"] = probe

    rng = np.random.default_rng(0)
    res = {"texts": n, "reader": [OBSERVER, PERFORMER], "seconds_per_text_cpu": round(float(z["sec"].mean()), 3),
           "by_score": {}}
    for name, s in scores.items():
        out = {}
        for d in DOMAINS:
            m_d, m_o = dom == d, (dom != d) & ~para
            clean = m_d & ~para
            hum, ai, pa = m_d & (y == 0) & ~para, m_d & (y == 1) & ~para, m_d & para
            # (a) transfer: 1% false alarms on the other domains' human texts
            thr_o = float(np.quantile(s[m_o & (y == 0)], 0.99))
            row = {"auroc": round(float(auroc(s[clean], y[clean].astype(float))), 3),
                   "transfer_1pct": {"false_alarms": round(float((s[hum] > thr_o).mean()), 3),
                                     "caught": round(float((s[ai] > thr_o).mean()), 3),
                                     "caught_paraphrased": round(float((s[pa] > thr_o).mean()), 3)}}
            # (b) promise from the domain's own human texts: 5% at 95% confidence from 150 of them
            fa, ca, cp, idx = [], [], [], np.where(hum)[0]
            for _ in range(a.splits):
                cal = rng.choice(idx, size=min(a.cal, len(idx) - 20), replace=False)
                test_h = np.setdiff1d(idx, cal)
                thr = guaranteed_threshold(s[cal], np.zeros(len(cal), int), a.fpr, 0.95)
                fa.append(float((s[test_h] > thr).mean()))
                ca.append(float((s[ai] > thr).mean()))
                cp.append(float((s[pa] > thr).mean()))
            row["promise_5pct_from_150_human"] = {
                "false_alarms_mean": round(float(np.mean(fa)), 3),
                "splits_over_5pct": round(float(np.mean(np.array(fa) > a.fpr)), 3),
                "caught": round(float(np.mean(ca)), 3), "caught_paraphrased": round(float(np.mean(cp)), 3)}
            out[d] = row
        allc = ~para
        out["all"] = {"auroc_pooled": round(float(auroc(s[allc], y[allc].astype(float))), 3),
                      "auroc_mean_over_domains": round(float(np.mean([out[d]["auroc"] for d in DOMAINS])), 3),
                      "transfer_false_alarms_max": max(out[d]["transfer_1pct"]["false_alarms"] for d in DOMAINS),
                      "promise_caught_mean": round(float(np.mean([out[d]["promise_5pct_from_150_human"]["caught"]
                                                                  for d in DOMAINS])), 3),
                      "promise_splits_over_5pct_mean": round(float(np.mean(
                          [out[d]["promise_5pct_from_150_human"]["splits_over_5pct"] for d in DOMAINS])), 3)}
        # which generators slip through: caught share per generator under the promise thresholds, pooled
        res["by_score"][name] = out
        print(name, json.dumps(out["all"]), flush=True)
    s = scores["probe (other domains)"]
    per_model = {}
    for mname in sorted(set(model[y == 1])):
        mm = (model == mname) & ~para
        per_model[mname] = round(float(auroc(np.r_[s[mm], s[(y == 0) & ~para]],
                                             np.r_[np.ones(mm.sum()), np.zeros(((y == 0) & ~para).sum())])), 3)
    res["probe_auroc_by_generator"] = per_model
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/raid_qwen3_0.6b.json", "w"), indent=1)
    print(json.dumps(per_model))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--per-domain", type=int, default=300)
    s.add_argument("--paraphrased", type=int, default=60)
    s.add_argument("--max-rows", type=int, default=0)
    r = sub.add_parser("read")
    r.add_argument("--batch", type=int, default=2)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    e = sub.add_parser("evaluate")
    e.add_argument("--C", type=float, default=0.01)
    e.add_argument("--cal", type=int, default=150)
    e.add_argument("--fpr", type=float, default=0.05)
    e.add_argument("--splits", type=int, default=50)
    a = ap.parse_args()
    {"sample": sample, "read": read, "evaluate": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
