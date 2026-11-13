"""Does the frozen reader's internals readout add anything to the best open RAGTruth detector?

LettuceDetect-large (ModernBERT fine-tuned on RAGTruth token labels, MIT licence; published response F1 79.22) is
the strongest open detector. Our frozen Qwen3-4B readout is at F1 72.1 (2,400 training answers; the learning curve
is flat and readout variants and token-level reading do not lift it). If the two see different errors, a
combination is worth more than either; if not, the internals readout is redundant on this benchmark.

On the same 600 RAGTruth test answers as results/ragtruth_supervised/ (seed 0):
    lettuce     LettuceDetect's answer score: its highest token probability (published rule: flag at 0.5)
    internals   our linear sentence readout trained on the first 2,400 training answers; worst sentence
    combined    logistic regression on the two logits
LettuceDetect was trained on RAGTruth's training split, so the combination's two weights and every threshold are
fitted on half of the test answers and scored on the other half (2-fold cross-fitting, repeated 20 times), never
on training answers whose labels LettuceDetect has seen.

The wandb/RAGTruth-processed rows give one context string and a query (the question, or the task instruction for
summaries and data-to-text); they go to LettuceDetect as context=[context], question=query. This is close to,
not identical with, the prompt it was trained on, so its F1 here can differ from the published 79.22.

    python experiments/lettuce_ensemble.py score --out <dir>        # LettuceDetect on the 600 test answers
    python experiments/lettuce_ensemble.py combine --out <dir> --shards <ragtruth_sota shard dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragtruth_sota import load, prf, response_scores, rows_for  # noqa: E402

MODEL = "KRLabsOrg/lettucedect-large-modernbert-en-v1"


def score(a):
    import torch
    from lettucedetect.models.inference import HallucinationDetector
    torch.set_num_threads(a.threads)
    det = HallucinationDetector(method="transformer", model_path=MODEL)
    rows = rows_for("test", a.n)
    path = os.path.join(a.out, "lettuce_test.jsonl")
    os.makedirs(a.out, exist_ok=True)
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    t0 = time.time()
    with open(path, "a") as f:
        for i, r in enumerate(rows[done:], done):
            t = time.time()
            toks = det.predict(context=[r["context"]], question=r["query"], answer=r["output"], output_format="tokens")
            probs = [float(x["prob"]) for x in toks] or [0.0]
            f.write(json.dumps({"i": i, "max_prob": max(probs), "any_flag": int(any(x["pred"] for x in toks)),
                                "n_tok": len(toks), "sec": round(time.time() - t, 3)}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} {(time.time() - t0) / (i + 1 - done):.2f}s an answer", flush=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def best_thr(s, y):
    ts = np.unique(s)
    f = [prf(s >= t, y)["f1"] for t in ts]
    return float(ts[int(np.argmax(f))])


def combine(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from vt.correctness import auroc
    let = [json.loads(l) for l in open(os.path.join(a.out, "lettuce_test.jsonl"))]
    tr, te = load(a.shards, "train"), load(a.shards, "test")
    keep = tr["resp"] < a.train_answers
    sc = StandardScaler().fit(tr["X"][keep].astype(np.float32))
    clf = LogisticRegression(C=0.001, max_iter=3000, class_weight="balanced").fit(
        sc.transform(tr["X"][keep].astype(np.float32)), tr["y"][keep])
    tmask = te["resp"] < len(let)
    p = clf.predict_proba(sc.transform(te["X"][tmask].astype(np.float32)))[:, 1]
    ids = te["resp_ids"][:len(let)]
    ours = response_scores(p, te["resp"][tmask], ids)
    y = te["resp_y"][:len(let)]
    task = te["resp_task"][:len(let)]
    lp = np.array([r["max_prob"] for r in let])
    Z = np.c_[logit(lp), logit(ours)]
    res = {"test_answers": len(let), "internals_train_answers": a.train_answers,
           "auroc": {"lettuce": round(float(auroc(lp, y.astype(float))), 3),
                     "internals": round(float(auroc(ours, y.astype(float))), 3)},
           "lettuce_published_rule": dict(prf(lp >= 0.5, y), by_task={t: prf((lp >= 0.5)[task == t], y[task == t])
                                                                        for t in sorted(set(task))}),
           "correlation_of_logits": round(float(np.corrcoef(Z.T)[0, 1]), 3)}
    rng = np.random.default_rng(0)
    out = {k: {"f1": [], "precision": [], "recall": [], "auroc": []} for k in ("lettuce", "internals", "combined")}
    weights = []
    for rep in range(20):
        perm = rng.permutation(len(y))
        halves = (perm[:len(y) // 2], perm[len(y) // 2:])
        pred = {k: np.zeros(len(y)) for k in out}
        flag = {k: np.zeros(len(y), bool) for k in out}
        for fit_i, ev_i in (halves, halves[::-1]):
            m = LogisticRegression().fit(Z[fit_i], y[fit_i])
            weights.append(m.coef_[0].tolist())
            for k, s in (("lettuce", Z[:, 0]), ("internals", Z[:, 1]), ("combined", m.decision_function(Z))):
                thr = best_thr(s[fit_i], y[fit_i])
                pred[k][ev_i], flag[k][ev_i] = s[ev_i], s[ev_i] >= thr
        for k in out:
            r = prf(flag[k], y)
            for m_ in ("f1", "precision", "recall"):
                out[k][m_].append(r[m_])
            out[k]["auroc"].append(float(auroc(pred[k], y.astype(float))))
    res["cross_fitted"] = {k: {m_: [round(float(np.mean(v)), 1 if m_ != "auroc" else 3),
                                    round(float(np.std(v)), 1 if m_ != "auroc" else 3)] for m_, v in d.items()}
                           for k, d in out.items()}
    res["combined_weights_mean"] = np.mean(weights, 0).round(3).tolist()
    # paired: how often the combination beats LettuceDetect on the same split
    res["combined_beats_lettuce_share"] = round(float(np.mean(np.array(out["combined"]["f1"]) >
                                                              np.array(out["lettuce"]["f1"]))), 2)
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.report, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "combine"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--shards", default="")
    ap.add_argument("--train-answers", type=int, default=2400)
    ap.add_argument("--report", default="results/ragtruth_supervised/lettuce_ensemble.json")
    a = ap.parse_args()
    {"score": score, "combine": combine}[a.cmd](a)


if __name__ == "__main__":
    main()
