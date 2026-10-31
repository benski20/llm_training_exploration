"""Trains the claim typer: is a sentence of a reply a factual claim (true or false about the world, or about a
text the user gave) or not (an opinion, a suggestion, a question, fiction, a greeting)? A sentence embedding
(bge-small, CPU) and a two-way softmax layer, from:

    ClaimBuster (CC-BY-4.0): 23.5k sentences from US presidential debates, labeled by people:
                             non-factual vs factual (unimportant or check-worthy)
    Dolly responses (CC-BY-SA): sentences of factual answers (open, closed and general QA, extraction,
                             summaries) as claims; of creative writing and brainstorming as not

Scores held out: a ClaimBuster split and results/claim_type_holdout.json (120 chatbot-reply sentences
labeled by hand, never trained on). Writes vt/claim_type_head.npz.

    python experiments/train_claim_type.py --out vt/claim_type_head.npz --report results/claim_type_train.json
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from train_router import log_loss, train  # noqa: E402

KINDS = ("not a claim", "fact")
DOLLY_FACT = {"open_qa", "closed_qa", "general_qa", "information_extraction", "summarization"}
DOLLY_NOT = {"creative_writing", "brainstorming"}


def load(seed=0, dolly_per_side=6000):
    from datasets import load_dataset

    from vt.claim_verify import BOLD, sentences
    cb = [(r["text"], int(r["checkworthiness"]) != 2) for r in load_dataset("Nithiwat/claimbuster")["train"]]
    rng = random.Random(seed)
    rng.shuffle(cb)
    cut = int(0.9 * len(cb))
    fact, nofact = [], []
    for r in load_dataset("databricks/databricks-dolly-15k")["train"]:
        side = fact if r["category"] in DOLLY_FACT else nofact if r["category"] in DOLLY_NOT else None
        if side is not None:
            side += [s.rstrip(BOLD) for s in sentences(r["response"]) if len(s.split()) >= 4]
    rng.shuffle(fact)
    rng.shuffle(nofact)
    dolly = [(s, True) for s in fact[:dolly_per_side]] + [(s, False) for s in nofact[:dolly_per_side]]
    held = {x["s"].lower() for x in json.load(open("results/claim_type_holdout.json"))}
    train_rows = [(s, y) for s, y in cb[:cut] + dolly if s.lower() not in held]
    rng.shuffle(train_rows)
    return train_rows, cb[cut:]


def scores(p_fact, y, threshold=0.5):
    pred = p_fact >= threshold
    y = np.asarray(y, bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    from vt.correctness import auroc
    return {"n": len(y), "accuracy": round(float((pred == y).mean()), 3),
            "fact_recall": round(tp / max(1, y.sum()), 3), "not_a_claim_taken_for_fact": round(fp / max(1, (~y).sum()), 3),
            "auroc": round(float(auroc(p_fact, y.astype(float))), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cache", help="save / reuse the training embeddings here (.npz)")
    ap.add_argument("--out", default="vt/claim_type_head.npz")
    ap.add_argument("--report", default="results/claim_type_train.json")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    rows, cb_test = load()
    emb = SentenceTransformer(a.embed)
    enc = lambda texts: emb.encode([t[:1000] for t in texts], batch_size=64, normalize_embeddings=True,  # noqa: E731
                                   show_progress_bar=False)
    y = np.array([int(f) for _, f in rows])
    if a.cache and os.path.exists(a.cache):
        X = np.load(a.cache)["X"]
    else:
        X = enc([s for s, _ in rows])
        if a.cache:
            np.savez(a.cache, X=X)
    cut = int(0.9 * len(rows))
    fits = {}
    for l2 in (1e-5, 1e-4, 1e-3):
        W, b = train(X[:cut], y[:cut], 2, l2=l2)
        fits[l2] = log_loss(X[cut:], y[cut:], W, b)
        print(f"l2 {l2:g}: validation log loss {fits[l2]:.3f}", flush=True)
    l2 = min(fits, key=fits.get)
    W, b = train(X, y, 2, l2=l2)
    np.savez(a.out, W=W.astype(np.float32), b=b.astype(np.float32), kinds=np.array(KINDS), embed=np.array(a.embed))

    def p_fact(texts):
        Z = enc(texts) @ W + b
        Z = np.exp(Z - Z.max(1, keepdims=True))
        return (Z / Z.sum(1, keepdims=True))[:, 1]
    hold = json.load(open("results/claim_type_holdout.json"))
    from vt.claim_verify import check_worthy
    report = {"embed": a.embed, "l2": l2, "train": {"n": len(rows), "facts": int(y.sum())},
              "claimbuster_heldout": scores(p_fact([s for s, _ in cb_test]), [f for _, f in cb_test]),
              "chatbot_sentences_heldout": scores(p_fact([x["s"] for x in hold]), [x["fact"] for x in hold]),
              "heuristic_on_chatbot_sentences": scores(np.array([float(check_worthy(x["s"])) for x in hold]),
                                                       [x["fact"] for x in hold])}
    pf = p_fact([x["s"] for x in hold])
    report["chatbot_sentences_wrong"] = [(x["fact"], round(float(p), 2), x["s"]) for x, p in zip(hold, pf)
                                         if (p >= 0.5) != x["fact"]]
    print(json.dumps(report, indent=1))
    json.dump(report, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
