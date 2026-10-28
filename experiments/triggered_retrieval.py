"""Retrieve only when the model doubts itself: does the internal-doubt trigger buy more accuracy per
retrieval than token probability (FLARE-style) or retrieving at random?

For each TriviaQA question (rc.wikipedia: each comes with its Wikipedia evidence pages):
  closed book   the model answers alone; the doubt-aware generator records the trigger scores
                (probe P(correct) averaged with token probability, and token probability alone)
  retrieved     the top two passages of the evidence pages (word overlap with the question) are given
                as a document and the model answers again
A policy retrieves for the share b of questions it is least sure about; accuracy is the retrieved
answer for those and the closed-book answer for the rest. Same answers for every policy, so the
curves differ only in *which* questions retrieve.

    python experiments/triggered_retrieval.py --n 200 --partial <rows.json>
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.correctness import SYSTEM, is_right, norm  # noqa: E402
from vt.guided import DoubtAwareGenerator  # noqa: E402
from vt.verichat import load_head  # noqa: E402

STOP = set("the a an of in on at to for and or is was were who what which when where how why did does do by with from as".split())


def retrieve(question, pages, k=2, size=600):
    q = {w for w in norm(question).split() if w not in STOP}
    chunks = []
    for page in pages:
        for i in range(0, min(len(page), 60000), size // 2):
            c = page[i:i + size]
            words = set(norm(c).split())
            chunks.append((len(q & words), c))
    chunks.sort(key=lambda x: -x[0])
    return "\n...\n".join(c for _, c in chunks[:k])


def answer(gen, question, doc=None):
    if doc:
        msgs = [{"role": "system", "content": "You are a helpful assistant. Use the document if it helps. Answer with a short answer."},
                {"role": "user", "content": f"Document:\n{doc}\n\nQuestion: {question}"}]
        r = gen.generate(msgs, 40, "flag", context=question)
    else:
        r = gen.generate([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}], 40, "flag")
    first = r["claims"][0] if r["claims"] else (r["first_token"] or {"p_correct": 1.0, "token_p": 1.0})
    return r["text"], float(first["p_correct"]), float(first["token_p"])


def curve(rows, score, budgets, seed=0):
    y_cb = np.array([r["right_closed"] for r in rows])
    y_rt = np.array([r["right_retrieved"] for r in rows])
    s = np.array([score(r) for r in rows]) if score else np.random.default_rng(seed).random(len(rows))
    order = np.argsort(s, kind="mergesort")                       # least sure first
    out = {}
    for b in budgets:
        k = int(round(b * len(rows)))
        retrieved = np.zeros(len(rows), bool)
        retrieved[order[:k]] = True
        out[f"{int(b * 100)}pct"] = round(float(np.where(retrieved, y_rt, y_cb).mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--head", default="results/kaggle/qwen-correctness/correctness/Qwen2.5-1.5B-Instruct_heads.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--partial", required=True)
    ap.add_argument("--out", default="results/triggered_retrieval.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    probe, layer = load_head(a.head)
    gen = DoubtAwareGenerator(model, tok, probe, layer, threshold=0.0, combine=False)
    rows = json.load(open(a.partial)) if os.path.exists(a.partial) else []
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation", streaming=True)
    t0 = time.time()
    for i, x in enumerate(ds):
        if i >= a.n:
            break
        if i < len(rows):
            continue
        q = x["question"]
        aliases = x["answer"]["normalized_aliases"] + [x["answer"]["value"]]
        doc = retrieve(q, x["entity_pages"]["wiki_context"])
        cb, p, tp = answer(gen, q)
        rt, _, _ = answer(gen, q, doc)
        rows.append({"q": q, "closed": cb, "retrieved": rt, "probe_p": p, "token_p": tp,
                     "right_closed": is_right(cb, aliases), "right_retrieved": is_right(rt, aliases),
                     "doc_has_answer": int(any(norm(al) and norm(al) in norm(doc) for al in aliases))})
        if (i + 1) % 10 == 0:
            json.dump(rows, open(a.partial, "w"))
            print(json.dumps({"done": i + 1, "closed": float(np.mean([r["right_closed"] for r in rows])),
                              "retrieved": float(np.mean([r["right_retrieved"] for r in rows])), "s": round(time.time() - t0)}), flush=True)
    json.dump(rows, open(a.partial, "w"))
    budgets = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
    res = {"model": a.model, "n": len(rows),
           "closed_book": float(np.mean([r["right_closed"] for r in rows])),
           "always_retrieve": float(np.mean([r["right_retrieved"] for r in rows])),
           "oracle_best_of_both": float(np.mean([max(r["right_closed"], r["right_retrieved"]) for r in rows])),
           "passage_contains_answer": float(np.mean([r["doc_has_answer"] for r in rows])),
           "curves": {"probe_and_token_p": curve(rows, lambda r: (r["probe_p"] + r["token_p"]) / 2, budgets),
                      "probe": curve(rows, lambda r: r["probe_p"], budgets),
                      "token_probability": curve(rows, lambda r: r["token_p"], budgets),
                      "random_mean_of_5": {k: round(float(np.mean([curve(rows, None, budgets, s)[k] for s in range(5)])), 4)
                                           for k in curve(rows, None, budgets, 0)}}}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
