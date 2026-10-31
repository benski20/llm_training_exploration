"""Does the evidence change the model's belief in what was written? A training-free signal from the model's own
probabilities, no judge and no probe: the reply is scored twice by teacher forcing, with the evidence in the
context and without it, and each token's log-probability shift is

    shift = log p(token | evidence, question, reply so far) - log p(token | question, reply so far)

Evidence that supports a claim makes its words more likely (shift > 0), evidence that contradicts it makes them
much less likely (shift << 0), evidence that does not bear on it leaves them alone (shift ~ 0). Scored on
RAGTruth's test set (answers written from documents, hallucinated spans labeled by people: "evident conflict"
contradicts the documents, "baseless info" is not in them), at token, sentence and response level, next to the
plain context-conditioned log-probability (no second pass). Reference points from this repo: the 32B's trained
risk head left-out-source RAGTruth sentence AUROC 0.58 (results/kaggle/risk-32b2), trained in-distribution
detector response AUROC 0.86, provenance (document share) token AUROC 0.76 (VISION.md).

    python experiments/evidence_shift.py --model Qwen/Qwen3-4B --n 200 --out results/evidence_shift_qwen3_4b.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def spans(row):
    """[(start, end, kind)] of the hallucinated character spans in the output."""
    out = []
    for h in json.loads(row["hallucination_labels"] or "[]"):
        out.append((int(h["start"]), int(h["end"]), h.get("label_type", "")))
    return out


def token_logprobs(model, tok, prefix_ids, reply_ids):
    """log p of each reply token given the prefix and the reply so far (one forward pass)."""
    import torch
    ids = torch.tensor([prefix_ids + reply_ids])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, len(prefix_ids) - 1:-1].float()
    lp = torch.log_softmax(logits, -1)
    return lp[torch.arange(len(reply_ids)), torch.tensor(reply_ids)].numpy()


def prefix(tok, user):
    msgs = [{"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(p, add_special_tokens=False).input_ids


def sentences(text):
    """[(start, end)] character ranges of the sentences of a text."""
    out, start = [], 0
    for m in re.finditer(r"[.!?](?=\s)|\n+", text):
        out.append((start, m.end()))
        start = m.end()
    if start < len(text):
        out.append((start, len(text)))
    return [(a, b) for a, b in out if text[a:b].strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-context-chars", type=int, default=4000)
    ap.add_argument("--out", default="evidence_shift.json")
    a = ap.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.correctness import auroc
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    ds = load_dataset("wandb/RAGTruth-processed", split="test").shuffle(seed=0)
    bad = [r for r in ds if spans(r)]
    good = [r for r in ds if not spans(r)]
    rows = bad[:a.n // 2] + good[:a.n - a.n // 2]               # balanced: half with hallucinations
    tok_rows, sent_rows, resp_rows = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        out = r["output"]
        enc = tok(out, add_special_tokens=False, return_offsets_mapping=True)
        reply_ids, offs = enc.input_ids, enc.offset_mapping
        ctx = r["context"][:a.max_context_chars]
        with_ev = token_logprobs(model, tok, prefix(tok, f"{r['query']}\n\n{ctx}"), reply_ids)
        without = token_logprobs(model, tok, prefix(tok, r["query"]), reply_ids)
        shift = with_ev - without
        sp = spans(r)

        def label(s, e):
            kinds = [k for a_, b_, k in sp if s < b_ and a_ < e]
            return (1 if kinds else 0), (kinds[0] if kinds else "")
        content = [bool(re.search(r"[A-Za-z0-9]", out[s:e])) for s, e in offs]
        for j, (s, e) in enumerate(offs):
            if content[j]:
                y, kind = label(s, e)
                tok_rows.append((y, kind, float(with_ev[j]), float(shift[j])))
        sent_scores = []
        for s, e in sentences(out):
            js = [j for j, (ts, te) in enumerate(offs) if content[j] and ts >= s and te <= e]
            if not js:
                continue
            y, kind = label(s, e)
            lw, sh = with_ev[js], shift[js]
            row = (y, kind, float(lw.min()), float(sh.min()), float(np.sort(sh)[:3].mean()), float(sh.mean()))
            sent_rows.append(row)
            sent_scores.append(row)
        if sent_scores:
            resp_rows.append((int(bool(sp)), min(x[2] for x in sent_scores), min(x[3] for x in sent_scores),
                              min(x[4] for x in sent_scores)))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)

    def au(rows_, col, sign=-1, keep=lambda r: True):
        rr = [r for r in rows_ if keep(r)]
        return round(float(auroc(np.array([sign * r[col] for r in rr]), np.array([r[0] for r in rr], float))), 3)
    conflict = lambda r: r[0] == 0 or "Conflict" in r[1]               # noqa: E731
    baseless = lambda r: r[0] == 0 or "Baseless" in r[1]               # noqa: E731
    res = {"model": a.model, "n_responses": len(resp_rows), "seconds": round(time.time() - t0),
           "token": {"n": len(tok_rows), "positives": sum(r[0] for r in tok_rows),
                     "logprob_with_evidence": au(tok_rows, 2), "evidence_shift": au(tok_rows, 3)},
           "sentence": {"n": len(sent_rows), "positives": sum(r[0] for r in sent_rows),
                        "min_logprob_with_evidence": au(sent_rows, 2), "min_shift": au(sent_rows, 3),
                        "lowest3_shift": au(sent_rows, 4), "mean_shift": au(sent_rows, 5),
                        "min_shift_conflicts_only": au(sent_rows, 3, keep=conflict),
                        "min_shift_baseless_only": au(sent_rows, 3, keep=baseless)},
           "response": {"n": len(resp_rows), "min_logprob_with_evidence": au(resp_rows, 1),
                        "min_shift": au(resp_rows, 2), "lowest3_shift": au(resp_rows, 3)}}
    print(json.dumps(res, indent=1))
    json.dump(dict(res, sentence_rows=sent_rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
