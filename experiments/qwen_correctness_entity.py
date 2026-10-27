"""Correctness signals read at the answer's entity rather than its first token.

The first answer token is often a function word ("The Chipmunks were created by Jim Henson"). Here
the explained position is the first token of the answer's first new name or number: a capitalised
or numeric word that is not in the question ("Jim"). Same questions, prompt and halves as
qwen_correctness.py; stronger output baselines: the whole answer's mean and minimum token
log-probability. Attribution groups: each question token, the chat template, and the answer's own
earlier tokens.

    python experiments/qwen_correctness_entity.py --n 1000 --rows <rows.pkl>
"""
import argparse
import json
import os
import pickle
import string
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "qwen_glassbox"))
import run as R  # noqa: E402
from qwen_correctness import question_groups  # noqa: E402
from vt.qwen_glassbox import explain  # noqa: E402

SKIP = {"the", "a", "an", "in", "it", "its", "this", "that", "yes", "no", "i"}


def entity_index(tok, ans_ids, question):
    """Index (into ans_ids) of the first token of the answer's first new capitalised or numeric word."""
    qwords = set(R.norm(question).split())
    starts = [j for j in range(len(ans_ids)) if j == 0 or tok.decode(ans_ids[j:j + 1]).startswith(" ")]
    for a, b in zip(starts, starts[1:] + [len(ans_ids)]):
        w = tok.decode(ans_ids[a:b]).strip().strip(string.punctuation + "\"'“”")
        if w and (w[0].isupper() or w[0].isdigit()) and R.norm(w) not in qwords and w.lower() not in SKIP:
            return a
    return 0


def stats(parts):
    pos = {n: v for n, v in parts.items() if v > 0}
    tot = sum(pos.values()) or 1.0
    q = sum(v for n, v in parts.items() if n.startswith("q@"))
    return {"question_signed": q, "answer_signed": parts.get("answer", 0.0),
            "template_signed": parts.get("template", 0.0),
            "question_share": sum(v for n, v in pos.items() if n.startswith("q@")) / tot,
            "answer_share": pos.get("answer", 0.0) / tot, "logit": sum(parts.values())}


@torch.no_grad()
def collect(model, tok, items, log_every=25):
    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        prompt = R.chat(tok, it["q"])
        ids = tok(prompt, return_tensors="pt").input_ids
        P = ids.shape[1]
        gen = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        ans_ids = gen[0, P:].tolist()
        while ans_ids and ans_ids[-1] == tok.eos_token_id:
            ans_ids.pop()
        if not ans_ids:
            continue
        ans = tok.decode(ans_ids, skip_special_tokens=True).strip()
        right = any(R.norm(al) and R.norm(al) in R.norm(ans) for al in it["aliases"])
        full = gen[:, :P + len(ans_ids)]
        out = model(full, output_hidden_states=True)
        lp = F.log_softmax(out.logits[0].float(), -1)
        tok_lp = lp[torch.arange(P - 1, P - 1 + len(ans_ids)), torch.tensor(ans_ids)]
        e = entity_index(tok, ans_ids, it["q"])
        t = P - 1 + e                                   # predicts the entity's first token
        target = ans_ids[e]
        groups = question_groups(tok, prompt, it["q"]) + ["answer"] * e
        parts, logit = explain(model, full[:, :t + 1], t, target, groups)
        r = {"q": it["q"], "answer": ans, "right": int(right), "entity": tok.decode([target]), "e": e,
             "entity_p": float(tok_lp[e].exp()), "first_p": float(tok_lp[0].exp()),
             "mean_lp": float(tok_lp.mean()), "min_lp": float(tok_lp.min()),
             "h_entity": [h[0, t].float().numpy() for h in out.hidden_states[1:]],
             "exact_err": abs(sum(parts.values()) - logit), **stats(parts)}
        rows.append(r)
        if (i + 1) % log_every == 0:
            R.log(phase="collect", done=i + 1, acc=float(np.mean([r["right"] for r in rows])),
                  s=round(time.time() - t0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rows", required=True, help="pickle the per-question rows here (large)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    rows = collect(model, tok, R.load_trivia(a.n, False))
    pickle.dump(rows, open(a.rows, "wb"))
    R.log(phase="done", n=len(rows), max_exact_err=max(r["exact_err"] for r in rows))


if __name__ == "__main__":
    main()
