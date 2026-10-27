"""Chat fact evaluation with enough questions for confidence intervals: is a difference
between two models larger than the evaluation noise?

    python experiments/chat_eval_large.py --model glassbox2_2 --n 1500
"""
import argparse
import json
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import glassbox_chat as C  # noqa: E402
from selective_answering import DATA, MODELS  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

MODELS = dict(MODELS, glassbox2_2=("Glassbox-2.2", lambda V: C.Glass2(V, 256, 0.002, 5, 32, 0, "hard_st"),
                                   "results/glassbox2_2_chat/glassbox2.glassbox2.pt"))


def rate(xs):
    """Mean and 95% Wilson interval."""
    n, k = len(xs), sum(xs)
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": round(p, 4), "ci95": [round(c - h, 4), round(c + h, 4)], "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/chat_eval_large.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    name, make, ckpt = MODELS[a.model]
    model = make(tok.get_vocab_size())
    model.m.load_state_dict(torch.load(ckpt)["model"])
    model.m.eval()
    world = C.ChatWorld(n_ent=300)
    out = {"name": name}
    for split, rels in (("chat_facts", C.QA_RELS), ("chat_facts_held_relations", C.HELD_RELS)):
        _, rows = C.chat_eval(model, tok, world, random.Random(a.seed), a.n, rels, 256)
        out[split] = {
            "hallucination_unstated": rate([not r["abstained"] for r in rows if r["kind"] == "absent"]),
            "hallucination_made_up": rate([not r["abstained"] for r in rows if r["kind"] == "absent_fresh"]),
            "over_refusal": rate([r["abstained"] for r in rows if r["kind"] in ("ctx", "mem", "ctx_fresh")]),
            "ctx_right": rate([r["right"] for r in rows if r["kind"] == "ctx"])}
        print(json.dumps({a.model: {split: out[split]}}), flush=True)
    path = a.out
    res = json.load(open(path)) if os.path.exists(path) else {}
    res[a.model] = out
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
