"""Exact attribution as a hallucination check.

For each reply that answers (does not abstain), take its riskiest token (lowest k) and decompose
that token's logit exactly into contributions of every input token, memory read and bias
(Glassbox2.explain). A grounded answer draws its content from a source token in the
conversation; a hallucination has none to draw from. Score: the largest share of the
positive contribution that comes from a single conversation token. Compared with the model's
other signals on the same replies (right answers vs hallucinations).

    python experiments/attribution_grounding.py --model glassbox2_3
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import glassbox_chat as C  # noqa: E402
from chat_eval_large import MODELS  # noqa: E402
from selective_answering import DATA  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


@torch.no_grad()
def grounding(m, ids, P):
    """At the reply token with the lowest k: (largest single-token share of the positive
    contribution, share from all conversation tokens, share from biases)."""
    x = torch.tensor([ids])
    m(x)
    k = m.know(m.trace["z_f"]).squeeze(-1)[0]
    span = list(range(P - 1, len(ids) - 1))                      # positions that emit reply tokens
    t = min(span, key=lambda i: float(k[i]))
    parts = m.explain(x, t, ids[t + 1])
    pos = {n: v for n, v in parts.items() if v > 0}
    tot = sum(pos.values()) or 1.0
    conv = {n: v for n, v in pos.items() if n.startswith("token@") and int(n[6:]) < P}
    return {"top_source_share": max(conv.values(), default=0.0) / tot,
            "conversation_share": sum(conv.values()) / tot,
            "bias_share": sum(v for n, v in pos.items() if n.startswith("bias@")) / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glassbox2_3")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/attribution_grounding.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    name, make, ckpt = MODELS[a.model]
    model = make(tok.get_vocab_size())
    model.m.load_state_dict(torch.load(ckpt)["model"])
    model.m.eval()
    world = C.ChatWorld(n_ent=300)
    rng = random.Random(202)
    rows = []
    for i in range(a.n):
        prompt, _, gold, kind = world.question(rng, world.KINDS[i % len(world.KINDS)], C.QA_RELS, test=True)
        out, f = C.respond(model, tok, prompt, 256)
        abst = any(mk in out.lower() for mk in C.ABSTAIN_MARKERS)
        if abst:
            continue
        ids = f["ids"][0].tolist()
        P = len(tok.encode(prompt).ids)
        if len(ids) <= P:
            continue
        rows.append({"kind": kind, "right": gold is not None and gold in out, "msp": f["msp"],
                     "neg_nll": -f["nll"], "k_min": f["k_min"], **grounding(model.m, ids, P)})
    y = np.array([r["right"] for r in rows], dtype=int)
    res = {"model": name, "answered": len(rows), "answered_right": int(y.sum()),
           "auroc": {s: C.auroc(np.array([r[s] for r in rows]), y)
                     for s in ("top_source_share", "conversation_share", "msp", "neg_nll", "k_min")},
           "auroc_neg_bias_share": C.auroc(-np.array([r["bias_share"] for r in rows]), y),
           "mean_by_outcome": {lab: {s: float(np.mean([r[s] for r in rows if r["right"] == v]))
                                     for s in ("top_source_share", "conversation_share", "bias_share")}
                               for lab, v in (("right", True), ("hallucinated", False))}}
    # grounding proper: answers to questions whose fact is in the conversation vs hallucinations
    # (memorised facts legitimately do not come from the conversation, so they are left out)
    g = [r for r in rows if r["kind"] in ("ctx", "ctx_fresh") and r["right"] or r["kind"].startswith("absent")]
    yg = np.array([r["right"] for r in g], dtype=int)
    res["grounding_subset"] = {"n": len(g), "grounded_right": int(yg.sum()),
                               "auroc": {s: C.auroc(np.array([r[s] for r in g]), yg)
                                         for s in ("top_source_share", "conversation_share", "msp", "neg_nll",
                                                   "k_min")}}
    # does attribution add to confidence? logistic fit on one half, AUROC on the other
    from interp_by_design import fit_probe
    for name_, sub, ys in (("all_answered", rows, y), ("grounding_subset", g, yg)):
        half = len(sub) // 2
        def feats(rs, cols):
            return np.array([[r[c] for c in cols] for r in rs], dtype=float)
        out = {}
        for label, cols in (("neg_nll", ["neg_nll"]), ("neg_nll+attribution", ["neg_nll", "top_source_share",
                                                                               "conversation_share", "bias_share"])):
            pr, _ = fit_probe(feats(sub[:half], cols), ys[:half])
            out[label] = C.auroc(pr(feats(sub[half:], cols)), ys[half:])
        res.setdefault("combined_heldout_auroc", {})[name_] = out
    json.dump(rows, open(a.out.replace(".json", f"_{a.model}_rows.json"), "w"))
    print(json.dumps(res, indent=1))
    allres = json.load(open(a.out)) if os.path.exists(a.out) else {}
    allres[a.model] = res
    json.dump(allres, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
