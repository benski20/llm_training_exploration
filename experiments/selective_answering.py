"""Selective answering: does a model's own signal turn its hallucinations into abstentions?

Every model answers; a reply is withheld (replaced by "I don't know") when a confidence
signal from the same forward pass falls below a threshold. The threshold is chosen on a
calibration set, as the most selective one whose over-refusal stays within a fixed budget,
then applied unchanged to the test questions (the same ones as in glassbox_chat.py). The
GPT gets the same rule with each of its own signals, including a post-hoc probe.

    python experiments/selective_answering.py --models baseline,glassbox2_1 --out results/selective_answering.json
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
from interp_by_design import fit_probe  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

DATA = "results/glassbox_chat_v2"
MODELS = {
    "baseline": ("GPT", lambda V: C.Base(V, 256), f"{DATA}/baseline.baseline.pt"),
    "glassbox2_1": ("Glassbox-2.1", lambda V: C.Glass2(V, 256, 0.002, 5, 32, 0, "hard"),
                    "results/glassbox2_1_chat/glassbox2.glassbox2.pt"),
    "glassbox3": ("Glassbox-3", lambda V: C.Glass3(V, 256, 0.002, 5, 32, 0),
                  "results/glassbox3_chat/glassbox3.glassbox3.pt"),
}
SIGNALS = ("msp", "neg_nll", "k_min", "probe")
ANSWERABLE = ("ctx", "mem", "ctx_fresh")


def signal(r, name):
    if name == "neg_nll":
        return -r["nll"]
    return r.get(name)


def outcome(rows, name, th):
    """Hallucination (answers to unanswerable questions), over-refusal and accuracy when
    replies with signal < th are withheld."""
    def abst(r):
        return r["abstained"] or (th is not None and signal(r, name) < th)

    def right(r):
        return abst(r) if r["kind"].startswith("absent") else (not abst(r) and r["right"])
    un = [r for r in rows if r["kind"] == "absent"]
    fresh = [r for r in rows if r["kind"] == "absent_fresh"]
    ans = [r for r in rows if r["kind"] in ANSWERABLE]
    return {"hallucination": float(np.mean([not abst(r) for r in un])),
            "hallucination_made_up": float(np.mean([not abst(r) for r in fresh])),
            "over_refusal": float(np.mean([abst(r) for r in ans])),
            "answered_right": float(np.mean([right(r) for r in ans])),
            "right": float(np.mean([right(r) for r in rows]))}


def choose(rows, name, budget):
    """The highest threshold (most withholding) whose over-refusal stays <= budget."""
    vals = sorted({signal(r, name) for r in rows if not r["abstained"]})
    best = None
    for th in vals:
        if outcome(rows, name, th)["over_refusal"] <= budget:
            best = th
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="baseline,glassbox2_1")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=float, default=0.267, help="over-refusal allowed on calibration questions")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/selective_answering.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    V = tok.get_vocab_size()
    world = C.ChatWorld(n_ent=300)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for key in a.models.split(","):
        name, make, ckpt = MODELS[key]
        model = make(V)
        model.m.load_state_dict(torch.load(ckpt)["model"])
        model.m.eval()
        sets = {}
        for split, seed in (("probe_fit", 21), ("calibration", 22), ("test", 11)):
            _, rows = C.chat_eval(model, tok, world, random.Random(seed), a.n, C.QA_RELS, 256)
            sets[split] = rows
        fit = sets["probe_fit"]                     # post-hoc probe on the last prompt token, best layer
        best = None
        for L in range(len(fit[0]["h"])):
            pr, _ = fit_probe(np.stack([r["h"][L] for r in fit]), np.array([r["right"] for r in fit], dtype=int))
            sc = C.auroc(pr(np.stack([r["h"][L] for r in sets["calibration"]])),
                         np.array([r["right"] for r in sets["calibration"]], dtype=int))
            if best is None or sc > best[0]:
                best = (sc, L, pr)
        for split in ("calibration", "test"):
            ps = best[2](np.stack([r["h"][best[1]] for r in sets[split]]))
            for r, p in zip(sets[split], ps):
                r["probe"] = float(p)
        out = {"no_rule": outcome(sets["test"], "msp", None)}
        for s in SIGNALS:
            if signal(sets["test"][0], s) is None:
                continue
            th = choose(sets["calibration"], s, a.budget)
            out[s] = {"threshold": th, "calibration": outcome(sets["calibration"], s, th),
                      "test": outcome(sets["test"], s, th)}
        res[key] = {"name": name, "budget": a.budget, **out}
        print(json.dumps({key: res[key]}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
