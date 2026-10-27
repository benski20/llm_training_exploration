"""How well do a model's internals tell whether its answer is correct?

On chat fact questions, for every reply that answers (does not abstain), at the answer word's
first token: is the answer right? Detectors, each fitted on one half of the replies and scored
on the other half (AUROC, and the error rate among the 50% most confident answers):

  both models   softmax probability and NLL of the answer token; a post-hoc logistic probe on
                the hidden state (best layer chosen on the fitting half)
  Glassbox      built-in confidence k; a logistic probe on which sparse features fired (exact,
                readable: its largest weights name the features that signal a wrong answer);
                exact attribution statistics of the answer token's score; all combined

    python experiments/correctness_from_internals.py --models baseline,glassbox2_4 --n 1500
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import glassbox_chat as C  # noqa: E402
from chat_eval_large import MODELS  # noqa: E402
from explain_demo import feature_labels  # noqa: E402
from interp_by_design import auroc, fit_probe  # noqa: E402
from selective_answering import DATA  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def answer_position(tok, ids, P):
    """Prediction position of the reply's last word (the answer value), and that token."""
    span = list(range(P - 1, len(ids) - 1))
    words = [i for i in span if tok.decode([ids[i + 1]]).startswith(" ")]
    t = words[-1] if words else span[-1]
    return t, ids[t + 1]


@torch.no_grad()
def collect(model, tok, world, n, rels, seed, glass):
    rows = []
    rng = random.Random(seed)
    for i in range(n):
        prompt, _, gold, kind = world.question(rng, world.KINDS[i % len(world.KINDS)], rels, test=True)
        out, f = C.respond(model, tok, prompt, 256)
        if any(mk in out.lower() for mk in C.ABSTAIN_MARKERS):
            continue
        ids = f["ids"][0].tolist()
        P = len(tok.encode(prompt).ids)
        if len(ids) <= P:
            continue
        t, target = answer_position(tok, ids, P)
        x = torch.tensor([ids])
        lp, ro, hs = model.readouts(x)
        r = {"kind": kind, "right": int(gold is not None and gold in out),
             "msp": float(lp[0, t].max().exp()), "neg_nll": float(lp[0, t, target]),
             "h": [h[0, t].numpy() for h in hs]}
        if glass:
            m = model.m
            r["k"] = float(ro["k"][0, t])
            r["feats"] = np.concatenate([(m.trace["feat"][L]["acts"][0, t] != 0).numpy()
                                         for L in range(m.config.n_layer)]).astype(np.float32)
            parts = m.explain(x, t, target)
            vals = np.array(list(parts.values()))
            pos = {n_: v for n_, v in parts.items() if v > 0}
            tot = sum(pos.values()) or 1.0
            conv = [v for n_, v in pos.items() if n_.startswith("token@") and int(n_[6:]) < P]
            p = np.array([v / tot for v in pos.values()])
            r["attr"] = np.array([sum(conv) / tot, max(conv, default=0.0) / tot,
                                  sum(v for n_, v in pos.items() if n_.startswith("bias@")) / tot,
                                  float(-(p * np.log(p + 1e-12)).sum()), float(vals.sum())], dtype=np.float32)
        rows.append(r)
    return rows


def risk_at_half(scores, y):
    keep = np.argsort(-np.asarray(scores))[: len(y) // 2]
    return float(1 - np.asarray(y)[keep].mean())


def evaluate(rows, glass):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    fit, test = [rows[i] for i in idx[: len(idx) // 2]], [rows[i] for i in idx[len(idx) // 2:]]
    yf, yt = np.array([r["right"] for r in fit]), np.array([r["right"] for r in test])
    out = {"answered": len(rows), "right": int(sum(r["right"] for r in rows)), "test_n": len(test),
           "error_rate_all": float(1 - yt.mean())}

    def score(name, s):
        out[name] = {"auroc": auroc(np.asarray(s, float), yt), "error_at_50pct": risk_at_half(s, yt)}

    score("softmax", [r["msp"] for r in test])
    score("neg_nll", [r["neg_nll"] for r in test])
    best = None                                  # post-hoc probe on hidden states, layer chosen on fit half
    for L in range(len(fit[0]["h"])):
        Xf = np.stack([r["h"][L] for r in fit])
        h = len(fit) // 2
        pr, _ = fit_probe(Xf[:h], yf[:h])
        a = auroc(pr(Xf[h:]), yf[h:])
        if best is None or a > best[0]:
            best = (a, L)
    pr, _ = fit_probe(np.stack([r["h"][best[1]] for r in fit]), yf)
    score("posthoc_probe", pr(np.stack([r["h"][best[1]] for r in test])))
    out["posthoc_probe"]["layer"] = best[1]
    if not glass:
        return out, None
    score("builtin_k", [r["k"] for r in test])
    pr_f, w = fit_probe(np.stack([r["feats"] for r in fit]), yf)
    score("feature_probe", pr_f(np.stack([r["feats"] for r in test])))
    pr_a, _ = fit_probe(np.stack([r["attr"] for r in fit]), yf)
    score("attribution", pr_a(np.stack([r["attr"] for r in test])))

    def comb(r):
        return np.concatenate([[r["msp"], r["neg_nll"], r["k"]], r["attr"], r["feats"]])
    pr_c, _ = fit_probe(np.stack([comb(r) for r in fit]), yf)
    score("combined", pr_c(np.stack([comb(r) for r in test])))
    return out, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="baseline,glassbox2_4")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/correctness_from_internals.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    world = C.ChatWorld(n_ent=300)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for key in a.models.split(","):
        name, make, ckpt = MODELS[key]
        model = make(tok.get_vocab_size())
        model.m.load_state_dict(torch.load(ckpt)["model"])
        model.m.eval()
        glass = key != "baseline" and not key.startswith("baseline")
        rows = collect(model, tok, world, a.n, C.QA_RELS, a.seed, glass)
        out, w = evaluate(rows, glass)
        out["name"] = name
        if w is not None:                       # which features signal a wrong answer (largest weights)
            F_ = model.m.config.n_features
            w = np.asarray(w)
            L = model.m.config.n_layer - 1
            labels = feature_labels(model.m, tok, torch.load(f"{DATA}/stream.pt"), L)
            last = w[L * F_:(L + 1) * F_]
            order = np.argsort(last)
            out["features_signalling_wrong"] = [{"feature": int(i), "weight": round(float(last[i]), 3),
                                                 "fires_on": labels.get(int(i), "(rare)")} for i in order[:8]]
            out["features_signalling_right"] = [{"feature": int(i), "weight": round(float(last[i]), 3),
                                                 "fires_on": labels.get(int(i), "(rare)")} for i in order[::-1][:8]]
        res[key] = out
        print(json.dumps({key: {k: v for k, v in out.items() if not k.startswith("features_")}}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
