"""Copying heads as a grounding signal: does the reply's token come from where the reader's copy machinery points?

Prompted by "Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, Nov 2026), which found that a model's
copying (induction) heads can be read off its weights, and that a sycophant shows itself in where one head looks
rather than in its weights. Here the reader's copying heads are found once, without labels (the standard
induction score on a random token sequence repeated twice), and then read on every check:

    a copying head at the position before reply token t attends to the evidence token that followed an earlier
    occurrence of the current token; that evidence token is its candidate for t.
    match_t    = copying-head attention on evidence positions holding exactly token t     (copied from the evidence)
    conflict_t = copying-head attention on evidence positions holding a DIFFERENT token    (the evidence says
                 something else here: a swapped number or name)

Per answer: the strongest conflict, the mean match, the share of word tokens with no copy support, and a plain
lexical baseline (the share of the answer's words absent from the evidence), each as an AUROC for "hallucinated"
on the same HaluBench sample as experiments/halubench_compare.py (seed 1), next to the grounding check's score.

    python experiments/copy_heads.py --out results/copy_heads_qwen3_4b.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def induction_scores(model, n=64, batch=4, seed=0):
    """Per layer and head: mean attention from each token of the second copy of a random sequence to the token
    after its first occurrence (no labels, no task data)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    vocab = model.config.vocab_size
    r = torch.randint(1000, min(vocab, 100000), (batch, n), generator=g)
    ids = torch.cat([r, r], 1)
    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True)
    q = torch.arange(n + 1, 2 * n)
    return np.stack([a[:, :, q, q - n + 1].float().mean((0, 2)).numpy() for a in out.attentions])   # layers x heads


def word_like(s):
    return bool(re.search(r"[A-Za-z0-9]", s))


def lexical_new(answer, evidence):
    ev = set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", evidence.lower()))
    ws = [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", answer.lower()) if len(w) > 2 or w.isdigit()]
    return float(np.mean([w not in ev for w in ws])) if ws else 0.0


def features(model, tok, heads, question, passage, answer, budget):
    import torch
    from evidence_shift import prefix
    from vt.grounding import select
    evidence = "\n\n".join(select([passage], f"{question} {answer}", budget))
    pre = prefix(tok, f"{question}\n\n{evidence}")
    rep = tok(str(answer), add_special_tokens=False).input_ids
    if not rep:
        return None
    ids = pre + rep
    P, R = len(pre), len(rep)
    qpos = list(range(P - 1, P + R - 1))                      # the positions that predict each reply token
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    got, hooks = {}, []
    for l, hs in by_layer.items():
        def hook(mod, inp, out, l=l, hs=hs):
            got[l] = out[1][0, hs][:, qpos, 1:P].float()      # heads x reply x evidence (position 0: the sink)
        hooks.append(model.model.layers[l].self_attn.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
    finally:
        for hk in hooks:
            hk.remove()
    att = torch.cat([got[l] for l in by_layer]).mean(0).numpy()   # reply x evidence, averaged over copying heads
    keys = np.array(pre[1:P])
    same = keys[None, :] == np.array(rep)[:, None]
    total, match = att.sum(1), (att * same).sum(1)
    conflict = total - match
    wl = np.array([word_like(tok.decode([t])) for t in rep])
    in_ev = same.any(1)
    w = wl & (np.arange(R) > 0)                                # word tokens after the first
    return {"conflict_max": float(conflict[w].max()) if w.any() else 0.0,
            "conflict_mean": float(conflict[w].mean()) if w.any() else 0.0,
            "match_mean": float(match[w].mean()) if w.any() else 0.0,
            # present in the evidence yet not copied from it, or not present at all
            "uncopied": float((match[w] < 0.05).mean()) if w.any() else 0.0,
            "conflict_on_present": float(conflict[w & in_ev].max()) if (w & in_ev).any() else 0.0,
            "lexical_new": lexical_new(str(answer), evidence), "tokens": P + R}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3-4B")
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16, help="how many of the strongest copying heads to read")
    ap.add_argument("--budget", type=int, default=8000, help="evidence characters (passage selection beyond)")
    ap.add_argument("--baseline", default="results/halubench_compare_qwen3_4b_seed1.json",
                    help="the grounding check's scores on the same sample")
    ap.add_argument("--out", default="results/copy_heads_qwen3_4b.json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halubench_compare import sample
    from vt.correctness import auroc
    tok = AutoTokenizer.from_pretrained(a.reader)
    model = AutoModelForCausalLM.from_pretrained(a.reader, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").eval()
    cache = a.out + ".jsonl"
    done = [json.loads(l) for l in open(cache)] if os.path.exists(cache) else []
    if os.path.exists(a.out + ".heads.json"):
        head_info = json.load(open(a.out + ".heads.json"))
    else:
        sc = induction_scores(model)
        order = np.dstack(np.unravel_index(np.argsort(-sc, None), sc.shape))[0]
        head_info = {"scores": sc.round(3).tolist(),
                     "top": [[int(l), int(h), round(float(sc[l, h]), 3)] for l, h in order[:40]]}
        json.dump(head_info, open(a.out + ".heads.json", "w"))
    heads = [(l, h) for l, h, _ in head_info["top"][:a.heads]]
    print("copying heads (layer, head, score):", head_info["top"][:a.heads], flush=True)
    rows = sample(a.per_source, a.seed)
    if a.limit:
        rows = rows[:a.limit]
    t0, start = time.time(), len(done)
    with open(cache, "a") as f:
        for i, r in enumerate(rows[len(done):], len(done)):
            ft = features(model, tok, heads, r["question"], r["passage"], r["answer"], a.budget) or {}
            rec = dict(ft, i=i, source=r["source_ds"], y=int(r["label"] == "FAIL"))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done.append(rec)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(rows)} {(time.time() - t0) / (i + 1 - start):.1f}s an answer", flush=True)
    out = done[:len(rows)]
    y = np.array([o["y"] for o in out], float)
    srcs = np.array([o["source"] for o in out])
    names = ["conflict_max", "conflict_mean", "conflict_on_present", "uncopied", "match_mean", "lexical_new"]
    sign = {"match_mean": -1}
    cols = {n: sign.get(n, 1) * np.array([o.get(n, 0.0) for o in out]) for n in names}
    if a.baseline and os.path.exists(a.baseline):
        base = json.load(open(a.baseline))["rows"]
        assert [b["source"] for b in base[:len(out)]] == list(srcs), "not the same sample"
        for n in ("grounding_total", "minicheck"):
            cols[n] = np.array([b[n] for b in base[:len(out)]])

    def summary(s):
        return {"auroc": round(float(auroc(s, y)), 3),
                "per_source": {src: round(float(auroc(s[srcs == src], y[srcs == src])), 3)
                               for src in sorted(set(srcs)) if 0 < y[srcs == src].sum() < (srcs == src).sum()}}
    res = {"n": len(out), "reader": a.reader, "seed": a.seed, "copying_heads": head_info["top"][:a.heads],
           "features": {n: summary(s) for n, s in cols.items()}}
    if "grounding_total" in cols and min(y.sum(), len(y) - y.sum()) >= 10:   # with labels: does it add to the grounding check?
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)        # noqa: E731
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        for name, keys in (("grounding_only", ["grounding_total"]),
                           ("grounding+copy", ["grounding_total", "conflict_max", "uncopied", "lexical_new"]),
                           ("copy_only", ["conflict_max", "uncopied", "lexical_new"])):
            X = np.stack([z(cols[k]) for k in keys], 1)
            p = cross_val_predict(LogisticRegression(C=1.0), X, y, cv=cv, method="predict_proba")[:, 1]
            res["features"]["cv5_" + name] = summary(p)
    print(json.dumps(res["features"], indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
