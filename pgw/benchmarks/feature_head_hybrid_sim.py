"""Tokens per pass on held-out greedy answers: the draft head alone, prompt lookup alone, and both (lookup when it fills the chain).

Replays ``feature_head_distill.py`` outputs (``ranks_<label>_<set>.json`` + ``eval_<set>.json``) through the strict
chain acceptance rule; lookup uses the SDK's settings (n-grams of 4 down to 2 tokens).

    PYTHONPATH=src python benchmarks/feature_head_hybrid_sim.py <out_dir> <label>   # e.g. d256_v16384
"""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from pgw.engine.generate import _lookup_drafts  # noqa: E402

out = sys.argv[1]
label = sys.argv[2]
for kind in ("chat", "code"):
    rk = json.load(open(f"{out}/ranks_{label}_{kind}.json"))
    ev = json.load(open(f"{out}/eval_{kind}.json"))
    res = {}
    for R in (2, 3, 4):
        k = R - 1
        tot = {"head": [0, 0], "lookup": [0, 0], "hybrid": [0, 0]}
        for a_i, (p, a) in enumerate(zip(ev["prompts"], ev["answers"], strict=True)):
            ranks = rk["ranks2d"][a_i]
            L = len(a)
            for mode in tot:
                i = 1
                toks, passes = 1, 1
                while i < L:
                    # pass with pending a[i-1]: drafts predict a[i], a[i+1], ...
                    head_acc = 0
                    while head_acc < k and i + head_acc < L - 1 and head_acc < len(ranks[i]) and ranks[i][head_acc] == 0:
                        head_acc += 1
                    prop = _lookup_drafts(p + a[:i], k, 4, 2)
                    lk_acc = 0
                    while lk_acc < len(prop) and i + lk_acc < L - 1 and prop[lk_acc] == a[i + lk_acc]:
                        lk_acc += 1
                    acc = {"head": head_acc, "lookup": lk_acc,
                           "hybrid": lk_acc if len(prop) >= k else head_acc}[mode]
                    em = min(1 + acc, L - i)
                    toks += em
                    passes += 1
                    i += em
                tot[mode][0] += toks
                tot[mode][1] += passes
        res[R] = {m: round(t / p, 3) for m, (t, p) in tot.items()}
    print(kind, json.dumps(res))
