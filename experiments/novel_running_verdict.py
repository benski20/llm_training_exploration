"""Running verdict: where does the reader's own verdict tip to "wrong" as it reads a reply step by step?

The reader (Qwen3-1.7B, CPU) reads [task, reply] once with a KV cache; after each sentence / step it is asked "is
everything so far correct? It may be unfinished" (vt.omni.Reader.step_verdicts) and we read j_k = log P(No) - log P(Yes).
Nothing is deleted from the reply (deleting steps, experiments/novel_causal_units.py, left gappy texts that judges
called wrong for being incomplete).

Corpus domains with unit labels: ProcessBench (the first wrong step) and FELM (each wrong segment). Rules set before
running:
  L  localisation on wrong replies, top-1 vs chance (#wrong units / #units):
       first_cross  the first unit with j_k > 0 (the sign of the verdict: no tuned threshold); none -> the last unit
       max_jump     the unit with the largest rise j_k - j_{k-1} (j_0 = the first unit's own value)
       surprisal    the unit with the lowest mean token log-probability (a training-free baseline)
  R  reply level (mean per-domain AUROC): the final running verdict, the largest jump, and their per-domain z-sum.

    python experiments/novel_running_verdict.py --per-domain 8 --out results/omni/novel_running_verdict.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vt.omni import Reader  # noqa: E402

DOMAINS = ["pb_gsm8k", "pb_math", "pb_olympiad", "pb_omnimath", "felm_math", "felm_reasoning", "felm_science",
           "felm_wk", "felm_writing"]


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return float("nan")
    return float(((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None])).mean())


def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--reader", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rd = Reader(a.reader, threads=a.threads)
    rows = [json.loads(l) for l in open(a.corpus)]
    rng = random.Random(0)                                    # the same sample as novel_causal_units.py
    pick = []
    for d in DOMAINS:
        x = [r for r in rows if r["domain"] == d and len(r["response"]) <= a.max_chars and len(r["units"]) >= 2]
        w = [r for r in x if r["label"] == 1 and 1 in r["unit_labels"]]
        c = [r for r in x if r["label"] == 0]
        pick += rng.sample(w, min(a.per_domain, len(w))) + rng.sample(c, min(a.per_domain, len(c)))
    out, t0 = [], time.time()
    for k, r in enumerate(pick):
        it = {"messages": r["messages"], "response": r["response"], "units": [tuple(u) for u in r["units"]],
              "doc": r.get("doc") or "", "question": r["messages"][-1]["content"]}
        st = rd.step_verdicts(it)
        f = rd.features(it)
        surpr = {u["index"]: float(-u["sc"][0]) for u in f["units"]} if f else {}
        idx = [it["units"].index((u["start"], u["end"])) for u in st]
        out.append({"id": r["id"], "domain": r["domain"], "label": r["label"],
                    "unit_labels": [r["unit_labels"][i] for i in idx], "j": [u["j"] for u in st],
                    "surprisal": [surpr.get(i, 0.0) for i in idx]})
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(pick)} {(time.time() - t0) / (k + 1):.1f} s per reply", flush=True)
            json.dump(out, open(a.out + ".partial", "w"))

    def first_cross(j):
        return next((k for k, x in enumerate(j) if x > 0), len(j) - 1)

    def max_jump(j):
        return int(np.argmax(np.diff([j[0]] + list(j)) if len(j) > 1 else [0]))

    loc = {"first_cross": [], "max_jump": [], "surprisal": []}
    chance = []
    for o in out:
        lab = np.array(o["unit_labels"])
        if o["label"] != 1 or not (lab == 1).any():
            continue
        chance.append(float((lab == 1).mean()))
        loc["first_cross"].append(float(lab[first_cross(o["j"])] == 1))
        loc["max_jump"].append(float(lab[max_jump(o["j"])] == 1))
        loc["surprisal"].append(float(lab[int(np.argmax(o["surprisal"]))] == 1))
    res = {"n_replies": len(out), "n_wrong_localised": len(chance), "chance_top1": float(np.mean(chance)),
           "top1": {m: float(np.mean(v)) for m, v in loc.items()}, "per_domain": {}}
    # paired bootstrap of first_cross - chance over wrong replies
    fc, ch = np.array(loc["first_cross"]), np.array(chance)
    b = np.random.default_rng(0)
    diffs = [float((fc[i] - ch[i]).mean()) for i in (b.integers(0, len(fc), len(fc)) for _ in range(2000))]
    res["first_cross_minus_chance_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    for d in DOMAINS:
        sel = [o for o in out if o["domain"] == d]
        y = np.array([o["label"] for o in sel])
        if len(set(y)) < 2:
            continue
        fin = np.array([o["j"][-1] for o in sel])
        jump = np.array([max(np.diff([o["j"][0]] + o["j"])) for o in sel])
        res["per_domain"][d] = {"final_verdict": auroc(fin, y), "max_jump": auroc(jump, y),
                                "max_verdict": auroc([max(o["j"]) for o in sel], y),
                                "final+jump": auroc(z(fin) + z(jump), y)}
    res["mean"] = {k: float(np.nanmean([p[k] for p in res["per_domain"].values()]))
                   for k in ("final_verdict", "max_jump", "max_verdict", "final+jump")}
    json.dump({"summary": res, "replies": out}, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_domain"}, indent=1))


if __name__ == "__main__":
    main()
