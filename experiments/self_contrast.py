"""Self-contrast: turn the verifier's strong RELATIVE judgement into an absolute one, with each reply as its own baseline.

The minimal-pair test (results/omni/nuance_pairs_q17_cpu.json) found the verifier ranks a reply above its own one-edit
corruption 90% of the time, but separates subtly wrong from correct replies across different replies at only 0.64:
correct replies vary among themselves more than one edit moves the score. Here every reply gets K automatic
single-claim corruptions (a number, a qualifier, a negation, an intermediate result: the rules of
kaggle/omni/nuance_pairs.py), all scored by the same verifier. A correct reply should stand out as better than its own
corruptions; a wrong one less so. Signal for "wrong": score(original) − mean score(corruptions), alone and added to the
plain score, per domain (zero labels: per-domain z-scores).

    python experiments/self_contrast.py --feats /data/omni/q17 --out results/omni/self_contrast_q17.json
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omni_corpus import load_features  # noqa: E402
from omni_eval import auroc  # noqa: E402

from vt.omni import OmniVerifier, feature_dicts, subset  # noqa: E402

spec = importlib.util.spec_from_file_location("nuance_pairs", os.path.join(ROOT, "kaggle", "omni", "nuance_pairs.py"))
NP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NP)


def corruptions(r, k, rng):
    kinds = ["number", "qualifier", "negation"] + (["reasoning_intermediate"] if r["format"] == "reasoning" else [])
    cands = [(kind, c) for kind in kinds for c in NP.sites(kind, r, rng) if c[3] is not None]
    rng.shuffle(cands)
    out, seen = [], set()
    for kind, (ui, a, b, new) in cands:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        t2, units2, msgs2, _, _ = NP.apply_edit(r, ui, a, b, new)
        out.append({"kind": kind, "messages": msgs2, "response": t2, "units": units2})
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--head", default="heads/omni_qwen3_1p7b.npz")
    ap.add_argument("--domains", nargs="+", default=["felm_wk", "ragtruth_qa", "ragtruth_summary", "pb_math"])
    ap.add_argument("--per-domain", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    for line in open(a.corpus):
        r = json.loads(line)
        rows[r["id"]] = r
    v = OmniVerifier.load(os.path.join(ROOT, a.head)).attach_reader(threads=4)
    F = load_features(a.feats)
    dom = np.array([rows[int(i)]["domain"] for i in F["ids"]])
    rng = random.Random(0)
    res, t0, done = {}, time.time(), 0
    for d in a.domains:
        fd = feature_dicts(subset(F, dom == d))
        for f in fd:
            f.setdefault("flags", 0)
        prof = v.profile(fd, d)                                        # labels unused
        pool = [r for r in rows.values() if r["domain"] == d]
        pos = [r for r in pool if r["label"] == 1]
        neg = [r for r in pool if r["label"] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        pick = pos[:a.per_domain // 2] + neg[:a.per_domain // 2]
        recs = []
        for r in pick:
            it = lambda m, t, u: {"messages": m, "response": t, "units": [tuple(x) for x in u], "doc": r["doc"],
                                  "question": m[-1]["content"]}
            s0 = float(v.score(prof, [v.features(it(r["messages"], r["response"], r["units"]))])[0])
            cs = [float(v.score(prof, [v.features(it(c["messages"], c["response"], c["units"]))])[0])
                  for c in corruptions(r, a.k, rng)]
            recs.append({"id": r["id"], "label": r["label"], "orig": s0, "corr": cs})
            done += 1
            if done % 20 == 0:
                print(f"{done} replies ({(time.time() - t0) / done:.1f} s each)", flush=True)
        y = np.array([x["label"] for x in recs])
        s0 = np.array([x["orig"] for x in recs])
        has = np.array([len(x["corr"]) > 0 for x in recs])
        gap = np.array([x["orig"] - np.mean(x["corr"]) if x["corr"] else np.nan for x in recs])
        z = lambda w: (w - np.nanmean(w)) / (np.nanstd(w) + 1e-9)
        comb = z(s0) + np.where(np.isnan(gap), 0, z(gap))
        res[d] = {"n": len(recs), "with_corruptions": float(has.mean()), "plain": auroc(s0, y),
                  "self_contrast": auroc(np.nan_to_num(gap, nan=np.nanmean(gap)), y), "plain_plus_contrast": auroc(comb, y),
                  "plain_on_corruptible": auroc(s0[has], y[has]), "contrast_on_corruptible": auroc(gap[has], y[has]),
                  "mean_gap_correct": float(np.nanmean(gap[y == 0])), "mean_gap_wrong": float(np.nanmean(gap[y == 1])),
                  "records": recs}
        print(d, {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in res[d].items() if k_ != "records"},
              flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
