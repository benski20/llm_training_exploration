"""Score the minimal-pair nuance set (results/omni/nuance_pairs_dataset.jsonl) with the CPU tier of the omni verifier:
Qwen3-1.7B reader, the shipped head, a zero-label profile per domain built from that domain's stored corpus features
(labels unused), the reader's own judge, and exact arithmetic flags. Each pair is an original correct reply and a twin
with one subtle edit (number, qualifier, negation, reasoning step, final answer).

    python experiments/nuance_eval_local.py --feats /data/omni/q17 --out results/omni/nuance_pairs_q17_cpu.json

Reports pairwise accuracy (the twin scored above its original), AUROC, by type / format / edit position, and
whether the reader's worst unit is the edited sentence (top-1 / top-3).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omni_corpus import load_features  # noqa: E402
from omni_eval import auroc  # noqa: E402

from vt.omni import OmniVerifier, feature_dicts, subset  # noqa: E402


def item(messages, response, units, doc):
    return {"messages": messages, "response": response, "units": [tuple(u) for u in units], "doc": doc or "",
            "question": messages[-1]["content"] if messages else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/omni/nuance_pairs_dataset.jsonl")
    ap.add_argument("--head", default="heads/omni_qwen3_1p7b.npz")
    ap.add_argument("--feats", help="stored corpus features of this reader for the domain profiles; without it, "
                    "--profile-n corpus replies per domain are read live (labels unused)")
    ap.add_argument("--profile-n", type=int, default=100)
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs)]
    v = OmniVerifier.load(a.head).attach_reader(threads=a.threads)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.corpus)}
    profiles = {}
    if a.feats:
        F = load_features(a.feats)
        dom = np.array([rows[int(i)]["domain"] for i in F["ids"]])
    used = {p["orig_id"] for p in pairs}
    for d in sorted({p["domain"] for p in pairs}):
        if a.feats:
            fd = feature_dicts(subset(F, dom == d))
            for f in fd:                                    # stored features carry no flags / NLI / judges
                f.setdefault("flags", 0)
        else:                                               # read live: replies of the domain not used in pairs
            pool = [r for r in rows.values() if r["domain"] == d and r["id"] not in used]
            np.random.default_rng(0).shuffle(pool)
            fd = [v.features(item(r["messages"], r["response"], r["units"], r["doc"])) for r in pool[:a.profile_n]]
            fd = [f for f in fd if f is not None]
            print(f"profile {d}: {len(fd)} replies read", flush=True)
        profiles[d] = v.profile(fd, d)                      # labels unused
    out, t0 = [], time.time()
    for k, p in enumerate(pairs):
        rec = {"pid": p["pid"], "type": p["type"], "format": p["format"], "domain": p["domain"], "third": p["third"],
               "orig_len": p["orig_len"]}
        prof = profiles[p["domain"]]
        for side, msgs, resp, units in (("orig", p["messages"], p["original"], p["original_units"]),
                                        ("twin", p.get("corrupted_messages") or p["messages"], p["corrupted"],
                                         p["corrupted_units"])):
            f = v.features(item(msgs, resp, units, p["doc"]))
            rec[side] = float(v.score(prof, [f])[0])
            if side == "twin":
                U = v.unit_matrix(f)
                us = ((U - np.array(prof["umu"])) / np.array(prof["usd"])) @ v.art["uw"] + float(v.art["ub"])
                idx = [u["index"] for u in f["units"]]
                order = [idx[j] for j in np.argsort(-us)]
                rec["edit_rank"] = order.index(p["edit_unit"]) + 1 if p["edit_unit"] in order else None
                rec["n_units"] = len(order)
        out.append(rec)
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(pairs)} ({(time.time() - t0) / (k + 1):.1f} s a pair)", flush=True)

    def summ(sel):
        o = np.array([r["orig"] for r in sel])
        t = np.array([r["twin"] for r in sel])
        ranks = [r["edit_rank"] for r in sel if r.get("edit_rank")]
        nu = [r["n_units"] for r in sel if r.get("edit_rank")]
        return {"n": len(sel), "pairwise_acc": float(np.mean(t > o)),
                "auroc": auroc(np.concatenate([o, t]), np.r_[np.zeros(len(o)), np.ones(len(t))]),
                "edit_top1": float(np.mean([x == 1 for x in ranks])) if ranks else None,
                "edit_top3": float(np.mean([x <= 3 for x in ranks])) if ranks else None,
                "chance_top1": float(np.mean([1 / n for n in nu])) if nu else None}
    rep = {"tier": "Qwen3-1.7B reader + shipped head + reader judge + arithmetic, zero-label domain profiles",
           "caveat": "the head was trained on the full corpus, so originals are training replies; twins are not",
           "all": summ(out)}
    for key in ("type", "format", "third"):
        rep["by_" + key] = {g: summ([r for r in out if r[key] == g]) for g in sorted({r[key] for r in out})}
    rep["pairs"] = out
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v_ for k, v_ in rep.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
