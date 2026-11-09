"""A third, different-family one-pass judge (IBM granite-3.3-2b-instruct, CPU) on the closed-book replies, added to the
8B stack + Qwen3-14B / phi-4 judges (results/omni/expert_scores_q8b.npz, judge_probs.npz). Lead from
docs/omni_experiments/novel_mech_bundle.md: granite's judge beat Qwen-1.7B's on chat (0.64 vs 0.48).

    python experiments/granite_judge.py --out results/omni/granite_judge.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vt.omni import JudgeScorer  # noqa: E402

DOMAINS = ("truthfulqa", "halueval_general", "felm_wk", "felm_science")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--model", default="ibm-granite/granite-3.3-2b-instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    rows = [json.loads(l) for l in open(a.corpus)]
    rows = [r for r in rows if r["domain"] in DOMAINS]
    j = JudgeScorer(a.model, load_in_4bit=False, max_tokens=2048)
    ids, sc, t0 = [], [], time.time()
    for k, r in enumerate(rows):
        it = {"response": r["response"], "doc": r["doc"], "question": r["messages"][-1]["content"]}
        sc.append(j(it, reasoning=False))
        ids.append(r["id"])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(rows)} ({(time.time() - t0) / (k + 1):.2f} s each)", flush=True)
            np.savez(a.out, ids=np.array(ids), logodds_wrong=np.array(sc))
    np.savez(a.out, ids=np.array(ids), logodds_wrong=np.array(sc))
    print("done", len(ids))


if __name__ == "__main__":
    main()
