"""Does sampling agreement add to the claim score? For each question: one greedy reply (its claim score,
as served), then K sampled replies; agreement = share of samples that contain the greedy reply's first
claim. AUROC for wrong-vs-right of the claim score, of disagreement, and of their rank mean; and the
cost if sampling runs only for replies in the claim score's uncertain band.

    python experiments/consistency.py --model Qwen/Qwen2.5-0.5B-Instruct --head <claims head> --n 150 --k 4
"""
import argparse
import json
import random
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from vt.correctness import auroc, auroc_ci, is_right, norm  # noqa: E402
from vt.verichat import load_head  # noqa: E402
from vt.guided import DoubtAwareGenerator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--head", default="results/kaggle/qwen-correctness/correctness/Qwen2.5-0.5B-Instruct_heads.pt")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--out", default="results/consistency.json")
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    probe, layer = load_head(a.head)
    g = DoubtAwareGenerator(m, tok, probe, layer, threshold=0.0)
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    qs = [(x["question"], x["answer"]["normalized_aliases"] + [x["answer"]["value"]])
          for x in d.select(random.Random(23).sample(range(len(d)), a.n))]
    rows = []
    for i in range(0, len(qs), 8):
        chunk = qs[i:i + 8]
        base = g.generate_batch([{"messages": [{"role": "user", "content": q}], "max_new_tokens": a.tokens, "mode": "flag"}
                                 for q, _ in chunk])
        samples = [g.generate_batch([{"messages": [{"role": "user", "content": q}], "max_new_tokens": a.tokens,
                                      "mode": "flag", "sampling": {"temperature": 1.0, "top_p": 0.95, "seed": 100 * s + i + j}}
                                     for j, (q, _) in enumerate(chunk)]) for s in range(a.k)]
        for j, ((q, al), r) in enumerate(zip(chunk, base)):
            key = norm(r["claims"][0]["text"]) if r["claims"] else norm(r["text"])[:20]
            agree = float(np.mean([bool(key) and key in norm(samples[s][j]["text"]) for s in range(a.k)]))
            rows.append({"wrong": 1 - is_right(r["text"], al), "claim": r["min_p_correct"] or 0.0, "agree": agree})
        print(len(rows), flush=True)
    y = np.array([r["wrong"] for r in rows])
    c = -np.array([r["claim"] for r in rows])
    dis = -np.array([r["agree"] for r in rows])

    def rk(v):
        return np.argsort(np.argsort(v + 1e-9 * np.arange(len(v)))) / (len(v) - 1)
    res = {"model": a.model, "n": len(y), "wrong": int(y.sum()), "k": a.k,
           "claim_auroc": auroc(c, y), "claim_ci": auroc_ci(c, y),
           "disagreement_auroc": auroc(dis, y), "disagreement_ci": auroc_ci(dis, y),
           "rank_mean_auroc": auroc((rk(c) + rk(dis)) / 2, y)}
    lo, hi = np.quantile(-c, [0.25, 0.75])                    # sample only for the middle half of claim scores
    band = (-c >= lo) & (-c <= hi)
    trig = np.where(band, (rk(c) + rk(dis)) / 2, rk(c))
    res["triggered"] = {"share_sampled": round(float(band.mean()), 3), "auroc": auroc(trig, y)}
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
