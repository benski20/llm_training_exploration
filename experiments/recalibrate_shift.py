"""Does a small labelled sample fix the correctness head's calibration on a new domain?

The Qwen2.5-0.5B head (trained on TriviaQA, kaggle/correctness) answers NQ-open questions it has
never seen. Its raw stacker logits are recorded with the true labels; the scale is refit
(`CorrectnessHead.calibrate`) on the first --n-fit questions and scored on the rest: ECE, and what
the 0.5 answer threshold means before and after (share of answers released, accuracy of those).

    python experiments/recalibrate_shift.py --n 600 --n-fit 200
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.correctness import CorrectnessHead, auroc, ece, is_right  # noqa: E402


def released(p, y, th=0.5):
    keep = p >= th
    return {"share_released": float(keep.mean()), "accuracy_released": float(y[keep].mean()) if keep.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--heads", default="results/kaggle/qwen-correctness/correctness/Qwen2.5-0.5B-Instruct_heads.pt")
    ap.add_argument("--variant", default="all")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--n-fit", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/recalibrate_shift.json")
    ap.add_argument("--partial", default="results/recalibrate_shift_rows.json", help="per-question rows, resumable")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager").eval()
    head = CorrectnessHead.from_state(torch.load(a.heads, weights_only=True)[a.variant])
    nq = load_dataset("google-research-datasets/nq_open", split="validation").select(range(a.n))
    rows = json.load(open(a.partial)) if os.path.exists(a.partial) else []      # resume after a restart
    t0 = time.time()
    for i, x in enumerate(nq):
        if i < len(rows):
            continue
        q = x["question"][0].upper() + x["question"][1:] + "?"
        r = head.answer(model, tok, q)
        rows.append({"q": q, "reply": r["reply"], "right": is_right(r["reply"], x["answer"]), "z": r["logit"]})
        if (i + 1) % 10 == 0:
            json.dump(rows, open(a.partial, "w"))
        if (i + 1) % 25 == 0:
            print(json.dumps({"done": i + 1, "acc": float(np.mean([r_["right"] for r_ in rows])),
                              "s": round(time.time() - t0)}), flush=True)
    z = np.array([r["z"] for r in rows])
    y = np.array([r["right"] for r in rows], dtype=float)
    fz, fy, tz, ty = z[:a.n_fit], y[:a.n_fit], z[a.n_fit:], y[a.n_fit:]
    p0 = 1 / (1 + np.exp(-tz))
    head.calibrate(fz, fy)
    ca, cb = head.calib
    p1 = 1 / (1 + np.exp(-(ca * tz + cb)))
    out = {"model": a.model, "variant": a.variant, "n": len(rows), "n_fit": a.n_fit, "accuracy": float(y.mean()),
           "calib": [ca, cb], "test_auroc": auroc(tz, ty),
           "before": {"ece": ece(p0, ty), "mean_p": float(p0.mean()), **released(p0, ty)},
           "after": {"ece": ece(p1, ty), "mean_p": float(p1.mean()), **released(p1, ty)},
           "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
