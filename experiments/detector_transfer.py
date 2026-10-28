"""Zero-shot transfer of a trained detector to domains it never saw: HaluBench's DROP (numerical
reasoning), PubMedQA, FinanceBench and CovidQA (the test split vt.detector uses, so a later run that
trains on these is scored on the same items).

    python experiments/detector_transfer.py --head results/kaggle/detector-3b/detector/Qwen2.5-3B-Instruct_detector.pt \
        --n 50 --partial <rows.json>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.correctness import auroc, auroc_ci  # noqa: E402
from vt.detector import HALUBENCH, Detector, load_halubench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=50, help="test items per domain")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--partial", required=True)
    ap.add_argument("--out", default="results/detector_transfer.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    head = torch.load(a.head, map_location="cpu", weights_only=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(head["model"])
    model = AutoModelForCausalLM.from_pretrained(head["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    det = Detector(model, tok, head)
    items = load_halubench("test", {s: a.n for s in HALUBENCH.values()})
    rows = json.load(open(a.partial)) if os.path.exists(a.partial) else []
    t0 = time.time()
    for i, it in enumerate(items):
        if i < len(rows):
            continue
        try:
            z = det.judge(it["messages"], it["response"], max_tokens=a.max_tokens)["logit"]   # its system prompt kept
        except ValueError:                                   # prompt too long for the cap
            z = None
        rows.append({"source": it["source"], "label": it["label"], "z": z})
        if (i + 1) % 10 == 0:
            json.dump(rows, open(a.partial, "w"))
            print(json.dumps({"done": i + 1, "of": len(items), "s": round(time.time() - t0)}), flush=True)
    json.dump(rows, open(a.partial, "w"))
    res = {"head": a.head, "model": head["model"], "threshold": det.threshold, "per_source": {}}
    for s in list(HALUBENCH.values()) + ["all"]:
        rs = [r for r in rows if r["z"] is not None and (s == "all" or r["source"] == s)]
        if len(rs) < 10:
            continue
        z, y = [r["z"] for r in rs], [r["label"] for r in rs]
        res["per_source"][s] = {"n": len(rs), "positives": int(sum(y)), "auroc": auroc(z, y), "auroc_ci": auroc_ci(z, y),
                                "accuracy": float(np.mean((np.array(z) >= det.threshold) == np.array(y))),
                                "accuracy_best_threshold": max(float(np.mean((np.array(z) >= t) == np.array(y))) for t in z)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
