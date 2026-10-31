"""The grounding probe end to end (vt/grounding.py): the 13 whole replies of results/judge_eval_items.json (right
and wrong facts, a wrong fact in a poem, arithmetic, code, reassurance, summaries of a pasted article), each
read once by the model with its evidence in context: the user's text, exact re-computation of its arithmetic,
and a Wikipedia lookup when the user gave nothing to check against. Prints each sentence's verdict and the
reply's total ungroundedness, and how well the total separates the wrong replies from the rest.

    python experiments/grounding_demo.py --model Qwen/Qwen3-4B --probe heads/grounding_probe_qwen3_4b.npz
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--probe", default="heads/grounding_probe_qwen3_4b.npz")
    ap.add_argument("--out", default="grounding_demo.json")
    a = ap.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.correctness import auroc
    from vt.grounding import Grounding
    from vt.retrieval import Wikipedia
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    g = Grounding(model, tok, a.probe, retriever=Wikipedia(full_text=True, pages=3))
    rows = []
    for x in json.load(open("results/judge_eval_items.json"))["replies"]:
        msgs = [{"role": "user", "content": x["q"]}]
        pasted = "\n\n" in x["q"]                               # a pasted article is the evidence itself
        out = g.check(msgs, x["reply"], documents=[x["q"].split("\n\n", 1)[1]] if pasted else ())
        wrong = x["expect"] == "likely wrong"
        rows.append({"q": x["q"][:60], "wrong": wrong, "total": out["total"], "verdict": out["verdict"],
                     "sentences": [(s["verdict"], s["z"], s["text"][:70]) for s in out["sentences"]]})
        print(("WRONG " if wrong else "right ") + f"total {out['total']:6.2f}  {x['q'][:50]!r}", flush=True)
        for s in out["sentences"]:
            print(f"      {s['verdict']:13s} z={s['z']:+.2f}  {s['text'][:80]}", flush=True)
    y = np.array([r["wrong"] for r in rows], float)
    res = {"model": a.model, "n": len(rows), "wrong": int(y.sum()),
           "total_auroc_wrong": round(float(auroc(np.array([r["total"] for r in rows]), y)), 3), "rows": rows}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
