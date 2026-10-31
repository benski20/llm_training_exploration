"""The ledger's judge (vt/ledger.py) offline, on a small model as a lower bound for the served one:

    kind        results/claim_type_holdout.json: 120 reply sentences labeled by hand, no evidence but a
                generic conversation; facts should not come back D, everything else should
    arithmetic  right and wrong calculations, told apart only by the exact re-computation in the evidence
    text        statements about a pasted article, true and false, the article in the conversation

    python experiments/judge_eval.py --model Qwen/Qwen3-1.7B --out results/judge_eval_qwen3_1.7b.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="judge_eval.json")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.ledger import Ledger
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if torch.cuda.is_available()
                                                 else torch.float32).eval()
    led = Ledger(model, tok)
    items = json.load(open("results/judge_eval_items.json"))
    kinds = json.load(open("results/claim_type_holdout.json"))
    t = time.time()
    res = {"model": a.model}

    def run(rows, messages):
        built = led.build_batch([{"text": s, "messages": messages} for s, _ in rows])
        return [(want, (b[0]["verdict"] if b else None), s) for (s, want), b in zip(rows, built)]

    kind = run([(x["s"], "fact" if x["fact"] else "not") for x in kinds],
               [{"role": "user", "content": "Can you help me with something?"}])
    res["kind"] = {"facts_not_D": sum(g != "not a factual claim" for w, g, _ in kind if w == "fact"),
                   "others_D": sum(g == "not a factual claim" for w, g, _ in kind if w == "not"),
                   "n_each": 60, "verdicts": {w: dict(Counter(g for ww, g, _ in kind if ww == w)) for w in ("fact", "not")},
                   "wrong": [(w, g, s) for w, g, s in kind if (w == "fact") == (g == "not a factual claim")]}
    for name, rows, msgs in (("arithmetic", items["arithmetic"], [{"role": "user", "content": "Can you work this out?"}]),
                             ("text", items["article_claims"],
                              [{"role": "user", "content": "Summarize this article.\n\n" + items["article"]}])):
        out = run(rows, msgs)
        res[name] = {"n": len(out), "correct": sum(w == g for w, g, _ in out),
                     "confusion": dict(Counter(f"{w} -> {g}" for w, g, _ in out if w != g)),
                     "wrong": [(w, g, s) for w, g, s in out if w != g]}
    res["seconds"] = round(time.time() - t, 1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
