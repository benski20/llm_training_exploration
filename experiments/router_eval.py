"""How well does vt.router sort messages? Offline: no server, no lookups, one forward pass per message the rules
leave open. Reports accuracy overall, per expected kind, for the rules alone and for the model alone, and the
confusion, on any items file of {"q", "expect"} (results/mixed_eval_items.json, results/router_holdout.json).

    python experiments/router_eval.py --model Qwen/Qwen3-0.6B --items results/router_holdout.json --out r.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# misroutes that cost little: a programming or text question answered from a search, a calculation looked up
HARMLESS = {("code", "information"), ("math", "information"), ("text", "information")}


def summarize(rows):
    out = {"n": len(rows), "accuracy": round(sum(r["route"] == r["expect"] for r in rows) / max(1, len(rows)), 3)}
    ruled = [r for r in rows if r["rule"]]
    asked = [r for r in rows if not r["rule"]]
    out["rules"] = {"decided": len(ruled),
                    "accuracy": round(sum(r["rule"] == r["expect"] for r in ruled) / max(1, len(ruled)), 3)}
    out["model"] = {"decided": len(asked),
                    "accuracy": round(sum(r["route"] == r["expect"] for r in asked) / max(1, len(asked)), 3)}
    out["acceptable"] = round(sum(r["route"] == r["expect"] or (r["expect"], r["route"]) in HARMLESS for r in rows)
                              / max(1, len(rows)), 3)
    out["unchecked_information"] = sum(r["expect"] == "information" and r["route"] in ("chat", "code") for r in rows)
    out["per_kind"] = {k: {"n": sum(r["expect"] == k for r in rows),
                           "accuracy": round(sum(r["route"] == k for r in rows if r["expect"] == k)
                                             / max(1, sum(r["expect"] == k for r in rows)), 3)}
                       for k in sorted({r["expect"] for r in rows})}
    out["confusion"] = {f"{e} -> {g}": n for (e, g), n in sorted(Counter((r["expect"], r["route"]) for r in rows).items())
                        if e != g}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", nargs="+", default=["results/mixed_eval_items.json", "results/router_holdout.json"])
    ap.add_argument("--out", default="router_eval.json")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.router import Router, rule
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if not torch.cuda.is_available()
                                                 else torch.bfloat16, device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    router = Router(model, tok)
    res = {"model": a.model}
    for path in a.items:
        items = json.load(open(path))
        t = time.time()
        routed = router.route_batch([[{"role": "user", "content": x["q"]}] for x in items])
        rows = [{"q": x["q"][:200], "expect": x["expect"], "kind": x.get("kind"), "rule": rule(x["q"]),
                 "route": k, "probs": p} for x, (k, p) in zip(items, routed)]
        s = summarize(rows)
        s["seconds"] = round(time.time() - t, 1)
        print(path, json.dumps(s, indent=1), flush=True)
        res[os.path.basename(path)] = dict(s, rows=rows)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
