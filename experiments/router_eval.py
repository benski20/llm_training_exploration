"""How well does vt.router sort messages? Offline: no server, no lookups, one forward pass per message the rules
leave open. Reports accuracy overall, per expected kind, for the rules alone and for the model alone, and the
confusion, on any items file of {"q", "expect"} (results/mixed_eval_items.json, results/router_holdout.json).

    python experiments/router_eval.py --out results/router_eval_cost.json          # the shipped router
    python experiments/router_eval.py --router model --model Qwen/Qwen3-0.6B --out r.json
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
    ap.add_argument("--router", choices=["text", "model"], default="text",
                    help="text: the shipped embedding router (vt/router_head.npz); model: the chat model's choice")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", nargs="+", default=["results/mixed_eval_items.json", "results/router_holdout.json",
                                                   "no_robots_test"])
    ap.add_argument("--out", default="router_eval.json")
    a = ap.parse_args()
    from vt.router import Router, TextRouter, rule
    if a.router == "text":
        router = TextRouter()
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32 if not torch.cuda.is_available()
                                                     else torch.bfloat16,
                                                     device_map="auto" if torch.cuda.is_available() else None)
        router = Router(model.eval(), tok)
    res = {"router": a.router, "model": a.model if a.router == "model" else None}
    for path in a.items:
        if path == "no_robots_test":                        # human-labeled categories (CC-BY-NC: evaluation only)
            from experiments.train_router import no_robots_test
            items = no_robots_test()
        else:
            items = json.load(open(path))
        t = time.time()
        routed = router.route_batch([[{"role": "user", "content": x["q"]}] for x in items])
        rows = [{"q": x["q"][:200], "expect": x["expect"], "kind": x.get("kind"), "rule": rule(x["q"]),
                 "route": k, "probs": p} for x, (k, p) in zip(items, routed)]
        s = summarize(rows)
        s["ms_per_message"] = round(1000 * (time.time() - t) / max(1, len(rows)), 1)
        if any(r["kind"] == "Rewrite" for r in rows):      # rewrites of pasted text are checked against it
            s["accuracy_counting_rewrites_of_pasted_text"] = round(
                sum(r["route"] == r["expect"] or (r["kind"] == "Rewrite" and r["route"] == "text") for r in rows)
                / len(rows), 3)
        print(path, json.dumps(s, indent=1), flush=True)
        res[os.path.basename(path)] = dict(s, rows=rows)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
