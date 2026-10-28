"""Kaggle kernel: a claim-level correctness probe for any reply format (vt/claims.py).

For each model: build conversations in five formats with automatically checkable claims (short
answer, several answers in one reply, a question after earlier turns, reasoning then 'Answer:', GSM8K
math), generate the model's own replies, label every claim, and read one forward pass per reply.
A single probe on the hidden state at each claim token is trained across formats and evaluated:

  per format, on TriviaQA validation / GSM8K test (in distribution) and NQ-open (shift),
  with 95% bootstrap intervals, against the claim token's probability and the word's least likely token;
  leave-one-format-out (trained without a format, tested on it);
  a training-size curve; a stacker (probe + output signals) with calibration; response-level risk.

Single pass, no sampling: the serving cost is one forward pass over the reply.

    python run.py --out /kaggle/working/claims --models Qwen/Qwen2.5-3B-Instruct
    python run.py --smoke --out /tmp/cl
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vtpkg"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch  # noqa: E402

from vt.claims import FORMATS, OUTPUT_KEYS, claim_features, label_claims, make_items, prompt_of  # noqa: E402
from vt.correctness import Probe, auroc, generate, report  # noqa: E402

MAX_NEW = {"short": 32, "multiturn": 32, "multi": 96, "reasoning": 320, "math": 400}


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def pools(a):
    if a.smoke:
        qa = [{"q": q, "aliases": al} for q, al in [
            ("What is the capital of France?", ["Paris"]), ("Who wrote Hamlet?", ["Shakespeare"]),
            ("What is the largest planet?", ["Jupiter"]), ("Who painted the Mona Lisa?", ["Leonardo da Vinci"]),
            ("What is the chemical symbol for gold?", ["Au"]), ("In which country is Kyoto?", ["Japan"]),
            ("Who discovered penicillin?", ["Alexander Fleming", "Fleming"]), ("What is the capital of Peru?", ["Lima"])]]
        gm = [{"q": "Tom has 3 apples and buys 4 more. How many apples does he have?", "answer": "7"},
              {"q": "A box holds 6 eggs. How many eggs are in 5 boxes?", "answer": "30"}]
        return {"train": qa, "test": qa, "nq": qa}, {"train": gm, "test": gm}
    from datasets import load_dataset

    def tq(split, n, sample):
        d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
        idx = random.Random(0).sample(range(len(d)), n) if sample else range(n)
        return [{"q": x["question"], "aliases": x["answer"]["normalized_aliases"] + [x["answer"]["value"]]}
                for x in d.select(idx)]
    nq = load_dataset("google-research-datasets/nq_open", split="validation").select(range(a.n_pool_test))
    nqp = [{"q": x["question"][0].upper() + x["question"][1:] + "?", "aliases": x["answer"]} for x in nq]

    def gsm(split):
        return [{"q": x["question"], "answer": x["answer"].split("####")[-1].strip().replace(",", "")}
                for x in load_dataset("openai/gsm8k", "main", split=split)]
    return ({"train": tq("train", a.n_pool_train, True), "test": tq("validation", a.n_pool_test, False), "nq": nqp},
            {"train": gsm("train"), "test": gsm("test")})


def build_sets(a, qa, gsm):
    """{split: [items]}; test and nq use disjoint question pools from train."""
    n_train = dict(zip(FORMATS, a.n_train))
    sets = {"train": [], "test": [], "nq": []}
    for k, fmt in enumerate(FORMATS):
        tr_pool = gsm["train"] if fmt == "math" else qa["train"]
        sets["train"] += make_items(fmt, tr_pool, n_train[fmt], seed=k)
        sets["test"] += make_items(fmt, gsm["test"] if fmt == "math" else qa["test"], a.n_test, seed=100 + k)
        if fmt != "math":
            sets["nq"] += make_items(fmt, qa["nq"], a.n_test, seed=200 + k)
    return sets


def adapt(messages, no_system):
    """Reasoning models (DeepSeek-R1 distills) are meant to run without a system prompt: fold it into
    the first user turn."""
    if not no_system or messages[0]["role"] != "system":
        return messages
    sys_, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    rest[0]["content"] = sys_ + "\n\n" + rest[0]["content"]
    return rest


def run_split(model, tok, items, layers, a):
    """Generate replies, label claims, extract claim features. Returns per-claim rows."""
    rows = []
    for fmt in FORMATS:
        its = [it for it in items if it["format"] == fmt]
        if not its:
            continue
        prompts = [prompt_of(tok, adapt(it["messages"], a.no_system)) for it in its]
        reps = generate(model, tok, prompts, max_new_tokens=int(MAX_NEW[fmt] * a.max_new_scale), batch=a.gen_batch)
        labs = [label_claims(tok, it, r) for it, r in zip(its, reps)]
        feats = claim_features(model, tok, [tok(p).input_ids for p in prompts], reps,
                               [[j for j, _ in lab] for lab in labs], layers, batch=a.feat_batch)
        for n, (it, r, lab, fs) in enumerate(zip(its, reps, labs, feats)):
            for c, ((j, y), f) in enumerate(zip(lab, fs)):
                rows.append({"format": fmt, "item": n, "claim": c, "label": y, **f,
                             "reply": tok.decode(r, skip_special_tokens=True)[:300] if c == 0 and n < 3 else None})
        log(phase="split_format", format=fmt, items=len(its), claims=sum(len(x) for x in labs),
            accuracy=float(np.mean([y for lab in labs for _, y in lab])) if any(labs) else None,
            coverage=float(np.mean([bool(lab) for lab in labs])))
    return rows


def X(rows, li=None):
    h = torch.stack([r["h"] for r in rows]).float()
    return (h[:, li] if li is not None else h.flatten(1)).numpy()


def Y(rows):
    return np.array([r["label"] for r in rows], dtype=np.float32)


def evaluate(train, tests, layers, device, a):
    R = {}
    idx = np.random.default_rng(0).permutation(len(train))
    fit = [train[i] for i in idx[: int(0.75 * len(idx))]]
    hold = [train[i] for i in idx[int(0.75 * len(idx)):]]
    by_layer = {}
    for li, L in enumerate(layers):
        by_layer[L] = auroc(Probe(device).fit(X(fit, li), Y(fit)).logit(X(hold, li)), Y(hold))
    li = max(range(len(layers)), key=lambda i: by_layer[layers[i]] or 0)
    R["probe_auroc_by_layer_on_holdout"], R["probe_layer"] = by_layer, layers[li]
    probe_fit = Probe(device).fit(X(fit, li), Y(fit))          # stacker is fitted on hold with these scores
    probe = Probe(device).fit(X(train, li), Y(train))

    def feats(rows, pr):
        return np.column_stack([pr.logit(X(rows, li))] + [[r[k] for r in rows] for k in OUTPUT_KEYS])
    stacker = Probe(device).fit(feats(hold, probe_fit), Y(hold), grid=(1e-4, 1e-3, 1e-2, 1e-1))
    R["per_format"] = {}
    for split, rows in tests.items():
        for fmt in FORMATS:
            rs = [r for r in rows if r["format"] == fmt]
            if len(rs) < 20 or len(set(Y(rs))) < 2:
                continue
            y = Y(rs)
            z = stacker.logit(feats(rs, probe))
            d = {"accuracy": float(y.mean()),
                 "probe": report(probe.logit(X(rs, li)), y, ci=True),
                 "stack": report(z, y, prob=1 / (1 + np.exp(-z)), ci=True)}
            for k in OUTPUT_KEYS:
                d[k] = report([r[k] for r in rs], y, ci=k == "claim_p")
            R["per_format"][f"{split}/{fmt}"] = d
        y = Y(rows)
        z = stacker.logit(feats(rows, probe))
        R["per_format"][f"{split}/all"] = {"accuracy": float(y.mean()), "probe": report(probe.logit(X(rows, li)), y, ci=True),
                                          "stack": report(z, y, prob=1 / (1 + np.exp(-z)), ci=True),
                                          "claim_p": report([r["claim_p"] for r in rows], y, ci=True)}
    R["leave_one_format_out"] = {}                            # trained without a format, tested on it
    for fmt in FORMATS:
        tr = [r for r in train if r["format"] != fmt]
        te = [r for r in tests["test"] if r["format"] == fmt]
        if len(te) < 20 or len(set(Y(te))) < 2:
            continue
        p = Probe(device).fit(X(tr, li), Y(tr))
        R["leave_one_format_out"][fmt] = {"held_out": report(p.logit(X(te, li)), Y(te), ci=True),
                                          "with_format": R["per_format"].get(f"test/{fmt}", {}).get("probe")}
    R["probe_by_train_size"] = {}
    for n in (1000, 3000, len(train)):
        if n <= len(train):
            sub = [train[i] for i in idx[:n]]
            p = Probe(device).fit(X(sub, li), Y(sub))
            R["probe_by_train_size"][n] = {s: report(p.logit(X(rows, li)), Y(rows)) for s, rows in tests.items()}
    multi = [r for r in tests["test"] if r["format"] == "multi"]  # reply-level: all claims right vs min P
    if multi:
        by = {}
        for r, z in zip(multi, stacker.logit(feats(multi, probe))):
            by.setdefault(r["item"], []).append((r["label"], z))
        ys = [min(l for l, _ in v) for v in by.values()]
        zs = [min(z for _, z in v) for v in by.values()]
        if len(set(ys)) > 1:
            R["multi_reply_level"] = report(zs, ys, ci=True)
    head = {"layer": layers[li], "probe": probe.state_dict(), "stacker": stacker.state_dict(), "output_keys": OUTPUT_KEYS}
    return R, head


def run_model(name, sets, a, device, checkpoint):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, a.dtype), attn_implementation=a.attn,
                                                 device_map="auto" if device == "cuda" else None).eval()
    if device != "cuda":
        model.to(device)
    L = model.config.num_hidden_layers
    layers = sorted({int(x) for x in np.linspace(L // 3, L - 1, 8).round()})
    R = {"model": name, "layers": layers, "dtype": a.dtype, "complete": False}
    rows = {}
    for split in ("test", "nq", "train"):                    # test sets first: they are the result
        rows[split] = run_split(model, tok, sets[split], layers, a)
        R.setdefault("claims", {})[split] = {fmt: sum(r["format"] == fmt for r in rows[split]) for fmt in FORMATS}
        R.setdefault("examples", {})[split] = [{"format": r["format"], "label": r["label"], "reply": r["reply"]}
                                                for r in rows[split] if r["reply"]]
        log(model=name, phase="split", split=split, claims=len(rows[split]), s=round(time.time() - t0))
        if any(torch.isnan(r["h"].float()).any() for r in rows[split][:100]):
            raise RuntimeError("NaN hidden states")
        checkpoint(R)
    ev, head = evaluate(rows["train"], {"test": rows["test"], "nq": rows["nq"]}, layers, device, a)
    R.update(ev)
    R["seconds"] = round(time.time() - t0)
    R["complete"] = True
    torch.save(head, os.path.join(a.out, name.split("/")[-1] + "_claims_head.pt"))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/claims")
    ap.add_argument("--models", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-train", type=int, nargs=5, default=[2500, 800, 1500, 1200, 1200],
                    help="train items per format: short multi multiturn reasoning math")
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-pool-train", type=int, default=20000)
    ap.add_argument("--n-pool-test", type=int, default=1000)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--feat-batch", type=int, default=4)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--attn", default="sdpa", help="eager attention overflows in float16 on Qwen2.5")
    ap.add_argument("--no-system", action="store_true", help="fold the system prompt into the user turn (R1 distills)")
    ap.add_argument("--max-new-scale", type=float, default=1.0, help="reasoning models think at length: e.g. 6")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.models, a.n_train, a.n_test, a.gen_batch = "Qwen/Qwen2.5-0.5B-Instruct", [8, 4, 4, 4, 4], 6, 4
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        a.dtype = "float32"
    qa, gsm = pools(a)
    sets = build_sets(a, qa, gsm)
    log(phase="data", **{k: len(v) for k, v in sets.items()}, device=device,
        gpus=torch.cuda.device_count() if device == "cuda" else 0,
        gpu_mem_gb=[round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
                    for i in range(torch.cuda.device_count())] if device == "cuda" else [])
    res = {"config": vars(a)}
    t0 = time.time()
    for name in a.models.split(","):
        def checkpoint(R, name=name):
            res[name] = R
            json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
        try:
            checkpoint(run_model(name, sets, a, device, checkpoint))
        except Exception as e:
            log(phase="model_failed", model=name, error=repr(e)[:500])
            res[name] = {**(res.get(name) or {}), "error": repr(e)[:500]}
            json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
            if "device-side assert" in repr(e):
                break
        log(phase="model_done", model=name, **{k: v for k, v in res[name].items() if k in ("per_format", "leave_one_format_out")})
    log(phase="done", minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
