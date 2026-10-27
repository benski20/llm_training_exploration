"""Kaggle kernel: how well can a pretrained model's internals predict whether its answer is right,
with enough training data, several signals and a dataset shift? (`vt/correctness.py`)

For each model (Qwen2.5-Instruct, several sizes):
  1. Greedy short answers to TriviaQA train questions (fitting), the 1,000 TriviaQA validation
     questions used throughout VISION.md (test), and 1,000 NQ-open questions (test under shift:
     nothing is fitted on NQ).
  2. Signals per answer: hidden states at the entity token (10 layers), output probabilities;
     on a stacking subset of train and both test sets also K sampled answers (consistency) and the
     exact attribution of the entity token (grounding).
  3. Detectors: unfitted signals; the entity probe by layer and by training-set size (500 -> all);
     a multi-layer probe; a probe trained to predict consistency (one pass at test time); stackers
     over probe + output (+ consistency) (+ grounding). AUROC, error among the 25% / 50% most
     confident answers, calibration (ECE).

    python run.py --out /kaggle/working/correctness
    python run.py --smoke --out /tmp/cq
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vtpkg"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch  # noqa: E402

from vt.correctness import (GROUND_KEYS, OUTPUT_KEYS, CorrectnessHead, Probe, auroc, chat, consistency,  # noqa: E402
                            features, generate, grounding, is_right, report, stack_features)


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def load_sets(n_train, n_test, smoke):
    if smoke:
        qa = [("What is the capital of France?", ["Paris"]), ("Who wrote Hamlet?", ["Shakespeare"]),
              ("What is the largest planet?", ["Jupiter"]), ("Who painted the Mona Lisa?", ["Leonardo da Vinci"]),
              ("What is the chemical symbol for gold?", ["Au"]), ("In which country is Kyoto?", ["Japan"])]
        rows = [{"q": q, "aliases": a} for q, a in qa]
        return {"train": rows * 4, "trivia_val": rows * 2, "nq": rows}
    from datasets import load_dataset
    tr = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    idx = random.Random(0).sample(range(len(tr)), n_train)
    sel = tr.select(idx)
    train = [{"q": x["question"], "aliases": x["answer"]["normalized_aliases"] + [x["answer"]["value"]]} for x in sel]
    va = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").select(range(n_test))
    val = [{"q": x["question"], "aliases": x["answer"]["normalized_aliases"] + [x["answer"]["value"]]} for x in va]
    nq = load_dataset("google-research-datasets/nq_open", split="validation").select(range(n_test))
    nqr = [{"q": x["question"][0].upper() + x["question"][1:] + "?", "aliases": x["answer"]} for x in nq]
    return {"train": train, "trivia_val": val, "nq": nqr}


def answer_set(model, tok, items, layers, a):
    prompts = [chat(tok, it["q"]) for it in items]
    ans = generate(model, tok, prompts, batch=a.gen_batch)
    rows = features(model, tok, [tok(p).input_ids for p in prompts], ans, [it["q"] for it in items], layers,
                    batch=a.feat_batch)
    for r, it, p, x in zip(rows, items, prompts, ans):
        r.update(q=it["q"], prompt=p, ans_ids=x, reply=tok.decode(x, skip_special_tokens=True).strip())
        r["right"] = is_right(r["reply"], it["aliases"])
    return rows


def enrich(model, tok, rows, a):
    """Consistency (K samples) and grounding (exact attribution) for a set of rows."""
    samples = generate(model, tok, [r["prompt"] for r in rows], batch=max(1, a.gen_batch // a.k), n=a.k, sample=True)
    for i, r in enumerate(rows):
        s = [tok.decode(x, skip_special_tokens=True) for x in samples[i * a.k:(i + 1) * a.k]]
        r["consistency"] = consistency(r["reply"], s)
        r.update(grounding(model, tok, r["prompt"], r["q"], r["ans_ids"]))


def H(rows, li=None):
    X = torch.stack([r["h"] for r in rows]).float()
    return (X[:, li] if li is not None else X.flatten(1)).numpy()


def run_model(name, sets, a, device, checkpoint):
    """checkpoint(R) is called with partial results after the cheap phase (answers, probes, probe+output
    stack), so a run cut off during the costly enrichment (sampling, exact grounding) still reports."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dtype = getattr(torch, a.dtype)          # float16 needs sdpa: eager overflows on Qwen2.5's raw q.k
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation=a.attn).to(device).eval()
    L = model.config.num_hidden_layers
    layers = sorted({int(x) for x in np.linspace(L // 4, L - 1, 10).round()})
    R = {"model": name, "layers": layers, "dtype": a.dtype, "attn": a.attn, "complete": False}
    data = {}
    for key, items in sets.items():
        data[key] = answer_set(model, tok, items, layers, a)
        acc = float(np.mean([r["right"] for r in data[key]]))
        log(model=name, phase="answers", set=key, n=len(items), accuracy=acc, s=round(time.time() - t0),
            samples=[(r["q"], r["reply"]) for r in data[key][:3]])
        if acc == 0.0 and len(items) >= 100:
            raise RuntimeError(f"{name}: no correct answers on {key}; generation is broken")
        if any(torch.isnan(r["h"].float()).any() for r in data[key][:50]):
            raise RuntimeError(f"{name}: NaN hidden states on {key}")
    n_stack = min(a.n_stack, len(data["train"]) // 3)
    B, A = data["train"][:n_stack], data["train"][n_stack:]
    R["accuracy"] = {k: float(np.mean([r["right"] for r in v])) for k, v in data.items()}
    y = {k: np.array([r["right"] for r in v]) for k, v in data.items()}
    yA, yB = np.array([r["right"] for r in A]), np.array([r["right"] for r in B])
    tests = ("trivia_val", "nq")
    R["unfitted"] = {t: {k: report([r[k] for r in data[t]], y[t]) for k in OUTPUT_KEYS} for t in tests}
    by_layer = {}                                     # entity probe: layer chosen on the stacking split
    for li, L_ in enumerate(layers):
        p = Probe(device).fit(H(A, li), yA)
        by_layer[L_] = auroc(p.logit(H(B, li)), yB)
    best_li = max(range(len(layers)), key=lambda i: by_layer[layers[i]] or 0)
    R["probe_auroc_by_layer_on_stack_split"] = by_layer
    R["probe_layer"] = layers[best_li]
    R["probe_by_train_size"] = {}
    for n in (500, 2000, len(A)):
        if n > len(A):
            continue
        p = Probe(device).fit(H(A[:n], best_li), yA[:n])
        R["probe_by_train_size"][n] = {t: report(p.logit(H(data[t], best_li)), y[t]) for t in tests}
    probe = Probe(device).fit(H(A, best_li), yA)
    multi = Probe(device).fit(H(A), yA)
    R["multi_layer_probe"] = {t: report(multi.logit(H(data[t])), y[t]) for t in tests}
    pl = {k: probe.logit(H(v, best_li)) for k, v in (("stack", B), ("trivia_val", data["trivia_val"]), ("nq", data["nq"]))}
    R["stacked"], heads = {}, {}

    def stack(name_s, uc, ug):            # stackers fitted on B, probe scores on B from the probe fitted on A
        Xs = np.stack([stack_features(pl["stack"][i], r, uc, ug) for i, r in enumerate(B)])
        st = Probe(device).fit(Xs, yB, grid=(1e-4, 1e-3, 1e-2, 1e-1))
        R["stacked"][name_s] = {}
        for t in tests:
            z = st.logit(np.stack([stack_features(pl[t][i], r, uc, ug) for i, r in enumerate(data[t])]))
            R["stacked"][name_s][t] = report(z, y[t], prob=1 / (1 + np.exp(-z)))
        heads[name_s] = CorrectnessHead([layers[best_li]], probe, st, uc, ug, a.k)

    def save_heads():
        torch.save({k: h.state_dict() for k, h in heads.items()}, os.path.join(a.out, name.split("/")[-1] + "_heads.pt"))
    stack("probe+output", False, False)
    R["seconds_cheap_phase"] = round(time.time() - t0)
    save_heads()
    checkpoint(R)
    log(model=name, phase="cheap_done", probe=R["probe_by_train_size"], stacked=R["stacked"], s=round(time.time() - t0))
    for key, rows in (("stack", B), ("trivia_val", data["trivia_val"]), ("nq", data["nq"])):
        enrich(model, tok, rows, a)
        log(model=name, phase="enrich", set=key, s=round(time.time() - t0))
    R["attribution_max_abs_error"] = max(r["exact_err"] for k in ("trivia_val", "nq") for r in data[k])
    for t in tests:
        R["unfitted"][t].update({k: report([r[k] for r in data[t]], y[t]) for k in ["consistency"] + GROUND_KEYS})
    sep = Probe(device).fit(H(B, best_li), (np.array([r["consistency"] for r in B]) >= 0.5).astype(np.float32))
    R["consistency_probe"] = {t: report(sep.logit(H(data[t], best_li)), y[t]) for t in tests}
    for name_s, uc, ug in (("probe+output+grounding", False, True), ("probe+output+consistency", True, False),
                           ("all", True, True)):
        stack(name_s, uc, ug)
    R["examples"] = []
    zs = heads["all"].stacker.logit(np.stack([stack_features(pl["trivia_val"][i], r, True, True)
                                              for i, r in enumerate(data["trivia_val"])]))
    for i, r in enumerate(data["trivia_val"][:12]):
        R["examples"].append({"q": r["q"], "reply": r["reply"], "right": r["right"],
                              "p_correct": round(float(1 / (1 + np.exp(-zs[i]))), 3),
                              "consistency": r["consistency"], "entity_p": round(r["entity_p"], 3),
                              "top_question_token": r["top_question_token"],
                              "question_share": round(r["question_share"], 3)})
    R["seconds"] = round(time.time() - t0)
    R["complete"] = True
    save_heads()
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/correctness")
    ap.add_argument("--models", default="Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-stack", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--gen-batch", type=int, default=48)
    ap.add_argument("--feat-batch", type=int, default=8)
    ap.add_argument("--budget-min", type=float, default=200, help="skip remaining models after this")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--attn", default="sdpa", help="sdpa: eager attention overflows in float16 on Qwen2.5")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.models, a.k, a.gen_batch, a.feat_batch = "Qwen/Qwen2.5-0.5B-Instruct", 2, 4, 4
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = load_sets(a.n_train, a.n_test, a.smoke)
    log(phase="data", **{k: len(v) for k, v in sets.items()}, device=device)
    t0, res = time.time(), {"config": vars(a)}
    for name in a.models.split(","):
        if (time.time() - t0) / 60 > a.budget_min:
            log(phase="skip", model=name, reason="time budget")
            continue
        def checkpoint(R, name=name):
            res[name] = R
            json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
        try:
            res[name] = run_model(name, sets, a, device, checkpoint)
        except Exception as e:                       # one model's failure must not lose the others
            log(phase="model_failed", model=name, error=repr(e)[:500])
            res[name] = {**(res.get(name) or {}), "error": repr(e)[:500]}
            json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
            if device == "cuda":
                torch.cuda.empty_cache()
            if "device-side assert" in repr(e):      # the CUDA context is unusable after this
                break
            continue
        log(phase="model_done", **{k: v for k, v in res[name].items() if k != "examples"})
        json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    log(phase="done", minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
