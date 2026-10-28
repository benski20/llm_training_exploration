"""Kaggle kernel: a general hallucination detector (vt/detector.py). A frozen reader model reads any
(prompt, optional documents, response) and linear probes on its hidden states score each response
token and the whole response. Responses come from other models (GPT-4, GPT-3.5, Llama-2, Mistral,
ChatGPT), so this judges closed models too.

  1. train pass: per response, mean hidden state at 4 depths; token states (2 depths, sampled) from
     the span-labelled sets
  2. fit: response probe and token probe (layer chosen on held-out train), plus one of each with every
     source left out; a stacker on a separate stacking split (probe, token max/mean/p95, log-probs)
  3. test pass (streamed, scored on the fly): every source's test split; exact provenance
     (vt.qwen_glassbox.explain_all: how much of each token's logit the document contributed) on a
     sample of document-grounded items
  4. report per source: response AUROC (+95% CI) and F1 at a threshold chosen on the stacking split,
     token AUROC for span sources, leave-one-source-out, against log-probability baselines

    python run.py --out /kaggle/working/detector --model Qwen/Qwen2.5-7B-Instruct
    python run.py --smoke --out /tmp/det
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

from vt.correctness import Probe, auroc, auroc_ci, report  # noqa: E402
from vt.detector import SOURCES, encode, load_all, response_states  # noqa: E402

SPAN_SOURCES = ("ragtruth", "halueval_general")


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def f1_at(scores, y, th):
    p = np.asarray(scores) >= th
    y = np.asarray(y).astype(bool)
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def best_threshold(scores, y):
    cands = np.unique(np.percentile(scores, np.linspace(1, 99, 99)))
    return float(max(cands, key=lambda t: f1_at(scores, y, t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/detector")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--train-caps", default="ragtruth=6000,halueval_general=3000,halueval_qa=2000,"
                                            "halueval_dialogue=2000,halueval_summary=1500,truthfulqa=1000")
    ap.add_argument("--test-caps", default="ragtruth=2700,halueval_general=1000,halueval_qa=500,"
                                           "halueval_dialogue=500,halueval_summary=400,truthfulqa=1000")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--token-cap", type=int, default=200000, help="training tokens kept per token layer")
    ap.add_argument("--neg-keep", type=float, default=0.15, help="share of non-hallucinated tokens kept for training")
    ap.add_argument("--stack-frac", type=float, default=0.15)
    ap.add_argument("--provenance", type=int, default=300, help="document-grounded test items with exact provenance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.model, a.train_caps, a.test_caps, a.provenance = ("Qwen/Qwen2.5-0.5B-Instruct",
                                                            "ragtruth=24,halueval_general=24,halueval_qa=8,halueval_dialogue=8,halueval_summary=4,truthfulqa=10",
                                                            "ragtruth=12,halueval_general=12,halueval_qa=4,halueval_dialogue=4,halueval_summary=3,truthfulqa=6", 3)
    caps = lambda s: {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(torch_dtype=torch.float16 if cuda else torch.float32, attn_implementation="sdpa")
    if cuda:
        kw["device_map"] = "auto"
    if a.quant != "none":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=a.quant == "8bit", load_in_4bit=a.quant == "4bit",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    L = model.config.num_hidden_layers
    layers = sorted({int(round(f * (L - 1))) for f in (0.4, 0.55, 0.7, 0.85)})
    tok_layers = [layers[1], layers[2]]
    res = {"config": vars(a), "model": a.model, "layers": layers, "token_layers": tok_layers,
           "gpus": torch.cuda.device_count() if cuda else 0, "complete": False}

    def save():
        json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)

    train = load_all("train", caps(a.train_caps))
    test = load_all("test", caps(a.test_caps))
    random.Random(0).shuffle(train)
    n_stack = int(len(train) * a.stack_frac)
    stack, fit_items = train[:n_stack], train[n_stack:]
    log(phase="data", fit=len(fit_items), stack=len(stack), test=len(test), gpus=res["gpus"],
        by_source={s: sum(x["source"] == s for x in train + test) for s in SOURCES})
    res["counts"] = {"fit": len(fit_items), "stack": len(stack), "test": len(test)}
    # 1. training pass
    R, RY, RS, TK, TY, TS = [], [], [], {L_: [] for L_ in tok_layers}, [], []
    kept = 0
    rng = np.random.default_rng(0)
    for i, it in enumerate(fit_items):
        e = encode(tok, it, a.max_tokens)
        if e is None or e["r0"] >= len(e["ids"]):
            continue
        h, lp = response_states(model, e["ids"], e["r0"], layers)
        R.append(h.float().mean(1).half())
        RY.append(it["label"])
        RS.append(it["source"])
        if e["labels"] is not None and kept < a.token_cap:
            lab = np.array(e["labels"])
            keep = np.where((lab == 1) | (rng.random(len(lab)) < a.neg_keep))[0]
            for L_ in tok_layers:
                TK[L_].append(h[layers.index(L_), keep])
            TY += lab[keep].tolist()
            TS += [it["source"]] * len(keep)
            kept += len(keep)
        if (i + 1) % 500 == 0:
            log(phase="train_pass", done=i + 1, of=len(fit_items), tokens=kept, s=round(time.time() - t0))
    R = torch.stack(R).float().numpy()                               # (N, nL, d)
    RY, RS = np.array(RY), np.array(RS)
    TY, TS = np.array(TY), np.array(TS)
    TK = {L_: torch.cat(v).float().numpy() for L_, v in TK.items()}
    pdev = "cuda" if cuda else "cpu"
    # 2. fit probes
    idx = rng.permutation(len(RY))
    a_, b_ = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
    by_layer = {L_: auroc(Probe(pdev).fit(R[a_, k], RY[a_]).logit(R[b_, k]), RY[b_]) for k, L_ in enumerate(layers)}
    rk = max(range(len(layers)), key=lambda k: by_layer[layers[k]] or 0)
    res["response_probe_by_layer"], res["response_layer"] = by_layer, layers[rk]
    resp_probe = Probe(pdev).fit(R[:, rk], RY)
    tidx = rng.permutation(len(TY))
    ta, tb = tidx[: int(0.8 * len(tidx))], tidx[int(0.8 * len(tidx)):]
    tby = {L_: auroc(Probe(pdev).fit(TK[L_][ta], TY[ta]).logit(TK[L_][tb]), TY[tb]) for L_ in tok_layers}
    tl = max(tok_layers, key=lambda L_: tby[L_] or 0)
    res["token_probe_by_layer"], res["token_layer"] = tby, tl
    tok_probe = Probe(pdev).fit(TK[tl], TY)
    lodo_resp = {s: Probe(pdev).fit(R[RS != s, rk], RY[RS != s]) for s in SOURCES if (RS == s).any()}
    lodo_tok = {s: Probe(pdev).fit(TK[tl][TS != s], TY[TS != s]) for s in SPAN_SOURCES if (TS == s).any() and (TS != s).any()}
    log(phase="fitted", response_layer=layers[rk], token_layer=tl, by_layer=by_layer, token_by_layer=tby, s=round(time.time() - t0))
    del TK, R
    save()

    def score(it, provenance=False):
        e = encode(tok, it, a.max_tokens)
        if e is None or e["r0"] >= len(e["ids"]):
            return None
        h, lp = response_states(model, e["ids"], e["r0"], [layers[rk], tl])
        ht = h[1].float().numpy()
        ts = tok_probe.logit(ht)
        r = {"source": it["source"], "task": it["task"], "label": it["label"], "token_labels": e["labels"],
             "resp": float(resp_probe.logit(h[0].float().mean(0, keepdim=True).numpy())[0]),
             "tok_max": float(ts.max()), "tok_mean": float(ts.mean()), "tok_p95": float(np.percentile(ts, 95)),
             "lp_mean": float(lp.mean()), "lp_min": float(lp.min()), "tok_scores": ts.tolist(), "tok_lp": lp.tolist(),
             "lodo_resp": {s: float(p.logit(h[0].float().mean(0, keepdim=True).numpy())[0]) for s, p in lodo_resp.items()},
             "lodo_tok_max": {s: float(p.logit(ht).max()) for s, p in lodo_tok.items()},
             "lodo_tok": {s: p.logit(ht).tolist() for s, p in lodo_tok.items() if s == it["source"]}}
        if provenance and "document" in e["groups"]:
            from vt.qwen_glassbox import explain_all
            dev = model.get_input_embeddings().weight.device
            names, c, _ = explain_all(model, torch.tensor([e["ids"]], device=dev), e["groups"], start=e["r0"] - 1)
            c = c.float().cpu().numpy()
            r["doc_share"] = (c[names.index("document")] / (np.abs(c).sum(0) + 1e-6)).tolist()
        return r

    def feats(r):
        return [r["resp"], r["tok_max"], r["tok_mean"], r["tok_p95"], r["lp_mean"], r["lp_min"]]
    # 3. stacking split
    S_ = [r for r in (score(it) for it in stack) if r]
    stacker = Probe(pdev).fit(np.array([feats(r) for r in S_]), np.array([r["label"] for r in S_]),
                              grid=(1e-4, 1e-3, 1e-2, 1e-1))
    zs = stacker.logit(np.array([feats(r) for r in S_]))
    th = best_threshold(zs, [r["label"] for r in S_])
    log(phase="stacked", n=len(S_), threshold=th, s=round(time.time() - t0))
    torch.save({"model": a.model, "response_layer": layers[rk], "token_layer": tl, "response_probe": resp_probe.state_dict(),
                "token_probe": tok_probe.state_dict(), "stacker": stacker.state_dict(), "threshold": th,
                "features": ["resp", "tok_max", "tok_mean", "tok_p95", "lp_mean", "lp_min"]},
               os.path.join(a.out, a.model.split("/")[-1] + "_detector.pt"))
    # 4. test pass
    rows, prov_left = [], a.provenance
    for i, it in enumerate(test):
        want = prov_left > 0 and bool(it["doc"]) and it["source"] in ("ragtruth", "halueval_qa", "halueval_summary")
        try:
            r = score(it, provenance=want)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            r = score(it, provenance=False)
        if r:
            rows.append(r)
            prov_left -= int("doc_share" in r)
        if (i + 1) % 500 == 0:
            log(phase="test_pass", done=i + 1, of=len(test), s=round(time.time() - t0))
    R_ = {}
    for s in SOURCES + ("all",):
        rs = [r for r in rows if s == "all" or r["source"] == s]
        if len(rs) < 10 or len({r["label"] for r in rs}) < 2:
            continue
        y = [r["label"] for r in rs]
        z = stacker.logit(np.array([feats(r) for r in rs]))
        d = {"n": len(rs), "positives": int(sum(y)),
             "stack": {**report(z, y, prob=1 / (1 + np.exp(-z)), ci=True), "f1": f1_at(z, y, th)},
             "response_probe": report([r["resp"] for r in rs], y, ci=True),
             "token_max": report([r["tok_max"] for r in rs], y),
             "baseline_low_mean_logprob": report([-r["lp_mean"] for r in rs], y, ci=True),
             "baseline_low_min_logprob": report([-r["lp_min"] for r in rs], y)}
        if s in SOURCES and s in rs[0]["lodo_resp"]:
            d["left_out_response_probe"] = report([r["lodo_resp"][s] for r in rs], y, ci=True)
        if s in SOURCES and rs[0]["lodo_tok_max"].get(s) is not None:
            d["left_out_token_max"] = report([r["lodo_tok_max"][s] for r in rs], y)
        tr = [(sc, lab, lp) for r in rs if r["token_labels"] for sc, lab, lp in zip(r["tok_scores"], r["token_labels"], r["tok_lp"])]
        if tr and len({t[1] for t in tr}) > 1:
            ty = [t[1] for t in tr]
            d["token_level"] = {"probe_auroc": auroc([t[0] for t in tr], ty), "probe_auroc_ci": auroc_ci([t[0] for t in tr], ty, n=200),
                                "low_logprob_auroc": auroc([-t[2] for t in tr], ty), "tokens": len(tr), "positives": int(sum(ty))}
            lo = [(sc, lab) for r in rs if r["token_labels"] and s in r["lodo_tok"] for sc, lab in zip(r["lodo_tok"][s], r["token_labels"])]
            if lo:
                d["token_level"]["left_out_probe_auroc"] = auroc([t[0] for t in lo], [t[1] for t in lo])
        pv = [(ds, lab, sc) for r in rs if r.get("doc_share") and r["token_labels"]
              for ds, lab, sc in zip(r["doc_share"], r["token_labels"], r["tok_scores"])]
        if pv and len({t[1] for t in pv}) > 1:
            d["provenance_tokens"] = {"low_doc_share_auroc": auroc([-t[0] for t in pv], [t[1] for t in pv]),
                                      "probe_auroc_same_tokens": auroc([t[2] for t in pv], [t[1] for t in pv]), "tokens": len(pv)}
        R_[s] = d
    res["per_source"], res["seconds"], res["complete"] = R_, round(time.time() - t0), True
    save()
    log(phase="done", per_source={s: {k: v.get("auroc") if isinstance(v, dict) else v for k, v in d.items()
                                      if k in ("stack", "response_probe", "baseline_low_mean_logprob")} for s, d in R_.items()},
        minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
