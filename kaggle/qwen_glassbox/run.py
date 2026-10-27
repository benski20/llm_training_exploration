"""Kaggle kernel: convert Qwen2.5-0.5B-Instruct into a glass-box model and test whether its exact
internals tell when an answer is correct.

  1. Distill: each dense MLP -> an exactly sparse feature layer (top-k), trained to reproduce the
     MLP's outputs on teacher chat data, plus the in-model reconstruction (SAE) objective.
  2. Fine-tune the converted model end to end (attention frozen) to match the original model's
     next-token distribution; train a self-knowledge head k.
  3. Evaluate: agreement with the original; TriviaQA accuracy of both models; correctness of each
     TriviaQA answer predicted from (a) softmax and a post-hoc hidden-state probe, for the original
     and the converted model, (b) k, (c) a probe on exactly which sparse features fired, (d) all
     combined; exactness of attribution; token purity of the features.

    python run.py --data /kaggle/input --out /kaggle/working/qwen_glassbox
    python run.py --smoke --data <dir with train.jsonl> --out /tmp/qg
"""
import argparse
import json
import math
import os
import random
import re
import string
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vtpkg"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from vt.qwen_glassbox import KHead, ScaleRecorder, convert, explain, set_original, sparse_layers  # noqa: E402

SYSTEM = "You are a helpful assistant. Answer the question with a short answer."


def log(**kw):
    print(json.dumps(kw), flush=True)


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if labels.all() or (~labels).all():
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - labels.sum() * (labels.sum() + 1) / 2) / (labels.sum() * (~labels).sum()))


def fit_probe(X, y, device, steps=400, l2=1e-3):
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(((X - mu) / sd) @ w + b, y) + l2 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    w_, b_, mu_, sd_ = w.detach(), b.detach(), mu, sd
    return lambda Z: (((torch.tensor(np.asarray(Z), dtype=torch.float32, device=device) - mu_) / sd_) @ w_
                      + b_).cpu().numpy(), (w_ / sd_).cpu().numpy()


def find_data(root):
    for dp, _, fs in os.walk(root):
        if "train.jsonl" in fs:
            return dp
    raise SystemExit(f"no train.jsonl under {root}")


def chat(tok, prompt, reply=None):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s + (reply if reply is not None else "")


def token_stream(tok, data_dir, n_docs, held_out):
    rows = [json.loads(l) for l in open(os.path.join(data_dir, "train.jsonl"))]
    random.Random(0).shuffle(rows)
    test, train = rows[:held_out], rows[held_out:held_out + n_docs]
    def enc(rs):
        ids = []
        for r in rs:
            ids += tok(chat(tok, r["prompt"], r["response"]) + "<|im_end|>\n").input_ids
        return torch.tensor(ids, dtype=torch.long)
    return enc(train), enc(test)


def batches(stream, bs, block, g):
    ix = torch.randint(len(stream) - block - 1, (bs,), generator=g)
    return torch.stack([stream[i:i + block] for i in ix.tolist()])


def distill(model, stream, a, device):
    """Phase 1: each sparse layer learns its MLP's input -> output map (plus reconstruction)."""
    sps = sparse_layers(model)
    caps = {}
    hooks = [sp.original.register_forward_hook(lambda m, i, o, k=k: caps.__setitem__(k, (i[0].detach(), o.detach())))
             for k, sp in enumerate(sps)]
    params = [p for sp in sps for p in list(sp.enc.parameters()) + list(sp.dec.parameters()) + list(sp.rec.parameters())]
    opt = torch.optim.Adam(params, lr=a.lr1)
    scaler1 = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    g = torch.Generator().manual_seed(1)
    set_original(model, True)
    t0 = time.time()
    for step in range(1, a.steps1 + 1):
        x = batches(stream, a.batch, a.block, g).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            model(x)
        loss, info = 0.0, []
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            for k, sp in enumerate(sps):
                xin, yout = caps[k]
                sp.use_original = False
                yhat = sp(xin).float()
                sp.use_original = True
                yout = yout.float()
                fvu = F.mse_loss(yhat, yout) / (yout.var() + 1e-6)
                loss = loss + fvu + a.recon * sp.recon_loss()
                info.append(fvu.item())
        opt.zero_grad(); scaler1.scale(loss / len(sps)).backward(); scaler1.step(opt); scaler1.update()
        if step % 100 == 0 or step == a.steps1:
            log(phase="distill", step=step, fvu_mean=round(float(np.mean(info)), 4),
                fvu_max=round(float(np.max(info)), 4), s=round(time.time() - t0))
    for h in hooks:
        h.remove()
    set_original(model, False)


def finetune(model, khead, stream, a, device):
    """Phase 2: the converted model matches the original's next-token distribution (attention frozen)."""
    for p in model.parameters():
        p.requires_grad_(False)
    sps = sparse_layers(model)
    params = [p for sp in sps for p in list(sp.enc.parameters()) + list(sp.dec.parameters()) + list(sp.rec.parameters())]
    params += list(khead.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=a.lr2)
    scaler2 = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    g = torch.Generator().manual_seed(2)
    t0 = time.time()
    for step in range(1, a.steps2 + 1):
        x = batches(stream, a.batch2, a.block, g).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            set_original(model, True)
            teacher = model(x).logits.float()
            set_original(model, False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(x, output_hidden_states=True)
            logits = out.logits.float()
            kd = F.kl_div(F.log_softmax(logits, -1).flatten(0, 1), F.log_softmax(teacher, -1).flatten(0, 1),
                          log_target=True, reduction="batchmean")
            rec = sum(sp.recon_loss() for sp in sps) / len(sps)
            k = khead(out.hidden_states[-1].float())[:, :-1]
            right = (logits[:, :-1].argmax(-1) == x[:, 1:]).float()
            kb = F.binary_cross_entropy_with_logits(k, right)
            loss = kd + a.recon * rec + 0.2 * kb
        opt.zero_grad(); scaler2.scale(loss).backward()
        scaler2.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler2.step(opt); scaler2.update()
        if step % 100 == 0 or step == a.steps2:
            log(phase="finetune", step=step, kl=round(kd.item(), 4), rec=round(rec.item(), 4),
                k_bce=round(kb.item(), 4), s=round(time.time() - t0))
    model.eval()


@torch.no_grad()
def agreement(model, stream, a, device, n=40):
    """Top-1 agreement and KL of the converted model against the original on held-out text."""
    g = torch.Generator().manual_seed(9)
    agree, kl, nll_c, nll_o, tokens = 0, 0.0, 0.0, 0.0, 0
    for _ in range(n):
        x = batches(stream, 4, a.block, g).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            set_original(model, True)
            t = model(x).logits.float()[:, :-1]
            set_original(model, False)
            s = model(x).logits.float()[:, :-1]
        y = x[:, 1:]
        agree += int((t.argmax(-1) == s.argmax(-1)).sum())
        kl += float(F.kl_div(F.log_softmax(s, -1).flatten(0, 1), F.log_softmax(t, -1).flatten(0, 1),
                             log_target=True, reduction="sum"))
        nll_c += float(F.cross_entropy(s.flatten(0, 1), y.flatten(), reduction="sum"))
        nll_o += float(F.cross_entropy(t.flatten(0, 1), y.flatten(), reduction="sum"))
        tokens += y.numel()
    return {"top1_agreement_with_original": agree / tokens, "kl_nats_per_token": kl / tokens,
            "bits_per_token_converted": nll_c / tokens / math.log(2),
            "bits_per_token_original": nll_o / tokens / math.log(2)}


def load_trivia(n, smoke):
    if smoke:
        return [{"q": "What is the capital of France?", "aliases": ["Paris"]},
                {"q": "Who wrote Hamlet?", "aliases": ["Shakespeare", "William Shakespeare"]}] * 6
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    rows = [{"q": x["question"], "aliases": x["answer"]["normalized_aliases"] + [x["answer"]["value"]]}
            for x in ds.select(range(min(n, len(ds))))]
    return rows


@torch.no_grad()
def trivia_rows(model, khead, tok, items, device, original):
    """Answer each question greedily; signals at the first answer token's prediction position."""
    set_original(model, original)
    rows = []
    for it in items:
        prompt = chat(tok, it["q"])
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        gen = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        ans_ids = gen[0, ids.shape[1]:]
        ans = tok.decode(ans_ids, skip_special_tokens=True).strip()
        right = any(norm(al) and norm(al) in norm(ans) for al in it["aliases"])
        out = model(ids, output_hidden_states=True)
        lp = F.log_softmax(out.logits[0, -1].float(), -1)
        first = int(ans_ids[0]) if len(ans_ids) else int(lp.argmax())
        r = {"q": it["q"], "answer": ans, "right": int(right), "msp": float(lp.max().exp()),
             "neg_nll": float(lp[first]),
             "h": [h[0, -1].float().cpu().numpy() for h in out.hidden_states[1:]]}
        if not original:
            r["k"] = float(khead(out.hidden_states[-1][0, -1].float()))
            r["feats"] = np.concatenate([sp.trace["mask"][0, -1].cpu().numpy() for sp in sparse_layers(model)]
                                        ).astype(np.float32)
        rows.append(r)
    set_original(model, False)
    return rows


def detectors(rows, device, glass):
    """Fit each detector on one half, score on the other: AUROC and error among the 50% most confident."""
    idx = np.random.default_rng(0).permutation(len(rows))
    fit, test = [rows[i] for i in idx[: len(idx) // 2]], [rows[i] for i in idx[len(idx) // 2:]]
    yf, yt = np.array([r["right"] for r in fit]), np.array([r["right"] for r in test])

    def risk(s):
        keep = np.argsort(-np.asarray(s))[: len(yt) // 2]
        return float(1 - yt[keep].mean())
    out = {"n": len(rows), "accuracy": float(np.mean([r["right"] for r in rows])), "test_n": len(test)}

    def score(name, s):
        out[name] = {"auroc": auroc(s, yt), "error_at_50pct": risk(s)}
    score("softmax", [r["msp"] for r in test])
    score("neg_nll", [r["neg_nll"] for r in test])
    best = None
    h = len(fit) // 2
    for L in range(len(fit[0]["h"])):
        Xf = np.stack([r["h"][L] for r in fit])
        pr, _ = fit_probe(Xf[:h], yf[:h], device)
        s = auroc(pr(Xf[h:]), yf[h:])
        if s is not None and (best is None or s > best[0]):
            best = (s, L)
    L = best[1] if best else len(fit[0]["h"]) - 1
    pr, _ = fit_probe(np.stack([r["h"][L] for r in fit]), yf, device)
    score("posthoc_probe", pr(np.stack([r["h"][L] for r in test])))
    out["posthoc_probe"]["layer"] = L
    weights = None
    if glass:
        score("builtin_k", [r["k"] for r in test])
        pr, weights = fit_probe(np.stack([r["feats"] for r in fit]), yf, device)
        score("feature_probe", pr(np.stack([r["feats"] for r in test])))

        def comb(r):
            return np.concatenate([[r["msp"], r["neg_nll"], r["k"]], r["h"][L], r["feats"]])
        pr, _ = fit_probe(np.stack([comb(r) for r in fit]), yf, device)
        score("combined", pr(np.stack([comb(r) for r in test])))
    return out, weights


@torch.no_grad()
def purity(model, stream, a, device, layers, windows=60, top=20):
    g = torch.Generator().manual_seed(11)
    acts = {L: [] for L in layers}
    toks = []
    for _ in range(windows // 4):
        x = batches(stream, 4, a.block, g).to(device)
        model(x)
        toks.append(x.flatten().cpu())
        for L in layers:
            acts[L].append(sparse_layers(model)[L].trace["acts"].flatten(0, 1).float().clamp_min(0).cpu())
    toks = torch.cat(toks).numpy()
    res = {}
    for L in layers:
        A = torch.cat(acts[L]).numpy()
        fires = (A > 0).sum(0)
        keep = np.nonzero(fires >= top)[0]
        pur = []
        for u in keep:
            idx = np.argpartition(-A[:, u], top)[:top]
            pur.append(Counter(toks[idx].tolist()).most_common(1)[0][1] / top)
        pur = np.array(pur)
        res[f"layer{L}"] = {"units_firing_20+": int(len(keep)), "median_top20_purity": float(np.median(pur)),
                            "share_purity_at_least_0.8": float((pur >= 0.8).mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/qwen_glassbox")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-features", type=int, default=8192)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--recon", type=float, default=0.5)
    ap.add_argument("--steps1", type=int, default=2500)
    ap.add_argument("--steps2", type=int, default=2000)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batch2", type=int, default=2)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--n-docs", type=int, default=20000)
    ap.add_argument("--n-trivia", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_features, a.k, a.steps1, a.steps2, a.batch, a.batch2, a.block, a.n_docs, a.n_trivia = \
            512, 16, 4, 3, 2, 1, 64, 60, 12
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager")
    model.to(device).eval()
    stream, held = token_stream(tok, find_data(a.data), a.n_docs, held_out=20 if a.smoke else 600)
    log(phase="data", train_tokens=len(stream), held_tokens=len(held), device=device)
    convert(model, a.n_features, a.k)
    model.to(device)
    khead = KHead(model.config.hidden_size).to(device)
    res = {"config": {k: v for k, v in vars(a).items()}}
    trivia = load_trivia(a.n_trivia, a.smoke)
    rows_o = trivia_rows(model, khead, tok, trivia, device, original=True)       # the original model
    res["original_trivia"], _ = detectors(rows_o, device, glass=False)
    log(phase="original", **{k: v for k, v in res["original_trivia"].items()})
    distill(model, stream, a, device)
    finetune(model, khead, stream, a, device)
    res["agreement"] = agreement(model, held, a, device, n=3 if a.smoke else 40)
    log(phase="agreement", **res["agreement"])
    rows_c = trivia_rows(model, khead, tok, trivia, device, original=False)
    res["converted_trivia"], w = detectors(rows_c, device, glass=True)
    log(phase="converted", **{k: v for k, v in res["converted_trivia"].items()})
    errs = []                                                                   # exactness of attribution
    for it in trivia[:5]:
        ids = tok(chat(tok, it["q"]), return_tensors="pt").input_ids.to(device)
        logits = model(ids).logits[0, -1]
        parts, logit = explain(model, ids, ids.shape[1] - 1, int(logits.argmax()))
        errs.append(abs(sum(parts.values()) - logit))
    res["attribution_max_abs_error"] = float(max(errs))
    nl = model.config.num_hidden_layers
    res["purity"] = purity(model, held, a, device, sorted({nl // 2, nl - 1}), windows=8 if a.smoke else 60)
    log(phase="inspect", attribution_max_abs_error=res["attribution_max_abs_error"], purity=res["purity"])
    if w is not None:                                         # features that most signal a wrong answer
        F_ = a.n_features
        order = np.argsort(w)
        res["features_signalling_wrong"] = [{"layer": int(i // F_), "feature": int(i % F_), "weight": float(w[i])}
                                            for i in order[:15]]
        res["features_signalling_right"] = [{"layer": int(i // F_), "feature": int(i % F_), "weight": float(w[i])}
                                            for i in order[::-1][:15]]
    res["samples"] = [{"q": r["q"], "answer": r["answer"], "right": r["right"], "k": r["k"]} for r in rows_c[:12]]
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    sd = {k: v.half().cpu() for sp_i, sp in enumerate(sparse_layers(model))
          for k, v in [(f"layer{sp_i}.{n}", p) for n, p in sp.state_dict().items() if not n.startswith("original")]}
    sd.update({f"khead.{n}": p.half().cpu() for n, p in khead.state_dict().items()})
    torch.save(sd, os.path.join(a.out, "sparse_layers.pt"))
    log(phase="done")


if __name__ == "__main__":
    main()
