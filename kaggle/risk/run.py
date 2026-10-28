"""Kaggle kernel: a learned, heuristic-free risk head for a chat model's own replies (vt/guided.py
risk_probe), plus its claim probe, from one set of forward passes.

Every written token gets P(this token is part of a hallucination), read from its own-position hidden
state; a sentence's risk is its riskiest content token. Labels come from three kinds of data, so the
head is not tied to names and numbers:

  own        the model's own replies in five formats (short, several answers, multi-turn, reasoning,
             GSM8K math), claims auto-labelled against gold answers (vt/claims.py): the claim word's
             tokens get the claim's label; its sentence too
  spans      RAGTruth and HaluEval-general responses with human-labelled hallucinated spans, read
             teacher-forced: token labels; a sentence is hallucinated if it overlaps a span
  steps      Math-Shepherd solution steps labelled correct / incorrect, read teacher-forced: the label at
             each step's last token; a step is a sentence

Evaluation on held-out items per source: token AUROC, sentence (step) AUROC with max-over-sentence,
reply AUROC, each against low token log-prob; leave-one-source-out; calibration of the sentence
threshold. The claim probe (predicting-position states at claim tokens, as kaggle/claims) is saved too.

    python run.py --model Qwen/Qwen3-32B --quant 4bit --out /kaggle/working/risk
    python run.py --smoke --out /tmp/risk
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

from vt.claims import FORMATS, label_claims, make_items, word_starts  # noqa: E402
from vt.correctness import Probe, auroc, auroc_ci, generate  # noqa: E402
from vt.detector import encode, load_all  # noqa: E402

MAX_NEW = {"short": 48, "multiturn": 48, "multi": 128, "reasoning": 384, "math": 512}
THINK = {"short": False, "multiturn": False, "multi": False, "reasoning": True, "math": True}


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def chat_prompt(tok, messages, thinking):
    """The chat template with Qwen3's thinking switch (ignored by templates without it)."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def after_thinking(tok, ids):
    text = ""
    for j, t in enumerate(ids):
        text += tok.decode([t])
        if "</think>" in text[-12:]:
            return j + 1
    return 0 if "<think>" not in tok.decode(ids[:8]) else None


def sentence_ids(tok, ids, lo=0):
    """Sentence number of every token of ids[lo:] (boundaries after . ! ? or a newline)."""
    out, k = [], 0
    for t in ids[lo:]:
        out.append(k)
        p = tok.decode([t])
        if p.rstrip(" \"')”’").endswith((".", "!", "?")) or "\n" in p:
            k += 1
    return out


@torch.no_grad()
def states(model, seqs, starts, layers, keep=None, batch_tokens=2048):
    """For each token sequence: (H, lp, pos). pos lists the reply positions kept (keep[i], or all);
    H (len(layers), len(pos), d) holds their own-position hidden states (the state after reading the
    token); lp the log-prob of every reply token. Left-padded batches of similar length."""
    dev = model.get_input_embeddings().weight.device
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [None] * len(seqs)

    def run_group(g):
        M = max(len(seqs[i]) for i in g)
        ids = torch.zeros((len(g), M), dtype=torch.long)
        mask = torch.zeros((len(g), M), dtype=torch.long)
        for b, i in enumerate(g):
            ids[b, M - len(seqs[i]):], mask[b, M - len(seqs[i]):] = torch.tensor(seqs[i]), 1
        ids, mask = ids.to(dev), mask.to(dev)
        kept, hooks = {}, []
        blocks = model.model.layers
        for L in layers:
            hooks.append(blocks[L].register_forward_hook(
                lambda m, a, o, L=L: kept.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())))
        try:
            hs = model.model(input_ids=ids, attention_mask=mask,
                             position_ids=(mask.cumsum(-1) - 1).clamp(min=0)).last_hidden_state
        finally:
            for h in hooks:
                h.remove()
        for b, i in enumerate(g):
            off = M - len(seqs[i])
            s0 = off + starts[i]
            n = M - s0
            pos = list(range(n)) if keep is None or keep[i] is None else [t for t in keep[i] if 0 <= t < n]
            sel = torch.tensor(pos, dtype=torch.long, device=hs.device) + s0
            H = torch.stack([kept[L][b].index_select(0, sel.to(kept[L].device)) for L in layers]).to("cpu", torch.float16)
            tgt = ids[b, s0:]
            parts = []
            for c in range(0, n, 256):                          # chunks: the full-vocab logits are large
                lg = model.lm_head(hs[b, s0 - 1 + c:min(M - 1, s0 - 1 + c + 256)]).float().log_softmax(-1)
                parts.append(lg.gather(-1, tgt[c:c + 256, None].to(lg.device))[:, 0].cpu())
            lp = torch.cat(parts)
            out[i] = (H, lp, pos)
        del kept, hs

    k = 0
    while k < len(order):
        g = [order[k]]
        while k + len(g) < len(order) and len(seqs[order[k + len(g)]]) * (len(g) + 1) <= batch_tokens:
            g.append(order[k + len(g)])
        k += len(g)
        try:
            run_group(g)
        except torch.OutOfMemoryError:                    # long padded batch: one sequence at a time
            torch.cuda.empty_cache()
            if len(g) == 1:
                raise
            for i in g:
                run_group([i])
    return out


def own_items(a, tok, model):
    """The model's own replies, claims labelled: per item {source, ids, start, token_labels, sent, claim}."""
    from datasets import load_dataset
    if a.smoke:
        qa = [{"q": q, "aliases": al} for q, al in [
            ("What is the capital of France?", ["Paris"]), ("Who wrote Hamlet?", ["Shakespeare"]),
            ("What is the largest planet?", ["Jupiter"]), ("Who painted the Mona Lisa?", ["Leonardo da Vinci"]),
            ("Who discovered penicillin?", ["Alexander Fleming", "Fleming"]), ("What is the capital of Peru?", ["Lima"])]]
        gm = [{"q": "Tom has 3 apples and buys 4 more. How many apples does he have?", "answer": "7"}]
        pools = {"train": (qa, gm), "test": (qa, gm)}
    else:
        def tq(split, n):
            d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
            idx = random.Random(0).sample(range(len(d)), n)
            return [{"q": x["question"], "aliases": x["answer"]["normalized_aliases"] + [x["answer"]["value"]]}
                    for x in d.select(idx)]

        def gsm(split):
            return [{"q": x["question"], "answer": x["answer"].split("####")[-1].strip().replace(",", "")}
                    for x in load_dataset("openai/gsm8k", "main", split=split)]
        pools = {"train": (tq("train", 4000), gsm("train")), "test": (tq("validation", 1000), gsm("test"))}
    out = {"train": [], "test": []}
    for split, n in (("train", a.own_train), ("test", a.own_test)):
        qa, gm = pools[split]
        for k, fmt in enumerate(FORMATS):
            its = make_items(fmt, gm if fmt == "math" else qa, n, seed=(0 if split == "train" else 100) + k)
            think = a.thinking and THINK[fmt]
            prompts = [chat_prompt(tok, it["messages"], think) for it in its]
            budget = int(MAX_NEW[fmt] * (a.think_scale if think else 1))
            reps = generate(model, tok, prompts, max_new_tokens=budget, batch=a.gen_batch)
            n_claims = 0
            for it, p, r in zip(its, prompts, reps):
                k0 = after_thinking(tok, r)
                if k0 is None:                            # never finished thinking: no answer to judge
                    continue
                lab = [(j + k0, y) for j, y in label_claims(tok, it, r[k0:])]
                pid = tok(p, add_special_tokens=False).input_ids
                sent = sentence_ids(tok, r)
                starts = word_starts(tok, r) + [len(r)]
                tl, sl, claims = {}, {}, []
                for j, y in lab:                          # y = 1: claim right; risk label = 1 - y
                    end = next((s for s in starts if s > j), len(r))
                    for t in range(j, end):
                        tl[t] = 1 - y
                    sl[sent[j]] = max(sl.get(sent[j], 0), 1 - y)
                    claims.append((j, y))
                n_claims += len(claims)
                out[split].append({"source": f"own_{fmt}", "ids": pid + r, "start": len(pid), "token_labels": tl,
                                   "sent": sent, "sent_labels": sl, "claims": claims,
                                   "reply_label": int(any(1 - y for _, y in claims)) if claims else None})
            log(phase="own", split=split, format=fmt, items=len(its), claims=n_claims, s=round(time.time() - T0))
    return out


def span_items(a, tok):
    """RAGTruth + HaluEval-general responses (token-labelled spans), teacher-forced."""
    caps = {"train": {"ragtruth": a.span_train, "halueval_general": a.span_train // 2},
            "test": {"ragtruth": a.span_test, "halueval_general": a.span_test // 2}}
    if a.smoke:
        caps = {"train": {"ragtruth": 12, "halueval_general": 8}, "test": {"ragtruth": 6, "halueval_general": 4}}
    out = {}
    for split in ("train", "test"):
        out[split] = []
        for it in load_all(split, caps[split]):
            e = encode(tok, it, a.max_tokens)
            if e is None or e["labels"] is None or e["r0"] >= len(e["ids"]):
                continue
            r = e["ids"][e["r0"]:]
            sent = sentence_ids(tok, r)
            tl = dict(enumerate(e["labels"]))
            sl = {}
            for t, y in tl.items():
                sl[sent[t]] = max(sl.get(sent[t], 0), y)
            out[split].append({"source": it["source"], "ids": e["ids"], "start": e["r0"], "token_labels": tl,
                               "sent": sent, "sent_labels": sl, "claims": [], "reply_label": it["label"]})
    return out


def step_items(a, tok):
    """Math-Shepherd: question as the user turn, its steps as the reply; step labels at step ends."""
    from datasets import load_dataset
    n = {"train": a.step_train, "test": a.step_test} if not a.smoke else {"train": 8, "test": 4}
    d = load_dataset("peiyi9979/Math-Shepherd", split="train", streaming=True)
    rows = []
    for x in d:
        if x["task"] != "GSM8K":
            continue
        rows.append(x)
        if len(rows) >= n["train"] + n["test"]:
            break
    out = {"train": [], "test": []}
    for i, x in enumerate(rows):
        split = "train" if i < n["train"] else "test"
        k = x["input"].find("Step 1:")
        if k < 0:
            continue
        q, body, lab = x["input"][:k].strip(), x["input"][k:], x["label"][k:]
        steps, marks = body.split("ки"), re.findall(r"([+-])(?:\n|$)", lab)
        steps = [s for s in steps if s.strip()]
        if not steps or len(marks) < len(steps):
            continue
        head = chat_prompt(tok, [{"role": "user", "content": q}], False)
        pid = tok(head, add_special_tokens=False).input_ids
        r, sent, tl, sl = [], [], {}, {}
        for si, (s, m) in enumerate(zip(steps, marks)):
            t = tok(s.strip() + "\n", add_special_tokens=False).input_ids
            sent += [si] * len(t)
            r += t
            y = int(m == "-")
            tl[len(r) - 1] = y                                 # the step's last token carries its label
            sl[si] = y
        if len(pid) + len(r) > a.max_tokens:
            continue
        out[split].append({"source": "math_steps", "ids": pid + r, "start": len(pid), "token_labels": tl,
                           "sent": sent, "sent_labels": sl, "claims": [], "reply_label": int(any(sl.values()))})
    return out


T0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--out", default="/kaggle/working/risk")
    ap.add_argument("--thinking", action="store_true", help="think on reasoning / math formats (Qwen3 switch)")
    ap.add_argument("--think-scale", type=float, default=2.0, help="token budget multiplier when thinking")
    ap.add_argument("--own-train", type=int, default=300, help="items per format")
    ap.add_argument("--own-test", type=int, default=100)
    ap.add_argument("--span-train", type=int, default=1200)
    ap.add_argument("--span-test", type=int, default=400)
    ap.add_argument("--step-train", type=int, default=1500)
    ap.add_argument("--step-test", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--neg-keep", type=float, default=0.2, help="share of unlabelled-negative tokens kept")
    ap.add_argument("--resume-data", help="data_items.json from an earlier run (bundled files/ searched): skip generation")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.model, a.own_train, a.own_test, a.gen_batch = "Qwen/Qwen2.5-0.5B-Instruct", 3, 2, 4
    os.makedirs(a.out, exist_ok=True)
    import subprocess
    from importlib.metadata import version
    tv = tuple(int(x) for x in version("transformers").split(".")[:2])
    if "qwen3" in a.model.lower() and tv < (4, 51):               # Qwen3 needs transformers >= 4.51
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51,<4.58"], check=False)
    prequant = "bnb-4bit" in a.model.lower()
    if prequant or a.quant != "none":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(phase="env", transformers=version("transformers"), torch=torch.__version__,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(torch_dtype=torch.float16 if cuda else torch.float32, attn_implementation="sdpa")
    if cuda:
        kw["device_map"] = "auto"
    if a.quant != "none" and not prequant:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=a.quant == "8bit", load_in_4bit=a.quant == "4bit",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if cuda and torch.cuda.get_device_capability(0)[0] < 8:          # T4: no bfloat16 matmuls; compute in float16
        for m in model.modules():
            if getattr(m, "compute_dtype", None) == torch.bfloat16:
                m.compute_dtype = torch.float16
    probe_q = [chat_prompt(tok, [{"role": "user", "content": "What is the capital of France? Answer briefly."}], False)]
    sanity = tok.decode(generate(model, tok, probe_q, max_new_tokens=16, batch=1)[0], skip_special_tokens=True)
    log(phase="sanity", reply=sanity, s=round(time.time() - T0))           # garbage here means float16 trouble
    L = model.config.num_hidden_layers
    layers = sorted({int(round(f * (L - 1))) for f in (0.55, 0.7, 0.85)})
    res = {"config": vars(a), "model": a.model, "layers": layers, "complete": False}

    def save():
        json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)

    data = {"train": [], "test": []}
    resume = a.resume_data and (a.resume_data if os.path.exists(a.resume_data)
                                else os.path.join(HERE, "files", os.path.basename(a.resume_data)))
    if resume and os.path.exists(resume):                 # replies generated by an earlier run: skip generation
        raw = json.load(open(resume))
        for split in data:
            for x in raw[split]:
                x["token_labels"] = {int(k): v for k, v in x["token_labels"].items()}
                x["sent_labels"] = {int(k): v for k, v in x["sent_labels"].items()}
                x["claims"] = [tuple(c) for c in x["claims"]]
                data[split].append(x)
        log(phase="resumed", path=resume)
    else:
        for part in (own_items(a, tok, model), span_items(a, tok), step_items(a, tok)):
            for split in data:
                data[split] += part[split]
        json.dump(data, open(os.path.join(a.out, "data_items.json"), "w"))   # kept even if a later step fails
    res["counts"] = {s: {src: sum(x["source"] == src for x in data[s]) for src in sorted({x["source"] for x in data[s]})}
                     for s in data}
    log(phase="data", counts=res["counts"], s=round(time.time() - T0))
    save()
    # one forward pass per item: own-position states at every reply token
    rng = np.random.default_rng(0)
    for x in data["train"]:                       # training keeps only labelled tokens (span negatives sampled)
        x["token_labels"] = {t: y for t, y in x["token_labels"].items()
                             if not (x["source"].startswith(("ragtruth", "halueval")) and y == 0
                                     and rng.random() > a.neg_keep)}
    keep_train = [sorted(set(x["token_labels"]) | {j - 1 for j, _ in x["claims"] if j >= 1}) for x in data["train"]]
    feats = {}
    for split, keep in (("train", keep_train), ("test", None)):
        feats[split] = states(model, [x["ids"] for x in data[split]], [x["start"] for x in data[split]], layers, keep)
        log(phase="states", split=split, items=len(feats[split]), s=round(time.time() - T0))
    # training rows: the labelled tokens
    Xs, ys, srcs = {L_: [] for L_ in layers}, [], []
    for x, (H, lp, pos) in zip(data["train"], feats["train"]):
        at = {t: r for r, t in enumerate(pos)}
        for t, y in x["token_labels"].items():
            if t not in at:
                continue
            for k, L_ in enumerate(layers):
                Xs[L_].append(H[k, at[t]].float().numpy())
            ys.append(y)
            srcs.append(x["source"])
    ys, srcs = np.array(ys, dtype=np.float32), np.array(srcs)
    Xs = {L_: np.stack(v) for L_, v in Xs.items()}
    pdev = "cpu"
    if cuda:
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        pdev = f"cuda:{int(np.argmax(free))}"
    idx = rng.permutation(len(ys))
    cut = int(0.8 * len(idx))
    by_layer = {L_: auroc(Probe(pdev).fit(Xs[L_][idx[:cut]], ys[idx[:cut]]).logit(Xs[L_][idx[cut:]]), ys[idx[cut:]])
                for L_ in layers}
    best = max(layers, key=lambda L_: by_layer[L_] or 0)
    probe = Probe(pdev).fit(Xs[best], ys)
    k_best = layers.index(best)
    lodo = {s: Probe(pdev).fit(Xs[best][srcs != s], ys[srcs != s]) for s in sorted(set(srcs.tolist()))
            if (srcs == s).any() and len(set(ys[srcs != s].tolist())) == 2}
    res.update(risk_layer=best, by_layer=by_layer, train_tokens=int(len(ys)), train_positives=int(ys.sum()))
    log(phase="fitted", risk_layer=best, by_layer=by_layer, tokens=len(ys), s=round(time.time() - T0))
    # claim probe: predicting-position states (the token before each claim) of the own replies
    cX, cY = [], []
    for x, (H, lp, pos) in zip(data["train"], feats["train"]):
        at = {t: r for r, t in enumerate(pos)}
        for j, y in x["claims"]:
            if j - 1 in at:
                cX.append(H[k_best, at[j - 1]].float().numpy())
                cY.append(y)
    claim_probe = Probe(pdev).fit(np.stack(cX), np.array(cY, dtype=np.float32)) if len(set(cY)) == 2 else None
    del Xs

    def scored(x, H, lp, p):                       # test items keep every reply token: H[:, t] is token t
        z = p.logit(H[k_best].float().numpy())
        risk = 1 / (1 + np.exp(-z))
        sents = {}
        for t, sn in enumerate(x["sent"][:len(risk)]):
            sents.setdefault(sn, []).append(t)
        return risk, sents
    # evaluation per source
    R = {}
    test_sources = sorted({x["source"] for x in data["test"]})
    sent_scores_all = []
    for s in test_sources + ["all"]:
        tok_s, tok_y, tok_lp, sen_s, sen_y, sen_lp, rep_s, rep_y, rep_lp, lo_tok, lo_sen = ([] for _ in range(11))
        for x, (H, lp, _) in zip(data["test"], feats["test"]):
            if s != "all" and x["source"] != s:
                continue
            risk, sents = scored(x, H, lp, probe)
            lpn = lp.numpy()
            for t, y in x["token_labels"].items():
                if t < len(risk):
                    tok_s.append(risk[t]), tok_y.append(y), tok_lp.append(-lpn[t])
            for sn, ts in sents.items():
                if sn in x["sent_labels"]:
                    sen_s.append(max(risk[t] for t in ts)), sen_y.append(x["sent_labels"][sn])
                    sen_lp.append(max(-lpn[t] for t in ts))
            if x["reply_label"] is not None and len(risk):
                rep_s.append(float(risk.max())), rep_y.append(x["reply_label"]), rep_lp.append(float(-lpn.min()))
            if s in lodo:
                lr, _ = scored(x, H, lp, lodo[s])
                for t, y in x["token_labels"].items():
                    if t < len(lr):
                        lo_tok.append((lr[t], y))
                for sn, ts in sents.items():
                    if sn in x["sent_labels"]:
                        lo_sen.append((max(lr[t] for t in ts), x["sent_labels"][sn]))
        d = {}
        if len(set(tok_y)) == 2:
            d["token"] = {"n": len(tok_y), "pos": int(sum(tok_y)), "risk_auroc": auroc(tok_s, tok_y),
                          "risk_ci": auroc_ci(tok_s, tok_y, n=200), "low_logprob_auroc": auroc(tok_lp, tok_y)}
        if len(set(sen_y)) == 2:
            d["sentence"] = {"n": len(sen_y), "pos": int(sum(sen_y)), "risk_auroc": auroc(sen_s, sen_y),
                             "risk_ci": auroc_ci(sen_s, sen_y, n=200), "low_logprob_auroc": auroc(sen_lp, sen_y)}
            if s == "all":
                sent_scores_all = list(zip(sen_s, sen_y))
        if len(set(rep_y)) == 2:
            d["reply"] = {"n": len(rep_y), "risk_auroc": auroc(rep_s, rep_y), "low_logprob_auroc": auroc(rep_lp, rep_y)}
        if lo_tok and len({y for _, y in lo_tok}) == 2:
            d["left_out_token_auroc"] = auroc([v for v, _ in lo_tok], [y for _, y in lo_tok])
        if lo_sen and len({y for _, y in lo_sen}) == 2:
            d["left_out_sentence_auroc"] = auroc([v for v, _ in lo_sen], [y for _, y in lo_sen])
        R[s] = d
    # sentence threshold: best F1 on the pooled test sentences (reported, used as the serving default)
    th = 0.5
    if sent_scores_all:
        sc = np.array([v for v, _ in sent_scores_all])
        yy = np.array([y for _, y in sent_scores_all])

        def f1(t):
            p = sc >= t
            tp = (p & (yy == 1)).sum()
            return 2 * tp / max(1, p.sum() + (yy == 1).sum())
        th = float(max(np.quantile(sc, np.linspace(0.05, 0.95, 37)), key=f1))
        res["sentence_threshold"], res["sentence_f1_at_threshold"] = th, float(f1(th))
    res["per_source"], res["complete"], res["seconds"] = R, True, round(time.time() - T0)
    torch.save({"model": a.model, "risk_layer": best, "risk_probe": probe.state_dict(), "risk_threshold": th,
                "claim_layer": best, "claim_probe": claim_probe.state_dict() if claim_probe else None},
               os.path.join(a.out, a.model.split("/")[-1] + "_risk_head.pt"))
    save()
    log(phase="done", per_source={s: {k: v.get("risk_auroc") for k, v in d.items() if isinstance(v, dict)}
                                  for s, d in R.items()}, minutes=round((time.time() - T0) / 60, 1))


if __name__ == "__main__":
    main()
