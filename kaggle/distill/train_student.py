"""Kernel 2 of the glass-box distillation: train students from scratch on the teacher's data.

Two students on the same data and token budget:
  glassbox : vt.glassbox2.Glassbox2: exactly sparse feature layers (the features
             are the neurons), exact attribution by construction, an internal
             memory layer behind a hard gate (exactly 0 when unused, so such
             tokens provably used no stored knowledge) whose slots record the
             training document that wrote them, and a per-token self-knowledge
             variable k that sets the sharpness of each next-token distribution;
  baseline : vt.lm.GPT with about the same parameters outside the memory (ALiBi).
Data: the teacher's replies (kernel 1) in chat form, plus WikiText-103.

Evaluation on held-out data: bits per token and top-1 agreement with the
teacher; per-token "my top prediction is wrong" AUROC from softmax confidence,
k and a post-hoc probe; the share of tokens and replies certified memory-free
(with a soundness check); TriviaQA and made-up-name questions (accuracy,
hallucination, abstention). The glass-box student is saved as a package that
`vt chat-serve` serves directly.

    python train_student.py --data /kaggle/input/glassbox-distill-data --out /kaggle/working/student
    python train_student.py --smoke --data /tmp/d --out /tmp/s
"""
import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import string
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vtpkg"))                    # Kaggle: bundled modules
sys.path.insert(0, os.path.join(HERE, "..", ".."))                 # repository checkout
from vt.glassbox2 import G2Config, Glassbox2  # noqa: E402
from vt.glassbox3 import G3Config, Glassbox3  # noqa: E402
from vt.lm import GPT, GPTConfig  # noqa: E402

ABSTAIN = ("don't know", "not sure", "no information", "do not know")
UNSURE = 1                                   # the tokenizer's <unsure> token (kernel 1)
GLASS = ("glassbox", "glassbox3")
HEDGE = "I'm not sure. I don't know the answer to that."


def log(**kw):
    print(json.dumps(kw), flush=True)


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def chat_text(prompt, reply):
    return f"User: {prompt}\nAssistant: {reply}"


# ---------------------------------------------------------------- data

def find_data(root):
    """Kernel outputs are mounted somewhere under /kaggle/input: find the directory with train.jsonl."""
    for dp, _, fs in os.walk(root):
        if "train.jsonl" in fs and "tokenizer.json" in fs:
            return dp
    raise SystemExit(f"no train.jsonl + tokenizer.json under {root}")


def load(a, mark_unsure=False):
    from tokenizers import Tokenizer
    a.data = find_data(a.data)
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    train = [json.loads(l) for l in open(os.path.join(a.data, "train.jsonl"))]
    ev = [json.loads(l) for l in open(os.path.join(a.data, "eval.jsonl"))]
    chat_docs = [(f"User: {x['prompt']}\nAssistant:", " " + x["response"]) for x in train]
    if a.smoke:
        wiki = ["The cat sat on the mat. It was a sunny day in the town."] * 50
    elif a.wiki_file:                                           # local text, one paragraph per line
        wiki = [ln.strip() for ln in open(a.wiki_file) if len(ln.strip()) > 80]
    else:
        from datasets import load_dataset
        w = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        wiki = [x["text"] for x in w if len(x["text"]) > 80]
    def stream(docs, offset, chunk=20000):
        """docs: texts, or (prompt, reply) pairs; with mark_unsure (Glassbox-3 students only),
        <unsure> goes before every abstaining reply. Tokenized in chunks into int32 arrays:
        holding every Encoding object or Python-list token at once runs out of memory."""
        pairs = isinstance(docs[0], tuple)
        parts = []
        for c0 in range(0, len(docs), chunk):
            sub = docs[c0:c0 + chunk]
            flat = [t for d in sub for t in d] if pairs else sub
            encs = tok.encode_batch(flat)
            for n in range(len(sub)):
                if pairs:
                    p, r = encs[2 * n].ids, encs[2 * n + 1].ids
                    mark = [UNSURE] if mark_unsure and any(mk in sub[n][1].lower() for mk in ABSTAIN) else []
                    one = p + mark + r
                else:
                    one = encs[n].ids
                parts.append(np.asarray(one + [0], dtype=np.int32))
            del encs
        lengths = np.fromiter((len(x) for x in parts), dtype=np.int64, count=len(parts))
        ids = torch.from_numpy(np.concatenate(parts))
        doc = torch.from_numpy(np.repeat(np.arange(offset, offset + len(parts), dtype=np.int32), lengths))
        return ids, doc
    cache = os.path.join(a.out, f"streams{'_unsure' if mark_unsure else ''}.pt")
    if os.path.exists(cache):                                  # built once by the parent process
        chat, wikis = torch.load(cache)
    else:
        chat, wikis = stream(chat_docs, 0), stream(wiki, 10_000_000)    # wiki documents: ids >= 1e7
        torch.save((chat, wikis), cache)
    sc_prompts = [x["prompt"] for x in train if x["kind"] == "trivia"]
    return tok, chat, wikis, ev, sc_prompts


def batch(chat, wiki, bs, block, g, p_chat=0.6):
    """Random windows; also the training-document id of every input token."""
    xs, ys, ds = [], [], []
    for _ in range(bs):
        src, doc = chat if torch.rand(1, generator=g).item() < p_chat else wiki
        i = int(torch.randint(len(src) - block - 1, (1,), generator=g))
        xs.append(src[i:i + block]); ys.append(src[i + 1:i + block + 1]); ds.append(doc[i:i + block])
    return torch.stack(xs).long(), torch.stack(ys).long(), torch.stack(ds).long()


# ---------------------------------------------------------------- models

def make(kind, a, V):
    if kind in GLASS:
        cfg = dict(vocab_size=V, block_size=a.block, n_layer=a.layers, n_head=a.heads, n_embd=a.embd,
                   n_features=a.n_features, k_active=a.k_active, mem_layer=a.layers - 1, mem_keys=a.mem_keys,
                   mem_topk=min(16, a.mem_keys), mix=a.mix, feat_recon=a.feat_recon > 0)
        if kind == "glassbox3":
            cfg.update(unsure_id=UNSURE)
            return Glassbox3(G3Config(**cfg)), cfg
        return Glassbox2(G2Config(**cfg)), cfg
    cfg = dict(vocab_size=V, block_size=a.block, n_layer=a.layers, n_head=a.heads, n_embd=a.embd,
               mlp_ratio=a.base_mlp, pos_encoding="alibi")
    return GPT(GPTConfig(**cfg)), cfg


def forward(kind, m, x):
    """logits, k (or None), hidden states per block."""
    if kind in GLASS:
        logits, k = m(x)
        return logits, k, m.trace["hs"]
    hs = []
    hooks = [b.register_forward_hook(lambda mod, i, o: hs.append(o)) for b in m.blocks]
    logits, _ = m(x)
    for h in hooks:
        h.remove()
    return logits, None, hs


def same(a, b):
    """Two normalized replies assert the same thing: one contains the other, or their words
    overlap with F1 >= 0.5 (short factual replies worded differently)."""
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    wa, wb = a.split(), b.split()
    common = sum(min(wa.count(w), wb.count(w)) for w in set(wa))
    return common > 0 and 2 * common / (len(wa) + len(wb)) >= 0.5


@torch.no_grad()
def sample_replies(m, tok, prompt, device, block, k, max_new=24):
    """k sampled replies to one prompt (normalized text; "<unsure>" if the reply starts with it)."""
    ids = tok.encode(f"User: {prompt}\nAssistant:").ids[-(block - max_new):]
    x = torch.tensor([ids] * k, device=device)
    outs, done = [[] for _ in range(k)], [False] * k
    for _ in range(max_new):
        lg = m(x)[0][:, -1].float()
        nxt = torch.multinomial(F.softmax(lg, -1), 1)
        for j in range(k):
            t = int(nxt[j])
            if done[j]:
                continue
            if t in (0, UNSURE) or "\n" in tok.decode(outs[j] + [t]):
                done[j] = True
                if t == UNSURE and not outs[j]:
                    outs[j] = None
                continue
            outs[j].append(t)
        if all(done):
            break
        x = torch.cat([x, nxt], 1)
    return ["<unsure>" if o is None else norm(tok.decode(o)) for o in outs]


def self_consistency_batch(m, tok, prompts, rng, a, device):
    """Self-consistency replay: prompts whose sampled replies disagree (largest group of
    agreeing replies below a.sc_agree of them) become training examples prompt + <unsure> + hedge.
    No label says which prompts those are: the model's own disagreement does."""
    m.eval()
    seqs, P = [], []
    for p in rng.sample(prompts, a.sc_prompts):
        outs = sample_replies(m, tok, p, device, a.block, a.sc_k)
        top = max(sum(same(o, q) for q in outs) for o in outs)
        if top / len(outs) < a.sc_agree:
            ids = tok.encode(f"User: {p}\nAssistant:").ids
            seqs.append(ids + [UNSURE] + tok.encode(" " + HEDGE).ids + [0])
            P.append(len(ids))
    m.train()
    if not seqs:
        return None, 0.0
    L = max(map(len, seqs))
    x = torch.zeros(len(seqs), L - 1, dtype=torch.long)
    y = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
    for i, (sq, p) in enumerate(zip(seqs, P)):
        x[i, :len(sq) - 1] = torch.tensor(sq[:-1])
        y[i, p - 1:len(sq) - 1] = torch.tensor(sq[p:])          # loss on the reply only
    return (x.to(device), y.to(device)), len(seqs) / a.sc_prompts


@torch.no_grad()
def val_loss(kind, m, tok, items, device, block):
    """Mean next-token loss (nats) on held-out teacher replies, as the evaluation scores them."""
    m.eval()
    tot, n = 0.0, 0
    for it in items:
        p = tok.encode(f"User: {it['prompt']}\nAssistant:").ids
        r = tok.encode(" " + it["response"]).ids
        ids = (p + r + [0])[:block + 1]
        P = min(len(p), block)
        x, y = torch.tensor([ids[:-1]], device=device), torch.tensor(ids[1:], device=device)
        logits = forward(kind, m, x)[0][0].float()
        tot += float(F.cross_entropy(logits[P - 1:], y[P - 1:], reduction="sum"))
        n += len(ids) - P
    m.train()
    return tot / max(1, n)


def train(kind, a, V, chat, wiki, device, log_fn, tok=None, sc_prompts=(), val_items=()):
    torch.manual_seed(0)
    m, cfg = make(kind, a, V)
    m.to(device)
    params = [p for p in m.parameters() if p.requires_grad]
    log_fn(phase="model", kind=kind, params=sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    budget = a.hours * 3600
    step, total = 0, None
    sc_rng, sc_rate = random.Random(3), None
    best = (float("inf"), None, 0)                  # held-out loss, weights, step: the teacher data is
    while True:                                     # small, so later steps overfit it
        el = time.time() - t0
        if total is None and step == 50:                              # fix the schedule from measured speed
            total = max(100, int(budget / (el / 50)))
            log_fn(phase="schedule", kind=kind, total_steps=total, s_per_step=round(el / 50, 3))
            mark = (step, el)
        elif total is not None and step - mark[0] == 500:            # re-estimate (replay costs time)
            rate = (el - mark[1]) / 500
            total = max(step + 1, step + int((budget - el) / rate))
            mark = (step, el)
        if (total is not None and step >= total) or el > budget * 1.05:
            break
        T = total or 10 ** 9
        lr = a.lr * min(1, (step + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / T))))
        for gr in opt.param_groups:
            gr["lr"] = lr
        m.train()
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):                                      # micro-batches: same effective batch
            x, y, docs = batch(chat, wiki, a.batch // a.accum, a.block, g)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits, k, _ = forward(kind, m, x)
                lm = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
                loss = lm
                info = {}
                if kind in GLASS:
                    right = (logits.argmax(-1) == y).float()
                    aux = F.binary_cross_entropy_with_logits(k.float(), right)
                    gate = m.trace["mem"]["gate"].float()
                    if a.temp_train == "decoupled" and kind == "glassbox":
                        # LM loss on the raw logits, as a GPT's; the k-set temperature learns separately
                        # as calibration of the (detached) raw logits. Coupled training lets a model that
                        # is still mostly wrong flatten its own logits and stall (the first GPU run).
                        raw = m.trace["raw_logits"].float()
                        lm = F.cross_entropy(raw.flatten(0, 1), y.flatten())
                        cal = F.cross_entropy((raw.detach() * m.temperature(k.float())[..., None]).flatten(0, 1),
                                              y.flatten())
                        loss = lm + cal + 0.2 * aux + a.mem_l1 * gate.mean()
                    else:
                        loss = lm + 0.2 * aux + a.mem_l1 * gate.mean()
                    if a.feat_recon > 0:                  # features must also be a sparse code of their input
                        rec = sum(f.recon_loss(d) for f, d in zip(m.feat, m.trace["feat"])) / len(m.feat)
                        loss = loss + a.feat_recon * rec.float()
                        info["feat_recon"] = round(rec.item(), 4)
                    info.update({"k_bce": round(aux.item(), 4), "gate_open": round((gate > 0).float().mean().item(), 4),
                                 "temp": round(m.temperature(k.float()).mean().item(), 4)})
                    m.record_writers(docs.to(device))         # which training document wrote each memory slot
                if kind == "glassbox3":
                    m.record_features()
                    epi = m.epi_loss(y)
                    loss = loss + epi
                    info.update(epi=round(epi.item(), 4), u=round(m.trace["u"].float().mean().item(), 4))
                    if (total and step >= a.sc_start * total and step % a.sc_every == 0 and sc_prompts):
                        xy, sc_rate = self_consistency_batch(m, tok, sc_prompts, sc_rng, a, device)
                        if xy is not None:
                            lg, _ = m(xy[0])
                            loss = loss + F.cross_entropy(lg.float().flatten(0, 1), xy[1].flatten(), ignore_index=-100)
                    if sc_rate is not None:
                        info["sc_disagree"] = round(sc_rate, 3)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}")
            scaler.scale(loss / a.accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        if step % 200 == 0:
            log_fn(phase="train", kind=kind, step=step, total=total, lm=round(lm.item(), 4),
                   s=round(time.time() - t0), **info)
        if val_items and step % a.val_every == 0:
            vl = val_loss(kind, m, tok, val_items, device, a.block)
            if vl < best[0]:
                best = (vl, {k_: v.detach().to("cpu", copy=True) for k_, v in m.state_dict().items()}, step)
            log_fn(phase="val", kind=kind, step=step, val_nats=round(vl, 4), best_step=best[2])
    if best[1] is not None:
        m.load_state_dict(best[1])
        log_fn(phase="best", kind=kind, step=best[2], val_nats=round(best[0], 4), last_step=step)
    m.eval()
    return m, cfg, best[2] or step


# ---------------------------------------------------------------- evaluation

def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if labels.all() or (~labels).all():
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - labels.sum() * (labels.sum() + 1) / 2) / (labels.sum() * (~labels).sum()))


def fit_probe(X, y, steps=300):
    X, y = torch.tensor(np.stack(X), dtype=torch.float), torch.tensor(y, dtype=torch.float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(((X - mu) / sd) @ w + b, y) + 1e-3 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda Z: (((torch.tensor(np.stack(Z), dtype=torch.float) - mu) / sd) @ w + b).detach().numpy()


@torch.no_grad()
def token_eval(kind, m, tok, items, device, block):
    rows = []
    for it in items:
        p = tok.encode(f"User: {it['prompt']}\nAssistant:").ids
        r = tok.encode(" " + it["response"]).ids
        ids = (p + r + [0])[:block + 1]
        P = min(len(p), block)
        x, y = torch.tensor([ids[:-1]], device=device), torch.tensor(ids[1:], device=device)
        logits, k, hs = forward(kind, m, x)
        lp = F.log_softmax(logits[0].float(), -1)
        free = m.memory_free()[0] if kind in GLASS else None
        u = m.trace["u"][0] if kind == "glassbox3" else None
        for t in range(P - 1, len(ids) - 1):
            rows.append({"ok": int(lp[t].argmax() == y[t]), "nll": -float(lp[t, y[t]]),
                         "msp": float(lp[t].max().exp()), "k": float(k[0, t]) if k is not None else None,
                         "free": bool(free[t]) if free is not None else None,
                         "u": float(u[t]) if u is not None else None,
                         "h": [hs[j][0, t].float().cpu().numpy() for j in (len(hs) // 2, len(hs) - 1)]})
    return rows


def token_metrics(rows):
    half = len(rows) // 2
    fit, test = rows[:half], rows[half:]
    y = np.array([r["ok"] for r in test])
    d = {"tokens": len(test), "top1_agreement": float(y.mean()),
         "bits_per_token": float(np.mean([r["nll"] for r in test]) / math.log(2)),
         "auroc_softmax": auroc([r["msp"] for r in test], y)}
    for li, name in enumerate(("middle", "last")):
        pr = fit_probe([r["h"][li] for r in fit], [r["ok"] for r in fit])
        d[f"auroc_probe_{name}"] = auroc(pr([r["h"][li] for r in test]), y)
    if test[0]["k"] is not None:
        d["auroc_k"] = auroc([r["k"] for r in test], y)
    if test[0]["u"] is not None:
        d["auroc_neg_u"] = auroc([-r["u"] for r in test], y)
    if test[0]["free"] is not None:
        free = np.array([r["free"] for r in test])
        d["certified_memory_free_tokens"] = float(free.mean())
        d["top1_when_certified"] = float(y[free].mean()) if free.any() else None
    return d


@torch.no_grad()
def reply(kind, m, tok, prompt, device, block, max_new=48):
    ids = tok.encode(f"User: {prompt}\nAssistant:").ids[-(block - max_new):]
    P, out = len(ids), []
    for i in range(max_new):
        logits, _, _ = forward(kind, m, torch.tensor([ids + out], device=device))
        lg = logits[0, -1].float()
        if i == 0:
            lg[0] = -float("inf")
        nxt = int(lg.argmax())
        if nxt == 0 or "\n" in tok.decode(out + [nxt]):
            break
        out.append(nxt)
    free, sig = None, {}
    if kind in GLASS and out:
        x = torch.tensor([ids + out], device=device)
        m(x)
        free = bool(m.memory_free()[0, P - 1:P - 1 + len(out)].all())
        if kind == "glassbox3":                      # epistemic signals where the reply starts
            sig = {"u_first": float(m.trace["u"][0, P - 1]), "novelty_first": float(m.novelty()[0, P - 1]),
                   "unsure": UNSURE in out}
    text = tok.decode(out).strip()                   # <unsure> is a special token: not decoded
    reply.signals = sig
    return ("<unsure> " + text if sig.get("unsure") else text), free, (ids + out), P


@torch.no_grad()
def soundness(m, seqs, device):
    """Certified positions must be bit-identical when the memory content is replaced by noise."""
    bad = 0
    saved = m.mem.values.weight.detach().clone()
    for s in seqs:
        x = torch.tensor([s], device=device)
        a_, _ = m(x)
        free = m.memory_free()
        m.mem.values.weight.copy_(torch.randn_like(saved) * 10)
        b_, _ = m(x)
        m.mem.values.weight.copy_(saved)
        bad += int(not torch.equal(a_[free], b_[free]))
    return bad


@torch.no_grad()
def reply_scores(kind, m, tok, prompt, seq, P, device, n_samples=5):
    """Truth signals for one reply: mean token probability, min k (glass-box) and self-consistency
    (share of sampled replies that agree with the greedy one)."""
    x = torch.tensor([seq], device=device)
    logits, k, _ = forward(kind, m, x)
    lp = F.log_softmax(logits[0].float(), -1)
    T = len(seq)
    idx = torch.arange(P - 1, T - 1, device=device)
    if len(idx) == 0:
        return {"msp": 0.0, "k_min": None, "consistency": 0.0}
    tgt = torch.tensor(seq[P:], device=device)
    d = {"msp": float(lp[idx, tgt].exp().mean()),
         "k_min": float(torch.sigmoid(k[0, idx].float()).min()) if k is not None else None}
    greedy = norm(tok.decode(seq[P:]))
    agree = 0
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(n_samples):
        ids = list(seq[:P])
        out = []
        for _ in range(32):
            lg = forward(kind, m, torch.tensor([ids + out], device=device))[0][0, -1].float()
            nxt = int(torch.multinomial(F.softmax(lg / 0.8, -1), 1, generator=g))
            if nxt == 0 or "\n" in tok.decode(out + [nxt]):
                break
            out.append(nxt)
        sampled = norm(tok.decode(out))
        agree += int(bool(sampled) and bool(greedy) and (sampled in greedy or greedy in sampled))
    d["consistency"] = agree / n_samples
    return d


def release_guarantee(cal, test, score, alpha=0.2, delta=0.05):
    """Choose the lowest score threshold whose calibration error rate among released answers has a
    one-sided Hoeffding upper bound <= alpha (each candidate threshold tested at level delta / n,
    a Bonferroni correction over thresholds). On fresh questions from the same distribution, the
    released answers' error rate is then <= alpha with probability >= 1 - delta.
    Reports the threshold and what happens on the held-out test half."""
    cands = sorted({r[score] for r in cal if r[score] is not None})
    chosen = None
    for th in cands:
        rel = [r for r in cal if r[score] is not None and r[score] >= th]
        n = len(rel)
        if n == 0:
            break
        err = 1 - np.mean([r["right"] for r in rel])
        ucb = err + math.sqrt(math.log(len(cands) / delta) / (2 * n))
        if ucb <= alpha:
            chosen = th
            break
    if chosen is None:
        return {"alpha": alpha, "threshold": None, "released_fraction": 0.0, "test_error_when_released": None}
    rel = [r for r in test if r[score] is not None and r[score] >= chosen]
    return {"alpha": alpha, "threshold": chosen, "released_fraction": len(rel) / max(1, len(test)),
            "test_error_when_released": float(1 - np.mean([r["right"] for r in rel])) if rel else None}


def qa_eval(kind, m, tok, items, device, block):
    out = {}
    cert_seqs = []
    for name in ("trivia_eval", "fictitious_absent_eval", "fictitious_ctx_eval"):
        rows = []
        for it in [x for x in items if x["kind"] == name]:
            text, free, seq, P = reply(kind, m, tok, it["prompt"], device, block)
            abst = any(mk in text.lower() for mk in ABSTAIN + ("<unsure>",))
            sig = dict(reply.signals) if kind == "glassbox3" else {}
            if name == "trivia_eval":
                right = (not abst) and any(norm(al) and norm(al) in norm(text) for al in it["aliases"])
            elif name == "fictitious_absent_eval":
                right = abst
            else:
                right = (not abst) and it["gold"] in text
            row = {"right": right, "abst": abst, "free": free, "q": it["prompt"], "a": text, **sig}
            if name == "trivia_eval":
                sc = reply_scores(kind, m, tok, it["prompt"], seq, P, device) if not abst else {}
                row.update({k_: sc.get(k_) for k_ in ("msp", "k_min", "consistency")})
            rows.append(row)
            if free:
                cert_seqs.append(seq)
        if not rows:
            continue
        d = {"n": len(rows), "accuracy": float(np.mean([r["right"] for r in rows])),
             "abstention": float(np.mean([r["abst"] for r in rows]))}
        if name != "fictitious_ctx_eval":
            d["hallucination"] = float(np.mean([(not r["abst"]) and not r["right"] for r in rows]))
        if kind == "glassbox3" and rows[0].get("u_first") is not None:
            d["unsure_emitted"] = float(np.mean([bool(r.get("unsure")) for r in rows]))
            d["u_first_mean"] = float(np.mean([r.get("u_first", 0.0) for r in rows]))
            if name == "trivia_eval":                    # does low u mean a right reply (abstentions included)?
                d["auroc_right_neg_u_first"] = auroc([-r.get("u_first", 0.0) for r in rows], [r["right"] for r in rows])
                d["auroc_right_neg_novelty"] = auroc([-r.get("novelty_first", 0.0) for r in rows],
                                                     [r["right"] for r in rows])
        if kind in GLASS:
            d["certified_memory_free_replies"] = float(np.mean([bool(r["free"]) for r in rows]))
            c = [r for r in rows if r["free"]]
            d["accuracy_when_certified"] = float(np.mean([r["right"] for r in c])) if c else None
        if name == "trivia_eval":                          # how well each signal measures truth
            answered = [r for r in rows if not r["abst"]]
            y = [r["right"] for r in answered]
            for sc in ("msp", "k_min", "consistency"):
                if answered and answered[0][sc] is not None:
                    d[f"auroc_true_{sc}"] = auroc([r[sc] for r in answered], y)
                    half = len(rows) // 2
                    d[f"release_{sc}"] = [release_guarantee(rows[:half], rows[half:], sc, alpha=al)
                                          for al in (0.2, 0.5)]
        d["samples"] = [{"q": r["q"], "a": r["a"], "right": r["right"], "certified": r["free"]} for r in rows[:6]]
        out[name] = d
    if kind in GLASS:
        out["certificate_soundness_violations"] = soundness(m, cert_seqs[:30], device)
    return out


@torch.no_grad()
def inspect(m, tok, items, device):
    """On the trained model: attribution sums exactly to the logits, features are exactly
    sparse, and the share of memory slots that have a recorded training-document writer."""
    err, sparse_ok = 0.0, True
    for it in items:
        ids = tok.encode(f"User: {it['prompt']}\nAssistant: {it['response']}").ids[:48]
        x = torch.tensor([ids], device=device)
        m(x)
        for t in (len(ids) // 2, len(ids) - 1):
            tok_id = int(m.trace["raw_logits"][0, t].argmax())
            parts = m.explain(x, t, tok_id)
            err = max(err, abs(sum(parts.values()) - float(m.trace["raw_logits"][0, t, tok_id])))
        sparse_ok &= all(int((d["acts"] != 0).sum(-1).max()) <= m.config.k_active for d in m.trace["feat"])
    return {"attribution_max_abs_error": err, "features_exactly_sparse": sparse_ok,
            "memory_slots_with_writer": float((m.writer_doc >= 0).float().mean())}


def save_package(m, cfg, tok_path, out):
    """vt-glassbox-chat/1 package: weights.pt, config.json, tokenizer.json, manifest.json."""
    os.makedirs(out, exist_ok=True)
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, buf)
    blobs = {"weights.pt": buf.getvalue(), "config.json": json.dumps(cfg, indent=1).encode(),
             "tokenizer.json": open(tok_path, "rb").read()}
    sha = lambda b: hashlib.sha256(b).hexdigest()                     # noqa: E731
    for n, b in blobs.items():
        open(os.path.join(out, n), "wb").write(b)
    files = {n: sha(b) for n, b in blobs.items()}
    fmt = "vt-glassbox3-chat/1" if isinstance(m, Glassbox3) else "vt-glassbox2-chat/1"
    json.dump({"format": fmt, "files": files,
               "package_sha256": sha("".join(files[n] for n in ("weights.pt", "config.json", "tokenizer.json")).encode())},
              open(os.path.join(out, "manifest.json"), "w"), indent=1)


def run(kind, a, device):
    tok, chat, wiki, ev, sc_prompts = load(a, mark_unsure=(kind == "glassbox3"))
    V = tok.get_vocab_size()
    log(phase="data", kind=kind, chat_tokens=len(chat[0]), wiki_tokens=len(wiki[0]), device=device,
        unsure_marked=int((chat[0] == UNSURE).sum()), sc_prompts=len(sc_prompts))
    val_items = [x for x in ev if x["kind"] == "instruction"][-a.n_val:] if a.n_val else []
    m, cfg, steps = train(kind, a, V, chat, wiki, device, log, tok, sc_prompts, val_items)
    gen_eval = [x for x in ev if x["kind"] == "instruction"][:a.n_token_eval]   # disjoint from val_items
    res = {"kind": kind, "config": cfg, "steps": steps,
           "params": sum(p.numel() for p in m.parameters() if p.requires_grad),
           "tokens": token_metrics(token_eval(kind, m, tok, gen_eval, device, a.block))}
    log(phase="eval_tokens", kind=kind, **res["tokens"])
    if kind in GLASS:                                    # inspectability checks on the trained model
        res["inspect"] = inspect(m, tok, gen_eval[:5], device)
        log(phase="inspect", **res["inspect"])
    res["qa"] = qa_eval(kind, m, tok, ev, device, a.block)
    log(phase="eval_qa", kind=kind, **{k: ({kk: vv for kk, vv in v.items() if kk != "samples"}
                                            if isinstance(v, dict) else v) for k, v in res["qa"].items()})
    res["samples"] = [{"prompt": x["prompt"], "reply": reply(kind, m, tok, x["prompt"], device, a.block)[0]}
                      for x in gen_eval[:8]]
    json.dump(res, open(os.path.join(a.out, f"{kind}.json"), "w"), indent=1, default=float)
    if kind in GLASS:
        save_package(m, cfg, os.path.join(a.data, "tokenizer.json"), os.path.join(a.out, "glassbox_package"))
    log(phase="done", kind=kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/glassbox-distill-data")
    ap.add_argument("--out", default="/kaggle/working/student")
    ap.add_argument("--models", default="glassbox,baseline")
    ap.add_argument("--hours", type=float, default=3.5, help="training time budget per student")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--embd", type=int, default=640)
    ap.add_argument("--n-features", type=int, default=4096)
    ap.add_argument("--k-active", type=int, default=32)
    ap.add_argument("--base-mlp", type=int, default=6)
    ap.add_argument("--mem-keys", type=int, default=256)
    ap.add_argument("--mem-l1", type=float, default=0.002, help="penalty on memory gate use")
    ap.add_argument("--feat-recon", type=float, default=0.0,
                    help="weight of the in-model feature reconstruction (SAE) objective (Glassbox-2.4: 1.0)")
    ap.add_argument("--mix", default="add", choices=["add", "convex"],
                    help="residual updates: standard sums (default; pilot 2.3) or convex mixing")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per step (batch is split across them)")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--n-token-eval", type=int, default=300)
    ap.add_argument("--sc-start", type=float, default=0.3, help="Glassbox-3: replay from this share of training")
    ap.add_argument("--sc-every", type=int, default=100)
    ap.add_argument("--sc-prompts", type=int, default=16)
    ap.add_argument("--sc-k", type=int, default=5)
    ap.add_argument("--sc-agree", type=float, default=0.6)
    ap.add_argument("--val-every", type=int, default=500, help="steps between held-out checks (best kept)")
    ap.add_argument("--n-val", type=int, default=200, help="held-out replies for checkpoint selection")
    ap.add_argument("--temp-train", default="decoupled", choices=["decoupled", "coupled"],
                    help="Glassbox-2: train the k temperature as calibration (decoupled) or through the LM loss")
    ap.add_argument("--wiki-file", default=None, help="local text instead of WikiText-103 (CPU reproductions)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.hours, a.layers, a.heads, a.embd, a.mem_keys, a.block, a.batch, a.n_token_eval = 0.01, 2, 2, 32, 8, 64, 4, 4
        a.val_every = 300
        a.n_features, a.k_active = 64, 4
    os.makedirs(a.out, exist_ok=True)
    kinds = a.models.split(",")
    for mark in sorted({k == "glassbox3" for k in kinds}):      # tokenize once, before the students start
        load(argparse.Namespace(**vars(a)), mark_unsure=mark)
    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2 and len(kinds) == 2:                               # one student per GPU, in parallel
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=run, args=(k, a, f"cuda:{i}")) for i, k in enumerate(kinds)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        if any(p.exitcode for p in ps):
            raise SystemExit("a student failed")
    else:
        if n_gpu == 1 and len(kinds) == 2:
            a.hours /= 2
        for k in kinds:
            run(k, a, "cuda" if n_gpu else "cpu")


if __name__ == "__main__":
    main()
