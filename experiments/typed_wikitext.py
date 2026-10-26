"""Typed generation on real text (WikiText-2): facts only by copying, checked exactly.

Task: given a source (article title + the preceding paragraph), continue the
article with the next paragraph. Fact words of the continuation that do not
occur in the source are replaced by a hole, for both models, so both learn
the same text distribution:

  typed    : vt.typed_lm.TypedLM. Fact words (uppercase / digit / non-ASCII)
             can only be produced by a whole-word pointer into the source.
  baseline : the same GPT body, unconstrained bytes, no pointer.

Reported on held-out articles (valid split):
  bpb             bits per target byte (same strings, so directly comparable)
  fact_top1       teacher-forced: next fact word predicted exactly
  samples         sampled continuations: fraction of fact words not in the
                  source (unsupported), hole rate, exact check pass rate

    python experiments/typed_wikitext.py --model typed --out results/typed_wikitext/typed.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.lm import GPT, GPTConfig  # noqa: E402
from vt import typed_lm as TL  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "wikitext-2")


def articles(path):
    arts, title = [], None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("= ") and s.endswith(" =") and not s.startswith("= ="):
            title = s[2:-2].strip()
            arts.append((title, []))
        elif s.startswith("= ="):
            continue
        elif arts:
            arts[-1][1].append(s)
    return arts


def tail(b: bytes, n):
    if len(b) <= n:
        return b
    cut = b[-n:]
    return cut[cut.find(b" ") + 1:]


def head(b: bytes, n):
    if len(b) <= n:
        return b
    cut = b[:n]
    return cut[:cut.rfind(b" ")]


def examples(arts, common, src_bytes, tgt_bytes):
    out = []
    for title, pars in arts:
        t = TL.normalize(title, common)
        for i in range(1, len(pars)):
            prev = TL.normalize(pars[i - 1], common)
            src = t + b" : " + tail(prev, src_bytes - len(t) - 3)
            if len(src) > src_bytes:
                continue
            tgt = TL.holeify(head(TL.normalize(pars[i], common), tgt_bytes), src)
            out.append((src, tgt))
    return out


def stats(exs):
    n_fact = n_hole = n_bytes = 0
    for s, t in exs:
        ws = t.split(b" ")
        n_hole += sum(w == bytes([TL.HOLE]) for w in ws)
        n_fact += sum(TL.is_fact(w) for w in ws)
        n_bytes += len(t)
    return {"examples": len(exs), "target_bytes": n_bytes,
            "copyable_fact_words": n_fact, "holes": n_hole,
            "copyable_frac": n_fact / max(1, n_fact + n_hole)}


class Baseline(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gpt = GPT(cfg)

    def token_logp(self, x, y, m):
        logits, _ = self.gpt(x)
        return F.log_softmax(logits, -1).gather(-1, y[..., None]).squeeze(-1)

    def loss(self, x, y, m):
        tok = self.token_logp(x, y, m)
        return -(tok * m["region"]).sum() / m["region"].sum()

    @torch.no_grad()
    def generate(self, source, max_bytes=200, temperature=0.8, generator=None):
        self.eval()
        x = list(source) + [TL.SEP]
        out = []
        while len(out) < max_bytes and len(x) + len(out) < self.gpt.config.block_size:
            logits, _ = self.gpt(torch.tensor([x + out]))
            a = int(torch.multinomial((logits[0, -1] / temperature).softmax(-1), 1, generator=generator))
            out.append(a)
            if a == TL.NL:
                break
        return bytes(out).rstrip(b"\n"), []


@torch.no_grad()
def evaluate(model, exs, block, typed, bs=16):
    model.eval()
    nll = nbytes = 0.0
    hit = tot = 0
    for i in range(0, len(exs), bs):
        x, y, m = TL.encode(exs[i:i + bs], block)
        tok = model.token_logp(x, y, m)
        nll -= float((tok * m["region"]).sum())
        nbytes += float(m["region"].sum())
        starts = m["tgt_id"] >= 0
        if typed:
            cand = m["cand_id"] >= 0
            lgen, lptr, lg, l1g = model.heads(x, cand)
            best_gen = (lgen.max(-1).values + l1g)
            j = lptr.argmax(-1)
            best_copy = lg + lptr.max(-1).values
            ok = (best_copy > best_gen) & (m["cand_id"].gather(1, j) == m["tgt_id"])
        else:
            logits, _ = model.gpt(x)
            right = logits.argmax(-1) == y
            ok = torch.zeros_like(starts)
            for b, t in starts.nonzero().tolist():
                e = t
                while e + 1 < y.shape[1] and m["inner"][b, e + 1]:
                    e += 1
                ok[b, t] = bool(right[b, t:e + 1].all())
        hit += int((ok & starts).sum())
        tot += int(starts.sum())
    return {"bpb": nll / nbytes / math.log(2), "fact_top1": hit / max(1, tot), "fact_positions": tot}


def sample_eval(model, exs, n, seed):
    g = torch.Generator().manual_seed(seed)
    unsup = facts = holes = words = ok = 0
    shown = []
    for k, (src, tgt) in enumerate(exs[:n]):
        out, tr = model.generate(src, generator=g)
        c = TL.check(src, out, tr)
        have = TL.fact_words(src)
        ws = out.split()
        facts += sum(TL.is_fact(w) for w in ws)
        unsup += sum(TL.is_fact(w) and w not in have for w in ws)
        holes += c["holes"]
        words += len(ws)
        ok += c["ok"]
        if k < 6:
            shown.append({"source": TL.render(src), "output": TL.render(out),
                          "reference": TL.render(tgt), "check_ok": c["ok"],
                          "unsupported": c["unsupported"][:10]})
    return {"n": n, "fact_words_per_100_words": 100 * facts / max(1, words),
            "unsupported_fact_frac": unsup / max(1, facts),
            "holes_per_100_words": 100 * holes / max(1, words),
            "check_pass": ok / n, "examples": shown}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["typed", "baseline"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--embd", type=int, default=192)
    ap.add_argument("--src", type=int, default=320)
    ap.add_argument("--tgt", type=int, default=190)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    block = a.src + a.tgt + 2
    train_arts = articles(os.path.join(DATA, "train.txt"))
    common = TL.common_lowercase(p for _, ps in train_arts for p in ps)
    tr = examples(train_arts, common, a.src, a.tgt)
    va = examples(articles(os.path.join(DATA, "valid.txt")), common, a.src, a.tgt)
    random.Random(1).shuffle(va)
    va = va[:a.n_eval]
    print(json.dumps({"train": stats(tr), "valid": stats(va)}), flush=True)
    cfg = GPTConfig(block_size=block, n_layer=a.layers, n_head=4, n_embd=a.embd, pos_encoding="alibi")
    model = TL.TypedLM(cfg) if a.model == "typed" else Baseline(cfg)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ckpt = a.out.replace(".json", ".pt")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    step = 0
    if os.path.exists(ckpt):
        st = torch.load(ckpt)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
    rng = random.Random(a.seed + step)
    t0 = time.time()
    while step < a.steps:
        model.train()
        lr = a.lr * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for gr in opt.param_groups:
            gr["lr"] = lr
        x, y, m = TL.encode(rng.sample(tr, a.batch), block)
        loss = model.loss(x, y, m)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
        if step % 250 == 0 or step == a.steps:
            print(json.dumps({"step": step, "loss_bits": round(loss.item() / math.log(2), 4),
                              "s_per_step": round((time.time() - t0) / 250, 3)}), flush=True)
            t0 = time.time()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt)
    res = {"model": a.model, "params": sum(p.numel() for p in model.parameters()), "args": vars(a),
           "data": {"train": stats(tr), "valid": stats(va)},
           "valid": evaluate(model, va, block, a.model == "typed")}
    print(json.dumps(res["valid"]), flush=True)
    res["samples"] = sample_eval(model, va, a.n_samples, a.seed)
    print(json.dumps({k: v for k, v in res["samples"].items() if k != "examples"}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
