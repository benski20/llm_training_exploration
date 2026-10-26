"""Provenance by architecture: facts can only be copied, and every copy is checked.

Same grounded-answering task as grounding_by_design.py. The model is a GPT
with a pointer head. Each output token's distribution is

    p(v) = (1 - g) * p_vocab(v) + g * sum_j a_j [x_j == v]       (copy from context)

and the fact-bearing tokens (digits) are masked out of p_vocab. So the model
cannot invent a year: digits come only from copying context positions, and
the argmax source position of each digit is recorded. A deterministic
provenance check then confirms the copied span is the year in a fact about
the entity that was asked about. If it isn't, the answer is withheld.
Released years are therefore correct relative to the context by construction;
what remains to measure is coverage, i.e. how many correct answers survive.

    python experiments/provenance_by_design.py --out results/provenance_by_design.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import grounding_by_design as G  # noqa: E402
from vt.lm import GPT, GPTConfig  # noqa: E402

DIGITS = torch.tensor([ord(c) for c in "0123456789"])


class PointerGPT(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.gpt = GPT(GPTConfig(block_size=block, n_layer=4, n_head=4, n_embd=128, pos_encoding=G.POS))
        d = 128
        self.q, self.k = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        self._h = None
        self.gpt.ln_f.register_forward_hook(lambda m, i, o: setattr(self, "_h", o))

    def forward(self, x):
        """Returns log p (B, T, 256) and the copy attention (B, T, T)."""
        logits, _ = self.gpt(x)
        h = self._h
        logits = logits.clone()
        logits[..., DIGITS] = -1e9                         # digits only via copying
        p_vocab = F.softmax(logits, -1)
        T = x.shape[1]
        att = (self.q(h) @ self.k(h).transpose(1, 2)) / math.sqrt(h.shape[-1])
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = F.softmax(att.masked_fill(~causal, -1e9), -1)
        g = torch.sigmoid(self.gate(h))
        p_copy = torch.zeros_like(p_vocab).scatter_add_(-1, x[:, None, :].expand(-1, T, -1), att)
        p = (1 - g) * p_vocab + g * p_copy
        return torch.log(p + 1e-12), att


def train(steps, seed, log):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = PointerGPT(G.BLOCK)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    for i in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 2e-3 * 0.5 * (1 + math.cos(math.pi * i / steps))
        batch = [G.example(rng, 1, 4) for _ in range(32)]
        x, y, _ = G.encode([b[0] for b in batch], [b[1] for b in batch])
        logp, _ = m(x)
        loss = F.nll_loss(logp.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 1000 == 0 or i == steps - 1:
            log({"step": i, "nll": round(loss.item(), 4)})
    return m


def provenance_ok(prompt, asked, sources, output=None):
    """Exact check: the copied digits are, in order, the year of a fact
    'NAME born YYYY' in the prompt whose NAME is the asked entity, and the
    released output is exactly that span (every emitted digit equals the
    character at its recorded source position)."""
    if len(sources) != 4 or any(s is None for s in sources):
        return False
    j = sources[0]
    if sources != [j, j + 1, j + 2, j + 3]:
        return False
    span = prompt[j:j + 4]
    if not (span.isdigit() and len(span) == 4):
        return False
    if output is not None and output != span:
        return False
    head = prompt[:j]
    if not head.endswith(" born "):
        return False
    name = head[: -len(" born ")].split("; ")[-1]
    return name == asked


@torch.no_grad()
def run(m, prompts):
    m.eval()
    outs, srcs = [], []
    for p in prompts:
        x = torch.tensor([list(p.encode("latin-1"))])
        L0 = x.shape[1]
        sources = []
        for _ in range(8):
            logp, att = m(x)
            nxt = int(logp[0, -1].argmax())
            if nxt in (46, 10):
                break
            sources.append(int(att[0, -1, :L0].argmax()) if chr(nxt).isdigit() else None)
            x = torch.cat([x, torch.tensor([[nxt]])], 1)
        outs.append(bytes(x[0, L0:].tolist()).decode("latin-1"))
        srcs.append([s for s in sources if s is not None] if any(s is not None for s in sources) else [])
    return outs, srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--names", default="single", choices=["multi", "single"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos", default="learned", choices=["learned", "alibi"])
    ap.add_argument("--out", default="results/provenance_by_design.json")
    a = ap.parse_args()
    G.NAMES, G.POS = a.names, a.pos
    torch.set_num_threads(os.cpu_count())
    t0 = time.time()
    ckpt = a.out.replace(".json", ".pt")
    if os.path.exists(ckpt):
        m = PointerGPT(G.BLOCK)
        m.load_state_dict(torch.load(ckpt))
    else:
        m = train(a.steps, a.seed, lambda e: print(json.dumps(e), flush=True))
        torch.save(m.state_dict(), ckpt)
    rng = random.Random(1000 + a.seed)
    res = {"names": a.names, "pos": a.pos, "train_s": time.time() - t0}
    for name, (kmin, kmax) in {"in_dist": (1, 4), "shift": (6, 8)}.items():
        data = [G.example(rng, kmin, kmax) for _ in range(a.n_test)]
        prompts = [d[0] for d in data]
        outs, srcs = run(m, prompts)
        raw = {"correct": 0, "hallucinated": 0, "wrong_attribution": 0}
        rel = {"released_correct": 0, "released_false": 0, "withheld": 0}
        n_present = sum(d[2] for d in data)
        for (prompt, ans, present), o, s in zip(data, outs, srcs):
            asked = prompt.split(" Q ")[1].split(" born?")[0]
            raw["correct"] += int(o == ans)
            raw["hallucinated"] += int((not present) and o != "unknown")
            raw["wrong_attribution"] += int(present and o not in (ans, "unknown"))
            if o == "unknown":
                rel["released_correct"] += int(not present)
                continue
            if provenance_ok(prompt, asked, s, output=o):
                rel["released_correct" if o == ans else "released_false"] += 1
            else:
                rel["withheld"] += 1
        res[name] = {"raw_accuracy": raw["correct"] / len(data),
                     "raw_hallucinated": raw["hallucinated"] / max(1, len(data) - n_present),
                     "raw_wrong_attribution": raw["wrong_attribution"] / max(1, n_present),
                     "released_false": rel["released_false"],
                     "released_accuracy": rel["released_correct"] / len(data),
                     "withheld_frac": rel["withheld"] / len(data)}
        print(json.dumps({name: res[name]}), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
