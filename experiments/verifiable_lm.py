"""Verifiable by construction: a typed answer, a causal bottleneck, exact provenance.

Combines the two semantic-layer mechanisms in one model and removes the parts
that were only "trained to be true":

  * typed output: the first answer token is either ABSTAIN ('u' of "unknown")
    or a digit, and digits only come from the pointer head (they are masked
    out of the vocabulary path). Each digit is emitted *as a pointer*: the
    decoder picks a source position j and outputs x[j], so every digit has an
    exact source.
  * causal bottleneck: the abstain/answer decision is a fixed monotone
    function of one named channel (mean of residual dims 120..127 after block
    1, at the decision position):

        P(abstain) = sigmoid(beta - softplus(alpha) * s)

    Nothing else in the network can influence that decision, so the channel is
    not a correlate found by a probe or a value the model was encouraged to
    use: it is the decision variable, by architecture. Interventions on s
    control abstention exactly, and the threshold s* = beta / softplus(alpha)
    is readable from two scalars.
  * exact release check: an answered year is released only if its pointers
    form one contiguous 4-digit span that is the year of a fact about the
    asked entity (provenance_ok). False years cannot be released, under any
    input distribution.

What is left to measure is coverage (how often a present fact is answered and
released) and how well s tracks "the fact is present".

    python experiments/verifiable_lm.py --out results/verifiable_lm.json
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
from interp_by_design import CH, KAPPA, LAYER, auroc  # noqa: E402
from provenance_by_design import DIGITS, PointerGPT, provenance_ok  # noqa: E402

U = ord("u")


class VerifiableLM(nn.Module):
    def __init__(self, block, layer=LAYER):
        super().__init__()
        self.layer = layer
        self.ptr = PointerGPT(block)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.write = None                  # (mask (B,T) bool, value) to overwrite the channel
        self._s = None
        self.ptr.gpt.blocks[layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.write is not None:
            mask, v = self.write
            out = out.clone()
            out[..., CH] = torch.where(mask[..., None], torch.as_tensor(v, dtype=out.dtype), out[..., CH])
        self._s = out[..., CH].mean(-1)
        return out

    def abstain_logit(self, s):
        return self.beta - F.softplus(self.alpha) * s

    def forward(self, x, dpos):
        """log p (B, T, 256), copy attention (B, T, T), channel s and abstain logit at dpos (B,)."""
        logp, att = self.ptr(x)
        b = torch.arange(x.shape[0])
        s = self._s[b, dpos]
        d = self.abstain_logit(s)
        a = att[b, dpos]                                                  # (B, T) copy weights
        is_dig = (x[..., None] == DIGITS).any(-1).float()                 # sources holding a digit
        p_dig = torch.zeros(x.shape[0], 256).scatter_add_(1, x, a * is_dig)[:, DIGITS]
        p_dig = p_dig / (p_dig.sum(-1, keepdim=True) + 1e-12)
        dec = torch.full((x.shape[0], 256), -30.0)
        dec[:, U] = F.logsigmoid(d)
        dec[:, DIGITS] = F.logsigmoid(-d)[:, None] + torch.log(p_dig + 1e-12)
        logp = logp.clone()
        logp[b, dpos] = dec
        return logp, att, s, d


def batch(rng, n, kmin, kmax):
    data = [G.example(rng, kmin, kmax) for _ in range(n)]
    x, y, pos = G.encode([d[0] for d in data], [d[1] for d in data])
    return data, x, y, pos + 1                                            # decision = the position after 'A'


def train(steps, seed, log, aux_w, layer, answer_only):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = VerifiableLM(G.BLOCK, layer)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    for i in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 2e-3 * 0.5 * (1 + math.cos(math.pi * i / steps))
        data, x, y, dpos = batch(rng, 32, 1, 4)
        logp, _, s, _ = m(x, dpos)
        tok = F.nll_loss(logp.reshape(-1, 256), y.reshape(-1), reduction="none").view(y.shape)
        if answer_only:                    # supervise the answer and its terminator, not the random context
            w = (torch.arange(y.shape[1])[None, :] >= dpos[:, None]).float()
            loss_lm = (tok * w).sum() / w.sum()
        else:
            loss_lm = tok.mean()
        present = torch.tensor([d[2] for d in data])
        loss_aux = F.smooth_l1_loss(s, torch.where(present, KAPPA, -KAPPA))
        loss = loss_lm + aux_w * loss_aux
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 1000 == 0 or i == steps - 1:
            log({"step": i, "lm": round(loss_lm.item(), 4), "aux": round(loss_aux.item(), 4),
                 "threshold": round((m.beta / F.softplus(m.alpha)).item(), 3)})
    return m


@torch.no_grad()
def answer(m, prompt, force=None):
    """Greedy typed decoding. Returns (output, digit sources, s, P(abstain))."""
    m.eval()
    x = torch.tensor([list(prompt.encode("latin-1"))])
    L0 = x.shape[1]
    dpos = torch.tensor([L0 - 1])
    out, sources, s0, pa = [], [], None, None
    for t in range(8):
        if force is not None:
            m.write = (torch.arange(x.shape[1])[None, :] == L0 - 1, force)
        logp, att, s, d = m(x, dpos)
        m.write = None
        if t == 0:
            s0, pa = float(s[0]), float(torch.sigmoid(d[0]))
        nxt = int(logp[0, -1].argmax())
        if t == 0 and nxt == U:                                           # typed output: ABSTAIN
            return "unknown", [], s0, pa
        if nxt in (46, 10):
            break
        if chr(nxt).isdigit():                                            # emit as a pointer
            w = att[0, -1, :L0].clone()
            w[x[0, :L0] != nxt] = -1.0
            j = int(w.argmax())
            sources.append(j)
            nxt = int(x[0, j])
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return bytes(out).decode("latin-1"), sources, s0, pa


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y, float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        sel = (p >= lo) & (p < lo + 1 / bins)
        if sel.any():
            e += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(e)


def evaluate(m, data):
    r = {"released_correct_year": 0, "released_false": 0, "correct_abstain": 0,
         "false_abstain": 0, "withheld": 0, "raw_correct": 0}
    S, P_ans, present = [], [], []
    for prompt, ans, pres in data:
        asked = prompt.split(" Q ")[1].split(" born?")[0]
        o, src, s, pa = answer(m, prompt)
        S.append(s); P_ans.append(1 - pa); present.append(pres)
        r["raw_correct"] += int(o == ans)
        if o == "unknown":
            r["correct_abstain" if not pres else "false_abstain"] += 1
        elif provenance_ok(prompt, asked, src, output=o):
            r["released_correct_year" if o == ans else "released_false"] += 1
        else:
            r["withheld"] += 1
    n, n_p = len(data), sum(present)
    pres = np.array(present, int)
    return {"raw_accuracy": r["raw_correct"] / n,
            "released_false": r["released_false"],
            "coverage": r["released_correct_year"] / max(1, n_p),          # present facts answered + released
            "correct_abstain": r["correct_abstain"] / max(1, n - n_p),
            "false_abstain": r["false_abstain"] / max(1, n_p),
            "withheld_frac": r["withheld"] / n,
            "auroc_channel_present": auroc(np.array(S), pres),
            "ece_p_answer_vs_present": ece(P_ans, pres)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux", type=float, default=0.5, help="weight of the channel supervision")
    ap.add_argument("--layer", type=int, default=LAYER, help="block whose output carries the channel")
    ap.add_argument("--answer-only", action="store_true", help="LM loss on answer tokens only")
    ap.add_argument("--pos", default="alibi", choices=["learned", "alibi"])
    ap.add_argument("--out", default="results/verifiable_lm.json")
    a = ap.parse_args()
    G.NAMES, G.POS = "single", a.pos
    torch.set_num_threads(os.cpu_count())
    t0 = time.time()
    ckpt = a.out.replace(".json", ".pt")
    if os.path.exists(ckpt):
        m = VerifiableLM(G.BLOCK, a.layer)
        m.load_state_dict(torch.load(ckpt))
    else:
        m = train(a.steps, a.seed, lambda e: print(json.dumps(e), flush=True), a.aux, a.layer,
                  a.answer_only)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        torch.save(m.state_dict(), ckpt)
    res = {"pos": a.pos, "aux": a.aux, "layer": a.layer, "answer_only": a.answer_only, "train_s": time.time() - t0,
           "threshold": (m.beta / F.softplus(m.alpha)).item(), "slope": F.softplus(m.alpha).item()}
    rng = random.Random(1000 + a.seed)
    for name, (kmin, kmax) in {"in_dist": (1, 4), "shift": (6, 8)}.items():
        data = [G.example(rng, kmin, kmax) for _ in range(a.n_test)]
        d = evaluate(m, data)
        # the decision is a function of s alone: forcing it controls abstention exactly
        d["force_low_abstains"] = float(np.mean([answer(m, p, force=-KAPPA)[0] == "unknown" for p, _, _ in data]))
        d["force_high_answers"] = float(np.mean([answer(m, p, force=+KAPPA)[0] != "unknown" for p, _, _ in data]))
        res[name] = d
        print(json.dumps({name: d}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
