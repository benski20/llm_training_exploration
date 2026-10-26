"""Interpretability by design vs. retrospective probing, for hallucination.

World: people P0..P(N-1). Every person appears in the corpus ("P12 lives T45.")
but only half have their birth year in the corpus ("P12 born 1843.", with
paraphrases). Questions "Q P12 born? A ..." must be answered with the year
when it was in the corpus, and with "unknown" when it was not. Questions are
trained on 70% of people and tested on the rest, whose facts were in the corpus
but whose questions were never seen.

by-design : a reserved 8-dim "epistemic channel" in the residual stream after
            block 1, with (a) an auxiliary loss making it encode "this fact was
            in my training corpus" at the answer position, and (b) intervention
            training: writing "unknown" into the channel must make the model
            abstain. The channel is a built-in monitor and a control knob.
baseline  : same model and data, no channel; a linear probe fitted afterwards
            on the same layer/position (the retrospective analogue).

    python experiments/interp_by_design.py --out results/interp_by_design.json
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
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.lm import GPT, GPTConfig  # noqa: E402

LAYER, CH = 1, slice(120, 128)      # channel: dims 120..127 after block LAYER
KAPPA = 4.0                         # canonical channel magnitude for interventions


def world(seed, n=600):
    rng = random.Random(seed)
    people = [f"P{i}" for i in range(n)]
    year = {p: rng.randint(1700, 1999) for p in people}
    town = {p: f"T{rng.randint(10, 99)}" for p in people}
    known = {p: rng.random() < 0.5 for p in people}
    order = people[:]
    rng.shuffle(order)
    train_q = set(order[: int(0.7 * n)])
    return people, year, town, known, train_q


def corpus(people, year, town, known, rng, reps=4):
    lines = []
    for _ in range(reps):
        for p in people:
            lines.append(f"{p} lives {town[p]}.")
            if known[p]:
                lines.append(rng.choice([f"{p} born {year[p]}.", f"{p} was born {year[p]}.",
                                         f"birth of {p}: {year[p]}."]))
    rng.shuffle(lines)
    return lines


def qa(p, year, known):
    return f"Q {p} born? A {year[p] if known[p] else 'unknown'}."


def encode(lines, block=48):
    """Batch of lines -> (x, y, answer_pos). answer_pos = index of the 'A' token."""
    xs, ys, pos = [], [], []
    for s in lines:
        b = list((s + "\n").encode())[:block + 1]
        b = b + [10] * (block + 1 - len(b))
        xs.append(b[:-1]); ys.append(b[1:])
        pos.append(s.find(" A ") + 1 if " A " in s else -1)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(pos)


class Channel:
    """Forward hook on block LAYER: records the channel and can overwrite it."""

    def __init__(self, model):
        self.read, self.write = None, None
        model.blocks[LAYER].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.read = out
        if self.write is not None:
            mask, value = self.write                 # mask: (B, T) bool, value: (B,) float
            out = out.clone()
            out[..., CH] = torch.where(mask[..., None], value[:, None, None].expand(-1, out.shape[1], 8),
                                       out[..., CH])
            self.read = out
            return out
        return out


def channel_score(read, pos):
    return read[torch.arange(len(pos)), pos][:, CH].mean(-1)


def train(by_design, data, steps, seed, log):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(block_size=48, n_layer=4, n_head=4, n_embd=128))
    ch = Channel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    corpus_lines, qa_lines, qa_known = data
    rng = random.Random(seed)
    for i in range(steps):
        lr = 2e-3 * 0.5 * (1 + math.cos(math.pi * i / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = rng.sample(corpus_lines, 24)
        qi = rng.sample(range(len(qa_lines)), 24)
        qb = [qa_lines[j] for j in qi]
        kb = torch.tensor([qa_known[j] for j in qi], dtype=torch.float)
        x, y, pos = encode(batch + qb)
        is_q = torch.arange(48) >= 0
        loss_aux = torch.tensor(0.0)
        ch.write = None
        if by_design and rng.random() < 0.3:
            # intervention training: force "unknown" into the channel of known-fact
            # questions from the answer position on, and require abstention
            flip = (kb > 0) & (torch.rand(24) < 0.7)
            if flip.any():
                for j in torch.nonzero(flip)[:, 0].tolist():
                    p = qb[j][2:qb[j].index(" born")]
                    qb[j] = f"Q {p} born? A unknown."
                x, y, pos = encode(batch + qb)
                mask = torch.zeros(48, 48, dtype=torch.bool)
                rows = torch.arange(24, 48)
                tpos = torch.arange(48)[None, :]
                qmask = torch.zeros(48, 48, dtype=torch.bool)
                qmask[rows] = tpos >= pos[rows][:, None]
                qmask[24:][~flip] = False
                ch.write = (qmask, torch.full((48,), -KAPPA))
        logits, loss_lm = model(x, y)
        if by_design:
            s = channel_score(ch.read[24:], pos[24:])
            target = torch.where(kb > 0, KAPPA, -KAPPA)
            if ch.write is not None:
                target = torch.where(flip, -KAPPA, target)
            loss_aux = F.smooth_l1_loss(s, target)
        loss = loss_lm + 0.5 * loss_aux
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 1000 == 0 or i == steps - 1:
            log({"by_design": by_design, "step": i, "lm": loss_lm.item(), "aux": float(loss_aux)})
    ch.write = None
    return model, ch


@torch.no_grad()
def answer(model, ch, people, force=None):
    """Greedy answers to 'Q p born? A ' plus channel score at the 'A' position.
    force: optional channel value written from the answer position on."""
    model.eval()
    prompts = [f"Q {p} born? A " for p in people]
    outs, scores = [], []
    for s in sorted(set(len(p) for p in prompts)):
        idx = [i for i, p in enumerate(prompts) if len(p) == s]
        xb = torch.tensor([list(prompts[i].encode()) for i in idx])
        pos = torch.full((len(idx),), s - 2)
        gen = xb
        for t in range(8):
            if force is not None:
                mask = torch.arange(gen.shape[1])[None, :] >= pos[:, None]
                ch.write = (mask, torch.full((len(idx),), force))
            logits, _ = model(gen)
            if t == 0:
                sc = channel_score(ch.read, pos)
            gen = torch.cat([gen, logits[:, -1].argmax(-1, keepdim=True)], 1)
        ch.write = None
        for k, i in enumerate(idx):
            outs.append((i, bytes(gen[k, s:].tolist()).decode("latin-1").split(".")[0]))
            scores.append((i, float(sc[k])))
    outs = [o for _, o in sorted(outs)]
    scores = [s for _, s in sorted(scores)]
    return outs, np.array(scores)


@torch.no_grad()
def residual_at_answer(model, ch, people):
    feats = []
    for p in people:
        s = f"Q {p} born? A "
        model(torch.tensor([list(s.encode())]))
        feats.append(ch.read[0, len(s) - 2].numpy())
    return np.stack(feats)


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def fit_probe(X, y, steps=500):
    Xt, yt = torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(((Xt - mu) / sd) @ w + b, yt) + 1e-3 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda Z: (((torch.tensor(Z, dtype=torch.float) - mu) / sd) @ w + b).detach().numpy(), w.detach() / sd


def behaviour(outs, people, year, known):
    k = [i for i, p in enumerate(people) if known[p]]
    u = [i for i, p in enumerate(people) if not known[p]]
    return {"known_correct": float(np.mean([outs[i] == str(year[people[i]]) for i in k])),
            "known_abstain": float(np.mean([outs[i] == "unknown" for i in k])),
            "unknown_abstain": float(np.mean([outs[i] == "unknown" for i in u])),
            "unknown_hallucinated": float(np.mean([outs[i] != "unknown" for i in u]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/interp_by_design.json")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    people, year, town, known, train_q = world(a.seed)
    rng = random.Random(a.seed)
    lines = corpus(people, year, town, known, rng)
    qa_people = [p for p in people if p in train_q]
    data = (lines, [qa(p, year, known) for p in qa_people], [known[p] for p in qa_people])
    test = [p for p in people if p not in train_q]
    labels = np.array([known[p] for p in test], dtype=int)
    res = {}
    for by_design in (False, True):
        t0 = time.time()
        model, ch = train(by_design, data, a.steps, a.seed, lambda e: print(json.dumps(e), flush=True))
        outs, ch_scores = answer(model, ch, test)
        r = {"train_s": time.time() - t0, "behaviour": behaviour(outs, test, year, known)}
        # retrospective probe on the same layer/position, fitted on training-question people
        Xtr = residual_at_answer(model, ch, qa_people)
        probe, direction = fit_probe(Xtr, np.array([known[p] for p in qa_people], dtype=int))
        r["probe_auroc"] = auroc(probe(residual_at_answer(model, ch, test)), labels)
        if by_design:
            r["channel_auroc"] = auroc(ch_scores, labels)
            force_k, _ = answer(model, ch, test, force=-KAPPA)
            force_u, _ = answer(model, ch, test, force=+KAPPA)
            r["intervene_unknown_on_known_abstain"] = behaviour(force_k, test, year, known)["known_abstain"]
            r["intervene_known_on_unknown_answer"] = behaviour(force_u, test, year, known)["unknown_hallucinated"]
        print(json.dumps({"by_design": by_design, **r}), flush=True)
        res["by_design" if by_design else "baseline"] = r
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
