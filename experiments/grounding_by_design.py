"""Grounded answering with a built-in epistemic channel vs. a post-hoc probe.

Context: k facts "P12 born 1843; P5 born 1777; ..." about various people, then
"Q P? born? A ...". If the asked person's fact is in the context the answer is
the year; otherwise "unknown". Training uses k in 1..4; the shift test uses
6..8 facts (more distractors, longer than any training context), where a
small model starts to make errors: wrong person's year, invented answers,
false abstentions.

by-design : reserved channel (dims 120..127 after block 1), trained to read
            "the asked fact is in my context" at the answer position, plus
            intervention training (forcing the channel to "absent" must yield
            "unknown"). At deployment the channel is a monitor, a control knob,
            and, after conformal calibration, a release gate with a stated bound.
baseline  : same model and data, no channel; a linear probe fitted afterwards on
            the same layer and position.

    python experiments/grounding_by_design.py --out results/grounding_by_design.json
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
sys.path.insert(0, os.path.dirname(__file__))
from interp_by_design import CH, KAPPA, LAYER, Channel, auroc, fit_probe  # noqa: E402
from vt.lm import GPT, GPTConfig  # noqa: E402
from vt.pcr import ConformalJudge  # noqa: E402

BLOCK = 200


NAMES = "multi"          # "multi": names like P123; "single": one byte per person (easier lookup)


def pname(p):
    return f"P{p}" if NAMES == "multi" else chr(0xA0 + p)


def example(rng, kmin, kmax, n_people=1000):
    if NAMES == "single":
        n_people = 90
    k = rng.randint(kmin, kmax)
    ppl = rng.sample(range(n_people), k + 1)
    asked, others = ppl[0], ppl[1:]
    present = rng.random() < 0.5
    facts = [(p, rng.randint(1700, 1999)) for p in others]
    year = rng.randint(1700, 1999)
    if present:
        facts.insert(rng.randint(0, len(facts)), (asked, year))
        if len(facts) > k:                       # keep k facts in total
            facts.pop(next(i for i, (p, _) in enumerate(facts) if p != asked))
    ctx = "; ".join(f"{pname(p)} born {y}" for p, y in facts)
    prompt = f"{ctx}. Q {pname(asked)} born? A "
    return prompt, (str(year) if present else "unknown"), present


def encode(prompts, answers):
    xs, ys, pos = [], [], []
    for p, a in zip(prompts, answers):
        b = list((p + a + ".\n").encode("latin-1"))[:BLOCK + 1]
        b += [10] * (BLOCK + 1 - len(b))
        xs.append(b[:-1]); ys.append(b[1:]); pos.append(len(p) - 2)     # index of 'A'
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(pos)


def train(by_design, steps, seed, log):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = GPT(GPTConfig(block_size=BLOCK, n_layer=4, n_head=4, n_embd=128))
    ch = Channel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = 2e-3 * 0.5 * (1 + math.cos(math.pi * i / steps))
        batch = [example(rng, 1, 4) for _ in range(32)]
        prompts = [b[0] for b in batch]
        answers = [b[1] for b in batch]
        present = torch.tensor([b[2] for b in batch])
        flip = torch.zeros(32, dtype=torch.bool)
        ch.write = None
        if by_design and rng.random() < 0.3:
            flip = present & (torch.rand(32) < 0.7)
            answers = ["unknown" if f else a for a, f in zip(answers, flip.tolist())]
        x, y, pos = encode(prompts, answers)
        if flip.any():
            mask = (torch.arange(BLOCK)[None, :] >= pos[:, None]) & flip[:, None]
            ch.write = (mask, torch.full((32,), -KAPPA))
        _, loss_lm = model(x, y)
        loss_aux = torch.tensor(0.0)
        if by_design:
            s = ch.read[torch.arange(32), pos][:, CH].mean(-1)
            target = torch.where(present & ~flip, KAPPA, -KAPPA)
            loss_aux = F.smooth_l1_loss(s, target)
        loss = loss_lm + 0.5 * loss_aux
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 1000 == 0 or i == steps - 1:
            log({"by_design": by_design, "step": i, "lm": round(loss_lm.item(), 4),
                 "aux": round(float(loss_aux.detach()), 4)})
    ch.write = None
    return model, ch


@torch.no_grad()
def run(model, ch, prompts, force=None):
    """Greedy answers, channel score and residual at the 'A' position."""
    model.eval()
    outs, scores, feats = [], [], []
    for p in prompts:
        x = torch.tensor([list(p.encode("latin-1"))])
        pos = len(p) - 2
        for t in range(8):
            if force is not None:
                ch.write = (torch.arange(x.shape[1])[None, :] >= pos, torch.full((1,), force))
            logits, _ = model(x)
            if t == 0:
                scores.append(float(ch.read[0, pos, CH].mean()))
                feats.append(ch.read[0, pos].numpy().copy())
            nxt = int(logits[0, -1].argmax())
            if nxt in (46, 10):                           # '.' or newline ends the answer
                break
            x = torch.cat([x, torch.tensor([[nxt]])], 1)
        ch.write = None
        outs.append(bytes(x[0, len(p):].tolist()).decode("latin-1"))
    return outs, np.array(scores), np.stack(feats)


def summarize(outs, answers, present):
    present = np.array(present)
    correct = np.array([o == a for o, a in zip(outs, answers)])
    answered = np.array([o != "unknown" for o in outs])
    return {"accuracy": float(correct.mean()),
            "hallucinated": float((answered & ~present).mean() / max(1e-9, (~present).mean())),
            "wrong_year": float((answered & present & ~correct).mean() / max(1e-9, present.mean())),
            "false_abstain": float((~answered & present).mean() / max(1e-9, present.mean()))}, correct, answered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/grounding_by_design.json")
    ap.add_argument("--names", default="multi", choices=["multi", "single"])
    a = ap.parse_args()
    global NAMES
    NAMES = a.names
    torch.set_num_threads(os.cpu_count())
    rng = random.Random(1000 + a.seed)
    sets = {"in_dist": [example(rng, 1, 4) for _ in range(a.n_test)],
            "shift": [example(rng, 6, 8) for _ in range(a.n_test)],
            "calib": [example(rng, 1, 8) for _ in range(a.n_test)],
            "probe_fit": [example(rng, 1, 4) for _ in range(a.n_test)]}
    res = {}
    for by_design in (False, True):
        t0 = time.time()
        model, ch = train(by_design, a.steps, a.seed, lambda e: print(json.dumps(e), flush=True))
        r = {"train_s": time.time() - t0, "names": NAMES}
        _, _, Fp = run(model, ch, [e[0] for e in sets["probe_fit"]])
        probe, _ = fit_probe(Fp, np.array([e[2] for e in sets["probe_fit"]], dtype=int))
        # conformal release gate on the monitor: bad = calibration examples whose fact is absent
        outs_c, sc_c, F_c = run(model, ch, [e[0] for e in sets["calib"]])
        absent_c = [not e[2] for e in sets["calib"]]
        monitor = (lambda s, f: s) if by_design else (lambda s, f: probe(f[None])[0])
        bad_scores = [-monitor(s, f) for s, f, b in zip(sc_c, F_c, absent_c) if b]
        judge = ConformalJudge(lambda v: v)
        judge.calibrate(bad_scores, a.alpha)
        for name in ("in_dist", "shift"):
            data = sets[name]
            outs, sc, Fx = run(model, ch, [e[0] for e in data])
            answers = [e[1] for e in data]
            present = [e[2] for e in data]
            summ, correct, answered = summarize(outs, answers, present)
            mon = sc if by_design else probe(Fx)
            pres = np.array(present, dtype=int)
            errors = answered & ~correct                     # the model's own wrong answers
            d = {**summ,
                 "auroc_grounding": auroc(mon, pres),
                 "auroc_own_errors": auroc(-mon[answered], errors[answered].astype(int))
                 if errors[answered].any() and (~errors[answered]).any() else None}
            # release gate: answer only if the monitor clears the conformal threshold
            released = np.array([judge.passes(-m) for m in mon]) & answered
            d["gated_hallucination_rate"] = float((released & ~np.array(present, bool)).sum() /
                                                  max(1, (~np.array(present, bool)).sum()))
            d["gated_kept_correct"] = float((released & correct & np.array(present, bool)).sum() /
                                            max(1, (correct & np.array(present, bool)).sum()))
            if by_design:
                fo, _, _ = run(model, ch, [e[0] for e in data], force=-KAPPA)
                fu, _, _ = run(model, ch, [e[0] for e in data], force=+KAPPA)
                d["force_absent_on_present_abstains"] = float(np.mean(
                    [o == "unknown" for o, p in zip(fo, present) if p]))
                d["force_present_on_absent_answers"] = float(np.mean(
                    [o != "unknown" for o, p in zip(fu, present) if not p]))
            r[name] = d
        r["conformal"] = judge.guarantee()
        print(json.dumps({"by_design": by_design, **r}), flush=True)
        res["by_design" if by_design else "baseline"] = r
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
