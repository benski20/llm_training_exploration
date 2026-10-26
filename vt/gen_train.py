"""Train a count-state generator whose support is exactly the spec language.

loss = NLL(data; eps) + lam * support hinge over EVERY (state, token) pair

The support hinge uses the enumerable product state graph (vt/gen_verify.py):
tokens that keep the spec satisfiable must score >= margin, tokens that make it
unsatisfiable must score <= -margin (so relu gives probability exactly 0).
Training therefore covers all of the generator's states, not a sample, and
the result is then certified exactly by verify_generator.
"""
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .gen import CountGen
from .gen_verify import state_graph, verify_generator


@dataclass
class GenConfig:
    L: int = 16
    lengths: tuple = ()
    d_model: int = 8
    d_hidden: int = 32
    window: int = 1
    steps: int = 3000
    lr: float = 1e-2
    batch: int = 256
    n_data: int = 20000
    margin: float = 0.5
    lam: float = 1.0
    eps: float = 1e-3
    seed: int = 0


def _layer_tensors(g, V, w):
    """Per layer: current token, recent tokens (offset-major) and older counts."""
    out = []
    for layer in g["layers"][:-1]:
        n = len(layer)
        cur = torch.full((n,), -1, dtype=torch.long)
        rec = torch.full((n, max(1, w - 1)), -1, dtype=torch.long)
        cnt = torch.zeros(n, V)
        for i, (counts, recent, _) in enumerate(layer):
            if recent:
                cur[i] = recent[-1]
                for o in range(1, min(w, len(recent))):
                    rec[i, o - 1] = recent[-1 - o]
            cnt[i] = torch.tensor(counts, dtype=torch.float32)
        out.append((cur, rec, cnt))
    return out


def _state_scores(model, cur, rec, cnt, r):
    """Raw next-token logits z (before relu) for a batch of prefix states with r
    tokens still to generate (including the next one)."""
    k, w, E = model.kern, model.w, model.E
    empty = cur < 0
    s = (1 + k[0]) * E[cur.clamp(min=0)] + r * model.u
    for o in range(1, w):
        m = (rec[:, o - 1] >= 0).float()[:, None]
        s = s + m * k[o] * E[rec[:, o - 1].clamp(min=0)]
    s = s + k[w] * (cnt @ E)
    z = F.relu(s @ model.W1.T + model.b1) @ model.Wo.T + model.bo
    return torch.where(empty[:, None], (model.z0 + r * model.u0).expand_as(z), z)


def support_hinge(model, graphs, margin):
    """Mean hinge over every live (state, token) pair. Live = the spec is still
    satisfiable from that state; dead states are unreachable once the generator
    is sound, and they can share features with live states, so they're excluded."""
    total, n = 0.0, 0
    for g, tens in graphs:
        L = len(g["layers"]) - 1
        for t, ((cur, rec, cnt), feas) in enumerate(zip(tens, g["feasible_t"])):
            live = feas.any(1)
            if not live.any():
                continue
            z = _state_scores(model, cur[live], rec[live], cnt[live], L - t)
            f = feas[live]
            total = total + F.relu(torch.where(f, margin - z, margin + z)).sum()
            n += z.numel()
    return total / max(1, n)


def train_generator(spec, cfg, V, log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    lengths = sorted(set(cfg.lengths) | {cfg.L})
    model = CountGen(V, cfg.d_model, cfg.d_hidden, cfg.window, max_len=max(lengths))
    graphs = []
    for L in lengths:
        g = state_graph(V, cfg.window, L, spec)
        g["feasible_t"] = [torch.from_numpy(f) for f in g["feasible"]]
        graphs.append((g, _layer_tensors(g, V, cfg.window)))
    data = {}
    for L in lengths:
        x = spec.sample(cfg.n_data * 3, L, rng)
        x = x[spec.label_batch(x) == 1][:cfg.n_data]
        data[L] = torch.from_numpy(x)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        L = lengths[step % len(lengths)]
        xb = data[L][torch.randint(len(data[L]), (cfg.batch,))]
        nll = -model.log_prob(xb, cfg.eps).mean() / L
        hinge = support_hinge(model, graphs, cfg.margin)
        loss = nll + cfg.lam * hinge
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == cfg.steps - 1:
            log({"step": step, "nll_per_token": nll.item(), "hinge": hinge.item(),
                 "elapsed": time.perf_counter() - t0})
    G = model.export()
    results = {L: verify_generator(G, spec, L, exact_mass=False) for L in lengths}
    return model, {"config": asdict(cfg), "lengths": lengths,
                   "verify": {str(L): r for L, r in results.items()},
                   "status": "proved" if all(r["status"] == "proved" for r in results.values())
                   else "failed", "elapsed": time.perf_counter() - t0}


@torch.no_grad()
def heldout_nll(model, spec, L, n=5000, seed=123):
    rng = np.random.default_rng(seed)
    x = spec.sample(n * 3, L, rng)
    x = torch.from_numpy(x[spec.label_batch(x) == 1][:n])
    return float(-model.log_prob(x, 0.0).mean() / L)
