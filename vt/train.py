"""Training with optional verifiability regularisers.

- rs_weight:  ReLU-stability loss (Xiao et al. 2019) on IBP pre-activation
              bounds, -tanh(1 + l*u), averaged over neurons. Pushes neurons to
              be provably always-on or always-off inside the eps-ball, which
              removes binary variables from the MILP.
- ibp_weight: certified (IBP) training (Gowal et al. 2018), as a reference.
- ternary:    set on the model; weights are quantised in the forward pass.
eps is ramped linearly from 0 over the first `warmup` fraction of training,
which keeps the bounds tight early on so the losses have useful gradients.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .model import MixerNet


@dataclass
class TrainConfig:
    name: str = "baseline"
    ternary: bool = False
    rs_weight: float = 0.0
    ibp_weight: float = 0.0
    eps: float = 0.1
    epochs: int = 30
    lr: float = 3e-3
    batch_size: int = 128
    warmup: float = 0.5
    d_model: int = 8
    d_hidden: int = 16
    n_blocks: int = 2
    seed: int = 0


def regulariser_terms(model, x, y, eps, cfg):
    lb, pre = model.margin_lower_bounds(x, y, eps)
    terms = {}
    if cfg.rs_weight:
        terms["rs"] = torch.cat([(-torch.tanh(1 + l * u)).flatten(1) for l, u in pre], 1).mean()
    if cfg.ibp_weight:
        terms["ibp"] = F.cross_entropy(-lb, y)
    return terms


def train(cfg, train_data, vocab, n_classes, log_every=0):
    torch.manual_seed(cfg.seed)
    X, Y = train_data
    model = MixerNet(X.shape[1], vocab, cfg.d_model, cfg.d_hidden, cfg.n_blocks,
                     n_classes, ternary=cfg.ternary)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    steps_per_epoch = (len(X) + cfg.batch_size - 1) // cfg.batch_size
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=total)
    step = 0
    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(X))
        for b in range(steps_per_epoch):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            x, y = X[idx], Y[idx]
            ce = F.cross_entropy(model(x), y)
            loss = ce
            if cfg.rs_weight or cfg.ibp_weight:
                eps_t = cfg.eps * min(1.0, step / max(1, cfg.warmup * total))
                t = regulariser_terms(model, x, y, eps_t, cfg)
                if "ibp" in t:
                    k = cfg.ibp_weight * min(1.0, step / max(1, cfg.warmup * total))
                    loss = (1 - k) * ce + k * t["ibp"]
                if "rs" in t:
                    loss = loss + cfg.rs_weight * t["rs"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  [{cfg.name}] epoch {epoch + 1} loss {loss.item():.4f} ce {ce.item():.4f}")
    return model


@torch.no_grad()
def evaluate(model, data, eps):
    X, Y = data
    acc = (model(X).argmax(1) == Y).float().mean().item()
    lb, pre = model.margin_lower_bounds(X, Y, eps)
    lb[torch.arange(len(Y)), Y] = float("inf")
    ibp_acc = (lb.min(1).values > 0).float().mean().item()
    unstable = sum(((l < 0) & (u > 0)).flatten(1).sum(1) for l, u in pre).float()
    n_relu = sum(l[0].numel() for l, _ in pre)
    return {"clean_acc": acc, "ibp_verified_acc": ibp_acc,
            "unstable_mean": unstable.mean().item(), "n_relu": n_relu,
            "weight_zero_frac": model.weight_zero_frac()}


__all__ = ["TrainConfig", "train", "evaluate", "asdict"]
