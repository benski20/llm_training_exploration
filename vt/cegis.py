"""Counterexample-guided training: train until the verifier PROVES the model
correct on every input in V^L, not just on a test set.

Each round:
  1. train on (sampled data  ∪  all counterexamples so far, upweighted),
     with a logit-margin hinge so proofs survive the tau safety margin;
  2. falsify cheaply by sampling a large batch and checking against the spec;
  3. if sampling finds nothing, run the complete MILP verifier;
  4. stop on a proof; otherwise add the counterexamples (and their one-token
     mutants) to the training set and repeat.
"""
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .global_verify import verify_global
from .model import MixerNet


@dataclass
class CegisConfig:
    L: int = 16
    d_model: int = 4
    d_hidden: int = 8
    n_blocks: int = 1
    ternary: bool = True
    causal: bool = False
    ternary_from_round: int = 0  # train float before this round, then quantise (QAT)
    n_train: int = 20000
    rounds: int = 20
    epochs_first: int = 60
    epochs_round: int = 15
    lr: float = 3e-3
    lr_decay: float = 0.7        # per-round lr multiplier, so later rounds fine-tune
    lr_min: float = 1e-4
    batch_size: int = 256
    margin: float = 1.0          # hinge target on the logit margin
    cex_weight: float = 5.0
    falsify_samples: int = 200000
    verify_time: float = 300.0
    verify_workers: int = 1
    collect_all_cex: bool = False  # run every case instead of stopping at the first counterexample
    verify_time_start: float = 60.0  # escalate x4 up to verify_time while undecided
    seed: int = 0


def _onehot(seqs, V):
    return torch.from_numpy(np.eye(V, dtype=np.float32)[seqs])


def _fit(model, X, Y, W, epochs, cfg, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    steps = epochs * ((n + cfg.batch_size - 1) // cfg.batch_size)
    # anneal to zero within the round: the round ends at a settled point
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            logits = model(X[idx])
            y, w = Y[idx], W[idx]
            ce = F.cross_entropy(logits, y, reduction="none")
            other = logits.clone()
            other[torch.arange(len(y)), y] = -float("inf")
            margin = logits[torch.arange(len(y)), y] - other.max(1).values
            loss = (w * (ce + F.relu(cfg.margin - margin))).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()


@torch.no_grad()
def _falsify(model, spec, cfg, V, rng):
    seqs = spec.sample(cfg.falsify_samples, cfg.L, rng)
    pred = model(_onehot(seqs, V)).argmax(1).numpy()
    bad = pred != spec.label_batch(seqs)
    return np.unique(seqs[bad], axis=0)


@torch.no_grad()
def _neighbourhood(model, spec, seeds, V, rng, n=50000):
    """Errors cluster: sample 1-3 token mutations of verifier counterexamples
    and keep every one the model gets wrong."""
    L = seeds.shape[1]
    base = seeds[rng.integers(len(seeds), size=n)].copy()
    k = rng.integers(1, 4, size=n)
    for j in range(3):
        rows = np.flatnonzero(k > j)
        base[rows, rng.integers(L, size=len(rows))] = rng.integers(V, size=len(rows))
    pred = model(_onehot(base, V)).argmax(1).numpy()
    return np.unique(base[pred != spec.label_batch(base)], axis=0)


def _mutants(seqs, V):
    out = [seqs]
    for t in range(seqs.shape[1]):
        m = seqs.copy()
        m[:, t] = (m[:, t] + 1) % V
        out.append(m)
    return np.concatenate(out)


def cegis(spec, cfg, V=2, log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = MixerNet(cfg.L, V, cfg.d_model, cfg.d_hidden, cfg.n_blocks,
                     spec.n_classes, ternary=cfg.ternary, causal=cfg.causal)
    base = spec.sample(cfg.n_train, cfg.L, rng)
    cex = np.zeros((0, cfg.L), dtype=np.int64)
    history = []
    t0 = time.perf_counter()
    for rnd in range(cfg.rounds):
        seqs = np.concatenate([base, cex])
        model.ternary = cfg.ternary and rnd >= cfg.ternary_from_round
        X, Y = _onehot(seqs, V), torch.from_numpy(spec.label_batch(seqs))
        W = torch.cat([torch.ones(len(base)), torch.full((len(cex),), cfg.cex_weight)])
        first = rnd == 0 or (cfg.ternary and rnd == cfg.ternary_from_round)
        lr = cfg.lr if first else max(cfg.lr_min, cfg.lr * cfg.lr_decay ** rnd)
        _fit(model, X, Y, W, cfg.epochs_first if first else cfg.epochs_round, cfg, lr)
        found = _falsify(model, spec, cfg, V, rng)
        entry = {"round": rnd, "n_cex_total": int(len(cex)), "sampled_errors": int(len(found))}
        if len(found) == 0:
            limit = min(cfg.verify_time_start, cfg.verify_time)
            while True:
                res = verify_global(model.export(), spec, time_limit=limit,
                                    workers=cfg.verify_workers,
                                    stop_on_cex=not cfg.collect_all_cex)
                if res["status"] != "unknown" or limit >= cfg.verify_time:
                    break
                limit = min(4 * limit, cfg.verify_time)
            entry["verify_limit"] = limit
            entry.update(verify_status=res["status"], verify_time=res["time"],
                         unstable=res["unstable"], verifier_cex=len(res["counterexamples"]),
                         slowest_case=max(q["time"] for q in res["queries"]))
            if res["status"] == "proved" and model.ternary == cfg.ternary:
                entry["elapsed"] = time.perf_counter() - t0
                history.append(entry)
                log(entry)
                return model, {"status": "proved", "rounds": rnd + 1, "history": history,
                               "config": asdict(cfg), "elapsed": entry["elapsed"]}
            found = np.array([c["seq"] for c in res["counterexamples"]], dtype=np.int64)
            if len(found) == 0:  # timeout: no new information; train more
                found = np.zeros((0, cfg.L), dtype=np.int64)
            else:
                near = _neighbourhood(model, spec, found, V, rng)
                entry["neighbourhood_errors"] = int(len(near))
                if len(near):
                    found = np.concatenate([found, near])
        if len(found):
            new = _mutants(found[:500], V)
            cex = np.unique(np.concatenate([cex, new]), axis=0)
        entry["elapsed"] = time.perf_counter() - t0
        history.append(entry)
        log(entry)
    return model, {"status": "not_proved", "rounds": cfg.rounds, "history": history,
                   "config": asdict(cfg), "elapsed": time.perf_counter() - t0}
