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
    bits: int = 0                # k-bit uniform weights instead of float (0 = off)
    shared: bool = False         # weight-tied across positions (Toeplitz mixing, pooled readout)
    lengths: tuple = ()          # extra lengths (< L) to train and prove at too; needs shared
    enumerate_up_to: int = 1 << 16  # lengths with V^L at most this are checked exhaustively
    tau: float = 1e-4            # required logit margin (same as the certificate's claim)
    ternary_from_round: int = 0  # train float before this round, then quantise (QAT; also for bits)
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
    cex_per_call: int = 20           # distinct counterexamples per verifier call (no-good cuts)
    seed: int = 0


def _onehot(seqs, V):
    return torch.from_numpy(np.eye(V, dtype=np.float32)[seqs])


def _fit(model, datasets, epochs, cfg, lr):
    """datasets: list of (X, Y, W), one per length; batches are interleaved."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    nb = [(len(X) + cfg.batch_size - 1) // cfg.batch_size for X, _, _ in datasets]
    # anneal to zero within the round: the round ends at a settled point
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * sum(nb)))
    for _ in range(epochs):
        perms = [torch.randperm(len(X)) for X, _, _ in datasets]
        order = torch.randperm(sum(nb)).tolist()
        owner = [k for k, n in enumerate(nb) for _ in range(n)]
        start = [0] * len(datasets)
        for o in order:
            k = owner[o]
            X, Y, W = datasets[k]
            idx = perms[k][start[k]:start[k] + cfg.batch_size]
            start[k] += cfg.batch_size
            logits = model(X[idx])
            y, w = Y[idx], W[idx]
            ce = F.cross_entropy(logits, y, reduction="none")
            other = logits.clone()
            other[torch.arange(len(y)), y] = -float("inf")
            margin = logits[torch.arange(len(y)), y] - other.max(1).values
            loss = (w * (ce + F.relu(cfg.margin - margin))).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()


@torch.no_grad()
def _falsify(model, spec, cfg, V, rng, L):
    if V ** L <= cfg.enumerate_up_to:
        seqs = _all_seqs(L, V)
    else:
        seqs = spec.sample(cfg.falsify_samples, L, rng)
    bad = []
    for i in range(0, len(seqs), 1 << 16):
        chunk = seqs[i:i + (1 << 16)]
        logits = model(_onehot(chunk, V))
        y = torch.from_numpy(spec.label_batch(chunk))
        true = logits[torch.arange(len(y)), y]
        logits[torch.arange(len(y)), y] = -float("inf")
        bad.append((true - logits.max(1).values <= cfg.tau).numpy())
    return np.unique(seqs[np.concatenate(bad)], axis=0)


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


def _all_seqs(L, V):
    idx = np.arange(V ** L)
    return (idx[:, None] // V ** np.arange(L)[::-1]) % V


def _mutants(seqs, V):
    out = [seqs]
    for t in range(seqs.shape[1]):
        m = seqs.copy()
        m[:, t] = (m[:, t] + 1) % V
        out.append(m)
    return np.concatenate(out)


def cegis(spec, cfg, V=2, log=print):
    """Train until every length in {cfg.L} ∪ cfg.lengths is proved (or rounds run out).
    Lengths with V^L <= cfg.enumerate_up_to are decided by exhaustive enumeration,
    the rest by the MILP verifier."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    lengths = sorted(set(cfg.lengths) | {cfg.L})
    if len(lengths) > 1 and not cfg.shared:
        raise ValueError("multiple lengths need a weight-tied (shared) model")
    if max(lengths) != cfg.L:
        raise ValueError("cfg.L must be the largest length")
    model = MixerNet(cfg.L, V, cfg.d_model, cfg.d_hidden, cfg.n_blocks,
                     spec.n_classes, ternary=cfg.ternary, causal=cfg.causal,
                     shared=cfg.shared)
    n_each = max(1000, cfg.n_train // len(lengths))
    base = {L: spec.sample(n_each, L, rng) for L in lengths}
    cex = {L: np.zeros((0, L), dtype=np.int64) for L in lengths}
    history = []
    t0 = time.perf_counter()
    for rnd in range(cfg.rounds):
        quant_on = rnd >= cfg.ternary_from_round
        model.ternary = cfg.ternary and quant_on
        model.bits = cfg.bits if (cfg.bits and quant_on) else None
        datasets = []
        for L in lengths:
            seqs = np.concatenate([base[L], cex[L]])
            W = torch.cat([torch.ones(len(base[L])), torch.full((len(cex[L]),), cfg.cex_weight)])
            datasets.append((_onehot(seqs, V), torch.from_numpy(spec.label_batch(seqs)), W))
        first = rnd == 0 or ((cfg.ternary or cfg.bits) and rnd == cfg.ternary_from_round)
        lr = cfg.lr if first else max(cfg.lr_min, cfg.lr * cfg.lr_decay ** rnd)
        _fit(model, datasets, cfg.epochs_first if first else cfg.epochs_round, cfg, lr)
        found = {L: _falsify(model, spec, cfg, V, rng, L) for L in lengths}
        entry = {"round": rnd, "n_cex_total": int(sum(len(c) for c in cex.values())),
                 "sampled_errors": int(sum(len(f) for f in found.values()))}
        if len(lengths) > 1:
            entry["errors_by_length"] = {str(L): int(len(found[L])) for L in lengths}
        if entry["sampled_errors"] == 0:
            per_length, all_proved = {}, True
            # enumerated lengths are already proved (complete check above)
            for L in lengths:
                if V ** L <= cfg.enumerate_up_to:
                    per_length[L] = {"status": "proved", "method": "enumeration"}
            # longest first: it is the most likely to fail, and a failure skips the rest
            for L in sorted([L for L in lengths if L not in per_length], reverse=True):
                res = _verify_escalating(model.export(L), spec, cfg)
                per_length[L] = {"status": res["status"], "method": "milp", "time": res["time"],
                                 "unstable": res["unstable"]}
                if res["status"] != "proved":
                    all_proved = False
                    if res["status"] == "unknown":
                        break
                    seqs = np.array([c["seq"] for c in res["counterexamples"]], dtype=np.int64)
                    if len(seqs):
                        near = _neighbourhood(model, spec, seqs, V, rng)
                        entry["neighbourhood_errors"] = int(len(near))
                        found[L] = np.concatenate([seqs, near]) if len(near) else seqs
                    break  # train on what we found before verifying longer lengths
            entry["verify"] = {str(L): v for L, v in per_length.items()}
            if all_proved and quant_on:
                entry["elapsed"] = time.perf_counter() - t0
                history.append(entry)
                log(entry)
                return model, {"status": "proved", "rounds": rnd + 1, "history": history,
                               "lengths": lengths, "config": asdict(cfg),
                               "elapsed": entry["elapsed"]}
        for L in lengths:
            if len(found[L]):
                new = _mutants(found[L][:500], V)
                cex[L] = np.unique(np.concatenate([cex[L], new]), axis=0)
        entry["elapsed"] = time.perf_counter() - t0
        history.append(entry)
        log(entry)
    return model, {"status": "not_proved", "rounds": cfg.rounds, "history": history,
                   "lengths": lengths, "config": asdict(cfg), "elapsed": time.perf_counter() - t0}


def _verify_escalating(P, spec, cfg):
    limit = min(cfg.verify_time_start, cfg.verify_time)
    while True:
        res = verify_global(P, spec, time_limit=limit, workers=cfg.verify_workers,
                            stop_on_cex=not cfg.collect_all_cex, n_solutions=cfg.cex_per_call)
        if res["status"] != "unknown" or limit >= cfg.verify_time:
            res["limit"] = limit
            return res
        limit = min(4 * limit, cfg.verify_time)
