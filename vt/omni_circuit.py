"""A hand-written verification circuit: the omni verifier's decision layer as a tiny network whose every weight is set by
hand and every neuron has a meaning (weight-space programming where it fits: an auditable program over learned
signals, not a detector written from scratch, which the repo found does not work: docs/omni_experiments/).

Inputs (each standardised by the domain profile from unlabelled replies; NaN = not available):
    r  reader probe on the reply        u  reader probe, worst sentence / step     j  reader's own "No"
    q  judge 1 "No" (Qwen3-14B)         p  judge 2 "No" (phi-4)                     n  NLI unsupported (documents)
    c  written-checker violations (count, raw)                                     a  arithmetic mismatch (0/1, raw)
Context gates, read from the request itself:  D  documents supplied     R  step-by-step reasoning

Layer 1, evidence neurons (linear, hand weights):
    JUDGES    = (q + p) / 2                       two independent verdicts
    INTERNAL  = (r + u + j) / 3                   what the reader's own states say
    STEPS     = u                                 the weakest step (gated on R)
    GROUNDING = n                                 support by the documents (gated on D)
    HARD      = c + a                             near-certain rule violations
Layer 2, decision (hand weights; gates multiply):
    score = 2.0 * JUDGES + 1.5 * INTERNAL + 1.0 * R * STEPS + 1.5 * D * GROUNDING + 4.0 * HARD
    DISAGREE = |JUDGES - INTERNAL|      not in the score: an uncertain case, routed to escalation (numeric recompute
                                        tier or a human) instead of being decided here
Output: score, the neurons that fired (each neuron's contribution), and whether to escalate.
"""
import numpy as np

W = {"JUDGES": 2.0, "INTERNAL": 1.5, "STEPS": 1.0, "GROUNDING": 1.5, "HARD": 4.0}


def _nz(v):
    return np.where(np.isnan(v), 0.0, v)


def circuit(r, u, j, q, p, n, c, a, D, R):
    """vectorised over replies; all inputs arrays of the same length. Returns (score, contributions, disagree)."""
    has_q, has_p = ~np.isnan(q), ~np.isnan(p)
    k = has_q.astype(float) + has_p.astype(float)
    judges = np.where(k > 0, (_nz(q) + _nz(p)) / np.maximum(k, 1), 0.0)
    internal = (_nz(r) + _nz(u) + _nz(j)) / 3
    steps = R * _nz(u)
    grounding = D * _nz(n)
    hard = _nz(c) + _nz(a)
    contrib = {"JUDGES": W["JUDGES"] * judges, "INTERNAL": W["INTERNAL"] * internal, "STEPS": W["STEPS"] * steps,
               "GROUNDING": W["GROUNDING"] * grounding, "HARD": W["HARD"] * hard}
    score = sum(contrib.values())
    disagree = np.where(k > 0, np.abs(judges - internal), 0.0)
    return score, contrib, disagree


def explain(contrib, i, top=3):
    """the neurons that pushed reply i toward "wrong", largest first."""
    items = sorted(((name, float(v[i])) for name, v in contrib.items()), key=lambda t: -t[1])
    return [f"{name} +{v:.2f}" for name, v in items[:top] if v > 0.25]


def is_reasoning(reply, n_units):
    import re
    return n_units >= 4 and bool(re.search(r"[=+*/^]|\bstep\b|\btherefore\b", reply, re.I))
