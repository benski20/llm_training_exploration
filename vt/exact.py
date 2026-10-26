"""Exact, solver-free global verification for uniform-mixing models.

If the (single) mixing block is uniform, M = a*I + b*(strictly lower ones), then
the state at position t is

    s'_t = (1 + a) E[x_t] + b * sum_v count_v(x_<t) E[v]      (E[v] = W_in[:, v] + b_in)

so the block output h_t depends only on (t, x_t, prefix token counts). The
logit difference logit_y - logit_j is a constant plus a sum over positions of
(W_out[y] - W_out[j])_t . h_t, which is additive along the sequence. Pairing
the counts with a DFA for the spec gives a dynamic program over
(t, counts, spec state) whose minimum is the exact worst-case margin over all
V^L inputs, for each true label.

All arithmetic is in exact rationals (float64 weights are converted exactly),
so a "proved" verdict does not rely on any floating-point solver tolerance.
Cost is polynomial in L: O(L * #count-vectors * #spec-states * V) block
evaluations, e.g. about 33k for Dyck at L=256.
"""
import time
from fractions import Fraction

import numpy as np


class StructureError(ValueError):
    """The artifact does not have the structure the exact verifier needs."""


def _fr(a):
    return np.vectorize(Fraction, otypes=[object])(np.asarray(a, dtype=np.float64))


def uniform_params(P):
    """Return (a, b) if P has one uniform causal mixing block, else raise."""
    if len(P["mix"]) != 1:
        raise StructureError("exact verifier needs exactly one block")
    M = np.asarray(P["mix"][0])
    L = P["L"]
    if np.any(np.triu(M, 1) != 0):
        raise StructureError("mixing is not causal")
    diag = np.diag(M)
    low = M[np.tril_indices(L, -1)]
    if np.any(diag != diag[0]) or (len(low) and np.any(low != low[0])):
        raise StructureError("mixing is not uniform (a*I + b*strict-lower)")
    return float(diag[0]), float(low[0]) if len(low) else 0.0


def _count_vectors(t, V):
    """All count vectors of length V summing to t."""
    if V == 1:
        yield (t,)
        return
    for c0 in range(t + 1):
        for rest in _count_vectors(t - c0, V - 1):
            yield (c0,) + rest


class _Block:
    """Exact evaluation of h_t for (x_t, prefix counts), cached."""

    def __init__(self, P):
        a, b = uniform_params(P)
        self.a, self.b = Fraction(a), Fraction(b)
        self.E = _fr(P["W_in"]).T + _fr(P["b_in"])         # (V, d): E[v]
        self.W1, self.b1 = _fr(P["W1"][0]), _fr(P["b1"][0])
        self.W2, self.b2 = _fr(P["W2"][0]), _fr(P["b2"][0])
        self.cache = {}

    def h(self, v, counts):
        key = (v, counts)
        out = self.cache.get(key)
        if out is None:
            s = (1 + self.a) * self.E[v]
            for u, c in enumerate(counts):
                if c:
                    s = s + self.b * c * self.E[u]
            z = self.W1.dot(s) + self.b1
            r = np.array([zi if zi > 0 else Fraction(0) for zi in z], dtype=object)
            out = s + self.W2.dot(r) + self.b2
            self.cache[key] = out
        return out


def exact_forward(P, seq):
    """Exact rational logits for one sequence (any model; dense, slow)."""
    L, V = P["L"], P["V"]
    x = np.eye(V, dtype=np.int64)[np.asarray(seq)]
    s = x.astype(object).dot(_fr(P["W_in"]).T) + _fr(P["b_in"])
    for M, W1, b1, W2, b2 in zip(P["mix"], P["W1"], P["b1"], P["W2"], P["b2"]):
        s = s + _fr(M).dot(s)
        z = s.dot(_fr(W1).T) + _fr(b1)
        z = np.vectorize(lambda q: q if q > 0 else Fraction(0), otypes=[object])(z)
        s = s + z.dot(_fr(W2).T) + _fr(b2)
    return _fr(P["W_out"]).dot(s.reshape(-1)) + _fr(P["b_out"])


def verify_exact(P, spec, tau=1e-4, max_states=5_000_000):
    """Exact proof (or exact counterexample) of global correctness.

    Returns {"status": "proved"|"counterexample", "min_margin": Fraction as str,
    "counterexamples": [...], "states": n, "time": s, "method": "exact-dp"}."""
    t0 = time.perf_counter()
    L, V, C, d = P["L"], P["V"], P["C"], P["d"]
    blk = _Block(P)
    W = _fr(P["W_out"]).reshape(C, L, d)
    bias = _fr(P["b_out"])
    tau = Fraction(tau)
    init, step, label = spec.dfa(L)
    pairs = [(y, j) for y in range(C) for j in range(C) if j != y]
    zero = (0,) * V
    # layer[t]: {(counts, q): {pair: (value, backpointer)}}
    frontier = {(zero, init): {p: (Fraction(0), None) for p in pairs}}
    back = []
    n_states = 0
    for t in range(L):
        nxt = {}
        for (counts, q), vals in frontier.items():
            for v in range(V):
                q2 = step(q, v)
                c2 = tuple(c + (u == v) for u, c in enumerate(counts))
                h = blk.h(v, counts)
                key = (c2, q2)
                slot = nxt.setdefault(key, {})
                for (y, j) in pairs:
                    val = vals[(y, j)][0] + (W[y, t] - W[j, t]).dot(h)
                    if (y, j) not in slot or val < slot[(y, j)][0]:
                        slot[(y, j)] = (val, ((counts, q), v))
        back.append(nxt)
        frontier = nxt
        n_states += len(nxt)
        if n_states > max_states:
            raise StructureError("state space too large for exact DP")
    worst, worst_key = None, None
    for (counts, q), vals in frontier.items():
        y = label(q)
        for j in range(C):
            if j == y:
                continue
            m = vals[(y, j)][0] + bias[y] - bias[j]
            if worst is None or m < worst:
                worst, worst_key = m, ((counts, q), (y, j))
    out = {"method": "exact-dp", "states": n_states, "min_margin": str(worst),
           "min_margin_float": float(worst), "counterexamples": []}
    if worst > tau:
        out["status"] = "proved"
    else:
        # walk back-pointers to the worst input
        (key, pair) = worst_key
        seq = []
        for t in range(L - 1, -1, -1):
            _, bp = back[t][key][pair]
            key, v = bp
            seq.append(v)
        seq = seq[::-1]
        logits = exact_forward(P, seq)
        y = pair[0]
        margin = logits[y] - max(logits[j] for j in range(C) if j != y)
        assert spec.label(seq) == y and margin == worst, "exact DP inconsistent with forward pass"
        out["status"] = "counterexample"
        out["counterexamples"] = [{"seq": [int(v) for v in seq], "label": int(y),
                                   "pred": int(np.argmax([float(l) for l in logits])),
                                   "margin": float(margin)}]
    out["time"] = time.perf_counter() - t0
    return out
