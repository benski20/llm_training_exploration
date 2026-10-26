"""Exact, solver-free global verification for windowed-uniform mixing models.

If the (single) mixing block's causal kernel is free for offsets < w and
constant for every offset >= w,

    s'_t = E[x_t] + sum_{i<w} k_i E[x_{t-i}] + k_w * sum_v count_v(x_{<=t-w}) E[v]
                                                        (E[v] = W_in[:, v] + b_in)

then the block output h_t depends only on (x_t, the previous w-1 tokens, the
token counts of everything older). The logit difference logit_y - logit_j is
a constant plus a per-position sum (W_out[y] - W_out[j])_t . h_t, additive
along the sequence. Pairing that state with a DFA for the spec gives a
dynamic program over (counts, recent tokens, spec state) whose minimum is the
exact worst-case margin over all V^L inputs, for each true label.

Arithmetic is exact: float64 weights are dyadic rationals, so every quantity
is an integer at a fixed power-of-two scale (Python big ints). A "proved"
verdict therefore does not rely on any floating-point tolerance. Cost is
polynomial in L for fixed V and w (w = 1, V = 2: O(L^2) states).
"""
import time
from fractions import Fraction

import numpy as np


class StructureError(ValueError):
    """The artifact does not have the structure the exact verifier needs."""


def _fr(a):
    return np.vectorize(Fraction, otypes=[object])(np.asarray(a, dtype=np.float64))


def kernel(P, max_window=4):
    """Return (k, w): kernel values k[0..w] with k[w] shared by all offsets >= w."""
    if len(P["mix"]) != 1:
        raise StructureError("exact verifier needs exactly one block")
    M = np.asarray(P["mix"][0])
    L = P["L"]
    if np.any(np.triu(M, 1) != 0):
        raise StructureError("mixing is not causal")
    diags = []
    for off in range(L):
        dg = np.diag(M, -off)
        if np.any(dg != dg[0]):
            raise StructureError("mixing is not Toeplitz")
        diags.append(float(dg[0]))
    w = L
    while w > 1 and diags[w - 1] == diags[-1]:
        w -= 1
    # now diags[w:] are all equal; L == 1 leaves no shared tail
    if w > max_window:
        raise StructureError(f"kernel window {w} > {max_window}: state space would explode")
    return diags[:w] + [diags[w] if w < L else 0.0], w


class _Exact:
    """Exact integer arithmetic at scale 2^K for all model quantities."""

    def __init__(self, P, max_window=4):
        self.k, self.w = kernel(P, max_window)
        arrays = [P["W_in"], P["b_in"], P["W1"][0], P["b1"][0], P["W2"][0], P["b2"][0],
                  P["W_out"], P["b_out"], np.array(self.k)]
        K = 0
        for a in arrays:
            for f in np.asarray(a, dtype=np.float64).ravel():
                if not np.isfinite(f):
                    raise StructureError("non-finite weight")
                K = max(K, Fraction(float(f)).denominator.bit_length() - 1)
        self.K = K
        S = 1 << K
        self.S = S

        def ints(a):
            return np.vectorize(lambda f: int(Fraction(float(f)) * S), otypes=[object])(
                np.asarray(a, dtype=np.float64))
        E = ints(P["W_in"]).T + ints(P["b_in"])             # scale S,   (V, d)
        k = ints(self.k)                                     # scale S
        self.E, self.kk = E, k
        self.W1, self.b1 = ints(P["W1"][0]), ints(P["b1"][0])
        self.W2, self.b2 = ints(P["W2"][0]), ints(P["b2"][0])
        self.W_out, self.b_out = ints(P["W_out"]), ints(P["b_out"])
        self.cache = {}

    def h(self, v, recent, counts):
        """Block output at scale S^4 for current token v, recent tokens
        (oldest first, at most w-1) and counts of older tokens."""
        key = (v, recent, counts)
        out = self.cache.get(key)
        if out is None:
            S, E, k = self.S, self.E, self.kk
            s = (S + k[0]) * E[v]                            # scale S^2
            for i, u in enumerate(reversed(recent), start=1):
                s = s + k[i] * E[u]
            for u, c in enumerate(counts):
                if c:
                    s = s + (k[self.w] * c) * E[u]
            z = self.W1.dot(s) + self.b1 * S * S             # scale S^3
            r = np.array([zi if zi > 0 else 0 for zi in z], dtype=object)
            out = s * S * S + self.W2.dot(r) + self.b2 * S ** 3   # scale S^4
            self.cache[key] = out
        return out


def exact_forward(P, seq):
    """Exact rational logits for one sequence (any model; dense, slow)."""
    V = P["V"]
    x = np.eye(V, dtype=np.int64)[np.asarray(seq)]
    s = x.astype(object).dot(_fr(P["W_in"]).T) + _fr(P["b_in"])
    for M, W1, b1, W2, b2 in zip(P["mix"], P["W1"], P["b1"], P["W2"], P["b2"]):
        s = s + _fr(M).dot(s)
        z = s.dot(_fr(W1).T) + _fr(b1)
        z = np.vectorize(lambda q: q if q > 0 else Fraction(0), otypes=[object])(z)
        s = s + z.dot(_fr(W2).T) + _fr(b2)
    return _fr(P["W_out"]).dot(s.reshape(-1)) + _fr(P["b_out"])


def verify_exact(P, spec, tau=1e-4, max_states=20_000_000, max_window=4):
    """Exact proof (or exact counterexample) of global correctness.

    Returns {"status": "proved"|"counterexample", "min_margin": exact rational
    as a string, "counterexamples": [...], "states": n, "method": "exact-dp"}."""
    t0 = time.perf_counter()
    L, V, C, d = P["L"], P["V"], P["C"], P["d"]
    ex = _Exact(P, max_window)
    w = ex.w
    W = ex.W_out.reshape(C, L, d)
    init, step, label = spec.dfa(L)
    pairs = [(y, j) for y in range(C) for j in range(C) if j != y]
    frontier = {((0,) * V, (), init): {p: (0, None) for p in pairs}}
    back = []
    n_states = 0
    for t in range(L):
        nxt = {}
        for (counts, recent, q), vals in frontier.items():
            for v in range(V):
                h = ex.h(v, recent, counts)
                r2 = recent + (v,)
                c2 = counts
                if len(r2) > w - 1:              # oldest recent token leaves the window
                    old, r2 = r2[0], r2[1:]
                    c2 = tuple(c + (u == old) for u, c in enumerate(counts))
                key = (c2, r2, step(q, v))
                slot = nxt.setdefault(key, {})
                for (y, j) in pairs:
                    val = vals[(y, j)][0] + (W[y, t] - W[j, t]).dot(h)
                    cur = slot.get((y, j))
                    if cur is None or val < cur[0]:
                        slot[(y, j)] = (val, ((counts, recent, q), v))
        back.append(nxt)
        frontier = nxt
        n_states += len(nxt)
        if n_states > max_states:
            raise StructureError("state space too large for exact DP")
    S4 = ex.S ** 4
    worst, worst_key = None, None
    for key, vals in frontier.items():
        y = label(key[2])
        for j in range(C):
            if j == y:
                continue
            m = vals[(y, j)][0] + (ex.b_out[y] - ex.b_out[j]) * S4      # scale S^5
            if worst is None or m < worst:
                worst, worst_key = m, (key, (y, j))
    worst_q = Fraction(worst, ex.S ** 5)
    out = {"method": "exact-dp", "states": n_states, "window": w, "scale_bits": ex.K,
           "min_margin": str(worst_q), "min_margin_float": float(worst_q), "counterexamples": []}
    if worst_q > Fraction(tau):
        out["status"] = "proved"
    else:
        key, pair = worst_key
        seq = []
        for t in range(L - 1, -1, -1):
            _, bp = back[t][key][pair]
            key, v = bp
            seq.append(v)
        seq = seq[::-1]
        logits = exact_forward(P, seq)
        y = pair[0]
        margin = logits[y] - max(logits[j] for j in range(C) if j != y)
        assert spec.label(seq) == y and margin == worst_q, "exact DP inconsistent with forward pass"
        out["status"] = "counterexample"
        out["counterexamples"] = [{"seq": [int(v) for v in seq], "label": int(y),
                                   "pred": int(np.argmax([float(l) for l in logits])),
                                   "margin": float(margin)}]
    out["time"] = time.perf_counter() - t0
    return out
