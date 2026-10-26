"""Global functional-correctness verification over the whole discrete domain.

Claim proved: for EVERY sequence in V^L, the model's predicted class equals the
specification's label, with logit margin > tau.

For each class c we ask the MILP solver for an input whose true label is not c
but on which the model predicts c with margin >= -tau. If every such query is
infeasible the model is proved correct on all V^L inputs. A feasible solution
is a counterexample, which is re-checked by a float64 forward pass and the
reference spec.
"""
import time

import numpy as np

from .bounds import OneHot, symbolic_bounds
from .verify import _MILP, encode_network, ibp_from_s0, np_forward


class DyckSpec:
    """Dyck-1 over tokens {0: '(', 1: ')'}; label 1 iff balanced."""
    n_classes = 2

    def label(self, seq):
        depth = 0
        for t in seq:
            depth += 1 if t == 0 else -1
            if depth < 0:
                return 0
        return int(depth == 0)

    def label_batch(self, seqs):
        depth = np.cumsum(1 - 2 * seqs, axis=1)
        return ((depth.min(1) >= 0) & (depth[:, -1] == 0)).astype(np.int64)

    def sample(self, n, L, rng):
        """A mix that covers easy and hard regions of the input space."""
        k = n // 4
        uni = rng.integers(0, 2, size=(k, L))
        perm = np.argsort(rng.random((k, L)), axis=1) >= L // 2   # equal counts
        bal = self._balanced(k, L, rng)
        flip = self._balanced(n - 3 * k, L, rng)
        i = rng.integers(L, size=len(flip))
        flip[np.arange(len(flip)), i] ^= 1
        return np.concatenate([uni, perm.astype(np.int64), bal, flip])

    @staticmethod
    def _balanced(n, L, rng):
        """Uniform-ish balanced strings via random walks conditioned to stay >= 0."""
        out = np.zeros((n, L), dtype=np.int64)
        depth = np.zeros(n, dtype=np.int64)
        for t in range(L):
            left = L - t
            must_close = depth >= left          # need all remaining to be ')'
            can_close = depth > 0
            close = np.where(must_close, True,
                             np.where(can_close, rng.random(n) < 0.5, False))
            out[:, t] = close
            depth += np.where(close, -1, 1)
        return out

    def encode_not_label(self, m, x, L, c):
        """Constrain the one-hot input variables x (L*2) to strings whose label != c.

        D_t = sum_{k<=t} (x[k,0] - x[k,1]) is the prefix depth (an integer)."""
        def depth(t):  # entries of D_t (positions 0..t inclusive)
            e = []
            for k in range(t + 1):
                e += [(x[2 * k], 1.0), (x[2 * k + 1], -1.0)]
            return e
        if c == 0:
            # label != 0  <=>  balanced: all prefixes >= 0 and final depth == 0
            for t in range(L - 1):
                m.row(depth(t), 0.0, np.inf)
            m.row(depth(L - 1), 0.0, 0.0)
        else:
            # label != 1  <=>  unbalanced: some prefix <= -1, or final depth >= 2
            g = m.vars([0.0] * (L + 1), [1.0] * (L + 1), integer=True)
            m.row([(gi, 1.0) for gi in g], 1.0, np.inf)
            for t in range(L):
                big = t + 2.0  # D_t <= t + 1, so D_t + 1 <= big always holds
                m.row(depth(t) + [(g[t], big)], -np.inf, -1.0 + big)
            big = L + 2.0
            m.row(depth(L - 1) + [(g[L], -big)], 2.0 - big, np.inf)


class SortedSpec:
    """Tokens 0..V-1; label 1 iff the sequence is non-decreasing."""
    n_classes = 2

    def __init__(self, V=4):
        self.V = V

    def label(self, seq):
        return int(all(a <= b for a, b in zip(seq[:-1], seq[1:])))

    def label_batch(self, seqs):
        return (np.diff(seqs, axis=1) >= 0).all(1).astype(np.int64)

    def sample(self, n, L, rng):
        k = n // 3
        uni = rng.integers(0, self.V, size=(k, L))
        srt = np.sort(rng.integers(0, self.V, size=(k, L)), axis=1)
        near = np.sort(rng.integers(0, self.V, size=(n - 2 * k, L)), axis=1)
        i = rng.integers(L, size=len(near))
        near[np.arange(len(near)), i] = rng.integers(0, self.V, size=len(near))
        return np.concatenate([uni, srt, near])

    def encode_not_label(self, m, x, L, c):
        V = self.V
        val = lambda t: [(x[t * V + v], float(v)) for v in range(1, V)]
        neg = lambda e: [(i, -w) for i, w in e]
        if c == 0:  # sorted: v_{t+1} - v_t >= 0 for all t
            for t in range(L - 1):
                m.row(val(t + 1) + neg(val(t)), 0.0, np.inf)
        else:       # unsorted: some t with v_t - v_{t+1} >= 1
            g = m.vars([0.0] * (L - 1), [1.0] * (L - 1), integer=True)
            m.row([(gi, 1.0) for gi in g], 1.0, np.inf)
            big = float(V)  # v_t - v_{t+1} >= -(V-1), so >= 1 - big when g = 0
            for t in range(L - 1):
                m.row(val(t) + neg(val(t + 1)) + [(g[t], -big)], 1.0 - big, np.inf)


SPECS = {"dyck": DyckSpec, "sorted": SortedSpec}


def global_bounds(P):
    """Interval bounds over all one-hot inputs: each position's embedding is
    exactly one column of W_in, so the box is the column-wise min/max."""
    L = P["L"]
    cols = P["W_in"].T + P["b_in"]  # (V, d)
    lo, hi = cols.min(0), cols.max(0)
    c = np.tile((lo + hi) / 2, (L, 1))
    r = np.tile((hi - lo) / 2, (L, 1))
    n_in = L * P["V"]
    ibp = ibp_from_s0(P, c, r, (np.zeros(n_in), np.ones(n_in)))
    return symbolic_bounds(P, OneHot(L, P["V"]), ibp)


def _query(P, spec, c, tau, bounds, time_limit):
    L, V, C = P["L"], P["V"], P["C"]
    m = _MILP()
    x = m.vars(*bounds["in"], integer=True)
    for t in range(L):
        m.row([(x[t * V + v], 1.0) for v in range(V)], 1.0, 1.0)
    s, n_bin = encode_network(m, P, x, bounds)
    # model predicts c (ties count against us): logit_c - logit_j >= -tau for all j
    for j in range(C):
        if j == c:
            continue
        coef = P["W_out"][c] - P["W_out"][j]
        const = P["b_out"][c] - P["b_out"][j]
        m.row([(si, w) for si, w in zip(s, coef) if w != 0], -tau - const, np.inf)
    spec.encode_not_label(m, x, L, c)
    res = m.solve(s[:1], [0.0], time_limit)  # pure feasibility
    return res, x, n_bin + L * V


def verify_global(P, spec, tau=1e-4, time_limit=600.0):
    """Returns {"status": proved|counterexample|unknown, "counterexamples": [...], ...}."""
    t0 = time.perf_counter()
    bounds = global_bounds(P)
    unstable = int(sum(((l < 0) & (u > 0)).sum() for l, u in bounds["z"]))
    out = {"status": "proved", "counterexamples": [], "queries": [], "unstable": unstable}
    for c in range(spec.n_classes):
        left = max(1.0, time_limit - (time.perf_counter() - t0))
        res, x, n_bin = _query(P, spec, c, tau, bounds, left)
        q = {"class": c, "milp_status": int(res.status), "binaries": n_bin,
             "time": None}
        if res.x is not None:
            onehot = np.round(res.x[x]).reshape(P["L"], P["V"])
            seq = onehot.argmax(1)
            logits = np_forward(P, onehot)
            pred = int(np.argmax(logits))
            true = spec.label(seq)
            # a genuine counterexample: wrong prediction, or right but inside the tau margin
            margin = logits[c] - np.delete(logits, c).max()
            if true != c and margin >= -tau:
                out["counterexamples"].append({"seq": seq.tolist(), "label": true,
                                               "pred": pred, "margin": float(margin)})
                q["result"] = "counterexample"
            else:
                q["result"] = "spurious"
        elif res.status == 2:  # infeasible: no bad input for this class
            q["result"] = "infeasible"
        else:
            q["result"] = "timeout"
        q["time"] = time.perf_counter() - t0
        out["queries"].append(q)
    results = {q["result"] for q in out["queries"]}
    if "counterexample" in results:
        out["status"] = "counterexample"
    elif results != {"infeasible"}:
        out["status"] = "unknown"
    out["time"] = time.perf_counter() - t0
    return out


def exhaustive_check(P, spec, tau=1e-4, chunk=1 << 16):
    """Independent check by enumerating all V^L inputs (feasible for small L).
    Returns (n_errors, min_correct_margin, first_error_seq or None)."""
    from .verify import np_forward_batch
    L, V = P["L"], P["V"]
    total = V ** L
    n_err, min_margin, first = 0, np.inf, None
    eye = np.eye(V)
    powers = V ** np.arange(L)[::-1]
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk))
        seqs = (idx[:, None] // powers) % V
        logits = np_forward_batch(P, eye[seqs])
        y = spec.label_batch(seqs)
        true_logit = logits[np.arange(len(y)), y]
        logits[np.arange(len(y)), y] = -np.inf
        margin = true_logit - logits.max(1)
        bad = margin <= tau
        if bad.any() and first is None:
            first = seqs[np.argmax(bad)].tolist()
        n_err += int(bad.sum())
        min_margin = min(min_margin, float(margin.min()))
    return n_err, min_margin, first
