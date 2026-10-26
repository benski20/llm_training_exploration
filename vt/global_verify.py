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

    def cases(self, c, L):
        """Disjoint cases covering 'label != c'. For c=1 (unbalanced): the depth
        first reaches -1 at an even position k, or never goes negative but ends >= 2."""
        return [None] if c == 0 else [k for k in range(0, L, 2)] + [L]

    def encode_case(self, m, x, L, c, k):
        if k is None:
            return self.encode_not_label(m, x, L, c)
        def depth(t):
            e = []
            for j in range(t + 1):
                e += [(x[2 * j], 1.0), (x[2 * j + 1], -1.0)]
            return e
        last = k if k < L else L - 1
        for t in range(last):
            m.row(depth(t), 0.0, np.inf)
        if k < L:
            m.row(depth(k), -1.0, -1.0)
        else:
            m.row(depth(L - 1), 2.0, np.inf)

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

    def cases(self, c, L):
        """For c=1 (unsorted): the first descent is at position k."""
        return [None] if c == 0 else list(range(L - 1))

    def encode_case(self, m, x, L, c, k):
        if k is None:
            return self.encode_not_label(m, x, L, c)
        V = self.V
        val = lambda t: [(x[t * V + v], float(v)) for v in range(1, V)]
        neg = lambda e: [(i, -w) for i, w in e]
        for t in range(k):
            m.row(val(t + 1) + neg(val(t)), 0.0, np.inf)
        m.row(val(k) + neg(val(k + 1)), 1.0, np.inf)

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


def _query(P, spec, c, k, tau, bounds, time_limit):
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
    if hasattr(spec, "encode_case"):
        spec.encode_case(m, x, L, c, k)
    else:
        spec.encode_not_label(m, x, L, c)
    res = m.solve(s[:1], [0.0], time_limit)  # pure feasibility
    return res, x, n_bin + L * V


def _run_case(args):
    P, spec, c, k, tau, bounds, time_limit = args
    t0 = time.perf_counter()
    res, x, n_bin = _query(P, spec, c, k, tau, bounds, time_limit)
    q = {"class": c, "case": k, "milp_status": int(res.status), "binaries": n_bin}
    if res.x is not None:
        onehot = np.round(res.x[x]).reshape(P["L"], P["V"])
        seq = onehot.argmax(1)
        logits = np_forward(P, onehot)
        true = spec.label(seq)
        # a genuine counterexample: wrong prediction, or right but inside the tau margin
        margin = logits[c] - np.delete(logits, c).max()
        if true != c and margin >= -tau:
            q["result"] = "counterexample"
            q["cex"] = {"seq": seq.tolist(), "label": true,
                        "pred": int(np.argmax(logits)), "margin": float(margin)}
        else:
            q["result"] = "spurious"
    elif res.status == 2:  # infeasible: no bad input in this case
        q["result"] = "infeasible"
    else:
        q["result"] = "timeout"
    q["time"] = time.perf_counter() - t0
    return q


def verify_global(P, spec, tau=1e-4, time_limit=600.0, workers=1, stop_on_cex=True):
    """Prove global correctness, splitting each class query into the spec's
    disjoint cases (solved in parallel when workers > 1).

    Returns {"status": proved|counterexample|unknown, "counterexamples": [...], ...}."""
    t0 = time.perf_counter()
    bounds = global_bounds(P)
    unstable = int(sum(((l < 0) & (u > 0)).sum() for l, u in bounds["z"]))
    out = {"status": "proved", "counterexamples": [], "queries": [], "unstable": unstable}
    jobs = []
    for c in range(spec.n_classes):
        cases = spec.cases(c, P["L"]) if hasattr(spec, "cases") else [None]
        jobs += [(P, spec, c, k, tau, bounds, time_limit) for k in cases]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for q in pool.imap_unordered(_run_case, jobs):
                out["queries"].append(q)
                if stop_on_cex and q["result"] == "counterexample":
                    pool.terminate()
                    break
    else:
        for job in jobs:
            q = _run_case(job)
            out["queries"].append(q)
            if stop_on_cex and q["result"] == "counterexample":
                break
    out["counterexamples"] = [q["cex"] for q in out["queries"] if "cex" in q]
    results = {q["result"] for q in out["queries"]}
    if "counterexample" in results:
        out["status"] = "counterexample"
    elif results != {"infeasible"} or len(out["queries"]) < len(jobs):
        out["status"] = "unknown"
    out["cases"] = len(jobs)
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
