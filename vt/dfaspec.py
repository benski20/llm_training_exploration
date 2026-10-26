"""User-defined specifications: any regular language, given as a DFA in JSON.

    {"name": "even_a", "alphabet": "ab", "start": "even",
     "accept": ["even"],
     "delta": {"even": {"a": "odd", "b": "even"}, "odd": {"a": "even", "b": "odd"}}}

label(seq) = 1 iff the DFA accepts. DFA specs are verified with the exact
verifier (vt/exact.py), so train them with --uniform (and --window if the
language depends on neighbouring tokens).
"""
import json

import numpy as np


class DFASpec:
    n_classes = 2

    def __init__(self, d):
        self.name = d.get("name", "dfa")
        self.alphabet = d["alphabet"]
        states = sorted(d["delta"])
        if d["start"] not in d["delta"]:
            raise ValueError("start state has no transitions")
        self.index = {s: i for i, s in enumerate(states)}
        self.states = states
        V = len(self.alphabet)
        T = np.zeros((len(states), V), dtype=np.int64)
        for s, row in d["delta"].items():
            for ch, s2 in row.items():
                if ch not in self.alphabet or s2 not in self.index:
                    raise ValueError(f"bad transition {s} --{ch}--> {s2}")
                T[self.index[s], self.alphabet.index(ch)] = self.index[s2]
            if set(row) != set(self.alphabet):
                raise ValueError(f"state {s} must have a transition for every symbol")
        self.T = T
        self.start = self.index[d["start"]]
        self.accept = np.zeros(len(states), dtype=bool)
        for s in d["accept"]:
            self.accept[self.index[s]] = True

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    @property
    def V(self):
        return len(self.alphabet)

    def label(self, seq):
        q = self.start
        for v in seq:
            q = self.T[q, int(v)]
        return int(self.accept[q])

    def label_batch(self, seqs):
        q = np.full(len(seqs), self.start)
        for t in range(seqs.shape[1]):
            q = self.T[q, seqs[:, t]]
        return self.accept[q].astype(np.int64)

    def dfa(self, L):
        return self.start, lambda q, v: int(self.T[q, v]), lambda q: int(self.accept[q])

    def _accept_counts(self, L):
        """N[r, q] = number of length-r suffixes from state q that end accepted
        (float64: exact enough for sampling weights, fine up to V^L ~ 1e308)."""
        N = np.zeros((L + 1, len(self.states)))
        N[0] = self.accept
        for r in range(1, L + 1):
            N[r] = N[r - 1][self.T].sum(1)
        return N

    def sample(self, n, L, rng):
        """Uniform strings, uniformly random accepted strings (via suffix counts),
        and accepted strings with one token changed (hard negatives)."""
        k = n // 3
        uni = rng.integers(0, self.V, size=(k, L))
        N = self._accept_counts(L)
        m = n - k
        if N[L, self.start] == 0:      # empty language at this length
            pos = rng.integers(0, self.V, size=(m, L))
        else:
            pos = np.zeros((m, L), dtype=np.int64)
            q = np.full(m, self.start)
            for t in range(L):
                w = N[L - t - 1][self.T[q]]                  # (m, V)
                cdf = np.cumsum(w, 1) / w.sum(1, keepdims=True)
                v = (rng.random((m, 1)) > cdf).sum(1)
                pos[:, t] = v
                q = self.T[q, v]
        near = pos[:k].copy()
        near[np.arange(k), rng.integers(L, size=k)] = rng.integers(0, self.V, size=k)
        return np.concatenate([uni, pos[k:], near])
