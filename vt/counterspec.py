"""Counter-automaton specs: beyond regular languages.

A deterministic counter automaton has finite control states and k integer
counters. Each (state, symbol) has a transition to a new state with a counter
update vector; a counter going negative kills the run. A string is accepted
if the run ends in an accepting state with every "zero" counter at 0.

    {"name": "anbncn", "alphabet": "abc", "start": "A", "counters": 2,
     "delta": {"A": {"a": ["A", [1, 0]], "b": ["B", [-1, 1]], "c": "dead"},
               "B": {"a": "dead", "b": ["B", [-1, 1]], "c": ["C", [0, -1]]},
               "C": {"a": "dead", "b": "dead", "c": ["C", [0, -1]]}},
     "accept": ["C"], "zero": [0, 1]}

(here a^n b^n c^n, n >= 1, which is not context-free). Counters are bounded by
the length, so the reachable state space at length L is polynomial and all of
vt's exact verifiers (classifiers and generators) apply unchanged: they only
need `dfa(L)` and `label`.
"""
import json

import numpy as np

DEAD = "dead"


class CounterSpec:
    n_classes = 2

    def __init__(self, d):
        self.name = d.get("name", "counter")
        self.alphabet = d["alphabet"]
        self.k = int(d.get("counters", 0))
        self.start = d["start"]
        self.accept = set(d["accept"])
        self.zero = list(d.get("zero", range(self.k)))
        self.delta = {}
        for s, row in d["delta"].items():
            if set(row) != set(self.alphabet):
                raise ValueError(f"state {s} must have a transition for every symbol")
            for ch, tr in row.items():
                if tr == DEAD:
                    self.delta[(s, ch)] = None
                else:
                    to, inc = tr
                    if to not in d["delta"] or len(inc) != self.k:
                        raise ValueError(f"bad transition {s} --{ch}--> {tr}")
                    self.delta[(s, ch)] = (to, tuple(int(i) for i in inc))
        if self.start not in d["delta"]:
            raise ValueError("start state has no transitions")
        self._raw = d

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def canonical(self):
        return json.dumps(self._raw, sort_keys=True)

    @property
    def V(self):
        return len(self.alphabet)

    def _step(self, q, v):
        if q == DEAD:
            return DEAD
        s, c = q
        tr = self.delta[(s, self.alphabet[v])]
        if tr is None:
            return DEAD
        to, inc = tr
        c2 = tuple(a + b for a, b in zip(c, inc))
        return DEAD if any(x < 0 for x in c2) else (to, c2)

    def _accepting(self, q):
        return q != DEAD and q[0] in self.accept and all(q[1][i] == 0 for i in self.zero)

    def dfa(self, L):
        return (self.start, (0,) * self.k), self._step, lambda q: int(self._accepting(q))

    def label(self, seq):
        q = (self.start, (0,) * self.k)
        for v in seq:
            q = self._step(q, int(v))
        return int(self._accepting(q))

    def label_batch(self, seqs):
        return np.array([self.label(s) for s in seqs], dtype=np.int64)

    def sample(self, n, L, rng):
        """Uniform strings, uniformly random accepted strings, and near misses."""
        k = n // 3
        uni = rng.integers(0, self.V, size=(k, L))
        pos = uniform_accepted(self, L, n - k, rng)
        near = pos[:k].copy()
        if len(near):
            near[np.arange(len(near)), rng.integers(L, size=len(near))] = rng.integers(0, self.V, size=len(near))
        return np.concatenate([uni, pos[k:], near]) if len(pos) else uni


def uniform_accepted(spec, L, n, rng):
    """n strings drawn uniformly from the length-L strings any `dfa`-style spec
    accepts (exact suffix counts over the reachable state layers). If the
    language is empty at this length, returns uniform random strings."""
    V = spec.V
    init, step, label = spec.dfa(L)
    layers = [[init]]
    for _ in range(L):
        seen = {}
        for q in layers[-1]:
            for v in range(V):
                seen.setdefault(step(q, v), None)
        layers.append(list(seen))
    count = [None] * (L + 1)
    count[L] = {q: label(q) for q in layers[L]}
    for t in range(L - 1, -1, -1):
        count[t] = {q: sum(count[t + 1][step(q, v)] for v in range(V)) for q in layers[t]}
    if count[0][init] == 0:
        return rng.integers(0, V, size=(n, L))
    out = np.zeros((n, L), dtype=np.int64)
    for i in range(n):
        q = init
        for t in range(L):
            w = np.array([float(count[t + 1][step(q, v)]) for v in range(V)])
            v = int(rng.choice(V, p=w / w.sum()))
            out[i, t] = v
            q = step(q, v)
    return out


def spec_from_def(d):
    """A spec object from its JSON definition: counter automaton or DFA."""
    if "counters" in d:
        return CounterSpec(d)
    from .dfaspec import DFASpec
    return DFASpec(d)
