"""Output guards: use a proved classifier to guarantee a property of generated text.

A GPT cannot be proved correct, but its *outputs* can be guaranteed: sample,
check each candidate with a guard model that is certified to equal the spec on
every input of that length, and return only accepted candidates. Because the
guard's certificate covers all V^L inputs, every returned string satisfies
the spec, with no exceptions. A merely accurate (tested) guard cannot promise
that.

    g = Guard("model.npz", "certificate.json")         # refuses unproved guards
    ok = g.accepts(["(()())()", "())(()()"])           # -> [True, False]
    outs = g.filter(candidates)                          # only spec-satisfying strings
"""
import numpy as np

from .serve import load_gated
from .verify import np_forward_batch


class Guard:
    def __init__(self, artifact, certificate, alphabet=None, allow_unverified=False):
        self.P, self.digest, self.cert = load_gated(artifact, certificate, allow_unverified)
        if alphabet is None:
            from .cli import TOKENS
            c = self.cert or {}
            alphabet = c["spec_dfa"]["alphabet"] if "spec_dfa" in c else TOKENS[c.get("spec", "dyck")]
        self.alphabet = alphabet
        self.L = self.P["L"]
        self.proved = bool(self.cert and self.cert.get("status") == "proved")

    def accepts(self, strings):
        """True where the guard predicts label 1. Strings of the wrong length or
        with foreign symbols are rejected without running the model."""
        out = np.zeros(len(strings), dtype=bool)
        idx, seqs = [], []
        for i, s in enumerate(strings):
            if len(s) == self.L and all(ch in self.alphabet for ch in s):
                idx.append(i)
                seqs.append([self.alphabet.index(ch) for ch in s])
        if seqs:
            logits = np_forward_batch(self.P, np.eye(self.P["V"])[np.array(seqs)])
            out[idx] = logits.argmax(1) == 1
        return out

    def filter(self, strings):
        keep = self.accepts(strings)
        return [s for s, k in zip(strings, keep) if k]
