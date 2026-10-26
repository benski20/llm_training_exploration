"""Statistically certified judges for properties with no formal definition.

Harm and tone beyond what a lexicon can capture need a learned judge, which
can't be proved correct. What it can carry is a distribution-free statistical
guarantee. Split-conformal calibration on held-out *bad* examples (e.g.
harmful responses) picks a threshold tau so that, for a new bad example
exchangeable with the calibration set,

    P(score(bad) < tau) <= alpha,

i.e. at most an alpha fraction of bad outputs pass the judge. It's a marginal
guarantee under exchangeability, not a proof about any single output, and
reports must say so.
"""
import math

import numpy as np


class ConformalJudge:
    def __init__(self, score_fn, name="judge"):
        self.score_fn, self.name = score_fn, name
        self.tau = None
        self.alpha = None
        self.n_cal = 0

    def calibrate(self, bad_examples, alpha):
        """bad_examples: held-out outputs known to be bad (labelled)."""
        s = np.sort(np.asarray([self.score_fn(x) for x in bad_examples], dtype=float))
        n = len(s)
        k = math.floor(alpha * (n + 1))          # number of bad scores allowed below tau
        if k < 1:
            raise ValueError(f"need at least {math.ceil(1 / alpha) - 1} calibration examples for alpha={alpha}")
        self.tau = s[k - 1]                      # pass iff score < tau
        self.alpha, self.n_cal = alpha, n
        return self.tau

    def passes(self, x):
        if self.tau is None:
            raise RuntimeError("judge is not calibrated")
        return self.score_fn(x) < self.tau

    def guarantee(self):
        return {"judge": self.name, "type": "split-conformal (exchangeability)",
                "bound": f"P(a bad output passes) <= {self.alpha}", "calibration_n": self.n_cal}
