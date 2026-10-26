"""Structural guarantees of the typed, bottlenecked, pointer-copying answerer."""
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import grounding_by_design as G  # noqa: E402
import verifiable_lm as V  # noqa: E402


def _model():
    G.NAMES, G.POS = "single", "alibi"
    torch.manual_seed(0)
    return V.VerifiableLM(G.BLOCK).eval()


def test_decision_is_typed_and_a_function_of_the_channel():
    m = _model()
    _, x, _, dpos = V.batch(random.Random(0), 4, 1, 4)
    for v in (-3.0, 0.5, 2.0):
        mask = torch.arange(x.shape[1])[None, :] == dpos[:, None]
        m.write = (mask, v)
        with torch.no_grad():
            logp, _, s, d = m(x, dpos)
        m.write = None
        p = logp[torch.arange(4), dpos].exp()
        allowed = torch.zeros(256, dtype=torch.bool)
        allowed[V.DIGITS] = True
        allowed[V.U] = True
        assert float(p[:, ~allowed].sum()) < 1e-6                   # only ABSTAIN or a copied digit
        assert torch.allclose(s, torch.full((4,), v))
        want = torch.sigmoid(m.beta - F.softplus(m.alpha) * v)
        assert torch.allclose(p[:, V.U], want.expand(4), atol=1e-5)  # P(abstain) depends on s alone


def test_digits_are_emitted_as_pointers():
    m = _model()
    rng = random.Random(1)
    for _ in range(5):
        prompt, _, _ = G.example(rng, 1, 4)
        o, src, _, _ = V.answer(m, prompt, force=-50.0)              # low s: abstain
        assert o == "unknown" and src == []
        o, src, _, _ = V.answer(m, prompt, force=50.0)
        digits = [c for c in o if c.isdigit()]
        assert len(digits) == len(src)
        assert all(prompt[j] == c for j, c in zip(src, digits))     # every digit is x[source]
