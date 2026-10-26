"""Exact DP verifier: agreement with enumeration, exact forward pass and MILP."""
import itertools
from fractions import Fraction

import numpy as np
import pytest
import torch

from test_global import hand_built_dyck
from vt.exact import StructureError, exact_forward, verify_exact
from vt.global_verify import DyckSpec, SortedSpec, exhaustive_check, verify_global
from vt.model import MixerNet
from vt.verify import np_forward


def test_dfa_matches_spec():
    for spec, V, L in [(DyckSpec(), 2, 8), (SortedSpec(3), 3, 5)]:
        init, step, label = spec.dfa(L)
        for seq in itertools.product(range(V), repeat=L):
            q = init
            for v in seq:
                q = step(q, v)
            assert label(q) == spec.label(seq)


def test_exact_forward_matches_float():
    torch.manual_seed(0)
    P = MixerNet(6, 2, 4, 6, 2, 2).export()
    x = [0, 1, 1, 0, 1, 0]
    assert np.allclose([float(v) for v in exact_forward(P, x)], np_forward(P, np.eye(2)[x]))


def test_hand_built_proved_exactly_at_long_lengths():
    for L in (16, 64, 128):
        r = verify_exact(hand_built_dyck(L), DyckSpec())
        assert r["status"] == "proved" and Fraction(r["min_margin"]) == Fraction(1, 2)


def test_broken_model_exact_counterexample():
    P = hand_built_dyck(12)
    P["W_out"][1, (12 - 1) * 3 + 2] = 0.0
    r = verify_exact(P, DyckSpec())
    assert r["status"] == "counterexample"
    c = r["counterexamples"][0]
    assert DyckSpec().label(c["seq"]) != c["pred"]


def _train(L, V, spec, steps, seed):
    torch.manual_seed(seed)
    seqs = np.array(list(itertools.product(range(V), repeat=L)))
    X = torch.tensor(np.eye(V, dtype=np.float32)[seqs])
    Y = torch.tensor(spec.label_batch(seqs))
    m = MixerNet(L, V, 4, 8, 1, 2, uniform=True)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    w = torch.where(Y == 1, float((Y == 0).sum() / max(1, (Y == 1).sum())), 1.0)
    for _ in range(steps):
        loss = (torch.nn.functional.cross_entropy(m(X), Y, reduction="none") * w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.export()


@pytest.mark.parametrize("spec,V,L", [(DyckSpec(), 2, 10), (SortedSpec(3), 3, 6)])
def test_exact_agrees_with_enumeration_and_milp(spec, V, L):
    outcomes = set()
    for steps, seed in [(5, 0), (300, 1), (1500, 2)]:
        P = _train(L, V, spec, steps, seed)
        n_err, min_margin, _ = exhaustive_check(P, spec)
        r = verify_exact(P, spec)
        outcomes.add(r["status"])
        assert (r["status"] == "proved") == (n_err == 0)
        assert abs(r["min_margin_float"] - min_margin) < 1e-9     # exact worst case
        m = verify_global(P, spec, time_limit=60)
        assert m["status"] == r["status"]
    assert "counterexample" in outcomes


def test_rejects_non_uniform_models():
    torch.manual_seed(0)
    with pytest.raises(StructureError):
        verify_exact(MixerNet(6, 2, 4, 6, 1, 2, shared=True).export(), DyckSpec())


@pytest.mark.parametrize("window", [2, 3])
def test_windowed_kernel_exact_matches_enumeration(window):
    spec, V, L = SortedSpec(3), 3, 6
    seqs = np.array(list(itertools.product(range(V), repeat=L)))
    X = torch.tensor(np.eye(V, dtype=np.float32)[seqs])
    Y = torch.tensor(spec.label_batch(seqs))
    for steps in (3, 800):
        torch.manual_seed(window)
        m = MixerNet(L, V, 4, 8, 1, 2, uniform=True, window=window)
        opt = torch.optim.Adam(m.parameters(), 3e-2)
        for _ in range(steps):
            loss = torch.nn.functional.cross_entropy(m(X), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        P = m.export()
        n_err, min_margin, _ = exhaustive_check(P, spec)
        r = verify_exact(P, spec)
        assert r["window"] == window
        assert (r["status"] == "proved") == (n_err == 0)
        assert abs(r["min_margin_float"] - min_margin) < 1e-9


def test_warm_start_and_extension_preserve_the_function():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from length_generalization import truncate
    torch.manual_seed(0)
    m = MixerNet(10, 2, 4, 8, 1, 2, uniform=True)
    P = m.export()
    big = MixerNet(20, 2, 4, 8, 1, 2, uniform=True)
    big.load_tied_artifact(P)
    x = np.eye(2)[[0, 1, 1, 0, 0, 0, 1, 1, 0, 1]]
    assert np.allclose(np_forward(big.export(10), x), np_forward(P, x), atol=1e-6)
    y = np.eye(2)[[0, 1] * 8]
    ext = truncate(P, 16)                     # extend the artifact itself
    assert np.allclose(np_forward(ext, y), np_forward(big.export(16), y), atol=1e-6)
