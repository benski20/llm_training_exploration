"""Counter-automaton specs (beyond regular): semantics, uniform sampler, exact proofs."""
import itertools
import os

import numpy as np
import torch

from vt.counterspec import CounterSpec, uniform_accepted
from vt.exact import verify_exact
from vt.global_verify import exhaustive_check
from vt.model import MixerNet

SPECS = os.path.join(os.path.dirname(__file__), "..", "specs")


def test_anbncn_semantics():
    s = CounterSpec.load(os.path.join(SPECS, "anbncn.json"))
    for L in range(1, 10):
        for seq in itertools.product(range(3), repeat=L):
            w = "".join("abc"[v] for v in seq)
            n = L // 3
            assert s.label(seq) == int(L % 3 == 0 and n > 0 and w == "a" * n + "b" * n + "c" * n)


def test_uniform_accepted_is_uniform():
    s = CounterSpec.load(os.path.join(SPECS, "anbn.json"))
    from vt.global_verify import DyckSpec

    class D(DyckSpec):
        V = 2
    d = D()
    x = uniform_accepted(d, 8, 14000, np.random.default_rng(0))    # Catalan(4) = 14 strings
    assert d.label_batch(x).all()
    _, counts = np.unique(x, axis=0, return_counts=True)
    assert len(counts) == 14 and counts.min() > 850
    assert s.label_batch(uniform_accepted(s, 10, 50, np.random.default_rng(1))).all()


def test_exact_verifier_agrees_with_enumeration_on_counter_spec():
    s, L = CounterSpec.load(os.path.join(SPECS, "anbn.json")), 8
    seqs = np.array(list(itertools.product(range(2), repeat=L)))
    X = torch.tensor(np.eye(2, dtype=np.float32)[seqs])
    Y = torch.tensor(s.label_batch(seqs))
    for steps in (3, 1500):
        torch.manual_seed(0)
        m = MixerNet(L, 2, 4, 8, 1, 2, uniform=True, window=2)
        opt = torch.optim.Adam(m.parameters(), 3e-2)
        w = torch.where(Y == 1, 100.0, 1.0)
        for _ in range(steps):
            loss = (torch.nn.functional.cross_entropy(m(X), Y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        P = m.export()
        n_err, min_margin, _ = exhaustive_check(P, s)
        r = verify_exact(P, s)
        assert (r["status"] == "proved") == (n_err == 0)
        assert abs(r["min_margin_float"] - min_margin) < 1e-9
