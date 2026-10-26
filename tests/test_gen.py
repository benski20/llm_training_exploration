"""Count-state generators: implementations agree; exact certificates match brute force."""
import itertools
from fractions import Fraction

import numpy as np
import pytest
import torch

from vt.gen import CountGen, np_scores
from vt.gen_train import GenConfig, _layer_tensors, _state_scores, train_generator
from vt.gen_verify import _ExactGen, read_generator, save_generator, state_graph, verify_generator
from vt.global_verify import DyckSpec, SortedSpec


@pytest.mark.parametrize("w", [1, 2])
def test_scores_agree_torch_numpy_exact_and_graph(w):
    torch.manual_seed(w)
    m = CountGen(3, 4, 6, window=w).double()
    with torch.no_grad():
        m.u.normal_(); m.u0.normal_(); m.kern.normal_()
    G = m.export()
    ex = _ExactGen(G)
    L = 7
    g = state_graph(3, w, L, SortedSpec(3))
    tens = _layer_tensors(g, 3, w)
    rng = np.random.default_rng(0)
    for _ in range(30):
        x = rng.integers(3, size=L)
        tor = m.scores(torch.tensor(x[None]), L)[0].detach().numpy()
        npy = np_scores(G, x, L)
        assert np.allclose(tor, npy, atol=1e-9)
        # walk the state graph along x and compare exact and batched-graph scores
        i = 0
        for t in range(L):
            st = g["layers"][t][i]
            exs = np.array([float(Fraction(v, ex.S ** (1 if t == 0 else 4))) for v in ex.scores(st, L - t)])
            assert np.allclose(exs, npy[t], atol=1e-9)
            cur, rec, cnt = (a[i:i + 1] for a in tens[t])
            z = _state_scores(m, cur, rec, cnt.double(), L - t).detach().numpy()[0]
            assert np.allclose(np.maximum(z, 0), npy[t], atol=1e-9)
            i = int(g["trans"][t][i, x[t]])


def _brute(G, spec, L):
    """Enumerate all V^L strings: support soundness/completeness and exact P(violation)."""
    from vt.gen_verify import _ExactGen
    V = G["V"]
    ex = _ExactGen(G)
    sound = complete = True
    pviol = Fraction(0)
    g = state_graph(V, G["w"], L, spec)
    for seq in itertools.product(range(V), repeat=L):
        p, i = Fraction(1), 0
        for t, v in enumerate(seq):
            sc = ex.scores(g["layers"][t][i], L - t)
            tot = sum(sc)
            p = p * (Fraction(sc[v], tot) if tot else 0)
            i = int(g["trans"][t][i, v])
        lab = spec.label(seq)
        if p > 0 and not lab:
            sound = False
            pviol += p
        if p == 0 and lab:
            complete = False
    return sound, complete, pviol


@pytest.mark.parametrize("lam", [0.0, 10.0])
def test_certificate_matches_brute_force(lam):
    spec, L = DyckSpec(), 8
    m, _ = train_generator(spec, GenConfig(L=L, steps=300, lam=lam, n_data=2000), 2,
                           log=lambda e: None)
    G = m.export()
    r = verify_generator(G, spec, L, exact_mass=True)
    sound, complete, pviol = _brute(G, spec, L)
    assert r["sound"] == sound and r["complete"] == complete
    assert Fraction(r["p_violation"]) == pviol


def test_untrained_generator_violation_mass_is_exact():
    torch.manual_seed(3)
    G = CountGen(2, 4, 6).export()
    G["bo"][:] = 1.0                                   # make every token possible
    r = verify_generator(G, DyckSpec(), 6, exact_mass=True)
    sound, complete, pviol = _brute(G, DyckSpec(), 6)
    assert not r["sound"] and Fraction(r["p_violation"]) == pviol > 0
    assert DyckSpec().label(r["violating_output"]) == 0


def test_multi_length_certificate_and_roundtrip(tmp_path):
    spec = DyckSpec()
    m, rep = train_generator(spec, GenConfig(L=12, lengths=(4, 8), steps=1500, lam=10.0,
                                             n_data=3000), 2, log=lambda e: None)
    assert rep["status"] == "proved", rep["verify"]
    path = str(tmp_path / "g.npz")
    save_generator(m.export(), path)
    G, _ = read_generator(path)
    for L in (4, 8, 12):
        assert verify_generator(G, spec, L)["status"] == "proved"
    x = m.double().sample(500, 12)
    assert spec.label_batch(x.numpy()).all()
