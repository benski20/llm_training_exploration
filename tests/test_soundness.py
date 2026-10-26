"""Soundness checks: exported weights match torch, bounds contain samples,
and MILP verdicts agree with brute-force sampling."""
import numpy as np
import pytest
import torch

from vt.model import MixerNet
from vt.verify import ibp_margins, np_forward, np_ibp, verify


@pytest.fixture(params=[False, True], ids=["float", "ternary"])
def model(request):
    torch.manual_seed(0)
    return MixerNet(seq_len=6, vocab=2, d_model=4, d_hidden=6, n_blocks=2,
                    n_classes=2, ternary=request.param)


def _x(L=6, V=2, seed=0):
    rng = np.random.default_rng(seed)
    return np.eye(V)[rng.integers(V, size=L)]


def test_export_matches_torch(model):
    P = model.export()
    x = _x()
    ref = model(torch.tensor(x[None], dtype=torch.float32))[0].detach().numpy()
    assert np.allclose(np_forward(P, x), ref, atol=1e-5)


def test_ibp_contains_samples(model):
    P, x, eps = model.export(), _x(), 0.2
    rng = np.random.default_rng(1)
    for y in range(2):
        lb = ibp_margins(P, x, y, eps)
        for _ in range(500):
            xp = x + rng.uniform(-eps, eps, x.shape)
            o = np_forward(P, xp)
            assert o[y] - o[1 - y] >= lb[0] - 1e-9


@pytest.mark.parametrize("eps", [0.02, 0.1, 0.3])
def test_milp_agrees_with_sampling(model, eps):
    P = model.export()
    rng = np.random.default_rng(2)
    for seed in range(5):
        x = _x(seed=seed)
        y = int(np.argmax(np_forward(P, x)))
        r = verify(P, x, y, eps, time_limit=20)
        assert r["status"] in ("verified", "falsified")
        if r["status"] == "verified":
            # no sampled point (including corners) may flip the prediction
            for _ in range(2000):
                xp = x + eps * rng.choice([-1.0, 1.0], size=x.shape) * rng.uniform(0.5, 1, x.shape)
                assert int(np.argmax(np_forward(P, xp))) == y
        else:
            assert r["cex_linf"] <= eps + 1e-6
            assert r["cex_margin"] <= 1e-6


def test_kbit_export_matches_torch_and_is_on_grid():
    torch.manual_seed(0)
    m = MixerNet(seq_len=6, vocab=2, d_model=4, d_hidden=6, n_blocks=1, n_classes=2, bits=4)
    P = m.export()
    x = _x()
    ref = m(torch.tensor(x[None], dtype=torch.float32))[0].detach().numpy()
    assert np.allclose(np_forward(P, x), ref, atol=1e-5)
    W = P["W1"][0]
    step = np.abs(W).max() / 7
    assert np.allclose(W / step, np.round(W / step), atol=1e-6)


def test_shared_model_is_toeplitz_and_exports_exactly():
    torch.manual_seed(0)
    m = MixerNet(seq_len=6, vocab=2, d_model=4, d_hidden=6, n_blocks=1, n_classes=2, shared=True)
    P = m.export()
    M = P["mix"][0]
    assert np.allclose(np.triu(M, 1), 0)                       # causal
    for k in range(6):                                          # constant along diagonals
        assert np.allclose(np.diag(M, -k), np.diag(M, -k)[0])
    W = P["W_out"].reshape(2, 6, 4)
    assert np.allclose(W[:, :-1], W[:, :1])                     # shared readout
    x = _x()
    ref = m(torch.tensor(x[None], dtype=torch.float32))[0].detach().numpy()
    assert np.allclose(np_forward(P, x), ref, atol=1e-5)
