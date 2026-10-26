"""Global verifier checks against a hand-built correct Dyck model and against
exhaustive enumeration on short sequences."""
import itertools

import numpy as np
import torch

from vt.global_verify import DyckSpec, verify_global
from vt.model import MixerNet
from vt.verify import np_forward


def hand_built_dyck(L):
    """Exact Dyck-1 recogniser as a 1-block MixerNet.

    ch0 <- +1/-1 per token; mixing makes ch0 the prefix depth D_t;
    MLP writes relu(-D_t) to ch1 and relu(D_t) to ch2;
    margin = 0.5 - sum_t relu(-D_t) - relu(D_L)."""
    d, h = 3, 2
    W_in = np.zeros((d, 2)); W_in[0] = [1, -1]
    M = np.tril(np.ones((L, L))) - np.eye(L)
    W1 = np.zeros((h, d)); W1[0, 0] = -1; W1[1, 0] = 1
    W2 = np.zeros((d, h)); W2[1, 0] = 1; W2[2, 1] = 1
    W_out = np.zeros((2, L * d))
    for t in range(L):
        W_out[1, t * d + 1] = -1
        W_out[1, t * d + 0] = 0
    W_out[1, (L - 1) * d + 2] = -1
    b_out = np.array([0.0, 0.5])
    return {"L": L, "V": 2, "d": d, "h": h, "C": 2, "W_in": W_in, "b_in": np.zeros(d),
            "mix": [M], "W1": [W1], "b1": [np.zeros(h)], "W2": [W2], "b2": [np.zeros(d)],
            "W_out": W_out, "b_out": b_out}


def exhaustive_errors(P, L):
    spec = DyckSpec()
    errs = []
    for seq in itertools.product([0, 1], repeat=L):
        pred = int(np.argmax(np_forward(P, np.eye(2)[list(seq)])))
        if pred != spec.label(seq):
            errs.append(seq)
    return errs


def test_hand_built_model_is_correct_and_proved():
    for L in (6, 10, 16):
        P = hand_built_dyck(L)
        if L <= 10:
            assert exhaustive_errors(P, L) == []
        r = verify_global(P, DyckSpec(), time_limit=60)
        assert r["status"] == "proved", r


def test_broken_model_yields_real_counterexample():
    P = hand_built_dyck(10)
    P["W_out"][1, (10 - 1) * 3 + 2] = 0.0  # forget the final-depth check
    r = verify_global(P, DyckSpec(), time_limit=60)
    assert r["status"] == "counterexample"
    for cex in r["counterexamples"]:
        assert cex["pred"] != cex["label"]
        assert DyckSpec().label(cex["seq"]) == cex["label"]


def test_agrees_with_enumeration_on_trained_models():
    spec = DyckSpec()
    L = 8
    seqs = np.array(list(itertools.product([0, 1], repeat=L)))
    X = torch.tensor(np.eye(2, dtype=np.float32)[seqs])
    Y = torch.tensor([spec.label(s) for s in seqs])
    outcomes = set()
    for seed, ternary in [(0, False), (1, True), (2, False), (3, False)]:
        torch.manual_seed(seed)
        model = MixerNet(L, 2, 4, 8, 1, 2, ternary=ternary)
        opt = torch.optim.Adam(model.parameters(), 1e-2)
        for _ in range(40 * (seed + 1)):  # varying amounts of training
            loss = torch.nn.functional.cross_entropy(model(X), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        P = model.export()
        n_err = len(exhaustive_errors(P, L))
        r = verify_global(P, spec, time_limit=60)
        assert r["status"] in ("proved", "counterexample")
        outcomes.add(r["status"])
        if n_err:
            assert r["status"] == "counterexample"
        elif r["status"] == "counterexample":
            # a correct model may still violate the tau safety margin
            assert all(abs(c["margin"]) <= 1e-4 for c in r["counterexamples"])
    assert outcomes == {"proved", "counterexample"}, outcomes


def test_symbolic_bounds_sound_and_tighter():
    from vt.global_verify import global_bounds
    from vt.verify import ibp_from_s0
    torch.manual_seed(0)
    L = 10
    model = MixerNet(L, 2, 6, 12, 2, 2, ternary=False)
    P = model.export()
    B = global_bounds(P)
    rng = np.random.default_rng(0)
    for _ in range(300):
        x = np.eye(2)[rng.integers(2, size=L)]
        s = x @ P["W_in"].T + P["b_in"]
        assert (B["s"][0][0] - 1e-9 <= s.reshape(-1)).all() and (s.reshape(-1) <= B["s"][0][1] + 1e-9).all()
        for i, (M, W1, b1, W2, b2) in enumerate(zip(P["mix"], P["W1"], P["b1"], P["W2"], P["b2"])):
            s = s + M @ s
            z = (s @ W1.T + b1).reshape(-1)
            l, u = B["z"][i]
            assert (l - 1e-9 <= z).all() and (z <= u + 1e-9).all()
            s = s + np.maximum(z.reshape(L, -1), 0) @ W2.T + b2
    # tighter than plain interval propagation
    cols = P["W_in"].T + P["b_in"]
    lo, hi = cols.min(0), cols.max(0)
    ibp = ibp_from_s0(P, np.tile((lo + hi) / 2, (L, 1)), np.tile((hi - lo) / 2, (L, 1)),
                      (np.zeros(2 * L), np.ones(2 * L)))
    width = lambda bb: sum((u - l).sum() for l, u in bb["z"])
    assert width(B) < 0.8 * width(ibp)


def test_sorted_spec_agrees_with_enumeration():
    from vt.global_verify import SortedSpec, exhaustive_check
    spec, L, V = SortedSpec(4), 5, 4
    seqs = np.array(list(itertools.product(range(V), repeat=L)))
    assert all(spec.label(s) == y for s, y in zip(seqs, spec.label_batch(seqs)))
    X = torch.tensor(np.eye(V, dtype=np.float32)[seqs])
    Y = torch.tensor(spec.label_batch(seqs))
    outcomes = set()
    for steps in (5, 400):
        torch.manual_seed(0)
        model = MixerNet(L, V, 6, 12, 1, 2)
        opt = torch.optim.Adam(model.parameters(), 1e-2)
        w = torch.where(Y == 1, 20.0, 1.0)
        for _ in range(steps):
            loss = (torch.nn.functional.cross_entropy(model(X), Y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        P = model.export()
        n_err, _, _ = exhaustive_check(P, spec)
        r = verify_global(P, spec, time_limit=60)
        outcomes.add(r["status"])
        assert (r["status"] == "proved") == (n_err == 0)
        for c in r["counterexamples"]:
            assert spec.label(c["seq"]) == c["label"] != c["pred"] or abs(c["margin"]) <= 1e-4
    assert outcomes == {"proved", "counterexample"}, outcomes


def test_parallel_cases_match_sequential():
    P = hand_built_dyck(12)
    assert verify_global(P, DyckSpec(), workers=2)["status"] == "proved"
    P["W_out"][1, (12 - 1) * 3 + 2] = 0.0
    r = verify_global(P, DyckSpec(), workers=2)
    assert r["status"] == "counterexample"
    assert all(c["pred"] != c["label"] for c in r["counterexamples"])
