"""Stack-state models: exact verifiers agree with brute force; Dyck-2 generators certify."""
import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vt.stack import (DyckKSpec, StackGen, StackNet, _gen_graph, stack_step, train_stackgen,
                      verify_stackgen, verify_stacknet)


def test_spec_semantics():
    s = DyckKSpec("()[]", max_depth=2)
    assert s.label([0, 1, 2, 3]) == 1 and s.label([0, 2, 3, 1]) == 1
    assert s.label([0, 3]) == 0 and s.label([0, 2, 0]) == 0       # mismatch / depth 3
    assert s.label([0, 0, 0, 1, 1, 1]) == 0                         # exceeds max depth 2


@pytest.mark.parametrize("steps", [3, 600])
def test_classifier_exact_margin_equals_brute_force(steps):
    spec, L = DyckKSpec("()[]", 3), 6
    seqs = np.array(list(itertools.product(range(4), repeat=L)))
    X, y = torch.tensor(seqs), torch.tensor(spec.label_batch(seqs))
    torch.manual_seed(0)
    c = StackNet(4, spec.roles, 3)
    opt = torch.optim.Adam(c.parameters(), 3e-2)
    for _ in range(steps):
        loss = F.cross_entropy(c(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    c = c.double()
    with torch.no_grad():
        lg = c(X)
    brute = ((lg[:, 1] - lg[:, 0]) * (2 * y - 1).double()).min().item()
    r = verify_stacknet(c.export(), spec, L)
    assert abs(r["min_margin_float"] - brute) < 1e-9


def _brute_support(G, spec, L):
    from vt.stack import _ints, EMPTY
    V, D, roles = G["V"], G["D"], G["roles"]
    m = StackGen(V, roles, D).double()
    with torch.no_grad():
        for k in ("T", "u", "ur", "W1", "b1", "Wo", "bo"):
            getattr(m, k).copy_(torch.as_tensor(G[k]))
    sound = complete = True
    for seq in itertools.product(range(V), repeat=L):
        st, dep, p_pos = (), 0, True
        for t, v in enumerate(seq):
            top = torch.tensor([st[-1] if st else EMPTY])
            sc = F.relu(m.logits(top, torch.tensor([float(dep)], dtype=torch.double),
                                 torch.tensor([float(L - t)], dtype=torch.double)))[0]
            if sc[v] <= 0:
                p_pos = False
                break
            st, dep = stack_step(st, dep, v, roles, D)
        lab = spec.label(seq)
        sound &= not (p_pos and not lab)
        complete &= not (lab and not p_pos)
    return sound, complete


@pytest.mark.parametrize("steps", [0, 1500])
def test_generator_certificate_equals_brute_force(steps):
    spec, L = DyckKSpec("()[]", 3), 6
    m = train_stackgen(spec, L, steps=steps, n_data=500, log=lambda e: None) if steps else \
        StackGen(4, spec.roles, 3)
    G = m.export()
    r = verify_stackgen(G, spec, L)
    assert (r["sound"], r["complete"]) == _brute_support(G, spec, L)


def test_dyck2_generator_certified_at_several_lengths():
    spec = DyckKSpec("()[]", 4)
    m = train_stackgen(spec, 12, lengths=(4, 8), steps=1500, n_data=3000, log=lambda e: None)
    G = m.export()
    for L in (4, 8, 12):
        assert verify_stackgen(G, spec, L)["status"] == "proved"
    assert spec.label_batch(m.sample(300, 12)).all()
