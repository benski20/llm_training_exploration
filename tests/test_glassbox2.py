"""Glassbox-2: exact attribution, exact sparsity, memory certificates and training-data writers."""
import torch

from vt.glassbox2 import G2Config, Glassbox2


def _m(**kw):
    torch.manual_seed(0)
    c = dict(vocab_size=60, block_size=32, n_layer=3, n_head=2, n_embd=16, n_features=64, k_active=4,
             mem_layer=1, mem_keys=4, mem_topk=2)
    c.update(kw)
    return Glassbox2(G2Config(**c)).eval()


def test_attribution_is_exact():
    for gate_bias in (0.5, -5.0):                       # memory open and closed
        m = _m()
        with torch.no_grad():
            m.mem.gate.bias.fill_(gate_bias)
            m.mem.values.weight.normal_(0, 0.5)
        x = torch.randint(0, 60, (1, 9))
        for t, tok in ((8, 3), (4, 17)):
            parts = m.explain(x, t, tok)
            raw = float(m.trace["raw_logits"][0, t, tok])
            assert abs(sum(parts.values()) - raw) < 1e-4, (sum(parts.values()), raw)
            assert all(parts[f"token@{j}"] == 0.0 for j in range(t + 1, 9))   # causal: no future tokens
            if gate_bias < 0:
                assert all(parts[f"memory@{j}"] == 0.0 for j in range(9))


def test_features_are_exactly_sparse():
    m = _m()
    m(torch.randint(0, 60, (2, 7)))
    for d in m.trace["feat"]:
        assert int((d["acts"] != 0).sum(-1).max()) <= 4


def test_memory_writers_are_recorded():
    m = _m()
    m.train()
    with torch.no_grad():
        m.mem.gate.bias.fill_(0.5)
    x = torch.randint(0, 60, (2, 7))
    m(x)
    m.record_writers(torch.tensor([[11] * 7, [22] * 7]))
    assert set(m.writer_doc[m.writer_doc >= 0].tolist()) <= {11, 22} and (m.writer_doc >= 0).any()
