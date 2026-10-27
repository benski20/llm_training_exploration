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


def _soundness(m, x):
    with torch.no_grad():
        base, _ = m(x)
        free = m.memory_free()
        saved = m.mem.values.weight.clone()
        m.mem.values.weight.normal_(0, 10)
        other, _ = m(x)
        m.mem.values.weight.copy_(saved)
    return free, torch.equal(base[free], other[free])


def test_memory_free_certificate_is_sound_for_every_layout():
    torch.manual_seed(3)
    for mem_layer, attn_topk in ((1, 0), (1, 2), (2, 0), (2, 2)):      # dense/sparse; memory mid or after last layer
        m = _m(mem_layer=mem_layer, attn_topk=attn_topk)
        with torch.no_grad():
            m.mem.gate.weight.normal_(0, 1); m.mem.gate.bias.fill_(-0.3)   # some gates open, some closed
            m.mem.values.weight.normal_(0, 1)
        x = torch.randint(0, 60, (2, 10))
        free, same = _soundness(m, x)
        assert same, (mem_layer, attn_topk)
        if mem_layer == 2:                                             # after the last layer: own gate only
            assert torch.equal(free, m.trace["mem"]["gate"] == 0)


def test_attribution_exact_with_sparse_attention():
    m = _m(attn_topk=2)
    x = torch.randint(0, 60, (1, 9))
    parts = m.explain(x, 8, 5)
    assert abs(sum(parts.values()) - float(m.trace["raw_logits"][0, 8, 5])) < 1e-4
    assert all(int((d["A"] > 0).sum(-1).max()) <= 2 for d in m.trace["attn"])


def test_straight_through_gate_is_exact_and_never_freezes():
    from vt.glassbox import GBConfig, PKMemory
    torch.manual_seed(0)
    m = PKMemory(GBConfig(vocab_size=10, n_embd=8, mem_keys=4, mem_topk=2, mem_gate="hard_st"))
    with torch.no_grad():
        m.gate.weight.mul_(20)                   # most gates saturated at 0 or 1
    z = torch.randn(64, 8)
    g = m.gate_value(z)
    assert torch.equal(g, m.gate(z).clamp(0, 1)) and (g == 0).any() and (g == 1).any()
    g.sum().backward()
    assert float(m.gate.bias.grad) == 64.0       # gradient 1 for every gate, saturated or not
