"""Glassbox-3: the epinet leaves the base prediction and its exact attribution untouched,
uncertainty raises <unsure>, and novelty falls on what was seen in training."""
import torch

from vt.glassbox3 import G3Config, Glassbox3


def _m():
    torch.manual_seed(0)
    return Glassbox3(G3Config(vocab_size=40, block_size=16, n_layer=2, n_head=2, n_embd=16, n_features=32,
                              k_active=4, mem_layer=1, mem_keys=4, mem_topk=2, unsure_id=1, epi_hidden=16))


def test_attribution_stays_exact_and_unsure_is_explicit():
    m = _m().eval()
    x = torch.randint(0, 40, (1, 7))
    logits, k_eff = m(x)
    raw = m.trace["raw_logits"]
    parts = m.explain(x, 6, 5)
    assert abs(sum(parts.values()) - float(raw[0, 6, 5])) < 1e-4
    t = m.temperature(m.trace["k_eff"])[..., None]
    expect = raw * t
    expect[..., 1] += torch.nn.functional.softplus(m.c_unsure) * m.trace["u"]
    assert torch.allclose(logits, expect, atol=1e-5)
    assert (m.trace["u"] >= -1e-6).all()


def test_epinet_does_not_touch_the_base_model():
    m = _m()
    x = torch.randint(0, 40, (2, 6)); y = torch.randint(0, 40, (2, 6))
    m(x)
    m.epi_loss(y).backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for n, p in m.named_parameters()
               if not n.startswith("epi."))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.epi.parameters())


def test_novelty_is_lower_for_seen_feature_patterns():
    m = _m().eval()
    seen = torch.randint(0, 40, (4, 8))
    for _ in range(5):
        m(seen); m.record_features()
    m(seen); a = m.novelty().mean()
    m(torch.randint(0, 40, (4, 8))); b = m.novelty().mean()
    assert a < b
