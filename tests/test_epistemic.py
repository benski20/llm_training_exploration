"""Epistemic channels: readout, supervision, causal intervention, isolation."""
import pytest
import torch
import torch.nn.functional as F

from vt.epistemic import ChannelSpec, EpistemicModel
from vt.lm import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_read_and_intervene_change_downstream():
    m = tiny()
    em = EpistemicModel(m, {"g": ChannelSpec(layer=0, dims=(24, 32))})
    x = torch.randint(0, 256, (2, 16))
    base, _ = m(x)
    r = em.read("g")
    assert r.shape == (2, 16)
    mask = torch.zeros(2, 16, dtype=torch.bool); mask[:, 5:] = True
    with em.intervene("g", mask, -4.0):
        moved, _ = m(x)
        assert torch.allclose(em.read("g")[:, 5:], torch.tensor(-4.0))
    assert torch.allclose(moved[:, :5], base[:, :5], atol=1e-6)     # causal: earlier positions untouched
    assert not torch.allclose(moved[:, 5:], base[:, 5:])            # later positions affected
    after, _ = m(x)
    assert torch.allclose(after, base)                              # intervention is scoped


def test_supervised_channel_learns_a_signal():
    m = tiny()
    em = EpistemicModel(m, {"g": ChannelSpec(layer=1, dims=(28, 32))})
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(150):
        x = torch.randint(0, 256, (16, 16))
        target = (x[:, 3] % 2 == 0)                                  # a computable signal
        m(x)
        loss = em.aux_loss("g", torch.full((16,), 8), target)
        opt.zero_grad(); loss.backward(); opt.step()
    x = torch.randint(0, 256, (64, 16))
    m(x)
    pred = em.read("g", torch.full((64,), 8)) > 0
    assert (pred == (x[:, 3] % 2 == 0)).float().mean() > 0.9


def test_rejects_overlapping_or_bad_channels():
    m = tiny()
    with pytest.raises(ValueError):
        EpistemicModel(m, {"a": ChannelSpec(0, (0, 8)), "b": ChannelSpec(0, (4, 12))})
    with pytest.raises(ValueError):
        EpistemicModel(m, {"a": ChannelSpec(5, (0, 8))})
