"""Certified attribution: the bounds are exact at zero width and sound for sampled memory contents."""
import torch

from vt.certify import certified_argmax, memory_free_bounds
from vt.glassbox import GBConfig, GlassBoxLM


def _models():
    torch.manual_seed(0)
    yield GlassBoxLM(GBConfig(vocab_size=40, block_size=16, n_layer=3, n_head=2, n_embd=16, mem_layer=1,
                              mem_keys=4, mem_topk=2)).eval()
    yield GlassBoxLM(GBConfig(vocab_size=40, block_size=16, n_layer=3, n_head=2, n_embd=16, mem_layer=1,
                              mem_keys=4, mem_topk=2, mlp_ratio=2, sharpness=True, fixed_embed=False)).eval()


def test_zero_width_box_is_the_exact_forward_pass():
    for m in _models():
        x = torch.randint(0, 40, (2, 7))
        ll, lh, kl, kh, raw, k = memory_free_bounds(m, x, positions=torch.zeros(2, 7, dtype=torch.bool))
        assert torch.allclose(ll, raw, atol=1e-4) and torch.allclose(lh, raw, atol=1e-4)
        assert torch.allclose(kl, k, atol=1e-4) and torch.allclose(kh, k, atol=1e-4)


def test_bounds_contain_every_sampled_memory_content():
    g = torch.Generator().manual_seed(1)
    for m in _models():
        x = torch.randint(0, 40, (1, 6))
        ll, lh, kl, kh, _, _ = memory_free_bounds(m, x)
        V = m.mem.values.weight
        for _ in range(30):
            w = torch.rand(6, V.shape[0], generator=g) ** 8                # random convex combinations
            reads = (w / w.sum(-1, keepdim=True)) @ V
            m.mem.patch = (torch.ones(1, 6, dtype=torch.bool), reads[None])
            logits, k, _, _ = m(x, evidence=False)
            m.mem.patch = None
            if m.config.sharpness:
                logits = logits / m.temperature(k)[..., None]
            assert (logits >= ll - 1e-4).all() and (logits <= lh + 1e-4).all()
            assert (k >= kl - 1e-4).all() and (k <= kh + 1e-4).all()


def test_certified_argmax():
    lo = torch.tensor([[0.0, 2.0, 1.0]]); hi = torch.tensor([[0.5, 3.0, 1.9]])
    assert certified_argmax(lo, hi).item() == 1
    assert certified_argmax(lo, torch.tensor([[0.5, 3.0, 2.5]])).item() == -1
