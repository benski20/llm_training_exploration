"""Glass-box LM invariants: the evidence stream is an exact convex account of mixing."""
import torch

from vt.glassbox import GBConfig, GlassBoxLM


def _m():
    torch.manual_seed(0)
    return GlassBoxLM(GBConfig(vocab_size=50, block_size=16, n_embd=32, mem_keys=8, mem_topk=4))


def test_evidence_is_a_distribution_over_past_sources():
    m = _m()
    x = torch.randint(0, 50, (3, 10))
    _, _, e, _ = m(x)
    assert torch.allclose(e.sum(-1), torch.ones(3, 10), atol=1e-5)
    assert (e >= -1e-7).all()
    future = torch.triu(torch.ones(10, 10, dtype=torch.bool), 1)
    assert float(e[..., :10][:, future].abs().max()) < 1e-7       # never attributes to future tokens


def test_evidence_matches_content_dependence():
    """Sanity: a source with positive evidence weight (the position's own token) does
    influence that position's content."""
    m = _m()
    x = torch.randint(0, 50, (1, 8))
    _, _, e, hs = m(x)
    x2 = x.clone(); x2[0, 7] = (x[0, 7] + 1) % 50                  # the last token itself
    _, _, _, hs2 = m(x2)
    assert float(e[0, 7, 7]) > 0 and not torch.allclose(hs[-1][0, 7], hs2[-1][0, 7])


def test_memory_patch_and_starved_core():
    m = _m()
    assert not m.embed.weight.requires_grad
    x = torch.randint(0, 50, (2, 6))
    m(x)
    r = m.mem.trace["read"][:, 5].clone()
    mask = torch.zeros(2, 6, dtype=torch.bool); mask[:, 5] = True
    m.mem.patch = (mask, r.flip(0))
    m(x)
    m.mem.patch = None
    assert torch.allclose(m.mem.trace["read"][:, 5], r.flip(0))


def test_per_position_memory_evidence():
    torch.manual_seed(0)
    m = GlassBoxLM(GBConfig(vocab_size=50, block_size=16, n_embd=32, mem_keys=8, mem_topk=4,
                            mem_evidence="per_position"))
    x = torch.randint(0, 50, (2, 7))
    _, _, e, _ = m(x)
    assert e.shape == (2, 7, 14)
    assert torch.allclose(e.sum(-1), torch.ones(2, 7), atol=1e-5)
    future = torch.triu(torch.ones(7, 7, dtype=torch.bool), 1)
    assert float(e[..., 7:][:, future].abs().max()) < 1e-7        # no credit to future memory reads
