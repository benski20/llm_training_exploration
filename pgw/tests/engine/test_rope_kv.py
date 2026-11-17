"""RoPE frequency computation (vs HF) and KV cache semantics."""

from __future__ import annotations

import pytest
import torch

from pgw.core import ConfigError
from pgw.engine import ContextOverflowError, KVCache, RotaryEmbedding, rope_inv_freq
from pgw.engine.rope import apply_rope, rotate_half

ROPE_CASES = [
    {"rope_type": "default"},
    {"rope_type": "linear", "factor": 2.0},
    {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
     "original_max_position_embeddings": 64},
    {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64},
    {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64, "beta_fast": 16, "beta_slow": 2,
     "mscale": 1.0, "mscale_all_dim": 0.5},
]


@pytest.mark.parametrize("scaling", ROPE_CASES, ids=lambda s: s["rope_type"] + str(len(s)))
@pytest.mark.parametrize("head_dim", [16, 64])
def test_inv_freq_matches_hf(scaling: dict, head_dim: int) -> None:
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    theta = 50000.0
    cfg = LlamaConfig(hidden_size=head_dim * 4, num_attention_heads=4, max_position_embeddings=256,
                      rope_parameters={**scaling, "rope_theta": theta})
    rtype = scaling["rope_type"]
    if rtype == "default":
        expected = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        attn = 1.0
    else:
        expected, attn = ROPE_INIT_FUNCTIONS[rtype](cfg, "cpu")
    got, got_attn = rope_inv_freq(head_dim, theta, None if rtype == "default" else scaling, 256)
    torch.testing.assert_close(got, expected.float(), rtol=1e-6, atol=0)
    assert got_attn == pytest.approx(attn)


def test_unsupported_rope_type() -> None:
    with pytest.raises(ConfigError):
        rope_inv_freq(16, 10000.0, {"rope_type": "longrope", "factor": 2.0})
    with pytest.raises(ConfigError):
        rope_inv_freq(15, 10000.0)


def test_rotary_tables_grow_and_rotate() -> None:
    rope = RotaryEmbedding(8, 10000.0)
    cos, _sin = rope.cos_sin(0, 4)
    assert cos.shape == (4, 8)
    cos2, sin2 = rope.cos_sin(1000, 3)  # forces growth
    assert cos2.shape == (3, 8)
    c_all, _s_all = rope.cos_sin(0, 1003)
    torch.testing.assert_close(c_all[1000:], cos2)
    x = torch.randn(3, 2, 8)
    y = apply_rope(x, cos2, sin2)
    # rotation preserves pairwise norms
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)
    assert torch.equal(rotate_half(rotate_half(x)), -x)


def test_kv_cache_append_advance_truncate() -> None:
    c = KVCache(2, 2, 4, max_len=10, initial_len=2)
    k = torch.randn(3, 2, 4)
    v = torch.randn(3, 2, 4)
    kk, vv = c.append(0, k, v)
    assert kk.shape == (3, 2, 4) and torch.equal(kk, k) and torch.equal(vv, v)
    assert c.length == 0  # not committed yet
    c.append(1, k, v)
    c.advance(3)
    assert c.length == 3 and c.remaining == 7
    k2 = torch.randn(1, 2, 4)
    kk, _ = c.append(0, k2, k2)
    assert kk.shape == (4, 2, 4) and torch.equal(kk[:3], k) and torch.equal(kk[3:], k2)
    c.advance(1)
    c.truncate(2)
    assert c.length == 2
    assert c.nbytes() > 0
    with pytest.raises(ConfigError):
        c.truncate(5)
    with pytest.raises(ContextOverflowError):
        c.append(0, torch.randn(9, 2, 4), torch.randn(9, 2, 4))
    with pytest.raises(ConfigError):
        c.append(5, k, v)
    with pytest.raises(ConfigError):
        c.append(0, torch.randn(1, 3, 4), torch.randn(1, 3, 4))
    c.clear()
    assert c.length == 0


def test_kv_cache_failed_pass_is_discarded() -> None:
    c = KVCache(1, 1, 2, max_len=8)
    c.append(0, torch.ones(2, 1, 2), torch.ones(2, 1, 2))
    c.advance(2)
    c.append(0, torch.full((3, 1, 2), 7.0), torch.full((3, 1, 2), 7.0))  # pass "fails": no advance
    kk, _ = c.append(0, torch.full((1, 1, 2), 5.0), torch.full((1, 1, 2), 5.0))
    assert kk.shape[0] == 3 and float(kk[2, 0, 0]) == 5.0


def test_kv_cache_rejects_bad_dims() -> None:
    with pytest.raises(ConfigError):
        KVCache(0, 1, 1, 4)
