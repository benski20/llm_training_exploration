"""``apply_rope`` (signed-table form) gives the bits of the HF form ``x * cos + rotate_half(x) * sin``."""

from __future__ import annotations

import gc

import pytest
import torch

from pgw.engine import rope as rope_mod
from pgw.engine.rope import RotaryEmbedding, apply_rope, apply_rope_reference

SCALINGS = [
    None,
    {"rope_type": "linear", "factor": 4.0},
    {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
     "original_max_position_embeddings": 8192},
    {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096},
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    view = {torch.float64: torch.int64, torch.float32: torch.int32, torch.float16: torch.int16,
            torch.bfloat16: torch.int16}[t.dtype]
    return t.contiguous().view(view)


def _same(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.stride() == b.stride()
    assert torch.equal(_bits(a), _bits(b))


@pytest.mark.parametrize("scaling", SCALINGS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_apply_rope_bits_match_reference(scaling: dict | None, dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(3)
    for hd, heads, kvh in ((128, 28, 4), (64, 14, 2), (2, 3, 1)):
        emb = RotaryEmbedding(hd, 1e6, scaling, max_position_embeddings=32768)
        for start, t in ((0, 1), (47, 1), (5000, 1), (0, 40), (13, 7), (2040, 3)):
            cos, sin = emb.cos_sin(start, t, dtype)
            # q and k as the model takes them: views into one fused qkv row block (non-contiguous for t > 1)
            y = (torch.randn(t, (heads + 2 * kvh) * hd, generator=g) * 3.0).to(dtype)
            y[:, ::17] = 0.0
            y[:, 1::19] = -0.0
            q = y[:, : heads * hd].reshape(t, heads, hd)
            k = y[:, heads * hd : (heads + kvh) * hd].reshape(t, kvh, hd)
            for x in (q, k, q.contiguous()):
                _same(apply_rope(x, cos, sin), apply_rope_reference(x, cos, sin))


def test_special_values() -> None:
    hd = 8
    vals = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), 1e-45, -3e38, 3e38, 1e-30])
    g = torch.Generator().manual_seed(1)
    x = vals[torch.randint(0, len(vals), (64, 3, hd), generator=g)]
    cs = vals[torch.randint(0, len(vals), (64, hd), generator=g)]
    sn = vals[torch.randint(0, len(vals), (64, hd), generator=g)]
    _same(apply_rope(x, cs, sn), apply_rope_reference(x, cs, sn))


def test_signed_table_cache_tracks_identity_and_in_place_changes() -> None:
    rope_mod._signed_sin_cache.clear()
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 4, 16, generator=g)
    cos = torch.randn(1, 16, generator=g)
    sin = torch.randn(1, 16, generator=g)
    a = apply_rope(x, cos, sin)
    assert len(rope_mod._signed_sin_cache) == 1
    assert torch.equal(apply_rope(x, cos, sin), a)
    sin.mul_(-0.5)  # in place: version bump, the cached signed table must not be reused
    _same(apply_rope(x, cos, sin), apply_rope_reference(x, cos, sin))
    other = sin.clone()  # equal values, another tensor
    _same(apply_rope(x, cos, other), apply_rope_reference(x, cos, other))
    del sin, other
    gc.collect()
    assert len(rope_mod._signed_sin_cache) == 0  # entries die with their tables
    for _ in range(3 * rope_mod._SIGNED_SIN_MAX):  # bounded
        s = torch.randn(1, 16, generator=g)
        _same(apply_rope(x, cos, s), apply_rope_reference(x, cos, s))
        keep = s  # noqa: F841 - keep one alive across iterations
    assert len(rope_mod._signed_sin_cache) <= rope_mod._SIGNED_SIN_MAX


@pytest.mark.parametrize("which", ["qwen_pack", "llama_pack"])
def test_model_layer_rope_unchanged_end_to_end(which: str, request: pytest.FixtureRequest) -> None:
    """A tiny model's prefill and decode logits with the new apply_rope equal those with the HF form."""
    from pgw.core import RingParams
    from pgw.engine import PlainBackend, TransformerModel
    from pgw.engine import model as model_mod

    pack = request.getfixturevalue(which)
    model = TransformerModel(pack, PlainBackend(pack, RingParams.from_name("K32A12C27")))
    ids = [3, 1, 4, 1, 5, 9, 2, 6]
    c1 = model.new_cache(64)
    fast = [model.forward(ids, c1, all_logits=True), model.forward([7], c1), model.forward([8], c1)]
    orig = model_mod.apply_rope
    model_mod.apply_rope = apply_rope_reference
    try:
        c2 = model.new_cache(64)
        ref = [model.forward(ids, c2, all_logits=True), model.forward([7], c2), model.forward([8], c2)]
    finally:
        model_mod.apply_rope = orig
    for a, b in zip(fast, ref, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_q_and_k_rotated_together_equal_separately(dtype: torch.dtype) -> None:
    """``_layer_standard`` rotates q and k in one call on the qkv output's view: the same bits as two calls."""
    g = torch.Generator().manual_seed(11)
    for hd, nh, kvh in ((64, 14, 2), (128, 28, 4)):
        emb = RotaryEmbedding(hd, 1e6, None, max_position_embeddings=32768)
        for start, t in ((0, 1), (300, 1), (0, 41), (17, 5)):
            cos, sin = emb.cos_sin(start, t, dtype)
            y = torch.randn(t, (nh + 2 * kvh) * hd, generator=g).to(dtype)
            nq, nk = nh * hd, kvh * hd
            q_ref = apply_rope(y[:, :nq].reshape(t, nh, hd), cos, sin)
            k_ref = apply_rope(y[:, nq : nq + nk].reshape(t, kvh, hd), cos, sin)
            qk = apply_rope(y[:, : nq + nk].reshape(t, nh + kvh, hd), cos, sin)
            assert torch.equal(_bits(qk[:, :nh]), _bits(q_ref)) and torch.equal(_bits(qk[:, nh:]), _bits(k_ref))
