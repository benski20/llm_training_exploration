"""Long prompts: chunked prefill (``prefill_chunk_rows``) and the row-tiled native attention.

On the native float path every row's bits depend only on that row and the cache prefix it sees, so a prompt
processed in one pass, in chunks of any size (chunk-major, every layer per chunk) or token by token gives the
same logits and the same KV cache bit for bit; the native attention's row tiles (bounded scratch for long
passes) keep that too. Masked == plain with chunking, and the provider sees the same rows per linear in total
(each call at most a chunk). The tiny model's context is stretched to thousands of positions (RoPE tables grow;
the bits tests do not need a trained context).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING, RING_K40_A12, native
from pgw.engine import InMemoryPack, LocalMaskReservation, PlainBackend, SequenceInput, TransformerModel
from pgw.engine.model import NativeFloatOps, native_float_available
from tests.engine.conftest import plain_and_masked

pytestmark = pytest.mark.skipif(
    not (native_float_available() and native.native_available("float_rope_attend_bf16")),
    reason="pgw_fast without the native float path")

#: (attention block, bf16 KV): v1, v2 with small blocks (many blocks per row), and the bf16 caches
VARIANTS = [(0, False), (16, False), (0, True), (256, True)]
VIDS = [f"block{b}-{'bf16' if h else 'f32'}" for b, h in VARIANTS]


def _ids(n: int) -> list[int]:
    return [(37 * i * i + 11 * i + 5) % 250 + 3 for i in range(n)]


def _model(pack: InMemoryPack, block: int, bf16: bool, chunk: int = 0, threads: int = 0) -> TransformerModel:
    model = TransformerModel(pack, PlainBackend(pack, RING_K40_A12))
    model.float_ops = NativeFloatOps(attn_block=block, kv_bf16=bf16, attn_threads=threads)
    model.prefill_chunk_rows = chunk
    return model


def _kv(cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    n = cache.length
    k, v = cache._k[:, :n].clone(), cache._v[:, :n].clone()
    return k, v


@pytest.mark.parametrize(("block", "bf16"), VARIANTS, ids=VIDS)
def test_one_pass_equals_chunks_and_token_by_token(block: int, bf16: bool, qwen_pack: InMemoryPack) -> None:
    n = 1500
    ids = torch.tensor(_ids(n))
    ref_model = _model(qwen_pack, block, bf16)
    cache = ref_model.new_cache(n + 8)
    ref = ref_model.forward(ids, cache, all_logits=True)
    ref_kv = _kv(cache)
    for chunk, threads in ((700, 0), (512, 3), (1, 1), (n - 1, 2)):
        model = _model(qwen_pack, block, bf16, chunk, threads)
        c = model.new_cache(n + 8)
        got = model.forward(ids, c, all_logits=True)
        assert torch.equal(got, ref), chunk
        k, v = _kv(c)
        assert torch.equal(k, ref_kv[0]) and torch.equal(v, ref_kv[1]), chunk
    # token by token (decode passes) from a 1000-row prompt pass: each row as plain decoding computes it
    model = _model(qwen_pack, block, bf16)
    c = model.new_cache(n + 8)
    rows = [model.forward(ids[:1000], c, all_logits=True)]
    for i in range(1000, n):
        rows.append(model.forward(ids[i : i + 1], c, all_logits=True))
    assert torch.equal(torch.cat(rows), ref)
    k, v = _kv(c)
    assert torch.equal(k, ref_kv[0]) and torch.equal(v, ref_kv[1])


def test_chunked_on_top_of_a_cached_prefix_and_batched(qwen_pack: InMemoryPack) -> None:
    """A chunked pass after earlier positions (prefix reuse, multi-turn) and a batch of two prompts of
    different lengths (sub-passes carry the sequences that still have rows) equal one pass each."""
    a, b = torch.tensor(_ids(900)), torch.tensor(_ids(1300)[::-1].copy())
    ref_model = _model(qwen_pack, 0, False)
    ca, cb = ref_model.new_cache(2400), ref_model.new_cache(2400)
    ref_model.forward(a[:333], ca)
    want_a = ref_model.forward(a[333:], ca, all_logits=True)
    want_b = ref_model.forward(b, cb, all_logits=True)
    model = _model(qwen_pack, 0, False, chunk=256)
    da, db = model.new_cache(2400), model.new_cache(2400)
    model.forward(a[:333], da)
    out = model.forward_batch([SequenceInput(a[333:], da, all_logits=True), SequenceInput(b, db, all_logits=True)])
    assert torch.equal(out[0], want_a) and torch.equal(out[1], want_b)
    assert torch.equal(_kv(da)[0], _kv(ca)[0]) and torch.equal(_kv(db)[1], _kv(cb)[1])


def test_chunked_capture_hidden_equals_one_pass(qwen_pack: InMemoryPack) -> None:
    ids = torch.tensor(_ids(700))
    outs = []
    for chunk in (0, 300):
        model = _model(qwen_pack, 16, False, chunk)
        s = SequenceInput(ids, model.new_cache(720), capture_hidden=(-1, 0, 1))
        outs.append((model.forward_batch([s])[0], s.hidden))
    assert torch.equal(outs[0][0], outs[1][0])
    assert outs[0][1] is not None and outs[1][1] is not None
    assert sorted(outs[0][1]) == sorted(outs[1][1]) == [-1, 0, 1]
    for lay in (-1, 0, 1):
        assert torch.equal(outs[0][1][lay], outs[1][1][lay]), lay


def test_torch_float_ops_never_chunk(qwen_pack: InMemoryPack) -> None:
    """The torch path's kernels choose blockings by the rows of a pass: chunking is not applied there."""
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    ids = torch.tensor(_ids(300))
    ref = model.forward(ids, model.new_cache(310), all_logits=True)
    model.prefill_chunk_rows = 64
    calls: list[int] = []
    orig = model.backend.forward_many

    def spy(cs: Any) -> Any:
        calls.extend(int(c.x.shape[0]) for c in cs)
        return orig(cs)

    model.backend.forward_many = spy  # type: ignore[method-assign]
    assert torch.equal(model.forward(ids, model.new_cache(310), all_logits=True), ref)
    assert set(calls) == {300}


def test_masked_chunked_equals_plain_and_provider_rows(qwen_pack: InMemoryPack) -> None:
    n, chunk = 600, 256
    ids = torch.tensor(_ids(n))
    plain, masked, client = plain_and_masked(qwen_pack, DEFAULT_RING, record=True)
    for m in (plain, masked):
        m.set_float_ops("native-blocked")
    want = plain.forward(ids, plain.new_cache(n + 4), all_logits=True)
    masked.prefill_chunk_rows = chunk
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, n + 4)
    got = masked.forward(ids, masked.new_cache(n + 4), session=res, all_logits=True)
    assert torch.equal(got, want)
    per: dict[str, int] = {}
    for lid, x in client.received:
        assert int(x.shape[0]) <= chunk
        per[lid] = per.get(lid, 0) + int(x.shape[0])
    assert per and set(per.values()) == {n}  # every linear: the prompt's rows, once
    # the masked pass took exactly the rows an unchunked pass takes
    for spec in qwen_pack.linear_specs():
        assert res.remaining(spec.id) == 4


def test_kv_cache_long_prompt_allocates_the_limit_once() -> None:
    """A prompt pass close to the cache limit allocates the limit at once: the first decode steps do not copy
    the whole cache into a doubled buffer; a short prompt with a far limit still grows geometrically."""
    from pgw.engine.kv_cache import KVCache

    c = KVCache(2, 2, 8, 10_000, initial_len=256)
    c.reserve(9_000)
    assert c.capacity == 10_000
    storage = c._k
    for n in (9_001, 9_500, 10_000):
        c.reserve(n)
        assert c._k is storage
    far = KVCache(2, 2, 8, 100_000, initial_len=256)
    far.reserve(300)
    assert far.capacity == 512
    far.reserve(5_000)
    assert far.capacity == 5_000


def test_native_attention_row_tiles_match_single_rows() -> None:
    """A long pass through ``float_attend`` (row tiles above the scratch budget) equals each row alone."""
    from pgw.engine.model import _np32

    torch.manual_seed(3)
    rows, nh, kvh, hd = 3000, 8, 2, 32  # per row ~190 KB of scores: several tiles
    q = torch.randn(rows, nh, hd)
    k = torch.randn(rows, kvh, hd)
    v = torch.randn(rows, kvh, hd)
    for block in (0, 256):
        out = torch.empty(rows, nh * hd)
        native.fast.float_attend(_np32(q), _np32(k), _np32(v), 0, 0.17, out.numpy(), threads=3, block=block)
        for t in (0, 1, 255, 256, 1999, 2999):
            o = torch.empty(1, nh * hd)
            native.fast.float_attend(_np32(q[t : t + 1]), _np32(k[: t + 1]), _np32(v[: t + 1]), t, 0.17, o.numpy(),
                                     threads=1, block=block)
            assert torch.equal(o[0], out[t]), (block, t)
