"""Native attention v2 ("blocked", parallel) and bfloat16 KV storage (``float_ops = "native-blocked[-bf16kv]"``).

The blocked formulation splits each row's prefix into fixed blocks (``NATIVE_ATTN_BLOCK`` = 256 positions in
``pgw``; 16 here so that the tiny model's prompts span several blocks) and merges them in block order, so the
bits do not depend on the thread count, the instruction set or the number of rows in a pass; masked == plain
holds bit for bit; logits stay close to HF. bf16 KV storage is a separate reference (quality vs float32 KV in
``test_bf16_kv_quality_qwen05b``).
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

from pgw.core import DEFAULT_RING, RING_K40_A12, ConfigError, native
from pgw.engine import (
    ChatTokenizer,
    FloatBackend,
    Generator,
    InMemoryPack,
    LocalMaskReservation,
    PlainBackend,
    SamplingParams,
    SequenceInput,
    TransformerModel,
)
from pgw.engine.model import FLOAT_OPS_NAMES, NATIVE_ATTN_BLOCK, NativeFloatOps, native_float_available
from tests.engine.conftest import plain_and_masked
from tests.engine.test_model import IDS, hf_logits

pytestmark = pytest.mark.skipif(
    not (native_float_available() and native.native_available("float_rope_attend_bf16")),
    reason="pgw_fast without the blocked native attention")

PROMPT = [256, 72, 105, 33, 10, 72, 105]
#: (block, bf16 KV): small blocks so the tiny prompts cross block boundaries
VARIANTS = [(16, False), (0, True), (16, True)]
VIDS = [f"block{b}-{'bf16' if h else 'f32'}" for b, h in VARIANTS]


def _ops(model: TransformerModel, block: int, bf16: bool, threads: int = 0) -> TransformerModel:
    model.float_ops = NativeFloatOps(attn_block=block, kv_bf16=bf16, attn_threads=threads)
    return model


def _long_ids(n: int) -> list[int]:
    return [(IDS[i % len(IDS)] + 7 * i) % 250 + 3 for i in range(n)]


@pytest.mark.parametrize(("block", "bf16"), VARIANTS, ids=VIDS)
def test_thread_count_and_isa_independent(block: int, bf16: bool, qwen_pack: InMemoryPack) -> None:
    ids = torch.tensor(_long_ids(120))
    outs = []
    for threads in (1, 2, 3, 4, 7):
        model = _ops(TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12)), block, bf16, threads)
        cache = model.new_cache(160)
        outs.append(model.forward(ids, cache, all_logits=True))
        # decode rows on top (one row per pass) on the same thread count
        for t in (5, 9, 11):
            outs.append(model.forward(torch.tensor([t]), cache, all_logits=True))
    n = len(outs) // 5
    for i in range(1, 5):
        for j in range(n):
            assert torch.equal(outs[i * n + j], outs[j]), (i, j)
    fast = native.fast
    model = _ops(TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12)), block, bf16, 3)
    try:
        for cap in (1, 0):
            fast.attn_set_isa_cap(cap)
            assert torch.equal(model.forward(ids, model.new_cache(160), all_logits=True), outs[0]), cap
    finally:
        fast.attn_set_isa_cap(2)


@pytest.mark.parametrize(("block", "bf16"), VARIANTS, ids=VIDS)
def test_rows_invariant(block: int, bf16: bool, qwen_pack: InMemoryPack) -> None:
    """R-row passes (prompt blocks, speculative verify rows) give each row the bits of 1-row decoding."""
    model = _ops(TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12)), block, bf16, 2)
    ids = _long_ids(70)
    last = qwen_pack.arch.num_layers - 1
    whole = SequenceInput(ids, model.new_cache(96), capture_hidden=(last,))
    model.forward_batch([whole])
    assert whole.hidden is not None
    cache = model.new_cache(96)
    rows = []
    for split in (ids[:21], *[[t] for t in ids[21:26]], ids[26:29], ids[29:47], *[[t] for t in ids[47:50]], ids[50:]):
        s = SequenceInput(split, cache, capture_hidden=(last,))
        model.forward_batch([s])
        assert s.hidden is not None
        rows.append(s.hidden[last])
    assert torch.equal(torch.cat(rows), whole.hidden[last])


@pytest.mark.parametrize(("block", "bf16"), VARIANTS, ids=VIDS)
def test_masked_equals_plain(block: int, bf16: bool, qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    params = SamplingParams(max_tokens=40, temperature=0.0, ignore_eos=True, logprobs=2)
    plain, masked, _client = plain_and_masked(qwen_pack, DEFAULT_RING)
    _ops(plain, block, bf16, 1)
    _ops(masked, block, bf16, 3)
    prompt = PROMPT * 3
    base = Generator(plain, tiny_tokenizer).generate(prompt, params)
    got = Generator(masked, tiny_tokenizer).generate(
        prompt, params, session=LocalMaskReservation(qwen_pack, DEFAULT_RING, len(prompt) + 40))
    assert got.token_ids == base.token_ids and got.logprobs == base.logprobs


@pytest.mark.parametrize(("block", "bf16"), VARIANTS, ids=VIDS)
def test_speculation_identical(block: int, bf16: bool, qwen_pack: InMemoryPack,
                               tiny_tokenizer: ChatTokenizer) -> None:
    params = SamplingParams(max_tokens=30, temperature=0.0, ignore_eos=True, logprobs=2)
    target = _ops(TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING)), block, bf16, 2)
    prompt = PROMPT * 3
    base = Generator(target, tiny_tokenizer).generate(prompt, params)
    draft = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    spec = Generator(target, tiny_tokenizer, draft=draft, spec_tokens=4).generate(prompt, params)
    assert spec.token_ids == base.token_ids and spec.logprobs == base.logprobs


@pytest.mark.parametrize("which", ["qwen", "llama"])
@pytest.mark.parametrize(("block", "bf16"), VARIANTS, ids=VIDS)
def test_close_to_hf(block: int, bf16: bool, which: str, request: pytest.FixtureRequest) -> None:
    d: Path = request.getfixturevalue("tiny_hf_dir" if which == "qwen" else "tiny_llama_hf_dir")
    pack = InMemoryPack.from_hf(d)
    ids = _long_ids(100)
    ref = hf_logits(d, pack, ids)
    model = _ops(TransformerModel(pack, FloatBackend(pack, RING_K40_A12)), block, bf16, 2)
    out = model.forward(torch.tensor(ids), model.new_cache(128), all_logits=True)
    tol = 2e-2 if bf16 else 1e-4  # bf16 K/V: 8-bit mantissa storage (a different reference)
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    if not bf16:
        v1 = _ops(TransformerModel(pack, FloatBackend(pack, RING_K40_A12)), 0, False)
        torch.testing.assert_close(out, v1.forward(torch.tensor(ids), v1.new_cache(128), all_logits=True),
                                   rtol=1e-5, atol=1e-5)


def test_blocked_is_v1_within_one_block(qwen_pack: InMemoryPack) -> None:
    """With the pinned block (256) and prompts shorter than it, 'native-blocked' has the 'native' bits."""
    ids = torch.tensor(_long_ids(150))
    a = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    a.set_float_ops("native")
    b = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    b.set_float_ops("native-blocked")
    assert isinstance(b.float_ops, NativeFloatOps) and b.float_ops.attn_block == NATIVE_ATTN_BLOCK
    assert torch.equal(a.forward(ids, a.new_cache(160), all_logits=True),
                       b.forward(ids, b.new_cache(160), all_logits=True))


def test_float_ops_names_and_cache_dtype(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    for name in FLOAT_OPS_NAMES[1:]:
        model.set_float_ops(name)
        assert isinstance(model.float_ops, NativeFloatOps) and model.float_ops.name == name
        want = torch.bfloat16 if name.endswith("bf16kv") else torch.float32
        assert model.new_cache(8).dtype == want
    with pytest.raises(ConfigError):
        model.set_float_ops("native-fp8kv")
    model.set_float_ops("native-bf16kv")
    cache = model.new_cache(16)
    model.set_float_ops("native")
    with pytest.raises(ConfigError):  # a bf16 cache under the float32-KV reference
        model.forward(torch.tensor(IDS), cache)
    monkeypatch.setenv("PGW_ATTN_THREADS", "3")
    model.set_float_ops("native-blocked")
    assert isinstance(model.float_ops, NativeFloatOps) and model.float_ops.attn_threads == 3
    monkeypatch.setenv("PGW_FLOAT_OPS", "native-blocked-bf16kv")
    m2 = TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12))
    assert isinstance(m2.float_ops, NativeFloatOps) and m2.float_ops.kv_bf16 and m2.float_ops.attn_block


def _quality(q_pack: InMemoryPack, ids: list[int], names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out = {}
    for name in names:
        model = TransformerModel(q_pack, FloatBackend(q_pack, RING_K40_A12))
        model.set_float_ops(name)
        assert isinstance(model.float_ops, NativeFloatOps)
        model.float_ops.attn_threads = 4
        with torch.no_grad():
            logits = model.forward(torch.tensor(ids), model.new_cache(len(ids) + 8), all_logits=True)
        out[name] = torch.log_softmax(logits.to(torch.float64), dim=-1)
        del model
        gc.collect()
    return out


@pytest.mark.network
@pytest.mark.slow
def test_bf16_kv_quality_qwen05b(qwen05b_dir: Path) -> None:
    """bf16 KV vs float32 KV on Qwen2.5-0.5B over a ~700-token text (several 256-blocks): max |delta logprob|
    of the float32-KV top-1 token, top-1 agreement; and blocked (v2) vs v1 at float32 KV."""
    q_pack = InMemoryPack.from_hf(qwen05b_dir, name="qwen2.5-0.5b-instruct")
    tok = ChatTokenizer(qwen05b_dir)
    text = " ".join([
        "The gateway keeps the key and value cache of every conversation in its own memory and computes",
        "attention locally; only masked linear layers leave the customer side. Long contexts make that",
        "attention the largest piece of gateway work at large model sizes, so it is split over heads and",
        "blocks of positions on many cores, with a fixed order of summation so that every thread count",
        "gives the same bits.",
    ] * 8)
    ids = tok.apply_chat_template([{"role": "user", "content": "Summarize: " + text}])
    assert len(ids) > 2 * NATIVE_ATTN_BLOCK
    lp = _quality(q_pack, ids, ("native", "native-blocked", "native-blocked-bf16kv"))
    ref, v2, bf = lp["native"], lp["native-blocked"], lp["native-blocked-bf16kv"]
    top = ref.argmax(-1)

    def stats(x: torch.Tensor) -> tuple[float, float, float]:
        d_top = (x.gather(1, top[:, None]) - ref.gather(1, top[:, None])).abs()
        return float(d_top.max()), float((x - ref).abs().max()), float((x.argmax(-1) == top).double().mean())

    v2_top, v2_all, v2_agree = stats(v2)
    bf_top, bf_all, bf_agree = stats(bf)
    print(f"\n[quality] Qwen2.5-0.5B, {len(ids)} positions: blocked vs v1: max|dlogprob| top-1 {v2_top:.2e}, "
          f"all {v2_all:.2e}, top-1 agreement {v2_agree:.4f}; bf16 KV vs f32 KV (blocked): max|dlogprob| top-1 "
          f"{bf_top:.3e}, all-vocab {bf_all:.3e}, top-1 agreement {bf_agree:.4f}")
    assert v2_agree == 1.0 and v2_top < 1e-3
    assert bf_agree >= 0.97 and bf_top < 0.25
