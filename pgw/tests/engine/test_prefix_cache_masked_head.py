"""KV prefix reuse with the masked LM head (``research/reviews/prefix-cache.md``): every turn that continues a
cached prefix gives the bits (tokens, logprobs) of recomputing its whole prompt on the same masked-head engine,
the provider sees only the new rows in the prompt pass (the head: one row, as always), and masks are taken only
for the new rows. Also with the client library's multi-entry cache (:class:`pgw.sdk.prefix_cache.ClientPrefixCache`):
a system prompt shared by several conversations, and strict speculation (filler and padded passes)."""

from __future__ import annotations

from typing import Any

import pytest

from pgw.core import DEFAULT_RING
from pgw.engine import (
    ChatTokenizer,
    Generator,
    InMemoryPack,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from pgw.engine.masked_head import MaskedHeadPack, enable_masked_head, head_linear_id
from pgw.engine.model import native_float_available
from pgw.sdk.prefix_cache import ClientPrefixCache, scope_key
from tests.engine.test_prefix_cache import P1, TURN

pytestmark = pytest.mark.skipif(not native_float_available(), reason="pgw_fast without the native float path")

PARAMS = {
    "greedy": SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True, logprobs=3),
    "seeded": SamplingParams(max_tokens=8, temperature=0.9, top_p=0.95, seed=99, ignore_eos=True, logprobs=2),
}
SYSTEM = [256, 115, 121, 115, 10, 89, 111, 117, 32, 97, 114, 101, 32, 116, 101, 114, 115, 101, 46, 258, 10]


@pytest.fixture(scope="module")
def head_pack(qwen_pack: InMemoryPack) -> MaskedHeadPack:
    return MaskedHeadPack(qwen_pack)


def _model(pack: MaskedHeadPack, masked: bool) -> tuple[TransformerModel, LocalComputeClient | None]:
    client = LocalComputeClient(pack, record=True) if masked else None
    be: Any = (MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint) if masked
               else PlainBackend(pack, DEFAULT_RING))
    m = TransformerModel(pack, be)
    m.set_float_ops("native")
    enable_masked_head(m)
    return m, client


def _store() -> ClientPrefixCache:
    return ClientPrefixCache(max_bytes=1 << 26, ttl_s=600.0, sweep_thread=False)


def test_masked_head_engine_supports_reuse(head_pack: MaskedHeadPack, tiny_tokenizer: ChatTokenizer) -> None:
    gen = Generator(_model(head_pack, True)[0], tiny_tokenizer)
    assert gen.prefix_reuse_supported() == (True, "ok")
    torch_head = TransformerModel(head_pack, PlainBackend(head_pack, DEFAULT_RING))
    enable_masked_head(torch_head)  # torch float ops: still refused
    assert not Generator(torch_head, tiny_tokenizer).prefix_reuse_supported()[0]


@pytest.mark.parametrize("sampling", ["greedy", "seeded"])
def test_reuse_bit_identical_masked_head(head_pack: MaskedHeadPack, tiny_tokenizer: ChatTokenizer,
                                         sampling: str) -> None:
    params = PARAMS[sampling]
    model, client = _model(head_pack, True)
    gen = Generator(model, tiny_tokenizer)
    ref_model, ref_client = _model(head_pack, True)
    ref = Generator(ref_model, tiny_tokenizer)
    plain = Generator(_model(head_pack, False)[0], tiny_tokenizer)  # the pinned plaintext reference
    assert client is not None and ref_client is not None
    head = head_linear_id(head_pack.arch.num_layers).key
    store = _store()
    key = scope_key("user:a", "c1")
    prompt = list(P1)
    for t in range(4):
        n0 = len(client.received)
        hit = store.take(key, prompt)
        reused = hit.length if hit is not None else 0
        need = gen.mask_rows_needed(len(prompt), params.max_tokens, reused)
        got = gen.generate(prompt, params, prefix=hit, prefix_sink=store.sink(key),
                           session=LocalMaskReservation(head_pack, DEFAULT_RING, need), own_session=True)
        want = ref.generate(prompt, params, session=LocalMaskReservation(
            head_pack, DEFAULT_RING, ref.mask_rows_needed(len(prompt), params.max_tokens)), own_session=True)
        base = plain.generate(prompt, params)
        assert got.token_ids == want.token_ids == base.token_ids, t
        assert got.logprobs == want.logprobs == base.logprobs, t
        assert got.masks_consumed == want.masks_consumed - reused == len(prompt) - reused + params.max_tokens - 1
        layer0 = [int(x.shape[0]) for lid, x in client.received[n0:] if lid == "0.qkv"]
        heads = [int(x.shape[0]) for lid, x in client.received[n0:] if lid == head]
        assert layer0 == [len(prompt) - reused] + [1] * (params.max_tokens - 1)
        assert heads == [1] * params.max_tokens  # the head sees one row per pass, hit or not
        if t:
            assert reused == len(prompt) - len(TURN[t - 1]) - 1  # the whole previous turn but its last token
            assert got.timings_ms["prefix_reused"] == reused
        prompt = prompt + got.token_ids + TURN[t]
    assert model.backend.stats.integrity_failures == 0
    assert store.stats["moved"] == 3 and store.stats["copied"] == 0


def test_shared_system_prompt_across_conversations(head_pack: MaskedHeadPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Two conversations of one scope reuse the shared system prompt (a copy, the entry stays) and each continues
    its own history (a move); bits as recomputing; another scope never sees them."""
    params = PARAMS["greedy"]
    gen = Generator(_model(head_pack, True)[0], tiny_tokenizer)
    ref = Generator(_model(head_pack, False)[0], tiny_tokenizer)
    store = _store()
    msgs = {"a": [104, 105, 10], "b": [111, 107, 63, 10]}
    prompts = {c: [*SYSTEM, 256, 117, 10, *m] for c, m in msgs.items()}
    seen: dict[str, int] = {}
    for rnd in range(2):
        for conv in ("a", "b"):
            key = scope_key("user:u", conv)
            p = prompts[conv]
            hit = store.take(key, p)
            reused = hit.length if hit is not None else 0
            got = gen.generate(p, params, prefix=hit, prefix_sink=store.sink(key),
                               session=LocalMaskReservation(head_pack, DEFAULT_RING,
                                                            gen.mask_rows_needed(len(p), params.max_tokens, reused)),
                               own_session=True)
            want = ref.generate(p, params)
            assert got.token_ids == want.token_ids and got.logprobs == want.logprobs
            seen[f"{conv}{rnd}"] = reused
            prompts[conv] = p + got.token_ids + TURN[rnd]
    assert seen["a0"] == 0
    assert seen["b0"] >= len(SYSTEM) + 3  # conversation b reuses the system prompt cached by a
    assert seen["a1"] > len(SYSTEM) + 10 and seen["b1"] > len(SYSTEM) + 10  # each continues its own history
    assert store.stats["copied"] >= 1 and store.stats["moved"] >= 2
    assert store.take(scope_key("user:other", "a"), prompts["a"]) is None


@pytest.mark.parametrize("pad", [None, 2], ids=["fillers", "padded"])
def test_reuse_strict_speculation_masked_head(qwen_pack: InMemoryPack, head_pack: MaskedHeadPack,
                                              tiny_tokenizer: ChatTokenizer, pad: int | None) -> None:
    """Strict speculation with the masked head and reuse: same bits as recomputing, same pass schedule but the
    prompt pass, the head one item of fixed rows per pass."""
    params = PARAMS["greedy"]
    extra: dict[str, Any] = {"spec_mask_rows": pad} if pad else {}

    def make() -> tuple[Generator, LocalComputeClient]:
        model, client = _model(head_pack, True)
        assert client is not None
        if pad and not model.supports_pad_rows:
            pytest.skip("padding rows need the native trip kernels")
        draft = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))  # proposals only
        return Generator(model, tiny_tokenizer, draft=draft, spec_rows=3, spec_strict=True, **extra), client

    gen, client = make()
    ref, client0 = make()
    head = head_linear_id(head_pack.arch.num_layers).key
    store = _store()
    key = scope_key("user:", "")
    prompt = list(P1)
    for t in range(3):
        n0, n1 = len(client0.received), len(client.received)
        hit = store.take(key, prompt)
        reused = hit.length if hit is not None else 0
        need = gen.mask_rows_needed(len(prompt), params.max_tokens, reused)
        got = gen.generate(prompt, params, prefix=hit, prefix_sink=store.sink(key),
                           session=LocalMaskReservation(head_pack, DEFAULT_RING, need), own_session=True)
        want = ref.generate(prompt, params, session=LocalMaskReservation(
            head_pack, DEFAULT_RING, ref.mask_rows_needed(len(prompt), params.max_tokens)), own_session=True)
        assert got.token_ids == want.token_ids and got.logprobs == want.logprobs
        seen = [int(x.shape[0]) for lid, x in client.received[n1:] if lid == "0.qkv"]
        seen0 = [int(x.shape[0]) for lid, x in client0.received[n0:] if lid == "0.qkv"]
        assert seen0[0] == len(prompt) and seen[0] == len(prompt) - reused
        assert seen[1:] == seen0[1:]
        heads = [int(x.shape[0]) for lid, x in client.received[n1:] if lid == head]
        heads0 = [int(x.shape[0]) for lid, x in client0.received[n0:] if lid == head]
        assert heads == heads0  # the head item does not depend on the hit
        if t:
            assert reused > 0
            assert need == ref.mask_rows_needed(len(prompt), params.max_tokens) - reused
        prompt = prompt + got.token_ids + TURN[t]
