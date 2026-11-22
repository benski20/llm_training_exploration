"""Conversation KV-prefix reuse (``pgw.engine.prefix_cache``): a turn that continues a cached conversation gives
the bits of recomputing its whole prompt (greedy, seeded sampling, speculation, strict speculation, masked), takes
masks only for the new rows, and the cache is bound to its key, bounded (TTL, budget), wiped and fail-closed."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING, ConfigError, IntegrityError
from pgw.engine import (
    ChatTokenizer,
    FloatBackend,
    Generator,
    InMemoryPack,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from pgw.engine.kv_cache import KVCache
from pgw.engine.model import native_float_available
from pgw.engine.prefix_cache import ConversationPrefixCache

pytestmark = pytest.mark.skipif(not native_float_available(), reason="pgw_fast without the native float path")

P1 = [256, 72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 10]
#: what a chat template adds between turns (end of the assistant message, the next user message)
TURN = [[258, 10, 256, 117, 115, 101, 114, 10, 104, 105, 10], [258, 10, 256, 111, 107, 63, 10],
        [258, 10, 256, 109, 111, 114, 101, 10], [258, 10, 256, 98, 121, 101, 10]]


def _native(pack: InMemoryPack, backend: Any = None) -> TransformerModel:
    m = TransformerModel(pack, backend if backend is not None else PlainBackend(pack, DEFAULT_RING))
    m.set_float_ops("native")
    return m


def _store(**kw: Any) -> ConversationPrefixCache:
    return ConversationPrefixCache(max_bytes=kw.pop("max_bytes", 1 << 26), ttl_s=kw.pop("ttl_s", 600.0),
                                   sweep_thread=False, **kw)


def _chat(gen: Generator, ref: Generator, params: SamplingParams, turns: int = 4, *,
          session: Callable[[list[int], int], Any] | None = None, store: ConversationPrefixCache | None = None,
          key: str = "key:a\x00user:") -> list[tuple[Any, Any, int]]:
    """Run ``turns`` chat turns with reuse on ``gen`` and the same prompts recomputed on ``ref``."""
    store = store or _store()
    prompt = list(P1)
    out = []
    for t in range(turns):
        hit = store.take(key, prompt)
        reused = hit.length if hit is not None else 0
        kw: dict[str, Any] = {"prefix": hit, "prefix_sink": store.sink(key)}
        if session is not None:
            kw.update(session=session(prompt, reused), own_session=True)
        got = gen.generate(prompt, params, **kw)
        kw0: dict[str, Any] = {}
        if session is not None:
            kw0.update(session=session(prompt, 0), own_session=True)
        want = ref.generate(prompt, params, **kw0)
        out.append((got, want, reused))
        prompt = prompt + got.token_ids + TURN[t % len(TURN)]
    return out


def _assert_same(rows: list[tuple[Any, Any, int]], prompt_rows: bool = True) -> None:
    for i, (got, want, reused) in enumerate(rows):
        assert got.token_ids == want.token_ids, i
        assert got.logprobs == want.logprobs, i
        assert got.finish_reason == want.finish_reason
        if i:
            assert reused > 0, "every later turn continues the cached conversation"
            assert got.timings_ms["prefix_reused"] == reused
        if prompt_rows:  # one mask row per prompt-pass row and per decode row
            assert got.masks_consumed == want.masks_consumed - reused


@pytest.mark.parametrize("sampling", ["greedy", "seeded"])
def test_reuse_bit_identical_plain(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer, sampling: str) -> None:
    params = (SamplingParams(max_tokens=9, temperature=0.0, ignore_eos=True, logprobs=3) if sampling == "greedy"
              else SamplingParams(max_tokens=9, temperature=0.9, top_p=0.95, seed=1234, ignore_eos=True,
                                  logprobs=3))
    gen = Generator(_native(qwen_pack), tiny_tokenizer)
    ref = Generator(_native(qwen_pack), tiny_tokenizer)
    rows = _chat(gen, ref, params, turns=5)
    _assert_same(rows)
    # the first reuse is the whole previous turn but its last token (never fed through the model)
    assert rows[1][2] == len(P1) + 9 - 1


def test_reuse_bit_identical_masked(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """Masked == the plaintext recompute, and the reservation holds only the new rows + max_tokens."""
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True, logprobs=2)
    client = LocalComputeClient(qwen_pack, record=True)
    masked = _native(qwen_pack, MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint))
    gen = Generator(masked, tiny_tokenizer)
    ref = Generator(_native(qwen_pack), tiny_tokenizer)
    sizes: list[int] = []

    def session(prompt: list[int], reused: int) -> Any:
        need = gen.mask_rows_needed(len(prompt), params.max_tokens, reused)
        sizes.append(need)
        return LocalMaskReservation(qwen_pack, DEFAULT_RING, need)

    store = _store()
    prompt = list(P1)
    for t in range(3):
        before = len(client.received)
        hit = store.take("k", prompt)
        reused = hit.length if hit is not None else 0
        got = gen.generate(prompt, params, prefix=hit, prefix_sink=store.sink("k"), session=session(prompt, reused),
                           own_session=True)
        want = ref.generate(prompt, params)
        assert got.token_ids == want.token_ids and got.logprobs == want.logprobs
        rows = [int(x.shape[0]) for lid, x in client.received[before:] if lid == "0.qkv"]
        # the provider sees the new rows' prefill pass, then one row per generated token
        assert rows == [len(prompt) - reused] + [1] * (params.max_tokens - 1)
        assert got.masks_consumed == len(prompt) - reused + params.max_tokens - 1
        assert sizes[-1] == len(prompt) - reused + params.max_tokens
        if t:
            assert reused == len(prompt) - len(TURN[t - 1]) - 1  # the last generated token was never fed
        prompt = prompt + got.token_ids + TURN[t]


@pytest.mark.parametrize("mode", ["draft", "lookup", "strict"])
def test_reuse_with_speculation(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer, mode: str) -> None:
    params = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True, logprobs=2)
    target = _native(qwen_pack)
    draft = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))  # proposals only
    kw: dict[str, Any]
    if mode == "draft":
        kw = {"draft": draft, "spec_tokens": 3}
    elif mode == "lookup":
        kw = {"lookup": 3, "lookup_min_n": 2, "spec_tokens": 3}
    else:  # padded strict speculation: test_reuse_strict_masked_pass_schedule
        kw = {"draft": draft, "spec_rows": 3, "spec_strict": True}
    gen = Generator(target, tiny_tokenizer, **kw)
    ref = Generator(_native(qwen_pack), tiny_tokenizer)
    rows = _chat(gen, ref, params, turns=4)
    _assert_same(rows, prompt_rows=False)
    if mode in ("draft", "strict"):
        assert gen.spec_stats["spec_accepted"] > 0  # the tiny model drafts for itself


@pytest.mark.parametrize("padded", [False, True])
def test_reuse_strict_masked_pass_schedule(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer,
                                           padded: bool) -> None:
    """Strict speculation with reuse: same bits as recomputing; the provider's pass schedule of a continued turn is
    the recompute's with only the prompt pass shorter (the new rows); the reservation is sized to the new rows."""
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True, logprobs=2)
    draft = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    extra: dict[str, Any] = {"spec_mask_rows": 3, "spec_mask_credit": 4} if padded else {}

    def make() -> tuple[Generator, LocalComputeClient]:
        client = LocalComputeClient(qwen_pack, record=True)
        be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint)
        model = _native(qwen_pack, be)
        if padded and not model.supports_pad_rows:
            pytest.skip("padding rows need the native trip kernels")
        return Generator(model, tiny_tokenizer, draft=draft, spec_rows=3, spec_strict=True, **extra), client

    gen, client = make()
    ref, client0 = make()
    store = _store()
    prompt = list(P1)
    for t in range(3):
        n0, n1 = len(client0.received), len(client.received)
        hit = store.take("k", prompt)
        reused = hit.length if hit is not None else 0
        need = gen.mask_rows_needed(len(prompt), params.max_tokens, reused)
        got = gen.generate(prompt, params, prefix=hit, prefix_sink=store.sink("k"),
                           session=LocalMaskReservation(qwen_pack, DEFAULT_RING, need), own_session=True)
        want = ref.generate(prompt, params, session=LocalMaskReservation(
            qwen_pack, DEFAULT_RING, ref.mask_rows_needed(len(prompt), params.max_tokens)), own_session=True)
        assert got.token_ids == want.token_ids and got.logprobs == want.logprobs
        seen = [int(x.shape[0]) for lid, x in client.received[n1:] if lid == "0.qkv"]
        seen0 = [int(x.shape[0]) for lid, x in client0.received[n0:] if lid == "0.qkv"]
        assert seen0[0] == len(prompt) and seen[0] == len(prompt) - reused
        assert seen[1:] == seen0[1:]  # same number of passes, same rows per decode pass
        assert got.timings_ms["passes"] == want.timings_ms["passes"]
        if t:
            assert reused > 0
            assert need == ref.mask_rows_needed(len(prompt), params.max_tokens) - reused
        prompt = prompt + got.token_ids + TURN[t]


def test_partial_and_wrong_prefix(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    params = SamplingParams(max_tokens=6, temperature=0.0, ignore_eos=True, logprobs=2)
    gen = Generator(_native(qwen_pack), tiny_tokenizer)
    ref = Generator(_native(qwen_pack), tiny_tokenizer)
    store = _store()
    r1 = gen.generate(P1, params, prefix_sink=store.sink("k"))
    cached = P1 + r1.token_ids[:-1]
    # an edited earlier message: only the common prefix is reused, exactly
    edited = P1[:6] + [65, 66, 67] + TURN[0]
    hit = store.take("k", edited)
    assert hit is not None and hit.length == 6
    got = gen.generate(edited, params, prefix=hit, prefix_sink=store.sink("k"))
    assert got.token_ids == ref.generate(edited, params).token_ids
    # a prompt sharing nothing: a miss, and the entry is wiped
    entry_cache = store._entries["k"].cache
    assert store.take("k", [65, 66, 67, 68]) is None
    assert store.stats["mismatches"] == 1 and len(store) == 0
    assert not entry_cache._k.any() and not entry_cache._v.any()
    # the prefix the engine was handed must match the prompt it runs
    store2 = _store()
    gen.generate(P1, params, prefix_sink=store2.sink("k"))
    hit = store2.take("k", [*cached, 1])
    assert hit is not None
    hit.length = len(cached) + 1  # longer than the prompt allows
    with pytest.raises(ConfigError):
        gen.generate([*cached, 1], params, prefix=hit)
    assert not hit.cache._k.any()  # wiped on the refusal


def test_cross_key_isolation(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    params = SamplingParams(max_tokens=5, temperature=0.0, ignore_eos=True)
    gen = Generator(_native(qwen_pack), tiny_tokenizer)
    store = _store()
    r1 = gen.generate(P1, params, prefix_sink=store.sink("key:a\x00user:"))
    p2 = P1 + r1.token_ids + TURN[0]
    assert store.take("key:b\x00user:", p2) is None  # another API key
    assert store.take("key:a\x00user:u2", p2) is None  # another user of the same key
    hit = store.take("key:a\x00user:", p2)
    assert hit is not None and hit.length == len(P1) + 4
    assert store.take("key:a\x00user:", p2) is None  # taken: never shared by two requests
    hit.wipe()


def _filled(store: ConversationPrefixCache, key: str, n: int = 10) -> KVCache:
    c = KVCache(2, 2, 16, 64)
    for layer in range(2):
        c.append(layer, torch.ones(n, 2, 16), torch.ones(n, 2, 16))
    c.advance(n)
    store.sink(key).commit(list(range(1, n + 1)), c)
    return c


def test_eviction_ttl_and_wipe() -> None:
    now = [0.0]
    one = 2 * 2 * 10 * 2 * 16 * 4 + 10 * 8  # KV bytes + tokens of a 10-position entry
    store = ConversationPrefixCache(max_bytes=2 * one, ttl_s=60.0, clock=lambda: now[0], sweep_thread=False)
    a = _filled(store, "a")
    now[0] = 1
    b = _filled(store, "b")
    now[0] = 2
    c = _filled(store, "c")  # over budget: the least recently used ("a") is wiped
    assert len(store) == 2 and store.stats["evicted"] == 1
    assert not a._k.any() and b._k.any() and c._k.any()
    assert store.status()["bytes"] == 2 * one
    now[0] = 61.5  # "b" unused for 60.5 s
    assert store.sweep() == 1 and not b._k.any() and c._k.any()
    assert store.take("b", list(range(1, 12))) is None
    # an entry larger than the whole budget is never stored
    big = KVCache(2, 2, 16, 64)
    for layer in range(2):
        big.append(layer, torch.ones(40, 2, 16), torch.ones(40, 2, 16))
    big.advance(40)
    assert not store.sink("big").commit(list(range(40)), big) and not big._k.any()
    # wipe_all (integrity failure, alarm, reload): everything zeroed; requests running then cannot store
    late = store.sink("late")
    assert store.wipe_all("test") == 1 and not c._k.any() and len(store) == 0
    d = KVCache(2, 2, 16, 64)
    d.append(0, torch.ones(3, 2, 16), torch.ones(3, 2, 16))
    d.advance(3)
    assert not late.commit([1, 2, 3], d) and not d._k.any() and store.stats["refused"] >= 1
    # the entry limit
    small = ConversationPrefixCache(max_bytes=1 << 20, ttl_s=60.0, max_entries=1, sweep_thread=False)
    e1 = _filled(small, "x")
    _filled(small, "y")
    assert len(small) == 1 and not e1._k.any()


def test_ttl_sweeper_thread() -> None:
    store = ConversationPrefixCache(max_bytes=1 << 20, ttl_s=0.01, sweep_thread=True)
    try:
        c = _filled(store, "a")
        import time

        deadline = time.monotonic() + 5
        while len(store) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(store) == 0 and not c._k.any()
    finally:
        store.close()


def test_integrity_failure_discards(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """A turn that fails its integrity check stores nothing and wipes the prefix it was given."""
    params = SamplingParams(max_tokens=6, temperature=0.0, ignore_eos=True)
    gen_ok = Generator(_native(qwen_pack), tiny_tokenizer)
    store = _store()
    r1 = gen_ok.generate(P1, params, prefix_sink=store.sink("k"))
    p2 = P1 + r1.token_ids + TURN[0]

    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        if lid.endswith("down"):
            y = y.clone()
            y[0, 0] += 1
        return y

    client = LocalComputeClient(qwen_pack, tamper=tamper)
    gen = Generator(_native(qwen_pack, MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint)),
                    tiny_tokenizer)
    hit = store.take("k", p2)
    assert hit is not None
    with pytest.raises(IntegrityError):
        gen.generate(p2, params, prefix=hit, prefix_sink=store.sink("k"),
                     session=LocalMaskReservation(qwen_pack, DEFAULT_RING, 64), own_session=True)
    assert len(store) == 0 and not hit.cache._k.any() and not hit.cache._v.any()
    assert not store.settling("k")


def test_failed_deferred_check_discards(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """The cache is stored only after the session's deferred checks joined cleanly."""
    params = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    model = _native(qwen_pack)
    gen = Generator(model, tiny_tokenizer)
    store = _store()
    model.backend.release_session_checks = lambda session: False  # type: ignore[attr-defined]
    gen.generate(P1, params, prefix_sink=store.sink("k"), session=LocalMaskReservation(qwen_pack, DEFAULT_RING, 64),
                 own_session=True)
    assert len(store) == 0 and not store.settling("k")
    model.backend.release_session_checks = lambda session: True  # type: ignore[attr-defined]
    gen.generate(P1, params, prefix_sink=store.sink("k"), session=LocalMaskReservation(qwen_pack, DEFAULT_RING, 64),
                 own_session=True)
    assert len(store) == 1


def test_cancelled_turn_is_not_stored(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    gen = Generator(_native(qwen_pack), tiny_tokenizer)
    store = _store()
    it = gen.stream(P1, params, prefix_sink=store.sink("k"))
    next(it)
    it.close()
    assert len(store) == 0 and not store.settling("k")


def test_refused_off_the_row_invariant_path(qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer) -> None:
    """torch float ops (shape-dependent kernel blockings) and the float backend refuse reuse."""
    params = SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True)
    torch_gen = Generator(TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING)), tiny_tokenizer)
    ok, why = torch_gen.prefix_reuse_supported()
    assert not ok and "native" in why
    store = _store()
    sink = store.sink("k")
    with pytest.raises(ConfigError, match="not exact"):
        torch_gen.generate(P1, params, prefix_sink=sink)
    assert sink.settled
    float_gen = Generator(_native(qwen_pack, FloatBackend(qwen_pack, DEFAULT_RING)), tiny_tokenizer)
    assert not float_gen.prefix_reuse_supported()[0]
    assert Generator(_native(qwen_pack), tiny_tokenizer).prefix_reuse_supported() == (True, "ok")


def test_native_rows_do_not_depend_on_the_split(qwen_pack: InMemoryPack) -> None:
    """The property reuse rests on: K/V and logits of a split prompt (block, single rows, block) are the bits of one
    block, also across reuse of a truncated cache."""
    model = _native(qwen_pack)
    ids = (P1 * 3)[:40]
    c1 = model.new_cache(64)
    full = model.forward(torch.tensor(ids), c1, all_logits=True)
    c2 = model.new_cache(64)
    parts = [model.forward(torch.tensor(ids[:17]), c2, all_logits=True)]
    for t in ids[17:21]:
        parts.append(model.forward(torch.tensor([t]), c2, all_logits=True))
    parts.append(model.forward(torch.tensor(ids[21:]), c2, all_logits=True))
    assert torch.equal(torch.cat(parts), full)
    assert torch.equal(c1._k[:, :40], c2._k[:, :40]) and torch.equal(c1._v[:, :40], c2._v[:, :40])
    c2.rebind(25, 64)
    tail = model.forward(torch.tensor(ids[25:]), c2, all_logits=True)
    assert torch.equal(tail, full[25:])
