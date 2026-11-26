"""The client library's KV prefix cache (:class:`pgw.sdk.prefix_cache.ClientPrefixCache`): several entries per
scope with the longest exact prefix winning, move (whole entry) vs copy (part of it), entries given back when the
engine never started, subsumed entries dropped, scopes isolated, bounded (budget, entries, TTL) and wiped."""

from __future__ import annotations

import torch

from pgw.engine.kv_cache import KVCache
from pgw.sdk.prefix_cache import ClientPrefixCache, copy_prefix, scope_key


def _cache(tokens: list[int]) -> KVCache:
    """A 2-layer cache whose position i holds the value tokens[i] (so copies can be checked)."""
    n = len(tokens)
    c = KVCache(2, 2, 4, 256)
    vals = torch.tensor(tokens, dtype=torch.float32).view(n, 1, 1).expand(n, 2, 4)
    for layer in range(2):
        c.append(layer, vals + layer, -vals - layer)
    c.advance(n)
    return c


def _put(store: ClientPrefixCache, key: str, tokens: list[int]) -> KVCache:
    c = _cache(tokens)
    store.sink(key).commit(tokens, c)
    return c


def _store(**kw: object) -> ClientPrefixCache:
    return ClientPrefixCache(max_bytes=int(kw.pop("max_bytes", 1 << 24)), ttl_s=float(kw.pop("ttl_s", 600.0)),  # type: ignore[arg-type]
                             sweep_thread=False, **kw)  # type: ignore[arg-type]


SYS = list(range(1, 21))
K = scope_key("user:u", "c1")


def test_longest_prefix_copy_and_move() -> None:
    store = _store()
    a = _put(store, scope_key("user:u", "a"), [*SYS, 100, 101, 102])
    _put(store, scope_key("user:u", "b"), [*SYS, 200, 201])
    assert len(store) == 2
    # a new conversation sharing the system prompt: a copy of 20 positions, both entries stay
    hit = store.take(K, [*SYS, 300, 301])
    assert hit is not None and hit.length == 20 and len(store) == 2
    assert hit.cache is not a and hit.cache.length == 20
    k, _ = hit.cache.layer_storage(1)
    assert torch.equal(k[:20, 0, 0], torch.tensor(SYS, dtype=torch.float32) + 1)
    hit.wipe()
    assert a._k.any()  # the copy's wipe leaves the entry
    # the next turn of conversation a: the whole entry is used and taken over (no copy)
    hit = store.take(K, [*SYS, 100, 101, 102, 103, 104])
    assert hit is not None and hit.length == 23 and hit.cache is a and len(store) == 1
    assert store.stats["moved"] == 1 and store.stats["copied"] == 1
    hit.claimed = True  # the engine owns it now
    hit.wipe()
    assert not a._k.any() and not a._v.any()


def test_unclaimed_moved_hit_goes_back_and_claimed_tokens_are_zeroed() -> None:
    store = _store()
    a = _put(store, K, SYS)
    hit = store.take(K, [*SYS, 5])
    assert hit is not None and len(store) == 0
    hit.wipe()  # the request never started (waited for masks, cancelled): back into the cache
    assert len(store) == 1 and a._k.any() and store.stats["given_back"] == 1
    hit = store.take(K, [*SYS, 5])
    assert hit is not None and hit.cache is a
    entry_tokens = hit.entry.tokens  # type: ignore[attr-defined]
    hit.claimed = True
    assert not entry_tokens.any() and hit.entry is None  # type: ignore[attr-defined]
    # after wipe_all (integrity failure, reload) a taken entry is never given back
    b = _put(store, K, SYS)
    hit = store.take(K, [*SYS, 6])
    assert hit is not None and hit.cache is b
    store.wipe_all("test")
    hit.wipe()
    assert len(store) == 0 and not b._k.any()


def test_subsumed_entries_are_dropped() -> None:
    store = _store()
    short = _put(store, K, SYS)
    longer = _put(store, K, [*SYS, 1, 2, 3])  # the old entry is a prefix of the new one
    assert len(store) == 1 and not short._k.any() and longer._k.any() and store.stats["subsumed"] == 1
    again = _put(store, K, [*SYS, 1])  # a prefix of a stored entry: not stored
    assert len(store) == 1 and not again._k.any() and store.stats["subsumed"] == 2
    other = _put(store, scope_key("user:v", "x"), SYS)  # another scope is never subsumed
    assert len(store) == 2 and other._k.any()


def test_scopes_are_isolated_and_keys_do_not_alias() -> None:
    store = _store()
    _put(store, scope_key("user:u", "a"), [*SYS, 7])
    assert store.take(scope_key("user:v", "a"), [*SYS, 7, 8]) is None
    assert store.take(scope_key("user:u\x00conv:", "a"), [*SYS, 7, 8]) is None
    assert store.take(scope_key("client", "a"), [*SYS, 7, 8]) is None
    hit = store.take(scope_key("user:u", "other"), [*SYS, 7, 8])  # same scope, any conversation
    assert hit is not None and hit.length == 21
    hit.wipe()


def test_min_tokens_and_ties() -> None:
    store = _store(min_tokens=4)
    _put(store, K, [1, 2, 3, 9, 9])
    assert store.take(K, [1, 2, 3, 4, 5]) is None  # 3 shared tokens < min_tokens
    assert store.stats["misses"] == 1
    hit = store.take(K, [1, 2, 3, 9, 9, 7])
    assert hit is not None and hit.length == 5
    hit.wipe()  # given back
    older = store._entries[next(iter(store._entries))].cache
    newer = _put(store, scope_key("user:u", "c2"), [1, 2, 3, 9, 8])
    hit = store.take(K, [1, 2, 3, 9, 7, 7])  # 4 tokens in both entries: the more recent one is copied
    assert hit is not None and hit.length == 4
    k, _ = hit.cache.layer_storage(0)
    assert torch.equal(k[:4, 0, 0], torch.tensor([1.0, 2.0, 3.0, 9.0]))
    assert older._k.any() and newer._k.any() and len(store) == 2
    hit.wipe()


def test_prompt_needs_one_new_row() -> None:
    store = _store()
    a = _put(store, K, SYS)
    hit = store.take(K, list(SYS))  # the same prompt again (a regenerate): all but its last token
    assert hit is not None and hit.length == len(SYS) - 1 and hit.cache is not a  # a copy: the entry stays
    assert len(store) == 1
    hit.wipe()
    assert store.take(K, [SYS[0]]) is None  # a one-token prompt cannot reuse anything


def test_budget_entries_and_ttl() -> None:
    probe = _store()
    _put(probe, K, SYS)
    one = probe.status()["bytes"]  # a stored entry: compacted KV + tokens
    now = [0.0]
    store = ClientPrefixCache(max_bytes=2 * one, ttl_s=60.0, max_entries=8, clock=lambda: now[0],
                              sweep_thread=False)
    a = _put(store, scope_key("s", "1"), [1, *SYS[1:]])
    now[0] = 1
    b = _put(store, scope_key("s", "2"), [2, *SYS[1:]])
    now[0] = 2
    hit = store.take(scope_key("s", "x"), [1, *SYS[1:], 5][:10])  # a copy touches entry a
    assert hit is not None
    hit.wipe()
    now[0] = 3
    c = _put(store, scope_key("s", "3"), [3, *SYS[1:]])  # over budget: the least recently used (b) goes
    assert len(store) == 2 and not b._k.any() and a._k.any() and c._k.any() and store.stats["evicted"] == 1
    now[0] = 70  # a unused since t=2, c since t=3
    assert store.sweep() == 2 and len(store) == 0 and not a._k.any() and not c._k.any()
    small = ClientPrefixCache(max_bytes=1 << 24, ttl_s=60.0, max_entries=1, sweep_thread=False)
    x = _put(small, scope_key("s", "1"), [1, 2, 3])
    _put(small, scope_key("s", "2"), [4, 5, 6])
    assert len(small) == 1 and not x._k.any()
    st = small.status()
    assert st["entries"] == 1 and "moved" in st and "copied" in st


def test_copy_prefix_is_independent() -> None:
    src = _cache([1, 2, 3, 4])
    dst = copy_prefix(src, 2)
    assert dst.length == 2 and dst.max_len == src.max_len
    src.clear()
    k, v = dst.layer_storage(0)
    assert torch.equal(k[:2, 0, 0], torch.tensor([1.0, 2.0])) and torch.equal(v[:2, 0, 0], torch.tensor([-1.0, -2.0]))


def _turn(store: ClientPrefixCache, key: str, prompt: list[int], answer: list[int]) -> tuple[int, KVCache]:
    """One request as the runtime runs it: take, then the sink; the engine stores prompt + answer[:-1]."""
    hit = store.take(key, prompt)
    sink = store.sink(key)
    reused = hit.length if hit is not None else 0
    if hit is not None:
        hit.claimed = True
    tokens = prompt + answer[:-1]
    c = _cache(tokens)
    sink.commit(tokens, c)
    return reused, c


def test_next_turn_drops_the_previous_turn_despite_retokenization() -> None:
    """A chat's next turn re-tokenizes the end of the previous answer: the previous turn's entry is replaced (not
    kept as a near-duplicate), while another conversation sharing only the system prompt is kept."""
    store = _store()
    a1 = [*SYS, 50, 51]
    _, c1 = _turn(store, K, a1, [60, 61, 62, 63])
    _, other = _turn(store, scope_key("user:u", "b"), [*SYS, 70, 71], [80, 81])
    assert len(store) == 2
    a2 = [*a1, 60, 61, 64, 65, 90, 91]  # the answer's last tokens re-tokenized (62 -> 64, 65), then a new message
    reused, c2 = _turn(store, K, a2, [92, 93])
    assert reused == len(a1) + 2 and store.stats["copied"] == 2  # partial hits: copies (b's first turn: SYS)
    assert len(store) == 2 and not c1._k.any() and c2._k.any() and other._k.any()
    assert store.stats["subsumed"] == 1
    # an answer that differs by more than tail_slack tokens (a regenerated long answer) keeps both branches
    store2 = _store(tail_slack=2)
    _turn(store2, K, a1, [60, 61, 62, 63, 66])
    _turn(store2, K, a1, [60, 67, 68, 69])
    assert len(store2) == 2


def test_new_conversation_waits_briefly_for_a_finishing_one_of_its_scope() -> None:
    import asyncio
    import time

    store = _store(scope_wait_s=0.25)
    other = scope_key("user:u", "b")

    async def run() -> tuple[float, int, float]:
        sink = store.sink(scope_key("user:u", "a"))
        sink.finishing()  # conversation a's answer is out; its cache is being stored
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, lambda: sink.commit([*SYS, 1, 2], _cache([*SYS, 1, 2])))
        t0 = time.monotonic()
        await store.wait_settled(other, 30.0)  # another conversation of the scope: waits for the commit
        waited = time.monotonic() - t0
        hit = store.take(other, [*SYS, 9])
        n = hit.length if hit is not None else 0
        if hit is not None:
            hit.wipe()
        stuck = store.sink(scope_key("user:u", "c"))
        stuck.finishing()  # never commits (say, strict-mode dummy passes): the wait is bounded
        t1 = time.monotonic()
        await store.wait_settled(other, 30.0)
        await store.wait_settled(scope_key("user:v", "x"), 30.0)  # another scope: no wait
        stuck.discard()
        return waited, n, time.monotonic() - t1

    waited, n, bounded = asyncio.run(run())
    assert 0.03 <= waited < 0.25 and n == len(SYS)
    assert 0.2 <= bounded < 1.0 and store.stats["scope_waits"] == 2
