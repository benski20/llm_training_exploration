"""The client library's KV prefix cache (``prefix_cache``; design: ``research/reviews/prefix-cache.md``).

Every prompt token the model has already seen in this process costs, without a cache, one masked row on every
round trip of the prompt pass and one single-use mask slot. :class:`ClientPrefixCache` keeps the plaintext KV
cache of finished requests in this process's RAM and hands a later request the longest exact token prefix it
shares with any of them, so only the rest of its prompt runs (no rows, no masks for the cached positions).

It is :class:`pgw.engine.prefix_cache.ConversationPrefixCache` (the gateway's one-entry-per-conversation cache,
SPEC 9.4b) with these differences, everything else (sinks, settle waits, epochs, TTL sweeper, wipes) inherited:

* **Several entries per scope, longest prefix wins.** A scope is the OpenAI ``user`` of the request (default) or
  the whole client (``prefix_cache_scope="client"``). A request looks at every entry of its scope; a system
  prompt shared by many conversations, a regenerated answer and an edited message all reuse what they share.
* **Move or copy.** A request that uses an entry whole (the next turn of a conversation) takes it over (no
  copy; its own cache replaces it when it ends). A request that uses part of an entry gets a copy of those
  positions and the entry stays for others. A taken entry that the engine never started on (the request waited
  for masks, timed out or was cancelled in the queue) goes back into the cache.
* **Subsumed entries are dropped.** Storing a request's cache drops the entries of its scope whose tokens are a
  prefix of it, or that it continues but for at most ``tail_slack`` trailing tokens of their answer (the next turn
  of a chat usually re-tokenizes the end of the previous answer); a cache that is itself a prefix of a stored
  entry is not stored.

Exactness is the engine's (:meth:`pgw.engine.generate.Generator.prefix_reuse_supported`: row-invariant native
float path, row-exact backend and head). The cache is bound to one model runtime (fingerprint, ring, head and
float path: the runtime builds it and closes it on reload or stop), RAM only, never written to disk, never sent
anywhere; every eviction, expiry and wipe zeroes the tensors and the token array.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pgw.engine.kv_cache import KVCache
from pgw.engine.prefix_cache import ConversationPrefixCache, PrefixHit, PrefixSink, _common_prefix, _Entry

__all__ = ["ClientPrefixCache", "copy_prefix", "scope_key"]

log = logging.getLogger(__name__)

#: Separates the lookup scope from the conversation part of a prefix key (:func:`scope_key`).
_CONV = "\x00conv:"


def scope_key(scope: str, conversation: str | None) -> str:
    """The prefix key of a request: its lookup ``scope`` (as a fixed-length digest, so no ``user`` or
    conversation string can alias another scope) and its conversation (settle waits are per key)."""
    return f"{hashlib.sha256(scope.encode()).hexdigest()}{_CONV}{conversation or ''}"


def _scope_of(key: str) -> str:
    return key[:64] if key[64 : 64 + len(_CONV)] == _CONV else key


def copy_prefix(src: KVCache, length: int) -> KVCache:
    """A new cache holding a copy of the first ``length`` committed positions of ``src``."""
    dst = KVCache(src.num_layers, src.num_kv_heads, src.head_dim, src.max_len, src.dtype, src.device,
                  initial_len=max(1, length))
    for layer in range(src.num_layers):
        k, v = src.layer_storage(layer)
        dst.append(layer, k[:length], v[:length])
    dst.advance(length)
    return dst


class _ScopedEntry(_Entry):
    __slots__ = ("prompt_len", "scope")

    def __init__(self, tokens: np.ndarray, cache: KVCache, now: float, scope: str, prompt_len: int | None = None):
        super().__init__(tokens, cache, now)
        self.scope = scope
        #: How many of ``tokens`` were the request's prompt (the rest it generated); None = unknown.
        self.prompt_len = prompt_len


class _ClientSink(PrefixSink):
    """A sink that knows its request's prompt length (for :attr:`ClientPrefixCache.tail_slack`)."""

    def __init__(self, owner: ConversationPrefixCache, key: str, epoch: int, prompt_len: int | None):
        super().__init__(owner, key, epoch)
        self.prompt_len = prompt_len


@dataclass
class _ClientHit(PrefixHit):
    """A hit; a taken (moved) entry goes back to its cache when the engine never claimed it."""

    owner: ClientPrefixCache | None = None
    entry: _ScopedEntry | None = None
    epoch: int = -1

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "claimed" and value:  # the engine owns the cache now: the token copy is no longer needed
            entry = self.__dict__.get("entry")
            if entry is not None:
                entry.tokens.fill(0)
                super().__setattr__("entry", None)
                super().__setattr__("owner", None)

    def wipe(self) -> None:
        owner, entry = self.owner, self.entry
        self.owner = self.entry = None
        if owner is not None and entry is not None and not self.claimed and owner._give_back(entry, self.epoch):
            return
        self.cache.clear()
        if entry is not None:
            entry.tokens.fill(0)


class ClientPrefixCache(ConversationPrefixCache):
    """Multi-entry, longest-prefix KV cache of one model runtime (see the module docstring).

    Args:
        max_bytes, ttl_s, max_entries, clock, sweep_thread: as :class:`ConversationPrefixCache` (the budget
            counts stored entries; a request's own copy is its live KV cache until it ends).
        scope_wait_s: How long a request waits for a just-finished request of another conversation of its scope
            to store its cache (default 0.25 s; its own conversation's previous turn: ``prefix_cache_wait_s``).
        scope_of: Maps a prefix key to its lookup scope (default: the scope digest of :func:`scope_key`; a key
            not made by it is its own scope).
        min_tokens: A hit must reuse at least this many tokens.
        tail_slack: Storing an entry also drops an entry of its scope whose whole prompt the new one holds and that
            differs from it only in its last ``tail_slack`` tokens (default 32): a conversation's next turn usually
            re-tokenizes the end of the previous answer, so the previous turn's entry is a prefix of the new one
            but for a few tokens.
    """

    def __init__(self, *, max_bytes: int, ttl_s: float, max_entries: int = 1024, min_tokens: int = 1,
                 tail_slack: int = 32, scope_wait_s: float = 0.25, scope_of: Callable[[str], str] = _scope_of,
                 **kw: Any):
        super().__init__(max_bytes=max_bytes, ttl_s=ttl_s, max_entries=max_entries, **kw)
        self.min_tokens = max(1, int(min_tokens))
        self.tail_slack = max(0, int(tail_slack))
        self._prompt_lens: dict[str, int] = {}
        self._scope_fn = scope_of
        self._ids = itertools.count(1)
        self.scope_wait_s = max(0.0, float(scope_wait_s))
        self.stats.update({"moved": 0, "copied": 0, "copied_tokens": 0, "given_back": 0, "subsumed": 0,
                           "scope_waits": 0})

    # ---- request side ----------------------------------------------------------------------------------
    def scope_settling(self, key: str) -> bool:
        """A finished request of ``key``'s scope (any conversation) is on its way into the cache."""
        scope = self._scope_fn(key)
        with self._lock:
            return any(self._scope_fn(k) == scope for k in self._marks)

    async def wait_settled(self, key: str, timeout_s: float) -> None:
        """Wait (at most ``timeout_s``) for the commit of a finished turn of ``key``'s own conversation, then (at
        most ``scope_wait_s``) for those of other conversations of its scope: a new conversation then finds the
        system prompt a just-finished one computed (its commit lands within milliseconds once the session's
        deferred checks are joined; the dummy passes of strict speculation are not waited for that long)."""
        await super().wait_settled(key, timeout_s)
        limit = min(float(timeout_s), self.scope_wait_s)
        if limit <= 0 or not self.scope_settling(key):
            return
        self.stats["scope_waits"] += 1
        deadline = time.monotonic() + limit
        delay = 0.001
        while self.scope_settling(key) and time.monotonic() < deadline:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.02)

    def sink(self, key: str) -> PrefixSink:
        """The sink of the request whose :meth:`take` just ran for ``key`` (the runtime calls the two together)."""
        with self._lock:
            return _ClientSink(self, key, self._epoch, self._prompt_lens.pop(key, None))

    def take(self, key: str, prompt: Sequence[int]) -> PrefixHit | None:
        """The longest cached prefix of ``prompt`` in ``key``'s scope (at most ``len(prompt) - 1`` tokens: the
        prompt pass needs one new row for the logits), moved when the entry is used whole, else copied."""
        scope = self._scope_fn(key)
        limit = len(prompt) - 1
        with self._lock:
            self._sweep_locked()
            if len(self._prompt_lens) > 4 * self.max_entries:  # keys whose sink was never asked for
                self._prompt_lens.clear()
            self._prompt_lens[key] = len(prompt)
            best_id, best_m = None, 0
            want = np.asarray(prompt, dtype=np.int64)
            if limit >= self.min_tokens:
                for eid, e in self._entries.items():  # oldest first: a later equal match (more recent) wins
                    if not isinstance(e, _ScopedEntry) or e.scope != scope:
                        continue
                    m = min(_common_prefix(e.tokens, want), limit, e.cache.length)
                    if m >= best_m and m > 0:
                        best_id, best_m = eid, m
            if best_id is None or best_m < self.min_tokens:
                self.stats["misses"] += 1
                return None
            e = self._entries[best_id]
            assert isinstance(e, _ScopedEntry)
            self.stats["hits"] += 1
            self.stats["reused_tokens"] += best_m
            if best_m == e.cache.length:  # the entry is used whole: the request takes it over
                del self._entries[best_id]
                self._bytes -= e.nbytes
                self.stats["moved"] += 1
                return _ClientHit(e.cache, best_m, owner=self, entry=e, epoch=self._epoch)
            # part of it: a copy (under the lock, so no eviction can wipe the source meanwhile)
            e.t_used = self._clock()
            self._entries.move_to_end(best_id)
            self.stats["copied"] += 1
            self.stats["copied_tokens"] += best_m
            return PrefixHit(copy_prefix(e.cache, best_m), best_m)

    # ---- internals ----------------------------------------------------------------------------------------
    def _give_back(self, e: _ScopedEntry, epoch: int) -> bool:
        """Put a taken entry the engine never started on back (False: wiped, the caller clears it)."""
        with self._lock:
            if epoch != self._epoch or self._stop.is_set():
                return False
            self.stats["given_back"] += 1
        return self._store(e, epoch, refused_stat=False)

    def _commit(self, sink: PrefixSink, tokens: Sequence[int], cache: KVCache) -> bool:
        n = cache.length
        if n <= 0 or len(tokens) < n:
            cache.clear()
            return False
        cache.compact()
        arr = np.asarray(tokens[:n], dtype=np.int64).copy()
        e = _ScopedEntry(arr, cache, self._clock(), self._scope_fn(sink.key), getattr(sink, "prompt_len", None))
        return self._store(e, sink.epoch)

    def _store(self, e: _ScopedEntry, epoch: int, refused_stat: bool = True) -> bool:
        wiped: list[_Entry] = []
        stored = False
        with self._lock:
            e.t_used = self._clock()
            if epoch != self._epoch or e.nbytes > self.max_bytes:
                if refused_stat:
                    self.stats["refused"] += 1
                wiped.append(e)
            else:
                self._sweep_locked()
                n = int(e.tokens.shape[0])
                covered = False
                for eid, old in list(self._entries.items()):
                    if not isinstance(old, _ScopedEntry) or old.scope != e.scope:
                        continue
                    k = int(old.tokens.shape[0])
                    cp = _common_prefix(old.tokens, e.tokens)
                    if cp == n and n < k:  # the new cache is a prefix of a stored one: keep that one
                        old.t_used = e.t_used
                        self._entries.move_to_end(eid)
                        covered = True
                        break
                    plen = old.prompt_len
                    if (cp == k and k <= n) or (plen is not None and cp >= plen and k - cp <= self.tail_slack):
                        # the old entry is a prefix of the new one, or the new one holds the old request's whole
                        # prompt and the two differ only in the last tail_slack tokens of the old answer (the next
                        # turn re-tokenizes the end of the previous answer): it would (almost) never be the better
                        # match again
                        del self._entries[eid]
                        self._bytes -= old.nbytes
                        self.stats["subsumed"] += 1
                        wiped.append(old)
                if covered:
                    self.stats["subsumed"] += 1
                    wiped.append(e)
                else:
                    while self._entries and (self._bytes + e.nbytes > self.max_bytes
                                             or len(self._entries) >= self.max_entries):
                        _, lru = self._entries.popitem(last=False)
                        self._bytes -= lru.nbytes
                        self.stats["evicted"] += 1
                        wiped.append(lru)
                    self._entries[f"e{next(self._ids)}"] = e
                    self._bytes += e.nbytes
                    self.stats["stored"] += 1
                    stored = True
        for w in wiped:
            w.wipe()
        return stored
