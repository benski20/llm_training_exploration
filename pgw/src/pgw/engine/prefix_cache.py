"""Exact KV-cache reuse across the turns of one conversation (``prefix_cache = "conversation"``; opt-in).

A chat client resends the whole conversation every turn. After the chat template, the new prompt usually
starts with the previous turn's prompt and the tokens the model generated for it. The gateway keeps that
turn's KV cache (customer-side secret state, plaintext, in gateway RAM only) under the conversation key, and
the next turn of the same key runs only the tokens after the longest common token prefix through the prompt
pass. The answer is bit-identical to recomputing the whole prompt because every K/V row and every logit row
depends only on its own token and the cache prefix it attends to (the native float path, integer linears
with per-row activation scales; :meth:`pgw.engine.Generator.prefix_reuse_supported` refuses any other path).

Security properties (SPEC 9.4b):

* RAM only, never spilled, never logged; bound to the conversation key (API key id, OpenAI ``user`` and an
  optional client conversation id), never shared across keys. One entry per key; using it removes it from
  the cache (no two requests share a cache), and a turn whose prompt does not start with any of the entry's
  tokens wipes it.
* A TTL and a memory budget with LRU eviction; every eviction, expiry, mismatch and failure zeroes the KV
  tensors and the token array before dropping them.
* An entry is stored only after a request finished cleanly (``stop`` / ``length``, strict-mode dummy passes
  done) and its session's deferred integrity checks were joined without a failure; :meth:`wipe_all` (any
  integrity failure or noise alarm of the model, reload, stop) discards everything and refuses entries of
  requests that were running at that moment (an epoch check).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from pgw.core.errors import ConfigError
from pgw.engine.kv_cache import KVCache

__all__ = ["ConversationPrefixCache", "PrefixHit", "PrefixSink"]

log = logging.getLogger(__name__)


def _common_prefix(a: np.ndarray, b: Sequence[int]) -> int:
    n = min(int(a.shape[0]), len(b))
    if n == 0:
        return 0
    bb = np.asarray(b[:n], dtype=np.int64)
    diff = np.flatnonzero(a[:n] != bb)
    return int(diff[0]) if diff.size else n


@dataclass
class PrefixHit:
    """A cached KV prefix handed to one request: ``cache`` holds exactly ``length`` committed positions, the
    KV rows of ``prompt[:length]`` (the request owns it now; the engine wipes it when it is done)."""

    cache: KVCache
    length: int
    #: Set by the engine when it takes the prefix (from then on the engine wipes or stores it).
    claimed: bool = False

    def wipe(self) -> None:
        self.cache.clear()


class _Entry:
    __slots__ = ("cache", "nbytes", "t_used", "tokens")

    def __init__(self, tokens: np.ndarray, cache: KVCache, now: float):
        self.tokens = tokens
        self.cache = cache
        self.nbytes = int(cache.nbytes() + tokens.nbytes)
        self.t_used = now

    def wipe(self) -> None:
        self.tokens.fill(0)
        self.cache.clear()


class PrefixSink:
    """Where one request's KV cache goes when the request ends (engine side of :class:`ConversationPrefixCache`).

    The engine calls :meth:`finishing` when the answer is complete (the next turn may then wait for the commit
    instead of missing it), then exactly one of :meth:`commit` (clean end, checks joined) or :meth:`discard`.
    Every method is idempotent after the first settle; a cache passed to a settled sink is wiped.
    """

    def __init__(self, owner: ConversationPrefixCache, key: str, epoch: int):
        self._owner = owner
        self.key = key
        self.epoch = epoch
        self._settled = False
        self._marked = False
        #: Set by the engine when it takes the sink (from then on the engine settles it).
        self.claimed = False

    @property
    def settled(self) -> bool:
        return self._settled

    def finishing(self) -> None:
        if not self._settled and not self._marked:
            self._marked = True
            self._owner._mark(self.key, self)

    def commit(self, tokens: Sequence[int], cache: KVCache) -> bool:
        """Store ``cache`` (whose committed positions are the KV rows of ``tokens``); False = wiped instead."""
        if self._settled:
            cache.clear()
            return False
        self._settled = True
        try:
            return self._owner._commit(self, tokens, cache)
        except BaseException:
            cache.clear()
            raise
        finally:
            self._owner._unmark(self.key, self)

    def discard(self, cache: KVCache | None = None) -> None:
        if cache is not None:
            cache.clear()
        if self._settled:
            return
        self._settled = True
        self._owner._unmark(self.key, self)


class ConversationPrefixCache:
    """Per-conversation KV caches of finished turns (gateway RAM; see the module docstring).

    Args:
        max_bytes: Memory budget for stored entries (KV tensors plus tokens); least recently used entries
            are wiped first. An entry larger than the budget is never stored.
        ttl_s: An entry unused for this long is wiped (swept on every call and by a background thread).
        max_entries: At most this many conversations are kept.
        clock: Monotonic clock (tests).
    """

    def __init__(self, *, max_bytes: int, ttl_s: float, max_entries: int = 1024,
                 clock: Callable[[], float] = time.monotonic, sweep_thread: bool = True):
        if max_bytes <= 0 or ttl_s <= 0 or max_entries <= 0:
            raise ConfigError("prefix cache: max_bytes, ttl_s and max_entries must be positive")
        self.max_bytes = int(max_bytes)
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._marks: dict[str, PrefixSink] = {}
        self._bytes = 0
        self._epoch = 0
        self.stats: dict[str, int] = {
            "hits": 0, "misses": 0, "mismatches": 0, "reused_tokens": 0, "stored": 0, "refused": 0,
            "evicted": 0, "expired": 0, "wiped_all": 0, "waits": 0,
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if sweep_thread:
            self._thread = threading.Thread(target=self._sweeper, name="pgw-prefix-cache-ttl", daemon=True)
            self._thread.start()

    # ---- request side ----------------------------------------------------------------------------------
    def sink(self, key: str) -> PrefixSink:
        with self._lock:
            return PrefixSink(self, key, self._epoch)

    def take(self, key: str, prompt: Sequence[int]) -> PrefixHit | None:
        """Remove ``key``'s entry and hand out its KV cache for ``prompt`` when they share a token prefix of at
        least one token (at most ``len(prompt) - 1``: the prompt pass needs one new row for the logits). A
        prompt that shares nothing wipes the entry (the conversation went elsewhere)."""
        with self._lock:
            self._sweep_locked()
            e = self._entries.pop(key, None)
            if e is None:
                self.stats["misses"] += 1
                return None
            self._bytes -= e.nbytes
            m = min(_common_prefix(e.tokens, prompt), len(prompt) - 1, e.cache.length)
            if m <= 0:
                self.stats["mismatches"] += 1
                e.wipe()
                return None
            self.stats["hits"] += 1
            self.stats["reused_tokens"] += m
        e.tokens.fill(0)
        e.cache.truncate(m)
        return PrefixHit(e.cache, m)

    def settling(self, key: str) -> bool:
        """A finished turn of ``key`` is on its way into the cache (its commit has not happened yet)."""
        with self._lock:
            return key in self._marks

    async def wait_settled(self, key: str, timeout_s: float) -> None:
        """Wait (at most ``timeout_s``) until no finished turn of ``key`` is still being committed."""
        if timeout_s <= 0 or not self.settling(key):
            return
        self.stats["waits"] += 1
        deadline = time.monotonic() + timeout_s
        delay = 0.001
        while self.settling(key) and time.monotonic() < deadline:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.05)

    # ---- maintenance -------------------------------------------------------------------------------------
    def wipe_all(self, reason: str = "") -> int:
        """Wipe every entry and refuse commits of requests that started before (integrity failure, alarm,
        reload, stop). Returns how many entries were wiped."""
        with self._lock:
            self._epoch += 1
            entries = list(self._entries.values())
            self._entries.clear()
            self._bytes = 0
            self.stats["wiped_all"] += 1
        for e in entries:
            e.wipe()
        if entries:
            log.info("prefix cache wiped (%s): %d entries", reason or "request", len(entries))
        return len(entries)

    def sweep(self) -> int:
        with self._lock:
            return self._sweep_locked()

    def close(self) -> None:
        self._stop.set()
        self.wipe_all("close")
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)

    def status(self) -> dict[str, int]:
        """Counters and sizes (no keys, no tokens)."""
        with self._lock:
            return {**self.stats, "entries": len(self._entries), "bytes": self._bytes, "max_bytes": self.max_bytes,
                    "ttl_s": int(self.ttl_s), "settling": len(self._marks)}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ---- internals ----------------------------------------------------------------------------------------
    def _sweeper(self) -> None:
        period = max(0.5, min(30.0, self.ttl_s / 4))
        while not self._stop.wait(period):
            try:
                self.sweep()
            except Exception:  # pragma: no cover - defensive
                log.exception("prefix cache sweep failed")

    def _sweep_locked(self) -> int:
        now = self._clock()
        dead = [k for k, e in self._entries.items() if now - e.t_used >= self.ttl_s]
        for k in dead:
            e = self._entries.pop(k)
            self._bytes -= e.nbytes
            e.wipe()
        self.stats["expired"] += len(dead)
        return len(dead)

    def _mark(self, key: str, sink: PrefixSink) -> None:
        with self._lock:
            self._marks[key] = sink

    def _unmark(self, key: str, sink: PrefixSink) -> None:
        with self._lock:
            if self._marks.get(key) is sink:
                del self._marks[key]

    def _commit(self, sink: PrefixSink, tokens: Sequence[int], cache: KVCache) -> bool:
        n = cache.length
        if n <= 0 or len(tokens) < n:
            cache.clear()
            return False
        cache.compact()
        arr = np.asarray(tokens[:n], dtype=np.int64).copy()
        e = _Entry(arr, cache, self._clock())
        wiped: list[_Entry] = []
        with self._lock:
            if sink.epoch != self._epoch or e.nbytes > self.max_bytes:
                self.stats["refused"] += 1
                wiped.append(e)
                stored = False
            else:
                old = self._entries.pop(sink.key, None)
                if old is not None:  # a concurrent turn of the same key stored first: the newer one wins
                    self._bytes -= old.nbytes
                    wiped.append(old)
                self._sweep_locked()
                while self._entries and (self._bytes + e.nbytes > self.max_bytes
                                         or len(self._entries) >= self.max_entries):
                    _, lru = self._entries.popitem(last=False)
                    self._bytes -= lru.nbytes
                    self.stats["evicted"] += 1
                    wiped.append(lru)
                self._entries[sink.key] = e
                self._bytes += e.nbytes
                self.stats["stored"] += 1
                stored = True
        for w in wiped:
            w.wipe()
        return stored
