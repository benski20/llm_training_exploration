"""Generation loop: prefill, decode, stop handling, streaming and cross-session batching."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from pgw.core.errors import ConfigError, MaskExhaustedError, PGWError
from pgw.engine.kv_cache import ContextOverflowError, KVCache
from pgw.engine.model import SequenceInput, TransformerModel
from pgw.engine.sampling import Sampler, SamplingParams
from pgw.engine.tokenizer import ChatTokenizer, IncrementalDecoder

__all__ = [
    "BuiltinDrafter",
    "DraftJob",
    "Drafter",
    "FinishReason",
    "GenerationResult",
    "Generator",
    "SessionFactory",
    "TokenEvent",
]

log = logging.getLogger(__name__)

#: Any linear id works for asking a mask reservation how many rows are left (all advance together).
_FIRST_LINEAR = "0.qkv"

#: ``fn(slots) -> session`` creating a mask reservation of ``slots`` positions (SPEC 9.4).
SessionFactory = Callable[[int], object]


class FinishReason(StrEnum):
    """OpenAI-compatible finish reasons."""

    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class TokenEvent:
    """One streamed step.

    ``text`` is the newly *final* text (stop-string look-ahead may delay text by a few tokens, and
    a stop token produces an event with empty text). ``finish_reason`` is set on the last event.
    """

    token_id: int
    text: str
    logprob: float | None = None
    finish_reason: FinishReason | None = None
    top_logprobs: list[tuple[int, float]] | None = None
    #: Mask rows consumed by this sequence so far (includes rejected speculative rows).
    masks: int = 0


@dataclass
class GenerationResult:
    """Completed generation."""

    token_ids: list[int]
    text: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    masks_consumed: int = 0
    logprobs: list[float] | None = None


@dataclass
class _SeqState:
    prompt: list[int]
    params: SamplingParams
    session: object | None
    own_session: bool
    cache: KVCache
    sampler: Sampler
    decoder: IncrementalDecoder | None
    stop_ids: frozenset[int]
    pending: list[int]
    generated: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    held: str = ""
    finish: FinishReason | None = None
    masks: int = 0
    t_start: float = field(default_factory=time.perf_counter)
    t_first: float | None = None
    t_end: float | None = None
    queue: asyncio.Queue[TokenEvent | BaseException] | None = None
    cancelled: bool = False
    # speculative decoding
    ctx: list[int] = field(default_factory=list)  # tokens held in the target KV cache
    draft_cache: KVCache | None = None
    drafts: list[int] = field(default_factory=list)  # proposals for the current step
    spec_proposed: int = 0  # draft rows sent for verification (accepted or not)
    spec_accepted: int = 0
    spec_passes: int = 0
    draft_ms: float = 0.0  # wall time of the draft-model passes this sequence took part in
    # fixed-row speculation (``spec_rows``) and strict-pass mode
    limit: int = 0  # LENGTH limit (positions); in fixed mode the KV cache has spare room for the fixed rows
    dstate: Any = None  # the drafter's per-sequence state
    fill: list[int] = field(default_factory=list)  # filler rows after the drafts (padding or masked fillers)
    pad: int = 0  # trailing rows of this pass sent as padding (random payload, no masks; pad-capable models)
    rows: int = 0  # rows of this pass (fixed mode, after the prefill pass)
    passes: int = 0  # target passes (provider-visible) this sequence took part in, prefill and dummies included
    dummy: bool = False  # this pass is a strict-mode dummy pass (no output)
    conv: str | None = None  # conversation key (strict mode queues a follow-up turn behind the dummy passes)
    conv_held: bool = False
    t_lane: float = 0.0  # when the sequence joined the prefill lane queue

    @property
    def done(self) -> bool:
        return self.finish is not None


#: Prompt lookup searches only the most recent tokens (O(window) per step).
_LOOKUP_WINDOW = 4096


def _lookup_drafts(seq: Sequence[int], k: int, max_n: int, min_n: int = 3) -> list[int]:
    """Prompt-lookup proposal: the ``k`` tokens that followed the latest earlier occurrence of the last
    ``n`` tokens of ``seq`` (``n`` from ``max_n`` down to ``min_n``).

    A match followed by a full ``k`` tokens is preferred over a later one cut short by the end of the
    context; when there is none, the continuation of the latest (longest-``n``) match is returned, which may
    be shorter than ``k`` (the caller pads it with the draft model or drops it). ``[]`` if nothing matches.
    ``min_n`` keeps short (1- and 2-gram) coincidences from proposing tokens.
    """
    if len(seq) > _LOOKUP_WINDOW:  # bound the per-step cost on long contexts
        seq = seq[-_LOOKUP_WINDOW:]
    seq = list(seq)
    size = len(seq)
    short: list[int] = []
    for n in range(min(max_n, size - 1), max(1, min_n) - 1, -1):
        tail = seq[size - n :]
        first = tail[0]
        for i in range(size - n - 1, -1, -1):  # latest match first, never the tail itself
            if seq[i] == first and seq[i : i + n] == tail:
                out = seq[i + n : i + n + k]
                if len(out) == k:
                    return out
                if out and not short:
                    short = out
    return short


def _dcache(st: _SeqState) -> KVCache:
    assert st.draft_cache is not None
    return st.draft_cache


def _lookup_drafts_fixed(seq: Sequence[int], k: int, max_n: int, min_n: int = 3) -> list[int]:
    """:func:`_lookup_drafts` with a fixed amount of work (same result).

    The search runs over a fixed-size window (the last ``_LOOKUP_WINDOW`` tokens, left-padded with a
    sentinel) for every n-gram length from ``max_n`` down to ``min_n``, with whole-array numpy operations
    and no early exit, so its running time does not depend on whether, where or how often the context
    repeats itself (fixed-row speculation keeps the gateway's per-pass work independent of the text).
    """
    w = _LOOKUP_WINDOW
    lo = max(1, min_n)
    if k <= 0 or max_n < lo:
        return []
    tail_seq = list(seq[-w:])
    size = len(tail_seq)
    arr = np.full(w + k, -1, dtype=np.int64)  # real tokens at [w - size, w); sentinels never match one
    if size:
        arr[w - size : w] = np.asarray(tail_seq, dtype=np.int64)
    pos = np.arange(w, dtype=np.int64)
    best_full: tuple[int, int] | None = None  # (n, start) of the chosen full-length match
    best_short: tuple[int, int] | None = None
    for n in range(max_n, lo - 1, -1):  # every length, always (no early exit)
        m = w - n  # candidate starts in [0, w - n): never the tail itself
        tail = arr[w - n : w]
        match = np.ones(m, dtype=bool)
        for j in range(n):
            match &= arr[j : j + m] == tail[j]
        full = match & (pos[:m] + n + k <= w)
        last_full = int(np.max(np.where(full, pos[:m], -1)))
        last_any = int(np.max(np.where(match, pos[:m], -1)))
        if n > size - 1:  # the tail would include sentinels: not a candidate length
            continue
        if best_full is None and last_full >= 0:
            best_full = (n, last_full)
        if best_short is None and last_any >= 0:
            best_short = (n, last_any)
    pick = best_full if best_full is not None else best_short
    if pick is None:
        return []
    n, i = pick
    return [int(t) for t in arr[i + n : min(i + n + k, w)]]


@dataclass
class DraftJob:
    """One sequence's draft request in a fixed-row speculative pass (see :class:`Drafter`).

    Attributes:
        state: The drafter's per-sequence state (from :meth:`Drafter.open`).
        context: Every token the target has processed so far plus the pending token: the drafts continue
            this sequence. In a strict-mode dummy pass the last token is a random filler. Between two calls
            for the same ``state``, ``context[:-1]`` only grows (the processed tokens are append-only).
        k: Number of draft tokens wanted.
        sampler: For sampled requests a fork of the request's sampler (same RNG state, penalties and
            filters): drawing draft ``j`` (1-based) with the fork advanced ``j - 1`` times uses exactly the
            noise the target's sample at that position will use. ``None`` for greedy requests.
        generated: Tokens generated so far (for penalties).
        out: Filled by the drafter: up to ``k`` proposals (the generator completes fewer with filler rows).
    """

    state: Any
    context: list[int]
    k: int
    sampler: Sampler | None = None
    generated: list[int] = field(default_factory=list)
    out: list[int] = field(default_factory=list)


@runtime_checkable
class Drafter(Protocol):
    """Draft source of fixed-row speculation (``Generator(spec_rows=R)``).

    Drafts are proposals only: every emitted token is decided by the masked target, so a drafter can
    never change the output, only how many tokens a pass yields. A drafter runs on the gateway in
    plaintext. Privacy contract: :meth:`propose` does the same work for every job whatever the text
    (a fixed number of model steps per pass, no early exit when a proposal is found or ends), so the
    gap the provider sees between two passes does not depend on how predictable the text is.

    ``needs_prefill``: :meth:`prefill` does real work (it then runs in a worker thread alongside the
    target's prefill pass, and ping-pong is not used).
    """

    needs_prefill: bool

    def open(self, prompt: Sequence[int], max_len: int) -> Any:
        """Per-sequence state for a request whose drafts continue ``prompt`` (``max_len`` positions)."""
        ...

    def prefill(self, states: Sequence[Any], prompts: Sequence[Sequence[int]]) -> None:
        """Read the prompts (runs alongside the target's prefill pass)."""
        ...

    def propose(self, jobs: Sequence[DraftJob]) -> None:
        """Fill ``job.out`` (up to ``job.k`` tokens) for every job, with content-independent work."""
        ...

    def close(self, state: Any) -> None:
        """Release a sequence's state."""
        ...


@dataclass
class _BuiltinState:
    cache: KVCache | None
    fed: list[int] = field(default_factory=list)  # tokens held in ``cache``, in order
    #: Leading tokens of ``fed`` known to equal the context's processed prefix (it only grows), so the
    #: catch-up compares only the tokens after it: O(accepted drafts), not O(context).
    ok: int = 0


class BuiltinDrafter:
    """Prompt lookup and/or a small local draft model, with a fixed number of draft-model steps per pass.

    Every :meth:`propose` call runs prompt lookup over the whole fixed window (:func:`_lookup_drafts_fixed`)
    and exactly ``k`` draft-model forward steps per job, each followed by exactly one sampler draw (or
    argmax): a lookup hit, a proposal cut short by an id the target does not have, or a full lookup match
    still runs the remaining steps on a filler token (cache rolled back, result dropped). A lookup match
    shorter than ``k`` is completed by the draft model (or by the generator's filler rows without one).

    The first model step of a job also feeds the tokens the draft has not seen yet (1 or 2 after a pass,
    plus lookup tokens); its row count is not padded. A draft model is memory-bound at these sizes (its
    weights are read once per step), so the step time hardly depends on it; the residual is listed in
    the timing audit.

    Args:
        model: Draft model (same tokenizer as the target) or ``None`` (prompt lookup only).
        lookup: Longest n-gram prompt lookup tries (0: no lookup).
        lookup_min_n: Shortest n-gram accepted.
        target_vocab: The target's vocabulary size (a proposal ends at the first id at or above it).
    """

    def __init__(self, model: TransformerModel | None, *, lookup: int = 0, lookup_min_n: int = 3,
                 target_vocab: int | None = None):
        if model is None and lookup <= 0:
            raise ConfigError("BuiltinDrafter needs a draft model or prompt lookup")
        self.model = model
        self.lookup = int(lookup)
        self.lookup_min_n = int(lookup_min_n)
        self.target_vocab = target_vocab
        dv = getattr(getattr(model, "arch", None), "vocab_size", None) if model is not None else None
        self.draft_vocab = int(dv) if dv is not None else None
        self.needs_prefill = model is not None
        #: ``calls`` propose calls that ran the model, ``steps`` draft forward steps, ``ms`` their wall time.
        self.stats: dict[str, float] = {"calls": 0.0, "steps": 0.0, "ms": 0.0}

    def _din(self, ids: Sequence[int]) -> list[int]:
        dv = self.draft_vocab
        return [int(t) for t in ids] if dv is None else [int(t) if t < dv else dv - 1 for t in ids]

    def _filler(self) -> int:
        return secrets.randbelow(self.draft_vocab or self.target_vocab or 1)

    def open(self, prompt: Sequence[int], max_len: int) -> _BuiltinState:
        return _BuiltinState(self.model.new_cache(max_len) if self.model is not None else None)

    def prefill(self, states: Sequence[Any], prompts: Sequence[Sequence[int]]) -> None:
        live = [(st, p) for st, p in zip(states, prompts, strict=True) if st.cache is not None]
        if not live or self.model is None:
            return
        try:
            for st, _ in live:  # a retried prefill pass starts over
                st.cache.truncate(0)
                st.fed = []
                st.ok = 0
            self.model.forward_batch([SequenceInput(self._din(p), st.cache) for st, p in live])
            for st, p in live:
                st.fed = [int(t) for t in p]
                st.ok = len(st.fed)
        except Exception:
            log.warning("draft prefill failed; model drafts disabled for %d sequence(s)", len(live), exc_info=True)
            for st, _ in live:
                self.close(st)

    def close(self, state: Any) -> None:
        if state.cache is not None:
            state.cache.clear()
        state.cache = None
        state.fed = []
        state.ok = 0

    def _lookup(self, context: list[int], k: int) -> list[int]:
        if not self.lookup:
            return []
        found = _lookup_drafts_fixed(context, k, max(self.lookup, self.lookup_min_n), self.lookup_min_n)
        tv = self.target_vocab
        if tv is not None:
            cut = next((j for j, t in enumerate(found) if t >= tv), len(found))
            found = found[:cut]
        return found

    def propose(self, jobs: Sequence[DraftJob]) -> None:
        seeds = [self._lookup(j.context, j.k) for j in jobs]
        for job, seed in zip(jobs, seeds, strict=True):
            job.out = list(seed[: job.k])
        mjobs = [(job, seed) for job, seed in zip(jobs, seeds, strict=True)
                 if job.k > 0 and isinstance(job.state, _BuiltinState) and job.state.cache is not None]
        if self.model is None or not mjobs:
            return
        t0 = time.perf_counter()
        try:
            self._model_steps(mjobs)
        except Exception:
            log.warning("draft model failed; model drafts disabled for %d sequence(s)", len(mjobs), exc_info=True)
            for job, seed in mjobs:
                self.close(job.state)
                job.out = list(seed[: job.k])
        finally:
            self.stats["calls"] += 1
            self.stats["ms"] += (time.perf_counter() - t0) * 1e3

    def _model_steps(self, mjobs: Sequence[tuple[DraftJob, list[int]]]) -> None:
        """Exactly ``job.k`` draft forward steps and ``job.k`` proposals (real or dropped) per job.

        Per job with a lookup seed of ``s`` tokens: ``d = min(s, k - 1)`` dummy steps first (their draws
        advance the coupled sampler past the seed positions), then the catch-up step (the context tokens
        the draft has not seen, plus the seed), then one-token steps; once the proposal is complete or has
        ended, the remaining steps are dummies on a filler token.
        """
        assert self.model is not None
        tv = self.target_vocab
        plan: list[dict[str, Any]] = []
        for mj, seed in mjobs:
            s = min(len(seed), mj.k)
            plan.append({"job": mj, "seed": seed[:s], "d": min(s, mj.k - 1), "stopped": s >= mj.k, "base": 0})
        steps = max(p["job"].k for p in plan)
        for t in range(steps):
            feeds: list[SequenceInput] = []
            kinds: list[tuple[dict[str, Any], str]] = []
            for p in plan:
                job: DraftJob = p["job"]
                st: _BuiltinState = job.state
                if t >= job.k:
                    continue
                cache = st.cache
                assert cache is not None
                if t < p["d"] or (t > p["d"] and p["stopped"]):  # dummy step on a filler token
                    p["base"] = cache.length
                    feeds.append(SequenceInput(self._din([self._filler()]), cache))
                    kinds.append((p, "dummy"))
                elif t == p["d"]:  # catch up with the context (dropping rejected drafts), then the seed
                    ctx = job.context
                    n = min(len(st.fed), len(ctx))
                    lcp = min(st.ok, n)  # fed[:ok] == ctx[:ok] (the processed prefix only grows)
                    while lcp < n and st.fed[lcp] == ctx[lcp]:
                        lcp += 1
                    lcp = min(lcp, len(ctx) - 1)  # always feed at least the last context token
                    cache.truncate(lcp)
                    del st.fed[lcp:]
                    new = [int(x) for x in ctx[lcp:]] + [int(x) for x in p["seed"]]
                    st.fed.extend(new)
                    st.ok = len(ctx) - 1  # the last context token may be a dummy pass's filler
                    feeds.append(SequenceInput(self._din(new), cache))
                    kinds.append((p, "full" if p["stopped"] else "real"))
                else:  # continue the proposal
                    last = int(job.out[-1])
                    st.fed.append(last)
                    feeds.append(SequenceInput(self._din([last]), cache))
                    kinds.append((p, "real"))
            if not feeds:
                break
            outs = self.model.forward_batch(feeds)
            self.stats["steps"] += 1
            for (p, kind), lg in zip(kinds, outs, strict=True):
                job = p["job"]
                tok = self._pick(job, lg)  # one draw (or argmax) per step and job, used or not
                if kind == "dummy":
                    job.state.cache.truncate(p["base"])
                    continue
                if kind == "full":  # the seed already completes the proposal; this step only catches up
                    continue
                if tok < 0 or (tv is not None and tok >= tv):  # not a target token: the proposal ends
                    p["stopped"] = True
                    continue
                job.out.append(tok)
                if len(job.out) >= job.k:
                    p["stopped"] = True

    def _pick(self, job: DraftJob, lg: torch.Tensor) -> int:
        fk = job.sampler
        if fk is None:
            return int(torch.argmax(lg))
        tv = self.target_vocab
        if tv is not None and lg.numel() != tv:  # align with the target's vocabulary (RNG draws)
            lg = lg[:tv] if lg.numel() > tv else torch.cat([lg, lg.new_full((tv - lg.numel(),), float("-inf"))])
        return fk.propose(lg, job.generated + job.out)


class _ConvGate:
    """Per-conversation FIFO: a follow-up turn starts only after the previous turn has left the engine
    (strict mode: after its dummy passes), so the provider never sees a new turn overlap them."""

    def __init__(self) -> None:
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    async def acquire(self, key: str) -> None:
        lock, users = self._locks.get(key, (asyncio.Lock(), 0))
        self._locks[key] = (lock, users + 1)
        try:
            await lock.acquire()
        except BaseException:
            self._drop(key)
            raise

    def release(self, key: str) -> None:
        if key not in self._locks:  # pragma: no cover - defensive
            return
        self._locks[key][0].release()
        self._drop(key)

    def _drop(self, key: str) -> None:
        lock, users = self._locks[key]
        if users <= 1:
            del self._locks[key]
        else:
            self._locks[key] = (lock, users - 1)


class Generator:
    """Drives :class:`TransformerModel` for complete requests.

    One mask slot is consumed per processed token position (prompt + generated). Callers get a
    mask reservation sized ``len(prompt) + max_tokens`` before starting (SPEC section 9.4) and pass
    it as ``session``; alternatively give ``session_factory`` and the generator reserves (and burns
    unused slots by closing) per request. Unused slots of a reservation are never returned.

    Concurrent requests are batched per decode step: :meth:`astream` registers each request with a
    background step loop that groups all active sessions at the same linear into one backend call
    (``forward_many``) to amortize round trips and server GEMMs. :meth:`generate_many` does the
    same synchronously for a fixed set of prompts.

    Args:
        model: Transformer runtime.
        tokenizer: Tokenizer (needed for text output and stop strings; ``None`` gives ids only).
        max_batch: Maximum sequences per batched step.
        session_factory: Optional ``fn(slots) -> session`` used when a call passes no session.
        batching: Use the shared continuous-batching loop in :meth:`astream`.
        draft: Optional small model (same tokenizer) run locally in plaintext for speculative
            decoding. Each step it proposes up to ``spec_tokens`` tokens; the target model checks
            them all in one pass (one masked round trip per linear for all of them). Tokens are
            accepted by sampling each position with the request's own sampler and keeping the draft
            while it agrees, so every emitted token is drawn exactly as without speculation (same
            greedy choice, same seeded sample). Rejected positions cost mask rows, never tokens.
        spec_tokens: Draft tokens proposed per step (0 disables speculation).
        lookup: Prompt lookup: longest n-gram tried (0 disables).
        spec_min_tokens: Adaptive draft length (draft model only): always ``spec_min_tokens`` drafts, the full
            ``spec_tokens`` only while the draft is confident (``draft_confidence``). ``None``: one fixed size.
        draft_confidence: Draft top probability needed for the long size (0: always the long size).
        lookup_min_n: Shortest n-gram prompt lookup accepts as a match (no 1-gram coincidences).
        spec_max_batch: Speculate only in steps with at most this many sequences. A verify pass adds
            ``k`` rows per sequence to every linear of a step that the other sequences must wait for; with
            several sequences sharing a step the extra rows cost more than the passes they save.
        pingpong: Two-cohort ping-pong for the async batching loop (off by default). A step with at least
            ``pingpong_min_sessions`` sequences is split into two cohorts (the first and second half of the
            step's sequences, in round-robin order) whose forward passes run concurrently: while one cohort's
            round trip is with the compute service, the gateway verifies, unmasks, runs the plaintext step
            and masks the next calls of the other. Both cohorts finish before the next step is picked, so the
            round-robin order, ``max_batch`` and per-sequence results are unchanged (each sequence's logits do
            not depend on the sequences it shares a pass with). Each cohort keeps the MaskExhaustedError
            isolation of a plain step. Used only when every sequence of the step has its own mask reservation
            (rows of one reservation are never taken by two passes at once) and there is no draft model.
        pingpong_min_sessions: Smallest step (sequences) that is split (at least 2; default 5: more than 4).
        spec_rows: Fixed-row speculation (``None``: off, the variable-row mode described below). Every pass
            after the prefill pass carries exactly ``spec_rows`` rows per sequence (2..8) in steps of at most
            ``spec_max_batch`` sequences, and 1 row in larger steps (a function of the public session count):
            the pending token, the drafts, then filler rows. It never falls back to 1 row because of the
            text, the drafts, the token budget or the mask reservation. On a model that supports padding
            rows (``supports_pad_rows``; the backend sends uniform random payloads that take no masks) the
            filler rows are padding and drafts use only spare mask rows; otherwise filler rows are ordinary
            masked rows (random tokens) and the request needs a reservation of
            ``prompt + (max_tokens - 1) * spec_rows`` rows, checked before any pass. Per-pass gateway work
            is fixed: the drafter runs a fixed number of steps (:class:`Drafter`), the LM head covers every
            row, and rows after the accepted ones are sampled with a throw-away sampler. Needs a draft
            source (``draft``, ``lookup`` or ``drafter``); the adaptive draft length is not used.
        spec_strict: Strict-pass mode (needs ``spec_rows``): after a sequence ends (or is cancelled), dummy
            passes on random data run through the same code path (drafter, every linear, LM head, sampling;
            mask-free on pad-capable models) until the provider has seen exactly as many passes as tokens
            were generated, which it already sees without speculation. The answer streams first; a
            follow-up turn of the same ``conversation`` waits until the dummy passes are done. ``spec_rows``
            is refused without ``spec_strict`` (non-strict fixed rows reveal the content-dependent pass count
            P instead of the generation length L; SPEC 3.4) unless ``spec_nonstrict_for_tests`` is set.
            Strict mode keeps a follow-up turn from overlapping the dummy passes only when the caller
            passes ``conversation=`` (the gateway must, before strict mode is enabled there).
        spec_nonstrict_for_tests: Allow ``spec_rows`` without ``spec_strict`` (tests and benchmarks only;
            never for served traffic).
        drafter: Custom :class:`Drafter` for fixed-row speculation (default: :class:`BuiltinDrafter` over
            ``draft`` and ``lookup``).
        prefill_lane: Run prompt prefills in a lane of their own, concurrently with the decode steps of
            running streams (off by default), so a long prompt does not freeze them for its whole prefill.
            Waiting prompts are taken shortest first with aging (a prompt's length minus
            ``prefill_aging`` tokens per second waited), up to ``prefill_lane_rows`` prompt rows per lane
            pass. A prompt uses the lane only with at least ``prefill_lane_min_tokens`` tokens and its own
            mask reservation (no running sequence shares it); others prefill in the decode step as before,
            and a sequence that shares the reservation of a prompt in a lane pass waits for that pass. Without
            a reservation (``session=None``: plain/dev mode) the lane is never used.
            Each sequence's tokens are unchanged (its logits do not depend on what shares a pass); the
            provider sees prefill items in passes of their own, whose sizes are the public prompt lengths.

    Privacy of speculation: the compute service sees how many rows each pass carries. Every pass of a
    sequence carries 1 or ``1 + k`` rows, ``k`` one fixed size (or, with the adaptive draft length, one of
    two fixed sizes): a prompt-lookup proposal is exactly ``k`` tokens (a short match is completed by the
    draft model or dropped), and near the end of the token budget, the context or the mask reservation the
    draft length drops to 0 rather than to a size that depends on how many drafts were accepted. So a pass
    tells the service at most about one bit about how predictable the text is, and the number of passes
    tells it how many tokens the drafts saved. ``spec_rows`` with ``spec_strict`` removes both: fixed rows
    and exactly as many passes as generated tokens (SPEC 3.4). The variable-row mode is kept for
    compatibility only (follow-up: deprecate it and steer the gateway to strict mode).
    """

    def __init__(
        self,
        model: TransformerModel,
        tokenizer: ChatTokenizer | None,
        *,
        max_batch: int = 32,
        session_factory: SessionFactory | None = None,
        batching: bool = True,
        draft: TransformerModel | None = None,
        spec_tokens: int = 2,
        lookup: int = 0,
        spec_min_tokens: int | None = None,
        draft_confidence: float = 0.0,
        lookup_min_n: int = 3,
        spec_max_batch: int = 1,
        pingpong: bool = False,
        pingpong_min_sessions: int = 5,
        spec_rows: int | None = None,
        spec_strict: bool = False,
        spec_nonstrict_for_tests: bool = False,
        drafter: Drafter | None = None,
        prefill_lane: bool = False,
        prefill_lane_min_tokens: int = 16,
        prefill_lane_rows: int = 2048,
        prefill_aging: float = 256.0,
    ):
        if max_batch < 1:
            raise ConfigError("max_batch must be >= 1")
        if pingpong_min_sessions < 2:
            raise ConfigError("pingpong_min_sessions must be >= 2")
        if spec_tokens < 0 or spec_tokens > 16:
            raise ConfigError("spec_tokens must be in [0, 16]")
        # The draft must share the target's tokenizer; its embedding table may be padded to another
        # size (Qwen2.5-0.5B: 151936 rows, 7B: 152064). Drafts are proposals only, so ids outside
        # either table are never fed to the target (a proposal there ends the draft) and are
        # clamped when fed to the draft (a worse proposal, never a different output).
        tv = getattr(getattr(model, "arch", None), "vocab_size", None)
        dv = getattr(getattr(draft, "arch", None), "vocab_size", None) if draft is not None else None
        self._target_vocab = int(tv) if tv is not None else None
        self._draft_vocab = int(dv) if dv is not None else None
        if lookup < 0 or lookup > 8:
            raise ConfigError("lookup must be in [0, 8]")
        if lookup_min_n < 2 or lookup_min_n > 8:
            raise ConfigError("lookup_min_n must be in [2, 8]")
        if spec_max_batch < 1:
            raise ConfigError("spec_max_batch must be >= 1")
        self.draft = draft if spec_tokens > 0 else None
        #: Prompt-lookup drafting: propose what followed the latest earlier match of the last
        #: ``lookup`` (down to ``lookup_min_n``) tokens of the context. Costs no model compute.
        self.lookup = lookup if spec_tokens > 0 else 0
        self.lookup_min_n = lookup_min_n
        self.spec_max_batch = spec_max_batch
        self.spec_tokens = spec_tokens if (draft is not None or self.lookup) else 0
        # Adaptive draft length: always draft spec_min_tokens; continue to spec_tokens only when each of the
        # first spec_min_tokens (at least one) draft tokens had top probability >= draft_confidence. Two fixed
        # sizes, so the row count of a pass tells the compute service little about how predictable the text is.
        lo = self.spec_tokens if spec_min_tokens is None else max(0, min(spec_min_tokens, self.spec_tokens))
        if not 0.0 <= draft_confidence < 1.0:
            raise ConfigError("draft_confidence must be in [0, 1)")
        self.spec_min_tokens = lo
        self.draft_confidence = draft_confidence
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.session_factory = session_factory
        self.batching = batching
        self._eos: tuple[int, ...] = tuple(tokenizer.eos_token_ids) if tokenizer is not None else ()
        arch_eos = tuple(getattr(model.arch, "eos_token_ids", ()) or ())
        self._eos = tuple(dict.fromkeys(self._eos + arch_eos))
        self._active: list[_SeqState] = []
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._rr = 0
        #: Reservation closes running in worker threads (never on the event loop; see _arelease).
        self._closing: set[asyncio.Future[None]] = set()
        #: Speculation counters over the generator's lifetime (read by metrics and benchmarks):
        #: ``spec_passes`` verify passes, ``spec_rows`` draft rows sent, ``spec_accepted`` drafts accepted,
        #: ``draft_calls`` / ``draft_ms`` draft-model planning steps and their wall time.
        self.spec_stats: dict[str, float] = {
            "spec_passes": 0.0, "spec_rows": 0.0, "spec_accepted": 0.0, "draft_calls": 0.0, "draft_ms": 0.0,
        }
        self.pingpong = bool(pingpong)
        self.pingpong_min_sessions = int(pingpong_min_sessions)
        #: Batching-loop counters (read by the gateway status and benchmarks): ``steps`` engine steps,
        #: ``pingpong_steps`` of them split in two cohorts, ``step_sequences`` sequences over all steps.
        self.sched_stats: dict[str, float] = {"steps": 0.0, "pingpong_steps": 0.0, "step_sequences": 0.0,
                                              "lane_steps": 0.0, "lane_sequences": 0.0}
        # ---- fixed-row speculation and strict-pass mode -----------------------------------------------
        if spec_strict and spec_rows is None:
            raise ConfigError("spec_strict needs spec_rows (a fixed row count per pass)")
        if drafter is not None and spec_rows is None:
            raise ConfigError("a drafter is used by fixed-row speculation only (set spec_rows)")
        self.spec_rows: int | None = None
        self.spec_strict = False
        self._drafter: Drafter | None = None
        #: Filler rows are padding (random payload, no masks): the model passes ``pad_rows`` to its backend.
        self._pad_ok = False
        if spec_rows is not None:
            if not 2 <= int(spec_rows) <= 8:
                raise ConfigError("spec_rows must be in [2, 8]")
            if drafter is None:
                if draft is None and lookup <= 0:
                    raise ConfigError("spec_rows needs a draft source (draft, lookup or drafter)")
                drafter = BuiltinDrafter(draft, lookup=lookup, lookup_min_n=lookup_min_n,
                                         target_vocab=self._target_vocab)
            if not spec_strict and not spec_nonstrict_for_tests:
                raise ConfigError("spec_rows needs spec_strict: non-strict fixed rows reveal the pass count "
                                  "(spec_nonstrict_for_tests=True allows it in tests only)")
            self.spec_rows = int(spec_rows)
            self.spec_strict = bool(spec_strict)
            self._drafter = drafter
            # the variable-row machinery stays inert in fixed mode
            self.draft = None
            self.lookup = 0
            self.spec_tokens = self.spec_rows - 1
            self.spec_min_tokens = self.spec_tokens
            self.draft_confidence = 0.0
            self._pad_ok = bool(getattr(model, "supports_pad_rows", False)) and "pad_rows" in getattr(
                SequenceInput, "__dataclass_fields__", {})
            self.spec_stats.update({"dummy_passes": 0.0, "fill_rows": 0.0, "pad_rows": 0.0})
        tv_fill = self._target_vocab if self._target_vocab is not None else getattr(model, "vocab_size", None)
        self._fill_vocab = int(tv_fill) if tv_fill is not None else 1
        #: Serializes the draft model between the decode loop and draft prefills in worker threads (the
        #: prefill lane runs them concurrently); a decode step that finds it busy plans no model drafts.
        self._draft_lock = threading.Lock()
        self._conv_gate = _ConvGate()
        #: Strict-mode drains of non-batching streams (dummy passes after the answer).
        self._draining: set[asyncio.Task[None]] = set()
        # ---- prefill lane --------------------------------------------------------------------------------
        if prefill_lane_min_tokens < 1:
            raise ConfigError("prefill_lane_min_tokens must be >= 1")
        if prefill_lane_rows < 1:
            raise ConfigError("prefill_lane_rows must be >= 1")
        if prefill_aging < 0:
            raise ConfigError("prefill_aging must be >= 0")
        self.prefill_lane = bool(prefill_lane)
        self.prefill_lane_min_tokens = int(prefill_lane_min_tokens)
        self.prefill_lane_rows = int(prefill_lane_rows)
        self.prefill_aging = float(prefill_aging)
        self._lane_wait: list[_SeqState] = []
        self._lane_busy: list[_SeqState] = []
        self._lane_task: asyncio.Task[None] | None = None

    # ---- state management ---------------------------------------------------------------------
    def _new_state(
        self,
        prompt_ids: Sequence[int],
        params: SamplingParams,
        session: object | None,
        own_session: bool = False,
    ) -> _SeqState:
        """Validate a request and build its state.

        Args:
            own_session: The generator takes ownership of ``session`` and closes it (burning unused
                mask slots) when the request ends. Sessions made by ``session_factory`` are always owned.
                On a validation error an owned session is closed before the error propagates.
        """
        own = bool(own_session and session is not None)
        try:
            prompt = [int(t) for t in prompt_ids]
            if not prompt:
                raise ConfigError("prompt must contain at least one token")
            if params.stop and self.tokenizer is None:
                raise ConfigError("stop strings need a tokenizer")
            ctx = self.model.arch.max_position_embeddings
            if len(prompt) >= ctx:
                raise ContextOverflowError(
                    f"prompt of {len(prompt)} tokens leaves no room in the {ctx}-token context",
                    details={"prompt_tokens": len(prompt), "max_len": ctx},
                )
            fixed = self.spec_rows
            if fixed is None:
                cap = min(len(prompt) + params.max_tokens + self.spec_tokens, ctx)
                alloc = cap
            else:  # LENGTH at the plain limit; the cache has room for a full pass of fixed rows beyond it
                cap = min(len(prompt) + params.max_tokens, ctx)
                alloc = cap + fixed
            need = self.mask_rows_needed(len(prompt), params.max_tokens)
            if session is None and self.session_factory is not None:
                session = self.session_factory(need)
                own = session is not None
            if fixed is not None and not self._pad_ok and session is not None:
                self._check_fixed_reservation(session, need)
            stop_ids = set(params.stop_token_ids)
            if not params.ignore_eos:
                stop_ids.update(self._eos)
            st = _SeqState(
                prompt=prompt,
                params=params,
                session=session,
                own_session=own,
                cache=self.model.new_cache(alloc),
                draft_cache=self.draft.new_cache(cap) if self.draft is not None else None,
                sampler=Sampler(params, prompt),
                decoder=self.tokenizer.incremental_decoder() if self.tokenizer is not None else None,
                stop_ids=frozenset(stop_ids),
                pending=list(prompt),
                limit=cap,
            )
            if self._drafter is not None and fixed is not None:
                st.dstate = self._drafter.open(prompt, cap + 2 * fixed)
            return st
        except BaseException:
            if own and session is not None:
                self._close_session(session)
            raise

    def mask_rows_needed(self, prompt_tokens: int, max_tokens: int) -> int:
        """Mask rows a request's reservation must hold: ``prompt + max_tokens`` (as without speculation:
        speculation only uses spare rows), or, with ``spec_rows`` on a model without padding rows (filler
        rows are masked rows), ``prompt + max_tokens + (max_tokens - 1) * (spec_rows - 1)``: every pass
        after the prefill then takes ``spec_rows`` rows (strict-mode dummy passes too)."""
        base = int(prompt_tokens) + int(max_tokens)
        if self.spec_rows is None or self._pad_ok:
            return base
        ctx = int(self.model.arch.max_position_embeddings)
        passes = max(0, min(int(max_tokens), ctx - int(prompt_tokens)) - 1)  # decode (and dummy) passes
        return base + passes * (self.spec_rows - 1)

    @staticmethod
    def _check_fixed_reservation(session: object, need: int) -> None:
        """Fixed rows with masked filler rows: refuse up front (before the provider sees anything) a
        reservation that could run out mid-request, rather than ever sending a pass of fewer rows."""
        remaining = getattr(session, "remaining", None)
        if not callable(remaining):
            return
        try:
            rows = int(remaining(_FIRST_LINEAR))
        except Exception:
            return
        if rows < need:
            raise MaskExhaustedError(
                "mask reservation too small for fixed-row speculation",
                details={"rows": rows, "needed": need},
            )

    def _filler(self) -> int:
        """A random token id (filler rows and dummy passes; masked or padding, never emitted)."""
        return secrets.randbelow(self._fill_vocab)

    @staticmethod
    def _close_session(session: object) -> None:
        close = getattr(session, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # pragma: no cover - defensive
                log.exception("closing session reservation failed")

    def _release_local(self, st: _SeqState) -> None:
        """Free a sequence's caches, drafter state and conversation slot (no reservation close)."""
        st.cache.clear()
        if st.draft_cache is not None:
            st.draft_cache.clear()
        if st.dstate is not None and self._drafter is not None:
            dstate, st.dstate = st.dstate, None
            try:
                self._drafter.close(dstate)
            except Exception:  # pragma: no cover - defensive
                log.exception("closing drafter state failed")
        if st.conv_held and st.conv is not None:
            st.conv_held = False
            self._conv_gate.release(st.conv)

    def _release(self, st: _SeqState) -> None:
        """Synchronous release (blocking API only: closes an owned session on this thread)."""
        if st.t_end is None:
            st.t_end = time.perf_counter()
        if st.own_session and st.session is not None:
            st.own_session = False
            Generator._close_session(st.session)
        self._release_local(st)

    def _arelease(self, st: _SeqState) -> None:
        """Release from async code without blocking the event loop.

        Closing a mask reservation takes the store's inter-process lock, commits a durable SQLite
        transaction and may erase a fully consumed segment file; that must never run on the event
        loop (it would stall every stream and the health probes). The close is handed to the default
        executor; :meth:`aclose` (and :meth:`wait_closed`) wait for outstanding closes.
        """
        if st.t_end is None:
            st.t_end = time.perf_counter()
        session = st.session if st.own_session else None
        st.own_session = False
        self._release_local(st)
        if session is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - not inside a loop: plain synchronous close
            self._close_session(session)
            return
        fut = loop.run_in_executor(None, self._close_session, session)
        self._closing.add(fut)
        fut.add_done_callback(self._closing.discard)

    async def wait_closed(self) -> None:
        """Wait until every reservation close handed to a worker thread has finished."""
        while True:
            # Prune finished closes here rather than relying on their done-callbacks having run:
            # re-gathering an already finished future in a loop can spin without ever yielding to them.
            pending = {f for f in self._closing if not f.done()}
            self._closing.intersection_update(pending)
            if not pending:
                return
            await asyncio.wait(pending)

    # ---- text / stop handling ---------------------------------------------------------------
    def _scan_stop(self, st: _SeqState, final: bool) -> tuple[str, bool]:
        """Split ``st.held`` into emittable text; returns ``(text, stop_string_hit)``."""
        buf = st.held
        stops = st.params.stop
        if stops:
            hit_at, hit = -1, ""
            for s in stops:
                i = buf.find(s)
                if i >= 0 and (hit_at < 0 or i < hit_at):
                    hit_at, hit = i, s
            if hit_at >= 0:
                out = buf[: hit_at + (len(hit) if st.params.include_stop_str_in_output else 0)]
                st.held = ""
                return out, True
            if not final:
                hold = 0
                for s in stops:
                    for k in range(min(len(s) - 1, len(buf)), 0, -1):
                        if buf.endswith(s[:k]):
                            hold = max(hold, k)
                            break
                out = buf[: len(buf) - hold]
                st.held = buf[len(buf) - hold :]
                return out, False
        st.held = ""
        return buf, False

    def _consume(self, st: _SeqState, logits: torch.Tensor) -> TokenEvent:
        """Sample from ``logits`` and update the state; returns the event to stream."""
        if st.t_first is None:
            st.t_first = time.perf_counter()
        tok, lp = st.sampler.sample(logits, st.generated)
        top = st.sampler.last_top_logprobs
        st.generated.append(tok)
        if tok in st.stop_ids:
            st.held += st.decoder.flush() if st.decoder is not None else ""
            text, _ = self._scan_stop(st, final=True)
            st.text_parts.append(text)
            st.finish = FinishReason.STOP
            return TokenEvent(tok, text, lp, FinishReason.STOP, top, st.masks)
        st.output_ids.append(tok)
        if lp is not None:
            st.logprobs.append(lp)
        if st.decoder is not None:
            st.held += st.decoder.push(tok)
        text, hit = self._scan_stop(st, final=False)
        if hit:
            st.finish = FinishReason.STOP
        elif len(st.generated) >= st.params.max_tokens or st.cache.length + 1 >= st.limit:
            st.finish = FinishReason.LENGTH
        if st.finish == FinishReason.LENGTH:
            st.held += st.decoder.flush() if st.decoder is not None else ""
            more, hit = self._scan_stop(st, final=True)
            text += more
            if hit:
                st.finish = FinishReason.STOP
        st.text_parts.append(text)
        if st.finish is None:
            st.pending = [tok]
        return TokenEvent(tok, text, lp, st.finish, top, st.masks)

    def _result(self, st: _SeqState) -> GenerationResult:
        end = st.t_end or time.perf_counter()
        first = st.t_first or end
        timings = {
            "ttft_ms": (first - st.t_start) * 1e3,
            "decode_ms": (end - first) * 1e3,
            "total_ms": (end - st.t_start) * 1e3,
        }
        n_decode = max(0, len(st.generated) - 1)
        if n_decode:
            timings["tpot_ms"] = timings["decode_ms"] / n_decode
        if st.spec_passes:
            timings["spec_passes"] = float(st.spec_passes)
            timings["spec_accept_rate"] = st.spec_accepted / max(1, st.spec_proposed)
            timings["spec_rows"] = float(st.spec_proposed)
            timings["spec_accepted"] = float(st.spec_accepted)
        if st.draft_ms:
            timings["draft_ms"] = st.draft_ms
        if self.spec_rows is not None:
            # provider-visible passes (dummy passes included); a strict-mode result returned before its
            # background dummy passes have run reports the final count and how many are still due
            owed = len(st.generated) - st.passes if self._owes(st) else 0
            timings["passes"] = float(st.passes + owed)
            if owed:
                timings["passes_pending"] = float(owed)
        return GenerationResult(
            token_ids=list(st.output_ids),
            text="".join(st.text_parts),
            finish_reason=st.finish or FinishReason.ERROR,
            prompt_tokens=len(st.prompt),
            completion_tokens=len(st.generated),
            timings_ms=timings,
            masks_consumed=st.masks,
            logprobs=list(st.logprobs) if st.params.logprobs is not None else None,
        )

    # ---- synchronous API -------------------------------------------------------------------------
    # ---- speculative decoding ---------------------------------------------------------------
    def _spec_budget(self, st: _SeqState) -> int:
        """Draft tokens for this step: ``spec_tokens`` (or, with the adaptive length, ``spec_min_tokens``) when
        that many fit, else 0. Never past max_tokens, the context, or the masks needed to finish.

        The size never shrinks to "whatever is left": near the end of the token budget, the context or the
        reservation's spare rows (which depend on how many drafts were rejected) speculation simply stops,
        so the rows of a pass stay in {1, 1 + k}.
        """
        if not self.spec_tokens:
            return 0
        left = st.params.max_tokens - len(st.generated)  # tokens still to emit, including this step's
        cap = min(left - 1, st.cache.max_len - st.cache.length - len(st.pending))
        sizes = [k for k in dict.fromkeys((self.spec_tokens, self.spec_min_tokens)) if 0 < k <= cap]
        if not sizes:
            return 0
        remaining = getattr(st.session, "remaining", None)
        if callable(remaining):
            try:
                rows = int(remaining(_FIRST_LINEAR))
                reserved = getattr(st.session, "reserved_rows", None)
                held = int(reserved(_FIRST_LINEAR)) if callable(reserved) else rows
            except Exception:
                return 0  # unknown session type: do not speculate
            # Never use rows that finishing without speculation needs (the cap), and never make the
            # store reserve more just to speculate: only rows the session already holds.
            spare = min(rows - (len(st.pending) + left - 1), held - len(st.pending))
            sizes = [k for k in sizes if k <= spare]
        return sizes[0] if sizes else 0

    def _plan_drafts(self, batch: Sequence[_SeqState]) -> None:
        """Propose exactly ``k`` tokens per sequence (``k`` from :meth:`_spec_budget`), or none.

        Prompt lookup first (free); a match shorter than ``k`` is completed by the draft model, or dropped
        when there is none. Otherwise the draft model (plaintext, local) proposes all ``k``. Steps with more
        than ``spec_max_batch`` sequences do not speculate.
        """
        if self.spec_rows is not None:
            self._plan_fixed(batch)
            return
        if not self.spec_tokens:
            return
        over = len(batch) > self.spec_max_batch
        tv = self._target_vocab
        # (state, draft size, lookup tokens the draft model continues from)
        active: list[tuple[_SeqState, int, list[int]]] = []
        for st in batch:
            st.drafts = []
            if st.draft_cache is not None and st.draft_cache.length > len(st.ctx):
                st.draft_cache.truncate(len(st.ctx))  # drop proposals of an earlier (failed) step
            if over:
                continue
            k = self._spec_budget(st)
            if k <= 0:
                continue
            can_draft = self.draft is not None and st.draft_cache is not None and not self._needs_draft_prefill(st)
            seed: list[int] = []
            if self.lookup:
                found = _lookup_drafts(st.ctx + st.pending, k, max(self.lookup, self.lookup_min_n), self.lookup_min_n)
                if tv is not None:  # a proposal ends at the first id the target does not have
                    cut = next((j for j, t in enumerate(found) if t >= tv), len(found))
                    found = found[:cut]
                if len(found) >= k:
                    st.drafts = found[:k]
                    continue
                if found and not can_draft:
                    continue  # a short match alone would reveal its length: no drafts this step
                seed = found
            if can_draft:
                active.append((st, k, seed))
        if not active or self.draft is None:
            return
        if not self._draft_lock.acquire(blocking=False):  # a draft prefill (prefill lane) holds the draft
            for st, _, _ in active:
                st.drafts = []  # no drafts this step (a lookup seed alone would reveal its length)
            return
        t0 = time.perf_counter()
        try:
            self._draft_tokens(active)
        finally:
            self._draft_lock.release()
            ms = (time.perf_counter() - t0) * 1e3
            for st, _, _ in active:
                st.draft_ms += ms
            self.spec_stats["draft_calls"] += 1
            self.spec_stats["draft_ms"] += ms

    def _draft_tokens(self, active: Sequence[tuple[_SeqState, int, list[int]]]) -> None:
        """Run the draft model so each ``(st, k, seed)`` ends with ``st.drafts`` of an allowed size: ``k``,
        ``spec_min_tokens`` (adaptive length, no lookup seed) or 0."""
        assert self.draft is not None
        dv, tv = self._draft_vocab, self._target_vocab
        mn = self.spec_min_tokens
        tau = self.draft_confidence

        def din(ids: list[int]) -> list[int]:
            return ids if dv is None else [t if t < dv else dv - 1 for t in ids]

        stopped: set[int] = set()
        # Coupled drafting for sampled requests: each draft token is drawn with a fork of the request's
        # own sampler (same RNG state, penalties, temperature and filters), so it uses exactly the noise
        # the target's sample at that position will use and agrees whenever the two distributions are
        # close. Acceptance is unchanged (target sample == draft), so the output is too.
        forks = {id(st): st.sampler.fork() for st, _, _ in active if not st.params.greedy}
        # Adaptive length: decided once the first max(1, spec_min_tokens) tokens are drafted. Only then is
        # the draft's top probability needed (a softmax and a host sync per token); never with one fixed size.
        probe = {id(st): max(1, mn) for st, k, seed in active if tau > 0 and mn < k and not seed}
        unsure: set[int] = set()
        finished: set[int] = set()
        pending_seed = {id(st): len(seed) for st, _, seed in active if seed}

        def take(st: _SeqState, lg: torch.Tensor) -> None:
            sid = id(st)
            n_probe = probe.get(sid, 0)
            if len(st.drafts) < n_probe and float(torch.softmax(lg.to(torch.float32), dim=-1).max()) < tau:
                unsure.add(sid)
            fk = forks.get(sid)
            if fk is None:
                t = int(torch.argmax(lg))
            else:
                if tv is not None and lg.numel() != tv:  # align with the target's vocabulary (RNG draws)
                    lg = lg[:tv] if lg.numel() > tv else torch.cat([lg, lg.new_full((tv - lg.numel(),), float("-inf"))])
                skip = pending_seed.pop(sid, 0)
                if skip:  # lookup tokens took these positions: advance the fork past their draws
                    flat = torch.zeros(lg.numel())
                    for _ in range(skip):
                        fk.propose(flat, [])
                t = fk.propose(lg, st.generated + st.drafts)
            if tv is not None and t >= tv:  # not a target token: end this sequence's draft here
                stopped.add(sid)
                return
            st.drafts.append(t)
            if n_probe and len(st.drafts) == n_probe and sid in unsure:
                del st.drafts[mn:]  # not confident: the short size
                finished.add(sid)

        for st, _, seed in active:
            st.drafts = list(seed)
        feeds = [SequenceInput(din(st.ctx[_dcache(st).length:] + st.pending + st.drafts), _dcache(st))
                 for st, _, _ in active]
        for (st, _, _), lg in zip(active, self.draft.forward_batch(feeds), strict=True):
            take(st, lg)
        while True:
            cur = [st for st, k, _ in active if len(st.drafts) < k and id(st) not in stopped
                   and id(st) not in finished]
            if not cur:
                break
            outs = self.draft.forward_batch([SequenceInput(din([st.drafts[-1]]), _dcache(st)) for st in cur])
            for st, lg in zip(cur, outs, strict=True):
                take(st, lg)
        for st, k, seed in active:  # a draft cut short (out-of-vocabulary id) falls back to an allowed size
            n = len(st.drafts)
            if n == k or (id(st) in finished and n == mn):
                continue
            del st.drafts[mn if (not seed and 0 < mn < k and n >= mn) else 0:]

    def _needs_draft_prefill(self, st: _SeqState) -> bool:
        """First step of a request: the draft has not seen the prompt yet. That step carries no model
        drafts; the draft reads the prompt while the target's prefill runs (:meth:`_draft_prefill`)."""
        if self.spec_rows is not None:
            return (st.dstate is not None and not st.passes
                    and bool(getattr(self._drafter, "needs_prefill", True)))
        return st.draft_cache is not None and st.draft_cache.length == 0 and not st.ctx and not st.drafts

    async def _forward_with_draft_prefill(self, batch: Sequence[_SeqState], warm: Sequence[_SeqState]) -> list[Any]:
        """The target's masked forward; the draft reads new prompts in its own thread meanwhile.

        The draft thread is always waited for before returning (also when the target fails or is
        cancelled), so no thread is left writing draft caches that a retry or a release would reuse.
        """
        if not warm:
            return await self.model.aforward_batch(self._inputs(batch))
        dfut = asyncio.ensure_future(asyncio.to_thread(self._draft_prefill_safe, warm))
        try:
            return await self.model.aforward_batch(self._inputs(batch))
        finally:
            await asyncio.shield(dfut)

    def _draft_prefill_safe(self, states: Sequence[_SeqState]) -> None:
        """:meth:`_draft_prefill`; a failing draft only disables speculation for these sequences."""
        try:
            self._draft_prefill(states)
        except Exception:
            log.warning("draft prefill failed; speculation disabled for %d sequence(s)", len(states), exc_info=True)
            for st in states:
                if st.draft_cache is not None:
                    st.draft_cache.clear()
                    st.draft_cache = None
                if st.dstate is not None and self._drafter is not None:
                    dstate, st.dstate = st.dstate, None  # fixed mode: filler rows from now on
                    with contextlib.suppress(Exception):
                        self._drafter.close(dstate)

    def _draft_prefill(self, states: Sequence[_SeqState]) -> None:
        """Feed the prompt to the draft (its logits are not needed); runs alongside the target's prefill."""
        if not states:
            return
        if self.spec_rows is not None:
            if self._drafter is None:
                return
            live = [st for st in states if st.dstate is not None]
            with self._draft_lock:
                self._drafter.prefill([st.dstate for st in live], [st.prompt for st in live])
            return
        if self.draft is None:
            return
        dv = self._draft_vocab
        feeds = [SequenceInput([t if dv is None or t < dv else dv - 1 for t in st.pending], _dcache(st))
                 for st in states]
        with self._draft_lock:
            self.draft.forward_batch(feeds)

    def _inputs(self, batch: Sequence[_SeqState]) -> list[SequenceInput]:
        if self.spec_rows is not None:
            return [self._input_fixed(st) for st in batch]
        return [SequenceInput(st.pending + st.drafts, st.cache, st.session, last_logits=len(st.drafts) + 1,
                              serial_tail=len(st.drafts))
                if st.drafts else SequenceInput(st.pending, st.cache, st.session) for st in batch]

    # ---- fixed-row speculation (spec_rows) and strict-pass mode -----------------------------------
    def _owes(self, st: _SeqState) -> bool:
        """Strict mode: the sequence has ended (or was cancelled) and the provider has seen fewer passes
        than tokens generated, so dummy passes are still due. Never after an error."""
        return (self.spec_strict and (st.done or st.cancelled) and st.finish != FinishReason.ERROR
                and st.passes < len(st.generated))

    def _wants_pass(self, st: _SeqState) -> bool:
        return (not st.done and not st.cancelled) or self._owes(st)

    def _plan_fixed(self, batch: Sequence[_SeqState]) -> None:
        """Rows of this pass per sequence: the prompt (prefill pass), else exactly ``rows``: the pending
        token, up to ``rows - 1`` drafts, then filler rows. A strict-mode dummy pass feeds a random token
        and only filler rows. The drafter runs for every sequence of a pass that has rows > 1 (dummy passes
        too), always with the same ``k``. ``rows`` is ``spec_rows`` in steps of at most ``spec_max_batch``
        sequences, else 1 (both public)."""
        assert self.spec_rows is not None
        rows = self.spec_rows if len(batch) <= self.spec_max_batch else 1
        k = rows - 1
        jobs: list[DraftJob] = []
        owners: list[_SeqState] = []
        for st in batch:
            st.drafts = []
            st.fill = []
            st.pad = 0
            st.dummy = st.done or st.cancelled
            if not st.passes:  # prefill pass: the prompt rows only (a public length)
                st.rows = len(st.pending)
                continue
            if st.dummy:
                st.pending = [self._filler()]
            st.rows = rows
            if k > 0 and st.dstate is not None:
                fork = None if st.params.greedy else st.sampler.fork()
                jobs.append(DraftJob(st.dstate, st.ctx + st.pending, k, fork, list(st.generated)))
                owners.append(st)
        if jobs and self._drafter is not None:
            if self._draft_lock.acquire(blocking=False):  # busy: a prefill-lane draft prefill (public event)
                t0 = time.perf_counter()
                try:
                    self._drafter.propose(jobs)
                except Exception:
                    log.warning("drafter failed; filler rows only this pass", exc_info=True)
                    for job in jobs:
                        job.out = []
                finally:
                    self._draft_lock.release()
                    ms = (time.perf_counter() - t0) * 1e3
                    for st in owners:
                        st.draft_ms += ms
                    self.spec_stats["draft_calls"] += 1
                    self.spec_stats["draft_ms"] += ms
        proposals = {id(st): job.out for st, job in zip(owners, jobs, strict=True)}
        tv = self._target_vocab
        for st in batch:
            if not st.passes or st.rows <= 1:
                if st.dummy and self._pad_ok:
                    st.pad = st.rows
                continue
            prop = [int(t) for t in proposals.get(id(st), [])][:k]
            cut = next((j for j, t in enumerate(prop) if t < 0 or (tv is not None and t >= tv)), len(prop))
            prop = prop[:cut]
            if st.dummy:
                n_real = 0
            elif self._pad_ok:  # drafts take masks, padding does not: only spare rows, never past the budget
                n_real = min(len(prop), self._spare_rows(st))
            else:  # every row is a masked row anyway (reserved up front)
                n_real = len(prop)
            st.drafts = prop[:n_real]
            st.fill = prop[n_real:] + [self._filler() for _ in range(k - len(prop))]
            if self._pad_ok:
                st.pad = len(st.fill) + (1 if st.dummy else 0)

    def _spare_rows(self, st: _SeqState) -> int:
        """Pad-capable models: drafts this pass may use (mask rows not needed to finish without drafts,
        already held by the reservation; positions inside the token budget and the context)."""
        left = st.params.max_tokens - len(st.generated)
        cap = min(left - 1, st.limit - st.cache.length - len(st.pending))
        remaining = getattr(st.session, "remaining", None)
        if callable(remaining):
            try:
                rows = int(remaining(_FIRST_LINEAR))
                reserved = getattr(st.session, "reserved_rows", None)
                held = int(reserved(_FIRST_LINEAR)) if callable(reserved) else rows
            except Exception:
                return 0
            cap = min(cap, rows - (len(st.pending) + left - 1), held - len(st.pending))
        return max(0, cap)

    def _input_fixed(self, st: _SeqState) -> SequenceInput:
        if not st.passes or st.rows <= 1:  # prefill pass, or a 1-row step
            if st.pad:
                return SequenceInput(st.pending, st.cache, st.session, pad_rows=st.pad)  # type: ignore[call-arg]
            return SequenceInput(st.pending, st.cache, st.session)
        ids = st.pending + st.drafts + st.fill
        n = len(ids)
        if self._pad_ok:
            return SequenceInput(ids, st.cache, st.session, last_logits=n, serial_tail=n - 1,  # type: ignore[call-arg]
                                 pad_rows=st.pad)
        return SequenceInput(ids, st.cache, st.session, last_logits=n, serial_tail=n - 1)

    def _dummy_samples(self, st: _SeqState, logits: Any, start: int, rows: int) -> None:
        """Sample rows ``start..rows-1`` with a throw-away fork of the request's sampler: the same work
        per pass whatever the number of accepted drafts (the output never sees these draws)."""
        if start >= rows:
            return
        fork = st.sampler.fork()
        gen = list(st.generated)
        for j in range(start, rows):
            fork.sample(logits if rows == 1 else logits[j], gen)

    def _consume_fixed(self, st: _SeqState, logits: Any) -> list[TokenEvent]:
        stats = self.spec_stats
        first = not st.passes
        st.passes += 1
        pending = st.pending
        if first or (st.rows <= 1 and not st.dummy):  # prefill pass, or a 1-row decode pass
            st.masks += len(pending) - st.pad
            st.ctx.extend(pending)
            return [self._consume(st, logits)]
        rows = st.rows
        base = st.cache.length - rows
        st.masks += rows - st.pad
        stats["fill_rows"] += len(st.fill)
        stats["pad_rows"] += st.pad
        if st.dummy:  # no output; the cache goes back to where it was
            self._dummy_samples(st, logits, 0, rows)
            st.cache.set_length(base)
            st.pending = []
            stats["dummy_passes"] += 1
            return []
        drafts = st.drafts
        events: list[TokenEvent] = []
        i = 0
        for i in range(len(drafts) + 1):
            st.cache.set_length(base + len(pending) + i)  # the cache holds pending + drafts[:i]
            events.append(self._consume(st, logits[i]))
            if st.done or i == len(drafts) or st.generated[-1] != drafts[i]:
                break
        self._dummy_samples(st, logits, i + 1, rows)
        st.ctx.extend(pending)
        st.ctx.extend(drafts[:i])
        st.spec_passes += 1
        st.spec_proposed += len(drafts)
        st.spec_accepted += i
        stats["spec_passes"] += 1
        stats["spec_rows"] += len(drafts)
        stats["spec_accepted"] += i
        st.drafts = []
        st.fill = []
        return events

    def _drain_sync(self, states: Sequence[_SeqState]) -> None:
        """Strict mode, blocking API: run the dummy passes still due (see ``spec_strict``)."""
        while True:
            owing = [st for st in states if self._owes(st)]
            if not owing:
                return
            for i in range(0, len(owing), self.max_batch):
                self._step_sync(owing[i : i + self.max_batch])

    def _consume_step(self, st: _SeqState, logits: torch.Tensor) -> list[TokenEvent]:
        """Emit this step's tokens: one, or (speculative) every draft the sampler agrees with plus one."""
        if self.spec_rows is not None:
            return self._consume_fixed(st, logits)
        pending = st.pending
        n_in = len(pending) + len(st.drafts)
        st.masks += n_in
        if not st.drafts:
            if self.spec_tokens:
                st.ctx.extend(pending)
            return [self._consume(st, logits)]
        drafts = st.drafts
        base = st.cache.length - n_in
        events: list[TokenEvent] = []
        i = 0
        for i in range(len(drafts) + 1):
            st.cache.set_length(base + len(pending) + i)  # the cache holds pending + drafts[:i]
            events.append(self._consume(st, logits[i]))
            if st.done or i == len(drafts) or st.generated[-1] != drafts[i]:
                break
        st.ctx.extend(pending)
        st.ctx.extend(drafts[:i])
        if st.draft_cache is not None and st.draft_cache.length > len(st.ctx):
            st.draft_cache.truncate(len(st.ctx))
        st.spec_passes += 1
        st.spec_proposed += len(drafts)
        st.spec_accepted += i
        stats = self.spec_stats
        stats["spec_passes"] += 1
        stats["spec_rows"] += len(drafts)
        stats["spec_accepted"] += i
        st.drafts = []
        return events

    def _step_sync(self, states: Sequence[_SeqState]) -> list[list[TokenEvent]]:
        self._plan_drafts(states)
        warm = [st for st in states if self._needs_draft_prefill(st)]
        logits = self.model.forward_batch(self._inputs(states))
        self._draft_prefill(warm)
        return [self._consume_step(st, lg) for st, lg in zip(states, logits, strict=True)]

    def stream(
        self,
        prompt_ids: Sequence[int],
        params: SamplingParams,
        *,
        session: object | None = None,
        own_session: bool = False,
    ) -> Iterator[TokenEvent]:
        """Blocking token stream. Closing the iterator early stops generation.

        ``own_session=True`` hands ``session`` to the generator, which closes it at the end. In strict
        mode the dummy passes still due run after the last event (or after an early close).
        """
        st = self._new_state(prompt_ids, params, session, own_session)
        drain = False
        try:
            while not st.done:
                yield from self._step_sync([st])[0]
            drain = True
        except GeneratorExit:
            drain = True
            raise
        finally:
            if st.finish is None:
                st.finish = FinishReason.CANCELLED
                st.cancelled = True
            try:
                if drain:
                    self._drain_sync([st])
            except Exception:
                log.exception("strict-mode dummy passes failed")
            finally:
                self._release(st)

    def generate(
        self,
        prompt_ids: Sequence[int],
        params: SamplingParams,
        *,
        session: object | None = None,
        own_session: bool = False,
    ) -> GenerationResult:
        """Blocking full generation (``own_session``: close ``session`` when done)."""
        st = self._new_state(prompt_ids, params, session, own_session)
        try:
            while not st.done:
                self._step_sync([st])
            self._drain_sync([st])
        finally:
            if st.finish is None:
                st.finish = FinishReason.ERROR
            self._release(st)
        return self._result(st)

    def generate_many(
        self,
        prompts: Sequence[Sequence[int]],
        params: SamplingParams | Sequence[SamplingParams],
        *,
        sessions: Sequence[object | None] | None = None,
    ) -> list[GenerationResult]:
        """Generate for several prompts with batched steps (up to ``max_batch`` sequences each)."""
        plist = list(params) if isinstance(params, Sequence) else [params] * len(prompts)
        slist = list(sessions) if sessions is not None else [None] * len(prompts)
        if len(plist) != len(prompts) or len(slist) != len(prompts):
            raise ConfigError("prompts, params and sessions must have the same length")
        states: list[_SeqState] = []
        try:
            for p, sp, ss in zip(prompts, plist, slist, strict=True):
                states.append(self._new_state(p, sp, ss))
            while True:
                active = [s for s in states if self._wants_pass(s)]  # strict mode: dummy passes too
                if not active:
                    break
                for i in range(0, len(active), self.max_batch):
                    self._step_sync(active[i : i + self.max_batch])
        finally:
            for st in states:
                if st.finish is None:
                    st.finish = FinishReason.ERROR
                self._release(st)
        return [self._result(s) for s in states]

    # ---- async API ---------------------------------------------------------------------------
    async def astream(
        self,
        prompt_ids: Sequence[int],
        params: SamplingParams,
        *,
        session: object | None = None,
        own_session: bool = False,
        conversation: str | None = None,
    ) -> AsyncIterator[TokenEvent]:
        """Async token stream (used by the gateway SSE endpoint); cancellation stops generation.

        With ``own_session=True`` the generator closes ``session`` (burning unused slots) at a safe
        point: never while a batched step that uses it is in flight. ``conversation``: in strict mode a
        request waits until the previous request of the same conversation has finished its dummy passes.
        """
        st = self._new_state(prompt_ids, params, session, own_session)
        st.conv = conversation if self.spec_strict else None
        inner = self._astream_state(st)
        try:
            async for ev in inner:
                yield ev
        finally:
            await inner.aclose()

    async def agenerate(
        self,
        prompt_ids: Sequence[int],
        params: SamplingParams,
        *,
        session: object | None = None,
        own_session: bool = False,
        conversation: str | None = None,
    ) -> GenerationResult:
        """Async full generation (batched with other concurrent requests; see :meth:`astream`).

        In strict mode the result is returned with the answer; the dummy passes still due run in the
        background (:meth:`wait_drained`).
        """
        st = self._new_state(prompt_ids, params, session, own_session)
        st.conv = conversation if self.spec_strict else None
        inner = self._astream_state(st)
        try:
            async for _ in inner:
                pass
        finally:
            await inner.aclose()
        return self._result(st)

    def _engine_holds(self, st: _SeqState) -> bool:
        """The batching loop or the prefill lane owns the state (and releases it)."""
        return st in self._active or st in self._lane_wait or st in self._lane_busy

    async def _astream_state(self, st: _SeqState) -> AsyncGenerator[TokenEvent, None]:
        if st.conv is not None:
            try:
                await self._conv_gate.acquire(st.conv)
            except BaseException:
                st.finish = FinishReason.CANCELLED
                self._arelease(st)
                raise
            st.conv_held = True
        if not self.batching:
            drain = False
            try:
                while not st.done:
                    await self._run_plain(self._plan_drafts, [st])
                    warm = [st] if self._needs_draft_prefill(st) else []
                    logits = (await self._forward_with_draft_prefill([st], warm))[0]
                    for ev in await self._run_plain(self._consume_step, st, logits):
                        yield ev
                drain = True
            except GeneratorExit:
                drain = True
                raise
            finally:
                if st.finish is None:
                    st.finish = FinishReason.CANCELLED
                    st.cancelled = True
                if drain and self._owes(st):  # the answer is out; the dummy passes run in the background
                    task = asyncio.ensure_future(self._drain_async(st))
                    self._draining.add(task)
                    task.add_done_callback(self._draining.discard)
                else:
                    self._arelease(st)
            return
        st.queue = asyncio.Queue()
        self._enqueue(st)
        try:
            while True:
                item = await st.queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finish_reason is not None:
                    break
        finally:
            if st.finish is None:
                st.cancelled = True
                st.finish = FinishReason.CANCELLED
            if not self._engine_holds(st):
                self._arelease(st)

    async def _drain_async(self, st: _SeqState) -> None:
        """Strict mode without the batching loop: the dummy passes still due, then the release."""
        try:
            while self._owes(st):
                await self._run_plain(self._plan_drafts, [st])
                logits = (await self.model.aforward_batch(self._inputs([st])))[0]
                await self._run_plain(self._consume_step, st, logits)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("strict-mode dummy passes failed", exc_info=True)
            st.finish = FinishReason.ERROR
        finally:
            self._arelease(st)

    async def wait_drained(self) -> None:
        """Wait until no sequence is generating or owes dummy passes, and every close has finished."""
        while self._active or self._lane_wait or self._lane_busy or self._draining:
            if self._draining:
                await asyncio.wait(set(self._draining))
                continue
            task = self._task if self._task is not None and not self._task.done() else self._lane_task
            if task is None or task.done():
                break
            await asyncio.wait({task}, timeout=0.05)
        await self.wait_closed()

    def _enqueue(self, st: _SeqState) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._wake is None:
            self._loop = loop
            self._wake = asyncio.Event()
            self._task = None
            self._lane_task = None
        if self._lane_eligible(st):
            st.t_lane = time.perf_counter()
            self._lane_wait.append(st)
            if self._lane_task is None or self._lane_task.done():
                self._lane_task = loop.create_task(self._run_lane(), name="pgw-engine-prefill-lane")
            return
        self._active.append(st)
        self._ensure_loop()

    def _ensure_loop(self) -> None:
        assert self._wake is not None and self._loop is not None
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = self._loop.create_task(self._run_loop(), name="pgw-engine-batcher")

    def _pick_batch(self) -> list[_SeqState]:
        live = [s for s in self._active if self._wants_pass(s)]
        if self._lane_busy:  # a reservation is never used by two passes at once: wait for the lane pass
            lane = {id(s.session) for s in self._lane_busy if s.session is not None}
            live = [s for s in live if s.session is None or id(s.session) not in lane]
        if len(live) <= self.max_batch:
            return live
        start = self._rr % len(live)
        self._rr += self.max_batch
        return (live[start:] + live[:start])[: self.max_batch]

    async def _run_step(self, batch: list[_SeqState], plan: bool = True) -> None:
        if self.spec_tokens and plan:
            await self._run_plain(self._plan_drafts, batch)
        warm = [st for st in batch if self._needs_draft_prefill(st)]
        logits = await self._forward_with_draft_prefill(batch, warm)
        events = await self._run_plain(self._consume_batch, batch, logits)
        for st, evs in zip(batch, events, strict=True):
            if st.queue is not None and not st.cancelled:
                for ev in evs:
                    st.queue.put_nowait(ev)

    def _consume_batch(self, batch: Sequence[_SeqState], logits: Sequence[torch.Tensor]) -> list[list[TokenEvent]]:
        return [self._consume_step(st, lg) for st, lg in zip(batch, logits, strict=True)]

    async def _run_plain(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Sampling/detokenization off the event loop (on the model's plaintext thread when it has one)."""
        run = getattr(self.model, "run_plain", None)
        if run is None:
            return fn(*args)
        return await run(fn, *args)

    def _fail(self, st: _SeqState, exc: BaseException) -> None:
        st.finish = FinishReason.ERROR
        if st.queue is not None and not st.cancelled:
            st.queue.put_nowait(exc)

    async def _run_loop(self) -> None:
        """Background continuous-batching loop (one per event loop)."""
        try:
            while True:
                for st in [s for s in self._active if not self._wants_pass(s)]:
                    self._active.remove(st)
                    self._arelease(st)
                if not self._active:
                    return
                batch = self._pick_batch()
                if not batch:
                    lane = self._lane_task
                    if self._lane_busy and lane is not None and not lane.done():
                        await asyncio.wait({lane}, timeout=0.005)  # held back behind a lane pass
                    else:
                        await asyncio.sleep(0)
                    continue
                self.sched_stats["steps"] += 1
                self.sched_stats["step_sequences"] += len(batch)
                cohorts = self._cohorts(batch)
                if cohorts is None:
                    await self._step_isolated(batch)
                else:
                    self.sched_stats["pingpong_steps"] += 1
                    await self._run_pingpong(batch, cohorts)
                await asyncio.sleep(0)  # let consumers run between steps
        except asyncio.CancelledError:
            for st in self._active:
                self._fail(st, PGWError("generation loop cancelled"))
                self._arelease(st)
            self._active.clear()
            raise
        except BaseException as exc:  # pragma: no cover - defensive: never leave streams hanging
            log.exception("engine batching loop crashed")
            for st in self._active:
                self._fail(st, exc if isinstance(exc, PGWError) else PGWError("generation loop failed"))
                self._arelease(st)
            self._active.clear()
            raise
        finally:
            if not self._active:
                self._task = None

    # ---- prefill lane ----------------------------------------------------------------------------
    def _lane_eligible(self, st: _SeqState) -> bool:
        return (self.prefill_lane and not st.passes and not st.ctx and st.session is not None
                and len(st.prompt) >= self.prefill_lane_min_tokens)

    def _pick_prefill(self) -> list[_SeqState]:
        """The next lane pass: waiting prompts shortest first with aging (length minus ``prefill_aging``
        tokens per second waited; ties in arrival order), up to ``prefill_lane_rows`` prompt rows (at least
        one prompt). A prompt whose reservation a running sequence (or another prompt of the pass) also
        uses cannot run concurrently with it: it joins the decode steps instead (the old behaviour)."""
        now = time.perf_counter()
        for st in [s for s in self._lane_wait if s.cancelled or s.done]:
            self._lane_wait.remove(st)
            self._active.append(st)  # released by the batching loop
        held = {id(s.session) for s in self._active if s.session is not None}
        order = sorted(self._lane_wait, key=lambda s: (len(s.prompt) - self.prefill_aging * (now - s.t_lane), s.t_lane))
        group: list[_SeqState] = []
        rows = 0
        for st in order:
            sid = id(st.session)
            if sid in held:
                self._lane_wait.remove(st)
                self._active.append(st)
                continue
            if group and rows + len(st.prompt) > self.prefill_lane_rows:
                continue
            group.append(st)
            held.add(sid)
            rows += len(st.prompt)
        for st in group:
            self._lane_wait.remove(st)
        return group

    async def _run_lane(self) -> None:
        """Prefill lane: prompt passes run concurrently with the batching loop's decode steps; a prefilled
        sequence then joins the decode steps (where its first token was already streamed)."""
        try:
            while self._lane_wait:
                group = self._pick_prefill()
                if self._active:
                    self._ensure_loop()
                if not group:
                    await asyncio.sleep(0)
                    continue
                self.sched_stats["lane_steps"] += 1
                self.sched_stats["lane_sequences"] += len(group)
                self._lane_busy = group
                try:
                    await self._step_isolated(group)
                except asyncio.CancelledError:
                    for st in group:
                        self._fail(st, PGWError("generation loop cancelled"))
                        self._arelease(st)
                    raise
                finally:
                    self._lane_busy = []
                self._active.extend(group)  # live ones decode; ended ones are released there
                self._ensure_loop()
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            for st in self._lane_wait:
                self._fail(st, PGWError("generation loop cancelled"))
                self._arelease(st)
            self._lane_wait.clear()
            raise
        finally:
            if not self._lane_wait:
                self._lane_task = None

    async def _step_isolated(self, batch: list[_SeqState], plan: bool = True) -> None:
        """One step of ``batch``; a failure ends only the sequences it concerns.

        A :class:`MaskExhaustedError` consumed nothing (the backend pre-checks every reservation before
        taking rows), so each sequence is retried alone and only the exhausted one fails. Any other error
        fails every sequence of the batch. Cancellation propagates. With ``spec_rows`` the retry reuses the
        batch's plan (rows, drafts, filler; strict-mode dummy passes included), so a sequence's row count
        stays a function of the step's public session count, not of a mask-exhaustion event.
        """
        try:
            await self._run_step(batch, plan)
        except MaskExhaustedError as exc:
            if len(batch) == 1:
                self._fail(batch[0], exc)
                return
            # Nothing was consumed (the backend pre-checks); isolate the exhausted session.
            fixed = self.spec_rows is not None
            for st in batch:
                skip = not self._wants_pass(st) if fixed else (st.done or st.cancelled)
                if skip:
                    continue
                try:
                    await self._run_step([st], plan=not fixed)
                except asyncio.CancelledError:
                    raise
                except Exception as one:
                    self._fail(st, one)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("batched step failed for %d sessions: %s", len(batch), type(exc).__name__)
            for st in batch:
                self._fail(st, exc)

    def _cohorts(self, batch: Sequence[_SeqState]) -> tuple[list[_SeqState], list[_SeqState]] | None:
        """The two cohorts of a ping-pong step, or ``None`` for a plain step.

        Split only with the flag on, at least ``pingpong_min_sessions`` sequences, no draft model (its
        prompt prefill runs in a thread of its own per pass) and a distinct, explicit mask reservation per
        sequence: a shared default reservation would hand rows to two concurrent passes in an order that
        depends on timing. The split is positional (first and second half of the round-robin order), so
        which cohort a sequence is in depends only on how many sequences are live, never on data.
        """
        n = len(batch)
        prefills = self.draft is not None or (
            self._drafter is not None and bool(getattr(self._drafter, "needs_prefill", True)))
        if not self.pingpong or n < self.pingpong_min_sessions or prefills:
            return None
        sessions = [st.session for st in batch]
        if any(s is None for s in sessions) or len({id(s) for s in sessions}) != n:
            return None
        half = (n + 1) // 2
        return list(batch[:half]), list(batch[half:])

    async def _run_pingpong(self, batch: list[_SeqState],
                            cohorts: tuple[list[_SeqState], list[_SeqState]]) -> None:
        """Run both cohorts of a step concurrently (see ``pingpong``) and wait for both.

        Drafts (prompt lookup only: there is no draft model here) are planned once for the whole step, so
        whether a step speculates depends on the step's size exactly as without ping-pong. Each cohort
        handles its own failures (:meth:`_step_isolated`). Neither cohort is left running when this returns
        or raises: on cancellation both are cancelled and awaited (their passes wipe prepared requests and
        finish any plaintext work on their KV caches first).
        """
        if self.spec_tokens:
            await self._run_plain(self._plan_drafts, batch)
        tasks = [asyncio.ensure_future(self._step_isolated(c, plan=False)) for c in cohorts]
        cancelled = False
        while True:
            pending = [t for t in tasks if not t.done()]
            if not pending:
                break
            try:
                await asyncio.wait(pending)
            except asyncio.CancelledError:
                cancelled = True
                for t in pending:
                    t.cancel()
        for t in tasks:  # retrieve every outcome (no "exception never retrieved" warnings)
            err = None if t.cancelled() else t.exception()
            if err is not None and not cancelled:
                raise err  # only BaseExceptions get here: _step_isolated handles every Exception
        if cancelled or any(t.cancelled() for t in tasks):
            raise asyncio.CancelledError

    async def aclose(self) -> None:
        """Cancel the prefill lane, the batching loop and strict-mode drains; fail outstanding streams."""
        for task in (self._lane_task, self._task, *self._draining):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._lane_task = None
        self._task = None
        for st in self._active:  # handed over by the lane after the loop stopped
            self._fail(st, PGWError("generation loop cancelled"))
            self._arelease(st)
        self._active.clear()
        await self.wait_closed()
