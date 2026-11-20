"""Generation loop: prefill, decode, stop handling, streaming and cross-session batching."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

from pgw.core.errors import ConfigError, MaskExhaustedError, PGWError
from pgw.engine.kv_cache import ContextOverflowError, KVCache
from pgw.engine.model import SequenceInput, TransformerModel
from pgw.engine.sampling import Sampler, SamplingParams
from pgw.engine.tokenizer import ChatTokenizer, IncrementalDecoder

__all__ = ["FinishReason", "GenerationResult", "Generator", "SessionFactory", "TokenEvent"]

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

    Privacy of speculation: the compute service sees how many rows each pass carries. Every pass of a
    sequence carries 1 or ``1 + k`` rows, ``k`` one fixed size (or, with the adaptive draft length, one of
    two fixed sizes): a prompt-lookup proposal is exactly ``k`` tokens (a short match is completed by the
    draft model or dropped), and near the end of the token budget, the context or the mask reservation the
    draft length drops to 0 rather than to a size that depends on how many drafts were accepted. So a pass
    tells the service at most about one bit about how predictable the text is.
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
        self.sched_stats: dict[str, float] = {"steps": 0.0, "pingpong_steps": 0.0, "step_sequences": 0.0}

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
            cap = min(len(prompt) + params.max_tokens + self.spec_tokens, ctx)
            if session is None and self.session_factory is not None:
                session = self.session_factory(len(prompt) + params.max_tokens)
                own = session is not None
            stop_ids = set(params.stop_token_ids)
            if not params.ignore_eos:
                stop_ids.update(self._eos)
            return _SeqState(
                prompt=prompt,
                params=params,
                session=session,
                own_session=own,
                cache=self.model.new_cache(cap),
                draft_cache=self.draft.new_cache(cap) if self.draft is not None else None,
                sampler=Sampler(params, prompt),
                decoder=self.tokenizer.incremental_decoder() if self.tokenizer is not None else None,
                stop_ids=frozenset(stop_ids),
                pending=list(prompt),
            )
        except BaseException:
            if own and session is not None:
                self._close_session(session)
            raise

    @staticmethod
    def _close_session(session: object) -> None:
        close = getattr(session, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # pragma: no cover - defensive
                log.exception("closing session reservation failed")

    @staticmethod
    def _release(st: _SeqState) -> None:
        """Synchronous release (blocking API only: closes an owned session on this thread)."""
        if st.t_end is None:
            st.t_end = time.perf_counter()
        if st.own_session and st.session is not None:
            st.own_session = False
            Generator._close_session(st.session)
        st.cache.clear()
        if st.draft_cache is not None:
            st.draft_cache.clear()

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
        st.cache.clear()
        if st.draft_cache is not None:
            st.draft_cache.clear()
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
        elif len(st.generated) >= st.params.max_tokens or st.cache.length + 1 >= st.cache.max_len:
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
        t0 = time.perf_counter()
        try:
            self._draft_tokens(active)
        finally:
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

    def _draft_prefill(self, states: Sequence[_SeqState]) -> None:
        """Feed the prompt to the draft (its logits are not needed); runs alongside the target's prefill."""
        if self.draft is None or not states:
            return
        dv = self._draft_vocab
        feeds = [SequenceInput([t if dv is None or t < dv else dv - 1 for t in st.pending], _dcache(st))
                 for st in states]
        self.draft.forward_batch(feeds)

    def _inputs(self, batch: Sequence[_SeqState]) -> list[SequenceInput]:
        return [SequenceInput(st.pending + st.drafts, st.cache, st.session, last_logits=len(st.drafts) + 1,
                              serial_tail=len(st.drafts))
                if st.drafts else SequenceInput(st.pending, st.cache, st.session) for st in batch]

    def _consume_step(self, st: _SeqState, logits: torch.Tensor) -> list[TokenEvent]:
        """Emit this step's tokens: one, or (speculative) every draft the sampler agrees with plus one."""
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

        ``own_session=True`` hands ``session`` to the generator, which closes it at the end.
        """
        st = self._new_state(prompt_ids, params, session, own_session)
        try:
            while not st.done:
                yield from self._step_sync([st])[0]
        finally:
            if st.finish is None:
                st.finish = FinishReason.CANCELLED
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
                active = [s for s in states if not s.done]
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
    ) -> AsyncIterator[TokenEvent]:
        """Async token stream (used by the gateway SSE endpoint); cancellation stops generation.

        With ``own_session=True`` the generator closes ``session`` (burning unused slots) at a safe
        point: never while a batched step that uses it is in flight.
        """
        st = self._new_state(prompt_ids, params, session, own_session)
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
    ) -> GenerationResult:
        """Async full generation (batched with other concurrent requests; see :meth:`astream`)."""
        st = self._new_state(prompt_ids, params, session, own_session)
        inner = self._astream_state(st)
        try:
            async for _ in inner:
                pass
        finally:
            await inner.aclose()
        return self._result(st)

    async def _astream_state(self, st: _SeqState) -> AsyncGenerator[TokenEvent, None]:
        if not self.batching:
            try:
                while not st.done:
                    await self._run_plain(self._plan_drafts, [st])
                    warm = [st] if self._needs_draft_prefill(st) else []
                    logits = (await self._forward_with_draft_prefill([st], warm))[0]
                    for ev in await self._run_plain(self._consume_step, st, logits):
                        yield ev
            finally:
                if st.finish is None:
                    st.finish = FinishReason.CANCELLED
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
            if st not in self._active:
                self._arelease(st)

    def _enqueue(self, st: _SeqState) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._wake is None:
            self._loop = loop
            self._wake = asyncio.Event()
            self._task = None
        self._active.append(st)
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run_loop(), name="pgw-engine-batcher")

    def _pick_batch(self) -> list[_SeqState]:
        live = [s for s in self._active if not s.cancelled and not s.done]
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
                for st in [s for s in self._active if s.cancelled or s.done]:
                    self._active.remove(st)
                    self._arelease(st)
                if not self._active:
                    return
                batch = self._pick_batch()
                if not batch:
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

    async def _step_isolated(self, batch: list[_SeqState], plan: bool = True) -> None:
        """One step of ``batch``; a failure ends only the sequences it concerns.

        A :class:`MaskExhaustedError` consumed nothing (the backend pre-checks every reservation before
        taking rows), so each sequence is retried alone and only the exhausted one fails. Any other error
        fails every sequence of the batch. Cancellation propagates.
        """
        try:
            await self._run_step(batch, plan)
        except MaskExhaustedError as exc:
            if len(batch) == 1:
                self._fail(batch[0], exc)
                return
            # Nothing was consumed (the backend pre-checks); isolate the exhausted session.
            for st in batch:
                if st.done or st.cancelled:
                    continue
                try:
                    await self._run_step([st])
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
        if not self.pingpong or n < self.pingpong_min_sessions or self.draft is not None:
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
        """Cancel the batching loop and fail outstanding streams."""
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._task = None
        await self.wait_closed()
