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
    spec_proposed: int = 0
    spec_accepted: int = 0
    spec_passes: int = 0

    @property
    def done(self) -> bool:
        return self.finish is not None


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
        spec_tokens: int = 4,
    ):
        if max_batch < 1:
            raise ConfigError("max_batch must be >= 1")
        if spec_tokens < 0 or spec_tokens > 16:
            raise ConfigError("spec_tokens must be in [0, 16]")
        if draft is not None:
            tv = getattr(model.arch, "vocab_size", None)
            dv = getattr(draft.arch, "vocab_size", None)
            if tv is not None and dv is not None and int(tv) != int(dv):
                raise ConfigError(f"draft model vocabulary ({dv}) differs from the target's ({tv})")
        self.draft = draft if spec_tokens > 0 else None
        self.spec_tokens = spec_tokens if draft is not None else 0
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
        while self._closing:
            await asyncio.gather(*list(self._closing), return_exceptions=True)

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
            return TokenEvent(tok, text, lp, FinishReason.STOP, top)
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
        return TokenEvent(tok, text, lp, st.finish, top)

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
        """Draft tokens for this step: never past max_tokens, the context, or the masks needed to finish."""
        if self.draft is None or st.draft_cache is None:
            return 0
        left = st.params.max_tokens - len(st.generated)  # tokens still to emit, including this step's
        k = min(self.spec_tokens, left - 1, st.cache.max_len - st.cache.length - len(st.pending))
        remaining = getattr(st.session, "remaining", None)
        if callable(remaining) and k > 0:
            try:
                rows = int(remaining(_FIRST_LINEAR))
                reserved = getattr(st.session, "reserved_rows", None)
                held = int(reserved(_FIRST_LINEAR)) if callable(reserved) else rows
            except Exception:
                return 0  # unknown session type: do not speculate
            # Never use rows that finishing without speculation needs (the cap), and never make the
            # store reserve more just to speculate: only rows the session already holds.
            k = min(k, rows - (len(st.pending) + left - 1), held - len(st.pending))
        return max(0, k)

    def _plan_drafts(self, batch: Sequence[_SeqState]) -> None:
        """Run the draft model (plaintext, local) to propose up to ``spec_tokens`` tokens per sequence."""
        if self.draft is None:
            return
        active: list[tuple[_SeqState, int]] = []
        for st in batch:
            st.drafts = []
            if st.draft_cache is not None and st.draft_cache.length > len(st.ctx):
                st.draft_cache.truncate(len(st.ctx))  # drop proposals of an earlier (failed) step
            k = self._spec_budget(st)
            if k > 0:
                active.append((st, k))
        if not active:
            return
        feeds = [SequenceInput(st.ctx[_dcache(st).length:] + st.pending, _dcache(st)) for st, _ in active]
        for (st, _), lg in zip(active, self.draft.forward_batch(feeds), strict=True):
            st.drafts.append(int(torch.argmax(lg)))
        while True:
            cur = [(st, k) for st, k in active if len(st.drafts) < k]
            if not cur:
                break
            outs = self.draft.forward_batch([SequenceInput([st.drafts[-1]], _dcache(st)) for st, _ in cur])
            for (st, _), lg in zip(cur, outs, strict=True):
                st.drafts.append(int(torch.argmax(lg)))

    def _inputs(self, batch: Sequence[_SeqState]) -> list[SequenceInput]:
        return [SequenceInput(st.pending + st.drafts, st.cache, st.session, last_logits=len(st.drafts) + 1)
                if st.drafts else SequenceInput(st.pending, st.cache, st.session) for st in batch]

    def _consume_step(self, st: _SeqState, logits: torch.Tensor) -> list[TokenEvent]:
        """Emit this step's tokens: one, or (speculative) every draft the sampler agrees with plus one."""
        pending = st.pending
        n_in = len(pending) + len(st.drafts)
        st.masks += n_in
        if not st.drafts:
            if st.draft_cache is not None:
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
        st.drafts = []
        return events

    def _step_sync(self, states: Sequence[_SeqState]) -> list[list[TokenEvent]]:
        self._plan_drafts(states)
        logits = self.model.forward_batch(self._inputs(states))
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
                    logits = (await self.model.aforward_batch(self._inputs([st])))[0]
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

    async def _run_step(self, batch: list[_SeqState]) -> None:
        if self.draft is not None:
            await self._run_plain(self._plan_drafts, batch)
        logits = await self.model.aforward_batch(self._inputs(batch))
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
                try:
                    await self._run_step(batch)
                except MaskExhaustedError as exc:
                    if len(batch) == 1:
                        self._fail(batch[0], exc)
                        continue
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

    async def aclose(self) -> None:
        """Cancel the batching loop and fail outstanding streams."""
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._task = None
        await self.wait_closed()
