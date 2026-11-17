"""Generation loop: prefill, decode, stop handling, streaming and cross-session batching."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import torch

from pgw.core.errors import ConfigError, MaskExhaustedError, PGWError
from pgw.engine.kv_cache import ContextOverflowError, KVCache
from pgw.engine.model import SequenceInput, TransformerModel
from pgw.engine.sampling import Sampler, SamplingParams
from pgw.engine.tokenizer import ChatTokenizer, IncrementalDecoder

__all__ = ["FinishReason", "GenerationResult", "Generator", "SessionFactory", "TokenEvent"]

log = logging.getLogger(__name__)

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

    @property
    def done(self) -> bool:
        return self.finish is not None


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
    """

    def __init__(
        self,
        model: TransformerModel,
        tokenizer: ChatTokenizer | None,
        *,
        max_batch: int = 32,
        session_factory: SessionFactory | None = None,
        batching: bool = True,
    ):
        if max_batch < 1:
            raise ConfigError("max_batch must be >= 1")
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

    # ---- state management ---------------------------------------------------------------------
    def _new_state(self, prompt_ids: Sequence[int], params: SamplingParams, session: object | None) -> _SeqState:
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
        cap = min(len(prompt) + params.max_tokens, ctx)
        own = False
        if session is None and self.session_factory is not None:
            session = self.session_factory(len(prompt) + params.max_tokens)
            own = True
        stop_ids = set(params.stop_token_ids)
        if not params.ignore_eos:
            stop_ids.update(self._eos)
        return _SeqState(
            prompt=prompt,
            params=params,
            session=session,
            own_session=own,
            cache=self.model.new_cache(cap),
            sampler=Sampler(params, prompt),
            decoder=self.tokenizer.incremental_decoder() if self.tokenizer is not None else None,
            stop_ids=frozenset(stop_ids),
            pending=list(prompt),
        )

    @staticmethod
    def _release(st: _SeqState) -> None:
        if st.t_end is None:
            st.t_end = time.perf_counter()
        if st.own_session and st.session is not None:
            close = getattr(st.session, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # pragma: no cover - defensive
                    log.exception("closing session reservation failed")
            st.own_session = False
        st.cache.clear()

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
    def _step_sync(self, states: Sequence[_SeqState]) -> list[TokenEvent]:
        inputs = [SequenceInput(st.pending, st.cache, st.session) for st in states]
        logits = self.model.forward_batch(inputs)
        events = []
        for st, lg in zip(states, logits, strict=True):
            st.masks += len(st.pending)
            events.append(self._consume(st, lg))
        return events

    def stream(
        self, prompt_ids: Sequence[int], params: SamplingParams, *, session: object | None = None
    ) -> Iterator[TokenEvent]:
        """Blocking token stream. Closing the iterator early stops generation."""
        st = self._new_state(prompt_ids, params, session)
        try:
            while not st.done:
                yield self._step_sync([st])[0]
        finally:
            if st.finish is None:
                st.finish = FinishReason.CANCELLED
            self._release(st)

    def generate(
        self, prompt_ids: Sequence[int], params: SamplingParams, *, session: object | None = None
    ) -> GenerationResult:
        """Blocking full generation."""
        st = self._new_state(prompt_ids, params, session)
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
        self, prompt_ids: Sequence[int], params: SamplingParams, *, session: object | None = None
    ) -> AsyncIterator[TokenEvent]:
        """Async token stream (used by the gateway SSE endpoint); cancellation stops generation."""
        st = self._new_state(prompt_ids, params, session)
        inner = self._astream_state(st)
        try:
            async for ev in inner:
                yield ev
        finally:
            await inner.aclose()

    async def agenerate(
        self, prompt_ids: Sequence[int], params: SamplingParams, *, session: object | None = None
    ) -> GenerationResult:
        """Async full generation (batched with other concurrent requests)."""
        st = self._new_state(prompt_ids, params, session)
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
                    logits = await self.model.aforward(st.pending, st.cache, session=st.session)
                    st.masks += len(st.pending)
                    yield self._consume(st, logits)
            finally:
                if st.finish is None:
                    st.finish = FinishReason.CANCELLED
                self._release(st)
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
                self._release(st)

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
        inputs = [SequenceInput(st.pending, st.cache, st.session) for st in batch]
        logits = await self.model.aforward_batch(inputs)
        for st, lg in zip(batch, logits, strict=True):
            st.masks += len(st.pending)
            ev = self._consume(st, lg)
            if st.queue is not None and not st.cancelled:
                st.queue.put_nowait(ev)

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
                    self._release(st)
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
                        try:
                            await self._run_step([st])
                        except PGWError as one:
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
                self._release(st)
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
