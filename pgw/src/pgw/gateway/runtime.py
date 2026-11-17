"""Wires engine + mask store + wire client per model.

Request life cycle (masked mode, SPEC 9.4/9.5)::

    open_session(prompt, params)
        breaker / noise-alarm check  -> admission (request queue)
        -> MaskStore.reserve(fp, ring, len(prompt) + max_tokens)   (durable, before any compute)
    session.run() / session.stream()
        Generator (continuous batching) -> TransformerModel -> MaskedBackend
            -> ComputeClient (egress-guarded) -> compute service
            <- Freivalds + range check on every response (IntegrityError trips the breaker)
    session.close()
        reservation.close() (unused slots are burned) -> admission slot released

Nothing in this module logs prompt or completion content.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from pgw.core.errors import (
    ConfigError,
    ErrorCode,
    IntegrityError,
    ModelNotFoundError,
    PGWError,
    ServiceUnavailableError,
)
from pgw.core.fingerprint import short_fingerprint
from pgw.core.ring import RingParams
from pgw.engine import (
    ChatTokenizer,
    FloatBackend,
    GenerationResult,
    Generator,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    TokenEvent,
    TransformerModel,
)
from pgw.gateway.config import GatewayModelConfig, GatewaySettings, origin_of
from pgw.gateway.egress import EgressGuardTransport
from pgw.gateway.metrics import GatewayMetrics
from pgw.gateway.safety import AdmissionController, CircuitBreaker

__all__ = [
    "ActiveSession",
    "InferenceRuntime",
    "InvalidRequestError",
    "ModelRuntime",
    "NoiseAlarmError",
]

log = logging.getLogger(__name__)

#: ``fn(model_config) -> client`` override (tests, custom transports).
ClientFactory = Callable[[GatewayModelConfig], Any]


class InvalidRequestError(PGWError):
    """The API request is malformed or violates a limit (HTTP 400)."""

    code = ErrorCode.PROTOCOL


class NoiseAlarmError(IntegrityError):
    """The outbound-noise monitor alarmed; masked traffic is blocked until an admin reset."""


def _kind_of(linear_key: str) -> str:
    return linear_key.split(".", 1)[1] if "." in linear_key else linear_key


class _InstrumentedClient:
    """Wraps the compute client: fail-closed gate and round-trip timing per linear kind.

    Every round trip first checks the model's integrity circuit breaker, so once any response
    fails a check, sessions that are already generating stop sending to the service as well
    (not only new requests). No payload inspection happens here.
    """

    def __init__(self, inner: Any, metrics: GatewayMetrics, model: str, breaker: CircuitBreaker | None = None):
        self.inner = inner
        self._hist = metrics.get("pgw_gateway_round_trip_seconds")
        self._model = model
        self._breaker = breaker
        self.round_trips = 0
        self.last_error: str | None = None

    async def compute(self, request: Any) -> Any:
        """Send one compute request (refused while the breaker is open)."""
        if self._breaker is not None:
            self._breaker.check()
        kinds = {_kind_of(it.linear) for it in request.items}
        kind = kinds.pop() if len(kinds) == 1 else "mixed"
        t0 = time.perf_counter()
        try:
            resp = await self.inner.compute(request)
        except PGWError as exc:
            self.last_error = str(exc.code)
            raise
        self.round_trips += 1
        self._hist.labels(model=self._model, linear_kind=kind).observe(time.perf_counter() - t0)
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass(eq=False)
class ActiveSession:
    """One admitted request: holds the admission slot and (masked mode) the mask reservation.

    Always close it (``await session.close()``); :meth:`run` and :meth:`stream` close it themselves
    when they finish. Closing stops generation, burns unused mask slots and frees the admission slot.
    """

    runtime: ModelRuntime
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    reservation: Any
    masks_reserved: int
    queue_s: float
    t_start: float = field(default_factory=time.perf_counter)
    result: GenerationResult | None = None
    error_code: str | None = None
    completion_tokens: int = 0
    masks_consumed: int = 0
    ttft_s: float | None = None
    finish_reason: str | None = None
    handed_over: bool = False
    _closed: bool = False
    _outer: Any = None
    _outer_done: bool = False

    def _engine_kwargs(self) -> dict[str, Any]:
        """Hand the reservation to the engine (masked mode).

        With ``own_session=True`` the engine closes (burns) the reservation at a safe point, i.e.
        never while a batched step that includes this sequence is in flight, and also when the
        request fails validation. Callers set :attr:`handed_over` right before the engine call,
        whose synchronous prologue takes ownership before any suspension point.
        """
        if self.reservation is None:
            return {}
        if self.handed_over:
            raise ConfigError("mask reservation already handed to the engine")
        self.handed_over = True
        return {"session": self.reservation, "own_session": True}

    async def run(self) -> GenerationResult:
        """Non-streaming generation."""
        if self._closed:
            raise ServiceUnavailableError("session already closed")
        try:
            res = await self.runtime.generator.agenerate(self.prompt_ids, self.params, **self._engine_kwargs())
            self.result = res
            self.completion_tokens = res.completion_tokens
            self.masks_consumed = res.masks_consumed
            self.finish_reason = str(res.finish_reason)
            if "ttft_ms" in res.timings_ms:
                self.ttft_s = res.timings_ms["ttft_ms"] / 1e3
            self.runtime._observe_result(res)
            return res
        except PGWError as exc:
            self.error_code = str(exc.code)
            raise
        except Exception:
            self.error_code = str(ErrorCode.INTERNAL)
            raise
        finally:
            await self.close()

    def stream(self) -> AsyncIterator[TokenEvent]:
        """Streaming generation (closing the iterator cancels generation and burns unused masks)."""
        if self._outer is not None:
            raise ConfigError("session already streamed")
        self._outer = self._stream()
        return self._outer  # type: ignore[no-any-return]

    async def _stream(self) -> AsyncIterator[TokenEvent]:
        t_prev = time.perf_counter()
        n = 0
        if self._closed:
            raise ServiceUnavailableError("session already closed")
        it = self.runtime.generator.astream(self.prompt_ids, self.params, **self._engine_kwargs())
        try:
            async for ev in it:
                now = time.perf_counter()
                if n == 0:
                    self.ttft_s = now - self.t_start
                    self.runtime._ttft.labels(model=self.runtime.name).observe(self.ttft_s)
                else:
                    self.runtime._tpot.labels(model=self.runtime.name).observe(now - t_prev)
                t_prev = now
                n += 1
                self.completion_tokens = n
                self.masks_consumed = len(self.prompt_ids) + n - 1
                if ev.finish_reason is not None:
                    self.finish_reason = str(ev.finish_reason)
                yield ev
            self.runtime._count_tokens(len(self.prompt_ids), n)
        except PGWError as exc:
            self.error_code = str(exc.code)
            raise
        except (asyncio.CancelledError, GeneratorExit):
            self.finish_reason = self.finish_reason or "cancelled"
            raise
        except Exception:
            self.error_code = str(ErrorCode.INTERNAL)
            raise
        finally:
            try:
                aclose = getattr(it, "aclose", None)
                if aclose is not None:  # marks the engine sequence cancelled; the engine burns the masks
                    await aclose()
            finally:
                self._outer_done = True
                await self.close()

    async def close(self) -> None:
        """Stop generation, burn unused masks and release the admission slot (idempotent)."""
        if self._closed:
            return
        self._closed = True
        try:
            outer = self._outer
            if outer is not None and not self._outer_done:
                with contextlib.suppress(RuntimeError):
                    await outer.aclose()
            if self.reservation is not None and not self.handed_over:
                close = getattr(self.reservation, "close", None)
                if close is not None:
                    try:
                        await asyncio.shield(asyncio.to_thread(close))
                    except Exception:
                        log.exception("closing mask reservation failed", extra={"request_id": self.request_id})
        finally:
            self.runtime._sessions.discard(self)
            self.runtime._active_gauge.labels(model=self.runtime.name).set(len(self.runtime._sessions))
            await self.runtime.admission.release()


class ModelRuntime:
    """Everything needed to serve one model: pack, tokenizer, :class:`pgw.engine.TransformerModel`,
    backend (masked/plain/float), mask store handle, compute client, admission control.

    Args:
        config: Model configuration.
        settings: Gateway settings.
        mask_store: Shared :class:`pgw.maskstore.MaskStore` (required for masked mode).
        metrics: Shared metrics.
        client_factory: Optional override building the compute client (tests).
        transport: Optional inner httpx transport for the compute client (wrapped by the egress
            guard; tests pass an ``ASGITransport`` to an in-process service).
    """

    def __init__(
        self,
        config: GatewayModelConfig,
        settings: GatewaySettings,
        *,
        mask_store: Any = None,
        metrics: GatewayMetrics | None = None,
        client_factory: ClientFactory | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        self.settings = settings
        self.name = config.name
        self.mode = config.mode
        self.ring: RingParams = config.ring_params
        self.mask_store = mask_store
        self.metrics = metrics or GatewayMetrics(enabled=False)
        self._client_factory = client_factory
        self._transport = transport
        self.admission = AdmissionController(config.max_concurrent_sessions, config.max_queue, config.queue_timeout_s)
        self.breaker = CircuitBreaker(config.name, settings.integrity_cooldown_s)
        self.pack: Any = None
        self.tokenizer: ChatTokenizer | None = None
        self.model: TransformerModel | None = None
        self.backend: Any = None
        self.generator: Generator = None  # type: ignore[assignment]
        self.client: Any = None
        self.noise_monitor: Any = None
        self.noise_monitor_state = "disabled"
        self.fingerprint: str = ""
        self.max_context = 0
        self.started = False
        self.started_at: float | None = None
        self.service_info: dict[str, Any] = {}
        self._sessions: set[ActiveSession] = set()
        self._lock = asyncio.Lock()
        self._integrity = self.metrics.get("pgw_gateway_integrity_failures_total")
        self._active_gauge = self.metrics.get("pgw_gateway_active_sessions")
        self._noise_gauge = self.metrics.get("pgw_gateway_noise_alarm")
        self._ttft = self.metrics.get("pgw_gateway_ttft_seconds")
        self._tpot = self.metrics.get("pgw_gateway_tpot_seconds")
        self._tokens = self.metrics.get("pgw_gateway_tokens_total")

    # ---- lifecycle ----------------------------------------------------------------------------
    def _open_pack(self) -> Any:
        from pgw.modelpack import ModelPack

        return ModelPack.open(
            self.config.pack_path, verify="fingerprint", expected_fingerprint=self.config.fingerprint
        )

    def _build_noise_monitor(self) -> Any:
        if not self.settings.noise_monitor:
            self.noise_monitor_state = "disabled"
            return None
        try:
            from pgw.observability.noise import OutboundNoiseMonitor

            mon = OutboundNoiseMonitor(
                self.ring, sample_rate=self.settings.noise_sample_rate,
                fail_closed=self.settings.noise_monitor_fail_closed,
            )
        except (NotImplementedError, ImportError) as exc:
            if self.settings.noise_monitor_required:
                raise ConfigError("outbound-noise monitor is required but unavailable") from exc
            log.warning("outbound-noise monitor unavailable (%s); running without it", type(exc).__name__)
            self.noise_monitor_state = "unavailable"
            return None
        self.noise_monitor_state = "active"
        return mon

    def _build_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(self.config)
        from pgw.wire.client import ClientConfig, ComputeClient, build_ssl_context

        c = self.config
        assert c.service_url is not None
        cc = ClientConfig(
            base_url=c.service_url,
            token=c.resolved_service_token(),
            ca_file=c.service_ca_file,
            client_cert_file=c.client_cert_file,
            client_key_file=c.client_key_file,
            request_timeout_s=c.service_timeout_s,
            max_retries=c.service_max_retries,
            max_connections=max(4, min(64, c.max_concurrent_sessions)),
            user_agent="pgw-gateway/1",
        )
        inner = self._transport
        if inner is None:
            inner = httpx.AsyncHTTPTransport(
                http2=cc.http2,
                verify=build_ssl_context(cc) if cc.uses_tls else True,
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=cc.max_connections,
                    max_keepalive_connections=cc.max_keepalive_connections,
                    keepalive_expiry=cc.keepalive_expiry_s,
                ),
            )
        allowed = self.settings.egress_origins([c])
        if origin_of(c.service_url) not in allowed:  # pragma: no cover - egress_origins includes it
            raise ConfigError("service URL is not on the egress allowlist")
        guard = EgressGuardTransport(inner, allowed)
        return ComputeClient(cc, transport=guard)

    def _on_integrity_failure(self, exc: BaseException) -> None:
        check = "freivalds" if "Freivalds" in str(getattr(exc, "message", "")) else "range"
        self._integrity.labels(model=self.name, check=check).inc()
        self.breaker.trip(exc)

    async def start(self) -> None:
        """Open pack (verify fingerprint), connect to the service (fingerprint/ring check), check masks."""
        async with self._lock:
            if self.started:
                return
            t0 = time.perf_counter()
            pack = await asyncio.to_thread(self._open_pack)
            try:
                if hasattr(pack, "check_ring"):
                    pack.check_ring(self.ring)
                self.fingerprint = pack.fingerprint
                tokenizer = await asyncio.to_thread(ChatTokenizer.from_pack, pack)
                ctx = int(pack.arch.max_position_embeddings)
                self.max_context = min(ctx, self.config.max_context) if self.config.max_context else ctx
                if self.mode == "masked":
                    if self.mask_store is None:
                        raise ConfigError(f"model {self.name!r}: masked mode needs a mask store")
                    client = self._build_client()
                    try:
                        models = await client.connect(self.fingerprint, self.ring.name)
                    except PGWError:
                        await _aclose(client)
                        raise
                    except Exception as exc:
                        await _aclose(client)
                        raise ServiceUnavailableError(
                            f"cannot reach the compute service: {type(exc).__name__}"
                        ) from exc
                    info = getattr(client, "model", None)
                    max_rows = int(getattr(info, "max_rows_per_item", 8192) or 8192)
                    self.service_info = {
                        "server": getattr(models, "server", ""),
                        "origin": "{}://{}:{}".format(*origin_of(self.config.service_url or "")),
                        "max_rows_per_item": max_rows,
                    }
                    self.noise_monitor = self._build_noise_monitor()
                    self.client = _InstrumentedClient(client, self.metrics, self.name, self.breaker)
                    self.backend = MaskedBackend(
                        pack, None, self.client, self.ring, self.fingerprint,
                        freivalds_rate=self.config.freivalds_rate,
                        noise_monitor=self.noise_monitor,
                        max_rows_per_item=max_rows,
                        deadline_ms=self.config.service_deadline_ms,
                        on_integrity_failure=self._on_integrity_failure,
                    )
                    avail = await asyncio.to_thread(self._available_slots)
                    if avail == 0:
                        log.warning("no mask slots available for model %s (%s %s)",
                                    self.name, short_fingerprint(self.fingerprint), self.ring.name)
                elif self.mode == "plain":
                    log.warning("model %s runs in DEV mode 'plain' (local compute, no compute service)", self.name)
                    self.backend = PlainBackend(pack, self.ring, device=self.config.device)
                else:
                    log.warning("model %s runs in DEV mode 'float' (local float compute)", self.name)
                    self.backend = FloatBackend(pack, self.ring, device=self.config.device)
                self.model = TransformerModel(pack, self.backend, device=self.config.device)
                self.generator = Generator(
                    self.model, tokenizer, max_batch=self.config.max_concurrent_sessions, batching=True,
                )
                self.tokenizer = tokenizer
                self.pack = pack
            except BaseException:
                close = getattr(pack, "close", None)
                if close is not None:
                    close()
                raise
            self.admission.reopen()
            self.started = True
            self.started_at = time.time()
            log.info(
                "model runtime started",
                extra={"model": self.name, "mode": self.mode, "fp": short_fingerprint(self.fingerprint),
                       "ring": self.ring.name, "ms": round((time.perf_counter() - t0) * 1e3, 1)},
            )

    async def stop(self) -> None:
        """Drain sessions and close connections."""
        async with self._lock:
            if not self.started:
                return
            idle = await self.admission.drain(self.settings.drain_timeout_s)
            if not idle:
                log.warning("model %s: %d sessions still active after drain timeout; cancelling",
                            self.name, self.admission.active)
            if self.generator is not None:
                await self.generator.aclose()
            for s in list(self._sessions):
                await s.close()
            if self.client is not None:
                await _aclose(self.client)
                self.client = None
            if self.pack is not None and hasattr(self.pack, "close"):
                self.pack.close()
            self.started = False
            log.info("model runtime stopped", extra={"model": self.name})

    # ---- requests ---------------------------------------------------------------------------------
    def _available_slots(self) -> int:
        if self.mask_store is None:
            return 0
        try:
            return int(sum(i.slots_available for i in self.mask_store.inventory(self.fingerprint, self.ring)))
        except PGWError:
            log.exception("mask inventory query failed")
            return 0

    def noise_alarm(self) -> bool:
        """Whether the outbound-noise monitor currently alarms."""
        if self.noise_monitor is None:
            return False
        try:
            alarm = bool(self.noise_monitor.report().alarm)
        except Exception:  # pragma: no cover - defensive
            log.exception("noise monitor report failed")
            return self.settings.noise_monitor_fail_closed
        self._noise_gauge.labels(model=self.name).set(1.0 if alarm else 0.0)
        return alarm

    def check_available(self) -> None:
        """Raise if the model cannot take masked traffic now (fail closed)."""
        if not self.started:
            raise ServiceUnavailableError(f"model {self.name!r} is not running", details={"retry_after_s": 5.0})
        if self.mode != "masked":
            return
        self.breaker.check()
        if self.noise_alarm() and self.settings.noise_monitor_fail_closed:
            raise NoiseAlarmError(
                "outbound-noise monitor alarm: masked traffic blocked until an admin reset",
                details={"model": self.name},
            )

    def encode_chat(self, messages: Sequence[dict[str, Any]]) -> list[int]:
        """Chat messages -> prompt ids with the model's chat template."""
        assert self.tokenizer is not None
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except ConfigError as exc:
            raise InvalidRequestError(f"cannot apply the chat template: {exc.message}") from exc

    def encode_text(self, text: str) -> list[int]:
        """Completion prompt -> ids."""
        assert self.tokenizer is not None
        return self.tokenizer.encode(text, add_special_tokens=True)

    def clamp_max_tokens(self, prompt_len: int, requested: int | None) -> int:
        """Validate the prompt length and return the effective ``max_tokens``."""
        if prompt_len <= 0:
            raise InvalidRequestError("prompt is empty")
        if prompt_len > self.settings.max_prompt_tokens:
            raise InvalidRequestError(
                f"prompt has {prompt_len} tokens; the limit is {self.settings.max_prompt_tokens}",
                details={"prompt_tokens": prompt_len},
            )
        room = self.max_context - prompt_len
        if room < 1:
            raise InvalidRequestError(
                f"prompt of {prompt_len} tokens leaves no room in the {self.max_context}-token context",
                details={"prompt_tokens": prompt_len, "max_context": self.max_context},
            )
        want = requested if requested is not None else self.settings.default_max_tokens
        if want > self.settings.max_tokens_limit:
            raise InvalidRequestError(
                f"max_tokens {want} exceeds the limit {self.settings.max_tokens_limit}",
                details={"max_tokens": want},
            )
        return max(1, min(want, room))

    async def open_session(
        self, prompt_ids: Sequence[int], params: SamplingParams, *, request_id: str | None = None
    ) -> ActiveSession:
        """Admit a request and reserve its masks (fails fast with ``mask_exhausted`` before compute)."""
        rid = request_id or uuid.uuid4().hex
        self.check_available()
        ids = [int(t) for t in prompt_ids]
        queue_s = await self.admission.acquire()
        reservation = None
        slots = 0
        try:
            self.check_available()  # state may have changed while queued
            if self.mode == "masked":
                slots = len(ids) + params.max_tokens
                reservation = await _reserve_shielded(
                    self.mask_store, self.fingerprint, self.ring, slots, rid
                )
        except BaseException:
            await self.admission.release()
            raise
        sess = ActiveSession(self, rid, ids, params, reservation, slots, queue_s)
        self._sessions.add(sess)
        self._active_gauge.labels(model=self.name).set(len(self._sessions))
        return sess

    async def generate(self, prompt_ids: Sequence[int], params: SamplingParams) -> GenerationResult:
        """Admit, run and close (convenience)."""
        sess = await self.open_session(prompt_ids, params)
        return await sess.run()

    async def chat(self, messages: Sequence[dict[str, Any]], params: Any) -> Any:
        """Non-streaming chat completion -> :class:`pgw.engine.GenerationResult`."""
        ids = await asyncio.to_thread(self.encode_chat, messages)
        return await self.generate(ids, params)

    async def chat_stream(self, messages: Sequence[dict[str, Any]], params: Any) -> AsyncIterator[Any]:
        """Streaming chat completion -> async iterator of :class:`pgw.engine.TokenEvent`."""
        ids = await asyncio.to_thread(self.encode_chat, messages)
        sess = await self.open_session(ids, params)
        async for ev in sess.stream():
            yield ev

    def _count_tokens(self, prompt: int, completion: int) -> None:
        self._tokens.labels(model=self.name, kind="prompt").inc(prompt)
        self._tokens.labels(model=self.name, kind="completion").inc(completion)

    def _observe_result(self, res: GenerationResult) -> None:
        self._count_tokens(res.prompt_tokens, res.completion_tokens)
        if "ttft_ms" in res.timings_ms:
            self._ttft.labels(model=self.name).observe(res.timings_ms["ttft_ms"] / 1e3)
        if "tpot_ms" in res.timings_ms:
            self._tpot.labels(model=self.name).observe(res.timings_ms["tpot_ms"] / 1e3)

    # ---- admin ------------------------------------------------------------------------------------
    def reset_safety(self, *, breaker: bool = True, noise: bool = True) -> dict[str, bool]:
        """Admin reset of the circuit breaker and/or the noise-monitor alarm."""
        out = {"breaker_was_open": False, "noise_was_alarmed": False}
        if breaker:
            out["breaker_was_open"] = self.breaker.reset()
        if noise and self.noise_monitor is not None:
            out["noise_was_alarmed"] = self.noise_alarm()
            self.noise_monitor.reset()
            self._noise_gauge.labels(model=self.name).set(0.0)
        return out

    def ready(self) -> tuple[bool, str]:
        """Readiness and a short reason."""
        if not self.started:
            return False, "not started"
        if self.mode == "masked":
            if self.breaker.is_open:
                return False, "integrity breaker open"
            if self.noise_alarm() and self.settings.noise_monitor_fail_closed:
                return False, "noise alarm"
            if self._available_slots() <= 0:
                return False, "no mask slots"
        return True, "ok"

    def status(self) -> dict[str, Any]:
        """Admin status: fingerprint, ring, mask inventory, sessions, service health."""
        ok, reason = self.ready() if self.started else (False, "not started")
        out: dict[str, Any] = {
            "name": self.name,
            "mode": self.mode,
            "started": self.started,
            "ready": ok,
            "reason": reason,
            "fingerprint": self.fingerprint,
            "fingerprint_short": short_fingerprint(self.fingerprint) if self.fingerprint else None,
            "ring": self.ring.name,
            "max_context": self.max_context,
            "sessions": self.admission.status(),
            "breaker": self.breaker.status(),
            "freivalds_rate": self.config.freivalds_rate if self.mode == "masked" else None,
        }
        if self.mode == "masked":
            out["service"] = dict(self.service_info)
            out["masks_available"] = self._available_slots() if self.started else None
            out["noise_monitor"] = {"state": self.noise_monitor_state}
            if self.noise_monitor is not None:
                try:
                    rep = self.noise_monitor.report()
                    out["noise_monitor"].update(
                        {"alarm": bool(rep.alarm), "samples": rep.samples, "windows": rep.windows,
                         "duplicate_rows": rep.duplicate_rows, "reasons": list(rep.reasons)}
                    )
                except Exception:  # pragma: no cover - defensive
                    out["noise_monitor"]["error"] = True
            if self.client is not None:
                st = getattr(self.client, "stats", None)
                if st is not None:
                    out["client"] = {k: v for k, v in vars(st).items() if isinstance(v, int | float)}
        if self.backend is not None and hasattr(self.backend, "stats"):
            out["backend"] = self.backend.stats.snapshot()
        return out


async def _reserve_shielded(store: Any, fingerprint: str, ring: RingParams, slots: int, rid: str) -> Any:
    """``store.reserve`` in a worker thread; if the caller is cancelled meanwhile, the reservation that
    the thread still completes is closed (its slots burned) instead of leaking until GC."""
    fut = asyncio.ensure_future(asyncio.to_thread(store.reserve, fingerprint, ring, slots, session_id=rid))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        def _burn(f: asyncio.Future[Any]) -> None:
            if f.cancelled() or f.exception() is not None:
                return
            res = f.result()
            close = getattr(res, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # pragma: no cover - defensive
                    log.exception("closing an orphaned mask reservation failed", extra={"request_id": rid})

        fut.add_done_callback(_burn)
        raise


async def _aclose(client: Any) -> None:
    aclose = getattr(client, "aclose", None)
    try:
        if aclose is not None:
            await aclose()
        elif hasattr(client, "close"):
            client.close()
    except Exception:  # pragma: no cover - best effort
        log.warning("closing compute client failed")


class InferenceRuntime:
    """All model runtimes of the gateway, keyed by served name.

    Args:
        settings: Gateway settings (policy is validated here: fail closed).
        mask_store: Shared mask store (default: opened from ``maskstore_root`` + ``PGW_MASKSTORE_*``
            when any model is masked).
        metrics: Shared metrics.
        client_factory: Compute-client override (tests).
        transports: ``{model_name: httpx transport}`` inner transports (tests / custom networking).
    """

    def __init__(
        self,
        settings: GatewaySettings,
        *,
        mask_store: Any = None,
        metrics: GatewayMetrics | None = None,
        client_factory: ClientFactory | None = None,
        transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    ):
        self.settings = settings
        self.metrics = metrics or GatewayMetrics(enabled=False)
        models = settings.all_models()
        settings.validate_policy(models)
        self._own_store = False
        if mask_store is None and any(m.mode == "masked" for m in models):
            from pgw.maskstore.admin import open_store
            from pgw.maskstore.config import MaskStoreSettings

            mask_store = open_store(MaskStoreSettings(root=settings.maskstore_root))
            self._own_store = True
        self.mask_store = mask_store
        transports = transports or {}
        self.runtimes: dict[str, ModelRuntime] = {
            m.name: ModelRuntime(
                m, settings, mask_store=mask_store, metrics=self.metrics,
                client_factory=client_factory, transport=transports.get(m.name),
            )
            for m in models
        }
        if mask_store is not None and hasattr(mask_store, "metrics"):
            self.metrics.add_collect_hook(lambda: self.metrics.update_masks(mask_store.metrics()))

    def get(self, model: str) -> ModelRuntime:
        """Runtime by name; raises :class:`pgw.core.ModelNotFoundError`."""
        rt = self.runtimes.get(model)
        if rt is None:
            raise ModelNotFoundError(f"model {model!r} does not exist", details={"model": model})
        return rt

    def names(self) -> list[str]:
        """Served model names."""
        return list(self.runtimes)

    async def start(self) -> None:
        """Start all model runtimes (fails if any cannot start: fail closed)."""
        for rt in self.runtimes.values():
            await rt.start()

    async def stop(self) -> None:
        """Stop all model runtimes."""
        for rt in self.runtimes.values():
            try:
                await rt.stop()
            except Exception:
                log.exception("stopping model %s failed", rt.name)
        if self._own_store and self.mask_store is not None:
            self.mask_store.close()

    async def reload(self, model: str) -> ModelRuntime:
        """Stop and restart one model (re-verifies pack and service)."""
        rt = self.get(model)
        await rt.stop()
        await rt.start()
        return rt

    def masks_status(self) -> dict[str, Any]:
        """``/admin/masks`` payload (store status, no secrets)."""
        if self.mask_store is None:
            return {"store": None, "models": {}}
        store_status = self.mask_store.status() if hasattr(self.mask_store, "status") else {}
        per_model = {}
        for rt in self.runtimes.values():
            if rt.mode == "masked":
                per_model[rt.name] = {
                    "fingerprint": rt.fingerprint, "ring": rt.ring.name,
                    "available_slots": rt._available_slots() if rt.started else None,
                    "low_water_slots": getattr(getattr(self.mask_store, "settings", None), "low_water_slots", None),
                }
        return {"store": store_status, "models": per_model}
