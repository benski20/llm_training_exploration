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
import math
import random
import threading
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
    MaskExhaustedError,
    ModelNotFoundError,
    PGWError,
    ServiceUnavailableError,
    ShapeError,
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
from pgw.engine.backends import Int8DraftBackend, int8_lm_head
from pgw.gateway.config import GatewayModelConfig, GatewaySettings, origin_of
from pgw.gateway.egress import EgressGuardTransport
from pgw.gateway.masks import WindowedReservation
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
        self._failures = metrics.get("pgw_gateway_round_trip_failures_total")
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
            self._failures.labels(model=self._model, code=str(exc.code)).inc()
            raise
        except Exception as exc:
            self.last_error = type(exc).__name__
            self._failures.labels(model=self._model, code="transport").inc()
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
                self.masks_consumed = ev.masks or (len(self.prompt_ids) + n - 1)
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
            if self.reservation is not None:
                self.masks_reserved = int(getattr(self.reservation, "slots", self.masks_reserved) or 0)
                burned = max(0, self.masks_reserved - self.masks_consumed)
                if burned:
                    self.runtime._burned.labels(model=self.runtime.name).inc(burned)
            self.runtime._sessions.discard(self)
            self.runtime._active_gauge.labels(model=self.runtime.name).set(len(self.runtime._sessions))
            await self.runtime.admission.release()

    @property
    def masks_burned(self) -> int:
        """Reserved mask slots this request never used (destroyed on close)."""
        return max(0, self.masks_reserved - self.masks_consumed) if self.reservation is not None else 0


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
        self.breaker = CircuitBreaker(
            config.name, settings.integrity_cooldown_s, max_auto_resets=settings.integrity_max_auto_resets
        )
        self.pack: Any = None
        self.tokenizer: ChatTokenizer | None = None
        self.model: TransformerModel | None = None
        self.backend: Any = None
        self.generator: Generator = None  # type: ignore[assignment]
        self.draft_model: TransformerModel | None = None
        self.client: Any = None
        self.noise_monitor: Any = None
        self.noise_monitor_state = "disabled"
        #: ``fn(event, model, reason)`` called on safety events (integrity failure, noise alarm).
        self.safety_listeners: list[Callable[[str, str, str], object]] = []
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
        self._burned = self.metrics.get("pgw_gateway_masks_burned_total")
        self._ready_gauge = self.metrics.get("pgw_gateway_model_ready")
        self._attempts = self.metrics.get("pgw_gateway_service_attempts_total")
        self._retries = self.metrics.get("pgw_gateway_service_retries_total")
        adm_active = self.metrics.get("pgw_gateway_admission_active")
        adm_waiting = self.metrics.get("pgw_gateway_admission_waiting")
        adm_rejected = self.metrics.get("pgw_gateway_admission_rejected_total")
        adm_wait = self.metrics.get("pgw_gateway_queue_wait_seconds")
        name = config.name

        def _adm_change(active: int, waiting: int) -> None:
            adm_active.labels(model=name).set(active)
            adm_waiting.labels(model=name).set(waiting)

        self.admission.on_change = _adm_change
        self.admission.on_reject = lambda reason: adm_rejected.labels(model=name, reason=reason).inc()
        self.admission.on_wait = lambda waited: adm_wait.labels(model=name).observe(waited)
        #: Life-cycle: "stopped", "connecting" (service not reachable yet; retried in the
        #: background), "ready" (started), "failed" (permanent misconfiguration).
        self.state = "stopped"
        self.last_error: str | None = None
        self.connect_attempts = 0
        self._connect_task: asyncio.Task[None] | None = None
        self._retries_seen = 0
        self._prepared: tuple[Any, ChatTokenizer] | None = None

    # ---- lifecycle ----------------------------------------------------------------------------
    def _open_pack(self) -> Any:
        from pgw.modelpack import ModelPack

        # "full": hash plain tensors, tokenizer files and every linear (incl. scales/biases) against
        # the manifest, and the manifest against the pinned fingerprint. Packs may come from images
        # the compute operator publishes; a swapped tokenizer/template must not go unnoticed.
        return ModelPack.open(
            self.config.pack_path, verify=self.settings.pack_verify, expected_fingerprint=self.config.fingerprint
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
        except ImportError as exc:
            if self.settings.noise_monitor_is_required:
                raise ConfigError("outbound-noise monitor is required but unavailable") from exc
            log.warning("outbound-noise monitor unavailable (%s); running without it", type(exc).__name__)
            self.noise_monitor_state = "unavailable"
            return None
        self.noise_monitor_state = "active"
        if hasattr(mon, "add_alarm_listener"):
            mon.add_alarm_listener(self._on_noise_alarm)
        return mon

    def _notify_safety(self, event: str, reason: str) -> None:
        for fn in list(self.safety_listeners):
            try:
                fn(event, self.name, reason)
            except Exception:  # a listener must never break the data path
                log.exception("safety listener failed for %s", event)

    def _on_noise_alarm(self, report: Any) -> None:
        reasons = list(getattr(report, "reasons", []) or [])
        # reasons are test names and p-values only (no payload values, SPEC 11.2)
        self._notify_safety("safety.noise_alarm", "; ".join(str(r) for r in reasons[-2:]) or "alarm")

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
        real_network = inner is None
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
        # The compute stream connects only to service_url, which is on the egress allowlist (checked above).
        return ComputeClient(cc, transport=guard, on_attempt=self._on_service_attempt,
                             stream_with_transport=real_network)

    def _on_service_attempt(self, path: str, status: int | None, seconds: float, code: str | None) -> None:
        """``ComputeClient`` hook: every HTTP attempt (incl. retries) by outcome."""
        self._attempts.labels(model=self.name, code="ok" if code is None else code).inc()

    def refresh_metrics(self) -> None:
        """Scrape-time gauges: readiness and the client's retry counter."""
        ok, _ = self.ready()
        self._ready_gauge.labels(model=self.name).set(1.0 if ok else 0.0)
        st = getattr(self.client, "stats", None)
        retries = int(getattr(st, "retries", 0) or 0)
        if retries > self._retries_seen:
            self._retries.labels(model=self.name).inc(retries - self._retries_seen)
        self._retries_seen = retries

    def _on_integrity_failure(self, exc: BaseException) -> None:
        check = "freivalds" if "Freivalds" in str(getattr(exc, "message", "")) else "range"
        self._integrity.labels(model=self.name, check=check).inc()
        self.breaker.trip(exc)
        self._notify_safety("safety.integrity_failure", check)

    @staticmethod
    def _transient(exc: BaseException) -> bool:
        """Connection problems worth retrying (vs. permanent misconfiguration)."""
        if isinstance(exc, PGWError):
            return bool(exc.retryable) and not isinstance(exc, ConfigError)
        return isinstance(exc, OSError | httpx.TransportError | TimeoutError)

    async def start(self) -> None:
        """Open pack (verify fingerprint), connect to the service (fingerprint/ring check), check masks.

        If the compute service is not reachable yet (e.g. it is being rolled out), the model goes
        to state ``connecting``: it reports not ready, requests fail fast with 503 + Retry-After,
        and the connection is retried in the background with exponential back-off. Permanent
        problems (fingerprint/ring/version mismatch, bad configuration) raise.
        """
        async with self._lock:
            if self.started or (self._connect_task is not None and not self._connect_task.done()):
                return
            t0 = time.perf_counter()
            self.state = "starting"
            try:
                pack = await asyncio.to_thread(self._open_pack)
            except BaseException as exc:
                self._fail(exc)
                raise
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
                    try:
                        client, models = await self._connect()
                    except BaseException as exc:
                        if not self._transient(exc):
                            raise
                        self._prepared = (pack, tokenizer)
                        self.state = "connecting"
                        self.last_error = getattr(exc, "message", None) or type(exc).__name__
                        log.warning(
                            "model %s: compute service not reachable yet (%s); retrying in the background",
                            self.name, self.last_error,
                        )
                        self._connect_task = asyncio.get_running_loop().create_task(
                            self._connect_loop(), name=f"pgw-connect-{self.name}"
                        )
                        return
                    await self._finish_masked(pack, tokenizer, client, models)
                else:
                    self._finish_local(pack, tokenizer)
            except BaseException as exc:
                close = getattr(pack, "close", None)
                if close is not None:
                    close()
                self._fail(exc)
                raise
            self._mark_started(t0)

    def _fail(self, exc: BaseException) -> None:
        self.state = "failed"
        self.last_error = getattr(exc, "message", None) or type(exc).__name__

    def _mark_started(self, t0: float) -> None:
        self.admission.reopen()
        self.started = True
        self.state = "ready"
        self.last_error = None
        self.started_at = time.time()
        log.info(
            "model runtime started",
            extra={"model": self.name, "mode": self.mode, "fp": short_fingerprint(self.fingerprint),
                   "ring": self.ring.name, "ms": round((time.perf_counter() - t0) * 1e3, 1)},
        )

    async def _connect(self) -> tuple[Any, Any]:
        client = self._build_client()
        self.connect_attempts += 1
        try:
            models = await client.connect(self.fingerprint, self.ring.name)
        except PGWError:
            await _aclose(client)
            raise
        except Exception as exc:
            await _aclose(client)
            raise ServiceUnavailableError(f"cannot reach the compute service: {type(exc).__name__}") from exc
        return client, models

    async def _connect_loop(self) -> None:
        """Background reconnect with exponential back-off until connected or permanently failed."""
        delay = 1.0
        while True:
            await asyncio.sleep(delay * random.uniform(0.8, 1.2))  # noqa: S311 - jitter only
            delay = min(delay * 2, self.settings.connect_retry_max_s)
            prepared = self._prepared
            if prepared is None:
                return
            t0 = time.perf_counter()
            try:
                client, models = await self._connect()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                if self._transient(exc):
                    self.last_error = getattr(exc, "message", None) or type(exc).__name__
                    log.info("model %s: compute service still unreachable (%s)", self.name, self.last_error)
                    continue
                log.error("model %s: permanent start-up failure: %s", self.name,
                          getattr(exc, "message", type(exc).__name__))
                self._prepared = None
                self._close_pack(prepared[0])
                self._fail(exc)
                return
            async with self._lock:
                if self._prepared is not prepared:  # stopped meanwhile
                    await _aclose(client)
                    return
                self._prepared = None
                try:
                    await self._finish_masked(prepared[0], prepared[1], client, models)
                except BaseException as exc:
                    self._close_pack(prepared[0])
                    self._fail(exc)
                    log.error("model %s: start-up failed after connecting: %s", self.name, self.last_error)
                    return
                self._mark_started(t0)
                log.info("model %s connected after %d attempts", self.name, self.connect_attempts)
                return

    @staticmethod
    def _close_pack(pack: Any) -> None:
        close = getattr(pack, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()

    async def _finish_masked(self, pack: Any, tokenizer: ChatTokenizer, client: Any, models: Any) -> None:
        try:
            info = getattr(client, "model", None)
            max_rows = int(getattr(info, "max_rows_per_item", 8192) or 8192)
            self.service_info = {
                "server": getattr(models, "server", ""),
                "origin": "{}://{}:{}".format(*origin_of(self.config.service_url or "")),
                "max_rows_per_item": max_rows,
            }
            self.noise_monitor = self._build_noise_monitor()
            self.client = _InstrumentedClient(client, self.metrics, self.name, self.breaker)
            self._retries_seen = 0
            self.backend = MaskedBackend(
                pack, None, self.client, self.ring, self.fingerprint,
                freivalds_rate=self.config.freivalds_rate,
                noise_monitor=self.noise_monitor,
                max_rows_per_item=max_rows,
                deadline_ms=self.config.service_deadline_ms,
                on_integrity_failure=self._on_integrity_failure,
                min_soundness_bits=self.settings.min_soundness_bits,
            )
            avail = await asyncio.to_thread(self._available_slots)
            if avail == 0:
                log.warning("no mask slots available for model %s (%s %s)",
                            self.name, short_fingerprint(self.fingerprint), self.ring.name)
            await asyncio.to_thread(self._check_mask_cache_budget)
            self._finish_engine(pack, tokenizer)
        except BaseException:
            if self.client is not None:
                await _aclose(self.client)
                self.client = None
            else:
                await _aclose(client)
            raise

    def _finish_local(self, pack: Any, tokenizer: ChatTokenizer) -> None:
        if self.mode == "plain":
            log.warning("model %s runs in DEV mode 'plain' (local compute, no compute service)", self.name)
            self.backend = PlainBackend(pack, self.ring, device=self.config.device)
        else:
            log.warning("model %s runs in DEV mode 'float' (local float compute)", self.name)
            self.backend = FloatBackend(pack, self.ring, device=self.config.device)
        self._finish_engine(pack, tokenizer)

    def _finish_engine(self, pack: Any, tokenizer: ChatTokenizer) -> None:
        self.model = TransformerModel(pack, self.backend, device=self.config.device)
        self.model.inline_max_rows = self.config.inline_max_rows
        if self.config.batch_lm_head is not None:
            self.model.batch_lm_head = self.config.batch_lm_head
        self.draft_model = self._load_draft()
        self.generator = Generator(
            self.model, tokenizer, max_batch=self.config.max_concurrent_sessions, batching=True,
            draft=self.draft_model, spec_tokens=self.config.spec_tokens, lookup=self.config.spec_lookup,
            spec_min_tokens=self.config.spec_min_tokens, draft_confidence=self.config.spec_confidence,
        )
        self.tokenizer = tokenizer
        self.pack = pack

    def _load_draft(self) -> TransformerModel | None:
        """The speculative-decoding draft model: plaintext, local to the gateway (never sent anywhere)."""
        path = self.config.draft_pack_path
        if path is None or self.config.spec_tokens == 0:
            return None
        from pgw.modelpack import ModelPack

        dpack = ModelPack.open(path, verify=self.settings.pack_verify)
        # Approximate compute is fine: the draft only proposes tokens, the masked target decides every one.
        # CPU: weight-only int8 when it measures faster here than float32 (CPU-dependent); GPU: float16.
        dev = self.config.draft_device or self.config.device
        import torch

        kind = "float32"
        on_cpu = torch.device(dev).type == "cpu"
        if on_cpu and Int8DraftBackend.available() and Int8DraftBackend.faster_than_float(dpack):
            draft = TransformerModel(dpack, Int8DraftBackend(dpack, self.ring), device=dev)
            draft.lm_head_fn = int8_lm_head(draft.lm_head)
            kind = "int8"
        elif torch.device(dev).type == "cuda":
            draft = TransformerModel(dpack, FloatBackend(dpack, self.ring, device=dev, dtype=torch.float16,
                                                         batch_rows=True), device=dev, dtype=torch.float16)
            w16 = draft.lm_head.to(torch.float16)
            draft.lm_head = w16  # drop the float32 copy (GPU memory next to the target's service)
            draft.lm_head_fn = lambda h: (h.to(torch.float16) @ w16.t()).to(torch.float32)
            kind = "float16"
        else:
            draft = TransformerModel(dpack, FloatBackend(dpack, self.ring, device=dev, batch_rows=True), device=dev)
        log.info("model %s: speculative decoding with draft %s (%s on %s, %d-%d tokens per step)", self.name,
                 getattr(dpack, "name", path), kind, dev, self.config.spec_min_tokens, self.config.spec_tokens)
        return draft

    @property
    def spec_factor(self) -> float:
        """Mask rows per generated token in the worst case (1 without speculation)."""
        speculating = (self.draft_model is not None or self.config.spec_lookup > 0) and self.config.spec_tokens > 0
        return min(1.0 + self.config.spec_tokens, self.config.spec_row_budget) if speculating else 1

    def mask_cache_estimate(self) -> int:
        """Worst-case decrypted mask bytes cached by ``max_concurrent_sessions`` sessions (0 if unknown)."""
        est = getattr(self.mask_store, "reservation_cache_bytes", None)
        if est is None:
            return 0
        try:
            per_session = int(est(self.fingerprint, self.ring))
        except PGWError:
            return 0
        return per_session * self.config.max_concurrent_sessions

    def _check_mask_cache_budget(self) -> None:
        budget = self.settings.mask_cache_budget_bytes
        if budget <= 0:
            return
        total = self.mask_cache_estimate()
        if total > budget:
            log.warning(
                "model %s: %d concurrent sessions may cache %.2f GB of decrypted masks (segment chunk size), "
                "above mask_cache_budget_bytes=%.2f GB; regenerate segments with the default (byte-sized) "
                "chunks or lower max_concurrent_sessions",
                self.name, self.config.max_concurrent_sessions, total / 1e9, budget / 1e9,
            )

    async def stop(self) -> None:
        """Drain sessions and close connections."""
        task, self._connect_task = self._connect_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        async with self._lock:
            prepared, self._prepared = self._prepared, None
            if prepared is not None:
                self._close_pack(prepared[0])
            if not self.started:
                if self.state != "failed":
                    self.state = "stopped"
                return
            idle = await self.admission.drain(self.settings.drain_timeout_s)
            if not idle:
                log.warning("model %s: %d sessions still active after drain timeout; cancelling",
                            self.name, self.admission.active)
            if self.generator is not None:
                await self.generator.aclose()
            for s in list(self._sessions):
                await s.close()
            if self.generator is not None:
                await self.generator.wait_closed()
            if self.model is not None:
                await asyncio.to_thread(self.model.close)
            if self.client is not None:
                await _aclose(self.client)
                self.client = None
            if self.pack is not None and hasattr(self.pack, "close"):
                self.pack.close()
            self.started = False
            self.state = "stopped"
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
            if self.state == "connecting":
                raise ServiceUnavailableError(
                    f"model {self.name!r} is waiting for the compute service",
                    details={"retry_after_s": 5.0, "model": self.name},
                )
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
        if self.tokenizer is None:
            self.check_available()
            raise ServiceUnavailableError(f"model {self.name!r} is not running", details={"retry_after_s": 5.0})
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except ConfigError as exc:
            raise InvalidRequestError(f"cannot apply the chat template: {exc.message}") from exc

    def encode_text(self, text: str) -> list[int]:
        """Completion prompt -> ids."""
        if self.tokenizer is None:
            self.check_available()
            raise ServiceUnavailableError(f"model {self.name!r} is not running", details={"retry_after_s": 5.0})
        return self.tokenizer.encode(text, add_special_tokens=True)

    def check_token_ids(self, ids: Sequence[int]) -> list[int]:
        """Validate a token-id prompt against the model vocabulary (``/v1/completions`` token arrays)."""
        vocab = self.vocab_size
        out = [int(t) for t in ids]
        if not out:
            raise InvalidRequestError("prompt is empty")
        bad = next((t for t in out if t < 0 or (vocab is not None and t >= vocab)), None)
        if bad is not None:
            raise InvalidRequestError(
                f"token id {bad} is outside the vocabulary [0, {vocab})", details={"token_id": bad}
            )
        return out

    @property
    def vocab_size(self) -> int | None:
        """Embedding rows of the model (``None`` while the model is not loaded)."""
        if self.model is not None:
            v = getattr(self.model.arch, "vocab_size", None)
            if v:
                return int(v)
        if self.tokenizer is not None:
            return len(self.tokenizer)
        return None

    def decode_ids(self, ids: Sequence[int]) -> str:
        """Token ids -> text (``echo`` of token-id prompts)."""
        if self.tokenizer is None:
            self.check_available()
            raise ServiceUnavailableError(f"model {self.name!r} is not running", details={"retry_after_s": 5.0})
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def clamp_max_tokens(self, prompt_len: int, requested: int | None) -> int:
        """Validate the prompt length and return the effective ``max_tokens``.

        Raises:
            ShapeError: (``bad_shape``) the prompt is longer than ``max_prompt_tokens`` or leaves
                no room in the model context.
            InvalidRequestError: (``protocol``) empty prompt, or ``max_tokens`` above
                ``max_tokens_limit``. A valid ``max_tokens`` is clamped to the context left.
        """
        if prompt_len <= 0:
            raise InvalidRequestError("prompt is empty")
        if prompt_len > self.settings.max_prompt_tokens:
            raise ShapeError(
                f"prompt has {prompt_len} tokens; the limit is {self.settings.max_prompt_tokens}",
                details={"prompt_tokens": prompt_len},
            )
        room = self.max_context - prompt_len
        if room < 1:
            raise ShapeError(
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
        return (await self.open_sessions([prompt_ids], params, request_id=request_id))[0]

    async def open_sessions(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams, *, request_id: str | None = None
    ) -> list[ActiveSession]:
        """Admit ``len(prompts)`` sequences atomically, then reserve their masks.

        All admission slots are taken in one step (no hold-and-wait between concurrent
        multi-prompt requests) and masks are reserved only once every slot is held, so a request
        that times out in the queue never burns masks. Each returned session owns one slot.
        """
        rid = request_id or uuid.uuid4().hex
        self.check_available()
        ids_list = [[int(t) for t in p] for p in prompts]
        n = len(ids_list)
        if n == 0:
            raise InvalidRequestError("no prompts")
        queue_s = await self.admission.acquire(n)
        sessions: list[ActiveSession] = []
        released = 0
        try:
            self.check_available()  # state may have changed while queued
            for i, ids in enumerate(ids_list):
                srid = rid if i == 0 else f"{rid}-{i}"
                reservation, slots = (None, 0)
                if self.mode == "masked":
                    reservation, slots = await self._reserve_for(ids, params, srid)
                sess = ActiveSession(self, srid, ids, params, reservation, slots, queue_s)
                sessions.append(sess)
                released += 1  # from now on the session releases its own slot
                self._sessions.add(sess)
        except BaseException:
            for sess in sessions:
                await sess.close()
            if n - released:
                await self.admission.release(n - released)
            raise
        self._active_gauge.labels(model=self.name).set(len(self._sessions))
        return sessions

    async def _reserve_for(self, ids: list[int], params: SamplingParams, rid: str) -> tuple[Any, int]:
        """Reserve the prompt plus a first window (or everything when windows are disabled)."""
        need = len(ids) + params.max_tokens
        # Speculative decoding may use more rows (rejected drafts); the cap allows it, but the engine
        # only speculates with rows the session already holds, so a short store never fails a request.
        cap = len(ids) + math.ceil(params.max_tokens * self.spec_factor)
        window = int(self.settings.mask_window_tokens)
        if window <= 0:  # everything up front: reserve what the request needs (speculation uses slack only)
            cap = need
        first = cap if window <= 0 else min(cap, len(ids) + window)
        if first < cap:
            # Fail fast (before any compute) when the store could not serve the whole request.
            avail = await asyncio.to_thread(self._available_slots)
            if avail < need:
                raise MaskExhaustedError(
                    "not enough mask slots for this request",
                    details={"needed": need, "available": avail, "model": self.name},
                )
        res = await _reserve_shielded(self.mask_store, self.fingerprint, self.ring, first, rid)
        if first >= cap:
            return res, first
        store, fp, ring = self.mask_store, self.fingerprint, self.ring

        def extend(slots: int) -> Any:  # runs in the backend's worker thread
            return store.reserve(fp, ring, slots, session_id=rid)

        return WindowedReservation(res, extend, cap=cap, window=window), first

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
            if self.state == "connecting":
                return False, "waiting for the compute service"
            if self.state == "failed":
                return False, f"failed: {self.last_error}"
            return False, "not started"
        if self.mode == "masked":
            if self.breaker.is_open:
                return False, "integrity breaker open"
            if self.noise_alarm() and self.settings.noise_monitor_fail_closed:
                return False, "noise alarm"
            if self.noise_monitor_state == "unavailable" and self.settings.noise_monitor_is_required:
                return False, "noise monitor unavailable"
            if self._available_slots() <= 0:
                return False, "no mask slots"
        return True, "ok"

    def status(self) -> dict[str, Any]:
        """Admin status: fingerprint, ring, mask inventory, sessions, service health."""
        ok, reason = self.ready()
        out: dict[str, Any] = {
            "name": self.name,
            "mode": self.mode,
            "started": self.started,
            "state": self.state,
            "last_error": self.last_error,
            "connect_attempts": self.connect_attempts,
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
            # Fail at start-up, not on the first request (which would otherwise get a 500 `config`):
            # reserve() refuses to hand out masks without an external monotonic counter (SPEC 7.6).
            require_counter = getattr(mask_store, "_require_external_counter", None)
            if require_counter is not None:
                try:
                    require_counter()
                except Exception:
                    mask_store.close()
                    raise
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
        self.metrics.add_collect_hook(self._refresh_model_metrics)
        self._gc_thread: threading.Thread | None = None
        self._gc_stop = threading.Event()
        self.gc_runs = 0
        self._started = False
        self._draining = False

    def _refresh_model_metrics(self) -> None:
        for rt in self.runtimes.values():
            rt.refresh_metrics()

    def ready(self) -> bool:
        """Process-level readiness (``/readyz``): start-up finished and the process can answer.

        Per-model readiness (mask exhaustion, breaker, service reachability) is deliberately *not*
        part of it: requests for an unavailable model fail fast with typed errors instead of the
        whole gateway leaving the load balancer.
        """
        return self._started and not self._draining

    def begin_drain(self) -> None:
        """Mark the process as shutting down: ``/readyz`` fails, admission refuses new work."""
        self._draining = True
        for rt in self.runtimes.values():
            rt.admission.stop_accepting()

    def add_safety_listener(self, fn: Callable[[str, str, str], object]) -> None:
        """Call ``fn(event, model, reason)`` on every safety event of any model (audit, paging)."""
        for rt in self.runtimes.values():
            rt.safety_listeners.append(fn)

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
        """Start all model runtimes independently.

        A model whose compute service is unreachable keeps retrying in the background (state
        ``connecting``); a model with a permanent problem (fingerprint/ring mismatch, bad pack) is
        marked ``failed`` and refuses its requests, without blocking the other models. Only when
        *every* model failed permanently does start-up fail (the process has nothing to serve).
        """
        errors: list[BaseException] = []
        for rt in self.runtimes.values():
            try:
                await rt.start()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("model %s failed to start: %s", rt.name, getattr(exc, "message", type(exc).__name__))
                errors.append(exc)
        if errors and len(errors) == len(self.runtimes):
            raise errors[0]
        self._start_housekeeping()
        self._started = True

    # ---- mask store housekeeping ------------------------------------------------------------------
    def _start_housekeeping(self) -> None:
        store = self.mask_store
        if store is None or not hasattr(store, "gc"):
            return
        claim = getattr(store, "claim_owner", None)
        if callable(claim):
            try:
                claim("gateway")
            except PGWError as exc:
                log.warning("cannot claim mask store ownership: %s", exc.message)
        interval = float(self.settings.maskstore_gc_interval_s)
        if interval <= 0 or self._gc_thread is not None:
            return
        self._gc_stop.clear()
        self._gc_thread = threading.Thread(
            target=self._gc_loop, args=(interval,), name="pgw-maskstore-gc", daemon=True
        )
        self._gc_thread.start()

    def _gc_loop(self, interval: float) -> None:
        """Periodic ``MaskStore.gc()`` (first run right after start-up) in a background thread."""
        while not self._gc_stop.is_set():
            self.run_gc()
            self._gc_stop.wait(interval)

    def run_gc(self) -> int:
        """One GC pass: close dead leases, shred consumed segments, prune old audit rows."""
        try:
            freed = int(self.mask_store.gc())
        except PGWError as exc:
            log.warning("mask store gc failed: %s", exc.message)
            return 0
        except Exception:  # pragma: no cover - never kill the thread
            log.exception("mask store gc failed")
            return 0
        self.gc_runs += 1
        if freed:
            log.info("mask store gc freed %.1f MB", freed / 1e6)
        return freed

    def _stop_housekeeping(self) -> None:
        self._gc_stop.set()
        t, self._gc_thread = self._gc_thread, None
        if t is not None:
            t.join(timeout=max(5.0, float(getattr(self.mask_store.settings, "lock_timeout_s", 60.0))
                               if hasattr(self.mask_store, "settings") else 5.0))

    async def stop(self) -> None:
        """Stop all model runtimes."""
        for rt in self.runtimes.values():
            try:
                await rt.stop()
            except Exception:
                log.exception("stopping model %s failed", rt.name)
        await asyncio.to_thread(self._stop_housekeeping)
        if self._own_store and self.mask_store is not None:
            await asyncio.to_thread(self.mask_store.close)

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
