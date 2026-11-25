"""The in-process gateway behind the client API.

One :class:`Engine` per (SDK key, cache directory, state directory) and process. It owns:

* a private asyncio event loop in a daemon thread (``pgw-sdk-loop``) on which the existing gateway runtime
  (:class:`pgw.gateway.runtime.ModelRuntime`: pack verification, the compute client and trip lane, the
  :class:`pgw.engine.MaskedBackend` with its Freivalds / p31 / range checks, the noise monitor, the integrity
  breaker, admission, incremental mask reservations and the conversation prefix cache) runs unchanged;
* the session (enrollment files), the encrypted mask store shared with the background factory, and one
  :class:`pgw.sdk.factory.FactoryProcess` per model.

Requests are converted with the same code as the gateway's HTTP API (:mod:`pgw.gateway.openai_render`), so a
request gives bit-identical tokens and logprobs through this library and through ``pgw gateway serve`` with the
same model settings and seed.

Nothing here logs prompt or completion content.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pgw.core.errors import (
    AuthError,
    ErrorCode,
    IntegrityError as CoreIntegrityError,
    MaskExhaustedError,
    PGWError,
    ServiceUnavailableError,
)
from pgw.gateway.safety import BreakerOpenError
from pgw.sdk import types as T
from pgw.sdk.cache import CacheLayout, ensure_private_dir, file_lock
from pgw.sdk.errors import (
    APITimeoutError,
    IntegrityError,
    NotFoundError,
    NotReadyError,
    PGWAPIError,
    from_pgw_error,
)
from pgw.sdk.factory import FactoryPlan, FactoryProcess, read_factory_state
from pgw.sdk.keys import SdkKey
from pgw.sdk.options import SDKOptions, canonical_model
from pgw.sdk.provisioning import Provisioned, Provisioner, adapt, default_provisioner
from pgw.sdk.session import Session, ensure_session

__all__ = ["NATIVE_ENV", "Engine"]

log = logging.getLogger(__name__)

R = TypeVar("R")

#: The measured native per-trip stack (deploy/modal/README.md: ``--trip-native --decode-native`` plus the
#: decrypt-ahead and lane knobs of the 28 tok/s run). Exact: same rows, bytes, checks and outputs; the lane knobs
#: are timing only (spin, prefetch hints). Set with ``setdefault``: an explicit environment value wins.
NATIVE_ENV: dict[str, str] = {
    "PGW_TRIP_FAST": "1",
    "PGW_TRIP_NATIVE": "1",
    "PGW_TRIP_DRIVER": "1",
    "PGW_DECODE_NATIVE": "1",
    "PGW_TRIP_DECRYPT_AHEAD": "1",
    "PGW_LANE_SPIN_AUTO_US": "400",
    "PGW_LANE_WAIT_PREFETCH": "1",
}

#: The ring of the proven stack (K32: p31 checks and pks-v3 need it).
PREFERRED_RING = "K32A12C27"
_RETRY_CODES = frozenset({str(ErrorCode.BACKEND_UNAVAILABLE), str(ErrorCode.DEADLINE_EXCEEDED),
                          str(ErrorCode.UNAVAILABLE)})


def apply_native_env() -> None:
    """Turn on the native trip / decode stack for runtimes built from now on (``setdefault``)."""
    for k, v in NATIVE_ENV.items():
        os.environ.setdefault(k, v)
    if os.environ.get("PGW_TRIP_DECRYPT_AHEAD", "") in ("1", "true", "yes", "on"):
        try:  # read by the mask store at import: also correct an import that happened before this call
            from pgw.maskstore import store as _store

            _store.TRIP_DECRYPT_AHEAD = True
        except Exception:  # pragma: no cover - defensive
            pass


@dataclass
class ModelHandle:
    """One model of the engine (public status only)."""

    name: str
    state: str = "new"  # new -> enrolling -> provisioning -> connecting -> ready | failed
    error: str | None = None
    provisioned: Provisioned | None = None
    runtime: Any = None
    factory: FactoryProcess | None = None
    masked_head: bool = False
    ring: str = PREFERRED_RING
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    requests: int = 0
    verified_requests: int = 0
    integrity_failures: int = 0
    last_verified_at: float | None = None
    last_integrity_failure_at: float | None = None
    started_at: float | None = None

    def public(self) -> dict[str, Any]:
        """Status view (no secrets, no content)."""
        out: dict[str, Any] = {"model": self.name, "state": self.state, "error": self.error,
                               "masked_head": self.masked_head, "ring": self.ring, "requests": self.requests,
                               "verified_requests": self.verified_requests,
                               "integrity_failures": self.integrity_failures,
                               "last_integrity_check_ok_at": self.last_verified_at,
                               "last_integrity_failure_at": self.last_integrity_failure_at}
        p = self.provisioned
        if p is not None:
            out["provisioned"] = {"fingerprint": p.fingerprint, "source": p.source, "pack_path": str(p.pack_path),
                                  "hint_dir": str(p.hint_dir) if p.hint_dir else None}
        rt = self.runtime
        if rt is not None:
            out["lane"] = getattr(rt, "lane_state", None)
            out["fingerprint"] = getattr(rt, "fingerprint", None)
            with contextlib.suppress(Exception):
                out["masks_available"] = rt._available_slots() if rt.started else None
            be = getattr(rt, "backend", None)
            if be is not None:
                out["native"] = {"trip_engine": bool(getattr(be, "trip_engine", False)),
                                 "trip_native": bool(getattr(be, "trip_native", False)),
                                 "decode_native": bool(getattr(be, "decode_native", False)),
                                 "prime_check": bool(getattr(be, "prime_check", False))}
                with contextlib.suppress(Exception):
                    out["backend"] = {k: v for k, v in be.stats.snapshot().items()
                                      if isinstance(v, int | float | str | bool)}
            with contextlib.suppress(Exception):
                out["breaker"] = rt.breaker.status()
        if self.factory is not None:
            with contextlib.suppress(Exception):
                out["factory"] = self.factory.state()
        return out


class Engine:
    """The in-process gateway of one SDK key (thread-safe; see the module docstring).

    Args:
        key: The parsed SDK key.
        options: Client options.
        provisioner: Provisioning function (default: :func:`pgw.provision.client.provision`, resolved lazily).
        enroll_transport: httpx transport for enrollment (tests).
    """

    def __init__(self, key: SdkKey, options: SDKOptions, *, provisioner: Provisioner | None = None,
                 enroll_transport: Any = None):
        self.key = key
        self.options = options
        self.layout = CacheLayout(options.cache_dir, options.state_dir)
        self.provisioner = provisioner or default_provisioner()
        self._enroll_transport = enroll_transport
        self.pid = os.getpid()
        self.session: Session | None = None
        self.store: Any = None
        self.models: dict[str, ModelHandle] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._mu = threading.Lock()
        self._closed = False
        self._session_lock: asyncio.Lock | None = None
        self._settings: Any = None
        self._status_lock = threading.Lock()

    # ---- event loop ----------------------------------------------------------------------------
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if os.getpid() != self.pid:
            raise PGWAPIError("this pgw client was created in another process (before a fork); create a new client")
        with self._mu:
            if self._closed:
                raise PGWAPIError("the pgw client is closed")
            if self._loop is None:
                loop = asyncio.new_event_loop()
                started = threading.Event()

                def run() -> None:
                    asyncio.set_event_loop(loop)
                    started.set()
                    loop.run_forever()

                t = threading.Thread(target=run, name="pgw-sdk-loop", daemon=True)
                t.start()
                started.wait()
                self._loop, self._thread = loop, t
            return self._loop

    def submit(self, coro: Coroutine[Any, Any, R]) -> concurrent.futures.Future[R]:
        """Run ``coro`` on the engine loop (from any thread)."""
        try:
            loop = self._ensure_loop()
        except BaseException:
            coro.close()
            raise
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def call(self, coro: Coroutine[Any, Any, R], timeout_s: float | None = None) -> R:
        """Run ``coro`` on the engine loop and wait for its result (blocking)."""
        fut = self.submit(coro)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise APITimeoutError("the request timed out") from exc
        except BaseException:
            if not fut.done():
                fut.cancel()
            raise

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The engine loop (started on first use)."""
        return self._ensure_loop()

    # ---- session, store ------------------------------------------------------------------------
    async def ensure_session(self, *, force: bool = False) -> Session:
        """The enrolled session (enrolls on first use; ``force`` enrolls again)."""
        if self._session_lock is None:
            self._session_lock = asyncio.Lock()
        async with self._session_lock:
            if self.session is None or force:
                self.layout.ensure()
                self.session = await asyncio.to_thread(
                    ensure_session, self.key, self.layout, force=force, transport=self._enroll_transport,
                    timeout_s=min(60.0, self.options.timeout_s))
            return self.session

    def _gateway_settings(self, session: Session) -> Any:
        from urllib.parse import urlsplit

        from pgw.gateway.config import GatewaySettings

        egress: list[str] = []
        u = urlsplit(session.service_url)
        if session.lane and u.scheme == "https" and u.hostname:
            egress.append(f"https://{u.hostname}:{int(session.lane['port'])}")
        return GatewaySettings(
            models=[], models_file=None, maskstore_root=self.layout.masks_dir(self.key.key_id_hex),
            allow_dev_modes=False, allow_anonymous=False, audit_log_file=None,
            noise_monitor=True, noise_monitor_fail_closed=True, egress_allowlist=egress, require_service_tls=True,
            prime_check=None, pack_verify="full", mask_reservation="incremental", mask_chunk_slots=8,
            default_max_tokens=512, max_tokens_limit=32768, maskstore_gc_interval_s=0,
            dashboard_enabled=False, console_enabled=False, setup_mode=False,
            config_dir=self.layout.session_dir(self.key.key_id_hex), data_dir=self.layout.session_dir(self.key.key_id_hex),
        )

    def _open_store(self) -> Any:
        from pgw.maskstore.admin import open_store
        from pgw.maskstore.config import MaskStoreSettings
        from pgw.maskstore.keys import write_key_file

        kid = self.key.key_id_hex
        sdir = ensure_private_dir(self.layout.session_state_dir(kid))
        ensure_private_dir(self.layout.session_dir(kid))
        root = ensure_private_dir(self.layout.masks_dir(kid))
        kek = sdir / "masks.kek"
        with file_lock(sdir / ".kek.lock"):
            if not kek.exists():
                write_key_file(kek)
        settings = MaskStoreSettings(root=root, key_provider="file", key_file=kek, counter="file",
                                     counter_file=sdir / "masks.counter", allow_no_external_counter=False,
                                     low_water_slots=64)
        return open_store(settings)

    # ---- models --------------------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        """Models the provider serves to this key and this release supports."""
        from pgw.sdk.options import SUPPORTED_MODELS

        s = await self.ensure_session()
        names = [str(m.get("name", "")).lower() for m in s.models]
        return [n for n in SUPPORTED_MODELS if n in names]

    async def ensure_model(self, model: str) -> ModelHandle:
        """The model's handle, started (enroll, provision, connect, factory) on first use."""
        name = canonical_model(model)
        h = self.models.get(name)
        if h is None:
            h = self.models[name] = ModelHandle(name)
        if h.state == "ready":
            return h
        async with h.lock:
            if h.state in ("ready", "connecting"):
                return h
            try:
                await asyncio.wait_for(self._start_model(h), timeout=self.options.ready_timeout_s)
            except TimeoutError as exc:
                h.state, h.error = "failed", "start-up timed out"
                raise NotReadyError(f"model {name!r} did not become ready within {self.options.ready_timeout_s:.0f}s"
                                    ) from exc
            except BaseException as exc:
                h.state = "failed"
                h.error = getattr(exc, "message", None) or type(exc).__name__
                self._write_status()
                if isinstance(exc, PGWAPIError):
                    raise
                if isinstance(exc, PGWError):
                    raise from_pgw_error(exc) from exc
                raise
        return h

    async def _start_model(self, h: ModelHandle) -> None:
        from pgw.gateway.metrics import GatewayMetrics
        from pgw.gateway.runtime import ModelRuntime

        old = h.runtime
        if old is not None and not getattr(old, "started", False):  # a failed earlier attempt
            h.runtime = None
            with contextlib.suppress(Exception):
                await old.stop()
        h.state = "enrolling"
        session = await self.ensure_session()
        entry = session.model_entry(h.name)
        if entry is None:
            session = await self.ensure_session(force=True)  # the provider may serve it since enrollment
            entry = session.model_entry(h.name)
        if entry is None:
            raise NotFoundError(f"the provider does not serve model {h.name!r} to this API key",
                                details={"model": h.name})
        h.state = "provisioning"
        self.layout.ensure()
        ensure_private_dir(self.layout.models_dir)
        raw = await asyncio.to_thread(self.provisioner, h.name, session=session, cache_dir=self.layout.models_dir,
                                      progress=lambda msg: log.info("provisioning %s: %s", h.name, msg))
        prov = adapt(raw, model=h.name)
        if prov.fingerprint != str(entry["fingerprint"]).lower():
            raise IntegrityError(f"provisioned artifacts of {h.name!r} do not match the provider's fingerprint",
                                 details={"model": h.name})
        h.provisioned = prov
        if self.store is None:
            self.store = await asyncio.to_thread(self._open_store)
        if self._settings is None:
            self._settings = self._gateway_settings(session)
        h.ring = self._pick_ring(entry)
        h.masked_head = await self._decide_masked_head(session, prov.fingerprint)
        if self.options.native:
            apply_native_env()
        cfg = self._model_config(h, session, prov)
        self._settings.validate_policy([cfg])
        h.state = "connecting"
        for attempt in (0, 1):
            rt = ModelRuntime(cfg, self._settings, mask_store=self.store, metrics=GatewayMetrics(enabled=False))
            try:
                await rt.start()
            except AuthError:
                if attempt:
                    raise
                # the provider no longer accepts the service token (rotated / revoked): enroll again once
                session = await self.ensure_session(force=True)
                cfg = self._model_config(h, session, prov)
                continue
            break
        h.runtime = rt
        h.started_at = time.time()
        if rt.started:
            self._on_ready(h)
        else:  # connecting in the background (the runtime retries): requests wait for it
            asyncio.get_running_loop().create_task(self._await_ready(h))

    async def _await_ready(self, h: ModelHandle) -> None:
        rt = h.runtime
        while rt is not None and not rt.started and rt.state == "connecting":
            await asyncio.sleep(0.5)
        if rt is not None and rt.started:
            self._on_ready(h)
        elif rt is not None:
            h.state, h.error = "failed", rt.last_error

    def _on_ready(self, h: ModelHandle) -> None:
        h.state = "ready"
        h.error = None
        h.ready_event.set()
        if self.options.factory and h.factory is None:
            try:
                h.factory = FactoryProcess(self._factory_plan(h)).start()
            except Exception as exc:  # the stock still serves; status shows it
                log.error("mask factory for %s not started: %s", h.name, getattr(exc, "message", type(exc).__name__))
        self._write_status()

    @staticmethod
    def _pick_ring(entry: dict[str, Any]) -> str:
        rings = [str(r) for r in (entry.get("rings") or [])]
        if entry.get("ring"):
            rings.append(str(entry["ring"]))
        if PREFERRED_RING in rings or not rings:
            return PREFERRED_RING
        k32 = [r for r in rings if r.upper().startswith("K32")]
        return k32[0] if k32 else rings[0]

    async def _decide_masked_head(self, session: Session, fp: str) -> bool:
        mode = self.options.masked_head
        if mode == "off" or self.options.prefix_cache is True:
            return False
        from pgw.modelpack.masked_head import head_fingerprint

        served = await self._served_fingerprints(session)
        ok = head_fingerprint(fp) in served
        if mode == "on" and not ok:
            raise NotFoundError("masked_head='on' but the provider does not serve the masked LM head of this model")
        return ok

    async def _served_fingerprints(self, session: Session) -> set[str]:
        from pgw.wire.client import ClientConfig, ComputeClient

        cc = ClientConfig(base_url=session.service_url, token=session.token(), ca_file=session.ca_file,
                          client_cert_file=session.client_cert_file, client_key_file=session.client_key_file,
                          request_timeout_s=min(60.0, self.options.timeout_s), max_retries=2,
                          user_agent="pgw-sdk/1")
        client = ComputeClient(cc)
        try:
            models = await client.models()
        except PGWError as exc:
            log.warning("cannot list the provider's models (%s); masked LM head off", exc.message)
            return set()
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()
        return {str(m.fingerprint) for m in getattr(models, "models", ())}

    def _model_config(self, h: ModelHandle, session: Session, prov: Provisioned) -> Any:
        from pgw.gateway.config import GatewayModelConfig

        o = self.options
        https = session.service_url.startswith("https://")
        if o.lane == "off":
            lane = "off"
        elif o.lane == "auto" and session.lane is not None and https and session.ca_file is not None:
            lane = "tls"
        else:
            lane = "stream"
        kw: dict[str, Any] = {
            "name": h.name, "pack_path": prov.pack_path, "fingerprint": prov.fingerprint, "mode": "masked",
            "ring": h.ring, "service_url": session.service_url, "service_token_file": session.token_file,
            "service_ca_file": session.ca_file, "client_cert_file": session.client_cert_file,
            "client_key_file": session.client_key_file, "service_timeout_s": min(120.0, o.timeout_s),
            "service_max_retries": 3, "freivalds_rate": 1.0, "max_concurrent_sessions": o.max_concurrent,
            "max_queue": 256, "queue_timeout_s": o.timeout_s, "device": "cpu", "lane": lane,
            "local_linears": "off", "float_ops": "native",
            "head_mode": "provider_int8" if h.masked_head else "gateway",
            "prefix_cache": "conversation" if self._prefix_on(h) else "off",
        }
        return GatewayModelConfig(**kw)

    def _prefix_on(self, h: ModelHandle) -> bool:
        """Conversation prefix reuse: explicit ``True``, or ``"auto"`` without the masked LM head (the engine
        proves reuse exact only with the gateway's own head)."""
        p = self.options.prefix_cache
        return p is True or (p == "auto" and not h.masked_head)

    def _factory_plan(self, h: ModelHandle) -> FactoryPlan:
        o = self.options
        assert h.provisioned is not None and self.session is not None
        rt = h.runtime
        kid = self.key.key_id_hex
        sdir = self.layout.session_state_dir(kid)
        prime = bool(getattr(getattr(rt, "backend", None), "prime_check", False))
        slot_bytes = self._slot_bytes(rt)
        target = o.mask_target_slots
        if target is None:
            target = int(max(64, min(16384, (o.mask_store_gb * 1e9) // max(1, slot_bytes))))
        low = o.mask_low_water_slots if o.mask_low_water_slots is not None else max(1, target // 2)
        low = min(low, target)
        seg = max(16, min(o.mask_segment_slots, target))
        scheme = o.mask_scheme
        hint_key = None
        if scheme == "pks-v3":
            hint_key = h.provisioned.hint_key_file or (sdir / "pks-hint.key")
        s = self.session
        return FactoryPlan(
            model=h.name, pack_path=h.provisioned.pack_path, ring=h.ring,
            state_dir=self.layout.factory_dir(kid) / h.provisioned.fingerprint[:16],
            store_root=self.layout.masks_dir(kid), kek_file=sdir / "masks.kek", counter_file=sdir / "masks.counter",
            scheme=scheme, check_field="p31" if prime else "ring", masked_head=h.masked_head,
            workers=o.workers, threads=max(1, o.mask_threads), nice=o.mask_nice, target_slots=target,
            low_water_slots=low, segment_slots=seg, first_segment_slots=max(16, min(64, seg)),
            service_url=s.service_url, token_file=s.token_file, ca_file=s.ca_file,
            client_cert_file=s.client_cert_file, client_key_file=s.client_key_file,
            hint_dir=h.provisioned.hint_dir or (self.layout.models_dir / h.provisioned.fingerprint[:16] / "hint"),
            hint_key_file=hint_key)

    @staticmethod
    def _slot_bytes(rt: Any) -> int:
        try:
            specs = rt.pack.linear_specs()
            return int(sum(int(s.out_features) for s in specs) * rt.ring.k // 8)
        except Exception:  # pragma: no cover - defensive
            return 4 << 20

    # ---- requests --------------------------------------------------------------------------------
    async def _ready_runtime(self, model: str, deadline: float) -> tuple[ModelHandle, Any]:
        h = await self.ensure_model(model)
        if h.state != "ready":
            left = deadline - time.monotonic()
            try:
                await asyncio.wait_for(h.ready_event.wait(), timeout=max(0.0, left))
            except TimeoutError as exc:
                raise NotReadyError(f"model {h.name!r} is not ready yet ({h.runtime.state if h.runtime else h.state}"
                                    f"{': ' + h.error if h.error else ''})") from exc
        return h, h.runtime

    async def reset_breaker(self, model: str) -> None:
        """Re-open a model blocked after an integrity failure (an operator decision, after investigating)."""
        h = self.models.get(canonical_model(model))
        if h is not None and h.runtime is not None:
            h.runtime.reset_safety(breaker=True, noise=True)

    async def _open(self, h: ModelHandle, rt: Any, ids: list[int], params: Any, rid: str, prefix_key: str | None,
                    deadline: float) -> Any:
        """Admit and reserve masks; waits (back-pressure) while the mask stock is empty or the service reconnects."""
        mask_deadline = min(deadline, time.monotonic() + self.options.mask_wait_s)
        delay = 0.1
        while True:
            try:
                return (await rt.open_sessions([ids], params, request_id=rid, prefix_key=prefix_key))[0]
            except MaskExhaustedError:
                if time.monotonic() + delay > mask_deadline:
                    raise
            except BreakerOpenError as exc:  # blocked after an integrity failure: hard error, never waited out
                raise IntegrityError(f"model {h.name!r} is blocked after an integrity failure of the provider "
                                     f"({exc.message}); call reset_integrity_breaker() after investigating",
                                     request_id=rid) from exc
            except ServiceUnavailableError as exc:
                retry = exc.details.get("retry_after_s") is not None or rt.state == "connecting"
                if not retry or time.monotonic() + delay > deadline:
                    raise
            await asyncio.sleep(delay)
            delay = min(1.0, delay * 1.5)

    async def chat_events(self, req: Any, *, conversation: str | None = None,
                          meta: dict[str, Any] | None = None) -> AsyncIterator[tuple[Any, Any]]:
        """Run one chat request; yields ``(ModelHandle, TokenEvent)`` (and fills ``meta``: id, usage, ...).

        Transport failures and temporary unavailability before the first token are retried
        (``max_retries``); integrity failures never are.
        """
        from pgw.gateway.openai_render import chat_messages, sampling_params

        meta = meta if meta is not None else {}
        t0 = time.monotonic()
        deadline = t0 + self.options.timeout_s
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        meta["id"] = rid
        h, rt = await self._ready_runtime(req.model, deadline + self.options.ready_timeout_s)
        meta["model"] = h.name
        meta["system_fingerprint"] = rt.fingerprint[:12] if rt.fingerprint else None
        msgs = chat_messages(req)
        ids = await asyncio.to_thread(rt.encode_chat, msgs)
        max_tokens = rt.clamp_max_tokens(len(ids), req.requested_max_tokens)
        lp = (req.top_logprobs or 0) if req.logprobs else None
        params = sampling_params(req, max_tokens, lp)
        meta["params"] = params
        prefix_key = None
        if rt.prefix_cache is not None:
            prefix_key = f"sdk\x00user:{req.user or ''}\x00conv:{conversation or ''}"
        attempt = 0
        while True:
            produced = 0
            sess = None
            try:
                sess = await self._open(h, rt, ids, params, rid if attempt == 0 else f"{rid}-r{attempt}",
                                        prefix_key, deadline)
                h.requests += 1
                meta["prompt_tokens"] = len(sess.prompt_ids)
                meta["cached_tokens"] = int(sess.prefix_reused)
                async for ev in sess.stream():
                    produced += 1
                    meta["completion_tokens"] = produced
                    yield h, ev
                h.verified_requests += 1
                h.last_verified_at = time.time()
                self._write_status_soon()
                return
            except (CoreIntegrityError, IntegrityError) as exc:
                h.integrity_failures += 1
                h.last_integrity_failure_at = time.time()
                self._write_status()
                if isinstance(exc, IntegrityError):
                    raise
                raise from_pgw_error(exc, request_id=rid) from exc
            except PGWError as exc:
                if (produced == 0 and str(exc.code) in _RETRY_CODES and attempt < self.options.max_retries
                        and time.monotonic() < deadline and not isinstance(exc, MaskExhaustedError)):
                    attempt += 1
                    log.info("request %s: retrying after %s (attempt %d)", rid, exc.code, attempt)
                    await asyncio.sleep(min(2.0, 0.25 * 2 ** attempt))
                    continue
                raise from_pgw_error(exc, request_id=rid) from exc
            finally:
                if sess is not None:
                    await sess.close()

    async def chat(self, req: Any, *, conversation: str | None = None) -> T.ChatCompletion:
        """Non-streaming chat completion."""
        from pgw.gateway.openai_render import TokenText, chat_logprobs, stop_ids, visible

        meta: dict[str, Any] = {}
        parts: list[str] = []
        events: list[Any] = []
        finish: str | None = None
        rt = None
        stops: frozenset[int] = frozenset()
        async with asyncio.timeout(self.options.timeout_s + self.options.ready_timeout_s):
            async for h, ev in self.chat_events(req, conversation=conversation, meta=meta):
                if rt is None:
                    rt = h.runtime
                    stops = stop_ids(rt, meta["params"])
                parts.append(ev.text)
                if visible(ev, stops):
                    events.append(ev)
                if ev.finish_reason is not None:
                    finish = str(ev.finish_reason)
        logprobs = None
        if rt is not None and any(ev.logprob is not None for ev in events):
            logprobs = chat_logprobs(events, TokenText(rt))
        p, c = int(meta.get("prompt_tokens", 0)), int(meta.get("completion_tokens", 0))
        usage: dict[str, Any] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
        if rt is not None and rt.prefix_cache is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": int(meta.get("cached_tokens", 0))}
        return T.ChatCompletion.model_validate({
            "id": meta["id"], "created": int(time.time()), "model": meta.get("model", req.model),
            "system_fingerprint": meta.get("system_fingerprint"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                         "logprobs": logprobs, "finish_reason": finish}],
            "usage": usage})

    async def chat_stream(self, req: Any, *, conversation: str | None = None) -> AsyncIterator[T.ChatCompletionChunk]:
        """Streaming chat completion (chunks as the gateway's SSE stream sends them)."""
        from pgw.gateway.openai_render import TokenText, chat_logprobs, stop_ids, visible

        meta: dict[str, Any] = {}
        created = int(time.time())
        include_usage = req.include_usage
        tt = None
        stops: frozenset[int] = frozenset()
        first = True
        prefix_cache = False

        def chunk(choices: list[dict[str, Any]], **extra: Any) -> T.ChatCompletionChunk:
            body = {"id": meta["id"], "created": created, "model": meta.get("model", req.model),
                    "system_fingerprint": meta.get("system_fingerprint"), "choices": choices}
            body.update(extra)
            return T.ChatCompletionChunk.model_validate(body)

        async for h, ev in self.chat_events(req, conversation=conversation, meta=meta):
            if first:
                first = False
                tt = TokenText(h.runtime)
                stops = stop_ids(h.runtime, meta["params"])
                prefix_cache = h.runtime.prefix_cache is not None
                yield chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None,
                              "finish_reason": None}])
            vis = visible(ev, stops)
            if not ev.text and ev.finish_reason is None and not (vis and ev.logprob is not None):
                continue
            delta: dict[str, Any] = {"content": ev.text} if ev.text else {}
            lp = chat_logprobs([ev], tt) if (vis and ev.logprob is not None and tt is not None) else None
            yield chunk([{"index": 0, "delta": delta, "logprobs": lp,
                          "finish_reason": str(ev.finish_reason) if ev.finish_reason else None}])
        if include_usage and "id" in meta:
            p, c = int(meta.get("prompt_tokens", 0)), int(meta.get("completion_tokens", 0))
            usage: dict[str, Any] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
            if prefix_cache:
                usage["prompt_tokens_details"] = {"cached_tokens": int(meta.get("cached_tokens", 0))}
            yield chunk([], usage=usage)

    # ---- status ------------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Public status: session, models (state, lane, native stack, masks, factory, integrity), cache paths."""
        out: dict[str, Any] = {"key_id": self.key.key_id_hex, "service_url": self.key.service_url,
                               "cache_dir": str(self.layout.root), "state_dir": str(self.layout.state),
                               "mask_scheme": self.options.mask_scheme, "pid": self.pid,
                               "session": self.session.public() if self.session else None,
                               "models": {n: h.public() for n, h in list(self.models.items())}}
        return out

    def _status_path(self) -> Path:
        return self.layout.session_dir(self.key.key_id_hex) / "status.json"

    def _write_status_soon(self) -> None:
        now = time.monotonic()
        if now - getattr(self, "_status_t", 0.0) >= 2.0:
            self._status_t = now
            self._write_status()

    def _write_status(self) -> None:
        """Merge this process's model states into ``<session>/status.json`` (for ``pgw sdk status``)."""
        try:
            path = self._status_path()
            if not path.parent.exists():
                return
            with self._status_lock, file_lock(path.with_suffix(".lock"), timeout_s=2.0):
                try:
                    cur = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    cur = {}
                models = cur.get("models") if isinstance(cur.get("models"), dict) else {}
                for n, h in self.models.items():
                    pub = h.public()
                    pub.pop("backend", None)
                    pub["pid"] = self.pid
                    pub["updated_at"] = time.time()
                    models[n] = pub
                cur.update({"key_id": self.key.key_id_hex, "service_url": self.key.service_url, "models": models,
                            "updated_at": time.time()})
                from pgw.core.fileio import atomic_write_text

                atomic_write_text(path, json.dumps(cur, indent=2, default=str) + "\n", mode=0o600)
        except Exception:  # status is best effort
            log.debug("status file not written", exc_info=True)

    # ---- shutdown ----------------------------------------------------------------------------------
    def close(self, timeout_s: float = 30.0) -> None:
        """Stop the models, the factories and the loop; close the mask store (idempotent)."""
        with self._mu:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if os.getpid() != self.pid:
            return
        for h in list(self.models.values()):
            if h.factory is not None:
                with contextlib.suppress(Exception):
                    h.factory.stop()
        if loop is not None and thread is not None and thread.is_alive():
            async def _stop() -> None:
                for h in list(self.models.values()):
                    if h.runtime is not None:
                        with contextlib.suppress(Exception):
                            await h.runtime.stop()

            try:
                asyncio.run_coroutine_threadsafe(_stop(), loop).result(timeout=timeout_s)
            except Exception:
                log.warning("stopping the models did not finish cleanly")
            self._write_status()
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)
        if self.store is not None:
            with contextlib.suppress(Exception):
                self.store.close()
            self.store = None
        if loop is not None and not loop.is_running():
            with contextlib.suppress(Exception):
                loop.close()


def read_status(layout: CacheLayout) -> dict[str, Any]:
    """Every session's ``status.json`` and factory state under ``layout`` (for ``pgw sdk status``; no network)."""
    out: dict[str, Any] = {"cache_dir": str(layout.root), "state_dir": str(layout.state), "sessions": []}
    root = layout.root / "sessions"
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        entry: dict[str, Any] = {"key_id": d.name}
        with contextlib.suppress(OSError, ValueError):
            meta = json.loads((d / "enrollment.json").read_text(encoding="utf-8"))
            entry["enrollment"] = {k: meta.get(k) for k in ("service_url", "tenant", "enrolled_at", "ca_fingerprint",
                                                            "client_cert", "lane")}
            entry["enrollment"]["models"] = [m.get("name") for m in meta.get("models") or []]
        with contextlib.suppress(OSError, ValueError):
            entry["status"] = json.loads((d / "status.json").read_text(encoding="utf-8"))
        fdir = d / "factory"
        if fdir.is_dir():
            entry["factories"] = {f.name: read_factory_state(f) for f in sorted(fdir.iterdir()) if f.is_dir()}
        out["sessions"].append(entry)
    return out
