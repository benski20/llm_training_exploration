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
import shutil
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
    MaskExhaustedError,
    PGWError,
    ServiceUnavailableError,
)
from pgw.core.errors import (
    IntegrityError as CoreIntegrityError,
)
from pgw.gateway.safety import BreakerOpenError
from pgw.sdk import types as T
from pgw.sdk.cache import CacheLayout, ensure_private_dir, file_lock
from pgw.sdk.errors import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    IntegrityError,
    InternalServerError,
    NotFoundError,
    NotReadyError,
    PGWAPIError,
    ProviderStalledError,
    from_pgw_error,
)
from pgw.sdk.factory import (
    FactoryPlan,
    FactoryProcess,
    describe_setup,
    provider_error,
    read_factory_state,
    write_demand,
)
from pgw.sdk.halt import HaltRecord, HaltStore
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
    # the next trip lease of each linear prepared on the lease-ahead thread (rows the session already holds; attached
    # only at its first row, else zeroed): tests/engine/test_lease_ahead.py, tests/sdk/test_concurrency_cpu.py
    "PGW_TRIP_LEASE_AHEAD": "1",
    "PGW_LANE_SPIN_AUTO_US": "400",
    "PGW_LANE_WAIT_PREFETCH": "1",
    # several answers in one batched trip: one lane-format-2 multi call when the service offers it (same items,
    # bytes and request id as the stream channel's request; tests/sdk/test_lane_multi.py)
    "PGW_LANE_MULTI": "1",
    # concurrent answers (docs/sdk/scaling.md): a decode step of several answers runs as one synchronous pass on
    # the plaintext thread when one multi call serves it (same calls and requests); its trips take one native
    # lease per answer (same rows, bytes, checks and outputs); the answers sample on a thread pool (each with its
    # own generator: the same tokens, logprobs and generator states). tests/sdk/test_concurrency_cpu.py
    "PGW_SYNC_MULTI": "1",
    "PGW_TRIP_NATIVE_MULTI": "1",
    # a decode step of several answers runs its decoder layers in the native decode driver (one native call per
    # stretch of trips, one lane multi call per trip; same requests, rows, bytes, checks and outputs):
    # tests/engine/test_decode_native_multi.py, tests/sdk/test_concurrency_cpu.py
    "PGW_DECODE_NATIVE_MULTI": "1",
    "PGW_SAMPLE_THREADS": "auto",
    # schedule="fused": the wide element-wise stages of a one-row fused pass (unmask of the second stage, its check,
    # the gate/up and output-layer scaling, SiLU) in fixed element ranges on the native pool (the same bits;
    # 32B widths on 4 vCPUs: 0.95-1.10 -> 0.75 ms per pair trip, benchmarks/results/fused2/
    # driver_cpu_32b_split_2026-11-27.json). Read by the fused2 driver only: the standard schedule is unaffected
    "PGW_FUSED2_SPLIT": "4" if (os.cpu_count() or 1) >= 4 else "2",
}

#: The ring of the proven stack (K32: p31 checks and pks-v3 need it).
PREFERRED_RING = "K32A12C27"
_RETRY_CODES = frozenset({str(ErrorCode.BACKEND_UNAVAILABLE), str(ErrorCode.DEADLINE_EXCEEDED),
                          str(ErrorCode.UNAVAILABLE)})
_AUTH_CODES = frozenset({str(ErrorCode.UNAUTHENTICATED), str(ErrorCode.PERMISSION_DENIED)})
_REFUSED = "the provider refused the session's credentials"
#: How often the client certificate is re-checked (renewal due, or replaced by another process).
CERT_CHECK_S = 600.0


def _accepts(fn: Any, name: str) -> bool:
    """Whether callable ``fn`` takes keyword argument ``name``."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def apply_native_env(on: bool = True) -> None:
    """Choose the native trip / decode stack for runtimes built from now on (``setdefault``: process-wide, so the
    first client's choice, or an explicit environment value, applies to every client of the process)."""
    for k, v in NATIVE_ENV.items():
        os.environ.setdefault(k, v if on else "0")
    if not on:
        return
    for name in ("PGW_TRIP_DECRYPT_AHEAD", "PGW_TRIP_LEASE_AHEAD"):
        if os.environ.get(name, "") in ("1", "true", "yes", "on"):
            try:  # read by the mask store at import: also correct an import that happened before this call
                from pgw.maskstore import store as _store

                setattr(_store, name[4:], True)
            except Exception:  # pragma: no cover - defensive
                log.debug("mask store flag %s not set", name, exc_info=True)


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
    #: the linear schedule of this model for the life of the client (``standard`` / ``fused``; pgw.sdk.schedule):
    #: decided once, at the first start, never changed afterwards (a reconnect keeps it)
    schedule: str = "standard"
    schedule_decision: Any = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    start_task: asyncio.Task[None] | None = None
    #: a mask reservation succeeded in this process (the first fill is over)
    masks_seen: bool = False
    #: last provisioning progress line (no secrets: stage and byte counts)
    progress: str | None = None
    requests: int = 0
    verified_requests: int = 0
    integrity_failures: int = 0
    last_verified_at: float | None = None
    last_integrity_failure_at: float | None = None
    started_at: float | None = None
    #: requests waiting for masks, in arrival order (the head polls the stock; see ``Engine._open``)
    mask_waiters: list[asyncio.Event] = field(default_factory=list)
    #: requests that failed before reaching the provider, by error code (status only; never reported, S12)
    local_errors: dict[str, int] = field(default_factory=dict)
    #: the persisted integrity halt of this (key, model) (:mod:`pgw.sdk.halt`), as last read
    halt: HaltRecord | None = None
    #: why the next request reconnects the model first (credentials refused, client certificate replaced)
    reconnect: str | None = None
    last_reconnect_at: float = 0.0

    def public(self) -> dict[str, Any]:
        """Status view (no secrets, no content)."""
        out: dict[str, Any] = {"model": self.name, "state": self.state, "error": self.error, "progress": self.progress,
                               "masked_head": self.masked_head, "ring": self.ring, "schedule": self.schedule,
                               "schedule_decision": (self.schedule_decision.public()
                                                     if self.schedule_decision is not None else None),
                               "requests": self.requests,
                               "verified_requests": self.verified_requests,
                               "integrity_failures": self.integrity_failures,
                               "requests_waiting_for_masks": len(self.mask_waiters),
                               "last_integrity_check_ok_at": self.last_verified_at,
                               "last_integrity_failure_at": self.last_integrity_failure_at}
        if self.local_errors:
            out["local_errors"] = dict(self.local_errors)
        if self.halt is not None:
            out["integrity_halt"] = self.halt.public()
        p = self.provisioned
        if p is not None:
            out["provisioned"] = {"fingerprint": p.fingerprint, "source": p.source, "pack_path": str(p.pack_path),
                                  "hint_dir": str(p.hint_dir) if p.hint_dir else None}
        rt = self.runtime
        if rt is not None:
            out["lane"] = getattr(rt, "lane_state", None)
            out["fingerprint"] = getattr(rt, "fingerprint", None)
            pc = getattr(rt, "prefix_cache", None)
            if pc is not None:  # counts and sizes only (no keys, no tokens)
                out["prefix_cache"] = pc.status()
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
            if getattr(rt, "fused2", False):  # schedule = "fused": public configuration and counters only
                with contextlib.suppress(Exception):
                    out["fused2"] = {k: v for k, v in rt._fused2_status().items()
                                     if isinstance(v, int | float | str | bool | type(None))}
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
        self.provisioner = provisioner or default_provisioner(options)
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
        self._status_task: asyncio.Task[None] | None = None
        self._gc_stop = threading.Event()
        self._gc_thread: threading.Thread | None = None
        #: requests that failed before reaching the provider, by error code (never reported, S12)
        self.local_errors: dict[str, int] = {}
        #: persisted integrity halts of this key (S11): a restart does not re-open a halted model
        self.halts = HaltStore(self.layout.session_state_dir(key.key_id_hex), key.key_id_hex)
        #: request coroutines running on the loop (``submit``): cancelled by ``close()``, so no caller waits forever
        self._tasks: set[asyncio.Task[Any]] = set()
        self._cert_task: asyncio.Task[None] | None = None
        #: cached packs deleted and provisioned again after a failed verification (never twice per process)
        self._repaired: set[str] = set()
        #: content-free request reports for the provider's console (``usage_reports``)
        self.reporter: Any = None
        if options.usage_reports:
            from pgw.sdk.reports import Reporter

            self.reporter = Reporter(lambda: self.session)

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
        """Run ``coro`` on the engine loop (from any thread). ``close()`` cancels it if it is still running."""
        try:
            loop = self._ensure_loop()
        except BaseException:
            coro.close()
            raise
        return asyncio.run_coroutine_threadsafe(self._tracked(coro), loop)

    async def _tracked(self, coro: Coroutine[Any, Any, R]) -> R:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            return await coro
        finally:
            if task is not None:
                self._tasks.discard(task)

    def closed_error(self, request_id: str | None = None) -> PGWAPIError:
        """The error of a request stopped because its client was closed (``close()``, interpreter exit)."""
        return APIConnectionError("the request was stopped because the pgw client was closed", request_id=request_id)

    def call(self, coro: Coroutine[Any, Any, R], timeout_s: float | None = None) -> R:
        """Run ``coro`` on the engine loop and wait for its result (blocking)."""
        fut = self.submit(coro)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise APITimeoutError("the request timed out") from exc
        except concurrent.futures.CancelledError:
            raise self.closed_error() from None
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
            prime_check=self._prime_check(), pack_verify="full", mask_reservation="incremental", mask_chunk_slots=8,
            # processes with the same key and cache directory share the mask store: admission counts every
            # process's promised masks, so a running answer's extensions never find its stock taken by another
            mask_ledger_dir=self.layout.session_dir(self.key.key_id_hex) / "mask-ledger",
            mask_admission=self.options.mask_admission,
            drain_timeout_s=2.0,  # close() cancels the requests still running first
            default_max_tokens=512, max_tokens_limit=32768, maskstore_gc_interval_s=0,
            dashboard_enabled=False, console_enabled=False, setup_mode=False,
            config_dir=self.layout.session_dir(self.key.key_id_hex),
            data_dir=self.layout.session_dir(self.key.key_id_hex),
        )

    def _prime_check(self) -> bool | None:
        """``None``: auto (p31 with the TripEngine). Strict speculation with masked filler rows makes no padded call,
        so the p31 check stays valid there, but the gateway's auto rule refuses p31 for any fixed-row mode: it is
        asked for explicitly, with the native trip stack only (which runs the prime check)."""
        if self.options.speculation != "off" and self.options.native:
            return True
        return None

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

    def _store_error(self, exc: PGWError) -> PGWAPIError:
        """The mask store cannot be opened (its database, key or rollback counter is damaged or was changed outside
        the library). Never repaired automatically: the counter is what stops an old copy of the store from serving
        used masks again. The message says how to start over."""
        kid = self.key.key_id_hex
        err = from_pgw_error(exc)
        if _disk_full(exc.message):
            return InternalServerError(f"the mask store of this API key cannot be opened: the disk holding "
                                       f"{self.layout.root} is full ({exc.message}). Free disk space there and retry",
                                       code=err.code)
        return type(err)(
            f"the mask store of this API key cannot be opened safely ({exc.message}). Nothing was sent. If its files "
            "were damaged (a full disk, a crash, a partial copy of the cache), stop every process using this cache "
            f"and delete {self.layout.masks_dir(kid)} together with masks.kek and masks.counter in "
            f"{self.layout.session_state_dir(kid)} to start over (the stored masks are discarded; enrollment and "
            "model files are kept)", code=err.code)

    def _start_gc(self, interval_s: float = 300.0) -> None:
        """Mask store housekeeping (shred consumed segments, close dead leases) in a daemon thread, at start and
        every ``interval_s``. Every process sharing the store may run it (the store serializes it)."""
        if self._gc_thread is not None:
            return

        def loop() -> None:
            while not self._gc_stop.is_set():
                store = self.store
                if store is None:
                    return
                try:
                    store.gc()
                except Exception as exc:  # never kill the thread
                    log.warning("mask store gc failed: %s", getattr(exc, "message", type(exc).__name__))
                self._gc_stop.wait(interval_s)

        self._gc_thread = threading.Thread(target=loop, name="pgw-sdk-gc", daemon=True)
        self._gc_thread.start()

    # ---- models --------------------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        """Models the provider serves to this key and this release supports."""
        from pgw.sdk.options import PREVIEW_ENV, SUPPORTED_MODELS, preview_models

        s = await self.ensure_session()
        names = [str(m.get("name", "")).lower() for m in s.models]
        preview = preview_models() if os.environ.get(PREVIEW_ENV, "").strip() else ()
        return [n for n in (*SUPPORTED_MODELS, *preview) if n in names]

    async def ensure_model(self, model: str) -> ModelHandle:
        """The model's handle, started (enroll, provision, connect, factory) on first use.

        The start-up runs as its own task: a caller that gives up (a short per-request ``timeout``) does not cancel
        a provisioning that is under way; the next request waits for the same start-up."""
        name = canonical_model(model)
        h = self.models.get(name)
        if h is None:
            h = self.models[name] = ModelHandle(name)
        if h.state == "ready":
            if h.reconnect is None:
                return h
            t = h.start_task
            if t is None or t.done():
                t = h.start_task = asyncio.get_running_loop().create_task(self._reconnect(h))
            await asyncio.shield(t)
            return h
        t = h.start_task
        if t is None or t.done():
            if (t is not None and not t.cancelled() and t.exception() is None
                    and h.state in ("ready", "connecting")):
                return h
            t = h.start_task = asyncio.get_running_loop().create_task(self._start_guarded(h))
        await asyncio.shield(t)
        return h

    async def _reconnect(self, h: ModelHandle) -> None:
        """Stop the model's runtime and start it again (enrolling again if the provider refuses the session): after
        the provider refused this installation's credentials, or after a re-enrollment replaced them."""
        async with h.lock:
            reason, h.reconnect = h.reconnect, None
            if reason is None or h.state != "ready":
                return
            log.warning("model %s: reconnecting to the provider (%s)", h.name, reason)
            h.last_reconnect_at = time.monotonic()
            old, fac = h.runtime, h.factory
            h.state, h.runtime, h.factory = "new", None, None
            h.ready_event.clear()
            if old is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(old.stop(), timeout=30.0)
            if fac is not None:  # restarted with the session's new files once the model is ready again
                await asyncio.to_thread(fac.stop)
            if reason.startswith(_REFUSED):
                try:  # the provider refused the session itself: a new one (a revoked key fails here, clearly)
                    await self.ensure_session(force=True)
                except BaseException as exc:
                    h.state = "failed"
                    h.error = getattr(exc, "message", None) or type(exc).__name__
                    raise
        await self._start_guarded(h)

    def _note_auth_failure(self, h: ModelHandle, exc: PGWError) -> None:
        """The provider refused the session's credentials during a request: the next request reconnects (and enrolls
        again if needed; a revoked key then fails with a clear AuthenticationError). At most once per minute."""
        if h.state == "ready" and h.reconnect is None and time.monotonic() - h.last_reconnect_at > 60.0:
            h.reconnect = f"{_REFUSED} ({exc.message})"

    async def _start_guarded(self, h: ModelHandle) -> None:
        async with h.lock:
            if h.state in ("ready", "connecting"):
                return
            try:
                await asyncio.wait_for(self._start_model(h), timeout=self.options.ready_timeout_s)
            except TimeoutError as exc:
                h.state, h.error = "failed", "start-up timed out"
                raise NotReadyError(f"model {h.name!r} did not become ready within "
                                    f"{self.options.ready_timeout_s:.0f}s") from exc
            except BaseException as exc:
                h.state = "failed"
                h.error = getattr(exc, "message", None) or type(exc).__name__
                self._write_status()
                if isinstance(exc, PGWAPIError):
                    raise
                if isinstance(exc, PGWError):
                    raise from_pgw_error(exc) from exc
                raise

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
        # the masked LM head is decided first: a provisioned hint store covers the head linear only when asked
        h.masked_head = await self._decide_masked_head(session, str(entry["fingerprint"]).lower())
        self._write_status()

        def progress(msg: str) -> None:
            h.progress = str(msg)[:200]
            log.info("provisioning %s: %s", h.name, msg)
            self._write_status_soon()

        pkw: dict[str, Any] = {"session": session, "cache_dir": self.layout.models_dir, "progress": progress}
        if _accepts(self.provisioner, "masked_head"):
            pkw["masked_head"] = h.masked_head
        try:
            raw = await asyncio.to_thread(self.provisioner, h.name, **pkw)
        except Exception as exc:
            if not isinstance(exc, AuthError) and str(getattr(exc, "code", "")) not in _AUTH_CODES:
                raise
            # the provider refused the cached session (its token or certificate is gone): enroll again once
            session = pkw["session"] = await self.ensure_session(force=True)
            raw = await asyncio.to_thread(self.provisioner, h.name, **pkw)
        prov = adapt(raw, model=h.name)
        if prov.fingerprint != str(entry["fingerprint"]).lower():
            raise IntegrityError(f"provisioned artifacts of {h.name!r} do not match the provider's fingerprint",
                                 details={"model": h.name})
        h.provisioned = prov
        if self.store is None:
            try:
                self.store = await asyncio.to_thread(self._open_store)
            except PGWError as exc:
                raise self._store_error(exc) from exc
            self._start_gc()
        if self._settings is None:
            self._settings = self._gateway_settings(session)
        if h.schedule_decision is None:  # once per model and client: the reference never changes afterwards
            h.schedule_decision = await self._decide_schedule(session, entry, prov)
            h.schedule = h.schedule_decision.schedule
        if h.schedule == "fused":
            from pgw.sdk.schedule import FUSED_RING

            h.ring = FUSED_RING
        else:
            h.ring = self._pick_ring(entry)
        apply_native_env(self.options.native)
        cfg = self._model_config(h, session, prov)
        self._settings.validate_policy([cfg])
        h.state = "connecting"
        repaired = False
        for attempt in (0, 1, 2):
            rt = ModelRuntime(cfg, self._settings, mask_store=self.store, metrics=GatewayMetrics(enabled=False))
            rt.prefix_cache_factory = self._prefix_cache_factory
            self._bind_halt(h, rt)
            try:
                await rt.start()
            except Exception as exc:
                if isinstance(exc, AuthError) or str(getattr(exc, "code", "")) in _AUTH_CODES:
                    if attempt == 2:
                        raise
                    # the provider no longer accepts the service token or certificate (rotated, lost, revoked):
                    # enroll again once (a revoked key then fails there with a clear AuthenticationError)
                    session = await self.ensure_session(force=True)
                    cfg = self._model_config(h, session, prov)
                    continue
                if (repaired or rt.fingerprint or not self._in_cache(prov.pack_path)
                        or not _damaged_pack(exc) or str(prov.pack_path) in self._repaired):
                    raise
                self._repaired.add(str(prov.pack_path))  # at most once per pack and process
                # the cached gateway pack failed its verification (or could not be read) before anything was sent:
                # a damaged or partly copied cache. Delete it and provision it again once; the new copy is checked
                # the same way against the pinned fingerprint
                repaired = True
                log.warning("model %s: the cached model files are damaged (%s); downloading them again", h.name,
                            getattr(exc, "message", None) or type(exc).__name__)
                await asyncio.to_thread(shutil.rmtree, prov.pack_path, True)
                prov = adapt(await asyncio.to_thread(self.provisioner, h.name, **pkw), model=h.name)
                if prov.fingerprint != str(entry["fingerprint"]).lower():
                    raise IntegrityError(f"provisioned artifacts of {h.name!r} do not match the provider's "
                                         "fingerprint", details={"model": h.name}) from exc
                h.provisioned = prov
                cfg = self._model_config(h, session, prov)
                continue
            break
        h.runtime = rt
        h.started_at = time.time()
        if rt.started:
            self._on_ready(h)
        else:  # connecting in the background (the runtime retries): requests wait for it
            asyncio.get_running_loop().create_task(self._await_ready(h))

    def _in_cache(self, path: Path) -> bool:
        """Whether ``path`` is a provisioned artifact inside this cache (never a pack the application supplied)."""
        try:
            return Path(path).resolve().is_relative_to(self.layout.models_dir.resolve())
        except OSError:
            return False

    async def _await_ready(self, h: ModelHandle) -> None:
        rt = h.runtime
        while rt is not None and not rt.started and rt.state == "connecting":
            await asyncio.sleep(0.5)
        if rt is not None and rt.started:
            self._on_ready(h)
        elif rt is not None:
            h.state, h.error = "failed", rt.last_error

    def _on_ready(self, h: ModelHandle) -> None:
        self._pace_strict_passes(h.runtime)
        h.state = "ready"
        h.error = None
        h.ready_event.set()
        if self._cert_task is None or self._cert_task.done():
            self._cert_task = asyncio.get_running_loop().create_task(self._cert_loop(), name="pgw-sdk-cert")
        if self.options.factory and h.factory is None:
            try:
                h.factory = FactoryProcess(self._factory_plan(h), stall_s=self.options.factory_stall_s,
                                           maskgen_stall_s=self.options.maskgen_stall_s).start()
            except Exception as exc:  # the stock still serves; status shows it
                log.error("mask factory for %s not started: %s", h.name, getattr(exc, "message", type(exc).__name__))
        self._write_status()

    def _pace_strict_passes(self, rt: Any) -> None:
        """Strict speculation: start a pass no earlier than ``spec_pass_gap_ms`` after the previous pass ended.

        The provider sees the gap between the last reply of one pass and the first request of the next. It holds the
        client's work between passes, which differs a little between the answer's passes and the dummy passes after
        it, most of all at the boundary, where the application wakes up with its answer (measured in the security
        review 2026-11, S20: ``benchmarks/sdk_dummy_pass_timing.py``). With the floor every gap below it looks the
        same. Rows, trips and outputs are unchanged; only the start of each forward pass waits."""
        floor = float(self.options.spec_pass_gap_ms) / 1e3
        gen = getattr(rt, "generator", None)
        model = getattr(gen, "model", None)
        if self.options.speculation == "off" or floor <= 0 or model is None or getattr(model, "_pgw_paced", False):
            return
        inner = model.aforward_batch
        last = [0.0]

        async def paced(seqs: Any) -> Any:
            wait = last[0] + floor - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                return await inner(seqs)
            finally:
                last[0] = time.monotonic()

        model.aforward_batch = paced
        model._pgw_paced = True

    async def _decide_schedule(self, session: Session, entry: dict[str, Any], prov: Provisioned) -> Any:
        """``schedule`` (pgw.sdk.schedule): standard, fused, or auto (link measured once, now)."""
        from pgw.sdk.schedule import FUSED_RING, decide

        o = self.options
        served = "fused" in (entry.get("schedules") or ()) and str(entry.get("fused_ring", FUSED_RING)) == FUSED_RING
        arch = None
        if o.schedule == "auto" and served:
            from pgw.modelpack import ModelPack

            def read_arch() -> Any:
                p = ModelPack.open(prov.pack_path, verify="fingerprint")
                try:
                    return p.arch
                finally:
                    with contextlib.suppress(Exception):
                        p.close()

            arch = await asyncio.to_thread(read_arch)
        d = await asyncio.to_thread(decide, o.schedule, fused_served=served, arch=arch, session=session,
                                    rtt_threshold_ms=o.schedule_auto_rtt_ms, margin=o.schedule_auto_margin)
        log.info("model %s: schedule %s (%s: %s)", entry.get("name"), d.schedule, d.requested, d.reason)
        return d

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
        if mode == "off":
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
        if h.schedule == "fused":
            if lane != "tls" and not (o.extra or {}).get("fused_lane_uds"):
                raise NotFoundError("schedule='fused' needs the provider's TCP+TLS trip lane (none was offered)")
            # fused2 + the folded head (head_mode provider_int8 with fused2) + layer 0's qkv local: 2L trips;
            # the pinned fused reference (migration 8 bits, derived from the verified pack like the provider's)
            kw.update({"schedule": "fused2", "head_mode": "provider_int8", "fused2_local_qkv0": True,
                       "fused2_migrate_bits": 8, "prefix_cache": "off", "float_ops": "native"})
        kw.update(self._spec_config(h.name))
        return GatewayModelConfig(**kw)

    def _spec_config(self, model: str | None = None) -> dict[str, Any]:
        """Strict speculation (``speculation``, opt-in): fixed ``spec_rows`` rows per verify pass, strict pass
        count (dummy passes after the answer), masked filler rows. The prompt-lookup draft proposes whenever a
        context match of at least 2 tokens exists: the rows are fixed, so a rejected proposal costs nothing more
        than the filler row it replaces. ``"draft_head"``: ``draft_head_path``, else the head pinned for
        ``model`` (``pgw.sdk.draft_heads``: shipped file or cache copy checked against its SHA-256)."""
        o = self.options
        if o.speculation == "off":
            return {}
        r = int(o.spec_rows)
        kw: dict[str, Any] = {"spec_rows": r, "spec_strict_passes": True, "spec_tokens": r - 1,
                              "spec_min_tokens": r - 1}
        if o.speculation == "lookup":
            kw.update(spec_lookup=4, spec_lookup_min_n=2)
        else:
            path = o.draft_head_path
            if path is None:
                from pgw.sdk.draft_heads import resolve_draft_head

                layout = getattr(self, "layout", None)
                path = resolve_draft_head(str(model), cache_dir=None if layout is None else layout.models_dir)
            # the head drafts; a prompt-lookup match that fills the whole chain replaces its drafts
            kw.update(draft_backend="feature_head", draft_head_path=path, spec_lookup=4, spec_lookup_min_n=2)
        return kw

    def _prefix_on(self, h: ModelHandle) -> bool:
        """KV prefix reuse (:mod:`pgw.sdk.prefix_cache`): explicit ``True`` (with or without the masked LM head), or
        ``"auto"`` without the masked head (the earlier default; with the masked head it stays opt-in: see
        ``research/reviews/prefix-cache.md``)."""
        p = self.options.prefix_cache
        if h.schedule == "fused":
            return False
        return p is True or (p == "auto" and not h.masked_head)

    def _prefix_cache_factory(self, cfg: Any) -> Any:
        """The runtime's prefix cache: the multi-entry, longest-prefix client cache (RAM only)."""
        from pgw.sdk.prefix_cache import ClientPrefixCache

        o = self.options
        return ClientPrefixCache(max_bytes=int(float(o.prefix_cache_mb) * (1 << 20)),
                                 ttl_s=float(o.prefix_cache_ttl_s), max_entries=int(cfg.prefix_cache_max_entries))

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
        fused = h.schedule == "fused"
        if scheme == "pks-v3":
            hint_key = h.provisioned.hint_key_file or (sdir / "pks-hint.key")
        s = self.session
        prov = h.provisioned
        state_dir = self.layout.factory_dir(kid) / prov.fingerprint[:16]
        stream = getattr(prov, "factory_weights", "pack") == "stream"
        if fused and stream:
            from pgw.core.errors import ConfigError

            raise ConfigError("schedule='fused' needs the int8 weights for its K56 masks (factory_weights='pack')")
        if o.factory_weights == "stream" and not stream:
            from pgw.core.errors import ConfigError

            raise ConfigError("factory_weights='stream' needs the provider provisioner (it provisions the weight-free "
                              "gateway pack and the hint without keeping the int8 weights)")
        return FactoryPlan(
            model=h.name, pack_path=prov.pack_path if stream else (prov.factory_pack_path or prov.pack_path),
            ring=h.ring,
            state_dir=state_dir, demand_file=state_dir / "demand.json",
            store_root=self.layout.masks_dir(kid), kek_file=sdir / "masks.kek", counter_file=sdir / "masks.counter",
            scheme=scheme, check_field="p31" if prime else "ring", masked_head=h.masked_head and not fused,
            workers=o.workers, threads=max(1, o.mask_threads), nice=o.mask_nice, target_slots=target,
            low_water_slots=low, segment_slots=seg, first_segment_slots=max(16, min(64, seg)),
            service_url=s.service_url, token_file=s.token_file, ca_file=s.ca_file,
            client_cert_file=s.client_cert_file, client_key_file=s.client_key_file,
            # fused: its own pks-v3 hint over the fused2 linears (built locally by the factory on first use)
            hint_dir=(self.layout.models_dir / h.provisioned.fingerprint[:16] / "hint-fused" if fused else
                      h.provisioned.hint_dir or (self.layout.models_dir / h.provisioned.fingerprint[:16] / "hint")),
            hint_key_file=hint_key, factory_weights="stream" if stream else "pack",
            weights_source=o.factory_weights_source, pack_fingerprint=prov.fingerprint, hf_repo=prov.hf_repo,
            hf_revision=prov.hf_revision, check_stock_segments=int(o.check_stock_segments),
            hint_work=self._hint_work(rt) if scheme == "pks-v3" else (),
            store_fingerprint=getattr(rt, "fingerprint", None) or None, schedule="fused2" if fused else "standard",
            # one private set per worker start and per failed or aborted segment
            check_stock_private=max(8, 2 * int(o.workers) + 2))

    @staticmethod
    def _hint_work(rt: Any) -> tuple[tuple[str, int], ...]:
        """``(linear key, rows x columns)`` of the runtime pack's linears: the weights of the hint build's progress
        (public shapes)."""
        try:
            tables = getattr(rt, "_f2_tables", None) if getattr(rt, "fused2", False) else None
            if tables is not None:  # fused: the hint covers the fused2 linears (file-safe keys, as the factory's)
                return tuple((str(lin.key).replace("/", "-"), int(lin.m) * int(lin.n)) for lin in tables.all_linears())
            return tuple((str(s.id.key), int(s.out_features) * int(s.in_features)) for s in rt.pack.linear_specs())
        except Exception:  # pragma: no cover - progress without a percentage
            return ()

    @staticmethod
    def _slot_bytes(rt: Any) -> int:
        try:
            tables = getattr(rt, "_f2_tables", None) if getattr(rt, "fused2", False) else None
            if tables is not None:  # fused: every fused2 linear's outputs (the folded head included), K56
                return int(sum(int(lin.m) for lin in tables.all_linears()) * rt.ring.k // 8)
            specs = rt.pack.linear_specs()
            return int(sum(int(s.out_features) for s in specs) * rt.ring.k // 8)
        except Exception:  # pragma: no cover - defensive
            return 4 << 20

    # ---- client certificate --------------------------------------------------------------------------
    async def _cert_loop(self) -> None:
        """Keep the session's mTLS client certificate valid for the life of the engine: renewed at 2/3 of its
        lifetime (``POST /pgw/v1/enroll/renew``, loaded into the live TLS contexts), enrolled again once it has
        expired (e.g. the machine slept past it; the models then reconnect), and reloaded when another process
        sharing the cache renewed it. Failures are retried with back-off (logged; requests keep working while the
        current certificate is valid)."""
        from pgw.wire.client import client_cert_validity, reload_client_cert

        failures = 0
        loaded: int | None = None
        while True:
            s = self.session
            if s is None or s.client_cert_file is None or s.client_key_file is None \
                    or not s.service_url.startswith("https://"):
                return
            try:
                mtime = s.client_cert_file.stat().st_mtime_ns
            except OSError:
                mtime = None
            if loaded is not None and mtime is not None and mtime != loaded:
                with contextlib.suppress(Exception):  # another process renewed the shared pair
                    await asyncio.to_thread(reload_client_cert, s.client_cert_file, s.client_key_file)
            loaded = mtime
            w = client_cert_validity(s.client_cert_file)
            if w is None:
                await asyncio.sleep(CERT_CHECK_S)
                continue
            due = w[0] + (w[1] - w[0]) * 2.0 / 3.0
            if time.time() < due:
                await asyncio.sleep(max(1.0, min(due - time.time(), CERT_CHECK_S)))
                continue
            try:
                outcome = await asyncio.to_thread(self._renew_or_enroll, s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                log.warning("client certificate renewal failed (attempt %d): %s", failures,
                            getattr(exc, "message", None) or type(exc).__name__)
                await asyncio.sleep(min(600.0, 15.0 * 2 ** min(failures, 6)))
                continue
            failures = 0
            with contextlib.suppress(OSError):
                loaded = (self.session or s).client_cert_file.stat().st_mtime_ns  # type: ignore[union-attr]
            if outcome == "enrolled":
                for h in self.models.values():
                    if h.state == "ready":
                        h.reconnect = "the session was enrolled again (its client certificate had expired)"
            w = client_cert_validity((self.session or s).client_cert_file)  # type: ignore[arg-type]
            if w is not None and time.time() >= w[0] + (w[1] - w[0]) * 2.0 / 3.0:
                log.error("the new client certificate is already due by this machine's clock: the clock is wrong "
                          "(check it, e.g. 'date -u', and enable NTP time sync)")
                await asyncio.sleep(3600.0)

    def _renew_or_enroll(self, s: Session) -> str:
        """``renewed`` (or ``reloaded`` when another process just did it) while the certificate is valid, else
        ``enrolled`` (a new session: the provider never renews an expired certificate). Blocking."""
        from pgw.enroll.client import renew_client_cert
        from pgw.wire.client import client_cert_validity, reload_client_cert

        assert s.client_cert_file is not None and s.client_key_file is not None
        with file_lock(s.dir / ".enroll.lock", timeout_s=60.0):
            w = client_cert_validity(s.client_cert_file)
            now = time.time()
            if w is not None and now < w[0] + (w[1] - w[0]) * 2.0 / 3.0:
                reload_client_cert(s.client_cert_file, s.client_key_file)
                return "reloaded"
            if w is not None and now < w[1] - 60.0:
                serial = renew_client_cert(s.service_url, s.token().get_secret_value(), s.client_cert_file,
                                           s.client_key_file, ca_file=s.ca_file, timeout_s=30.0, reload=True)
                log.info("renewed the client certificate (serial %x)", serial)
                return "renewed"
        self.session = ensure_session(self.key, self.layout, force=True, transport=self._enroll_transport,
                                      timeout_s=min(60.0, self.options.timeout_s))
        return "enrolled"

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
        """Re-open a model blocked after an integrity failure (an operator decision, after investigating): the
        in-process breaker and the persisted halt of this key (as ``pgw sdk clear-halt``)."""
        name = canonical_model(model)
        old = await asyncio.to_thread(self.halts.clear, name, "reset_integrity_breaker()")
        if old.halted or old.tampered:
            log.warning("integrity halt of %s cleared by reset_integrity_breaker() (%d failure(s) recorded)",
                        name, old.failures)
        h = self.models.get(name)
        if h is not None:
            h.halt = self.halts.load(name)
            if h.runtime is not None:
                h.runtime.reset_safety(breaker=True, noise=True)
        self._write_status()

    # ---- persisted integrity halt (S11) ---------------------------------------------------------------
    def _bind_halt(self, h: ModelHandle, rt: Any) -> None:
        """Apply the persisted halt to a new runtime and persist its integrity failures from now on."""
        def on_safety(event: str, _model: str, reason: str) -> None:
            if event == "safety.integrity_failure":
                self._persist_failure(h, reason)

        rt.safety_listeners.append(on_safety)
        self._apply_halt(h, rt)

    def _apply_halt(self, h: ModelHandle, rt: Any) -> None:
        """Trip ``rt``'s breaker when the persisted record says halted (or cannot be verified). Cheap: the record
        is re-read only when its file changed, so another process's halt reaches this one on its next request."""
        try:
            rec = self.halts.load(h.name)
        except Exception as exc:  # fail closed
            rec = HaltRecord(h.name, halted=True, tampered=f"unreadable: {type(exc).__name__}")
        h.halt = rec
        if rec.failures > h.integrity_failures:
            h.integrity_failures = rec.failures
        if rec.last_failure_at and (h.last_integrity_failure_at or 0) < rec.last_failure_at:
            h.last_integrity_failure_at = rec.last_failure_at
        if (rec.halted or rec.tampered) and not rt.breaker.is_open:
            log.critical("model %s halted: %s", h.name, rec.reason)
            rt.breaker.trip(reason="persisted_halt" if not rec.tampered else "halt_record_unverifiable")

    def _persist_failure(self, h: ModelHandle, reason: str) -> None:
        try:
            h.halt = self.halts.record_failure(h.name, reason)
        except Exception:  # the in-memory breaker is open anyway; never break the data path
            log.exception("could not persist the integrity halt of %s", h.name)

    async def _open(self, h: ModelHandle, rt: Any, ids: list[int], params: Any, rid: str, prefix_key: str | None,
                    deadline: float, setup_deadline: float | None = None, conversation: str | None = None) -> Any:
        """Admit and reserve masks; waits (back-pressure) while the mask stock is empty or the service reconnects.

        ``mask_wait_s`` bounds the wait for masks, except
        * during the model's first fill on this host (no mask seen yet, the factory running and making progress: it
          may still be building the one-time pks-v3 hint, tens of minutes at 7B), when the request waits up to
          ``setup_deadline`` (the start-up budget ``ready_timeout_s``);
        * while the factory is visibly making the segment that will cover this request and expects it within the
          request's ``timeout_s`` (:meth:`_extended_limit`).
        A stalled factory (no progress for ``factory_stall_s`` because the provider cannot be reached, or its runs
        keep failing: :meth:`pgw.sdk.factory.FactoryProcess.health`) fails the request at once, naming why."""
        t_open = time.monotonic()
        mask_deadline = min(deadline, t_open + self.options.mask_wait_s)
        delay = 0.1
        # Requests waiting for masks are served in arrival order (FIFO): once one waits, later requests queue behind
        # it instead of trying first, so a large request is never starved by a stream of smaller ones that each fit
        # the stock left. Only the head of the queue polls the stock.
        ticket: asyncio.Event | None = None
        waiters = h.mask_waiters
        try:
            while True:
                if ticket is None and waiters:
                    ticket = asyncio.Event()
                    waiters.append(ticket)
                if ticket is not None and waiters and waiters[0] is not ticket:
                    left = min(self._mask_limit(h, mask_deadline, setup_deadline, t_open, deadline) - time.monotonic(),
                               self.options.factory_stall_s / 4)  # re-check the factory's health now and then
                    try:
                        await asyncio.wait_for(ticket.wait(), timeout=max(0.0, left))
                    except TimeoutError as exc:
                        await self._raise_if_stalled(h, rid, exc)
                        if time.monotonic() < self._mask_limit(h, mask_deadline, setup_deadline, t_open, deadline):
                            continue
                        why = self._waiting_why(h)
                        raise NotReadyError(f"no masks in stock for model {h.name!r} after waiting (queued behind "
                                            f"{waiters.index(ticket)} earlier request(s) waiting for masks){why}",
                                            code=str(ErrorCode.MASK_EXHAUSTED), request_id=rid) from exc
                    ticket.clear()
                    continue
                try:
                    self._apply_halt(h, rt)
                    sess = (await rt.open_sessions([ids], params, request_id=rid, prefix_key=prefix_key,
                                                   conversation=conversation))[0]
                    h.masks_seen = True
                    return sess
                except MaskExhaustedError as exc:
                    self._never_fits(h, exc, rid)
                    if ticket is None:
                        ticket = asyncio.Event()
                        waiters.append(ticket)
                    fac = h.factory
                    demand = getattr(fac.plan, "demand_file", None) if fac is not None else None
                    if demand is not None and self.options.factory:
                        # the factory refills only below its low-water mark: ask it for what this request waits for
                        want = int(exc.details.get("entitled") or exc.details.get("needed") or 0)
                        write_demand(demand, want + int(exc.details.get("promised") or 0))
                    broken = fac is not None and fac.owner and fac.failures >= 3 and fac.last_error
                    if not broken:
                        await self._raise_if_stalled(h, rid, exc)
                    fst = fac.state() if fac is not None else {}
                    disk = fst.get("disk_limited") if isinstance(fst.get("disk_limited"), dict) else None
                    hl: dict[str, Any] = _as_dict(fst.get("health"))
                    # a run error of this factory child (the factory retries with back-off; a persistent one stalls)
                    fac_error = hl.get("last_error") if isinstance(hl.get("last_error"), str) else None
                    stock_why = _check_stock_trouble(fac)
                    first_fill = self._first_fill(h, setup_deadline) and disk is None
                    deficit = (int(exc.details.get("entitled") or exc.details.get("needed") or 0)
                               + int(exc.details.get("promised") or 0) - int(exc.details.get("available") or 0))
                    limit = self._mask_limit(h, mask_deadline, setup_deadline, t_open, deadline, deficit) if (
                        disk is None) else mask_deadline
                    if broken or stock_why or time.monotonic() + delay > limit:
                        why = f"; the mask factory is failing: {fac.last_error}" if broken and fac is not None else (
                            "" if self.options.factory else "; the background mask factory is off (factory=False)")
                        if stock_why:
                            why += f"; {stock_why}"
                        if disk is not None:
                            why += (f"; the mask factory stopped producing because the disk of {self.layout.root} is "
                                    f"almost full (it keeps {int(disk.get('min_free_bytes') or 0) / 2**30:.1f} GiB "
                                    "free; free disk space there, or lower PGW_FACTORY_MIN_FREE_BYTES)")
                        elif fst.get("halted"):
                            why += "; the mask factory halted after provider integrity failures (see pgw sdk status)"
                        elif fac_error is not None and not broken:
                            why += f"; the mask factory's last run failed: {fac_error[:300]} (it retries)"
                        if first_fill:
                            why += self._setup_why(hl)
                        elif not broken and not stock_why and disk is None:
                            why += self._eta_why(hl)
                        raise NotReadyError(f"no masks in stock for model {h.name!r} after waiting "
                                            f"{time.monotonic() - t_open:.0f}s{why}",
                                            code=str(exc.code), request_id=rid) from exc
                except BreakerOpenError as exc:  # blocked after an integrity failure: hard error, never waited out
                    why = (h.halt.reason if h.halt is not None else None) or exc.message
                    raise IntegrityError(f"model {h.name!r} is blocked after an integrity failure of the provider "
                                         f"({why}); after investigating, clear it with 'pgw sdk clear-halt --model "
                                         f"{h.name}' or reset_integrity_breaker()", request_id=rid) from exc
                except ServiceUnavailableError as exc:
                    retry = exc.details.get("retry_after_s") is not None or rt.state == "connecting"
                    if not retry or time.monotonic() + delay > deadline:
                        raise
                await asyncio.sleep(delay)
                delay = min(1.0, delay * 1.5)
        finally:
            if ticket is not None:
                head = bool(waiters) and waiters[0] is ticket
                with contextlib.suppress(ValueError):
                    waiters.remove(ticket)
                if head and waiters:
                    waiters[0].set()  # the next request in line tries now

    def _first_fill(self, h: ModelHandle, setup_deadline: float | None) -> bool:
        """The model's first fill on this host (no mask seen yet, the factory running and not stalled)."""
        fac = h.factory
        broken = fac is not None and fac.owner and fac.failures >= 3 and fac.last_error
        if fac is None or broken or h.masks_seen or setup_deadline is None:
            return False
        st = fac.state()
        return not st.get("halted") and not (st.get("health") or {}).get("stalled")

    def _mask_limit(self, h: ModelHandle, mask_deadline: float, setup_deadline: float | None,
                    t_open: float | None = None, deadline: float | None = None, deficit: int = 0) -> float:
        if setup_deadline is not None and self._first_fill(h, setup_deadline):
            return max(mask_deadline, setup_deadline)
        if t_open is not None and deadline is not None:
            return max(mask_deadline, self._extended_limit(h, t_open, deadline, deficit))
        return mask_deadline

    def _extended_limit(self, h: ModelHandle, t_open: float, deadline: float, deficit: int = 0) -> float:
        """How long a request may wait beyond ``mask_wait_s`` because the factory is visibly making its masks: while
        a segment is being made (or one completed within ``factory_stall_s``), the factory is not stalled, and the
        segments this request still lacks are due (at the last measured rate) within ``max(mask_wait_s,
        timeout_s)`` of the request's start and before its deadline; never beyond either. ``0.0``: no extension."""
        fac = h.factory
        if fac is None or self.options.mask_wait_s <= 0 or not self.options.factory:
            return 0.0
        hl = _health_of(fac)
        if hl.get("stalled") or hl.get("phase") not in ("producing", "first_segment", "idle", "retrying"):
            return 0.0
        seg = hl.get("segment") if isinstance(hl.get("segment"), dict) else None
        full = hl.get("segment_s")
        if not isinstance(full, int | float) or full <= 0:
            return 0.0
        progress_at = float(hl.get("progress_at") or 0.0)
        if seg is None and time.time() - progress_at > self.options.factory_stall_s:
            return 0.0  # nothing being made and nothing made lately
        first = float(seg["eta_s"]) if seg is not None and isinstance(seg.get("eta_s"), int | float) else float(full)
        per = max(1, int(fac.plan.segment_slots))
        more = max(0, -(-max(0, deficit) // per) - 1)  # whole segments after the next one
        eta = first + more * float(full)
        cap = min(deadline, t_open + max(self.options.mask_wait_s, self.options.timeout_s))
        now = time.monotonic()
        if now + eta > cap:
            return 0.0
        return min(cap, now + eta + min(60.0, self.options.mask_wait_s))

    async def _raise_if_stalled(self, h: ModelHandle, rid: str, cause: BaseException | None = None) -> None:
        """Fail a request waiting for masks at once while the factory is stalled (see
        :meth:`pgw.sdk.factory.FactoryProcess.health`; confirmed by one more probe of the provider first):
        ``APIConnectionError`` when the provider cannot be reached, else ``NotReadyError``; both retryable, both name
        the factory's last error and any first-time setup."""
        fac = h.factory
        if fac is None or not self.options.factory:
            return
        hl = _health_of(fac)
        if not hl.get("stalled"):
            return
        again = getattr(fac, "recheck", None)
        if callable(again):
            with contextlib.suppress(Exception):
                hl = await asyncio.to_thread(again) or {}
        stalled = hl.get("stalled")
        if not stalled:
            return
        if hl.get("phase") == "maskgen_stalled":
            mg = _as_dict(hl.get("maskgen"))
            raise ProviderStalledError(
                f"no masks for model {h.name!r}: {str(stalled)[:600]}. The mask factory waits for the provider and "
                "resumes by itself once it makes progress again; retry later (pgw sdk status shows the factory)",
                code=str(ErrorCode.BACKEND_UNAVAILABLE), request_id=rid,
                details={"reason": "provider_maskgen_stalled", "phase": "maskgen_stalled",
                         "oldest_running_s": mg.get("oldest_running_s"), "bound_s": mg.get("bound_s"),
                         "maskgen_stall_s": hl.get("maskgen_stall_s")}) from cause
        msg = (f"no masks for model {h.name!r}: the mask factory has made no progress for at least "
               f"{self.options.factory_stall_s:.0f}s: {str(stalled)[:600]}. It keeps retrying with back-off and "
               "resumes by itself when the provider is back; retry later (pgw sdk status shows the factory)")
        setup = describe_setup(hl)
        if setup:
            msg += f"; {setup} (it continues in the background)"
        prov = _as_dict(hl.get("provider"))
        details = {"phase": hl.get("phase"), "stall_s": self.options.factory_stall_s,
                   "provider_error": prov.get("error"), "factory_error": hl.get("last_error")}
        if prov.get("reachable") is False or provider_error(hl.get("last_error")):
            raise APIConnectionError(msg, code=str(ErrorCode.BACKEND_UNAVAILABLE), request_id=rid,
                                     details=details) from cause
        raise NotReadyError(msg, code=str(ErrorCode.MASK_EXHAUSTED), request_id=rid, details=details) from cause

    @staticmethod
    def _setup_why(hl: dict[str, Any]) -> str:
        setup = describe_setup(hl)
        return (f"; {setup or 'the one-time mask setup of this model is still running'} (see pgw sdk status; run "
                "'pgw sdk prepare' before the first traffic to do it ahead of time)")

    @staticmethod
    def _eta_why(hl: dict[str, Any]) -> str:
        seg = hl.get("segment") if isinstance(hl.get("segment"), dict) else None
        if seg is not None and isinstance(seg.get("eta_s"), int | float):
            return (f"; the mask factory is making {seg.get('slots')} more mask slots, due in about "
                    f"{float(seg['eta_s']):.0f}s (retry then, raise mask_wait_s, or lower max_tokens)")
        if isinstance(hl.get("segment_s"), int | float):
            return (f"; the mask factory makes a segment of masks in about {float(hl['segment_s']):.0f}s here "
                    "(retry then, raise mask_wait_s, or lower max_tokens)")
        return ""

    def _waiting_why(self, h: ModelHandle) -> str:
        fac = h.factory
        if fac is None:
            return ""
        hl = _health_of(fac)
        return self._setup_why(hl) if hl.get("setup") else self._eta_why(hl)

    def _never_fits(self, h: ModelHandle, exc: MaskExhaustedError, rid: str) -> None:
        """Fail at once (no wait) when the request is entitled to more mask slots than this client's stock can ever
        hold (the factory's target plus one segment): waiting could not help."""
        fac = h.factory
        entitled = exc.details.get("entitled") if isinstance(exc.details, dict) else None
        if fac is None or not isinstance(entitled, int):
            return
        cap = int(fac.plan.target_slots) + int(fac.plan.segment_slots)
        if entitled > cap:
            raise BadRequestError(
                f"this request may use up to {entitled} mask slots (prompt + max_tokens), more than the {cap} the "
                f"mask stock of model {h.name!r} can hold; lower max_tokens or raise mask_store_gb "
                "(docs/sdk/scaling.md)", code=str(ErrorCode.PAYLOAD_TOO_LARGE), request_id=rid,
                details={"entitled": entitled, "stock_capacity": cap}) from exc

    def _budget(self, timeout_s: float | None) -> float:
        """Whole-request budget: an explicit per-request ``timeout`` bounds everything (waiting for a model that is
        still starting included); the default allows ``ready_timeout_s`` for a first start-up on top."""
        if timeout_s is not None:
            return float(timeout_s)
        return self.options.timeout_s + self.options.ready_timeout_s

    async def chat_events(self, req: Any, *, conversation: str | None = None,
                          meta: dict[str, Any] | None = None,
                          timeout_s: float | None = None) -> AsyncIterator[tuple[Any, Any]]:
        """:meth:`_chat_events` plus the content-free request report (:mod:`pgw.sdk.reports`)."""
        meta = meta if meta is not None else {}
        started, t0 = time.time(), time.monotonic()
        first: float | None = None
        status: str = "error"
        code: str | None = "cancelled"
        try:
            async for item in self._chat_events(req, conversation=conversation, meta=meta, timeout_s=timeout_s):
                if first is None:
                    first = time.monotonic()
                yield item
            status, code = "ok", None
        except PGWError as exc:
            code = str(exc.code)
            raise
        except TimeoutError:
            code = "deadline_exceeded"
            raise
        except Exception as exc:
            code = str(getattr(exc, "code", "") or "internal")
            raise
        finally:
            if code == "cancelled" and time.monotonic() - t0 >= 0.98 * self._budget(timeout_s):
                code = "deadline_exceeded"  # the request's budget ran out (asyncio.timeout cancels the generator)
            if not meta.get("opened"):
                # never reached the provider (context length, no masks, model blocked or not ready, ...): kept in
                # the local log and status only, never reported (security review 2026-11, S12)
                self._local_error(meta.get("model"), code or "internal")
            elif self.reporter is not None and "model" in meta:
                from pgw.sdk.reports import build_record

                self.reporter.add(build_record(
                    ts=started, model=meta["model"], status=status, error_code=code,
                    prompt_tokens=int(meta.get("prompt_tokens", 0)), cached_tokens=int(meta.get("cached_tokens", 0)),
                    completion_tokens=int(meta.get("completion_tokens", 0)),
                    latency_ms=round((time.monotonic() - t0) * 1000.0, 3),
                    ttft_ms=round((first - t0) * 1000.0, 3) if first is not None else None,
                    stream=bool(meta.get("stream")), speculation=self.options.speculation != "off"))

    async def _chat_events(self, req: Any, *, conversation: str | None = None,
                           meta: dict[str, Any] | None = None,
                           timeout_s: float | None = None) -> AsyncIterator[tuple[Any, Any]]:
        """Run one chat request; yields ``(ModelHandle, TokenEvent)`` (and fills ``meta``: id, usage, ...).

        Transport failures and temporary unavailability before the first token are retried
        (``max_retries``); integrity failures never are. ``timeout_s``: this request's budget (default
        ``options.timeout_s``, plus ``ready_timeout_s`` while the model starts).
        """
        from pgw.gateway.openai_render import chat_messages, sampling_params

        meta = meta if meta is not None else {}
        t0 = time.monotonic()
        deadline = t0 + (float(timeout_s) if timeout_s is not None else self.options.timeout_s)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        meta["id"] = rid
        meta["t0"] = t0
        h, rt = await self._ready_runtime(
            req.model, deadline if timeout_s is not None else deadline + self.options.ready_timeout_s)
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
            from pgw.sdk.prefix_cache import scope_key

            scope = "sdk\x00client" if self.options.prefix_cache_scope == "client" else f"sdk\x00user:{req.user or ''}"
            prefix_key = scope_key(scope, conversation)
        # Strict speculation orders a request behind the dummy passes of the previous request of the same OpenAI
        # ``user`` (as the gateway does per key and user): a follow-up issued after an answer must not start before
        # the provider has seen one pass per token of that answer. A client-local key, never sent to the provider.
        strict = f"sdk\x00user:{req.user or ''}" if self.options.speculation != "off" else None
        attempt = 0
        while True:
            produced = 0
            sess = None
            try:
                sess = await self._open(h, rt, ids, params, rid if attempt == 0 else f"{rid}-r{attempt}",
                                        prefix_key, deadline,
                                        None if timeout_s is not None else deadline + self.options.ready_timeout_s,
                                        conversation=strict)
                h.requests += 1
                meta["opened"] = True  # a session is open: its trips reach the provider (reported, S12)
                meta["prompt_tokens"] = len(sess.prompt_ids)
                meta["cached_tokens"] = int(sess.prefix_reused)
                async for ev in sess.stream():
                    produced += 1
                    meta["completion_tokens"] = produced
                    yield h, ev
                h.verified_requests += 1
                h.last_verified_at = time.time()
                self._status_after_passes(rt)
                return
            except (CoreIntegrityError, IntegrityError) as exc:
                h.integrity_failures += 1
                h.last_integrity_failure_at = time.time()
                if isinstance(exc, CoreIntegrityError) and not (h.halt is not None and h.halt.halted):
                    self._persist_failure(h, "integrity_check_failed")  # (normally done by the breaker's listener)
                self._write_status()
                if isinstance(exc, IntegrityError):
                    raise
                raise from_pgw_error(exc, request_id=rid) from exc
            except PGWError as exc:
                if str(exc.code) in _AUTH_CODES:
                    self._note_auth_failure(h, exc)
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

    async def chat(self, req: Any, *, conversation: str | None = None,
                   timeout_s: float | None = None) -> T.ChatCompletion:
        """Non-streaming chat completion."""
        from pgw.gateway.openai_render import TokenText, chat_logprobs, stop_ids, visible

        meta: dict[str, Any] = {}
        parts: list[str] = []
        events: list[Any] = []
        finish: str | None = None
        rt = None
        stops: frozenset[int] = frozenset()
        async with asyncio.timeout(self._budget(timeout_s)):
            async for h, ev in self.chat_events(req, conversation=conversation, meta=meta, timeout_s=timeout_s):
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

    async def chat_stream(self, req: Any, *, conversation: str | None = None,
                          timeout_s: float | None = None) -> AsyncIterator[T.ChatCompletionChunk]:
        """Streaming chat completion (chunks as the gateway's SSE stream sends them)."""
        meta: dict[str, Any] = {"stream": True}
        created = int(time.time())

        def chunk(choices: list[dict[str, Any]], **extra: Any) -> T.ChatCompletionChunk:
            body = {"id": meta["id"], "created": created, "model": meta.get("model", req.model),
                    "system_fingerprint": meta.get("system_fingerprint"), "choices": choices}
            body.update(extra)
            return T.ChatCompletionChunk.model_validate(body)

        async with asyncio.timeout(self._budget(timeout_s)):
            async for item in self._chat_stream_body(req, conversation, meta, timeout_s, chunk):
                yield item

    async def _chat_stream_body(self, req: Any, conversation: str | None, meta: dict[str, Any],
                                timeout_s: float | None, chunk: Any) -> AsyncIterator[T.ChatCompletionChunk]:
        from pgw.gateway.openai_render import TokenText, chat_logprobs, stop_ids, visible

        include_usage = req.include_usage
        tt = None
        stops: frozenset[int] = frozenset()
        first = True
        prefix_cache = False
        async for h, ev in self.chat_events(req, conversation=conversation, meta=meta, timeout_s=timeout_s):
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

    def _local_error(self, model: str | None, code: str) -> None:
        """Count a request that failed before any of its trips reached the provider (status and log only)."""
        code = str(code)[:64]
        with self._status_lock:
            self.local_errors[code] = self.local_errors.get(code, 0) + 1
        h = self.models.get(model) if model else None
        if h is not None:
            h.local_errors[code] = h.local_errors.get(code, 0) + 1
        log.info("request failed locally before reaching the provider: %s (model %s; not reported)", code,
                 model or "-")

    # ---- status ------------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Public status: session, models (state, lane, native stack, masks, factory, integrity), cache paths."""
        out: dict[str, Any] = {"key_id": self.key.key_id_hex, "service_url": self.key.service_url,
                               "cache_dir": str(self.layout.root), "state_dir": str(self.layout.state),
                               "mask_scheme": self.options.mask_scheme, "pid": self.pid,
                               "session": self.session.public() if self.session else None,
                               "models": {n: h.public() for n, h in list(self.models.items())},
                               "local_errors": dict(self.local_errors)}
        return out

    def _status_path(self) -> Path:
        return self.layout.session_dir(self.key.key_id_hex) / "status.json"

    def _status_after_passes(self, rt: Any) -> None:
        """The status write at the end of a request. With strict speculation it waits until the generator has run
        every dummy pass that is due: a file write (and fsync) on the event loop between the answer's last pass
        and the first dummy pass was a gap the provider could see, marking the end of the real passes (security
        review 2026-11, S20). After the dummy passes its timing depends on the public pass count only."""
        gen = getattr(rt, "generator", None)
        if self.options.speculation == "off" or gen is None:
            self._write_status_soon()
            return
        if self._status_task is not None and not self._status_task.done():
            return

        async def later() -> None:
            # wait for the generator to go idle without polling the loop (a periodic wake-up during the dummy
            # passes only would itself be a timing difference); a busy generator gets the write after 30 s
            with contextlib.suppress(Exception):
                busy = {t for t in (getattr(gen, "_task", None), getattr(gen, "_lane_task", None))
                        if t is not None and not t.done()} | set(getattr(gen, "_draining", ()) or ())
                if busy:
                    await asyncio.wait(busy, timeout=30.0)
            self._write_status_soon()

        self._status_task = asyncio.get_running_loop().create_task(later())

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
        """Stop the models, the factories and the loop; close the mask store (idempotent).

        Order: the dummy passes still due (strict speculation) run first; requests still running are then cancelled
        (their callers get :class:`APIConnectionError`; their unused masks are burned, never reused); the runtimes
        stop; every task left on the loop is cancelled before the loop stops, so no caller waits forever. Bounded by
        about ``timeout_s`` in total."""
        with self._mu:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if os.getpid() != self.pid:
            return
        deadline = time.monotonic() + timeout_s
        for h in list(self.models.values()):
            if h.factory is not None:
                with contextlib.suppress(Exception):
                    h.factory.stop(timeout_s=max(1.0, min(30.0, 0.3 * timeout_s)))
        if loop is not None and thread is not None and thread.is_alive():
            async def _stop() -> None:
                for h in list(self.models.values()):
                    gen = getattr(h.runtime, "generator", None) if h.runtime is not None else None
                    if gen is not None and self.options.speculation != "off":
                        # strict speculation: let the dummy passes still due run first, so the provider sees one
                        # pass per generated token even when the application closes right after an answer
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(gen.wait_drained(), timeout=max(1.0, 0.6 * timeout_s))
                await _cancel_all(self._tasks, max(1.0, 0.1 * timeout_s))
                for h in list(self.models.values()):
                    if h.runtime is not None:
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(h.runtime.stop(), timeout=max(1.0, 0.2 * timeout_s))

            try:
                asyncio.run_coroutine_threadsafe(_stop(), loop).result(
                    timeout=max(1.0, deadline - time.monotonic()))
            except Exception:
                log.warning("stopping the models did not finish cleanly")
            try:  # whatever is still pending (a stuck runtime stop, background tasks): cancel, then stop the loop
                asyncio.run_coroutine_threadsafe(_cancel_all(None, 2.0), loop).result(timeout=5.0)
            except Exception:
                log.warning("tasks of the pgw client were still running at close")
            self._write_status()
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)
        if self.reporter is not None:
            with contextlib.suppress(Exception):
                self.reporter.close(timeout_s=5.0)
        self._gc_stop.set()
        gct, self._gc_thread = self._gc_thread, None
        if gct is not None:
            gct.join(timeout=30)
        if self.store is not None:
            with contextlib.suppress(Exception):
                self.store.close()
            self.store = None
        if loop is not None and not loop.is_running():
            with contextlib.suppress(Exception):
                loop.close()


def _damaged_pack(exc: BaseException) -> bool:
    """A pack that fails its verification or cannot be parsed (bytes changed, file truncated or missing)."""
    if isinstance(exc, CoreIntegrityError | OSError | ValueError):
        return True
    m = str(getattr(exc, "message", "") or exc).lower()
    return any(w in m for w in ("manifest", "truncated", "inconsistent offsets", "safetensors", "no such file"))


def _disk_full(message: str) -> bool:
    m = message.lower()
    return "disk is full" in m or "no space left" in m or "not enough free space" in m or "quota" in m


async def _cancel_all(tasks: set[asyncio.Task[Any]] | None, timeout_s: float) -> None:
    """Cancel ``tasks`` (default: every task of the running loop but this one) and wait up to ``timeout_s``."""
    me = asyncio.current_task()
    todo = [t for t in (list(tasks) if tasks is not None else asyncio.all_tasks()) if t is not me and not t.done()]
    for t in todo:
        t.cancel()
    if todo:
        await asyncio.wait(todo, timeout=timeout_s)


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
        halts = HaltStore(layout.session_state_dir(d.name), d.name)
        with contextlib.suppress(OSError, ValueError):
            recs = {m: halts.load(m).public() for m in halts.models()}
            if recs:
                entry["integrity_halts"] = recs
        fdir = d / "factory"
        if fdir.is_dir():
            entry["factories"] = {f.name: read_factory_state(f) for f in sorted(fdir.iterdir()) if f.is_dir()}
        out["sessions"].append(entry)
    return out


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _health_of(fac: Any) -> dict[str, Any]:
    """:meth:`pgw.sdk.factory.FactoryProcess.health` (``{}`` when unknown)."""
    fn = getattr(fac, "health", None)
    try:
        out = fn() if callable(fn) else None
    except Exception:  # pragma: no cover - status only
        return {}
    return out if isinstance(out, dict) else {}


def _check_stock_trouble(fac: Any) -> str | None:
    """Why a weight-free factory (``factory_weights="stream"``) cannot make masks: its check stock is empty and its
    weight passes keep failing (3 in a row), else ``None``. Requests then fail closed with this reason instead of
    waiting out the setup budget."""
    if fac is None or getattr(fac.plan, "factory_weights", "pack") != "stream":
        return None
    try:
        st = (fac.state() or {}).get("check_stock") or {}
    except Exception:  # pragma: no cover - status only
        return None
    if st.get("segments") or int(st.get("pass_failures") or 0) < 3:
        return None
    done = int(st.get("passes") or 0)
    return ("the weight-free factory has no check material: its stock is empty and "
            + ("no fingerprint-verified weight pass has completed" if not done else "its refill passes are failing")
            + f" (last pass error: {st.get('last_pass_error') or '?'}); see pgw sdk status")
