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
    APITimeoutError,
    BadRequestError,
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
    if os.environ.get("PGW_TRIP_DECRYPT_AHEAD", "") in ("1", "true", "yes", "on"):
        try:  # read by the mask store at import: also correct an import that happened before this call
            from pgw.maskstore import store as _store

            _store.TRIP_DECRYPT_AHEAD = True
        except Exception:  # pragma: no cover - defensive
            log.debug("mask store decrypt-ahead flag not set", exc_info=True)


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

    def public(self) -> dict[str, Any]:
        """Status view (no secrets, no content)."""
        out: dict[str, Any] = {"model": self.name, "state": self.state, "error": self.error, "progress": self.progress,
                               "masked_head": self.masked_head, "ring": self.ring, "requests": self.requests,
                               "verified_requests": self.verified_requests,
                               "integrity_failures": self.integrity_failures,
                               "requests_waiting_for_masks": len(self.mask_waiters),
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
        self._gc_stop = threading.Event()
        self._gc_thread: threading.Thread | None = None
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
            prime_check=self._prime_check(), pack_verify="full", mask_reservation="incremental", mask_chunk_slots=8,
            # processes with the same key and cache directory share the mask store: admission counts every
            # process's promised masks, so a running answer's extensions never find its stock taken by another
            mask_ledger_dir=self.layout.session_dir(self.key.key_id_hex) / "mask-ledger",
            mask_admission=self.options.mask_admission,
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
        from pgw.sdk.options import SUPPORTED_MODELS

        s = await self.ensure_session()
        names = [str(m.get("name", "")).lower() for m in s.models]
        return [n for n in SUPPORTED_MODELS if n in names]

    async def ensure_model(self, model: str) -> ModelHandle:
        """The model's handle, started (enroll, provision, connect, factory) on first use.

        The start-up runs as its own task: a caller that gives up (a short per-request ``timeout``) does not cancel
        a provisioning that is under way; the next request waits for the same start-up."""
        name = canonical_model(model)
        h = self.models.get(name)
        if h is None:
            h = self.models[name] = ModelHandle(name)
        if h.state == "ready":
            return h
        t = h.start_task
        if t is None or t.done():
            if (t is not None and not t.cancelled() and t.exception() is None
                    and h.state in ("ready", "connecting")):
                return h
            t = h.start_task = asyncio.get_running_loop().create_task(self._start_guarded(h))
        await asyncio.shield(t)
        return h

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
        raw = await asyncio.to_thread(self.provisioner, h.name, **pkw)
        prov = adapt(raw, model=h.name)
        if prov.fingerprint != str(entry["fingerprint"]).lower():
            raise IntegrityError(f"provisioned artifacts of {h.name!r} do not match the provider's fingerprint",
                                 details={"model": h.name})
        h.provisioned = prov
        if self.store is None:
            self.store = await asyncio.to_thread(self._open_store)
            self._start_gc()
        if self._settings is None:
            self._settings = self._gateway_settings(session)
        h.ring = self._pick_ring(entry)
        apply_native_env(self.options.native)
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
        kw.update(self._spec_config())
        return GatewayModelConfig(**kw)

    def _spec_config(self) -> dict[str, Any]:
        """Strict speculation (``speculation``, opt-in): fixed ``spec_rows`` rows per verify pass, strict pass
        count (dummy passes after the answer), masked filler rows. The prompt-lookup draft proposes whenever a
        context match of at least 2 tokens exists: the rows are fixed, so a rejected proposal costs nothing more
        than the filler row it replaces."""
        o = self.options
        if o.speculation == "off":
            return {}
        r = int(o.spec_rows)
        kw: dict[str, Any] = {"spec_rows": r, "spec_strict_passes": True, "spec_tokens": r - 1,
                              "spec_min_tokens": r - 1}
        if o.speculation == "lookup":
            kw.update(spec_lookup=4, spec_lookup_min_n=2)
        else:
            kw.update(draft_backend="feature_head", draft_head_path=o.draft_head_path)
        return kw

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
            model=h.name, pack_path=h.provisioned.factory_pack_path or h.provisioned.pack_path, ring=h.ring,
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
                    deadline: float, setup_deadline: float | None = None, conversation: str | None = None) -> Any:
        """Admit and reserve masks; waits (back-pressure) while the mask stock is empty or the service reconnects.

        ``mask_wait_s`` bounds the wait for masks, except during the model's first fill on this host (no mask seen
        yet, the factory running and healthy: it may still be building the one-time pks-v3 hint, tens of minutes at
        7B), when the request waits up to ``setup_deadline`` (the start-up budget ``ready_timeout_s``)."""
        mask_deadline = min(deadline, time.monotonic() + self.options.mask_wait_s)
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
                    left = self._mask_limit(h, mask_deadline, setup_deadline) - time.monotonic()
                    try:
                        await asyncio.wait_for(ticket.wait(), timeout=max(0.0, left))
                    except TimeoutError as exc:
                        raise NotReadyError(f"no masks in stock for model {h.name!r} after waiting (queued behind "
                                            f"{waiters.index(ticket)} earlier request(s) waiting for masks)",
                                            code=str(ErrorCode.MASK_EXHAUSTED), request_id=rid) from exc
                    ticket.clear()
                    continue
                try:
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
                    broken = fac is not None and fac.owner and fac.failures >= 3 and fac.last_error
                    first_fill = self._first_fill(h, setup_deadline)
                    limit = self._mask_limit(h, mask_deadline, setup_deadline)
                    if broken or time.monotonic() + delay > limit:
                        why = f"; the mask factory is failing: {fac.last_error}" if broken and fac is not None else (
                            "" if self.options.factory else "; the background mask factory is off (factory=False)")
                        if first_fill:
                            why += "; the one-time mask setup of this model is still running (see pgw sdk status)"
                        raise NotReadyError(f"no masks in stock for model {h.name!r} after waiting{why}",
                                            code=str(exc.code), request_id=rid) from exc
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
        finally:
            if ticket is not None:
                head = bool(waiters) and waiters[0] is ticket
                with contextlib.suppress(ValueError):
                    waiters.remove(ticket)
                if head and waiters:
                    waiters[0].set()  # the next request in line tries now

    def _first_fill(self, h: ModelHandle, setup_deadline: float | None) -> bool:
        """The model's first fill on this host (no mask seen yet, the factory running and healthy)."""
        fac = h.factory
        broken = fac is not None and fac.owner and fac.failures >= 3 and fac.last_error
        return bool(fac is not None and not broken and not h.masks_seen and setup_deadline is not None
                    and not fac.state().get("halted"))

    def _mask_limit(self, h: ModelHandle, mask_deadline: float, setup_deadline: float | None) -> float:
        if setup_deadline is not None and self._first_fill(h, setup_deadline):
            return max(mask_deadline, setup_deadline)
        return mask_deadline

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
                "(docs/sdk/scaling.md)", code=str(exc.code), request_id=rid,
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
            if self.reporter is not None and "model" in meta:
                self.reporter.add({
                    "ts": started, "model": meta["model"], "status": status, "error_code": code,
                    "prompt_tokens": int(meta.get("prompt_tokens", 0)),
                    "completion_tokens": int(meta.get("completion_tokens", 0)),
                    "latency_ms": round((time.monotonic() - t0) * 1000.0, 3),
                    "ttft_ms": round((first - t0) * 1000.0, 3) if first is not None else None,
                    "stream": bool(meta.get("stream")), "checks_ok": status == "ok"})

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
            prefix_key = f"sdk\x00user:{req.user or ''}\x00conv:{conversation or ''}"
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
                    gen = getattr(h.runtime, "generator", None) if h.runtime is not None else None
                    if gen is not None and self.options.speculation != "off":
                        # strict speculation: let the dummy passes still due run first, so the provider sees one
                        # pass per generated token even when the application closes right after an answer
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(gen.wait_drained(), timeout=max(1.0, 0.8 * timeout_s))
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
