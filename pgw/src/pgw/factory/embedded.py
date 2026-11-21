"""Mask factory running as threads inside the gateway, filling its in-memory rings (design §5.3, v1).

:class:`EmbeddedFactory` builds ``FactorySettings`` from ``PGW_FACTORY_*`` (pack paths and rings
default to the gateway's masked models), one generator per ``(pack, ring)``, and a
:class:`pgw.factory.FactoryScheduler` whose ``ring_sink`` is a :class:`pgw.factory.sinks.RingSink`
over the gateway's :class:`pgw.maskstore.tiered.TieredMaskSource`. Overflow goes to the disk store
(fresh seeds; the seed registry is shared) when one exists.

Exclusivity: one GPU must never run two factories. Besides the scheduler's ``factory.lock`` on its
state directory (the store root, or ``run_dir`` in ring-only mode), the embedded factory takes a
host-wide lock per device (``<lock_dir>/pgw-factory-<device>.lock``, :func:`acquire_device_lock`; the
v2 factory child's supervisor takes the same lock). When it cannot, it logs an error and the gateway
serves from the store only (availability, not safety).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError
from pgw.factory.scheduler import FactoryLock, FactoryScheduler
from pgw.factory.sinks import RingSink, StoreSink
from pgw.maskstore.ringbuf import GLOBAL_SEEDS

__all__ = ["EmbeddedFactory", "acquire_device_lock", "default_lock_dir"]

log = logging.getLogger(__name__)


def default_lock_dir() -> Path:
    """``/run/lock/pgw`` when writable, else ``<tmp>/pgw-lock``."""
    for cand in (Path("/run/lock/pgw"), Path(tempfile.gettempdir()) / "pgw-lock"):
        try:
            cand.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            continue
        if os.access(cand, os.W_OK):
            return cand
    raise ConfigError("no writable directory for the factory device lock")


def acquire_device_lock(device: str, lock_dir: Path | None = None) -> FactoryLock:
    """Take the host-wide lock ``<lock_dir>/pgw-factory-<device>.lock`` (v1 embedded factory and the
    v2 factory child's supervisor alike); :class:`ConfigError` if another factory holds it.

    ``cuda`` and ``cuda:0`` name the same GPU and share one lock.
    """
    name = str(device).strip().lower()
    if name == "cuda":
        name = "cuda:0"
    dev = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    lock = FactoryLock(Path(lock_dir or default_lock_dir()) / f"pgw-factory-{dev}.lock")
    lock.acquire()
    return lock


class EmbeddedFactory:
    """In-process factory (one daemon thread) for the gateway's mask rings.

    Args:
        source: The gateway's tiered mask source.
        settings: ``FactorySettings``; default built from the environment with ``defaults``.
        defaults: Field defaults (e.g. ``pack_path``/``rings`` from the gateway models).
        run_dir: State/lock directory when there is no store (ring-only).
        lock_dir: Directory of the host-wide per-device lock.
        generators: ``{(fp, ring): MaskGenerator}`` override (tests).
        ring_low_fraction: Scheduler priority threshold for the rings.
    """

    def __init__(self, source: Any, *, settings: Any = None, defaults: dict[str, Any] | None = None,
                 run_dir: Path | None = None, lock_dir: Path | None = None,
                 generators: dict[tuple[str, str], Any] | None = None, ring_low_fraction: float = 0.5,
                 metrics: Any = False):
        self.source = source
        self._settings = settings
        self._defaults = dict(defaults or {})
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.lock_dir = lock_dir
        self._generators = generators
        self.ring_low_fraction = ring_low_fraction
        self._metrics = metrics
        self.scheduler: FactoryScheduler | None = None
        self._thread: threading.Thread | None = None
        self._device_lock: FactoryLock | None = None
        self.error: str | None = None

    def _build_settings(self) -> Any:
        from pgw.factory.config import FactorySettings

        if self._settings is not None:
            st = self._settings
        else:
            env = {k for k in os.environ if k.startswith("PGW_FACTORY_")}
            kw = {k: v for k, v in self._defaults.items() if f"PGW_FACTORY_{k.upper()}" not in env}
            st = FactorySettings(**kw)
        if self.source.store is None and st.state_dir is None:
            if self.run_dir is None:
                raise ConfigError("ring-only embedded factory needs mask_ring.run_dir (factory state and lock)")
            st = st.model_copy(update={"state_dir": self.run_dir})
        return st

    def start(self) -> bool:
        """Build and start; returns False (gateway keeps serving from the store) if another factory runs."""
        settings = self._build_settings()
        if settings.skip_kinds:
            self.error = ("skip_linear_kinds is not supported by the embedded factory (in-memory rings hold every "
                          "linear)")
            log.error("embedded mask factory not started: %s; serving from the mask store only", self.error)
            return False
        try:
            lock = acquire_device_lock(str(settings.device), self.lock_dir)
        except ConfigError as exc:
            self.error = exc.message
            log.error("embedded mask factory not started: %s; serving from the mask store only", exc.message)
            return False
        self._device_lock = lock
        try:
            gens = self._generators
            if gens is None:
                from pgw.factory.governor import Throttle
                from pgw.factory.runtime import build_generators, open_packs

                gens = build_generators(settings, open_packs(settings),
                                        Throttle(settings.max_slots_per_hour, settings.duty_cycle))
            self._check_generators(gens)
            store = self.source.store
            sink = StoreSink(store, seeds=GLOBAL_SEEDS) if store is not None else None
            self.scheduler = FactoryScheduler(settings, sink, gens, ring_sink=RingSink(self.source, store=store),
                                              ring_low_fraction=self.ring_low_fraction, metrics=self._metrics)
            self.scheduler._start()  # takes factory.lock on the state dir (ConfigError if held)
        except ConfigError as exc:
            self.error = exc.message
            self._release()
            log.error("embedded mask factory not started: %s; serving from the mask store only", exc.message)
            return False
        except BaseException:
            self._release()
            raise
        self._thread = threading.Thread(target=self._run, name="pgw-embedded-factory", daemon=True)
        self._thread.start()
        log.info("embedded mask factory started (%d generator(s))", len(gens))
        return True

    def _check_generators(self, gens: dict[tuple[str, str], Any]) -> None:
        for (fp, ring), gen in gens.items():
            mr = self.source.ring_for(fp, ring)
            specs = getattr(gen, "specs", None)
            if mr is not None and specs is not None:
                mr.check_specs(fp, ring, specs)  # ConfigError on a mismatch

    def _run(self) -> None:
        assert self.scheduler is not None
        try:
            self.scheduler.run_forever()
        except Exception as exc:  # ConfigError from a run: the gateway keeps serving from what it has
            self.error = getattr(exc, "message", None) or type(exc).__name__
            log.error("embedded mask factory stopped: %s", self.error)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        if self.scheduler is None:
            return {"running": False, "error": self.error}
        ok, health = self.scheduler.health()
        return {"running": self.running, "error": self.error, "healthy": ok, **health,
                "ring_segments": self.scheduler.ring_segments}

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the loop (an in-flight ring writer aborts, which wipes its slab) and release the locks."""
        sch = self.scheduler
        if sch is not None:
            sch.stop()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout)
            if t.is_alive():  # pragma: no cover - a stuck GEMM; the daemon thread dies with the process
                log.error("embedded mask factory did not stop within %.0fs", timeout)
        self._release()

    def _release(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        if self._device_lock is not None:
            self._device_lock.release()
            self._device_lock = None
