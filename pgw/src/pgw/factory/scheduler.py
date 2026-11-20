"""Keeps mask store inventories between low-water and target levels.

Policy (hysteresis): a ``(fingerprint, ring)`` starts refilling when its available slots drop
below ``low_water_slots`` and keeps refilling until ``target_slots`` is reached. While several
models need masks, the one with the lowest fill ratio gets the next segment.

Ramp: when ``first_segment_slots`` is set (smaller than ``segment_slots``) and a store is below low
water, each plan holds exactly one segment of :func:`ramp_segment_slots` slots (256, 512, 1024, ...),
and the store is re-planned after it lands. At or above low water, full segments are planned.

Crash safety and resume:

* Segments are committed atomically by the store writer; a crash mid-segment leaves only a
  temporary file that the writer/store discards. Nothing partial is ever visible.
* The scheduler keeps ``factory-state.json`` (atomic replace + fsync) with lifetime totals and the
  segment in progress. After a crash the next start logs the interrupted segment, counts the
  recovery and simply re-plans from the real inventory, which is the source of truth.
* A ``factory.lock`` file lock prevents two factory processes from driving the same store or
  export directory at once.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError, PGWError
from pgw.core.fingerprint import short_fingerprint
from pgw.core.ring import RingParams
from pgw.factory.governor import FactoryStopped, Throttle
from pgw.factory.metrics import FactoryMetrics
from pgw.factory.sinks import StoreSink

__all__ = ["FactoryLock", "FactoryScheduler", "FactoryState", "RefillPlan", "ramp_segment_slots"]

log = logging.getLogger(__name__)

STATE_FILE = "factory-state.json"
LOCK_FILE = "factory.lock"
_DEFAULT_LOW = 4096
_DEFAULT_TARGET = 65536
_MAX_BACKOFF_S = 600.0


@dataclass(frozen=True)
class RefillPlan:
    """What the scheduler intends to produce."""

    fingerprint: str
    ring: str
    segments: int
    slots_per_segment: int
    reason: str
    available: int = 0
    low_water: int = 0
    target: int = 0

    @property
    def slots(self) -> int:
        """Total slots this plan produces."""
        return self.segments * self.slots_per_segment

    @property
    def fill_ratio(self) -> float:
        """``available / target`` (0 when the target is 0)."""
        return self.available / self.target if self.target > 0 else 1.0


def _next_pow2(n: int) -> int:
    """Smallest power of two >= ``n`` (1 for ``n <= 1``)."""
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def ramp_segment_slots(available: int, first_segment_slots: int, segment_slots: int) -> int:
    """Size of the next segment while a store is below low water and ``first_segment_slots`` is set.

    ``min(segment_slots, max(first_segment_slots, next_pow2(2 * available)))``: with 256 / 4096 an empty
    store gets 256, then 512, 1024, 2048 and 4096 as the inventory grows; a store with at least
    ``segment_slots / 2`` slots already gets full segments.
    """
    return min(int(segment_slots), max(int(first_segment_slots), _next_pow2(2 * int(available))))


# ---- persistent state ------------------------------------------------------------------------
@dataclass
class FactoryState:
    """Persistent factory bookkeeping (no secrets: ids, counts and timings only)."""

    version: int = 1
    factory_id: str = ""
    totals: dict[str, dict[str, float]] = field(default_factory=dict)
    in_progress: dict[str, Any] | None = None
    crash_recoveries: int = 0
    last_error: str | None = None
    last_run_at: float | None = None
    last_segment_id: str | None = None
    throughput: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def key(fingerprint: str, ring: str) -> str:
        """``"<fp>/<ring>"`` (full fingerprint)."""
        return f"{fingerprint}/{ring}"

    @classmethod
    def load(cls, path: Path) -> FactoryState:
        """Load ``path``; a missing file gives a fresh state, a corrupt one is preserved and reset."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("state is not a JSON object")
            known = {f for f in cls.__dataclass_fields__}
            st = cls(**{k: v for k, v in data.items() if k in known})
            if not isinstance(st.totals, dict) or not isinstance(st.throughput, dict) or not (
                st.in_progress is None or isinstance(st.in_progress, dict)
            ):
                raise ValueError("state has fields of the wrong type")
            return st
        except (OSError, ValueError, TypeError) as exc:
            bad = path.with_suffix(f".corrupt-{int(time.time())}")
            with contextlib.suppress(OSError):
                path.replace(bad)
            log.warning("factory state %s unreadable (%s); starting fresh (old file kept as %s)", path, exc, bad.name)
            return cls()

    def save(self, path: Path) -> None:
        """Atomically write the state (temp file, fsync, rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        data = json.dumps(asdict(self), indent=1, sort_keys=True).encode()
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        with contextlib.suppress(OSError):  # make the rename durable (not supported everywhere)
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)

    def record_segment(self, fingerprint: str, ring: str, slots: int, seconds: float, segment_id: str) -> None:
        """Account a committed segment."""
        t = self.totals.setdefault(self.key(fingerprint, ring), {"segments": 0, "slots": 0, "seconds": 0.0})
        t["segments"] = int(t.get("segments", 0)) + 1
        t["slots"] = int(t.get("slots", 0)) + slots
        t["seconds"] = float(t.get("seconds", 0.0)) + seconds
        self.throughput[self.key(fingerprint, ring)] = slots / seconds if seconds > 0 else 0.0
        self.last_segment_id = segment_id


class FactoryLock:
    """Exclusive, non-blocking ``flock`` on ``<state_dir>/factory.lock`` (released on close or exit)."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        """Take the lock or raise :class:`ConfigError` if another factory holds it."""
        if self._fd is not None:
            return
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise ConfigError(
                f"another mask factory is already running on {self.path.parent} (lock {self.path.name} held)"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd

    @property
    def held(self) -> bool:
        """True while this process holds the lock."""
        return self._fd is not None

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            import fcntl

            with contextlib.suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


# ---- scheduler -----------------------------------------------------------------------------------
class FactoryScheduler:
    """Polls inventory and runs the generator until each ``(fingerprint, ring)`` reaches target.

    Resumable and crash-safe (partial segments are aborted by the store writer). Exposes progress
    metrics (``pgw_factory_*``). Runs as ``pgw factory run``.

    Args:
        settings: :class:`pgw.factory.FactorySettings`.
        store: A :class:`pgw.maskstore.MaskStore` or a :class:`pgw.factory.sinks.SegmentSink`.
        generators: ``{(fingerprint, ring_name): MaskGenerator}``.
        throttle: Shared :class:`Throttle` (default: built from the settings).
        metrics: ``PGWMetrics``-like object, ``None`` (auto) or ``False`` (disabled).
    """

    def __init__(
        self,
        settings: Any,
        store: Any,
        generators: dict[tuple[str, str], Any],
        *,
        throttle: Throttle | None = None,
        metrics: Any = None,
    ):
        if not generators:
            raise ConfigError("factory scheduler needs at least one generator")
        self.settings = settings
        self.sink: Any = store if hasattr(store, "available_slots") else StoreSink(store)
        self.generators: dict[tuple[str, str], Any] = {}
        for (fp, ring), gen in generators.items():
            rname = RingParams.from_name(ring).name
            gk = getattr(gen, "key", None)
            if gk is not None and tuple(gk) != (fp, rname):
                raise ConfigError(
                    f"generator for {short_fingerprint(gk[0])}/{gk[1]} registered under {short_fingerprint(fp)}/{rname}"
                )
            self.generators[(fp, rname)] = gen
        defaults = self.sink.default_levels() if hasattr(self.sink, "default_levels") else None
        low = settings.low_water_slots
        if low is None:
            low = defaults[0] if defaults else _DEFAULT_LOW
        target = settings.target_slots
        if target is None:
            target = defaults[1] if defaults else _DEFAULT_TARGET
        if target < low:
            raise ConfigError(f"target_slots ({target}) must be >= low_water_slots ({low})")
        self.low_water_slots = int(low)
        self.target_slots = int(target)
        self.stop_event = throttle.stop_event if throttle is not None else threading.Event()
        self.throttle = throttle or Throttle(settings.max_slots_per_hour, settings.duty_cycle, self.stop_event)
        self.metrics = FactoryMetrics(metrics)
        sdir = settings.state_dir or (self.sink.state_dir() if hasattr(self.sink, "state_dir") else None)
        self.state_dir = Path(sdir) if sdir is not None else Path.cwd() / ".pgw-factory"
        self.state_path = self.state_dir / STATE_FILE
        self.lock = FactoryLock(self.state_dir / LOCK_FILE)
        self.state = FactoryState.load(self.state_path)
        self._refilling: set[tuple[str, str]] = set()
        #: True when the last plan was cut short by the disk budget.
        self.disk_limited = False
        #: Liveness: wall time of the last completed plan or segment, and the failure streak.
        self.last_progress_at = time.time()
        self.consecutive_errors = 0
        self.last_run_error: str | None = None
        for (fp, ring), gen in self.generators.items():
            max_slots = getattr(gen, "max_segment_slots", None)
            if callable(max_slots) and int(settings.segment_slots) > int(max_slots()):
                raise ConfigError(
                    f"segment_slots={settings.segment_slots} exceeds the PRNG stream range for "
                    f"{short_fingerprint(fp)}/{ring} (max {int(max_slots())} slots per segment)",
                )
        self._wake = threading.Event()
        self._resumed = False
        if hasattr(self.sink, "add_low_water_listener"):
            self.sink.add_low_water_listener(self._on_low_water)

    # ---- lifecycle ---------------------------------------------------------------------------
    def _on_low_water(self, inventory: Any) -> None:
        log.info("low-water notification for %s/%s; waking scheduler",
                 short_fingerprint(getattr(inventory, "fingerprint", "") or "0" * 12), getattr(inventory, "ring", "?"))
        self._wake.set()

    def _start(self) -> None:
        """Take the lock and recover from an interrupted previous run (once)."""
        self.lock.acquire()
        if self._resumed:
            return
        self._resumed = True
        st = self.state
        st.factory_id = self.settings.factory_id
        crashed = bool(st.in_progress)
        if crashed:
            ip = st.in_progress or {}
            st.crash_recoveries += 1
            log.warning(
                "previous factory run was interrupted while generating a segment for %s/%s (pid %s); "
                "the partial segment was never committed and is discarded; resuming from inventory",
                short_fingerprint(str(ip.get("fingerprint", "0" * 12))), ip.get("ring"), ip.get("pid"),
            )
            st.in_progress = None
            self._save_state()
        # Export sinks own a private staging store: always safe to recover (we hold the factory lock).
        recover = getattr(self.sink, "recover", None)
        if callable(recover):
            try:
                recover()
            except Exception as exc:  # never block production on cleanup
                log.error("export staging recovery failed: %s", getattr(exc, "message", exc))
        elif crashed:
            # A shared customer store: only remove partial files after a crash of *this* factory.
            cleanup = getattr(getattr(self.sink, "store", self.sink), "cleanup_partials", None)
            if callable(cleanup):
                try:
                    removed = cleanup()
                    if removed:
                        log.warning("removed %d partial segment file(s) left by the crash", len(removed))
                except Exception as exc:
                    log.error("cleanup of partial segment files failed: %s", getattr(exc, "message", exc))

    def _save_state(self) -> None:
        try:
            self.state.save(self.state_path)
        except OSError as exc:
            log.error("cannot persist factory state to %s: %s", self.state_path, exc)

    def stop(self) -> None:
        """Request a stop: the current segment is aborted and :meth:`run_forever` returns."""
        self.stop_event.set()
        self._wake.set()

    def close(self) -> None:
        """Release the factory lock."""
        self.lock.release()

    def __enter__(self) -> FactoryScheduler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- planning ----------------------------------------------------------------------------
    def plan(self) -> list[RefillPlan]:
        """Compute refill work from the current inventory (most urgent first)."""
        seg = int(self.settings.segment_slots)
        plans: list[RefillPlan] = []
        for (fp, ring) in self.generators:
            avail = int(self.sink.available_slots(fp, RingParams.from_name(ring)))
            key = (fp, ring)
            below = avail < self.low_water_slots
            self.metrics.inventory(fp, ring, avail, below)
            if below:
                self._refilling.add(key)
            if avail >= self.target_slots:
                self._refilling.discard(key)
            if key not in self._refilling:
                continue
            first = int(self.settings.first_segment_slots or seg)
            if below and first < seg:
                # Ramp: one segment at a time, sized to about double the inventory, so a new (or drained)
                # store can serve after a small segment and supply grows geometrically
                # (256 -> 512 -> 1024 -> ... -> segment_slots); re-planned after each segment lands.
                size = ramp_segment_slots(avail, first, seg)
                reason = (f"no masks yet: a first segment of {size} slots" if avail == 0 else
                          f"{avail} slots available < low water {self.low_water_slots}: ramp segment of {size} slots")
                plans.append(RefillPlan(fp, ring, 1, size, reason, avail, self.low_water_slots, self.target_slots))
                continue
            deficit = max(self.target_slots - avail, 1)
            n = max(1, math.ceil(deficit / seg))
            if self.settings.max_segments_per_run:
                n = min(n, int(self.settings.max_segments_per_run))
            reason = (
                f"{avail} slots available < low water {self.low_water_slots}" if below
                else f"refilling: {avail} slots available < target {self.target_slots}"
            )
            plans.append(RefillPlan(fp, ring, n, seg, reason, avail, self.low_water_slots, self.target_slots))
        plans.sort(key=lambda p: (p.fill_ratio, p.available))
        self._heartbeat()
        return self._fit_to_disk(plans)

    def _heartbeat(self) -> None:
        self.last_progress_at = time.time()
        self.metrics.run_ok(self.last_progress_at)

    # ---- disk budget ---------------------------------------------------------------------------
    def _segment_bytes(self, plan: RefillPlan) -> int:
        gen = self.generators[(plan.fingerprint, plan.ring)]
        fn = getattr(gen, "segment_bytes", None)
        if callable(fn):
            return int(fn(plan.slots_per_segment))
        bps = getattr(gen, "bytes_per_slot", None)
        return int(bps) * plan.slots_per_segment if isinstance(bps, int) else 0

    def disk_budget(self) -> int | None:
        """Bytes that may still be written (free space minus ``min_free_bytes``, and the
        ``max_store_bytes`` cap); ``None`` when the sink has no local volume to measure."""
        path = self.sink.state_dir() if hasattr(self.sink, "state_dir") else None
        if path is None:
            return None
        try:
            free = shutil.disk_usage(Path(path)).free
        except OSError as exc:
            log.warning("cannot measure free space of %s: %s", path, exc)
            return None
        budget = int(free) - int(getattr(self.settings, "min_free_bytes", 0) or 0)
        cap = int(getattr(self.settings, "max_store_bytes", 0) or 0)
        if cap and hasattr(self.sink, "stored_bytes"):
            budget = min(budget, cap - int(self.sink.stored_bytes()))
        return max(budget, 0)

    def _fit_to_disk(self, plans: list[RefillPlan]) -> list[RefillPlan]:
        """Cap planned segments to what fits on the volume (the factory must never fill it up:
        the gateway's state database lives on the same volume and stops serving on ENOSPC)."""
        budget = self.disk_budget()
        self.disk_limited = False
        if budget is None or not plans:
            return plans
        out: list[RefillPlan] = []
        for p in plans:
            seg_bytes = self._segment_bytes(p)
            if seg_bytes <= 0:
                out.append(p)
                continue
            fit = min(p.segments, budget // seg_bytes)
            if fit < p.segments:
                self.disk_limited = True
                log.warning(
                    "disk budget: %s/%s needs %d segment(s) of %.2f GB but only %d fit (budget %.2f GB; "
                    "min_free_bytes=%d, max_store_bytes=%d); not producing more",
                    short_fingerprint(p.fingerprint), p.ring, p.segments, seg_bytes / 1e9, fit, budget / 1e9,
                    int(getattr(self.settings, "min_free_bytes", 0) or 0),
                    int(getattr(self.settings, "max_store_bytes", 0) or 0),
                )
            if fit > 0:
                out.append(dataclasses.replace(p, segments=int(fit)))
                budget -= fit * seg_bytes
        return out

    # ---- execution ---------------------------------------------------------------------------
    def _produce(self, plan: RefillPlan) -> str:
        gen = self.generators[(plan.fingerprint, plan.ring)]
        st = self.state
        st.in_progress = {"fingerprint": plan.fingerprint, "ring": plan.ring, "slots": plan.slots_per_segment,
                          "started_at": time.time(), "pid": os.getpid()}
        self._save_state()
        t0 = time.perf_counter()
        kw: dict[str, Any] = {}
        if getattr(self.settings, "segment_chunk_slots", None) is not None:
            kw["store_chunk_slots"] = int(self.settings.segment_chunk_slots)
        if getattr(self.settings, "segment_chunk_bytes", None) is not None:
            kw["store_chunk_bytes"] = int(self.settings.segment_chunk_bytes)
        seg_id: str = gen.generate_into_store(
            self.sink, plan.slots_per_segment, checks=self.settings.checks, chunk_slots=self.settings.chunk_slots,
            factory_id=self.settings.factory_id, throttle=self.throttle, **kw,
        )
        dt = time.perf_counter() - t0
        st.in_progress = None
        st.record_segment(plan.fingerprint, plan.ring, plan.slots_per_segment, dt, seg_id)
        st.last_run_at = time.time()
        st.last_error = None
        self._save_state()
        self.metrics.segment_done(plan.fingerprint, plan.ring, str(getattr(gen, "device", "?")),
                                  plan.slots_per_segment, plan.slots_per_segment / dt if dt > 0 else 0.0)
        self._heartbeat()
        return seg_id

    def run_once(self) -> list[str]:
        """Execute one plan; return produced segment ids.

        Segments are interleaved across models by fill ratio. A failing model is skipped for the
        rest of this run (others continue); if every planned model fails, the first error is raised.

        Raises:
            FactoryStopped: a stop was requested (segments committed so far are kept).
        """
        self._start()
        plans = self.plan()
        if not plans:
            log.debug("all inventories at target; nothing to do")
            return []
        remaining = {(p.fingerprint, p.ring): p for p in plans}
        avail = {(p.fingerprint, p.ring): p.available for p in plans}
        left = {(p.fingerprint, p.ring): p.segments for p in plans}
        produced: list[str] = []
        errors: list[PGWError | Exception] = []
        log.info("factory plan: %s", "; ".join(f"{short_fingerprint(p.fingerprint)}/{p.ring} x{p.segments} ({p.reason})"
                                               for p in plans))
        while left:
            self.throttle.check()
            key = min(left, key=lambda k: (avail[k] / self.target_slots if self.target_slots else 0.0, avail[k]))
            plan = remaining[key]
            try:
                seg_id = self._produce(plan)
            except FactoryStopped:
                self.state.in_progress = None
                self._save_state()
                raise
            except Exception as exc:
                self.state.in_progress = None
                self.state.last_error = f"{type(exc).__name__}: {getattr(exc, 'message', str(exc))}"[:500]
                self._save_state()
                log.error("segment generation failed for %s/%s: %s", short_fingerprint(key[0]), key[1],
                          self.state.last_error)
                errors.append(exc)
                del left[key]
                continue
            produced.append(seg_id)
            avail[key] += plan.slots_per_segment
            left[key] -= 1
            if left[key] <= 0:
                del left[key]
        if errors and not produced:
            raise errors[0]
        return produced

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        """Loop ``plan``/``run_once`` every ``poll_interval_s`` until stopped.

        A low-water notification from the store wakes the loop early. Failures back off
        exponentially (``error_backoff_s`` doubling up to 10 minutes).
        """
        if stop_event is not None:
            self.stop_event = stop_event
            self.throttle.stop_event = stop_event
        stop = self.stop_event
        backoff = float(self.settings.error_backoff_s)
        log.info("mask factory scheduler started (%d model/ring pair(s), low water %d, target %d, poll %.0fs)",
                 len(self.generators), self.low_water_slots, self.target_slots, self.settings.poll_interval_s)
        try:
            while not stop.is_set():
                wait = float(self.settings.poll_interval_s)
                try:
                    produced = self.run_once()
                    backoff = float(self.settings.error_backoff_s)
                    self.consecutive_errors = 0
                    self.last_run_error = None
                    if produced:
                        wait = 0.0  # re-plan immediately; inventory may still be below target
                except FactoryStopped:
                    break
                except ConfigError as exc:
                    self._record_error(exc)
                    raise
                except Exception as exc:
                    self._record_error(exc)
                    log.error("factory run failed (%s); retrying in %.0fs", exc, backoff)
                    wait = backoff
                    backoff = min(backoff * 2, _MAX_BACKOFF_S)
                if wait > 0:
                    self._wake.wait(wait)
                    self._wake.clear()
        finally:
            log.info("mask factory scheduler stopped")
            self.close()

    def _record_error(self, exc: BaseException) -> None:
        self.consecutive_errors += 1
        self.last_run_error = f"{type(exc).__name__}: {getattr(exc, 'message', str(exc))}"[:300]
        self.metrics.run_failed()

    # ---- reporting ---------------------------------------------------------------------------
    def health(self) -> tuple[bool, dict[str, Any]]:
        """Liveness for ``/healthz``: recent progress and no persistent failure streak."""
        now = time.time()
        stale_s = now - self.last_progress_at
        max_stale = float(getattr(self.settings, "health_max_stale_s", 3600.0))
        max_err = int(getattr(self.settings, "health_max_consecutive_errors", 5))
        reasons = []
        if stale_s > max(max_stale, 3 * float(self.settings.poll_interval_s)):
            reasons.append(f"no progress for {stale_s:.0f}s")
        if self.consecutive_errors >= max_err:
            reasons.append(f"{self.consecutive_errors} consecutive failed runs")
        return not reasons, {
            "healthy": not reasons, "reasons": reasons, "seconds_since_progress": round(stale_s, 1),
            "consecutive_errors": self.consecutive_errors, "last_error": self.last_run_error,
            "disk_limited": self.disk_limited,
        }

    def status(self) -> dict[str, Any]:
        """Inventory, plan and lifetime totals (JSON-serialisable; no secrets)."""
        rows = []
        for (fp, ring), gen in self.generators.items():
            avail = int(self.sink.available_slots(fp, RingParams.from_name(ring)))
            tot = self.state.totals.get(FactoryState.key(fp, ring), {})
            rows.append({
                "model": getattr(getattr(gen, "pack", None), "name", None),
                "fingerprint": fp,
                "ring": ring,
                "available_slots": avail,
                "low_water_slots": self.low_water_slots,
                "target_slots": self.target_slots,
                "below_low_water": avail < self.low_water_slots,
                "segments_generated": int(tot.get("segments", 0)),
                "slots_generated": int(tot.get("slots", 0)),
                "last_slots_per_second": self.state.throughput.get(FactoryState.key(fp, ring)),
                "bytes_per_slot": getattr(gen, "bytes_per_slot", None),
            })
        return {
            "sink": getattr(self.sink, "kind", "store"),
            "state_dir": str(self.state_dir),
            "factory_id": self.settings.factory_id,
            "running_elsewhere": self._lock_held_elsewhere(),
            "crash_recoveries": self.state.crash_recoveries,
            "disk_limited": self.disk_limited,
            "disk_budget_bytes": self.disk_budget(),
            "interrupted_segment": self.state.in_progress,
            "last_error": self.state.last_error,
            "last_segment_id": self.state.last_segment_id,
            "models": rows,
            "plan": [asdict(p) for p in self.plan()],
        }

    def _lock_held_elsewhere(self) -> bool:
        if self.lock.held:
            return False
        probe = FactoryLock(self.lock.path)
        try:
            probe.acquire()
        except ConfigError:
            return True
        probe.release()
        return False
