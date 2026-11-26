"""Horizontal scale-out of mask generation: N factory worker processes feeding one mask store.

``PGW_FACTORY_WORKERS=N`` (``pgw factory run --workers N``) starts a :class:`FactoryFleet`: a
supervisor that spawns N worker processes. Each worker is an ordinary factory (its own generators,
``factory-state.json`` and ``factory.lock`` under ``<state>/fleet/w<k>``) running a
:class:`FleetScheduler`, which differs from :class:`pgw.factory.scheduler.FactoryScheduler` in two
ways:

* **One segment per claim.** Before producing a segment, a worker takes the fleet's short plan lock,
  re-plans from the real store inventory **plus the segments its live peers are producing right now**,
  records its own claim in the :class:`ProductionLedger` and releases the lock. Together the workers
  therefore stop at ``target_slots`` (at most one segment per worker in flight beyond what each saw),
  and a ramp (``first_segment_slots``) still grows geometrically across workers. The disk budget
  subtracts the peers' in-flight segment bytes too.
* **Per-worker resources.** Worker k uses ``devices[k % len]`` for local generation and, in encrypted
  mode, starts on compute service ``he_service_urls[k % len]`` (:class:`SegmentShardedRemote`).

Liveness is an ``flock`` per worker (``<state>/fleet/inflight/<id>.lock``), held for the life of the
process: a claim of a worker whose lock is free (crashed, killed) is ignored, so a crash never blocks
production. Claims only steer *how much* the fleet produces; they never decide which slots a gateway
gets.

**Single use is unchanged** and does not depend on this module: every segment has a fresh 32-byte
seed and a uuid4 id, both UNIQUE in the store's SQLite state; registration and every consuming
``reserve`` run under the store's inter-process ``flock`` plus ``BEGIN IMMEDIATE``, so concurrent
writers and readers are serialised by the store itself; a segment is visible only after the atomic
rename and its registration. Partial files of live writers are ``flock``-ed
(:func:`pgw.maskstore.segment_io.partial_in_use`), so a worker that restarts after a crash cleans up
only the partial files of dead writers.

**Privacy.** Sharding is by segment, never by data: all ciphertexts of one segment go to one compute
service, each request is the same MLWE ciphertext format and size as with one service, and the number,
size and timing of requests depend only on public configuration (segment size, groups per call,
worker count). Nothing secret leaves the worker; seeds never cross the process boundary.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError
from pgw.core.fingerprint import short_fingerprint
from pgw.factory.governor import FactoryStopped
from pgw.factory.scheduler import HALT_FILE, FactoryScheduler, RefillPlan, is_provider_integrity_failure

__all__ = [
    "FactoryFleet", "FleetScheduler", "ProductionLedger", "SegmentShardedRemote", "fleet_root", "worker_settings",
]

log = logging.getLogger(__name__)

FLEET_DIR = "fleet"
_MAX_RESTART_BACKOFF_S = 300.0


def _flock_try(fd: int, flags: int) -> bool:
    try:
        fcntl.flock(fd, flags | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            return False
        raise
    return True


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# ---- in-flight ledger ------------------------------------------------------------------------------
class ProductionLedger:
    """Segments the fleet's workers are producing right now (``<root>/inflight``).

    Files per worker: ``<id>.lock`` (exclusive ``flock`` for the life of the worker = liveness) and
    ``<id>.json`` (the current claim: fingerprint, ring, slots, bytes; no seeds or segment ids).
    ``plan.lock`` serialises plan + claim across workers (held for milliseconds).

    Args:
        root: Directory shared by the fleet's workers (local filesystem: ``flock``).
        worker_id: Unique id of this worker in the fleet.
    """

    def __init__(self, root: Path, worker_id: str):
        if not worker_id or "/" in worker_id or worker_id.startswith("."):
            raise ConfigError(f"invalid fleet worker id {worker_id!r}")
        self.dir = Path(root) / "inflight"
        self.worker_id = worker_id
        self._fd: int | None = None
        self._plan_fd: int | None = None
        self._tlock = threading.RLock()

    # -- lifecycle
    def open(self) -> ProductionLedger:
        """Take this worker's liveness lock (fails if another live worker uses the same id)."""
        if self._fd is not None:
            return self
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.dir / f"{self.worker_id}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        if not _flock_try(fd, fcntl.LOCK_EX):
            os.close(fd)
            raise ConfigError(f"fleet worker {self.worker_id!r} is already running on {self.dir.parent}")
        self._fd = fd
        self.clear()  # a claim left by a previous incarnation of this worker is void
        return self

    def close(self) -> None:
        """Drop the claim and the liveness lock."""
        if self._fd is None:
            return
        with contextlib.suppress(OSError):
            self.clear()
        with contextlib.suppress(OSError):
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def __enter__(self) -> ProductionLedger:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- plan lock
    @contextlib.contextmanager
    def plan_lock(self, timeout_s: float = 120.0) -> Iterator[None]:
        """Exclusive fleet-wide lock around one plan + claim."""
        with self._tlock:
            self.dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.dir / "plan.lock", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                deadline = time.monotonic() + timeout_s
                while not _flock_try(fd, fcntl.LOCK_EX):
                    if time.monotonic() > deadline:
                        raise ConfigError("timed out waiting for the factory fleet plan lock")
                    time.sleep(0.005)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # -- claims
    def claim(self, fingerprint: str, ring: str, slots: int, nbytes: int) -> None:
        """Record the segment this worker is about to produce (call under :meth:`plan_lock`)."""
        if self._fd is None:
            raise ConfigError("production ledger is not open")
        rec = {"pid": os.getpid(), "fingerprint": fingerprint, "ring": ring, "slots": int(slots),
               "bytes": int(nbytes), "since": time.time()}
        _atomic_write(self.dir / f"{self.worker_id}.json", json.dumps(rec).encode())

    def clear(self) -> None:
        """No segment in progress."""
        with contextlib.suppress(FileNotFoundError):
            (self.dir / f"{self.worker_id}.json").unlink()

    def _alive(self, wid: str) -> bool:
        try:
            fd = os.open(self.dir / f"{wid}.lock", os.O_RDONLY)
        except OSError:
            return False
        try:
            # A shared probe fails while the owner holds its exclusive lock.
            got = _flock_try(fd, fcntl.LOCK_SH)
            return not got
        finally:
            os.close(fd)

    def peer_claims(self) -> list[dict[str, Any]]:
        """Claims of the *other* live workers (a dead worker's claim is ignored)."""
        out: list[dict[str, Any]] = []
        if not self.dir.is_dir():
            return out
        for p in sorted(self.dir.glob("*.json")):
            wid = p.stem
            if wid == self.worker_id or not self._alive(wid):
                continue
            try:
                rec = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(rec, dict) and isinstance(rec.get("slots"), int):
                rec["worker"] = wid
                out.append(rec)
        return out

    def live_workers(self) -> list[str]:
        """Ids of workers whose liveness lock is held (this one included)."""
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.lock")
                      if p.name != "plan.lock" and ((p.stem == self.worker_id and self._fd is not None)
                                                    or self._alive(p.stem)))


# ---- scheduler ---------------------------------------------------------------------------------------
class _InflightSink:
    """The store sink seen by a fleet worker's planner: inventory + the peers' in-flight segments."""

    def __init__(self, sink: Any, inflight: Callable[[str, str], int]):
        self._sink = sink
        self._inflight = inflight

    def available_slots(self, fingerprint: str, ring: Any) -> int:
        name = getattr(ring, "name", str(ring))
        return int(self._sink.available_slots(fingerprint, ring)) + int(self._inflight(fingerprint, name))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sink, name)


class FleetScheduler(FactoryScheduler):
    """A :class:`FactoryScheduler` that is one worker of a fleet sharing a store (see the module doc).

    Args:
        ledger: This worker's :class:`ProductionLedger` (opened by :meth:`_start`).
        Other arguments as :class:`FactoryScheduler` (no ``ring_sink``: in-memory gateway rings have one
        writer, the gateway's own child).
    """

    def __init__(self, settings: Any, store: Any, generators: dict[tuple[str, str], Any], *,
                 ledger: ProductionLedger, **kw: Any):
        if kw.get("ring_sink") is not None:
            raise ConfigError("a fleet worker writes into the mask store, not into a gateway ring")
        super().__init__(settings, store, generators, **kw)
        self.ledger = ledger
        self._peers: list[dict[str, Any]] = []
        self.sink = _InflightSink(self.sink, self._peer_slots)
        #: Segments this worker produced (process lifetime).
        self.segments_produced = 0
        #: A halt on provider integrity failures stops every worker of the fleet (``<fleet root>/factory-halted.json``;
        #: worker state dirs are ``<fleet root>/w<k>``); ``pgw factory clear-halt`` clears it.
        self.shared_halt_path = self.state_dir.parent / HALT_FILE

    def _peer_slots(self, fingerprint: str, ring: str) -> int:
        return sum(int(c.get("slots", 0)) for c in self._peers
                   if c.get("fingerprint") == fingerprint and c.get("ring") == ring)

    def disk_budget(self) -> int | None:
        """The single-factory budget minus the bytes of the segments peers are writing."""
        b = super().disk_budget()
        if b is None:
            return None
        return max(0, b - sum(int(c.get("bytes", 0)) for c in self._peers))

    def _start(self) -> None:
        self.ledger.open()
        super()._start()

    def close(self) -> None:
        super().close()
        self.ledger.close()

    def _next_plan(self, skip: set[tuple[str, str]]) -> RefillPlan | None:
        """Most urgent single segment (fleet-wide view), claimed in the ledger; ``None`` = nothing to do."""
        with self.ledger.plan_lock():
            self._peers = self.ledger.peer_claims()
            plans = [p for p in self.plan() if (p.fingerprint, p.ring) not in skip and p.tier == "store"]
            if not plans:
                return None
            plan = dataclasses.replace(plans[0], segments=1)
            self.ledger.claim(plan.fingerprint, plan.ring, plan.slots_per_segment, self._segment_bytes(plan))
            return plan

    def run_once(self) -> list[str]:
        """Produce segments one claim at a time until the fleet as a whole reaches target.

        A model that fails is skipped for the rest of this run; if nothing was produced and something
        failed, the first error is raised (as :meth:`FactoryScheduler.run_once`).
        """
        self._start()
        produced: list[str] = []
        errors: list[BaseException] = []
        failed: set[tuple[str, str]] = set()
        limit = int(getattr(self.settings, "max_segments_per_run", 0) or 0)
        try:
            while not (limit and len(produced) >= limit):
                self.throttle.check()
                plan = self._next_plan(failed)
                if plan is None:
                    break
                log.info("fleet worker %s: segment of %d slots for %s/%s (%s)", self.ledger.worker_id,
                         plan.slots_per_segment, short_fingerprint(plan.fingerprint), plan.ring, plan.reason)
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
                    log.error("segment generation failed for %s/%s: %s", short_fingerprint(plan.fingerprint),
                              plan.ring, self.state.last_error)
                    if is_provider_integrity_failure(exc, self.generators.get((plan.fingerprint, plan.ring))):
                        self._provider_integrity_failure(exc, plan.fingerprint, plan.ring)  # raises at the limit
                    errors.append(exc)
                    failed.add((plan.fingerprint, plan.ring))
                    continue
                finally:
                    # After the commit (inventory already counts the segment) or the abort: a peer that plans
                    # in between counts it twice and under-produces for one plan, never over-produces.
                    self.ledger.clear()
                produced.append(seg_id)
                self.segments_produced += 1
        finally:
            self._peers = []
        if errors and not produced:
            raise errors[0]
        return produced

    def status(self) -> dict[str, Any]:
        out = super().status()
        self._peers = self.ledger.peer_claims() if self.ledger._fd is not None else []
        try:
            out["fleet"] = {"worker": self.ledger.worker_id, "live_workers": self.ledger.live_workers(),
                            "peer_claims": [{k: c[k] for k in ("worker", "ring", "slots") if k in c}
                                            for c in self._peers],
                            "segments_produced": self.segments_produced}
        finally:
            self._peers = []
        return out


# ---- provider sharding ----------------------------------------------------------------------------
class SegmentShardedRemote:
    """Several compute services behind one ``remote(linear, b)`` callable, sharded **by segment**.

    Between segments the fleet scheduler calls :meth:`begin_segment`; every request of the segment
    then goes to the selected service (requests of one segment are never split across services). With
    ``rotate`` the selection moves to the next service each segment (one worker, several providers);
    otherwise the worker stays on its service (``start``) and moves on only after a segment on it failed
    (failover; the failed segment was aborted and its seed is never used again).
    """

    def __init__(self, remotes: Sequence[Callable[..., Any]], *, start: int = 0, rotate: bool = False,
                 names: Sequence[str] | None = None):
        if not remotes:
            raise ConfigError("SegmentShardedRemote needs at least one remote")
        self.remotes = list(remotes)
        self.names = list(names) if names is not None else [f"service-{i}" for i in range(len(self.remotes))]
        self.rotate = bool(rotate)
        self._cur = int(start) % len(self.remotes)
        self._started = False
        self._failed = False
        self._lock = threading.Lock()
        #: Segments begun per service (observability; public counts only).
        self.segments = [0] * len(self.remotes)

    @property
    def current(self) -> int:
        """Index of the service the current segment uses."""
        return self._cur

    def begin_segment(self) -> int:
        """Select the service for the next segment; returns its index."""
        with self._lock:
            if self._started and (self.rotate or self._failed):
                self._cur = (self._cur + 1) % len(self.remotes)
            self._started = True
            self._failed = False
            self.segments[self._cur] += 1
            return self._cur

    def __call__(self, linear: str, b: Any) -> Any:
        r = self.remotes[self._cur]
        try:
            return r(linear, b)
        except Exception:
            self._failed = True
            raise

    def __getattr__(self, name: str) -> Any:  # bytes_up, calls, ... of the current service
        if name.startswith("_") or name in ("remotes", "names", "rotate", "segments"):
            raise AttributeError(name)
        return getattr(self.remotes[self._cur], name)


# ---- supervisor -----------------------------------------------------------------------------------
def fleet_root(settings: Any, store_root: Path | None) -> Path:
    """``<state_dir or store root>/fleet``: ledger and worker state directories."""
    base = settings.state_dir or store_root
    if base is None:
        raise ConfigError("a factory fleet needs state_dir or a mask store root")
    return Path(base) / FLEET_DIR


def worker_settings(settings: Any, k: int, root: Path) -> Any:
    """Settings of worker ``k``: own state dir and factory id, its device and first compute service."""
    n = int(settings.workers)
    upd: dict[str, Any] = {
        "state_dir": root / f"w{k}",
        "factory_id": f"{settings.factory_id}-w{k}",
        "workers": 1,
    }
    if settings.devices:
        upd["device"] = settings.devices[k % len(settings.devices)]
    if settings.metrics_port is not None:
        upd["metrics_port"] = int(settings.metrics_port) + k
    if getattr(settings, "he_service_urls", None) or getattr(settings, "he_service_url", None):
        # One hint W8.A for the whole fleet (the classic factory's default location), not one per worker
        # state dir: building it is minutes and GBs for 7B. Workers build it one at a time (_hint_build_lock).
        upd["he_hint_dir"] = settings.he_hint_dir or Path(settings.state_dir or ".") / "hemask"
    if getattr(settings, "factory_weights", "pack") == "stream":
        # one check stock for the whole fleet (each worker takes its own items from it; one pass at a time)
        from pgw.factory.runtime import check_stock_root

        upd["check_stock_dir"] = check_stock_root(settings)
    if not settings.cpu_threads and n > 1:
        upd["cpu_threads"] = max(1, (os.cpu_count() or 1) // n)
    return settings.model_copy(update=upd)


@contextlib.contextmanager
def _hint_build_lock(settings: Any) -> Iterator[None]:
    """Serialise the one-time HE hint build across a fleet's workers sharing one hint directory.

    ``HintStore.build`` replaces an existing hint directory on rename, and hints are mmapped lazily, so
    two workers building at once could leave one of them with another's ``W8.A`` (its segments would
    then fail the Freivalds check). With the lock the first worker builds and the others open its hint.
    """
    hint_dir = getattr(settings, "he_hint_dir", None)
    if hint_dir is None:
        yield
        return
    Path(hint_dir).mkdir(parents=True, exist_ok=True)
    fd = os.open(Path(hint_dir) / ".fleet-build.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _worker_main(settings_json: str, k: int, n: int, root: str, store_overrides: dict[str, Any],
                 ready: Any = None) -> None:
    """Entry point of one worker process (spawned)."""
    from pgw.core.proc_tuning import exit_with_parent

    # a worker ends with its supervisor, even when the supervisor is SIGKILLed (out of memory): never an orphan
    exit_with_parent(os.getppid())
    from pgw.core.log import configure_logging
    from pgw.factory.config import FactorySettings
    from pgw.factory.governor import Throttle, apply_process_priority
    from pgw.factory.runtime import build_generators, build_sink, open_packs, open_store

    with contextlib.suppress(Exception):
        configure_logging()
    base = FactorySettings.model_validate_json(settings_json)
    s = worker_settings(base, k, Path(root))
    apply_process_priority(s.nice, s.cpu_threads)
    store = open_store(**store_overrides)
    throttle = Throttle(s.max_slots_per_hour / n if s.max_slots_per_hour else 0.0, s.duty_cycle)
    packs = open_packs(s)
    with _hint_build_lock(s):
        gens = build_generators(s, packs, throttle, worker_index=k, workers=n)
    sched = FleetScheduler(s, build_sink(s, store), gens, throttle=throttle,
                           ledger=ProductionLedger(Path(root), f"w{k}"))
    health = None
    if s.metrics_port is not None:
        from pgw.factory.health import FactoryHealthServer

        health = FactoryHealthServer(sched, s.metrics_host, s.metrics_port).start()

    def _stop(signum: int, frame: Any) -> None:
        sched.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if ready is not None:
        ready.set()
    try:
        sched.run_forever()
    finally:
        if health is not None:
            health.stop()
        with contextlib.suppress(Exception):
            store.close()


class FactoryFleet:
    """Supervisor of ``settings.workers`` factory worker processes (see the module doc).

    Workers are spawned (fresh interpreters: no inherited CUDA context, keys or file locks), restarted
    with exponential back-off when they die, and stopped with SIGTERM (each aborts its current segment;
    nothing partial is ever committed).

    Args:
        settings: :class:`pgw.factory.FactorySettings` with ``workers >= 1`` and ``output='store'``.
        store_root: Mask store root (default: from ``PGW_MASKSTORE_*``); the fleet directory lives under
            ``settings.state_dir`` or this root.
        store_overrides: ``open_store`` overrides passed to every worker (e.g. ``root``, ``key_file``).
        target: Worker entry point (tests).
    """

    def __init__(self, settings: Any, *, store_root: Path | None = None,
                 store_overrides: dict[str, Any] | None = None, target: Callable[..., None] | None = None):
        if settings.output != "store":
            raise ConfigError("a factory fleet writes into a mask store (output='store')")
        self.settings = settings
        self.store_overrides = dict(store_overrides or {})
        if store_root is None and settings.state_dir is None:
            from pgw.maskstore import MaskStoreSettings

            store_root = Path(self.store_overrides.get("root") or MaskStoreSettings().root)
        self.root = fleet_root(settings, store_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.n = int(settings.workers)
        self._target = target or _worker_main
        self._ctx = mp.get_context("spawn")
        self._procs: dict[int, Any] = {}
        self._restarts = [0] * self.n
        self._next_start = [0.0] * self.n
        self.stop_event = threading.Event()

    def _spawn(self, k: int) -> None:
        p = self._ctx.Process(
            target=self._target, name=f"pgw-factory-w{k}",
            args=(self.settings.model_dump_json(), k, self.n, str(self.root), self.store_overrides),
        )
        p.start()
        self._procs[k] = p
        log.info("factory fleet: worker w%d started (pid %s)", k, p.pid)

    def start(self) -> FactoryFleet:
        """Spawn every worker."""
        for k in range(self.n):
            self._spawn(k)
        return self

    def stop(self, timeout_s: float = 60.0) -> None:
        """SIGTERM every worker, wait, then kill stragglers."""
        self.stop_event.set()
        for p in self._procs.values():
            if p.is_alive():
                with contextlib.suppress(OSError):
                    os.kill(p.pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_s
        for p in self._procs.values():
            p.join(max(0.0, deadline - time.monotonic()))
            if p.is_alive():
                log.warning("factory fleet: worker %s did not stop; killing it", p.name)
                p.kill()
                p.join(5)

    def supervise_once(self) -> list[int]:
        """Restart dead workers whose back-off expired; return the restarted indices."""
        out = []
        now = time.monotonic()
        for k, p in list(self._procs.items()):
            if p.is_alive() or self.stop_event.is_set():
                continue
            if self._next_start[k] == 0.0:
                self._restarts[k] += 1
                delay = min(_MAX_RESTART_BACKOFF_S, float(self.settings.error_backoff_s) * 2 ** (self._restarts[k] - 1))
                self._next_start[k] = now + delay
                log.error("factory fleet: worker w%d exited with code %s; restarting in %.0fs", k, p.exitcode, delay)
            if now >= self._next_start[k]:
                self._next_start[k] = 0.0
                self._spawn(k)
                out.append(k)
        return out

    def run_forever(self, poll_s: float = 1.0) -> None:
        """Supervise until :meth:`stop` (or SIGTERM routed to it)."""
        try:
            while not self.stop_event.wait(poll_s):
                self.supervise_once()
        finally:
            self.stop()

    def status(self) -> dict[str, Any]:
        """Worker pids, liveness and restart counts (no secrets)."""
        return {"workers": self.n, "root": str(self.root),
                "processes": [{"worker": f"w{k}", "pid": p.pid, "alive": p.is_alive(), "restarts": self._restarts[k]}
                              for k, p in sorted(self._procs.items())]}


def run_fleet(settings: Any, store_overrides: dict[str, Any] | None = None) -> None:
    """``pgw factory run`` with ``workers > 1``: start the fleet and supervise until SIGTERM/SIGINT."""
    fleet = FactoryFleet(settings, store_overrides=store_overrides).start()
    prev = {}
    if threading.current_thread() is threading.main_thread():
        def _h(signum: int, frame: Any) -> None:
            fleet.stop_event.set()

        prev = {sig: signal.signal(sig, _h) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        fleet.run_forever()
    finally:
        for sig, h in prev.items():
            signal.signal(sig, h)
