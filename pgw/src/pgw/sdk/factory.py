"""The background mask factory of the client library: one supervised child process per model and key.

The child runs the existing factory (:mod:`pgw.factory`: scheduler, fleet of ``mask_workers`` worker processes,
the pks-v3 / mlwe encrypted backends or local generation, the 40-bit self-check of provider products, the halt
after repeated provider integrity failures) at low CPU priority (``nice``), with ``torch`` threads capped, and
keeps the model's mask stock between the low-water mark and the target.

Several processes on one host that use the same key and cache directory share one mask store and one factory
per model: the factory is run by whichever process holds ``<factory dir>/owner.lock``; the others retry the lock
every few seconds and take over when its holder exits. The child exits with its parent (``PR_SET_PDEATHSIG``).
Its output goes to ``<factory dir>/factory.log`` (nothing on the application's stdout or stderr).

The owner's supervisor also watches the child from outside (:meth:`FactoryProcess.health`): the first-time setup
(the pks-v3 hint build: percent of the work and an ETA; the first segment), segment progress (completed segments,
writes to the segment being made), the child's heartbeat, and whether the provider answers (``GET /readyz`` with
the session's pinned CA, only while that matters: first fill, a run error, a request waiting). When the factory
has made no progress for ``stall_s`` because the provider cannot be reached (or its runs keep failing), the
supervisor reports a *stall*; requests waiting for masks then fail at once with that reason instead of waiting
out their budget, while the child keeps retrying with back-off. When the provider answers again after an outage,
a child waiting out a long back-off is restarted at once (never during a hint build or a running segment), so
production resumes within seconds. What it observes is written to ``<factory dir>/supervisor.json`` (public
values only) for ``pgw sdk status`` and the other processes sharing the factory.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pgw.sdk._factory_child import HEARTBEAT_FILE
from pgw.sdk.cache import HeldLock, ensure_private_dir

__all__ = ["FactoryPlan", "FactoryProcess", "describe_setup", "factory_env", "write_demand"]

log = logging.getLogger(__name__)

_LOG_MAX_BYTES = 4 << 20
#: What the owner's supervisor observes (setup progress, provider reachability, a stall); public values only.
WATCH_FILE = "supervisor.json"
#: A ``supervisor.json`` older than this is not used by other processes (its owner is gone or stuck).
_WATCH_FRESH_S = 60.0
#: Factory errors that say the provider could not be reached or could not serve (wording of messages only: the
#: supervisor's own probe decides whether the provider is reachable).
_UNREACHABLE = re.compile(r"transport error|unavailable|unreachable|timed? ?out|timeout|deadline|connect|reset|"
                          r"refused|broken pipe|eof|tls|ssl|certificate|\b5\d\d\b|bad gateway", re.I)


def provider_error(message: str | None) -> bool:
    """Whether a factory error line says the provider could not be reached or could not serve (transport, TLS,
    timeouts, 5xx): wording only."""
    return bool(message) and bool(_UNREACHABLE.search(str(message)))


@dataclass(frozen=True)
class FactoryPlan:
    """Everything the factory child of one model needs (no secret values: only file paths)."""

    model: str
    pack_path: Path
    ring: str
    state_dir: Path
    store_root: Path
    kek_file: Path
    counter_file: Path
    scheme: str
    check_field: str
    masked_head: bool
    workers: int
    threads: int
    nice: int
    target_slots: int
    low_water_slots: int
    segment_slots: int
    first_segment_slots: int
    service_url: str | None = None
    token_file: Path | None = None
    ca_file: Path | None = None
    client_cert_file: Path | None = None
    client_key_file: Path | None = None
    hint_dir: Path | None = None
    hint_key_file: Path | None = None
    #: written by the engine while a request waits for more masks than are in stock (see FactorySettings)
    demand_file: Path | None = None
    #: ``"stream"``: ``pack_path`` is the weight-free gateway pack; check material comes from a stock refilled by
    #: fingerprint-checked weight passes from ``weights_source`` (:mod:`pgw.factory.checkstock`)
    factory_weights: str = "pack"
    weights_source: str = "provider"
    pack_fingerprint: str | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None
    check_stock_segments: int = 64
    check_stock_private: int = 8
    #: ``(linear key, rows x columns)`` of every linear the pks-v3 hint covers: weights of the hint build's
    #: progress (its time is proportional to them); empty: no percentage is shown
    hint_work: tuple[tuple[str, int], ...] = ()
    #: the mask store's fingerprint directory of this model (segment-write progress); ``None``: any
    store_fingerprint: str | None = None


#: pks-v3 encrypts in groups of 256 slots and amortises the upload's public NTTs over the groups of one request:
#: a segment is rounded down to whole groups (a partly filled group costs as much as a full one).
PKS_GROUP_SLOTS = 256


def segment_slots(scheme: str, slots: int) -> int:
    """The factory's steady-state segment size: ``slots``, for pks-v3 rounded down to whole 256-slot groups (at
    least one). Per mask slot the client's encryption costs about 1.8x as much with one group per request as with
    four (``benchmarks/pks_client_slot_bench.py``); masks and outputs do not depend on the segmentation."""
    if scheme != "pks-v3" or slots < PKS_GROUP_SLOTS:
        return int(slots)
    return int(slots) // PKS_GROUP_SLOTS * PKS_GROUP_SLOTS


def factory_env(plan: FactoryPlan) -> dict[str, str]:
    """``PGW_FACTORY_*`` / ``PGW_MASKSTORE_*`` for the child (paths only; secrets stay in their 0600 files)."""
    env: dict[str, str] = {
        "PGW_FACTORY_PACK_PATH": str(plan.pack_path),
        "PGW_FACTORY_RINGS": json.dumps([plan.ring]),
        "PGW_FACTORY_RING": plan.ring,
        "PGW_FACTORY_DEVICE": "cpu",
        "PGW_FACTORY_STATE_DIR": str(plan.state_dir),
        "PGW_FACTORY_SEGMENT_SLOTS": str(segment_slots(plan.scheme, plan.segment_slots)),
        "PGW_FACTORY_FIRST_SEGMENT_SLOTS": str(plan.first_segment_slots),
        "PGW_FACTORY_LOW_WATER_SLOTS": str(plan.low_water_slots),
        "PGW_FACTORY_TARGET_SLOTS": str(plan.target_slots),
        "PGW_FACTORY_NICE": str(plan.nice),
        "PGW_FACTORY_CPU_THREADS": str(plan.threads),
        "PGW_FACTORY_WORKERS": str(plan.workers),
        "PGW_FACTORY_POLL_INTERVAL_S": "2",
        "PGW_FACTORY_CHECK_FIELD": plan.check_field,
        "PGW_FACTORY_MASKED_HEAD": "true" if plan.masked_head else "false",
        "PGW_FACTORY_FACTORY_ID": "sdk",
        "PGW_FACTORY_ERROR_BACKOFF_S": "5",
        "PGW_MASKSTORE_ROOT": str(plan.store_root),
        "PGW_MASKSTORE_KEY_PROVIDER": "file",
        "PGW_MASKSTORE_KEY_FILE": str(plan.kek_file),
        "PGW_MASKSTORE_COUNTER": "file",
        "PGW_MASKSTORE_COUNTER_FILE": str(plan.counter_file),
        "PGW_MASKSTORE_TARGET_SLOTS": str(plan.target_slots),
        "PGW_MASKSTORE_LOW_WATER_SLOTS": str(plan.low_water_slots),
    }
    if plan.demand_file is not None:
        env["PGW_FACTORY_DEMAND_FILE"] = str(plan.demand_file)
    if plan.scheme in ("pks-v3", "mlwe"):
        assert plan.service_url is not None
        env["PGW_FACTORY_HE_SERVICE_URL"] = plan.service_url
        env["PGW_FACTORY_HE_SCHEME"] = plan.scheme
        if plan.token_file is not None:
            env["PGW_FACTORY_HE_TOKEN_FILE"] = str(plan.token_file)
        if plan.ca_file is not None:
            env["PGW_FACTORY_HE_CA_FILE"] = str(plan.ca_file)
        if plan.client_cert_file is not None and plan.client_key_file is not None:
            env["PGW_FACTORY_HE_CLIENT_CERT_FILE"] = str(plan.client_cert_file)
            env["PGW_FACTORY_HE_CLIENT_KEY_FILE"] = str(plan.client_key_file)
        if plan.hint_dir is not None:
            env["PGW_FACTORY_HE_HINT_DIR"] = str(plan.hint_dir)
        if plan.scheme == "pks-v3":
            from pgw.hemask.pks import PKS_V3_ACK

            # the library's explicit acknowledgement: pks-v3 is pending external review (SPEC 7.9, 16)
            env["PGW_FACTORY_HE_PKS_ACK"] = PKS_V3_ACK
            if plan.hint_key_file is not None:
                env["PGW_FACTORY_HE_HINT_KEY_FILE"] = str(plan.hint_key_file)
    if plan.pack_fingerprint is not None:
        env["PGW_FACTORY_PACK_FINGERPRINT"] = plan.pack_fingerprint
    if plan.factory_weights == "stream":
        env["PGW_FACTORY_FACTORY_WEIGHTS"] = "stream"
        env["PGW_FACTORY_WEIGHTS_SOURCE"] = plan.weights_source
        env["PGW_FACTORY_CHECK_STOCK_SEGMENTS"] = str(plan.check_stock_segments)
        env["PGW_FACTORY_CHECK_STOCK_PRIVATE"] = str(plan.check_stock_private)
        if plan.weights_source == "hf":
            env["PGW_FACTORY_WEIGHTS_HF_REPO"] = str(plan.hf_repo or "")
            if plan.hf_revision:
                env["PGW_FACTORY_WEIGHTS_HF_REVISION"] = plan.hf_revision
    return env


def _child_env(plan: FactoryPlan) -> dict[str, str]:
    # Only what a Python process needs plus the factory's own settings: no PGW_* of the application (its
    # gateway settings, API key) and no proxies for the provider other than the standard ones.
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV", "HTTPS_PROXY", "https_proxy",
            "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "LD_LIBRARY_PATH")
    env = {k: v for k, v in os.environ.items() if k in keep}
    for k in ("PGW_FACTORY_MIN_FREE_BYTES", "PGW_MASKSTORE_MIN_FREE_BYTES"):  # disk reserves (2 GiB, 1 GiB)
        if os.environ.get(k, "").strip():
            env[k] = os.environ[k].strip()
    env.update(factory_env(plan))
    env["PGW_SDK_PARENT_PID"] = str(os.getpid())
    env["OMP_NUM_THREADS"] = str(max(1, plan.threads))
    env["PGW_LOG_LEVEL"] = "INFO"
    return env


class FactoryProcess:
    """Supervises the factory child of one model (start, restart on change or crash, stop).

    Args:
        plan: The child's configuration.
        poll_s: How often a non-owner retries the owner lock and the owner checks the child.
    """

    #: the child's command line (tests substitute a stand-in)
    command: tuple[str, ...] = (sys.executable, "-m", "pgw.sdk._factory_child")

    def __init__(self, plan: FactoryPlan, *, poll_s: float = 3.0, hang_s: float | None = None,
                 stall_s: float | None = None, probe_s: float | None = None):
        self.plan = plan
        self.poll_s = poll_s
        #: a running child whose heartbeat is older than this is killed and restarted (``PGW_SDK_FACTORY_HANG_S``)
        self.hang_s = float(hang_s if hang_s is not None else os.environ.get("PGW_SDK_FACTORY_HANG_S", "") or 300.0)
        #: no progress for this long because the provider is unreachable (or the factory's runs keep failing): a
        #: stall, and requests waiting for masks fail at once (``SDKOptions.factory_stall_s``)
        self.stall_s = float(stall_s if stall_s is not None else os.environ.get("PGW_FACTORY_STALL_S", "") or 120.0)
        #: how often the provider is probed while it matters (``GET /readyz``)
        self.probe_s = float(probe_s if probe_s is not None else min(10.0, max(1.0, self.stall_s / 6)))
        ensure_private_dir(plan.state_dir)
        self._lock = HeldLock(plan.state_dir / "owner.lock")
        self._proc: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mu = threading.Lock()
        self.starts = 0
        self.last_exit: int | None = None
        #: The child's last error line (``error [code]: message``; no secrets) and consecutive failed exits.
        self.last_error: str | None = None
        self.failures = 0
        self._started_at = 0.0
        self._next_start = 0.0
        self._log_offset = 0
        #: the delay before the latest restart (exponential in the consecutive failures)
        self.backoff_s = 0.0
        #: restarts of a child waiting out a long back-off after the provider came back (not failures)
        self.nudges = 0
        # ---- the watch (owner only; see the module docstring) ----
        self._wmu = threading.Lock()
        self._spawn_wall = 0.0
        self._err_since: float | None = None  # monotonic: first sight of the child's current run-error streak
        self._down_in_err = False  # the probe failed during that streak (the provider went away)
        self._probe_thread: threading.Thread | None = None
        self._next_probe = 0.0
        self._probe_fail_since: float | None = None
        self._last_probe_at = 0.0
        self._probe_error: str | None = None
        self._probe_ok_at: float | None = None
        self._back_at: float | None = None  # monotonic: the probe succeeded again after failing
        self._last_nudge = 0.0
        self._hint_t0: tuple[str, float, float] | None = None  # (build dir, monotonic, fraction) at first sight
        self._health: dict[str, Any] = {}
        self._watch_written = 0.0
        self._watch_sig = ""

    @property
    def log_file(self) -> Path:
        """The child's log (truncated when it grows past 4 MiB at a restart)."""
        return self.plan.state_dir / "factory.log"

    @property
    def owner(self) -> bool:
        """Whether this process runs the factory (else another process on the host does)."""
        return self._lock.held

    @property
    def running(self) -> bool:
        """Whether this process's child is alive."""
        p = self._proc
        return p is not None and p.poll() is None

    def start(self) -> FactoryProcess:
        """Start supervising (returns at once)."""
        if self._thread is None:
            self._stop.clear()
            self._supervise_once()
            self._thread = threading.Thread(target=self._loop, name=f"pgw-sdk-factory-{self.plan.model}", daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self._supervise_once()
            except Exception:  # pragma: no cover - never kill the supervisor
                log.exception("mask factory supervisor error")

    def _supervise_once(self) -> None:
        with self._mu:
            if self._stop.is_set():
                return
            if not self._lock.try_acquire():
                return  # another process on this host runs this model's factory
            p = self._proc
            hung = None
            if p is not None and p.poll() is None:
                hung = self._hung()
                if hung is None:
                    if time.monotonic() - self._started_at > 120:
                        self.failures = 0  # has been running for a while: healthy
                    try:
                        nudge = self._observe(running=True)
                    except Exception:  # pragma: no cover - the watch must never stop supervision
                        log.exception("mask factory watch error")
                        nudge = False
                    if nudge:
                        self._restart_now(p)
                    return
                with contextlib.suppress(OSError):
                    p.kill()  # stopped or stuck: its fleet workers follow it (they exit with their parent)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    p.wait(timeout=10)
            if p is not None:
                self._proc = None
                self.last_exit = p.returncode
                self.last_error = hung or _exit_reason(p.returncode, _last_error_line(self.log_file,
                                                                                         self._log_offset))
                self.failures += 1
                delay = min(300.0, self.poll_s * 2 ** min(self.failures - 1, 8))
                self.backoff_s = delay
                self._next_start = time.monotonic() + delay
                log.warning("mask factory for %s exited with code %s (%s); restarting in %.0fs", self.plan.model,
                            p.returncode, self.last_error, delay)
            if time.monotonic() < self._next_start:
                with contextlib.suppress(Exception):
                    self._observe(running=False)
                return
            self._spawn()

    def _hung(self) -> str | None:
        """Why the running child counts as hung (heartbeat older than ``hang_s``), else ``None``."""
        if self._started_at <= 0 or time.monotonic() - self._started_at < self.hang_s:
            return None
        try:
            age = time.time() - (self.plan.state_dir / HEARTBEAT_FILE).stat().st_mtime
        except OSError:
            age = time.monotonic() - self._started_at  # never beat since it started
        if age < self.hang_s:
            return None
        log.error("mask factory for %s stopped responding (no heartbeat for %.0fs); restarting it", self.plan.model,
                  age)
        return f"the mask factory stopped responding (no heartbeat for {age:.0f}s) and was restarted"

    def _spawn(self) -> None:
        lf = self.log_file
        with contextlib.suppress(OSError):
            if lf.exists() and lf.stat().st_size > _LOG_MAX_BYTES:
                lf.unlink()
        fd = os.open(lf, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0), 0o600)
        self._log_offset = os.fstat(fd).st_size
        with contextlib.suppress(OSError):  # a heartbeat of an earlier child must not count for this one
            (self.plan.state_dir / HEARTBEAT_FILE).unlink()
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - our own factory child (sys.executable -m ...)
                list(self.command), env=_child_env(self.plan),
                stdin=subprocess.DEVNULL, stdout=fd, stderr=fd, close_fds=True, cwd=str(self.plan.state_dir))
        finally:
            os.close(fd)
        self.starts += 1
        self._started_at = time.monotonic()
        with self._wmu:  # a run error in the state file from before this child is not this child's
            self._spawn_wall = time.time()
            self._err_since = None
            self._down_in_err = False
        log.info("mask factory for %s started (pid %d, %d worker(s), scheme %s)", self.plan.model, self._proc.pid,
                 self.plan.workers, self.plan.scheme)

    # ---- the watch: setup progress, provider reachability, stalls -------------------------------------------------
    def _restart_now(self, p: subprocess.Popen[bytes]) -> None:
        """Restart a child that waits out a long back-off although the provider answers again (not a failure)."""
        log.info("the provider of %s answers again: restarting the mask factory now instead of waiting out its "
                 "back-off", self.plan.model)
        with contextlib.suppress(OSError):
            p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                p.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=10)
        self._proc = None
        self.nudges += 1
        self._last_nudge = time.monotonic()
        self._spawn()

    def _probe_due(self, st: dict[str, Any], err: bool) -> bool:
        """Whether the provider's reachability matters now: the first fill, a run error, a request waiting."""
        if not self.plan.service_url or not (self.plan.scheme in ("pks-v3", "mlwe") or (
                self.plan.factory_weights == "stream" and self.plan.weights_source == "provider")):
            return False
        if err or self._probe_fail_since is not None or not st.get("totals"):
            return True
        if self.plan.demand_file is not None:
            try:
                return float(json.loads(self.plan.demand_file.read_text(encoding="utf-8"))["until"]) > time.time()
            except (OSError, ValueError, KeyError, TypeError):
                return False
        return False

    def _start_probe(self) -> None:
        t = self._probe_thread
        if t is not None and t.is_alive():
            return

        def run() -> None:
            self._note_probe(self._probe_once())

        self._probe_thread = threading.Thread(target=run, name=f"pgw-sdk-probe-{self.plan.model}", daemon=True)
        self._probe_thread.start()

    def _note_probe(self, err: str | None) -> None:
        now = time.monotonic()
        with self._wmu:
            self._last_probe_at = now
            if err is None:
                if self._probe_fail_since is not None:
                    self._back_at = now
                    if self._err_since is not None:  # errors while it was away are not the factory's failures
                        self._err_since = now
                    log.info("the provider of %s answers again after %.0fs", self.plan.model,
                             now - self._probe_fail_since)
                    if self._health.get("stalled"):
                        self._health = {**self._health, "stalled": None, "phase": "retrying"}
                self._probe_fail_since, self._probe_error, self._probe_ok_at = None, None, time.time()
                if self._health:
                    self._health["provider"] = {"reachable": True, "down_for_s": None, "error": None,
                                                "checked_ok_at": self._probe_ok_at}
            else:
                if self._probe_fail_since is None:
                    self._probe_fail_since = now
                    log.warning("the provider of %s does not answer: %s", self.plan.model, err)
                self._probe_error = err
                self._back_at = None
                if self._err_since is not None:
                    self._down_in_err = True

    def recheck(self) -> dict[str, Any]:
        """Before a request fails on a stall: probe the provider once more now (at most once a second), so a
        provider that just came back is not reported as gone; returns :meth:`health`. Blocks up to the probe's
        timeout (call it off the event loop)."""
        if self.owner and self.plan.service_url:
            with self._wmu:
                due = time.monotonic() - self._last_probe_at >= 1.0 and self._health.get("stalled") is not None
            if due:
                self._note_probe(self._probe_once())
        return self.health()

    def _probe_once(self) -> str | None:
        """``GET /readyz`` of the provider with the session's TLS settings (pinned CA, client certificate; no
        token, nothing else sent); ``None`` when it answers ready, else why not (no secrets)."""
        import httpx

        from pgw.wire.client import ClientConfig, build_ssl_context

        url = str(self.plan.service_url).rstrip("/")
        timeout = max(2.0, min(10.0, self.probe_s))
        try:
            verify: Any = True
            if url.startswith("https://"):
                cfg = ClientConfig(base_url=url, ca_file=self.plan.ca_file,
                                   client_cert_file=self.plan.client_cert_file,
                                   client_key_file=self.plan.client_key_file)
                verify = build_ssl_context(cfg, register=False)
            with httpx.Client(verify=verify, trust_env=False, timeout=timeout, follow_redirects=False) as c:
                r = c.get(f"{url}/readyz")
        except httpx.TimeoutException as exc:
            return f"no answer within {timeout:.0f}s ({type(exc).__name__})"
        except (httpx.HTTPError, OSError) as exc:
            return f"connection failed ({type(exc).__name__})"
        except Exception as exc:  # TLS context errors and the like: reported, never raised
            return f"probe failed ({type(exc).__name__})"
        if r.status_code != 200:
            return f"not ready (HTTP {r.status_code})"
        return None

    def _hint_progress(self) -> dict[str, Any] | None:
        """The pks-v3 hint build under way in ``hint_dir`` (a ``*.building-*`` / ``*.partial`` directory), else
        ``None``: weighted fraction done, ETA, the newest write (wall time)."""
        root = self.plan.hint_dir
        if root is None or self.plan.scheme != "pks-v3":
            return None
        try:
            dirs = [d for d in root.iterdir() if d.is_dir() and (".building-" in d.name or d.name.endswith(".partial"))]
        except OSError:
            return None
        live: list[tuple[float, Path]] = []
        for d in dirs:
            try:
                newest = max([d.stat().st_mtime] + [f.stat().st_mtime for f in d.iterdir()])
            except (OSError, ValueError):
                continue
            pid = re.search(r"\.building-(\d+)-", d.name)
            if pid is not None:
                if not _pid_alive(int(pid.group(1))):
                    continue  # left behind by a killed build
            elif time.time() - newest > 900:
                continue  # an interrupted resumable build nobody continues
            live.append((newest, d))
        if not live:
            return None
        newest, d = max(live)
        try:
            done = {f.name[:-4] for f in d.iterdir() if f.name.endswith(".npy")}
        except OSError:
            done = set()
        work = dict(self.plan.hint_work)
        total = sum(work.values())
        frac = None if not total else min(0.999, sum(work.get(k, 0) for k in done) / total)
        now = time.monotonic()
        if self._hint_t0 is None or self._hint_t0[0] != d.name:
            self._hint_t0 = (d.name, now, frac or 0.0)
        eta = None
        _, t0, f0 = self._hint_t0
        if frac is not None and frac > f0 and now - t0 >= 5:
            eta = (now - t0) * (1.0 - frac) / (frac - f0)
        return {"stage": "hint", "percent": None if frac is None else round(100 * frac, 1),
                "eta_s": None if eta is None else round(eta), "linears_done": len(done),
                "linears": len(work) or None, "newest_write_at": newest}

    def _part_mtime(self) -> float:
        """The newest write to a segment being made for this model (wall time; 0 if none)."""
        fp = self.plan.store_fingerprint or "*"
        best = 0.0
        with contextlib.suppress(OSError):
            for f in self.plan.store_root.glob(f"{fp}/*/segments/.*.part"):
                with contextlib.suppress(OSError):
                    best = max(best, f.stat().st_mtime)
        return best

    def _observe(self, running: bool) -> bool:
        """One look at the child from outside (every ``poll_s``): updates :meth:`health` and ``supervisor.json``.
        Returns whether the child should be restarted now (the provider answers again after an outage and the
        child waits out its back-off)."""
        now, wall = time.monotonic(), time.time()
        sd = self.plan.state_dir
        st = read_factory_state(sd)
        try:
            st_mtime = (sd / "factory-state.json").stat().st_mtime
        except OSError:
            st_mtime = 0.0
        err = st.get("last_error") if isinstance(st.get("last_error"), str) else None
        # a run error saved before this child started is an earlier child's (or an earlier process's)
        fresh_err = err is not None and running and st_mtime >= self._spawn_wall - 1.0
        in_prog = st.get("in_progress") if isinstance(st.get("in_progress"), dict) else None
        hint = self._hint_progress() if running else None
        seg_at = max(float(st.get("last_run_at") or 0.0), self._part_mtime())
        seg_recent = wall - seg_at < self.stall_s
        with self._wmu:
            if fresh_err:
                if self._err_since is None:
                    self._err_since = now
                    if self._probe_fail_since is not None:
                        self._down_in_err = True
            else:
                self._err_since, self._down_in_err = None, False
            due = running and self._probe_due(st, fresh_err)
            if due and now >= self._next_probe:
                self._next_probe = now + self.probe_s
                self._start_probe()
            down_for = now - self._probe_fail_since if self._probe_fail_since is not None else 0.0
            err_for = now - self._err_since if self._err_since is not None else 0.0
            probe_ok = self._probe_fail_since is None
            stalled = None
            if down_for >= self.stall_s and not seg_recent:
                stalled = (f"the provider ({self.plan.service_url}) has not answered for {down_for:.0f}s "
                           f"({self._probe_error})" + (f"; the mask factory's last error: {err}" if fresh_err else ""))
            elif err_for >= self.stall_s and not seg_recent and not (in_prog is not None and probe_ok):
                stalled = (f"mask generation has failed for {err_for:.0f}s without producing a segment"
                           + (" (the provider answers its health check but cannot serve mask generation)"
                              if probe_ok and self._probe_ok_at is not None and provider_error(err) else "")
                           + f"; the mask factory's last error: {err}")
            nudge = bool(running and fresh_err and self._down_in_err and probe_ok and self._back_at is not None
                         and now - self._back_at >= min(10.0, 2 * self.probe_s) and in_prog is None and hint is None
                         and now - self._last_nudge >= 30.0)
            # first-time setup: no segment was ever made for this model on this host
            setup: dict[str, Any] | None = None
            if not st.get("totals"):
                if hint is not None:
                    setup = {k: hint[k] for k in ("stage", "percent", "eta_s", "linears_done", "linears")}
                elif in_prog is not None:
                    setup = {"stage": "first_segment", "slots": in_prog.get("slots"),
                             "running_s": round(max(0.0, wall - float(in_prog.get("started_at") or wall)))}
                elif fresh_err:
                    setup = {"stage": "first_segment_retry"}
                elif running:
                    setup = {"stage": "starting"}
            tput: dict[str, Any] = st["throughput"] if isinstance(st.get("throughput"), dict) else {}
            rate = max((float(v) for v in tput.values() if isinstance(v, int | float)), default=0.0)
            seg: dict[str, Any] | None = None
            if in_prog is not None:
                slots = int(in_prog.get("slots") or 0)
                run_s = max(0.0, wall - float(in_prog.get("started_at") or wall))
                seg = {"slots": slots, "running_s": round(run_s, 1),
                       "eta_s": round(max(0.0, slots / rate - run_s), 1) if rate > 0 else None}
            if stalled:
                phase = "waiting_for_provider" if (not probe_ok or provider_error(err)) else "failing"
            elif setup is not None:
                phase = {"hint": "hint_build", "first_segment": "first_segment",
                         "first_segment_retry": "retrying"}.get(str(setup["stage"]), "starting")
            elif in_prog is not None:
                phase = "producing"
            elif fresh_err:
                phase = "retrying"
            else:
                phase = "idle" if running else "restarting"
            self._health = {
                "phase": phase, "setup": setup, "segment": seg, "stalled": stalled,
                "segment_s": round(self.plan.segment_slots / rate, 1) if rate > 0 else None,
                "progress_at": max(seg_at, float(hint["newest_write_at"]) if hint else 0.0) or None,
                "failing_for_s": round(err_for, 1) if fresh_err else None, "last_error": err if fresh_err else None,
                "provider": {"reachable": None if (self._probe_ok_at is None and probe_ok) else probe_ok,
                             "down_for_s": round(down_for, 1) if not probe_ok else None,
                             "error": self._probe_error, "checked_ok_at": self._probe_ok_at},
                "stall_s": self.stall_s, "updated_at": wall,
            }
            health = dict(self._health)
        self._write_watch(health)
        return nudge

    def _write_watch(self, health: dict[str, Any]) -> None:
        """``supervisor.json`` when it changed, or every 15 s while not idle: for ``pgw sdk status`` and the other
        processes sharing this factory."""
        from pgw.core.fileio import atomic_write_text

        sig = json.dumps({k: v for k, v in health.items()
                          if k in ("phase", "setup", "stalled", "last_error")}, sort_keys=True, default=str)
        if sig == self._watch_sig and (health.get("phase") == "idle"
                                       or time.monotonic() - self._watch_written < 15.0):
            return
        self._watch_sig, self._watch_written = sig, time.monotonic()
        with contextlib.suppress(Exception):
            atomic_write_text(self.plan.state_dir / WATCH_FILE, json.dumps(health, default=str) + "\n", mode=0o600)

    def health(self) -> dict[str, Any]:
        """What the supervisor observes (see the module docstring): ``phase`` (``hint_build``, ``first_segment``,
        ``starting``, ``producing``, ``idle``, ``retrying``, ``waiting_for_provider``, ``failing``, ...), ``setup``
        (first-time setup: stage, percent, ETA), ``segment`` (the one being made: slots, running time, ETA),
        ``segment_s`` (a whole segment at the last measured rate), ``stalled`` (why requests waiting for masks fail
        at once, else ``None``) and ``provider`` (reachability). Another process's factory: its
        ``supervisor.json`` while fresh, else ``{}``."""
        if self.owner:
            with self._wmu:
                return dict(self._health)
        return read_watch(self.plan.state_dir)

    def stop(self, timeout_s: float = 30.0) -> None:
        """Stop the child (SIGTERM, then SIGKILL) and release the owner lock (idempotent)."""
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=self.poll_s + 5)
        with self._mu:
            p, self._proc = self._proc, None
            if p is not None and p.poll() is None:
                with contextlib.suppress(OSError):
                    p.send_signal(signal.SIGTERM)
                try:
                    p.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=10)
            self._lock.release()

    def state(self) -> dict[str, Any]:
        """Factory state for status displays (``factory-state.json`` of the child, halt flag; no secrets)."""
        out: dict[str, Any] = {"owner": self.owner, "running": self.running, "starts": self.starts,
                               "consecutive_failures": self.failures, "last_child_error": self.last_error,
                               "workers": self.plan.workers, "scheme": self.plan.scheme,
                               "factory_weights": self.plan.factory_weights,
                               "target_slots": self.plan.target_slots, "low_water_slots": self.plan.low_water_slots}
        out.update(read_factory_state(self.plan.state_dir))
        out.pop("watch", None)
        out["nudges"] = self.nudges
        out["health"] = self.health()
        return out


def write_demand(path: Path, slots: int, ttl_s: float = 30.0) -> None:
    """Tell the factory a request waits for ``slots`` masks (it refills up to its target now, whatever its
    low-water mark). Best effort: a failed write (full disk) only means the factory refills on its own schedule."""
    from pgw.core.fileio import atomic_write_text

    with contextlib.suppress(Exception):
        atomic_write_text(path, json.dumps({"slots": int(slots), "until": time.time() + ttl_s}) + "\n", mode=0o600)


def _exit_reason(code: int | None, last_error: str | None) -> str:
    """The child's error line, or what its exit code says (a signal: SIGKILL is most often the out-of-memory
    killer)."""
    if last_error:
        return last_error
    if code is not None and code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        hint = " (often the out-of-memory killer: lower mask_workers or give the process more memory)" if (
            -code == signal.SIGKILL) else ""
        return f"killed by {name}{hint}"
    return f"exit code {code}"


def _last_error_line(log_file: Path, since: int = 0) -> str | None:
    """The last ``error [...]`` line the factory CLI printed after byte ``since`` of its log (this child's own
    output; its error messages carry no secrets)."""
    try:
        with log_file.open("rb") as fh:
            fh.seek(max(since, log_file.stat().st_size - 8192))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    lines = [ln.strip() for ln in tail.splitlines() if ln.startswith("error [")]
    return lines[-1][:500] if lines else None


def read_factory_state(state_dir: Path) -> dict[str, Any]:
    """Public fields of a factory state directory (``factory-state.json``, ``factory-halted.json``)."""
    out: dict[str, Any] = {}
    try:
        st = json.loads((state_dir / "factory-state.json").read_text(encoding="utf-8"))
        for k in ("totals", "provider_integrity_failures", "last_run_at", "last_error", "halted", "throughput",
                  "disk_limited", "in_progress"):
            if k in st:
                out[k] = st[k]
    except (OSError, ValueError):
        pass
    watch = read_watch(state_dir)
    if watch:
        out["watch"] = watch
    out["halted"] = bool(out.get("halted")) or (state_dir / "factory-halted.json").exists()
    stock = check_stock_state(state_dir)
    if stock:
        out["check_stock"] = stock
    return out


def read_watch(state_dir: Path, fresh_s: float = _WATCH_FRESH_S) -> dict[str, Any]:
    """The owner supervisor's ``supervisor.json`` (see :meth:`FactoryProcess.health`) while fresh, else ``{}``."""
    try:
        w = json.loads((state_dir / WATCH_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(w, dict) or time.time() - float(w.get("updated_at") or 0) > fresh_s:
        return {}
    return w


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # another user's process
    return True


def _minutes(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"


def describe_setup(health: dict[str, Any] | None) -> str | None:
    """One line on the model's first-time setup from :meth:`FactoryProcess.health`, e.g. ``"first-time setup in
    progress: building the one-time pks-v3 hint, 45% (about 9 min left)..."``; ``None`` when no setup runs."""
    su = (health or {}).get("setup")
    if not isinstance(su, dict):
        return None
    stage = su.get("stage")
    if stage == "hint":
        pct, eta = su.get("percent"), su.get("eta_s")
        what = "building the one-time pks-v3 hint" + (f", {pct:.0f}% done" if isinstance(pct, int | float) else "")
        what += f" (about {_minutes(float(eta))} left)" if isinstance(eta, int | float) else ""
        what += ", then the first mask segment"
    elif stage == "first_segment":
        what = (f"making the first mask segment ({su.get('slots') or '?'} slots, running for "
                f"{_minutes(float(su.get('running_s') or 0))})")
    elif stage == "first_segment_retry":
        what = "the first mask segment failed and is retried with back-off"
    else:
        what = "starting the mask factory (loading the weights and the one-time setup)"
    return f"first-time setup in progress: {what}"


def check_stock_state(state_dir: Path) -> dict[str, Any] | None:
    """The weight-free factory's check stock (``factory_weights="stream"``): counts, passes, the last pass and its
    error, from ``<state_dir>/check-stock/*/status.json`` (public counters; no secrets), or ``None``."""
    from pgw.factory.checkstock import stock_status

    root = state_dir / "check-stock"
    if not root.is_dir():
        return None
    stocks = {d.name: stock_status(d) for d in sorted(root.iterdir()) if d.is_dir()}
    stocks = {k: v for k, v in stocks.items() if v}
    if not stocks:
        return None
    if len(stocks) == 1:
        return next(iter(stocks.values()))
    return {"stocks": stocks}


def wait_for(predicate: Any, timeout_s: float, poll_s: float = 0.2) -> bool:
    """Poll ``predicate()`` until true or ``timeout_s`` passed."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return bool(predicate())
