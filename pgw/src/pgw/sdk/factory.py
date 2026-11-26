"""The background mask factory of the client library: one supervised child process per model and key.

The child runs the existing factory (:mod:`pgw.factory`: scheduler, fleet of ``mask_workers`` worker processes,
the pks-v3 / mlwe encrypted backends or local generation, the 40-bit self-check of provider products, the halt
after repeated provider integrity failures) at low CPU priority (``nice``), with ``torch`` threads capped, and
keeps the model's mask stock between the low-water mark and the target.

Several processes on one host that use the same key and cache directory share one mask store and one factory
per model: the factory is run by whichever process holds ``<factory dir>/owner.lock``; the others retry the lock
every few seconds and take over when its holder exits. The child exits with its parent (``PR_SET_PDEATHSIG``).
Its output goes to ``<factory dir>/factory.log`` (nothing on the application's stdout or stderr).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
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

__all__ = ["FactoryPlan", "FactoryProcess", "factory_env", "write_demand"]

log = logging.getLogger(__name__)

_LOG_MAX_BYTES = 4 << 20


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

    def __init__(self, plan: FactoryPlan, *, poll_s: float = 3.0, hang_s: float | None = None):
        self.plan = plan
        self.poll_s = poll_s
        #: a running child whose heartbeat is older than this is killed and restarted (``PGW_SDK_FACTORY_HANG_S``)
        self.hang_s = float(hang_s if hang_s is not None else os.environ.get("PGW_SDK_FACTORY_HANG_S", "") or 300.0)
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
            self._proc = subprocess.Popen(
                list(self.command), env=_child_env(self.plan),
                stdin=subprocess.DEVNULL, stdout=fd, stderr=fd, close_fds=True, cwd=str(self.plan.state_dir))
        finally:
            os.close(fd)
        self.starts += 1
        self._started_at = time.monotonic()
        log.info("mask factory for %s started (pid %d, %d worker(s), scheme %s)", self.plan.model, self._proc.pid,
                 self.plan.workers, self.plan.scheme)

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
                  "disk_limited"):
            if k in st:
                out[k] = st[k]
    except (OSError, ValueError):
        pass
    out["halted"] = bool(out.get("halted")) or (state_dir / "factory-halted.json").exists()
    stock = check_stock_state(state_dir)
    if stock:
        out["check_stock"] = stock
    return out


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
