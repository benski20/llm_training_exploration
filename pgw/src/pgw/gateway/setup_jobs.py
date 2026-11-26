"""Background work of the setup wizard: model pack build, encrypted-mask hint build, mask factory.

* :class:`PackBuildJob` downloads the open-weight checkpoint (HF hub by default; the source is
  injectable), builds the pack with :func:`pgw.modelpack.pack_checkpoint` exactly as the service did
  (name, revision, dtype, license) and refuses the result unless its fingerprint equals the one the
  service serves.
* :class:`HintBuildJob` builds the one-time hint ``W8 A`` of encrypted mask generation
  (:class:`pgw.hemask.backend.HintStore`) so the factory can start producing at once.
* :class:`FactorySupervisor` runs ``pgw factory run`` (encrypted mode) as a child process and
  restarts it with back-off when it exits.

Progress is kept in memory (``status()``) and polled by the console.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError, PGWError

__all__ = [
    "CheckpointResolver", "FactorySupervisor", "HintBuildJob", "PackBuildJob", "hf_hub_resolver", "hint_params",
    "mask_estimate",
]

log = logging.getLogger(__name__)

#: ``(hf_repo, revision) -> local checkpoint directory`` (or anything :func:`pack_checkpoint` accepts).
CheckpointResolver = Callable[[str, str | None], "str | Path"]

#: Typical figures shown before the model's own shapes are known (docs/operations/encrypted-masks.md).
TYPICAL_7B_HINT_BYTES = 34_200_000_000
HE_RANK = 16


def hf_hub_resolver(repo: str, revision: str | None) -> Path:
    """Download the checkpoint files from the Hugging Face hub (default resolver)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency of the pack extra
        raise ConfigError("huggingface_hub is required to download models (pip install 'pgw-sdk[pack]')") from exc
    from pgw.modelpack.packer import _HUB_PATTERNS

    return Path(snapshot_download(repo, revision=revision, allow_patterns=_HUB_PATTERNS))


def hint_params(ring: str) -> tuple[int, int]:
    """``(rank, q_bits)`` of the hint the factory uses for ``ring`` (same rule as the factory)."""
    from pgw.core.ring import RingParams
    from pgw.hemask.lwe import HEParams

    return HE_RANK, HEParams(k=RingParams.from_name(ring).k, rank=HE_RANK).q


def mask_estimate(pack_path: Path | None, ring: str, buffer_slots: int) -> dict[str, Any]:
    """Disk needed by encrypted mask generation: the one-time hint plus the mask buffer.

    Without a pack yet, returns the typical 7B figure (about 34 GB of hint, see encrypted-masks.md).
    """
    from pgw.core.ring import RingParams

    k = RingParams.from_name(ring).k
    if pack_path is None or not (Path(pack_path) / "manifest.json").is_file():
        return {"basis": "typical_7b", "hint_bytes": TYPICAL_7B_HINT_BYTES, "buffer_bytes": None,
                "total_bytes": None, "buffer_slots": buffer_slots}
    man = json.loads((Path(pack_path) / "manifest.json").read_text(encoding="utf-8"))
    m_total = sum(int(e["out_features"]) for e in man["linears"])
    rank, q_bits = hint_params(ring)
    hint = m_total * (q_bits // 8 - 1) * 256 * rank
    buffer = m_total * (k // 8) * buffer_slots
    return {"basis": "model", "hint_bytes": hint, "buffer_bytes": buffer, "total_bytes": hint + buffer,
            "buffer_slots": buffer_slots, "mask_bytes_per_token": m_total * (k // 8)}


def disk_free(path: Path) -> int | None:
    """Free bytes on the volume holding ``path`` (or its nearest existing parent)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


class _Job(threading.Thread):
    """Thread with a status dict: ``{phase, done, total, message, error}``."""

    def __init__(self, name: str):
        super().__init__(name=name, daemon=True)
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {"phase": "queued", "done": 0, "total": 0, "message": "", "error": None,
                                        "started_at": time.time(), "finished_at": None}

    def update(self, **kw: Any) -> None:
        with self._lock:
            self._status.update(kw)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    @property
    def finished(self) -> bool:
        return self.status()["phase"] in ("done", "failed")

    def _fail(self, exc: BaseException) -> None:
        msg = exc.message if isinstance(exc, PGWError) else f"{type(exc).__name__}: {exc}"
        log.error("%s failed: %s", self.name, msg)
        self.update(phase="failed", error=msg, finished_at=time.time())


class PackBuildJob(_Job):
    """Download + pack + fingerprint check for one enrolled model.

    Args:
        model: Enrolled model info (``name``, ``fingerprint``, ``hf_repo``, ``revision``, ``plain_dtype``,
            ``license``, ``pack_name``).
        out_dir: Pack directory to create.
        resolver: Checkpoint source (default: Hugging Face hub).
        on_done: Called with the final status from the job thread.
    """

    def __init__(self, model: dict[str, Any], out_dir: Path, resolver: CheckpointResolver | None = None,
                 on_done: Callable[[dict[str, Any]], None] | None = None):
        super().__init__(f"pgw-setup-pack-{model.get('name')}")
        self.model = model
        self.out_dir = Path(out_dir)
        self.resolver = resolver or hf_hub_resolver
        self.on_done = on_done

    def run(self) -> None:
        from pgw.modelpack import ModelPack, pack_checkpoint

        m = self.model
        try:
            if self._reuse_existing():
                return
            repo = m.get("hf_repo")
            if not repo:
                raise ConfigError("the service did not say which open-weight checkpoint this model uses; ask your "
                                  "provider to re-issue the enrollment code")
            self.update(phase="downloading", message=f"Downloading {repo}")
            src = self.resolver(str(repo), m.get("revision"))
            total = 0
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                cfg = json.loads((Path(src) / "config.json").read_text(encoding="utf-8"))
                total = 4 * int(cfg["num_hidden_layers"])
            self.update(phase="packing", done=0, total=total, message="Preparing the model for private inference")
            done = [0]

            def progress(_: str) -> None:
                done[0] += 1
                self.update(done=done[0])

            self.out_dir.parent.mkdir(parents=True, exist_ok=True)
            man = pack_checkpoint(src, self.out_dir, name=m.get("pack_name") or m["name"], revision=m.get("revision"),
                                  plain_dtype=m.get("plain_dtype") or "source", license=m.get("license"),
                                  overwrite=True, progress=progress)
            self.update(phase="verifying", message="Checking the model fingerprint against the service")
            with ModelPack.open(self.out_dir, verify="full") as p:
                fp = p.fingerprint
            if fp != m["fingerprint"] or man.fingerprint != fp:
                shutil.rmtree(self.out_dir, ignore_errors=True)
                raise ConfigError(
                    f"the model built here (fingerprint {fp[:12]}) is not the one the service runs "
                    f"({str(m['fingerprint'])[:12]}); the pack was deleted. Ask your provider which checkpoint "
                    "revision they serve")
            self.update(phase="done", done=total or done[0], message="Model ready", finished_at=time.time(),
                        fingerprint=fp)
        except BaseException as exc:
            self._fail(exc)
        finally:
            if self.on_done is not None:
                try:
                    self.on_done(self.status())
                except Exception:  # pragma: no cover - defensive
                    log.exception("pack job callback failed")

    def _reuse_existing(self) -> bool:
        """``PGW_SETUP_REUSE_PACK=1`` (off by default): a pack already in ``out_dir`` (e.g. restored from a
        deployment's cache of public packs) is used without downloading when its full verification (every tensor,
        the tokenizer and the linears hashed) gives the enrolled fingerprint; otherwise it is deleted and the model
        is downloaded and packed as usual. The fingerprint is what the service serves, so a reused pack is the
        same model a fresh build would give."""
        from pgw.modelpack import ModelPack

        if os.environ.get("PGW_SETUP_REUSE_PACK", "").strip().lower() not in ("1", "true", "yes", "on"):
            return False
        if not (self.out_dir / "manifest.json").is_file():
            return False
        self.update(phase="verifying", message="Checking the prepared model against the service's fingerprint")
        try:
            with ModelPack.open(self.out_dir, verify="full") as p:
                fp = p.fingerprint
        except Exception as exc:  # unreadable or tampered: build from the checkpoint instead
            log.warning("prepared pack in %s is not usable (%s): building it from the checkpoint", self.out_dir, exc)
            fp = None
        if fp != self.model["fingerprint"]:
            if fp is not None:
                log.warning("prepared pack in %s has fingerprint %s, not the enrolled %s: building it from the "
                            "checkpoint", self.out_dir, str(fp)[:12], str(self.model["fingerprint"])[:12])
            shutil.rmtree(self.out_dir, ignore_errors=True)
            return False
        log.info("reusing the prepared pack in %s (full verification: fingerprint %s)", self.out_dir, fp[:12])
        self.update(phase="done", message="Model ready", finished_at=time.time(), fingerprint=fp)
        return True


class HintBuildJob(_Job):
    """Build (or find) the one-time hint of encrypted mask generation for a pack and ring."""

    def __init__(self, pack_path: Path, ring: str, hint_root: Path,
                 on_done: Callable[[dict[str, Any]], None] | None = None):
        super().__init__("pgw-setup-hint")
        self.pack_path = Path(pack_path)
        self.ring = ring
        self.hint_root = Path(hint_root)
        self.on_done = on_done

    def run(self) -> None:
        from pgw.hemask.backend import HintStore
        from pgw.modelpack import ModelPack

        try:
            rank, q_bits = hint_params(self.ring)
            with ModelPack.open(self.pack_path, verify="fingerprint") as pack:
                if HintStore.open(self.hint_root, pack.fingerprint, rank, q_bits) is None:
                    total = len(list(pack.linear_specs()))
                    final = HintStore.dir_for(self.hint_root, pack.fingerprint, rank, q_bits)
                    self.update(phase="building", total=total, message="Building the one-time encryption hint")
                    stop = threading.Event()

                    def watch() -> None:  # progress = finished linear files in the temporary build directory
                        while not stop.wait(1.0):
                            n = sum(len(list(d.glob("*.npy"))) for d in final.parent.glob(final.name + ".building-*"))
                            self.update(done=max(0, n - 1))

                    w = threading.Thread(target=watch, daemon=True)
                    w.start()
                    gpu = self._gpu_planes(pack)
                    try:
                        t0 = time.monotonic()
                        HintStore.build(self.hint_root, pack, rank, q_bits=q_bits, planes_fn=gpu)
                        log.info("encryption hint built on the %s in %.0f s", "GPU" if gpu is not None else "CPU",
                                 time.monotonic() - t0)
                    finally:
                        stop.set()
                        if gpu is not None:
                            gpu.release()
                    self.update(done=total)
            self.update(phase="done", message="Encryption hint ready", finished_at=time.time())
        except BaseException as exc:
            self._fail(exc)
        finally:
            if self.on_done is not None:
                with contextlib.suppress(Exception):
                    self.on_done(self.status())


    @staticmethod
    def _gpu_planes(pack: Any) -> Any:
        """``PGW_HINT_DEVICE=cuda`` (off by default): the hint's GEMMs on the gateway's own GPU
        (:class:`pgw.hemask.hint_gpu.GpuHintPlanes`, the same bits), after a self-check against the CPU path on
        the pack's linear shapes; ``None`` (the CPU path) when unset, unavailable or the check fails."""
        dev = os.environ.get("PGW_HINT_DEVICE", "").strip().lower()
        if dev in ("", "cpu", "0", "off"):
            return None
        try:
            from pgw.hemask.hint_gpu import GpuHintPlanes

            gpu = GpuHintPlanes(dev)
        except Exception as exc:
            log.warning("PGW_HINT_DEVICE=%s unavailable (%s); building the hint on the CPU", dev, exc)
            return None
        shapes = sorted({(int(sp.out_features), int(sp.in_features)) for sp in pack.linear_specs()})
        if not gpu.self_check(shapes):
            return None
        log.info("encryption hint: GEMMs on %s (self-check passed on %d linear shapes)", dev, len(shapes))
        return gpu


class FactorySupervisor:
    """Runs ``pgw factory run`` with ``env`` as a child process; restarts it with back-off when it exits.

    Args:
        env: ``PGW_FACTORY_*`` / ``PGW_MASKSTORE_*`` settings (added to the inherited environment).
        argv: Command (default: this interpreter running ``pgw factory run``).
    """

    def __init__(self, env: dict[str, str], argv: list[str] | None = None, *, max_backoff_s: float = 300.0):
        self.env = dict(env)
        self.argv = argv or [sys.executable, "-m", "pgw.cli", "factory", "run"]
        self.max_backoff_s = max_backoff_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.restarts = 0
        self.last_exit: int | None = None
        self.started_at: float | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="pgw-factory-supervisor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        backoff = 5.0
        env = {**os.environ, **self.env}
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._proc = subprocess.Popen(self.argv, env=env, stdin=subprocess.DEVNULL)  # noqa: S603
            except OSError as exc:
                log.error("cannot start the mask factory: %s", exc.strerror)
                self.last_exit = -1
            else:
                self.started_at = time.time()
                log.info("mask factory started (pid %d)", self._proc.pid)
                self.last_exit = self._proc.wait()
                log.warning("mask factory exited with status %s", self.last_exit)
            if self._stop.is_set():
                break
            if time.monotonic() - t0 > 600:
                backoff = 5.0
            self.restarts += 1
            self._stop.wait(backoff)
            backoff = min(self.max_backoff_s, backoff * 2)

    def stop(self, timeout_s: float = 30.0) -> None:
        self._stop.set()
        p = self._proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout_s)
            except subprocess.TimeoutExpired:
                p.kill()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def status(self) -> dict[str, Any]:
        return {"running": self.running, "restarts": self.restarts, "last_exit": self.last_exit,
                "started_at": self.started_at}


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines (``#`` comments, blank lines ignored; no shell expansion)."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out
