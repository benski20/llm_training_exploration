"""Where the client library keeps its files, and the locks that let several processes share them.

Two private (0700) directories:

* the **cache** (``~/.cache/pgw``; ``PGW_CACHE_DIR`` or ``cache_dir=``): provisioned model artifacts (public,
  fingerprint-checked), per-key session files (service token and client key 0600, client certificate, CA),
  the encrypted mask store, factory state and logs. Deleting it is safe: the library enrolls and provisions
  again (masks in it are lost).
* the **state** directory (``$XDG_STATE_HOME/pgw``, default ``~/.local/state/pgw``; ``PGW_STATE_DIR`` or
  ``state_dir=``): small secrets that must not live next to the masks: the mask store's key-encryption key,
  its monotonic rollback counter (SPEC 7.6: restoring an old copy of the mask store must not re-serve used
  masks) and the pks-v3 hint key.

Locks are ``flock`` locks on files inside these directories (local filesystems only, as the mask store
requires): every process on the host that uses the same directories coordinates through them.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import stat
import time
from collections.abc import Iterator
from pathlib import Path

from pgw.core.errors import ConfigError

__all__ = ["CacheLayout", "default_cache_dir", "default_state_dir", "ensure_private_dir", "file_lock"]


def default_cache_dir() -> Path:
    """``PGW_CACHE_DIR``, else ``$XDG_CACHE_HOME/pgw``, else ``~/.cache/pgw``."""
    env = os.environ.get("PGW_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "pgw"


def default_state_dir() -> Path:
    """``PGW_STATE_DIR``, else ``$XDG_STATE_HOME/pgw``, else ``~/.local/state/pgw``."""
    env = os.environ.get("PGW_STATE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state") / "pgw"


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) and make it 0700; refuse a directory owned by another user or a symlink.

    Raises:
        ConfigError: not a directory, owned by someone else, or not writable.
    """
    path = Path(path).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        st = os.lstat(path)
    except OSError as exc:
        raise ConfigError(f"cannot create the directory {path}: {exc.strerror or type(exc).__name__}") from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ConfigError(f"{path} must be a real directory (not a symlink or a file)")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise ConfigError(f"{path} belongs to another user; set a private directory (PGW_CACHE_DIR / PGW_STATE_DIR)")
    if stat.S_IMODE(st.st_mode) != 0o700:
        try:
            os.chmod(path, 0o700)
        except OSError as exc:
            raise ConfigError(f"cannot make {path} private (0700): {exc.strerror}") from exc
    return path


@contextlib.contextmanager
def file_lock(path: Path, *, shared: bool = False, timeout_s: float | None = None,
              poll_s: float = 0.05) -> Iterator[None]:
    """``flock`` on ``path`` (created 0600). ``timeout_s=None`` waits forever; else raises :class:`TimeoutError`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600)
    op = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    try:
        if timeout_s is None:
            fcntl.flock(fd, op)
        else:
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    fcntl.flock(fd, op | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out waiting for the lock {path.name}") from None
                    time.sleep(poll_s)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class HeldLock:
    """A non-blocking exclusive ``flock`` held until :meth:`release` (e.g. "this process runs the factory")."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def try_acquire(self) -> bool:
        """Take the lock if free; ``True`` when this object holds it."""
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                     0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    @property
    def held(self) -> bool:
        """Whether this object holds the lock."""
        return self._fd is not None

    def release(self) -> None:
        """Release (idempotent)."""
        fd, self._fd = self._fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


class CacheLayout:
    """Paths of the library's files (directories are created 0700 on first use).

    Args:
        cache_dir: The cache root (default :func:`default_cache_dir`).
        state_dir: The state root (default :func:`default_state_dir`).
    """

    def __init__(self, cache_dir: Path | str | None = None, state_dir: Path | str | None = None):
        self.root = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
        self.state = Path(state_dir).expanduser() if state_dir is not None else default_state_dir()

    def ensure(self) -> CacheLayout:
        """Create both roots (0700)."""
        ensure_private_dir(self.root)
        ensure_private_dir(self.state)
        return self

    # ---- per SDK key -----------------------------------------------------------------------------
    def session_dir(self, key_id: str) -> Path:
        """Enrollment files of one key (token and client key under ``secrets/``, 0600)."""
        return self.root / "sessions" / key_id

    def session_state_dir(self, key_id: str) -> Path:
        """Secrets of one key's mask store (KEK, rollback counter, pks-v3 hint key)."""
        return self.state / "sessions" / key_id

    def masks_dir(self, key_id: str) -> Path:
        """The key's encrypted mask store."""
        return self.session_dir(key_id) / "masks"

    def factory_dir(self, key_id: str) -> Path:
        """The key's mask factory state (locks, ledger, ``factory-state.json``, log)."""
        return self.session_dir(key_id) / "factory"

    # ---- shared --------------------------------------------------------------------------------------
    @property
    def models_dir(self) -> Path:
        """Provisioned model artifacts (public; checked against the pack fingerprint), shared by every key."""
        return self.root / "models"

    def lock_path(self, name: str) -> Path:
        """A lock file under ``<cache>/locks``."""
        return self.root / "locks" / f"{name}.lock"
