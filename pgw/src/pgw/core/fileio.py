"""Small file helpers for secrets and configuration files (mode 0600, no symlinks, atomic replace)."""

from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator
from pathlib import Path

from pgw.core.errors import ConfigError

__all__ = ["atomic_write_text", "file_lock", "write_private"]


def write_private(path: Path, text: str, *, append: bool) -> None:
    """Write ``text`` to ``path`` with mode 0600, refusing symlinks and tightening a looser existing mode.

    Raises:
        ConfigError: the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ConfigError(f"cannot write {path}: {exc.strerror}") from exc
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "a" if append else "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Replace ``path`` atomically (temp file in the same directory, fsync, rename, fsync dir).

    Raises:
        ConfigError: the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), mode)
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError as exc:
        raise ConfigError(f"cannot write {path}: {exc.strerror or type(exc).__name__}") from exc


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive inter-process lock on ``path`` (created 0600 if absent; ``flock``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
