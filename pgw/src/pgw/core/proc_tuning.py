"""Process runtime tuning shared by the service, gateway and factory entry points.

* :func:`apply_malloc_tuning` raises glibc's trim threshold, top pad and mmap threshold so the
  large per-call buffers of the hot path (masked rows, GEMM inputs and outputs, msgpack payloads:
  hundreds of KiB to tens of MiB) are reused from the heap instead of being mapped, zero-filled and
  unmapped on every call. The cost is a higher resident floor (up to about 320 MiB per process).
  The same values can be set before start-up with the ``MALLOC_TRIM_THRESHOLD_``,
  ``MALLOC_TOP_PAD_`` and ``MALLOC_MMAP_THRESHOLD_`` environment variables (the Dockerfiles and
  systemd units do); calling ``mallopt`` as well is harmless. ``PGW_MALLOC_TUNING=0`` turns the
  call off; it does nothing on a C library other than glibc.
* :func:`new_event_loop` returns a uvloop loop when uvloop is importable (cheaper callbacks,
  transports and timers on the request path), else the standard asyncio loop.
  ``PGW_UVLOOP=0`` forces the standard loop. Use it as ``asyncio.run(main(), loop_factory=new_event_loop)``.

* :func:`os_thread_name` (``PGW_THREAD_NAMES=1``, default off): copies the calling Python thread's name
  into the kernel's thread name (``/proc/<pid>/task/<tid>/comm``, 15 bytes), so per-thread CPU time can be
  attributed to a stage (diagnostics: the encrypted factory's stage threads call it).

None of these changes any computed value: they only change how memory is obtained and how the event loop
dispatches callbacks.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
from collections.abc import Callable
from typing import Any

__all__ = [
    "MALLOC_ENV",
    "M_MMAP_THRESHOLD",
    "M_TOP_PAD",
    "M_TRIM_THRESHOLD",
    "apply_malloc_tuning",
    "glibc_version",
    "loop_name",
    "new_event_loop",
    "os_thread_name",
]

log = logging.getLogger(__name__)

#: ``mallopt`` parameter numbers (glibc ``malloc.h``).
M_TRIM_THRESHOLD = -1
M_TOP_PAD = -2
M_MMAP_THRESHOLD = -3

TRIM_THRESHOLD_BYTES = 256 << 20
TOP_PAD_BYTES = 64 << 20
MMAP_THRESHOLD_BYTES = 32 << 20

#: Environment equivalents (read by glibc at process start; set in deploy/docker and deploy/systemd).
MALLOC_ENV: dict[str, str] = {
    "MALLOC_TRIM_THRESHOLD_": str(TRIM_THRESHOLD_BYTES),
    "MALLOC_TOP_PAD_": str(TOP_PAD_BYTES),
    "MALLOC_MMAP_THRESHOLD_": str(MMAP_THRESHOLD_BYTES),
}

_FALSE = frozenset({"0", "false", "no", "off"})
_applied: bool | None = None


def _disabled(var: str) -> bool:
    return os.environ.get(var, "").strip().lower() in _FALSE


def _libc() -> Any | None:
    try:
        return ctypes.CDLL(None)
    except OSError:  # pragma: no cover - no C library handle (unusual platforms)
        return None


def glibc_version(libc: Any | None = None) -> str | None:
    """glibc version string, or ``None`` when the process does not run on glibc (musl, macOS, Windows)."""
    lib = libc if libc is not None else _libc()
    if lib is None:
        return None
    fn = getattr(lib, "gnu_get_libc_version", None)
    if fn is None:
        return None
    try:
        fn.restype = ctypes.c_char_p
        v = fn()
    except Exception:  # pragma: no cover - defensive
        return None
    return v.decode("ascii", "replace") if isinstance(v, bytes) else None


def apply_malloc_tuning(*, libc: Any | None = None, force: bool = False) -> bool:
    """Set glibc's trim threshold (256 MiB), top pad (64 MiB) and mmap threshold (32 MiB) together.

    Idempotent (applied once per process unless ``force``). Does nothing and returns ``False``
    when ``PGW_MALLOC_TUNING=0`` or the C library is not glibc.

    Args:
        libc: C library handle (tests); default: the process's own.
        force: Apply again even if already applied.

    Returns:
        True if all three ``mallopt`` calls succeeded.
    """
    global _applied
    if _applied is not None and not force:
        return _applied
    if _disabled("PGW_MALLOC_TUNING"):
        _applied = False
        return False
    lib = libc if libc is not None else _libc()
    if lib is None or glibc_version(lib) is None:
        _applied = False
        return False
    mallopt = getattr(lib, "mallopt", None)
    if mallopt is None:  # pragma: no cover - every glibc has it
        _applied = False
        return False
    ok = True
    for param, value in ((M_TRIM_THRESHOLD, TRIM_THRESHOLD_BYTES), (M_TOP_PAD, TOP_PAD_BYTES),
                         (M_MMAP_THRESHOLD, MMAP_THRESHOLD_BYTES)):
        try:
            ok = bool(mallopt(ctypes.c_int(param), ctypes.c_int(value))) and ok
        except Exception:  # pragma: no cover - defensive
            ok = False
    if ok:
        log.debug("malloc tuning applied (trim %d, top pad %d, mmap %d bytes)",
                  TRIM_THRESHOLD_BYTES, TOP_PAD_BYTES, MMAP_THRESHOLD_BYTES)
    else:  # pragma: no cover - glibc returns 1 for these parameters
        log.warning("malloc tuning: mallopt rejected a parameter")
    _applied = ok
    return ok


def _uvloop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    if _disabled("PGW_UVLOOP"):
        return None
    try:
        import uvloop
    except ImportError:
        return None
    return uvloop.new_event_loop


def new_event_loop() -> asyncio.AbstractEventLoop:
    """A new event loop: uvloop when importable (and ``PGW_UVLOOP`` is not 0), else asyncio's."""
    factory = _uvloop_factory()
    return factory() if factory is not None else asyncio.new_event_loop()


def loop_name() -> str:
    """``"uvloop"`` or ``"asyncio"``: what :func:`new_event_loop` returns in this process."""
    return "uvloop" if _uvloop_factory() is not None else "asyncio"


_PR_SET_NAME = 15


def os_thread_name(name: str | None = None) -> bool:
    """With ``PGW_THREAD_NAMES=1``: set the kernel name of the calling thread to ``name`` (default: the Python
    thread's name, shortened: ``pgw-`` dropped, ``hemask-`` -> ``he-``, ``factory-`` -> ``fac-``, the last 15
    bytes kept). Diagnostics only; returns whether it was set."""
    if os.environ.get("PGW_THREAD_NAMES", "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    import threading

    short = (name or threading.current_thread().name).removeprefix("pgw-").replace("hemask-", "he-")
    raw = short.replace("factory-", "fac-").encode(errors="replace")[-15:]
    lib = _libc()
    if lib is None:  # pragma: no cover - unusual platforms
        return False
    try:
        return int(lib.prctl(_PR_SET_NAME, ctypes.c_char_p(raw), 0, 0, 0)) == 0
    except (AttributeError, OSError):  # pragma: no cover - no prctl (not Linux)
        return False
