"""In-memory mask ring: hand masks from a co-located factory to the gateway through locked RAM.

Design: ``research/plans/mask-ringbuffer-design.md`` (WP8, plan item L5), including the design-review
fixes listed there in section 15. In short:

* One :class:`MaskRing` per masked model ``(fingerprint, ring)``. Its bytes live in one
  :class:`LockedArena`: an anonymous private ``mmap`` that is ``mlock``-ed, ``MADV_DONTDUMP`` and
  ``MADV_DONTFORK`` (all fail closed). The arena is split into fixed-size *slabs*; each slab holds one
  RAM segment laid out like the plaintext of a ``.pgwm`` file (seed, ``uW`` rows, ``W r`` rows).
* The factory fills a ``FREE`` slab through :class:`RingSegmentWriter` (the ``SegmentWriter``
  contract) and publishes it (``READY``). The gateway reserves slot ranges oldest first with cursors
  that only move forward (:meth:`MaskRing.reserve`) and gets an ordinary
  :class:`pgw.maskstore.MaskReservation` whose segment objects are :class:`RamSegment` handles.
* A slab is wiped and reused only after every slot was reserved and every handle on it was closed.

Single use (SPEC 3.3) rests on these rules, all enforced here:

R1  Only ``FREE`` slabs are granted; only ``WIPING`` becomes ``FREE``, after the wipe returned.
R2  ``reserve()`` only reads ``READY`` slabs and moves ``next_slot`` forward under the ring lock,
    before any byte is copied out. A fully reserved slab is ``DRAINING`` and never ``READY`` again.
R3  A slab with open handles is never wiped or granted (except by :meth:`MaskRing.close`, see R4).
R4  (primary guard) Every read (:meth:`RamSegment.decrypt_into`) and every factory write checks,
    under the slab's own lock, that the ring is open, the slab generation matches the handle and the
    state allows the access, and copies while still holding that lock. Every transition out of
    ``READY``/``DRAINING``/``FILLING`` takes the same lock, and :meth:`MaskRing.close` marks every
    slab ``WIPING`` before anything is wiped, so a late reader always raises and never sees zeros or
    another segment's masks.

Further guards from the design review:

* The secrets read verifies ``seed_hash(seed)`` against the hash recorded at commit and refuses an
  all-zero seed (``r`` from a zero seed is public).
* :meth:`RamSegment.close` takes no lock (it may run from a GC finalizer while this thread holds the
  ring lock); it queues the handle on a :class:`queue.SimpleQueue` that the ring reconciles.
* Copy-out goes straight from the arena into a fresh ``bytearray`` (``ctypes.memmove``): no immutable
  ``bytes`` copy of masks or seeds is ever made by this module.
* The arena's buffer export lives as long as the arena; :meth:`MaskRing.close` joins the wiper and
  wipes the whole arena synchronously before ``munlock``/``munmap``.

Nothing in this module writes a ring byte to a file, a socket or the disk store.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import errno
import hashlib
import hmac
import logging
import mmap
import os
import queue
import resource
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, MaskExhaustedError, MaskReuseError, MaskStoreError, ShapeError
from pgw.core.fingerprint import short_fingerprint, validate_fingerprint
from pgw.core.prng import SEED_BYTES, max_stream_rows
from pgw.core.ring import RingParams
from pgw.core.settings import ring_from_setting
from pgw.core.types import LinearId, LinearSpec
from pgw.maskstore.format import ChunkRef, SegmentHeader, SegmentLayout, SegmentLinear
from pgw.maskstore.store import MaskReservation, SlotExtent

__all__ = [
    "DEFAULT_MAX_CHECKS",
    "LockedArena",
    "MaskRing",
    "RamSegment",
    "RingFull",
    "RingSegmentWriter",
    "RingShort",
    "SeedRegistry",
    "SlabState",
    "harden_process",
    "init_on_free_status",
    "raise_memlock_limit",
    "ring_geometry",
    "seed_digest",
    "swap_status",
]

log = logging.getLogger(__name__)

#: Freivalds vectors per linear a slab has room for (secrets region sized for this many).
DEFAULT_MAX_CHECKS = 8
#: Wipes run in pieces of this size (each piece is one ``explicit_bzero`` call, GIL released).
WIPE_PIECE_BYTES = 64 << 20
_PAGE = mmap.PAGESIZE
_ALIGN = 64
_WIPER_POLL_S = 0.25
_SEED_ID_PREFIX = b"pgw/v1/seed-id\0"  # pgw.maskstore.format.seed_hash domain separator

_PR_SET_DUMPABLE = 4
_PR_GET_DUMPABLE = 3


# =================================================================================================
# errors
# =================================================================================================
class RingShort(MaskExhaustedError):  # noqa: N818 - reads as a state, like MaskExhausted
    """The ring holds fewer unreserved slots than requested (``details["available"]``)."""


class RingFull(MaskStoreError):  # noqa: N818
    """No ``FREE`` slab can be granted right now (the factory re-plans; not an error)."""


# =================================================================================================
# libc helpers (ctypes calls release the GIL)
# =================================================================================================
_libc: Any = None


def _lib() -> Any:
    global _libc
    if _libc is None:
        name = ctypes.util.find_library("c") or "libc.so.6"
        lib = ctypes.CDLL(name, use_errno=True)
        for fn in ("mlock", "munlock"):
            f = getattr(lib, fn)
            f.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            f.restype = ctypes.c_int
        try:
            lib.explicit_bzero.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.explicit_bzero.restype = None
        except AttributeError:  # pragma: no cover - old libc: memset through ctypes
            pass
        lib.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        lib.prctl.restype = ctypes.c_int
        lib.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        lib.mmap.restype = ctypes.c_void_p
        lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.munmap.restype = ctypes.c_int
        lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.madvise.restype = ctypes.c_int
        _libc = lib
    return _libc


def _bzero(addr: int, n: int) -> None:
    lib = _lib()
    fn = getattr(lib, "explicit_bzero", None)
    if fn is not None:
        fn(addr, n)
    else:  # pragma: no cover
        ctypes.memset(addr, 0, n)


def seed_digest(seed: bytes | bytearray | memoryview) -> str:
    """:func:`pgw.maskstore.format.seed_hash` without an immutable copy of the seed."""
    h = hashlib.sha256(_SEED_ID_PREFIX)
    h.update(memoryview(seed)[:SEED_BYTES])
    return h.hexdigest()


def _all_zero(buf: bytes | bytearray | memoryview) -> bool:
    a = np.frombuffer(buf, dtype=np.uint8)
    z = not bool(a.any())
    del a
    return z


def _zero(buf: bytearray) -> None:
    if len(buf):
        a = np.frombuffer(buf, dtype=np.uint8)
        a.fill(0)
        del a


# =================================================================================================
# process hardening (mandatory when the ring is enabled)
# =================================================================================================
def harden_process() -> None:
    """``prctl(PR_SET_DUMPABLE, 0)``: no core dumps, no same-uid ptrace or ``/proc/<pid>/mem`` reads.

    The arena is ``DONTDUMP`` itself, but reservation chunk caches, prefetch stages and returned
    tensors live on the ordinary heap; only a non-dumpable process keeps those out of dumps and away
    from other processes of the same uid. Fails closed.
    """
    lib = _lib()
    if lib.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        e = ctypes.get_errno()
        raise ConfigError(f"prctl(PR_SET_DUMPABLE, 0) failed: {os.strerror(e)}")
    if lib.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:  # pragma: no cover - kernel refused silently
        raise ConfigError("process is still dumpable after PR_SET_DUMPABLE=0")


def swap_status(proc_swaps: str = "/proc/swaps", sys_block: str = "/sys/block") -> dict[str, Any]:
    """Active swap areas and whether each is RAM-backed (zram) or encrypted (dm-crypt).

    Returns ``{"active": [...], "unprotected": [...]}``; ``unprotected`` lists swap areas that could
    write heap copies of masks to disk in the clear.
    """
    active: list[str] = []
    unprotected: list[str] = []
    try:
        with open(proc_swaps, encoding="ascii") as f:
            lines = f.read().splitlines()[1:]
    except OSError:
        return {"active": [], "unprotected": [], "unknown": True}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        dev = parts[0]
        active.append(dev)
        base = os.path.basename(os.path.realpath(dev)) if dev.startswith("/dev/") else ""
        if base.startswith("zram"):
            continue
        if base.startswith("dm-"):
            try:
                with open(os.path.join(sys_block, base, "dm", "uuid"), encoding="ascii") as f:
                    if f.read().strip().startswith("CRYPT-"):
                        continue
            except OSError:
                pass
        unprotected.append(dev)
    return {"active": active, "unprotected": unprotected}


def raise_memlock_limit(need: int | None = None) -> None:
    """Raise the soft ``RLIMIT_MEMLOCK`` to the hard limit (best effort).

    Used by every ``mlock`` of ring memory (v1 arena, v2 memfd slabs in the gateway and in the
    factory child). ``need``: skip when the soft limit already covers this many bytes; ``None``
    always raises it (v2 locks many slabs one at a time, so the total is not known up front).
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY or (need is not None and soft >= need):
        return
    if hard == resource.RLIM_INFINITY or hard > soft:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))


# =================================================================================================
# arena
# =================================================================================================
class LockedArena:
    """Anonymous private memory, ``mlock``-ed, ``MADV_DONTDUMP`` and ``MADV_DONTFORK``.

    Args:
        size: Bytes (rounded up to a page).
        allow_unlocked: DEV ONLY: when ``mlock`` or ``madvise`` fails, log a warning and continue
            (``locked`` is then False). Otherwise a failure raises :class:`ConfigError` (fail closed).

    The ctypes buffer export taken at construction stays alive until :meth:`close`, so the raw
    address used for wipes and copies always refers to this mapping.
    """

    def __init__(self, size: int, *, allow_unlocked: bool = False, mlock: Callable[[int, int], int] | None = None):
        if size <= 0:
            raise ConfigError("arena size must be positive")
        self.size = -(-int(size) // _PAGE) * _PAGE
        flags = mmap.MAP_PRIVATE | getattr(mmap, "MAP_POPULATE", 0)
        try:
            self._mm: mmap.mmap | None = mmap.mmap(-1, self.size, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except OSError as exc:
            raise ConfigError(f"cannot map a {self.size / 2**20:.0f} MiB mask ring arena: {exc.strerror}") from exc
        self._cbuf: Any = (ctypes.c_char * self.size).from_buffer(self._mm)  # export held for the lifetime
        self.addr = ctypes.addressof(self._cbuf)
        self.locked = False
        self.dontdump = False
        self._closed = False
        try:
            self._lock_pages(mlock, allow_unlocked)
            self._advise(allow_unlocked)
        except BaseException:
            self._unmap()
            raise

    def _lock_pages(self, mlock: Callable[[int, int], int] | None, allow_unlocked: bool) -> None:
        raise_memlock_limit(self.size)
        rc = (mlock or _lib().mlock)(self.addr, self.size)
        if rc == 0:
            self.locked = True
            return
        e = ctypes.get_errno() or errno.ENOMEM
        msg = (
            f"cannot mlock the {self.size / 2**20:.0f} MiB mask ring ({os.strerror(e)}); raise the memlock limit: "
            "Helm securityContext.capabilities.add: [IPC_LOCK], systemd LimitMEMLOCK=, docker --ulimit memlock=-1 "
            "or --cap-add IPC_LOCK"
        )
        if not allow_unlocked:
            raise ConfigError(msg, details={"errno": e, "bytes": self.size})
        log.warning("%s; continuing UNLOCKED because allow_unlocked is set (dev only)", msg)

    def _advise(self, allow_unlocked: bool) -> None:
        assert self._mm is not None
        for name in ("MADV_DONTDUMP", "MADV_DONTFORK"):
            flag = getattr(mmap, name, None)
            try:
                if flag is None:
                    raise OSError(errno.ENOSYS, f"{name} unsupported")
                self._mm.madvise(flag)
            except OSError as exc:
                if not allow_unlocked:
                    raise ConfigError(f"madvise({name}) failed for the mask ring: {exc.strerror}") from exc
                log.warning("madvise(%s) failed (%s); continuing because allow_unlocked is set (dev only)",
                            name, exc.strerror)
                continue
            if name == "MADV_DONTDUMP":
                self.dontdump = True

    @property
    def closed(self) -> bool:
        return self._closed

    def view(self, off: int, n: int) -> memoryview:
        """Writable view of ``[off, off + n)`` (callers must release it before :meth:`close`)."""
        if self._closed:
            raise MaskStoreError("mask ring arena is closed")
        return memoryview(self._cbuf).cast("B")[off : off + n]

    def copy_out(self, off: int, n: int) -> bytearray:
        """A fresh zeroable copy of ``[off, off + n)`` (no intermediate ``bytes`` object)."""
        if self._closed or off < 0 or off + n > self.size:
            raise MaskStoreError("mask ring read out of range")
        out = bytearray(n)
        if n:
            dst = (ctypes.c_char * n).from_buffer(out)
            ctypes.memmove(dst, self.addr + off, n)
            del dst
        return out

    def copy_in(self, off: int, data: bytes | bytearray | memoryview) -> None:
        """Copy ``data`` to ``off`` (bounds-checked)."""
        mv = memoryview(data).cast("B")
        n = mv.nbytes
        if self._closed or off < 0 or off + n > self.size:
            raise MaskStoreError("mask ring write out of range")
        if n:
            dst = self.view(off, n)
            dst[:] = mv
            dst.release()

    def zero(self, off: int, n: int) -> None:
        """``explicit_bzero`` of ``[off, off + n)`` in pieces of :data:`WIPE_PIECE_BYTES`."""
        if self._closed:
            return
        if off < 0 or off + n > self.size:
            raise MaskStoreError("mask ring wipe out of range")
        end = off + n
        while off < end:
            k = min(WIPE_PIECE_BYTES, end - off)
            _bzero(self.addr + off, k)
            off += k

    def is_zero(self, off: int = 0, n: int | None = None) -> bool:
        """True if ``[off, off + n)`` is all zero (tests and audits)."""
        n = self.size - off if n is None else n
        mv = self.view(off, n)
        try:
            return _all_zero(mv)
        finally:
            mv.release()

    def _unmap(self) -> None:
        self._cbuf = None
        mm, self._mm = self._mm, None
        if mm is not None:
            mm.close()

    def close(self) -> None:
        """Wipe everything, ``munlock`` and ``munmap`` (idempotent). Callers stop every user first."""
        if self._closed:
            return
        self.zero(0, self.size)
        self._closed = True
        if self.locked:
            _lib().munlock(self.addr, self.size)
            self.locked = False
        self._unmap()


# =================================================================================================
# shared-memory slabs (v2): one memfd per grant, mapped through libc (no Python buffer exports)
# =================================================================================================
_MAP_FAILED = ctypes.c_void_p(-1).value
# fcntl(2) memfd sealing (linux/fcntl.h); not every Python build exports them
_F_ADD_SEALS = 1024 + 9
_F_SEAL_SEAL, _F_SEAL_SHRINK, _F_SEAL_GROW, _F_SEAL_WRITE = 0x1, 0x2, 0x4, 0x8


class FdMapping:
    """A ``MAP_SHARED`` mapping of a memfd made with libc ``mmap`` (``mlock``, ``MADV_DONTDUMP`` and
    ``MADV_DONTFORK``, all fail closed unless ``allow_unlocked``). Used by the gateway (read-only)
    and by the factory child (read-write)."""

    def __init__(self, fd: int, size: int, *, writable: bool, allow_unlocked: bool = False):
        lib = _lib()
        prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
        addr = lib.mmap(None, size, prot, mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0), fd, 0)
        if addr is None or addr == _MAP_FAILED:
            e = ctypes.get_errno()
            raise ConfigError(f"cannot map a mask ring slab: {os.strerror(e)}")
        self.addr = int(addr)
        self.size = int(size)
        self.writable = writable
        self.locked = False
        self._closed = False
        try:
            raise_memlock_limit()
            if lib.mlock(self.addr, self.size) == 0:
                self.locked = True
            elif not allow_unlocked:
                raise ConfigError(f"cannot mlock a mask ring slab ({os.strerror(ctypes.get_errno())}); "
                                  "raise the memlock limit (IPC_LOCK)")
            for name in ("MADV_DONTDUMP", "MADV_DONTFORK"):
                flag = getattr(mmap, name, None)
                if (flag is None or lib.madvise(self.addr, self.size, flag) != 0) and not allow_unlocked:
                    raise ConfigError(f"madvise({name}) failed for a mask ring slab")
        except BaseException:
            self.close(wipe=False)
            raise

    def view(self, off: int, n: int) -> memoryview:
        if self._closed or off < 0 or off + n > self.size:
            raise MaskStoreError("mask ring slab access out of range")
        return memoryview((ctypes.c_char * n).from_address(self.addr + off)).cast("B")

    def copy_out(self, off: int, n: int) -> bytearray:
        if self._closed or off < 0 or off + n > self.size:
            raise MaskStoreError("mask ring slab read out of range")
        out = bytearray(n)
        if n:
            dst = (ctypes.c_char * n).from_buffer(out)
            ctypes.memmove(dst, self.addr + off, n)
            del dst
        return out

    def zero(self) -> None:
        if not self._closed and self.writable:
            _bzero(self.addr, self.size)

    def close(self, *, wipe: bool = True) -> None:
        if self._closed:
            return
        if wipe:
            self.zero()
        self._closed = True
        lib = _lib()
        if self.locked:
            lib.munlock(self.addr, self.size)
            self.locked = False
        lib.munmap(self.addr, self.size)


class MemfdSlab:
    """Gateway side of one shared slab: a fresh sealed-size memfd, mapped read-only and locked.

    The factory child maps the fd read-write and fills it. At publish the gateway adds
    ``F_SEAL_WRITE``, which the kernel refuses (``EBUSY``) while any writable mapping exists, so a
    published slab can never be written again by anyone. Sealed pages cannot be zeroed either: they
    are dropped (``munmap`` + ``close``); the kernel zeroes a page before handing it to any process,
    but until then the freed physical page still holds the seed, ``uW`` and ``W r`` unless the kernel
    runs with ``init_on_free=1`` (see :func:`init_on_free_status`, SPEC 7.8). Unsealed slabs (aborted
    fills, refused publishes before the seal) are zeroed through the fd before they are dropped.
    """

    def __init__(self, size: int, *, allow_unlocked: bool = False):
        import fcntl

        self.size = int(size)
        self.fd = os.memfd_create("pgw-maskslab", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            os.ftruncate(self.fd, self.size)
            fcntl.fcntl(self.fd, _F_ADD_SEALS, _F_SEAL_SHRINK | _F_SEAL_GROW)
            # Map through a read-only reopen: a shared mapping of the O_RDWR memfd would count as
            # "may write" and block F_SEAL_WRITE at publish. This one stays locked through the fill.
            ro = os.open(f"/proc/self/fd/{self.fd}", os.O_RDONLY | os.O_CLOEXEC)
            try:
                self._map = FdMapping(ro, self.size, writable=False, allow_unlocked=allow_unlocked)
            finally:
                os.close(ro)
        except BaseException:
            os.close(self.fd)
            raise
        self.sealed = False

    @property
    def locked(self) -> bool:
        return self._map.locked

    def seal(self) -> None:
        """Forbid every future write (``EBUSY`` while the factory still maps the slab writable)."""
        import fcntl

        try:
            fcntl.fcntl(self.fd, _F_ADD_SEALS, _F_SEAL_WRITE | _F_SEAL_SEAL)
        except OSError as exc:
            raise MaskStoreError(
                f"cannot seal a published ring slab ({exc.strerror}): the factory still maps it writable",
            ) from exc
        self.sealed = True

    def copy_out(self, off: int, n: int) -> bytearray:
        return self._map.copy_out(off, n)

    def close(self) -> None:
        """Unmap and close. An unsealed slab is overwritten with zeros first (``pwrite`` on the memfd
        reaches the same pages); a sealed one cannot be written and is only dropped."""
        self._map.close(wipe=False)
        if not self.sealed:
            try:
                _zero_fd(self.fd, self.size)
            except OSError as exc:  # pragma: no cover - EPERM if a write seal was added meanwhile
                log.debug("could not zero an unsealed ring slab: %s", exc.strerror)
        with contextlib.suppress(OSError):
            os.close(self.fd)


_ZERO_PIECE = bytes(1 << 20)


def _zero_fd(fd: int, size: int) -> None:
    """Overwrite ``[0, size)`` of a (memfd) file with zeros."""
    off = 0
    while off < size:
        n = os.pwrite(fd, memoryview(_ZERO_PIECE)[: min(len(_ZERO_PIECE), size - off)], off)
        if n <= 0:  # pragma: no cover
            raise OSError(errno.EIO, "short write while zeroing a ring slab")
        off += n


def init_on_free_status(cmdline: str = "/proc/cmdline", config_paths: Sequence[str] | None = None) -> bool | None:
    """Whether the kernel zeroes pages when they are freed (``init_on_free``).

    v2 drops sealed memfd slabs without zeroing them (they cannot be written); only ``init_on_free``
    clears the freed pages at once. Reads the boot command line, then the kernel config
    (``CONFIG_INIT_ON_FREE_DEFAULT_ON``). Returns ``None`` when neither says.
    """
    try:
        with open(cmdline, encoding="ascii", errors="replace") as f:
            args = f.read().split()
    except OSError:
        args = []
    for a in reversed(args):  # the last occurrence wins
        if a.startswith("init_on_free="):
            return a.split("=", 1)[1].strip().lower() in ("1", "on", "y", "yes", "true")
    if config_paths is None:
        config_paths = (f"/boot/config-{os.uname().release}", "/proc/config.gz")
    for path in config_paths:
        try:
            if path.endswith(".gz"):
                import gzip

                with gzip.open(path, "rt", encoding="ascii", errors="replace") as g:
                    text = g.read()
            else:
                with open(path, encoding="ascii", errors="replace") as f:
                    text = f.read()
        except OSError:
            continue
        return "CONFIG_INIT_ON_FREE_DEFAULT_ON=y" in text
    return None


# =================================================================================================
# seeds
# =================================================================================================
class SeedRegistry:
    """Seed hashes granted in this process (ring writers and guarded store writers), bounded.

    Shared by every ring and by the factory's store sink, so the same seed can never land in both
    tiers (I3). Seeds are 256-bit ``os.urandom`` values, so this registry only guards against bugs
    (a seed passed twice, a generator that repeats); it does not need to remember forever:

    * *live* digests (``claim(..., live=True)``: a ring slab being filled or still holding the
      segment) are kept until :meth:`release` (the slab's wipe), so their number is bounded by the
      slab count;
    * every other digest (released ring seeds, store seeds, which the store's own ``seed_hash``
      table also remembers forever) stays in a recent window of ``max_recent`` entries, oldest
      evicted first.

    Memory is therefore ``O(slabs + max_recent)`` (about 150 bytes per entry), not one entry per
    segment for the gateway's lifetime.
    """

    def __init__(self, max_recent: int = 1 << 16) -> None:
        if max_recent < 1:
            raise ValueError("max_recent must be >= 1")
        self.max_recent = int(max_recent)
        self._lock = threading.Lock()
        self._live: dict[str, str] = {}
        self._recent: OrderedDict[str, str] = OrderedDict()
        self.evicted = 0

    def claim(self, digest: str, owner: str, *, live: bool = False) -> None:
        """Record ``digest`` for ``owner``; :class:`MaskReuseError` if it is known.

        ``live=True`` pins it until :meth:`release` (ring slabs)."""
        with self._lock:
            if digest in self._live or digest in self._recent:
                raise MaskReuseError("this seed was already used for a mask segment in this process (mask reuse)")
            if live:
                self._live[digest] = owner
            else:
                self._remember_locked(digest, owner)

    def release(self, digest: str | None) -> None:
        """Unpin a live digest (its slab was wiped); it stays in the recent window."""
        if digest is None:
            return
        with self._lock:
            owner = self._live.pop(digest, None)
            if owner is not None:
                self._remember_locked(digest, owner)

    def _remember_locked(self, digest: str, owner: str) -> None:
        self._recent[digest] = owner
        while len(self._recent) > self.max_recent:
            self._recent.popitem(last=False)
            self.evicted += 1

    def owned_by(self, digest: str, owner: str) -> bool:
        """True if ``digest`` is pinned (live) for ``owner``."""
        with self._lock:
            return self._live.get(digest) == owner

    @property
    def live(self) -> int:
        with self._lock:
            return len(self._live)

    def __contains__(self, digest: object) -> bool:
        with self._lock:
            return digest in self._live or digest in self._recent

    def __len__(self) -> int:
        with self._lock:
            return len(self._live) + len(self._recent)


#: Process-wide default registry.
GLOBAL_SEEDS = SeedRegistry()


# =================================================================================================
# geometry
# =================================================================================================
@dataclass(frozen=True)
class RingGeometry:
    """Sizes of one ring (see design §3.2)."""

    bpe: int
    slot_bytes: int
    secrets_max: int  #: seed + uW for ``max_checks`` vectors
    wr_base: int  #: slab-relative offset of the first W r byte (aligned)
    segment_slots: int
    slab_bytes: int
    n_slabs: int

    @property
    def capacity_slots(self) -> int:
        return self.n_slabs * self.segment_slots


def ring_geometry(
    specs: Sequence[LinearSpec], ring: RingParams, capacity_bytes: int, *, segment_slots: int | None = None,
    max_checks: int = DEFAULT_MAX_CHECKS, min_slabs: int = 4,
) -> RingGeometry:
    """Slab geometry for ``capacity_bytes`` (``ConfigError`` when fewer than ``min_slabs`` fit)."""
    bpe = ring.bytes_per_element
    slot_bytes = sum(int(s.out_features) for s in specs) * bpe
    sum_n = sum(int(s.in_features) for s in specs)
    secrets_max = SEED_BYTES + max_checks * sum_n * bpe
    wr_base = -(-secrets_max // _ALIGN) * _ALIGN
    stream_max = min(max_stream_rows(int(s.in_features)) for s in specs)
    if segment_slots is None:
        guess = (int(capacity_bytes) // 8 - wr_base) // max(1, slot_bytes)
        segment_slots = max(16, min(256, guess))
    segment_slots = int(segment_slots)
    if segment_slots <= 0:
        raise ConfigError("segment_slots must be positive")
    if segment_slots > stream_max:
        raise ConfigError(f"ring segment_slots={segment_slots} exceeds the PRNG stream range (max {stream_max})")
    slab_bytes = -(-(wr_base + segment_slots * slot_bytes) // _PAGE) * _PAGE
    n_slabs = int(capacity_bytes) // slab_bytes
    if n_slabs < min_slabs:
        raise ConfigError(
            f"mask ring capacity {capacity_bytes / 2**20:.1f} MiB holds only {n_slabs} slab(s) of "
            f"{segment_slots} slots ({slab_bytes / 2**20:.1f} MiB each); at least {min_slabs} are needed "
            "(raise capacity_bytes or lower segment_slots)",
            details={"slab_bytes": slab_bytes, "n_slabs": n_slabs},
        )
    return RingGeometry(bpe, slot_bytes, secrets_max, wr_base, segment_slots, slab_bytes, n_slabs)


# =================================================================================================
# slabs and handles
# =================================================================================================
class SlabState:
    FREE = "free"
    FILLING = "filling"
    READY = "ready"
    DRAINING = "draining"
    WIPING = "wiping"


_READABLE = (SlabState.READY, SlabState.DRAINING)


class _Slab:
    __slots__ = (
        "chunk_slots",
        "digest",
        "gen",
        "header",
        "index",
        "layout",
        "lock",
        "mem",
        "next_slot",
        "off",
        "refs",
        "secrets_len",
        "slots",
        "state",
        "wipe_queued",
        "wr_offsets",
    )

    def __init__(self, index: int, off: int):
        self.index = index
        self.off = off
        self.lock = threading.Lock()  # R4: reads, writes and state transitions out of READY/DRAINING/FILLING
        self.state = SlabState.FREE
        self.gen = 0
        self.header: SegmentHeader | None = None
        self.layout: SegmentLayout | None = None
        self.digest: str | None = None
        self.next_slot = 0
        self.refs = 0
        self.slots = 0
        self.chunk_slots = 0
        self.secrets_len = 0
        self.wr_offsets: list[int] = []
        self.wipe_queued = False
        #: Shared-memory (v2) slabs: the per-grant memfd mapping; ``None`` = this ring's arena.
        self.mem: Any = None

    def reset(self) -> None:
        self.header = None
        self.layout = None
        self.digest = None
        self.next_slot = 0
        self.slots = 0
        self.chunk_slots = 0
        self.secrets_len = 0
        self.wr_offsets = []


class RamSegment:
    """Duck-typed :class:`pgw.maskstore.format.SegmentFile` over one published slab (one handle).

    Provides ``header``, ``layout``, ``decrypt_into(ref)`` and ``close()``, which is all
    :class:`MaskReservation` uses. Every handle holds one reference on its slab until closed.
    """

    __slots__ = ("_ring", "_slab", "closed", "ext_count", "from_finalizer", "gen", "header", "layout",
                 "reconciled", "rid")

    def __init__(self, ring: MaskRing, slab: _Slab, rid: str, ext_count: int):
        assert slab.header is not None and slab.layout is not None
        self._ring = ring
        self._slab = slab
        self.gen = slab.gen
        self.header = slab.header
        self.layout = slab.layout
        self.rid = rid
        self.ext_count = ext_count
        self.closed = False
        self.from_finalizer = False
        self.reconciled = False

    @property
    def segment_id(self) -> str:
        return self.header.segment_id

    def decrypt_into(self, ref: ChunkRef) -> bytearray:
        """Copy chunk ``ref``'s plaintext out of the slab into a fresh ``bytearray`` (R4 checks under the
        slab lock). The secrets chunk is verified against the seed hash recorded at commit."""
        if self.closed:
            raise MaskStoreError(f"ring segment {self.header.segment_id} handle is closed")
        ring, slab = self._ring, self._slab
        with slab.lock:
            if ring.closed or slab.gen != self.gen or slab.state not in _READABLE:
                raise MaskStoreError(
                    f"ring segment {self.header.segment_id} is no longer readable (ring closed or slab recycled)",
                    details={"segment_id": self.header.segment_id},
                )
            off, n = ring._chunk_span(slab, ref)
            out = ring._copy_out(slab, off, n)
            digest = slab.digest
        if ref.index == 0:
            seed = memoryview(out)[:SEED_BYTES]
            bad = _all_zero(seed) or digest is None or not hmac.compare_digest(seed_digest(seed), digest)
            seed.release()
            if bad:
                _zero(out)
                raise MaskStoreError(
                    f"ring segment {self.header.segment_id}: seed check failed (zero or altered seed)",
                    details={"segment_id": self.header.segment_id},
                )
        return out

    def close(self) -> None:
        """Release the slab reference. Lock-free (safe from finalizers): the ring reconciles later."""
        if self.closed:
            return
        self.closed = True
        self._ring._closeq.put(self)


class _RingOwner:
    """``store`` stand-in for ring reservations (only ``_on_reservation_closed`` is used)."""

    __slots__ = ("_ring",)

    def __init__(self, ring: MaskRing):
        self._ring = weakref.ref(ring)

    def _on_reservation_closed(self, res: MaskReservation, burned: int) -> None:
        ring = self._ring()
        if ring is not None:
            ring._reservation_closed(res, burned)


class RingReservation(MaskReservation):
    """A :class:`MaskReservation` whose finalizer marks its handles so the ring counts them as leaked."""

    def __del__(self) -> None:  # pragma: no cover - exercised through gc in tests
        if not getattr(self, "_closed", True):
            for e in getattr(self, "_extents", ()):
                seg = getattr(e, "seg", None)
                if isinstance(seg, RamSegment):
                    seg.from_finalizer = True
        super().__del__()


class _OpenRec:
    __slots__ = ("finalized", "handles", "open_handles", "res", "rid", "slots")

    def __init__(self, rid: str, slots: int, handles: list[RamSegment]):
        self.rid = rid
        self.slots = slots
        self.handles = handles
        self.open_handles = len(handles)
        self.res: weakref.ref[MaskReservation] | None = None
        self.finalized = False


# =================================================================================================
# ring
# =================================================================================================
class MaskRing:
    """Locked RAM ring of mask segments for one ``(fingerprint, ring)``.

    Args:
        fingerprint: Model fingerprint.
        ring: Ring parameters.
        specs: The model's linear specs (generator order).
        capacity_bytes: Locked bytes for this ring.
        segment_slots: Slots per slab (default: about 8 slabs, clamped to 16..256).
        min_soundness_bits: Commit refuses segments whose Freivalds checks give fewer bits.
        allow_unlocked: DEV ONLY (see :class:`LockedArena`).
        seeds: Seed registry (default: the process-wide one).
        max_checks: Freivalds vectors a slab has room for.
        name: Model name for logs and status.
    """

    def __init__(
        self,
        fingerprint: str,
        ring: RingParams | str,
        specs: Sequence[LinearSpec],
        *,
        capacity_bytes: int,
        segment_slots: int | None = None,
        min_soundness_bits: float = 20.0,
        allow_unlocked: bool = False,
        seeds: SeedRegistry | None = None,
        max_checks: int = DEFAULT_MAX_CHECKS,
        name: str | None = None,
        shared: bool = False,
        _mlock: Callable[[int, int], int] | None = None,
    ):
        self.fingerprint = validate_fingerprint(fingerprint)
        #: v2: slabs live in per-grant memfds filled by a factory child (pgw.maskstore.ringshm).
        self.shared = bool(shared)
        self.allow_unlocked = bool(allow_unlocked)
        self.ring = ring_from_setting(ring)
        self.specs = list(specs)
        if not self.specs:
            raise ConfigError("mask ring needs the model's linear specs")
        self.name = name or short_fingerprint(self.fingerprint)
        self._lin_sig = tuple((s.id.key, int(s.out_features), int(s.in_features)) for s in self.specs)
        self.linears = tuple(SegmentLinear(id=k, out_features=m, in_features=n) for k, m, n in self._lin_sig)
        self.max_checks = int(max_checks)
        self.min_soundness_bits = float(min_soundness_bits)
        self.geometry = ring_geometry(self.specs, self.ring, capacity_bytes, segment_slots=segment_slots,
                                      max_checks=self.max_checks)
        g = self.geometry
        self.segment_slots = g.segment_slots
        self.seeds = seeds if seeds is not None else GLOBAL_SEEDS
        self._arena: LockedArena | None = (
            None if self.shared else LockedArena(g.n_slabs * g.slab_bytes, allow_unlocked=allow_unlocked, mlock=_mlock)
        )
        self._slabs = [_Slab(i, i * g.slab_bytes) for i in range(g.n_slabs)]
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._closeq: queue.SimpleQueue[RamSegment] = queue.SimpleQueue()
        self._fifo: list[_Slab] = []  # READY slabs in publish order
        self._open: dict[str, _OpenRec] = {}
        self._wipe_todo: list[_Slab] = []
        self._writers: dict[int, RingSegmentWriter] = {}
        self._closed = False
        self._stop = False
        self._free_listeners: list[Callable[[], None]] = []
        self._owner = _RingOwner(self)
        # accounting (I9): published = reserved + available + lost ; reserved = taken + burned + leaked + open
        self.published_total = 0
        self.published_segments = 0
        self.reserved_total = 0
        self.taken_total = 0
        self.burned_total = 0
        self.leaked_total = 0
        self.lost_total = 0
        self.aborted_total = 0
        self.reservations_total = 0
        self.wipes_total = 0
        self._wiper = threading.Thread(target=self._wiper_loop, name="pgw-maskring-wiper", daemon=True)
        self._wiper.start()
        log.info(
            "mask ring for %s (%s) %s: %d slabs x %d slots (%.1f MiB each), %.1f MiB %s",
            self.name, short_fingerprint(self.fingerprint), self.ring.name, g.n_slabs, g.segment_slots,
            g.slab_bytes / 2**20, g.n_slabs * g.slab_bytes / 2**20,
            "shared" if self.shared else "locked" if self._arena and self._arena.locked else "UNLOCKED",
        )

    def __repr__(self) -> str:
        return (f"MaskRing({self.name}, {self.ring.name}, slabs={len(self._slabs)}, "
                f"segment_slots={self.segment_slots}, closed={self._closed})")

    # ---- introspection ---------------------------------------------------------------------------
    @property
    def key(self) -> tuple[str, str]:
        return self.fingerprint, self.ring.name

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def locked_bytes(self) -> int:
        if self._arena is None:
            return sum(s.mem.size for s in self._slabs if s.mem is not None and s.mem.locked)
        return self._arena.size if self._arena.locked else 0

    # ---- slab memory (arena slice in v1, per-grant memfd mapping in v2) -------------------------
    def _copy_out(self, slab: _Slab, off: int, n: int) -> bytearray:
        if slab.mem is not None:
            return slab.mem.copy_out(off, n)  # type: ignore[no-any-return]
        if self._arena is None:
            raise MaskStoreError("ring slab has no memory")
        return self._arena.copy_out(slab.off + off, n)

    def _release_slab_memory(self, slab: _Slab) -> None:
        """Wipe (arena) or drop (memfd: unmap and close; sealed pages cannot be written) a slab's bytes."""
        if slab.mem is not None:
            # Under the slab lock: a concurrent copy (publish_shared, decrypt_into) holds it while it
            # reads the mapping, so the munmap never runs under a reader.
            with slab.lock:
                mem, slab.mem = slab.mem, None
                if mem is not None:
                    mem.close()
        elif self._arena is not None:
            self._arena.zero(slab.off, self.geometry.slab_bytes)

    @property
    def capacity_slots(self) -> int:
        return self.geometry.capacity_slots

    @property
    def n_slabs(self) -> int:
        return len(self._slabs)

    def _available_locked(self) -> int:
        return sum(s.slots - s.next_slot for s in self._fifo if s.state == SlabState.READY)

    @property
    def available(self) -> int:
        """Unreserved slots in READY slabs."""
        with self._lock:
            self._reconcile_locked()
            return self._available_locked()

    def counts(self) -> dict[str, int]:
        """Slab counts per state plus accounting totals (ids and counts only)."""
        with self._lock:
            self._reconcile_locked()
            by: dict[str, int] = dict.fromkeys(
                (SlabState.FREE, SlabState.FILLING, SlabState.READY, SlabState.DRAINING, SlabState.WIPING), 0)
            for s in self._slabs:
                by[s.state] += 1
            return {
                **{f"{k}_slabs": v for k, v in by.items()},
                "available_slots": self._available_locked(),
                "open_reservations": len(self._open),
                "open_slots": sum(r.slots for r in self._open.values()),
                "published_slots": self.published_total,
                "published_segments": self.published_segments,
                "reserved_slots": self.reserved_total,
                "taken_slots": self.taken_total,
                "burned_slots": self.burned_total,
                "leaked_slots": self.leaked_total,
                "lost_slots": self.lost_total,
                "aborted_slots": self.aborted_total,
                "reservations": self.reservations_total,
                "wipes": self.wipes_total,
            }

    def status(self) -> dict[str, Any]:
        """JSON-serialisable status (ids, counts and sizes only; never seeds, rows or addresses)."""
        c = self.counts()
        return {
            "model": self.name, "fingerprint": self.fingerprint, "ring": self.ring.name,
            "segment_slots": self.segment_slots, "slab_bytes": self.geometry.slab_bytes,
            "capacity_slots": self.capacity_slots, "locked_bytes": self.locked_bytes, "closed": self._closed,
            **c,
        }

    def slab_states(self) -> list[str]:
        """Per-slab states (tests)."""
        with self._lock:
            self._reconcile_locked()
            return [s.state for s in self._slabs]

    def add_free_listener(self, fn: Callable[[], None]) -> None:
        """Call ``fn()`` (from the wiper thread, no lock held) whenever slabs become FREE."""
        with self._lock:
            self._free_listeners.append(fn)

    # ---- factory side ----------------------------------------------------------------------------
    def check_specs(self, fingerprint: str, ring: RingParams | str, specs: Sequence[LinearSpec]) -> None:
        """``ConfigError`` unless ``(fingerprint, ring, specs)`` is exactly this ring's model."""
        rp = ring_from_setting(ring)
        if fingerprint != self.fingerprint or rp.name != self.ring.name:
            raise ConfigError("ring writer for a different model or ring")
        sig = tuple((s.id.key, int(s.out_features), int(s.in_features)) for s in specs)
        if sig != self._lin_sig:
            raise ConfigError("ring writer: linear specs differ from the ring's model (ids, shapes or order)")

    def writer(
        self,
        fingerprint: str,
        ring: RingParams | str,
        specs: Sequence[LinearSpec],
        *,
        slots: int,
        seed: bytes,
        checks: int = 2,
        chunk_slots: int = 256,
        factory_id: str = "local",
        destination: str | None = None,
    ) -> RingSegmentWriter:
        """Grant a FREE slab to a new segment (``SegmentWriter`` contract). Never blocks.

        Raises:
            ConfigError: wrong model/ring/specs, a ``destination``, too many slots or checks.
            MaskReuseError: the seed was used before in this process (or is all zero).
            RingFull: no FREE slab right now.
        """
        if destination is not None:
            raise ConfigError("ring segments are never exported (destination is not allowed)")
        if self.shared:
            raise ConfigError("a shared-memory ring is filled by its factory child (pgw.maskstore.ringshm)")
        self.check_specs(fingerprint, ring, specs)
        if not isinstance(slots, int) or isinstance(slots, bool) or not 0 < slots <= self.segment_slots:
            raise ConfigError(f"ring segments hold 1..{self.segment_slots} slots, got {slots!r}")
        if not 0 <= int(checks) <= self.max_checks:
            raise ConfigError(f"ring segments hold at most {self.max_checks} check vectors, got {checks}")
        if not isinstance(chunk_slots, int) or chunk_slots <= 0:
            raise ConfigError("chunk_slots must be a positive int")
        if not isinstance(seed, bytes | bytearray) or len(seed) != SEED_BYTES:
            raise ConfigError(f"seed must be exactly {SEED_BYTES} bytes")
        if _all_zero(seed):
            raise MaskReuseError("refusing an all-zero segment seed")
        digest = seed_digest(seed)
        token = uuid.uuid4().hex
        with self._lock:
            if self._closed:
                raise MaskStoreError("mask ring is closed")
            self._reconcile_locked()
            slab = next((s for s in self._slabs if s.state == SlabState.FREE), None)
            if slab is None:
                raise RingFull("no free ring slab", details={"ring": self.ring.name})
            self.seeds.claim(digest, token, live=True)  # at grant: pinned until the slab's wipe
            with slab.lock:
                slab.state = SlabState.FILLING
                slab.gen += 1
                slab.reset()
                slab.digest = digest  # released from the registry's live set when this slab is wiped
                slab.slots = slots
                slab.chunk_slots = min(int(chunk_slots), slots)
            w = RingSegmentWriter(self, slab, slab.gen, seed=seed, digest=digest, token=token, slots=slots,
                                  checks=int(checks), chunk_slots=slab.chunk_slots, factory_id=factory_id)
            self._writers[slab.index] = w
        return w

    def _write_rows(self, w: RingSegmentWriter, li: int, row0: int, a: np.ndarray) -> None:
        """Copy W r rows (already range-checked) into the slab region of linear ``li`` (R4)."""
        slab = w._slab
        g = self.geometry
        m = int(a.shape[1])
        bpe = g.bpe
        off = slab.off + g.wr_base + w._wr_rel[li] + row0 * m * bpe
        with slab.lock:
            if self._closed or slab.gen != w.gen or slab.state != SlabState.FILLING or self._arena is None:
                raise MaskStoreError("ring slab was revoked while writing (ring closed)")
            mv = self._arena.view(off, a.shape[0] * m * bpe)
            try:
                _pack_into(mv, a, self.ring)
            finally:
                mv.release()

    def _publish(self, w: RingSegmentWriter, header: SegmentHeader, secrets: bytearray) -> None:
        slab = w._slab
        with self._lock:
            if self._closed:
                raise MaskStoreError("mask ring is closed")
            if not self.seeds.owned_by(w._digest, w._token):
                raise MaskReuseError("segment seed is not owned by this writer (mask reuse)")
            with slab.lock:
                if slab.gen != w.gen or slab.state != SlabState.FILLING:
                    raise MaskStoreError("ring slab was revoked before commit")
                assert self._arena is not None
                self._arena.copy_in(slab.off, secrets)
                slab.header = header
                slab.layout = SegmentLayout(header, 0)
                slab.digest = w._digest
                slab.secrets_len = len(secrets)
                slab.wr_offsets = list(w._wr_rel)
                slab.next_slot = 0
                slab.refs = 0
                slab.state = SlabState.READY
            self._writers.pop(slab.index, None)
            self._fifo.append(slab)
            self.published_total += slab.slots
            self.published_segments += 1
            self._cv.notify_all()

    def _abort_fill(self, w: RingSegmentWriter) -> None:
        slab = w._slab
        with self._lock:
            if self._writers.get(slab.index) is w:
                del self._writers[slab.index]
            with slab.lock:
                if slab.gen != w.gen or slab.state != SlabState.FILLING:
                    return
                slab.state = SlabState.WIPING
            self.aborted_total += slab.slots
            self._queue_wipe_locked(slab)

    # ---- v2: shared-memory slabs filled by the factory child (pgw.maskstore.ringshm) -------------
    def grant_shared(self, fingerprint: str, ring: RingParams | str, spec_sig: Sequence[Sequence[Any]], *,
                     slots: int, checks: int, chunk_slots: int) -> tuple[int, int, int]:
        """Grant a FREE slab as a fresh memfd; returns ``(slab index, generation, fd)``.

        The fd stays owned by the ring (the broker sends a copy to the child). Raises like
        :meth:`writer` (``RingFull`` when no slab is free)."""
        if not self.shared:
            raise ConfigError("grant_shared needs a shared-memory ring")
        rp = ring_from_setting(ring)
        if fingerprint != self.fingerprint or rp.name != self.ring.name:
            raise ConfigError("ring grant for a different model or ring")
        if tuple(tuple(x) for x in spec_sig) != tuple(tuple(x) for x in self._lin_sig):
            raise ConfigError("ring grant: linear specs differ from the ring's model")
        if not isinstance(slots, int) or not 0 < slots <= self.segment_slots:
            raise ConfigError(f"ring segments hold 1..{self.segment_slots} slots, got {slots!r}")
        if not 0 <= int(checks) <= self.max_checks or int(chunk_slots) <= 0:
            raise ConfigError("ring grant: bad checks or chunk_slots")
        with self._lock:
            if self._closed:
                raise MaskStoreError("mask ring is closed")
            self._reconcile_locked()
            slab = next((s for s in self._slabs if s.state == SlabState.FREE), None)
            if slab is None:
                raise RingFull("no free ring slab", details={"ring": self.ring.name})
            with slab.lock:
                slab.state = SlabState.FILLING
                slab.gen += 1
                slab.reset()
                slab.slots = slots
                slab.chunk_slots = min(int(chunk_slots), slots)
                slab.secrets_len = SEED_BYTES + int(checks) * sum(lin.in_features for lin in self.linears) * \
                    self.geometry.bpe
            gen = slab.gen
        try:
            mem = MemfdSlab(self.geometry.slab_bytes, allow_unlocked=self.allow_unlocked)
        except BaseException:
            self._abort_index(slab.index, gen)
            raise
        with self._lock, slab.lock:
            if self._closed or slab.gen != gen or slab.state != SlabState.FILLING:
                mem.close()
                raise MaskStoreError("ring slab was revoked during the grant")
            slab.mem = mem
        return slab.index, gen, mem.fd

    def publish_shared(self, index: int, gen: int, *, slots: int, checks: int, chunk_slots: int,
                       factory_id: str) -> str:
        """Seal, validate and publish a slab the factory child filled; returns the segment id.

        Refused (and the slab dropped) when the child still maps it writable, the seed is zero or
        was used before, the soundness floor is not met, or the geometry differs from the grant."""
        slab = self._slabs[int(index)]
        with self._lock:
            ok = (not self._closed and slab.gen == gen and slab.state == SlabState.FILLING and slab.mem is not None
                  and slab.slots == slots and slab.chunk_slots == min(int(chunk_slots), slots))
            mem = slab.mem
            secrets_len = slab.secrets_len
        if not ok or mem is None:
            self._abort_index(index, gen)
            raise MaskStoreError("ring publish does not match its grant")
        secrets: bytearray | None = None
        token = uuid.uuid4().hex
        try:
            with slab.lock:  # the mapping cannot be dropped (close, abort) while it is read
                if slab.mem is not mem or slab.gen != gen or slab.state != SlabState.FILLING:
                    raise MaskStoreError("ring slab was revoked before publish")
                mem.seal()
                secrets = mem.copy_out(0, secrets_len)
            seed = memoryview(secrets)[:SEED_BYTES]
            zero_seed = _all_zero(seed)
            digest = seed_digest(seed)
            seed.release()
            if zero_seed:
                raise MaskReuseError("refusing an all-zero segment seed")
            if int(checks) != (secrets_len - SEED_BYTES) // max(
                    1, sum(lin.in_features for lin in self.linears) * self.geometry.bpe):
                raise MaskStoreError("ring publish: checks differ from the grant")
            _check_floor(self, int(checks))
            header = SegmentHeader(
                segment_id=uuid.uuid4().hex, fingerprint=self.fingerprint, ring=self.ring.name, slots=slots,
                chunk_slots=slab.chunk_slots, checks=int(checks), linears=self.linears, kek_id="ram",
                wrapped_dek="", nonce_prefix="00000000", factory_id=str(factory_id)[:64],
                created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            wr_rel, off = [], 0
            for lin in self.linears:
                wr_rel.append(off)
                off += slots * lin.out_features * self.geometry.bpe
            with self._lock:
                if self._closed:
                    raise MaskStoreError("mask ring is closed")
                with slab.lock:
                    if slab.gen != gen or slab.state != SlabState.FILLING:
                        raise MaskStoreError("ring slab was revoked before publish")
                    # unique across both tiers and every ring; pinned until this slab's wipe
                    self.seeds.claim(digest, token, live=True)
                    slab.header = header
                    slab.layout = SegmentLayout(header, 0)
                    slab.digest = digest
                    slab.wr_offsets = wr_rel
                    slab.next_slot = 0
                    slab.refs = 0
                    slab.state = SlabState.READY
                self._fifo.append(slab)
                self.published_total += slab.slots
                self.published_segments += 1
                self._cv.notify_all()
        except BaseException:
            self._abort_index(index, gen)
            raise
        finally:
            if secrets is not None:
                _zero(secrets)
        log.info("ring %s: published shared segment %s (%d slots, slab %d)", self.name, header.segment_id, slots,
                 slab.index)
        return header.segment_id

    def abort_shared(self, index: int, gen: int) -> None:
        """The child aborted (or died): the slab is dropped and freed."""
        self._abort_index(int(index), int(gen))

    def _abort_index(self, index: int, gen: int) -> None:
        slab = self._slabs[index]
        with self._lock:
            with slab.lock:
                if slab.gen != gen or slab.state != SlabState.FILLING:
                    return
                slab.state = SlabState.WIPING
            self.aborted_total += slab.slots
            self._queue_wipe_locked(slab)

    # ---- gateway side ----------------------------------------------------------------------------
    def reserve(self, slots: int, *, session_id: str = "", wait_s: float = 0.0) -> MaskReservation:
        """Reserve ``slots`` slots, oldest READY slabs first. Never does I/O; waits only up to ``wait_s``.

        Raises:
            RingShort: fewer slots are available (after waiting ``wait_s``).
            MaskStoreError: the ring is closed.
        """
        if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
            raise ConfigError(f"slots must be a positive int, got {slots!r}")
        deadline = time.monotonic() + max(0.0, float(wait_s))
        rid = uuid.uuid4().hex
        with self._lock:
            while True:
                if self._closed:
                    raise MaskStoreError("mask ring is closed")
                self._reconcile_locked()
                avail = self._available_locked()
                if avail >= slots:
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    filling = sum(1 for s in self._slabs if s.state == SlabState.FILLING)
                    raise RingShort(
                        "not enough mask slots in the ring",
                        details={"fingerprint": short_fingerprint(self.fingerprint), "ring": self.ring.name,
                                 "needed": slots, "available": avail, "ring_available": avail, "filling": filling},
                    )
                self._cv.wait(left)
            picked: list[tuple[SlotExtent, RamSegment]] = []
            need = slots
            for slab in self._fifo:
                if need == 0:
                    break
                if slab.state != SlabState.READY:
                    continue
                take = min(need, slab.slots - slab.next_slot)
                if take <= 0:
                    continue
                assert slab.header is not None
                ext = SlotExtent(slab.header.segment_id, slab.next_slot, take)
                slab.next_slot += take  # R2: the cursor moves before anything is handed out
                slab.refs += 1
                if slab.next_slot == slab.slots:
                    with slab.lock:
                        slab.state = SlabState.DRAINING
                picked.append((ext, RamSegment(self, slab, rid, take)))
                need -= take
            self._fifo = [s for s in self._fifo if s.state == SlabState.READY]
            rec = _OpenRec(rid, slots, [h for _, h in picked])
            self._open[rid] = rec  # registered inside the critical section (close() sees it)
            self.reserved_total += slots
            self.reservations_total += 1
        try:
            res = RingReservation(self._owner, rid, session_id, self.fingerprint, self.ring, picked)  # type: ignore[arg-type]
        except BaseException:
            for _, h in picked:
                h.close()
            with self._lock:
                self._reconcile_locked()
                if self._open.pop(rid, None) is not None:
                    self.burned_total += slots  # reserved, never handed out, never reused
            raise
        with self._lock:
            rec.res = weakref.ref(res)
            closed = self._closed
        if closed:
            res.close()
            raise MaskStoreError("mask ring was closed during the reservation")
        log.debug("ring %s: reserved %d slots (%d extents) rid=%s", self.name, slots, len(picked), rid[:8])
        return res

    def _reservation_closed(self, res: MaskReservation, burned: int) -> None:
        with self._lock:
            self._reconcile_locked()
            rec = self._open.pop(res.id, None)
            if rec is not None:
                self.burned_total += int(burned)
                self.taken_total += rec.slots - int(burned)
            if self._wipe_todo:
                self._cv.notify_all()

    def _chunk_span(self, slab: _Slab, ref: ChunkRef) -> tuple[int, int]:
        """Slab-relative ``(offset, length)`` of chunk ``ref`` (validated against the segment layout)."""
        layout = slab.layout
        assert layout is not None
        idx = int(ref.index)
        if idx == 0:
            n = slab.secrets_len
            if ref.plain_len != n:
                raise MaskStoreError("ring secrets chunk length mismatch")
            return 0, n
        cpl = layout.chunks_per_linear
        li, j = divmod(idx - 1, cpl)
        if not 0 <= li < len(slab.wr_offsets):
            raise MaskStoreError(f"ring chunk index {idx} out of range")
        m = self.linears[li].out_features
        bpe = self.geometry.bpe
        n = layout.chunk_rows(j) * m * bpe
        if ref.plain_len != n:
            raise MaskStoreError("ring chunk length mismatch")
        return self.geometry.wr_base + slab.wr_offsets[li] + j * slab.chunk_slots * m * bpe, n

    # ---- reconcile / wipe ------------------------------------------------------------------------
    def _reconcile_locked(self) -> None:
        """Apply queued handle closes (refs -= 1); fully drained slabs go to the wiper. Lock held."""
        while True:
            try:
                h = self._closeq.get_nowait()
            except queue.Empty:
                return
            if h.reconciled:
                continue
            h.reconciled = True
            slab = h._slab
            if slab.gen == h.gen and slab.refs > 0:
                slab.refs -= 1
            rec = self._open.get(h.rid)
            if rec is not None:
                rec.open_handles -= 1
                if h.from_finalizer:
                    rec.finalized = True
                    self.leaked_total += h.ext_count
                if rec.open_handles <= 0 and rec.finalized:
                    del self._open[h.rid]
                    log.warning("ring %s: reservation %s was garbage-collected without close(); %d slots leaked",
                                self.name, h.rid[:8], rec.slots)
            if slab.gen == h.gen and slab.refs == 0 and slab.state == SlabState.DRAINING:
                with slab.lock:
                    slab.state = SlabState.WIPING
                self._queue_wipe_locked(slab)

    def _queue_wipe_locked(self, slab: _Slab) -> None:
        if not slab.wipe_queued:
            slab.wipe_queued = True
            self._wipe_todo.append(slab)
            self._cv.notify_all()

    def _wiper_loop(self) -> None:
        while True:
            with self._lock:
                if not self._stop and not self._wipe_todo:
                    self._cv.wait(_WIPER_POLL_S)
                if self._stop:
                    return
                self._reconcile_locked()
                todo, self._wipe_todo = self._wipe_todo, []
            if not todo:
                continue
            for slab in todo:
                self._release_slab_memory(slab)
            listeners: list[Callable[[], None]] = []
            with self._lock:
                if self._stop:
                    return
                for slab in todo:
                    with slab.lock:
                        self.seeds.release(slab.digest)
                        slab.reset()
                        slab.refs = 0
                        slab.wipe_queued = False
                        slab.state = SlabState.FREE
                    self.wipes_total += 1
                listeners = list(self._free_listeners)
                self._cv.notify_all()
            for fn in listeners:
                try:
                    fn()
                except Exception:  # pragma: no cover - listener bugs must not stop the wiper
                    log.exception("mask ring free listener failed")

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Wait until no slab is queued for or in a wipe (tests, shutdown)."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while True:
                self._reconcile_locked()
                if not self._wipe_todo and not any(s.state == SlabState.WIPING for s in self._slabs):
                    return True
                left = deadline - time.monotonic()
                if left <= 0 or self._closed:
                    return False
                self._cv.notify_all()
                self._cv.wait(min(left, 0.05))

    # ---- shutdown --------------------------------------------------------------------------------
    def close(self) -> None:
        """Revoke and wipe everything (idempotent).

        Order (design review): set closed and move every slab to WIPING under its lock (late readers
        and writers raise from now on), force-close open reservations, join the wiper, wipe the whole
        arena synchronously, ``munlock``, ``munmap``. The embedded factory should be stopped first.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for slab in self._slabs:
                with slab.lock:
                    if slab.state == SlabState.READY:
                        self.lost_total += slab.slots - slab.next_slot
                    elif slab.state == SlabState.FILLING:
                        self.aborted_total += slab.slots
                    slab.state = SlabState.WIPING
            recs = list(self._open.values())
            self._fifo = []
            self._writers.clear()
            self._cv.notify_all()
        for rec in recs:
            res = rec.res() if rec.res is not None else None
            if res is not None:
                try:
                    res.close()
                except Exception:  # pragma: no cover - close() of a reservation does not raise
                    log.exception("force-closing a ring reservation failed")
            else:  # placeholder (still being built) or already collected: revoke the handles
                for h in rec.handles:
                    h.close()
        with self._lock:
            self._stop = True
            self._cv.notify_all()
        if self._wiper is not threading.current_thread():
            self._wiper.join()
        with self._lock:
            self._reconcile_locked()
        for slab in self._slabs:
            if slab.mem is not None:
                self._release_slab_memory(slab)
            self.seeds.release(slab.digest)
        if self._arena is not None:
            self._arena.close()
        log.info("mask ring %s closed and wiped (%d unreserved slots lost)", self.name, self.lost_total)

    def __enter__(self) -> MaskRing:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- metrics ---------------------------------------------------------------------------------
    def metrics(self) -> dict[str, float]:
        """``pgw_masks_ring_*`` values for this ring (labels are added by the caller)."""
        c = self.counts()
        return {
            "pgw_masks_ring_capacity_slots": float(self.capacity_slots),
            "pgw_masks_ring_available_slots": float(c["available_slots"]),
            "pgw_masks_ring_free_slabs": float(c["free_slabs"]),
            "pgw_masks_ring_draining_slabs": float(c["draining_slabs"]),
            "pgw_masks_ring_published_slots_total": float(c["published_slots"]),
            "pgw_masks_ring_reserved_slots_total": float(c["reserved_slots"]),
            "pgw_masks_ring_burned_slots_total": float(c["burned_slots"]),
            "pgw_masks_ring_leaked_slots_total": float(c["leaked_slots"]),
            "pgw_masks_ring_lost_slots_total": float(c["lost_slots"]),
            "pgw_masks_ring_reservations_total": float(c["reservations"]),
        }


# =================================================================================================
# writer
# =================================================================================================
def _pack_into(dst: memoryview, a: np.ndarray, ring: RingParams) -> None:
    """Write ring values ``a`` (canonical, ``< 2^K``) into ``dst`` in :func:`pgw.core.packing.pack_ring`
    encoding without an intermediate ``bytes`` object."""
    bpe = ring.bytes_per_element
    rows, m = a.shape
    if bpe in (2, 4):
        view = np.frombuffer(dst, dtype="<u4" if bpe == 4 else "<u2", count=rows * m).reshape(rows, m)
        np.copyto(view, a, casting="unsafe")  # values are range-checked: the wrap is exact
        del view
        return
    u = np.ascontiguousarray(a, dtype=np.uint64).astype("<u8", copy=False)
    by = u.view(np.uint8).reshape(rows * m, 8)
    view = np.frombuffer(dst, dtype=np.uint8, count=rows * m * bpe).reshape(rows * m, bpe)
    np.copyto(view, by[:, :bpe])
    del view, by
    u.fill(0)


def _as_ring_array(t: torch.Tensor, shape: tuple[int, int], ring: RingParams, what: str) -> np.ndarray:
    """Validated canonical ring values (int64, or 32-bit compact for K <= 32) as a numpy array."""
    if not isinstance(t, torch.Tensor):
        raise ShapeError(f"{what} must be a tensor")
    if tuple(t.shape) != shape:
        raise ShapeError(f"{what} has shape {tuple(t.shape)}, expected {shape}")
    a = t.detach().cpu().numpy()
    if a.dtype == np.int64:
        if a.size and (int(a.min()) < 0 or int(a.max()) > ring.mask):
            raise ShapeError(f"{what} has values outside the canonical ring range [0, 2^{ring.k})")
    elif a.dtype in (np.int32, np.uint32) and ring.k <= 32:
        u = a.view(np.uint32)
        if ring.k < 32 and u.size and int(u.max()) > ring.mask:  # copyto wraps mod 2^32, not 2^K
            raise ShapeError(f"{what} has values outside the canonical ring range [0, 2^{ring.k})")
        a = u
    else:
        raise ShapeError(f"{what} must be an int64 tensor (or 32-bit at K <= 32)")
    return a


def _check_floor(ring: MaskRing, checks: int) -> None:
    """``MaskStore._check_soundness`` for ring segments (the gateway's floor in ring-only mode)."""
    floor = ring.min_soundness_bits
    if floor <= 0:
        return
    from pgw.core.freivalds import soundness_bits

    rp = ring.ring
    weakest = min(ring.linears, key=lambda lin: soundness_bits(rp, lin.in_features, checks))
    bits = soundness_bits(rp, weakest.in_features, checks)
    if bits < floor:
        raise MaskStoreError(
            f"segment gives only {bits:.0f} bits of Freivalds soundness for {weakest.id} "
            f"(checks={checks}); the ring requires {floor:.0f}",
            details={"checks": checks, "min_soundness_bits": floor},
        )


class RingSegmentWriter:
    """Writes one RAM segment into a granted slab (``SegmentWriter`` contract). Obtain via
    :meth:`MaskRing.writer` (or :class:`pgw.factory.sinks.RingSink`)."""

    def __init__(self, ring: MaskRing, slab: _Slab, gen: int, *, seed: bytes, digest: str, token: str, slots: int,
                 checks: int, chunk_slots: int, factory_id: str):
        self._ring = ring
        self._slab = slab
        self.gen = gen
        self._seed = bytearray(seed)
        self._digest = digest
        self._token = token
        self.slots = slots
        self.checks = checks
        self.chunk_slots = chunk_slots
        self.factory_id = factory_id
        self.segment_id = uuid.uuid4().hex
        self._linears = {lin.id: (i, lin) for i, lin in enumerate(ring.linears)}
        bpe = ring.geometry.bpe
        rel, off = [], 0
        for lin in ring.linears:
            rel.append(off)
            off += slots * lin.out_features * bpe
        self._wr_rel = rel
        self._rows_done: dict[str, int] = dict.fromkeys(self._linears, 0)
        self._uw: dict[str, bytearray] = {}
        self._state = "open"
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"RingSegmentWriter(slab={self._slab.index}, slots={self.slots}, state={self._state})"

    @property
    def header_ring(self) -> RingParams:
        return self._ring.ring

    def write_linear(self, lid: LinearId | str, wr: torch.Tensor, uw: torch.Tensor) -> None:
        """Copy ``W r`` rows (consecutive chunks of any size, linears in any order) and check ``uW``."""
        key = str(lid)
        ring = self._ring.ring
        with self._lock:
            if self._state != "open":
                raise MaskStoreError(f"ring segment writer is {self._state}")
            if key not in self._linears:
                raise ConfigError(f"linear {key} is not part of this segment")
            li, lin = self._linears[key]
            if wr.ndim != 2:
                raise ShapeError(f"W r for {key} must be 2-D (rows, {lin.out_features})")
            rows = int(wr.shape[0])
            if rows <= 0:
                raise ShapeError("W r chunk has no rows")
            done = self._rows_done[key]
            if done + rows > self.slots:
                raise ShapeError(f"{key}: {done + rows} rows exceed the segment's {self.slots} slots")
            a = _as_ring_array(wr, (rows, lin.out_features), ring, f"W r of {key}")
            ua = _as_ring_array(uw, (self.checks, lin.in_features), ring, f"uW of {key}")
            packed = bytearray(self.checks * lin.in_features * ring.bytes_per_element)
            if len(packed):
                _pack_into(memoryview(packed), ua, ring)
            prev = self._uw.get(key)
            if prev is None:
                self._uw[key] = packed
            else:
                same = hmac.compare_digest(prev, packed)
                _zero(packed)
                if not same:
                    raise MaskStoreError(f"uW of {key} changed between chunks")
            self._ring._write_rows(self, li, done, a)
            self._rows_done[key] = done + rows

    def commit(self) -> str:
        """Validate, write seed and ``uW``, publish (READY). Returns the segment id."""
        ring = self._ring
        with self._lock:
            if self._state == "committed":
                return self.segment_id
            if self._state != "open":
                raise MaskStoreError(f"ring segment writer is {self._state}")
            try:
                missing = [k for k, n in self._rows_done.items() if n != self.slots]
                if missing:
                    raise MaskStoreError(f"segment incomplete: {len(missing)} linears lack rows (first {missing[0]})")
                self._check_soundness()
                header = SegmentHeader(
                    segment_id=self.segment_id, fingerprint=ring.fingerprint, ring=ring.ring.name, slots=self.slots,
                    chunk_slots=self.chunk_slots, checks=self.checks, linears=ring.linears, kek_id="ram",
                    wrapped_dek="", nonce_prefix="00000000", factory_id=self.factory_id,
                    created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
                secrets = bytearray(self._seed)
                for lin in ring.linears:
                    secrets += self._uw[lin.id]
                try:
                    ring._publish(self, header, secrets)
                finally:
                    _zero(secrets)
            except BaseException:
                self._state = "aborted"
                ring._abort_fill(self)
                self._wipe()
                raise
            self._state = "committed"
            self._wipe()
        log.info("ring %s: published segment %s (%d slots, slab %d)", ring.name, self.segment_id, self.slots,
                 self._slab.index)
        return self.segment_id

    def _check_soundness(self) -> None:
        _check_floor(self._ring, self.checks)

    def abort(self) -> None:
        """Discard the partial segment: the slab is wiped and freed."""
        with self._lock:
            if self._state != "open":
                return
            self._state = "aborted"
            self._ring._abort_fill(self)
            self._wipe()
        log.info("ring %s: aborted segment in slab %d", self._ring.name, self._slab.index)

    def _wipe(self) -> None:
        _zero(self._seed)
        for b in self._uw.values():
            _zero(b)
        self._uw.clear()

    def __enter__(self) -> RingSegmentWriter:
        return self

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None, tb: object) -> None:
        if et is None:
            self.commit()
        else:
            self.abort()
