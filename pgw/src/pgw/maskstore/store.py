"""Mask store API: segment writing (factory side), reservation and consumption (gateway side).

Single use is the one property that must never fail (SPEC.md 3.3 and 7.5). It is enforced by:

1. **Durable cursor before use.** :meth:`MaskStore.reserve` advances per-segment cursors and the
   global generation in one ``BEGIN IMMEDIATE`` transaction with ``synchronous=FULL`` (fsync), and
   only then decrypts anything for the caller. A crash loses masks but never reuses them.
2. **Serialisation.** An in-process lock plus an inter-process ``flock`` on ``<root>/.lock``
   serialise every consuming operation across gateway workers; SQLite's write lock is a second line.
3. **Rollback detection.** The generation and consumed totals are mirrored to an HMAC-chained
   journal and optionally a monotonic counter (file / TPM / HSM), see :mod:`pgw.maskstore.counters`.
   Segment rows and the meta record are HMAC-authenticated, and ``consumed_total`` must equal the
   sum of all cursors, so editing or replaying single rows is detected too.
4. **Seed uniqueness forever.** ``segment_id`` and ``SHA-256(seed)`` are UNIQUE across the whole
   history of the store; importing or writing the same segment twice is refused with
   :class:`MaskReuseError`.
5. **In-order hand-out.** A :class:`MaskReservation` hands out rows strictly in order per linear,
   never returns rows to the pool, and zeroises its buffers on :meth:`MaskReservation.close`.
"""

from __future__ import annotations

import base64
import collections
import contextlib
import errno
import fcntl
import hmac
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from pgw.core.errors import ConfigError, MaskExhaustedError, MaskReuseError, MaskStoreError, PGWError, ShapeError
from pgw.core.fingerprint import short_fingerprint, validate_fingerprint
from pgw.core.packing import pack_ring, unpack_ring
from pgw.core.prng import SEED_BYTES, SeedExpander
from pgw.core.ring import RingParams
from pgw.core.settings import ring_from_setting
from pgw.core.types import LinearId, LinearSpec
from pgw.maskstore.config import MaskStoreSettings
from pgw.maskstore.counters import (
    FileMonotonicCounter,
    GenerationJournal,
    MonotonicCounter,
    ProviderCounter,
    TpmNvCounter,
    fsync_dir,
)
from pgw.maskstore.format import (
    DEK_BYTES,
    NONCE_PREFIX_BYTES,
    SEGMENT_SUFFIX,
    SegmentFile,
    SegmentHeader,
    SegmentLinear,
    seed_hash,
)
from pgw.maskstore.keys import KeyProvider, RecipientKeyPair, WrappedKey
from pgw.maskstore.partner import TrustedPartners, read_signature
from pgw.maskstore.segment_io import RawSegmentWriter, shred_file
from pgw.maskstore.state import (
    ON_DISK,
    SCHEMA_VERSION,
    SELECT_SEGMENTS,
    SegmentRow,
    SegmentStatus,
    connect,
    meta_message,
    segment_message,
)

__all__ = [
    "IntegrityReport",
    "Inventory",
    "LowWaterListener",
    "MaskReservation",
    "MaskStore",
    "SegmentWriter",
    "SlotExtent",
]

log = logging.getLogger(__name__)



def _owner_host() -> str:
    """Identity of this host *and PID namespace* recorded with every reservation.

    Liveness checks by pid are only meaningful inside one PID namespace: two containers of one
    pod share the hostname but not their pids (both may be pid 1). Reservations made in another
    namespace are therefore treated as foreign (they only expire by ``lease_timeout_s``).
    """
    host = socket.gethostname()
    try:
        ns = os.readlink("/proc/self/ns/pid")
    except OSError:
        return host
    return f"{host}|{ns}"


_HOST = _owner_host()
_PART_MAX_AGE_S = 3600.0
_MAX_SLOTS_PER_SEGMENT = 1 << 24


#: Ids of reservations that are open in *this process*, across every :class:`MaskStore` instance.
#: Stale-lease detection consults it, so a second store object on the same root (gateway and
#: factory in one process, or a short-lived admin call) can never close or shred the segment of a
#: reservation another object is still reading.
_LIVE_RESERVATIONS: set[str] = set()
_LIVE_LOCK = threading.Lock()


def _live_add(rid: str) -> None:
    with _LIVE_LOCK:
        _LIVE_RESERVATIONS.add(rid)


def _live_discard(rid: str) -> None:
    with _LIVE_LOCK:
        _LIVE_RESERVATIONS.discard(rid)


def _is_live(rid: str) -> bool:
    with _LIVE_LOCK:
        return rid in _LIVE_RESERVATIONS


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ring_name(ring: RingParams | str) -> str:
    return ring_from_setting(ring).name


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True)
class SlotExtent:
    """A contiguous run of slots ``[start, start + count)`` inside one segment."""

    segment_id: str
    start: int
    count: int


@dataclass(frozen=True)
class Inventory:
    """Availability for one ``(fingerprint, ring)``."""

    fingerprint: str
    ring: str
    segments: int
    slots_total: int
    slots_available: int
    bytes_on_disk: int


@dataclass
class IntegrityReport:
    """Result of :meth:`MaskStore.verify_integrity` (problems that were repaired or tolerated)."""

    generation: int = 0
    consumed_total: int = 0
    segments_checked: int = 0
    resynced_witnesses: list[str] = field(default_factory=list)
    missing_segments: list[str] = field(default_factory=list)
    quarantined_segments: list[str] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)
    stale_reservations: int = 0
    warnings: list[str] = field(default_factory=list)


#: Called with the inventory when available slots drop below ``low_water_slots``.
LowWaterListener = Callable[[Inventory], None]


# =================================================================================================
# Reservation
# =================================================================================================
class _Extent:
    """Runtime state of one extent of a reservation (an open, keyed segment file)."""

    def __init__(self, ext: SlotExtent, seg: SegmentFile, first_row: int):
        self.ext = ext
        self.seg = seg
        self.first_row = first_row  #: reservation-global index of the extent's first row
        self.end_row = first_row + ext.count
        self._secrets: bytearray | None = None
        self.seed: bytes | None = None
        self._uw_views: list[memoryview] = []

    def secrets(self) -> tuple[bytes, list[memoryview]]:
        if self._secrets is None:
            plain = self.seg.decrypt(self.seg.layout.secrets_chunk())
            self._secrets = bytearray(plain)
            del plain
            self.seed, self._uw_views = self.seg.layout.secrets_split(self._secrets)
        assert self.seed is not None
        return self.seed, self._uw_views

    def wipe(self) -> None:
        self._uw_views = []
        if self._secrets is not None:
            for i in range(len(self._secrets)):
                self._secrets[i] = 0
            self._secrets = None
        self.seed = None
        self.seg.close()


class MaskReservation:
    """Slots exclusively owned by one generation session.

    Slots are already marked consumed in durable state when the reservation is created. Rows are
    handed out strictly in order: ``take(lid, T)`` returns the next ``T`` rows for linear ``lid``;
    each linear has its own cursor (every token position consumes one row of every linear).
    Unused rows are destroyed on :meth:`close` (never returned to the pool).

    Not created directly; use :meth:`MaskStore.reserve`. Thread-safe.
    """

    def __init__(
        self,
        store: MaskStore | None,
        reservation_id: str,
        session_id: str,
        fingerprint: str,
        ring: RingParams,
        extents: Sequence[tuple[SlotExtent, SegmentFile]],
    ):
        if not extents:
            raise MaskStoreError("reservation without extents")
        self.id = reservation_id
        self.session_id = session_id
        self.fingerprint = fingerprint
        self.ring = ring
        self._store = store
        self._lock = threading.Lock()
        self._closed = False
        self._extents: list[_Extent] = []
        row = 0
        for ext, seg in extents:
            self._extents.append(_Extent(ext, seg, row))
            row += ext.count
        self._slots = row
        h0 = self._extents[0].seg.header
        self._linears: dict[str, tuple[int, SegmentLinear]] = {lin.id: (i, lin) for i, lin in enumerate(h0.linears)}
        for e in self._extents[1:]:
            h = e.seg.header
            if h.fingerprint != h0.fingerprint or h.ring != h0.ring or h.linears != h0.linears:
                raise MaskStoreError("segments of one reservation disagree on model, ring or linears")
        if h0.fingerprint != fingerprint or h0.ring != ring.name:
            raise MaskStoreError("segment header does not match the reserved fingerprint/ring")
        self._checks = h0.checks
        self._cursor: dict[str, int] = dict.fromkeys(self._linears, 0)
        # (extent index, linear key) -> (chunk index, packed plaintext)
        self._chunk_cache: dict[tuple[int, str], tuple[int, bytearray]] = {}
        self._uw_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def __repr__(self) -> str:
        return (
            f"MaskReservation(id={self.id[:8]}, fp={short_fingerprint(self.fingerprint)}, ring={self.ring.name}, "
            f"slots={self._slots}, extents={len(self._extents)}, closed={self._closed})"
        )

    # ---- introspection ---------------------------------------------------------------------------
    @property
    def slots(self) -> int:
        """Total slots reserved."""
        return self._slots

    @property
    def extents(self) -> Sequence[SlotExtent]:
        """Where the slots live."""
        return tuple(e.ext for e in self._extents)

    @property
    def closed(self) -> bool:
        """True after :meth:`close`."""
        return self._closed

    @property
    def linear_keys(self) -> tuple[str, ...]:
        """Linear ids covered by this reservation (header order)."""
        return tuple(self._linears)

    def _key(self, lid: LinearId | str) -> str:
        key = str(lid)
        if key not in self._linears:
            raise ConfigError(f"linear {key} is not covered by this reservation")
        return key

    def _alive(self) -> None:
        if self._closed:
            raise MaskReuseError("mask reservation is closed; its unused rows were burned")

    def _extent_index(self, row: int) -> int:
        for i, e in enumerate(self._extents):
            if e.first_row <= row < e.end_row:
                return i
        raise MaskExhaustedError("mask reservation exhausted", details={"slots": self._slots})

    def remaining(self, lid: LinearId | str) -> int:
        """Rows not yet taken for ``lid``."""
        key = self._key(lid)
        with self._lock:
            return self._slots - self._cursor[key]

    def taken(self, lid: LinearId | str) -> int:
        """Rows already taken for ``lid``."""
        key = self._key(lid)
        with self._lock:
            return self._cursor[key]

    def rows_until_boundary(self, lid: LinearId | str) -> int:
        """Rows of ``lid`` left in the current segment extent."""
        key = self._key(lid)
        with self._lock:
            cur = self._cursor[key]
            if cur >= self._slots:
                return 0
            return self._extents[self._extent_index(cur)].end_row - cur

    # ---- masks -----------------------------------------------------------------------------------
    def take(self, lid: LinearId | str, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Next ``rows`` masks for ``lid``: ``(r (rows, n), W r (rows, m))`` int64 ring tensors.

        The call must not cross a segment-extent boundary (see :meth:`rows_until_boundary`),
        because Freivalds vectors differ per segment.

        Raises:
            MaskExhaustedError: fewer than ``rows`` remain.
            MaskReuseError: the reservation is closed.
            ConfigError: unknown linear, ``rows <= 0`` or the call crosses an extent boundary.
            MaskStoreError: decryption/authentication failure (the rows stay consumed).
        """
        key = self._key(lid)
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise ConfigError(f"rows must be a positive int, got {rows!r}")
        with self._lock:
            self._alive()
            start = self._cursor[key]
            if start + rows > self._slots:
                raise MaskExhaustedError(
                    "mask reservation exhausted",
                    details={"linear": key, "needed": rows, "remaining": self._slots - start},
                )
            ei = self._extent_index(start)
            ex = self._extents[ei]
            if start + rows > ex.end_row:
                raise ConfigError(
                    f"take({key}, {rows}) crosses a segment extent boundary; split at rows_until_boundary()",
                    details={"rows_until_boundary": ex.end_row - start},
                )
            # Advance first: whatever happens below, these rows are never handed out again.
            self._cursor[key] = start + rows
            li, lin = self._linears[key]
            slot0 = ex.ext.start + (start - ex.first_row)
            seed, _ = ex.secrets()
            r = SeedExpander(seed).ring_uniform_rows(lin.in_features, self.ring.k, slot0, rows, "r", key)
            wr = self._read_wr(ei, ex, li, lin, slot0, rows)
            if slot0 + rows >= ex.ext.start + ex.ext.count:
                self._chunk_cache.pop((ei, key), None)  # this linear is done with the extent
        return r, wr

    def _read_wr(self, ei: int, ex: _Extent, li: int, lin: SegmentLinear, slot0: int, rows: int) -> torch.Tensor:
        layout = ex.seg.layout
        cs = ex.seg.header.chunk_slots
        bpe = layout.bpe
        row_bytes = lin.out_features * bpe
        out = bytearray(rows * row_bytes)
        done = 0
        while done < rows:
            slot = slot0 + done
            j = slot // cs
            cached = self._chunk_cache.get((ei, lin.id))
            if cached is None or cached[0] != j:
                if cached is not None:
                    _zero(cached[1])
                plain = ex.seg.decrypt(layout.wr_chunk(li, j))
                cached = (j, bytearray(plain))
                del plain
                self._chunk_cache[(ei, lin.id)] = cached
            buf = cached[1]
            in_chunk = slot - j * cs
            n = min(rows - done, layout.chunk_rows(j) - in_chunk)
            out[done * row_bytes : (done + n) * row_bytes] = buf[in_chunk * row_bytes : (in_chunk + n) * row_bytes]
            done += n
        t = unpack_ring(out, (rows, lin.out_features), self.ring)
        _zero(out)
        return t

    def checks(self, lid: LinearId | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Freivalds vectors ``(u (c, m), uW (c, n))`` valid for the rows of the current extent.

        "Current" is the extent the next :meth:`take` of ``lid`` reads from (call this *before*
        ``take``); after the last row was taken it is the last extent. When a reservation spans
        segments, the engine must split calls at extent boundaries (``rows_until_boundary``)
        because ``u`` differs per segment.
        """
        key = self._key(lid)
        with self._lock:
            self._alive()
            if self._checks == 0:
                raise ConfigError("segments of this reservation carry no Freivalds vectors (checks=0)")
            cur = self._cursor[key]
            # Before take(): the extent the next rows come from. Once every row of the linear was
            # taken: the last extent (so a caller may also verify after taking).
            ei = self._extent_index(cur) if cur < self._slots else len(self._extents) - 1
            hit = self._uw_cache.get((ei, key))
            if hit is None:
                ex = self._extents[ei]
                li, lin = self._linears[key]
                seed, uw_views = ex.secrets()
                u = SeedExpander(seed).ring_uniform((self._checks, lin.out_features), self.ring.k, "u", key)
                uw = unpack_ring(uw_views[li], (self._checks, lin.in_features), self.ring)
                hit = (u, uw)
                self._uw_cache[(ei, key)] = hit
            return hit[0].clone(), hit[1].clone()

    # ---- lifecycle -------------------------------------------------------------------------------
    def close(self) -> None:
        """Zeroize in-memory secrets and burn unused rows (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for _, buf in self._chunk_cache.values():
                _zero(buf)
            self._chunk_cache.clear()
            for u, uw in self._uw_cache.values():
                u.zero_()
                uw.zero_()
            self._uw_cache.clear()
            for e in self._extents:
                e.wipe()
            used = max(self._cursor.values(), default=0)
            burned = self._slots - used
        _live_discard(self.id)
        store, self._store = self._store, None
        if store is not None:
            store._on_reservation_closed(self, burned)

    def __enter__(self) -> MaskReservation:
        return self

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort; close() is the contract
        if not getattr(self, "_closed", True):
            # Do not touch the store from a finalizer (locks); the lease is cleaned up by gc().
            self._store = None
            with contextlib.suppress(Exception):
                self.close()


def _zero(buf: bytearray) -> None:
    buf[:] = bytes(len(buf))


# =================================================================================================
# Writer
# =================================================================================================
class SegmentWriter:
    """Writes one segment (factory side). Obtain via :meth:`MaskStore.writer`.

    Usage::

        with store.writer(fp, ring, specs, slots=4096, seed=seed, checks=2) as w:
            for lid in ids:
                w.write_linear(lid, wr_rows, u_w)   # wr (slots, m), uW (checks, n)
        # commit on successful exit: fsync + atomic rename; abort deletes the temp file

    ``write_linear`` may be called several times per linear with consecutive row chunks of any size;
    linears may come in any order. ``uW`` must be identical on every call for a linear.
    """

    def __init__(
        self,
        store: MaskStore,
        header: SegmentHeader,
        final_path: Path,
        dek: bytes,
        seed: bytes,
    ):
        self._store = store
        self.header = header
        self.segment_id = header.segment_id
        self.ring = header.ring_params
        self._seed = bytearray(seed)
        self._seed_hash = seed_hash(seed)
        self._raw = RawSegmentWriter(final_path, header, dek, fsync=store.settings.fsync)
        self._layout = self._raw.layout
        self._linears = {lin.id: (i, lin) for i, lin in enumerate(header.linears)}
        self._rows_done: dict[str, int] = dict.fromkeys(self._linears, 0)
        self._pending: dict[str, bytearray] = {k: bytearray() for k in self._linears}
        self._uw: dict[str, bytes] = {}
        self._state = "open"
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"SegmentWriter(segment_id={self.segment_id}, slots={self.header.slots}, state={self._state})"

    def _check_tensor(self, t: torch.Tensor, shape: tuple[int, int], what: str) -> None:
        if not isinstance(t, torch.Tensor) or t.dtype != torch.int64:
            raise ShapeError(f"{what} must be an int64 tensor")
        if tuple(t.shape) != shape:
            raise ShapeError(f"{what} has shape {tuple(t.shape)}, expected {shape}")
        if t.numel() and (int(t.min()) < 0 or int(t.max()) > self.ring.mask):
            raise ShapeError(f"{what} has values outside the canonical ring range [0, 2^{self.ring.k})")

    def write_linear(self, lid: LinearId | str, wr: torch.Tensor, uw: torch.Tensor) -> None:
        """Encrypt and append ``W r`` rows (may be called in slot chunks, in order) and ``uW``."""
        key = str(lid)
        with self._lock:
            if self._state != "open":
                raise MaskStoreError(f"segment writer is {self._state}")
            if key not in self._linears:
                raise ConfigError(f"linear {key} is not part of this segment")
            li, lin = self._linears[key]
            if wr.ndim != 2:
                raise ShapeError(f"W r for {key} must be 2-D (rows, {lin.out_features})")
            rows = int(wr.shape[0])
            if rows <= 0:
                raise ShapeError("W r chunk has no rows")
            done = self._rows_done[key]
            if done + rows > self.header.slots:
                raise ShapeError(f"{key}: {done + rows} rows exceed the segment's {self.header.slots} slots")
            self._check_tensor(wr.cpu(), (rows, lin.out_features), f"W r of {key}")
            uw_c = uw.cpu()
            self._check_tensor(uw_c, (self.header.checks, lin.in_features), f"uW of {key}")
            packed_uw = pack_ring(uw_c, self.ring)
            prev = self._uw.get(key)
            if prev is None:
                self._uw[key] = packed_uw
            elif not hmac.compare_digest(prev, packed_uw):
                raise MaskStoreError(f"uW of {key} changed between chunks")
            pend = self._pending[key]
            pend += pack_ring(wr, self.ring)
            self._rows_done[key] = done + rows
            self._flush(key, li, lin)

    def _flush(self, key: str, li: int, lin: SegmentLinear) -> None:
        cs = self.header.chunk_slots
        row_bytes = lin.out_features * self._layout.bpe
        pend = self._pending[key]
        written_rows = self._rows_done[key] - len(pend) // row_bytes
        while pend:
            j = written_rows // cs
            need = self._layout.chunk_rows(j) * row_bytes
            if len(pend) < need:
                break
            self._raw.write_chunk(self._layout.wr_chunk(li, j), memoryview(pend)[:need])
            _zero_prefix(pend, need)
            del pend[:need]
            written_rows += self._layout.chunk_rows(j)

    def commit(self) -> str:
        """Finalize; returns the segment id."""
        with self._lock:
            if self._state == "committed":
                return self.segment_id
            if self._state != "open":
                raise MaskStoreError(f"segment writer is {self._state}")
            try:
                missing = [k for k, n in self._rows_done.items() if n != self.header.slots]
                if missing:
                    raise MaskStoreError(
                        f"segment incomplete: {len(missing)} linears lack rows (first {missing[0]})",
                        details={"segment_id": self.segment_id},
                    )
                secrets = bytearray(self._seed)
                for lin in self.header.linears:
                    secrets += self._uw[lin.id]
                self._raw.write_chunk(self._layout.secrets_chunk(), secrets)
                _zero(secrets)
                size = self._raw.publish()
                self._store._register_segment(self.header, self._seed_hash, self._raw.final_path, size)
            except BaseException:
                self._state = "aborted"
                self._raw.abort()
                with contextlib.suppress(OSError):
                    if self._raw.final_path.exists() and not self._store._knows_segment(self.segment_id):
                        self._raw.final_path.unlink()
                self._wipe()
                raise
            self._state = "committed"
            self._wipe()
        log.info(
            "committed segment %s: %s %s, %d slots, %.1f MB",
            self.segment_id, short_fingerprint(self.header.fingerprint), self.header.ring, self.header.slots,
            size / 1e6,
        )
        self._store._after_inventory_change(self.header.fingerprint, self.header.ring)
        return self.segment_id

    def abort(self) -> None:
        """Discard the partial segment."""
        with self._lock:
            if self._state != "open":
                return
            self._state = "aborted"
            self._raw.abort()
            self._wipe()
        log.info("aborted segment %s", self.segment_id)

    def _wipe(self) -> None:
        _zero(self._seed)
        for buf in self._pending.values():
            _zero(buf)
        self._pending.clear()
        self._uw.clear()

    def __enter__(self) -> SegmentWriter:
        return self

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None) -> None:
        if et is None:
            self.commit()
        else:
            self.abort()


def _zero_prefix(buf: bytearray, n: int) -> None:
    buf[:n] = bytes(n)


# =================================================================================================
# Store
# =================================================================================================
class _ProcessLock:
    """Re-entrant (per thread) lock that also holds an exclusive ``flock`` across processes."""

    def __init__(self, path: Path, timeout: float):
        self.path = path
        self.timeout = timeout
        self._rlock = threading.RLock()
        self._depth = 0
        self._fd: int | None = None

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        if not self._rlock.acquire(timeout=self.timeout):
            raise MaskStoreError("timed out waiting for the mask store lock (in-process)")
        try:
            if self._depth == 0:
                self._acquire_flock()
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    self._release_flock()
        finally:
            self._rlock.release()

    def _acquire_flock(self) -> None:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + self.timeout
        delay = 0.001
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    os.close(fd)
                    raise MaskStoreError(f"cannot lock mask store: {exc.strerror}") from exc
            if time.monotonic() >= deadline:
                os.close(fd)
                raise MaskStoreError("timed out waiting for the mask store lock (another process holds it)")
            time.sleep(delay)
            delay = min(delay * 2, 0.05)

    def _release_flock(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


@dataclass
class _Meta:
    store_id: str
    generation: int
    consumed: int
    journal_bound: int
    counter_kind: str
    counter_offset: int
    mac: str


class MaskStore:
    """Directory-backed mask store.

    Layout: ``<root>/<fingerprint>/<ring>/segments/<segment_id>.pgwm`` plus ``<root>/state.sqlite``
    (WAL mode) with per-segment consumption cursors and a global monotonic generation counter.
    Thread- and process-safe (SQLite ``BEGIN IMMEDIATE`` around cursor moves, plus an ``flock``).

    Args:
        settings: Store settings.
        key_provider: KEK provider (wraps segment DEKs; MACs state and journal when supported).
        recipient: X25519 key pair for importing partner segments (SPEC 7.7).
        counter: External monotonic counter; default built from ``settings.counter``.
        trusted_partners: Ed25519 keys of partner factories allowed to deliver segments; default
            built from ``settings.trusted_partner_keys``.
    """

    def __init__(
        self,
        settings: MaskStoreSettings,
        key_provider: KeyProvider,
        *,
        recipient: RecipientKeyPair | None = None,
        counter: MonotonicCounter | None = None,
        trusted_partners: TrustedPartners | None = None,
    ):
        self.settings = settings
        self.root = Path(settings.root)
        self.key_provider = key_provider
        self.recipient = recipient
        if self.recipient is None and settings.recipient_key_file is not None:
            self.recipient = RecipientKeyPair(settings.recipient_key_file)
        self.trusted_partners = (
            trusted_partners if trusted_partners is not None else TrustedPartners(settings.trusted_partner_keys)
        )
        self._pid = os.getpid()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = _ProcessLock(self.root / ".lock", settings.lock_timeout_s)
        self._conn: sqlite3.Connection | None = connect(
            self.root / "state.sqlite", timeout=settings.lock_timeout_s, fsync=settings.fsync
        )
        self._mac_ok = self._probe_mac()
        self.counter = counter if counter is not None else self._build_counter()
        self._journal: GenerationJournal | None = None
        self._dek_cache: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        self._listeners: list[LowWaterListener] = []
        self._below: dict[tuple[str, str], bool] = {}
        #: Reservations of this process that are not closed (None while being constructed).
        self._open_reservations: dict[str, MaskReservation | None] = {}
        self._tainted: str | None = None
        self._verified = False
        #: Test hook: called with a point name ("after_commit", "after_witness").
        self._fault_hook: Callable[[str], None] | None = None
        with self._lock.hold():
            meta = self._init_meta()
        if settings.journal and self._mac_ok:
            self._journal = GenerationJournal(
                settings.journal_file or (self.root / "generation.journal"),
                meta.store_id,
                self._mac,
                fsync=settings.fsync,
            )
            with self._lock.hold():
                self._journal.recover_pending()
        elif settings.journal:
            log.warning(
                "key provider %s cannot MAC: generation journal and row authentication disabled; "
                "rollback detection relies on the monotonic counter (%s)",
                type(key_provider).__name__, self.counter.kind if self.counter else "none",
            )
        if self.counter is None and self._journal is not None and not getattr(self.counter, "hardware", False):
            log.info("rollback protection: HMAC journal only (no hardware counter configured)")
        if settings.verify_on_open:
            self.verify_integrity()
        log.info(
            "opened mask store %s (store %s, generation %d)", self.root, meta.store_id[:8], meta.generation
        )

    # ---- construction helpers --------------------------------------------------------------------
    @classmethod
    def open(cls, root: str | Path, key_provider: KeyProvider, **overrides: object) -> MaskStore:
        """Open (creating if needed) a store at ``root``.

        ``overrides`` are :class:`MaskStoreSettings` fields, plus ``recipient`` (a
        :class:`RecipientKeyPair`), ``trusted_partners`` (a :class:`TrustedPartners`) and
        ``counter``, which may be a settings value (``"file"``, ``"tpm"``, ...) or a
        :class:`MonotonicCounter` instance.
        """
        recipient = overrides.pop("recipient", None)
        trusted = overrides.pop("trusted_partners", None)
        counter = overrides.get("counter")
        if isinstance(counter, MonotonicCounter):
            overrides.pop("counter")
        else:
            counter = None
        settings = MaskStoreSettings(root=Path(root), **overrides)  # type: ignore[arg-type]
        return cls(
            settings, key_provider, recipient=recipient, counter=counter, trusted_partners=trusted  # type: ignore[arg-type]
        )

    def _probe_mac(self) -> bool:
        fn = getattr(self.key_provider, "mac", None)
        if fn is None:
            return False
        try:
            return fn(b"pgw/v1/probe") is not None
        except NotImplementedError:
            return False

    def _mac(self, data: bytes) -> bytes | None:
        if not self._mac_ok:
            return None
        return self.key_provider.mac(data)

    def _mac_hex(self, data: bytes) -> str:
        m = self._mac(data)
        return m.hex() if m is not None else ""

    def _check_mac(self, data: bytes, mac_hex: str, what: str) -> None:
        if not self._mac_ok:
            return
        expected = self._mac_hex(data)
        if not hmac.compare_digest(expected, mac_hex):
            self._taint(f"{what} failed authentication (state database was modified outside the store)")

    def _build_counter(self) -> MonotonicCounter | None:
        s = self.settings
        if s.counter == "none":
            return None
        if s.counter == "file":
            if s.counter_file is None:
                raise ConfigError("counter='file' needs counter_file (ideally on a separate volume)")
            return FileMonotonicCounter(s.counter_file, fsync=s.fsync)
        if s.counter == "tpm":
            if s.tpm_nv_index is None:
                raise ConfigError("counter='tpm' needs tpm_nv_index")
            return TpmNvCounter(s.tpm_nv_index, tcti=s.tpm_tcti)
        return ProviderCounter(self.key_provider)

    @property
    def conn(self) -> sqlite3.Connection:
        """The state database connection."""
        if self._conn is None:
            raise MaskStoreError("mask store is closed")
        if os.getpid() != self._pid:
            raise MaskStoreError("MaskStore used across fork(); open a new MaskStore in the child process")
        return self._conn

    @contextlib.contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise MaskStoreError(f"cannot start state transaction: {exc}") from exc
        try:
            yield conn
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        try:
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise MaskStoreError(f"state commit failed: {exc}") from exc

    # ---- meta ------------------------------------------------------------------------------------
    def _read_meta(self, conn: sqlite3.Connection) -> _Meta:
        rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        try:
            return _Meta(
                rows["store_id"], int(rows["generation"]), int(rows["consumed_total"]),
                int(rows.get("journal_bound", "0")), rows.get("counter_kind", "none"),
                int(rows.get("counter_offset", "0")), rows.get("meta_mac", ""),
            )
        except (KeyError, ValueError) as exc:
            raise MaskStoreError("mask store meta record is missing or corrupt") from exc

    def _meta_msg(self, m: _Meta) -> bytes:
        return meta_message(m.store_id, m.generation, m.consumed, m.journal_bound, m.counter_kind, m.counter_offset)

    def _write_meta(self, conn: sqlite3.Connection, m: _Meta) -> None:
        m.mac = self._mac_hex(self._meta_msg(m))
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [
                ("store_id", m.store_id), ("generation", str(m.generation)), ("consumed_total", str(m.consumed)),
                ("journal_bound", str(m.journal_bound)), ("counter_kind", m.counter_kind),
                ("counter_offset", str(m.counter_offset)), ("meta_mac", m.mac),
                ("schema_version", str(SCHEMA_VERSION)),
            ],
        )

    def _load_meta(self, conn: sqlite3.Connection) -> _Meta:
        m = self._read_meta(conn)
        self._check_mac(self._meta_msg(m), m.mac, "meta record")
        return m

    def _init_meta(self) -> _Meta:
        with self._txn() as conn:
            exists = conn.execute("SELECT value FROM meta WHERE key='store_id'").fetchone()
            if exists is None:
                m = _Meta(uuid.uuid4().hex, 0, 0, 0, "none", 0, "")
                self._write_meta(conn, m)
                conn.execute("INSERT INTO meta(key, value) VALUES('created_at', ?)", (_now_iso(),))
                log.info("initialised new mask store %s at %s", m.store_id[:8], self.root)
                return m
            ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if ver is not None and int(ver[0]) > SCHEMA_VERSION:
                raise MaskStoreError(f"state database schema {ver[0]} is newer than this software ({SCHEMA_VERSION})")
            return self._load_meta(conn)

    @property
    def store_id(self) -> str:
        """Random id of this store (binds journal records and MACs)."""
        with self._lock.hold():
            return self._read_meta(self.conn).store_id

    @property
    def generation(self) -> int:
        """Current global generation counter."""
        with self._lock.hold():
            return self._read_meta(self.conn).generation

    # ---- integrity / witnesses ---------------------------------------------------------------
    def _taint(self, reason: str) -> None:
        self._tainted = reason
        log.critical("mask store refuses to serve: %s", reason)
        raise MaskReuseError(f"mask store integrity violation: {reason}", details={"root": str(self.root)})

    def _ready(self) -> None:
        if self._tainted is not None:
            raise MaskReuseError(f"mask store integrity violation: {self._tainted}")
        if not self._verified:
            raise MaskStoreError("mask store has not passed verify_integrity()")

    def _sync_witnesses(self, conn: sqlite3.Connection, m: _Meta, report: IntegrityReport | None) -> None:
        """Compare DB with witnesses; refuse on rollback, resync when the DB is (safely) ahead.

        Must run inside a transaction while holding the store lock.
        """
        if self._journal is not None:
            head = self._journal.verify() if report is not None else self._journal.head()
            if head is None:
                if m.journal_bound:
                    self._taint("generation journal is missing (deleted, or the store directory was restored)")
                self._journal.append(m.generation, m.consumed)
                m.journal_bound = 1
                self._write_meta(conn, m)
                if report is not None:
                    report.resynced_witnesses.append("journal:created")
            elif head.generation > m.generation or head.consumed > m.consumed:
                self._taint(
                    f"rollback detected: journal at generation {head.generation} (consumed {head.consumed}) "
                    f"but state database at {m.generation} (consumed {m.consumed})"
                )
            elif head.generation < m.generation or head.consumed < m.consumed:
                log.warning(
                    "generation journal behind state (%d < %d); resynchronising after an interrupted operation",
                    head.generation, m.generation,
                )
                self._journal.append(m.generation, m.consumed)
                if report is not None:
                    report.resynced_witnesses.append("journal")
            if not m.journal_bound:
                m.journal_bound = 1
                self._write_meta(conn, m)
        elif m.journal_bound and report is not None:
            msg = "generation journal was bound to this store but is now disabled (weaker rollback protection)"
            log.warning(msg)
            report.warnings.append(msg)
        if self.counter is not None:
            kind = self.counter.kind
            value = self.counter.read()
            if m.counter_kind != kind:
                if m.counter_kind != "none" and report is not None:
                    report.warnings.append(f"monotonic counter changed from {m.counter_kind} to {kind}")
                m.counter_kind = kind
                m.counter_offset = value - m.generation
                self._write_meta(conn, m)
                if report is not None:
                    report.resynced_witnesses.append(f"counter:{kind}:bound")
                return
            derived = value - m.counter_offset
            if derived > m.generation:
                self._taint(
                    f"rollback detected: monotonic counter implies generation {derived} "
                    f"but state database is at {m.generation}"
                )
            if derived < m.generation:
                log.warning("monotonic counter behind state by %d; resynchronising", m.generation - derived)
                if m.generation - derived <= 16:
                    for _ in range(m.generation - derived):
                        value = self.counter.increment()
                m.counter_offset = value - m.generation
                self._write_meta(conn, m)
                if report is not None:
                    report.resynced_witnesses.append(f"counter:{kind}")
        elif m.counter_kind != "none" and report is not None:
            msg = f"monotonic counter {m.counter_kind} was bound to this store but is not configured now"
            log.warning(msg)
            report.warnings.append(msg)

    def _advance_witnesses(self, m: _Meta) -> None:
        if self._journal is not None:
            self._journal.append(m.generation, m.consumed)
        if self.counter is not None and m.counter_kind == self.counter.kind:
            value = self.counter.increment()
            if value - m.counter_offset != m.generation:
                self._taint(
                    f"monotonic counter out of step (counter implies {value - m.counter_offset}, "
                    f"generation {m.generation})"
                )

    def _check_consumed_sum(self, conn: sqlite3.Connection, m: _Meta) -> None:
        (total,) = conn.execute("SELECT COALESCE(SUM(next_slot), 0) FROM segments").fetchone()
        if int(total) != m.consumed:
            self._taint(
                f"consumption cursors were modified: sum of cursors {total} != consumed_total {m.consumed}"
            )

    def _row_mac_msg(self, store_id: str, r: SegmentRow) -> bytes:
        return segment_message(
            store_id, r.segment_id, r.seed_hash, r.fingerprint, r.ring, r.slots, r.next_slot, r.status, r.updated_gen
        )

    def _check_row(self, store_id: str, r: SegmentRow) -> None:
        self._check_mac(self._row_mac_msg(store_id, r), r.mac, f"segment row {r.segment_id}")
        if r.status != SegmentStatus.ACTIVE and r.next_slot != r.slots:
            self._taint(f"segment {r.segment_id} is {r.status} but its cursor is {r.next_slot}/{r.slots}")

    def _update_row(
        self, conn: sqlite3.Connection, store_id: str, r: SegmentRow, *, next_slot: int, status: str, gen: int
    ) -> None:
        if next_slot < r.next_slot:
            raise MaskReuseError("refusing to move a consumption cursor backwards")
        nr = SegmentRow(
            r.seq, r.segment_id, r.seed_hash, r.fingerprint, r.ring, r.slots, next_slot, status, r.relpath,
            r.bytes, r.kek_id, r.factory_id, r.created_at, r.added_at, gen, "",
        )
        mac = self._mac_hex(self._row_mac_msg(store_id, nr))
        cur = conn.execute(
            "UPDATE segments SET next_slot=?, status=?, updated_gen=?, mac=? WHERE segment_id=? AND next_slot=?",
            (next_slot, status, gen, mac, r.segment_id, r.next_slot),
        )
        if cur.rowcount != 1:
            raise MaskStoreError(f"concurrent modification of segment {r.segment_id}")

    def verify_integrity(self) -> IntegrityReport:
        """Check state DB vs. segment files and the provider's monotonic counter (rollback detection).

        Repairs what is safe to repair (witnesses behind after a crash, missing or corrupt segment
        files are written off, dead reservations closed) and refuses to serve (raises
        :class:`MaskReuseError`, store stays unusable) on rollback or state tampering.
        """
        report = IntegrityReport()
        written_off: list[tuple[str, str]] = []
        with self._lock.hold():
            with self._txn() as conn:
                m = self._load_meta(conn)
                rows = [SegmentRow.from_row(t) for t in conn.execute(SELECT_SEGMENTS)]
                for r in rows:
                    self._check_row(m.store_id, r)
                self._check_consumed_sum(conn, m)
                self._sync_witnesses(conn, m, report)
                # segment files
                for r in rows:
                    if r.status not in ON_DISK:
                        continue
                    report.segments_checked += 1
                    problem = self._check_segment_file(r)
                    if problem is None:
                        continue
                    status = SegmentStatus.MISSING if problem == "missing" else SegmentStatus.QUARANTINED
                    (report.missing_segments if problem == "missing" else report.quarantined_segments).append(
                        r.segment_id
                    )
                    if r.status == SegmentStatus.ACTIVE or status == SegmentStatus.MISSING:
                        m.generation += 1
                        m.consumed += r.slots - r.next_slot
                        self._update_row(conn, m.store_id, r, next_slot=r.slots, status=status, gen=m.generation)
                        written_off.append((r.fingerprint, r.ring))
                if written_off:
                    self._write_meta(conn, m)
                report.stale_reservations = self._close_stale_reservations(conn)
                report.orphan_files = self._find_orphans({r.relpath for r in rows})
                report.generation, report.consumed_total = m.generation, m.consumed
            if written_off:
                self._advance_witnesses(m)
            self._verified = True
            self._tainted = None
        for fp, ring in set(written_off):
            self._after_inventory_change(fp, ring)
        if report.missing_segments or report.quarantined_segments:
            log.error(
                "mask store: %d segment files missing, %d quarantined; their remaining slots were written off",
                len(report.missing_segments), len(report.quarantined_segments),
            )
        return report

    def _check_segment_file(self, r: SegmentRow) -> str | None:
        path = self.root / r.relpath
        if not path.exists():
            return "missing"
        try:
            with SegmentFile(path) as seg:
                h = seg.header
                if (h.segment_id, h.fingerprint, h.ring, h.slots) != (r.segment_id, r.fingerprint, r.ring, r.slots):
                    log.error("segment %s: header does not match its state row", r.segment_id)
                    return "quarantined"
        except MaskStoreError as exc:
            log.error("segment %s failed validation: %s", r.segment_id, exc.message)
            return "quarantined"
        return None

    def _find_orphans(self, known: set[str]) -> list[str]:
        out: list[str] = []
        now = time.time()
        for p in self.root.glob(f"*/*/segments/*{SEGMENT_SUFFIX}*"):
            rel = str(p.relative_to(self.root))
            if p.name.endswith(".part"):
                with contextlib.suppress(OSError):
                    if now - p.stat().st_mtime > _PART_MAX_AGE_S:
                        p.unlink()
                        log.info("removed stale partial segment %s", p.name)
                continue
            if rel not in known:
                out.append(rel)
                log.warning("segment file %s is not registered in the state database; it will never be used", rel)
        return out

    def _close_stale_reservations(self, conn: sqlite3.Connection) -> int:
        now = time.time()
        stale = 0
        for rid, pid, host, created in conn.execute(
            "SELECT id, pid, host, created_at FROM reservations WHERE closed_at IS NULL"
        ).fetchall():
            if rid in self._open_reservations or _is_live(rid):
                continue
            dead = host == _HOST and (int(pid) == self._pid or not _pid_alive(int(pid)))
            if dead or now - float(created) > self.settings.lease_timeout_s:
                conn.execute("UPDATE reservations SET closed_at=?, burned=-1 WHERE id=?", (now, rid))
                stale += 1
        return stale

    def rebind_witnesses(self, *, i_understand_this_can_hide_a_rollback: bool = False) -> None:
        """Operator escape hatch: accept the current DB state as truth and re-create the witnesses.

        Only after investigating a rollback alarm (e.g. a known, intentional restore *and* making
        sure no reservation made after the restored state was ever used).
        """
        if not i_understand_this_can_hide_a_rollback:
            raise ConfigError("rebind_witnesses requires explicit acknowledgement")
        with self._lock.hold(), self._txn() as conn:
            m = self._read_meta(conn)
            if self._journal is not None:
                with contextlib.suppress(FileNotFoundError):
                    self._journal.path.unlink()
                self._journal.append(m.generation, m.consumed)
                m.journal_bound = 1
            if self.counter is not None:
                m.counter_kind = self.counter.kind
                m.counter_offset = self.counter.read() - m.generation
            self._write_meta(conn, m)
            rows = [SegmentRow.from_row(t) for t in conn.execute(SELECT_SEGMENTS)]
            for r in rows:
                mac = self._mac_hex(self._row_mac_msg(m.store_id, r))
                conn.execute("UPDATE segments SET mac=? WHERE segment_id=?", (mac, r.segment_id))
        self._tainted = None
        log.critical("mask store witnesses re-bound by operator at generation %d", m.generation)

    # ---- writing / import ------------------------------------------------------------------------
    def _segment_dir(self, fingerprint: str, ring: str) -> Path:
        return self.root / fingerprint / ring / "segments"

    def _knows_segment(self, segment_id: str) -> bool:
        with self._lock.hold():
            return (
                self.conn.execute("SELECT 1 FROM segments WHERE segment_id=?", (segment_id,)).fetchone() is not None
            )

    def _check_new(self, segment_id: str, shash: str) -> None:
        row = self.conn.execute(
            "SELECT segment_id, seed_hash FROM segments WHERE segment_id=? OR seed_hash=?", (segment_id, shash)
        ).fetchone()
        if row is not None:
            what = "segment id" if row[0] == segment_id else "seed"
            raise MaskReuseError(
                f"this {what} was already registered in the store (duplicate segment = mask reuse)",
                details={"segment_id": segment_id},
            )

    def writer(
        self,
        fingerprint: str,
        ring: RingParams | str,
        specs: Sequence[LinearSpec],
        *,
        slots: int,
        seed: bytes | None = None,
        checks: int = 2,
        chunk_slots: int = 256,
        factory_id: str = "local",
    ) -> SegmentWriter:
        """Start writing a new segment.

        Args:
            fingerprint: Model pack fingerprint.
            ring: Ring preset.
            specs: Linear specs in canonical order (header order).
            slots: Token positions.
            seed: 32-byte segment seed (default: fresh ``os.urandom``). Must be unique forever.
            checks: Freivalds vectors per linear.
            chunk_slots: Slots per encrypted ``W r`` chunk (smaller = less to decrypt per decode step).
            factory_id: Producer id recorded in the header.
        """
        validate_fingerprint(fingerprint)
        rp = ring_from_setting(ring)
        if not specs:
            raise ConfigError("a segment needs at least one linear")
        if not 0 < slots <= _MAX_SLOTS_PER_SEGMENT:
            raise ConfigError(f"slots must be in [1, {_MAX_SLOTS_PER_SEGMENT}]")
        if chunk_slots <= 0:
            raise ConfigError("chunk_slots must be positive")
        if not 0 <= checks <= 8:
            raise ConfigError("checks must be in [0, 8]")
        if seed is None:
            seed = os.urandom(SEED_BYTES)
        if not isinstance(seed, bytes | bytearray) or len(seed) != SEED_BYTES:
            raise ConfigError(f"seed must be exactly {SEED_BYTES} bytes")
        for s in specs:
            rp.check_in_features(s.in_features)
        segment_id = uuid.uuid4().hex
        with self._lock.hold():
            self._ready()
            self._check_new(segment_id, seed_hash(bytes(seed)))
        dek = os.urandom(DEK_BYTES)
        wrapped = self.key_provider.wrap(dek)
        header = SegmentHeader(
            segment_id=segment_id,
            fingerprint=fingerprint,
            ring=rp.name,
            slots=slots,
            chunk_slots=min(chunk_slots, slots),
            checks=checks,
            linears=tuple(
                SegmentLinear(id=s.id.key, out_features=s.out_features, in_features=s.in_features) for s in specs
            ),
            kek_id=wrapped.kek_id,
            wrapped_dek=base64.b64encode(wrapped.wrapped).decode(),
            nonce_prefix=os.urandom(NONCE_PREFIX_BYTES).hex(),
            factory_id=factory_id,
            created_at=_now_iso(),
        )
        path = self._segment_dir(fingerprint, rp.name) / f"{segment_id}{SEGMENT_SUFFIX}"
        try:
            return SegmentWriter(self, header, path, dek, bytes(seed))
        finally:
            del dek

    def _register_segment(self, header: SegmentHeader, shash: str, path: Path, size: int) -> None:
        rel = str(path.relative_to(self.root))
        with self._lock.hold():
            self._ready()
            with self._txn() as conn:
                m = self._load_meta(conn)
                self._check_new(header.segment_id, shash)
                r = SegmentRow(
                    0, header.segment_id, shash, header.fingerprint, header.ring, header.slots, 0,
                    SegmentStatus.ACTIVE.value, rel, size, header.kek_id, header.factory_id, header.created_at,
                    time.time(), m.generation, "",
                )
                mac = self._mac_hex(self._row_mac_msg(m.store_id, r))
                try:
                    conn.execute(
                        "INSERT INTO segments(segment_id, seed_hash, fingerprint, ring, slots, next_slot, status,"
                        " relpath, bytes, kek_id, factory_id, created_at, added_at, updated_gen, mac)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            r.segment_id, r.seed_hash, r.fingerprint, r.ring, r.slots, 0, r.status, rel, size,
                            r.kek_id, r.factory_id, r.created_at, r.added_at, r.updated_gen, mac,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise MaskReuseError("duplicate segment id or seed (mask reuse refused)") from exc
                conn.execute(
                    "INSERT INTO counters(fingerprint, ring) VALUES(?, ?) ON CONFLICT DO NOTHING",
                    (header.fingerprint, header.ring),
                )

    def import_segment(self, path: str | Path, *, recipient: RecipientKeyPair | None = None) -> str:
        """Import a segment file produced elsewhere (partner factory); validates header and AEAD.

        Every chunk is authenticated; the segment is re-encrypted under a fresh DEK wrapped by the
        local KEK (so the partner-held wrapping is not needed any more). The source file is left in
        place; delete it after a successful import. Importing the same segment (or any segment with
        the same seed) twice raises :class:`MaskReuseError`.

        Partner (HPKE-sealed) segments must carry a valid ``<file>.sig`` from a trusted partner key
        when ``trusted_partner_keys`` is configured or ``require_partner_signature`` is set (see
        :mod:`pgw.maskstore.partner`: HPKE base mode does not authenticate the sender, so without a
        signature anyone holding the public recipient key could inject masks whose seed they know).

        Raises:
            MaskReuseError: segment id or seed already known to this store.
            MaskStoreError: bad file, failed authentication, missing/invalid partner signature.
        """
        src_path = Path(path)
        rec = recipient or self.recipient
        with SegmentFile(src_path) as src:
            h = src.header
            for lin in h.linears:
                h.ring_params.check_in_features(lin.in_features)
            with self._lock.hold():
                self._ready()
                row = self.conn.execute("SELECT 1 FROM segments WHERE segment_id=?", (h.segment_id,)).fetchone()
            if row is not None:
                raise MaskReuseError(
                    "segment was already imported (duplicate segment = mask reuse)",
                    details={"segment_id": h.segment_id},
                )
            signer = self._check_partner_signature(src_path, src, h)
            wrapped = WrappedKey(h.kek_id, h.wrapped_dek_bytes)
            if h.kek_id.startswith("hpke:"):
                if rec is None:
                    raise MaskStoreError(
                        "segment is HPKE-sealed to a recipient key but no recipient key pair is configured",
                        details={"kek_id": h.kek_id},
                    )
                src_dek = rec.unwrap(wrapped)
            else:
                src_dek = self.key_provider.unwrap(wrapped)
            src.set_dek(src_dek)
            del src_dek
            secrets = bytearray(src.decrypt(src.layout.secrets_chunk()))
            final = self._segment_dir(h.fingerprint, h.ring) / f"{h.segment_id}{SEGMENT_SUFFIX}"
            published = False
            try:
                seed, _ = src.layout.secrets_split(secrets)
                shash = seed_hash(seed)
                del seed
                with self._lock.hold():
                    self._check_new(h.segment_id, shash)
                dek = os.urandom(DEK_BYTES)
                new_wrapped = self.key_provider.wrap(dek)
                new_header = h.model_copy(
                    update={
                        "kek_id": new_wrapped.kek_id,
                        "wrapped_dek": base64.b64encode(new_wrapped.wrapped).decode(),
                        "nonce_prefix": os.urandom(NONCE_PREFIX_BYTES).hex(),
                    }
                )
                size = self._reencrypt(src, final, new_header, dek, secrets)
                published = True
                del dek
                self._register_segment(new_header, shash, final, size)
            except BaseException:
                _zero(secrets)
                # Only remove a file *this* call published: a concurrent import of the same segment
                # may own the file at ``final`` (its registration then refuses ours as a duplicate).
                if published and not self._knows_segment(h.segment_id):
                    with contextlib.suppress(OSError):
                        final.unlink()
                raise
        log.info(
            "imported segment %s (%s %s, %d slots) from %s (factory %s%s)",
            h.segment_id, short_fingerprint(h.fingerprint), h.ring, h.slots, src_path.name, h.factory_id,
            f", signed by partner key {signer}" if signer else "",
        )
        self._after_inventory_change(h.fingerprint, h.ring)
        return h.segment_id

    def _check_partner_signature(self, src_path: Path, src: SegmentFile, h: SegmentHeader) -> str | None:
        """Enforce the partner-signature policy for HPKE segments; return the signer key id."""
        if not h.kek_id.startswith("hpke:"):
            return None  # wrapped under our own KEK: only a holder of the KEK can have produced it
        required = bool(self.trusted_partners) or self.settings.require_partner_signature
        sig = read_signature(src_path)
        if required:
            if not self.trusted_partners:
                raise ConfigError("require_partner_signature is set but no trusted_partner_keys are configured")
            return self.trusted_partners.verify(src.header_raw, h.segment_id, sig)
        log.warning(
            "importing partner segment %s WITHOUT sender authentication (%s); configure "
            "trusted_partner_keys so only segments signed by your partner factory are accepted",
            h.segment_id, "signature present but no trusted keys" if sig is not None else "unsigned",
        )
        return None

    def _reencrypt(
        self,
        src: SegmentFile,
        final: Path,
        new_header: SegmentHeader,
        dek: bytes,
        secrets: bytearray,
        *,
        replace: bool = False,
    ) -> int:
        """Re-encrypt every chunk of ``src`` (DEK installed) into ``final`` under ``dek``/``new_header``.

        ``secrets`` is the decrypted chunk 0 (zeroised here). Returns the new file size. Every
        source chunk is authenticated on the way; on any failure the partial file is removed.
        """
        raw = RawSegmentWriter(final, new_header, dek, fsync=self.settings.fsync)
        try:
            raw.write_chunk(raw.layout.secrets_chunk(), secrets)
            _zero(secrets)
            # the new header may have a different length, so offsets differ: pair chunks by index
            for sref, dref in zip(src.iter_chunks()[1:], raw.layout.chunks()[1:], strict=True):
                plain = bytearray(src.decrypt(sref))
                raw.write_chunk(dref, plain)
                _zero(plain)
            return raw.publish(replace=replace)
        except BaseException:
            raw.abort()
            _zero(secrets)
            raise

    def export_segment(self, segment_id: str, dest_dir: str | Path) -> Path:
        """Move an untouched segment out of this store (partner staging store -> outbox).

        The segment is marked ``exported`` (its slots can never be reserved here) before the file
        leaves, so a crash cannot result in the same segment being both exported and used.
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        with self._lock.hold():
            self._ready()
            with self._txn() as conn:
                m = self._load_meta(conn)
                t = conn.execute(SELECT_SEGMENTS + " WHERE segment_id=?", (segment_id,)).fetchone()
                if t is None:
                    raise MaskStoreError(f"unknown segment {segment_id}")
                r = SegmentRow.from_row(t)
                self._check_row(m.store_id, r)
                if r.status != SegmentStatus.ACTIVE or r.next_slot != 0:
                    raise MaskReuseError(
                        f"segment {segment_id} was already (partly) used or exported; refusing to export",
                        details={"status": r.status, "next_slot": r.next_slot},
                    )
                m.generation += 1
                m.consumed += r.slots
                self._update_row(
                    conn, m.store_id, r, next_slot=r.slots, status=SegmentStatus.EXPORTED, gen=m.generation
                )
                self._write_meta(conn, m)
            self._advance_witnesses(m)
            src = self.root / r.relpath
            out = dest / src.name
            try:
                os.replace(src, out)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise MaskStoreError(f"cannot move segment to {dest}: {exc.strerror}") from exc
                tmp = out.with_name(f".{out.name}.part")
                with src.open("rb") as fi, tmp.open("wb") as fo:
                    while chunk := fi.read(1 << 20):
                        fo.write(chunk)
                    fo.flush()
                    os.fsync(fo.fileno())
                os.replace(tmp, out)
                src.unlink()
            if self.settings.fsync:
                fsync_dir(dest)
                fsync_dir(src.parent)
        self._after_inventory_change(r.fingerprint, r.ring)
        return out

    # ---- reservation -----------------------------------------------------------------------------
    def reserve(
        self, fingerprint: str, ring: RingParams | str, slots: int, *, session_id: str = ""
    ) -> MaskReservation:
        """Atomically consume ``slots`` slots (oldest segments first) and return them.

        Raises:
            MaskExhaustedError: not enough slots available.
            MaskReuseError: the store detected rollback/tampering and refuses to serve.
        """
        if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
            raise ConfigError(f"slots must be a positive int, got {slots!r}")
        rname = _ring_name(ring)
        rp = RingParams.from_name(rname)
        rid = uuid.uuid4().hex
        exhausted: MaskExhaustedError | None = None
        with self._lock.hold():
            self._ready()
            try:
                with self._txn() as conn:
                    m = self._load_meta(conn)
                    self._check_consumed_sum(conn, m)
                    self._sync_witnesses(conn, m, None)
                    rows = [
                        SegmentRow.from_row(t)
                        for t in conn.execute(
                            SELECT_SEGMENTS + " WHERE fingerprint=? AND ring=? AND status=?"
                            " ORDER BY seq",
                            (fingerprint, rname, SegmentStatus.ACTIVE.value),
                        )
                    ]
                    picked: list[tuple[SegmentRow, int]] = []
                    need = slots
                    for r in rows:
                        if need == 0:
                            break
                        self._check_row(m.store_id, r)
                        take = min(need, r.slots - r.next_slot)
                        if take > 0:
                            picked.append((r, take))
                            need -= take
                    if need > 0:
                        raise MaskExhaustedError(
                            "not enough mask slots in the store",
                            details={
                                "fingerprint": short_fingerprint(fingerprint), "ring": rname, "needed": slots,
                                "available": slots - need,
                            },
                        )
                    m.generation += 1
                    m.consumed += slots
                    extents: list[tuple[SlotExtent, SegmentRow]] = []
                    for r, take in picked:
                        nxt = r.next_slot + take
                        status = SegmentStatus.CONSUMED if nxt == r.slots else SegmentStatus.ACTIVE
                        self._update_row(conn, m.store_id, r, next_slot=nxt, status=status, gen=m.generation)
                        extents.append((SlotExtent(r.segment_id, r.next_slot, take), r))
                    self._write_meta(conn, m)
                    conn.execute(
                        "INSERT INTO reservations(id, session_id, fingerprint, ring, slots, generation, created_at,"
                        " pid, host) VALUES(?,?,?,?,?,?,?,?,?)",
                        (rid, session_id[:200], fingerprint, rname, slots, m.generation, time.time(), self._pid, _HOST),
                    )
                    conn.executemany(
                        "INSERT INTO reservation_extents(reservation_id, segment_id, start, count) VALUES(?,?,?,?)",
                        [(rid, e.segment_id, e.start, e.count) for e, _ in extents],
                    )
                    conn.execute(
                        "INSERT INTO counters(fingerprint, ring, consumed) VALUES(?,?,?) ON CONFLICT(fingerprint, ring)"
                        " DO UPDATE SET consumed = consumed + excluded.consumed",
                        (fingerprint, rname, slots),
                    )
                    # lease the segments in this process before anyone can see the commit
                    self._open_reservations[rid] = None
                    _live_add(rid)
            except MaskExhaustedError as exc:
                exhausted = exc
            except BaseException:
                # Commit failed or never happened: nothing was consumed, drop the in-process lease.
                self._open_reservations.pop(rid, None)
                _live_discard(rid)
                raise
            else:
                try:
                    self._fault("after_commit")
                    self._advance_witnesses(m)
                    self._fault("after_witness")
                except BaseException:
                    # The slots are durably consumed; they are simply lost (never reused). Witnesses
                    # behind the DB are resynchronised by the next reserve/verify.
                    self._open_reservations.pop(rid, None)
                    _live_discard(rid)
                    self._mark_closed(rid, slots)
                    raise
        if exhausted is not None:
            self._after_inventory_change(fingerprint, rname)
            raise exhausted
        # Durable: these slots are consumed. Open the segment files outside the lock.
        opened: list[tuple[SlotExtent, SegmentFile]] = []
        try:
            for ext, r in extents:
                seg = SegmentFile(self.root / r.relpath)
                opened.append((ext, seg))
                if seg.header.segment_id != r.segment_id:
                    raise MaskStoreError(f"segment file for {r.segment_id} has a different header")
                seg.set_dek(self._dek_for(seg.header))
            res = MaskReservation(self, rid, session_id, fingerprint, rp, opened)
        except BaseException as exc:
            for _, seg in opened:
                seg.close()
            with self._lock.hold():
                self._open_reservations.pop(rid, None)
            _live_discard(rid)
            self._mark_closed(rid, slots)
            if isinstance(exc, PGWError):
                exc.details.setdefault("reservation_lost_slots", slots)
            raise
        with self._lock.hold():
            self._open_reservations[rid] = res
        log.debug(
            "reserved %d slots (%d extents) for %s %s at generation %d",
            slots, len(extents), short_fingerprint(fingerprint), rname, m.generation,
        )
        self._after_inventory_change(fingerprint, rname)
        return res

    def _fault(self, point: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(point)

    def _dek_for(self, header: SegmentHeader) -> bytes:
        with self._lock.hold():
            hit = self._dek_cache.get(header.segment_id)
            if hit is not None:
                self._dek_cache.move_to_end(header.segment_id)
                return hit
        dek = self.key_provider.unwrap(WrappedKey(header.kek_id, header.wrapped_dek_bytes))
        if self.settings.dek_cache_size > 0:
            with self._lock.hold():
                self._dek_cache[header.segment_id] = dek
                while len(self._dek_cache) > self.settings.dek_cache_size:
                    self._dek_cache.popitem(last=False)
        return dek

    def _mark_closed(self, rid: str, burned: int) -> None:
        try:
            with self._lock.hold(), self._txn() as conn:
                row = conn.execute("SELECT fingerprint, ring FROM reservations WHERE id=?", (rid,)).fetchone()
                conn.execute("UPDATE reservations SET closed_at=?, burned=? WHERE id=?", (time.time(), burned, rid))
                if row is not None and burned > 0:
                    conn.execute(
                        "UPDATE counters SET burned = burned + ? WHERE fingerprint=? AND ring=?", (burned, *row)
                    )
        except PGWError as exc:
            log.warning("could not record reservation close: %s", exc.message)

    def _on_reservation_closed(self, res: MaskReservation, burned: int) -> None:
        if self._conn is None:
            return
        with self._lock.hold():
            self._open_reservations.pop(res.id, None)
        self._mark_closed(res.id, burned)
        if burned:
            log.debug("reservation %s closed; burned %d unused slots", res.id[:8], burned)
        if self.settings.shred_consumed:
            try:
                self._gc_segments({e.segment_id for e in res.extents})
            except PGWError as exc:
                log.warning("shredding after reservation close failed: %s", exc.message)

    # ---- inventory / metrics -------------------------------------------------------------------
    def inventory(self, fingerprint: str | None = None, ring: RingParams | str | None = None) -> list[Inventory]:
        """Availability per ``(fingerprint, ring)``."""
        q = (
            "SELECT fingerprint, ring, COUNT(*),"
            " SUM(CASE WHEN status='active' THEN slots ELSE 0 END),"
            " SUM(CASE WHEN status='active' THEN slots - next_slot ELSE 0 END),"
            " SUM(bytes) FROM segments WHERE status IN (?, ?)"
        )
        args: list[Any] = list(ON_DISK)
        if fingerprint is not None:
            q += " AND fingerprint=?"
            args.append(fingerprint)
        if ring is not None:
            q += " AND ring=?"
            args.append(_ring_name(ring))
        q += " GROUP BY fingerprint, ring ORDER BY fingerprint, ring"
        with self._lock.hold():
            rows = self.conn.execute(q, args).fetchall()
        out = [Inventory(fp, rg, int(n), int(tot or 0), int(av or 0), int(b or 0)) for fp, rg, n, tot, av, b in rows]
        if not out and fingerprint is not None and ring is not None:
            out = [Inventory(fingerprint, _ring_name(ring), 0, 0, 0, 0)]
        return out

    def available_slots(self, fingerprint: str, ring: RingParams | str) -> int:
        """Unconsumed slots for ``(fingerprint, ring)``."""
        return sum(i.slots_available for i in self.inventory(fingerprint, ring))

    def add_low_water_listener(self, fn: LowWaterListener) -> None:
        """Register a callback fired (at most once per crossing) when inventory drops below low water."""
        with self._lock.hold():
            self._listeners.append(fn)

    def _after_inventory_change(self, fingerprint: str, ring: str) -> None:
        if self._conn is None:
            return
        try:
            inv = self.inventory(fingerprint, ring)[0]
        except (PGWError, IndexError):  # pragma: no cover - defensive
            return
        key = (fingerprint, ring)
        below = inv.slots_available < self.settings.low_water_slots
        with self._lock.hold():
            was = self._below.get(key, False)
            self._below[key] = below
            listeners = list(self._listeners) if below and not was else []
        if below and not was:
            log.warning(
                "mask inventory low: %s %s has %d slots (< %d)",
                short_fingerprint(fingerprint), ring, inv.slots_available, self.settings.low_water_slots,
            )
        for fn in listeners:
            try:
                fn(inv)
            except Exception:
                log.exception("low-water listener failed")

    def metrics(self) -> dict[str, dict[tuple[str, str], float]]:
        """Values for the ``pgw_masks_*`` metrics (SPEC 11.1), keyed by ``(fingerprint, ring)``.

        The gateway exports them; this package does not depend on ``pgw.observability``.
        """
        out: dict[str, dict[tuple[str, str], float]] = {
            "pgw_masks_available_slots": {}, "pgw_masks_consumed_slots_total": {},
            "pgw_masks_burned_slots_total": {}, "pgw_masks_low_water": {},
        }
        with self._lock.hold():
            counters = self.conn.execute("SELECT fingerprint, ring, consumed, burned FROM counters").fetchall()
        for fp, rg, consumed, burned in counters:
            out["pgw_masks_consumed_slots_total"][(fp, rg)] = float(consumed)
            out["pgw_masks_burned_slots_total"][(fp, rg)] = float(burned)
            out["pgw_masks_available_slots"][(fp, rg)] = 0.0
        for inv in self.inventory():
            out["pgw_masks_available_slots"][(inv.fingerprint, inv.ring)] = float(inv.slots_available)
        for key, avail in out["pgw_masks_available_slots"].items():
            out["pgw_masks_low_water"][key] = 1.0 if avail < self.settings.low_water_slots else 0.0
        return out

    def status(self) -> dict[str, Any]:
        """Non-secret status summary (CLI ``pgw maskstore status``, gateway ``/admin/masks``)."""
        with self._lock.hold():
            m = self._read_meta(self.conn)
            open_res = self.conn.execute("SELECT COUNT(*) FROM reservations WHERE closed_at IS NULL").fetchone()[0]
            by_status = dict(self.conn.execute("SELECT status, COUNT(*) FROM segments GROUP BY status").fetchall())
        return {
            "root": str(self.root),
            "store_id": m.store_id,
            "generation": m.generation,
            "consumed_total": m.consumed,
            "open_reservations": int(open_res),
            "segments_by_status": by_status,
            "key_provider": self.key_provider.describe() if hasattr(self.key_provider, "describe") else {},
            "rollback_protection": {
                "journal": self._journal is not None,
                "counter": self.counter.kind if self.counter is not None else "none",
                "hardware_counter": bool(self.counter is not None and self.counter.hardware),
                "row_macs": self._mac_ok,
            },
            "low_water_slots": self.settings.low_water_slots,
            "inventory": [i.__dict__ for i in self.inventory()],
            "tainted": self._tainted,
        }

    # ---- housekeeping ------------------------------------------------------------------------
    def _leased_segments(self, conn: sqlite3.Connection, segment_ids: set[str] | None) -> set[str]:
        now = time.time()
        leased: set[str] = set()
        q = (
            "SELECT e.segment_id, r.id, r.pid, r.host, r.created_at FROM reservation_extents e"
            " JOIN reservations r ON r.id = e.reservation_id WHERE r.closed_at IS NULL"
        )
        for seg_id, rid, pid, host, created in conn.execute(q).fetchall():
            if segment_ids is not None and seg_id not in segment_ids:
                continue
            if now - float(created) > self.settings.lease_timeout_s:
                continue
            if host == _HOST and rid not in self._open_reservations and not _is_live(rid) and not (
                int(pid) != self._pid and _pid_alive(int(pid))
            ):
                continue  # owner is gone (crashed) or it is ours and already closed
            leased.add(seg_id)
        return leased

    def _gc_segments(self, segment_ids: set[str] | None) -> int:
        freed = 0
        with self._lock.hold():
            with self._txn() as conn:
                m = self._load_meta(conn)
                q = SELECT_SEGMENTS + " WHERE status=?"
                rows = [SegmentRow.from_row(t) for t in conn.execute(q, (SegmentStatus.CONSUMED.value,))]
                if segment_ids is not None:
                    rows = [r for r in rows if r.segment_id in segment_ids]
                if not rows:
                    return 0
                leased = self._leased_segments(conn, {r.segment_id for r in rows})
                done: list[SegmentRow] = []
                for r in rows:
                    if r.segment_id in leased:
                        continue
                    self._check_row(m.store_id, r)
                    freed += shred_file(self.root / r.relpath, fsync=self.settings.fsync)
                    self._update_row(
                        conn, m.store_id, r, next_slot=r.slots, status=SegmentStatus.SHREDDED, gen=r.updated_gen
                    )
                    self._dek_cache.pop(r.segment_id, None)
                    done.append(r)
            for r in done:
                log.info("shredded consumed segment %s (%.1f MB)", r.segment_id, r.bytes / 1e6)
                with contextlib.suppress(OSError):
                    d = (self.root / r.relpath).parent
                    if not any(d.iterdir()):
                        d.rmdir()
        return freed

    def gc(self) -> int:
        """Shred fully consumed segments; return bytes freed.

        Also closes reservations whose owning process died and prunes old audit records.
        """
        with self._lock.hold():
            self._ready()
            with self._txn() as conn:
                self._close_stale_reservations(conn)
                cutoff = time.time() - self.settings.audit_retention_days * 86400
                old = [r for (r,) in conn.execute("SELECT id FROM reservations WHERE closed_at < ?", (cutoff,))]
                conn.executemany("DELETE FROM reservation_extents WHERE reservation_id=?", [(r,) for r in old])
                conn.executemany("DELETE FROM reservations WHERE id=?", [(r,) for r in old])
            freed = self._gc_segments(None)
            self._remove_partials(_PART_MAX_AGE_S)
        return freed

    def _remove_partials(self, max_age_s: float) -> list[str]:
        removed: list[str] = []
        now = time.time()
        for p in self.root.glob(f"*/*/segments/.*{SEGMENT_SUFFIX}.part"):
            with contextlib.suppress(OSError):
                if now - p.stat().st_mtime >= max_age_s:
                    p.unlink()
                    removed.append(p.name)
                    log.info("removed partial segment file %s", p.name)
        return removed

    def cleanup_partials(self, max_age_s: float = 0.0) -> list[str]:
        """Delete temp files of segment writes interrupted by a crash; return their names.

        Called by the factory scheduler at start-up. Partial files hold encrypted, never-registered
        masks, so deleting them cannot cause reuse. With the default ``max_age_s=0`` every partial
        file is removed: only call it when no writer of this store is running (factory start-up).
        """
        with self._lock.hold():
            return self._remove_partials(max_age_s)

    # ---- key rotation ------------------------------------------------------------------------
    def rotate_kek(self, new_provider: KeyProvider, *, reencrypt: bool = True) -> dict[str, Any]:
        """Switch the store to a new key-encryption key (offline operation).

        Stop all gateway workers and factories of this store first: other processes still hold
        the old key and would refuse the re-authenticated state afterwards (they must be restarted
        with the new provider). The call fails if any reservation is open.

        Steps, all under the store lock:

        1. :meth:`verify_integrity` under the current provider.
        2. With ``reencrypt`` (default), every segment still on disk that is wrapped under another
           KEK is re-encrypted under a fresh DEK wrapped by ``new_provider`` (file replaced
           atomically; the same slots stay consumed). Without it, ``new_provider`` must still be
           able to unwrap the old DEKs (e.g. ``FileKeyProvider(new, previous=[old])``).
        3. State rows and the meta record are re-MACed under ``new_provider`` and the generation
           journal is replaced by a checkpoint under the new key. The journal is staged before the
           DB commit and promoted after it; an interruption is resolved on the next open
           (:meth:`GenerationJournal.recover_pending`).

        Returns:
            Summary ``{"old_kek_id", "new_kek_id", "segments_reencrypted", "bytes_reencrypted"}``.
        """
        if self._mac_ok and new_provider.mac(b"pgw/v1/probe") is None:
            raise ConfigError("the new key provider cannot MAC; rotating would disable rollback detection")
        if not new_provider.can_unwrap:
            raise ConfigError("the new key provider is wrap-only")
        old_kek = self.key_provider.kek_id
        summary: dict[str, Any] = {
            "old_kek_id": old_kek, "new_kek_id": new_provider.kek_id, "segments_reencrypted": 0,
            "bytes_reencrypted": 0,
        }
        with self._lock.hold():
            self.verify_integrity()
            self._ready()
            with self._txn() as conn:
                self._close_stale_reservations(conn)
                (open_res,) = conn.execute("SELECT COUNT(*) FROM reservations WHERE closed_at IS NULL").fetchone()
            if open_res or self._open_reservations:
                raise MaskStoreError(
                    "cannot rotate the KEK while reservations are open; stop the gateways first",
                    details={"open_reservations": int(open_res)},
                )
            rows = [SegmentRow.from_row(t) for t in self.conn.execute(SELECT_SEGMENTS)]
            jhead = self._journal.verify() if self._journal is not None else None
            # 1) re-encrypt segment files (no state change: same segment id, same slots). Resumable:
            #    files already under the new KEK (an earlier, interrupted rotation) are skipped.
            sizes: dict[str, int] = {}
            for r in rows:
                if not reencrypt or r.status not in ON_DISK or r.kek_id == new_provider.kek_id:
                    continue
                path = self.root / r.relpath
                with SegmentFile(path) as src:
                    if src.header.kek_id == new_provider.kek_id:
                        sizes[r.segment_id] = src.layout.total_size
                        continue
                    src.set_dek(self.key_provider.unwrap(WrappedKey(src.header.kek_id, src.header.wrapped_dek_bytes)))
                    secrets = bytearray(src.decrypt(src.layout.secrets_chunk()))
                    dek = os.urandom(DEK_BYTES)
                    wrapped = new_provider.wrap(dek)
                    header = src.header.model_copy(
                        update={
                            "kek_id": wrapped.kek_id,
                            "wrapped_dek": base64.b64encode(wrapped.wrapped).decode(),
                            "nonce_prefix": os.urandom(NONCE_PREFIX_BYTES).hex(),
                        }
                    )
                    sizes[r.segment_id] = self._reencrypt(src, path, header, dek, secrets, replace=True)
                    del dek
                summary["segments_reencrypted"] += 1
                summary["bytes_reencrypted"] += sizes[r.segment_id]
            # 2) switch keys and re-authenticate the state
            old_provider, old_mac_ok = self.key_provider, self._mac_ok
            self.key_provider = new_provider
            self._mac_ok = self._probe_mac()
            self._dek_cache.clear()
            try:
                with self._txn() as conn:
                    m = self._read_meta(conn)
                    for t in conn.execute(SELECT_SEGMENTS).fetchall():
                        r = SegmentRow.from_row(t)
                        kek = new_provider.kek_id if r.segment_id in sizes else r.kek_id
                        size = sizes.get(r.segment_id, r.bytes)
                        conn.execute(
                            "UPDATE segments SET mac=?, kek_id=?, bytes=? WHERE segment_id=?",
                            (self._mac_hex(self._row_mac_msg(m.store_id, r)), kek, size, r.segment_id),
                        )
                    self._write_meta(conn, m)
                    if self._journal is not None:
                        prev = jhead.mac if jhead is not None else bytes(32)
                        self._journal.stage_checkpoint(m.generation, m.consumed, prev)
            except BaseException:
                self.key_provider, self._mac_ok = old_provider, old_mac_ok
                if self._journal is not None:
                    self._journal.pending_path.unlink(missing_ok=True)
                raise
            if self._journal is not None:
                self._journal.promote_pending()
        log.warning(
            "rotated mask store KEK %s -> %s (%d segments re-encrypted)",
            old_kek, new_provider.kek_id, summary["segments_reencrypted"],
        )
        return summary

    def close(self) -> None:
        """Close the state database; open reservations of this process are closed (burned) first."""
        for res in list(self._open_reservations.values()):
            if res is not None:
                res.close()
        with self._lock.hold():
            for k in list(self._dek_cache):
                self._dek_cache.pop(k)
            if self._conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._conn.close()
                self._conn = None

    def __enter__(self) -> MaskStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"MaskStore(root={str(self.root)!r})"
