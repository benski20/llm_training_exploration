"""Durable consumption state of a mask store (``<root>/state.sqlite``, WAL mode). SPEC.md 7.5.

Tables::

    meta(key, value)                 store_id, schema, generation, consumed_total, witness bindings, meta_mac
    segments(seq, segment_id, seed_hash, fingerprint, ring, slots, next_slot, status, relpath,
             bytes, kek_id, factory_id, created_at, added_at, updated_gen, mac)
    reservations(id, session_id, fingerprint, ring, slots, generation, created_at, pid, host,
                 closed_at, burned)
    reservation_extents(reservation_id, segment_id, start, count)
    counters(fingerprint, ring, consumed, burned)

Invariants (checked by :meth:`pgw.maskstore.MaskStore.verify_integrity`):

* ``next_slot`` of a segment only grows; slots ``[0, next_slot)`` are consumed forever.
* every unavailable status (``consumed``, ``shredded``, ``exported``, ``missing``, ``quarantined``)
  has ``next_slot == slots``.
* ``meta.consumed_total == SUM(segments.next_slot)`` over *all* rows; rows are never deleted.
* ``segment_id`` and ``seed_hash`` are UNIQUE forever, so the same seed can never be imported twice.
* ``meta.generation`` grows by one per consuming transaction and is mirrored to the witnesses.
* each segment row and the meta record carry an HMAC (when the key provider supports MACs).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pgw.core.errors import MaskStoreError

__all__ = [
    "SCHEMA_VERSION", "SELECT_SEGMENTS", "SegmentRow", "SegmentStatus", "connect", "meta_message", "segment_message",
]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id  TEXT NOT NULL UNIQUE,
    seed_hash   TEXT NOT NULL UNIQUE,
    fingerprint TEXT NOT NULL,
    ring        TEXT NOT NULL,
    slots       INTEGER NOT NULL CHECK (slots > 0),
    next_slot   INTEGER NOT NULL CHECK (next_slot >= 0 AND next_slot <= slots),
    status      TEXT NOT NULL,
    relpath     TEXT NOT NULL,
    bytes       INTEGER NOT NULL,
    kek_id      TEXT NOT NULL,
    factory_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    added_at    REAL NOT NULL,
    updated_gen INTEGER NOT NULL,
    mac         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS segments_avail ON segments (fingerprint, ring, status, seq);
CREATE TABLE IF NOT EXISTS reservations (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    ring        TEXT NOT NULL,
    slots       INTEGER NOT NULL,
    generation  INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    pid         INTEGER NOT NULL,
    host        TEXT NOT NULL,
    closed_at   REAL,
    burned      INTEGER
);
CREATE INDEX IF NOT EXISTS reservations_open ON reservations (closed_at);
CREATE TABLE IF NOT EXISTS reservation_extents (
    reservation_id TEXT NOT NULL,
    segment_id     TEXT NOT NULL,
    start          INTEGER NOT NULL,
    count          INTEGER NOT NULL,
    PRIMARY KEY (reservation_id, segment_id)
);
CREATE INDEX IF NOT EXISTS extents_segment ON reservation_extents (segment_id);
CREATE TABLE IF NOT EXISTS counters (
    fingerprint TEXT NOT NULL,
    ring        TEXT NOT NULL,
    consumed    INTEGER NOT NULL DEFAULT 0,
    burned      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (fingerprint, ring)
);
"""


class SegmentStatus(StrEnum):
    """Lifecycle of a segment row."""

    ACTIVE = "active"  #: has unconsumed slots (or none yet consumed)
    CONSUMED = "consumed"  #: every slot handed out; file awaits shredding
    SHREDDED = "shredded"  #: file securely deleted
    EXPORTED = "exported"  #: moved out of this store (partner staging store)
    MISSING = "missing"  #: file vanished; remaining slots written off
    QUARANTINED = "quarantined"  #: file failed validation; remaining slots written off


#: Statuses whose file is still expected on disk.
ON_DISK = (SegmentStatus.ACTIVE.value, SegmentStatus.CONSUMED.value)


@dataclass(frozen=True, slots=True)
class SegmentRow:
    """One ``segments`` row."""

    seq: int
    segment_id: str
    seed_hash: str
    fingerprint: str
    ring: str
    slots: int
    next_slot: int
    status: str
    relpath: str
    bytes: int
    kek_id: str
    factory_id: str
    created_at: str
    added_at: float
    updated_gen: int
    mac: str

    COLUMNS = (
        "seq, segment_id, seed_hash, fingerprint, ring, slots, next_slot, status, relpath, bytes, kek_id, "
        "factory_id, created_at, added_at, updated_gen, mac"
    )

    @classmethod
    def from_row(cls, row: tuple[object, ...]) -> SegmentRow:
        """Build from a ``SELECT COLUMNS`` tuple."""
        return cls(*row)  # type: ignore[arg-type]

    @property
    def available(self) -> int:
        """Unconsumed slots."""
        return self.slots - self.next_slot if self.status == SegmentStatus.ACTIVE else 0


#: ``SELECT`` of all :class:`SegmentRow` columns (append ``WHERE ...`` with bound parameters only).
SELECT_SEGMENTS = (
    "SELECT seq, segment_id, seed_hash, fingerprint, ring, slots, next_slot, status, relpath, bytes, kek_id, "
    "factory_id, created_at, added_at, updated_gen, mac FROM segments"
)


def segment_message(
    store_id: str,
    segment_id: str,
    seed_hash: str,
    fingerprint: str,
    ring: str,
    slots: int,
    next_slot: int,
    status: str,
    updated_gen: int,
) -> bytes:
    """Bytes authenticated by a segment row MAC."""
    return (
        f"pgw/v1/segment-row|{store_id}|{segment_id}|{seed_hash}|{fingerprint}|{ring}|{slots}|{next_slot}|"
        f"{status}|{updated_gen}"
    ).encode()


def meta_message(
    store_id: str, generation: int, consumed: int, journal_bound: int, counter_kind: str, counter_offset: int
) -> bytes:
    """Bytes authenticated by the meta MAC."""
    return (
        f"pgw/v1/meta|{store_id}|{generation}|{consumed}|{journal_bound}|{counter_kind}|{counter_offset}"
    ).encode()


def connect(path: Path, *, timeout: float, fsync: bool) -> sqlite3.Connection:
    """Open the state DB (autocommit mode; callers use explicit ``BEGIN IMMEDIATE``)."""
    try:
        conn = sqlite3.connect(str(path), timeout=timeout, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        # FULL: the WAL is fsynced on every commit, so a committed consumption survives power loss.
        conn.execute(f"PRAGMA synchronous={'FULL' if fsync else 'NORMAL'}")
        conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
    except sqlite3.DatabaseError as exc:
        raise MaskStoreError(f"cannot open mask store state database: {exc}") from exc
    return conn
