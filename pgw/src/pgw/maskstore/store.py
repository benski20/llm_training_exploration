"""Mask store API: segment writing (factory side), reservation and consumption (gateway side)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import torch

from pgw.core.ring import RingParams
from pgw.core.types import LinearId, LinearSpec
from pgw.maskstore.config import MaskStoreSettings
from pgw.maskstore.keys import KeyProvider

__all__ = ["Inventory", "LowWaterListener", "MaskReservation", "MaskStore", "SegmentWriter", "SlotExtent"]


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


#: Called with the inventory when available slots drop below ``low_water_slots``.
LowWaterListener = Callable[[Inventory], None]


class MaskReservation:
    """Slots exclusively owned by one generation session.

    Slots are already marked consumed in durable state when the reservation is created. Rows are
    handed out strictly in order: ``take(lid, T)`` returns the next ``T`` rows for linear ``lid``;
    each linear has its own cursor (every token position consumes one row of every linear).
    Unused rows are destroyed on :meth:`close` (never returned to the pool).
    """

    @property
    def slots(self) -> int:
        """Total slots reserved."""
        raise NotImplementedError

    @property
    def extents(self) -> Sequence[SlotExtent]:
        """Where the slots live."""
        raise NotImplementedError

    def remaining(self, lid: LinearId) -> int:
        """Rows not yet taken for ``lid``."""
        raise NotImplementedError

    def take(self, lid: LinearId, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Next ``rows`` masks for ``lid``: ``(r (rows, n), W r (rows, m))`` int64 ring tensors.

        Raises:
            MaskExhaustedError: fewer than ``rows`` remain.
        """
        raise NotImplementedError

    def checks(self, lid: LinearId) -> tuple[torch.Tensor, torch.Tensor]:
        """Freivalds vectors ``(u (c, m), uW (c, n))`` valid for the rows of the current extent.

        When a reservation spans segments, the engine must split calls at extent boundaries
        (``rows_until_boundary``) because ``u`` differs per segment.
        """
        raise NotImplementedError

    def rows_until_boundary(self, lid: LinearId) -> int:
        """Rows of ``lid`` left in the current segment extent."""
        raise NotImplementedError

    def close(self) -> None:
        """Zeroize in-memory secrets and burn unused rows."""
        raise NotImplementedError

    def __enter__(self) -> MaskReservation:
        return self

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None) -> None:
        self.close()


class SegmentWriter:
    """Writes one segment (factory side). Obtain via :meth:`MaskStore.writer`.

    Usage::

        with store.writer(fp, ring, specs, slots=4096, seed=seed, checks=2) as w:
            for lid in ids:
                w.write_linear(lid, wr_rows, u_w)   # wr (slots, m), uW (checks, n)
        # commit on successful exit: fsync + atomic rename; abort deletes the temp file
    """

    def write_linear(self, lid: LinearId, wr: torch.Tensor, uw: torch.Tensor) -> None:
        """Encrypt and append ``W r`` rows (may be called in slot chunks, in order) and ``uW``."""
        raise NotImplementedError

    def commit(self) -> str:
        """Finalize; returns the segment id."""
        raise NotImplementedError

    def abort(self) -> None:
        """Discard the partial segment."""
        raise NotImplementedError

    def __enter__(self) -> SegmentWriter:
        return self

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None) -> None:
        if et is None:
            self.commit()
        else:
            self.abort()


class MaskStore:
    """Directory-backed mask store.

    Layout: ``<root>/<fingerprint>/<ring>/segments/<segment_id>.pgwm`` plus ``<root>/state.sqlite``
    (WAL mode) with per-segment consumption cursors and a global monotonic generation counter.
    Thread- and process-safe (SQLite ``BEGIN IMMEDIATE`` around cursor moves).
    """

    def __init__(self, settings: MaskStoreSettings, key_provider: KeyProvider):
        raise NotImplementedError("maskstore team: MaskStore")

    @classmethod
    def open(cls, root: str | Path, key_provider: KeyProvider, **overrides: object) -> MaskStore:
        """Open (creating if needed) a store at ``root``."""
        raise NotImplementedError

    def writer(
        self,
        fingerprint: str,
        ring: RingParams,
        specs: Sequence[LinearSpec],
        *,
        slots: int,
        seed: bytes,
        checks: int = 2,
        chunk_slots: int = 256,
        factory_id: str = "local",
    ) -> SegmentWriter:
        """Start writing a new segment."""
        raise NotImplementedError

    def import_segment(self, path: str | Path) -> str:
        """Import a segment file produced elsewhere (partner factory); validates header and AEAD."""
        raise NotImplementedError

    def reserve(self, fingerprint: str, ring: RingParams, slots: int, *, session_id: str = "") -> MaskReservation:
        """Atomically consume ``slots`` slots (oldest segments first) and return them.

        Raises:
            MaskExhaustedError: not enough slots available.
        """
        raise NotImplementedError

    def inventory(self, fingerprint: str | None = None, ring: RingParams | None = None) -> list[Inventory]:
        """Availability per ``(fingerprint, ring)``."""
        raise NotImplementedError

    def add_low_water_listener(self, fn: LowWaterListener) -> None:
        """Register a callback fired (at most once per crossing) when inventory drops below low water."""
        raise NotImplementedError

    def gc(self) -> int:
        """Shred fully consumed segments; return bytes freed."""
        raise NotImplementedError

    def verify_integrity(self) -> None:
        """Check state DB vs. segment files and the provider's monotonic counter (rollback detection)."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the state database."""
        raise NotImplementedError
