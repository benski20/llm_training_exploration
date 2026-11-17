"""Incremental mask reservations: reserve a window ahead of generation instead of ``max_tokens``.

A request is entitled to ``len(prompt) + max_tokens`` mask slots, but most requests stop early
(stop sequence, EOS, client disconnect). Reserving everything up front burns every unused slot
when the request ends (slots are never returned: reuse would leak plaintext differences). With
:class:`WindowedReservation` the gateway reserves the prompt plus ``window`` tokens and extends by
further reservations while generating, so an early stop burns at most one window.

The engine's masked backend calls :meth:`WindowedReservation.ensure` before it takes any row of a
step (in its worker thread), so an extension that fails (store exhausted) raises
:class:`pgw.core.MaskExhaustedError` before any mask of the batch is consumed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import torch

from pgw.core.errors import MaskExhaustedError, MaskReuseError
from pgw.core.types import LinearId

__all__ = ["WindowedReservation"]


def _ring_of(part: Any) -> Any:
    from pgw.core.ring import RingParams

    r = part.ring
    return RingParams.from_name(r if isinstance(r, str) else r.name)

log = logging.getLogger(__name__)

#: ``fn(slots) -> reservation`` (a :class:`pgw.maskstore.MaskReservation`-like object).
Reserve = Callable[[int], Any]


class WindowedReservation:
    """A growing chain of mask reservations presented to the engine as one session.

    Rows of every linear are handed out in order across the chained reservations. Each part keeps
    its own extents (and Freivalds vectors); :meth:`rows_until_boundary` never crosses a part, so
    the engine splits calls exactly as it does at segment boundaries.

    Args:
        first: The initial reservation (prompt + first window).
        reserve: Makes a further reservation of ``n`` slots (blocking; runs in a worker thread).
        cap: Total slots the request may use (``len(prompt) + max_tokens``).
        window: Slots added per extension.
    """

    def __init__(self, first: Any, reserve: Reserve, *, cap: int, window: int):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._parts: list[Any] = [first]
        self._reserve = reserve
        self.cap = int(cap)
        self.window = int(window)
        self._lock = threading.Lock()
        self._closed = False
        self.extensions = 0
        self.ring = getattr(first, "ring", None)
        self.fingerprint = getattr(first, "fingerprint", None)

    def __repr__(self) -> str:
        return f"WindowedReservation(parts={len(self._parts)}, slots={self.slots}, cap={self.cap})"

    # ---- introspection ------------------------------------------------------------------------
    @property
    def slots(self) -> int:
        """Slots reserved so far (all parts)."""
        return sum(int(p.slots) for p in self._parts)

    @property
    def closed(self) -> bool:
        """True after :meth:`close`."""
        return self._closed

    @property
    def parts(self) -> tuple[Any, ...]:
        """The chained reservations."""
        return tuple(self._parts)

    def _current(self, lid: LinearId | str) -> Any:
        for p in self._parts:
            if int(p.remaining(lid)) > 0:
                return p
        return self._parts[-1]

    def _reserved_left(self, lid: LinearId | str) -> int:
        return sum(int(p.remaining(lid)) for p in self._parts)

    # ---- engine interface ----------------------------------------------------------------------
    def remaining(self, lid: LinearId | str) -> int:
        """Rows of ``lid`` this session may still use (reserved or reservable up to ``cap``)."""
        with self._lock:
            return self._reserved_left(lid) + max(0, self.cap - self.slots)

    def taken(self, lid: LinearId | str) -> int:
        """Rows of ``lid`` already taken."""
        with self._lock:
            return sum(int(p.taken(lid)) for p in self._parts)

    def ensure(self, lid: LinearId | str, rows: int) -> None:
        """Make sure ``rows`` rows of ``lid`` are reserved, extending by windows if needed.

        Raises:
            MaskExhaustedError: the cap is reached or the store has no slots left (nothing taken).
            MaskReuseError: the reservation is closed.
        """
        with self._lock:
            if self._closed:
                raise MaskReuseError("mask reservation is closed; its unused rows were burned")
            have = self._reserved_left(lid)
            if have >= rows:
                return
            room = self.cap - self.slots
            missing = rows - have
            if missing > room:
                raise MaskExhaustedError(
                    "session mask reservation exhausted",
                    details={"linear": str(lid), "needed": rows, "remaining": have + room},
                )
            n = min(room, max(missing, self.window))
            part = self._reserve(n)
            self._parts.append(part)
            self.extensions += 1
            log.debug("extended mask reservation by %d slots (%d parts)", n, len(self._parts))

    def rows_until_boundary(self, lid: LinearId | str) -> int:
        """Rows of ``lid`` left in the current part's current extent."""
        with self._lock:
            return int(self._current(lid).rows_until_boundary(lid))

    def take(self, lid: LinearId | str, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Next ``rows`` masks of ``lid`` (must not cross a boundary; see :meth:`rows_until_boundary`)."""
        with self._lock:
            if self._closed:
                raise MaskReuseError("mask reservation is closed; its unused rows were burned")
            part = self._current(lid)
        return part.take(lid, rows)  # type: ignore[no-any-return]

    def checks(self, lid: LinearId | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Freivalds vectors of the extent the next :meth:`take` of ``lid`` reads from."""
        with self._lock:
            part = self._current(lid)
        return part.checks(lid)  # type: ignore[no-any-return]

    def checks_u64(self, lid: LinearId | str) -> Any:
        """Read-only uint64 views of the current extent's check vectors (see ``MaskReservation``)."""
        with self._lock:
            part = self._current(lid)
        fn = getattr(part, "checks_u64", None)
        if fn is None:
            from pgw.core.freivalds import checks_to_u64

            u, uw = part.checks(lid)
            return checks_to_u64(u, uw, part.ring if hasattr(part.ring, "mask") else _ring_of(part))
        return fn(lid)

    # ---- lifecycle --------------------------------------------------------------------------------
    def close(self) -> None:
        """Close every part (burning unused rows; idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            parts = list(self._parts)
        errors: list[BaseException] = []
        for p in parts:
            try:
                p.close()
            except Exception as exc:  # close the others regardless
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> WindowedReservation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
