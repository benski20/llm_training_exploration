"""Incremental mask reservations: reserve a window ahead of generation instead of ``max_tokens``.

A request is entitled to ``len(prompt) + max_tokens`` mask slots, but most requests stop early
(stop sequence, EOS, client disconnect). Reserving everything up front burns every unused slot
when the request ends (slots are never returned: reuse would leak plaintext differences). With
:class:`WindowedReservation` the gateway reserves the prompt plus ``window`` tokens and extends by
further reservations while generating, so an early stop burns at most one window.

The engine's masked backend calls :meth:`WindowedReservation.ensure` before it takes any row of a
step (in its worker thread), so an extension that fails (store exhausted) raises
:class:`pgw.core.MaskExhaustedError` before any mask of the batch is consumed.

Parts every linear is done with are closed eagerly, at the start of the next token step (the
:meth:`ensure` call for the model's first linear), never mid-step: closing zeroes a part's check
vectors. A long session therefore holds open only the parts it still reads, so it never pins every
slab of an in-memory mask ring (:mod:`pgw.maskstore.ringbuf`) until it ends.
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
        ahead: Reserve the next window in a background thread once fewer than ``window / 2`` reserved
            rows are left (at the start of a step), so a fail-fast extension (in-memory mask ring)
            rarely has to happen inside a batch step.
    """

    def __init__(self, first: Any, reserve: Reserve, *, cap: int, window: int, ahead: bool = False):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._parts: list[Any] = [first]
        self._reserve = reserve
        self.cap = int(cap)
        self.window = int(window)
        self._lock = threading.Lock()
        self._closed = False
        self.extensions = 0
        #: linear key -> index of the part its rows currently come from (parts are used in order,
        #: so earlier, exhausted parts are never scanned again: per-call cost stays flat).
        self._cur: dict[str, int] = {}
        self.ring = getattr(first, "ring", None)
        self.fingerprint = getattr(first, "fingerprint", None)
        keys = getattr(first, "linear_keys", None)
        #: The first linear of a forward pass: its ensure() marks the start of a token step.
        self._step_key: str | None = keys[0] if keys else None
        #: Parts before this index are closed (retired).
        self._live = 0
        self.retired = 0
        self.ahead = bool(ahead)
        self._pending = 0  #: slots being reserved ahead in the background
        self._ahead_thread: threading.Thread | None = None
        self.ahead_reservations = 0

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

    @staticmethod
    def _k(lid: LinearId | str) -> str:
        return lid if type(lid) is str else lid.key if type(lid) is LinearId else str(lid)

    def _current_index(self, lid: LinearId | str) -> int:
        key = self._k(lid)
        parts = self._parts
        i = self._cur.get(key, 0)
        last = len(parts) - 1
        while i < last and int(parts[i].remaining(lid)) <= 0:
            i += 1
        self._cur[key] = i
        return i

    def _current(self, lid: LinearId | str) -> Any:
        return self._parts[self._current_index(lid)]

    def _reserved_left(self, lid: LinearId | str) -> int:
        i = self._current_index(lid)
        return sum(int(p.remaining(lid)) for p in self._parts[i:])

    # ---- engine interface ----------------------------------------------------------------------
    def remaining(self, lid: LinearId | str) -> int:
        """Rows of ``lid`` this session may still use (reserved or reservable up to ``cap``)."""
        with self._lock:
            return self._reserved_left(lid) + max(0, self.cap - self.slots)

    def reserved_rows(self, lid: LinearId | str) -> int:
        """Rows of ``lid`` already reserved and not yet taken (no store extension needed for them)."""
        with self._lock:
            return self._reserved_left(lid)

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
            step_start = self._step_key is not None and self._k(lid) == self._step_key
            if step_start:
                self._retire_locked()
            have = self._reserved_left(lid)
            if have >= rows:
                if step_start and self.ahead and have - rows < max(1, self.window // 2):
                    self._start_ahead_locked()
                return
            pending = self._ahead_thread
        if pending is not None:  # a background extension is in flight: use it (one bounded reserve call)
            pending.join()
        with self._lock:
            if self._closed:
                raise MaskReuseError("mask reservation is closed; its unused rows were burned")
            have = self._reserved_left(lid)
            if have >= rows:
                return
            # Slots a background extension is still reserving count against the cap too (only one
            # thread calls ensure() per session today, so _pending is 0 here; this keeps it exact).
            room = self.cap - self.slots - self._pending
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

    def _start_ahead_locked(self) -> None:
        room = self.cap - self.slots - self._pending
        if self._ahead_thread is not None or room <= 0:
            return
        n = min(room, self.window)
        self._pending = n

        def run() -> None:
            part = None
            try:
                part = self._reserve(n)
            except Exception as exc:  # the step's own (synchronous) extension reports exhaustion
                log.debug("background mask extension failed: %s", getattr(exc, "message", type(exc).__name__))
            with self._lock:
                self._pending = 0
                self._ahead_thread = None
                if part is not None and not self._closed:
                    self._parts.append(part)
                    self.extensions += 1
                    self.ahead_reservations += 1
                    part = None
            if part is not None:  # the session closed meanwhile: burn it
                part.close()

        t = threading.Thread(target=run, name="pgw-mask-ahead", daemon=True)
        self._ahead_thread = t
        t.start()

    def _retire_locked(self) -> None:
        """Close leading parts (never the last) whose every linear is fully taken; chained parts retire
        their own exhausted pieces. Called only at the start of a token step (see module docstring)."""
        parts = self._parts
        last = len(parts) - 1
        while self._live < last:
            p = parts[self._live]
            keys = getattr(p, "linear_keys", None)
            if not keys or any(int(p.remaining(k)) > 0 for k in keys):
                break
            try:
                p.close()
            except Exception:  # pragma: no cover - close() burns; never fail a step for it
                log.exception("closing an exhausted mask reservation part failed")
            self._live += 1
            self.retired += 1
        p = parts[min(self._live, last)]
        retire = getattr(p, "retire_exhausted", None)
        if retire is not None:
            retire()

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

    def take_checked(self, lid: LinearId | str, max_rows: int, checks: bool) -> Any:
        """``MaskReservation.take_checked`` of the current part (never crosses a part)."""
        with self._lock:
            if self._closed:
                raise MaskReuseError("mask reservation is closed; its unused rows were burned")
            part = self._current(lid)
        fn = getattr(part, "take_checked", None)
        if fn is not None:
            return fn(lid, max_rows, checks)
        rows = min(int(max_rows), int(part.rows_until_boundary(lid)))
        if rows <= 0:
            raise MaskExhaustedError("mask reservation has no rows left", details={"linear": str(lid)})
        chk = (None, *self._part_checks_u64(part, lid)) if checks else None
        r, wr = part.take(lid, rows)
        return rows, r, wr, chk

    def prefetch(self, lid: LinearId | str, rows: int) -> int:
        """``MaskReservation.prefetch`` of the current part (never another part, never an extension)."""
        with self._lock:
            if self._closed:
                return 0
            part = self._current(lid)
        fn = getattr(part, "prefetch", None)
        return 0 if fn is None else int(fn(lid, rows))

    def checks_compact(self, lid: LinearId | str) -> Any:
        """``MaskReservation.checks_compact`` of the current part (token ``None`` if it has none)."""
        with self._lock:
            part = self._current(lid)
        fn = getattr(part, "checks_compact", None)
        if fn is not None:
            return fn(lid)
        return (None, *self._part_checks_u64(part, lid))

    @staticmethod
    def _part_checks_u64(part: Any, lid: LinearId | str) -> Any:
        fn = getattr(part, "checks_u64", None)
        if fn is not None:
            return fn(lid)
        from pgw.core.freivalds import checks_to_u64

        u, uw = part.checks(lid)
        return checks_to_u64(u, uw, part.ring if hasattr(part.ring, "mask") else _ring_of(part))

    def checks(self, lid: LinearId | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Freivalds vectors of the extent the next :meth:`take` of ``lid`` reads from."""
        with self._lock:
            part = self._current(lid)
        return part.checks(lid)  # type: ignore[no-any-return]

    def checks_u64(self, lid: LinearId | str) -> Any:
        """Read-only uint64 views of the current extent's check vectors (see ``MaskReservation``)."""
        with self._lock:
            part = self._current(lid)
        return self._part_checks_u64(part, lid)

    # ---- lifecycle --------------------------------------------------------------------------------
    def close(self) -> None:
        """Close every part (burning unused rows; idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            parts = list(self._parts)
            pending = self._ahead_thread
        if pending is not None:
            pending.join()  # its result is burned by the thread itself (closed)
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
