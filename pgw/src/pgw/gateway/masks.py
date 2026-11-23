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

Incremental mode (``chunked=True``; gateway ``mask_reservation = "incremental"``): the same chain with a small,
fixed, public chunk size. Extensions are whole chunks (never clipped to what the request may still use, so their
sizes and the moments the store is asked depend only on the rows taken, never on ``max_tokens``), the next chunk is
reserved ahead in the background, and retired parts are closed (their close commit) on a background thread
instead of inside the token step. The gateway pairs it with an entitlement ledger (:class:`EntitlementLedger`)
that admits a request only when the store can cover every live session's remaining entitlement, so the
on-demand extensions do not run short in normal operation (and fail closed if they ever do).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import torch

from pgw.core.errors import MaskExhaustedError, MaskReuseError
from pgw.core.types import LinearId


def _decrypt_ahead() -> bool:
    """``PGW_TRIP_DECRYPT_AHEAD`` (read by the mask store at import)."""
    from pgw.maskstore import store

    return bool(store.TRIP_DECRYPT_AHEAD)

__all__ = ["EntitlementLedger", "WindowedReservation", "chunk_entitlement"]


def chunk_entitlement(first: int, cap: int, chunk: int) -> int:
    """Most slots an incremental session can ever reserve: ``first`` plus whole chunks up to ``cap``."""
    first, cap, chunk = int(first), int(cap), int(chunk)
    return first + chunk * max(0, -(-(cap - first) // chunk))


_closer_lock = threading.Lock()
_closer: Any = None


def _close_pool() -> Any:
    """One shared background thread for retired parts' closes (incremental mode)."""
    global _closer
    with _closer_lock:
        if _closer is None:
            from concurrent.futures import ThreadPoolExecutor

            _closer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-mask-retire")
        return _closer


def _ring_of(part: Any) -> Any:
    from pgw.core.ring import RingParams

    r = part.ring
    return RingParams.from_name(r if isinstance(r, str) else r.name)

log = logging.getLogger(__name__)

#: ``fn(slots) -> reservation`` (a :class:`pgw.maskstore.MaskReservation`-like object).
Reserve = Callable[[int], Any]


def _close_quietly(part: Any) -> None:
    try:
        part.close()
    except Exception:  # pragma: no cover - close() burns; never fail a step for it
        log.exception("closing an exhausted mask reservation part failed")


class EntitlementLedger:
    """Admission for incremental reservations (one per model runtime, i.e. per fingerprint and ring).

    An incremental session holds only a chunk or two of masks, but is *entitled* to more (up to its
    chunk-aligned cap). A request is admitted only when the store's available slots cover the remaining
    entitlement of every live session plus the new request's, so a running answer's extensions find slots
    (the guarantee an up-front reservation gave) while nothing beyond what is used is consumed. Only this
    process reserves from the store for this ``(fingerprint, ring)`` (a store is never shared between
    hosts); mask factories only add slots. Stock lost otherwise (retired segments) can still make an
    extension fail: that fails closed (:class:`MaskExhaustedError`, nothing taken; the answer ends with an
    error and its reservation is burned).

    The ledger lives in gateway memory only: a crash loses entitlements, never consumed slots.
    """

    def __init__(self) -> None:
        import weakref

        self._live: weakref.WeakSet[WindowedReservation] = weakref.WeakSet()
        self.refused = 0

    def outstanding(self) -> int:
        """Slots live sessions may still reserve."""
        total = 0
        for w in list(self._live):
            if w.closed:
                self._live.discard(w)
                continue
            total += w.outstanding
        return total

    def admit(self, available: int, entitled: int) -> bool:
        """True when ``available`` slots cover the live sessions' outstanding entitlement plus ``entitled``."""
        ok = int(available) - self.outstanding() >= int(entitled)
        if not ok:
            self.refused += 1
        return ok

    def add(self, w: WindowedReservation) -> None:
        self._live.add(w)

    def __len__(self) -> int:
        return sum(1 for w in list(self._live) if not w.closed)


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
        public: Public schedule (strict speculation with ``spec_mask_rows``, L3): the store is asked for
            slots only by :meth:`charge_to`, whose targets depend on public quantities alone (prompt length,
            pass index), never on how many rows were taken. :meth:`ensure` then never extends (a step that
            would need more than the charged slots fails closed), the ahead reservation is started from
            :meth:`charge_to` (charged minus reserved slots, public), and no part is retired before
            :meth:`close` (when a part runs out depends on the drafts; closing it early frees store or ring
            capacity at a data-dependent time).
        chunked: Incremental mode: ``window`` is a fixed chunk; on-demand extensions are whole chunks
            (``chunk * ceil(missing / chunk)``, at most up to ``cap``, which the gateway sets to the chunk-aligned
            entitlement, :func:`chunk_entitlement`), and retired parts are closed on a background thread
            (:meth:`close` waits for those closes).
    """

    def __init__(self, first: Any, reserve: Reserve, *, cap: int, window: int, ahead: bool = False,
                 public: bool = False, chunked: bool = False):
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
        self.public = bool(public)
        #: Public mode: the highest charge target so far (slots the schedule entitles the session to).
        self.charged = 0
        #: linear key -> the current part's native trip lease (``PGW_TRIP_NATIVE``; see :meth:`trip_lease`).
        self.trip_leases: dict[str, Any] = {}
        self.chunked = bool(chunked)
        #: chunked mode: background closes of retired parts (joined by :meth:`close`)
        self._retiring: list[Any] = []

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
    def outstanding(self) -> int:
        """Slots this session may still ask the store for (``cap`` minus reserved and in-flight; 0 once closed)."""
        if self._closed:
            return 0
        return max(0, self.cap - self.slots - self._pending)

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

    def _reserved_left(self, lid: LinearId | str, limit: int | None = None) -> int:
        """Rows of ``lid`` reserved and not yet taken; with ``limit``, the sum may stop early once it
        reaches ``limit`` (a value ``>= limit`` then only says "at least ``limit``")."""
        i = self._current_index(lid)
        if limit is None:
            return sum(int(p.remaining(lid)) for p in self._parts[i:])
        total = 0
        for p in self._parts[i:]:
            total += int(p.remaining(lid))
            if total >= limit:
                break
        return total

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
            if self.public:  # slots come from charge_to only (public schedule)
                have = self._reserved_left(lid)
                if have < rows:
                    raise MaskExhaustedError(
                        "session mask reservation exhausted (public schedule)",
                        details={"linear": str(lid), "needed": rows, "remaining": have},
                    )
                return
            step_start = self._step_key is not None and self._k(lid) == self._step_key
            if step_start:
                self._retire_locked()
            # the decode hot path: both decisions below only need to know whether ``have`` reaches
            # ``rows + max(1, window // 2)``, so the per-part sum stops there (same decisions)
            have = self._reserved_left(lid, rows + max(1, self.window // 2))
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
            if self.chunked:  # whole chunks: the size depends on the rows taken only
                n = min(room, self.window * -(-missing // self.window))
            else:
                n = min(room, max(missing, self.window))
            part = self._reserve(n)
            self._parts.append(part)
            self.extensions += 1
            log.debug("extended mask reservation by %d slots (%d parts)", n, len(self._parts))

    def charge_to(self, target: int) -> None:
        """Public schedule: make the session hold at least ``min(target, cap)`` slots in total.

        Extends by whole windows (``max(window, missing)``, at most up to ``cap``), where ``missing`` is the
        target minus the slots reserved so far: both are public, so when the store is asked and for how much
        never depends on how many rows were taken. With ``ahead``, the next window is reserved in the
        background once the reserved slots exceed the target by less than half a window (also public).

        Raises:
            MaskExhaustedError: the store has no slots left. MaskReuseError: the reservation is closed.
        """
        target = min(int(target), self.cap)
        with self._lock:
            if self._closed:
                raise MaskReuseError("mask reservation is closed; its unused rows were burned")
            self.charged = max(self.charged, target)
            if self.slots >= target:
                if self.ahead and self.slots - target < max(1, self.window // 2):
                    self._start_ahead_locked()
                return
            pending = self._ahead_thread
        if pending is not None:
            pending.join()
        with self._lock:
            if self._closed:
                raise MaskReuseError("mask reservation is closed; its unused rows were burned")
            missing = target - self.slots
            if missing <= 0:
                return
            room = self.cap - self.slots - self._pending
            if self.chunked:
                n = min(room, self.window * -(-missing // self.window))
            else:
                n = min(room, max(missing, self.window))
            part = self._reserve(n)
            self._parts.append(part)
            self.extensions += 1
            log.debug("charged mask reservation to %d slots (%d parts)", self.slots, len(self._parts))

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
            if self.chunked:
                # every row of the part is taken: its close (zeroing, the durable close record) runs off the
                # token step; close() joins it. The part is never read again (cursors moved past it).
                self._retiring = [f for f in self._retiring if not f.done()]
                self._retiring.append(_close_pool().submit(_close_quietly, p))
            else:
                _close_quietly(p)
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

    take_checked_verify = True

    def take_checked(self, lid: LinearId | str, max_rows: int, checks: bool, verify: Any = None) -> Any:
        """``MaskReservation.take_checked`` of the current part (never crosses a part); ``verify``
        (with ``checks``) sees the part's checks before any row is taken."""
        with self._lock:
            if self._closed:
                raise MaskReuseError("mask reservation is closed; its unused rows were burned")
            part = self._current(lid)
        fn = getattr(part, "take_checked", None)
        if fn is not None:
            if verify is not None:
                from pgw.maskstore.tiered import _take_verified

                return _take_verified(part, lid, max_rows, checks, verify)
            return fn(lid, max_rows, checks)
        rows = min(int(max_rows), int(part.rows_until_boundary(lid)))
        if rows <= 0:
            raise MaskExhaustedError("mask reservation has no rows left", details={"linear": str(lid)})
        chk = (None, *self._part_checks_u64(part, lid)) if checks else None
        if verify is not None and chk is not None:
            verify(chk)
        r, wr = part.take(lid, rows)
        return rows, r, wr, chk

    # ---- native trip leases (PGW_TRIP_NATIVE) -----------------------------------------------------
    @property
    def trip_step_key(self) -> str | None:
        """The linear whose every trip goes through :meth:`trip_lease` (and so :meth:`ensure`): the
        step-start linear, whose ``ensure`` retires parts and starts the ahead reservation. ``None`` in
        public mode (``ensure`` has no side effects there)."""
        return None if self.public else self._step_key

    def trip_lease(self, lid: LinearId | str, rows: int, verify: Any = None, wnorm: float = 0.0, *,
                   prime_ok: bool = False) -> Any:
        """:meth:`ensure` (exactly as before a take), then the current part's native trip lease
        (``MaskReservation.trip_lease``), cached in :attr:`trip_leases`; ``None``: the ordinary path.

        With ``PGW_TRIP_DECRYPT_AHEAD``, a lease that covers the rest of its part starts decrypting the
        next part's first chunk of the linear (rows this session already reserved; nothing handed out)."""
        self.ensure(lid, rows)
        with self._lock:
            if self._closed:
                raise MaskReuseError("mask reservation is closed; its unused rows were burned")
            i = self._current_index(lid)
            part = self._parts[i]
            nxt = self._parts[i + 1] if i + 1 < len(self._parts) else None
        fn = getattr(part, "trip_lease", None)
        lease = None if fn is None else fn(lid, rows, verify, wnorm, strict=False, prime_ok=prime_ok)
        key = self._k(lid)
        if lease is None:
            self.trip_leases.pop(key, None)
        else:
            self.trip_leases[key] = lease
            ahead = getattr(nxt, "decrypt_ahead", None) if nxt is not None and _decrypt_ahead() else None
            if ahead is not None and int(lease.left) >= int(part.remaining(lid)):
                ahead(lid)
        return lease

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
        for f in self._retiring:  # background closes of retired parts (chunked mode; _close_quietly logs)
            f.result()
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
