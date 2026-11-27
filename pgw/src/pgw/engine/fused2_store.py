"""fused2 masks from the encrypted mask store (opt-in, default off): the gateway's pair rows come from store
reservations, exactly as today's masks do.

A fused2 session holds a :class:`pgw.maskstore.MaskReservation` (or a growing
:class:`pgw.gateway.masks.WindowedReservation`, incremental mode included) over **fused2 pair-mask segments**
(:mod:`pgw.maskstore.format`, ``schedule = "fused2"``: one entry per fused2 linear, ``r`` / ``W r`` = ``r_x`` /
``W1 r_x`` for a trip's first linear and ``v`` / ``W2' v`` for a pair's second; p61 check words), written by the
customer-side factory (:mod:`pgw.factory.fused2_gen`). :class:`Fused2StoreMasks` gives
:class:`pgw.engine.fused2.Fused2MaskedBackend` and the native driver (:mod:`pgw.engine.fused2_native`) the
``take(trip)`` interface of the in-process source over it:

* a trip takes the next rows of its first linear and, for a pair, the same number of rows of its second
  (``take_checked``: rows and the check vectors of the extent they came from, atomically); a take never crosses
  a segment extent (the check vectors differ per segment), so it may return fewer rows than the trip: the
  backend then runs the trip in row pieces (rows are independent);
* single use: the reservation's cursors only move forward (durably consumed at reserve time; unused rows
  burned on close); a failure after a take never hands the rows out again (the backend never retries a trip
  with new masks);
* the gateway holds no weights: ``r_R = v - hmul * W1 r_x`` from the stored products and the public ``hmul``;
* secrets: the rows are fresh tensors the backend wipes after the trip; the check vectors are copied once per
  (segment, linear) into arrays this object owns (immune to the store's shared-cache wipe at a concurrent close)
  and zeroed when the next segment's replace them and in :meth:`wipe`.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from pgw.core.errors import ConfigError, IntegrityError, MaskExhaustedError
from pgw.engine.fused2 import Fused2Trip, TripMasks, _Checks

__all__ = ["Fused2StoreMasks", "fused2_reservation_keys", "is_store_session", "open_fused2_session"]

_P61 = (1 << 61) - 1


def is_store_session(session: Any) -> bool:
    """Whether ``session`` is a mask-store reservation a fused2 backend can take pair rows from."""
    return session is not None and hasattr(session, "take_checked") and hasattr(session, "rows_until_boundary")


def fused2_reservation_keys(weights: Any) -> list[str]:
    """The linear keys a fused2 reservation must cover, in pass order (the segment header order)."""
    return [lin.key for lin in weights.all_linears()]


def _schedule_of(session: Any) -> str | None:
    sch = getattr(session, "schedule", None)
    if sch is None:
        parts = getattr(session, "parts", None)
        if parts:
            sch = getattr(parts[0], "schedule", None)
    return sch


class Fused2StoreMasks:
    """The fused2 mask source of one session over a store reservation (see the module docstring).

    Args:
        session: A ``MaskReservation`` / ``WindowedReservation`` of fused2 pair-mask segments.
        weights: The gateway's public fused2 tables (keys and shapes; ``w8`` is never read).
        ring: The fused2 ring.
    """

    #: ``take`` may return fewer rows than the trip asks for (a segment-extent boundary).
    partial = True

    def __init__(self, session: Any, weights: Any, ring: Any):
        if not is_store_session(session):
            raise ConfigError("fused2 store masks need a mask-store reservation")
        sch = _schedule_of(session)
        if sch != "fused2":
            raise ConfigError(f"the reservation's segments are not fused2 pair-mask segments (schedule {sch!r})")
        rname = getattr(getattr(session, "ring", None), "name", getattr(session, "ring", None))
        if rname is not None and rname != ring.name:
            raise ConfigError(f"reservation ring {rname} != fused2 ring {ring.name}")
        keys = set(getattr(session, "linear_keys", None) or [])
        need = fused2_reservation_keys(weights)
        if keys and not set(need) <= keys:
            raise ConfigError("the reservation does not cover every fused2 linear")
        self.session = session
        self.w = weights
        self.ring = ring
        self._ensure = getattr(session, "ensure", None)
        self._lock = threading.Lock()
        #: linear key -> (check token, owned copy of the vectors)
        self._checks: dict[str, tuple[Any, _Checks]] = {}
        self.macs = 0  # the gateway computes no factory products
        self.takes = 0
        self.split_takes = 0

    # ---- the Fused2MaskSource interface -------------------------------------------------------------------
    def check_vectors(self, lin: Any) -> _Checks:
        """Store masks carry their check vectors with every take (``TripMasks.c1`` / ``c2``); a caller that asks
        for static vectors is refused (they differ per segment)."""
        raise ConfigError("fused2 store masks have per-segment check vectors: use TripMasks.c1 / c2")

    def _owned(self, key: str, chk: Any) -> _Checks:
        token, u, uw = chk
        if not (isinstance(token, tuple) and len(token) == 3 and token[2] == "p61"):
            raise IntegrityError(f"fused2 {key}: the segment does not carry p61 check vectors")
        with self._lock:
            hit = self._checks.get(key)
            if hit is not None and hit[0] == token:
                return hit[1]
            uc = np.array(u, dtype=np.uint64).astype(np.int64)
            uwc = np.array(uw, dtype=np.uint64).astype(np.int64)
            if not uc.any() or not uc.size or int(uc.max()) >= _P61 or int(uwc.max(initial=0)) >= _P61:
                uc.fill(0)
                uwc.fill(0)
                # all-zero u accepts anything: the vectors were wiped (a closed reservation) or never valid
                raise IntegrityError(f"fused2 {key}: check vectors unavailable (reservation closed?)")
            ch = _Checks(uc, uwc)
            ch.token = token  # type: ignore[attr-defined]  # (segment id, linear, "p61"): names the vectors
            if hit is not None:
                _wipe_checks(hit[1])
            self._checks[key] = (token, ch)
            return ch

    def _take(self, key: str, rows: int) -> tuple[int, Any, Any, _Checks]:
        if self._ensure is not None:
            self._ensure(key, rows)
        n, r, wr, chk = self.session.take_checked(key, rows, True)
        try:
            if chk is None:
                raise IntegrityError(f"fused2 {key}: the reservation returned no check vectors")
            ch = self._owned(key, chk)
        except BaseException:
            r.zero_()
            wr.zero_()
            raise
        return int(n), r, wr.to(r.dtype) if wr.dtype != r.dtype else wr, ch

    def take(self, trip: Fused2Trip) -> TripMasks:
        """The next rows of the trip's linears: ``min(rows, rows left in the current extent)`` of each (equal)."""
        rows = int(trip.x.shape[0])
        if rows <= 0:
            raise ConfigError("fused2 trip without rows")
        n1, r_x, wr_x, c1 = self._take(trip.l1.key, rows)
        tm = TripMasks(r_x, wr_x)
        tm.c1 = c1
        if trip.l2 is not None:
            try:
                n2, v, wv, c2 = self._take(trip.l2.key, n1)
            except BaseException:
                tm.wipe()
                raise
            tm.v, tm.wv, tm.c2 = v, wv, c2
            if n2 != n1:  # cursors of one pass move in lockstep: never happens with a well-formed reservation
                tm.wipe()
                raise IntegrityError(f"fused2 {trip.l1.key}+{trip.l2.key}: mask rows misaligned ({n1} != {n2})")
        self.takes += 1
        if n1 < rows:
            self.split_takes += 1
        return tm

    def pass_fits(self, keys: list[str], rows: int) -> bool:
        """Whether every key can take ``rows`` rows from one extent (the native driver's whole-pass take);
        extends a growing reservation first (as the takes would)."""
        for key in keys:
            if self._ensure is not None:
                try:
                    self._ensure(key, rows)
                except MaskExhaustedError:
                    return False
            if int(self.session.rows_until_boundary(key)) < rows:
                return False
        return True

    def stocked_rows(self) -> int:
        keys = fused2_reservation_keys(self.w)
        return min(int(self.session.remaining(k)) for k in keys)

    def slots_used(self) -> int:
        keys = fused2_reservation_keys(self.w)
        return max(int(self.session.taken(k)) for k in keys)

    @property
    def closed(self) -> bool:
        return bool(getattr(self.session, "closed", False))

    def wipe(self) -> None:
        """Zero the owned check-vector copies."""
        with self._lock:
            for _, ch in self._checks.values():
                _wipe_checks(ch)
            self._checks.clear()


def _wipe_checks(ch: _Checks) -> None:
    for a in (ch.u, ch.uw):
        if a is not None:
            a.fill(0)
    for limbs in (ch._ul, ch._uwl):
        if limbs is not None:
            for a in limbs:
                a.fill(0)
    ch._ul = ch._uwl = None


def open_fused2_session(store: Any, fingerprint: str, ring: Any, weights: Any, rows0: int, *, cap: int,
                        chunk: int = 16, ahead: bool = True) -> Any:
    """An incremental fused2 session over ``store`` (the gateway's ``mask_reservation = "incremental"``): the
    prompt plus one ``chunk`` now, whole chunks later (:class:`pgw.gateway.masks.WindowedReservation`,
    ``chunked``), every part a ``schedule = "fused2"`` reservation of the fused2 lane ``fingerprint`` (from the
    lane ticket: the gateway's weight-free tables cannot recompute it). Segments lacking a fused2 linear are
    retired (``require_linears``), never handed out. ``cap``: the request's ``len(prompt) + max_tokens``."""
    from pgw.gateway.masks import WindowedReservation, chunk_entitlement

    store.require_linears(fingerprint, ring, fused2_reservation_keys(weights))
    first = min(int(cap), int(rows0) + int(chunk))

    def reserve(n: int) -> Any:
        return store.reserve(fingerprint, ring, int(n), schedule="fused2")

    return WindowedReservation(reserve(first), reserve, cap=chunk_entitlement(first, int(cap), int(chunk)),
                               window=int(chunk), ahead=ahead, chunked=True)
