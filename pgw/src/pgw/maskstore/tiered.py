"""Mask source with two tiers: the in-memory :class:`pgw.maskstore.ringbuf.MaskRing` per model and the
disk :class:`pgw.maskstore.MaskStore` (design ``research/plans/mask-ringbuffer-design.md`` §6).

:class:`TieredMaskSource` has the store's interface (``reserve``, ``inventory``, ``metrics``,
``status``, ...). ``reserve()`` tries the preferred tier, then the other one, and when neither tier
alone holds the request it returns a :class:`ChainedReservation` of a ring part and a store part
(both closed if either fails), so a request that passed the fail-fast check (which counts both
tiers) is never refused because of the split. The store may be absent (``ring_only``).

Rings are attached per model when the model runtime starts (it has the pack's linear specs) and
closed and wiped when it stops or reloads.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, MaskExhaustedError, MaskReuseError, MaskStoreError
from pgw.core.fingerprint import short_fingerprint
from pgw.core.ring import RingParams
from pgw.core.settings import ring_from_setting
from pgw.core.types import LinearId, LinearSpec
from pgw.maskstore.format import auto_chunk_slots, chunk_cache_bytes
from pgw.maskstore.ringbuf import (
    MaskRing,
    RingShort,
    SeedRegistry,
    harden_process,
    init_on_free_status,
    swap_status,
)
from pgw.maskstore.store import Inventory, SlotExtent

__all__ = ["ChainedReservation", "TieredMaskSource", "build_tiered_source", "check_init_on_free"]

log = logging.getLogger(__name__)


def _key(lid: LinearId | str) -> str:
    return lid if type(lid) is str else lid.key if type(lid) is LinearId else str(lid)


class ChainedReservation:
    """A fixed chain of reservations (a ring part, then a store part) with the
    :class:`pgw.maskstore.MaskReservation` interface.

    Rows of every linear come from the parts in order; :meth:`rows_until_boundary` never crosses a
    part (parts have their own Freivalds vectors), so the engine splits calls exactly as it does at
    segment boundaries. :meth:`retire_exhausted` closes parts every linear is done with; the gateway
    calls it only at the start of a step (never mid-step: closing zeroes the check vectors).
    """

    def __init__(self, parts: Sequence[Any]):
        if len(parts) < 2:
            raise MaskStoreError("a chained reservation needs at least two parts")
        self._parts = list(parts)
        p0 = self._parts[0]
        self.id = "+".join(str(p.id)[:16] for p in self._parts)
        self.session_id = getattr(p0, "session_id", "")
        self.fingerprint = p0.fingerprint
        self.ring = p0.ring
        for p in self._parts[1:]:
            same_ring = getattr(p.ring, "name", p.ring) == getattr(self.ring, "name", self.ring)
            if p.fingerprint != self.fingerprint or not same_ring:
                raise MaskStoreError("parts of a chained reservation disagree on model or ring")
        self._lock = threading.Lock()
        self._closed = False
        self._cur: dict[str, int] = {}
        self.staged_burned = 0

    def __repr__(self) -> str:
        return f"ChainedReservation(parts={len(self._parts)}, slots={self.slots}, closed={self._closed})"

    @property
    def parts(self) -> tuple[Any, ...]:
        return tuple(self._parts)

    @property
    def slots(self) -> int:
        return sum(int(p.slots) for p in self._parts)

    @property
    def extents(self) -> Sequence[SlotExtent]:
        return tuple(e for p in self._parts for e in p.extents)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def linear_keys(self) -> tuple[str, ...]:
        return tuple(self._parts[0].linear_keys)

    def _alive(self) -> None:
        if self._closed:
            raise MaskReuseError("mask reservation is closed; its unused rows were burned")

    def _index(self, lid: LinearId | str) -> int:
        key = _key(lid)
        i = self._cur.get(key, 0)
        last = len(self._parts) - 1
        while i < last and int(self._parts[i].remaining(lid)) <= 0:
            i += 1
        self._cur[key] = i
        return i

    def _current(self, lid: LinearId | str) -> Any:
        with self._lock:
            self._alive()
            return self._parts[self._index(lid)]

    def remaining(self, lid: LinearId | str) -> int:
        with self._lock:
            i = self._index(lid)
            return sum(int(p.remaining(lid)) for p in self._parts[i:])

    def taken(self, lid: LinearId | str) -> int:
        return sum(int(p.taken(lid)) for p in self._parts)

    def rows_until_boundary(self, lid: LinearId | str) -> int:
        with self._lock:
            return int(self._parts[self._index(lid)].rows_until_boundary(lid))

    def take(self, lid: LinearId | str, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._current(lid).take(lid, rows)  # type: ignore[no-any-return]

    def take_checked(self, lid: LinearId | str, max_rows: int, checks: bool) -> Any:
        return self._current(lid).take_checked(lid, max_rows, checks)

    def prefetch(self, lid: LinearId | str, rows: int) -> int:
        with self._lock:
            if self._closed:
                return 0
            part = self._parts[self._index(lid)]
        return int(part.prefetch(lid, rows))

    def staged_rows(self, lid: LinearId | str) -> int:
        with self._lock:
            part = self._parts[self._index(lid)]
        return int(part.staged_rows(lid))

    def checks(self, lid: LinearId | str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._current(lid).checks(lid)  # type: ignore[no-any-return]

    def checks_u64(self, lid: LinearId | str) -> tuple[np.ndarray, np.ndarray]:
        return self._current(lid).checks_u64(lid)  # type: ignore[no-any-return]

    def checks_compact(self, lid: LinearId | str) -> Any:
        return self._current(lid).checks_compact(lid)

    def retire_exhausted(self) -> int:
        """Close parts (all but the last) whose every linear is fully taken; returns how many."""
        with self._lock:
            if self._closed:
                return 0
            done = [p for p in self._parts[:-1] if not p.closed and all(int(p.remaining(k)) <= 0
                                                                         for k in p.linear_keys)]
        for p in done:
            p.close()
            self.staged_burned += int(getattr(p, "staged_burned", 0))
        return len(done)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        errors: list[BaseException] = []
        for p in self._parts:
            try:
                p.close()
            except Exception as exc:
                errors.append(exc)
        self.staged_burned = sum(int(getattr(p, "staged_burned", 0)) for p in self._parts)
        if errors:
            raise errors[0]

    def __enter__(self) -> ChainedReservation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TieredMaskSource:
    """Ring tier plus store tier behind the store's interface.

    Args:
        store: The disk store, or ``None`` (``ring_only``).
        ring_settings: :class:`pgw.gateway.config.MaskRingSettings`-like object.
        min_soundness_bits: Floor for ring segments (the store's own setting when a store exists,
            else the gateway's ``min_soundness_bits``).
        masked_models: Number of masked models sharing ``capacity_bytes``.
        seeds: Seed registry shared with the factory's sinks (default: process-wide).
    """

    def __init__(self, store: Any, ring_settings: Any, *, min_soundness_bits: float = 20.0, masked_models: int = 1,
                 seeds: SeedRegistry | None = None, _mlock: Callable[[int, int], int] | None = None):
        self.store = store
        self.cfg = ring_settings
        self.prefer = str(getattr(ring_settings, "prefer", "ring"))
        if self.prefer not in ("ring", "store"):
            raise ConfigError("mask_ring.prefer must be 'ring' or 'store'")
        self.ring_only = store is None
        self.ring_only_wait_s = float(getattr(ring_settings, "ring_only_wait_s", 2.0))
        st = getattr(store, "settings", None)
        self.min_soundness_bits = float(getattr(st, "min_soundness_bits", min_soundness_bits))
        self.masked_models = max(1, int(masked_models))
        self.seeds = seeds
        self._mlock = _mlock
        self._rings: dict[tuple[str, str], MaskRing] = {}
        self._lock = threading.Lock()
        self.fallback_reservations = 0
        #: Per ``(fingerprint, ring)`` count of reservations the other tier served (metric labels).
        self.fallback_by_key: dict[tuple[str, str], int] = {}
        self.spanning_reservations = 0
        self._ring_listeners: list[Callable[[MaskRing], None]] = []

    def __getattr__(self, name: str) -> Any:
        # Everything else (verify_integrity, writer, cleanup_partials, ...) is the store's business.
        if name.startswith("__") or name in ("store", "_rings"):
            raise AttributeError(name)
        store = self.__dict__.get("store")
        if store is None:
            raise AttributeError(name)
        return getattr(store, name)

    def __repr__(self) -> str:
        return f"TieredMaskSource(rings={len(self._rings)}, store={'yes' if self.store is not None else 'no'})"

    # ---- rings -----------------------------------------------------------------------------------
    @property
    def rings(self) -> dict[tuple[str, str], MaskRing]:
        with self._lock:
            return dict(self._rings)

    def ring_for(self, fingerprint: str, ring: RingParams | str) -> MaskRing | None:
        with self._lock:
            mr = self._rings.get((fingerprint, ring_from_setting(ring).name))
        return None if mr is None or mr.closed else mr

    def add_ring_listener(self, fn: Callable[[MaskRing], None]) -> None:
        """``fn(ring)`` whenever a ring is attached (the embedded factory hooks its free listener)."""
        with self._lock:
            self._ring_listeners.append(fn)
            rings = list(self._rings.values())
        for r in rings:
            fn(r)

    def capacity_for(self, name: str) -> int:
        per = dict(getattr(self.cfg, "per_model_bytes", {}) or {})
        if name in per:
            return int(per[name])
        cap = getattr(self.cfg, "capacity_bytes", None)
        if not cap:
            raise ConfigError("mask_ring.capacity_bytes is required when the mask ring is enabled")
        return int(cap) // self.masked_models

    def attach_model(self, name: str, fingerprint: str, ring: RingParams | str, specs: Sequence[LinearSpec], *,
                     max_session_slots: int | None = None) -> MaskRing:
        """Build (or keep) the ring of a masked model; closes a ring of the same model name with another
        fingerprint (reload). In ``ring_only`` mode refuses models whose longest session would not fit
        well below the ring's capacity."""
        rname = ring_from_setting(ring).name
        with self._lock:
            hit = self._rings.get((fingerprint, rname))
            stale = [k for k, r in self._rings.items() if r.name == name and k != (fingerprint, rname)]
        if hit is not None and not hit.closed:
            self._check_fit(hit, max_session_slots)
            return hit
        for k in stale:
            self.detach_model(*k)
        mr = MaskRing(
            fingerprint, rname, specs, capacity_bytes=self.capacity_for(name),
            segment_slots=getattr(self.cfg, "segment_slots", None), min_soundness_bits=self.min_soundness_bits,
            allow_unlocked=bool(getattr(self.cfg, "allow_unlocked", False)), seeds=self.seeds, name=name,
            shared=bool(getattr(self.cfg, "factory_process", False)), _mlock=self._mlock,
        )
        try:
            self._check_fit(mr, max_session_slots)
        except BaseException:
            mr.close()
            raise
        with self._lock:
            self._rings[(fingerprint, rname)] = mr
            listeners = list(self._ring_listeners)
        for fn in listeners:
            fn(mr)
        return mr

    def _check_fit(self, mr: MaskRing, max_session_slots: int | None) -> None:
        if not self.ring_only or max_session_slots is None:
            return
        frac = float(getattr(self.cfg, "max_session_fraction", 0.5))
        limit = int(mr.capacity_slots * frac)
        if int(max_session_slots) > limit:
            raise ConfigError(
                f"ring_only: a session of {max_session_slots} slots (max_context + mask_window_tokens) exceeds "
                f"{frac:.0%} of the ring's {mr.capacity_slots} slots for model {mr.name}; raise capacity_bytes, "
                "lower max_context or add the disk store",
                details={"max_session_slots": int(max_session_slots), "capacity_slots": mr.capacity_slots},
            )

    def detach_model(self, fingerprint: str, ring: RingParams | str) -> None:
        """Close and wipe the ring of ``(fingerprint, ring)`` (masks in it are lost, never reused)."""
        with self._lock:
            mr = self._rings.pop((fingerprint, ring_from_setting(ring).name), None)
        if mr is not None:
            mr.close()

    # ---- reservations ----------------------------------------------------------------------------
    def reserve(self, fingerprint: str, ring: RingParams | str, slots: int, *, session_id: str = "",
                wait: bool = True) -> Any:
        """Reserve ``slots`` slots: preferred tier, other tier, or a chain across both.

        ``wait=False`` (window extensions) never waits for the factory in ``ring_only`` mode.
        """
        if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
            raise ConfigError(f"slots must be a positive int, got {slots!r}")
        mr = self.ring_for(fingerprint, ring)
        store = self.store
        if mr is None:
            if store is None:
                raise MaskExhaustedError("no mask ring for this model", details={
                    "fingerprint": short_fingerprint(fingerprint), "needed": slots, "available": 0})
            return store.reserve(fingerprint, ring, slots, session_id=session_id)
        ring_av = store_av = 0
        if self.prefer == "ring" or store is None:
            try:
                return mr.reserve(slots, session_id=session_id)
            except RingShort as exc:
                ring_av = int(exc.details.get("available", 0))
            if store is None:
                if wait and self.ring_only_wait_s > 0:
                    try:
                        return mr.reserve(slots, session_id=session_id, wait_s=self.ring_only_wait_s)
                    except RingShort as exc:
                        ring_av = int(exc.details.get("available", 0))
                raise MaskExhaustedError("not enough mask slots in the ring", details={
                    "fingerprint": short_fingerprint(fingerprint), "needed": slots, "available": ring_av,
                    "ring_available": ring_av})
            try:
                res = store.reserve(fingerprint, ring, slots, session_id=session_id)
            except MaskExhaustedError as exc:
                store_av = int(exc.details.get("available", 0))
            else:
                self._count_fallback(mr)
                return res
        else:
            try:
                return store.reserve(fingerprint, ring, slots, session_id=session_id)
            except MaskExhaustedError as exc:
                store_av = int(exc.details.get("available", 0))
            try:
                res = mr.reserve(slots, session_id=session_id)
            except RingShort as exc:
                ring_av = int(exc.details.get("available", 0))
            else:
                self._count_fallback(mr)
                return res
        return self._span(mr, fingerprint, ring, slots, session_id, ring_av, store_av)

    def _count_fallback(self, mr: MaskRing) -> None:
        with self._lock:
            self.fallback_reservations += 1
            self.fallback_by_key[mr.key] = self.fallback_by_key.get(mr.key, 0) + 1

    def reserve_extension(self, fingerprint: str, ring: RingParams | str, slots: int, *, session_id: str = "") -> Any:
        """A window extension: like :meth:`reserve` but never waits (fails fast)."""
        return self.reserve(fingerprint, ring, slots, session_id=session_id, wait=False)

    def _span(self, mr: MaskRing, fingerprint: str, ring: RingParams | str, slots: int, session_id: str,
              ring_av: int, store_av: int) -> Any:
        exhausted = MaskExhaustedError("not enough mask slots (ring and store together)", details={
            "fingerprint": short_fingerprint(fingerprint), "ring": ring_from_setting(ring).name, "needed": slots,
            "available": ring_av + store_av, "ring_available": ring_av, "store_available": store_av})
        if ring_av <= 0 or store_av <= 0 or ring_av + store_av < slots:
            raise exhausted
        part_a = None
        for _ in range(2):  # the ring may have shrunk since RingShort: retry once with its new level
            take = min(ring_av, slots - 1)
            if take <= 0:
                raise exhausted
            try:
                part_a = mr.reserve(take, session_id=session_id)
                break
            except RingShort as exc:
                ring_av = int(exc.details.get("available", 0))
        if part_a is None:
            raise exhausted
        try:
            part_b = self.store.reserve(fingerprint, ring, slots - int(part_a.slots), session_id=session_id)
        except BaseException:
            part_a.close()
            raise
        try:
            chain = ChainedReservation([part_a, part_b])
        except BaseException:
            part_a.close()
            part_b.close()
            raise
        with self._lock:
            self.spanning_reservations += 1
        log.debug("reservation of %d slots spans ring (%d) and store (%d)", slots, part_a.slots, part_b.slots)
        return chain

    # ---- inventory / metrics ---------------------------------------------------------------------
    def inventory(self, fingerprint: str | None = None, ring: RingParams | str | None = None) -> list[Inventory]:
        out: dict[tuple[str, str], Inventory] = {}
        if self.store is not None:
            for inv in self.store.inventory(fingerprint, ring):
                out[(inv.fingerprint, inv.ring)] = inv
        rname = ring_from_setting(ring).name if ring is not None else None
        for (fp, rg), mr in self.rings.items():
            if mr.closed or (fingerprint is not None and fp != fingerprint) or (rname is not None and rg != rname):
                continue
            av = mr.available
            inv = out.get((fp, rg)) or Inventory(fp, rg, 0, 0, 0, 0)
            out[(fp, rg)] = Inventory(fp, rg, inv.segments, inv.slots_total + mr.capacity_slots,
                                      inv.slots_available + av, inv.bytes_on_disk)
        if not out and fingerprint is not None and rname is not None:
            return [Inventory(fingerprint, rname, 0, 0, 0, 0)]
        return [out[k] for k in sorted(out)]

    def available_slots(self, fingerprint: str, ring: RingParams | str) -> int:
        return sum(i.slots_available for i in self.inventory(fingerprint, ring))

    def reservation_cache_bytes(self, fingerprint: str, ring: RingParams | str) -> int:
        best = 0
        if self.store is not None and hasattr(self.store, "reservation_cache_bytes"):
            best = int(self.store.reservation_cache_bytes(fingerprint, ring))
        mr = self.ring_for(fingerprint, ring)
        if mr is not None:
            outs = [lin.out_features for lin in mr.linears]
            best = max(best, chunk_cache_bytes(auto_chunk_slots(outs, mr.ring), outs, mr.ring))
        return best

    def metrics(self) -> dict[str, dict[tuple[str, ...], float]]:
        out: dict[str, dict[tuple[str, ...], float]] = {}
        if self.store is not None and hasattr(self.store, "metrics"):
            out = {k: dict(v) for k, v in self.store.metrics().items()}
        low = int(getattr(getattr(self.store, "settings", None), "low_water_slots", 0) or 0)
        frac = float(getattr(self.cfg, "ring_low_fraction", 0.5))
        locked = 0
        for (fp, rg), mr in self.rings.items():
            if mr.closed:
                continue
            key = (fp, rg)
            for name, v in mr.metrics().items():
                out.setdefault(name, {})[key] = v
            total = self.available_slots(fp, rg)
            out.setdefault("pgw_masks_available_slots", {})[key] = float(total)
            lw = low if self.store is not None else int(mr.capacity_slots * frac)
            out.setdefault("pgw_masks_low_water", {})[key] = 1.0 if total < lw else 0.0
            with self._lock:
                fb = self.fallback_by_key.get(key, 0)
            out.setdefault("pgw_masks_ring_fallback_reservations_total", {})[key] = float(fb)
            locked += mr.locked_bytes
        out["pgw_masks_ring_locked_bytes"] = {(): float(locked)}
        return out

    def status(self) -> dict[str, Any]:
        st: dict[str, Any] = {}
        if self.store is not None and hasattr(self.store, "status"):
            st = dict(self.store.status())
        st["ring"] = {f"{short_fingerprint(fp)}/{rg}": mr.status() for (fp, rg), mr in self.rings.items()}
        st["ring_fallback_reservations"] = self.fallback_reservations
        st["ring_spanning_reservations"] = self.spanning_reservations
        st["ring_only"] = self.ring_only
        return st

    # ---- delegated store housekeeping ----------------------------------------------------------
    def gc(self) -> int:
        return int(self.store.gc()) if self.store is not None else 0

    def claim_owner(self, role: str = "gateway") -> bool:
        return bool(self.store.claim_owner(role)) if self.store is not None else True

    def add_low_water_listener(self, fn: Callable[[Inventory], None]) -> None:
        if self.store is not None:
            self.store.add_low_water_listener(fn)

    def _require_external_counter(self) -> None:
        if self.store is not None:
            self.store._require_external_counter()

    def close(self) -> None:
        """Close and wipe every ring (closing the store stays with its owner)."""
        with self._lock:
            rings = list(self._rings.values())
            self._rings.clear()
        for mr in rings:
            mr.close()


def check_init_on_free(policy: str = "warn", *,
                       status_fn: Callable[[], bool | None] = init_on_free_status) -> bool | None:
    """v2 drops sealed slabs without zeroing them (SPEC 7.8): warn, or refuse (``policy='refuse'``),
    unless the kernel zeroes freed pages (``init_on_free=1``). Returns the detected status."""
    on = status_fn()
    if on:
        return True
    msg = ("mask_ring.factory_process: the kernel does not zero freed pages (init_on_free=1 "
           f"{'is off' if on is False else 'not detected'}): consumed shared slabs are dropped unzeroed (sealed, "
           "not writable), so their seed, uW and W r stay in freed physical pages until the kernel reuses them; "
           "boot with init_on_free=1")
    if policy == "refuse":
        raise ConfigError(msg + " (or set mask_ring.init_on_free_policy=warn)")
    log.warning(msg)
    return on


def build_tiered_source(store: Any, ring_settings: Any, *, min_soundness_bits: float, masked_models: int,
                        harden: bool = True) -> TieredMaskSource:
    """The gateway hook: process hardening, swap policy, then a :class:`TieredMaskSource`.

    ``harden`` (default) makes the process non-dumpable (mandatory with the ring) and applies
    ``swap_policy``: ``refuse`` (default) fails when unencrypted disk swap is active, ``warn`` logs.
    """
    if harden:
        harden_process()
        sw = swap_status()
        if sw["unprotected"]:
            msg = (f"unencrypted swap is active ({', '.join(sw['unprotected'])}): mask copies on the heap "
                   "(reservation caches, tensors) could be written to disk; disable swap or encrypt it")
            if str(getattr(ring_settings, "swap_policy", "refuse")) == "refuse":
                raise ConfigError(msg + " (or set mask_ring.swap_policy=warn)")
            log.warning(msg)
        if getattr(ring_settings, "factory_process", False):
            check_init_on_free(str(getattr(ring_settings, "init_on_free_policy", "warn")))
    if getattr(ring_settings, "ring_only", False):
        store = None
    return TieredMaskSource(store, ring_settings, min_soundness_bits=min_soundness_bits,
                            masked_models=masked_models)
