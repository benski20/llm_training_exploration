"""In-process stand-ins for the compute service and the mask store.

* :class:`LocalComputeClient` implements the ``ComputeClient`` surface the engine uses
  (``compute_sync`` / async ``compute`` / ``models``) by evaluating ``W8 x_hat mod 2^K`` locally with
  :class:`pgw.core.PreparedWeight`. It validates requests like the real service (fingerprint, ring,
  shapes) and never keeps payloads. Used for tests, benchmarks and single-box development.
* :class:`LocalMaskReservation` is an in-memory, single-use mask reservation that derives ``r``
  and ``u`` from a seed exactly as the mask factory does (SPEC 7.3) and computes ``W8 r`` and
  ``u^T W8`` on the gateway ("self-factory" mode, SPEC 16). It enforces per-linear in-order,
  never-reused rows and zeroizes on close.
* :class:`LocalMaskSource` hands out such reservations with the ``MaskStore.reserve`` signature.

None of these change the protocol: the bytes a ``LocalComputeClient`` sees are exactly what the
remote service would see.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from pgw.core.errors import (
    ConfigError,
    FingerprintMismatchError,
    MaskExhaustedError,
    MaskReuseError,
    ProtocolError,
    ShapeError,
)
from pgw.core.freivalds import freivalds_precompute
from pgw.core.packing import pack_ring, packed_size, unpack_ring
from pgw.core.prng import SEED_BYTES, SeedExpander
from pgw.core.ring import RingParams
from pgw.core.ringmm import PreparedWeight
from pgw.core.types import LinearId, LinearSpec

__all__ = ["LocalComputeClient", "LocalMaskReservation", "LocalMaskSource"]

log = logging.getLogger(__name__)


class LocalComputeClient:
    """In-process compute "service" with the :class:`pgw.wire.ComputeClient` interface.

    Args:
        pack: Model pack (int8 linears are loaded lazily).
        rings: Ring names served (default: all presets).
        device: Device of the ring matmuls (``cuda`` uses the int8 tensor-core path).
        backend: ``ringmm`` backend override.
        tamper: Test hook ``fn(linear_id, y_hat) -> y_hat`` applied to results (simulates a
            malicious service). ``None`` in production.
        record: Keep a copy of every received ``x_hat`` (test hook for privacy assertions).
    """

    def __init__(
        self,
        pack: Any,
        *,
        rings: Sequence[str] = ("K32A8", "K40A12", "K48A20", "K32A12C27"),
        device: str = "cpu",
        backend: str = "auto",
        tamper: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
        record: bool = False,
    ):
        self.pack = pack
        self.rings = tuple(rings)
        self.device = torch.device(device)
        self.backend = backend
        self.tamper = tamper
        self.record = record
        self.received: list[tuple[str, torch.Tensor]] = []
        self.requests = 0
        self._specs: dict[str, LinearSpec] = {s.id.key: s for s in pack.linear_specs()}
        self._weights: dict[str, PreparedWeight] = {}
        self._lock = threading.Lock()

    def _weight(self, key: str) -> PreparedWeight:
        w = self._weights.get(key)
        if w is None:
            with self._lock:
                w = self._weights.get(key)
                if w is None:
                    lw = self.pack.load_linear(LinearId.parse(key))
                    w = PreparedWeight(lw.w8, device=self.device, backend=self.backend)  # type: ignore[arg-type]
                    self._weights[key] = w
        return w

    def compute_sync(self, request: Any) -> Any:
        """Evaluate a :class:`pgw.wire.ComputeRequest`; returns a ``ComputeResponse``."""
        from pgw.wire.messages import ComputeResponse, ComputeResult, ServerTiming

        t0 = time.perf_counter()
        if request.fingerprint != self.pack.fingerprint:
            raise FingerprintMismatchError("unknown model fingerprint", details={"fp": str(request.fingerprint)[:12]})
        if request.ring not in self.rings:
            raise ConfigError(f"ring {request.ring} not served")
        ring = RingParams.from_name(request.ring)
        out = []
        rows_total = 0
        for item in request.items:
            spec = self._specs.get(item.linear)
            if spec is None:
                raise ProtocolError(f"unknown linear {item.linear!r}")
            if item.rows <= 0:
                raise ShapeError("rows must be positive")
            if len(item.x_hat) != packed_size((item.rows, spec.in_features), ring):
                raise ShapeError(f"{item.linear}: x_hat length does not match rows * n * K/8")
            x_hat = unpack_ring(item.x_hat, (item.rows, spec.in_features), ring)
            if self.record:
                self.received.append((item.linear, x_hat.clone()))
            y_hat = self._weight(item.linear).matmul(x_hat.to(self.device), ring).cpu()
            if self.tamper is not None:
                y_hat = self.tamper(item.linear, y_hat) & ring.mask
            out.append(ComputeResult(item.linear, item.rows, pack_ring(y_hat, ring)))
            rows_total += item.rows
        self.requests += 1
        ms = (time.perf_counter() - t0) * 1e3
        return ComputeResponse(request.request_id, tuple(out), ServerTiming(0.0, ms, rows_total))

    async def compute(self, request: Any) -> Any:
        """Async wrapper (runs the matmuls in a worker thread)."""
        return await asyncio.to_thread(self.compute_sync, request)

    async def connect(self, expected_fingerprint: str | None = None, ring: str | None = None) -> Any:
        """Mirror of ``ComputeClient.connect``: check fingerprint and ring."""
        if expected_fingerprint is not None and expected_fingerprint != self.pack.fingerprint:
            raise FingerprintMismatchError("service does not serve the expected fingerprint")
        if ring is not None and ring not in self.rings:
            raise ConfigError(f"ring {ring} not served")
        return await self.models()

    async def models(self) -> Any:
        """``ModelsResponse`` describing the local pack."""
        from pgw.wire.messages import LinearInfo, ModelInfo, ModelsResponse

        linears = tuple(LinearInfo(s.id.key, s.out_features, s.in_features) for s in self._specs.values())
        info = ModelInfo(self.pack.name, self.pack.fingerprint, self.rings, linears, 8192, 256 << 20)
        return ModelsResponse((info,), server="pgw-engine-local")

    async def aclose(self) -> None:
        """Release prepared weights."""
        self._weights.clear()

    def close(self) -> None:
        """Synchronous close."""
        self._weights.clear()


class LocalMaskReservation:
    """In-memory single-use reservation of ``slots`` mask slots (one extent, one seed).

    Rows of each linear are derived lazily per :meth:`take` from the seed with the normative labels
    (``"r", linear_id`` / ``"u", linear_id``), so memory stays proportional to one call.

    Args:
        pack: Model pack (needs int8 weights to compute ``W8 r`` and ``u^T W8``).
        ring: Ring parameters.
        slots: Number of token positions.
        seed: 32-byte seed (default ``os.urandom``). Never reuse a seed across reservations.
        checks: Freivalds vectors per linear.
        slot_offset: First slot index within the seed's stream (to mimic a segment extent).
        weights: Optional shared ``{linear_key: PreparedWeight}`` cache (factory side).
        backend: ``ringmm`` backend for ``W8 r`` (``cpu_f32`` halves resident memory vs ``cpu_f64``).
        check_field: Freivalds check format, as a segment header's (``None``: the ring check; ``"p31"``:
            the prime-field check, served by :meth:`checks_compact` only).
    """

    def __init__(
        self,
        pack: Any,
        ring: RingParams,
        slots: int,
        *,
        seed: bytes | None = None,
        checks: int = 2,
        slot_offset: int = 0,
        weights: dict[str, PreparedWeight] | None = None,
        backend: str = "auto",
        check_field: str | None = None,
    ):
        from pgw.core.freivalds import field_modulus

        if field_modulus(check_field) is not None and (ring.k != 32 or checks < 1):
            raise ConfigError("check_field='p31' needs a K32 ring and checks >= 1")
        self._check_field = check_field if field_modulus(check_field) is not None else None
        self._tid = os.urandom(8).hex()
        if slots <= 0:
            raise ConfigError("slots must be positive")
        if checks < 0:
            raise ConfigError("checks must be >= 0")
        self.pack = pack
        self.ring = ring
        self.fingerprint: str = pack.fingerprint
        self._slots = slots
        self._offset = slot_offset
        self._checks = checks
        self._exp: SeedExpander | None = SeedExpander(seed if seed is not None else os.urandom(SEED_BYTES))
        self._specs: dict[str, LinearSpec] = {s.id.key: s for s in pack.linear_specs()}
        self._cursor: dict[str, int] = dict.fromkeys(self._specs, 0)
        self._weights = weights if weights is not None else {}
        self._backend = backend
        self._uw: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._closed = False

    # ---- MaskReservation surface ----------------------------------------------------------------
    @property
    def slots(self) -> int:
        """Total slots reserved."""
        return self._slots

    @property
    def extents(self) -> Sequence[Any]:
        """Single in-memory extent."""
        return ({"segment_id": "local", "start": self._offset, "count": self._slots},)

    def _spec(self, lid: LinearId | str) -> LinearSpec:
        spec = self._specs.get(str(lid))
        if spec is None:
            raise ConfigError(f"unknown linear {lid}")
        return spec

    def _weight(self, key: str) -> PreparedWeight:
        w = self._weights.get(key)
        if w is None:
            w = PreparedWeight(self.pack.load_linear(LinearId.parse(key)).w8, backend=self._backend)  # type: ignore[arg-type]
            self._weights[key] = w
        return w

    def _alive(self) -> SeedExpander:
        if self._closed or self._exp is None:
            raise MaskReuseError("mask reservation is closed")
        return self._exp

    def remaining(self, lid: LinearId) -> int:
        """Rows not yet taken for ``lid``."""
        return self._slots - self._cursor[self._spec(lid).id.key]

    def rows_until_boundary(self, lid: LinearId) -> int:
        """Single extent: same as :meth:`remaining`."""
        return self.remaining(lid)

    def take(self, lid: LinearId, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Next ``rows`` masks for ``lid``: ``(r (rows, n), W8 r (rows, m))``."""
        spec = self._spec(lid)
        key = spec.id.key
        if rows <= 0:
            raise ConfigError("rows must be positive")
        with self._lock:
            exp = self._alive()
            start = self._cursor[key]
            if start + rows > self._slots:
                raise MaskExhaustedError(
                    "mask reservation exhausted",
                    details={"linear": key, "needed": rows, "remaining": self._slots - start},
                )
            self._cursor[key] = start + rows
        r = exp.ring_uniform_rows(spec.in_features, self.ring.k, self._offset + start, rows, "r", key)
        wr = self._weight(key).matmul(r, self.ring)
        return r, wr

    def checks(self, lid: LinearId) -> tuple[torch.Tensor, torch.Tensor]:
        """Freivalds vectors ``(u (c, m), u^T W8 (c, n))`` (ring check format only)."""
        if self._check_field is not None:
            raise ConfigError(f"{lid}: prime-field check vectors are served by checks_compact only")
        return tuple(t.clone() for t in self._vectors(lid))  # type: ignore[return-value]

    def _vectors(self, lid: LinearId | str) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self._spec(lid)
        key = spec.id.key
        exp = self._alive()
        if self._checks == 0:
            raise ConfigError("reservation has no Freivalds vectors (checks=0)")
        hit = self._uw.get(key)
        if hit is None:
            if self._check_field is not None:
                from pgw.core.freivalds import prime_check_vectors, prime_precompute

                seed = exp.derive_seed("local-prime-check")
                u = prime_check_vectors(seed, self._checks, spec.out_features, key)
                uw = prime_precompute(u, self.pack.load_linear(spec.id).w8)
            else:
                u = exp.ring_uniform((self._checks, spec.out_features), self.ring.k, "u", key)
                uw = freivalds_precompute(u, self.pack.load_linear(spec.id).w8, self.ring)
            hit = (u, uw)
            self._uw[key] = hit
        return hit

    def check_field(self, lid: LinearId | str) -> str | None:
        """Check format of the reservation (one extent): ``None`` (ring) or ``"p31"``."""
        return self._check_field

    def checks_compact(self, lid: LinearId | str) -> tuple[tuple[str, ...], Any, Any]:
        """``(token, u, uw)`` as uint32 (``K <= 32``) or uint64 copies; prime-field tokens end in ``"p31"``."""
        from pgw.core.freivalds import PRIME_TAG

        u, uw = self._vectors(lid)
        dt = np.uint32 if self.ring.k <= 32 else np.uint64
        key = self._spec(lid).id.key
        tid = f"local-{id(self):x}-{self._tid}"  # names this reservation's vectors only
        token = (tid, key, PRIME_TAG) if self._check_field is not None else (tid, key)
        return token, (u.numpy() & self.ring.mask).astype(dt), (uw.numpy() & self.ring.mask).astype(dt)

    def close(self) -> None:
        """Burn unused rows and drop secrets."""
        with self._lock:
            self._closed = True
            self._exp = None
            for u, uw in self._uw.values():
                u.zero_()
                uw.zero_()
            self._uw.clear()
            burned = sum(self._slots - c for c in self._cursor.values())
            if burned:
                log.debug("local reservation closed; burned %d unused rows", burned)

    def __enter__(self) -> LocalMaskReservation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _auto_checks(pack: Any, ring: RingParams, min_bits: float = 20.0) -> int:
    """Smallest ``checks`` in ``[2, 8]`` giving ``min_bits`` of soundness for every linear (else 8)."""
    from pgw.core.freivalds import soundness_bits

    ns = {s.in_features for s in pack.linear_specs()}
    for c in range(2, 9):
        if all(soundness_bits(ring, n, c) >= min_bits for n in ns):
            return c
    return 8


class LocalMaskSource:
    """Creates :class:`LocalMaskReservation` objects with the ``MaskStore.reserve`` signature.

    A fresh random seed is drawn per reservation, so masks are never reused across sessions.
    ``W8`` is prepared once and shared between reservations.
    """

    def __init__(self, pack: Any, *, checks: int | None = None, backend: str = "auto"):
        self.pack = pack
        #: ``None`` = per ring, the smallest number of vectors giving 20 bits of soundness (max 8).
        self.checks = checks
        self.backend = backend
        self._weights: dict[str, PreparedWeight] = {}
        self.reserved_slots = 0

    def reserve(self, fingerprint: str, ring: RingParams, slots: int, *, session_id: str = "") -> LocalMaskReservation:
        """Reserve ``slots`` fresh slots."""
        if fingerprint != self.pack.fingerprint:
            raise FingerprintMismatchError("mask source serves a different model fingerprint")
        self.reserved_slots += slots
        checks = self.checks if self.checks is not None else _auto_checks(self.pack, ring)
        return LocalMaskReservation(
            self.pack, ring, slots, checks=checks, weights=self._weights, backend=self.backend
        )
