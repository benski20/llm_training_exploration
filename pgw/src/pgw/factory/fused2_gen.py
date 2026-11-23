"""fused2 pair-mask factory: encrypted store segments for the fused2 schedule (opt-in, default off).

The customer-side plain factory for fused2 (research/plans/fused2-two-trips-per-layer.md section 11): it holds
the **public** fused2 weights (``Fused2Weights``: the pack's int8 ``o`` / ``down`` and the folded int8
``qkv/f2`` / ``gate_up/f2``, plus the public multipliers ``hmul``), never anything of the provider's, and writes
encrypted segments in the mask store's fused2 format (:mod:`pgw.maskstore.format`, ``schedule = "fused2"``,
``check_field = "p61"``) for a weight-free gateway:

* per fused2 linear (pass order ``0.qkv/f2, 0.o, 0.gate_up/f2, 0.down, 1.qkv/f2, ...``) and slot, the row
  ``r = SeedExpander(seed).ring_uniform_rows(n, K, slot, 1, "r", key)`` (``r_x`` for a trip's first linear, ``v``
  for a pair's second) and the exact single-stage product ``W r mod 2^K`` (``W1 r_x`` or ``W2' v``): wrapping
  64-bit integer arithmetic only (:class:`pgw.core.ringmm.PreparedWeight`: ``cpu_f64`` exact digit GEMMs on the
  CPU, ``cuda_int8`` tensor-core digit GEMMs on a customer GPU, the GPU mask service's arithmetic);
* per linear the secret p61 check words ``uW = (u * hmul) W mod (2^61 - 1)`` with
  ``u = u64le(stream("u-p61", key)) mod p`` (:func:`pgw.maskstore.fused2.p61_check_vectors`), exact integers.

Guards (as the GPU mask service's): :meth:`Fused2MaskGenerator.startup_check` compares sampled rows of the
production device's products and sampled ``uW`` columns bit for bit with independent references (numpy uint64
wrapping products, Python integers mod p) before the first segment; every chunk of every linear is then
verified before it is written with private ring check vectors (``u_s . (W r) == (u_s W) . r mod 2^K`` on
sampled rows, numpy uint64, independent of the GEMM backend); on CUDA the device memory the caching allocator
holds is scrubbed after every segment. A mismatch raises :class:`pgw.core.IntegrityError` and aborts the segment.

``python -m pgw.factory.fused2_gen generate|bench`` runs it as its own process (see :func:`main`).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pgw.core import native
from pgw.core.errors import ConfigError, IntegrityError
from pgw.core.prng import SEED_BYTES, SeedExpander, new_seed
from pgw.core.ring import RingParams
from pgw.core.ringmm import PreparedWeight
from pgw.maskstore.format import FUSED2_SCHEDULE, MAX_CHUNK_BYTES, P61, auto_chunk_slots
from pgw.maskstore.fused2 import fused2_soundness_bits, p61_check_vectors

__all__ = ["Fused2GenStats", "Fused2MaskGenerator", "Fused2Spec", "fused2_specs"]

log = logging.getLogger(__name__)

_NATIVE_RNG = native.native_available("mask_rows_with_key")
#: Rows of ``W`` per block of the exact numpy references (bounds the uint64 weight copy to 512 x n x 8 bytes).
_REF_BLOCK = 512


@dataclass(frozen=True)
class Fused2Spec:
    """One fused2 segment entry: a fused2 linear key and its shape (``W (m, n)``)."""

    id: str
    out_features: int
    in_features: int


def fused2_specs(weights: Any) -> list[Fused2Spec]:
    """The segment entries of a fused2 weight set, in pass order (``Fused2Weights.linears()``)."""
    return [Fused2Spec(lin.key, lin.m, lin.n) for lin in weights.linears()]


@dataclass
class Fused2GenStats:
    """Work of the last segment (no secret)."""

    slots: int = 0
    seconds: float = 0.0
    gemm_seconds: float = 0.0
    rng_seconds: float = 0.0
    check_seconds: float = 0.0
    write_seconds: float = 0.0
    uw_seconds: float = 0.0
    macs: int = 0
    rows_checked: int = 0
    device: str = "cpu"
    backend: str = ""
    segment_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def slots_per_second(self) -> float:
        return self.slots / self.seconds if self.seconds > 0 else 0.0


def _w_u64(w8: torch.Tensor, r0: int, r1: int) -> np.ndarray:
    return np.ascontiguousarray(w8[r0:r1].numpy().astype(np.int64)).view(np.uint64)


def reference_wr(r: torch.Tensor, w8: torch.Tensor, k: int) -> np.ndarray:
    """``r @ W8^T mod 2^K`` with numpy uint64 wrapping products (no BLAS floats, no digits): the independent
    reference of the start-up check. ``r`` int64 ``(t, n)`` in ``[0, 2^K)``; returns uint64 ``(t, m)``."""
    ru = np.ascontiguousarray(r.numpy()).view(np.uint64)
    m = int(w8.shape[0])
    out = np.empty((ru.shape[0], m), dtype=np.uint64)
    for r0 in range(0, m, _REF_BLOCK):
        r1 = min(m, r0 + _REF_BLOCK)
        out[:, r0:r1] = ru @ _w_u64(w8, r0, r1).T
    out &= np.uint64((1 << k) - 1)
    return out


#: Limb width of the float64 p61 product: ``2^21 * 2^7 * m < 2^53`` for ``m < 2^25`` (every partial sum exact).
_UW_LIMB = 21
_UW_BLOCK = 4096


def p61_uw(u: torch.Tensor, w8: torch.Tensor, hmul: torch.Tensor | None) -> torch.Tensor:
    """``(u * hmul) W8 mod p`` (int64 ``(c, n)`` in ``[0, p)``), exact.

    ``u * hmul mod p`` (Python integers on the ``(c, m)`` vector) is split in three 21-bit limbs; each limb times
    the int8 weight is one float64 BLAS product whose every partial sum is an integer below ``2^21 * 128 * m <
    2^53`` (``m < 2^25``), so it is exact in any summation order; the three ``(c, n)`` results are recombined mod
    ``p`` in Python integers. ``W8`` is converted to float64 in blocks of rows (bounded memory)."""
    m = int(w8.shape[0])
    if m >= 1 << 25:
        raise ConfigError("out_features too large for the exact float64 p61 product")
    uh = u.numpy().astype(np.int64)
    if hmul is not None:
        uh = (uh.astype(object) * hmul.numpy().astype(np.int64).astype(object)[None, :] % P61).astype(np.int64)
    lm = (1 << _UW_LIMB) - 1
    limbs = np.stack([(uh >> (_UW_LIMB * i)) & lm for i in range(3)]).astype(np.float64)  # (3, c, m)
    c = uh.shape[0]
    acc = np.zeros((3 * c, int(w8.shape[1])), dtype=np.float64)
    flat = limbs.reshape(3 * c, m)
    for r0 in range(0, m, _UW_BLOCK):
        r1 = min(m, r0 + _UW_BLOCK)
        acc += flat[:, r0:r1] @ w8[r0:r1].to(torch.float64).numpy()
    parts = np.rint(acc).astype(np.int64).reshape(3, c, -1)
    out = np.zeros(parts.shape[1:], dtype=object)
    for i in range(3):
        out += parts[i].astype(object) * (1 << (_UW_LIMB * i))
    res = torch.from_numpy(np.asarray(out % P61, dtype=np.int64))
    for a in (uh, limbs, acc, parts):
        a.fill(0)
    return res


class Fused2MaskGenerator:
    """Writes fused2 pair-mask segments of one fused2 weight set into a mask store (customer side).

    Args:
        weights: The full public :class:`pgw.engine.fused2.Fused2Weights` (``w8`` and ``hmul`` per linear; a
            provider-only file is enough: scales and norms are not used).
        ring: The fused2 ring (e.g. ``K56A12C53``).
        device: ``"cpu"`` (plain CPU factory), ``"cuda"`` (the GPU mask service's device) or ``"auto"``.
        backend: :class:`PreparedWeight` backend (``auto``: ``cuda_int8`` on CUDA, ``cpu_f64`` on the CPU).
        lane_fingerprint: The fused2 lane fingerprint the segments carry (default: computed from ``weights``
            with :func:`pgw.service.fused2_lane.fused2_fingerprint`; it binds every ``w8`` and ``hmul``).
        self_check_rows: Rows per chunk verified with private ring vectors before writing (``-1``: all rows;
            ``0`` is refused: a wrong mask would silently corrupt outputs only the gateway's checks would see).
        chunk_rows: Rows per GEMM batch (throughput only).
        he: Optional :class:`pgw.factory.fused2_he.DigitSplitProducts`: the products ``W r mod 2^K`` come from the
            existing encrypted factory (mlwe v2 / pks-v3) as independent uniform digits recombined exactly (no
            local GEMM); every guard below still runs on the recombined products.
        pack: The pack the weights were derived from (status only).
    """

    def __init__(self, weights: Any, ring: RingParams | str, *, device: str = "cpu", backend: str = "auto",
                 lane_fingerprint: str | None = None, self_check_rows: int = 2, chunk_rows: int = 64,
                 native_rng: bool | None = None, he: Any = None, pack: Any = None):
        self.w = weights
        self.ring = ring if isinstance(ring, RingParams) else RingParams.from_name(ring)
        if self.ring.k % 8 or not self.ring.cap or self.ring.cap + 1 >= 61:
            raise ConfigError("fused2 masks need a capped ring, K a multiple of 8, 2^(C+1) < 2^61 - 1")
        if self_check_rows == 0 or self_check_rows < -1:
            raise ConfigError("self_check_rows must be -1 (all) or positive")
        if chunk_rows < 1:
            raise ConfigError("chunk_rows must be >= 1")
        if lane_fingerprint is None:
            from pgw.service.fused2_lane import fused2_fingerprint

            lane_fingerprint = fused2_fingerprint(weights, self.ring)
        self.fingerprint = str(lane_fingerprint)
        dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.device = torch.device(dev)
        self.backend = backend
        self.self_check_rows = int(self_check_rows)
        self.chunk_rows = int(chunk_rows)
        if native_rng and not _NATIVE_RNG:
            raise ConfigError("native_rng=True but pgw_fast.mask_rows_with_key is not available")
        self.native_rng = _NATIVE_RNG if native_rng is None else bool(native_rng)
        self.lins = {lin.key: lin for lin in weights.linears()}
        self.he = he
        self.pack = pack
        if he is not None and self.device.type != "cpu":
            raise ConfigError("the encrypted (digit-split) fused2 factory runs on the CPU ")
        self.specs = fused2_specs(weights)
        self._pw: dict[str, PreparedWeight] = {}
        self._priv: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # private ring self-check vectors per linear
        self._lock = threading.Lock()
        self.last_stats = Fused2GenStats(device=str(self.device))
        #: Scrub the device's cached-but-unused memory after every segment (on CUDA; the GPU mask service).
        self.scrub_device = self.device.type == "cuda"

    # ---- geometry ---------------------------------------------------------------------------------------
    @property
    def key(self) -> tuple[str, str]:
        return self.fingerprint, self.ring.name

    @property
    def bytes_per_slot(self) -> int:
        return sum(s.out_features for s in self.specs) * self.ring.bytes_per_element

    @property
    def macs_per_slot(self) -> int:
        return sum(s.out_features * s.in_features for s in self.specs)

    def store_chunk_slots(self, target_bytes: int = 1 << 20) -> int:
        cs = auto_chunk_slots([s.out_features for s in self.specs], self.ring, target_bytes)
        widest = max(s.out_features for s in self.specs) * self.ring.bytes_per_element
        return max(1, min(cs, MAX_CHUNK_BYTES // widest))

    # ---- arithmetic -------------------------------------------------------------------------------------
    def _prepared(self, key: str) -> PreparedWeight:
        pw = self._pw.get(key)
        if pw is None:
            pw = self._pw[key] = PreparedWeight(self.lins[key].w8, device=self.device, backend=self.backend,  # type: ignore[arg-type]
                                                cache_float=self.device.type == "cpu")
        return pw

    def products(self, key: str, r: torch.Tensor) -> torch.Tensor:
        """``W r mod 2^K`` (int64 ``(t, m)`` on the CPU) on the generator's device (or the encrypted factory)."""
        if self.he is not None:
            return self.he.matmul(key, r)
        pw = self._prepared(key)
        x = r.to(self.device, non_blocking=True) if self.device.type != "cpu" else r
        y = pw.matmul(x, self.ring.k)
        if self.device.type != "cpu":
            out = y.cpu()
            y.zero_()
            x.zero_()
            return out
        return y

    def _rows(self, rkey: bytes | bytearray, n: int, start: int, rows: int) -> torch.Tensor:
        if self.native_rng:
            return torch.from_numpy(native.fast.mask_rows_with_key(rkey, n, self.ring.k, start, rows))
        return SeedExpander.ring_uniform_rows_with_key(rkey, n, self.ring.k, start, rows)

    def _private(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Private ring self-check vectors of a linear (2 x m uniform mod 2^K, never stored) and ``u_s W mod 2^K``."""
        hit = self._priv.get(key)
        if hit is None:
            lin = self.lins[key]
            us = np.frombuffer(os.urandom(16 * lin.m), dtype=np.uint64).reshape(2, lin.m) & np.uint64(self.ring.mask)
            acc = np.zeros((2, lin.n), dtype=np.uint64)
            for r0 in range(0, lin.m, _REF_BLOCK):
                r1 = min(lin.m, r0 + _REF_BLOCK)
                acc += us[:, r0:r1] @ _w_u64(lin.w8, r0, r1)
            acc &= np.uint64(self.ring.mask)
            hit = self._priv[key] = (us.copy(), acc)
        return hit

    def _self_check(self, key: str, r: torch.Tensor, wr: torch.Tensor) -> int:
        """Ring Freivalds identity on sampled rows (all with ``self_check_rows = -1``); returns rows checked."""
        t = int(r.shape[0])
        if self.self_check_rows < 0 or self.self_check_rows >= t:
            idx = np.arange(t)
        else:
            idx = np.sort(np.frombuffer(os.urandom(8 * self.self_check_rows), dtype=np.uint64) % np.uint64(t))
            idx = np.unique(idx.astype(np.int64))
        us, usw = self._private(key)
        rr = np.ascontiguousarray(r.numpy()[idx]).view(np.uint64)
        yy = np.ascontiguousarray(wr.numpy()[idx]).view(np.uint64)
        mask = np.uint64(self.ring.mask)
        lhs = (yy @ us.T) & mask
        rhs = (rr @ usw.T) & mask
        rr.fill(0)
        yy.fill(0)
        if not np.array_equal(lhs, rhs):
            raise IntegrityError(f"fused2 factory self-check failed for {key} (device {self.device}); segment aborted")
        return int(idx.size)

    def check_words(self, seed: bytes, checks: int) -> dict[str, torch.Tensor]:
        """``uW = (u * hmul) W mod p`` of every linear for segment ``seed`` (secret)."""
        out: dict[str, torch.Tensor] = {}
        for s in self.specs:
            lin = self.lins[s.id]
            u = p61_check_vectors(seed, checks, lin.m, s.id)
            out[s.id] = p61_uw(u, lin.w8, lin.hmul)
            u.zero_()
        return out

    # ---- segments ---------------------------------------------------------------------------------------
    def tokens_per_second(self) -> float:
        """Slots per second of the last segment (the ``pgw factory generate`` report)."""
        return self.last_stats.slots_per_second

    def segment_bytes(self, slots: int) -> int:
        """Disk bytes of a ``slots``-slot segment (ring products; the p61 words are small)."""
        return int(slots) * self.bytes_per_slot

    def generate_into_store(self, store: Any, slots: int, *, checks: int | None = 1, seed: bytes | None = None,
                            store_chunk_slots: int | None = None, factory_id: str = "fused2",
                            linears: Sequence[str] | None = None, chunk_slots: int | None = None,
                            throttle: Any = None, store_chunk_bytes: int | None = None, **_ignored: Any) -> str:
        """Generate one fresh-seeded fused2 segment of ``slots`` slots into ``store``; returns its id.

        ``store`` is anything with :meth:`pgw.maskstore.MaskStore.writer` (a store or an export sink). On any
        error the partial segment is aborted, never committed. ``seed``: tests only (production: a fresh
        ``os.urandom`` seed per segment). ``linears``: tests only (write a subset; the header lists them all).
        The scheduler's arguments (:class:`pgw.factory.scheduler.FactoryScheduler`): ``chunk_slots`` (rows per
        product batch), ``throttle`` (rate limit / duty cycle / stop, accounted per chunk in slot-equivalents),
        ``store_chunk_bytes`` (encrypted chunk size).
        """
        if checks is None:
            checks = 1
        if chunk_slots is not None and int(chunk_slots) < 1:
            raise ConfigError("chunk_slots must be >= 1")
        chunk_rows = int(chunk_slots) if chunk_slots is not None else self.chunk_rows
        if store_chunk_slots is None and store_chunk_bytes is not None:
            store_chunk_slots = self.store_chunk_slots(int(store_chunk_bytes))
        total_mn = max(1, self.macs_per_slot)
        if checks < 1 or checks > 8:
            raise ConfigError("checks must be in [1, 8]")
        if fused2_soundness_bits(self.ring, checks) <= 0:
            raise ConfigError("the p61 check is not sound for this ring")
        seg_seed = bytes(seed) if seed is not None else new_seed()
        if len(seg_seed) != SEED_BYTES:
            raise ConfigError(f"seed must be exactly {SEED_BYTES} bytes")
        st = Fused2GenStats(slots=int(slots), device=str(self.device))
        with self._lock:
            t0 = time.perf_counter()
            writer = store.writer(
                self.fingerprint, self.ring, self.specs, slots=int(slots), seed=seg_seed, checks=int(checks),
                chunk_slots=store_chunk_slots if store_chunk_slots is not None else self.store_chunk_slots(),
                factory_id=factory_id, check_field="p61", schedule=FUSED2_SCHEDULE,
            )
            exp = SeedExpander(seg_seed)
            try:
                for s in self.specs:
                    if linears is not None and s.id not in linears:
                        continue
                    t1 = time.perf_counter()
                    uw = p61_uw(p61_check_vectors(seg_seed, checks, s.out_features, s.id), self.lins[s.id].w8,
                                self.lins[s.id].hmul)
                    st.uw_seconds += time.perf_counter() - t1
                    rkey = bytearray(exp.stream_key("r", s.id))
                    try:
                        for a in range(0, int(slots), chunk_rows):
                            rows = min(chunk_rows, int(slots) - a)
                            if throttle is not None:
                                throttle.check()
                            t1 = time.perf_counter()
                            r = self._rows(rkey, s.in_features, a, rows)
                            t2 = time.perf_counter()
                            wr = self.products(s.id, r)
                            t3 = time.perf_counter()
                            st.rows_checked += self._self_check(s.id, r, wr)
                            t4 = time.perf_counter()
                            r.zero_()
                            writer.write_linear(s.id, wr, uw)
                            wr.zero_()
                            t5 = time.perf_counter()
                            st.rng_seconds += t2 - t1
                            st.gemm_seconds += t3 - t2
                            st.check_seconds += t4 - t3
                            st.write_seconds += t5 - t4
                            st.macs += rows * s.in_features * s.out_features
                            if throttle is not None:
                                throttle.account(t5 - t1, rows * s.in_features * s.out_features / total_mn)
                    finally:
                        rkey[:] = bytes(len(rkey))
                        uw.zero_()
                t1 = time.perf_counter()
                seg_id = str(writer.commit())
                st.write_seconds += time.perf_counter() - t1
            except BaseException:
                try:
                    writer.abort()
                except Exception as exc:  # pragma: no cover - best effort
                    log.error("aborting partial fused2 segment failed: %r", exc)
                raise
            finally:
                self._scrub()
            st.seconds = time.perf_counter() - t0
            st.segment_id = seg_id
            st.backend = (self.he.resolved_backend if self.he is not None else
                          next(iter(self._pw.values())).resolved_backend if self._pw else "")
            if self.he is not None:
                st.extra["he"] = dict(self.he.stats)
                st.extra["he_backend"] = dict(getattr(self.he.backend, "stats", {}) or {})
            self.last_stats = st
        log.info("fused2 segment %s: %d slots in %.2f s (%.1f slots/s, %s/%s)", seg_id, slots, st.seconds,
                 st.slots_per_second, self.device, st.backend)
        return seg_id

    def _scrub(self) -> None:
        if self.scrub_device and self.device.type == "cuda":  # pragma: no cover - GPU only
            from pgw.factory.mask_service import scrub_device_memory

            scrub_device_memory(self.device)

    # ---- start-up self-check ----------------------------------------------------------------------------
    def startup_check(self, *, rows: int = 2, linears: int = 8, uw_columns: int = 8) -> dict[str, Any]:
        """Before the first segment: products of the production device vs numpy uint64 wrapping products for
        ``rows`` fresh random rows of sampled linears (the first layer's, the last layer's and evenly spaced
        ones; ``linears <= 0``: all), and sampled ``uW`` columns vs Python integers mod p. Bit for bit; any
        difference raises :class:`IntegrityError`. Returns a report (no secret)."""
        t0 = time.perf_counter()
        n = len(self.specs)
        if linears <= 0 or linears >= n:
            pick = list(range(n))
        else:
            base = {0, 1, 2, 3, n - 4, n - 3, n - 2, n - 1}
            base |= {round(i * (n - 1) / max(1, linears - 1)) for i in range(linears)}
            pick = sorted(i for i in base if 0 <= i < n)
        seed = new_seed()
        exp = SeedExpander(seed)
        compared = 0
        uw_compared = 0
        for i in pick:
            s = self.specs[i]
            lin = self.lins[s.id]
            r = exp.ring_uniform_rows(s.in_features, self.ring.k, 0, rows, "startup", s.id)
            got = self.products(s.id, r).numpy().view(np.uint64)
            ref = reference_wr(r, lin.w8, self.ring.k)
            if not np.array_equal(got, ref):
                raise IntegrityError(f"fused2 factory start-up check: products of {s.id} differ from the reference "
                                     f"on {self.device}; refusing to produce masks")
            compared += int(ref.size)
            u = p61_check_vectors(seed, 1, lin.m, s.id)
            uw = p61_uw(u, lin.w8, lin.hmul).numpy()
            cols = np.unique(np.frombuffer(os.urandom(8 * uw_columns), dtype=np.uint64) % np.uint64(lin.n))
            hm = [1] * lin.m if lin.hmul is None else [int(v) for v in lin.hmul.tolist()]
            ul = [int(v) for v in u[0].tolist()]
            w = lin.w8.numpy()
            for j in cols.tolist():
                col = [int(v) for v in w[:, j].tolist()]
                want = sum(a * h * c for a, h, c in zip(ul, hm, col, strict=True)) % P61
                if int(uw[0, j]) != want:
                    raise IntegrityError(f"fused2 factory start-up check: uW of {s.id} differs from the reference")
                uw_compared += 1
            r.zero_()
            u.zero_()
        self._scrub()
        return {"ok": True, "device": str(self.device), "linears_checked": [self.specs[i].id for i in pick],
                "wr_elements_compared": compared, "uw_elements_compared": uw_compared,
                "seconds": time.perf_counter() - t0}


# ---- process entry ----------------------------------------------------------------------------------------------
def _shape_weights(model: str, layers: int) -> Any:
    """Random public weights with the shapes of a Qwen2.5 checkpoint (benchmark only; multipliers 1..255)."""
    from types import SimpleNamespace

    from pgw.engine.fused2 import F2Linear

    shapes = {"qwen2.5-7b": (3584, 4608, 37888, 18944), "qwen2.5-0.5b": (896, 1152, 9728, 4864),
              "qwen2.5-32b": (5120, 7168, 55296, 27648), "qwen2.5-72b": (8192, 10240, 59136, 29568)}
    if model not in shapes:
        raise ConfigError(f"unknown model shapes {model!r}; known: {sorted(shapes)}")
    d, qkv, gu, inter = shapes[model]
    g = torch.Generator().manual_seed(0)

    def lin(key: str, m: int, n: int, hm: bool) -> F2Linear:
        w8 = torch.randint(-127, 128, (m, n), dtype=torch.int8, generator=g)
        h = torch.randint(1, 256, (m,), dtype=torch.int64, generator=g) if hm else None
        return F2Linear(key, w8, torch.ones(m, dtype=torch.float64), None, 0.0, hmul=h)

    w = SimpleNamespace(num_layers=layers, fingerprint="", qkv=[], o=[], gu=[], down=[])
    for i in range(layers):
        w.qkv.append(lin(f"{i}.qkv/f2", qkv, d, False))
        w.o.append(lin(f"{i}.o", d, d, True))
        w.gu.append(lin(f"{i}.gate_up/f2", gu, d, False))
        w.down.append(lin(f"{i}.down", d, inter, i + 1 < layers))

    def linears() -> list[F2Linear]:
        out: list[F2Linear] = []
        for i in range(layers):
            out += [w.qkv[i], w.o[i], w.gu[i], w.down[i]]
        return out

    w.linears = linears
    return w


class _NullWriter:
    """A writer that discards rows (benchmark: generation cost without disk)."""

    def __init__(self) -> None:
        self.rows = 0

    def write_linear(self, lid: Any, wr: torch.Tensor, uw: torch.Tensor) -> None:
        self.rows += int(wr.shape[0])

    def commit(self) -> str:
        return "bench"

    def abort(self) -> None:
        pass


class _NullStore:
    def writer(self, *a: Any, **kw: Any) -> _NullWriter:
        return _NullWriter()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - process entry point
    """``python -m pgw.factory.fused2_gen generate --weights F2.pt --store-root DIR --key-file KEK --slots N``
    writes fused2 segments into an encrypted store; ``bench --model qwen2.5-7b --layers 2 --slots 16`` measures
    generation throughput at a checkpoint's shapes (random public weights; per-layer cost scales linearly)."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="fused2 pair-mask factory (opt-in)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--weights", required=True, help="Fused2Weights.save() file (provider_only is enough)")
    g.add_argument("--ring", default="K56A12C53")
    g.add_argument("--store-root", required=True)
    g.add_argument("--key-file", required=True)
    g.add_argument("--slots", type=int, required=True)
    g.add_argument("--segments", type=int, default=1)
    g.add_argument("--checks", type=int, default=1)
    g.add_argument("--device", default="auto")
    b = sub.add_parser("bench")
    b.add_argument("--model", default="qwen2.5-7b")
    b.add_argument("--layers", type=int, default=1)
    b.add_argument("--slots", type=int, default=16)
    b.add_argument("--ring", default="K56A12C53")
    b.add_argument("--device", default="auto")
    b.add_argument("--chunk-rows", type=int, default=64)
    b.add_argument("--json", default="")
    b.add_argument("--store-root", default="", help="also encrypt + write into a fresh store here (disk!)")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        from pgw.engine.fused2 import Fused2Weights
        from pgw.maskstore import FileKeyProvider, MaskStore, MaskStoreSettings

        gen = Fused2MaskGenerator(Fused2Weights.load(args.weights), args.ring, device=args.device)
        print(json.dumps(gen.startup_check()), flush=True)
        store = MaskStore(MaskStoreSettings(root=args.store_root), FileKeyProvider(args.key_file))
        try:
            for _ in range(args.segments):
                sid = gen.generate_into_store(store, args.slots, checks=args.checks)
                st = gen.last_stats
                print(f"segment {sid}: {args.slots} slots, {st.slots_per_second:.2f} slots/s", flush=True)
        finally:
            store.close()
        return
    w = _shape_weights(args.model, args.layers)
    gen = Fused2MaskGenerator(w, args.ring, device=args.device, lane_fingerprint="00" * 32,
                              chunk_rows=args.chunk_rows)
    t0 = time.perf_counter()
    gen.products(gen.specs[0].id, torch.zeros((1, gen.specs[0].in_features), dtype=torch.int64))  # warm-up
    for s in gen.specs:
        gen._prepared(s.id)
    gen.generate_into_store(_NullStore(), min(args.slots, args.chunk_rows))  # warm-up: private self-check vectors
    prep = time.perf_counter() - t0
    target: Any = _NullStore()
    if args.store_root:  # encrypt and write too (a real store; KEK in memory)
        from pgw.maskstore import MaskStore, StaticKeyProvider

        target = MaskStore.open(args.store_root, StaticKeyProvider(os.urandom(32)), low_water_slots=0)
    try:
        gen.generate_into_store(target, args.slots)
    finally:
        if args.store_root:
            target.close()
    st = gen.last_stats
    layers_full = {"qwen2.5-7b": 28, "qwen2.5-0.5b": 24, "qwen2.5-32b": 64, "qwen2.5-72b": 80}[args.model]
    per_slot_layer = st.seconds / args.slots / args.layers
    gemm_slot_layer = st.gemm_seconds / args.slots / args.layers
    rep = {"model": args.model, "ring": gen.ring.name, "device": str(gen.device), "backend": st.backend,
           "torch_threads": torch.get_num_threads(), "layers_measured": args.layers, "slots": args.slots,
           "seconds": st.seconds, "gemm_s": st.gemm_seconds, "rng_s": st.rng_seconds, "check_s": st.check_seconds,
           "write_s": st.write_seconds, "uw_s_per_segment": st.uw_seconds, "prepare_s": prep,
           "encrypted_store": bool(args.store_root),
           "macs_per_slot_full_model": gen.macs_per_slot // args.layers * layers_full,
           "gmacs_per_s": st.macs / max(st.gemm_seconds, 1e-9) / 1e9,
           "slots_per_s_full_model": 1.0 / ((st.seconds - st.uw_seconds) / args.slots / args.layers * layers_full),
           "slots_per_s_full_model_gemm_only": 1.0 / (gemm_slot_layer * layers_full),
           "s_per_slot_layer": per_slot_layer}
    print(json.dumps(rep, indent=1), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":  # pragma: no cover
    main()
