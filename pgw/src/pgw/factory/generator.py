"""Core mask generation: for each linear, ``r`` from the segment seed and ``W r mod 2^K``.

Derivation (normative, SPEC.md 7.3) for a segment with seed ``s``::

    r[slot, :] = SeedExpander(s).ring_uniform_rows(n, K, slot, 1, "r", linear_key)
    u          = SeedExpander(s).ring_uniform((checks, m), K, "u", linear_key)

The generator stores ``W8 r`` (per slot) and ``u W8`` (per segment); ``r`` and ``u`` are never
stored - the gateway regenerates them from the seed.

Work is linear-major (the segment file layout is linear-major, SPEC 7.2): one weight is on the
device at a time, which is what makes weight streaming for 70B-class models possible. With
``pipeline=True`` three stages overlap: ChaCha20 expansion of the next ``r`` chunk (thread), the
ring GEMMs (caller thread; torch releases the GIL) and encryption/writing (writer thread).

On CUDA the ChaCha20 expansion and the digit split run on the GPU instead
(:mod:`pgw.factory.gpu_prng`, ``PGW_FACTORY_GPU_PRNG``): the stream key is derived on the CPU by the
same HKDF code, the keystream is computed straight into the int8 digit planes of the GEMM operand, and
``r`` never crosses the host bus. The kernel is self-tested against the CPU expander before first use
and the output self-check re-derives its sampled rows on the CPU.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError, PGWError
from pgw.core.fingerprint import short_fingerprint
from pgw.core.prng import SEED_BYTES, SeedExpander, max_stream_rows, new_seed
from pgw.core.ring import RingParams
from pgw.core.settings import ring_from_setting
from pgw.core.types import LinearId, LinearSpec
from pgw.factory import gpu_prng as _gp
from pgw.factory.governor import Throttle
from pgw.factory.kernels import RingKernel, resolve_backend, resolve_device, ring_vecmat
from pgw.factory.streaming import WeightStreamer, auto_resident_bytes
from pgw.maskstore.format import DEFAULT_CHUNK_BYTES, auto_chunk_slots

__all__ = ["GeneratedLinear", "GenerationStats", "MaskGenerator", "slot_bytes"]

log = logging.getLogger(__name__)

#: Default rows per chunk verified by the output self-check (see :class:`MaskGenerator`).
DEFAULT_SELF_CHECK_ROWS = 8


@dataclass
class GeneratedLinear:
    """Output for one linear and one slot chunk.

    Attributes:
        id: Linear id.
        slot_start: First slot of this chunk.
        wr: ``(rows, m)`` int64 ring elements ``W8 r``.
        uw: ``(checks, n)`` int64 ``u^T W8`` (only on the first chunk of each linear, else ``None``).
    """

    id: LinearId
    slot_start: int
    wr: torch.Tensor
    uw: torch.Tensor | None


@dataclass
class GenerationStats:
    """Timing breakdown of one :meth:`MaskGenerator.generate` run."""

    slots: int = 0
    linears: int = 0
    seconds: float = 0.0
    rng_seconds: float = 0.0
    gemm_seconds: float = 0.0
    check_seconds: float = 0.0
    write_seconds: float = 0.0
    throttle_seconds: float = 0.0
    wr_bytes: int = 0
    self_checked_rows: int = 0
    backend: str = ""
    device: str = ""
    segment_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def slots_per_second(self) -> float:
        """Slots (token positions, all linears) per wall-clock second."""
        return self.slots / self.seconds if self.seconds > 0 else 0.0


def slot_bytes(specs: list[LinearSpec], ring: RingParams) -> int:
    """Stored ``W r`` bytes per slot: ``sum(m) * K/8``."""
    return sum(s.out_features for s in specs) * ring.bytes_per_element


def _u64(t: torch.Tensor) -> np.ndarray:
    """Canonical int64 ring elements as a uint64 array (values in ``[0, 2^K)`` are unchanged)."""
    return t.detach().cpu().numpy().astype(np.uint64)


class _RemoteKernel:
    """Adapter giving :class:`pgw.hemask.HEMaskBackend` the ``RingKernel.matmul`` interface."""

    def __init__(self, backend: Any, linear: str):
        self.backend = backend
        self.linear = linear

    def matmul(self, r: torch.Tensor, ring: RingParams) -> torch.Tensor:
        out: torch.Tensor = self.backend.matmul(self.linear, r)
        return out


class _Stop:
    """Writer-queue sentinel."""


class MaskGenerator:
    """Computes masks for segments of a model pack.

    Derivation is normative (see :mod:`pgw.maskstore.format`): ``r`` rows come from
    ``SeedExpander(seed).ring_uniform_rows(n, K, slot, rows, "r", linear_id)`` and the check vectors
    from ``ring_uniform((checks, m), K, "u", linear_id)``. Uses exact int8 ring GEMMs
    (``cuda_int8`` on GPU: ~0.0066 s/token for 8B at K=48 on a T4; ``cpu_int8`` via CPU
    ``torch._int_mm`` when exact, else float64).

    Args:
        pack: Open :class:`pgw.modelpack.ModelPack` (``components=("linears",)`` suffices).
        ring: Ring parameters (or preset name).
        device: Torch device for GEMMs (``"auto"`` picks CUDA when available).
        backend: ``auto``, ``cuda_int8``, ``cpu_int8``, ``cpu_f64`` or ``cpu_f32``.
        streamer: Optional :class:`WeightStreamer`; by default all weights stay resident when they
            fit within ``memory_fraction`` of the device, else they are streamed.
        throttle: Optional :class:`Throttle` (rate limit / duty cycle / stop).
        pipeline: Overlap ``r`` expansion, GEMMs and writing in threads.
        memory_fraction: Device memory fraction for resident weights (CUDA).
        self_check_rows: Rows per chunk whose output is verified before it is written, using the
            Freivalds identity ``u . (W8 r) == (u W8) . r (mod 2^K)`` with the segment's check
            vectors (or two private ones when ``checks=0``). An error with 2-adic valuation ``v``
            escapes one vector with probability ``2^-(K-v)``; errors in the top bits are also
            caught later by the gateway's output range check. A wrong mask would not be caught
            anywhere else: the gateway would silently produce garbage, because the service's
            Freivalds check only covers ``W8 x_hat``. A kernel fault (e.g. a saturating int8 GEMM
            or a GPU bit flip) affects many rows, so a sample suffices. ``0`` disables the
            check, ``-1`` verifies every row (about 5% extra CPU time). A mismatch raises
            :class:`pgw.core.IntegrityError` and the segment is aborted. The rows of ``r`` it uses
            are always the CPU expander's, so with the GPU PRNG a keystream fault fails it too.
        gpu_prng: Expand ``r`` (ChaCha20 + digit split) on the GPU: ``None`` = ``PGW_FACTORY_GPU_PRNG``
            (default ``auto``: on for ``cuda_int8``), ``False`` = CPU ChaCha20, ``True`` = on. Used only
            after :func:`pgw.factory.gpu_prng.self_test` passed on the device (checked on first use; on
            failure an error is logged and the CPU path is used). Never with ``remote``.
        gpu_prng_impl: ``auto`` (the Triton kernel on CUDA). Tests may pass ``triton`` or ``torch`` with
            ``gpu_prng=True`` to run the device path on a ``cpu_int8`` generator (Triton interpreter or
            the torch emulation).
    """

    def __init__(
        self,
        pack: Any,
        ring: RingParams | str,
        device: str = "cpu",
        backend: str = "auto",
        streamer: WeightStreamer | None = None,
        *,
        throttle: Throttle | None = None,
        pipeline: bool = True,
        memory_fraction: float = 0.85,
        self_check_rows: int = DEFAULT_SELF_CHECK_ROWS,
        remote: Any = None,
        gpu_prng: bool | str | None = None,
        gpu_prng_impl: str = "auto",
    ):
        self.pack = pack
        # Encrypted mask generation (pgw.hemask.HEMaskBackend): W8 r comes from the provider on
        # MLWE ciphertexts. Every row is then verified with private check vectors (never the
        # segment's stored u, which the provider could otherwise probe through rejections).
        self.remote = remote
        if remote is not None:
            self_check_rows = -1
        self.ring = ring_from_setting(ring)
        if hasattr(pack, "check_ring"):
            pack.check_ring(self.ring)
        self.specs: list[LinearSpec] = list(pack.linear_specs())
        self._weak_warned: set[int] = set()
        if not self.specs:
            raise ConfigError("model pack has no linear groups")
        for s in self.specs:
            self.ring.check_in_features(s.in_features)
        self.fingerprint: str = pack.fingerprint
        self.device = resolve_device(device)
        self.backend = resolve_backend(self.device, backend)
        if streamer is None:
            max_lin = max(s.out_features * s.in_features for s in self.specs)
            reserve = 2 * max_lin + (1 << 30)
            budget = auto_resident_bytes(self.device, memory_fraction, reserve)
            streamer = WeightStreamer(pack, self.device, budget)
        elif torch.device(streamer.device) != self.device:
            raise ConfigError(f"streamer device {streamer.device} != generator device {self.device}")
        self.streamer = streamer
        self.throttle = throttle
        self.pipeline = pipeline
        if self_check_rows < -1:
            raise ConfigError(f"self_check_rows must be -1 (all), 0 (off) or positive, got {self_check_rows}")
        self.self_check_rows = int(self_check_rows)
        self.gpu_prng_requested = _gp.gpu_prng_setting(gpu_prng)
        if gpu_prng_impl not in ("auto", "triton", "torch"):
            raise ConfigError(f"gpu_prng_impl must be auto, triton or torch, got {gpu_prng_impl!r}")
        self.gpu_prng_impl = gpu_prng_impl
        self._dev_prng: str | None = None
        self._dev_prng_resolved = False
        self._dev_prng_lock = threading.Lock()
        self.last_stats = GenerationStats(backend=self.backend, device=str(self.device))
        self._lock = threading.Lock()  # one segment at a time per generator
        log.info(
            "mask generator for %s (%s) ring %s on %s/%s, %d linears, %.2f MB/slot",
            getattr(pack, "name", "?"), short_fingerprint(self.fingerprint), self.ring.name, self.device,
            self.backend, len(self.specs), slot_bytes(self.specs, self.ring) / 1e6,
        )

    # ---- properties --------------------------------------------------------------------------
    @property
    def key(self) -> tuple[str, str]:
        """``(fingerprint, ring name)`` this generator produces for."""
        return self.fingerprint, self.ring.name

    @property
    def bytes_per_slot(self) -> int:
        """Stored ``W r`` bytes per slot."""
        return slot_bytes(self.specs, self.ring)

    def store_chunk_slots(self, target_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
        """Encrypted-chunk size (slots) giving about ``target_bytes`` for the widest linear."""
        return auto_chunk_slots([s.out_features for s in self.specs], self.ring, target_bytes)

    def max_segment_slots(self) -> int:
        """Largest segment the PRNG stream range allows for this model (see ``max_stream_rows``)."""
        return min(max_stream_rows(s.in_features) for s in self.specs)

    def segment_bytes(self, slots: int) -> int:
        """Approximate on-disk size of a segment of ``slots`` slots (W r rows dominate)."""
        return int(slots) * self.bytes_per_slot + (1 << 20)

    def tokens_per_second(self) -> float:
        """Measured throughput of the last run (slots/s)."""
        return self.last_stats.slots_per_second

    # ---- GPU PRNG ------------------------------------------------------------------------------
    def device_prng(self) -> str | None:
        """Implementation used to expand ``r`` on the device (``"triton"``/``"torch"``), or ``None`` (CPU).

        Resolved once, on first use: runs the GPU PRNG self-test (compiles the kernel) when it applies.
        """
        with self._dev_prng_lock:
            if not self._dev_prng_resolved:
                self._dev_prng = self._resolve_device_prng()
                self._dev_prng_resolved = True
            return self._dev_prng

    def _resolve_device_prng(self) -> str | None:
        want = self.gpu_prng_requested
        if want is False or self.remote is not None:
            return None
        if self.self_check_rows == 0:
            # the per-chunk output self-check is what re-derives sampled r rows on the CPU; without it
            # the startup self-test alone would guard the keystream at production shapes
            (log.warning if want else log.info)(
                "GPU mask PRNG needs the output self-check (self_check_rows != 0); using CPU ChaCha20")
            return None
        impl = self.gpu_prng_impl
        if self.backend == "cuda_int8":
            impl = "triton" if impl == "auto" else impl
        elif self.backend == "cpu_int8" and want is True and impl != "auto":
            pass  # test hook: the device path on CPU tensors
        else:
            if want is True:
                log.info("GPU mask PRNG requested but the factory runs %s on %s; using CPU ChaCha20",
                         self.backend, self.device)
            return None
        if impl == "triton" and not _gp.triton_available():
            (log.warning if want else log.info)("GPU mask PRNG: triton is not available; using CPU ChaCha20")
            return None
        if not _gp.self_test(self.device, impl, (self.ring.k,)):  # type: ignore[arg-type]
            return None  # self_test logged the error
        log.info("mask generator: r expansion and digit split on %s (%s)", self.device, impl)
        return impl

    def _disable_device_prng(self, why: str) -> None:
        with self._dev_prng_lock:
            if self._dev_prng is not None:
                log.error("mask generator: disabling the GPU mask PRNG on %s after %s; using CPU ChaCha20",
                          self.device, why)
            self._dev_prng = None
            self._dev_prng_resolved = True

    # ---- generation --------------------------------------------------------------------------
    @staticmethod
    def _validate(seed: bytes, slots: int, checks: int, chunk_slots: int) -> None:
        if not isinstance(seed, bytes | bytearray) or len(seed) != SEED_BYTES:
            raise ConfigError(f"seed must be exactly {SEED_BYTES} bytes")
        if slots <= 0:
            raise ConfigError(f"slots must be positive, got {slots}")
        if not 0 <= checks <= 8:
            raise ConfigError(f"checks must be in [0, 8], got {checks}")
        if chunk_slots <= 0:
            raise ConfigError(f"chunk_slots must be positive, got {chunk_slots}")

    #: Target bits of Freivalds soundness per call (range check + check vectors). ``checks=None``
    #: picks the smallest number of vectors that reaches it; below it a warning is logged. Gateways
    #: and stores refuse segments under their own floor (default 20, ``min_soundness_bits``).
    MIN_SOUNDNESS_BITS = 20

    def checks_for(self, min_bits: float | None = None) -> int:
        """Smallest ``checks`` in ``[2, 8]`` giving ``min_bits`` of soundness for every linear (else 8)."""
        from pgw.core.freivalds import soundness_bits

        target = self.MIN_SOUNDNESS_BITS if min_bits is None else min_bits
        for c in range(2, 9):
            if all(soundness_bits(self.ring, sp.in_features, c) >= target for sp in self.specs):
                return c
        return 8

    def _warn_weak_integrity(self, checks: int) -> None:
        """Warn once per ``checks`` value when the integrity margin is weak for some linear (docs finding 2)."""
        if checks == 0 or checks in self._weak_warned:
            return
        self._weak_warned.add(checks)
        from pgw.core.freivalds import soundness_bits

        worst = min(self.specs, key=lambda sp: soundness_bits(self.ring, sp.in_features, checks))
        bits = soundness_bits(self.ring, worst.in_features, checks)
        if bits < self.MIN_SOUNDNESS_BITS:
            need = next((c for c in range(checks + 1, 9)
                         if soundness_bits(self.ring, worst.in_features, c) >= self.MIN_SOUNDNESS_BITS), None)
            log.warning(
                "weak integrity margin: %s with checks=%d gives only %.0f bits of Freivalds soundness for %s "
                "(n=%d); %s",
                self.ring.name, checks, bits, worst.id.key, worst.in_features,
                f"use checks>={need}" if need else "use a wider ring (K40A12 or K48A20)",
            )

    def generate(
        self, seed: bytes, slots: int, *, checks: int = 2, chunk_slots: int = 256, throttle: Throttle | None = None
    ) -> Iterator[GeneratedLinear]:
        """Yield chunks for every linear (canonical order, then slot order) of one segment.

        The ``r`` buffers are zeroized after use; ``wr`` tensors belong to the caller.
        """
        self._validate(seed, slots, checks, chunk_slots)
        thr = throttle or self.throttle
        exp = SeedExpander(bytes(seed))
        k = self.ring.k
        specs = self.specs
        total_mn = float(sum(s.out_features * s.in_features for s in specs))
        cuda = self.device.type == "cuda"
        tasks = [
            (i, start, min(chunk_slots, slots - start))
            for i in range(len(specs))
            for start in range(0, slots, chunk_slots)
        ]
        stats = GenerationStats(slots=slots, linears=len(specs), backend=self.backend, device=str(self.device))
        self.last_stats = stats
        rng_time = [0.0]
        dev_prng = self.device_prng() if self.remote is None else None
        stats.extra["gpu_prng"] = dev_prng is not None
        n_limbs = _gp.limbs(k)

        def make_r(task: tuple[int, int, int]) -> torch.Tensor:
            i, start, rows = task
            t0 = time.perf_counter()
            key = specs[i].id.key
            r = exp.ring_uniform_rows(specs[i].in_features, k, start, rows, "r", key)
            if cuda:  # pragma: no cover - GPU only
                pinned = r.pin_memory()
                r.zero_()
                r = pinned
            rng_time[0] += time.perf_counter() - t0
            return r

        pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-factory-rng")
                if self.pipeline and dev_prng is None else None)
        stream: _gp.DeviceRingStream | None = None
        pending: Future[torch.Tensor] | None = None
        t_start = time.perf_counter()
        ti = 0
        try:
            if pool is not None:
                pending = pool.submit(make_r, tasks[0])
            for li, (lid, w) in enumerate(self.streamer.iter_device_weights()):
                spec = specs[li]
                if lid != spec.id:  # pragma: no cover - streamer contract
                    raise PGWError(f"weight streamer order mismatch: got {lid}, expected {spec.id}")
                if tuple(w.shape) != (spec.out_features, spec.in_features):
                    raise ConfigError(f"{lid}: weight shape {tuple(w.shape)} does not match manifest")
                kern: Any = RingKernel(w, self.backend) if self.remote is None else _RemoteKernel(self.remote, lid.key)
                t0 = time.perf_counter()
                cu: np.ndarray | None = None
                cw: np.ndarray | None = None
                if checks:
                    u = exp.ring_uniform((checks, spec.out_features), k, "u", lid.key)
                    uw: torch.Tensor = ring_vecmat(u, w, self.ring)
                    if self.self_check_rows and self.remote is None:
                        cu, cw = _u64(u.t()), _u64(uw.t())
                    u.zero_()
                else:
                    uw = torch.zeros((0, spec.in_features), dtype=torch.int64)
                if cu is None and self.self_check_rows:  # private check vectors, never stored
                    v = SeedExpander(new_seed()).ring_uniform((2, spec.out_features), k, "v")
                    cu, cw = _u64(v.t()), _u64(ring_vecmat(v, w, self.ring).t())
                    v.zero_()
                stats.check_seconds += time.perf_counter() - t0
                if dev_prng is not None:
                    # stream key derived on the CPU (SPEC 7.3, unchanged); expansion on the device
                    stream = _gp.DeviceRingStream(exp.stream_key("r", lid.key), spec.in_features, k, w.device,
                                                  dev_prng)  # type: ignore[arg-type]
                for start in range(0, slots, chunk_slots):
                    task = tasks[ti]
                    ti += 1
                    r: torch.Tensor | None = None
                    if stream is not None:
                        t0 = time.perf_counter()
                        p, np_ = kern.plane_shape(task[2])
                        d8 = torch.zeros((n_limbs * p, np_), dtype=torch.int8, device=w.device)
                        try:
                            try:
                                stream.digit_planes(start, task[2], d8)
                            except Exception as exc:
                                # a launch/compile fault at production shapes: this segment fails, later
                                # segments of this generator use the CPU path
                                self._disable_device_prng(f"an error in the GPU mask PRNG ({type(exc).__name__})")
                                raise
                            rng_time[0] += time.perf_counter() - t0
                            t0 = time.perf_counter()
                            wr = kern.matmul_planes(d8, task[2], self.ring)
                        finally:
                            d8.zero_()
                            del d8
                    else:
                        if pending is not None:
                            r = pending.result()
                            pending = pool.submit(make_r, tasks[ti]) if pool is not None and ti < len(tasks) else None
                        else:
                            r = make_r(task)
                        t0 = time.perf_counter()
                        wr = kern.matmul(r, self.ring)
                    wr = wr.cpu() if cuda else wr
                    dt = time.perf_counter() - t0
                    if cu is not None and cw is not None:
                        t1 = time.perf_counter()
                        try:
                            stats.self_checked_rows += self._self_check(lid, r, wr, cu, cw, stream=stream, row0=start)
                        except BaseException as exc:
                            if r is not None:
                                r.zero_()
                            wr.zero_()
                            if stream is not None and isinstance(exc, IntegrityError):
                                self._disable_device_prng("a failed output self-check")
                            raise
                        stats.check_seconds += time.perf_counter() - t1
                    if r is not None:
                        r.zero_()
                    del r
                    stats.gemm_seconds += dt
                    stats.wr_bytes += wr.numel() * self.ring.bytes_per_element
                    if thr is not None:
                        before = thr.slept_s
                        thr.account(dt, task[2] * spec.out_features * spec.in_features / total_mn)
                        stats.throttle_seconds += thr.slept_s - before
                    yield GeneratedLinear(lid, start, wr, uw if start == 0 else None)
                if stream is not None:
                    stream.wipe()
                    stream = None
                del kern, w
        finally:
            if stream is not None:
                stream.wipe()
            if pending is not None:
                pending.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            if pending is not None and pending.done() and not pending.cancelled() and pending.exception() is None:
                pending.result().zero_()  # prefetched r of a chunk that was never used
            stats.rng_seconds = rng_time[0]
            stats.seconds = time.perf_counter() - t_start

    def _self_check(self, lid: LinearId, r: torch.Tensor | None, wr: torch.Tensor, cu: np.ndarray, cw: np.ndarray,
                    *, stream: _gp.DeviceRingStream | None = None, row0: int = 0) -> int:
        """Verify ``wr[t] @ cu == r[t] @ cw (mod 2^K)`` on sampled rows; return the rows checked.

        ``cu`` is ``(m, c)`` (check vectors as columns) and ``cw`` is ``(n, c)`` (``u W8`` transposed).
        With the GPU PRNG (``r is None``) the sampled rows ``row0 + t`` of ``r`` are re-derived by the CPU
        expander from ``stream``'s key, so a wrong GPU keystream fails the check like a wrong GEMM.

        uint64 arithmetic wraps mod ``2^64``, which is exact mod ``2^K``. Row selection is not
        secret (it only has to be unpredictable to a faulty kernel, not to an adversary).

        Raises:
            IntegrityError: the GEMM output (or ``u W8``) is wrong; the segment must be discarded.
        """
        rows = int(wr.shape[0])
        n_check = rows if self.self_check_rows < 0 else min(rows, self.self_check_rows)
        if n_check <= 0:
            return 0
        idx = torch.randperm(rows)[:n_check] if n_check < rows else torch.arange(rows)
        ws = wr.index_select(0, idx) if n_check < rows else wr
        if r is None:
            if stream is None:
                raise PGWError("self-check needs r or its stream")
            rs = stream.cpu_rows(row0, idx)
        else:
            rs = r.index_select(0, idx) if n_check < rows else r
        try:
            lhs = (_u64(ws) @ cu) & np.uint64(self.ring.mask)
            rhs = (_u64(rs) @ cw) & np.uint64(self.ring.mask)
        finally:
            if rs is not r:
                rs.zero_()
        if not np.array_equal(lhs, rhs):
            bad = int((lhs != rhs).any(axis=1).sum())
            log.error("factory self-check failed for %s on %s/%s: %d of %d sampled rows wrong",
                      lid, self.device, self.backend, bad, n_check)
            raise IntegrityError(
                "mask factory output failed its self-check (ring GEMM is not exact on this device/backend)",
                details={"linear": str(lid), "backend": self.backend, "device": str(self.device),
                         "bad_rows": bad, "checked_rows": n_check},
            )
        return n_check

    def generate_into_store(
        self,
        store: Any,
        slots: int,
        *,
        checks: int | None = None,
        chunk_slots: int = 256,
        factory_id: str = "local",
        seed: bytes | None = None,
        throttle: Throttle | None = None,
        store_chunk_slots: int | None = None,
        store_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> str:
        """Generate a fresh-seeded segment straight into a :class:`pgw.maskstore.MaskStore`; return its id.

        ``store`` may be anything with a compatible ``writer(...)`` (e.g. an export sink). On any
        error or stop the partial segment is aborted, never committed.

        Args:
            checks: Freivalds vectors per linear; ``None`` (default) = :meth:`checks_for` (meets
                :attr:`MIN_SOUNDNESS_BITS` for this ring and model where possible).
            seed: For tests only; production always uses a fresh ``os.urandom`` seed per segment.
            chunk_slots: Rows per GEMM batch (throughput only).
            store_chunk_slots: Slots per encrypted chunk in the segment file; default sized by
                ``store_chunk_bytes`` (bounds the decrypted masks a gateway caches per session).
        """
        if checks is None:
            checks = self.checks_for()
        seg_seed = bytes(seed) if seed is not None else new_seed()
        self._validate(seg_seed, slots, checks, chunk_slots)
        if slots > self.max_segment_slots():
            raise ConfigError(
                f"segment_slots={slots} exceeds the PRNG stream range of this model "
                f"(max {self.max_segment_slots()} slots per segment)",
                details={"slots": slots, "max_slots": self.max_segment_slots()},
            )
        self._warn_weak_integrity(checks)
        with self._lock:
            t0 = time.perf_counter()
            writer = store.writer(
                self.fingerprint, self.ring, self.specs, slots=slots, seed=seg_seed, checks=checks,
                chunk_slots=self.store_chunk_slots(store_chunk_bytes) if store_chunk_slots is None
                else store_chunk_slots,
                factory_id=factory_id,
            )
            write_time = [0.0]
            try:
                self._drive(writer, self.generate(seg_seed, slots, checks=checks, chunk_slots=chunk_slots,
                                                  throttle=throttle), write_time)
                t1 = time.perf_counter()
                seg_id = str(writer.commit())
                write_time[0] += time.perf_counter() - t1
            except BaseException:
                try:
                    writer.abort()
                except Exception as exc:  # pragma: no cover - best effort
                    log.error("aborting partial segment failed: %r", exc)
                raise
            stats = self.last_stats
            stats.seconds = time.perf_counter() - t0
            stats.write_seconds = write_time[0]
            stats.segment_id = seg_id
            log.info(
                "segment %s: %s %s, %d slots in %.2fs (%.1f slots/s; gemm %.2fs, rng %.2fs, write %.2fs, "
                "throttle %.2fs)",
                seg_id, short_fingerprint(self.fingerprint), self.ring.name, slots, stats.seconds,
                stats.slots_per_second, stats.gemm_seconds, stats.rng_seconds, stats.write_seconds,
                stats.throttle_seconds,
            )
            return seg_id

    def _drive(self, writer: Any, chunks: Iterator[GeneratedLinear], write_time: list[float]) -> None:
        """Feed generated chunks to ``writer.write_linear`` (in a writer thread when pipelined)."""
        uw_by_linear: dict[str, torch.Tensor] = {}

        def write(item: GeneratedLinear) -> None:
            if item.uw is not None:
                uw_by_linear[item.id.key] = item.uw
            t = time.perf_counter()
            writer.write_linear(item.id, item.wr, uw_by_linear[item.id.key])
            write_time[0] += time.perf_counter() - t
            item.wr.zero_()

        if not self.pipeline:
            try:
                for item in chunks:
                    write(item)
            finally:
                close = getattr(chunks, "close", None)
                if close is not None:
                    close()
            return

        q: queue.Queue[GeneratedLinear | type[_Stop]] = queue.Queue(maxsize=2)
        failure: list[BaseException] = []

        def worker() -> None:
            while True:
                item = q.get()
                if item is _Stop:
                    return
                if failure:
                    assert isinstance(item, GeneratedLinear)
                    item.wr.zero_()  # drain after an error
                    continue
                try:
                    assert isinstance(item, GeneratedLinear)
                    write(item)
                except BaseException as exc:
                    failure.append(exc)

        th = threading.Thread(target=worker, name="pgw-factory-writer", daemon=True)
        th.start()
        try:
            for item in chunks:
                if failure:
                    break
                q.put(item)
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
            q.put(_Stop)
            th.join()
        if failure:
            raise failure[0]
