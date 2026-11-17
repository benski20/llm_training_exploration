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

import torch

from pgw.core.errors import ConfigError, PGWError
from pgw.core.fingerprint import short_fingerprint
from pgw.core.prng import SEED_BYTES, SeedExpander, new_seed
from pgw.core.ring import RingParams
from pgw.core.settings import ring_from_setting
from pgw.core.types import LinearId, LinearSpec
from pgw.factory.governor import Throttle
from pgw.factory.kernels import RingKernel, resolve_backend, resolve_device, ring_vecmat
from pgw.factory.streaming import WeightStreamer, auto_resident_bytes

__all__ = ["GeneratedLinear", "GenerationStats", "MaskGenerator", "slot_bytes"]

log = logging.getLogger(__name__)


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
    ):
        self.pack = pack
        self.ring = ring_from_setting(ring)
        if hasattr(pack, "check_ring"):
            pack.check_ring(self.ring)
        self.specs: list[LinearSpec] = list(pack.linear_specs())
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

    def tokens_per_second(self) -> float:
        """Measured throughput of the last run (slots/s)."""
        return self.last_stats.slots_per_second

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

        def make_r(task: tuple[int, int, int]) -> torch.Tensor:
            i, start, rows = task
            t0 = time.perf_counter()
            key = specs[i].id.key
            r = exp.ring_uniform_rows(specs[i].in_features, k, start, rows, "r", key)
            if cuda:  # pragma: no cover - GPU only
                r = r.pin_memory()
            rng_time[0] += time.perf_counter() - t0
            return r

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-factory-rng") if self.pipeline else None
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
                kern = RingKernel(w, self.backend)
                t0 = time.perf_counter()
                if checks:
                    u = exp.ring_uniform((checks, spec.out_features), k, "u", lid.key)
                    uw: torch.Tensor = ring_vecmat(u, w, self.ring)
                    u.zero_()
                else:
                    uw = torch.zeros((0, spec.in_features), dtype=torch.int64)
                stats.check_seconds += time.perf_counter() - t0
                for start in range(0, slots, chunk_slots):
                    task = tasks[ti]
                    ti += 1
                    if pending is not None:
                        r = pending.result()
                        pending = pool.submit(make_r, tasks[ti]) if pool is not None and ti < len(tasks) else None
                    else:
                        r = make_r(task)
                    t0 = time.perf_counter()
                    wr = kern.matmul(r, self.ring)
                    wr = wr.cpu() if cuda else wr
                    dt = time.perf_counter() - t0
                    r.zero_()
                    del r
                    stats.gemm_seconds += dt
                    stats.wr_bytes += wr.numel() * self.ring.bytes_per_element
                    if thr is not None:
                        before = thr.slept_s
                        thr.account(dt, task[2] * spec.out_features * spec.in_features / total_mn)
                        stats.throttle_seconds += thr.slept_s - before
                    yield GeneratedLinear(lid, start, wr, uw if start == 0 else None)
                del kern, w
        finally:
            if pending is not None:
                pending.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            stats.rng_seconds = rng_time[0]
            stats.seconds = time.perf_counter() - t_start

    def generate_into_store(
        self,
        store: Any,
        slots: int,
        *,
        checks: int = 2,
        chunk_slots: int = 256,
        factory_id: str = "local",
        seed: bytes | None = None,
        throttle: Throttle | None = None,
    ) -> str:
        """Generate a fresh-seeded segment straight into a :class:`pgw.maskstore.MaskStore`; return its id.

        ``store`` may be anything with a compatible ``writer(...)`` (e.g. an export sink). On any
        error or stop the partial segment is aborted, never committed.

        Args:
            seed: For tests only; production always uses a fresh ``os.urandom`` seed per segment.
        """
        seg_seed = bytes(seed) if seed is not None else new_seed()
        self._validate(seg_seed, slots, checks, chunk_slots)
        with self._lock:
            t0 = time.perf_counter()
            writer = store.writer(
                self.fingerprint, self.ring, self.specs, slots=slots, seed=seg_seed, checks=checks,
                chunk_slots=chunk_slots, factory_id=factory_id,
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
                    continue  # drain after an error
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
