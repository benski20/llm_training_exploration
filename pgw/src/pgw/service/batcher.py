"""Cross-request batching: coalesce rows for the same (fingerprint, ring, linear) into one GEMM.

Mixing tenants in one batch is safe: every row is uniform noise to us, and results are split back
by row ranges, so no request ever sees another request's data.

Scheduling model
----------------
* Each ``(fingerprint, ring, linear)`` key has at most one *open* batch. The first rows to arrive
  open it and start a timer of ``batch_window_ms``; when the timer fires (or the batch reaches
  ``max_batch_rows``) the batch is put on its device's ready queue.
* A batch keeps accepting rows **while it waits in the ready queue**, until a worker picks it up.
  Under load (device busy) batches therefore grow beyond what the window alone would collect,
  which is exactly when batching pays off; when idle, latency is bounded by the window.
* Every device has its own FIFO ready queue and a fixed number of execution slots
  (``gpu_workers_per_device``, default 1, so GPU work is serialized per device; ``cpu_workers`` for
  the CPU). Kernels run in a per-device thread pool, so the event loop never blocks on a GEMM. With
  multi-GPU layer placement the devices work in parallel on different layers.
* Dispatch is callback driven, without worker tasks: a batch that becomes ready while its device
  has a free slot is handed to the thread pool synchronously, and the pool's completion callback
  resolves the callers' futures and starts the next ready batch. An idle device therefore costs
  one thread hand-off per round trip and no extra event-loop hops (decode-sized calls are latency
  bound), while a busy device still queues batches that keep growing.
* Rows whose caller went away (deadline, disconnect) before the batch starts are dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future as _CFuture
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from pgw.core.errors import PGWError, ServiceUnavailableError, ShapeError
from pgw.core.ring import RingParams
from pgw.service.gpu_exec import from_compact
from pgw.service.kernels import ComputeKernel

__all__ = ["BatchScheduler", "BatchStats", "BatchTiming"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchTiming:
    """Timing of one submission (the slowest of its chunks).

    Attributes:
        queue_ms: Time from submission until its batch started computing.
        compute_ms: Kernel time of its batch.
        batch_rows: Rows in its (largest) batch, including other requests' rows.
    """

    queue_ms: float = 0.0
    compute_ms: float = 0.0
    batch_rows: int = 0


@dataclass
class BatchStats:
    """Counters for tests, logs and ``/readyz`` details."""

    submitted_rows: int = 0
    batches: int = 0
    batched_rows: int = 0
    dropped_rows: int = 0
    failed_batches: int = 0
    max_batch_rows_seen: int = 0

    @property
    def mean_batch_rows(self) -> float:
        """Average rows per executed batch."""
        return self.batched_rows / self.batches if self.batches else 0.0


@dataclass(eq=False)
class _Entry:
    x: torch.Tensor
    fut: asyncio.Future[tuple[torch.Tensor, BatchTiming]]
    t_enq: float
    #: the caller accepts the kernel's compact result (int32 bit patterns for K <= 32 on CUDA)
    compact: bool = False


@dataclass(eq=False)
class _Batch:
    key: tuple[str, str, str]
    kernel: ComputeKernel
    ring: RingParams
    device: str
    entries: list[_Entry] = field(default_factory=list)
    rows: int = 0
    queued: bool = False
    started: bool = False
    timer: asyncio.TimerHandle | None = None
    #: entries still wanted when the batch started, and their rows
    live: list[_Entry] = field(default_factory=list)
    live_rows: int = 0
    #: failed by ``stop(drain=False)`` while its kernel was still running
    aborted: bool = False


@dataclass(eq=False)
class _Device:
    executor: ThreadPoolExecutor
    slots: int
    busy: int = 0
    ready: deque[_Batch] = field(default_factory=deque)
    running: set[_Batch] = field(default_factory=set)


class BatchScheduler:
    """Async micro-batcher.

    Requests for the same ``(fingerprint, ring, linear)`` arriving within ``batch_window_ms`` (or
    until ``max_batch_rows``) are concatenated, computed with one kernel call and split back.
    Batching is the main throughput lever (batch 32 is ~30x cheaper per token than batch 1).
    A single worker per device serializes GPU work; CPU work runs in a thread pool.

    Args:
        registry: A :class:`pgw.service.ModelRegistry` (anything with ``get(fp).kernel(linear)``).
        batch_window_ms: Max time the first rows of a batch wait for company.
        max_batch_rows: Max rows per kernel call (larger submissions are split).
        cpu_workers: Concurrent kernel executions on the CPU device.
        gpu_workers_per_device: Concurrent kernel executions per GPU.
        max_queued_rows: Admission limit on rows waiting or running (``unavailable`` above it).
        inline_max_rows: Run a batch of at most this many rows directly on the event loop when its
            device is completely idle (nothing running or queued), saving the two thread hand-offs
            of the pool (0 = always use the pool). Blocks the loop for the kernel's duration.
        metrics: Optional :class:`pgw.service.metrics.ServiceMetrics`.
    """

    def __init__(
        self,
        registry: object,
        *,
        batch_window_ms: float = 0.0,
        max_batch_rows: int = 4096,
        cpu_workers: int = 2,
        gpu_workers_per_device: int = 1,
        max_queued_rows: int = 1 << 20,
        metrics: Any | None = None,
        inline_max_rows: int = 0,
    ):
        if batch_window_ms < 0:
            raise ValueError("batch_window_ms must be >= 0")
        if max_batch_rows <= 0 or cpu_workers <= 0 or gpu_workers_per_device <= 0 or max_queued_rows <= 0:
            raise ValueError("max_batch_rows, worker counts and max_queued_rows must be positive")
        self.registry = registry
        self.window_s = batch_window_ms / 1000.0
        self.max_batch_rows = max_batch_rows
        self.cpu_workers = cpu_workers
        self.gpu_workers_per_device = gpu_workers_per_device
        self.max_queued_rows = max_queued_rows
        self.metrics = metrics
        self.inline_max_rows = max(0, int(inline_max_rows))
        self.stats = BatchStats()
        self._open: dict[tuple[str, str, str], _Batch] = {}
        self._devices: dict[str, _Device] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._accepting = False
        self._queued_rows = 0
        self._queued_items = 0
        self._pending_batches = 0  # enqueued and not finished
        self._idle: asyncio.Event | None = None

    # ---- lifecycle -----------------------------------------------------------------------
    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        return self._accepting

    @property
    def queued_rows(self) -> int:
        """Rows submitted and not yet finished."""
        return self._queued_rows

    @property
    def queued_items(self) -> int:
        """Submissions (chunks) waiting for a worker."""
        return self._queued_items

    async def start(self) -> None:
        """Start accepting work (idempotent). Device thread pools are created lazily on first use."""
        self._loop = asyncio.get_running_loop()
        if self._idle is None:
            self._idle = asyncio.Event()
            self._idle.set()
        self._accepting = True

    def _ensure_device(self, device: str) -> _Device:
        d = self._devices.get(device)
        if d is not None:
            return d
        n = self.cpu_workers if device.startswith("cpu") else self.gpu_workers_per_device
        d = _Device(ThreadPoolExecutor(max_workers=n, thread_name_prefix=f"pgw-kernel-{device}"), n)
        self._devices[device] = d
        log.debug("batch executor started for %s (%d slot(s))", device, n)
        return d

    async def stop(self, drain: bool = True, timeout: float | None = None) -> None:
        """Stop accepting work, then finish (``drain=True``) or fail queued work, and stop the pools.

        Args:
            drain: Finish everything already submitted (graceful shutdown).
            timeout: Max seconds to wait for the drain; remaining work is failed afterwards.
        """
        self._accepting = False
        for b in list(self._open.values()):
            self._enqueue(b)
        if drain and self._pending_batches and self._idle is not None:
            try:
                await asyncio.wait_for(self._idle.wait(), timeout=timeout)
            except TimeoutError:
                log.warning("batch drain timed out after %.1fs; failing remaining work", timeout or 0.0)
        # fail whatever is left (not drained or timed out)
        stop_err = ServiceUnavailableError("service is shutting down", details={"retry_after_s": 1.0})
        for d in self._devices.values():
            while d.ready:
                b = d.ready.popleft()
                self._fail(b, stop_err)
                self._batch_done()
            for b in list(d.running):  # kernels still running: their callers get an error now
                b.aborted = True
                d.running.discard(b)
                self._queued_rows -= b.live_rows
                for e in b.live:
                    if not e.fut.done():
                        e.fut.set_exception(stop_err)
        for d in self._devices.values():
            d.executor.shutdown(wait=True, cancel_futures=True)
        self._devices.clear()
        self._open.clear()
        log.info("batch scheduler stopped (drain=%s, batches=%d, mean_rows=%.1f)",
                 drain, self.stats.batches, self.stats.mean_batch_rows)

    # ---- submission ----------------------------------------------------------------------
    async def submit(self, fingerprint: str, ring: RingParams, linear: str, x_hat: torch.Tensor) -> torch.Tensor:
        """Queue rows and await their results (``(rows, m)`` int64 in ``[0, 2^K)``)."""
        y, _ = await self.submit_timed(fingerprint, ring, linear, x_hat)
        return y

    async def submit_timed(
        self, fingerprint: str, ring: RingParams, linear: str, x_hat: torch.Tensor, *, compact: bool = False
    ) -> tuple[torch.Tensor, BatchTiming]:
        """Like :meth:`submit`, also returning :class:`BatchTiming`.

        With ``compact=True`` the result may be the kernel's compact form (int32 bit patterns for
        ``K <= 32`` on CUDA, see :meth:`ComputeKernel.run_compact`), which :func:`pgw.core.pack_ring`
        packs directly; otherwise it is always int64 in ``[0, 2^K)``.

        Raises:
            ServiceUnavailableError: not started, shutting down, or queue full.
            ModelNotFoundError / ProtocolError: unknown model or linear.
            ShapeError: ``x_hat`` is not ``(rows, n)`` int64.
        """
        if not self._accepting:
            raise ServiceUnavailableError("compute scheduler is not accepting work", details={"retry_after_s": 1.0})
        kernel: ComputeKernel = self.registry.get(fingerprint).kernel(linear)  # type: ignore[attr-defined]
        if x_hat.ndim != 2 or x_hat.shape[1] != kernel.in_features or x_hat.dtype != torch.int64:
            raise ShapeError(
                f"{linear}: x_hat must be int64 (rows, {kernel.in_features}), got {x_hat.dtype} {tuple(x_hat.shape)}",
                details={"linear": linear},
            )
        rows = int(x_hat.shape[0])
        if rows == 0:
            return torch.zeros((0, kernel.out_features), dtype=torch.int64), BatchTiming()
        if self._queued_rows + rows > self.max_queued_rows:
            raise ServiceUnavailableError(
                "compute queue is full", details={"retry_after_s": 1.0, "queued_rows": self._queued_rows}
            )
        key = (fingerprint, ring.name, linear)
        if rows <= self.inline_max_rows and self.window_s <= 0 and key not in self._open:
            dev = str(kernel.device)
            d = self._devices.get(dev) or self._ensure_device(dev)
            if d.busy == 0 and not d.ready:  # idle device: no batch to join, nothing to wait for
                return self._run_inline(d, kernel, ring, x_hat, rows, compact)
        if rows <= self.max_batch_rows:  # the common case: one chunk, await its future directly
            fut = self._add(key, kernel, ring, x_hat, compact)
            self.stats.submitted_rows += rows
            try:
                return await fut
            except BaseException:
                fut.cancel()  # caller gone or failed: drop our rows if not started yet
                raise
        futs: list[asyncio.Future[tuple[torch.Tensor, BatchTiming]]] = []
        for r0 in range(0, rows, self.max_batch_rows):
            futs.append(self._add(key, kernel, ring, x_hat[r0 : r0 + self.max_batch_rows], compact))
        self.stats.submitted_rows += rows
        try:
            results = await asyncio.gather(*futs)
        except BaseException:
            for f in futs:  # caller gone or failed: drop our not-yet-started rows
                f.cancel()
            raise
        ys = [r[0] for r in results]
        timing = BatchTiming(
            queue_ms=max(r[1].queue_ms for r in results),
            compute_ms=max(r[1].compute_ms for r in results),
            batch_rows=max(r[1].batch_rows for r in results),
        )
        return (ys[0] if len(ys) == 1 else torch.cat(ys, dim=0)), timing

    def _run_inline(self, d: _Device, kernel: ComputeKernel, ring: RingParams, x: torch.Tensor, rows: int,
                    compact: bool) -> tuple[torch.Tensor, BatchTiming]:
        """A one-submission batch on an idle device, run synchronously on the event loop.

        What :meth:`_add` + :meth:`_start` (inline branch) + :meth:`_resolve` do for a batch that
        starts and finishes at once, without the batch, entry and future objects: same stats,
        metrics, error mapping and results. Nothing can join it (it is never open), and no other
        batch can start on the device while it runs (the event loop is busy with it).
        """
        t_enq = time.perf_counter()
        self.stats.submitted_rows += rows
        d.busy += 1
        try:
            t_start = time.perf_counter()
            y, secs = _run_kernel(kernel, (x,), ring)
        except Exception as exc:
            self.stats.failed_batches += 1
            if isinstance(exc, PGWError):
                raise
            log.error("kernel failed for %s (%d rows) on %s", kernel.linear, rows, kernel.device, exc_info=exc)
            raise PGWError("compute kernel failed") from exc
        finally:
            d.busy -= 1
        st = self.stats
        st.batches += 1
        st.batched_rows += rows
        if rows > st.max_batch_rows_seen:
            st.max_batch_rows_seen = rows
        if self.metrics is not None:
            self.metrics.batch(kernel.kind, kernel.backend, rows, secs)
        timing = BatchTiming(queue_ms=(t_start - t_enq) * 1e3, compute_ms=secs * 1e3, batch_rows=rows)
        return (y if compact else from_compact(y, ring.k)), timing

    def _add(
        self, key: tuple[str, str, str], kernel: ComputeKernel, ring: RingParams, x: torch.Tensor,
        compact: bool = False,
    ) -> asyncio.Future[tuple[torch.Tensor, BatchTiming]]:
        loop = self._loop
        assert loop is not None
        rows = int(x.shape[0])
        b = self._open.get(key)
        new = False
        if b is None or b.started or b.rows + rows > self.max_batch_rows:
            if b is not None and not b.queued:
                self._enqueue(b)  # full: send it off now
            b = _Batch(key=key, kernel=kernel, ring=ring, device=str(kernel.device))
            self._open[key] = b
            new = True
        fut: asyncio.Future[tuple[torch.Tensor, BatchTiming]] = loop.create_future()
        b.entries.append(_Entry(x, fut, time.perf_counter(), compact))
        b.rows += rows
        self._queued_rows += rows
        self._queued_items += 1
        self._set_depth()
        if not b.queued and (self.window_s <= 0 or b.rows >= self.max_batch_rows):
            self._enqueue(b)  # starts it right away when the device has a free slot
        elif new and not b.queued:
            b.timer = loop.call_later(self.window_s, self._enqueue, b)
        return fut

    def _enqueue(self, b: _Batch) -> None:
        if b.queued:
            return
        b.queued = True
        if b.timer is not None:
            b.timer.cancel()
            b.timer = None
        d = self._ensure_device(b.device)
        d.ready.append(b)
        self._pending_batches += 1
        if self._idle is not None:
            self._idle.clear()
        self._pump(d)

    def _pump(self, d: _Device) -> None:
        """Start ready batches while the device has free slots (FIFO across keys and tenants)."""
        while d.busy < d.slots and d.ready:
            self._start(d, d.ready.popleft())

    def _batch_done(self) -> None:
        self._pending_batches -= 1
        if self._pending_batches <= 0:
            self._pending_batches = 0
            if self._idle is not None:
                self._idle.set()

    def _set_depth(self) -> None:
        if self.metrics is not None:
            self.metrics.queue_depth(self._queued_items)

    # ---- execution -----------------------------------------------------------------------
    def _start(self, d: _Device, b: _Batch) -> None:
        """Hand ``b`` to the device's thread pool (runs on the event loop, never blocks)."""
        b.started = True
        if self._open.get(b.key) is b:
            del self._open[b.key]
        live: list[_Entry] = []
        for e in b.entries:
            self._queued_items -= 1
            if e.fut.done():  # cancelled by its caller
                n = int(e.x.shape[0])
                self._queued_rows -= n
                self.stats.dropped_rows += n
            else:
                live.append(e)
        b.entries = []
        self._set_depth()
        if not live:
            self._batch_done()
            return
        b.live = live
        b.live_rows = sum(int(e.x.shape[0]) for e in live)
        loop = self._loop
        assert loop is not None
        t_start = time.perf_counter()
        parts = [e.x for e in live]  # concatenated in the worker thread, off the event loop
        if b.live_rows <= self.inline_max_rows and d.busy == 0 and not d.ready:
            # idle device, small batch: run it here (no pool hand-off and wake-up in either direction)
            d.busy += 1
            try:
                out = _run_kernel(b.kernel, parts, b.ring)
            except Exception as exc:
                self._finish(d, b, t_start, None, exc)
            else:
                self._finish(d, b, t_start, out, None)
            return
        try:
            cf = d.executor.submit(_run_kernel, b.kernel, parts, b.ring)
        except RuntimeError:  # pool already shut down
            self._queued_rows -= b.live_rows
            err = ServiceUnavailableError("service is shutting down", details={"retry_after_s": 1.0})
            for e in live:
                if not e.fut.done():
                    e.fut.set_exception(err)
            self._batch_done()
            return
        d.busy += 1
        d.running.add(b)

        def done(f: _CFuture[tuple[torch.Tensor, float]]) -> None:  # runs in the worker thread
            try:
                loop.call_soon_threadsafe(self._finish_future, d, b, t_start, f)
            except RuntimeError:  # event loop already closed (process shutting down)
                pass

        cf.add_done_callback(done)

    def _finish_future(self, d: _Device, b: _Batch, t_start: float, f: _CFuture[tuple[torch.Tensor, float]]) -> None:
        """Pool completion (on the event loop): unwrap the future and :meth:`_finish`."""
        if f.cancelled():
            self._finish(d, b, t_start, None,
                         ServiceUnavailableError("service is shutting down", details={"retry_after_s": 1.0}))
            return
        exc = f.exception()
        self._finish(d, b, t_start, None if exc is not None else f.result(), exc)

    def _finish(self, d: _Device, b: _Batch, t_start: float, out: tuple[torch.Tensor, float] | None,
                exc: BaseException | None) -> None:
        """Kernel finished (on the event loop): resolve the callers, then start the next ready batch.

        Exactly one of ``out`` (``(y, seconds)``) and ``exc`` is set.
        """
        d.busy -= 1
        d.running.discard(b)
        try:
            if not b.aborted:
                self._resolve(b, t_start, out, exc)
        finally:
            self._batch_done()
            self._pump(d)

    def _resolve(self, b: _Batch, t_start: float, out: tuple[torch.Tensor, float] | None,
                 exc: BaseException | None) -> None:
        live, rows = b.live, b.live_rows
        self._queued_rows -= rows
        if exc is not None or out is None:
            exc = exc if exc is not None else PGWError("compute kernel returned no result")
            self.stats.failed_batches += 1
            err = exc if isinstance(exc, PGWError) else PGWError("compute kernel failed")
            if not isinstance(exc, PGWError):
                log.error("kernel failed for %s (%d rows) on %s", b.key[2], rows, b.device, exc_info=exc)
            for e in live:
                if not e.fut.done():
                    e.fut.set_exception(err)
            return
        y, secs = out
        self.stats.batches += 1
        self.stats.batched_rows += rows
        self.stats.max_batch_rows_seen = max(self.stats.max_batch_rows_seen, rows)
        if self.metrics is not None:
            self.metrics.batch(b.kernel.kind, b.kernel.backend, rows, secs)
        compute_ms = secs * 1e3
        r0 = 0
        for e in live:
            n = int(e.x.shape[0])
            if not e.fut.done():
                timing = BatchTiming(queue_ms=(t_start - e.t_enq) * 1e3, compute_ms=compute_ms, batch_rows=rows)
                ye = y[r0 : r0 + n]
                e.fut.set_result((ye if e.compact else from_compact(ye, b.ring.k), timing))
            r0 += n

    def _fail(self, b: _Batch, exc: PGWError) -> None:
        for e in b.entries:
            self._queued_items -= 1
            self._queued_rows -= int(e.x.shape[0])
            if not e.fut.done():
                e.fut.set_exception(exc)
        self._set_depth()

    def devices(self) -> Sequence[str]:
        """Devices with an active thread pool."""
        return list(self._devices)


def _run_kernel(
    kernel: ComputeKernel, parts: Sequence[torch.Tensor], ring: RingParams
) -> tuple[torch.Tensor, float]:
    """Concatenate the batch's row blocks and run the kernel (executes in a worker thread).

    Returns:
        ``(y, seconds)``; ``y`` (:meth:`ComputeKernel.run_compact`) has ``sum(rows)`` rows in the
        order of ``parts``.

    Raises:
        PGWError: shape errors, or a kernel that returned a wrong shape (``internal``).
    """
    t0 = time.perf_counter()
    x = parts[0] if len(parts) == 1 else torch.cat(list(parts), dim=0)
    y = kernel.run_compact(x, ring)
    if tuple(y.shape) != (int(x.shape[0]), kernel.out_features):
        raise PGWError(f"kernel {kernel.linear} returned shape {tuple(y.shape)}")
    return y, time.perf_counter() - t0
