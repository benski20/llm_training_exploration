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
* Every device has its own ready queue and worker tasks (``gpu_workers_per_device``, default 1, so
  GPU work is serialized per device; ``cpu_workers`` for the CPU). Kernels run in a per-device
  thread pool, so the event loop never blocks on a GEMM. With multi-GPU layer placement the
  devices work in parallel on different layers.
* Rows whose caller went away (deadline, disconnect) before the batch starts are dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from pgw.core.errors import PGWError, ServiceUnavailableError, ShapeError
from pgw.core.ring import RingParams
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
        self.stats = BatchStats()
        self._open: dict[tuple[str, str, str], _Batch] = {}
        self._queues: dict[str, asyncio.Queue[_Batch]] = {}
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._accepting = False
        self._queued_rows = 0
        self._queued_items = 0
        self._busy = 0

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
        """Start worker tasks (idempotent). Workers for devices are created lazily on first use."""
        self._loop = asyncio.get_running_loop()
        self._accepting = True

    def _ensure_device(self, device: str) -> asyncio.Queue[_Batch]:
        q = self._queues.get(device)
        if q is not None:
            return q
        assert self._loop is not None
        n = self.cpu_workers if device.startswith("cpu") else self.gpu_workers_per_device
        q = asyncio.Queue()
        self._queues[device] = q
        self._executors[device] = ThreadPoolExecutor(max_workers=n, thread_name_prefix=f"pgw-kernel-{device}")
        for i in range(n):
            self._workers.append(self._loop.create_task(self._worker(device, q), name=f"pgw-batch-{device}-{i}"))
        log.debug("batch workers started for %s (%d)", device, n)
        return q

    async def stop(self, drain: bool = True, timeout: float | None = None) -> None:
        """Stop accepting work, then finish (``drain=True``) or fail queued work, and stop workers.

        Args:
            drain: Finish everything already submitted (graceful shutdown).
            timeout: Max seconds to wait for the drain; remaining work is failed afterwards.
        """
        self._accepting = False
        for b in list(self._open.values()):
            self._enqueue(b)
        if drain and self._queues:
            joins = [q.join() for q in self._queues.values()]
            try:
                await asyncio.wait_for(asyncio.gather(*joins), timeout=timeout)
            except TimeoutError:
                log.warning("batch drain timed out after %.1fs; failing remaining work", timeout or 0.0)
        # fail whatever is left (not drained or timed out)
        for q in self._queues.values():
            while not q.empty():
                b = q.get_nowait()
                self._fail(b, ServiceUnavailableError("service is shutting down", details={"retry_after_s": 1.0}))
                q.task_done()
        for t in self._workers:
            t.cancel()
        for t in self._workers:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._workers.clear()
        self._queues.clear()
        for ex in self._executors.values():
            ex.shutdown(wait=True, cancel_futures=True)
        self._executors.clear()
        self._open.clear()
        log.info("batch scheduler stopped (drain=%s, batches=%d, mean_rows=%.1f)",
                 drain, self.stats.batches, self.stats.mean_batch_rows)

    # ---- submission ----------------------------------------------------------------------
    async def submit(self, fingerprint: str, ring: RingParams, linear: str, x_hat: torch.Tensor) -> torch.Tensor:
        """Queue rows and await their results (``(rows, m)`` int64 in ``[0, 2^K)``)."""
        y, _ = await self.submit_timed(fingerprint, ring, linear, x_hat)
        return y

    async def submit_timed(
        self, fingerprint: str, ring: RingParams, linear: str, x_hat: torch.Tensor
    ) -> tuple[torch.Tensor, BatchTiming]:
        """Like :meth:`submit`, also returning :class:`BatchTiming`.

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
        futs: list[asyncio.Future[tuple[torch.Tensor, BatchTiming]]] = []
        for r0 in range(0, rows, self.max_batch_rows):
            futs.append(self._add(key, kernel, ring, x_hat[r0 : r0 + self.max_batch_rows]))
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

    def _add(
        self, key: tuple[str, str, str], kernel: ComputeKernel, ring: RingParams, x: torch.Tensor
    ) -> asyncio.Future[tuple[torch.Tensor, BatchTiming]]:
        loop = self._loop
        assert loop is not None
        rows = int(x.shape[0])
        b = self._open.get(key)
        if b is None or b.started or b.rows + rows > self.max_batch_rows:
            if b is not None and not b.queued:
                self._enqueue(b)  # full: send it off now
            b = _Batch(key=key, kernel=kernel, ring=ring, device=str(kernel.device))
            self._open[key] = b
            if self.window_s <= 0:
                self._enqueue(b)
            else:
                b.timer = loop.call_later(self.window_s, self._enqueue, b)
        fut: asyncio.Future[tuple[torch.Tensor, BatchTiming]] = loop.create_future()
        b.entries.append(_Entry(x, fut, time.perf_counter()))
        b.rows += rows
        self._queued_rows += rows
        self._queued_items += 1
        self._set_depth()
        if b.rows >= self.max_batch_rows and not b.queued:
            self._enqueue(b)
        return fut

    def _enqueue(self, b: _Batch) -> None:
        if b.queued:
            return
        b.queued = True
        if b.timer is not None:
            b.timer.cancel()
            b.timer = None
        self._ensure_device(b.device).put_nowait(b)

    def _set_depth(self) -> None:
        if self.metrics is not None:
            self.metrics.queue_depth(self._queued_items)

    # ---- execution -----------------------------------------------------------------------
    async def _worker(self, device: str, q: asyncio.Queue[_Batch]) -> None:
        loop = asyncio.get_running_loop()
        while True:
            b = await q.get()
            try:
                await self._run_batch(loop, device, b)
            except Exception:  # pragma: no cover - _run_batch already routes errors to futures
                log.exception("batch worker error")
            finally:
                q.task_done()

    async def _run_batch(self, loop: asyncio.AbstractEventLoop, device: str, b: _Batch) -> None:
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
        self._set_depth()
        if not live:
            return
        rows = sum(int(e.x.shape[0]) for e in live)
        t_start = time.perf_counter()
        try:
            parts = [e.x for e in live]  # concatenated in the worker thread, off the event loop
            y, secs = await loop.run_in_executor(self._executors[device], _run_kernel, b.kernel, parts, b.ring)
        except asyncio.CancelledError:  # scheduler stopped without drain while this batch ran
            self._queued_rows -= rows
            stop_err = ServiceUnavailableError("service is shutting down", details={"retry_after_s": 1.0})
            for e in live:
                if not e.fut.done():
                    e.fut.set_exception(stop_err)
            raise
        except Exception as exc:
            self.stats.failed_batches += 1
            self._queued_rows -= rows
            err = exc if isinstance(exc, PGWError) else PGWError("compute kernel failed")
            if not isinstance(exc, PGWError):
                log.error("kernel failed for %s (%d rows) on %s", b.key[2], rows, device, exc_info=exc)
            for e in live:
                if not e.fut.done():
                    e.fut.set_exception(err)
            return
        self._queued_rows -= rows
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
                e.fut.set_result((y[r0 : r0 + n], timing))
            r0 += n

    def _fail(self, b: _Batch, exc: PGWError) -> None:
        for e in b.entries:
            self._queued_items -= 1
            self._queued_rows -= int(e.x.shape[0])
            if not e.fut.done():
                e.fut.set_exception(exc)
        self._set_depth()

    def devices(self) -> Sequence[str]:
        """Devices with active workers."""
        return list(self._queues)


def _run_kernel(
    kernel: ComputeKernel, parts: Sequence[torch.Tensor], ring: RingParams
) -> tuple[torch.Tensor, float]:
    """Concatenate the batch's row blocks and run the kernel (executes in a worker thread).

    Returns:
        ``(y, seconds)``; ``y`` has ``sum(rows)`` rows in the order of ``parts``.

    Raises:
        PGWError: shape errors, or a kernel that returned a wrong shape (``internal``).
    """
    t0 = time.perf_counter()
    x = parts[0] if len(parts) == 1 else torch.cat(list(parts), dim=0)
    y = kernel.run(x, ring)
    if tuple(y.shape) != (int(x.shape[0]), kernel.out_features):
        raise PGWError(f"kernel {kernel.linear} returned shape {tuple(y.shape)}")
    return y, time.perf_counter() - t0
