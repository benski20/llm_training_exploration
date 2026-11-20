"""Low-overhead execution of the CUDA ring matmul: CUDA graphs, pinned staging, compact output.

A decode-sized call (1-8 rows) on a 7B layer is bound by launch and copy overheads, not by the
GEMM: the eager offset-byte path (:func:`pgw.core.ringmm.offset_byte_core`) is about a dozen
kernel launches plus padding, a synchronous pageable host-to-device copy and an int64 result.
:class:`DeviceExecutor` removes most of that:

* **CUDA graphs** per ``(linear, K, row bucket)``. A bucket is a padded GEMM row count ``p``
  (default 32, 64, 128); it serves every call with ``rows * K/8 <= p`` (``p = 32`` covers up to
  8 rows at ``K = 32``). Each graph reads a static input view ``(p // (K/8), n)`` and writes a
  static output view; rows past the call's are stale and their outputs are discarded (every
  output row depends only on its own input row). Graphs are captured lazily on first use (or by
  :meth:`pgw.service.ComputeKernel.warmup`), checked once against the eager result, and replayed
  afterwards. Calls that fit no bucket run eagerly. A capture that fails is logged once and that
  ``(linear, K, bucket)`` runs eagerly from then on; a replay that disagrees with the eager result
  disables graphs on the device.
* **Fused ring GEMV** (:meth:`DeviceExecutor.run_gemv`): for ``K <= 32`` and at most
  :data:`pgw.core.triton_gemv.TRITON_GEMV_MAX_ROWS` rows, one Triton kernel computes the product
  with wrapping 32-bit integer arithmetic (no digits, no padding); only the low 32 bits of ``x`` are
  copied in. Used only after :func:`pgw.core.triton_gemv.self_check` passed on the device; a result
  that disagrees with the digit-GEMM path disables it (:meth:`DeviceExecutor.disable_gemv`).
* **Pinned staging** buffers per device and non-blocking copies both ways (one stream sync per call).
* **Compact output**: for ``K <= 32`` the device returns int32 bit patterns (the low 32 bits of
  each ring element; :func:`from_compact` restores int64) which halves the device-to-host bytes;
  :func:`pgw.core.pack_ring` packs them directly.

Serialization invariant: graph replays share per-device static buffers and one CUDA graph memory
pool (graph temporaries are reused across graphs), so at most one call per device may run at a
time. :class:`DeviceExecutor` enforces this with a per-device lock held for the whole call (stage,
replay, copy back, sync). The service's :class:`pgw.service.BatchScheduler` already runs one GPU
batch per device at a time (``gpu_workers_per_device = 1``, inline batches only on an idle
device), so the lock is uncontended there; it also covers other callers (warm-up, health checks,
``gpu_workers_per_device > 1``).

Everything except the actual CUDA capture runs on CPU tensors too (with a ``capturer`` that just
calls the body), which is how the CPU tests exercise buckets, static buffers and fallbacks.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from pgw.core.errors import ConfigError
from pgw.core.ringmm import offset_byte_consts, offset_byte_core, offset_byte_rows
from pgw.core.triton_gemv import TRITON_GEMV_KS, low32, max_rows, ring_gemv, self_check

__all__ = [
    "GRAPH_BUCKETS",
    "Capturer",
    "DeviceExecutor",
    "ExecutorStats",
    "bucket_for",
    "compact_dtype",
    "cuda_graph_capturer",
    "device_executor",
    "eager_capturer",
    "from_compact",
    "to_compact",
    "validate_buckets",
]

log = logging.getLogger(__name__)

#: Default padded GEMM row counts with a captured graph (multiples of 32: cuBLASLt int8 on T4).
GRAPH_BUCKETS: tuple[int, ...] = (32, 64, 128)
#: Largest staging transfer (per direction) that goes through the pinned buffers; larger ones
#: (prefill-sized batches) use plain pageable copies.
PINNED_MAX_BYTES = 64 << 20

#: ``capturer(body) -> replay``: runs ``body`` once eagerly (warm-up), records it, and returns a
#: callable that re-executes the recording. Raises if capture is impossible.
Capturer = Callable[[Callable[[], None]], Callable[[], None]]

GraphKey = tuple[int, int]  # (K, bucket p)


def validate_buckets(buckets: Sequence[int]) -> tuple[int, ...]:
    """Sorted unique buckets; each a positive multiple of 32 and at most 4096.

    Raises:
        ConfigError: invalid bucket.
    """
    out = sorted({int(p) for p in buckets})
    for p in out:
        if p <= 0 or p % 32 or p > 4096:
            raise ConfigError(f"CUDA graph buckets must be multiples of 32 in [32, 4096], got {p}")
    return tuple(out)


def bucket_for(rows: int, k: int, buckets: Sequence[int]) -> int | None:
    """Smallest bucket ``p`` (sorted ``buckets``) with ``rows * K/8 <= p``; ``None`` if none fits."""
    need = rows * (k // 8)
    if rows <= 0:
        return None
    for p in buckets:
        if need <= p:
            return int(p)
    return None


def compact_dtype(k: int) -> torch.dtype:
    """dtype of the compact form: int32 for ``K <= 32``, else int64."""
    return torch.int32 if k <= 32 else torch.int64


def to_compact(y: torch.Tensor, k: int) -> torch.Tensor:
    """int64 ring values in ``[0, 2^K)`` -> compact form (int32 bit patterns for ``K <= 32``).

    For ``K = 32`` the value is first centered into ``[-2^31, 2^31)`` so the int32 cast is exact
    (no reliance on out-of-range integer conversion); its two's-complement bits equal the ring value.
    Device-only ops (graph capturable).
    """
    if k > 32:
        return y
    if k == 32:
        y = ((y + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    return y.to(torch.int32)


def from_compact(y: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of :func:`to_compact`: int64 in ``[0, 2^K)``."""
    if y.dtype == torch.int64:
        return y
    return y.to(torch.int64) & ((1 << k) - 1)


def eager_capturer(body: Callable[[], None]) -> Callable[[], None]:
    """A "capturer" that records nothing: runs ``body`` once and replays by calling it (CPU tests)."""
    body()
    return body


def cuda_graph_capturer(device: torch.device) -> Capturer:
    """Capture ``body`` into a :class:`torch.cuda.CUDAGraph` on ``device`` (one shared memory pool).

    The body is first run eagerly on the capture stream (cuBLAS handle and workspace for that
    stream get created outside the graph pool), then captured with ``thread_local`` error mode so
    unrelated CUDA work in other threads (e.g. the maskgen GEMM) does not invalidate the capture.
    """
    with torch.cuda.device(device):  # pragma: no cover - GPU only
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device)  # type: ignore[no-untyped-call]

    def capture(body: Callable[[], None]) -> Callable[[], None]:  # pragma: no cover - GPU only
        with torch.cuda.device(device):
            cur = torch.cuda.current_stream(device)
            stream.wait_stream(cur)
            with torch.cuda.stream(stream):
                body()
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                g.capture_begin(pool=pool, capture_error_mode="thread_local")
                try:
                    body()
                finally:
                    g.capture_end()
            cur.wait_stream(stream)
        return g.replay

    return capture


@dataclass
class ExecutorStats:
    """Counters (tests, benchmarks, logs)."""

    replays: int = 0
    eager: int = 0
    captures: int = 0
    capture_failures: int = 0
    pinned_copies: int = 0
    gemv: int = 0


@dataclass(eq=False)
class _Graph:
    replay: Callable[[], None]
    x: torch.Tensor  # (bcap, n) int64 static input (view of the device input arena)
    out: torch.Tensor  # (bcap, m) compact static output (view of the device output arena)


@dataclass(eq=False)
class KernelGraphs:
    """Per-linear graph cache: ``(K, p) -> graph`` or ``None`` (capture failed: run eagerly)."""

    graphs: dict[GraphKey, _Graph | None] = field(default_factory=dict)

    def __len__(self) -> int:
        return sum(1 for g in self.graphs.values() if g is not None)


class DeviceExecutor:
    """Runs offset-byte ring matmuls on one device (see the module docstring).

    Args:
        device: The device (``cuda:i``; ``cpu`` only for tests).
        capturer: Graph capturer; default :func:`cuda_graph_capturer` on CUDA, none (eager) on CPU.
        pinned_max_bytes: Largest transfer staged through pinned host buffers.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        capturer: Capturer | None = None,
        pinned_max_bytes: int = PINNED_MAX_BYTES,
    ):
        self.device = torch.device(device)
        self._cuda = self.device.type == "cuda"
        self._capturer = capturer
        self._capturer_ready = capturer is not None
        self.graphs_enabled = True
        self.pinned_max_bytes = int(pinned_max_bytes)
        #: held for the whole of every call (see the serialization invariant above)
        self.lock = threading.Lock()
        self.stats = ExecutorStats()
        self._warned_capture = False
        self._consts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._x_arena: torch.Tensor | None = None
        self._out_arena: torch.Tensor | None = None
        self._reserve_x = 0
        self._reserve_out = 0
        self._h_in: torch.Tensor | None = None
        self._h_out: torch.Tensor | None = None
        #: fused Triton ring GEMV: None = not checked yet, then the self-check verdict (False after a mismatch)
        self._gemv_ok: bool | None = None

    # ---- fused ring GEMV (pgw.core.triton_gemv) ---------------------------------------------
    def gemv_ready(self) -> bool:
        """True if the fused Triton GEMV may serve calls on this device (self-check once, then cached)."""
        if self._gemv_ok is None:
            ok = self_check(self.device)
            with self.lock:
                if self._gemv_ok is None:
                    self._gemv_ok = ok
        return bool(self._gemv_ok)

    def gemv_eligible(self, rows: int, k: int) -> bool:
        """A call of ``rows`` rows in ring ``K`` fits the fused kernel and it passed its self-check."""
        return k in TRITON_GEMV_KS and self.gemv_ready() and 1 <= rows <= max_rows(self.device)

    def disable_gemv(self, reason: str) -> None:
        """Stop using the fused kernel on this device (a result disagreed with the digit-GEMM path)."""
        self._gemv_ok = False
        log.error("fused ring GEMV disabled on %s: %s", self.device, reason)

    def run_gemv(self, x: torch.Tensor, k: int, w: torch.Tensor, n: int, m: int) -> torch.Tensor:
        """Compact ``x @ W^T mod 2^K`` with the fused kernel (one launch); host int32 ``(b, m)``.

        Args:
            x: Host int64 ``(b, n)`` (any int64; only the low 32 bits are sent: ``K <= 32``).
            k: Ring bits (one of :data:`pgw.core.triton_gemv.TRITON_GEMV_KS`).
            w: Padded int8 weight ``(mp, np_)`` on this device.
            n: Inner dimension (``<= np_``).
            m: Output features (``<= mp``).
        """
        with self.lock:
            xd = self._stage_in32(x)
            self.stats.gemv += 1
            return self._stage_out(ring_gemv(xd, w, k, n=n, m=m))

    def _stage_in32(self, x: torch.Tensor) -> torch.Tensor:
        """Copy the low 32 bits of host int64 ``x`` to the device (half the bytes of :meth:`_stage_in`)."""
        b, n = x.shape
        lo = low32(x)
        h = self._host("_h_in", b * n * 4) if self._cuda else None
        if h is None:
            return lo.contiguous().to(self.device, non_blocking=True)
        src = h[: b * n * 4].view(torch.int32).view(b, n)
        src.copy_(lo)
        self.stats.pinned_copies += 1
        return src.to(self.device, non_blocking=True)

    # ---- setup ---------------------------------------------------------------------------
    def reserve(self, in_features: int, out_features: int, buckets: Sequence[int] = GRAPH_BUCKETS) -> None:
        """Size hint for the static device buffers (call for every linear before the first capture).

        With ``K >= 32`` (at least 4 byte planes) a bucket ``p`` holds ``p // 4`` rows; smaller K
        just grows the buffers on first use.
        """
        if not buckets:
            return
        bcap = max(buckets) // 4
        with self.lock:
            self._reserve_x = max(self._reserve_x, bcap * in_features * 8)
            self._reserve_out = max(self._reserve_out, bcap * out_features * 8)

    def _get_capturer(self) -> Capturer | None:
        if not self._capturer_ready:
            self._capturer_ready = True
            if self._cuda:  # pragma: no cover - GPU only
                try:
                    self._capturer = cuda_graph_capturer(self.device)
                except Exception as exc:
                    log.warning("CUDA graphs unavailable on %s (%s: %s); running eagerly",
                                self.device, type(exc).__name__, str(exc)[:200])
                    self.graphs_enabled = False
        return self._capturer

    def _arena(self, name: str, nbytes: int, reserve: int) -> torch.Tensor:
        """A device byte buffer of at least ``nbytes`` (graphs keep views of older buffers alive)."""
        cur: torch.Tensor | None = getattr(self, name)
        if cur is None or cur.numel() < nbytes:
            size = max(nbytes, reserve, 0 if cur is None else 2 * cur.numel())
            cur = torch.zeros(size, dtype=torch.uint8, device=self.device)
            setattr(self, name, cur)
        return cur

    def _host(self, name: str, nbytes: int) -> torch.Tensor | None:
        """Pinned host byte buffer of at least ``nbytes`` (``None`` above :attr:`pinned_max_bytes`)."""
        if nbytes > self.pinned_max_bytes:
            return None
        cur: torch.Tensor | None = getattr(self, name)
        if cur is None or cur.numel() < nbytes:
            size = 1 << max(16, (nbytes - 1).bit_length())
            cur = torch.empty(size, dtype=torch.uint8, pin_memory=self._cuda)
            setattr(self, name, cur)
        return cur

    def _const(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self._consts.get(k)
        if c is None:
            c = offset_byte_consts(k, self.device)
            self._consts[k] = c
        return c

    # ---- staging -------------------------------------------------------------------------
    def _stage_in(self, x: torch.Tensor, dst: torch.Tensor | None) -> torch.Tensor:
        """Copy host ``x`` to the device (into ``dst`` if given); non-blocking via pinned memory."""
        b, n = x.shape
        h = self._host("_h_in", b * n * 8) if self._cuda else None
        src = x
        if h is not None:
            src = h[: b * n * 8].view(torch.int64).view(b, n)
            src.copy_(x)
            self.stats.pinned_copies += 1
        if dst is None:
            return src.to(self.device, non_blocking=True)
        dst.copy_(src, non_blocking=True)
        return dst

    def _stage_out(self, y: torch.Tensor) -> torch.Tensor:
        """Copy device ``y`` to a fresh host tensor (one stream sync)."""
        if not self._cuda:
            return y.clone()
        nbytes = y.numel() * y.element_size()  # pragma: no cover - GPU only
        h = self._host("_h_out", nbytes)  # pragma: no cover - GPU only
        if h is None:  # pragma: no cover - GPU only
            return y.cpu()
        hv = h[:nbytes].view(y.dtype).view(y.shape)  # pragma: no cover - GPU only
        hv.copy_(y, non_blocking=True)  # pragma: no cover - GPU only
        torch.cuda.current_stream(self.device).synchronize()  # pragma: no cover - GPU only
        return hv.clone()  # pragma: no cover - GPU only

    # ---- compute -------------------------------------------------------------------------
    def run(
        self,
        x: torch.Tensor,
        k: int,
        w: torch.Tensor,
        rowsum: torch.Tensor,
        m: int,
        *,
        graphs: KernelGraphs | None,
        buckets: Sequence[int] = GRAPH_BUCKETS,
    ) -> torch.Tensor:
        """Compact ``x @ W^T mod 2^K`` for host int64 ``x (b, n)``; returns a host ``(b, m)`` tensor.

        Args:
            x: Host int64 ``(b, n)`` (any int64; taken mod ``2^K``), ``b >= 1``.
            k: Ring bits (multiple of 8 in [8, 56]).
            w: Padded int8 weight ``(mp, np_)`` on this device (:func:`pgw.core.ringmm.pad_weight_cuda`).
            rowsum: int64 ``(mp,)`` row sums of ``w`` on this device.
            m: Output features.
            graphs: The linear's graph cache, or ``None`` to run eagerly.
            buckets: Sorted graph buckets.

        Returns:
            int32 bit patterns (``K <= 32``) or int64 in ``[0, 2^K)`` (:func:`from_compact`).
        """
        with self.lock:
            p = bucket_for(int(x.shape[0]), k, buckets) if graphs is not None and self.graphs_enabled else None
            if p is not None:
                assert graphs is not None
                key = (k, p)
                g = graphs.graphs.get(key, False)
                if g is False:
                    return self._capture_and_run(x, k, w, rowsum, m, p, graphs)
                if g is not None:
                    assert isinstance(g, _Graph)
                    b = int(x.shape[0])
                    self._stage_in(x, g.x[:b])
                    g.replay()
                    self.stats.replays += 1
                    return self._stage_out(g.out[:b])
            return self._eager(x, k, w, rowsum, m)

    def _eager(self, x: torch.Tensor, k: int, w: torch.Tensor, rowsum: torch.Tensor, m: int) -> torch.Tensor:
        xd = self._stage_in(x, None)
        self.stats.eager += 1
        sh, pm = self._const(k)  # cached: no per-call host-to-device constant copies
        y = offset_byte_core(xd, k, w, rowsum, sh, pm, offset_byte_rows(k, int(x.shape[0])))[:, :m]
        return self._stage_out(to_compact(y, k))

    def _capture_and_run(
        self, x: torch.Tensor, k: int, w: torch.Tensor, rowsum: torch.Tensor, m: int, p: int, graphs: KernelGraphs
    ) -> torch.Tensor:
        b, n = (int(s) for s in x.shape)
        bcap = p // (k // 8)
        dt = compact_dtype(k)
        esize = torch.empty((), dtype=dt).element_size()
        capturer = self._get_capturer()
        if capturer is None or not self.graphs_enabled:
            graphs.graphs[(k, p)] = None
            return self._eager(x, k, w, rowsum, m)
        x_arena = self._arena("_x_arena", bcap * n * 8, self._reserve_x)
        out_arena = self._arena("_out_arena", bcap * m * esize, self._reserve_out)
        gx = x_arena[: bcap * n * 8].view(torch.int64).view(bcap, n)
        gout = out_arena[: bcap * m * esize].view(dt).view(bcap, m)
        sh, pm = self._const(k)

        def body() -> None:
            gout.copy_(to_compact(offset_byte_core(gx, k, w, rowsum, sh, pm, p)[:, :m], k))

        self._stage_in(x, gx[:b])
        try:
            replay = capturer(body)  # runs body once eagerly first: gout holds the eager result
            ref = gout[:b].clone()
            gout.fill_(-0x5A5A5A5A)  # a replay that does nothing cannot pass the check below
            replay()
            ok = bool(torch.equal(gout[:b], ref))
        except Exception as exc:
            self.stats.capture_failures += 1
            graphs.graphs[(k, p)] = None
            lvl = logging.DEBUG if self._warned_capture else logging.WARNING
            self._warned_capture = True
            log.log(lvl, "CUDA graph capture failed on %s (K=%d, rows<=%d, n=%d, m=%d): %s: %s; running eagerly",
                    self.device, k, bcap, n, m, type(exc).__name__, str(exc)[:300])
            return self._eager(x, k, w, rowsum, m)
        if not ok:
            self.graphs_enabled = False
            graphs.graphs[(k, p)] = None
            log.error("CUDA graph replay disagrees with the eager result on %s (K=%d, n=%d, m=%d); "
                      "CUDA graphs disabled on this device", self.device, k, n, m)
            return self._eager(x, k, w, rowsum, m)
        graphs.graphs[(k, p)] = _Graph(replay, gx, gout)
        self.stats.captures += 1
        self.stats.replays += 1
        return self._stage_out(gout[:b])


_EXECUTORS: dict[str, DeviceExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def device_executor(device: torch.device | str) -> DeviceExecutor:
    """The process-wide :class:`DeviceExecutor` of ``device`` (one per device: shared buffers + lock)."""
    key = str(torch.device(device))
    with _EXECUTORS_LOCK:
        ex = _EXECUTORS.get(key)
        if ex is None:
            ex = DeviceExecutor(device)
            _EXECUTORS[key] = ex
        return ex
