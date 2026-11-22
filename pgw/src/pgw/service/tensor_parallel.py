"""Output-row tensor parallel serving (scaling plan, provider target 4; opt-in, default off).

A PGW trip is one linear whose output goes straight back to the gateway (the nonlinear steps are the
gateway's), so a linear too large for one GPU is split **by output rows**: shard ``j`` of an
``(m, n)`` weight holds rows ``[m0_j, m1_j)`` on device ``j``. A trip's masked input ``x_hat`` is
broadcast to every shard, each computes ``x_hat @ W_j^T mod 2^K``, and the slices are gathered in
shard order into the reply. Every output element depends only on its own weight row and the input
row, and the arithmetic is exact modulo ``2^K``, so the reply is **byte-identical** to the single
device's: no reduction, no all-reduce, nothing about the computation or what the provider sees
changes (the provider still sees only the masked ``x_hat`` and returns ``y_hat``). The pack's
fingerprint is unchanged: shards are derived at load.

:class:`ShardedKernel` is a drop-in for :class:`pgw.service.kernels.ComputeKernel` (same metadata,
``run``/``run_compact``, maskgen entry points, ``warmup``). Shards run concurrently: the caller's
thread runs shard 0 and a small per-group pool runs the others (each shard's
:class:`~pgw.service.gpu_exec.DeviceExecutor` uses its own device, stream, pinned staging, lock and
CUDA graphs, so the shards overlap; torch releases the GIL in the GEMMs and copies). The group is
scheduled as **one** unit: :attr:`ShardedKernel.device` is the group's first device, so the
:class:`pgw.service.BatchScheduler` serializes trips per TP group exactly as it does per GPU.

Failure isolation: a call returns only when every shard finished; if any shard fails, the call
raises (no partial reply is ever assembled) and the next call is unaffected.

The fake multi-device backend for CPU tests is simply ``devices = ["cpu", "cpu", ...]`` (several CPU
shards in one process); duplicate CUDA devices are refused.

Configuration: ``ServiceSettings.tp_devices`` (``PGW_SERVICE_TP_DEVICES='["cuda:0","cuda:1",...]'``),
empty by default (no tensor parallelism).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, PGWError, ShapeError
from pgw.core.ring import RingParams
from pgw.core.types import LinearSpec
from pgw.service.gpu_exec import GRAPH_BUCKETS, from_compact
from pgw.service.kernels import ComputeKernel, resolve_device

__all__ = ["ShardedKernel", "TPGroup", "shard_bounds", "validate_tp_devices"]

log = logging.getLogger(__name__)

#: Shard boundaries are multiples of this when every shard can still get rows (padding friendly).
SHARD_ALIGN = 16


def shard_bounds(m: int, shards: int, align: int = SHARD_ALIGN) -> list[tuple[int, int]]:
    """Contiguous, balanced ``[m0, m1)`` output-row ranges covering ``0..m`` in order.

    Boundaries are multiples of ``align`` when ``m >= shards * align`` (else plain balanced splits).

    Raises:
        ConfigError: ``shards < 1`` or ``m < shards`` (a shard would be empty).
    """
    if shards < 1:
        raise ConfigError("tensor parallel needs at least one shard")
    if m < shards:
        raise ConfigError(f"cannot split {m} output rows over {shards} shards")
    a = align if align > 0 and m >= shards * align else 1
    units = -(-m // a)
    cuts = [min(m, (units * j // shards) * a) for j in range(shards)] + [m]
    out = [(cuts[j], cuts[j + 1]) for j in range(shards)]
    if any(m1 <= m0 for m0, m1 in out):  # cannot happen with the rules above; keep the invariant explicit
        raise ConfigError(f"empty shard splitting {m} rows over {shards}")
    return out


def validate_tp_devices(devices: Sequence[str | torch.device]) -> list[torch.device]:
    """Resolved TP devices: all CUDA or all CPU; a CUDA device at most once (CPU may repeat: the fake
    multi-device backend of the tests).

    Raises:
        ConfigError: mixed device types or a repeated CUDA device.
    """
    devs = [resolve_device(d) for d in devices]
    if not devs:
        raise ConfigError("tensor parallel needs at least one device")
    if len({d.type for d in devs}) != 1:
        raise ConfigError(f"tensor parallel devices must all be CUDA or all CPU, got {[str(d) for d in devs]}")
    cuda = [str(d) for d in devs if d.type == "cuda"]
    if len(cuda) != len(set(cuda)):
        raise ConfigError(f"a CUDA device appears twice in tp_devices: {cuda}")
    return devs


class TPGroup:
    """The devices of one tensor-parallel group and the helper threads that drive shards 1..N-1.

    One group is shared by every :class:`ShardedKernel` of a model (``N - 1`` threads in all).
    """

    def __init__(self, devices: Sequence[str | torch.device]):
        self.devices = validate_tp_devices(devices)
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Number of shards."""
        return len(self.devices)

    def _get_pool(self) -> ThreadPoolExecutor | None:
        if self.size == 1:
            return None
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.size - 1, thread_name_prefix="pgw-tp")
            return self._pool

    def map(self, fns: Sequence[Callable[[], Any]]) -> list[Any]:
        """Run ``fns[0]`` on the calling thread and the others on the pool; wait for **all** of them
        (also when one fails), then return their results in order or raise the first shard's error."""
        pool = self._get_pool()
        if pool is None or len(fns) == 1:
            return [f() for f in fns]
        futs: list[Future[Any]] = [pool.submit(f) for f in fns[1:]]
        first: BaseException | None = None
        res0: Any = None
        try:
            res0 = fns[0]()
        except BaseException as exc:
            first = exc
        out = [res0]
        for f in futs:
            try:
                out.append(f.result())
            except BaseException as exc:
                if first is None:
                    first = exc
                out.append(None)
        if first is not None:
            raise first
        return out

    def close(self) -> None:
        """Stop the helper threads (after in-flight calls)."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)


class ShardedKernel:
    """Exact ring matmul for one linear split by output rows over a :class:`TPGroup`.

    Args:
        spec: Linear spec (the whole linear).
        w8: int8 ``(m, n)`` weight (any device; each shard copies its rows to its device).
        group: The TP group (or a device list, which makes a private group).
        backend, cache_float, cuda_graphs, graph_buckets, triton_gemv: As for
            :class:`~pgw.service.kernels.ComputeKernel` (per shard).
        align: Shard boundary alignment (:func:`shard_bounds`).
    """

    def __init__(
        self,
        spec: LinearSpec,
        w8: torch.Tensor,
        group: TPGroup | Sequence[str | torch.device],
        backend: str = "auto",
        *,
        cache_float: bool = True,
        cuda_graphs: bool = True,
        graph_buckets: tuple[int, ...] | list[int] = GRAPH_BUCKETS,
        triton_gemv: bool = True,
        align: int = SHARD_ALIGN,
    ):
        if w8.dtype != torch.int8 or w8.ndim != 2:
            raise ShapeError(f"{spec.id}: weight must be 2-D int8, got {w8.dtype} {tuple(w8.shape)}")
        if tuple(w8.shape) != (spec.out_features, spec.in_features):
            raise ShapeError(f"{spec.id}: weight shape {tuple(w8.shape)} does not match spec "
                             f"({spec.out_features}, {spec.in_features})")
        self.group = group if isinstance(group, TPGroup) else TPGroup(group)
        self.spec = spec
        self.bounds = shard_bounds(spec.out_features, self.group.size, align)
        self.shards: list[ComputeKernel] = []
        for dev, (m0, m1) in zip(self.group.devices, self.bounds, strict=True):
            sspec = replace(spec, out_features=m1 - m0)
            self.shards.append(ComputeKernel(
                sspec, w8[m0:m1], device=dev, backend=backend, cache_float=cache_float, cuda_graphs=cuda_graphs,
                graph_buckets=graph_buckets, triton_gemv=triton_gemv,
            ))
        self.backend = self.shards[0].backend
        #: test hook: ``fault(shard_index)`` runs before each shard's compute (failure-isolation tests)
        self.fault: Callable[[int], None] | None = None

    # ---- metadata (ComputeKernel interface) ----------------------------------------------
    @property
    def linear(self) -> str:
        """Canonical linear id."""
        return self.spec.id.key

    @property
    def kind(self) -> str:
        """Linear kind (metrics label)."""
        return self.spec.id.kind.value

    @property
    def out_features(self) -> int:
        """``m`` of the whole linear."""
        return self.spec.out_features

    @property
    def in_features(self) -> int:
        """``n``."""
        return self.spec.in_features

    @property
    def device(self) -> torch.device:
        """The group's first device: the scheduling key (a TP group runs one trip at a time)."""
        return self.group.devices[0]

    @property
    def devices(self) -> list[torch.device]:
        """Shard devices in shard order."""
        return list(self.group.devices)

    @property
    def tp(self) -> int:
        """Number of shards."""
        return len(self.shards)

    @property
    def device_bytes(self) -> int:
        """Device memory of all shards."""
        return sum(s.device_bytes for s in self.shards)

    @property
    def triton_gemv(self) -> bool:
        """True if every shard serves eligible calls with the fused Triton GEMV."""
        return all(s.triton_gemv for s in self.shards)

    @property
    def cuda_graphs(self) -> int:
        """CUDA graphs captured over all shards."""
        return sum(s.cuda_graphs for s in self.shards)

    # ---- compute -------------------------------------------------------------------------
    def _check_input(self, x_hat: torch.Tensor) -> None:
        if x_hat.ndim != 2 or x_hat.shape[1] != self.in_features:
            raise ShapeError(f"{self.linear}: x_hat must be (B, {self.in_features}), got {tuple(x_hat.shape)}",
                             details={"linear": self.linear})
        if x_hat.dtype != torch.int64:
            raise ShapeError(f"{self.linear}: x_hat must be int64, got {x_hat.dtype}", details={"linear": self.linear})

    def _each(self, fn: Callable[[int, ComputeKernel], Any]) -> list[Any]:
        def job(j: int) -> Callable[[], Any]:
            def run() -> Any:
                if self.fault is not None:
                    self.fault(j)
                return fn(j, self.shards[j])
            return run

        try:
            return self.group.map([job(j) for j in range(len(self.shards))])
        except PGWError:
            raise
        except Exception as exc:
            log.error("tensor-parallel shard failed for %s: %s: %s", self.linear, type(exc).__name__, str(exc)[:300])
            raise PGWError(f"{self.linear}: a tensor-parallel shard failed") from exc

    def run_compact(self, x_hat: torch.Tensor, ring: RingParams) -> torch.Tensor:
        """``(B, n)`` int64 -> ``(B, m)``: the shards' compact results gathered in row order (int32 bit
        patterns when every shard returned them, else int64 in ``[0, 2^K)``)."""
        self._check_input(x_hat)
        if x_hat.shape[0] == 0:
            return torch.zeros((0, self.out_features), dtype=torch.int64)
        x = x_hat if x_hat.device.type == "cpu" else x_hat.cpu()
        parts = self._each(lambda _j, s: s.run_compact(x, ring))
        if len({p.dtype for p in parts}) != 1:
            parts = [from_compact(p, ring.k) for p in parts]
        return torch.cat(parts, dim=1)

    def run(self, x_hat: torch.Tensor, ring: RingParams) -> torch.Tensor:
        """``(B, n)`` int64 -> ``(B, m)`` int64 in ``[0, 2^K)`` (input and output on CPU)."""
        return from_compact(self.run_compact(x_hat, ring), ring.k)

    __call__ = run

    # ---- encrypted mask generation (row blocks of W8 b) --------------------------------------
    def mulmod64(self, b: np.ndarray, nbytes: int = 8) -> np.ndarray:
        """``W8 b mod 2^64``: each shard's ``(m_j, cols)`` rows stacked in order."""
        parts = self._each(lambda _j, s: s.mulmod64(b, nbytes))
        return np.concatenate(parts, axis=0)

    def mulmod64_wire(self, data: bytes | bytearray | memoryview, cols: int, q_bits: int = 64,
                      drop_bits: int = 0) -> bytes:
        """Wire to wire (:meth:`ComputeKernel.mulmod64_wire`): the reply is whole bytes of each
        row-major output word, so the shards' replies concatenate in row order."""
        buf = bytes(data)
        parts = self._each(lambda _j, s: s.mulmod64_wire(buf, cols, q_bits, drop_bits))
        return b"".join(bytes(p) for p in parts)

    def synchronize(self) -> None:
        """Wait for queued work on every shard's device."""
        for s in self.shards:
            s.synchronize()

    def warmup(self, ring: RingParams, rows: int = 1, *, all_buckets: bool = False) -> float:
        """Warm every shard (captures its graphs on CUDA); returns seconds taken."""
        t0 = time.perf_counter()
        self._each(lambda _j, s: s.warmup(ring, rows, all_buckets=all_buckets))
        return time.perf_counter() - t0

    def __repr__(self) -> str:
        return (f"ShardedKernel({self.linear}, m={self.out_features}, n={self.in_features}, "
                f"tp={self.tp}, devices={[str(d) for d in self.devices]}, bounds={self.bounds})")
