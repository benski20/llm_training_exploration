"""Compute kernels: exact ``W8 x_hat mod 2^K`` for one linear on the service device.

Backends (all exact modulo ``2^K``; see SPEC.md section 5):

``cuda_int8``
    ``K/8`` int8 tensor-core GEMMs (``torch._int_mm``) on balanced base-256 digits of ``x_hat``,
    int32 accumulation, inner dimension chunked so no partial sum overflows. Rows are padded to
    at least 32 and to a multiple of 8, ``n`` and ``m`` to multiples of 8 (cuBLASLt constraints).
    Implemented by :class:`pgw.core.PreparedWeight`.
``cpu_int8``
    The same digit algorithm on CPU via ``torch._int_mm`` (oneDNN/fbgemm, VNNI/AMX where present).
    Selected only after an exactness probe with adversarial operands passes
    (:func:`pgw.core.ringmm.cpu_int8_available`), because some int8 CPU kernels saturate int16
    intermediates. The digit GEMM is :func:`pgw.core.ringmm.int8_ring_matmul` (shared with the
    engine and the factory; CPU inner chunks of ``INT_MM_MAX_INNER_CPU``).
``cpu_f64`` / ``cpu_f32``
    Float GEMMs on wide balanced digits (:class:`pgw.core.PreparedWeight`). Always exact.
``auto``
    ``cuda_int8`` on CUDA devices; on CPU ``cpu_int8`` if the probe passes, else ``cpu_f64``.

On CUDA, :meth:`ComputeKernel.run` goes through the device's
:class:`pgw.service.gpu_exec.DeviceExecutor`: CUDA graphs per ``(linear, K, row bucket)`` for
decode-sized calls, pinned staging and non-blocking copies, and (:meth:`ComputeKernel.run_compact`)
32-bit results for ``K <= 32``. Results are bit-identical to the eager path.

Decode-sized CUDA calls (``K <= 32``, at most :func:`pgw.core.triton_gemv.max_rows` rows) use the
fused Triton ring GEMV (:mod:`pgw.core.triton_gemv`: one kernel, int8 weight streamed once,
wrapping 32-bit integer arithmetic) when ``triton_gemv`` is on and its self-check passed on the
device; the first call per ``(K, rows)`` of each linear is also compared with the digit-GEMM path,
and any disagreement switches the device back to it. The launch configuration comes from the
device's table (measured, each rule self-checked) or the heuristic; 2-8 rows use the memory-bound
``stream`` variant where it is enabled (``PGW_SERVICE_TRITON_GEMV_MULTIROW``: on GPUs without the
tensor-core variant, such as a T4, after a startup timing gate). Everything else is unchanged.

Measured: Llama-3-8B, T4, K=48: 0.45 s/token at batch 1, 0.014 s/token amortized at batch 32.
Future: fused CUDA/Triton kernel doing digit split + K/8 GEMMs + shift-accumulate in one pass.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams
from pgw.core.ringmm import (
    PreparedWeight,
    cpu_int8_available,
    cpu_int8_fast,
    cuda_int8_available,
    int8_ring_matmul,
)
from pgw.core.types import LinearSpec
from pgw.service.gpu_exec import (
    GRAPH_BUCKETS,
    DeviceExecutor,
    KernelGraphs,
    device_executor,
    from_compact,
    validate_buckets,
)

__all__ = [
    "SERVICE_BACKENDS",
    "ComputeKernel",
    "ServiceBackend",
    "cpu_int8_available",
    "int8_ring_matmul",
    "resolve_backend",
    "resolve_device",
]

log = logging.getLogger(__name__)

ServiceBackend = Literal["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"]
SERVICE_BACKENDS: tuple[str, ...] = ("auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32")

def resolve_device(device: str | torch.device) -> torch.device:
    """``"auto"`` -> ``cuda`` (device 0) when available, else ``cpu``. Validates explicit devices.

    Raises:
        ConfigError: malformed or unsupported device string.
        BackendUnavailableError: a CUDA device is requested but not present.
    """
    if isinstance(device, str) and device.strip().lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ConfigError(f"invalid device {device!r}") from exc
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise BackendUnavailableError(f"device {device!s} requested but CUDA is not available")
        idx = 0 if dev.index is None else dev.index
        if idx >= torch.cuda.device_count():
            raise BackendUnavailableError(f"device {device!s} requested but only {torch.cuda.device_count()} GPU(s)")
        return torch.device("cuda", idx)
    if dev.type != "cpu":
        raise ConfigError(f"unsupported device type {dev.type!r}; use cpu or cuda[:i]")
    return dev


def resolve_backend(device: torch.device, backend: str = "auto") -> str:
    """Concrete backend for ``device``.

    Raises:
        ConfigError: unknown backend name.
        BackendUnavailableError: backend incompatible with the device or unavailable on this host.
    """
    if backend not in SERVICE_BACKENDS:
        raise ConfigError(f"unknown service backend {backend!r}; choose one of {SERVICE_BACKENDS}")
    if backend == "auto":
        if device.type == "cuda":
            return "cuda_int8"
        return "cpu_int8" if cpu_int8_fast() else "cpu_f64"
    if backend == "cuda_int8":
        if device.type != "cuda":
            raise BackendUnavailableError("cuda_int8 needs a CUDA device")
        if not cuda_int8_available():  # pragma: no cover - GPU only
            raise BackendUnavailableError("torch._int_mm does not work on this CUDA device")
        return backend
    if device.type != "cpu":
        raise BackendUnavailableError(f"{backend} needs device 'cpu'")
    if backend == "cpu_int8" and not cpu_int8_available():
        raise BackendUnavailableError("cpu_int8 requested but CPU torch._int_mm is unavailable or inexact")
    return backend


class ComputeKernel:
    """Exact ring matmul for one public int8 linear weight resident on one device.

    Args:
        spec: Linear spec (``id``, ``out_features m``, ``in_features n``).
        w8: int8 ``(m, n)`` weight (any device; copied to ``device``).
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:i"``.
        backend: See module docstring.
        cache_float: CPU float backends keep a float copy of the weight (faster, 4-8x memory).
        cuda_graphs: On CUDA, replay captured CUDA graphs for calls that fit a row bucket
            (``rows * K/8 <= p``); other calls run eagerly. Ignored on CPU.
        graph_buckets: Padded GEMM row counts that get a graph (multiples of 32).
        triton_gemv: On CUDA, serve small ``K <= 32`` calls with the fused Triton kernel once its
            device self-check passed (run here, once per device). Ignored on CPU.

    Thread-safety: :meth:`run` may be called from several threads (weights are read-only). On CUDA
    every call holds its device executor's lock (graphs share per-device static buffers), and the
    service also serializes GPU work per device through :class:`pgw.service.BatchScheduler`.
    """

    def __init__(
        self,
        spec: LinearSpec,
        w8: torch.Tensor,
        device: str | torch.device = "auto",
        backend: str = "auto",
        *,
        cache_float: bool = True,
        cuda_graphs: bool = True,
        graph_buckets: tuple[int, ...] | list[int] = GRAPH_BUCKETS,
        triton_gemv: bool = True,
    ):
        if w8.dtype != torch.int8 or w8.ndim != 2:
            raise ShapeError(f"{spec.id}: weight must be 2-D int8, got {w8.dtype} {tuple(w8.shape)}")
        if tuple(w8.shape) != (spec.out_features, spec.in_features):
            raise ShapeError(
                f"{spec.id}: weight shape {tuple(w8.shape)} does not match spec "
                f"({spec.out_features}, {spec.in_features})"
            )
        self.spec = spec
        self.device = resolve_device(device)
        self.backend = resolve_backend(self.device, backend)
        self._pw: PreparedWeight | None = None
        self._w: torch.Tensor | None = None
        self._exec: DeviceExecutor | None = None
        self._graphs: KernelGraphs | None = None
        self._buckets = validate_buckets(graph_buckets)
        self._gemv = False
        self._gemv_checked: set[tuple[int, int]] = set()
        # Own the memory: a pack's tensors may be memory-mapped views of files.
        wd = w8.detach().to(self.device)
        if wd.data_ptr() == w8.data_ptr():
            wd = wd.clone()
        wd = wd.contiguous()
        if self.backend == "cpu_int8":
            self._w = wd
        else:
            self._pw = PreparedWeight(
                wd, device=self.device, backend=self.backend,  # type: ignore[arg-type]
                cache_float=cache_float,
            )
        if self.backend == "cuda_int8":  # pragma: no cover - GPU only
            self._exec = device_executor(self.device)
            self._exec.reserve(self.in_features, self.out_features, self._buckets)
            self._graphs = KernelGraphs() if cuda_graphs else None
            self._gemv = bool(triton_gemv) and self._exec.gemv_ready()  # startup self-check (once per device)

    # ---- metadata ------------------------------------------------------------------------
    @property
    def linear(self) -> str:
        """Canonical linear id (``"3.qkv"``)."""
        return self.spec.id.key

    @property
    def kind(self) -> str:
        """Linear kind (``qkv``, ``o``, ``gate_up``, ``down``); used as a metrics label."""
        return self.spec.id.kind.value

    @property
    def out_features(self) -> int:
        """``m``."""
        return self.spec.out_features

    @property
    def in_features(self) -> int:
        """``n``."""
        return self.spec.in_features

    @property
    def device_bytes(self) -> int:
        """Device memory held by this kernel (weights and cached float copies)."""
        if self._w is not None:
            return int(self._w.numel() * self._w.element_size())
        assert self._pw is not None
        total = 0
        for t in (self._pw._w_dev, self._pw._w_float):
            if t is not None:
                total += int(t.numel() * t.element_size())
        return total

    # ---- compute -------------------------------------------------------------------------
    def run(self, x_hat: torch.Tensor, ring: RingParams) -> torch.Tensor:
        """``(B, n)`` int64 -> ``(B, m)`` int64 in ``[0, 2^K)``; input and output on CPU.

        Raises:
            ShapeError: wrong shape or dtype.
        """
        self._check_input(x_hat)
        if x_hat.shape[0] == 0:
            return torch.zeros((0, self.out_features), dtype=torch.int64)
        if self._exec is not None and self._exec_ok(ring.k):  # pragma: no cover - GPU only
            return from_compact(self._run_exec(x_hat, ring.k), ring.k)
        if self._w is not None:
            out = int8_ring_matmul(x_hat.to(self.device), self._w, ring.k)
        else:
            assert self._pw is not None
            xin = x_hat.to(self.device, non_blocking=True) if self.device.type == "cuda" else x_hat
            out = self._pw.matmul(xin, ring)
        return out.cpu() if out.device.type != "cpu" else out

    def run_compact(self, x_hat: torch.Tensor, ring: RingParams) -> torch.Tensor:
        """Like :meth:`run`, but for ``K <= 32`` on CUDA the result may be int32 bit patterns.

        They are the low 32 bits of each ring element (half the device-to-host bytes);
        :func:`pgw.core.pack_ring` packs either form and :func:`pgw.service.gpu_exec.from_compact`
        restores int64. CPU backends return :meth:`run`'s int64 result.
        """
        if self._exec is not None and self._exec_ok(ring.k):  # pragma: no cover - GPU only
            self._check_input(x_hat)
            if x_hat.shape[0] == 0:
                return torch.zeros((0, self.out_features), dtype=torch.int64)
            return self._run_exec(x_hat, ring.k)
        return self.run(x_hat, ring)

    def _check_input(self, x_hat: torch.Tensor) -> None:
        if x_hat.ndim != 2 or x_hat.shape[1] != self.in_features:
            raise ShapeError(
                f"{self.linear}: x_hat must be (B, {self.in_features}), got {tuple(x_hat.shape)}",
                details={"linear": self.linear},
            )
        if x_hat.dtype != torch.int64:
            raise ShapeError(f"{self.linear}: x_hat must be int64, got {x_hat.dtype}", details={"linear": self.linear})

    @staticmethod
    def _exec_ok(k: int) -> bool:
        return k % 8 == 0 and 8 <= k <= 56

    def _run_exec(self, x_hat: torch.Tensor, k: int) -> torch.Tensor:  # pragma: no cover - GPU only
        assert self._exec is not None and self._pw is not None and self._pw._w_dev is not None
        x = x_hat if x_hat.device.type == "cpu" else x_hat.cpu()
        rows = int(x.shape[0])
        if self._gemv and self._exec.gemv_eligible(rows, k):
            y = self._exec.run_gemv(x, k, self._pw._w_dev, self.in_features, self.out_features)
            if (k, rows) in self._gemv_checked:
                return y
            ref = self._exec.run(x, k, self._pw._w_dev, self._pw._rowsum(), self.out_features, graphs=None)
            if torch.equal(from_compact(y, k), from_compact(ref, k)):
                self._gemv_checked.add((k, rows))
                return y
            self._exec.disable_gemv(f"{self.linear} K={k} rows={rows}: result differs from the digit-GEMM path")
            return ref
        return self._exec.run(x, k, self._pw._w_dev, self._pw._rowsum(), self.out_features,
                              graphs=self._graphs, buckets=self._buckets)

    @property
    def triton_gemv(self) -> bool:
        """True if this kernel serves eligible calls with the fused Triton GEMV (CUDA, self-check passed)."""
        return self._gemv and self._exec is not None and self._exec.gemv_ready()

    @property
    def cuda_graphs(self) -> int:
        """Number of CUDA graphs captured for this linear (0 on CPU or with graphs off)."""
        return len(self._graphs) if self._graphs is not None else 0

    __call__ = run

    def mulmod64(self, b: np.ndarray, nbytes: int = 8) -> np.ndarray:
        """``W8 b mod 2^64`` for MLWE mask ciphertexts ``b`` (``(n, cols)`` uint64 -> ``(m, cols)``).

        Used by ``POST /pgw/v1/maskgen`` (:mod:`pgw.hemask`). On CUDA, ``b`` is split into eight
        unsigned byte planes, offset to signed int8, multiplied with int8 tensor-core GEMMs
        (``|partial| <= 127 * 128 * n < 2^31``) and recombined with wrapping int64 shifts. With
        ``nbytes < 8`` (coefficients below ``2^(8 nbytes)``, e.g. q = 2^56) the zero top planes are skipped.
        """
        if not 1 <= nbytes <= 8:
            raise ShapeError("nbytes must be in [1, 8]")
        if b.ndim != 2 or b.shape[0] != self.in_features or b.dtype != np.uint64:
            raise ShapeError(f"{self.linear}: maskgen operand must be ({self.in_features}, cols) uint64, "
                             f"got {b.dtype} {tuple(b.shape)}")
        w8 = self._w if self._w is not None else self._pw.w8  # type: ignore[union-attr]
        if self.device.type != "cuda":
            from pgw.hemask.fastmm import engine, small_mm_big
            from pgw.hemask.lwe import matmul_mod64

            if engine() == "cpu_f64":
                return matmul_mod64(w8.cpu().numpy(), b, fast=False)
            return small_mm_big(w8.cpu().numpy(), b, nbytes=nbytes)
        return self._mulmod64_cuda(b, nbytes)  # pragma: no cover - GPU only

    def _mulmod64_cuda(self, b: np.ndarray, nbytes: int = 8) -> np.ndarray:  # pragma: no cover - GPU only
        assert self._pw is not None and self._pw._w_dev is not None
        wpad = self._pw._w_dev  # (mp, np_) int8, zero padded to multiples of 32
        n, cols = b.shape
        m = self.out_features
        mp, np_ = wpad.shape
        colp = -(-cols // 32) * 32
        bt = torch.from_numpy(np.ascontiguousarray(b).view(np.int64)).to(self.device)
        rowsum = wpad.to(torch.int64).sum(1, keepdim=True)  # for the -128 digit offset
        acc = torch.zeros((mp, colp), dtype=torch.int64, device=self.device)
        plane = torch.zeros((colp, np_), dtype=torch.int8, device=self.device)
        for d in range(nbytes):  # coefficients < 2^(8 nbytes): the top planes are zero
            byte = (bt >> (8 * d)) & 0xFF  # arithmetic shift is fine: the mask keeps the byte
            plane[:cols, :n] = (byte - 128).to(torch.int8).t()
            part = torch._int_mm(wpad, plane.t()).to(torch.int64) + 128 * rowsum
            acc += part << (8 * d)  # int64 two's-complement wrap == mod 2^64
        out = acc[:m, :cols].contiguous().cpu().numpy().view(np.uint64)
        return np.asarray(out)

    def synchronize(self) -> None:
        """Wait for queued device work (no-op on CPU)."""
        if self.device.type == "cuda":  # pragma: no cover - GPU only
            torch.cuda.synchronize(self.device)

    def warmup(self, ring: RingParams, rows: int = 1, *, all_buckets: bool = False) -> float:
        """Run one dummy product (allocator / cuBLAS warm-up); returns seconds taken.

        On CUDA this captures the graph of ``rows``'s bucket; ``all_buckets`` captures every bucket.
        """
        t0 = time.perf_counter()
        sizes = [rows]
        if all_buckets and self._graphs is not None:
            nb = ring.k // 8
            sizes += [p // nb for p in self._buckets if p // nb > 0]
        for r in dict.fromkeys(sizes):
            self.run(torch.zeros((r, self.in_features), dtype=torch.int64), ring)
        self.synchronize()
        return time.perf_counter() - t0

    def __repr__(self) -> str:
        return (
            f"ComputeKernel({self.linear}, m={self.out_features}, n={self.in_features}, "
            f"device={self.device}, backend={self.backend})"
        )
