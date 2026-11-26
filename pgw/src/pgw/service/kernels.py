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
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import torch

from pgw.core.errors import BackendUnavailableError, ConfigError, ShapeError
from pgw.core.ring import RingParams
from pgw.core.ringmm import (
    INT_MM_MAX_INNER_CPU,
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

#: Byte budget of one maskgen GEMM's int32 partial sums (a request is processed in column blocks
#: that fit it, which bounds the working memory per concurrent request).
MASKGEN_PART_BYTES_CPU = 64 << 20
MASKGEN_PART_BYTES_CUDA = 512 << 20


def _maskgen_block_cols(rows: int, planes: int, cols: int, budget: int, align: int) -> int:
    """Columns per maskgen block: ``rows x planes x cols`` int32 partials within ``budget`` bytes,
    split evenly (no small tail block)."""
    step = max(align, budget // max(1, rows * planes * 4) // align * align)
    nblk = -(-cols // step)
    even = -(-cols // nblk)
    return -(-even // align) * align


def _combine_planes(planes: list[torch.Tensor], const: torch.Tensor) -> torch.Tensor:
    """``sum_d planes[d] << 8d + const mod 2^64`` as int64 bits (``planes``: int32 ``(m, c)`` views).

    Planes 0-3 and 4-7 are summed separately without overflow (``|plane| < 2^31``, so every term
    is below ``2^55``); only ``const`` (any int64 bit pattern) and the final ``hi << 32`` wrap, and
    int64 wrap-around is two's complement, i.e. reduction mod 2^64.
    """
    m, c = planes[0].shape
    acc = torch.empty((m, c), dtype=torch.int64, device=planes[0].device)
    acc.copy_(planes[0])
    for d in range(1, min(len(planes), 4)):
        acc.add_(planes[d], alpha=1 << (8 * d))
    acc.add_(const)
    if len(planes) > 4:
        hi = torch.empty((m, c), dtype=torch.int64, device=acc.device)
        hi.copy_(planes[4])
        for d in range(5, len(planes)):
            hi.add_(planes[d], alpha=1 << (8 * (d - 4)))
        acc.add_(hi.bitwise_left_shift_(32))
    return acc


def _combine_planes_np(planes: list[np.ndarray], const: np.ndarray) -> np.ndarray:
    """NumPy twin of :func:`_combine_planes` for the CPU path: uint64 ``(m, c)``, wrap-around in uint64
    (well defined). ``const`` is ``(m, 1)`` uint64. Faster than torch's strided int64 kernels on CPU."""
    acc = planes[0].astype(np.int64)
    tmp = np.empty_like(acc)
    for d in range(1, min(len(planes), 4)):
        np.multiply(planes[d], 1 << (8 * d), out=tmp, dtype=np.int64)  # |.| < 2^55: exact
        acc += tmp
    out = acc.view(np.uint64)
    out += const
    if len(planes) > 4:
        hi = planes[4].astype(np.int64)
        for d in range(5, len(planes)):
            np.multiply(planes[d], 1 << (8 * (d - 4)), out=tmp, dtype=np.int64)
            hi += tmp
        hu = hi.view(np.uint64)
        hu <<= np.uint64(32)
        out += hu
    return out


_COMBINE_ROWS = 128  # row chunk of the CPU combine (keeps its int64 temporaries in L2)
_combine_pool: ThreadPoolExecutor | None = None
_combine_pool_lock = threading.Lock()


def _combine_rows_np(part: np.ndarray, nb: int, const: np.ndarray, out: np.ndarray, ylo: int, yhi: int) -> None:
    """``out[r, c, :] = bytes ylo..yhi-1 of (sum_d part[r, d*cb + c] << 8d + const[r]) mod 2^64``.

    ``part`` is ``(m, nb * cb)`` int32 (plane-major column blocks), ``out`` a ``(m, cb, yhi - ylo)``
    uint8 view. Rows go in L2-sized chunks; large calls are split over a small thread pool (NumPy
    releases the GIL).
    """
    m = part.shape[0]
    cb = part.shape[1] // nb

    def rows(r0: int, r1: int) -> None:
        for a in range(r0, r1, _COMBINE_ROWS):
            b = min(a + _COMBINE_ROWS, r1)
            y = _combine_planes_np([part[a:b, d * cb:(d + 1) * cb] for d in range(nb)], const[a:b])
            out[a:b] = y.view(np.uint8).reshape(b - a, cb, 8)[:, :, ylo:yhi]

    workers = min(torch.get_num_threads(), max(1, (m * cb) // (1 << 16)), -(-m // _COMBINE_ROWS))
    if workers <= 1:
        rows(0, m)
        return
    global _combine_pool
    with _combine_pool_lock:
        if _combine_pool is None:
            _combine_pool = ThreadPoolExecutor(max(2, os.cpu_count() or 2), thread_name_prefix="pgw-maskgen")
        pool = _combine_pool
    per = -(-m // workers)
    for f in [pool.submit(rows, r0, min(m, r0 + per)) for r0 in range(0, m, per)]:
        f.result()


def _maskgen_block_tn(wpad: torch.Tensor, raw: torch.Tensor, m: int, n: int, const: torch.Tensor,
                      ylo: int, yhi: int) -> torch.Tensor:
    """One maskgen column block on ``wpad``'s device (the CUDA path; device-agnostic, so testable on CPU).

    ``wpad``: ``(mp, np_)`` int8 weight zero-padded to multiples of 32; ``raw``: ``(n, cb, nb)`` uint8
    coefficient bytes on the same device. Returns the ``(m, cb, yhi - ylo)`` uint8 view of bytes
    ``ylo..yhi-1`` of ``(W8 b + const) mod 2^64``. The planes are laid out ``(nb, cbp, np_)`` and used as
    the column-major right operand (the TN form cuBLASLt int8 GEMMs need); the padding is left
    uninitialised: padded inputs meet zero weight columns (any int8 times 0 is 0) and padded output
    columns are dropped.
    """
    _, cb, nb = raw.shape
    mp, np_ = wpad.shape
    cbp = -(-cb // 32) * 32
    planes = torch.empty((nb, cbp, np_), dtype=torch.uint8, device=wpad.device)
    torch.bitwise_xor(raw.permute(2, 1, 0), 0x80, out=planes[:, :cb, :n])
    part = torch._int_mm(wpad, planes.view(torch.int8).view(nb * cbp, np_).t()).view(mp, nb, cbp)
    y = _combine_planes([part[:m, d, :cb] for d in range(nb)], const)
    return y.view(torch.uint8).view(m, cb, 8)[:, :, ylo:yhi]


_STREAMS: dict[torch.device, list[torch.cuda.Stream]] = {}
_STREAMS_LOCK = threading.Lock()


def _borrow_stream(dev: torch.device) -> torch.cuda.Stream:  # pragma: no cover - GPU only
    """A CUDA stream for one maskgen call (pooled per device; one per concurrent call)."""
    with _STREAMS_LOCK:
        pool = _STREAMS.setdefault(dev, [])
        if pool:
            return pool.pop()
    return torch.cuda.Stream(dev)  # type: ignore[no-untyped-call]


def _return_stream(dev: torch.device, stream: torch.cuda.Stream) -> None:  # pragma: no cover - GPU only
    with _STREAMS_LOCK:
        _STREAMS.setdefault(dev, []).append(stream)

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
        self._mg_consts: dict[tuple[int, int], torch.Tensor] = {}  # maskgen per-row constants (cached rowsum)
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
            if self._pw._resolved == "cuda_int8" and self._pw._w_dev is not None:  # pragma: no cover - GPU only
                # The CUDA path keeps the weight once, padded (``_w_dev``); ``w8`` becomes a view of it. Keeping the
                # unpadded device copy as well doubled the provider's GPU memory (13.2 GiB for 7B's linears: no room
                # on a 16 GB T4).
                self._pw.w8 = self._pw._w_dev[: wd.shape[0], : wd.shape[1]]
                del wd
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

    # ---- encrypted mask generation (POST /pgw/v1/maskgen) ---------------------------------
    def mulmod64(self, b: np.ndarray, nbytes: int = 8) -> np.ndarray:
        """``W8 b mod 2^64`` for MLWE mask ciphertexts ``b`` (``(n, cols)`` uint64 -> ``(m, cols)``).

        Used by ``POST /pgw/v1/maskgen`` (:mod:`pgw.hemask`; the service itself calls
        :meth:`mulmod64_wire`). ``b`` is split into unsigned byte planes, offset to signed int8,
        multiplied with one int8 GEMM (``|partial| <= 128 * 128 * n < 2^31``) and recombined with
        wrapping int64 shifts (see :meth:`mulmod64_wire`). With ``nbytes < 8`` (coefficients below
        ``2^(8 nbytes)``, e.g. q = 2^56) the zero top planes are skipped.
        """
        if not 1 <= nbytes <= 8:
            raise ShapeError("nbytes must be in [1, 8]")
        if b.ndim != 2 or b.shape[0] != self.in_features or b.dtype != np.uint64:
            raise ShapeError(f"{self.linear}: maskgen operand must be ({self.in_features}, cols) uint64, "
                             f"got {b.dtype} {tuple(b.shape)}")
        n, cols = b.shape
        if cols == 0:
            return np.zeros((self.out_features, 0), dtype=np.uint64)
        if not self._maskgen_planes_ok():
            return self._mulmod64_f64(b)
        raw = np.ascontiguousarray(b, dtype="<u8").view(np.uint8).reshape(n, cols, 8)[:, :, :nbytes]
        out = self._maskgen_planes(raw, 0, 8, 0)
        return out.view("<u8").reshape(self.out_features, cols).astype(np.uint64, copy=False)

    def mulmod64_wire(self, data: bytes | bytearray | memoryview, cols: int, q_bits: int = 64,
                      drop_bits: int = 0) -> bytes:
        """Wire to wire: ``pack_y(W8 unpack_b(data) mod 2^64, drop_bits, q_bits)``, byte for byte.

        ``data`` is a maskgen item's ``b``: ``(n, cols)`` coefficients as ``q/8`` little-endian bytes
        each (row-major). The bytes go to the device as they are; byte ``d`` of every coefficient is
        a GEMM plane (``byte ^ 0x80 == byte - 128`` as int8), all planes in one int8 GEMM. The
        result is recombined in int64 (planes 0-3 and 4-7 exactly, only the final ``hi << 32`` and
        the constant wrap, two's complement == mod 2^64) with one cached per-row constant:
        ``128 * rowsum(W8) * sum_d 2^(8d)`` (the -128 plane offset) plus the rounding term
        ``2^(drop-1)``. The reply is bytes ``drop/8 .. q/8 - 1`` of each little-endian word (the
        bits of ``round(y / 2^drop) mod 2^(q - drop)``), cut out on the device so only
        ``(q - drop)/8`` bytes per element come back. On CUDA each call runs on its own stream
        (any number of threads may call concurrently; the service bounds how many).
        """
        n, m = self.in_features, self.out_features
        if q_bits % 8 or not 8 <= q_bits <= 64 or drop_bits % 8 or not 0 <= drop_bits < q_bits:
            raise ShapeError(f"{self.linear}: maskgen q_bits={q_bits}, drop_bits={drop_bits} unsupported")
        if isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0:
            raise ShapeError(f"{self.linear}: maskgen cols must be a positive integer, got {cols!r}")
        nb = q_bits // 8
        if len(data) != nb * n * cols:
            raise ShapeError(f"{self.linear}: maskgen b has {len(data)} bytes, expected {nb * n * cols}")
        from pgw.hemask.wire import pack_y, unpack_b, y_round_offset, y_wire_bytes

        if not self._maskgen_planes_ok():
            return pack_y(self._mulmod64_f64(unpack_b(data, (n, cols), q_bits)), drop_bits, q_bits)
        ylo, yhi = y_wire_bytes(q_bits, drop_bits)  # pack_y == these bytes of y + round offset
        raw = np.frombuffer(data, dtype=np.uint8).reshape(n, cols, nb)
        out = self._maskgen_planes(raw, ylo, yhi, y_round_offset(drop_bits))
        assert out.shape == (m, cols, yhi - ylo)
        return out.tobytes()

    def _maskgen_planes_ok(self) -> bool:
        """True when the byte-plane int8 GEMM path serves maskgen here (CUDA, or exact CPU int8 GEMMs)."""
        if sys.byteorder != "little":  # pragma: no cover - the byte slicing assumes little-endian words
            return False
        if self.device.type == "cuda":  # pragma: no cover - GPU only
            return True
        from pgw.hemask.fastmm import engine

        eng = engine()  # PGW_HEMASK_ENGINE=cpu_f64 forces the float64 path, as before
        if self.in_features > INT_MM_MAX_INNER_CPU:  # beyond the probed exact inner length
            return False
        # A CPU-device kernel with engine() == "cuda" and slow CPU int8 takes the float64 path
        # (exact, possibly slower than the old small_mm_big, which ran its GEMMs on the GPU).
        return (eng == "cpu_int8" or (eng == "cuda" and cpu_int8_fast())) and cpu_int8_available()

    def _mulmod64_f64(self, b: np.ndarray) -> np.ndarray:
        """Float64 bit-slice reference (CPUs without exact fast int8 GEMMs); reuses the cached float weight."""
        from pgw.hemask.lwe import matmul_mod64

        pw = self._pw
        if pw is not None and pw._w_float is not None and pw._w_float.dtype == torch.float64 \
                and pw._w_float.device.type == "cpu":
            small = pw._w_float.numpy().T  # (m, n) view of the cached (n, m) float64 copy
        else:
            w8 = self._w if self._w is not None else pw.w8  # type: ignore[union-attr]
            small = w8.cpu().numpy()
        return matmul_mod64(small, b, fast=False)

    def _maskgen_const(self, nb: int, rnd: int) -> torch.Tensor:
        """Cached ``(m, 1)`` int64 bits of ``128 * rowsum * sum_{d<nb} 2^(8d) + rnd mod 2^64`` on the device."""
        key = (nb, rnd)
        c = self._mg_consts.get(key)
        if c is not None:
            return c
        m = self.out_features
        if self.device.type == "cuda":  # pragma: no cover - GPU only
            assert self._pw is not None and self._pw._w_dev is not None
            cached = self._pw._w_rowsum  # the compute path's cached rowsum, if it already exists
            rs_dev = cached if cached is not None else self._pw._w_dev.sum(1, dtype=torch.int64)
            rs = rs_dev[:m].cpu().numpy()  # padded rows are 0; sum(dtype=) avoids an int64 weight copy
        else:
            rs = self._maskgen_w_cpu().sum(1, dtype=torch.int64).numpy()  # no int64 copy of the weight
        scale = (128 * sum(1 << (8 * d) for d in range(nb))) % (1 << 64)
        cu = rs.astype(np.int64).view(np.uint64) * np.uint64(scale) + np.uint64(rnd)  # wraps mod 2^64
        t = torch.from_numpy(cu.view(np.int64).reshape(m, 1).copy())
        if self.device.type == "cuda":  # pragma: no cover - GPU only
            t = t.to(self.device)
            torch.cuda.current_stream(self.device).synchronize()  # visible to every maskgen stream
        self._mg_consts[key] = t
        return t

    def _maskgen_w_cpu(self) -> torch.Tensor:
        w = self._w if self._w is not None else self._pw.w8  # type: ignore[union-attr]
        return w if w.is_contiguous() else w.contiguous()

    def _maskgen_planes(self, raw: np.ndarray, ylo: int, yhi: int, rnd: int) -> np.ndarray:
        """``(m, cols, yhi - ylo)`` uint8: bytes ``ylo..yhi-1`` of ``(W8 b + rnd) mod 2^64``.

        ``raw`` is ``(n, cols, nb)`` uint8, byte ``d`` of coefficient ``b[i, c]`` at ``[i, c, d]``
        (``b < 2^(8 nb)``); it may be a strided view. The result may be a strided view too.
        """
        n, cols, nb = raw.shape
        const = self._maskgen_const(nb, rnd)
        if self.device.type == "cuda":  # pragma: no cover - GPU only
            return self._maskgen_planes_cuda(raw, ylo, yhi, const)
        m = self.out_features
        w = self._maskgen_w_cpu()
        cnp = const.numpy().view(np.uint64)
        nbo = yhi - ylo
        out = np.empty((m, cols, nbo), dtype=np.uint8)
        step = _maskgen_block_cols(m, nb, cols, MASKGEN_PART_BYTES_CPU, 1)
        xor = np.uint8(0x80)
        for c0 in range(0, cols, step):
            cb = min(step, cols - c0)
            # Row-major (n, nb * cb) right operand: the wire's (n, cb, nb) bytes with the two inner
            # axes swapped (cache-local), so the planes come out as contiguous column blocks.
            rhs = torch.empty((n, nb * cb), dtype=torch.int8)
            np.bitwise_xor(raw[:, c0:c0 + cb].transpose(0, 2, 1), xor,
                           out=rhs.numpy().view(np.uint8).reshape(n, nb, cb))
            part = torch._int_mm(w, rhs).numpy()  # (m, nb * cb) int32
            _combine_rows_np(part, nb, cnp, out[:, c0:c0 + cb], ylo, yhi)
        return out

    def _maskgen_planes_cuda(self, raw: np.ndarray, ylo: int, yhi: int,
                             const: torch.Tensor) -> np.ndarray:  # pragma: no cover - GPU only
        assert self._pw is not None and self._pw._w_dev is not None
        wpad = self._pw._w_dev  # (mp, np_) int8, zero padded to multiples of 32
        n, cols, nb = raw.shape
        m = self.out_features
        mp = wpad.shape[0]
        dev = self.device
        nbo = yhi - ylo
        stream = _borrow_stream(dev)
        try:
            with torch.cuda.device(dev), torch.cuda.stream(stream):
                host = torch.empty((n, cols, nb), dtype=torch.uint8, pin_memory=True)
                host.numpy()[...] = raw  # the only host-side pass over the request bytes
                draw = host.to(dev, non_blocking=True)
                dout = torch.empty((m, cols, nbo), dtype=torch.uint8, device=dev)
                step = _maskgen_block_cols(mp, nb, cols, MASKGEN_PART_BYTES_CUDA, 32)
                for c0 in range(0, cols, step):
                    cb = min(step, cols - c0)
                    dout[:, c0:c0 + cb] = _maskgen_block_tn(wpad, draw[:, c0:c0 + cb], m, n, const, ylo, yhi)
                res = torch.empty((m, cols, nbo), dtype=torch.uint8, pin_memory=True)
                res.copy_(dout, non_blocking=True)
                stream.synchronize()
            return res.numpy()
        finally:
            _return_stream(dev, stream)

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
