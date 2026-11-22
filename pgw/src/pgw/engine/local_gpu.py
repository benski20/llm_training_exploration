"""Exact local linears on a gateway-owned GPU (``local_linears_device = "cuda"``; K = 32 rings).

The gateway's local linears (E2) are the masked trip with a zero mask and the product computed on the
gateway instead of the provider. This module runs exactly that, with the product on the gateway's own GPU:

1. ``pgw_fast.trip_prepare_into`` with ``r = 0`` quantizes the activation row(s) into the K-bit ring payload
   (the TripEngine's prepare kernel, self-checked against the reference quantizer by the backend), written
   straight into a pinned host buffer;
2. a captured CUDA graph ``[H2D, ring_gemv, D2H]`` computes ``x_hat @ W8^T mod 2^32``: the provider's fused
   ring GEMV (:mod:`pgw.core.triton_gemv`; wrapping unsigned 32-bit arithmetic, so any reduction order gives
   the same bits; gated on its own device self-check). Calls with more rows than the fused kernel serves
   (prompt passes) take the digit-GEMM path :func:`pgw.engine.backends.exact_int_matmul` instead (exact int64
   product, then its low 32 bits), or, with ``PGW_LOCAL_GPU_PREFILL_ROWS=P``, a second captured graph per
   linear on P-row blocks (two int8 digits, one exact int32 GEMM, ``lo + (hi << 8)`` mod 2^32);
3. ``pgw_fast.trip_finish_into`` with ``W r = 0`` lifts the centered result, applies the range check and
   rescales (the TripEngine's finish kernel).

Every step is integer-exact or the same CPU kernel as the masked path, so the outputs are the reference's
bits; the backend additionally compares this path with its CPU kernel on public test rows per linear shape
before using it (``MaskedBackend._local_gpu_agrees``) and falls back to the CPU on any mismatch or error.

With the native kernel ``pgw_fast.local_gpu_linear_into`` (default; ``PGW_LOCAL_GPU_NATIVE=0`` disables it), a
graph-sized call is all three steps in one GIL-free call: prepare into the pinned buffer, ``cuGraphLaunch`` of the
captured graph's exec through the CUDA driver, a spin on ``cuStreamQuery`` (``PGW_LOCAL_GPU_SPIN_US``, default
2000 us, then a blocking wait), the range check and rescale (the finish kernel's expressions), and the pinned
buffers zeroed. No Python runs between the launch and the result.

Privacy: the GPU belongs to the gateway (the co-located demo gives each party its own device and hides the
other one with ``CUDA_VISIBLE_DEVICES``). Nothing here talks to the provider; the masked traffic is unchanged.
The pinned staging buffers are zeroed after every call (the device buffers are overwritten by the next one).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pgw.core.native import fast as _fast
from pgw.core.ringmm import pad_weight_cuda

log = logging.getLogger(__name__)

__all__ = ["LocalGpuProducts", "local_gpu_unavailable"]


def local_gpu_unavailable(device: torch.device, k: int) -> str | None:
    """``None`` if :class:`LocalGpuProducts` can serve ring width ``k`` on ``device``, else why not."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return "no CUDA device"
    if k != 32:
        return f"ring K={k} (the fused ring GEMV path needs K = 32)"
    if _fast is None or not all(hasattr(_fast, f) for f in ("trip_prepare_into", "trip_finish_into")):
        return "native trip kernels unavailable"
    from pgw.core import triton_gemv

    if not triton_gemv.triton_available():
        return "triton unavailable"
    if not triton_gemv.self_check(device):
        return "fused ring GEMV failed its self-check on this device"
    return None


@dataclass
class _Linear:
    n: int
    m: int
    w: torch.Tensor  # int8 (mp, np_) on the device: W8 zero padded to multiples of 32 (as the provider's weights)
    graphs: dict[int, Any]  # rows -> torch.cuda.CUDAGraph
    execs: dict[int, int]  # rows -> cudaGraphExec_t address (native launch)


#: Measured 1-row launch configurations of the local graphs per GPU model and ``(n, m)``:
#: ``(block_m, block_n, split_k, num_warps)`` (``benchmarks/local_gemv_sweep.py``).
#: Tesla T4 (2026-11-22, graph replay + synchronize / kernel device time, default -> chosen): Qwen2.5-0.5B
#: qkv 43 -> 36 us / 13.0 -> 10.0 us, o 35 -> 35 us / 9.9 -> 7.6 us, down 66 -> 46 us / 31.5 -> 14.2 us (no
#: split-K, so no memset node in the graph).
ONE_ROW_CONFIGS: dict[str, dict[tuple[int, int], tuple[int, int, int, int]]] = {
    "Tesla T4": {(896, 1152): (8, 512, 1, 1), (896, 896): (8, 1024, 1, 1), (4864, 896): (4, 1024, 1, 1)},
}


def _env_configs() -> dict[tuple[int, int], tuple[int, int, int, int]]:
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for item in os.environ.get("PGW_LOCAL_GPU_GEMV", "").split(";"):
        if ":" not in item:
            continue
        shape, cfg = item.split(":", 1)
        n, m = (int(v) for v in shape.lower().split("x"))
        bm, bn, sk, nw = (int(v) for v in cfg.split(","))
        out[(n, m)] = (bm, bn, sk, nw)
    return out


def _as_ptr(handle: Any) -> int:
    """A CUDA handle as an integer address (torch returns an int or a cuda-python object)."""
    if isinstance(handle, int):
        return handle
    get = getattr(handle, "getPtr", None)
    return int(get()) if callable(get) else int(handle)


def _native_reason() -> str | None:
    """``None`` if the native GPU local kernel can run here, else why not."""
    if os.environ.get("PGW_LOCAL_GPU_NATIVE", "1").strip().lower() in ("0", "false", "no", "off"):
        return "PGW_LOCAL_GPU_NATIVE=0"
    if _fast is None or not hasattr(_fast, "local_gpu_linear_into"):
        return "native kernel unavailable"
    if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph_exec"):
        return f"torch {torch.__version__} has no CUDAGraph.raw_cuda_graph_exec"
    why = _fast.local_gpu_driver_status()
    return None if why is None else str(why)


class LocalGpuProducts:
    """Per-linear exact products on one gateway GPU (module docstring). Thread-safe (one call at a time)."""

    def __init__(self, device: torch.device, k: int, a: int, cap: int, *, graph_rows: tuple[int, ...] = (1,),
                 max_n: int, max_m: int, staging_rows: int = 8, spin: bool = True):
        from pgw.core import triton_gemv

        self.device = device
        self._index = device.index if device.index is not None else torch.cuda.current_device()
        self.k, self.a, self.cap = int(k), int(a), int(cap)
        self.fused_rows = max(1, int(triton_gemv.max_rows(device)))
        self.graph_rows = tuple(sorted({int(r) for r in graph_rows if 1 <= int(r) <= self.fused_rows}))
        self.staging_rows = max(max(self.graph_rows, default=1), int(staging_rows))
        self.spin = bool(spin)
        self._lock = threading.Lock()
        self._lin: dict[str, _Linear] = {}
        r = self.staging_rows
        with torch.cuda.device(device):
            self._stream = torch.cuda.Stream(device)  # type: ignore[no-untyped-call]
            self._pool = torch.cuda.graph_pool_handle()
            self._event = torch.cuda.Event()
            self._h_in = torch.zeros(r * max_n, dtype=torch.int32, pin_memory=True)
            self._h_out = torch.zeros(r * max_m, dtype=torch.int32, pin_memory=True)
            self._d_in = torch.zeros(r * max_n, dtype=torch.int32, device=device)
            self._d_out = torch.zeros(r * max_m, dtype=torch.int32, device=device)
        self._pin_in = self._h_in.numpy().view(np.uint8)
        self._pin_out = self._h_out.numpy().view(np.uint8)
        self._zr = np.zeros(r * max_n, dtype=np.int64)  # zero masks (flat: any (rows, n) view is contiguous)
        self._zwr = np.zeros(r * max_m, dtype=np.int64)
        self.stats: dict[str, Any] = {"graph_calls": 0, "native_calls": 0, "eager_calls": 0, "native_s": 0.0,
                                      "eager_s": 0.0}
        self.native_reason = _native_reason()
        self.native = self.native_reason is None
        self.spin_us = int(os.environ.get("PGW_LOCAL_GPU_SPIN_US", "2000") or 2000)
        #: keep the GIL during a native call (``PGW_LOCAL_GPU_HOLD_GIL``, default on): a call is tens of
        #: microseconds, and releasing the GIL lets another thread take it for up to the switch interval
        self.hold_gil = os.environ.get("PGW_LOCAL_GPU_HOLD_GIL", "1").strip().lower() not in ("0", "false", "no", "off")
        self._stream_ptr = int(self._stream.cuda_stream)
        self._h_in_ptr, self._h_in_bytes = int(self._h_in.data_ptr()), int(self._h_in.numel()) * 4
        self._h_out_ptr, self._h_out_bytes = int(self._h_out.data_ptr()), int(self._h_out.numel()) * 4
        #: Prompt passes through one more captured graph per linear (``PGW_LOCAL_GPU_PREFILL_ROWS=P``, a multiple
        #: of 8 from 16; 0 = off, the default): ``[H2D, two balanced int8 digits, one int8 GEMM (cuBLASLt
        #: _int_mm, int32 accumulation, exact), lo + (hi << 8) mod 2^32, D2H]`` on P rows. The backend splits a
        #: call of more than P rows into P-row blocks; a shorter block runs on zero-padded rows (the pinned
        #: input is all zero outside a call, and a zero row's product is zero). Each output row is its own row's
        #: exact product mod 2^32, the ring GEMV's bits (self-checked per linear like every path here).
        try:
            p_rows = int(os.environ.get("PGW_LOCAL_GPU_PREFILL_ROWS", "0") or 0)
        except ValueError:
            p_rows = 0
        if p_rows < 16 or p_rows % 8 or self.a > 15 or max_n > 131_040:
            p_rows = 0  # digits: |x| < 2^14 keeps the high digit in int8 and every int32 partial sum exact
        self.prefill_block_rows = p_rows
        if p_rows:
            with torch.cuda.device(device):
                self._hp_in = torch.zeros(p_rows * max_n, dtype=torch.int32, pin_memory=True)
                self._hp_out = torch.zeros(p_rows * max_m, dtype=torch.int32, pin_memory=True)
                self._dp_in = torch.zeros(p_rows * max_n, dtype=torch.int32, device=device)
                self._dp_a8 = torch.zeros(2 * p_rows * max_n, dtype=torch.int8, device=device)
                self._dp_out = torch.zeros(p_rows * max_m, dtype=torch.int32, device=device)
            self._pp_in = self._hp_in.numpy().view(np.uint8)
            self._pp_out = self._hp_out.numpy().view(np.uint8)
            self._pzr = np.zeros(p_rows * max_n, dtype=np.int64)
            self._pzwr = np.zeros(p_rows * max_m, dtype=np.int64)
        self.stats["prefill_calls"] = 0

    def gemv_config(self, rows: int, n: int, m: int) -> Any:
        """Launch configuration of the graph's ring GEMV for ``rows`` rows of an ``(m, n)`` weight: an entry of
        ``PGW_LOCAL_GPU_GEMV`` (``"NxM:block_m,block_n,split_k,num_warps;..."``, 1-row graphs), else the
        measured table of this GPU model (:data:`ONE_ROW_CONFIGS`), else ``None`` (the provider's heuristic).
        Any configuration computes the same integers (wrapping 32-bit sums); the backend self-checks every
        linear against the CPU reference before use."""
        if rows != 1:
            return None
        from pgw.core.triton_gemv import GemvConfig, device_name

        spec = _env_configs().get((n, m))
        if spec is None and os.environ.get("PGW_LOCAL_GPU_TABLE", "1").strip().lower() not in ("0", "no", "off"):
            spec = ONE_ROW_CONFIGS.get(device_name(self.device), {}).get((n, m))
        if spec is None:
            return None
        bm, bn, sk, nw = spec
        cfg = GemvConfig(1, bm, bn, sk, nw)
        cfg.validate()
        return cfg

    def add(self, key: str, w8: torch.Tensor) -> None:
        """Upload ``w8`` (int8 ``(m, n)``, any device) and capture its graphs. Call at start-up."""
        from pgw.core.triton_gemv import ring_gemv

        m, n = (int(s) for s in w8.shape)
        if n * self.staging_rows > self._h_in.numel() or m * self.staging_rows > self._h_out.numel():
            raise ValueError(f"{key}: shape ({m}, {n}) exceeds the staging buffers")
        w = pad_weight_cuda(w8.detach().cpu(), self.device)
        lin = _Linear(n, m, w, {}, {})
        stream = self._stream
        with torch.cuda.device(self.device):
            for rows in self.graph_rows:
                d_in = self._d_in[: rows * n].view(rows, n)
                h_in = self._h_in[: rows * n].view(rows, n)
                d_out = self._d_out[: rows * m].view(rows, m)
                h_out = self._h_out[: rows * m].view(rows, m)

                cfg = self.gemv_config(rows, n, m)

                def body(d_in: torch.Tensor = d_in, h_in: torch.Tensor = h_in, d_out: torch.Tensor = d_out,
                         h_out: torch.Tensor = h_out, cfg: Any = cfg) -> None:
                    d_in.copy_(h_in, non_blocking=True)
                    ring_gemv(d_in, w, 32, n=n, m=m, out=d_out, config=cfg)
                    h_out.copy_(d_out, non_blocking=True)

                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    body()  # compile outside the capture
                stream.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._pool, stream=stream, capture_error_mode="thread_local"):
                    body()
                lin.graphs[rows] = g
                if self.native:
                    try:
                        lin.execs[rows] = _as_ptr(g.raw_cuda_graph_exec())
                    except Exception as exc:  # the Python replay serves it
                        self.native, self.native_reason = False, f"raw_cuda_graph_exec: {exc}"
            if self.prefill_block_rows and n % 32 == 0 and m % 32 == 0:
                try:
                    lin.graphs[-1] = self._capture_prefill(w, n, m)
                except Exception as exc:  # the prompt-block graphs are an option: eager prompt products instead
                    log.warning("local GPU prompt graphs off (%s: %s)", type(exc).__name__, str(exc)[:200])
                    self.prefill_block_rows = 0
        self._h_in.zero_()
        self._h_out.zero_()
        self._lin[key] = lin

    def _capture_prefill(self, w: torch.Tensor, n: int, m: int) -> Any:
        """The prompt-block graph of one linear (see :attr:`prefill_block_rows`)."""
        p = self.prefill_block_rows
        d_in = self._dp_in[: p * n].view(p, n)
        h_in = self._hp_in[: p * n].view(p, n)
        a8 = self._dp_a8[: 2 * p * n].view(2 * p, n)
        d_out = self._dp_out[: p * m].view(p, m)
        h_out = self._hp_out[: p * m].view(p, m)
        wt = w[:m, :n].t()  # column-major view of the (m, n) weight (n and m: multiples of 32, so no padding)

        def body() -> None:
            d_in.copy_(h_in, non_blocking=True)
            lo = ((d_in + 128) & 255) - 128
            a8[:p].copy_(lo)
            a8[p:].copy_((d_in - lo) >> 8)
            y = torch._int_mm(a8, wt)  # (2p, m) int32: exact (|digit| <= 128, n <= 131040)
            torch.add(y[:p], torch.bitwise_left_shift(y[p:], 8), out=d_out)  # mod 2^32 (int32 wraps)
            h_out.copy_(d_out, non_blocking=True)

        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            body()
        stream.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool, stream=stream, capture_error_mode="thread_local"):
            body()
        self._hp_in.zero_()
        self._hp_out.zero_()
        return g

    def has(self, key: str) -> bool:
        return key in self._lin

    def forward(self, key: str, xf: np.ndarray, wnorm: float, scale: np.ndarray, bias: np.ndarray | None,
                bound: int, out: np.ndarray, sx: np.ndarray) -> int:
        """Exact local product of ``xf`` (float32 ``(rows, n)``, C-contiguous, finite) into ``out``
        (float32 ``(rows, m)``); returns the finish status (0 = ok, 2 = range check failed). ``sx`` is a
        float64 ``(rows,)`` scratch. Raises ``ValueError`` from the prepare kernel (e.g. ``ring_overflow``)."""
        lin = self._lin[key]
        rows, n = xf.shape
        m = lin.m
        k, a, cap = self.k, self.a, self.cap
        with self._lock:
            ex = lin.execs.get(rows) if self.native else None
            if ex:
                t0 = time.perf_counter()
                status = _fast.local_gpu_linear_into(
                    xf, scale, bias, k, a, cap, wnorm, bound, sx, out, self._index, ex, self._stream_ptr,
                    self._h_in_ptr, self._h_in_bytes, self._h_out_ptr, self._h_out_bytes, self.spin_us,
                    self.hold_gil)
                self.stats["native_s"] += time.perf_counter() - t0
                self.stats["native_calls"] += 1
                return int(status)
            zr, zwr = self._zeros(rows, n, m)
            g = lin.graphs.get(rows)
            pin_in_all, pin_out_all, g_rows = self._pin_in, self._pin_out, rows
            if g is None and 1 < rows <= self.prefill_block_rows and -1 in lin.graphs:
                # a prompt block: the digit-GEMM graph on P rows, the rows past ``rows`` zero
                g, g_rows = lin.graphs[-1], self.prefill_block_rows
                pin_in_all, pin_out_all = self._pp_in, self._pp_out
                zr, zwr = self._pzr[: rows * n].reshape(rows, n), self._pzwr[: rows * m].reshape(rows, m)
                self.stats["prefill_calls"] += 1
            if g is not None:
                pin_in, pin_out = pin_in_all[: rows * n * 4], pin_out_all[: rows * m * 4]
                try:
                    _fast.trip_prepare_into(xf, zr, k, a, cap, wnorm, pin_in, sx)
                    ev = self._event
                    if torch.cuda.current_device() == self._index:
                        # the usual case (the gateway sees one GPU): replay on the calling thread's current
                        # stream, without entering device/stream contexts (a few microseconds each)
                        g.replay()
                        ev.record()
                    else:
                        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                            g.replay()
                            ev.record(self._stream)
                    if self.spin:  # a few tens of microseconds: poll instead of a blocking wait
                        while not ev.query():
                            pass
                    else:
                        ev.synchronize()
                    status, _ = _fast.trip_finish_into(pin_out, zwr, sx, scale, bias, k, bound, out)
                finally:
                    pin_in.fill(0)
                    pin_out_all[: g_rows * m * 4].fill(0)
                self.stats["graph_calls"] += 1
                return int(status)
            # more rows than a graph serves (prompt passes): the same three steps without a graph
            t0 = time.perf_counter()
            payload = np.empty(rows * n * 4, dtype=np.uint8)
            try:
                _fast.trip_prepare_into(xf, zr, k, a, cap, wnorm, payload, sx)
                y32 = self._eager(lin, payload, rows)
                status, _ = _fast.trip_finish_into(y32.numpy().view(np.uint8).reshape(-1), zwr, sx, scale, bias, k,
                                                   bound, out)
                y32.zero_()
            finally:
                payload.fill(0)
            self.stats["eager_calls"] += 1
            self.stats["eager_s"] += time.perf_counter() - t0
            return int(status)

    def one_row(self, key: str, wnorm: float, scale: np.ndarray, bias: np.ndarray | None, bound: int,
                on_error: Any) -> Any:
        """A lean callable ``f(x) -> y`` for 1-row decode calls of ``key`` through the native kernel (or ``None``
        when the native path or a 1-row graph is unavailable): ``x`` float32 ``(1, n)`` C-contiguous CPU tensor
        without grad, ``y`` a new float32 ``(1, m)`` tensor. The same call as :meth:`forward` with every per-call
        lookup bound once; non-finite inputs are refused by the kernel's quantizer (``ring_overflow``).
        ``on_error(status_or_exc)`` maps a non-zero status or a ``ValueError`` to the backend's exception."""
        lin = self._lin.get(key)
        ex = lin.execs.get(1) if (lin is not None and self.native) else None
        if not ex:
            return None
        m = lin.m
        k, a, cap = self.k, self.a, self.cap
        fn = _fast.local_gpu_linear_into
        args_tail = (self._index, ex, self._stream_ptr, self._h_in_ptr, self._h_in_bytes, self._h_out_ptr,
                     self._h_out_bytes, self.spin_us, self.hold_gil)
        lock, stats = self._lock, self.stats
        from_numpy, empty, f32, f64 = torch.from_numpy, np.empty, np.float32, np.float64

        def call(x: torch.Tensor) -> torch.Tensor:
            out = empty((1, m), f32)
            sx = empty(1, f64)
            try:
                with lock:
                    status = fn(x.numpy(), scale, bias, k, a, cap, wnorm, bound, sx, out, *args_tail)
                    stats["native_calls"] += 1
            except ValueError as exc:
                on_error(exc)
                raise
            finally:
                sx[0] = 0.0
            if status:
                out.fill(0)
                on_error(int(status))
            return from_numpy(out)

        return call

    def _zeros(self, rows: int, n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Zero mask ``r`` (``(rows, n)``) and zero ``W r`` (``(rows, m)``), int64, C-contiguous."""
        if rows <= self.staging_rows:
            return self._zr[: rows * n].reshape(rows, n), self._zwr[: rows * m].reshape(rows, m)
        return np.zeros((rows, n), dtype=np.int64), np.zeros((rows, m), dtype=np.int64)


    def _eager(self, lin: _Linear, payload: np.ndarray, rows: int) -> torch.Tensor:
        """``x_hat @ W8^T mod 2^32`` as int32 bit patterns on the CPU, for any row count."""
        from pgw.core.triton_gemv import low32, ring_gemv
        from pgw.engine.backends import exact_int_matmul

        n, m = lin.n, lin.m
        x32 = torch.from_numpy(payload.view(np.int32).reshape(rows, n))
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            xd = x32.to(self.device, non_blocking=False)
            try:
                if rows <= self.fused_rows:
                    yd = ring_gemv(xd, lin.w, 32, n=n, m=m)
                else:
                    # int32 view of a K = 32 payload is the centered activation itself (|xq| < 2^(A-1))
                    y64 = exact_int_matmul(xd.to(torch.int64), lin.w[:m, :n].t(), self.a)
                    yd = low32(y64.contiguous()).contiguous()  # a copy: y64 is wiped next
                    y64.zero_()
                y = yd.cpu()
                yd.zero_()
            finally:
                xd.zero_()
            self._stream.synchronize()
        return y
