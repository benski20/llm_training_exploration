"""Persistent GPU trip worker (scaling plan, provider target 3: <= 15-25 us of service overhead per trip; opt-in).

Today a lane trip replays a CUDA graph per call (``cuGraphLaunch`` + stream wait, ~66 us of service time on the
H100 run at 7B, most of it launch and synchronisation). Here one **resident kernel** per GPU is launched once and
polls a control block in mapped pinned host memory; a trip is a memcpy, a doorbell store and a spin, done natively
by the lane (``rust/pgw-lane/src/persistent.rs``), with no launch, no graph replay and no Python per trip.

Protocol (one outstanding trip per worker; the device lock keeps it so):

* control block (:data:`CTRL_BYTES`, 128-byte aligned, mapped pinned host memory; u32 little endian):
  ``doorbell @0`` (host: trip sequence number, written last), ``linear @4`` (table index), ``rows @8``,
  ``quit @12`` (host: stop the kernel), ``done @128`` (device: last finished sequence number, written last),
  ``status @132`` (device: 0 ok, else an error code);
* one mapped input buffer per group: ``rows x n`` u32, the packed K = 32 ``x_hat`` (little-endian low 32 bits);
  written once by the host and read by every shard of a tensor-parallel group (broadcast);
* one mapped output buffer per group: ``rows x m`` u32 row-major; shard ``j`` writes columns
  ``[col0_j, col0_j + m_j)`` of every row, so the gathered reply needs no extra copy.

Device side (:data:`KERNEL_SOURCE`, compiled at run time with NVRTC, loaded and launched through the driver
API with ``ctypes``): block 0 polls the doorbell (``__nanosleep`` between reads) and publishes it in device
memory for the other blocks; the grid then (1) copies ``x`` from host memory to a device buffer, (2) runs the
ring GEMV ``y = W8 x mod 2^32`` with wrapping u32 arithmetic (one warp per group of output rows, 16-byte weight
loads, x from L2; 4 instead of 16 rows per warp when 16 would leave warps idle) and each warp stores its outputs
straight into the mapped reply, and (3) after a grid barrier whose arrival is preceded by one system-scope fence
per block, block 0 publishes ``status`` and ``done``. Grid-wide barriers between the phases (all blocks are
co-resident: grid <= SMs x occupancy). System fences are per block, never per thread: per-thread fences cost
~2.8 ms per trip on the first H100 run (2026-11-22, exact but 30x slower than graphs). Exact: every product
and sum is a u32 wrap, i.e. arithmetic mod 2^32, the
same values as the fused Triton GEMV and the digit-GEMM path. Rows per trip: 1..:data:`MAX_ROWS` (larger calls
fall back to the graph path).

Validation: :func:`start_lane_persistent` registers the workers with the lane only after a self-check through
the native serving path against the digit-GEMM path (random, all-ones and zero inputs, every row count); any
mismatch or error disables the persistent path (calls then use graphs or the kernel path). A worker that stops
answering disables it too (``rust/pgw-lane``).

CPU: :class:`EmulatedWorker` runs the same protocol on the same memory layout in a Python thread (numpy
arithmetic mod 2^32), so the lane's native executor, the tensor-parallel gather and the self-check are tested
without a GPU. The CUDA code path cannot be exercised here; it is guarded (``torch.cuda.is_available()``, the
NVRTC library, the driver) and measured by ``benchmarks/provider_trip_bench.py`` on Modal.

Configuration (environment, all default off): ``PGW_SERVICE_EXEC=persistent`` (default ``graph``),
``PGW_SERVICE_PERSISTENT_BLOCKS_PER_SM`` (default 1: one 256-thread block per SM, so other kernels (the graph
path for larger calls, maskgen) still find registers and threads on every SM while the resident kernel spins),
``PGW_SERVICE_PERSISTENT_MAX_ROWS`` (default and maximum 4),
``PGW_SERVICE_PERSISTENT_TIMEOUT_MS`` (default 2000).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import logging
import os
import struct
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch

from pgw.core.errors import BackendUnavailableError, ConfigError

__all__ = [
    "CTRL_BYTES",
    "KERNEL_SOURCE",
    "MAX_ROWS",
    "CudaWorker",
    "EmulatedWorker",
    "HostBuffer",
    "LinearEntry",
    "PersistentGroup",
    "exec_mode",
    "start_lane_persistent",
]

log = logging.getLogger(__name__)

CTRL_DOORBELL = 0
CTRL_LINEAR = 4
CTRL_ROWS = 8
CTRL_QUIT = 12
CTRL_DONE = 128
CTRL_STATUS = 132
CTRL_BYTES = 256
#: Largest rows per trip the resident kernel accepts (compile-time accumulator tiles).
MAX_ROWS = 4
ENV_EXEC = "PGW_SERVICE_EXEC"


def exec_mode() -> str:
    """``graph`` (default) or ``persistent`` (``PGW_SERVICE_EXEC``)."""
    v = (os.environ.get(ENV_EXEC) or "graph").strip().lower()
    if v not in ("graph", "persistent"):
        raise ConfigError(f"{ENV_EXEC} must be graph or persistent, got {v!r}")
    return v


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        v = int(raw) if raw else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    return min(hi, max(lo, v))


def max_rows() -> int:
    """Rows per trip served persistently (``PGW_SERVICE_PERSISTENT_MAX_ROWS``, 1..4)."""
    return _env_int("PGW_SERVICE_PERSISTENT_MAX_ROWS", MAX_ROWS, 1, MAX_ROWS)


# ---- the resident kernel ------------------------------------------------------------------------
KERNEL_SOURCE = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

struct LinDesc { u64 w; u64 ld; u32 n; u32 m; u32 col0; u32 m_total; };  // 32 bytes, see LinearEntry

#define C_DOORBELL 0
#define C_LINEAR 1
#define C_ROWS 2
#define C_QUIT 3
#define C_DONE 32
#define C_STATUS 33

// bar: [0] arrivals, [1] generation, [2] published doorbell, [3] quit
// SYS: thread 0 makes the block's earlier writes (ordered before it by __syncthreads) visible system-wide
// (mapped host memory) before it arrives. One system fence per block, not per thread: a per-thread
// __threadfence_system() serialises on every SM and cost ~2.8 ms per trip on an H100 (132 x 256 threads).
template <bool SYS>
__device__ __forceinline__ void grid_sync(u32* bar) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile u32* vgen = bar + 1;
    const u32 gen = *vgen;
    if (SYS) __threadfence_system(); else __threadfence();
    if (atomicAdd(bar, 1u) == gridDim.x - 1u) {
      atomicExch(bar, 0u);
      __threadfence();
      atomicAdd(bar + 1, 1u);
    } else {
      while (*vgen == gen) { __nanosleep(20); }
    }
    __threadfence();
  }
  __syncthreads();
}

// y = W8 x mod 2^32 for R rows; warp g owns output rows [g*RPW, g*RPW+RPW) and writes them straight into the
// mapped reply (row-major, row stride m_total, columns from col0): lane k < RPW stores output row row0+k.
template <int R, int RPW>
__device__ __forceinline__ void gemv(const LinDesc& L, const u32* __restrict__ d_x, u32* __restrict__ h_out,
                                     u32 warp, u32 nwarps, u32 lane) {
  const u32 n = L.n, m = L.m;
  const signed char* W = (const signed char*)L.w;
  const u32 nv = ((n & 15u) == 0u && (L.ld & 15u) == 0u) ? n : 0u;
  for (u32 g = warp; g * RPW < m; g += nwarps) {
    const u32 row0 = g * RPW;
    u32 acc[RPW][R];
#pragma unroll
    for (int k = 0; k < RPW; ++k)
#pragma unroll
      for (int r = 0; r < R; ++r) acc[k][r] = 0u;
    for (u32 j = lane * 16u; j < nv; j += 512u) {
      u32 xv[R][16];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4* xp = (const uint4*)(d_x + (u64)r * n + j);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 t = __ldcg(xp + q);
          xv[r][4 * q] = t.x; xv[r][4 * q + 1] = t.y; xv[r][4 * q + 2] = t.z; xv[r][4 * q + 3] = t.w;
        }
      }
#pragma unroll
      for (int k = 0; k < RPW; ++k) {
        const u32 row = row0 + k;
        if (row < m) {
          const int4 wv = __ldg((const int4*)(W + (u64)row * L.ld + j));
          const u32 w4[4] = {(u32)wv.x, (u32)wv.y, (u32)wv.z, (u32)wv.w};
#pragma unroll
          for (int b = 0; b < 16; ++b) {
            const u32 wb = (u32)(int)(signed char)(w4[b >> 2] >> (8 * (b & 3)));
#pragma unroll
            for (int r = 0; r < R; ++r) acc[k][r] += wb * xv[r][b];
          }
        }
      }
    }
    for (u32 j = nv + lane; j < n; j += 32u) {
#pragma unroll
      for (int k = 0; k < RPW; ++k) {
        const u32 row = row0 + k;
        if (row < m) {
          const u32 wb = (u32)(int)W[(u64)row * L.ld + j];
#pragma unroll
          for (int r = 0; r < R; ++r) acc[k][r] += wb * __ldcg(d_x + (u64)r * n + j);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < RPW; ++k)
#pragma unroll
      for (int r = 0; r < R; ++r) {
        u32 v = acc[k][r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);  // every lane: the sum
        acc[k][r] = v;
      }
#pragma unroll
    for (int k = 0; k < RPW; ++k) {
      const u32 row = row0 + k;
      if (lane == (u32)k && row < m) {
#pragma unroll
        for (int r = 0; r < R; ++r) h_out[(u64)r * L.m_total + L.col0 + row] = acc[k][r];
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) pgw_persistent(const LinDesc* __restrict__ lins, u32 nlin,
                                                                 volatile u32* ctrl,
                                          const u32* h_in, u32* h_out, u32* d_x, u32* d_y, u32* bar,
                                          u32 max_rows, u32 x_cap, u32 y_cap) {
  __shared__ u32 s_seq, s_go, s_li, s_rows;
  const u32 tid = threadIdx.x;
  const u32 gtid = blockIdx.x * blockDim.x + tid;
  const u32 gthreads = gridDim.x * blockDim.x;
  u32 seen = ctrl[C_DONE];
  for (;;) {
    if (tid == 0) {
      volatile u32* pub = bar + 2;
      volatile u32* pquit = bar + 3;
      s_go = 0u;
      for (;;) {
        if (blockIdx.x == 0) {
          const u32 d = ctrl[C_DOORBELL];
          if (d != seen) { *pub = d; __threadfence(); s_seq = d; s_go = 1u; __threadfence_system(); break; }
          if (ctrl[C_QUIT] != 0u) { *pquit = 1u; __threadfence(); break; }
        } else {
          const u32 d = *pub;
          if (d != seen) { s_seq = d; s_go = 1u; __threadfence_system(); break; }
          if (*pquit != 0u) break;
        }
        __nanosleep(64);
      }
      if (s_go) {  // one read of the trip's words per block (mapped host memory), after the system fence
        s_li = ctrl[C_LINEAR];
        s_rows = ctrl[C_ROWS];
      }
    }
    __syncthreads();
    if (s_go == 0u) return;
    seen = s_seq;  // thread 0's system fence after the doorbell, then __syncthreads, order the reads below
    const u32 li = s_li;
    const u32 rows = s_rows;
    u32 bad = (li >= nlin || rows == 0u || rows > max_rows || rows > 4u) ? 1u : 0u;
    LinDesc L = {0ull, 0ull, 0u, 0u, 0u, 0u};
    if (!bad) {
      L = lins[li];
      if (rows * L.n > x_cap || rows * L.m > y_cap) bad = 2u;
    }
    if (!bad) {
      const u32 total = rows * L.n;
      if ((total & 3u) == 0u) {  // 16-byte reads over the bus (h_in and d_x are 128-byte aligned)
        const uint4* src = (const uint4*)h_in;
        uint4* dst = (uint4*)d_x;
        for (u32 i = gtid; i < total / 4u; i += gthreads) dst[i] = __ldcv(src + i);
      } else {
        for (u32 i = gtid; i < total; i += gthreads) d_x[i] = __ldcv(h_in + i);
      }
    }
    grid_sync<false>(bar);
    if (!bad) {
      const u32 lane = tid & 31u, warp = gtid >> 5, nwarps = gthreads >> 5;
      // fewer rows per warp when 16 would leave warps idle (small m): more warps stream the weight
      const bool wide = L.m >= 16u * nwarps;
      switch (rows) {
        case 1: if (wide) gemv<1, 16>(L, d_x, h_out, warp, nwarps, lane);
                else gemv<1, 4>(L, d_x, h_out, warp, nwarps, lane); break;
        case 2: if (wide) gemv<2, 8>(L, d_x, h_out, warp, nwarps, lane);
                else gemv<2, 4>(L, d_x, h_out, warp, nwarps, lane); break;
        case 3: gemv<3, 5>(L, d_x, h_out, warp, nwarps, lane); break;
        default: gemv<4, 4>(L, d_x, h_out, warp, nwarps, lane); break;
      }
    }
    grid_sync<true>(bar);  // every block's reply stores visible to the host before done
    if (blockIdx.x == 0 && tid == 0) {
      ctrl[C_STATUS] = bad;
      __threadfence_system();
      ctrl[C_DONE] = seen;
      __threadfence_system();
    }
  }
}
"""

#: ``struct LinDesc`` of :data:`KERNEL_SOURCE`.
LINDESC = struct.Struct("<QQIIII")


@dataclass
class LinearEntry:
    """One row of a worker's linear table: the shard's weight and where its outputs go."""

    w: torch.Tensor  # int8 (m_rows, ld) with at least n columns, row stride ld
    n: int
    m: int  # rows of this shard
    col0: int  # first output column of this shard in the reply row
    m_total: int  # reply row length (whole linear)

    @property
    def ld(self) -> int:
        """Row stride of :attr:`w` in elements."""
        return int(self.w.stride(0))

    def pack(self) -> bytes:
        """The 32-byte device descriptor."""
        return LINDESC.pack(int(self.w.data_ptr()), self.ld, self.n, self.m, self.col0, self.m_total)


# ---- host memory ------------------------------------------------------------------------------------
class HostBuffer:
    """Host memory with a stable, 128-byte aligned address: mapped pinned memory (``cuMemHostAlloc`` with
    ``PORTABLE | DEVICEMAP``, visible to every GPU at the same address under unified addressing) for CUDA
    workers, plain memory for the emulator."""

    def __init__(self, nbytes: int, *, cuda: bool = False):
        self.nbytes = int(nbytes)
        self.cuda = cuda
        self._raw: Any = None
        if cuda:  # pragma: no cover - GPU only
            drv = _driver()
            p = ctypes.c_void_p()
            _check(drv.cuMemHostAlloc(ctypes.byref(p), ctypes.c_size_t(self.nbytes + 128), 0x01 | 0x02),
                   "cuMemHostAlloc")
            self._raw = p
            base = int(p.value or 0)
            ctypes.memset(base, 0, self.nbytes + 128)
        else:
            self._raw = np.zeros(self.nbytes + 128, dtype=np.uint8)
            base = int(self._raw.ctypes.data)
        self.addr = (base + 127) & ~127
        self.array = np.ctypeslib.as_array((ctypes.c_uint8 * self.nbytes).from_address(self.addr))

    def u32(self, off: int = 0, count: int | None = None) -> np.ndarray:
        """A u32 view starting at byte ``off``."""
        c = (self.nbytes - off) // 4 if count is None else count
        return self.array[off: off + 4 * c].view("<u4")

    def free(self) -> None:
        """Release the memory (only after every user is gone)."""
        if self.cuda and self._raw is not None:  # pragma: no cover - GPU only
            _driver().cuMemFreeHost(self._raw)
        self._raw = None


def _ctrl_word(ctrl: HostBuffer, off: int) -> int:
    return int(ctrl.u32(off, 1)[0])


# ---- workers ------------------------------------------------------------------------------------------
class EmulatedWorker:
    """The resident kernel's protocol in a Python thread (CPU tests): same control words, same order, numpy
    arithmetic mod 2^32. ``stall`` stops answering; ``fail`` reports status 1."""

    def __init__(self, device: int, ctrl: HostBuffer, h_in: HostBuffer, h_out: HostBuffer,
                 table: Sequence[LinearEntry], *, max_rows_: int = MAX_ROWS):
        self.device = device
        self.ctrl, self.h_in, self.h_out = ctrl, h_in, h_out
        self.table = list(table)
        self.max_rows = max_rows_
        self.stall = False
        self.fail = False
        self.delay_s = 0.0  # tests: make trips overlap
        self.trips = 0
        self._w = [np.ascontiguousarray(e.w[:, : e.n].cpu().numpy().astype(np.int64)) for e in self.table]
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start polling."""
        self._thread = threading.Thread(target=self._loop, name=f"pgw-persistent-emu-{self.device}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        ctl = self.ctrl.u32()
        seen = int(ctl[CTRL_DONE // 4])
        while True:
            if ctl[CTRL_QUIT // 4]:
                return
            d = int(ctl[CTRL_DOORBELL // 4])
            if d == seen or self.stall:
                time.sleep(0)
                continue
            seen = d
            if self.delay_s:
                time.sleep(self.delay_s)
            li, rows = int(ctl[CTRL_LINEAR // 4]), int(ctl[CTRL_ROWS // 4])
            bad = 0 if (li < len(self.table) and 1 <= rows <= self.max_rows) else 1
            if not bad:
                e = self.table[li]
                x = self.h_in.u32(0, rows * e.n).astype(np.int64).reshape(rows, e.n)
                y = (x @ self._w[li].T) & 0xFFFFFFFF  # exact: |terms| < 2^39, n < 2^24
                out = self.h_out.u32(0, rows * e.m_total).reshape(rows, e.m_total)
                out[:, e.col0: e.col0 + e.m] = y.astype(np.uint32)
            self.trips += 1
            ctl[CTRL_STATUS // 4] = 1 if (bad or self.fail) else 0
            ctl[CTRL_DONE // 4] = seen

    def stop(self) -> None:
        """Set ``quit`` and join."""
        self.ctrl.u32()[CTRL_QUIT // 4] = 1
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class CudaWorker:  # pragma: no cover - GPU only
    """The resident kernel on one GPU (NVRTC-compiled :data:`KERNEL_SOURCE`, launched once on its own stream)."""

    BLOCK = 256

    def __init__(self, device: int, ctrl: HostBuffer, h_in: HostBuffer, h_out: HostBuffer,
                 table: Sequence[LinearEntry], *, max_rows_: int = MAX_ROWS):
        if not torch.cuda.is_available():
            raise BackendUnavailableError("the persistent worker needs CUDA")
        self.device = int(device)
        self.ctrl, self.h_in, self.h_out = ctrl, h_in, h_out
        self.table = list(table)
        self.max_rows = int(max_rows_)
        self.dev = torch.device("cuda", self.device)
        n_max = max(e.n for e in self.table)
        m_max = max(e.m for e in self.table)
        self.x_cap, self.y_cap = self.max_rows * n_max, self.max_rows * m_max
        with torch.cuda.device(self.dev):
            self.d_x = torch.zeros(self.x_cap + 64, dtype=torch.int32, device=self.dev)
            self.d_y = torch.zeros(self.y_cap + 64, dtype=torch.int32, device=self.dev)
            self.bar = torch.zeros(64, dtype=torch.int32, device=self.dev)
            desc = b"".join(e.pack() for e in self.table)
            self.lins = torch.frombuffer(bytearray(desc), dtype=torch.uint8).to(self.dev)
            self.stream = torch.cuda.Stream(self.dev, priority=-100)  # type: ignore[no-untyped-call]
        self._fn: Any = None
        self._args: Any = None
        self.grid = 0

    def start(self) -> None:
        drv = _driver()
        with torch.cuda.device(self.dev):
            torch.cuda.synchronize(self.dev)
            ctx = ctypes.c_void_p()
            _check(drv.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
            fn = _compile(self.device)
            sms = torch.cuda.get_device_properties(self.dev).multi_processor_count
            occ = ctypes.c_int()
            _check(drv.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(occ), fn, self.BLOCK,
                                                                      ctypes.c_size_t(0)), "occupancy")
            per_sm = min(_env_int("PGW_SERVICE_PERSISTENT_BLOCKS_PER_SM", 1, 1, 32), max(1, occ.value))
            self.grid = sms * per_sm
            seen = _ctrl_word(self.ctrl, CTRL_DONE)
            self.bar[2] = seen  # published doorbell starts at the last finished trip
            torch.cuda.synchronize(self.dev)
            vals = [ctypes.c_uint64(int(self.lins.data_ptr())), ctypes.c_uint32(len(self.table)),
                    ctypes.c_uint64(self.ctrl.addr), ctypes.c_uint64(self.h_in.addr), ctypes.c_uint64(self.h_out.addr),
                    ctypes.c_uint64(int(self.d_x.data_ptr())), ctypes.c_uint64(int(self.d_y.data_ptr())),
                    ctypes.c_uint64(int(self.bar.data_ptr())), ctypes.c_uint32(self.max_rows),
                    ctypes.c_uint32(self.x_cap), ctypes.c_uint32(self.y_cap)]
            ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
            self._args = (vals, ptrs)
            _check(drv.cuLaunchKernel(fn, self.grid, 1, 1, self.BLOCK, 1, 1, 0,
                                      ctypes.c_void_p(int(self.stream.cuda_stream)), ptrs, None), "cuLaunchKernel")
            self._fn = fn

    def stop(self, timeout_s: float = 5.0) -> None:
        self.ctrl.u32()[CTRL_QUIT // 4] = 1
        t0 = time.monotonic()
        while not self.stream.query():
            if time.monotonic() - t0 > timeout_s:
                log.error("persistent kernel on cuda:%d did not exit within %.1fs", self.device, timeout_s)
                return
            time.sleep(0.001)


# ---- driver / NVRTC (GPU only) -------------------------------------------------------------------------
_DRV: Any = None
_NVRTC: Any = None
_FUNCS: dict[int, Any] = {}


def _check(rc: int, what: str) -> None:  # pragma: no cover - GPU only
    if rc != 0:
        raise BackendUnavailableError(f"{what} failed (CUDA error {rc})")


def _driver() -> Any:  # pragma: no cover - GPU only
    global _DRV
    if _DRV is None:
        name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        drv = ctypes.CDLL(name)
        for fn in ("cuMemHostAlloc", "cuMemFreeHost", "cuCtxGetCurrent", "cuModuleLoadData", "cuModuleGetFunction",
                   "cuLaunchKernel", "cuOccupancyMaxActiveBlocksPerMultiprocessor"):
            if not hasattr(drv, fn):
                raise BackendUnavailableError(f"libcuda has no {fn}")
        drv.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p]
        drv.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                                     ctypes.c_size_t]
        drv.cuModuleGetFunction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        drv.cuModuleLoadData.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _DRV = drv
    return _DRV


def _nvrtc() -> Any:  # pragma: no cover - GPU only
    global _NVRTC
    if _NVRTC is None:
        cands: list[str] = []
        try:
            import nvidia  # the CUDA wheels torch depends on

            for root in list(getattr(nvidia, "__path__", [])):
                cands += sorted(glob.glob(os.path.join(root, "**", "libnvrtc.so*"), recursive=True), reverse=True)
        except ImportError:
            pass
        found = ctypes.util.find_library("nvrtc")
        if found:
            cands.append(found)
        cands = [c for c in cands if "builtins" not in os.path.basename(c)]
        last: Exception | None = None
        for c in cands:
            try:
                _NVRTC = ctypes.CDLL(c)
                break
            except OSError as exc:
                last = exc
        if _NVRTC is None:
            raise BackendUnavailableError(f"NVRTC not found ({last})")
    return _NVRTC


def _compile(device: int) -> Any:  # pragma: no cover - GPU only
    """Compile :data:`KERNEL_SOURCE` for ``device``'s architecture (cubin; PTX if this NVRTC has no cubin
    output) and return the ``pgw_persistent`` function handle (cached per device)."""
    if device in _FUNCS:
        return _FUNCS[device]
    nv, drv = _nvrtc(), _driver()
    major, minor = torch.cuda.get_device_capability(device)
    prog = ctypes.c_void_p()
    _check(nv.nvrtcCreateProgram(ctypes.byref(prog), KERNEL_SOURCE.encode(), b"pgw_persistent.cu", 0, None, None),
           "nvrtcCreateProgram")
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17", b"-default-device"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = nv.nvrtcCompileProgram(prog, len(opts), arr)
    if rc != 0:
        size = ctypes.c_size_t()
        nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        nv.nvrtcGetProgramLog(prog, buf)
        raise BackendUnavailableError(f"NVRTC compile failed: {buf.value.decode(errors='replace')[:2000]}")
    size = ctypes.c_size_t()
    if hasattr(nv, "nvrtcGetCUBINSize") and nv.nvrtcGetCUBINSize(prog, ctypes.byref(size)) == 0 and size.value:
        image = ctypes.create_string_buffer(size.value)
        _check(nv.nvrtcGetCUBIN(prog, image), "nvrtcGetCUBIN")
    else:
        _check(nv.nvrtcGetPTXSize(prog, ctypes.byref(size)), "nvrtcGetPTXSize")
        image = ctypes.create_string_buffer(size.value)
        _check(nv.nvrtcGetPTX(prog, image), "nvrtcGetPTX")
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    _check(drv.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
    _check(drv.cuModuleGetFunction(ctypes.byref(fn), mod, b"pgw_persistent"), "cuModuleGetFunction")
    _FUNCS[device] = fn
    return fn


# ---- groups ---------------------------------------------------------------------------------------------
@dataclass
class PersistentGroup:
    """The workers of one GPU (or one tensor-parallel group) and their shared mapped buffers.

    Build with :meth:`add_kernel` for every linear, then :meth:`start`. ``emulate`` runs
    :class:`EmulatedWorker` threads instead of CUDA kernels (CPU tests).
    """

    devices: list[int]
    emulate: bool = False
    rows: int = MAX_ROWS
    tables: list[list[LinearEntry]] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)
    shapes: list[tuple[int, int]] = field(default_factory=list)
    workers: list[Any] = field(default_factory=list)
    ctrls: list[HostBuffer] = field(default_factory=list)
    h_in: HostBuffer | None = None
    h_out: HostBuffer | None = None
    _seq: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tables:
            self.tables = [[] for _ in self.devices]

    def add_kernel(self, kernel: Any) -> int:
        """Add a :class:`~pgw.service.kernels.ComputeKernel` (one device) or
        :class:`~pgw.service.tensor_parallel.ShardedKernel` (one shard per device, in order); returns its
        table index (the same in every worker)."""
        if kernel.linear in self.index:
            return self.index[kernel.linear]
        shards = list(getattr(kernel, "shards", [kernel]))
        bounds = list(getattr(kernel, "bounds", [(0, kernel.out_features)]))
        if len(shards) != len(self.devices):
            raise ConfigError(f"{kernel.linear}: {len(shards)} shards for {len(self.devices)} workers")
        n, m = int(kernel.in_features), int(kernel.out_features)
        for j, (s, (m0, m1)) in enumerate(zip(shards, bounds, strict=True)):
            w = _shard_weight(s)
            self.tables[j].append(LinearEntry(w=w, n=n, m=m1 - m0, col0=m0, m_total=m))
        self.index[kernel.linear] = len(self.shapes)
        self.shapes.append((n, m))
        return self.index[kernel.linear]

    def start(self) -> None:
        """Allocate the mapped buffers and start one worker per device."""
        if not self.shapes:
            raise ConfigError("persistent group without linears")
        n_max = max(n for n, _ in self.shapes)
        m_max = max(m for _, m in self.shapes)
        cuda = not self.emulate
        self.h_in = HostBuffer(4 * self.rows * n_max, cuda=cuda)
        self.h_out = HostBuffer(4 * self.rows * m_max, cuda=cuda)
        for dev, table in zip(self.devices, self.tables, strict=True):
            ctrl = HostBuffer(CTRL_BYTES, cuda=cuda)
            cls = EmulatedWorker if self.emulate else CudaWorker
            w = cls(dev, ctrl, self.h_in, self.h_out, table, max_rows_=self.rows)
            self.ctrls.append(ctrl)
            self.workers.append(w)
            self._seq.append(0)
            w.start()

    @property
    def in_cap(self) -> int:
        """Bytes of the input buffer."""
        assert self.h_in is not None
        return self.h_in.nbytes

    @property
    def out_cap(self) -> int:
        """Bytes of the output buffer."""
        assert self.h_out is not None
        return self.h_out.nbytes

    def run(self, table: int, body: bytes, rows: int, timeout_s: float = 5.0) -> bytes:
        """One trip driven from Python (tests and benchmarks; serving goes through the lane's native
        executor, which implements the same steps). Not thread-safe."""
        assert self.h_in is not None and self.h_out is not None
        n, m = self.shapes[table]
        if len(body) != rows * n * 4 or not 1 <= rows <= self.rows:
            raise ValueError("bad trip shape")
        self.h_in.array[: len(body)] = np.frombuffer(body, dtype=np.uint8)
        want = []
        for j, c in enumerate(self.ctrls):
            ctl = c.u32()
            self._seq[j] = (max(self._seq[j], int(ctl[CTRL_DONE // 4])) + 1) & 0xFFFFFFFF
            ctl[CTRL_LINEAR // 4] = table
            ctl[CTRL_ROWS // 4] = rows
            ctl[CTRL_DOORBELL // 4] = self._seq[j]
            want.append(self._seq[j])
        t0 = time.perf_counter()
        for c, s in zip(self.ctrls, want, strict=True):
            ctl = c.u32()
            while int(ctl[CTRL_DONE // 4]) != s:
                if time.perf_counter() - t0 > timeout_s:
                    raise TimeoutError("persistent worker did not answer")
            if ctl[CTRL_STATUS // 4]:
                raise RuntimeError(f"persistent worker status {int(ctl[CTRL_STATUS // 4])}")
        return bytes(self.h_out.array[: rows * m * 4])

    def stop(self) -> None:
        """Stop the workers (the lane must have forgotten them first: ``clear_persistent``)."""
        for w in self.workers:
            w.stop()
        self.workers.clear()


def _shard_weight(kern: Any) -> torch.Tensor:
    """The int8 weight a worker reads for one (shard) kernel: the padded device copy on CUDA (row stride =
    padded n), else the CPU tensor."""
    pw = getattr(kern, "_pw", None)
    w = getattr(pw, "_w_dev", None) if pw is not None else None
    if w is None:
        w = getattr(kern, "_w", None)
    if w is None and pw is not None:
        w = getattr(pw, "w8", None) or getattr(pw, "_w8", None)
    if w is None:
        raise ConfigError(f"{kern.linear}: no int8 weight to serve persistently")
    if w.dtype != torch.int8 or w.ndim != 2 or w.stride(1) != 1:
        raise ConfigError(f"{kern.linear}: persistent worker needs a row-major int8 weight")
    return cast(torch.Tensor, w)


# ---- lane integration -------------------------------------------------------------------------------------
@dataclass
class PersistentReport:
    """Outcome of :func:`start_lane_persistent`."""

    groups: int = 0
    linears: int = 0
    checked: int = 0
    disabled: str | None = None
    seconds: float = 0.0


def start_lane_persistent(mgr: Any, *, emulate: bool = False, inject_mismatch: bool = False,
                          rows_list: Sequence[int] | None = None) -> PersistentReport:
    """Start persistent workers for every model the lane serves (one group per distinct device set), register
    them with the native lane and self-check every linear through the native serving path against the
    digit-GEMM path. Any error or mismatch disables the persistent path (calls use graphs / the kernel path).

    Only K = 32 rings qualify (the packed body is the u32 the kernel reads). ``emulate`` uses
    :class:`EmulatedWorker` (CPU tests); otherwise CUDA kernels are required.
    """
    from pgw.core.packing import pack_ring

    rep = PersistentReport()
    t0 = time.perf_counter()
    groups: dict[tuple[int, ...], PersistentGroup] = {}
    gid: dict[tuple[int, ...], int] = {}
    rows_cap = max_rows()
    entries: list[tuple[int, int, int, Any, Any, tuple[int, ...]]] = []
    try:
        for fp, slot in mgr._slots.items():
            t = mgr._tables[slot]
            model = mgr.state.registry.get(fp)
            for li, name in enumerate(t.linears):
                kern = model.kernel(name)
                devs = [torch.device(d) for d in getattr(kern, "devices", [kern.device])]
                if not emulate and any(d.type != "cuda" for d in devs):
                    continue
                key = tuple(int(d.index or 0) for d in devs) if not emulate else tuple(range(len(devs)))
                for ri, ring in enumerate(t.rings):
                    if ring.k != 32:
                        continue
                    g = groups.get(key)
                    if g is None:
                        g = groups[key] = PersistentGroup(list(key), emulate=emulate, rows=rows_cap)
                        gid[key] = len(gid)
                    g.add_kernel(kern)
                    entries.append((slot, li, ri, kern, ring, key))
        # reference results first: the kernel path must not have to share the GPUs with spinning workers
        rng = np.random.default_rng(0x9E35)
        cases: list[list[tuple[int, bytes, bytes]]] = []
        for _slot, _li, _ri, kern, ring, _key in entries:
            n = int(kern.in_features)
            mine: list[tuple[int, bytes, bytes]] = []
            for rows in (rows_list or range(1, rows_cap + 1)):
                for case in range(3):
                    if case == 0:
                        x = torch.from_numpy(rng.integers(0, 1 << 32, size=(rows, n), dtype=np.int64))
                    else:
                        x = torch.full((rows, n), 0 if case == 2 else (1 << 32) - 1, dtype=torch.int64)
                    want = pack_ring(kern.run_compact(x, ring), ring)
                    if inject_mismatch:
                        want = bytes([want[0] ^ 1]) + want[1:]
                    mine.append((rows, pack_ring(x, ring), want))
            cases.append(mine)
        for key, g in groups.items():
            g.start()
            mgr.svc.register_persistent_group(gid[key], list(zip(g.devices, [c.addr for c in g.ctrls], strict=True)),
                                              g.h_in.addr, g.in_cap, g.h_out.addr, g.out_cap)  # type: ignore[union-attr]
            rep.groups += 1
        mgr._persistent = list(groups.values())
        tmo = _env_int("PGW_SERVICE_PERSISTENT_TIMEOUT_MS", 2000, 1, 600_000)
        mgr.svc.set_persistent_timeout_us(tmo * 1000)
        for (slot, li, ri, kern, _ring, key), mine in zip(entries, cases, strict=True):
            g = groups[key]
            n, m = int(kern.in_features), int(kern.out_features)
            mgr.svc.register_persistent_linear(slot, li, ri, gid[key], g.index[kern.linear], n, m, rows_cap)
            rep.linears += 1
            for i, (rows, body, want) in enumerate(mine):
                if bytes(mgr.svc.run_persistent(slot, li, ri, rows, body)) != want:
                    rep.disabled = (f"persistent trip differs from the kernel path ({kern.linear}, rows={rows}, "
                                    f"case {i % 3})")
                    break
            if rep.disabled:
                break
            rep.checked += 1
    except Exception as exc:
        rep.disabled = f"persistent workers failed: {type(exc).__name__}: {str(exc)[:200]}"
    if rep.disabled:
        mgr.svc.disable_persistent(rep.disabled)
        log.error("lane persistent workers disabled: %s; lane calls use graphs or the kernel path", rep.disabled)
    rep.seconds = time.perf_counter() - t0
    return rep


def stop_lane_persistent(mgr: Any) -> None:
    """Forget the workers in the lane (waits for in-flight trips), then stop their kernels."""
    groups = getattr(mgr, "_persistent", None) or []
    if mgr.svc is not None and hasattr(mgr.svc, "clear_persistent"):
        mgr.svc.clear_persistent()
    for g in groups:
        g.stop()
    mgr._persistent = []
