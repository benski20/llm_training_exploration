"""Fused exact ring products over ``Z/2^K`` for ``K <= 64`` (the fused2 provider op), as Triton kernels.

The fused2 provider op (``research/plans/fused2-two-trips-per-layer.md``) is, on masked ring elements only::

    pair:   H = (R + hmul * (W1 x)) mod 2^K,   Z = (W2' H) mod 2^K       (W1, W2' public int8, hmul public)
    single: Y = (W1 x) mod 2^K

with ``K = 56`` for the migrated reference (7 bytes per element). The harness of 2026-11-23 ran these as eager
digit GEMMs (:class:`pgw.core.ringmm.PreparedWeight`: ~a dozen launches per product, host round trips for the
add and the multiply): 0.97 ms of provider compute per 7B pair trip. Here one trip is a handful of kernels that
a CUDA graph captures together with the copies (:class:`PairOp`):

``unpack``   packed little-endian ``K/8``-byte elements -> int64 (the body as the lane carries it);
``gemv``     split-K partial products ``x W^T mod 2^64`` (``sum``: uint64 multiply-adds on CUDA cores, the
             memory-bound one/few-row kernel; ``dot``: ``K/8`` balanced signed base-256 digit planes of
             ``x``, one int8 tensor-core ``tl.dot`` per plane and tile, int32-exact per tile, recombined in
             wrapping uint64);
``epilogue`` sum of the partials, ``* hmul``, ``+ R``, ``mod 2^K``; writes the int64 ``H`` for stage 2 and
             the packed reply bytes.

**Exactness.** ``Z_{2^K}`` is a quotient of ``Z_{2^64}``. Inputs are read as uint64, int8 weights are
sign-extended to 64 bits (the same residue), and every product, partial sum, cross-split sum, multiplier
product and residual add is a wrapping unsigned 64-bit operation: associative and commutative, so tile shape,
split-K, warp reductions or digit order cannot change the bits. The ``dot`` variant's digit products are
exact int32 integers per tile (``|d| <= 128``, ``|w| <= 128``, ``BLOCK_N <= 2^16``) and are recombined mod
``2^64``. Only the final mask reduces to ``2^K``.

The kernels serve any ``K`` that is a multiple of 8 up to 64 (the epilogue packs ``K/8`` bytes). Triton is
optional; with ``TRITON_INTERPRET=1`` (``tests/conftest.py`` on hosts without CUDA) the same kernels run on CPU
tensors. :func:`self_check` compares every variant and configuration with the Python-int reference
(:func:`pgw.core.ringmm.ringmm_reference`) before a service uses them.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ShapeError

__all__ = [
    "PairConfig",
    "PairOp",
    "digit_planes",
    "gemm_planes",
    "gemv64",
    "pair_config",
    "pair_reference",
    "self_check",
    "unpack64",
]

log = logging.getLogger(__name__)

#: Largest tile inner size of the ``dot`` variant (keeps each tile's int32 digit product exact).
_DOT_MAX_BLOCK_N = 1 << 16
#: Largest split length of the ``planes`` variant: its int32 accumulator holds ``sum_k W d`` over a whole split,
#: ``|.| <= 128 * 128 * n_per_split``, exact while ``n_per_split <= (2^31 - 1) // 2^14``.
PLANES_MAX_SPLIT_N = ((1 << 31) - 1) // (1 << 14)
#: Most rows of the ``planes`` variant (8 digit columns per row, ``8 * rows_pow2 <= 128``).
PLANES_MAX_ROWS = 16
VARIANTS = ("sum", "dot", "planes")
#: ``PGW_SERVICE_FUSED2_VARIANT``: ``auto`` (default; ``dot`` for >= 2 rows on devices with int8 tensor cores,
#: ``sum`` for one row), ``sum``, ``dot`` or ``planes`` everywhere (``planes``: opt-in, the digit-plane
#: tensor-core GEMV with the fused unpack / epilogue, :func:`gemm_planes`).
ENV_VARIANT = "PGW_SERVICE_FUSED2_VARIANT"


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


@functools.cache
def _kernels(interpret: bool) -> dict[str, Any]:
    del interpret  # cache key only (triton.jit picks the mode when it wraps)
    import triton

    from pgw.core import _ring64_kernel as k

    return {name: triton.jit(getattr(k, name)) for name in
            ("unpack_kernel", "gemv64_sum_kernel", "gemv64_dot_kernel", "epilogue_kernel", "unpack_planes_kernel",
             "gemm_planes_kernel")}


def _kern(name: str) -> Any:
    return _kernels(_interpret())[name]


@functools.cache
def _sm_count(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


@dataclass(frozen=True)
class PairConfig:
    """Launch configuration of one product: tile sizes, split-K, warps and variant (``sum`` or ``dot``)."""

    block_m: int = 16
    block_n: int = 256
    split: int = 1
    num_warps: int = 4
    variant: str = "sum"

    def block_r(self, rows: int) -> int:
        b = 1
        while b < rows:
            b *= 2
        if self.variant == "dot":
            return max(16, b)
        if self.variant == "planes":  # 8 digit columns per row; tl.dot needs >= 16 columns
            return max(2, b)
        return b

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ShapeError(f"ring64: unknown variant {self.variant!r}")
        if self.variant == "planes" and (self.block_m < 16 or self.block_n < 32):
            # int8 tl.dot on GPUs needs K >= 32 (block_n is the K tile of the weight x digit-plane product)
            raise ShapeError("ring64: the planes variant needs block_m >= 16 and block_n >= 32")
        for v in (self.block_m, self.block_n):
            if v < 1 or v & (v - 1):
                raise ShapeError("ring64: tile sizes must be powers of two")
        if self.variant == "dot" and (self.block_m < 16 or self.block_n < 16 or self.block_n > _DOT_MAX_BLOCK_N):
            raise ShapeError("ring64: the dot variant needs 16 <= block sizes, block_n <= 2^16")
        if self.split < 1:
            raise ShapeError("ring64: split must be >= 1")


def _variant_for(rows: int, device: torch.device) -> str:
    v = os.environ.get(ENV_VARIANT, "auto").strip().lower()
    if v in VARIANTS:
        return v
    if device.type == "cuda" and rows >= 2 and torch.cuda.get_device_capability(device)[0] >= 8:
        return "dot"
    return "sum"


def pair_config(rows: int, m: int, n: int, device: torch.device, variant: str | None = None) -> PairConfig:
    """Heuristic launch configuration for a ``(rows, n) x (m, n)^T`` product on ``device``.

    Aims at about four programs per SM (memory-bound: every weight byte is streamed once per call).
    """
    variant = variant or _variant_for(rows, device)
    if variant == "planes":
        return _planes_config(rows, m, n, device)
    if variant == "dot":
        bm, bn, warps = 32, 128, 4
    else:
        bm, bn, warps = (16, 256, 4) if rows <= 2 else (8, 128, 4)
    sms = _sm_count(int(device.index or 0)) if device.type == "cuda" else 4
    progs_m = -(-m // bm)
    want = max(1, -(-4 * sms // progs_m))
    split = max(1, min(want, -(-n // bn), 64))
    return PairConfig(bm, bn, split, warps, variant)


def _planes_config(rows: int, m: int, n: int, device: torch.device) -> PairConfig:
    """``planes`` heuristic: 64-row weight tiles (one Hopper warpgroup MMA), 256-byte K steps, split-K until about
    two programs per SM stream weights (memory-bound GEMV), splits no shorter than 4 K steps."""
    bm = 64 if rows <= 4 else 128
    bk = 256
    sms = _sm_count(int(device.index or 0)) if device.type == "cuda" else 4
    progs_m = -(-m // bm)
    want = max(1, -(-2 * sms // progs_m))
    split = max(1, min(want, -(-n // (4 * bk)), 64))
    split = max(split, -(-n // PLANES_MAX_SPLIT_N))
    return PairConfig(bm, bk, split, 4, "planes")


def _per_split(n: int, cfg: PairConfig) -> tuple[int, int]:
    per = -(-(-(-n // cfg.split)) // cfg.block_n) * cfg.block_n
    if cfg.variant == "planes" and per > PLANES_MAX_SPLIT_N:  # keep the int32 accumulator exact
        per = (PLANES_MAX_SPLIT_N // cfg.block_n) * cfg.block_n
    return per, -(-n // per)


def gemv64(x: torch.Tensor, w: torch.Tensor, cfg: PairConfig, part: torch.Tensor | None = None,
           ndig: int = 8) -> torch.Tensor:
    """Split-K partial products ``part[s] = x W^T`` of int64 ``x (b, n)`` and int8 ``w (m, n)``.

    Returns int64 ``(splits, b, m)``; their wrapping sum is the product mod ``2^64`` (``sum``) or mod
    ``2^(8 ndig)`` on the low bits (``dot`` with ``ndig`` digit planes; ``ndig = K/8`` suffices for a result
    used mod ``2^K``).
    """
    cfg.validate()
    if x.dtype != torch.int64 or w.dtype != torch.int8 or x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ShapeError(f"gemv64 needs int64 x (b, n), int8 w (m, n); got {tuple(x.shape)} {tuple(w.shape)}")
    if x.stride(1) != 1 or w.stride(1) != 1:
        raise ShapeError("gemv64: unit column stride required")
    b, n = (int(s) for s in x.shape)
    m = int(w.shape[0])
    per, splits = _per_split(n, cfg)
    if part is None:
        part = torch.empty((splits, b, m), dtype=torch.int64, device=w.device)
    elif tuple(part.shape[:1]) < (splits,) or part.dtype != torch.int64:
        raise ShapeError("gemv64: partial buffer too small")
    br = cfg.block_r(b)
    if br > 16 and cfg.variant == "sum":
        raise ShapeError("gemv64: the sum variant serves at most 16 rows")
    grid = (-(-m // cfg.block_m), splits)
    if cfg.variant == "dot":
        _kern("gemv64_dot_kernel")[grid](x, w, part, b, n, m, per, x.stride(0), w.stride(0), b * m,
                                         BLOCK_R=br, BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, NDIG=int(ndig),
                                         num_warps=cfg.num_warps)
    else:
        _kern("gemv64_sum_kernel")[grid](x, w, part, b, n, m, per, x.stride(0), w.stride(0), b * m,
                                         BLOCK_R=br, BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n,
                                         num_warps=cfg.num_warps)
    return part[:splits]


def digit_planes(x: torch.Tensor, rp: int | None = None) -> torch.Tensor:
    """Host-side digit planes of int64 ``x (b, n)`` (tests): int8 ``(8 rp, n)``, row ``r * 8 + i`` = digit ``i`` of
    row ``r`` (balanced, ``[-128, 127]``, ``x == sum_i d_i 256^i mod 2^64``); rows ``>= b`` zero."""
    b, n = (int(s) for s in x.shape)
    rp = rp or PairConfig(variant="planes").block_r(b)
    v = x.cpu().numpy().astype(np.int64).view(np.uint64).copy()
    out = np.zeros((rp, 8, n), dtype=np.int8)
    for i in range(8):
        d = ((v & np.uint64(0xFF)).astype(np.int64) ^ 0x80) - 0x80
        out[:b, i] = d.astype(np.int8)
        v = (v - d.astype(np.uint64)) >> np.uint64(8)
    return torch.from_numpy(out.reshape(rp * 8, n))


def gemm_planes(planes: torch.Tensor, rows: int, w: torch.Tensor, cfg: PairConfig, part: torch.Tensor | None = None
                ) -> torch.Tensor:
    """Split-K partials ``part[s] = x W^T mod 2^64`` of the ``planes`` variant (``planes``: int8 ``(>= 8 rp, n)`` digit
    planes of the ``rows`` input rows, :func:`digit_planes`; ``w`` int8 ``(m, n)``).
    Returns int64 ``(splits, rows, m)``.
    """
    cfg.validate()
    if cfg.variant != "planes" or planes.dtype != torch.int8 or w.dtype != torch.int8 or planes.shape[1] != w.shape[1]:
        raise ShapeError("gemm_planes needs int8 digit planes (8 rp, n) and int8 w (m, n) with a planes config")
    n, m = int(w.shape[1]), int(w.shape[0])
    rp = cfg.block_r(rows)
    if rp > PLANES_MAX_ROWS or int(planes.shape[0]) < 8 * rp:
        raise ShapeError("gemm_planes: too many rows / too few digit planes")
    per, splits = _per_split(n, cfg)
    if part is None:
        part = torch.empty((splits, rows, m), dtype=torch.int64, device=w.device)
    _launch_planes(planes, rows, w, cfg, per, splits, part=part)
    return part[:splits]


def _launch_planes(planes: torch.Tensor, rows: int, w: torch.Tensor, cfg: PairConfig, per: int, splits: int, *,
                   part: torch.Tensor | None = None, out_u8: torch.Tensor | None = None, k: int = 64,
                   r: torch.Tensor | None = None, hmul: torch.Tensor | None = None,
                   hp: torch.Tensor | None = None) -> None:
    n, m = int(w.shape[1]), int(w.shape[0])
    fused = part is None
    if fused and splits != 1:
        raise ShapeError("ring64: the fused planes epilogue needs one split")
    dummy = w
    grid = (-(-m // cfg.block_m), splits)
    _kern("gemm_planes_kernel")[grid](
        w, planes, dummy if part is None else part, dummy if r is None else r, dummy if hmul is None else hmul,
        dummy if hp is None else hp, dummy if out_u8 is None else out_u8,
        rows, n, m, per, w.stride(0), planes.stride(0), rows * m,
        RP=cfg.block_r(rows), BLOCK_M=cfg.block_m, BLOCK_K=cfg.block_n,
        FUSED=fused, HAS_R=r is not None, HAS_HMUL=hmul is not None, STORE_PLANES=hp is not None, BPE=k // 8,
        num_warps=cfg.num_warps)


def unpack64(src: torch.Tensor, dst: torch.Tensor, bpe: int) -> torch.Tensor:
    """Unpack ``dst.numel()`` little-endian ``bpe``-byte elements of uint8 ``src`` into int64 ``dst``."""
    numel = int(dst.numel())
    if numel:
        blk = 1024
        _kern("unpack_kernel")[(-(-numel // blk),)](src, dst, numel, BPE=bpe, BLOCK=blk)
    return dst


def _epilogue(part: torch.Tensor, splits: int, b: int, m: int, k: int, out_u8: torch.Tensor, *,
              r: torch.Tensor | None = None, hmul: torch.Tensor | None = None, h: torch.Tensor | None = None,
              hp: torch.Tensor | None = None) -> None:
    numel = b * m
    if not numel:
        return
    blk = 512
    dummy = part
    _kern("epilogue_kernel")[(-(-numel // blk),)](
        part, splits, numel, m,
        dummy if r is None else r, dummy if hmul is None else hmul, dummy if h is None else h, out_u8,
        dummy if hp is None else hp,
        HAS_R=r is not None, HAS_HMUL=hmul is not None, STORE_H=h is not None, BPE=k // 8, BLOCK=blk,
        STORE_PLANES=hp is not None)


def pair_reference(k: int, w1: torch.Tensor, x: torch.Tensor, w2: torch.Tensor | None = None,
                   r: torch.Tensor | None = None, hmul: torch.Tensor | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Python-int reference of the op (tests and the self-check): ``(H, Z)`` or ``(Y, None)``, int64 in [0, 2^K)."""
    mod = 1 << k

    def u(t: torch.Tensor) -> np.ndarray:  # ring elements as Python ints in [0, 2^64)
        return np.ascontiguousarray(t.cpu().numpy().astype(np.int64)).view(np.uint64).astype(object)

    def to_t(v: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((v % mod).astype(np.uint64).view(np.int64))

    def mm(a: np.ndarray, w: torch.Tensor) -> np.ndarray:
        return (a @ w.cpu().numpy().astype(np.int64).astype(object).T) % mod

    y = mm(u(x), w1)
    if w2 is None:
        return to_t(y), None
    if hmul is not None:
        y = y * hmul.cpu().numpy().astype(object)[None, :]
    assert r is not None
    hv = (y + u(r)) % mod
    return to_t(hv), to_t(mm(hv, w2))


class PairOp:
    """One fused2 provider op on one device: a pair ``(W1, hmul, W2')`` or a single ``W1``.

    Serves packed bodies exactly as the lane carries them: ``body = pack(x_hat) || pack(r_hat)`` (pair) or
    ``pack(x_hat)`` (single), each row-major ``K/8``-byte little-endian elements; the reply is
    ``pack(H_hat) || pack(Z_hat)`` (pair) or ``pack(Y_hat)``. Device buffers are sized for ``max_rows``.
    :meth:`run` is the eager path; :meth:`capture` returns a CUDA graph of ``[H2D, unpack, gemv, epilogue,
    (gemv, epilogue), D2H]`` reading and writing pinned host buffers (no Python per call when the lane replays
    it).
    """

    def __init__(self, k: int, w1: torch.Tensor, w2: torch.Tensor | None = None, hmul: torch.Tensor | None = None,
                 *, device: torch.device | str = "cpu", max_rows: int = 8, variant: str | None = None):
        if k % 8 or not 8 <= k <= 64:
            raise ShapeError(f"ring64: K must be a multiple of 8 in [8, 64], got {k}")
        self.k = int(k)
        self.bpe = self.k // 8
        self.device = torch.device(device)
        self.w1 = w1.to(self.device, torch.int8).contiguous()
        self.w2 = None if w2 is None else w2.to(self.device, torch.int8).contiguous()
        self.m1, self.n1 = (int(s) for s in self.w1.shape)
        if self.w2 is not None and int(self.w2.shape[1]) != self.m1:
            raise ShapeError(f"ring64 pair: W2' has {int(self.w2.shape[1])} inputs, W1 {self.m1} outputs")
        self.m2 = 0 if self.w2 is None else int(self.w2.shape[0])
        self.hmul = None
        if hmul is not None and self.w2 is not None:
            hm = torch.as_tensor(hmul, dtype=torch.int64)
            if tuple(hm.shape) != (self.m1,):
                raise ShapeError("ring64 pair: hmul must have one entry per W1 row")
            self.hmul = hm.to(self.device).contiguous()
        self.max_rows = int(max_rows)
        self._variant = variant
        self._cfg: dict[int, tuple[PairConfig, PairConfig | None]] = {}
        r = self.max_rows
        dev = self.device
        self.in_elems = self.n1 + (self.m1 if self.pair else 0)
        self.out_elems = self.m1 + self.m2
        self.d_in = torch.zeros(r * self.in_elems * self.bpe, dtype=torch.uint8, device=dev)
        self.d_x = torch.zeros(r * self.in_elems, dtype=torch.int64, device=dev)
        self.d_h = torch.zeros(r * self.m1, dtype=torch.int64, device=dev)
        self.d_out = torch.zeros(r * self.out_elems * self.bpe, dtype=torch.uint8, device=dev)
        s1 = max(self.config(rr)[0].split for rr in range(1, r + 1))
        s2 = max((c[1].split for c in (self.config(rr) for rr in range(1, r + 1)) if c[1] is not None), default=1)
        self.d_p1 = torch.zeros((s1 * r * self.m1,), dtype=torch.int64, device=dev)
        self.d_p2 = torch.zeros((s2 * r * max(1, self.m2),), dtype=torch.int64, device=dev)
        self._planes_bufs: tuple[torch.Tensor, torch.Tensor | None] | None = None

    @property
    def pair(self) -> bool:
        return self.w2 is not None

    def in_bytes(self, rows: int) -> int:
        return rows * self.in_elems * self.bpe

    def out_bytes(self, rows: int) -> int:
        return rows * self.out_elems * self.bpe

    def config(self, rows: int) -> tuple[PairConfig, PairConfig | None]:
        hit = self._cfg.get(rows)
        if hit is None:
            c1 = pair_config(rows, self.m1, self.n1, self.device, self._variant)
            c2 = None if self.w2 is None else pair_config(rows, self.m2, self.m1, self.device, self._variant)
            hit = self._cfg[rows] = (c1, c2)
        return hit

    def set_config(self, rows: int, c1: PairConfig, c2: PairConfig | None) -> None:
        """Pin the launch configuration of ``rows`` (before :meth:`capture`; grows the partial buffers)."""
        c1.validate()
        if c2 is not None and (c1.variant == "planes") != (c2.variant == "planes"):
            raise ShapeError("ring64: the planes variant serves both stages of a pair or neither")
        need1 = _per_split(self.n1, c1)[1] * rows * self.m1
        if need1 > self.d_p1.numel():
            self.d_p1 = torch.zeros(need1, dtype=torch.int64, device=self.device)
        if c2 is not None:
            c2.validate()
            need2 = _per_split(self.m1, c2)[1] * rows * self.m2
            if need2 > self.d_p2.numel():
                self.d_p2 = torch.zeros(need2, dtype=torch.int64, device=self.device)
        self._cfg[rows] = (c1, c2)

    def _planes(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Digit-plane buffers of the ``planes`` variant (zeroed once; allocated on first use, i.e. in the
        warm-up before a capture): int8 ``(8 rp_max, n1)`` for ``x`` and ``(8 rp_max, m1)`` for ``H``."""
        if self._planes_bufs is None:
            rp = PairConfig(variant="planes").block_r(self.max_rows)
            if rp > PLANES_MAX_ROWS:
                raise ShapeError(f"ring64: the planes variant serves at most {PLANES_MAX_ROWS} rows")
            xp = torch.zeros((8 * rp, self.n1), dtype=torch.int8, device=self.device)
            hp = torch.zeros((8 * rp, self.m1), dtype=torch.int8, device=self.device) if self.pair else None
            self._planes_bufs = (xp, hp)
        return self._planes_bufs

    def _compute_planes(self, rows: int, c1: PairConfig, c2: PairConfig | None) -> None:
        """``planes`` variant: fused unpack to digit planes (+ int64 ``R``), one tensor-core GEMV per stage with
        its epilogue fused when unsplit (else a partial + epilogue launch); stage 1 writes ``H`` as stage 2's
        digit planes. 3-5 kernels per pair."""
        xp, hp = self._planes()
        nx, total = rows * self.n1, rows * self.in_elems
        blk = 1024
        _kern("unpack_planes_kernel")[(-(-total // blk),)](self.d_in, xp, self.d_x, nx, self.n1, total,
                                                         BPE=self.bpe, BLOCK=blk)
        r = self.d_x[: rows * self.m1] if self.pair else None
        hp1 = hp if self.pair else None
        per1, sp1 = _per_split(self.n1, c1)
        if sp1 == 1:
            _launch_planes(xp, rows, self.w1, c1, per1, 1, out_u8=self.d_out, k=self.k, r=r, hmul=self.hmul, hp=hp1)
        else:
            p1 = self.d_p1[: sp1 * rows * self.m1].view(sp1, rows, self.m1)
            _launch_planes(xp, rows, self.w1, c1, per1, sp1, part=p1)
            _epilogue(p1, sp1, rows, self.m1, self.k, self.d_out, r=r, hmul=self.hmul, hp=hp1)
        if not self.pair:
            return
        assert self.w2 is not None and c2 is not None and hp is not None
        out2 = self.d_out[rows * self.m1 * self.bpe:]
        per2, sp2 = _per_split(self.m1, c2)
        if sp2 == 1:
            _launch_planes(hp, rows, self.w2, c2, per2, 1, out_u8=out2, k=self.k)
        else:
            p2 = self.d_p2[: sp2 * rows * self.m2].view(sp2, rows, self.m2)
            _launch_planes(hp, rows, self.w2, c2, per2, sp2, part=p2)
            _epilogue(p2, sp2, rows, self.m2, self.k, out2)

    def _compute(self, rows: int) -> None:
        """Device part of one call on ``d_in[:in_bytes(rows)]`` -> ``d_out[:out_bytes(rows)]``."""
        c1, c2 = self.config(rows)
        if c1.variant == "planes":
            self._compute_planes(rows, c1, c2)
            return
        x_all = self.d_x[: rows * self.in_elems]
        unpack64(self.d_in, x_all, self.bpe)
        x = x_all[: rows * self.n1].view(rows, self.n1)
        _, sp1 = _per_split(self.n1, c1)
        p1 = gemv64(x, self.w1, c1, self.d_p1[: sp1 * rows * self.m1].view(sp1, rows, self.m1), self.bpe)
        if not self.pair:
            _epilogue(p1, sp1, rows, self.m1, self.k, self.d_out)
            return
        assert self.w2 is not None and c2 is not None
        r = x_all[rows * self.n1:]
        h = self.d_h[: rows * self.m1]
        _epilogue(p1, sp1, rows, self.m1, self.k, self.d_out, r=r, hmul=self.hmul, h=h)
        _, sp2 = _per_split(self.m1, c2)
        p2 = gemv64(h.view(rows, self.m1), self.w2, c2, self.d_p2[: sp2 * rows * self.m2].view(sp2, rows, self.m2),
                    self.bpe)
        _epilogue(p2, sp2, rows, self.m2, self.k, self.d_out[rows * self.m1 * self.bpe:])

    def run(self, rows: int, body: bytes | bytearray | memoryview) -> bytes:
        """Eager call: packed body in, packed reply out (the lane's Python executor and the self-check)."""
        if not 1 <= rows <= self.max_rows:
            raise ShapeError(f"ring64: rows {rows} outside 1..{self.max_rows}")
        if len(body) != self.in_bytes(rows):
            raise ShapeError(f"ring64: body has {len(body)} bytes, expected {self.in_bytes(rows)}")
        if self.device.type == "cpu" and _cpu_exact():
            return self._run_cpu(rows, body)
        src = torch.frombuffer(bytearray(body), dtype=torch.uint8)
        self.d_in[: len(body)].copy_(src)
        self._compute(rows)
        return bytes(self.d_out[: self.out_bytes(rows)].cpu().numpy().tobytes())

    def _run_cpu(self, rows: int, body: bytes | bytearray | memoryview) -> bytes:
        """The CPU provider's op without the Triton interpreter: exact ``W x mod 2^K`` by
        :class:`pgw.core.ringmm.PreparedWeight` (digit GEMMs in float64 whose every partial sum is an exact
        integer), the epilogue in wrapping int64. The same bytes as the kernels (the start-up self-check compares
        it with the Python-int reference); CPU test providers and benchmarks only (a GPU runs the kernels)."""
        from pgw.core.packing import pack_ring, unpack_ring
        from pgw.core.ringmm import PreparedWeight

        if getattr(self, "_pw1", None) is None:
            self._pw1 = PreparedWeight(self.w1.cpu(), cache_float=True)
            self._pw2 = None if self.w2 is None else PreparedWeight(self.w2.cpu(), cache_float=True)
        k, mask = self.k, (1 << self.k) - 1
        nx = rows * self.n1 * self.bpe
        x = unpack_ring(memoryview(body)[:nx], (rows, self.n1), k)
        y = self._pw1.matmul(x, k)
        if self.w2 is None:
            return pack_ring(y, k)
        r = unpack_ring(memoryview(body)[nx:], (rows, self.m1), k)
        if self.hmul is not None:
            y = (y * self.hmul.cpu()) & mask
        h = (r + y) & mask
        assert self._pw2 is not None
        return pack_ring(h, k) + pack_ring(self._pw2.matmul(h, k), k)

    def capture(self, rows: int, h_in: torch.Tensor, h_out: torch.Tensor, *, stream: Any, pool: Any) -> Any:
        """CUDA graph of one call reading pinned uint8 ``h_in[:in_bytes(rows)]`` and writing ``h_out``."""  # pragma: no cover
        nin, nout = self.in_bytes(rows), self.out_bytes(rows)

        def body() -> None:
            self.d_in[:nin].copy_(h_in[:nin], non_blocking=True)
            self._compute(rows)
            h_out[:nout].copy_(self.d_out[:nout], non_blocking=True)

        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            body()  # warm-up: compiles the kernels outside the capture
        stream.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool, stream=stream, capture_error_mode="thread_local"):
            body()
        return g


def _cpu_exact() -> bool:
    """CPU ops use the exact float64-digit path unless ``PGW_RING64_CPU=triton`` (the kernels in the Triton
    interpreter: the GPU code path, for its CPU tests)."""
    return os.environ.get("PGW_RING64_CPU", "exact").strip().lower() != "triton"


def _rand_ring(gen: np.random.Generator, shape: tuple[int, ...], k: int) -> torch.Tensor:
    v = gen.integers(0, 1 << 63, size=shape, dtype=np.int64).astype(np.uint64) * np.uint64(2)
    v |= gen.integers(0, 2, size=shape, dtype=np.int64).astype(np.uint64)
    if k < 64:
        v &= np.uint64((1 << k) - 1)
    return torch.from_numpy(v.view(np.int64))


def check_op(op: PairOp, rows: int, *, seed: int = 0, extreme: bool = False) -> str | None:
    """Run ``op`` on random (or extreme) ring inputs and compare with :func:`pair_reference`; None if equal."""
    from pgw.core.packing import pack_ring

    gen = np.random.default_rng(seed)
    k = op.k
    if extreme:
        top = (1 << k) - 1
        x = torch.from_numpy(np.full((rows, op.n1), top, dtype=np.uint64).view(np.int64))
        r = torch.from_numpy(np.full((rows, op.m1), top, dtype=np.uint64).view(np.int64))
    else:
        x = _rand_ring(gen, (rows, op.n1), k)
        r = _rand_ring(gen, (rows, op.m1), k)
    body = pack_ring(x, k) + (pack_ring(r, k) if op.pair else b"")
    got = op.run(rows, body)
    hm = None if op.hmul is None else op.hmul.cpu()
    h, z = pair_reference(k, op.w1.cpu(), x, None if op.w2 is None else op.w2.cpu(), r if op.pair else None, hm)
    want = pack_ring(h, k) + (pack_ring(z, k) if z is not None else b"")
    if got != want:
        return f"ring64 {'pair' if op.pair else 'single'} K={k} rows={rows} differs from the reference"
    return None


def self_check(device: torch.device | str, *, k: int = 56, rows: tuple[int, ...] = (1, 2, 3, 8),
               variants: tuple[str, ...] = VARIANTS) -> list[str]:
    """Compare every variant on small random and extreme pairs with the Python-int reference; list failures."""
    dev = torch.device(device)
    gen = torch.Generator().manual_seed(20261123)
    fails: list[str] = []
    for variant in variants:
        if variant in ("dot", "planes") and dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] < 8:
            continue
        for m1, n1, m2 in ((24, 40, 33), (64, 300, 96)):
            w1 = torch.randint(-128, 128, (m1, n1), generator=gen, dtype=torch.int64).to(torch.int8)
            w2 = torch.randint(-128, 128, (m2, m1), generator=gen, dtype=torch.int64).to(torch.int8)
            hm = torch.randint(1, 1 << 12, (m1,), generator=gen, dtype=torch.int64)
            w1[0].fill_(-128)
            w2[-1].fill_(-128)
            for op in (PairOp(k, w1, w2, hm, device=dev, max_rows=max(rows), variant=variant),
                       PairOp(k, w1, device=dev, max_rows=max(rows), variant=variant)):
                for rr in rows:
                    for case in range(2):
                        why = check_op(op, rr, seed=rr * 7 + case, extreme=case == 1)
                        if why:
                            fails.append(f"{variant}: {why}")
                # small tiles and splits too: several partials per output, the unfused epilogue
                small = PairConfig(16, 32 if variant == "planes" else 16, 3, 4, variant)
                rr = rows[-1]
                op.set_config(rr, small, small if op.pair else None)
                for case in range(2):
                    why = check_op(op, rr, seed=rr * 11 + case, extreme=case == 1)
                    if why:
                        fails.append(f"{variant} (split): {why}")
    return fails
