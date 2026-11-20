"""Fused exact ring GEMV ``y = x @ W8^T mod 2^K`` for few rows (``K <= 32``) as one Triton kernel.

The digit-GEMM path (:func:`pgw.core.ringmm.offset_byte_core`) splits ``x`` into ``K/8`` byte
planes, pads them to a ``torch._int_mm`` shape and recombines the plane products: about a dozen
kernel launches per call, which dominates decode-sized calls (1-16 rows). For ``K <= 32`` the ring
product needs no digits at all: arithmetic modulo ``2^32`` is exactly what 32-bit integer
multiply-add does, and ``Z_{2^K}`` is a quotient of ``Z_{2^32}``. So one kernel streams the int8
weight once and computes, per output element::

    y[r, j] = sum_i  u32(x[r, i]) * u32(sext(W8[j, i]))   (mod 2^32),   then  & (2^K - 1)

Exactness argument:

* ``x`` enters as the low 32 bits of each ring element (``x mod 2^32``; ``x mod 2^K`` is
  determined by it for ``K <= 32``), reinterpreted as ``uint32``.
* ``W8`` is sign-extended to int32 and reinterpreted as ``uint32``: the same residue mod ``2^32``.
* All products, partial sums, the in-register reduction and the cross-program ``atomic_add``
  (split-K) are **unsigned** 32-bit operations. Unsigned overflow wraps by definition (LLVM
  ``mul``/``add`` on ``i32``, PTX ``mul.lo.u32``/``add.u32``/``red.add.u32``, numpy ``uint32`` in the
  interpreter), there is no signed-overflow UB and no float anywhere. Wrapping addition is
  associative and commutative, so any reduction order (tile shape, warp shuffles, split-K
  atomics) gives the same bits.
* The result is reduced mod ``2^K`` (a mask) and returned as int32 bit patterns: the compact form
  of :func:`pgw.service.gpu_exec.to_compact` (the low 32 bits of the ring element).

That CUDA-core kernel is compute-bound beyond one row (``rows`` integer multiply-adds per weight
byte), so several rows use a tensor-core variant (``GemvConfig.dot``) inside the same single launch:
each ``x`` tile is split in registers into four balanced signed base-256 digits
``x = sum_i d_i 256^i (mod 2^32)``, ``d_i in [-128, 127]`` (peeled with wrapping unsigned subtract and
logical shift), each digit plane is multiplied with the int8 weight tile by ``tl.dot`` (int32
accumulation, exact because ``128 * 128 * n < 2^31``; the host caps the inner range per program), and
the four products are recombined with unsigned shifts and adds mod ``2^32``. Masked ``x`` entries are
0, i.e. all-zero digits, so padding columns contribute nothing in either variant.

Triton is optional (it ships with CUDA builds of torch); :func:`triton_available` says whether it
imports. Set ``TRITON_INTERPRET=1`` before triton is first imported to run the kernels on CPU tensors
(tests; ``tests/conftest.py`` does this on hosts without CUDA).
Callers that serve traffic gate the kernel on :func:`self_check`, which compares it with the
digit-GEMM path on the target device (random and extreme operands, every supported row count and K).
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any

import torch

from pgw.core.errors import ShapeError

__all__ = [
    "TRITON_GEMV_KS",
    "TRITON_GEMV_MAX_ROWS",
    "GemvConfig",
    "gemv_config",
    "low32",
    "max_rows",
    "ring_gemv",
    "self_check",
    "triton_available",
]

log = logging.getLogger(__name__)

#: Largest row count the fused kernel serves (decode, concurrent sessions, speculative verify).
TRITON_GEMV_MAX_ROWS = 16
#: Ring widths the fused kernel serves (``K <= 32``; multiples of 8 as :class:`RingParams` requires).
TRITON_GEMV_KS = (16, 24, 32)


@functools.cache
def triton_available() -> bool:
    """True if ``triton`` imports (it ships with CUDA builds of torch)."""
    try:
        import triton
        import triton.language  # noqa: F401
    except Exception as exc:  # ImportError, or a broken install
        log.info("triton unavailable (%s: %s); fused ring GEMV disabled", type(exc).__name__, str(exc)[:200])
        return False
    return True


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


@functools.cache
def _kernel(interpret: bool, dot: bool) -> Any:
    """The jitted kernel (lazily: importing this module must not need triton).

    ``triton.jit`` picks the interpreter or the compiler from ``TRITON_INTERPRET`` when it wraps the
    function, so the wrapper is cached per mode.
    """
    del interpret  # cache key only
    import triton

    from pgw.core import _triton_gemv_kernel as kmod

    return triton.jit(kmod.ring_gemv_dot_kernel if dot else kmod.ring_gemv_kernel)


@dataclass(frozen=True)
class GemvConfig:
    """Launch configuration: tile sizes, split-K factor, warps and kernel variant.

    ``dot=False``: CUDA-core kernel (``uint32`` multiply-adds of ``x`` with the sign-extended weight).
    ``dot=True``: tensor-core kernel (four balanced signed byte digits of ``x``, int8 ``tl.dot`` with
    int32 accumulation, recombined mod 2^32); needs ``block_r, block_m >= 16`` and ``block_n >= 32``.
    """

    block_r: int
    block_m: int
    block_n: int
    split_k: int = 1
    num_warps: int = 4
    dot: bool = False


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _pow2_at_least(v: int) -> int:
    return 1 << max(0, (v - 1).bit_length())


@functools.cache
def _sm_count(device_index: int) -> int:
    try:
        return int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    except Exception:
        return 40


def gemv_config(b: int, m: int, nw: int, device: torch.device | None = None) -> GemvConfig:
    """Heuristic launch configuration for ``b`` rows, ``m`` outputs and ``nw`` weight columns.

    One row uses the CUDA-core kernel (``BLOCK_N = 1024``); several rows use the tensor-core kernel
    (rows padded to 16) where it compiled and passed the self-check. Narrow outputs (few programs)
    split the inner dimension across programs that combine with wrapping atomic adds (aiming at
    about 2048 programs, at most 8 splits). Tuned on a T4 with the 7B shapes
    (``benchmarks/triton_gemv_bench.py --sweep``; results in ``benchmarks/results/kaggle/triton_gemv``).
    """
    sms = _sm_count(device.index or 0) if device is not None and device.type == "cuda" else 40
    if b <= 1:
        block_r, block_m, block_n, dot = 1, (32 if m >= 16384 else 16), 1024, False
    elif device is not None and str(device) in _NO_DOT:
        block_r, block_m, block_n, dot = _pow2_at_least(b), 8, 512, False  # correct but compute-bound
    else:
        block_r, block_m, block_n, dot = max(16, _pow2_at_least(b)), 64, 128, True
    programs = _cdiv(m, block_m)
    target = max(2048, 16 * sms)
    split = min(8, _cdiv(target, programs), max(1, nw // 512))
    return GemvConfig(block_r, block_m, block_n, max(1, split), 4, dot)


def low32(x: torch.Tensor) -> torch.Tensor:
    """The low 32 bits of int64 ``x`` as an int32 tensor of the same shape (exact, no range cast)."""
    if x.dtype != torch.int64:
        raise ShapeError(f"expected int64 ring elements, got {x.dtype}")
    if sys.byteorder == "little" and x.is_contiguous():
        return x.view(torch.int32).view(*x.shape, 2)[..., 0]
    return (((x & 0xFFFFFFFF) + (1 << 31)) & 0xFFFFFFFF).sub_(1 << 31).to(torch.int32)


def ring_gemv(
    x32: torch.Tensor,
    w: torch.Tensor,
    k: int,
    *,
    n: int | None = None,
    m: int | None = None,
    out: torch.Tensor | None = None,
    config: GemvConfig | None = None,
) -> torch.Tensor:
    """``x @ w[:m, :n]^T mod 2^K`` as int32 bit patterns ``(b, m)`` with one kernel launch.

    Args:
        x32: int32 ``(b, n)`` low 32 bits of the ring elements (:func:`low32`), unit stride in ``n``;
            ``1 <= b <= TRITON_GEMV_MAX_ROWS`` (larger ``b`` works but is not tuned).
        w: int8 ``(mw, nw)`` weight with unit column stride, ``mw >= m`` and ``nw >= n``. Columns past
            ``n`` (zero padding of :func:`pgw.core.ringmm.pad_weight_cuda`) are streamed but meet zero
            ``x``, so their content does not matter; whole padded rows keep loads aligned.
        k: Ring bits, ``K <= 32``.
        n: Inner dimension (default ``x32.shape[1]``).
        m: Output features (default ``w.shape[0]``).
        out: Optional int32 ``(b, m)`` output with unit column stride.
        config: Launch configuration (default :func:`gemv_config`).

    Returns:
        int32 ``(b, m)``: ``y mod 2^K`` (for ``K = 32`` the two's-complement bits of ``y mod 2^32``).
    """
    if not 1 <= k <= 32:
        raise ShapeError(f"fused ring GEMV needs K <= 32, got {k}")
    if x32.dtype != torch.int32 or x32.ndim != 2 or w.dtype != torch.int8 or w.ndim != 2:
        raise ShapeError(f"ring_gemv needs int32 x (b, n) and int8 w (m, n), got {x32.dtype} {tuple(x32.shape)}, "
                         f"{w.dtype} {tuple(w.shape)}")
    b = int(x32.shape[0])
    n = int(x32.shape[1]) if n is None else int(n)
    m = int(w.shape[0]) if m is None else int(m)
    mw, nw = (int(s) for s in w.shape)
    if n > int(x32.shape[1]) or n > nw or m > mw:
        raise ShapeError(f"ring_gemv: n={n}, m={m} exceed x {tuple(x32.shape)} / w {tuple(w.shape)}")
    if x32.stride(1) != 1 and n > 1:
        x32 = x32.contiguous()
    if w.stride(1) != 1 and nw > 1:
        raise ShapeError("ring_gemv: weight needs unit column stride")
    if x32.device != w.device:
        raise ShapeError(f"ring_gemv: x on {x32.device}, w on {w.device}")
    cfg = config or gemv_config(b, m, nw, w.device)
    if cfg.block_r < b:
        raise ShapeError(f"ring_gemv: block_r={cfg.block_r} < rows={b}")
    split = max(1, int(cfg.split_k))
    if out is None:
        out = (torch.zeros if split > 1 else torch.empty)((b, m), dtype=torch.int32, device=w.device)
    else:
        if out.dtype != torch.int32 or tuple(out.shape) != (b, m) or (out.stride(1) != 1 and m > 1):
            raise ShapeError(f"ring_gemv: bad out {out.dtype} {tuple(out.shape)}")
        if split > 1:
            out.zero_()
    if cfg.dot and (cfg.block_r < 16 or cfg.block_m < 16 or cfg.block_n < 32):
        raise ShapeError(f"ring_gemv: tensor-core tiles need block_r, block_m >= 16 and block_n >= 32, got {cfg}")
    if b == 0 or m == 0:
        return out
    # inner range per split-K program, a multiple of BLOCK_N (keeps loads aligned)
    per = -(-_cdiv(nw, split) // cfg.block_n) * cfg.block_n
    if cfg.dot and per > _DOT_MAX_INNER:  # keep every int32 dot accumulator exact (no wrap at all)
        per = _DOT_MAX_INNER // cfg.block_n * cfg.block_n
    split = -(-nw // per)
    kmask = 0 if k == 32 else (1 << k) - 1
    grid = (-(-m // cfg.block_m), split)
    # split-K partial sums are added mod 2^32 by the atomics and masked afterwards
    kmask_launch = -1 if (kmask == 0 or split > 1) else kmask
    with _launch_guard(w.device):
        _kernel(_interpret(), cfg.dot)[grid](
            x32, w, out, b, n, m, nw, per, kmask_launch,
            x32.stride(0), w.stride(0), out.stride(0),
            BLOCK_R=cfg.block_r, BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n,
            ATOMIC=split > 1, num_warps=cfg.num_warps,
        )
    if split > 1 and kmask:
        out.bitwise_and_(kmask)
    return out


_GUARD = threading.Lock()
#: ``128 * 128 * n < 2^31`` keeps each tensor-core digit dot exact in int32
_DOT_MAX_INNER = 131_040


class _launch_guard:  # noqa: N801 - context manager used like a function
    """Selects the weight's CUDA device for the launch (no-op on CPU)."""

    def __init__(self, device: torch.device):
        self._dev = device
        self._ctx: Any = None

    def __enter__(self) -> None:
        if self._dev.type == "cuda":
            self._ctx = torch.cuda.device(self._dev)
            self._ctx.__enter__()

    def __exit__(self, *exc: object) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*exc)


def _reference(x: torch.Tensor, w8: torch.Tensor, k: int) -> torch.Tensor:
    """Existing digit-GEMM path on ``w8``'s device (int64 ``[0, 2^K)``), the self-check's oracle."""
    from pgw.core.ringmm import PreparedWeight, ringmm_reference

    if w8.device.type == "cuda":
        return PreparedWeight(w8, device=w8.device, backend="cuda_int8").matmul(x.to(w8.device), k).cpu()
    return torch.from_numpy(ringmm_reference(x, w8, k))


def _check_cases(device: torch.device, quick: bool) -> list[tuple[str, torch.Tensor, torch.Tensor, int]]:
    gen = torch.Generator().manual_seed(20261120)
    cases: list[tuple[str, torch.Tensor, torch.Tensor, int]] = []
    shapes = [(1, 1), (37, 129), (64, 4097)] if quick else [(1, 1), (37, 129), (300, 1000), (64, 4097), (96, 18944)]
    rows_all = [1, 2, 3, 5, 8, 9, 16] if not quick else [1, 3, 8, 16]
    for k in TRITON_GEMV_KS:
        top = (1 << k) - 1
        for m, n in shapes:
            for b in rows_all:
                x = torch.randint(0, 1 << k, (b, n), dtype=torch.int64, generator=gen)
                w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
                cases.append((f"random K={k} b={b} m={m} n={n}", x, w, k))
        m, n = shapes[-1]
        for wv in (-128, 127):
            for xv in (top, 1 << (k - 1), (1 << (k - 1)) - 1):
                b = TRITON_GEMV_MAX_ROWS
                x = torch.full((b, n), xv, dtype=torch.int64)
                w = torch.full((m, n), wv, dtype=torch.int8)
                cases.append((f"extreme K={k} W={wv} x={xv} m={m} n={n}", x, w, k))
    return cases


def check_against_reference(device: torch.device | str, *, quick: bool = False) -> list[str]:
    """Run the fused kernel on ``device`` for the self-check cases; returns the failing case names."""
    dev = torch.device(device)
    from pgw.core.ringmm import pad_weight_cuda

    bad: list[str] = []
    for name, x, w8, k in _check_cases(dev, quick):
        if int(x.shape[0]) > max_rows(dev):
            continue  # not served on this device
        want = _reference(x, w8.to(dev), k).to(torch.int64)
        wpad = pad_weight_cuda(w8, dev) if dev.type == "cuda" else w8
        got = ring_gemv(low32(x).to(dev), wpad, k, n=int(x.shape[1]), m=int(w8.shape[0]))
        got64 = got.cpu().to(torch.int64) & ((1 << k) - 1)
        if not torch.equal(got64, want):
            bad.append(name)
    return bad


_SELF_CHECK: dict[str, bool] = {}
#: devices where the tensor-core variant failed its check: only one-row calls use the fused kernel there
_NO_DOT: set[str] = set()


def max_rows(device: torch.device | str) -> int:
    """Most rows the fused kernel serves on ``device``: :data:`TRITON_GEMV_MAX_ROWS`, or 1 if only the
    CUDA-core variant works there (it is slower than the digit-GEMM path for several rows)."""
    return 1 if str(torch.device(device)) in _NO_DOT else TRITON_GEMV_MAX_ROWS


def _run_check(key: str, device: torch.device | str) -> tuple[bool, str]:
    try:
        bad = check_against_reference(device)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
    return not bad, f"{len(bad)} failing cases, e.g. {bad[:3]}"


def self_check(device: torch.device | str) -> bool:
    """True if triton imports and the fused kernel matches the digit-GEMM path bit for bit on ``device``.

    Cached per device. Any exception (compile failure, unsupported GPU) counts as a failure.
    """
    key = str(torch.device(device))
    with _GUARD:
        if key in _SELF_CHECK:
            return _SELF_CHECK[key]
        ok = False
        if triton_available():
            ok, why = _run_check(key, device)
            if not ok:
                # retry with the CUDA-core kernel alone (one-row calls only)
                log.warning("fused ring GEMV (tensor-core variant) failed its self-check on %s (%s); retrying "
                            "with the CUDA-core variant for one-row calls", key, why)
                _NO_DOT.add(key)
                ok, why = _run_check(key, device)
            if ok:
                log.info("fused ring GEMV self-check passed on %s (up to %d rows)", key, max_rows(key))
            else:
                log.error("fused ring GEMV self-check FAILED on %s (%s); using the digit-GEMM path", key, why)
        _SELF_CHECK[key] = ok
        return ok
