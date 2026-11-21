"""Provider-side encrypted maskgen on CUDA at scale: device-side unpack/pack and a fused combine.

``POST /pgw/v1/maskgen`` multiplies MLWE ciphertexts by ``W8`` (``W8 b mod 2^64``, byte-plane int8
GEMMs). For large models this is the provider's dominant mask cost, and before this module the CUDA
path spent most of its time on the *host*: wire v2 items were bit-unpacked to ``uint64`` and the
result bit-packed again on the CPU (about ``8 + 8`` bytes of host passes per coefficient and output),
ran 8 byte planes whatever ``bb`` was, and moved ``8`` bytes per output element back to the host.
The plane recombination also ran as a dozen separate int64 torch kernels.

Here one item runs end to end on the device, on its own stream:

1. **Upload as sent.** The request bytes go to a pinned staging buffer and to the device once
   (``bb/8`` bytes per coefficient for v2, ``q/8`` for v1). No host-side unpack.
2. **Device-side planes.** The v2 bit-packing (:func:`pgw.hemask.wire.pack_bits`) stores each
   field as a contiguous little-endian array, so whole-byte fields *are* byte planes; only the
   ``bb mod 8`` tail bits need decoding. ``ceil(bb/8)`` planes are used (7 for ``bb = 53``, the
   default upload; 8 before). Planes are written straight into the column-major GEMM operand.
3. **One int8 GEMM per column block** (all planes stacked, cuBLASLt TN form, as before).
4. **Fused combine.** A Triton kernel reads the ``nb`` int32 partials of each output once and
   writes ``sum_d part_d << 8d + const mod 2^64`` (int64 wrap-around == mod 2^64). Gated by a
   per-device self-check against the torch reference (:func:`pgw.service.kernels._combine_planes`);
   without Triton, or if the check fails, the torch reference runs.
5. **Device-side pack.** v1 cuts bytes ``drop/8 .. q/8-1``; v2 applies the rounding shift and
   produces :func:`pgw.hemask.wire.pack_y_v2`'s bit-packed layout on the device, so only
   ``yb/8`` bytes per element come back.

Every step is integer-exact and byte-identical to the reference path (tests compare against
:func:`pgw.hemask.wire.pack_y_v2` / :func:`pgw.hemask.lwe.matmul_mod64` on CPU torch, the same
code on CUDA in the ``gpu`` tests, and the service re-checks the fused path once per device before
it serves with it). The provider sees exactly the same bytes as before (privacy unchanged).

The functions are device-agnostic torch code (CPU tensors work, which is how the CPU tests run
them); the service uses them for CUDA kernels only (``PGW_SERVICE_MASKGEN_FUSED=0`` restores the
previous path).

**Decode priority.** CUDA cannot give a stream lower priority than the default one, so maskgen
cannot be pushed below decode by stream priority alone. With ``PGW_SERVICE_MASKGEN_YIELD_MS > 0``
(off by default) each maskgen column block is synchronised and, while the decode batcher has
queued rows, the next block waits up to that many milliseconds: decode GEMMs then queue behind at
most one maskgen block per stream.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any

import torch

from pgw.hemask.wire import bit_fields, y_round_offset, y_wire_bytes

__all__ = [
    "ENV_FUSED",
    "ENV_YIELD_MS",
    "combine",
    "fused_enabled",
    "maskgen_device",
    "pack_v2_block",
    "run_item",
    "triton_combine_ok",
    "v2_planes_source",
    "yield_ms",
]

log = logging.getLogger(__name__)

ENV_FUSED = "PGW_SERVICE_MASKGEN_FUSED"
ENV_YIELD_MS = "PGW_SERVICE_MASKGEN_YIELD_MS"
#: Byte budget of one block's int32 partial sums on CUDA (as :data:`pgw.service.kernels.MASKGEN_PART_BYTES_CUDA`).
PART_BYTES = 512 << 20
#: Smaller blocks while the decode yield is on (each block is one wait point; ~10 ms of T4 GEMM at 7B).
PART_BYTES_YIELD = 64 << 20
_ALIGN = 32


def fused_enabled() -> bool:
    """``PGW_SERVICE_MASKGEN_FUSED`` (default on): the CUDA maskgen path of this module."""
    raw = os.environ.get(ENV_FUSED, "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def yield_ms() -> float:
    """``PGW_SERVICE_MASKGEN_YIELD_MS`` (default 0 = off): longest wait per block while decode is queued."""
    raw = os.environ.get(ENV_YIELD_MS, "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        log.warning("ignoring %s=%r (not a number)", ENV_YIELD_MS, raw)
        return 0.0


# ---- fused combine -------------------------------------------------------------------------------
@functools.cache
def _combine_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(part_ptr, const_ptr, out_ptr, cb, row_stride, plane_stride,
               NB: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = c < cb
        base = r * row_stride + c
        acc = tl.load(const_ptr + r).to(tl.uint64, bitcast=True)
        for d in tl.static_range(NB):
            p = tl.load(part_ptr + base + d * plane_stride, mask=mask, other=0)
            acc += p.to(tl.int64).to(tl.uint64, bitcast=True) << (8 * d)  # wraps: mod 2^64
        tl.store(out_ptr + r * cb + c, acc.to(tl.int64, bitcast=True), mask=mask)

    return kernel


def _triton_ready(dev: torch.device) -> bool:
    from pgw.core.triton_gemv import triton_available

    if not triton_available():
        return False
    interp = os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")
    return dev.type == "cuda" or interp


def _combine_triton(part: torch.Tensor, m: int, cb: int, const: torch.Tensor) -> torch.Tensor:
    _, nb, cbp = part.shape
    out = torch.empty((m, cb), dtype=torch.int64, device=part.device)
    block = _block_of(cb)
    grid = (m, -(-cb // block))
    _combine_kernel()[grid](part, const.contiguous(), out, cb, nb * cbp, cbp, NB=nb, BLOCK=block)
    return out


def _combine_torch(part: torch.Tensor, m: int, cb: int, const: torch.Tensor) -> torch.Tensor:
    from pgw.service.kernels import _combine_planes

    return _combine_planes([part[:m, d, :cb] for d in range(part.shape[1])], const)


_TRITON_OK: dict[str, bool] = {}
_TRITON_LOCK = threading.Lock()


def triton_combine_ok(dev: torch.device) -> bool:
    """True once the Triton combine matched the torch reference on ``dev`` (checked once per device)."""
    key = str(dev)
    ok = _TRITON_OK.get(key)
    if ok is not None:
        return ok
    with _TRITON_LOCK:
        ok = _TRITON_OK.get(key)
        if ok is not None:
            return ok
        ok = False
        if _triton_ready(dev):
            try:
                ok = _self_check(dev)
            except Exception as exc:  # compile or launch failure: use the torch reference
                log.warning("maskgen fused combine unavailable on %s (%s: %s); using torch ops",
                            dev, type(exc).__name__, str(exc)[:200])
                ok = False
            if not ok:
                log.warning("maskgen fused combine disabled on %s (self-check failed)", dev)
        _TRITON_OK[key] = ok
        return ok


def _check_case(dev: torch.device, nb: int, m: int, cb: int, g: torch.Generator) -> bool:
    """One Triton-vs-torch combine comparison with extreme operands (int32 limits, int64 wrap)."""
    cbp = -(-cb // 32) * 32 + 32
    mp = m + 3
    lim = 128 * 128 * 4096
    part = torch.randint(-lim, lim, (mp, nb, cbp), generator=g, dtype=torch.int32)
    part[0] = lim
    part[1] = -lim
    part[2, :, ::2] = 2**31 - 1
    part[2, :, 1::2] = -(2**31)
    const = torch.randint(-(2**62), 2**62, (m, 1), generator=g, dtype=torch.int64) * 2 - 1
    const[0] = -(2**63)
    part, const = part.to(dev), const.to(dev)
    return torch.equal(_combine_triton(part, m, cb, const), _combine_torch(part, m, cb, const))


def _self_check(dev: torch.device) -> bool:
    g = torch.Generator().manual_seed(1234)
    return all(_check_case(dev, nb, m, cb, g) for nb, m, cb in ((7, 37, 96), (8, 64, 288), (1, 5, 32), (6, 33, 64)))


_SPEC_OK: dict[tuple[str, int, int], bool] = {}


def _block_of(cb: int) -> int:
    return 256 if cb >= 256 else 32


def _spec_ok(dev: torch.device, nb: int, cb: int) -> bool:
    """True once the kernel specialisation ``(NB, BLOCK)`` used for ``cb`` columns matched the torch
    reference on ``dev`` (each compiled specialisation is checked before it serves)."""
    block = _block_of(cb)
    key = (str(dev), nb, block)
    ok = _SPEC_OK.get(key)
    if ok is not None:
        return ok
    with _TRITON_LOCK:
        ok = _SPEC_OK.get(key)
        if ok is not None:
            return ok
        try:
            ok = _check_case(dev, nb, 37, 288 if block == 256 else 96, torch.Generator().manual_seed(4321 + nb))
        except Exception as exc:  # compile or launch failure: use the torch reference
            log.warning("maskgen fused combine NB=%d BLOCK=%d unavailable on %s (%s: %s); using torch ops",
                        nb, block, dev, type(exc).__name__, str(exc)[:200])
            ok = False
        if not ok:
            log.warning("maskgen fused combine NB=%d BLOCK=%d disabled on %s (self-check failed)", nb, block, dev)
        _SPEC_OK[key] = ok
        return ok


def combine(part: torch.Tensor, m: int, cb: int, const: torch.Tensor, *, fused: bool | None = None) -> torch.Tensor:
    """``(m, cb)`` int64 bits of ``sum_d part[:m, d, :cb] << 8d + const mod 2^64``.

    ``part``: ``(mp, nb, cbp)`` int32 contiguous (the stacked-plane GEMM output), ``const``: ``(m, 1)``
    int64. ``fused=None`` uses the Triton kernel when it passed its self-check on this device, and the
    kernel specialisation for this ``nb`` and block width passed its own check there.
    """
    if fused is None:
        fused = triton_combine_ok(part.device) and _spec_ok(part.device, part.shape[1], cb)
    if fused:
        return _combine_triton(part.contiguous(), m, cb, const)
    return _combine_torch(part, m, cb, const)


# ---- wire v2 on the device ---------------------------------------------------------------------
def v2_planes_source(packed: torch.Tensor, n: int, cols: int, bits: int) -> list[tuple[torch.Tensor, int]]:
    """Byte-plane sources of a bit-packed ``(n, cols)`` operand (:func:`pgw.hemask.wire.pack_bits`).

    Returns ``nb = ceil(bits/8)`` entries ``(src, j)``: plane ``d`` (byte ``d`` of every value) is
    ``src[:, :, j]`` for an ``(n, cols, w)`` uint8 view ``src``. Whole-byte fields are views of
    ``packed``; the ``bits mod 8`` tail bits are decoded into one fresh ``(n, cols, 1)`` tensor.
    """
    count = n * cols
    if count % 8 or packed.numel() != count * bits // 8:
        raise ValueError(f"packed operand has {packed.numel()} bytes, expected {count * bits // 8} "
                         f"({count} values of {bits} bits, a multiple of 8 values)")
    fields, t = bit_fields(bits)
    out: list[tuple[torch.Tensor, int]] = []
    pos = 0
    for _off, width in fields:
        wb = width // 8
        src = packed[pos:pos + count * wb].view(n, cols, wb)
        out.extend((src, j) for j in range(wb))
        pos += count * wb
    if t:
        tb = packed[pos:pos + count * t // 8].view(count // 8, t).to(torch.int64)
        word = tb[:, 0].clone()
        for j in range(1, t):
            word |= tb[:, j] << (8 * j)
        shifts = torch.arange(0, 8 * t, t, dtype=torch.int64, device=packed.device)
        vals = ((word.unsqueeze(1) >> shifts) & ((1 << t) - 1)).to(torch.uint8)  # (count/8, 8)
        out.append((vals.view(n, cols, 1), 0))
    return out


def pack_v2_block(y: torch.Tensor, b_bits: int, y_bits: int, out: torch.Tensor, cols: int, c0: int) -> None:
    """Write columns ``c0 .. c0+cb-1`` of :func:`pgw.hemask.wire.pack_y_v2` into ``out`` (on the device).

    ``y``: ``(m, cb)`` int64 bits of ``W8 b`` modulo a multiple of ``2^b_bits``; ``out``: the whole
    item's packed reply, ``m * cols * y_bits / 8`` uint8; ``c0`` and ``cb`` multiples of 8.
    """
    m, cb = y.shape
    if c0 % 8 or cb % 8:
        raise ValueError("v2 pack blocks must be multiples of 8 columns")
    d = b_bits - y_bits
    if d:
        y = (y + (1 << (d - 1))) >> d  # wraps mod 2^64; bits >= 64 - d (sign fill) are never packed
    vb = y.contiguous().view(torch.uint8).view(m, cb, 8)
    count = m * cols
    fields, t = bit_fields(y_bits)
    pos = 0
    for off, width in fields:
        wb = width // 8
        dst = out[pos:pos + count * wb].view(m, cols, wb)
        dst[:, c0:c0 + cb, :] = vb[:, :, off // 8:off // 8 + wb]
        pos += count * wb
    if t:
        tv = (vb[:, :, (y_bits - t) // 8] & ((1 << t) - 1)).to(torch.int64).view(m, cb // 8, 8)
        shifts = torch.arange(0, 8 * t, t, dtype=torch.int64, device=y.device)
        word = (tv << shifts).sum(-1)  # disjoint bit ranges: sum == or
        dst = out[pos:pos + count * t // 8].view(m, cols // 8, t)
        dst[:, c0 // 8:(c0 + cb) // 8, :] = word.contiguous().view(torch.uint8).view(m, cb // 8, 8)[:, :, :t]


# ---- one item ------------------------------------------------------------------------------------
def _block_cols(rows: int, planes: int, cols: int, budget: int) -> int:
    from pgw.service.kernels import _maskgen_block_cols

    return _maskgen_block_cols(rows, planes, cols, budget, _ALIGN)


def maskgen_device(wpad: torch.Tensor, m: int, n: int, dev_data: torch.Tensor, cols: int, const: torch.Tensor, *,
                   in_bytes: int = 0, b_bits: int = 0, y_bits: int = 0, ylo: int = 0, yhi: int = 8,
                   part_bytes: int = PART_BYTES, fused: bool | None = None,
                   after_block: Callable[[], None] | None = None) -> torch.Tensor:
    """The packed reply of one maskgen item, computed where ``wpad`` lives (returns a uint8 device tensor).

    ``wpad``: ``(mp, np_)`` int8 weight zero-padded to multiples of 32; ``dev_data``: the item's ``b``
    bytes (1-D uint8, same device). v1: ``in_bytes = q/8`` bytes per coefficient, ``(n, cols, in_bytes)``,
    reply bytes ``ylo .. yhi-1`` of ``W8 b + const`` per element (``const`` carries the rounding
    offset). v2: ``b_bits``/``y_bits`` set, reply :func:`pack_y_v2` bit-packed. ``const`` is
    :meth:`ComputeKernel._maskgen_const` for the plane count (``in_bytes``, or ``ceil(b_bits/8)``).
    """
    mp, np_ = wpad.shape
    dev = wpad.device
    if b_bits:
        srcs = v2_planes_source(dev_data, n, cols, b_bits)
        out = torch.empty(m * cols * y_bits // 8, dtype=torch.uint8, device=dev)
    else:
        raw = dev_data.view(n, cols, in_bytes)
        srcs = [(raw, j) for j in range(in_bytes)]
        out = torch.empty((m, cols, yhi - ylo), dtype=torch.uint8, device=dev)
    nb = len(srcs)
    step = _block_cols(mp, nb, cols, part_bytes)
    for c0 in range(0, cols, step):
        cb = min(step, cols - c0)
        cbp = -(-cb // _ALIGN) * _ALIGN
        planes = torch.empty((nb, cbp, np_), dtype=torch.uint8, device=dev)  # padding: see _maskgen_block_tn
        for d, (src, j) in enumerate(srcs):
            torch.bitwise_xor(src[:, c0:c0 + cb, j].t(), 0x80, out=planes[d, :cb, :n])
        part = torch._int_mm(wpad, planes.view(torch.int8).view(nb * cbp, np_).t()).view(mp, nb, cbp)
        y = combine(part, m, cb, const, fused=fused)
        if b_bits:
            pack_v2_block(y, b_bits, y_bits, out, cols, c0)
        else:
            out[:, c0:c0 + cb] = y.view(torch.uint8).view(m, cb, 8)[:, :, ylo:yhi]
        if after_block is not None:
            after_block()
    return out.view(-1)


#: torch warns on every view of read-only ``bytes``; the views here are only read (thread-safe, narrow filter)
warnings.filterwarnings("ignore", message=r"The given buffer is not writable", category=UserWarning)


def _upload(data: bytes | bytearray | memoryview, dev: torch.device) -> torch.Tensor:
    """The request bytes on ``dev``: one parallel host copy into pinned memory, then an async H2D copy."""
    # read-only request bytes are never written through this view (see the warning filter above)
    src = torch.frombuffer(memoryview(data).cast("B"), dtype=torch.uint8) if len(data) else \
        torch.empty(0, dtype=torch.uint8)
    if dev.type != "cuda":
        return src.clone()
    host = torch.empty(src.numel(), dtype=torch.uint8, pin_memory=True)  # torch's caching host allocator
    host.copy_(src)  # multi-threaded, releases the GIL
    return host.to(dev, non_blocking=True)


def run_item(kern: Any, it: Any, *, decode_busy: Callable[[], bool] | None = None) -> memoryview:
    """One maskgen item (v1 or v2) on a CUDA :class:`~pgw.service.kernels.ComputeKernel`: wire bytes in, out.

    Byte-identical to the previous path (``mulmod64_wire`` for v1, the CPU unpack/pack around it for
    v2). Runs on a pooled stream of the kernel's device; any number of threads may call it.
    Returns a read-only byte view of the pinned reply buffer (no host copy; msgpack sends it as
    ``bin``; the buffer stays alive while the view does).
    """
    from pgw.service.kernels import _borrow_stream, _return_stream

    if sys.byteorder != "little":  # pragma: no cover
        raise RuntimeError("maskgen_gpu assumes little-endian words")
    pw = kern._pw
    wpad = pw._w_dev
    dev = kern.device
    n, m, cols = kern.in_features, kern.out_features, int(it.cols)
    if it.v2:
        nb = -(-it.b_bits // 8)
        kw: dict[str, Any] = {"b_bits": it.b_bits, "y_bits": it.y_bits}
        const = kern._maskgen_const(nb, 0)
        want = n * cols * it.b_bits // 8
    else:
        nb = it.q_bits // 8
        ylo, yhi = y_wire_bytes(it.q_bits, it.drop_bits)
        kw = {"in_bytes": nb, "ylo": ylo, "yhi": yhi}
        const = kern._maskgen_const(nb, y_round_offset(it.drop_bits))
        want = n * cols * nb
    if len(it.b) != want:
        raise ValueError(f"{kern.linear}: maskgen b has {len(it.b)} bytes, expected {want}")
    wait_s = yield_ms() / 1e3
    stream = _borrow_stream(dev)
    try:
        with torch.cuda.device(dev), torch.cuda.stream(stream):
            after = None
            if wait_s > 0 and decode_busy is not None:
                def after() -> None:
                    stream.synchronize()
                    t_end = time.monotonic() + wait_s
                    while decode_busy() and time.monotonic() < t_end:
                        time.sleep(2e-4)
            dev_data = _upload(it.b, dev)
            out = maskgen_device(wpad, m, n, dev_data, cols, const, fused=None, after_block=after,
                                 part_bytes=PART_BYTES_YIELD if after is not None else PART_BYTES, **kw)
            res = torch.empty(out.numel(), dtype=torch.uint8, pin_memory=True)
            res.copy_(out, non_blocking=True)
            stream.synchronize()
        return memoryview(res.numpy()).toreadonly()
    finally:
        _return_stream(dev, stream)
