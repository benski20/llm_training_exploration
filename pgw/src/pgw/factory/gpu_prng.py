"""Mask ``r`` expansion on the factory's GPU: ChaCha20 keystream and digit split in one Triton kernel.

The factory's CPU path (:meth:`pgw.core.prng.SeedExpander.ring_uniform_rows`) runs ChaCha20 on the host,
copies ``8 x n`` bytes per mask row to the GPU and splits them into int8 digit planes there. On a T4 this,
not the GEMMs, bounds throughput (STATUS.md, factory profile). This module produces the *same* ring
elements on the GPU:

* The 32-byte stream key is derived **on the CPU** by the unchanged HKDF code
  (:meth:`SeedExpander.stream_key`, SPEC 7.3) and uploaded as eight little-endian 32-bit words. Nothing
  about the derivation, the stream or the stored format changes; only where the keystream is computed.
* The kernel (:mod:`pgw.factory._triton_chacha_kernel`) is the RFC 8439 ChaCha20 block function with the
  16-byte ``counter || nonce`` the CPU path uses: counter = the block holding the slice's first byte
  (``8 * cols * row_start // 64``), nonce = 0. Each 64-bit little-endian word becomes one ring element
  mod ``2^K``, exactly as ``ring_uniform_rows`` does (including slices that do not start on a block
  boundary).
* It writes either int64 ring elements (``rows``) or, for the factory, the ``ceil(K/8)`` int8 planes of
  balanced base-256 digits that :func:`pgw.core.ringmm.int8_ring_matmul` would compute, directly into the
  zero-padded GEMM operand (``digit_planes``). ``r`` never exists on the host and never as int64 on the GPU.

Safety rails:

* :func:`self_test` compares the kernel with the CPU expander (random keys, every K the factory uses,
  unaligned offsets, odd shapes, counters above ``2^31``, padding untouched) the first time a device is
  used; the generator falls back to the CPU path if it fails (:class:`pgw.factory.MaskGenerator`).
* The generator's output self-check re-derives the sampled rows of ``r`` with the CPU expander, so a GPU
  keystream fault (not only a GEMM fault) fails the segment.
* The device key tensor, the stream's host copy of the key and the digit planes are zeroed after use
  (as the CPU path zeroes ``r``). The ``bytes`` key returned by ``SeedExpander.stream_key`` is immutable
  and is only dropped, exactly as on the CPU path.

:func:`chacha20_blocks_torch`, :func:`ring_rows_torch` and :func:`digit_planes_torch` are a plain-torch
emulation of the same algorithm (unsigned 32-bit arithmetic in int64 tensors, the kernel's index math).
They run on any device and are what the CPU-only tests check against ``cryptography``'s ChaCha20; the
Triton kernel itself also runs on CPU tensors in the Triton interpreter (``TRITON_INTERPRET=1``).

Setting: ``PGW_FACTORY_GPU_PRNG`` (``FactorySettings.gpu_prng``): unset/``auto`` = on for CUDA factories
whose self-test passes, ``0`` = off (CPU ChaCha20), ``1`` = on (still self-tested; never used if it fails).
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Any, Literal

import numpy as np
import torch

from pgw.core.errors import ConfigError, ShapeError
from pgw.core.prng import SeedExpander, new_seed

__all__ = [
    "GPU_PRNG_ENV",
    "DeviceRingStream",
    "PrngImpl",
    "chacha20_blocks_torch",
    "check_against_cpu",
    "digit_planes_torch",
    "gpu_prng_setting",
    "key_words",
    "limbs",
    "ring_rows_torch",
    "self_test",
    "stream_layout",
]

log = logging.getLogger(__name__)

GPU_PRNG_ENV = "PGW_FACTORY_GPU_PRNG"
PrngImpl = Literal["triton", "torch"]
#: ChaCha20 blocks per Triton program (one block per lane).
_BLOCK = 256
_NUM_WARPS = 4
_M32 = 0xFFFFFFFF
_SIGMA = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def gpu_prng_setting(value: bool | str | None = None) -> bool | None:
    """Resolve the GPU PRNG switch: ``True`` (on), ``False`` (off) or ``None`` (auto: on for CUDA).

    ``value`` wins when given; otherwise ``PGW_FACTORY_GPU_PRNG`` is read (unset/empty/``auto`` = auto).
    """
    if isinstance(value, bool):
        return value
    raw = os.environ.get(GPU_PRNG_ENV, "") if value is None else str(value)
    v = raw.strip().lower()
    if v in ("", "auto"):
        return None
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ConfigError(f"{GPU_PRNG_ENV} must be 0, 1 or auto, got {raw!r}")


def limbs(k: int) -> int:
    """Number of balanced base-256 digit planes for ring width ``k``."""
    return -(-k // 8)


def _check_k(k: int) -> None:
    if not 1 <= k <= 63:
        raise ConfigError(f"GPU ring expansion supports 1 <= K <= 63, got {k}")


def stream_layout(cols: int, k: int, row_start: int, row_count: int) -> tuple[int, int, int, int]:
    """``(block0, s0, total, nblk)`` of rows ``[row_start, row_start + row_count)`` of a ``cols``-wide stream.

    ``block0``: ChaCha20 counter of the first block (as the CPU path seeks), ``s0``: 64-bit words of that
    block before the slice, ``total``: ring elements, ``nblk``: blocks to compute.

    Raises:
        ConfigError: bad arguments, or the slice runs past the 2^32-block counter range. This is
            stricter than the CPU path, which only refuses a slice that *starts* at block ``>= 2^32``
            (one that runs across the end would carry into the nonce word there); ``max_stream_rows``
            keeps factory streams inside the range, so the two agree on every slice the factory uses.
    """
    if cols <= 0 or row_start < 0 or row_count < 0:
        raise ConfigError("cols must be > 0 and row_start/row_count >= 0")
    _check_k(k)
    offset = 8 * cols * row_start
    total = cols * row_count
    block0, skip = divmod(offset, 64)
    s0 = skip // 8  # offset is a multiple of 8: whole 64-bit words
    nblk = -(-(s0 + total) // 8)
    if block0 >= 1 << 32 or (total and block0 + nblk > 1 << 32):
        raise ConfigError("stream offset beyond the 2^32-block ChaCha20 counter range")
    return block0, s0, total, nblk


def key_words(key: bytes | bytearray) -> torch.Tensor:
    """The 32-byte ChaCha20 key as eight little-endian words: int32 bit patterns on the CPU (caller zeroes)."""
    if len(key) != 32:
        raise ConfigError("ChaCha20 key must be 32 bytes")
    # one copy (astype), owned by the returned tensor so the caller can zero it
    return torch.from_numpy(np.frombuffer(key, dtype="<i4").astype(np.int32))


# ---- plain-torch emulation (reference; any device) ---------------------------------------------------
def _rotl(x: torch.Tensor, n: int) -> torch.Tensor:
    return ((x << n) | (x >> (32 - n))) & _M32


def _qr(s: list[torch.Tensor], a: int, b: int, c: int, d: int) -> None:
    s[a] = (s[a] + s[b]) & _M32
    s[d] = _rotl(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _M32
    s[b] = _rotl(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & _M32
    s[d] = _rotl(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & _M32
    s[b] = _rotl(s[b] ^ s[c], 7)


def chacha20_blocks_torch(key: torch.Tensor | bytes, counters: torch.Tensor) -> torch.Tensor:
    """RFC 8439 ChaCha20 blocks for 32-bit ``counters`` (nonce 0) as int64 ``(B, 16)`` words in ``[0, 2^32)``.

    Unsigned 32-bit arithmetic is emulated in int64 (every intermediate stays below ``2^48``). Words are
    the little-endian 32-bit words of the 64-byte keystream blocks.

    Args:
        key: 32 key bytes, or the eight words of :func:`key_words` (int32 bit patterns, any device).
        counters: int64 ``(B,)`` block counters in ``[0, 2^32)`` (on the device to compute on).
    """
    kw = key_words(key) if isinstance(key, bytes | bytearray) else key
    kw = kw.to(device=counters.device, dtype=torch.int64) & _M32
    ctr = counters.to(torch.int64)
    if bool(((ctr < 0) | (ctr > _M32)).any()):
        raise ConfigError("ChaCha20 block counters must be in [0, 2^32)")
    zero = torch.zeros_like(ctr)
    init = [zero + c for c in _SIGMA] + [zero + kw[i] for i in range(8)] + [ctr, zero, zero, zero]
    s = list(init)
    for _ in range(10):
        _qr(s, 0, 4, 8, 12)
        _qr(s, 1, 5, 9, 13)
        _qr(s, 2, 6, 10, 14)
        _qr(s, 3, 7, 11, 15)
        _qr(s, 0, 5, 10, 15)
        _qr(s, 1, 6, 11, 12)
        _qr(s, 2, 7, 8, 13)
        _qr(s, 3, 4, 9, 14)
    out = torch.stack([(s[i] + init[i]) & _M32 for i in range(16)], dim=1)
    del s, init
    return out


def _words_to_ring(lo: torch.Tensor, hi: torch.Tensor, k: int) -> torch.Tensor:
    """``(hi << 32 | lo) mod 2^k`` as non-negative int64 (the kernel's formula)."""
    if k > 32:
        return lo | ((hi & ((1 << (k - 32)) - 1)) << 32)
    if k == 32:
        return lo
    return lo & ((1 << k) - 1)


def ring_rows_torch(key: torch.Tensor | bytes, cols: int, k: int, row_start: int, row_count: int,
                    device: torch.device | str = "cpu") -> torch.Tensor:
    """Torch emulation of the kernel's ``rows`` output: equals ``SeedExpander.ring_uniform_rows_with_key``."""
    block0, s0, total, nblk = stream_layout(cols, k, row_start, row_count)
    dev = torch.device(device)
    if total == 0:
        return torch.zeros((row_count, cols), dtype=torch.int64, device=dev)
    words = chacha20_blocks_torch(key, block0 + torch.arange(nblk, dtype=torch.int64, device=dev))
    vals = _words_to_ring(words[:, 0::2], words[:, 1::2], k).reshape(-1)  # (nblk * 8,) in stream order
    out = vals[s0 : s0 + total].reshape(row_count, cols).clone()
    vals.zero_()
    words.zero_()  # (vals may be a view of words for K = 32)
    return out


def _balanced_planes(v: torch.Tensor, k: int) -> list[torch.Tensor]:
    """The kernel's digit loop (identical to ``balanced_digits(v, k, 8)``)."""
    out = []
    cur = v
    for _ in range(limbs(k)):
        d = cur & 255
        d = torch.where(d >= 128, d - 256, d)
        out.append(d.to(torch.int8))
        cur = (cur - d) >> 8
    return out


def digit_planes_torch(key: torch.Tensor | bytes, cols: int, k: int, row_start: int, row_count: int,
                       out: torch.Tensor) -> torch.Tensor:
    """Torch emulation of the kernel's ``digit_planes`` output (see :meth:`DeviceRingStream.digit_planes`)."""
    p, np_ = _check_planes(out, cols, k, row_count)
    r = ring_rows_torch(key, cols, k, row_start, row_count, out.device)
    planes = _balanced_planes(r, k)
    r.zero_()
    view = out.view(limbs(k), p, np_)
    for i, d in enumerate(planes):
        view[i, :row_count, :cols] = d
        d.zero_()
    return out


def _check_planes(out: torch.Tensor, cols: int, k: int, row_count: int) -> tuple[int, int]:
    if out.dtype != torch.int8 or out.ndim != 2 or not out.is_contiguous():
        raise ShapeError(f"digit planes must be a contiguous 2-D int8 tensor, got {out.dtype} {tuple(out.shape)}")
    n_l = limbs(k)
    rows_total, np_ = (int(s) for s in out.shape)
    if rows_total % n_l:
        raise ShapeError(f"digit planes need {n_l} x P rows, got {rows_total}")
    p = rows_total // n_l
    if p < row_count or np_ < cols:
        raise ShapeError(f"digit planes {tuple(out.shape)} too small for {row_count} x {cols} (K={k})")
    return p, np_


# ---- Triton kernel -----------------------------------------------------------------------------------
def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


@functools.cache
def _kernel(interpret: bool) -> Any:
    """The jitted kernel (lazily, per interpreter mode; importing this module never needs triton)."""
    del interpret  # cache key only
    import triton

    from pgw.factory import _triton_chacha_kernel as kmod

    # Runtime scalars that change per call must not be specialised (one compile per K and output mode).
    return triton.jit(kmod.chacha_ring_kernel,
                      do_not_specialize=["block0", "s0", "total", "cols", "nblk", "row_stride", "plane_stride"])


def _launch(key_t: torch.Tensor, cols: int, k: int, row_start: int, row_count: int, *,
            r_out: torch.Tensor | None = None, d_out: torch.Tensor | None = None,
            row_stride: int = 0, plane_stride: int = 0) -> None:
    block0, s0, total, nblk = stream_layout(cols, k, row_start, row_count)
    if total == 0:
        return
    dummy = key_t
    grid = (-(-nblk // _BLOCK),)
    dev = key_t.device
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else _Null()
    with ctx:
        _kernel(_interpret())[grid](
            key_t, r_out if r_out is not None else dummy, d_out if d_out is not None else dummy,
            block0, s0, total, cols, nblk, row_stride, plane_stride,
            K=k, LIMBS=limbs(k), BLOCK=_BLOCK, WRITE_R=r_out is not None, WRITE_D=d_out is not None,
            num_warps=_NUM_WARPS,
        )


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


@functools.cache
def triton_available() -> bool:
    """True if triton imports (it ships with CUDA builds of torch)."""
    try:
        import triton
        import triton.language  # noqa: F401
    except Exception as exc:
        log.info("triton unavailable (%s); GPU mask PRNG disabled", type(exc).__name__)
        return False
    return True


class DeviceRingStream:
    """One labelled ``r`` stream (one linear of one segment) expanded on ``device``.

    Holds the stream key (derived on the CPU) as a device tensor until :meth:`wipe`. Use as a context
    manager so the key is zeroed on every exit path.

    Args:
        key: 32-byte stream key (:meth:`SeedExpander.stream_key` ``("r", linear_key)``).
        cols: Row width ``n`` (the linear's ``in_features``).
        k: Ring width.
        device: Where to expand.
        impl: ``triton`` (the kernel; CUDA, or CPU tensors in the Triton interpreter) or ``torch`` (the
            emulation; tests).
    """

    def __init__(self, key: bytes | bytearray, cols: int, k: int, device: torch.device | str, impl: PrngImpl):
        if impl not in ("triton", "torch"):
            raise ConfigError(f"unknown GPU PRNG implementation {impl!r}")
        if cols <= 0:
            raise ConfigError("cols must be positive")
        _check_k(k)
        self.cols = int(cols)
        self.k = int(k)
        self.device = torch.device(device)
        self.impl: PrngImpl = impl
        # for CPU re-derivation of sampled rows (self-check); a bytearray so wipe() can zero it
        self._key: bytearray | None = bytearray(key)
        staging = key_words(key)
        try:
            self._key_t: torch.Tensor | None = staging.to(self.device, copy=True)
        finally:
            staging.zero_()

    def __enter__(self) -> DeviceRingStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.wipe()

    def __repr__(self) -> str:  # never print key material
        return f"DeviceRingStream(<redacted>, cols={self.cols}, k={self.k}, device={self.device}, impl={self.impl})"

    def _kt(self) -> torch.Tensor:
        if self._key_t is None:
            raise ConfigError("stream key already wiped")
        return self._key_t

    def wipe(self) -> None:
        """Zero the device key and this object's host copy of it (idempotent).

        The caller's ``key`` argument is not touched (the CPU path holds stream keys as ``bytes`` too).
        """
        if self._key_t is not None:
            self._key_t.zero_()
            self._key_t = None
        if self._key is not None:
            self._key[:] = bytes(len(self._key))
            self._key = None

    def rows(self, row_start: int, row_count: int) -> torch.Tensor:
        """int64 ``(row_count, cols)`` ring elements on the device (== ``ring_uniform_rows`` of the stream)."""
        if self.impl == "torch":
            return ring_rows_torch(self._kt(), self.cols, self.k, row_start, row_count, self.device)
        out = torch.empty((row_count, self.cols), dtype=torch.int64, device=self.device)
        _launch(self._kt(), self.cols, self.k, row_start, row_count, r_out=out)
        return out

    def digit_planes(self, row_start: int, row_count: int, out: torch.Tensor) -> torch.Tensor:
        """Write the balanced int8 digit planes of the rows into ``out`` and return it.

        ``out`` is int8 ``(limbs * P, NP)`` with ``P >= row_count`` and ``NP >= cols``: plane ``i`` is
        ``out[i*P:(i+1)*P]``, element ``(t, j)`` at ``[i*P + t, j]`` -- exactly the stacked operand
        :func:`pgw.core.ringmm.int8_ring_matmul` builds from a zero-padded ``(P, NP)`` copy of ``r``.
        Positions outside the ``row_count x cols`` block are not written (pass a zeroed buffer).
        """
        p, np_ = _check_planes(out, self.cols, self.k, row_count)
        if out.device != self.device:
            raise ShapeError(f"digit planes on {out.device}, stream on {self.device}")
        if self.impl == "torch":
            return digit_planes_torch(self._kt(), self.cols, self.k, row_start, row_count, out)
        _launch(self._kt(), self.cols, self.k, row_start, row_count, d_out=out, row_stride=np_, plane_stride=p * np_)
        return out

    def cpu_rows(self, row_start: int, idx: torch.Tensor) -> torch.Tensor:
        """Rows ``row_start + idx`` re-derived by the CPU expander (int64 ``(len(idx), cols)``, caller zeroes)."""
        if self._key is None:
            raise ConfigError("stream key already wiped")
        parts = [SeedExpander.ring_uniform_rows_with_key(self._key, self.cols, self.k, row_start + int(i), 1)
                 for i in idx.tolist()]
        out = torch.cat(parts) if parts else torch.zeros((0, self.cols), dtype=torch.int64)
        for t in parts:
            t.zero_()
        return out


# ---- self-test ---------------------------------------------------------------------------------------
def _cases(ks: tuple[int, ...], quick: bool) -> list[tuple[int, int, int, int]]:
    shapes = [(1, 0, 1), (3, 5, 7), (13, 3, 5), (37, 11, 9), (129, 1, 33), (1000, 7, 3), (8, 0, 0)]
    if not quick:
        shapes += [(5, 1, 64), (896, 13, 4), (3584, 2, 3), (18944, 1, 2)]
    out = [(k, c, s, r) for k in ks for (c, s, r) in shapes]
    out.append((max(ks), 8, (1 << 32) - 3, 2))  # counters above 2^31, last blocks of the counter range
    return out


def check_against_cpu(device: torch.device | str, impl: PrngImpl = "triton", *, ks: tuple[int, ...] = (32, 40, 48),
                      quick: bool = True, keys: int = 2) -> list[str]:
    """Compare :class:`DeviceRingStream` (rows and digit planes) with the CPU expander; return failing cases."""
    from pgw.core.ring import balanced_digits

    dev = torch.device(device)
    bad: list[str] = []
    for ki in range(keys):
        key = SeedExpander(new_seed()).stream_key("r", f"selftest.{ki}")
        for k, cols, start, rows in _cases(tuple(sorted(set(ks))), quick):
            name = f"K={k} cols={cols} start={start} rows={rows}"
            want = SeedExpander.ring_uniform_rows_with_key(key, cols, k, start, rows)
            with DeviceRingStream(key, cols, k, dev, impl) as st:
                got = st.rows(start, rows)
                if not torch.equal(got.cpu(), want):
                    bad.append("rows " + name)
                got.zero_()
                n_l = limbs(k)
                p, np_ = rows + 3, cols + 5
                out = torch.full((n_l * p, np_), 0x5A, dtype=torch.int8, device=dev)  # sentinel padding
                st.digit_planes(start, rows, out)
                exp = torch.full((n_l, p, np_), 0x5A, dtype=torch.int8)
                for i, d in enumerate(balanced_digits(want, k, 8)):
                    exp[i, :rows, :cols] = d.to(torch.int8)
                if not torch.equal(out.cpu().view(n_l, p, np_), exp):
                    bad.append("planes " + name)
                out.zero_()
            want.zero_()
    return bad


_SELF_TEST: dict[tuple[str, str, tuple[int, ...]], bool] = {}
_LOCK = threading.Lock()


def self_test(device: torch.device | str, impl: PrngImpl = "triton", ks: tuple[int, ...] = (32, 40, 48)) -> bool:
    """True if the GPU expansion matches the CPU expander bit for bit on ``device`` (cached per device/impl/K set).

    Any exception (compile failure, unsupported GPU, missing triton) counts as a failure and is logged as
    an error; the factory then keeps the CPU path.
    """
    ks = tuple(sorted({32, 40, 48, *ks}))
    key = (str(torch.device(device)), impl, ks)
    with _LOCK:
        if key in _SELF_TEST:
            return _SELF_TEST[key]
        ok = False
        why = ""
        if impl == "triton" and not triton_available():
            why = "triton not importable"
        else:
            try:
                bad = check_against_cpu(device, impl, ks=ks)
                ok = not bad
                why = f"{len(bad)} mismatching cases, e.g. {bad[:3]}"
            except Exception as exc:
                why = f"{type(exc).__name__}: {str(exc)[:300]}"
        if ok:
            log.info("GPU mask PRNG self-test passed on %s (%s, K in %s)", key[0], impl, ks)
        else:
            log.error("GPU mask PRNG self-test FAILED on %s (%s): %s; the factory uses the CPU ChaCha20 path",
                      key[0], impl, why)
        _SELF_TEST[key] = ok
        return ok
