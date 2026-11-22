"""Gateway-GPU engine of the encrypted mask factory: device-resident operands and fused device passes.

:mod:`pgw.hemask.fastmm` runs each byte-plane product on a GPU but moves both operands over the bus
on every call (the public matrix ``A`` per group, the hint per pass) and finishes on the CPU. This
module keeps the work on the device:

* **Encryption** (:meth:`DeviceHE.encrypt`): ``A`` stays on the device as int8 byte planes. All
  groups of one provider request are encrypted with ONE int8 GEMM ``A[:n] @ [S_1 | ... | S_g]``;
  the plane combine, noise, message and ``mod 2^q`` reduction run on the device and only the
  ciphertext ``b`` comes back. Secrets and noise are still sampled on the CPU by the same code
  (:func:`pgw.hemask.lwe.sample_secret`, :func:`pgw.hemask.lwe.sample_noise`), from a fresh
  ``os.urandom`` seed per group, so the ciphertexts are exactly those of
  :func:`pgw.hemask.lwe.encrypt_masks` for the same seeds.
* **Decryption** (:meth:`DeviceHE.decrypt_into`): the hint ``H`` (``m x P x dim`` int8 planes) is
  kept on the device while it fits a memory budget (LRU per linear, else streamed through a pinned
  staging buffer). ``H S``, the subtraction from the reply, the rounding and the transpose into slot
  order run on the device; the host receives ``W8 r`` rows only.

Exactness: every step is integer arithmetic mod ``2^64`` (int8 GEMMs with int32 partial sums that
cannot overflow, int64 adds and shifts that wrap). :func:`self_test` compares encryption and
decryption on the device with the float64 bit-slice reference (``fast=False``) at the real ``dim``
before first use; on any mismatch or error the device is not used (:func:`device_for` returns
``None`` and :func:`pgw.hemask.fastmm.engine` stops choosing ``cuda``). The factory's every-row
Freivalds check of the decrypted output still runs on the CPU.

Privacy: this is customer-side work only. The provider still receives exactly the same ciphertexts
(same distribution, same sizes, same request schedule). Device temporaries that hold secret values
(``A s``, ``H S``, masks, noise) are zeroed after use like their CPU counterparts.

Environment: ``PGW_HEMASK_GPU_HINT_MB`` caps the hint cache (MiB; ``auto`` = 70% of the free device
memory at start-up minus 1 GiB; ``0`` streams every pass). ``PGW_HEMASK_GPU_HINT_STALE_S`` (default 600)
protects hints used that recently from eviction (``0``: plain LRU). ``PGW_HEMASK_HINT_PREFETCH=0`` turns
off the double-buffered hint stream (next block copied on a helper thread while the device works).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from pgw.core.prng import new_seed
from pgw.core.proc_tuning import os_thread_name
from pgw.hemask.lwe import (
    DEGREE,
    HEParams,
    MaskCiphertext,
    matmul_mod64,
    public_matrix,
    sample_noise,
    sample_secret,
)

__all__ = ["DeviceHE", "device_for", "self_test", "status"]

log = logging.getLogger(__name__)

# torch.from_numpy warns on read-only memory-mapped hints; torch only reads them (see backend.py).
warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning,
                        module=__name__)

#: int32 GEMM output per block of a device pass (bounds device temporaries).
_BLOCK_BYTES = 256 << 20
_TESTED: dict[str, bool] = {}
_TEST_LOCK = threading.Lock()
_STATUS: dict[str, Any] = {}


def _signed(v: int) -> int:
    """A uint64 constant as the int64 with the same bits."""
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >= 1 << 63 else v


def int_mm(lhs: torch.Tensor, rhs_t: torch.Tensor) -> torch.Tensor:
    """``lhs (M, K) @ rhs_t (N, K)^T`` int8 -> int32 on the operands' device.

    On CUDA ``rhs_t.t()`` is the column-major operand cuBLASLt int8 GEMMs accept on every GPU; shapes
    are zero-padded only when ``torch._int_mm`` needs it (rows > 16 and a multiple of 8, inner and
    output dims multiples of 32 on T4-class cuBLASLt), so the aligned shapes of the factory (``dim``
    a multiple of 256, ``256 g`` columns) never copy.
    """
    m, k = lhs.shape
    n = rhs_t.shape[0]
    if lhs.device.type != "cuda":
        return torch._int_mm(lhs, rhs_t.t())
    mp, kp, np_ = max(32, -(-m // 8) * 8), -(-k // 32) * 32, -(-n // 32) * 32  # pragma: no cover - GPU only
    a = lhs  # pragma: no cover - GPU only
    if (mp, kp) != (m, k) or not lhs.is_contiguous():  # pragma: no cover - GPU only
        a = torch.zeros((mp, kp), dtype=torch.int8, device=lhs.device)
        a[:m, :k] = lhs
    b = rhs_t  # pragma: no cover - GPU only
    if (np_, kp) != (n, k) or not rhs_t.is_contiguous():  # pragma: no cover - GPU only
        b = torch.zeros((np_, kp), dtype=torch.int8, device=lhs.device)
        b[:n, :k] = rhs_t
    out = torch._int_mm(a, b.t())  # pragma: no cover - GPU only
    return out if (mp, np_) == (m, n) else out[:m, :n]  # pragma: no cover - GPU only


def combine(part: torch.Tensor, offset: torch.Tensor, first_byte: int) -> torch.Tensor:
    """``sum_i (part[:, i] + offset) << 8 (first_byte + i)`` mod ``2^64`` as int64 (wrapping) on the device.

    ``part`` is ``(rows, P, cols)`` int32 with ``|part| < 2^22`` (ternary secrets, ``dim <= 2^14``);
    ``offset`` is ``(cols,)`` int64 (``128 * colsum(S)``, from the -128 plane bias). Neighbouring planes
    are paired in int32 (``p_i + 256 p_{i+1}``, below ``2^31``) before widening, as in
    :func:`pgw.hemask.fastmm._combine`.
    """
    rows, npl, cols = part.shape
    acc = torch.zeros((rows, cols), dtype=torch.int64, device=part.device)
    weight = 0
    i = 0
    while i < npl:
        shift = 8 * (first_byte + i)
        if i + 1 < npl:
            term = torch.add(part[:, i], part[:, i + 1], alpha=256).to(torch.int64)
            weight += 257 << shift
            i += 2
        else:
            term = part[:, i].to(torch.int64)
            weight += 1 << shift
            i += 1
        if shift:
            term <<= shift
        acc += term
        term.zero_()  # partial products of the secret (H S or A s planes)
        del term
    bias = offset.view(1, cols) * _signed(weight)
    acc += bias
    bias.zero_()  # column sums of the secret
    return acc


def _stale_s() -> float:
    raw = os.environ.get("PGW_HEMASK_GPU_HINT_STALE_S", "").strip()
    try:
        return max(0.0, float(raw)) if raw else 600.0
    except ValueError:
        return 600.0


class _HintCache:
    """Device copies of hints within ``budget`` bytes.

    The factory reads every linear's hint once per pass, in the same order, so the access pattern is a
    cycle over all hints. When they do not all fit (7B: 34 GB of hints against a 16 GB GPU), least
    recently used eviction evicts each hint just before it is needed again: every access misses and
    re-uploads the whole hint. So entries used within the last ``stale_s`` seconds
    (``PGW_HEMASK_GPU_HINT_STALE_S``, default 600) are never evicted: a hint that does not fit is
    streamed instead, the resident ones keep hitting, and hints of a pack or ring that is no longer
    generated make room once they go stale (least recently used first). ``stale_s = 0`` is plain LRU.
    Which hints are resident never changes a result (cached and streamed passes compute the same bits).
    """

    def __init__(self, budget: int, stale_s: float | None = None):
        self.budget = max(0, int(budget))
        self.used = 0
        self.stale_s = _stale_s() if stale_s is None else max(0.0, float(stale_s))
        self._d: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._last: dict[str, float] = {}

    @staticmethod
    def _size(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def get(self, key: str) -> torch.Tensor | None:
        t = self._d.get(key)
        if t is not None:
            self._d.move_to_end(key)
            self._last[key] = time.monotonic()
        return t

    def _evictable(self, now: float) -> list[str]:
        """Keys that may be evicted, least recently used first."""
        return [k for k in self._d if now - self._last.get(k, -math.inf) >= self.stale_s]

    def fits(self, size: int) -> bool:
        """Whether ``put`` of ``size`` bytes would succeed (without evicting anything yet)."""
        if size > self.budget:
            return False
        if self.used + size <= self.budget:
            return True
        free = self.budget - self.used + sum(self._size(self._d[k]) for k in self._evictable(time.monotonic()))
        return size <= free

    def put(self, key: str, t: torch.Tensor) -> bool:
        size = self._size(t)
        if not self.fits(size):
            return False
        now = time.monotonic()
        for k in self._evictable(now):
            if self.used + size <= self.budget:
                break
            self.used -= self._size(self._d.pop(k))
            self._last.pop(k, None)
        self._d[key] = t
        self._last[key] = now
        self.used += size
        return True

    def shrink(self, nbytes: int) -> int:
        """Lower the budget by ``nbytes`` (at least to what is resident minus ``nbytes``) and drop the most
        recently cached hints until the rest fits, stale or not. Returns the bytes released."""
        self.budget = max(0, min(self.budget, self.used) - max(0, int(nbytes)))
        freed = 0
        while self._d and self.used > self.budget:
            k, t = self._d.popitem(last=True)
            self._last.pop(k, None)
            self.used -= self._size(t)
            freed += self._size(t)
        return freed

    def pop(self, key: str) -> None:
        """Drop ``key`` (if cached)."""
        t = self._d.pop(key, None)
        self._last.pop(key, None)
        if t is not None:
            self.used -= self._size(t)

    def clear(self) -> None:
        self._d.clear()
        self._last.clear()
        self.used = 0


def _prefetch_enabled() -> bool:
    """``PGW_HEMASK_HINT_PREFETCH`` (default on): streamed hint blocks are copied ahead on a helper thread."""
    return os.environ.get("PGW_HEMASK_HINT_PREFETCH", "1").strip().lower() not in ("0", "false", "no", "off")


#: Bytes of cached hints dropped per out-of-memory retry of a decryption pass.
_OOM_SHRINK = 2 << 30


def _auto_budget(dev: torch.device) -> int:
    raw = os.environ.get("PGW_HEMASK_GPU_HINT_MB", "auto").strip().lower()
    if raw not in ("", "auto"):
        return max(0, int(float(raw) * (1 << 20)))
    if dev.type != "cuda":
        return 1 << 30
    try:  # pragma: no cover - GPU only
        free, _total = torch.cuda.mem_get_info(dev)
    except Exception:  # pragma: no cover - GPU only
        return 0
    return max(0, int(0.7 * free) - (1 << 30))  # pragma: no cover - GPU only


class DeviceHE:
    """Encryption and decryption of one :class:`pgw.hemask.backend.HEMaskBackend` on ``device``.

    Args:
        a_planes: Public matrix rows as int8 byte planes ``(rows, P, dim)`` (host); uploaded once.
        params: Encryption parameters.
        device: Torch device (``cuda``; tests use ``cpu`` to run the same code).
        hint_budget: Bytes of hints kept on the device (default :envvar:`PGW_HEMASK_GPU_HINT_MB`).
    """

    def __init__(self, a_planes: np.ndarray, params: HEParams, device: torch.device | str,
                 hint_budget: int | None = None):
        self.dev = torch.device(device)
        self.params = params
        self.a = torch.from_numpy(np.ascontiguousarray(a_planes)).to(self.dev)
        self.cache = _HintCache(_auto_budget(self.dev) if hint_budget is None else hint_budget)
        self._stage: torch.Tensor | None = None  # pinned staging buffer for streamed hint blocks
        self._stage_ev: Any = None
        self._stage2: list[torch.Tensor] | None = None  # two staging buffers of the prefetching hint stream
        self._stage2_ev: list[Any] = [None, None]
        self._prefetch: ThreadPoolExecutor | None = None
        self.stats = {"hint_uploads": 0, "hint_streamed_blocks": 0}

    # ---- helpers ---------------------------------------------------------------------------------
    def _to_dev(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if self.dev.type != "cuda":
            return t.clone()
        # always a private pinned copy: t.pin_memory() returns t itself when the caller's buffer is already
        # pinned (the factory pins r), and wiping that would destroy the caller's data
        pinned = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)  # pragma: no cover - GPU only
        pinned.copy_(t)  # pragma: no cover - GPU only
        try:  # pragma: no cover - GPU only
            return pinned.to(self.dev)  # synchronous: the pinned copy can be wiped right after
        finally:  # pragma: no cover - GPU only
            pinned.zero_()

    def _upload_block(self, blk: np.ndarray) -> torch.Tensor:
        """A streamed hint block through the reusable pinned staging buffer."""
        if self.dev.type != "cuda":
            return torch.from_numpy(np.ascontiguousarray(blk))
        n = blk.size  # pragma: no cover - GPU only
        if self._stage is None or self._stage.numel() < n:  # pragma: no cover - GPU only
            self._stage = torch.empty(n, dtype=torch.int8, pin_memory=True)
        elif self._stage_ev is not None:  # pragma: no cover - GPU only
            self._stage_ev.synchronize()  # the previous block's copy has left the staging buffer
        st = self._stage[:n].view(blk.shape)  # pragma: no cover - GPU only
        st.numpy()[...] = blk  # pragma: no cover - GPU only
        out = st.to(self.dev, non_blocking=True)  # pragma: no cover - GPU only
        self._stage_ev = torch.cuda.Event()  # type: ignore[no-untyped-call]  # pragma: no cover - GPU only
        self._stage_ev.record()  # pragma: no cover - GPU only
        self.stats["hint_streamed_blocks"] += 1  # pragma: no cover - GPU only
        return out  # pragma: no cover - GPU only

    def _streamed_blocks(self, hint: np.ndarray, block: int) -> Generator[tuple[int, int, torch.Tensor], None, None]:
        """``(r0, rows, device copy of hint[r0:r0 + rows])`` for the hint's row blocks, in order.

        Two (pinned) staging buffers: while the device multiplies block ``i``, a helper thread copies block
        ``i + 1`` out of the hint (a memory map, so this is where the disk reads of a hint larger than RAM
        happen). The one-buffer path did the host copy, the upload and the product one after another. A
        buffer is refilled only after its previous upload has completed (CUDA event). Same bytes reach the
        device either way, so decryption gives the same bits.
        """
        m, npl, dim = hint.shape
        cuda = self.dev.type == "cuda"
        size = min(block, m) * npl * dim
        if self._stage2 is None or self._stage2[0].numel() < size:
            self._stage2 = [torch.empty(size, dtype=torch.int8, pin_memory=cuda) for _ in range(2)]
            self._stage2_ev = [None, None]
        bufs, evs = self._stage2, self._stage2_ev
        if self._prefetch is None:
            self._prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-hemask-hint",
                                                initializer=os_thread_name)
        starts = list(range(0, m, block))

        def fill(i: int) -> torch.Tensor:
            r0 = starts[i]
            nb = min(block, m - r0)
            ev = evs[i % 2]
            if ev is not None:  # pragma: no cover - GPU only
                ev.synchronize()  # the upload that last read this buffer has finished
            dst = bufs[i % 2][:nb * npl * dim].view(nb, npl, dim)
            np.copyto(dst.numpy(), hint[r0:r0 + nb])
            return dst

        fut: Future[torch.Tensor] | None = self._prefetch.submit(fill, 0)
        try:
            for i, r0 in enumerate(starts):
                assert fut is not None
                host = fut.result()
                fut = None
                if cuda:  # pragma: no cover - GPU only
                    dev_blk = host.to(self.dev, non_blocking=True)
                    ev = torch.cuda.Event()  # type: ignore[no-untyped-call]
                    ev.record()
                    evs[i % 2] = ev
                else:
                    dev_blk = host  # CPU tensors (tests): the caller is done with it before the next fill
                if i + 1 < len(starts):
                    fut = self._prefetch.submit(fill, i + 1)
                self.stats["hint_streamed_blocks"] += 1
                yield r0, int(host.shape[0]), dev_blk
        finally:
            if fut is not None:
                with contextlib.suppress(Exception):
                    fut.result()  # never leave a copy into the staging buffer running past this pass

    def hint_on_device(self, key: str, hint: np.ndarray) -> torch.Tensor | None:
        """The device copy of ``hint`` (uploaded on first use), or ``None`` when it does not fit the budget."""
        t = self.cache.get(key)
        if t is not None:
            return t
        if not self.cache.fits(hint.nbytes):  # streamed: uploading it would evict a hint still in use
            self.stats["hint_misses_streamed"] = self.stats.get("hint_misses_streamed", 0) + 1
            return None
        try:
            t = torch.empty(hint.shape, dtype=torch.int8, device=self.dev)
            rows = max(1, (64 << 20) // max(1, hint.shape[1] * hint.shape[2]))
            for r0 in range(0, hint.shape[0], rows):
                t[r0:r0 + rows].copy_(self._upload_block(hint[r0:r0 + rows]))
        except torch.cuda.OutOfMemoryError:
            # the budget is estimated per backend at start-up; other backends of this process (several
            # packs/rings) or another process on the same GPU may have taken the memory since: keep the
            # hints already resident and stream the rest (same bits, slower)
            self.cache.budget = self.cache.used
            self.stats["hint_oom"] = self.stats.get("hint_oom", 0) + 1
            log.warning("encrypted mask factory: device memory full while caching the hint of %s; hint cache "
                        "capped at %.2f GiB, the other hints are streamed", key, self.cache.used / 2**30)
            if self.dev.type == "cuda":  # pragma: no cover - GPU only
                torch.cuda.empty_cache()
            return None
        if not self.cache.put(key, t):  # pragma: no cover - checked above
            return None
        self.stats["hint_uploads"] += 1
        return t

    # ---- encryption ----------------------------------------------------------------------------
    def encrypt(self, r: np.ndarray, n: int, *, seeds: list[bytes] | None = None
                ) -> tuple[list[MaskCiphertext], np.ndarray]:
        """Encrypt the masks ``r`` (``rows x n``) in groups of 256 slots with one device GEMM.

        Returns the groups (each with its secret; ``b`` left empty) and their ciphertexts side by side
        (``n x 256 g`` uint64), equal to ``encrypt_masks(group, ...)`` for each group's seed.
        ``seeds`` is a test hook (fixed randomness exposes the masks); production draws a fresh
        ``os.urandom`` seed per group.
        """
        p = self.params
        rows = int(r.shape[0])
        g = -(-rows // DEGREE)
        seeds = [new_seed() for _ in range(g)] if seeds is None else list(seeds)
        if len(seeds) != g:
            raise ValueError("one seed per group")
        cts: list[MaskCiphertext] = []
        host_e = np.empty((n, DEGREE * g), dtype=np.int8)
        host_st = np.empty((DEGREE * g, p.dim), dtype=np.int8)
        st: torch.Tensor | None = None
        ed: torch.Tensor | None = None
        rd: torch.Tensor | None = None
        try:
            for gi, seed in enumerate(seeds):
                smat = sample_secret(seed, p.rank)
                cts.append(MaskCiphertext(b=_NO_B, secret=smat, rows=min(DEGREE, rows - gi * DEGREE)))
                host_st[gi * DEGREE:(gi + 1) * DEGREE] = smat.T
                e = sample_noise(seed, n, p.eta)
                host_e[:, gi * DEGREE:(gi + 1) * DEGREE] = e  # |e| <= eta: exact in int8
                e.fill(0)
            st = self._to_dev(host_st)
            ed = self._to_dev(host_e)
            rd = self._to_dev(np.ascontiguousarray(r, dtype=np.int64))
        except BaseException:
            for t in (st, ed, rd):  # an upload failed part-way: wipe what reached the device
                if t is not None:
                    t.zero_()
            raise
        finally:
            host_st.fill(0)
            host_e.fill(0)
        assert st is not None and ed is not None and rd is not None
        npl = self.a.shape[1]
        part = pad = msg = None
        try:
            part = int_mm(self.a[:n].reshape(n * npl, p.dim), st).view(n, npl, DEGREE * g)
            offset = st.sum(1, dtype=torch.int64) * 128
            pad = combine(part, offset, 0)
            part.zero_()
            offset.zero_()
            msg = torch.zeros((DEGREE * g, n), dtype=torch.int64, device=self.dev)
            msg[:rows] = rd
            rd.zero_()
            pad += ed  # int8 noise, promoted in place
            ed.zero_()
            msg <<= p.delta_bits
            pad += msg.t()
            msg.zero_()
            if p.q < 64:
                pad &= (1 << p.q) - 1
            b = pad.to("cpu", copy=True).numpy().view(np.uint64)
        finally:
            for t in (part, pad, msg, st, ed, rd):
                if t is not None:
                    t.zero_()
        return cts, b

    # ---- decryption ----------------------------------------------------------------------------
    def decrypt_into(self, key: str, hint: np.ndarray, items: list[tuple[np.ndarray, list[MaskCiphertext], int]],
                     out: np.ndarray, delta_bits: int, k: int) -> None:
        """``out[slot] = round((y - H S) / 2^delta) mod 2^K`` for every slot of ``items`` (one hint pass).

        ``items`` are ``(y, cts, s0)`` as in :meth:`pgw.hemask.backend.HEMaskBackend._decrypt_into`;
        ``hint`` holds byte planes ``1 .. P`` of ``H mod 2^q`` (byte 0 is not needed).

        When the device runs out of memory for the pass itself (the hint cache took the memory the pass
        workspace needs: 7B hints on a 16 GB T4), cached hints are dropped (``_OOM_SHRINK`` bytes at a time,
        the budget lowered with them) and the pass runs again from the start; it rewrites every output row,
        so the result is the same bits. With no cached hint left the error is raised.
        """
        while True:
            oom = False
            try:
                self._decrypt_pass(key, hint, items, out, delta_bits, k)
                return
            except torch.cuda.OutOfMemoryError:
                if not self.cache.used:
                    raise
                oom = True
            if oom:  # outside the handler: the failed pass's frames (and device tensors) are released
                freed = self.cache.shrink(_OOM_SHRINK)
                self.stats["pass_oom_shrinks"] = self.stats.get("pass_oom_shrinks", 0) + 1
                log.warning("encrypted mask factory: device memory full during a decryption pass; dropped %.2f GiB "
                            "of cached hints (cache now %.2f GiB) and retried", freed / 2**30, self.cache.used / 2**30)
                if self.dev.type == "cuda":  # pragma: no cover - GPU only
                    torch.cuda.empty_cache()

    def _decrypt_pass(self, key: str, hint: np.ndarray, items: list[tuple[np.ndarray, list[MaskCiphertext], int]],
                      out: np.ndarray, delta_bits: int, k: int) -> None:
        m, npl, dim = hint.shape
        smat_t = np.concatenate([ct.secret.T for it in items for ct in it[1]], axis=0)  # (cols, dim)
        ys = np.concatenate([np.ascontiguousarray(y, dtype=np.uint64) for y, _, _ in items], axis=1)
        cols = smat_t.shape[0]
        slot = np.full(cols, -1, dtype=np.int64)
        c0 = 0
        for y, cts, s0 in items:
            row = s0
            for gi, ct in enumerate(cts):
                slot[c0 + gi * DEGREE:c0 + gi * DEGREE + ct.rows] = np.arange(row, row + ct.rows)
                row += ct.rows
            c0 += y.shape[1]
        valid = np.flatnonzero(slot >= 0)
        dst = slot[valid]
        st: torch.Tensor | None = None
        try:
            st = self._to_dev(smat_t)
            yd = self._to_dev(ys.view(np.int64))
        except BaseException:
            if st is not None:  # the secrets reached the device but the replies did not: wipe them
                st.zero_()
            raise
        finally:
            smat_t.fill(0)
            ys.fill(0)
        assert st is not None
        # Every device temporary holding secret-derived values (S column sums, H S partials, y - H S, the
        # selected mask rows) is zeroed in the finally below, also when the pass fails part-way (device
        # out of memory, retried by decrypt_into): freed device memory is handed to the next allocation.
        offset: torch.Tensor | None = None
        part: torch.Tensor | None = None
        corr: torch.Tensor | None = None
        v: torch.Tensor | None = None
        sel: torch.Tensor | None = None
        host: np.ndarray | None = None
        blocks: Iterator[tuple[int, int, torch.Tensor]] | None = None
        try:
            vidx = torch.from_numpy(valid).to(self.dev)
            dev_hint = self.hint_on_device(key, hint)
            half = 1 << (delta_bits - 1)
            kmask = (1 << k) - 1
            block = max(8, min(m, _BLOCK_BYTES // (npl * cols * 4)))
            offset = st.sum(1, dtype=torch.int64) * 128
            if dev_hint is not None:
                blocks = ((r0, min(block, m - r0), dev_hint[r0:r0 + block]) for r0 in range(0, m, block))
            elif _prefetch_enabled():
                blocks = self._streamed_blocks(hint, block)
            else:  # PGW_HEMASK_HINT_PREFETCH=0: one staging buffer, host copy / upload / product in turn
                blocks = ((r0, min(block, m - r0), self._upload_block(hint[r0:r0 + block]))
                          for r0 in range(0, m, block))
            for r0, nb, h in blocks:
                part = int_mm(h.reshape(nb * npl, dim), st).view(nb, npl, cols)
                corr = combine(part, offset, 1)
                part.zero_()
                v = yd[r0:r0 + nb] - corr
                corr.zero_()
                v += half
                v >>= delta_bits  # arithmetic shift: the K bits kept are the same (delta + K <= 64)
                v &= kmask
                sel = v.t().index_select(0, vidx)  # (valid slots, nb): slot-major, as the store wants
                v.zero_()
                host = sel.to("cpu", copy=True).numpy()
                sel.zero_()
                out[dst, r0:r0 + nb] = host
                host.fill(0)
        finally:
            if isinstance(blocks, Generator):
                blocks.close()  # an error part-way: the next block's host copy is waited for, not left running
            for t in (part, corr, v, sel, offset, st, yd):
                if t is not None:
                    t.zero_()
            if host is not None:
                host.fill(0)

_NO_B = np.empty((0, DEGREE), dtype=np.uint64)


def _reference_encrypt(r: np.ndarray, a: np.ndarray, p: HEParams, seed: bytes) -> np.ndarray:
    smat = sample_secret(seed, p.rank)
    e = sample_noise(seed, a.shape[0], p.eta)
    msg = np.zeros((a.shape[0], DEGREE), dtype=np.uint64)
    msg[:, :r.shape[0]] = r.astype(np.uint64).T
    pad = matmul_mod64(smat, a, small_left=False, fast=False)
    return (pad + e.astype(np.uint64) + (msg << np.uint64(p.delta_bits))) & p.q_mask


def self_test(device: torch.device | str, k: int = 32, rank: int = 16) -> bool:
    """Device encryption and decryption equal the float64 reference bit for bit (cached per device/ring).

    Runs at the real ``dim`` (``256 rank``) with a few rows and two groups (one partial).
    """
    from pgw.hemask.fastmm import to_planes
    from pgw.hemask.lwe import a_plane_count

    dev = torch.device(device)
    tag = f"{dev}/k{k}/r{rank}"
    with _TEST_LOCK:
        if tag in _TESTED:
            return _TESTED[tag]
        ok = False
        why = ""
        try:
            p = HEParams(k=k, rank=rank)
            rng = np.random.default_rng(1234)
            n, m, rows = 40, 24, DEGREE + 7
            a = public_matrix(bytes(range(32)), n, p)
            a_pl = to_planes(a)[:, :a_plane_count(p)]
            eng = DeviceHE(a_pl, p, dev, hint_budget=0)
            r = rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64)
            seeds = [bytes([7 + i]) * 32 for i in range(2)]
            # the factory hands over pinned host buffers on CUDA: they must come back untouched
            r_in = torch.from_numpy(r.copy()).pin_memory().numpy() if dev.type == "cuda" else r.copy()
            cts, b = eng.encrypt(r_in, n, seeds=seeds)
            ref = np.concatenate([_reference_encrypt(r[i * DEGREE:(i + 1) * DEGREE], a, p, s)
                                  for i, s in enumerate(seeds)], axis=1)
            if not np.array_equal(r_in, r):
                why = "encryption modified its input masks"
            elif not np.array_equal(b, ref):
                why = "encryption differs from the reference"
            else:
                w = rng.integers(-127, 128, size=(m, n)).astype(np.int64)
                hint = to_planes(matmul_mod64(w, a, fast=False))[:, 1:p.q // 8]
                y = matmul_mod64(w, b, fast=False) & p.q_mask
                out = np.zeros((rows, m), dtype=np.int64)
                eng.decrypt_into("selftest", hint, [(y, cts, 0)], out, p.delta_bits, k)
                want = np.array((r.astype(object) @ w.T.astype(object)) % (1 << k), dtype=np.int64)
                # streamed and cached hints, and a pass split into two items
                eng2 = DeviceHE(a_pl, p, dev, hint_budget=1 << 30)
                out2 = np.zeros((rows, m), dtype=np.int64)
                eng2.decrypt_into("selftest", hint, [(y[:, :DEGREE], cts[:1], 0),
                                                     (y[:, DEGREE:], cts[1:], DEGREE)], out2, p.delta_bits, k)
                if not (np.array_equal(out, want) and np.array_equal(out2, want)):
                    why = "decryption differs from the reference"
                else:
                    ok = True
            for ct in cts:
                ct.wipe()
        except Exception as exc:  # a missing kernel, a cuBLAS refusal: the device is not used
            why = f"{type(exc).__name__}: {exc}"
        if ok:
            log.info("encrypted mask factory: device engine on %s passed its self-test (K%d, rank %d)", dev, k, rank)
        else:
            log.error("encrypted mask factory: device engine on %s FAILED its self-test (%s); using the CPU",
                      dev, why)
        _TESTED[tag] = ok
        _STATUS[tag] = {"ok": ok, **({"error": why[:300]} if why else {})}
        return ok


def device_for(params: HEParams) -> torch.device | None:
    """The device the encrypted factory's products run on (``None``: the CPU engines).

    ``cuda`` when :func:`pgw.hemask.fastmm.engine` picks it and :func:`self_test` passed; a failed test
    turns the CUDA engine off for the whole process (:func:`pgw.hemask.fastmm.disable_cuda`).
    ``PGW_HEMASK_DEVICE_PASSES=0`` keeps the per-call fastmm products instead of the fused device passes.
    """
    from pgw.hemask import fastmm

    if fastmm.engine() != "cuda":
        return None
    if os.environ.get("PGW_HEMASK_DEVICE_PASSES", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    dev = torch.device("cuda")
    if not self_test(dev, params.k, params.rank):
        fastmm.disable_cuda("device self-test failed")
        return None
    return dev


def status() -> dict[str, Any]:
    """Self-test results of this process (for ``pgw doctor`` and the admin status)."""
    return dict(_STATUS)
