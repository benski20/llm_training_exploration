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
memory at start-up minus 1 GiB; ``0`` streams every pass).
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from pgw.core.prng import new_seed
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
        del term
    acc += offset.view(1, cols) * _signed(weight)
    return acc


class _HintCache:
    """Device copies of hints, least recently used first out, within ``budget`` bytes."""

    def __init__(self, budget: int):
        self.budget = max(0, int(budget))
        self.used = 0
        self._d: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, key: str) -> torch.Tensor | None:
        t = self._d.get(key)
        if t is not None:
            self._d.move_to_end(key)
        return t

    def put(self, key: str, t: torch.Tensor) -> bool:
        size = t.numel() * t.element_size()
        if size > self.budget:
            return False
        while self.used + size > self.budget and self._d:
            _, old = self._d.popitem(last=False)
            self.used -= old.numel() * old.element_size()
        self._d[key] = t
        self.used += size
        return True

    def clear(self) -> None:
        self._d.clear()
        self.used = 0


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
        self.stats = {"hint_uploads": 0, "hint_streamed_blocks": 0}

    # ---- helpers ---------------------------------------------------------------------------------
    def _to_dev(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if self.dev.type != "cuda":
            return t.clone()
        pinned = t.pin_memory()  # pragma: no cover - GPU only
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

    def hint_on_device(self, key: str, hint: np.ndarray) -> torch.Tensor | None:
        """The device copy of ``hint`` (uploaded on first use), or ``None`` when it does not fit the budget."""
        t = self.cache.get(key)
        if t is not None:
            return t
        if hint.nbytes > self.cache.budget:
            return None
        t = torch.empty(hint.shape, dtype=torch.int8, device=self.dev)
        rows = max(1, (64 << 20) // max(1, hint.shape[1] * hint.shape[2]))
        for r0 in range(0, hint.shape[0], rows):
            t[r0:r0 + rows].copy_(self._upload_block(hint[r0:r0 + rows]))
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
        finally:
            host_st.fill(0)
            host_e.fill(0)
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
        """
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
        try:
            st = self._to_dev(smat_t)
            yd = self._to_dev(ys.view(np.int64))
        finally:
            smat_t.fill(0)
            ys.fill(0)
        vidx = torch.from_numpy(valid).to(self.dev)
        dev_hint = self.hint_on_device(key, hint)
        half = 1 << (delta_bits - 1)
        kmask = (1 << k) - 1
        block = max(8, min(m, _BLOCK_BYTES // (npl * cols * 4)))
        offset = st.sum(1, dtype=torch.int64) * 128
        try:
            for r0 in range(0, m, block):
                nb = min(block, m - r0)
                h = dev_hint[r0:r0 + nb] if dev_hint is not None else self._upload_block(hint[r0:r0 + nb])
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
            offset.zero_()
            st.zero_()
            yd.zero_()


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
            cts, b = eng.encrypt(r, n, seeds=seeds)
            ref = np.concatenate([_reference_encrypt(r[i * DEGREE:(i + 1) * DEGREE], a, p, s)
                                  for i, s in enumerate(seeds)], axis=1)
            if not np.array_equal(b, ref):
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
