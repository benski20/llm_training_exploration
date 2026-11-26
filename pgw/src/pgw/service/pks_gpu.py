"""Torch path of the ``pks-v3`` provider key switch (:func:`pgw.hemask.pks_native.keyswitch`, reply form).

Exact int64 arithmetic on the device: every product of two residues below 2^31 is below 2^62, and the
modulus switches use 14-bit digits (as :func:`pgw.hemask.pks.round_mul_shift`). Used by
:class:`pgw.service.pks_provider.PKSProvider` on CUDA kernels only after :func:`self_test` matched the CPU
path bit for bit; it runs on any torch device (the tests run it on the CPU).

:func:`item` computes a whole item on the device: the provider-side hint rows (``W8 A``, switched to ``q1``), the
item's ``y = W8 b`` and the key switch, row block by row block, so the ``m x 4096`` hint is never cached or copied
(used by ``PKSProvider`` when the model's hints do not fit its cache; self-tested by :func:`item_self_test`).

The key switch's NTTs run either as eager torch operators or as the fused Triton kernels of
:mod:`pgw.service.pks_ntt` (one read and one write per transform), chosen by ``PGW_SERVICE_PKS_GPU_NTT``
(:func:`ntt_mode`): ``auto`` (default) uses the fused kernels on CUDA devices once :func:`fused_self_test` matched the
torch path bit for bit there, and falls back to torch for good on any mismatch or error; ``torch`` and ``fused``
force one.

The device item's other fused kernels follow the same rule (:class:`FusedChoice`, each with its own start-up
self-test and permanent fallback): ``PGW_SERVICE_PKS_GPU_FWD4`` (the forward NTTs of all four primes in one launch
with a coalesced gather, :func:`pgw.service.pks_ntt.forward4`), ``PGW_SERVICE_PKS_GPU_SWITCH`` (the 7-plane
recombination mod ``2^56`` and the hint switch in one pass, :func:`pgw.service.pks_fused.combine`) and
``PGW_SERVICE_PKS_GPU_RING`` (the public ring rows expanded by the device, ChaCha20,
:func:`pgw.service.pks_fused.ring_rows`; the fallback runs the host stream on a thread pool, off the GIL, overlapped
with the item's uploads), ``PGW_SERVICE_PKS_GPU_WIRE`` (the upload unpacked and the reply packed on the device) and
``PGW_SERVICE_PKS_GPU_KEYS`` (the reply keys unpacked and range-checked on the device, :func:`_unpack_ksk_dev`).
``PGW_SERVICE_PKS_PINNED`` (default on) stages the uploads and reply copies in pinned memory and overlaps them with
device work (a side stream for the uploads, staged in pieces by the host thread pool so the host copies overlap the
transfers, :func:`staged_copy`; double-buffered reply copies).
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from functools import cache
from typing import Any

import numpy as np
import torch

from pgw.hemask import pks
from pgw.hemask.pks import NR, NU, PACK, RANK, D, PKSParams

__all__ = [
    "FWD4",
    "KEYS",
    "RING",
    "SWITCH",
    "FusedChoice",
    "fused_self_test",
    "fused_status",
    "gemm_into",
    "host_ring_rows",
    "item",
    "item_bytes",
    "item_rows",
    "item_self_test",
    "keys_layout_ok",
    "keyswitch",
    "ntt_impl",
    "ntt_mode",
    "ring_rows",
    "ring_rows_async",
    "self_test",
    "staged_copy",
]

_PAR = PKSParams()
log = logging.getLogger(__name__)


@cache
def _tables(p: int) -> tuple[np.ndarray, np.ndarray, int]:
    nt = pks.Ntt.of(p, NR)
    return nt.z.astype(np.int64), nt.zi.astype(np.int64), int(nt.ninv)


def _dev_tables(p: int, dev: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
    z, zi, ninv = _tables(p)
    return torch.from_numpy(z).to(dev), torch.from_numpy(zi).to(dev), ninv


#: Butterfly stages of length below ``_S`` run on the transposed layout ``(rows, _S, NR / _S)`` (index
#: ``i = hi _S + lo`` stored at ``lo NR/_S + hi``): their pairs differ in the low bits ``lo``, so every operand is a
#: contiguous run of ``NR / _S = 128`` values instead of a stride-2 sequence of 1-32 values (in the natural layout
#: those stages were ~95 % of the key switch's CPU time, and uncoalesced on a GPU). Same butterflies, same
#: arithmetic, same twiddle per pair: bit-identical (``tests/service/test_pks_device_item.py``).
_S = 64
_H = NR // _S


def _tw_t(w: torch.Tensor, length: int) -> torch.Tensor:
    """Twiddles ``w[nb + b]`` of the stage of ``length`` (``nb = NR / 2 length`` blocks ``b = hi G + g``) laid
    out ``(G, 1, H)`` for the transposed layout (``G = _S / 2 length``)."""
    g = _S // (2 * length)
    nb = NR // (2 * length)
    return w[nb:2 * nb].view(_H, g).t().unsqueeze(1)


def _bf_fwd(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, p: int) -> None:
    """Cooley-Tukey butterfly in place: ``a, b <- (a + w b) mod p, (a - w b) mod p``."""
    t = b * w
    t.remainder_(p)
    d = a - t
    a.add_(t).remainder_(p)
    b.copy_(d.remainder_(p))


def _bf_inv(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, p: int) -> None:
    """Gentleman-Sande butterfly in place: ``a, b <- (a + b) mod p, (a - b) w mod p``."""
    d = a - b
    a.add_(b).remainder_(p)
    b.copy_(d.remainder_(p).mul_(w).remainder_(p))


def _fwd(x: torch.Tensor, p: int, z: torch.Tensor) -> torch.Tensor:
    """Negacyclic forward NTT mod ``p`` (bit-reversed output), as :meth:`pgw.hemask.pks.Ntt.fwd`."""
    lead = x.shape[:-1]
    x = x.reshape(-1, NR).clone()
    r = x.shape[0]
    length = NR // 2
    while length >= _S:
        nb = NR // (2 * length)
        v = x.view(r, nb, 2, length)
        _bf_fwd(v[:, :, 0], v[:, :, 1], z[nb:2 * nb, None], p)
        length //= 2
    t = x.view(r, _H, _S).transpose(1, 2).contiguous()
    del x
    while length >= 1:
        v = t.view(r, _S // (2 * length), 2, length, _H)
        _bf_fwd(v[:, :, 0], v[:, :, 1], _tw_t(z, length), p)
        length //= 2
    return t.transpose(1, 2).reshape(*lead, NR)


def _inv(x: torch.Tensor, p: int, zi: torch.Tensor, ninv: int) -> torch.Tensor:
    """Negacyclic inverse NTT mod ``p`` (bit-reversed input), as :meth:`pgw.hemask.pks.Ntt.inv`."""
    lead = x.shape[:-1]
    x = x.reshape(-1, NR)
    r = x.shape[0]
    t = x.view(r, _H, _S).transpose(1, 2).contiguous()  # a copy: the input is left alone
    length = 1
    while length < _S:
        v = t.view(r, _S // (2 * length), 2, length, _H)
        _bf_inv(v[:, :, 0], v[:, :, 1], _tw_t(zi, length), p)
        length *= 2
    x = t.transpose(1, 2).contiguous().view(r, NR)
    del t
    while length < NR:
        nb = NR // (2 * length)
        v = x.view(r, nb, 2, length)
        _bf_inv(v[:, :, 0], v[:, :, 1], zi[nb:2 * nb, None], p)
        length *= 2
    return x.mul_(ninv).remainder_(p).view(*lead, NR)


def _round_mul_shift(x: torch.Tensor, num: int, shift: int) -> torch.Tensor:
    """``floor((x num + 2^(shift-1)) / 2^shift)`` for ``x < 2^56``, ``num < 2^48`` (int64, 14-bit digits)."""
    m14 = (1 << 14) - 1
    t = [((x >> (14 * i)) & m14) * num for i in range(4)]
    i0, b0 = divmod(shift - 1, 14)
    t[i0] = t[i0] + (1 << b0)
    s = t[0]
    low = s & m14
    for i in (1, 2):
        s = (s >> 14) + t[i]
        low = low | ((s & m14) << (14 * i))
    s3 = (s >> 14) + t[3]
    if shift >= 42:
        return s3 >> (shift - 42)
    return (s3 << (42 - shift)) + (low >> shift)


def _round_pow2_div(x: torch.Tensor, bits: int, q: int) -> torch.Tensor:
    """``floor((x 2^bits + floor(q/2)) / q)`` for ``x < q < 2^48`` (12-bit long division in int64)."""
    quo = torch.zeros_like(x)
    rem = x.clone()
    left = bits
    while left > 0:
        s = min(12, left)
        left -= s
        a = rem << s
        if left == 0:
            a = a + q // 2
        quo = (quo << s) + torch.div(a, q, rounding_mode="floor")
        rem = a % q
    return quo


#: Device working memory of one key-switch call before its output rows are split into blocks
#: (``PGW_SERVICE_PKS_GPU_WORK_MB``). Every reply ciphertext (32 output rows) is computed independently.
WORK_BYTES = 512 << 20


def _block_bytes(groups: int) -> int:
    """Approximate peak device bytes per reply ciphertext (32 rows): the hint rows (4096 int64 each) and their
    copies and NTTs (~8x), the per-group products (2G x), the rows of ``y`` (3 x 256 G int64 each)."""
    return PACK * NU * 8 * (8 + 2 * groups) + PACK * D * groups * 8 * 3


def _work_bytes() -> int:
    import os

    raw = (os.environ.get("PGW_SERVICE_PKS_GPU_WORK_MB") or "").strip()
    return (max(16, int(raw)) << 20) if raw else WORK_BYTES


def keyswitch(hs: np.ndarray, ops: np.ndarray, y: np.ndarray, dev: Any, *, work_bytes: int | None = None) -> np.ndarray:
    """Reply ``(G, m/32, 8192)`` uint64, equal to :func:`pgw.hemask.pks_native.keyswitch` (reply form).

    Output rows go in blocks of whole reply ciphertexts so the device working memory stays near ``work_bytes``
    (:data:`WORK_BYTES`): unblocked, one item held ~``(8 + 2G) x 32 KiB`` per output row at once (e.g. ~19 GB
    for a 4-group item of 72B's gate_up, ~50 GB for 7B's 152k-row masked head). Bit-identical: every reply
    ciphertext depends only on its own 32 hint rows and 32 rows of ``y``."""
    m = hs.shape[0]
    nb, groups = m // PACK, ops.shape[0]
    budget = _work_bytes() if work_bytes is None else int(work_bytes)
    step = max(1, budget // _block_bytes(groups))
    ops_t = _Keys(torch.from_numpy(np.ascontiguousarray(ops).astype(np.int64)).to(dev))
    if step >= nb:
        return _keyswitch_rows(hs, ops_t, y, dev)
    out = np.empty((groups, nb, NR), dtype=np.uint64)
    for b0 in range(0, nb, step):
        b1 = min(nb, b0 + step)
        r0, r1 = b0 * PACK, b1 * PACK
        out[:, b0:b1] = _keyswitch_rows(hs[r0:r1], ops_t, y[r0:r1], dev)
    return out


def _keyswitch_rows(hs: np.ndarray, ops_t: _Keys, y: np.ndarray, dev: Any) -> np.ndarray:
    hs_t = torch.from_numpy(np.ascontiguousarray(hs).view(np.int64)).to(dev)
    y_t = torch.from_numpy(np.ascontiguousarray(y & np.uint64((1 << 56) - 1)).view(np.int64)).to(dev)
    return _keyswitch_t(hs_t, ops_t, y_t).cpu().numpy().astype(np.uint64)


# ---- NTT implementation choice (PGW_SERVICE_PKS_GPU_NTT) -----------------------------------------------------


def ntt_mode() -> str:
    """``PGW_SERVICE_PKS_GPU_NTT``: ``auto`` (default), ``torch`` or ``fused``."""
    raw = os.environ.get("PGW_SERVICE_PKS_GPU_NTT", "").strip().lower()
    if raw in ("", "auto"):
        return "auto"
    if raw in ("torch", "0", "off", "false", "no"):
        return "torch"
    if raw in ("fused", "triton", "1", "on", "true", "yes"):
        return "fused"
    raise ValueError(f"PGW_SERVICE_PKS_GPU_NTT={raw!r}: use auto, torch or fused")


_ntt_lock = threading.Lock()
#: Per device (``str(device)``): ``"fused"`` once its self-test passed under ``auto``, ``"torch"`` for good after a
#: failed self-test or any error of the fused kernels.
_ntt_choice: dict[str, str] = {}


def ntt_impl(dev: Any) -> str:
    """The NTT implementation the key switch uses on ``dev`` now: ``"fused"`` or ``"torch"``. Under ``auto`` the
    first call on a CUDA device runs :func:`fused_self_test` there (other devices use torch)."""
    mode = ntt_mode()
    if mode != "auto":
        return mode
    dev = torch.device(dev)
    key = str(dev)
    with _ntt_lock:
        got = _ntt_choice.get(key)
        if got is None:
            got = "torch"
            if dev.type == "cuda":
                try:
                    fused_self_test(dev)
                    got = "fused"
                except Exception as e:
                    log.warning("pks-v3 fused NTT unavailable on %s, using torch: %r", key, e)
            _ntt_choice[key] = got
        return got


def _ntt_fallback(dev: Any, err: BaseException) -> None:
    key = str(torch.device(dev))
    log.warning("pks-v3 fused NTT failed on %s, using torch from now on: %r", key, err)
    with _ntt_lock:
        _ntt_choice[key] = "torch"


class FusedChoice:
    """``auto`` / ``torch`` / ``fused`` selection of one fused kernel of the device item, as the NTT's
    (:func:`ntt_impl`): under ``auto`` (default) the kernel is used on a CUDA device only after ``test(dev)`` passed
    there, and any error of the self-test or of a later call falls back to the plain path on that device for good
    (same bits); ``fused`` forces it (errors propagate), ``torch`` disables it. ``choice``: per device ``"fused"`` or
    ``"torch"`` once decided."""

    def __init__(self, name: str, env: str, test: Any, default: str = "auto"):
        self.name, self.env, self.test, self.default = name, env, test, default
        self.choice: dict[str, str] = {}
        self._lock = threading.Lock()

    def mode(self) -> str:
        raw = os.environ.get(self.env, "").strip().lower()
        if raw == "":
            return self.default
        if raw == "auto":
            return "auto"
        if raw in ("torch", "host", "0", "off", "false", "no"):
            return "torch"
        if raw in ("fused", "triton", "device", "1", "on", "true", "yes"):
            return "fused"
        raise ValueError(f"{self.env}={raw!r}: use auto, torch or fused")

    def impl(self, dev: Any) -> str:
        mode = self.mode()
        if mode != "auto":
            return mode
        dev = torch.device(dev)
        key = str(dev)
        with self._lock:
            got = self.choice.get(key)
            if got is None:
                got = "torch"
                if dev.type == "cuda":
                    try:
                        self.test(dev)
                        got = "fused"
                    except Exception as e:
                        log.warning("pks-v3 fused %s unavailable on %s, using the plain path: %r", self.name, key, e)
                self.choice[key] = got
            return got

    def fallback(self, dev: Any, err: BaseException) -> None:
        key = str(torch.device(dev))
        log.warning("pks-v3 fused %s failed on %s, using the plain path from now on: %r", self.name, key, err)
        with self._lock:
            self.choice[key] = "torch"

    def run(self, dev: Any, fused: Any, plain: Any) -> Any:
        """``fused()`` when this kernel is chosen on ``dev`` (falling back to ``plain()`` for good on an error under
        ``auto``), else ``plain()``."""
        if self.impl(dev) == "fused":
            try:
                return fused()
            except Exception as e:
                if self.mode() == "fused":
                    raise
                self.fallback(dev, e)
        return plain()


def _fwd4_self_test(dev: Any, seed: int = 13) -> None:
    """Raise unless :func:`pgw.service.pks_ntt.forward4` equals the per-prime :func:`pgw.service.pks_ntt.forward`
    (itself self-tested against torch, :func:`fused_self_test`) on edge-case hint rows: 3 reply ciphertexts (one at
    the centring boundary, one at ``q1 - 1``) and 1."""
    from pgw.service import pks_ntt

    rng = np.random.default_rng(seed)
    hs = _edge_hint_rows(rng, 3 * PACK)
    hs[PACK:2 * PACK] = (_PAR.q1 + 1) // 2
    hs[-PACK:] = _PAR.q1 - 1
    hs_t = torch.from_numpy(hs.view(np.int64)).to(dev)
    for rows in (hs_t, hs_t[PACK:2 * PACK]):
        got = pks_ntt.forward4(rows)
        for k, p in enumerate(pks.QP):
            if not torch.equal(got[k], pks_ntt.forward(rows, p)):
                raise RuntimeError("pks-v3 fused 4-prime forward NTT does not match the per-prime kernel")


def _switch_self_test(dev: Any, seed: int = 17) -> None:
    """Raise unless :func:`pgw.service.pks_fused.combine` equals the torch recombination (and switch) bit for bit:
    int32 partials at ``+-(2^31 - 1)``, ``-2^31`` and random, constants at ``0``, ``-1``, ``+-2^63`` and random, shapes
    not multiples of the kernel's blocks, one row, and a strided output view."""
    from pgw.service import pks_fused
    from pgw.service.kernels import _combine_planes

    dev = torch.device(dev)
    rng = np.random.default_rng(seed)
    for r, c in ((40, 300), (32, 256), (1, 512)):
        part = rng.integers(-(1 << 31), 1 << 31, (r, PLANES * c), dtype=np.int64)
        part[0, :c] = (1 << 31) - 1
        if r > 2:
            part[1] = -(1 << 31)
            part[2, ::3] = 0
        const = rng.integers(-(1 << 63), (1 << 63) - 1, (r, 1), dtype=np.int64)
        const[0] = -1
        if r > 3:
            const[1], const[2], const[3] = 0, -(1 << 63), (1 << 63) - 1
        p_t = torch.from_numpy(part.astype(np.int32)).to(dev)
        c_t = torch.from_numpy(const).to(dev)
        want = _combine_planes([p_t[:, d * c:(d + 1) * c] for d in range(PLANES)], c_t).bitwise_and_(_M56)
        for switch in (False, True):
            buf = torch.full((r, c + 9), -7, dtype=torch.int64, device=dev)
            got = pks_fused.combine(p_t, c_t, buf[:, 4:4 + c], switch)
            ref = _round_mul_shift(want, _PAR.q1, 56) % _PAR.q1 if switch else want
            kept = bool((buf[:, :4] == -7).all()) and bool((buf[:, 4 + c:] == -7).all())
            if not torch.equal(got, ref) or not kept:
                raise RuntimeError("pks-v3 fused recombination / hint switch does not match the torch path")


def _ring_self_test(dev: Any) -> None:
    """Raise unless :func:`pgw.service.pks_fused.ring_rows` equals :func:`pgw.hemask.pks.public_ring_rows` (the host
    ChaCha20 stream) bit for bit for two public seeds and linears, 1 and 3 ring rows (a partial last program)."""
    from pgw.service import pks_fused

    for seed, n, linear in ((bytes(range(32)), RANK, "self-test"), (b"\xff" * 32, 3 * RANK, "0.down")):
        got = pks_fused.ring_rows(seed, n, linear, dev).cpu().numpy().view(np.uint64)
        if not np.array_equal(got, pks.public_ring_rows(seed, n, linear)):
            raise RuntimeError("pks-v3 device ring rows do not match the host stream")


#: ``PGW_SERVICE_PKS_GPU_FWD4``: the key switch's forward NTTs of all four primes in one launch, coalesced gather
#: (:func:`pgw.service.pks_ntt.forward4`); used only while the fused NTTs are (:func:`ntt_impl`). Off unless set
#: (``auto`` or ``fused``): exact on the Kaggle T4 but slower there (3.14 against 2.50 us per output row for 4 groups;
#: the transforms are bound by the SMs, and the per-prime kernel's strided gather costs less than four on-chip
#: transposes of the tile); the RunPod session measures both on an H100.
FWD4 = FusedChoice("4-prime forward NTT", "PGW_SERVICE_PKS_GPU_FWD4", _fwd4_self_test, default="torch")
#: ``PGW_SERVICE_PKS_GPU_SWITCH``: the device item's plane recombination and hint switch in one pass
#: (:func:`pgw.service.pks_fused.combine`).
SWITCH = FusedChoice("hint switch + recombination", "PGW_SERVICE_PKS_GPU_SWITCH", _switch_self_test)
#: ``PGW_SERVICE_PKS_GPU_RING``: the customer's public ring rows expanded by the device (ChaCha20,
#: :func:`pgw.service.pks_fused.ring_rows`) instead of on the host.
RING = FusedChoice("ring-row stream", "PGW_SERVICE_PKS_GPU_RING", _ring_self_test)


def fused_status(dev: Any) -> dict[str, str]:
    """What the device item uses on ``dev`` now (runs pending ``auto`` self-tests): NTT, 4-prime forward, switch,
    ring rows, wire forms, reply keys, each ``"fused"`` or ``"torch"``."""
    ntt = ntt_impl(dev)
    return {"ntt": ntt, "fwd4": FWD4.impl(dev) if ntt == "fused" else "torch", "switch": SWITCH.impl(dev),
            "ring": RING.impl(dev), "wire": WIRE.impl(dev), "keys": KEYS.impl(dev)}


class _Keys:
    """An item's reply keys on the device: ``t`` ``(G, 4, 16, 8192)`` int64 (the torch key switch's form) and
    ``u32()`` ``(4, G, 16, 8192)`` uint32 (the fused kernels' form), each made from the other on first use. The keys
    unpacked on the device (:func:`_unpack_ksk_dev`) start as the uint32 form."""

    __slots__ = ("_t", "_u")

    def __init__(self, t: torch.Tensor | None = None, *, u32: torch.Tensor | None = None):
        if (t is None) == (u32 is None):
            raise ValueError("_Keys: give the int64 or the uint32 form")
        self._t, self._u = t, u32

    @property
    def t(self) -> torch.Tensor:
        if self._t is None:
            assert self._u is not None
            # residues below 2^30: the int32 bits are the value
            self._t = self._u.view(torch.int32).to(torch.int64).transpose(0, 1).contiguous()
        return self._t

    @property
    def shape(self) -> tuple[int, ...]:
        if self._t is not None:
            return tuple(self._t.shape)
        assert self._u is not None
        f, g, r, n = self._u.shape
        return (g, f, r, n)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The forms made so far (for ``record_stream``)."""
        return tuple(x for x in (self._t, self._u) if x is not None)

    def u32(self) -> torch.Tensor:
        if self._u is None:
            from pgw.service import pks_ntt

            self._u = pks_ntt.ops_u32(self.t)
        return self._u


def _keyswitch_t(hs_t: torch.Tensor, ops_t: _Keys | torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """The key switch of whole reply ciphertexts on ``hs_t``'s device: ``hs_t`` ``(R, 4096)`` int64 in ``[0, q1)``
    (switched hint rows), ``y_t`` ``(R, 256 G)`` int64 below ``2^56``; returns ``(G, R/32, 8192)`` int64. Uses the
    fused NTT kernels when :func:`ntt_impl` says so (bit-identical), else :func:`_keyswitch_torch`."""
    keys = ops_t if isinstance(ops_t, _Keys) else _Keys(ops_t)
    dev = hs_t.device
    if ntt_impl(dev) == "fused":
        try:
            from pgw.service import pks_ntt

            u32 = keys.u32()
            return FWD4.run(dev, lambda: pks_ntt.keyswitch(hs_t, u32, y_t, fwd4=True),  # type: ignore[no-any-return]
                            lambda: pks_ntt.keyswitch(hs_t, u32, y_t))
        except Exception as e:
            if ntt_mode() == "fused":
                raise
            _ntt_fallback(dev, e)
    return _keyswitch_torch(hs_t, keys.t, y_t)


def _keyswitch_torch(hs_t: torch.Tensor, ops_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """:func:`_keyswitch_t` with eager torch NTTs (:func:`_fwd`, :func:`_inv`)."""
    q1, pm = _PAR.q1, _PAR.pmod
    m = hs_t.shape[0]
    nb, groups = m // PACK, ops_t.shape[0]
    dev = hs_t.device
    # A~[b, l, 32 u + j] = hs[32 b + j, 256 l + u], centred
    at = hs_t.view(nb, PACK, RANK, D).permute(0, 2, 3, 1).reshape(nb, RANK, NR)
    at = torch.where(at >= (q1 + 1) // 2, at - q1, at)
    ys = _round_mul_shift(y_t, q1, 56) % q1  # (m, 256 G)
    body = ys.view(nb, PACK, groups, D).permute(2, 0, 3, 1).reshape(groups, nb, NR)
    res = []
    for i, p in enumerate(pks.QP):
        z, zi, ninv = _dev_tables(p, dev)
        ah = _fwd(at % p, p, z)  # (nb, 16, NR)
        acc = (ah[None] * ops_t[:, i][:, None] % p).sum(dim=2) % p  # (G, nb, NR)
        acc = _inv(acc, p, zi, ninv)
        res.append((body % p * (pm % p) - acc) % p)
    p3, p4 = pks.PS
    k = (res[3] - res[2] % p4) % p4 * pow(p3, -1, p4) % p4
    cp = res[2] + p3 * k
    cp = torch.where(cp >= (pm + 1) // 2, cp - pm, cp)
    o = [(res[i] - cp % p) % p * pow(pm, -1, p) % p for i, p in enumerate(pks.Q1)]
    p1, p2 = pks.Q1
    kk = (o[1] - o[0] % p2) % p2 * pow(p1, -1, p2) % p2
    c0 = o[0] + p1 * kk
    return _round_pow2_div(c0, _PAR.yb, q1) & ((1 << _PAR.yb) - 1)


# ---- one whole item on the device: hint GEMM, modulus switch, upload GEMM and key switch ---------------------
#
# The provider-side hint ``round(W8 A q1 / 2^56) mod q1`` is ``m x 4096`` uint64 per (model, customer seed, linear):
# 50 / 157 / 230 GB per seed at 7B / 32B / 72B, far above a host cache. Instead of rebuilding it on the host for
# every item (a 4096-column GEMM, its result copied to the host, the host modulus switch, the switched hint copied
# back for the key switch), :func:`item` recomputes it on the device per item, in row blocks of whole reply
# ciphertexts, right next to the item's own GEMM ``y = W8 b mod 2^56``: per row block the hint rows and ``y`` rows
# exist only on the device and only for that block. Only the customer's public ring rows ``a_C`` (``n/16 x 4096``,
# 1/16 of ``A``) go up, the uploaded ``b`` once, and the ``(G, R/32, 8192)`` reply rows come back.

#: Columns per hint GEMM block (two upload components of 256 columns).
COL_BLOCK = 512
PLANES = 7  # values below 2^56: the top byte plane is zero
_M56 = (1 << 56) - 1


@cache
def _kl_index() -> np.ndarray:
    """``(16, 16)`` source index of entry ``(k, l)`` of :func:`pgw.hemask.pks.a_mlwe_rows` in ``cat(a, X a)``:
    ``a_{k-l}`` for ``l <= k``, ``X a_{k-l+16}`` (index ``16 + k - l + 16``) otherwise."""
    k = np.arange(RANK)[:, None]
    li = np.arange(RANK)[None, :]
    return np.where(li <= k, k - li, k - li + 2 * RANK).astype(np.int64)


def ring_rows(a_seed: bytes, n: int, linear: str, dev: Any) -> torch.Tensor:
    """The customer's public ring rows ``a_C`` of one linear (``(n/16, 4096)`` int64 below ``2^56``) on ``dev``
    (:func:`pgw.hemask.pks.public_ring_rows`: the same public stream the host path expands). Generated by the device
    (ChaCha20, :func:`pgw.service.pks_fused.ring_rows`) when :data:`RING` says so, else on the host in parallel
    threads (:func:`host_ring_rows`, off the GIL) and uploaded."""
    dev = torch.device(dev)

    def fused() -> torch.Tensor:
        from pgw.service import pks_fused

        return pks_fused.ring_rows(bytes(a_seed), int(n), linear, dev)

    return RING.run(dev, fused, lambda: _upload(host_ring_rows(a_seed, n, linear), dev))  # type: ignore[no-any-return]


class _HostRing:
    """The host ring-row stream of one item running in the background (:func:`ring_rows_async`); ``result()`` waits
    for it and uploads it on the caller's current stream."""

    def __init__(self, fut: Any, dev: torch.device):
        self.fut, self.dev = fut, dev

    def result(self) -> torch.Tensor:
        return _upload(self.fut.result(), self.dev)


_outer_pool_obj: Any = None


def ring_rows_async(a_seed: bytes, n: int, linear: str, dev: Any) -> torch.Tensor | _HostRing:
    """:func:`ring_rows` for :func:`item`, overlapped: the device kernel is asynchronous already; the host fallback
    starts in the background (off the GIL) and :func:`item` waits for it only after queuing the upload of ``b``."""
    global _outer_pool_obj
    dev = torch.device(dev)
    if RING.impl(dev) == "fused":
        return ring_rows(a_seed, n, linear, dev)
    with _host_pool_lock:
        if _outer_pool_obj is None:
            from concurrent.futures import ThreadPoolExecutor

            _outer_pool_obj = ThreadPoolExecutor(4, thread_name_prefix="pgw-pks-ring-item")
    return _HostRing(_outer_pool_obj.submit(host_ring_rows, bytes(a_seed), int(n), linear), dev)


_RING_ROWS_PER_TASK = 64  # ring rows (512 KiB of keystream each 64) per host task


def _host_threads() -> int:
    raw = (os.environ.get("PGW_SERVICE_PKS_HOST_THREADS") or "").strip()
    return max(1, int(raw)) if raw else max(1, min(8, os.cpu_count() or 1))


_host_pool_obj: Any = None
_host_pool_lock = threading.Lock()


def _host_pool() -> Any:
    global _host_pool_obj
    with _host_pool_lock:
        if _host_pool_obj is None:
            from concurrent.futures import ThreadPoolExecutor

            _host_pool_obj = ThreadPoolExecutor(_host_threads(), thread_name_prefix="pgw-pks-ring")
        return _host_pool_obj


def host_ring_rows(a_seed: bytes, n: int, linear: str) -> np.ndarray:
    """:func:`pgw.hemask.pks.public_ring_rows` (``(n/16, 4096)`` int64 view) computed in parallel: the stream is cut at
    ring-row boundaries (ChaCha20 block-counter seeks, :func:`pgw.core.prng._ring_stream_into`) and the pieces run on
    a thread pool (OpenSSL's ChaCha20 runs without the GIL). The same bytes as the one-shot stream."""
    from pgw.core.prng import _ring_stream_into
    from pgw.service.pks_fused import chacha_key

    if n % RANK:
        raise ValueError(f"in_features {n} is not a multiple of {RANK}")
    rows = int(n) // RANK
    key = chacha_key(a_seed, linear)
    out = np.empty((rows, NU), dtype=np.int64)
    step = _RING_ROWS_PER_TASK
    if rows <= step or _host_threads() == 1:
        _ring_stream_into(key, 0, out, 56)
        return out
    futs = [_host_pool().submit(_ring_stream_into, key, 8 * NU * r0, out[r0:r0 + step], 56)
            for r0 in range(0, rows, step)]
    for f in futs:
        f.result()
    return out


def _upload(a: np.ndarray, dev: torch.device) -> torch.Tensor:
    """``a`` on ``dev`` (uint64 as int64 bits): through a pinned staging buffer and a non-blocking copy on the current
    stream on CUDA (PyTorch's caching host allocator keeps the buffer until the copy finished), plain elsewhere."""
    a = np.ascontiguousarray(a)
    src = torch.from_numpy(a.view(np.int64) if a.dtype == np.uint64 else a)
    if dev.type != "cuda" or not _pinned_ok():
        return src.to(dev)
    raw = _stage_upload(a.reshape(-1).view(np.uint8), dev)
    return raw.view(src.dtype).view(src.shape)


#: Bytes per piece of a staged upload (:func:`_stage_upload`): the host copies piece ``k + 1`` into pinned memory
#: while piece ``k`` crosses PCIe.
UPLOAD_CHUNK = 8 << 20
_copy_pool_obj: Any = None


def _copy_pool() -> Any:
    global _copy_pool_obj
    with _host_pool_lock:
        if _copy_pool_obj is None:
            from concurrent.futures import ThreadPoolExecutor

            _copy_pool_obj = ThreadPoolExecutor(_host_threads(), thread_name_prefix="pgw-pks-stage")
        return _copy_pool_obj


def staged_copy(src: np.ndarray, dst: np.ndarray, on_piece: Any, chunk: int = UPLOAD_CHUNK) -> None:
    """Copy the bytes ``src`` into ``dst`` (both flat uint8, the same size) in pieces of ``chunk`` bytes on the host
    thread pool (numpy copies run without the GIL), calling ``on_piece(a, b)`` in order as soon as bytes ``a .. b - 1``
    are in ``dst`` (:func:`_stage_upload` queues their transfer there, so the copies and the transfers overlap). One
    piece, or ``PGW_SERVICE_PKS_HOST_THREADS=1``: one copy on the calling thread."""
    if src.shape != dst.shape:
        raise ValueError("staged_copy: size mismatch")
    size = int(src.size)
    bounds = [(a, min(size, a + chunk)) for a in range(0, size, max(1, int(chunk)))]
    if len(bounds) <= 1 or _host_threads() == 1:
        np.copyto(dst, src)
        if size:
            on_piece(0, size)
        return
    pool = _copy_pool()
    futs = [pool.submit(np.copyto, dst[a:b], src[a:b]) for a, b in bounds]
    try:
        for (a, b), f in zip(bounds, futs, strict=True):
            f.result()
            on_piece(a, b)
    finally:
        for f in futs:  # never leave a copy running into a buffer the caller may release
            f.cancel()
        for f in futs:
            if not f.cancelled():
                f.exception()


def _stage_upload(src: np.ndarray, dev: torch.device) -> torch.Tensor:
    """The flat uint8 bytes ``src`` as a uint8 tensor on the CUDA device ``dev``: copied into a pinned buffer in pieces
    by the host thread pool, each piece's non-blocking transfer queued on the current stream as soon as it is there
    (:func:`staged_copy`), so the host copy (~7 GB/s on one thread) and PCIe overlap instead of adding up. PyTorch's
    caching host allocator keeps the pinned buffer until its transfers finished."""
    pin = torch.empty(src.shape, dtype=torch.uint8, pin_memory=True)
    out = torch.empty(src.shape, dtype=torch.uint8, device=dev)

    def piece(a: int, b: int) -> None:
        out[a:b].copy_(pin[a:b], non_blocking=True)

    staged_copy(src, pin.numpy(), piece)
    return out


def _pinned_ok() -> bool:
    """``PGW_SERVICE_PKS_PINNED`` (default on): pinned, overlapped uploads and reply copies of the device item."""
    raw = (os.environ.get("PGW_SERVICE_PKS_PINNED") or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _a_source(ring_t: torch.Tensor) -> torch.Tensor:
    """``(n/16, 32, 256)``: the components ``a_j`` of every ring row and their negacyclic shifts ``X a_j``."""
    nb = ring_t.shape[0]
    comps = ring_t.view(nb, D, RANK).transpose(1, 2)  # comps[C, j, u] = a_C[16 u + j]
    x = torch.roll(comps, 1, dims=-1)
    x[..., 0] = torch.neg(x[..., 0]).bitwise_and_(_M56)
    return torch.cat([comps, x], dim=1)


def a_columns(src: torch.Tensor, l0: int, l1: int) -> torch.Tensor:
    """Columns ``256 l0 .. 256 l1 - 1`` of ``A = a_mlwe_rows(ring)`` (``(n, 256 (l1 - l0))`` int64) from
    :func:`_a_source`, built on its device (bit-identical to the host expansion)."""
    nb = src.shape[0]
    idx = torch.from_numpy(_kl_index()[:, l0:l1].copy()).to(src.device)  # (16, L)
    return src[:, idx].reshape(nb * RANK, (l1 - l0) * D)


def planes_of(vals: torch.Tensor, np_: int) -> torch.Tensor:
    """``(7, c, np_)`` uint8 GEMM operand of ``vals`` (``(n, c)`` int64 below ``2^56``): entry ``[d, j, i]`` is
    ``byte_d(vals[i, j]) ^ 0x80`` (``byte - 128`` as int8); inputs ``n .. np_ - 1`` are zero."""
    n, c = vals.shape
    out = torch.empty((PLANES, c, np_), dtype=torch.uint8, device=vals.device)
    raw = vals.contiguous().view(torch.uint8).view(n, c, 8)  # little-endian words (checked by the caller)
    torch.bitwise_xor(raw[:, :, :PLANES].permute(2, 1, 0), 0x80, out=out[:, :, :n])
    if np_ > n:
        out[:, :, n:].zero_()
    return out


def gemm_mod56(w: torch.Tensor, planes: torch.Tensor, const: torch.Tensor) -> torch.Tensor:
    """``W8 x mod 2^56`` (``(R, c)`` int64) for weight rows ``w`` (``(R, np_)`` int8, zero padded inputs) and
    :func:`planes_of` of ``x``: one int8 GEMM over the 7 planes and the exact int64 recombination of
    :func:`pgw.service.kernels._combine_planes` (``const``: ``(R, 1)`` int64, ``128 rowsum sum_d 2^(8d)`` mod
    ``2^64``, the ``-128`` plane offset). The same arithmetic as :meth:`ComputeKernel.mulmod64` with 7 bytes."""
    from pgw.service.kernels import _combine_planes

    nb, c, np_ = planes.shape
    part = torch._int_mm(w, planes.view(torch.int8).view(nb * c, np_).t()).view(w.shape[0], nb, c)
    y = _combine_planes([part[:, d, :] for d in range(nb)], const)
    del part
    return y.bitwise_and_(_M56)


def gemm_into(w: torch.Tensor, planes: torch.Tensor, const: torch.Tensor, out: torch.Tensor, switch: bool) -> None:
    """``out = gemm_mod56(w, planes, const)``, switched to ``round(. q1 / 2^56) mod q1`` when ``switch`` (``out``: an
    ``(R, c)`` int64 view). The recombination (and switch) runs as one fused kernel when :data:`SWITCH` says so
    (:func:`pgw.service.pks_fused.combine`: the int32 partials read once, one int64 written), else as eager torch
    operators; the same bits."""
    nb, c, np_ = planes.shape
    part = torch._int_mm(w, planes.view(torch.int8).view(nb * c, np_).t())  # (R, 7 c): plane-major columns

    def fused() -> None:
        from pgw.service import pks_fused

        pks_fused.combine(part, const, out, switch)

    def plain() -> None:
        from pgw.service.kernels import _combine_planes

        h = _combine_planes([part[:, d * c:(d + 1) * c] for d in range(nb)], const).bitwise_and_(_M56)
        out.copy_(_round_mul_shift(h, _PAR.q1, 56) % _PAR.q1 if switch else h)

    SWITCH.run(w.device, fused, plain)


def _row_bytes(n_pad: int, cols: int, groups: int) -> int:
    """Device bytes per output row of one row block of :func:`item`: the switched hint row (4096 int64), the row of
    ``y`` (``cols`` int64), the per-block GEMM partials and switch temporaries (``COL_BLOCK`` wide, int32 partials
    of 7 planes, ~7 int64 temporaries of the 14-bit-digit switch) and the key switch's share
    (:func:`_block_bytes` per 32 rows)."""
    return 8 * NU + 8 * cols + PLANES * COL_BLOCK * 4 + 8 * 8 * COL_BLOCK + _block_bytes(groups) // PACK


def item_rows(m: int, n_pad: int, cols: int, groups: int, work_bytes: int | None = None) -> int:
    """Rows per block of :func:`item` (a multiple of 32, at most ``m``) within ``work_bytes``
    (``PGW_SERVICE_PKS_GPU_WORK_MB``, default :data:`WORK_BYTES`)."""
    budget = _work_bytes() if work_bytes is None else int(work_bytes)
    r = max(PACK, budget // _row_bytes(n_pad, cols, groups) // PACK * PACK)
    return min(r, -(-m // PACK) * PACK)


def item_bytes(m: int, n: int, n_pad: int, cols: int, groups: int, work_bytes: int | None = None) -> int:
    """Modelled peak device bytes of one :func:`item` (checked against measured allocations on CPU,
    ``tests/service/test_pks_device_item.py``): the hint's GEMM operand (``7 x 4096 x n_pad`` bytes, held for the
    item), the upload's (``7 x cols x n_pad``), the ring rows, the plane build's transients and one row block."""
    rows = item_rows(m, n_pad, cols, groups, work_bytes)
    held = PLANES * (NU + cols) * n_pad + 8 * (n // RANK) * NU
    build = 2 * 8 * (n // RANK) * 2 * NU + 2 * 8 * n * COL_BLOCK  # cat(a, X a) + roll, one column block (+ copy)
    # on CUDA the upload's planes are built on a side stream at the same time: its wire bytes and one column block
    upload = 7 * n * cols + 8 * n * COL_BLOCK
    return held + max(build + upload, rows * _row_bytes(n_pad, cols, groups))


def item(w: torch.Tensor, const: torch.Tensor, m: int, n: int, ring_t: Any, b: np.ndarray | PackedUpload,
         ops: np.ndarray | PackedKeys, *, work_bytes: int | None = None, out: np.ndarray | PackedReply | None = None,
         out_b0: int = 0) -> np.ndarray | PackedReply:
    """The reply ``(G, m/32, 8192)`` uint64 of one pks-v3 item, computed on ``w``'s device.

    ``w``: the kernel's int8 weight (``(>= m, n_pad)``, rows and inputs beyond ``(m, n)`` zero), ``const``: its
    ``(>= m, 1)`` maskgen constant for 7 byte planes (:meth:`ComputeKernel._maskgen_const`), ``ring_t``:
    :func:`ring_rows` (or :func:`ring_rows_async`), ``b``: the item's ``(n, 256 G)`` upload below ``2^56``, ``ops``:
    ``(G, 4, 16, 8192)`` reply
    keys. Equal bit for bit to ``keyswitch(switch_hint(W8 A mod 2^56), ops, W8 b mod 2^56)`` (the host path of
    :class:`pgw.service.pks_provider.PKSProvider`): the same GEMM arithmetic (:func:`gemm_mod56`), the same
    exact 14-bit-digit switch (:func:`_round_mul_shift`) and the same key switch (:func:`_keyswitch_t`, whose
    reply ciphertexts depend only on their own 32 rows).

    ``b`` may be the upload's wire bytes (:class:`PackedUpload`, unpacked on the device when :data:`WIRE` says so,
    else on the host), ``ops`` the reply keys' wire bytes (:class:`PackedKeys`, unpacked and range-checked on the
    device when :data:`KEYS` says so, else on the host; a residue out of range raises ``ProtocolError`` before any
    key switch); ``out`` an array or :class:`PackedReply` to write the reply rows into at reply ciphertext
    ``out_b0`` (tensor-parallel shards), returned."""
    dev = w.device
    np_ = int(w.shape[1])
    groups, cols = int(ops.shape[0]), int(b.shape[1])
    if m % PACK or n % RANK or b.shape[0] != n or cols != D * groups or w.shape[0] < m or np_ < n:
        raise ValueError("pks-v3 device item: bad shapes")
    if isinstance(b, PackedUpload) and WIRE.impl(dev) != "fused":
        b = b.host()
    side = _side_stream(dev) if dev.type == "cuda" and _pinned_ok() else None
    aplanes = _a_planes(ring_t, np_) if isinstance(ring_t, torch.Tensor) else None  # queued first
    if aplanes is not None:
        del ring_t
    # the uploads (reply keys, ``b`` and its planes) on a side stream: their host copies into pinned buffers and the
    # transfers overlap the ring rows, the hint operand's planes and the first row block's hint GEMMs (or, with the
    # host ring-row stream still running, that stream)
    check: _KeyCheck | None = None
    with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
        if isinstance(ops, PackedKeys):
            # the keys' wire bytes go up and the device unpacks and range-checks them (:data:`KEYS`), else the host
            # unpacks them (raising ``ProtocolError`` on a residue out of range) and they go up as uint32
            keys_in = ops
            ops_t, check = KEYS.run(dev, lambda: _device_keys(keys_in, dev),
                                    lambda: (_host_keys(keys_in.host(), dev), None))
        else:
            ops_t = _host_keys(ops, dev)
        bplanes = []
        raw = _upload_bytes(b.data, dev) if isinstance(b, PackedUpload) else None
        for c0 in range(0, cols, COL_BLOCK):
            if raw is not None:
                bt = _unpack56_cols(raw, n, cols, c0, min(cols, c0 + COL_BLOCK))
            else:
                bt = _upload(b[:, c0:c0 + COL_BLOCK], dev)  # type: ignore[index]
            bplanes.append(planes_of(bt, np_))
            del bt
        del raw
    if aplanes is None:
        aplanes = _a_planes(ring_t.result(), np_)
        del ring_t
    waited = side is None
    rows = item_rows(m, np_, cols, groups, work_bytes)
    if out is None:
        out = np.empty((groups, m // PACK, NR), dtype=np.uint64)
    copies = _ReplyCopies(out, dev, out_b0)
    for r0 in range(0, m, rows):
        r1 = min(m, r0 + rows)
        wr, cr = w[r0:r1], const[r0:r1]
        hs = torch.empty((r1 - r0, NU), dtype=torch.int64, device=dev)
        for j, pl in enumerate(aplanes):
            gemm_into(wr, pl, cr, hs[:, j * COL_BLOCK:(j + 1) * COL_BLOCK], True)
        if not waited:
            cur = torch.cuda.current_stream(dev)
            cur.wait_stream(side)  # type: ignore[arg-type]
            for t in (*ops_t.tensors(), *bplanes):  # allocated on the side stream, used (and freed) on this one
                t.record_stream(cur)
            waited = True
        if check is not None:  # the device's range check of the reply keys, before any key switch uses them
            check.verify()
            check = None
        y = torch.empty((r1 - r0, cols), dtype=torch.int64, device=dev)
        for j, pl in enumerate(bplanes):
            c0 = j * COL_BLOCK
            gemm_into(wr, pl, cr, y[:, c0:c0 + pl.shape[1]], False)
        res = _keyswitch_t(hs, ops_t, y)
        del hs, y
        copies.put(res, r0 // PACK, r1 // PACK)
        del res
    copies.finish()
    return out


def _host_keys(ops: np.ndarray, dev: torch.device) -> _Keys:
    """Reply keys unpacked on the host (``(G, 4, 16, 8192)``, residues below their primes) on ``dev``: up as uint32
    (half the bytes) and widened on the device (residues below 2^30: exact)."""
    return _Keys(_upload(np.ascontiguousarray(ops, dtype=np.uint32).view(np.int32), dev).to(torch.int64))


def _a_planes(ring_t: torch.Tensor, np_: int) -> list[torch.Tensor]:
    """The hint GEMM's operand: :func:`planes_of` of ``A = a_mlwe_rows(ring)``, column block by column block."""
    src = _a_source(ring_t)
    step = COL_BLOCK // D
    return [planes_of(a_columns(src, l0, l0 + step), np_) for l0 in range(0, RANK, step)]


_side = threading.local()


def _side_stream(dev: torch.device) -> Any:
    """This thread's upload stream on ``dev`` (one per thread and device: concurrent items run in their own threads
    on their own streams)."""
    cache = getattr(_side, "streams", None)
    if cache is None:
        cache = _side.streams = {}
    key = str(dev)
    if key not in cache:
        cache[key] = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call]
    return cache[key]


class _ReplyCopies:
    """The reply rows of :func:`item`'s row blocks back to the host, into an array or a :class:`PackedReply` (the wire
    bytes; packed on the device when :data:`WIRE` says so: 4.5 instead of 8 bytes per value cross PCIe and the host
    only copies, else packed on the host block by block). On CUDA (``PGW_SERVICE_PKS_PINNED``, default on) each
    block goes into pinned buffers by non-blocking copies on the current stream, and the host copies the previous
    block out while the device computes the next one; elsewhere plain copies."""

    def __init__(self, out: np.ndarray | PackedReply, dev: torch.device, b_off: int = 0):
        self.out, self.b_off = out, b_off
        self.overlap = dev.type == "cuda" and _pinned_ok()
        self.dev_pack = isinstance(out, PackedReply) and WIRE.impl(dev) == "fused"
        self.pending: tuple[Any, tuple[torch.Tensor, ...], int, int] | None = None

    def put(self, res: torch.Tensor, b0: int, b1: int) -> None:
        b0, b1 = b0 + self.b_off, b1 + self.b_off
        parts = _pack36_dev(res) if self.dev_pack else (res,)
        if not self.overlap:
            self._store(tuple(p.cpu() for p in parts), b0, b1)
            return
        host = []
        for p in parts:
            h = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
            h.copy_(p, non_blocking=True)
            host.append(h)
        ev = torch.cuda.Event()  # type: ignore[no-untyped-call]
        ev.record()
        self._drain()
        self.pending = (ev, tuple(host), b0, b1)

    def _store(self, parts: tuple[torch.Tensor, ...], b0: int, b1: int) -> None:
        if isinstance(self.out, PackedReply):
            if len(parts) == 2:
                self.out.low[:, b0:b1] = parts[0].numpy().view(np.uint32)
                self.out.tail[:, b0:b1] = parts[1].numpy()
            else:
                self.out.put_host(parts[0].numpy().view(np.uint64), b0, b1)
        else:
            self.out[:, b0:b1] = parts[0].numpy().view(np.uint64)

    def _drain(self) -> None:
        if self.pending is not None:
            ev, host, b0, b1 = self.pending
            ev.synchronize()
            self._store(host, b0, b1)
            self.pending = None

    def finish(self) -> None:
        self._drain()


# ---- the wire forms on the device (PGW_SERVICE_PKS_GPU_WIRE) ------------------------------------------------
#
# ``run_item`` used to unpack the upload ``b`` (56-bit wire fields) and pack the reply (36-bit fields) with numpy on
# one host thread: ~8 and ~20 ns per value on the development box, i.e. ~37 + ~147 s per 1024-slot cycle at 72B
# (4.4e9 upload and 7.2e9 reply values), far above the device work. Both formats are plain bit fields
# (:func:`pgw.hemask.wire.pack_bits`): the upload as 32 + 16 + 8-bit little-endian arrays, the reply as one 32-bit
# array and a 4-bit tail stream (two values per byte, the first in the low nibble). So the raw upload bytes go up
# (7 instead of 8 bytes per value) and are unpacked by the device, and the reply is packed by the device row block
# by row block (4.5 instead of 8 bytes per value come back) straight into the reply's wire buffer. Integer bit
# operations only: the same bytes (self-tested against ``pack_bits`` / ``unpack_bits`` before first use).

_M32 = (1 << 32) - 1


def wire_layout_ok() -> bool:
    """True when the v3 wire fields are the ones the device pack / unpack assume (56-bit upload: 32 + 16 + 8 bits;
    36-bit reply: 32 bits + a 4-bit tail)."""
    from pgw.hemask.wire import V3_Y_BITS, bit_fields

    return V3_Y_BITS == 36 and bit_fields(36) == ([(0, 32)], 4) and \
        bit_fields(56) == ([(0, 32), (32, 16), (48, 8)], 0)


class PackedUpload:
    """An item's upload ``b`` (``(n, cols)`` values below ``2^56``) in its wire form (``pack_bits(b, 56)``)."""

    def __init__(self, data: bytes | bytearray | memoryview, n: int, cols: int):
        if len(data) != n * cols * 7:
            raise ValueError("pks-v3 upload: wrong size")
        self.data, self.n, self.cols = data, int(n), int(cols)
        self.shape = (self.n, self.cols)
        self._host: np.ndarray | None = None

    def host(self) -> np.ndarray:
        """``unpack_bits`` on the host (cached): the fallback and the cache path."""
        if self._host is None:
            from pgw.hemask.wire import unpack_bits

            self._host = unpack_bits(self.data, self.n * self.cols, 56).reshape(self.n, self.cols)
        return self._host


class PackedReply:
    """The reply's wire bytes ``pack_bits(reply.reshape(-1), 36)`` (``reply``: ``(G, nb, 8192)``), filled row block
    by row block: ``low`` ``(G, nb, 8192)`` uint32 and ``tail`` ``(G, nb, 4096)`` uint8 views of ``buf``."""

    def __init__(self, groups: int, nb: int):
        count = groups * nb * NR
        self.buf = bytearray(count * 36 // 8)
        self.low = np.frombuffer(self.buf, dtype=np.uint32, count=count).reshape(groups, nb, NR)
        self.tail = np.frombuffer(self.buf, dtype=np.uint8, count=count // 2, offset=4 * count).reshape(groups, nb,
                                                                                                         NR // 2)

    def put_host(self, vals: np.ndarray, b0: int, b1: int) -> None:
        """Pack ``(G, b1 - b0, 8192)`` uint64 reply values below ``2^36`` on the host."""
        np.copyto(self.low[:, b0:b1], vals & np.uint64(_M32), casting="unsafe")
        t = ((vals >> np.uint64(32)) & np.uint64(15)).astype(np.uint8)
        self.tail[:, b0:b1] = t[..., 0::2] | (t[..., 1::2] << np.uint8(4))


def _pack36_dev(res: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(low, tail)`` of reply values ``res`` (int64 below ``2^36``) on their device: the low 32 bits as int32 (the
    same bits as uint32) and the 4-bit tails two per byte (uint8, the first value in the low nibble)."""
    low = res & _M32
    low = torch.where(low > (1 << 31) - 1, low - (1 << 32), low).to(torch.int32)
    t = (res >> 32) & 15
    tail = (t[..., 0::2] | (t[..., 1::2] << 4)).to(torch.uint8)
    return low, tail


def _upload_bytes(data: bytes | bytearray | memoryview, dev: torch.device) -> torch.Tensor:
    """Raw bytes as a uint8 tensor on ``dev`` (pinned and non-blocking on CUDA, as :func:`_upload`)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if dev.type != "cuda" or not _pinned_ok():
        return torch.from_numpy(arr.copy()).to(dev)
    return _stage_upload(arr, dev)


def _unpack56_cols(raw: torch.Tensor, n: int, cols: int, c0: int, c1: int) -> torch.Tensor:
    """Columns ``c0 .. c1 - 1`` of the ``(n, cols)`` upload from its wire bytes ``raw`` (uint8 on the device):
    ``low32 | mid16 << 32 | hi8 << 48`` (int64, below ``2^56``)."""
    cnt = n * cols
    low = raw[:4 * cnt].view(torch.int32).view(n, cols)[:, c0:c1].to(torch.int64) & _M32
    mid = raw[4 * cnt:6 * cnt].view(torch.int16).view(n, cols)[:, c0:c1].to(torch.int64) & 0xFFFF
    hi = raw[6 * cnt:7 * cnt].view(n, cols)[:, c0:c1].to(torch.int64)
    return low | (mid << 32) | (hi << 48)


def _wire_self_test(dev: Any, seed: int = 19) -> None:
    """Raise unless the device pack of a reply equals ``pack_bits(., 36)`` and the device unpack of an upload equals
    ``unpack_bits(., 56)``, on edge values (``0``, ``2^31 - 1``, ``2^31``, ``2^32 - 1``, ``2^32``, ``2^36 - 1``;
    ``2^56 - 1`` and the field boundaries of the upload)."""
    from pgw.hemask.wire import pack_bits

    if not wire_layout_ok():
        raise RuntimeError("pks-v3 wire fields changed: device pack / unpack disabled")
    dev = torch.device(dev)
    rng = np.random.default_rng(seed)
    groups, nb = 2, 3
    vals = rng.integers(0, 1 << 36, (groups, nb, NR), dtype=np.uint64)
    vals[0, 0, :8] = [0, (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32, (1 << 36) - 1, 1, (1 << 35)]
    vals[1, 2] = (1 << 36) - 1
    pr = PackedReply(groups, nb)
    low, tail = _pack36_dev(torch.from_numpy(vals.view(np.int64)).to(dev)[:, 1:])
    pr.low[:, 1:] = low.cpu().numpy().view(np.uint32)
    pr.tail[:, 1:] = tail.cpu().numpy()
    pr.put_host(vals[:, :1], 0, 1)
    if bytes(pr.buf) != bytes(pack_bits(vals.reshape(-1), 36)):
        raise RuntimeError("pks-v3 device reply pack does not match pack_bits")
    n, cols = 48, 1024 + 256
    b = rng.integers(0, 1 << 56, (n, cols), dtype=np.uint64)
    b[0, :10] = [0, (1 << 56) - 1, (1 << 32) - 1, 1 << 32, (1 << 48) - 1, 1 << 48, (1 << 31), (1 << 47), 1 << 55, 1]
    raw = _upload_bytes(pack_bits(b.reshape(-1), 56), dev)
    got = torch.cat([_unpack56_cols(raw, n, cols, c0, min(cols, c0 + COL_BLOCK)) for c0 in range(0, cols, COL_BLOCK)],
                    dim=1)
    if not np.array_equal(got.cpu().numpy().view(np.uint64), b):
        raise RuntimeError("pks-v3 device upload unpack does not match unpack_bits")


#: ``PGW_SERVICE_PKS_GPU_WIRE``: the upload unpacked and the reply packed on the device (the wire's bit fields).
WIRE = FusedChoice("wire pack / unpack", "PGW_SERVICE_PKS_GPU_WIRE", _wire_self_test)


# ---- the reply keys unpacked on the device (PGW_SERVICE_PKS_GPU_KEYS) -----------------------------------------
#
# ``run_item`` unpacked the item's reply keys (``G x 4 x 16 x 8192`` residues, 24 / 24 / 30 / 30-bit wire fields)
# with numpy on the host and checked every residue below its prime: ~3.5-4.4 ns per residue, ~7-9 ms per 4-group
# item, whatever the linear (~1.0 / 2.4 / 3.0 s per 1024-slot cycle at 7B / 32B / 72B: a quarter of 7B's cycle).
# Now the wire bytes go up as they are (1.77 MB per group instead of 2.1 MB of uint32) and the device unpacks them
# straight into the fused kernels' uint32 form. The range check stays: the device reduces it to one flag, and the
# item reads that flag before its first key switch and refuses the item (``ProtocolError``) exactly as the host did.


def keys_layout_ok() -> bool:
    """True when every reply-key field fits the device unpack (residues below ``2^31``, byte fields of at most 32
    bits and a tail below 8 bits: :func:`pgw.hemask.wire.bit_fields`)."""
    from pgw.hemask.wire import V3_KSK_BITS, V3_KSK_BYTES, bit_fields

    if len(V3_KSK_BITS) != len(pks.QP) or V3_KSK_BYTES != RANK * NR * sum(V3_KSK_BITS) // 8:
        return False
    for bits, p in zip(V3_KSK_BITS, pks.QP, strict=True):
        fields, _t = bit_fields(bits)
        if bits > 31 or p > 1 << bits or any(w > 32 for _o, w in fields):
            return False
    return True


class PackedKeys:
    """An item's reply keys in their wire form (``groups x V3_KSK_BYTES``, :func:`pgw.service.pks_provider.pack_ksk`),
    unpacked on the device when :data:`KEYS` says so. A wrong size is refused here, as the host unpack does."""

    def __init__(self, data: bytes | bytearray | memoryview, groups: int):
        from pgw.core.errors import ProtocolError
        from pgw.hemask.wire import V3_KSK_BYTES

        if len(data) != int(groups) * V3_KSK_BYTES:
            raise ProtocolError("maskgen v3: reply key size mismatch")
        self.data, self.groups = data, int(groups)
        self.shape = (self.groups, len(pks.QP), RANK, NR)
        self._host: np.ndarray | None = None

    def host(self) -> np.ndarray:
        """:func:`pgw.service.pks_provider.unpack_ksk` (cached; raises ``ProtocolError`` on a residue out of range)."""
        if self._host is None:
            from pgw.service.pks_provider import unpack_ksk

            self._host = unpack_ksk(self.data, self.groups)
        return self._host


def _unpack_ksk_dev(raw: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(u32, bad)`` from the reply keys' wire bytes ``raw`` (flat uint8 on the device): ``u32`` ``(4, G, 16, 8192)``
    uint32 (:func:`pgw.service.pks_ntt.ops_u32`'s layout), ``bad`` a 0-dim bool, true when a residue is not below its
    prime. The fields of :func:`pgw.hemask.wire.pack_bits`: little-endian byte fields, then a ``t``-bit tail stream
    (eight values per ``t`` bytes, the first in the low bits). Integer ops only (int32: residues below ``2^31``)."""
    from pgw.hemask.wire import V3_KSK_BITS, V3_KSK_BYTES, bit_fields

    dev = raw.device
    cnt = RANK * NR
    r = raw.view(groups, V3_KSK_BYTES)
    out = torch.empty((len(V3_KSK_BITS), groups, RANK, NR), dtype=torch.int32, device=dev)
    bad = torch.zeros((), dtype=torch.bool, device=dev)
    pos = i0 = 0
    while i0 < len(V3_KSK_BITS):  # runs of primes with the same width (24, 24 | 30, 30): one pass per run
        bits = V3_KSK_BITS[i0]
        k = 1
        while i0 + k < len(V3_KSK_BITS) and V3_KSK_BITS[i0 + k] == bits:
            k += 1
        nb = cnt * bits // 8
        seg = r[:, pos:pos + k * nb].reshape(groups, k, nb)
        pos += k * nb
        fields, t = bit_fields(bits)
        v: torch.Tensor | None = None
        fp = 0
        for off, width in fields:
            wb = width // 8
            fb = seg[:, :, fp:fp + cnt * wb].reshape(groups, k, cnt, wb).to(torch.int32)
            fp += cnt * wb
            for j in range(wb):
                part = fb[..., j] << (off + 8 * j) if off + 8 * j else fb[..., j]
                v = part.clone() if v is None else v.bitwise_or_(part)
        if t:
            tb = seg[:, :, fp:fp + cnt * t // 8].reshape(groups, k, cnt // 8, t).to(torch.int64)
            fp += cnt * t // 8
            word = tb[..., 0].clone()
            for j in range(1, t):
                word.bitwise_or_(tb[..., j] << (8 * j))
            sh = torch.arange(8, dtype=torch.int64, device=dev) * t
            tails = (word[..., None] >> sh).bitwise_and_((1 << t) - 1).reshape(groups, k, cnt).to(torch.int32)
            tails = tails << (bits - t)
            v = tails if v is None else v.bitwise_or_(tails)
        if v is None or fp != nb:  # pragma: no cover - keys_layout_ok guards the layout
            raise RuntimeError("pks-v3 reply key layout mismatch")
        primes = torch.tensor(pks.QP[i0:i0 + k], dtype=torch.int32).to(dev).view(1, k, 1)
        bad.logical_or_((v >= primes).any())
        out[i0:i0 + k] = v.transpose(0, 1).reshape(k, groups, RANK, NR)
        i0 += k
    if pos != V3_KSK_BYTES:  # pragma: no cover - keys_layout_ok guards the layout
        raise RuntimeError("pks-v3 reply key layout mismatch")
    return out.view(torch.uint32), bad


class _KeyCheck:
    """The device's verdict on an item's reply keys (``bad``): copied to pinned memory without blocking where it may,
    read by :meth:`verify` before the item's first key switch."""

    def __init__(self, bad: torch.Tensor, dev: torch.device):
        self.bad = bad
        self.ev: Any = None
        self.host: torch.Tensor | None = None
        if dev.type == "cuda" and _pinned_ok():  # pragma: no cover - GPU only
            self.host = torch.empty((1,), dtype=torch.bool, pin_memory=True)
            self.host.copy_(bad.view(1), non_blocking=True)
            self.ev = torch.cuda.Event()  # type: ignore[no-untyped-call]
            self.ev.record()

    def verify(self) -> None:
        """Raise ``ProtocolError`` when a reply key residue is out of range (as :func:`unpack_ksk`)."""
        from pgw.core.errors import ProtocolError

        if self.ev is not None:  # pragma: no cover - GPU only
            self.ev.synchronize()
            flag = bool(self.host[0])  # type: ignore[index]
        else:
            flag = bool(self.bad)
        if flag:
            raise ProtocolError("maskgen v3: reply key residue out of range")


def _device_keys(ops: PackedKeys, dev: torch.device) -> tuple[_Keys, _KeyCheck]:
    u32, bad = _unpack_ksk_dev(_upload_bytes(ops.data, dev), ops.groups)
    return _Keys(u32=u32), _KeyCheck(bad, dev)


def _keys_self_test(dev: Any, seed: int = 23) -> None:
    """Raise unless the device unpack of reply keys equals :func:`pgw.service.pks_provider.unpack_ksk` bit for bit
    (2 groups, residues at ``0``, ``1``, ``p - 2``, ``p - 1`` and every byte boundary of each field) and flags a
    residue out of range (``p`` and ``2^bits - 1`` in each prime, one group only) without flagging the good ones."""
    from pgw.hemask.wire import V3_KSK_BITS, pack_bits
    from pgw.service.pks_provider import unpack_ksk

    if not keys_layout_ok():
        raise RuntimeError("pks-v3 reply key fields changed: device unpack disabled")
    dev = torch.device(dev)
    rng = np.random.default_rng(seed)
    groups = 2
    beta = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                     for _ in range(groups)])
    for i, p in enumerate(pks.QP):
        edge = [0, 1, p - 2, p - 1, 255, 256, 65535, 65536, (1 << 24) - 1, 1 << 24]
        beta[0, i, 0, :len(edge)] = [e for e in edge if e < p] + [0] * sum(e >= p for e in edge)
        beta[1, i, RANK - 1, -8:] = p - 1
    beta[1, :, 3] = 0

    def packed(b: np.ndarray) -> bytes:
        out = bytearray()
        for g in range(b.shape[0]):
            for i, bits in enumerate(V3_KSK_BITS):
                out += pack_bits(np.ascontiguousarray(b[g, i]).reshape(-1), bits)
        return bytes(out)

    data = packed(beta)
    u32, bad = _unpack_ksk_dev(_upload_bytes(data, dev), groups)
    want = unpack_ksk(data, groups)
    if bool(bad) or not np.array_equal(u32.cpu().view(torch.int32).numpy().view(np.uint32),
                                       want.transpose(1, 0, 2, 3)):
        raise RuntimeError("pks-v3 device reply-key unpack does not match unpack_ksk")
    for i, (bits, p) in enumerate(zip(V3_KSK_BITS, pks.QP, strict=True)):
        for v in (p, (1 << bits) - 1):
            b1 = beta[1:].copy()
            b1[0, i, 5, 7] = v
            _u, bad = _unpack_ksk_dev(_upload_bytes(packed(b1), dev), 1)
            if not bool(bad):
                raise RuntimeError("pks-v3 device reply-key unpack misses a residue out of range")


#: ``PGW_SERVICE_PKS_GPU_KEYS``: the item's reply keys unpacked (and range-checked) on the device.
KEYS = FusedChoice("reply-key unpack", "PGW_SERVICE_PKS_GPU_KEYS", _keys_self_test)


def item_self_test(dev: Any, seed: int = 5) -> None:
    """Raise unless :func:`item` on ``dev`` equals the host path (exact numpy GEMM, :func:`pgw.hemask.pks.switch_hint`
    and the CPU key switch) on public random data, with a zero-padded weight as on CUDA (64 x 48 -> 64 x 64)."""
    from pgw.hemask.lwe import matmul_mod64
    from pgw.hemask.pks_native import keyswitch as cpu

    rng = np.random.default_rng(seed)
    m, n, groups = 2 * PACK, 3 * RANK, 1
    w8 = rng.integers(-128, 128, (m, n), dtype=np.int64).astype(np.int8)
    w8[0, :] = -128  # extreme operands
    w8[1, :] = 127
    wpad = torch.zeros((m, 64), dtype=torch.int8)
    wpad[:, :n] = torch.from_numpy(w8)
    scale = (128 * sum(1 << (8 * d) for d in range(PLANES))) % (1 << 64)
    cu = w8.astype(np.int64).sum(1).view(np.uint64) * np.uint64(scale)
    const = torch.from_numpy(cu.view(np.int64).reshape(m, 1).copy())
    a_seed = bytes(range(32))
    b = rng.integers(0, 1 << 56, (n, D * groups), dtype=np.uint64)
    b[0] = (1 << 56) - 1
    ops = np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])[None].astype(np.uint32)
    got = item(wpad.to(dev), const.to(dev), m, n, ring_rows(a_seed, n, "self-test", dev), b, ops)
    a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, n, "self-test"))
    hs = pks.switch_hint(matmul_mod64(w8, a) & np.uint64(_M56), _PAR)
    want = cpu(hs, ops, matmul_mod64(w8, b) & np.uint64(_M56))
    if not np.array_equal(got, want):
        raise RuntimeError("pks-v3 device item does not match the host path")
    if wire_layout_ok():  # the wire forms: the upload's bytes in, the reply's bytes out (as PKSProvider)
        from pgw.hemask.wire import V3_Y_BITS, pack_bits
        from pgw.service.pks_provider import pack_ksk

        keys: np.ndarray | PackedKeys = PackedKeys(pack_ksk(ops), ops.shape[0]) if keys_layout_ok() else ops
        pr = item(wpad.to(dev), const.to(dev), m, n, ring_rows(a_seed, n, "self-test", dev),
                  PackedUpload(pack_bits(b.reshape(-1), 56), n, b.shape[1]), keys,
                  out=PackedReply(ops.shape[0], m // PACK))
        if not isinstance(pr, PackedReply) or bytes(pr.buf) != bytes(pack_bits(want.reshape(-1), V3_Y_BITS)):
            raise RuntimeError("pks-v3 device item (wire forms) does not match the host path")


def self_test(dev: Any, seed: int = 7) -> None:
    """Raise unless :func:`keyswitch` on ``dev`` equals the CPU path on public random data (64 rows, 1 group)."""
    from pgw.hemask.pks_native import keyswitch as cpu

    rng = np.random.default_rng(seed)
    m = 2 * PACK
    hs = rng.integers(0, _PAR.q1, (m, NU), dtype=np.uint64)
    ops = np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])[None].astype(np.uint32)
    y = rng.integers(0, 1 << 56, (m, D), dtype=np.uint64)
    got = keyswitch(hs, ops, y, dev)
    want = cpu(hs, ops, y)
    if not np.array_equal(got, want):
        raise RuntimeError("pks-v3 device key switch does not match the CPU path")


def _edge_hint_rows(rng: np.random.Generator, m: int) -> np.ndarray:
    """``(m, 4096)`` switched hint rows in ``[0, q1)``: random, with the centring boundary, ``0``, ``q1 - 1`` and the
    24-bit split boundaries of the fused input reduction in the first rows, and two whole rows at ``0`` / ``q1 - 1``."""
    q1 = _PAR.q1
    hs = rng.integers(0, q1, (m, NU), dtype=np.uint64)
    edge = np.array([0, 1, q1 - 1, q1 - 2, (q1 - 1) // 2, (q1 + 1) // 2, (q1 + 3) // 2, (1 << 24) - 1, 1 << 24,
                     (1 << 24) + 1, *pks.QP, *(p - 1 for p in pks.QP), *(2 * p for p in pks.Q1)], dtype=np.uint64)
    hs[0, :edge.size] = edge
    hs[2] = 0
    hs[3] = q1 - 1
    return hs


def fused_self_test(dev: Any, seed: int = 11) -> None:
    """Raise unless the fused key switch (:func:`pgw.service.pks_ntt.keyswitch`) on ``dev`` equals the torch path
    (:func:`_keyswitch_torch`) bit for bit: 96 rows (3 reply ciphertexts) x 2 groups and 32 rows x 1 group of public
    random data with edge values (hint rows at ``0``, ``q1 - 1`` and the centring boundary, ``y`` at ``0`` and
    ``2^56 - 1``, keys at ``0`` and ``p - 1``)."""
    from pgw.service import pks_ntt

    dev = torch.device(dev)
    rng = np.random.default_rng(seed)
    m, groups = 3 * PACK, 2
    hs = _edge_hint_rows(rng, m)
    y = rng.integers(0, 1 << 56, (m, D * groups), dtype=np.uint64)
    y[0, :6] = [0, 1, (1 << 56) - 1, (1 << 55) - 1, 1 << 55, (1 << 55) + 1]
    y[1] = (1 << 56) - 1
    y[2] = 0
    ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.int64) for p in pks.QP]) for _ in range(groups)])
    for k, p in enumerate(pks.QP):
        ops[0, k, 0] = p - 1
        ops[1, k, 1] = 0
        ops[1, k, 2, :4] = [0, 1, p - 1, p - 2]
    hs_t = torch.from_numpy(hs.view(np.int64)).to(dev)
    y_t = torch.from_numpy(y.view(np.int64)).to(dev)
    keys = _Keys(torch.from_numpy(ops).to(dev))
    got = pks_ntt.keyswitch(hs_t, keys.u32(), y_t)
    want = _keyswitch_torch(hs_t, keys.t, y_t)
    if not torch.equal(got.cpu(), want.cpu()):
        raise RuntimeError("pks-v3 fused NTT key switch does not match the torch path")
    # one group and one reply ciphertext too: grid dimensions of 1 (Triton specializes arguments equal to 1)
    k1 = _Keys(keys.t[1:])
    got = pks_ntt.keyswitch(hs_t[:PACK], k1.u32(), y_t[:PACK, D:])
    if not torch.equal(got.cpu(), _keyswitch_torch(hs_t[:PACK], k1.t, y_t[:PACK, D:]).cpu()):
        raise RuntimeError("pks-v3 fused NTT key switch does not match the torch path (1 group)")
