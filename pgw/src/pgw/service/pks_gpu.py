"""Torch path of the ``pks-v3`` provider key switch (:func:`pgw.hemask.pks_native.keyswitch`, reply form).

Exact int64 arithmetic on the device: every product of two residues below 2^31 is below 2^62, and the
modulus switches use 14-bit digits (as :func:`pgw.hemask.pks.round_mul_shift`). Used by
:class:`pgw.service.pks_provider.PKSProvider` on CUDA kernels only after :func:`self_test` matched the CPU
path bit for bit; it runs on any torch device (the tests run it on the CPU).

:func:`item` computes a whole item on the device: the provider-side hint rows (``W8 A``, switched to ``q1``), the
item's ``y = W8 b`` and the key switch, row block by row block, so the ``m x 4096`` hint is never cached or copied
(used by ``PKSProvider`` when the model's hints do not fit its cache; self-tested by :func:`item_self_test`).
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import torch

from pgw.hemask import pks
from pgw.hemask.pks import NR, NU, PACK, RANK, D, PKSParams

__all__ = ["item", "item_bytes", "item_rows", "item_self_test", "keyswitch", "ring_rows", "self_test"]

_PAR = PKSParams()


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


def _tw_t(w: np.ndarray | torch.Tensor, length: int) -> torch.Tensor:
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
    ops_t = torch.from_numpy(np.ascontiguousarray(ops).astype(np.int64)).to(dev)
    if step >= nb:
        return _keyswitch_rows(hs, ops_t, y, dev)
    out = np.empty((groups, nb, NR), dtype=np.uint64)
    for b0 in range(0, nb, step):
        b1 = min(nb, b0 + step)
        r0, r1 = b0 * PACK, b1 * PACK
        out[:, b0:b1] = _keyswitch_rows(hs[r0:r1], ops_t, y[r0:r1], dev)
    return out


def _keyswitch_rows(hs: np.ndarray, ops_t: torch.Tensor, y: np.ndarray, dev: Any) -> np.ndarray:
    hs_t = torch.from_numpy(np.ascontiguousarray(hs).view(np.int64)).to(dev)
    y_t = torch.from_numpy(np.ascontiguousarray(y & np.uint64((1 << 56) - 1)).view(np.int64)).to(dev)
    return _keyswitch_t(hs_t, ops_t, y_t).cpu().numpy().astype(np.uint64)


def _keyswitch_t(hs_t: torch.Tensor, ops_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """The key switch of whole reply ciphertexts on ``hs_t``'s device: ``hs_t`` ``(R, 4096)`` int64 in ``[0, q1)``
    (switched hint rows), ``y_t`` ``(R, 256 G)`` int64 below ``2^56``; returns ``(G, R/32, 8192)`` int64."""
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
    (:func:`pgw.hemask.pks.public_ring_rows`: the same public stream the host path expands)."""
    ring = pks.public_ring_rows(bytes(a_seed), int(n), linear)
    return torch.from_numpy(np.ascontiguousarray(ring).view(np.int64)).to(dev)


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
    return held + max(build, rows * _row_bytes(n_pad, cols, groups))


def item(w: torch.Tensor, const: torch.Tensor, m: int, n: int, ring_t: torch.Tensor, b: np.ndarray, ops: np.ndarray,
         *, work_bytes: int | None = None) -> np.ndarray:
    """The reply ``(G, m/32, 8192)`` uint64 of one pks-v3 item, computed on ``w``'s device.

    ``w``: the kernel's int8 weight (``(>= m, n_pad)``, rows and inputs beyond ``(m, n)`` zero), ``const``: its
    ``(>= m, 1)`` maskgen constant for 7 byte planes (:meth:`ComputeKernel._maskgen_const`), ``ring_t``:
    :func:`ring_rows`, ``b``: the item's ``(n, 256 G)`` upload below ``2^56``, ``ops``: ``(G, 4, 16, 8192)`` reply
    keys. Equal bit for bit to ``keyswitch(switch_hint(W8 A mod 2^56), ops, W8 b mod 2^56)`` (the host path of
    :class:`pgw.service.pks_provider.PKSProvider`): the same GEMM arithmetic (:func:`gemm_mod56`), the same
    exact 14-bit-digit switch (:func:`_round_mul_shift`) and the same key switch (:func:`_keyswitch_t`, whose
    reply ciphertexts depend only on their own 32 rows)."""
    dev = w.device
    np_ = int(w.shape[1])
    groups, cols = int(ops.shape[0]), int(b.shape[1])
    if m % PACK or n % RANK or b.shape[0] != n or cols != D * groups or w.shape[0] < m or np_ < n:
        raise ValueError("pks-v3 device item: bad shapes")
    q1 = _PAR.q1
    ops_t = torch.from_numpy(np.ascontiguousarray(ops).astype(np.int64)).to(dev)
    src = _a_source(ring_t)
    step = COL_BLOCK // D
    aplanes = [planes_of(a_columns(src, l0, l0 + step), np_) for l0 in range(0, RANK, step)]
    del src
    bplanes = []
    for c0 in range(0, cols, COL_BLOCK):
        bt = torch.from_numpy(np.ascontiguousarray(b[:, c0:c0 + COL_BLOCK]).view(np.int64)).to(dev)
        bplanes.append(planes_of(bt, np_))
        del bt
    rows = item_rows(m, np_, cols, groups, work_bytes)
    out = np.empty((groups, m // PACK, NR), dtype=np.uint64)
    for r0 in range(0, m, rows):
        r1 = min(m, r0 + rows)
        wr, cr = w[r0:r1], const[r0:r1]
        hs = torch.empty((r1 - r0, NU), dtype=torch.int64, device=dev)
        for j, pl in enumerate(aplanes):
            h = gemm_mod56(wr, pl, cr)
            hs[:, j * COL_BLOCK:(j + 1) * COL_BLOCK] = _round_mul_shift(h, q1, 56) % q1
            del h
        y = torch.empty((r1 - r0, cols), dtype=torch.int64, device=dev)
        for j, pl in enumerate(bplanes):
            c0 = j * COL_BLOCK
            y[:, c0:c0 + pl.shape[1]] = gemm_mod56(wr, pl, cr)
        res = _keyswitch_t(hs, ops_t, y)
        del hs, y
        out[:, r0 // PACK:r1 // PACK] = res.cpu().numpy().view(np.uint64)
        del res
    return out


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
