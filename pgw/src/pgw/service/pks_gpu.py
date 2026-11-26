"""Torch path of the ``pks-v3`` provider key switch (:func:`pgw.hemask.pks_native.keyswitch`, reply form).

Exact int64 arithmetic on the device: every product of two residues below 2^31 is below 2^62, and the
modulus switches use 14-bit digits (as :func:`pgw.hemask.pks.round_mul_shift`). Used by
:class:`pgw.service.pks_provider.PKSProvider` on CUDA kernels only after :func:`self_test` matched the CPU
path bit for bit; it runs on any torch device (the tests run it on the CPU).
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import torch

from pgw.hemask import pks
from pgw.hemask.pks import NR, NU, PACK, RANK, D, PKSParams

__all__ = ["keyswitch", "self_test"]

_PAR = PKSParams()


@cache
def _tables(p: int) -> tuple[np.ndarray, np.ndarray, int]:
    nt = pks.Ntt.of(p, NR)
    return nt.z.astype(np.int64), nt.zi.astype(np.int64), int(nt.ninv)


def _dev_tables(p: int, dev: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
    z, zi, ninv = _tables(p)
    return torch.from_numpy(z).to(dev), torch.from_numpy(zi).to(dev), ninv


def _fwd(x: torch.Tensor, p: int, z: torch.Tensor) -> torch.Tensor:
    lead = x.shape[:-1]
    x = x.reshape(-1, NR).clone()
    length, k = NR // 2, 1
    while length >= 1:
        nb = NR // (2 * length)
        v = x.view(-1, nb, 2, length)
        t = v[:, :, 1, :] * z[k:k + nb, None] % p
        u = v[:, :, 0, :].clone()
        v[:, :, 0, :] = (u + t) % p
        v[:, :, 1, :] = (u - t) % p
        k += nb
        length //= 2
    return x.view(*lead, NR)


def _inv(x: torch.Tensor, p: int, zi: torch.Tensor, ninv: int) -> torch.Tensor:
    lead = x.shape[:-1]
    x = x.reshape(-1, NR).clone()
    length = 1
    while length < NR:
        nb = NR // (2 * length)
        v = x.view(-1, nb, 2, length)
        u, w = v[:, :, 0, :].clone(), v[:, :, 1, :].clone()
        v[:, :, 0, :] = (u + w) % p
        v[:, :, 1, :] = (u - w) % p * zi[nb:2 * nb, None] % p
        length *= 2
    return (x * ninv % p).view(*lead, NR)


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
    q1, pm = _PAR.q1, _PAR.pmod
    m = hs.shape[0]
    nb, groups = m // PACK, ops_t.shape[0]
    hs_t = torch.from_numpy(np.ascontiguousarray(hs).view(np.int64)).to(dev)
    # A~[b, l, 32 u + j] = hs[32 b + j, 256 l + u], centred
    at = hs_t.view(nb, PACK, RANK, D).permute(0, 2, 3, 1).reshape(nb, RANK, NR)
    at = torch.where(at >= (q1 + 1) // 2, at - q1, at)
    y_t = torch.from_numpy(np.ascontiguousarray(y & np.uint64((1 << 56) - 1)).view(np.int64)).to(dev)
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
    out = _round_pow2_div(c0, _PAR.yb, q1) & ((1 << _PAR.yb) - 1)
    return out.cpu().numpy().astype(np.uint64)


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
