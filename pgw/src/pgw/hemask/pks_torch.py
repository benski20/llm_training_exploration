"""Customer-side ``pks-v3`` kernels in exact torch integer arithmetic (prototype; opt-in, not wired by default).

Purpose (``research/reviews/mask-cost-consumer.md``, section 6): run the client mask factory's NTT-heavy parts on the
device's GPU (CUDA here; the same integer algorithm is what a Metal / wgpu compute port would run) instead of CPU
cores. Every function equals its reference in :mod:`pgw.hemask.pks` bit for bit (``tests/hemask/test_pks_torch.py``):

* :func:`pad` - the upload pad ``a_C t mod (Y^4096 + 1, 2^56)`` (:func:`pgw.hemask.pks.negacyclic_mul_mod2`): three
  31-bit-prime NTTs and the centred Garner CRT, int64 only (every product of two residues is below ``2^62``; the CRT
  is done mod ``2^56`` with 28-bit halves, never relying on wrapping);
* :func:`encrypt_body` - the upload bodies ``b`` of one group (:func:`pgw.hemask.pks.encrypt_rlwe`) from the
  customer's secrets ``t`` and noise ``e`` (sampled by the existing samplers: the streams are unchanged);
* :func:`reply_key` - ``beta_l = k_l t' + e_l + P iota(t_l)`` (:func:`pgw.hemask.pks.make_reply_key`);
* :func:`decrypt` - ``W8 r mod 2^32`` from the rounded reply, the compact hint and ``t'``
  (:func:`pgw.hemask.pks.decrypt_pks`).

Secrets are inputs (sampled on the host by the reference samplers, so the request bytes are the same as today's);
the device holds them only for the call. Nothing here changes the scheme, its parameters or its randomness.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import torch

from pgw.hemask import pks
from pgw.hemask.pks import EP, NR, NU, PACK, PS, Q1, QP, RANK, D, PKSParams

__all__ = ["TorchNtt", "decrypt", "decrypt_fused", "encrypt_body", "hint_residues_t", "pad", "pad_ahat",
           "reply_key"]

_PAR = PKSParams()
_M56 = (1 << 56) - 1
_M28 = (1 << 28) - 1


class TorchNtt:
    """Negacyclic NTT of length ``n`` modulo a prime ``p < 2^31`` on ``dev``: the same butterflies and twiddles as
    :class:`pgw.hemask.pks.Ntt` (Cooley-Tukey to bit-reversed order, Gentleman-Sande back), int64 throughout."""

    def __init__(self, p: int, n: int, dev: torch.device):
        if p >= 1 << 31:
            raise ValueError("TorchNtt: primes below 2^31 only (products below 2^62)")
        ref = pks.Ntt.of(p, n)
        self.p, self.n = int(p), int(n)
        self.z = torch.from_numpy(ref.z.astype(np.int64)).to(dev)
        self.zi = torch.from_numpy(ref.zi.astype(np.int64)).to(dev)
        self.ninv = int(ref.ninv)

    def fwd(self, a: torch.Tensor) -> torch.Tensor:
        p, n = self.p, self.n
        lead = a.shape[:-1]
        x = a.reshape(-1, n).clone()
        length, k = n // 2, 1
        while length >= 1:
            nb = n // (2 * length)
            v = x.view(-1, nb, 2, length)
            t = v[:, :, 1, :] * self.z[k:k + nb, None]
            t.remainder_(p)
            u = v[:, :, 0, :]
            d = u - t
            u.add_(t).remainder_(p)
            v[:, :, 1, :] = d.remainder_(p)
            k += nb
            length //= 2
        return x.view(*lead, n)

    def inv(self, a: torch.Tensor) -> torch.Tensor:
        p, n = self.p, self.n
        lead = a.shape[:-1]
        x = a.reshape(-1, n).clone()
        length = 1
        while length < n:
            nb = n // (2 * length)
            v = x.view(-1, nb, 2, length)
            u, w = v[:, :, 0, :], v[:, :, 1, :]
            d = u - w
            u.add_(w).remainder_(p)
            v[:, :, 1, :] = d.remainder_(p).mul_(self.zi[nb:2 * nb, None]).remainder_(p)
            length *= 2
        return x.mul_(self.ninv).remainder_(p).view(*lead, n)


@cache
def _ntt(p: int, n: int, dev: str) -> TorchNtt:
    return TorchNtt(p, n, torch.device(dev))


def _t(x: Any, dev: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dev, torch.int64)
    arr = np.asarray(x)
    if arr.dtype == np.uint64:
        arr = arr.view(np.int64)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64)).to(dev)


# ---- upload --------------------------------------------------------------------------------------------------

def pad_ahat(a_ring: Any, dev: torch.device) -> list[torch.Tensor]:
    """Forward NTTs of the public ``a_C`` (``(n/16, 4096)`` below ``2^56``) per pad prime: public, reusable by every
    group of the linear (a cache trades device memory for one NTT per block and group)."""
    a = _t(a_ring, dev)
    return [_ntt(p, NU, str(dev)).fwd(a % p) for p in EP]


def _crt3_mod56(res: list[torch.Tensor]) -> torch.Tensor:
    """Centred Garner CRT of three residues, reduced mod ``2^56`` (:func:`pgw.hemask.pks._crt3_mod2`), without
    relying on int64 wrap-around: ``(p0 p1 mod 2^56) k2`` is formed from 28-bit halves."""
    p0, p1, p2 = EP
    r0, r1, r2 = res
    inv01, inv012 = pow(p0, -1, p1), pow(p0 * p1, -1, p2)
    k1 = (r1 - r0 % p1).remainder_(p1).mul_(inv01).remainder_(p1)
    base = (r0 % p2 + (p0 % p2) * k1 % p2) % p2
    k2 = (r2 - base).remainder_(p2).mul_(inv012).remainder_(p2)
    c = (p0 * p1) & _M56
    ck2 = (k2 * (c & _M28) + (((k2 * (c >> 28)) & _M28) << 28)) & _M56
    x = (r0 + p0 * k1 + ck2) & _M56
    big = (p0 * p1 * p2) & _M56
    return torch.where(k2 > p2 // 2, (x + (1 << 56) - big) & _M56, x)


def pad(a_ring: Any, t: Any, dev: torch.device, ahat: list[torch.Tensor] | None = None) -> torch.Tensor:
    """``a_C t mod (Y^4096 + 1, 2^56)`` for every block ``C`` (``(n/16, 4096)`` int64), ``t`` ternary ``(4096,)``."""
    tt = _t(t, dev)
    ah = pad_ahat(a_ring, dev) if ahat is None else ahat
    res = []
    for i, p in enumerate(EP):
        nt = _ntt(p, NU, str(dev))
        tf = nt.fwd(tt.remainder(p))
        res.append(nt.inv(ah[i] * tf % p))
    return _crt3_mod56(res)


def encrypt_body(r: Any, a_ring: Any, t: Any, e: Any, dev: torch.device,
                 ahat: list[torch.Tensor] | None = None) -> torch.Tensor:
    """The ``(n, 256)`` upload bodies of one group (:func:`pgw.hemask.pks.encrypt_rlwe` with its ``t`` and ``e``):
    ``b[16 C + k, u] = (a_C t)[16 u + k] + e[16 C + k, u] + 2^24 r[u, 16 C + k] mod 2^56``."""
    rr = _t(r, dev)
    rows, n = rr.shape
    pd = pad(a_ring, t, dev, ahat)  # (n/16, 4096)
    b = pd.view(n // RANK, D, RANK).transpose(1, 2).reshape(n, D)
    msg = torch.zeros((n, D), dtype=torch.int64, device=dev)
    msg[:, :rows] = (rr & 0xFFFFFFFF).t()
    return (b + _t(e, dev) + (msg << (_PAR.q - _PAR.k))) & _M56


# ---- reply key ---------------------------------------------------------------------------------------------

def reply_key(t_components: Any, k_hat: Any, t_reply: Any, e: Any, dev: torch.device) -> torch.Tensor:
    """``beta_hat`` ``(4, 16, 8192)`` (:func:`pgw.hemask.pks.make_reply_key` with its ``t'`` and noise ``e``)."""
    from pgw.service import pks_gpu

    kh = _t(k_hat, dev)
    tr, ee = _t(t_reply, dev), _t(e, dev)
    emb = torch.zeros((RANK, NR), dtype=torch.int64, device=dev)
    emb[:, ::PACK] = _t(t_components, dev)
    out = []
    pm = PS[0] * PS[1]
    for i, p in enumerate(QP):
        z, zi, ninv = pks_gpu._dev_tables(p, dev)
        kt = pks_gpu._inv(kh[i] * pks_gpu._fwd(tr.remainder(p), p, z) % p, p, zi, ninv)
        msg = emb.remainder(p) * (pm % p) % p
        out.append(pks_gpu._fwd((kt + ee.remainder(p) + msg) % p, p, z))
    return torch.stack(out)


# ---- decryption --------------------------------------------------------------------------------------------

def hint_residues_t(hint: Any, dev: torch.device) -> list[torch.Tensor]:
    """The stored compact hint (``(nb, 2, 8192, 3)`` uint8) as NTT-domain residues per ``Q1`` prime on ``dev``
    (:func:`pgw.hemask.pks.hint_residues`; once per linear)."""
    return [_t(h, dev) for h in pks.hint_residues(np.asarray(hint))]


def decrypt(c0r: Any, hr: list[torch.Tensor], t_reply: Any, rows: int, dev: torch.device) -> torch.Tensor:
    """``W8 r mod 2^32`` (``rows x m`` int64) from the rounded reply ``(m/32, 8192)``, the hint residues
    (:func:`hint_residues_t`) and ``t'`` (:func:`pgw.hemask.pks.decrypt_pks`)."""
    from pgw.service import pks_gpu

    q1 = _PAR.q1
    c0 = pks_gpu._round_mul_shift(_t(c0r, dev), q1, _PAR.yb) % q1
    tr = _t(t_reply, dev)
    res = []
    for i, p in enumerate(Q1):
        z, zi, ninv = pks_gpu._dev_tables(p, dev)
        prod = hr[i] * pks_gpu._fwd(tr.remainder(p), p, z) % p
        res.append((c0 % p + pks_gpu._inv(prod, p, zi, ninv)) % p)
    p1, p2 = Q1
    k = (res[1] - res[0] % p2).remainder_(p2).mul_(pow(p1, -1, p2)).remainder_(p2)
    v = res[0] + p1 * k
    msk = pks_gpu._round_pow2_div(v, _PAR.k, q1) & 0xFFFFFFFF
    nb = msk.shape[0]
    return msk.view(nb, D, PACK).transpose(1, 2).reshape(nb * PACK, D)[:, :rows].t()


def decrypt_fused(c0r: Any, hr: list[torch.Tensor], t_reply: Any, rows: int, dev: torch.device) -> torch.Tensor:
    """:func:`decrypt` with the products and inverse NTTs in the provider's fused Triton kernel
    (:func:`pgw.service.pks_ntt.mac_inverse`'s kernel with one product per value: ``inv(c1_hat * t'_hat)``; Shoup /
    Montgomery 32-bit arithmetic, bit-identical to :class:`pgw.hemask.pks.Ntt`); CUDA only. The tail (``c0``'s
    rounding, the CRT and the mask rounding) is :func:`decrypt`'s."""
    from pgw.service import pks_gpu, pks_ntt

    q1 = _PAR.q1
    c0 = pks_gpu._round_mul_shift(_t(c0r, dev), q1, _PAR.yb) % q1
    tr = _t(t_reply, dev)
    nb = int(c0.shape[0])
    res = []
    for i, p in enumerate(Q1):
        z, _zi, _ninv = pks_gpu._dev_tables(p, dev)
        th = pks_gpu._fwd(tr.remainder(p), p, z).to(torch.int32).view(torch.uint32).view(1, 1, NR)
        ah = hr[i].to(torch.int32).view(torch.uint32).view(nb, 1, NR)
        t = pks_ntt.tables(p, dev)
        out = torch.empty((1, nb, NR), dtype=torch.uint32, device=dev)
        pks_ntt._mac_inv_kernel[(1, nb)](ah, th, out, t.zi, t.zis, p, t.pinv, t.scale, t.scales, N=NR,
                                          LOGN=pks_ntt.LOG_NR, RK=1, num_warps=pks_ntt._warps())
        res.append((c0 % p + out[0].view(torch.int32).to(torch.int64)) % p)
    p1, p2 = Q1
    k = (res[1] - res[0] % p2).remainder_(p2).mul_(pow(p1, -1, p2)).remainder_(p2)
    v = res[0] + p1 * k
    msk = pks_gpu._round_pow2_div(v, _PAR.k, q1) & 0xFFFFFFFF
    return msk.view(nb, D, PACK).transpose(1, 2).reshape(nb * PACK, D)[:, :rows].t()
