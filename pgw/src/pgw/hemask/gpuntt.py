"""Gateway HE on the GPU with NTTs: encryption with ``A`` in NTT form and decryption's hint product.

The customer side of the encrypted mask factory computes two ring products by the small ternary
secret ``S`` (see :mod:`pgw.hemask.ntt`): the encryption pad ``A[:n] @ S`` and the decryption
correction ``H @ S`` (``H = W8 A`` the hint). Each row is a sum of ``rank`` negacyclic products in
``Z[X]/(X^256 + 1)``. The CPU path (``he_encrypt="ntt_pre"``, native crate ``pgw_ntt``) computes them
with NTTs over three ~2^29 primes and CRT; the dense device path (:mod:`pgw.hemask.devmm`) multiplies
by ``S`` as a ``dim x 256`` matrix (``7.3e6`` int8 MACs per row and group). This module runs the NTT
algorithm on the gateway GPU:

* the public matrix ``A`` is forward-transformed **once** and stays on the device as int32 residues
  ``(3, rank, 256, rows)`` (12 bytes per coefficient, like ``pgw_ntt``'s ``TransformedOperand``);
* the hint is kept as byte planes (in device memory while it fits, else streamed, as in
  :mod:`pgw.hemask.devmm`) and forward-transformed per decryption pass, block by block;
* per group, the secret polynomials are transformed (tiny), then one fused kernel per block does the
  pointwise multiply-accumulate over the ``rank`` polynomials, the inverse NTT and the CRT to
  ``mod 2^64`` for all three primes (Triton on CUDA); the encryption/decryption epilogues (noise,
  message, ``mod 2^q``; subtract, round, transpose) run on the device as in :mod:`pgw.hemask.devmm`.

Exactness: every residue operation is exact int64 arithmetic (operands below ``2^29``, products below
``2^58``, sums of up to 16 products below ``2^62``). An output coefficient is ``sum +- a s`` with
``|a| < 2^b`` and ``sum |s| = L1(S)``; CRT with a centred last digit recovers every
``|Z| <= HALF = p0 p1 (p2 - 1) / 2 ~ 2^86``, checked per pass from the secrets' L1 norms (at rank 16
the bound is ``2^68`` for ``q = 2^56``). The result is ``Z mod 2^64``, i.e. the dense product **bit for
bit**, hence the same ciphertexts and decrypted masks as ``ntt_pre`` and the dense paths.
:func:`self_test` compares the device engine with the float64 reference (encryption and decryption,
real ``dim``) before first use; on any mismatch or error it is not used (:func:`gpu_ntt_for` returns
``None`` and the backend falls back to the CPU ``ntt_pre`` path or the dense path).

Privacy: customer-side work only. The provider receives the same ciphertexts (same distribution,
sizes, counts and schedule). Secrets and noise are sampled on the CPU by the same code
(:func:`pgw.hemask.lwe.sample_secret`, :func:`pgw.hemask.lwe.sample_noise`) from a fresh OS seed per
group; device temporaries holding secret-dependent values (transformed secrets, ``A s``, ``H S``,
noise, masks) are zeroed after use.

Environment: ``PGW_HEMASK_GPU_NTT_KERNEL=eager`` forces the torch (unfused) implementation on CUDA
(the kernel is otherwise Triton when importable); ``PGW_HEMASK_GPU_HINT_MB`` caps the hint cache as in
:mod:`pgw.hemask.devmm`.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError
from pgw.core.prng import new_seed
from pgw.hemask.devmm import DeviceHE, _auto_budget, _reference_encrypt
from pgw.hemask.lwe import DEGREE, HEParams, MaskCiphertext, matmul_mod64, public_matrix, sample_noise, sample_secret

__all__ = ["PRIMES", "GpuNTTHE", "gpu_ntt_for", "self_test", "status"]

log = logging.getLogger(__name__)

#: NTT primes (``p = 1 mod 512``, below ``2^29``): the same as the native ``pgw_ntt`` crate.
PRIMES = (536870401, 536865281, 536856577)
NP = len(PRIMES)
_P01 = PRIMES[0] * PRIMES[1]
#: Largest ``|Z|`` the centred CRT recovers.
HALF = _P01 * (PRIMES[2] - 1) // 2
_INV01 = pow(PRIMES[0] % PRIMES[1], -1, PRIMES[1])
_INV012 = pow(_P01 % PRIMES[2], -1, PRIMES[2])
_P01_S = _P01 - (1 << 64) if _P01 >= 1 << 63 else _P01  # fits int64 (< 2^58)
#: Temporaries per block (bytes): bounds device memory of a pass.
_BLOCK_BYTES = 192 << 20
_BR = 32  # columns per Triton program


def _brv8(k: int) -> int:
    return int(f"{k:08b}"[::-1], 2)


def _psi(p: int) -> int:
    """A primitive 512th root of unity mod ``p``."""
    for x in range(2, p):
        c = pow(x, (p - 1) // 512, p)
        if pow(c, 256, p) == p - 1:
            return c
    raise ValueError(p)  # pragma: no cover - the primes are NTT friendly


def _zetas() -> np.ndarray:
    """``(3, 256)`` int64: ``zetas[k] = psi^brv8(k)`` (the Dilithium-style negacyclic NTT table)."""
    out = np.empty((NP, DEGREE), dtype=np.int64)
    for i, p in enumerate(PRIMES):
        psi = _psi(p)
        out[i] = [pow(psi, _brv8(k), p) for k in range(DEGREE)]
    return out


_ZETAS = _zetas()
_NINV = [pow(DEGREE, -1, p) for p in PRIMES]


# ---------------------------------------------------------------------------------------------
# Eager torch implementation (CPU tests, CUDA without Triton, and the one-time transform of A)
# ---------------------------------------------------------------------------------------------


class _Tables:
    """Per-device constants."""

    def __init__(self, dev: torch.device):
        self.dev = dev
        self.p = torch.tensor(PRIMES, dtype=torch.int64, device=dev)
        self.zetas = torch.from_numpy(_ZETAS).to(dev)
        self.fwd = []  # stage s: (3, B, 1, 1), B = 2^s
        self.inv = []  # stage s: (3, B, 1, 1), B = 128 >> s
        for s in range(8):
            b = 1 << s
            self.fwd.append(self.zetas[:, b:2 * b].reshape(NP, b, 1, 1).contiguous())
            b = 128 >> s
            idx = torch.arange(2 * b - 1, b - 1, -1, device=dev)
            self.inv.append(self.zetas[:, idx].reshape(NP, b, 1, 1).contiguous())
        self.ninv = torch.tensor(_NINV, dtype=torch.int64, device=dev)


_TABLES: dict[str, _Tables] = {}


def _tables(dev: torch.device) -> _Tables:
    t = _TABLES.get(str(dev))
    if t is None:
        t = _TABLES[str(dev)] = _Tables(dev)
    return t


def _copy(t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """A fresh contiguous copy (never a view of ``t``, which the callers wipe)."""
    out = torch.empty(t.shape, dtype=dtype or t.dtype, device=t.device)
    out.copy_(t)
    return out


def fwd_ntt(x: torch.Tensor, tb: _Tables) -> torch.Tensor:
    """Negacyclic forward NTT along dim 1 of ``x`` ``(3, 256, R)`` int64 residues (natural order in,
    bit-reversed out), in place; returns ``x``."""
    p4 = tb.p.view(NP, 1, 1, 1)
    r = x.shape[2]
    for s in range(8):
        b, ln = 1 << s, 128 >> s
        v = x.view(NP, b, 2, ln, r)
        lo, hi = v[:, :, 0], v[:, :, 1]
        t = hi * tb.fwd[s]
        t.remainder_(p4)
        torch.sub(lo, t, out=hi)
        hi.add_(p4)
        lo.add_(t)
        del t
        hi.remainder_(p4)
        lo.remainder_(p4)
    return x


def inv_ntt(x: torch.Tensor, tb: _Tables) -> torch.Tensor:
    """Inverse of :func:`fwd_ntt` without the ``1/256`` factor (bit-reversed in, natural out), in place."""
    p4 = tb.p.view(NP, 1, 1, 1)
    r = x.shape[2]
    for s in range(8):
        ln, b = 1 << s, 128 >> s
        v = x.view(NP, b, 2, ln, r)
        lo, hi = v[:, :, 0], v[:, :, 1]
        d = hi - lo
        d.add_(p4)
        d.mul_(tb.inv[s])
        d.remainder_(p4)
        lo.add_(hi)
        lo.remainder_(p4)
        hi.copy_(d)
        del d
    return x


def crt64(x: torch.Tensor) -> torch.Tensor:
    """``(3, ...)`` residues -> the centred CRT value mod ``2^64`` as int64 (wrapping)."""
    p0, p1, p2 = PRIMES
    v0 = x[0]
    v1 = (x[1] - v0.remainder(p1) + p1).mul_(_INV01).remainder_(p1)
    u = v1 * p0
    u += v0
    v2 = (x[2] - u.remainder(p2) + p2).mul_(_INV012).remainder_(p2)
    v2 -= (v2 > (p2 - 1) // 2).to(torch.int64) * p2
    v2 *= _P01_S  # wraps mod 2^64
    u += v2
    return u


def transform_rows(vals: torch.Tensor, rank: int, tb: _Tables) -> torch.Tensor:
    """``(rows, rank*256)`` int64 values (any integers, congruent mod 2^64) -> ``(3, rank, 256, rows)``
    int32 NTT residues (the layout of a transformed operand)."""
    rows = vals.shape[0]
    src = vals.view(rows, rank, DEGREE).permute(2, 0, 1).reshape(1, DEGREE, rows * rank)
    x = torch.empty((NP, DEGREE, rows * rank), dtype=torch.int64, device=vals.device)
    torch.remainder(src, tb.p.view(NP, 1, 1), out=x)
    fwd_ntt(x, tb)
    out = _copy(x.view(NP, DEGREE, rows, rank).permute(0, 3, 1, 2), torch.int32)
    x.zero_()
    return out


def transform_secrets(polys: torch.Tensor, tb: _Tables) -> torch.Tensor:
    """``(G, rank, 256)`` ternary secret polynomials -> ``(3, rank, 256, G)`` int64 NTT residues
    pre-multiplied by ``1/256`` (so the inverse transform needs no scaling)."""
    g, rank, _ = polys.shape
    src = polys.to(torch.int64).permute(2, 1, 0).reshape(1, DEGREE, rank * g)
    x = torch.empty((NP, DEGREE, rank * g), dtype=torch.int64, device=polys.device)
    torch.remainder(src, tb.p.view(NP, 1, 1), out=x)
    src.zero_()
    fwd_ntt(x, tb)
    x.mul_(tb.ninv.view(NP, 1, 1)).remainder_(tb.p.view(NP, 1, 1))
    out = _copy(x.view(NP, DEGREE, rank, g).permute(0, 2, 1, 3))
    x.zero_()
    return out


def mac_inv_crt_eager(t_op: torch.Tensor, c0: int, nb: int, sh: torch.Tensor, tb: _Tables) -> torch.Tensor:
    """``(nb, G*256)`` int64: rows ``c0 .. c0+nb`` of the transformed operand ``t_op`` ``(3, rank, 256, rows)``
    times the transformed secrets ``sh`` ``(3, rank, 256, G)``, mod ``2^64``."""
    rank, g = sh.shape[1], sh.shape[3]
    acc = torch.zeros((NP, DEGREE, nb, g), dtype=torch.int64, device=sh.device)
    for li in range(rank):
        acc.addcmul_(t_op[:, li, :, c0:c0 + nb, None].to(torch.int64), sh[:, li, :, None, :])
    x = acc.view(NP, DEGREE, nb * g)
    x.remainder_(tb.p.view(NP, 1, 1))
    inv_ntt(x, tb)
    z = crt64(x)  # (256, nb*g)
    x.zero_()
    out = _copy(z.view(DEGREE, nb, g).permute(1, 2, 0)).view(nb, g * DEGREE)
    z.zero_()
    return out


def fwd_planes_eager(h: torch.Tensor, first_byte: int, rank: int, tb: _Tables) -> torch.Tensor:
    """Hint rows as byte planes ``(rows, P, dim)`` int8 (:func:`pgw.hemask.fastmm.to_planes`: byte - 128, bytes
    ``first_byte ..``) -> transformed operand."""
    rows, npl, dim = h.shape
    v = torch.zeros((rows, dim), dtype=torch.int64, device=h.device)
    for i in range(npl):
        v += (h[:, i].to(torch.int64) + 128) << (8 * (first_byte + i))  # planes hold byte - 128
    out = transform_rows(v, rank, tb)
    v.zero_()
    return out


# ---------------------------------------------------------------------------------------------
# Triton kernels (CUDA): fused MAC + inverse NTT + CRT, and the forward transform of hint planes
# ---------------------------------------------------------------------------------------------

_triton: Any = None
_TRITON_ERR = ""
try:  # pragma: no cover - depends on the environment
    import triton
    import triton.language as tl

    @triton.jit
    def _mac_inv_crt_kernel(t_ptr, s_ptr, x_ptr, z_ptr, zt_ptr, t_rows, c0, ncols, g, rank,
                            P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                            INV01: tl.constexpr, INV012: tl.constexpr, P01: tl.constexpr,
                            BR: tl.constexpr):
        pid = tl.program_id(0)
        r = pid * BR + tl.arange(0, BR)
        rm = r < ncols
        rr = tl.where(rm, r, 0)
        c = rr // g
        gi = rr % g
        jj = tl.arange(0, 128)
        m2 = (jj[:, None] < 128) & rm[None, :]
        for pr in tl.static_range(3):
            if pr == 0:
                p = P0
            elif pr == 1:
                p = P1
            else:
                p = P2
            for h in tl.static_range(2):
                j = h * 128 + jj
                acc = tl.zeros((128, BR), dtype=tl.int64)
                for li in range(rank):
                    base = (pr * rank + li) * 256 + j[:, None]
                    tv = tl.load(t_ptr + base.to(tl.int64) * t_rows + c0 + c[None, :], mask=m2, other=0)
                    sv = tl.load(s_ptr + base * g + gi[None, :], mask=m2, other=0)
                    acc += tv.to(tl.int64) * sv
                acc = acc % p
                tl.store(x_ptr + ((pr * 256 + j[:, None]).to(tl.int64) * ncols + r[None, :]), acc.to(tl.int32),
                         mask=m2)
            tl.debug_barrier()
            for st in range(8):
                ln = 1 << st
                b = jj // ln
                jlo = b * 2 * ln + jj % ln
                z = tl.load(zt_ptr + pr * 256 + (2 * (128 >> st) - 1 - b)).to(tl.int64)
                plo = x_ptr + ((pr * 256 + jlo[:, None]).to(tl.int64) * ncols + r[None, :])
                phi = plo + ln * ncols
                lo = tl.load(plo, mask=m2, other=0).to(tl.int64)
                hi = tl.load(phi, mask=m2, other=0).to(tl.int64)
                tl.debug_barrier()
                sm = lo + hi
                sm = tl.where(sm >= p, sm - p, sm)
                d = ((hi - lo + p) * z[:, None]) % p
                tl.store(plo, sm.to(tl.int32), mask=m2)
                tl.store(phi, d.to(tl.int32), mask=m2)
                tl.debug_barrier()
        for h in tl.static_range(2):
            j = h * 128 + jj
            off = j[:, None].to(tl.int64) * ncols + r[None, :]
            x0 = tl.load(x_ptr + off, mask=m2, other=0).to(tl.int64)
            x1 = tl.load(x_ptr + 256 * ncols + off, mask=m2, other=0).to(tl.int64)
            x2 = tl.load(x_ptr + 512 * ncols + off, mask=m2, other=0).to(tl.int64)
            v1 = ((x1 - x0 % P1 + P1) * INV01) % P1
            u = x0 + v1 * P0
            v2 = ((x2 - u % P2 + P2) * INV012) % P2
            v2 = tl.where(v2 > (P2 - 1) // 2, v2 - P2, v2)
            zv = u + v2 * P01
            tl.store(z_ptr + (c[None, :].to(tl.int64) * (g * 256) + gi[None, :] * 256 + j[:, None]), zv, mask=m2)

    @triton.jit
    def _fwd_planes_kernel(h_ptr, t_ptr, zt_ptr, rows, npl, dim, FIRST: tl.constexpr, NPL: tl.constexpr,
                           P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, BR: tl.constexpr):
        pid = tl.program_id(0)
        li = tl.program_id(1)
        rank = tl.num_programs(1)
        c = pid * BR + tl.arange(0, BR)
        cm = c < rows
        jj = tl.arange(0, 128)
        m2 = (jj[:, None] < 128) & cm[None, :]
        for pr in tl.static_range(3):
            if pr == 0:
                p = P0
            elif pr == 1:
                p = P1
            else:
                p = P2
            for h in tl.static_range(2):
                j = h * 128 + jj
                v = tl.zeros((128, BR), dtype=tl.int64)
                src = h_ptr + c[None, :].to(tl.int64) * (npl * dim) + li * 256 + j[:, None]
                for i in tl.static_range(NPL):
                    byte = tl.load(src + i * dim, mask=m2, other=0).to(tl.int64) + 128  # byte - 128
                    v += byte << (8 * (FIRST + i))
                v = v % p
                v = tl.where(v < 0, v + p, v)
                tl.store(t_ptr + (((pr * rank + li) * 256 + j[:, None]).to(tl.int64) * rows + c[None, :]),
                         v.to(tl.int32), mask=m2)
            tl.debug_barrier()
            for st in range(8):
                ln = 128 >> st
                b = jj // ln
                jlo = b * 2 * ln + jj % ln
                z = tl.load(zt_ptr + pr * 256 + (1 << st) + b).to(tl.int64)
                plo = t_ptr + (((pr * rank + li) * 256 + jlo[:, None]).to(tl.int64) * rows + c[None, :])
                phi = plo + ln * rows
                lo = tl.load(plo, mask=m2, other=0).to(tl.int64)
                hi = tl.load(phi, mask=m2, other=0).to(tl.int64)
                tl.debug_barrier()
                t = (hi * z[:, None]) % p
                a = lo + t
                a = tl.where(a >= p, a - p, a)
                d = lo - t + p
                d = tl.where(d >= p, d - p, d)
                tl.store(plo, a.to(tl.int32), mask=m2)
                tl.store(phi, d.to(tl.int32), mask=m2)
                tl.debug_barrier()

    _triton = triton
except Exception as exc:  # pragma: no cover - no Triton
    _TRITON_ERR = f"{type(exc).__name__}: {exc}"


def triton_available() -> bool:
    return _triton is not None


def mac_inv_crt_triton(t_op: torch.Tensor, c0: int, nb: int, sh: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """Fused :func:`mac_inv_crt_eager` (Triton)."""
    rank, g = sh.shape[1], sh.shape[3]
    ncols = nb * g
    x = torch.empty((NP, DEGREE, ncols), dtype=torch.int32, device=sh.device)
    z = torch.empty((nb, g * DEGREE), dtype=torch.int64, device=sh.device)
    grid = (-(-ncols // _BR),)
    _mac_inv_crt_kernel[grid](t_op, sh, x, z, zt, t_op.shape[3], c0, ncols, g, rank,
                              P0=PRIMES[0], P1=PRIMES[1], P2=PRIMES[2], INV01=_INV01, INV012=_INV012,
                              P01=_P01, BR=_BR, num_warps=8)
    x.zero_()
    return z


def fwd_planes_triton(h: torch.Tensor, first_byte: int, rank: int, zt: torch.Tensor) -> torch.Tensor:
    """Fused :func:`fwd_planes_eager` (Triton)."""
    rows, npl, dim = h.shape
    out = torch.empty((NP, rank, DEGREE, rows), dtype=torch.int32, device=h.device)
    grid = (-(-rows // _BR), rank)
    _fwd_planes_kernel[grid](h, out, zt, rows, npl, dim, FIRST=first_byte, NPL=npl,
                             P0=PRIMES[0], P1=PRIMES[1], P2=PRIMES[2], BR=_BR, num_warps=8)
    return out


# ---------------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------------


def _l1_ok(polys: np.ndarray, big_bits: int) -> bool:
    """``(2^big_bits) * max_g L1(s_g) <= HALF`` (the CRT recovers every coefficient exactly)."""
    l1 = int(np.abs(polys.reshape(polys.shape[0], -1).astype(np.int64)).sum(1).max())
    return (1 << big_bits) * l1 <= HALF


def _secret_polys(smats: list[np.ndarray], rank: int) -> np.ndarray:
    """``(G, rank, 256)`` int8 secret polynomials of block-negacyclic matrices (row ``256 l`` of block ``l``
    is ``s_l``, see :func:`pgw.hemask.lwe.secret_matrix`)."""
    out = np.empty((len(smats), rank, DEGREE), dtype=np.int8)
    for gi, sm in enumerate(smats):
        out[gi] = sm[::DEGREE][:rank]
    return out


class GpuNTTHE(DeviceHE):
    """Encryption (:meth:`encrypt`, the :class:`pgw.hemask.mlwe_ring.PretransformedEncryptor` contract) and
    decryption (:meth:`decrypt_into`, the :class:`pgw.hemask.devmm.DeviceHE` contract) with NTTs on ``device``.

    Args:
        a: Public matrix ``(n_max, dim)`` uint64 (:func:`pgw.hemask.lwe.public_matrix`); transformed once on
            the device (``12 n_max dim`` bytes).
        params: Encryption parameters.
        device: Torch device (``cuda``; tests run the same code on ``cpu``).
        kernel: ``"triton"`` (fused kernels, CUDA only), ``"eager"`` (torch ops) or ``None`` (Triton when
            available on CUDA, :envvar:`PGW_HEMASK_GPU_NTT_KERNEL` overrides).
        hint_budget: Bytes of hint planes kept on the device (default: as :mod:`pgw.hemask.devmm`, minus a
            reserve for the pass temporaries).
    """

    def __init__(self, a: np.ndarray, params: HEParams, device: torch.device | str, *,
                 kernel: str | None = None, hint_budget: int | None = None, groups_per_pass: int = 4):
        dev = torch.device(device)
        if a.ndim != 2 or a.shape[1] != params.dim:
            raise ConfigError(f"public matrix shape {a.shape} does not match dimension {params.dim}")
        want = kernel or os.environ.get("PGW_HEMASK_GPU_NTT_KERNEL", "").strip().lower() or "auto"
        if want == "auto":
            want = "triton" if dev.type == "cuda" and _triton is not None else "eager"
        if want == "triton" and (dev.type != "cuda" or _triton is None):
            raise ConfigError(f"Triton NTT kernels need CUDA and triton ({_TRITON_ERR or dev})")
        if want not in ("triton", "eager"):
            raise ConfigError(f"unknown GPU NTT kernel {want!r}")
        self.kernel = want
        self.tb = _tables(dev)
        self.n_max = int(a.shape[0])
        self.groups_per_pass = max(1, min(16, int(groups_per_pass)))
        # A mod 2^q (the ciphertext is reduced mod 2^q): non-negative below 2^q for q < 64, else the int64 with
        # the same bits (|a| <= 2^63, congruent mod 2^64).
        self._a_bits = params.q if params.q < 64 else 63
        rows = max(1, (64 << 20) // (8 * params.dim * NP * 3))
        t_a = torch.empty((NP, params.rank, DEGREE, self.n_max), dtype=torch.int32, device=dev)
        for r0 in range(0, self.n_max, rows):
            blk = np.ascontiguousarray(a[r0:r0 + rows])
            if params.q < 64:
                blk = blk & params.q_mask
            v = torch.from_numpy(blk.view(np.int64)).to(dev)
            t_a[:, :, :, r0:r0 + v.shape[0]] = transform_rows(v, params.rank, self.tb)
        self.t_a = t_a
        if hint_budget is None:
            hint_budget = max(0, _auto_budget(dev) - (1 << 30)) if dev.type == "cuda" else 1 << 30
        super().__init__(np.empty((0, 1, params.dim), dtype=np.int8), params, dev, hint_budget=hint_budget)
        self.stats.update({"encrypt_groups": 0, "decrypt_groups": 0})

    @property
    def nbytes(self) -> int:
        """Device memory of the transformed public matrix."""
        return int(self.t_a.numel() * 4)

    # ---- kernels ----------------------------------------------------------------------------
    def _mac(self, t_op: torch.Tensor, c0: int, nb: int, sh: torch.Tensor) -> torch.Tensor:
        if self.kernel == "triton":
            return mac_inv_crt_triton(t_op, c0, nb, sh, self.tb.zetas)
        return mac_inv_crt_eager(t_op, c0, nb, sh, self.tb)

    def _fwd_planes(self, h: torch.Tensor, first_byte: int) -> torch.Tensor:
        if self.kernel == "triton":
            return fwd_planes_triton(h.contiguous(), first_byte, self.params.rank, self.tb.zetas)
        return fwd_planes_eager(h, first_byte, self.params.rank, self.tb)

    def _block_rows(self, cols: int) -> int:
        # per row: the int32 scratch (3 x 256 per column) and int64 output and epilogue temporaries
        per_row = cols * (NP * 4 + 4 * 8) + (NP * DEGREE * 8 * cols if self.kernel == "eager" else 0)
        return max(8, _BLOCK_BYTES // max(1, per_row))

    def _secrets_on_device(self, smats: list[np.ndarray], big_bits: int) -> torch.Tensor:
        polys = _secret_polys(smats, self.params.rank)
        try:
            if not _l1_ok(polys, big_bits):  # cannot happen for ternary secrets at rank <= 2^13
                raise ConfigError("secret too large for an exact NTT product")
            pd = self._to_dev(polys)
            try:
                return transform_secrets(pd, self.tb)
            finally:
                pd.zero_()
        finally:
            polys.fill(0)

    # ---- encryption ----------------------------------------------------------------------------
    def encrypt(self, r: np.ndarray, *, _seeds: list[bytes] | None = None) -> list[MaskCiphertext]:  # type: ignore[override]
        """Encrypt ``(rows, n)`` masks as ``ceil(rows / 256)`` groups; ciphertexts equal those of
        :func:`pgw.hemask.lwe.encrypt_masks` for the same per-group seeds (``_seeds``: test hook only)."""
        r = np.asarray(r)
        rows, n = r.shape
        if n > self.n_max:
            raise ConfigError(f"linear has {n} inputs but the public matrix has {self.n_max} rows")
        groups = -(-rows // DEGREE)
        if _seeds is not None and len(_seeds) != groups:
            raise ConfigError(f"need {groups} seeds, got {len(_seeds)}")
        out: list[MaskCiphertext] = []
        gp = self.groups_per_pass
        for p0 in range(0, groups, gp):
            p1 = min(groups, p0 + gp)
            seeds = list(_seeds[p0:p1]) if _seeds is not None else [new_seed() for _ in range(p0, p1)]
            try:
                out.extend(self._encrypt_pass(r[p0 * DEGREE:min(rows, p1 * DEGREE)], n, seeds))
            finally:
                for i in range(len(seeds)):
                    seeds[i] = b""
        return out

    def _encrypt_pass(self, r: np.ndarray, n: int, seeds: list[bytes]) -> list[MaskCiphertext]:
        """One pass: the pads ``A s`` of all groups are launched on the device first (asynchronous on CUDA),
        then the noise is sampled on the CPU while the device works, then the epilogue runs per block."""
        p = self.params
        g = len(seeds)
        rows = int(r.shape[0])
        smats = [sample_secret(s, p.rank) for s in seeds]
        sh = ed = rd = None
        pads: list[torch.Tensor] = []
        try:
            sh = self._secrets_on_device(smats, self._a_bits)
            block = self._block_rows(g * DEGREE)
            starts = list(range(0, n, block))
            for c0 in starts:
                pads.append(self._mac(self.t_a, c0, min(block, n - c0), sh))  # (nb, g*256) = A s mod 2^64
            sh.zero_()
            host_e = np.empty((n, DEGREE * g), dtype=np.int8)
            try:
                for gi, seed in enumerate(seeds):  # overlaps the device work launched above
                    e = sample_noise(seed, n, p.eta)
                    host_e[:, gi * DEGREE:(gi + 1) * DEGREE] = e  # |e| <= eta: exact in int8
                    e.fill(0)
                ed = self._to_dev(host_e)
                rd = self._to_dev(np.ascontiguousarray(r, dtype=np.int64))
            finally:
                host_e.fill(0)
            b = np.empty((n, DEGREE * g), dtype=np.uint64)
            qmask = (1 << p.q) - 1 if p.q < 64 else None
            for c0, pad in zip(starts, pads, strict=True):
                nb = pad.shape[0]
                pad += ed[c0:c0 + nb]
                msg = torch.zeros((nb, DEGREE * g), dtype=torch.int64, device=self.dev)
                msg[:, :rows] = rd[:, c0:c0 + nb].t()
                msg <<= p.delta_bits
                pad += msg
                msg.zero_()
                if qmask is not None:
                    pad &= qmask
                host = pad.to("cpu", copy=True)
                pad.zero_()
                b[c0:c0 + nb] = host.numpy().view(np.uint64)
                host.zero_()
            self.stats["encrypt_groups"] += g
        except BaseException:
            for sm in smats:
                sm.fill(0)
            raise
        finally:
            for t in (sh, ed, rd, *pads):
                if t is not None:
                    t.zero_()
        b3 = b.reshape(n, g, DEGREE)
        return [MaskCiphertext(b=b3[:, gi, :], secret=smats[gi], rows=min(DEGREE, rows - gi * DEGREE))
                for gi in range(g)]

    # ---- decryption ----------------------------------------------------------------------------
    def decrypt_into(self, key: str, hint: np.ndarray, items: list[tuple[np.ndarray, list[MaskCiphertext], int]],
                     out: np.ndarray, delta_bits: int, k: int) -> None:
        """Same contract and bits as :meth:`pgw.hemask.devmm.DeviceHE.decrypt_into`: one hint pass,
        ``out[slot] = round((y - H S) / 2^delta) mod 2^K``. ``hint`` holds byte planes ``1 .. P``."""
        m, npl, dim = hint.shape
        if dim != self.params.dim:
            raise ConfigError(f"hint dimension {dim} != {self.params.dim}")
        cts_all = [ct for it in items for ct in it[1]]
        cols = sum(int(y.shape[1]) for y, _, _ in items)
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
        big_bits = min(63, 8 * (1 + npl))
        sh = self._secrets_on_device([ct.secret for ct in cts_all], big_bits)
        try:
            # the replies W8 b are what the provider sent (not secret): uploaded straight into one device array
            yd = torch.empty((m, cols), dtype=torch.int64, device=self.dev)
            c0 = 0
            for y, _, _ in items:
                yh = torch.from_numpy(np.ascontiguousarray(y, dtype=np.uint64).view(np.int64))
                yd[:, c0:c0 + yh.shape[1]].copy_(yh)
                c0 += yh.shape[1]
        except BaseException:
            sh.zero_()
            raise
        vidx = torch.from_numpy(valid).to(self.dev)
        dev_hint = self.hint_on_device(key, hint)
        half = 1 << (delta_bits - 1)
        kmask = (1 << k) - 1
        block = max(8, min(m, self._block_rows(cols) // 2))
        try:
            for r0 in range(0, m, block):
                nb = min(block, m - r0)
                h = dev_hint[r0:r0 + nb] if dev_hint is not None else self._upload_block(hint[r0:r0 + nb])
                th = self._fwd_planes(h, 1)  # (3, rank, 256, nb): public (the hint is not secret)
                corr = self._mac(th, 0, nb, sh)  # (nb, cols) = H S mod 2^64
                del th
                v = yd[r0:r0 + nb] - corr
                corr.zero_()
                v += half
                v >>= delta_bits  # arithmetic shift: the K bits kept are the same (delta + K <= 64)
                v &= kmask
                sel = v.t().index_select(0, vidx)
                v.zero_()
                if k <= 32:  # download 4 bytes per value: [0, 2^K) as the int32 with the same low 32 bits
                    sel -= (sel >= 1 << 31).to(torch.int64) << 32
                    sel32 = sel.to(torch.int32)
                    sel.zero_()
                    host = sel32.to("cpu", copy=True).numpy()
                    sel32.zero_()
                    out[dst, r0:r0 + nb] = host.view(np.uint32)
                else:
                    host = sel.to("cpu", copy=True).numpy()
                    sel.zero_()
                    out[dst, r0:r0 + nb] = host
                host.fill(0)
            self.stats["decrypt_groups"] += len(cts_all)
        finally:
            sh.zero_()
            yd.zero_()


# ---------------------------------------------------------------------------------------------
# Self-test and selection
# ---------------------------------------------------------------------------------------------

_TESTED: dict[str, bool] = {}
_TEST_LOCK = threading.Lock()
_STATUS: dict[str, Any] = {}


def self_test(device: torch.device | str, k: int = 32, rank: int = 16, kernel: str | None = None) -> bool:
    """Encryption and decryption of :class:`GpuNTTHE` equal the float64 reference bit for bit at the real
    ``dim`` (cached per device, ring, rank and kernel)."""
    from pgw.hemask.fastmm import to_planes

    dev = torch.device(device)
    tag = f"{dev}/k{k}/r{rank}/{kernel or 'auto'}"
    with _TEST_LOCK:
        if tag in _TESTED:
            return _TESTED[tag]
        ok = False
        why = ""
        try:
            p = HEParams(k=k, rank=rank)
            rng = np.random.default_rng(4321)
            n, m, rows = 70, 45, DEGREE + 9
            a = public_matrix(bytes(range(1, 33)), n, p)
            eng = GpuNTTHE(a, p, dev, kernel=kernel, hint_budget=0)
            r = rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64)
            seeds = [bytes([11 + i]) * 32 for i in range(2)]
            cts = eng.encrypt(r, _seeds=seeds)
            b = np.concatenate([ct.b for ct in cts], axis=1)
            ref = np.concatenate([_reference_encrypt(r[i * DEGREE:(i + 1) * DEGREE], a, p, s)
                                  for i, s in enumerate(seeds)], axis=1)
            if not np.array_equal(b, ref):
                why = "encryption differs from the reference"
            else:
                w = rng.integers(-127, 128, size=(m, n)).astype(np.int64)
                hint = to_planes(matmul_mod64(w, a, fast=False))[:, 1:p.q // 8]
                y = matmul_mod64(w, b, fast=False) & p.q_mask
                want = np.array((r.astype(object) @ w.T.astype(object)) % (1 << k), dtype=np.int64)
                out = np.zeros((rows, m), dtype=np.int64)
                eng.decrypt_into("selftest", hint, [(y, cts, 0)], out, p.delta_bits, k)
                eng2 = GpuNTTHE(a, p, dev, kernel=kernel, hint_budget=1 << 30)
                out2 = np.zeros((rows, m), dtype=np.int64)
                eng2.decrypt_into("selftest", hint, [(y[:, :DEGREE], cts[:1], 0),
                                                     (y[:, DEGREE:], cts[1:], DEGREE)], out2, p.delta_bits, k)
                if not (np.array_equal(out, want) and np.array_equal(out2, want)):
                    why = "decryption differs from the reference"
                else:
                    ok = True
            for ct in cts:
                ct.wipe()
        except Exception as exc:  # a missing kernel, a compiler error: not used
            why = f"{type(exc).__name__}: {exc}"
        if ok:
            log.info("encrypted mask factory: GPU NTT engine on %s (%s) passed its self-test (K%d, rank %d)",
                     dev, kernel or "auto", k, rank)
        else:
            log.error("encrypted mask factory: GPU NTT engine on %s FAILED its self-test (%s)", dev, why)
        _TESTED[tag] = ok
        _STATUS[tag] = {"ok": ok, **({"error": why[:300]} if why else {})}
        return ok


def gpu_ntt_for(a: np.ndarray, params: HEParams, device: str | None = "auto",
                **kw: Any) -> GpuNTTHE | None:
    """A :class:`GpuNTTHE` on ``device`` (``"auto"``: ``cuda`` when available) after a passing
    :func:`self_test`, else ``None`` (the caller falls back to the CPU paths)."""
    if device in (None, "auto"):
        if not torch.cuda.is_available():
            log.warning("he_encrypt='ntt_gpu': no CUDA device; falling back to the CPU path")
            return None
        device = "cuda"
    dev = torch.device(device)
    kernel = kw.get("kernel")
    if not self_test(dev, params.k, params.rank, kernel):
        return None
    try:
        return GpuNTTHE(a, params, dev, **kw)
    except Exception as exc:  # out of device memory for A in NTT form, ...
        log.error("he_encrypt='ntt_gpu': could not start the GPU NTT engine (%s); falling back to the CPU path",
                  exc)
        return None


def status() -> dict[str, Any]:
    """Self-test results of this process."""
    return dict(_STATUS)
