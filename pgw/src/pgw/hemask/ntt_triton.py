# mypy: disable-error-code="no-untyped-def,untyped-decorator"
"""Fused Triton kernel for the device NTT hot loop (:mod:`pgw.hemask.ntt_dev`).

One program handles ``BR`` rows of one group: for each of the three primes it multiply-accumulates
``rank`` pointwise products ``a_hat[r, l] * s_hat[g, l]`` (int64, reduced every 16 terms), runs the
inverse negacyclic NTT in registers (reshape / split / join butterflies, Barrett multiplication), and
then combines the three residues with Garner's CRT into the exact integer mod ``2^64``. Only the int64
result leaves the kernel (8 bytes per coefficient instead of 24 bytes of residues per prime pass), so
the whole per-row computation is one launch.

The arithmetic is the same integer arithmetic as the torch reference (:class:`ntt_dev.DeviceNTT`) and
the result is checked against the CPU at start-up (:func:`pgw.hemask.ntt_dev.self_test`); a failing or
unavailable kernel falls back to the torch ops. Tests run the kernel on CPU tensors in Triton's
interpreter (``TRITON_INTERPRET=1``).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import triton
import triton.language as tl

from pgw.hemask.ntt_dev import PRIMES

__all__ = ["Kernels"]

_P0, _P1, _P2 = PRIMES


@triton.jit
def _mulmod(a, b, p, m):
    """``a b mod p`` for ``0 <= a, b < p < 2^29`` (Barrett, ``m = floor(2^58 / p)``)."""
    x = a * b
    q = ((x >> 28) * m) >> 30
    x = x - q * p
    x = tl.where(x >= p, x - p, x)
    x = tl.where(x >= p, x - p, x)
    x = tl.where(x >= p, x - p, x)
    return x


@triton.jit
def _one_prime(A, S, TW, PSI, rows, rmask, g, PI: tl.constexpr, p, m, RANK: tl.constexpr,
               BR: tl.constexpr):
    cols = tl.arange(0, 256)
    acc = tl.zeros((BR, 256), dtype=tl.int64)
    for li in tl.static_range(RANK):
        a_off = ((rows[:, None].to(tl.int64) * RANK + li) * 3 + PI) * 256 + cols[None, :]
        a = tl.load(A + a_off, mask=rmask[:, None], other=0).to(tl.int64)
        s = tl.load(S + ((g * RANK + li) * 3 + PI) * 256 + cols).to(tl.int64)
        acc += a * s[None, :]
        if li % 16 == 15:
            acc = acc % p
    acc = acc % p
    x = acc
    for i in tl.static_range(8):
        s_: tl.constexpr = 7 - i
        H: tl.constexpr = 128 >> s_
        B: tl.constexpr = 1 << s_
        y = tl.reshape(x, (BR, B, 2, H))
        y = tl.permute(y, (0, 1, 3, 2))
        u, v = tl.split(y)
        w = tl.load(TW + (PI * 8 + s_) * 128 + tl.arange(0, H)).to(tl.int64)
        v = _mulmod(v, w[None, None, :], p, m)
        sm = u + v
        sm = tl.where(sm >= p, sm - p, sm)
        df = u - v
        df = tl.where(df < 0, df + p, df)
        y = tl.join(sm, df)
        y = tl.permute(y, (0, 1, 3, 2))
        x = tl.reshape(y, (BR, 256))
    psi = tl.load(PSI + PI * 256 + cols).to(tl.int64)
    return _mulmod(x, psi[None, :], p, m)


@triton.jit
def _mac_inv_crt(A, S, TW, PSI, OUT, R, G, RANK: tl.constexpr, BR: tl.constexpr,
                 P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                 M0: tl.constexpr, M1: tl.constexpr, M2: tl.constexpr,
                 INV01: tl.constexpr, P0M2: tl.constexpr, INV012: tl.constexpr, P01: tl.constexpr):
    g = tl.program_id(0)  # groups vary fastest: programs on the same rows run together and share A in L2
    pid = tl.program_id(1)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    r0 = _one_prime(A, S, TW, PSI, rows, rmask, g, 0, P0, M0, RANK, BR)
    r1 = _one_prime(A, S, TW, PSI, rows, rmask, g, 1, P1, M1, RANK, BR)
    r2 = _one_prime(A, S, TW, PSI, rows, rmask, g, 2, P2, M2, RANK, BR)
    # Garner: x0 = r0, x1 = (r1 - x0) / p0 mod p1, x2 = (r2 - x0 - x1 p0) / (p0 p1) mod p2 (centred)
    x0m1 = tl.where(r0 >= P1, r0 - P1, r0)
    d = r1 - x0m1
    d = tl.where(d < 0, d + P1, d)
    x1 = _mulmod(d, INV01, P1, M1)
    t = _mulmod(x1, P0M2, P2, M2)
    x0m2 = tl.where(r0 >= P2, r0 - P2, r0)
    t = t + x0m2
    t = tl.where(t >= P2, t - P2, t)
    d = r2 - t
    d = tl.where(d < 0, d + P2, d)
    x2 = _mulmod(d, INV012, P2, M2)
    x2 = tl.where(x2 > (P2 - 1) // 2, x2 - P2, x2)
    z = r0 + x1 * P0 + x2 * P01  # wraps mod 2^64
    cols = tl.arange(0, 256)
    off = (rows[:, None].to(tl.int64) * G + g) * 256 + cols[None, :]
    tl.store(OUT + off, z, mask=rmask[:, None])


class Kernels:
    """Launcher bound to one :class:`pgw.hemask.ntt_dev.DeviceNTT` (its device and twiddle tables)."""

    def __init__(self, ntt: Any, block_rows: int | None = None):
        self.ntt = ntt
        # rows per program (power of two); PGW_NTT_DEV_BLOCK_ROWS tunes it on a new GPU
        self.br = int(block_rows or os.environ.get("PGW_NTT_DEV_BLOCK_ROWS", "4"))
        if self.br < 1 or self.br & (self.br - 1):
            raise ValueError(f"block rows must be a power of two, got {self.br}")
        tw = torch.zeros((3, 8, 128), dtype=torch.int64, device=ntt.dev)
        for s in range(8):
            tw[:, s, :128 >> s] = ntt.tw_i[s].view(3, -1)
        self.tw = tw.contiguous()
        self.psi_inv = ntt.psi_inv.contiguous()
        self.consts = dict(P0=_P0, P1=_P1, P2=_P2, M0=(1 << 58) // _P0, M1=(1 << 58) // _P1,
                           M2=(1 << 58) // _P2, INV01=ntt.c_inv01, P0M2=ntt.c_p0m2, INV012=ntt.c_inv012,
                           P01=ntt.c_p01)

    def mac_inverse_crt(self, a_hat: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        rows, rank = int(a_hat.shape[0]), int(a_hat.shape[1])
        g = int(s_hat.shape[0])
        a = a_hat if a_hat.is_contiguous() else a_hat.contiguous()
        s = s_hat.contiguous()
        out = torch.empty((rows, g, 256), dtype=torch.int64, device=a.device)
        grid = (g, triton.cdiv(rows, self.br))
        _mac_inv_crt[grid](a, s, self.tw, self.psi_inv, out, rows, g, RANK=rank, BR=self.br, **self.consts)
        return out
