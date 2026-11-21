"""Gateway-GPU NTT for the encrypted mask factory: ``A s`` and ``H S`` exactly, on a torch device.

The CPU path :class:`pgw.hemask.mlwe_ring.PretransformedEncryptor` (native crate ``pgw_ntt``) keeps the
public matrix ``A`` in NTT form and computes the encryption pad ``A[:n] @ S`` (``S`` the
block-negacyclic matrix of the ternary secret) as ``rank`` pointwise products plus one inverse NTT per
row, group and prime. This module runs the same mathematics on a torch device (a gateway GPU; tests run
it on CPU tensors):

* :class:`DeviceNTT` -- three NTT primes ``p < 2^29``, ``p = 1 mod 512`` (negacyclic length-256
  transforms), Barrett multiplication in int64 (products ``< 2^58``), Garner CRT with a centred last
  digit and wrapping int64 arithmetic for the result mod ``2^64``.
* :class:`DeviceTransformedA` -- ``A`` (low ``q`` bits) forward-transformed once and kept on the device
  as int32 residues (12 bytes per coefficient); ``mm_secrets(secrets, rows=n)`` has the contract of
  :meth:`pgw.hemask.ntt.TransformedOperand.mm_secrets` (``A[:n] @ S mod 2^64``, uint64 on the host), so
  :class:`~pgw.hemask.mlwe_ring.PretransformedEncryptor` uses it unchanged.
* :meth:`DeviceNTT.hint_mm` -- the decryption correction ``H @ S mod 2^64`` from the hint's int8 byte
  planes (``first_byte=1``), same bits as :func:`pgw.hemask.ntt.big_planes_mm_small`. The hint rows are
  forward-transformed on the fly per pass (one transform serves every group of the pass).

Optionally (CUDA only) the per-row hot loop -- pointwise multiply-accumulate over the rank, inverse NTT,
twist and Garner CRT -- runs in one fused Triton kernel (:mod:`pgw.hemask.ntt_triton`); the torch
implementation is the reference and the fallback.

Exactness: an output coefficient is ``Z = sum +- a * s`` with ``|a| < 2^b`` (``b <= 64``) and
``sum |s| = L1 <= 256 rank``; the CRT recovers it exactly when ``(2^b - 1) L1 <= HALF = p0 p1 (p2 - 1) / 2
~ 2^86`` (rank 16: ``2^68``-``2^76``). :func:`exact` checks the bound on every call. All arithmetic is
integer; the result mod ``2^64`` equals the dense product bit for bit. Before first use,
:func:`self_test` compares the device results with the CPU references (native NTT when present, else
the dense product) at the real dimension; on any mismatch or error the device path is not used and the
caller falls back to the CPU (:func:`device_ntt_for`).

Privacy: this is customer-side work only; the provider receives exactly the same ciphertexts (same
seeds -> same bits, same sizes and schedule). Device temporaries holding secret-derived values (secret
transforms, ``A s``, ``H S``) are zeroed before they are released.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError
from pgw.hemask.lwe import DEGREE, HEParams

__all__ = [
    "PRIMES",
    "DeviceNTT",
    "DeviceTransformedA",
    "device_ntt_for",
    "exact",
    "secrets_from_matrix",
    "self_test",
    "status",
]

log = logging.getLogger(__name__)

_LOGN = 8
#: Bytes of int64 temporaries per row block (bounds device memory of one pass).
_BLOCK_BYTES = 192 << 20
_OFF = ("0", "false", "no", "off")


# ---------------------------------------------------------------------------------------------
# Primes and constants
# ---------------------------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % d == 0:
            return n == d
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):  # deterministic below 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_primes(count: int = 3) -> tuple[int, ...]:
    """The ``count`` largest primes below ``2^29`` with ``p = 1 mod 512``."""
    out: list[int] = []
    p = ((1 << 29) - 1) // 512 * 512 + 1
    while len(out) < count:
        if p < (1 << 29) and _is_prime(p):
            out.append(p)
        p -= 512
    return tuple(out)


#: The three NTT primes (all in ``(2^28, 2^29)``, ``= 1 mod 512``).
PRIMES: tuple[int, ...] = _find_primes()
_P0, _P1, _P2 = PRIMES
#: Largest exactly recoverable ``|Z|`` (centred last CRT digit).
HALF = _P0 * _P1 * ((_P2 - 1) // 2)


def exact(big_bits: int, l1: int) -> bool:
    """``(2^big_bits - 1) * l1 <= HALF``: products of a ``big_bits`` operand with a secret of L1 norm ``l1``
    are exact."""
    return ((1 << big_bits) - 1) * int(l1) <= HALF


def _psi(p: int) -> int:
    """A primitive 512-th root of unity mod ``p``."""
    for g in range(2, 1000):
        c = pow(g, (p - 1) // (2 * DEGREE), p)
        if pow(c, DEGREE, p) == p - 1:
            return c
    raise AssertionError("no root of unity")  # pragma: no cover


def _signed(v: int) -> int:
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >= 1 << 63 else v


def secrets_from_matrix(small: np.ndarray) -> np.ndarray:
    """The ``(G, rank, 256)`` int8 secret polynomials of a block-negacyclic matrix ``small``
    (``rank*256 x G*256``, :func:`pgw.hemask.lwe.secret_matrix` blocks along columns).

    Row ``t = 0`` of block ``l`` is ``s_l`` itself; the whole matrix is then checked against the
    structure (raises :class:`ConfigError` otherwise), so a non-secret operand can never be multiplied
    as if it were one. The returned array is a fresh copy the caller must zero.
    """
    from pgw.hemask.lwe import secret_matrix

    s = np.asarray(small)
    if s.ndim != 2 or s.shape[0] % DEGREE or s.shape[1] % DEGREE or s.size == 0:
        raise ConfigError(f"secret matrix shape {s.shape} is not (rank*256, G*256)")
    rank, g = s.shape[0] // DEGREE, s.shape[1] // DEGREE
    out = np.empty((g, rank, DEGREE), dtype=np.int8)
    for gi in range(g):
        blk = s[:, gi * DEGREE:(gi + 1) * DEGREE]
        out[gi] = blk[::DEGREE, :]
        ref = secret_matrix(out[gi])
        try:
            if not np.array_equal(ref, blk):
                out.fill(0)
                raise ConfigError("operand is not a block-negacyclic secret matrix (lwe.secret_matrix)")
        finally:
            ref.fill(0)
    return out


# ---------------------------------------------------------------------------------------------
# The device NTT
# ---------------------------------------------------------------------------------------------


class DeviceNTT:
    """Exact negacyclic NTT products over three primes on ``device``.

    Data layout: ``(..., 3, 256)`` int64 residues in ``[0, p)`` (prime axis second to last).
    """

    def __init__(self, device: torch.device | str = "cpu", *, use_triton: bool | None = None):
        self.dev = torch.device(device)
        dev = self.dev
        p = torch.tensor(PRIMES, dtype=torch.int64, device=dev)
        self.p = p.view(3, 1)  # broadcast over the coefficient axis
        self.m = torch.tensor([(1 << 58) // q for q in PRIMES], dtype=torch.int64, device=dev).view(3, 1)
        fwd, inv = [], []
        psi_pow, psi_inv_pow = [], []
        for q in PRIMES:
            psi = _psi(q)
            om = psi * psi % q
            om_inv = pow(om, q - 2, q)
            psi_inv = pow(psi, q - 2, q)
            n_inv = pow(DEGREE, q - 2, q)
            psi_pow.append([pow(psi, i, q) for i in range(DEGREE)])
            psi_inv_pow.append([pow(psi_inv, i, q) * n_inv % q for i in range(DEGREE)])
            fwd.append([[pow(om, j << s, q) for j in range(128 >> s)] for s in range(_LOGN)])
            inv.append([[pow(om_inv, j << s, q) for j in range(128 >> s)] for s in range(_LOGN)])
        self.psi = torch.tensor(psi_pow, dtype=torch.int64, device=dev)  # (3, 256)
        self.psi_inv = torch.tensor(psi_inv_pow, dtype=torch.int64, device=dev)  # psi^-i / 256
        # stage s: (3, 1, half) twiddles omega^(j 2^s), half = 128 >> s
        self.tw_f = [torch.tensor([fwd[i][s] for i in range(3)], dtype=torch.int64, device=dev).view(3, 1, -1)
                     for s in range(_LOGN)]
        self.tw_i = [torch.tensor([inv[i][s] for i in range(3)], dtype=torch.int64, device=dev).view(3, 1, -1)
                     for s in range(_LOGN)]
        # Garner constants
        self.c_inv01 = pow(_P0, _P1 - 2, _P1)  # p0^-1 mod p1
        self.c_p0m2 = _P0 % _P2
        self.c_inv012 = pow(_P0 * _P1 % _P2, _P2 - 2, _P2)  # (p0 p1)^-1 mod p2
        self.c_p01 = _signed(_P0 * _P1)
        self.triton: Any = None
        want = _triton_wanted() if use_triton is None else use_triton
        if want and self.dev.type == "cuda":  # pragma: no cover - GPU only
            try:
                from pgw.hemask import ntt_triton

                self.triton = ntt_triton.Kernels(self)
            except Exception as e:
                log.warning("device NTT: Triton kernels unavailable (%s); using torch ops", e)
        elif want and os.environ.get("TRITON_INTERPRET", "") == "1":  # tests: the kernel on CPU tensors
            from pgw.hemask import ntt_triton

            self.triton = ntt_triton.Kernels(self)

    # ---- modular helpers (broadcast over a (..., 3, k) tensor) ---------------------------------
    def _p(self, x: torch.Tensor) -> torch.Tensor:
        """The prime column broadcastable against ``x`` (prime axis = ``x.ndim - 2``)."""
        return self.p

    def mulmod(self, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor | None = None,
               m: torch.Tensor | None = None) -> torch.Tensor:
        """``a b mod p`` for ``0 <= a, b < p < 2^29`` (Barrett: ``a b < 2^58``)."""
        p = self.p if p is None else p
        m = self.m if m is None else m
        x = a * b
        q = ((x >> 28) * m) >> 30  # floor(x/p) - 3 <= q <= floor(x/p)
        x -= q * p
        for _ in range(3):
            x = torch.where(x >= p, x - p, x)
        return x

    def _addsub(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = u + v
        s = torch.where(s >= self.p.unsqueeze(-1), s - self.p.unsqueeze(-1), s)
        d = u - v
        d = torch.where(d < 0, d + self.p.unsqueeze(-1), d)
        return s, d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Negacyclic forward NTT of residues ``(..., 3, 256)`` (output in bit-reversed order)."""
        lead = x.shape[:-2]
        x = self.mulmod(x, self.psi)
        p3 = self.p.unsqueeze(-1)
        for s in range(_LOGN):
            half = 128 >> s
            y = x.reshape(*lead, 3, 1 << s, 2, half)
            u, v = y[..., 0, :], y[..., 1, :]
            a, d = self._addsub(u, v)
            b = self.mulmod(d, self.tw_f[s], p3, self.m.unsqueeze(-1))
            x = torch.stack((a, b), dim=-2).reshape(*lead, 3, DEGREE)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`forward` (bit-reversed input, natural output, scaled and untwisted)."""
        lead = x.shape[:-2]
        p3 = self.p.unsqueeze(-1)
        m3 = self.m.unsqueeze(-1)
        for s in reversed(range(_LOGN)):
            half = 128 >> s
            y = x.reshape(*lead, 3, 1 << s, 2, half)
            u = y[..., 0, :]
            v = self.mulmod(y[..., 1, :], self.tw_i[s], p3, m3)
            a, d = self._addsub(u, v)
            x = torch.stack((a, d), dim=-2).reshape(*lead, 3, DEGREE)
        return self.mulmod(x, self.psi_inv)

    def crt(self, r: torch.Tensor) -> torch.Tensor:
        """Residues ``(..., 3, k)`` -> the centred integer mod ``2^64`` as int64 ``(..., k)`` (wrapping)."""
        r0, r1, r2 = r.unbind(-2)
        p1 = torch.tensor(_P1, dtype=torch.int64, device=r.device)
        p2 = torch.tensor(_P2, dtype=torch.int64, device=r.device)
        m1 = torch.tensor((1 << 58) // _P1, dtype=torch.int64, device=r.device)
        m2 = torch.tensor((1 << 58) // _P2, dtype=torch.int64, device=r.device)
        x0m1 = torch.where(r0 >= _P1, r0 - _P1, r0)  # r0 < p0 < 2 p1
        d = r1 - x0m1
        d = torch.where(d < 0, d + _P1, d)
        x1 = self.mulmod(d, torch.tensor(self.c_inv01, device=r.device), p1, m1)
        t = self.mulmod(x1, torch.tensor(self.c_p0m2, device=r.device), p2, m2)
        x0m2 = torch.where(r0 >= _P2, r0 - _P2, r0)
        t = t + x0m2
        t = torch.where(t >= _P2, t - _P2, t)
        d = r2 - t
        d = torch.where(d < 0, d + _P2, d)
        x2 = self.mulmod(d, torch.tensor(self.c_inv012, device=r.device), p2, m2)
        x2 = torch.where(x2 > (_P2 - 1) // 2, x2 - _P2, x2)  # centred last digit
        return r0 + x1 * _P0 + x2 * self.c_p01  # wraps mod 2^64

    # ---- residues ---------------------------------------------------------------------------------
    def residues_u64(self, x: torch.Tensor) -> torch.Tensor:
        """int64 tensor holding uint64 bits ``(...,)`` -> residues ``(..., 3)`` ... as ``(3, ...)``."""
        lo = x & 0xFFFFFFFF
        hi = (x >> 32) & 0xFFFFFFFF
        out = []
        for i, q in enumerate(PRIMES):
            c = (1 << 32) % q
            h = hi % q
            out.append(self.mulmod(h, torch.tensor(c, device=x.device), self.p[i], self.m[i]) + lo % q)
            out[-1] = torch.where(out[-1] >= q, out[-1] - q, out[-1])
        return torch.stack(out, dim=0)

    def secrets_hat(self, secrets: np.ndarray) -> torch.Tensor:
        """Forward transforms of int8 secrets ``(G, rank, 256)`` -> ``(G, rank, 3, 256)`` int64 on the device."""
        # a private device copy (never the caller's buffer: it is wiped below)
        s = torch.from_numpy(np.ascontiguousarray(secrets, dtype=np.int8)).to(self.dev, copy=True)
        try:
            s64 = s.to(torch.int64).unsqueeze(-2)  # (G, rank, 1, 256)
            res = torch.where(s64 < 0, s64 + self.p, s64)  # (G, rank, 3, 256)
            s64.zero_()
            out = self.forward(res)
            res.zero_()
            return out
        finally:
            s.zero_()

    # ---- the hot loop: pointwise MAC, inverse NTT, CRT ---------------------------------------------
    def mac_inverse_crt(self, a_hat: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        """``a_hat`` ``(R, rank, 3, 256)`` (int32/int64 residues), ``s_hat`` ``(G, rank, 3, 256)`` int64 ->
        ``sum_l a_l * s_{g,l}`` as int64 ``(R, G, 256)`` (the exact product mod 2^64)."""
        if self.triton is not None:
            fused: torch.Tensor = self.triton.mac_inverse_crt(a_hat, s_hat)
            return fused
        rows, rank = a_hat.shape[0], a_hat.shape[1]
        g = s_hat.shape[0]
        acc = torch.zeros((rows, g, 3, DEGREE), dtype=torch.int64, device=self.dev)
        try:
            for l0 in range(0, rank, 16):  # 16 products < 2^58 sum below 2^62
                for li in range(l0, min(rank, l0 + 16)):
                    acc += a_hat[:, None, li].to(torch.int64) * s_hat[None, :, li]
                acc.remainder_(self.p)
            z = self.inverse(acc)
            try:
                return self.crt(z)
            finally:
                z.zero_()
        finally:
            acc.zero_()

    def _rows_per_block(self, rank: int, g: int) -> int:
        per_row = 8 * 3 * DEGREE * (rank + 4 * g)
        return max(16, min(4096, _BLOCK_BYTES // per_row))

    def hint_mm(self, hint: np.ndarray | torch.Tensor, small: np.ndarray, *, first_byte: int = 1,
                secrets: np.ndarray | None = None) -> np.ndarray:
        """``U @ S mod 2^64`` for byte planes ``U`` ``(m, P, dim)`` int8 holding bytes ``first_byte ..``
        (missing bytes zero) and the block-negacyclic secret matrix ``small``; uint64 ``(m, cols)``,
        same bits as :func:`pgw.hemask.ntt.big_planes_mm_small`."""
        m, npl, dim = hint.shape
        if dim % DEGREE or small.shape[0] != dim:
            raise ConfigError(f"hint dim {dim} does not match the secret matrix {small.shape}")
        if first_byte < 0 or first_byte + npl > 8:
            raise ConfigError("planes out of range")
        sec = secrets_from_matrix(small) if secrets is None else secrets
        rank, g = dim // DEGREE, sec.shape[0]
        if not exact(8 * (first_byte + npl), int(np.abs(sec.astype(np.int64)).sum((1, 2)).max())):
            raise ConfigError("secret too large for an exact NTT product")
        out = np.empty((m, g * DEGREE), dtype=np.uint64)
        s_hat = self.secrets_hat(sec)
        if secrets is None:
            sec.fill(0)
        weights = [_signed(1 << (8 * (first_byte + i))) for i in range(npl)]
        try:
            blk = self._rows_per_block(rank, g)
            for r0 in range(0, m, blk):
                h = hint[r0:r0 + blk]
                ht = (h if isinstance(h, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(h))).to(self.dev)
                v = torch.zeros(ht.shape[0], dim, dtype=torch.int64, device=self.dev)
                for i in range(npl):
                    v += (ht[:, i].to(torch.int64) + 128) * weights[i]  # byte = plane + 128
                res = self.residues_u64(v)  # (3, R, dim)
                a_hat = self.forward(res.view(3, -1, rank, DEGREE).permute(1, 2, 0, 3))
                z = self.mac_inverse_crt(a_hat, s_hat)
                out[r0:r0 + ht.shape[0]] = z.reshape(ht.shape[0], g * DEGREE).cpu().numpy().view(np.uint64)
                z.zero_()
        finally:
            s_hat.zero_()
        return out


class DeviceTransformedA:
    """The public matrix in NTT form on a device; drop-in for :class:`pgw.hemask.ntt.TransformedOperand`
    inside :class:`pgw.hemask.mlwe_ring.PretransformedEncryptor`.

    Args:
        a: Public matrix ``(rows, dim)`` uint64 (host).
        q_bits: Only the low ``q_bits`` of ``A`` are transformed (``b`` is reduced mod ``2^q``; the CPU
            path keeps the same low planes), so ``big_bits = q_bits``.
        ntt: The :class:`DeviceNTT` to use.
    """

    def __init__(self, a: np.ndarray, q_bits: int, ntt: DeviceNTT):
        self.ntt = ntt
        self.big_bits = int(q_bits)
        rows, dim = a.shape
        if dim % DEGREE:
            raise ConfigError(f"dimension {dim} is not a multiple of {DEGREE}")
        self.rank = dim // DEGREE
        self._rows: int = int(rows)
        self.t = torch.empty((rows, self.rank, 3, DEGREE), dtype=torch.int32, device=ntt.dev)
        mask = (1 << q_bits) - 1 if q_bits < 64 else (1 << 64) - 1
        blk = 1024
        for r0 in range(0, rows, blk):
            x = np.ascontiguousarray(a[r0:r0 + blk], dtype=np.uint64) & np.uint64(mask)
            xt = torch.from_numpy(x.view(np.int64)).to(ntt.dev)
            res = ntt.residues_u64(xt)  # (3, R, dim)
            hat = ntt.forward(res.view(3, -1, self.rank, DEGREE).permute(1, 2, 0, 3))
            self.t[r0:r0 + blk] = hat.to(torch.int32)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def nbytes(self) -> int:
        return int(self.t.numel() * self.t.element_size())

    def mm_secrets(self, secrets: np.ndarray, rows: int | None = None) -> np.ndarray:
        """``A[:rows] @ S mod 2^64`` (uint64 ``(rows, G*256)``) for secrets ``(G, rank, 256)`` int8."""
        n = self._rows if rows is None else int(rows)
        if secrets.shape[1:] != (self.rank, DEGREE):
            raise ConfigError(f"secrets shape {secrets.shape} does not match rank {self.rank}")
        if not exact(self.big_bits, int(np.abs(secrets.astype(np.int64)).sum((1, 2)).max())):
            raise ConfigError("secret too large for an exact NTT product")
        g = secrets.shape[0]
        out = np.empty((n, g * DEGREE), dtype=np.uint64)
        s_hat = self.ntt.secrets_hat(secrets)
        try:
            blk = self.ntt._rows_per_block(self.rank, g)
            for r0 in range(0, n, blk):
                r1 = min(n, r0 + blk)
                z = self.ntt.mac_inverse_crt(self.t[r0:r1], s_hat)
                try:
                    out[r0:r1] = z.reshape(r1 - r0, g * DEGREE).cpu().numpy().view(np.uint64)
                finally:
                    z.zero_()
        finally:
            s_hat.zero_()
        return out


# ---------------------------------------------------------------------------------------------
# Start-up self-check and device choice
# ---------------------------------------------------------------------------------------------

_TESTED: dict[str, bool] = {}
_LOCK = threading.Lock()
_STATUS: dict[str, Any] = {}


def _triton_wanted() -> bool:
    return os.environ.get("PGW_NTT_DEV_TRITON", "1").strip().lower() not in _OFF


def self_test(ntt: DeviceNTT, params: HEParams, *, rows: int = 8, groups: int = 2) -> bool:
    """Compare encryption pads and hint products on the device with the CPU references at the real
    ``dim`` (random 64-bit ``A`` rows, worst-case all-``+-1`` and random ternary secrets)."""
    from pgw.hemask.fastmm import to_planes
    from pgw.hemask.lwe import matmul_mod64, secret_matrix

    rng = np.random.default_rng(20261121)
    rank = params.rank
    a = rng.integers(0, 1 << 64, size=(rows, params.dim), dtype=np.uint64)
    a[0] = np.uint64((1 << 64) - 1)  # largest operand
    secs = rng.integers(-1, 2, size=(groups, rank, DEGREE), dtype=np.int64).astype(np.int8)
    secs[0] = 1  # largest L1 norm
    smat = np.concatenate([secret_matrix(s) for s in secs], axis=1)
    try:
        for q_bits in sorted({params.q, 64}):
            aq = a & np.uint64((1 << q_bits) - 1) if q_bits < 64 else a
            want = matmul_mod64(smat, aq, small_left=False)
            got = DeviceTransformedA(a, q_bits, ntt).mm_secrets(secs, rows=rows)
            if not np.array_equal(got, want):
                return False
        npl = params.q // 8 - 1
        planes = to_planes(a)[:, 1:1 + npl]
        hv = a & np.uint64(((1 << (8 * (1 + npl))) - 1) ^ 0xFF)  # bytes 1..npl
        want = matmul_mod64(smat, hv, small_left=False)
        got = ntt.hint_mm(planes, smat, first_byte=1)
        return bool(np.array_equal(got, want))
    finally:
        smat.fill(0)
        secs.fill(0)


def status() -> dict[str, Any]:
    """Last self-test outcome per device (for logs and benchmarks)."""
    return dict(_STATUS)


def device_ntt_for(params: HEParams, device: str | torch.device | None = "auto") -> DeviceNTT | None:
    """A self-tested :class:`DeviceNTT` on ``device`` (``"auto"``: CUDA when available), or ``None``
    (no device, or the self-test failed: use the CPU path)."""
    if device is None:
        return None
    if device == "auto":
        if not torch.cuda.is_available():
            return None
        device = "cuda"
    dev = torch.device(device)
    rows = 67 if dev.type == "cuda" else 8  # several Triton row blocks on a GPU; cheap CPU-tensor tests
    with _LOCK:
        # candidates: the fused Triton kernel (when wanted and available), then plain torch ops
        for use_triton in ([True, False] if _triton_wanted() else [False]):
            key = f"{dev}:{params.rank}:{params.q}:{use_triton}"
            if _TESTED.get(key) is False:
                continue
            try:
                ntt = DeviceNTT(dev, use_triton=use_triton)
                if use_triton and ntt.triton is None:
                    continue
                if key not in _TESTED:
                    _TESTED[key] = self_test(ntt, params, rows=rows)
                    _STATUS[str(dev)] = {"ok": _TESTED[key], "triton": use_triton, "rank": params.rank}
            except Exception as e:
                log.warning("device NTT on %s (triton=%s) unavailable: %s", dev, use_triton, e)
                _TESTED[key] = False
                _STATUS[str(dev)] = {"ok": False, "triton": use_triton, "error": repr(e)}
                continue
            if _TESTED[key]:
                return ntt
            log.warning("device NTT on %s (triton=%s) failed its self-test", dev, use_triton)
        log.warning("device NTT on %s unavailable; using the CPU path", dev)
        return None
