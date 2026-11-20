"""Module-LWE encryption of one-time masks, so the compute service can produce ``W8 r`` blind.

The mask factory needs ``W8 r mod 2^K`` for uniform one-time masks ``r``. Computing it locally costs as
much as running the model's linear layers, which is what PGW outsources. Here the gateway (customer)
encrypts ``r`` under module-LWE, the provider multiplies the ciphertext by the plaintext int8 weights,
and the gateway decrypts ``W8 r``. The provider sees only MLWE samples, which are pseudorandom, so it
learns nothing about ``r`` (single provider, no second party, post-quantum assumption).

Layout (token axis, "coefficient encoding"): a group of ``G = degree = 256`` future token slots is
encrypted together. For input feature ``c`` of a linear, polynomial coefficient ``j`` carries
``r[j, c]``::

    b_c = sum_l a_{c,l} * s_l + e_c + 2^(64-K) * r_c      (mod 2^64, in Z[X]/(X^256 + 1))

``a_{c,l}`` is a public uniform polynomial (one row of the public matrix ``A``, expanded from a
customer-chosen seed), ``s`` a fresh ternary secret per (linear, group), ``e`` centred-binomial noise.
Because ``W8`` multiplies each feature by an integer scalar, the provider's work is a plain matrix
product mod ``2^64`` over the coefficients: ``Y = W8 B``. With the one-time hint ``H = W8 A`` (computed
once per pack and public seed), the gateway decrypts::

    Y_i - sum_l H_{i,l} * s_l = (W8 e)_i + 2^(64-K) (W8 r)_i      (mod 2^64)

and rounds away ``W8 e`` (|W8 e| << 2^(63-K)) to get ``W8 r mod 2^K`` exactly.

Negacyclic products by the small secret are written as one GEMM with the secret's block matrix ``S``
(``dim x 256``, entries in {-1, 0, 1}). All products mod 2^64 are computed exactly with float64 GEMMs
on bit slices of the large operand (see :func:`matmul_mod64`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pgw.core.errors import ConfigError
from pgw.core.prng import SeedExpander, new_seed

__all__ = [
    "DEGREE",
    "HEParams",
    "MaskCiphertext",
    "a_plane_count",
    "decrypt_wr",
    "encrypt_masks",
    "matmul_mod64",
    "public_matrix",
    "sample_noise",
    "sample_secret",
    "secret_matrix",
]

DEGREE = 256
_U64 = np.uint64
_F53 = 53  # float64 mantissa: integer sums below 2^53 are exact


@dataclass(frozen=True)
class HEParams:
    """Parameters of the MLWE mask encryption.

    Attributes:
        k: Mask ring bits ``K`` (from the pack's ring preset). The plaintext modulus is ``2^K``.
        rank: Module rank; the LWE dimension is ``rank * 256``. Default 16 (dimension 4096): at
            ``q = 2^64`` this is inside the HomomorphicEncryption.org standard's 128-bit table
            (``log q <= 109`` for ``n = 4096``, ternary secret). Smaller ranks shrink the hint and
            gateway CPU but must be confirmed with the lattice estimator first.
        eta: Centred-binomial noise parameter (standard deviation ``sqrt(eta / 2)``; 21 gives 3.24).
        tail_sigmas: Decryption must succeed with ``|W8 e| < tail_sigmas * sigma * ||w||_2``.
    """

    k: int
    rank: int = 16
    eta: int = 21
    tail_sigmas: float = 24.0
    #: Ciphertext modulus bits (0 = automatic: 56 for K <= 32, else 64). The scaling factor
    #: ``2^(q-K)`` must stay >= 2^24 to absorb the noise, so a 32-bit mask ring fits q = 2^56:
    #: 7-byte uploads, 5-byte downloads, 6-plane hints, and more lattice security at the same rank.
    q_bits: int = 0

    def __post_init__(self) -> None:
        if not 16 <= self.k <= 48 or self.k % 8:
            raise ConfigError(f"HE mask encryption needs a ring K in {{16, 24, 32, 40, 48}}, got {self.k}")
        if self.rank < 8:
            raise ConfigError(f"rank {self.rank} gives LWE dimension {self.rank * DEGREE} < 2048: insecure at q=2^64")
        if not 1 <= self.eta <= 32:
            raise ConfigError(f"eta must be in [1, 32], got {self.eta}")
        if self.q_bits not in (0, 56, 64):
            raise ConfigError(f"q_bits must be 0 (auto), 56 or 64, got {self.q_bits}")
        if self.q - self.k < 24:
            raise ConfigError(f"K={self.k} leaves only {self.q - self.k} bits of noise room at q=2^{self.q} "
                              f"(24 needed): encrypted masks support rings up to K40 (K32 uses q=2^56)")

    @property
    def dim(self) -> int:
        """LWE dimension ``rank * 256`` (columns of the public matrix and of the hint)."""
        return self.rank * DEGREE

    @property
    def q(self) -> int:
        """Ciphertext modulus bits."""
        return self.q_bits or (56 if self.k <= 32 else 64)

    @property
    def q_mask(self) -> np.uint64:
        return np.uint64((1 << self.q) - 1) if self.q < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)

    @property
    def b_bytes(self) -> int:
        """Bytes per uploaded ciphertext coefficient."""
        return self.q // 8

    @property
    def delta_bits(self) -> int:
        """``log2`` of the scaling factor ``2^(q-K)`` that lifts masks above the noise."""
        return self.q - self.k

    @property
    def sigma(self) -> float:
        return math.sqrt(self.eta / 2)

    #: Low bits of the provider's output dropped on the wire (rounding error <= 2^(drop_bits-1)).
    DROP_BITS = 16
    #: The stored hint omits byte 0 of each word (decryption only): error <= 255 * sum|S| <= 255 * dim.
    HINT_DROP_ERROR = 255

    def noise_bound(self, max_row_l2: float) -> float:
        """High-probability bound on the decryption error: ``|W8 e|`` for weight rows of Euclidean norm
        ``<= max_row_l2`` plus the download rounding."""
        return (self.tail_sigmas * self.sigma * max_row_l2 + 2.0 ** (self.DROP_BITS - 1)
                + self.HINT_DROP_ERROR * self.dim)

    def check_noise(self, max_row_l2: float, label: str = "") -> None:
        """Raise :class:`ConfigError` if decryption could fail for rows of this norm."""
        if self.noise_bound(max_row_l2) >= 2.0 ** (self.delta_bits - 1):
            raise ConfigError(
                f"HE mask noise budget exceeded{(' for ' + label) if label else ''}: |W e| bound "
                f"2^{math.log2(self.noise_bound(max_row_l2)):.1f} >= 2^{self.delta_bits - 1}; use a smaller ring K"
            )


def matmul_mod64(small: np.ndarray, big: np.ndarray, *, small_left: bool = True, fast: bool | None = None
                 ) -> np.ndarray:
    """Exact ``small @ big`` (or ``big @ small``) mod ``2^64``.

    Uses int8 GEMMs on byte planes (:mod:`pgw.hemask.fastmm`) when the CPU has fast exact int8 GEMMs
    and ``small`` fits in int8; ``fast=False`` forces the float64 bit-slice reference path.

    ``small`` holds small signed integers (int8 weights or a ternary secret matrix), ``big`` holds
    arbitrary ``uint64`` values. ``big`` is split into bit slices narrow enough that every float64
    partial sum stays below ``2^53``, so each slice GEMM is exact; the slices are recombined with
    wrapping ``uint64`` shifts and adds (two's complement wrap is reduction mod 2^64).
    """
    if fast is None:
        fast = _fast_ok(small)
    if fast:
        from pgw.hemask.fastmm import big_planes_mm_small, small_mm_big, to_planes

        s8 = np.asarray(small).astype(np.int8)
        if small_left:
            return small_mm_big(s8, np.asarray(big, dtype=_U64))
        return big_planes_mm_small(to_planes(np.asarray(big, dtype=_U64)), s8)
    small_f = np.asarray(small, dtype=np.float64)
    big = np.ascontiguousarray(big, dtype=_U64)
    inner = small_f.shape[1] if small_left else small_f.shape[0]
    if (big.shape[0] if small_left else big.shape[1]) != inner:
        raise ConfigError(f"shape mismatch: {small_f.shape} vs {big.shape}")
    max_abs = float(np.abs(small_f).max()) if small_f.size else 0.0
    if max_abs == 0.0:
        shape = (small_f.shape[0], big.shape[1]) if small_left else (big.shape[0], small_f.shape[1])
        return np.zeros(shape, dtype=_U64)
    bits = int(_F53 - 1 - math.ceil(math.log2(max_abs * inner)))
    if bits < 8:
        raise ConfigError(f"operands too large for exact float64 slicing (max |small| {max_abs}, inner {inner})")
    bits = min(bits, 32)
    out: np.ndarray | None = None
    for shift in range(0, 64, bits):
        part = ((big >> _U64(shift)) & _U64((1 << bits) - 1)).astype(np.float64)
        z = small_f @ part if small_left else part @ small_f
        term = np.rint(z).astype(np.int64).astype(_U64) << _U64(shift)
        out = term if out is None else out + term
    assert out is not None
    return out


def _fast_ok(small: np.ndarray) -> bool:
    from pgw.hemask.fastmm import engine

    if small.size == 0 or engine() == "cpu_f64":
        return False
    if np.asarray(small).dtype == np.int8:
        return True
    return bool(np.abs(np.asarray(small)).max() <= 127)


def public_matrix(seed: bytes, rows: int, params: HEParams) -> np.ndarray:
    """Public uniform matrix ``A`` (``rows x dim`` uint64) expanded from a customer-chosen seed.

    Row ``c`` is the module element ``(a_{c,0}, ..., a_{c,rank-1})`` (coefficients concatenated). A
    linear with ``n`` inputs uses the first ``n`` rows. The seed must be chosen by the gateway (a
    provider-chosen ``A``, e.g. all zeros, would expose the masks).
    """
    if len(seed) != 32:
        raise ConfigError("public matrix seed must be 32 bytes")
    raw = SeedExpander(seed).random_bytes(rows * params.dim * 8, "pgw/hemask/v1/A", params.rank)
    return np.frombuffer(raw, dtype="<u8").reshape(rows, params.dim).astype(_U64, copy=True)


def secret_matrix(s: np.ndarray) -> np.ndarray:
    """Block matrix ``S`` (``rank*256 x 256``, int8) with ``a @ S == sum_l a_l * s_l`` in ``Z[X]/(X^256+1)``."""
    rank, deg = s.shape
    if deg != DEGREE:
        raise ConfigError(f"secret polynomials must have degree {DEGREE}")
    idx = np.arange(DEGREE)
    diff = idx[None, :] - idx[:, None]  # j - t
    wrap = diff < 0
    blocks = []
    for li in range(rank):
        m = s[li][np.where(wrap, diff + DEGREE, diff)].astype(np.int8)
        m[wrap] = -m[wrap]
        blocks.append(m)
    return np.concatenate(blocks, axis=0)


def _cbd(sx: SeedExpander, shape: tuple[int, ...], eta: int, label: str) -> np.ndarray:
    """Centred binomial samples: popcount(eta random bits) - popcount(eta random bits).

    Python reference of ``pgw_fast.he_cbd`` (same stream, same values)."""
    n = int(np.prod(shape))
    if eta > 32:
        raise ConfigError("eta must be <= 32")
    raw = np.frombuffer(sx.random_bytes(8 * n, label), dtype="<u8")
    mask = np.uint64((1 << eta) - 1)
    pos = np.bitwise_count(raw & mask).astype(np.int64)
    neg = np.bitwise_count((raw >> np.uint64(eta)) & mask).astype(np.int64)
    return (pos - neg).reshape(shape)


def _ternary(sx: SeedExpander, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Uniform {-1, 0, 1} samples by rejection from random bytes (values 0..251 kept, mod 3).

    Python reference of ``pgw_fast.he_ternary`` (same streams, same values)."""
    n = int(np.prod(shape))
    out = np.empty(0, dtype=np.int64)
    ctr = 0
    while out.size < n:
        raw = np.frombuffer(sx.random_bytes(2 * (n - out.size) + 64, label, ctr), dtype=np.uint8)
        ctr += 1
        keep = raw[raw < 252].astype(np.int64) % 3 - 1
        out = np.concatenate([out, keep])
    return out[:n].reshape(shape)


def _native(*names: str) -> Any:
    """The ``pgw_fast`` module when it provides ``names`` (bit-identical native kernels), else ``None``."""
    from pgw.core.native import fast, native_available

    return fast if native_available(*names) else None


def sample_secret(seed: bytes, rank: int) -> np.ndarray:
    """Secret block matrix ``S`` (``rank*256 x 256`` int8) of the ternary secret drawn from ``seed``.

    Equal to ``secret_matrix(_ternary(SeedExpander(seed), (rank, 256), "s"))``; native when available.
    """
    nat = _native("he_secret_matrix")
    if nat is not None:
        out: np.ndarray = nat.he_secret_matrix(seed, rank)
        return out
    s = _ternary(SeedExpander(seed), (rank, DEGREE), "s")
    smat = secret_matrix(s)
    s.fill(0)
    return smat


def sample_noise(seed: bytes, rows: int, eta: int) -> np.ndarray:
    """CBD(``eta``) noise ``(rows, 256)`` int64 drawn from ``seed`` (``_cbd(..., "e")``); native when available."""
    nat = _native("he_cbd")
    if nat is not None:
        out: np.ndarray = nat.he_cbd(seed, rows, DEGREE, eta)
        return out
    return _cbd(SeedExpander(seed), (rows, DEGREE), eta, "e")


@dataclass
class MaskCiphertext:
    """One encrypted group: ``b`` (``n x 256`` uint64) goes to the provider; ``secret`` stays local."""

    b: np.ndarray
    secret: np.ndarray  # int8 block matrix S (dim x 256)
    rows: int  # real token slots in the group (<= 256)

    def wipe(self) -> None:
        self.secret.fill(0)


def a_plane_count(params: HEParams) -> int:
    """Byte planes of the public matrix that encryption needs: ``ceil(q / 8)``.

    Plane ``d`` only contributes ``2^(8d)`` multiples to the pad, so for ``q = 2^56`` byte 7 vanishes
    in the final ``mod 2^q`` and 7 planes give exactly the same ciphertexts as 8.
    """
    return -(-params.q // 8)


#: int32 GEMM output kept per block of the fused native encryption (bounds temporary memory).
_ENC_BLOCK_BYTES = 32 << 20


def encrypt_masks(r: np.ndarray, a: np.ndarray | None, params: HEParams, *,
                  a_planes: np.ndarray | None = None, _seed: bytes | None = None) -> MaskCiphertext:
    """Encrypt the masks of up to 256 slots of one linear.

    Args:
        r: ``(rows, n)`` masks in ``[0, 2^K)`` (int64/uint64), ``rows <= 256``.
        a: Public matrix rows for this linear (``n x dim`` uint64), e.g. ``public_matrix(...)[:n]``.
        params: Encryption parameters.
        a_planes: The same rows as int8 byte planes (``n x P x dim``, :func:`pgw.hemask.fastmm.to_planes`,
            bytes ``0 .. P-1``); used instead of ``a`` (faster: int8 GEMM, no per-call conversion).
            ``P = 8``, or ``P = a_plane_count(params)`` (7 at ``q = 2^56``: same ciphertexts).
        _seed: Test hook only (private on purpose): fixes the randomness of the secret and the noise
            for reproducibility tests. A fixed seed reuses the MLWE secret and noise across calls,
            which exposes the masks; production always takes the default, a fresh ``os.urandom``
            seed per call that is never stored.
    """
    rows, n = r.shape
    if rows > DEGREE:
        raise ConfigError(f"at most {DEGREE} slots per ciphertext group, got {rows}")
    shape = a_planes.shape if a_planes is not None else (a.shape if a is not None else None)
    want: tuple[int, ...] = (n, params.dim)
    if a_planes is not None:
        npl = a_planes.shape[1] if a_planes.ndim == 3 else 0
        want = (n, npl if npl in (8, a_plane_count(params)) else 8, params.dim)
    if shape != want:
        raise ConfigError(f"public matrix shape {shape} != {want}")
    seed = new_seed() if _seed is None else _seed  # fresh secret and noise from the OS CSPRNG; never stored
    smat = sample_secret(seed, params.rank)
    if a_planes is not None:
        from pgw.hemask.fastmm import engine

        nat = _native("he_encrypt_combine") if engine() == "cpu_int8" else None
        if nat is not None:
            b = _encrypt_native(nat, r, a_planes, smat, seed, params)
            return MaskCiphertext(b=b, secret=smat, rows=rows)
    e = sample_noise(seed, n, params.eta)
    msg = np.zeros((n, DEGREE), dtype=_U64)
    msg[:, :rows] = np.asarray(r, dtype=np.int64).astype(_U64).T
    if a_planes is not None:
        from pgw.hemask.ntt import big_planes_mm_small  # exact NTT/CRT product; falls back to fastmm

        pad = big_planes_mm_small(a_planes, smat, first_byte=0)
    else:
        assert a is not None
        pad = matmul_mod64(smat, a, small_left=False)
    b = (pad + e.astype(_U64) + (msg << _U64(params.delta_bits))) & params.q_mask
    pad.fill(0)  # A s is as secret as the masks (b - A s = e + Delta r)
    e.fill(0)
    msg.fill(0)
    return MaskCiphertext(b=b, secret=smat, rows=rows)


def _encrypt_native(nat: Any, r: np.ndarray, a_planes: np.ndarray, smat: np.ndarray, seed: bytes,
                    params: HEParams) -> np.ndarray:
    """``(pad + e + (msg << delta)) & q_mask`` with the pad GEMM in ``torch._int_mm`` (AMX/VNNI) and
    the noise sampling, plane combine, mask transpose and reduction fused in ``pgw_fast``."""
    import torch

    n, npl, dim = a_planes.shape
    r64 = np.ascontiguousarray(r, dtype=np.int64)
    planes = np.ascontiguousarray(a_planes)
    rhs = torch.from_numpy(smat)  # (dim, 256) row-major: no transposed copy of the secret
    offset = (128 * smat.sum(0, dtype=np.int64)).astype(np.int32)
    b = np.empty((n, DEGREE), dtype=_U64)
    block = max(1, _ENC_BLOCK_BYTES // (npl * DEGREE * 4))
    try:
        for r0 in range(0, n, block):
            blk = planes[r0:r0 + block]
            nb = blk.shape[0]
            part = torch._int_mm(torch.from_numpy(blk.reshape(nb * npl, dim)), rhs).numpy()
            try:
                nat.he_encrypt_combine(part.reshape(nb, npl, DEGREE), offset, 0, seed, params.eta, r64, r0,
                                       params.delta_bits, params.q, b)
            finally:
                part.fill(0)  # A s is as secret as the masks (b - A s = e + Delta r)
    finally:
        offset.fill(0)  # 128 * column sums of S
    return b


def decrypt_wr(y: np.ndarray, hint: np.ndarray, ct: MaskCiphertext, params: HEParams) -> np.ndarray:
    """Recover ``W8 r mod 2^K`` (``rows x m`` int64) from the provider's ``y = W8 b mod 2^64``.

    ``hint`` is ``W8 A[:n] mod 2^64``, either ``m x dim`` uint64 or ``m x 8 x dim`` int8 planes.
    Wipes the ciphertext secret.
    """
    try:
        planes = hint.ndim == 3
        if hint.shape[-1] != params.dim or y.shape != (hint.shape[0], DEGREE):
            raise ConfigError(f"decrypt shapes do not match: y {y.shape}, hint {hint.shape}")
        if planes:
            from pgw.hemask.ntt import big_planes_mm_small  # exact NTT/CRT product; falls back to fastmm

            corr = big_planes_mm_small(np.asarray(hint), ct.secret)
        else:
            corr = matmul_mod64(ct.secret, np.asarray(hint, dtype=_U64), small_left=False)
        v = (np.asarray(y, dtype=_U64) - corr) & params.q_mask
        corr.fill(0)  # H s: with y it gives W8 r
        half = _U64(1 << (params.delta_bits - 1))
        m = ((v + half) >> _U64(params.delta_bits)) & _U64((1 << params.k) - 1)
        v.fill(0)
        out: np.ndarray = m[:, : ct.rows].T.astype(np.int64)
        m.fill(0)
        return out
    finally:
        ct.wipe()
