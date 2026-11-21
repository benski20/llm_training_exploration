"""Fast gateway-side encryption of one-time masks: the public matrix kept in NTT form.

The encrypted mask factory (:mod:`pgw.hemask.lwe`) already uses a *structured* public matrix: row
``c`` of ``A`` is a module element ``(a_{c,0}, ..., a_{c,rank-1})`` of ``rank`` polynomials in
``R = Z[X]/(X^256 + 1)``, and the pad of feature ``c`` is the module inner product
``sum_l a_{c,l} * s_l``. The default code path computes that pad as a *dense* product with the
secret's block-negacyclic matrix ``S`` (``dim x 256``): ``n x dim x 256`` MACs on each of 7 byte
planes per 256-slot group (about ``7.3e6 n``), which is what made gateway encryption expensive.

:class:`PretransformedEncryptor` uses the ring structure instead. ``A`` is public and fixed per hint
store, so its forward NTTs (three ~2^29 CRT primes, native crate ``pgw_ntt``) are computed **once**
(:class:`pgw.hemask.ntt.TransformedOperand`, 12 bytes per coefficient). Per group and row the pad
then costs ``rank`` pointwise multiply-accumulates per coefficient and prime plus one length-256
inverse NTT, about ``3 x (4096 + 1024)`` modular operations instead of ``7.3e6`` int8 MACs, and the
secret transforms are shared across all rows. Several groups go through one pass over ``A``.

Nothing cryptographic changes: same public matrix, same secret and noise distributions and
samplers (:func:`pgw.hemask.lwe.sample_secret`, :func:`pgw.hemask.lwe.sample_noise`), same
ciphertext layout. For the same per-group seed the ciphertext is **bit-identical** to
:func:`pgw.hemask.lwe.encrypt_masks` (``tests/hemask/test_mlwe_ring.py``), so the provider, the wire,
the hint and decryption are untouched and the security argument is the existing one. The NTT
product is exact by the CRT bound documented in :mod:`pgw.hemask.ntt` (checked on every call).

The module also carries the parameter *profiles* of the encrypted factory (:data:`PROFILES`): the
default rank-16 profile and the opt-in reduced-rank profile ``mlwe-r12`` (LWE dimension 3072). The
latter **is** a cryptographic parameter change (smaller lattice dimension, same assumption family)
and is refused unless the operator sets the acknowledgement string :data:`R12_ACK`; see
``research/plans/he-structured-encryption.md`` for the security estimate and what a cryptographer
must review. Ranks below 12 are refused by :func:`check_rank`.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from pgw.core.errors import ConfigError
from pgw.core.prng import new_seed
from pgw.hemask.lwe import DEGREE, HEParams, MaskCiphertext, sample_noise, sample_secret

__all__ = [
    "DEFAULT_RANK",
    "PROFILES",
    "R12_ACK",
    "HEProfile",
    "PretransformedEncryptor",
    "check_rank",
    "core_svp_beta",
    "encrypt_cost_ops",
]

_U64 = np.uint64
#: Ciphertext rows combined per task (keeps one block of every group in L2).
_COMBINE_ROWS = 256
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1)),
                                       thread_name_prefix="pgw-hemask-combine")
        return _POOL

# ---------------------------------------------------------------------------------------------
# Parameter profiles and the acknowledgement gate
# ---------------------------------------------------------------------------------------------

#: Rank of the reviewed default profile (LWE dimension 4096).
DEFAULT_RANK = 16

#: Acknowledgement an operator must set (``PGW_FACTORY_HE_PARAMS_ACK``) to use the reduced-rank
#: profile. Its wording is deliberately explicit: the profile has not had a human cryptographic review.
R12_ACK = "i-accept-unreviewed-he-params:mlwe-r12"


@dataclass(frozen=True)
class HEProfile:
    """One parameter set of the encrypted factory.

    Attributes:
        name: Profile name.
        rank: Module rank (LWE dimension ``256 * rank``).
        reviewed: ``True`` for the shipped default; ``False`` for opt-in profiles awaiting review.
        ack: Acknowledgement string required to enable an unreviewed profile (``""`` for reviewed ones).
    """

    name: str
    rank: int
    reviewed: bool
    ack: str = ""

    @property
    def dim(self) -> int:
        return self.rank * DEGREE


#: Known profiles, by rank.
PROFILES: dict[int, HEProfile] = {
    16: HEProfile("mlwe-r16", 16, reviewed=True),
    12: HEProfile("mlwe-r12", 12, reviewed=False, ack=R12_ACK),
}


def check_rank(rank: int, ack: str = "") -> HEProfile:
    """The profile of ``rank``, or :class:`ConfigError` when it is refused.

    * rank 16 (default) and larger: always allowed (larger ranks only add security);
    * rank 12: only with ``ack == R12_ACK`` (estimated core-SVP ~163 bits classical at ``q = 2^56``,
      see the memo; unreviewed);
    * any other rank below 16: refused (rank 8 at ``q = 2^56`` is below 128-bit security under the
      HomomorphicEncryption.org table, ranks 9-11 and 13-15 are not analysed).
    """
    if rank >= DEFAULT_RANK:
        return PROFILES.get(rank, HEProfile(f"mlwe-r{rank}", rank, reviewed=True))
    prof = PROFILES.get(rank)
    if prof is None:
        raise ConfigError(
            f"HE rank {rank} is refused: only rank >= 16 (reviewed) or the opt-in rank-12 profile are allowed "
            "(see research/plans/he-structured-encryption.md)"
        )
    if (ack or "").strip() != prof.ack:
        raise ConfigError(
            f"HE profile {prof.name} (rank {rank}, LWE dimension {prof.dim}) changes the lattice parameters and "
            f"has not had a cryptographic review; set he_params_ack={prof.ack!r} to opt in"
        )
    return prof


def core_svp_beta(n: int, log_q: float, sigma_e: float, sigma_s: float = math.sqrt(2 / 3)) -> int:
    """Smallest BKZ block size ``beta`` for which the primal uSVP attack succeeds (2016 estimate).

    The "core-SVP" methodology of NewHope/Kyber: an LWE instance of dimension ``n``, modulus ``2^log_q``,
    error standard deviation ``sigma_e`` and secret standard deviation ``sigma_s`` (``sqrt(2/3)`` for
    uniform ternary), embedded with the secret rescaled by ``nu = sigma_e / sigma_s`` (Bai-Galbraith), with
    any number of samples ``m`` (the provider sees ``n x 256`` samples per secret, far more than the
    optimum). The attack succeeds when ``sigma_e sqrt(beta) <= delta^(2 beta - d - 1) Vol^(1/d)``. The
    classical cost is ``2^(0.292 beta)`` (quantum ``2^(0.265 beta)``); this is a lower bound used to
    *compare* parameter sets, not a replacement for the lattice estimator.
    """
    nu = sigma_e / sigma_s
    ln_q = log_q * math.log(2)
    for beta in range(50, 4000):
        ld = math.log(((math.pi * beta) ** (1 / beta) * beta / (2 * math.pi * math.e)) ** (1 / (2 * (beta - 1))))
        for m in range(max(1, n // 4), 3 * n, 8):
            d = m + n + 1
            lv = (m * ln_q + n * math.log(nu)) / d
            if math.log(sigma_e) + 0.5 * math.log(beta) <= (2 * beta - d - 1) * ld + lv:
                return beta
    return 4000


def encrypt_cost_ops(n: int, rank: int, groups: int = 1, *, planes: int = 7) -> dict[str, float]:
    """Operation counts of one linear's encryption (``n`` inputs, ``groups`` 256-slot groups).

    ``dense``: int8 MACs of the byte-plane product ``A[:n] @ S``. ``ntt_pre``: modular multiplications
    with ``A`` pre-transformed (pointwise multiply-accumulate plus inverse NTT butterflies, 3 primes).
    """
    dense = float(n) * planes * rank * DEGREE * DEGREE * groups
    inv_ntt = DEGREE // 2 * int(math.log2(DEGREE))
    pre = 3.0 * n * groups * (rank * DEGREE + inv_ntt)
    return {"dense": dense, "ntt_pre": pre, "ratio": dense / pre}


# ---------------------------------------------------------------------------------------------
# Exact fast encryption
# ---------------------------------------------------------------------------------------------


class PretransformedEncryptor:
    """Encrypt mask groups with the public matrix held in NTT form; same ciphertexts as
    :func:`pgw.hemask.lwe.encrypt_masks`.

    Args:
        a: Public matrix ``(n_max, dim)`` uint64 (:func:`pgw.hemask.lwe.public_matrix`); only its
            NTTs are kept (``12 * n_max * dim`` bytes, rows padded to 16).
        params: Encryption parameters (``a.shape[1] == params.dim``).
        groups_per_pass: Groups whose pads are computed in one pass over ``A`` (the transformed rows
            are streamed once per pass; 4 is a good default on CPUs with 1-2 MB of L2 per core).
    """

    def __init__(self, a: np.ndarray, params: HEParams, *, groups_per_pass: int = 4):
        from pgw.hemask import ntt

        if not ntt.available():
            raise ConfigError("he_encrypt='ntt_pre' needs the native NTT module pgw_ntt "
                              "(build it with python rust/tools/build_ntt.py)")
        if a.ndim != 2 or a.shape[1] != params.dim:
            raise ConfigError(f"public matrix shape {a.shape} does not match dimension {params.dim}")
        self.params = params
        self.n_max = int(a.shape[0])
        self.groups_per_pass = max(1, min(16, int(groups_per_pass)))
        # The low q bits of A determine the ciphertext (b is reduced mod 2^q); masking first keeps the
        # CRT bound for 56-bit operands (the native transform derives its bound from big_bits).
        if params.q < 64:
            from pgw.hemask.fastmm import to_planes

            npl = -(-params.q // 8)
            planes = np.empty((a.shape[0], npl, a.shape[1]), dtype=np.int8)
            for r0 in range(0, a.shape[0], 1024):
                planes[r0:r0 + 1024] = to_planes(a[r0:r0 + 1024])[:, :npl]
            self._t = ntt.TransformedOperand.from_planes(planes, first_byte=0)
            del planes
        else:
            self._t = ntt.TransformedOperand.from_u64(a)

    @property
    def nbytes(self) -> int:
        """Memory of the transformed public matrix."""
        return self._t.nbytes

    def encrypt(self, r: np.ndarray, *, _seeds: list[bytes] | None = None) -> list[MaskCiphertext]:
        """Encrypt ``(rows, n)`` masks (any number of rows) as ``ceil(rows / 256)`` groups.

        ``_seeds`` is a test hook only (one per group, as ``encrypt_masks(_seed=...)``): production
        always draws a fresh OS-CSPRNG seed per group that is never stored.
        """
        r = np.asarray(r)
        rows, n = r.shape
        if n > self.n_max:
            raise ConfigError(f"linear has {n} inputs but the public matrix has {self.n_max} rows")
        groups = -(-rows // DEGREE)
        if _seeds is not None and len(_seeds) != groups:
            raise ConfigError(f"need {groups} seeds, got {len(_seeds)}")
        out: list[MaskCiphertext] = []
        g = self.groups_per_pass
        for p0 in range(0, groups, g):
            p1 = min(groups, p0 + g)
            seeds = list(_seeds[p0:p1]) if _seeds is not None else [new_seed() for _ in range(p0, p1)]
            out.extend(self._encrypt_pass(r[p0 * DEGREE:min(rows, p1 * DEGREE)], n, seeds))
            for i in range(len(seeds)):
                seeds[i] = b""
        return out

    def _encrypt_pass(self, r: np.ndarray, n: int, seeds: list[bytes]) -> list[MaskCiphertext]:
        from pgw.hemask import ntt

        p = self.params
        smats = [sample_secret(s, p.rank) for s in seeds]
        cts: list[MaskCiphertext] = []
        try:
            secs = np.empty((len(seeds), p.rank, DEGREE), dtype=np.int8)
            try:
                for gi, sm in enumerate(smats):
                    sec = ntt.secrets_of(sm)
                    if sec is None:  # pragma: no cover - native module present by construction
                        raise ConfigError("secret matrix is not block-negacyclic")
                    secs[gi] = sec[0]
                    sec.fill(0)
                pad = self._t.mm_secrets(secs, rows=n)  # (n, G*256) uint64, A s mod 2^64
            finally:
                secs.fill(0)
            try:
                # The pad becomes the ciphertexts in place: b_g = pad[:, g] + e_g + (r_g^T << delta),
                # combined in blocks of rows on a few threads (numpy releases the GIL).
                pad3 = pad.reshape(n, len(seeds), DEGREE)
                delta = _U64(p.delta_bits)
                noise = [sample_noise(seed, n, p.eta) for seed in seeds]  # native, multi-threaded
                rgs = [np.ascontiguousarray(r[gi * DEGREE:(gi + 1) * DEGREE], dtype=np.int64)
                       for gi in range(len(seeds))]
                try:
                    def combine(c0: int) -> None:
                        c1 = min(n, c0 + _COMBINE_ROWS)
                        for gi, (e, rg) in enumerate(zip(noise, rgs, strict=True)):
                            v = pad3[c0:c1, gi, :]
                            np.add(v, e[c0:c1].view(_U64), out=v)  # int64 noise, two's complement wrap
                            msg = rg[:, c0:c1].view(_U64) << delta
                            vr = v[:, :rg.shape[0]]
                            np.add(vr, msg.T, out=vr)
                            msg.fill(0)
                        np.bitwise_and(pad[c0:c1], p.q_mask, out=pad[c0:c1])

                    starts = range(0, n, _COMBINE_ROWS)
                    if len(starts) > 1:
                        list(_pool().map(combine, starts))
                    else:
                        combine(0)
                finally:
                    for e in noise:
                        e.fill(0)
                    for rg in rgs:
                        if not np.may_share_memory(rg, r):  # only a converted copy, never the caller's masks
                            rg.fill(0)
                cts = [MaskCiphertext(b=pad3[:, gi, :], secret=smats[gi],
                                      rows=min(DEGREE, r.shape[0] - gi * DEGREE)) for gi in range(len(seeds))]
            except BaseException:
                pad.fill(0)  # A s is as secret as the masks (b - A s = e + Delta r)
                raise
        except BaseException:
            for sm in smats:
                sm.fill(0)
            raise
        return cts
