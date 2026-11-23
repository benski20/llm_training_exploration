"""Compact-hint encrypted masks with packed key-switched replies: scheme ``pks-v3`` (numpy reference).

Opt-in and off by default (``PGW_FACTORY_HE_SCHEME=pks-v3`` plus :data:`PKS_V3_ACK` until the external
review signs off, ``PGW_SERVICE_MASKGEN_V3`` on the provider). Scheme, parameters, noise and hint integrity:
``research/plans/he-compact-hint-pks.md`` (section 7: production parameters) and the review packet
``research/reviews/pks-v3-review-packet.md``. The fast kernels are ``rust/pgw-pks`` (``pgw_pks``, wrapped by
:mod:`pgw.hemask.pks_native`); every function here is their bit-exact reference.

Why: the v1/v2 decryption needs the hint ``H = W8 A`` (4096 coefficients per output row: 34 GB for
Qwen2.5-7B, 169 GB for 72B) and 16 NTT products per output and slot. The provider, which holds ``W8``
and computes ``H`` itself (it is public), instead

1. modulus-switches its reply ``y = W8 b mod 2^56`` and ``H`` to ``q1 = p0 p1`` (~2^48),
2. packs ``PACK = 32`` output rows of one 256-slot group into one element of the reply ring
   ``R_r = Z[Z]/(Z^8192 + 1)`` (row ``j``, slot ``u`` -> coefficient ``32 u + j``; the degree-256 ring embeds
   as ``X = Z^32``), and
3. key-switches the packed ciphertext from the upload secret (16 components of degree 256) to a fresh
   reply secret ``t'`` of degree 8192 with the key-switching key the customer sends with the request
   (hybrid key switching, one digit, special modulus ``P = p2 p3`` ~2^60; ``log2(q1 P) = 108``).

The packed ciphertext's ``c1`` depends only on public data (``W8``, ``A``, the seeded key-switching
``a``-parts ``k``), so it is a fixed per-linear *compact hint*: 8192 coefficients per 32 rows = 256 per
output row, stored in NTT form as two 24-bit residues (1.5 KB per row: 2.1 GB for 7B). The provider sends
``c0`` rounded to ``yb = 36`` bits; the customer decrypts with one pointwise product and one inverse NTT per
prime and 32-row block: ``v = c0 + c1 t' mod q1``, ``W8 r mod 2^32 = round(2^32 v / q1)``.

Upload: 16 features x 256 slots are ONE ring-LWE sample over ``R_u = Z[Y]/(Y^4096 + 1)`` mod ``2^56``
(``B_C = a_C t + e_C + 2^24 M_C``) sent in today's ``(n, 256)`` layout (``b[16 C + k, u] = B_C[16 u + k]``):
in the provider's view feature ``16 C + k`` is a module-LWE sample under the components ``t_l`` of ``t``
with ``a``-part row ``k`` of the multiplication matrix of ``a_C`` (:func:`a_mlwe_rows`), so the provider's
GEMM is unchanged. ``a_C`` is derived per linear from the customer's public seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import numpy as np

from pgw.core.errors import ConfigError
from pgw.core.prng import SeedExpander, new_seed
from pgw.hemask.lwe import _cbd, _ternary

__all__ = [
    "NR",
    "NU",
    "PACK",
    "PKS_V3_ACK",
    "RANK",
    "SCHEME_ID",
    "D",
    "Ntt",
    "PKSParams",
    "ReplyKey",
    "a_mlwe_rows",
    "compact_hint",
    "decrypt_pks",
    "encrypt_rlwe",
    "ks_a",
    "make_reply_key",
    "negacyclic_mul_mod2",
    "provider_reply",
    "public_ring_rows",
    "switch_hint",
]

SCHEME_ID = "pks-v3"
#: Acknowledgement string (``PGW_FACTORY_HE_PKS_ACK``) required until the external cryptographic review of
#: ``research/reviews/pks-v3-review-packet.md`` signs off.
PKS_V3_ACK = "pks-v3-unreviewed-n8192"
D = 256  # module degree seen by the provider = slots per group
RANK = 16  # upload module rank (provider view)
NU = RANK * D  # upload ring degree (4096)
NR = 8192  # reply ring degree (production; core-SVP 251 classical / 228 quantum at log2(q1 P) = 108, HE-std limit 218)
PACK = NR // D  # 32 output rows per reply ciphertext
_U64 = np.uint64
_M56 = _U64((1 << 56) - 1)


# ------------------------------------------------------------------------------------------------
# Primes and NTTs
# ------------------------------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
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


@cache
def ntt_primes(bits: int, count: int, step: int = 2 * NR) -> tuple[int, ...]:
    """The ``count`` largest primes below ``2^bits`` with ``p = 1 mod step`` (NTT-friendly for degree ``step/2``)."""
    out: list[int] = []
    c = ((1 << bits) - 1) // step * step + 1
    while len(out) < count:
        if c < (1 << bits) and _is_prime(c):
            out.append(c)
        c -= step
    return tuple(out)


#: Reply modulus q1 = Q1[0] * Q1[1] (~2^48). Customer-side decryption works modulo these two primes.
Q1 = ntt_primes(24, 2)
#: Special modulus P = PS[0] * PS[1] (~2^60) of the hybrid key switching.
PS = ntt_primes(30, 2)
#: Key-switching RNS (q1 P, ~2^108).
QP = Q1 + PS
#: Primes for exact products of 56-bit and ternary polynomials over R_u (upload pad): range 2^92 > 2^69.
EP = ntt_primes(31, 3, 2 * NU)


def _primitive_root(p: int) -> int:
    n, fac, f = p - 1, [], 2
    while f * f <= n:
        if n % f == 0:
            fac.append(f)
            while n % f == 0:
                n //= f
        f += 1
    if n > 1:
        fac.append(n)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fac):
            return g
    raise AssertionError("no generator")


def _brv(x: int, bits: int) -> int:
    return int(f"{x:0{bits}b}"[::-1], 2)




class Ntt:
    """Negacyclic length-``n`` NTT modulo one prime (Cooley-Tukey forward to bit-reversed evaluation
    order, Gentleman-Sande inverse). Vectorised over leading axes. Reference speed only."""

    _cache: ClassVar[dict[tuple[int, int], Ntt]] = {}

    def __init__(self, p: int, n: int = NR):
        self.p, self.n = p, n
        lg = n.bit_length() - 1
        psi = pow(_primitive_root(p), (p - 1) // (2 * n), p)
        assert pow(psi, n, p) == p - 1
        z = [pow(psi, _brv(k, lg), p) for k in range(n)]
        self.z = np.array(z, dtype=_U64)
        self.zi = np.array([pow(v, p - 2, p) for v in z], dtype=_U64)
        self.ninv = _U64(pow(n, p - 2, p))

    @classmethod
    def of(cls, p: int, n: int = NR) -> Ntt:
        if (p, n) not in cls._cache:
            cls._cache[(p, n)] = cls(p, n)
        return cls._cache[(p, n)]

    def fwd(self, a: np.ndarray) -> np.ndarray:
        p = _U64(self.p)
        lead = a.shape[:-1]
        n = self.n
        x = np.array(a, dtype=_U64, copy=True).reshape(-1, n)
        length, k = n // 2, 1
        while length >= 1:
            nb = n // (2 * length)
            v = x.reshape(-1, nb, 2, length)
            t = (v[:, :, 1, :] * self.z[k:k + nb, None]) % p
            u = v[:, :, 0, :].copy()
            v[:, :, 0, :] = (u + t) % p
            v[:, :, 1, :] = (u + p - t) % p
            k += nb
            length //= 2
        return x.reshape(*lead, n)

    def inv(self, a: np.ndarray) -> np.ndarray:
        p = _U64(self.p)
        lead = a.shape[:-1]
        n = self.n
        x = np.array(a, dtype=_U64, copy=True).reshape(-1, n)
        length = 1
        while length < n:
            nb = n // (2 * length)
            v = x.reshape(-1, nb, 2, length)
            u, w = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
            v[:, :, 0, :] = (u + w) % p
            v[:, :, 1, :] = ((u + p - w) % p * self.zi[nb:2 * nb, None]) % p
            length *= 2
        return ((x * self.ninv) % p).reshape(*lead, n)


def _residues(x_signed: np.ndarray, p: int) -> np.ndarray:
    """Residues mod ``p`` of signed int64 values."""
    return np.mod(np.asarray(x_signed, dtype=np.int64), p).astype(_U64)


def negacyclic_mul_mod2(a: np.ndarray, t: np.ndarray, bits: int = 56) -> np.ndarray:
    """Exact ``a * t mod (Y^NU + 1, 2^bits)`` for ``a < 2^56`` (uint64, ``(..., NU)``) and ternary ``t``
    (``(NU,)`` int): three-prime NTT products and a centred Garner CRT (``|a t| < 2^69 << E/2``)."""
    res = []
    for p in EP:
        nt = Ntt.of(p, NU)
        res.append(nt.inv(nt.fwd(np.asarray(a, dtype=_U64) % _U64(p)) * nt.fwd(_residues(t, p)) % _U64(p)))
    return _crt3_mod2(res, bits)


def _crt3_mod2(res: list[np.ndarray], bits: int) -> np.ndarray:
    p0, p1, p2 = (_U64(p) for p in EP)
    r0, r1, r2 = res
    inv01 = _U64(pow(EP[0], -1, EP[1]))
    inv012 = _U64(pow(EP[0] * EP[1], -1, EP[2]))
    k1 = ((r1 + p1 - r0 % p1) % p1) * inv01 % p1
    base = (r0 % p2 + (p0 % p2) * k1 % p2) % p2
    k2 = ((r2 + p2 - base) % p2) * inv012 % p2
    x = r0 + p0 * k1 + _U64((EP[0] * EP[1]) % (1 << 64)) * k2  # wraps mod 2^64
    neg = k2 > (p2 // _U64(2))
    x = np.where(neg, x - _U64((EP[0] * EP[1] * EP[2]) % (1 << 64)), x)
    return x & _U64((1 << bits) - 1)


# ------------------------------------------------------------------------------------------------
# Exact scaled rounding helpers (all inputs < 2^56, scalars < 2^60)
# ------------------------------------------------------------------------------------------------

def round_mul_shift(x: np.ndarray, num: int, shift: int) -> np.ndarray:
    """``floor((x * num + 2^(shift-1)) / 2^shift)`` exactly, for ``x < 2^56``, ``num < 2^48``, ``1 <= shift <= 56``
    (the result must fit in 64 bits). Base-2^14 digits of ``x`` keep every partial product below 2^62."""
    if not (num < (1 << 48) and 1 <= shift <= 56):
        raise ConfigError("round_mul_shift: unsupported operand sizes")
    x = np.asarray(x, dtype=_U64)
    m14 = _U64((1 << 14) - 1)
    nm = _U64(num)
    t = [((x >> _U64(14 * i)) & m14) * nm for i in range(4)]
    i0, b0 = divmod(shift - 1, 14)
    t[i0] = t[i0] + _U64(1 << b0)  # + 2^(shift-1)
    s = t[0]
    low = s & m14
    for i in (1, 2):
        s = (s >> _U64(14)) + t[i]
        low = low | ((s & m14) << _U64(14 * i))
    s3 = (s >> _U64(14)) + t[3]  # T = s3 2^42 + low, low < 2^42
    if shift >= 42:
        return s3 >> _U64(shift - 42)
    return (s3 << _U64(42 - shift)) + (low >> _U64(shift))


def round_pow2_div(x: np.ndarray, bits: int, q: int) -> np.ndarray:
    """``floor((x * 2^bits + floor(q/2)) / q)`` exactly for ``x < q < 2^48`` (long division, 16-bit digits)."""
    x = np.asarray(x, dtype=_U64)
    qq = _U64(q)
    quo = np.zeros_like(x)
    rem = x.copy()
    left = bits
    while left > 0:
        s = min(16, left)
        left -= s
        a = rem << _U64(s)
        if left == 0:
            a = a + _U64(q // 2)
        quo = (quo << _U64(s)) + a // qq
        rem = a % qq
    return quo


# ------------------------------------------------------------------------------------------------
# Parameters and noise
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PKSParams:
    """``k``: mask ring bits (32). ``q``: upload modulus bits (56, unchanged wire). ``yb``: reply bits.
    ``tail``: tail bound in standard deviations (Gaussian terms) / Hoeffding ``t`` (rounding sums), as SPEC 7.9."""

    k: int = 32
    q: int = 56
    eta: int = 21
    yb: int = 36
    tail: float = 24.0

    def __post_init__(self) -> None:
        if self.k != 32 or self.q != 56 or self.yb != 36:
            raise ConfigError("pks-v3 covers K32 rings, q = 2^56 uploads and 36-bit replies only")

    @property
    def q1(self) -> int:
        return Q1[0] * Q1[1]

    @property
    def pmod(self) -> int:
        return PS[0] * PS[1]

    @property
    def qks_bits(self) -> float:
        """log2 of the key-switching modulus ``q1 P`` (RLWE over ``R_r``; HE-standard 128-bit limit 218 at 8192)."""
        return math.log2(self.q1 * self.pmod)

    @property
    def sigma(self) -> float:
        return math.sqrt(self.eta / 2)

    def noise_terms(self, max_row_l2: float) -> dict[str, float]:
        """High-probability bounds of the reply's decryption error (units of 1 mod q1); budget ``q1 / 2^(K+1)``.

        Gaussian-like terms at ``tail`` standard deviations, sums of independent bounded rounding errors by
        Hoeffding (``P(|S| > tail c sqrt(N)) <= 2 exp(-tail^2 / 2)``, 2^-414 at 24), worst cases otherwise.
        """
        sc = self.q1 / 2.0 ** self.q
        t = self.tail
        return {
            # sum_k w_k e_k: independent CBD(eta), sd sigma ||w||, scaled to q1
            "W8 e (upload noise)": t * self.sigma * max_row_l2 * sc,
            # body = round(y q1 / 2^56): |err| <= 1/2
            "mod switch of y": 0.5,
            # (round(H q1/2^56) - H q1/2^56) . s: 4096 terms in [-1/2, 1/2] times ternary s
            "mod switch of H times s (Hoeffding)": t * 0.5 * math.sqrt(NU),
            # sum_l A~_l e_l / P: 16 x 8192 terms |A~| <= q1/2, |e| <= eta (worst-case magnitude, Gaussian count)
            "key switch sum A~ e / P": t * self.eta * (self.q1 / 2) * math.sqrt(RANK * NR) / self.pmod,
            # ModDown roundings of c0 (1/2) and of c1 (1/2 per coefficient, times t': 8192 terms)
            "ModDown of c0": 0.5,
            "ModDown of c1 times t' (Hoeffding)": t * 0.5 * math.sqrt(NR),
            # c0 rounded to yb bits: q1 / 2^(yb+1), and the decryptor's own rounding of c0 back to q1
            "reply rounding": self.q1 / 2.0 ** (self.yb + 1) + 0.5,
        }

    def noise_bound(self, max_row_l2: float) -> float:
        return sum(self.noise_terms(max_row_l2).values())

    def budget(self, margin_bits: float = 0.0) -> float:
        return self.q1 / 2.0 ** (self.k + 1 + margin_bits)

    def noise_ok(self, max_row_l2: float, margin_bits: float = 1.0) -> bool:
        return self.noise_bound(max_row_l2) < self.budget(margin_bits)

    def check_noise(self, max_row_l2: float, label: str = "", margin_bits: float = 1.0) -> None:
        if not self.noise_ok(max_row_l2, margin_bits):
            raise ConfigError(f"pks-v3 noise budget exceeded{(' for ' + label) if label else ''}: bound "
                              f"2^{math.log2(self.noise_bound(max_row_l2)):.2f} >= 2^{math.log2(self.budget(margin_bits)):.2f}")


# ------------------------------------------------------------------------------------------------
# Upload: ring-LWE over R_u, provider sees today's (n x 256) module-LWE bodies
# ------------------------------------------------------------------------------------------------

A_LABEL = "pgw/hemask/pks-v3/A"
KS_LABEL = "pgw/hemask/pks-v3/ksk-a"


def public_ring_rows(seed: bytes, n: int, linear: str) -> np.ndarray:
    """Public ring elements ``a_C`` of one linear (``n/16 x 4096``, uniform mod 2^56), derived from the
    customer's public seed and the linear id (the stream :func:`pgw.core.prng.SeedExpander.stream_key`
    ``(A_LABEL, linear)`` that the native kernel reads)."""
    if n % RANK:
        raise ConfigError(f"in_features {n} is not a multiple of {RANK}")
    raw = SeedExpander(seed).random_bytes((n // RANK) * NU * 8, A_LABEL, linear)
    return np.frombuffer(raw, dtype="<u8").reshape(n // RANK, NU) & _M56


def _neg_shift(p: np.ndarray, bits: int = 56) -> np.ndarray:
    """``X * p`` in ``Z_2^bits[X]/(X^256 + 1)`` (last axis)."""
    out = np.roll(p, 1, axis=-1)
    out[..., 0] = (_U64(0) - out[..., 0]) & _U64((1 << bits) - 1)
    return out


def a_mlwe_rows(a_ring: np.ndarray) -> np.ndarray:
    """The provider-view module ``a``-parts (``n x 4096``: 16 components of degree 256 per feature).

    Feature ``16 C + k`` has component ``l`` equal to entry ``(k, l)`` of the multiplication matrix of
    ``a_C`` over ``R_256`` (basis ``1, Y, .., Y^15``, ``Y^16 = X``): ``a_{C,k-l}`` for ``l <= k`` and
    ``X a_{C,k-l+16}`` otherwise. ``H = W8 @ a_mlwe_rows(a) mod 2^56`` is the hint of this upload.
    """
    nb = a_ring.shape[0]
    comps = a_ring.reshape(nb, D, RANK).transpose(0, 2, 1)  # (nb, 16, 256): comps[:, k] = a_k
    xcomps = _neg_shift(comps)
    out = np.empty((nb, RANK, RANK, D), dtype=_U64)
    for k in range(RANK):
        for li in range(RANK):
            out[:, k, li] = comps[:, k - li] if li <= k else xcomps[:, k - li + RANK]
    return out.reshape(nb * RANK, RANK * D)


@dataclass
class UploadCiphertext:
    b: np.ndarray  # (n, 256) uint64 mod 2^56: today's wire layout
    t: np.ndarray  # (4096,) int64 ternary upload secret (customer only)
    rows: int

    @property
    def t_components(self) -> np.ndarray:
        """``(16, 256)``: ``t_l[u] = t[16 u + l]`` (the module secret the provider's reply is under)."""
        return self.t.reshape(D, RANK).T.copy()


def encrypt_rlwe(r: np.ndarray, a_ring: np.ndarray, params: PKSParams, *, _seed: bytes | None = None
                 ) -> UploadCiphertext:
    """Encrypt the masks ``r`` (``rows x n``, ``rows <= 256``) of one linear and group.

    ``B_C = a_C t + e_C + 2^(q-K) M_C`` over ``R_u`` with ``M_C[16 u + k] = r[u, 16 C + k]``; returned as the
    ``(n, 256)`` bodies ``b[16 C + k, u] = B_C[16 u + k]``. Fresh ternary ``t`` (stream ``("pks-t", 0)``) and
    CBD noise ``e[f, u]`` (stream ``"pks-e"``) per call, from the seed ``_seed`` (fresh by default).
    """
    rows, n = r.shape
    if rows > D or n % RANK or a_ring.shape != (n // RANK, NU):
        raise ConfigError("encrypt_rlwe: bad shapes")
    sx = SeedExpander(new_seed() if _seed is None else _seed)
    t = _ternary(sx, (NU,), "pks-t").astype(np.int64)
    e = _cbd(sx, (n, D), params.eta, "pks-e").astype(np.int64)
    pad = negacyclic_mul_mod2(a_ring, t)  # (n/16, 4096)
    b = pad.reshape(n // RANK, D, RANK).transpose(0, 2, 1).reshape(n, D).copy()
    msg = np.zeros((n, D), dtype=_U64)
    msg[:, :rows] = (np.asarray(r, dtype=np.int64) & 0xFFFFFFFF).astype(_U64).T
    b = (b + e.astype(_U64) + (msg << _U64(params.q - params.k))) & _U64((1 << params.q) - 1)
    return UploadCiphertext(b=b, t=t, rows=rows)


# ------------------------------------------------------------------------------------------------
# Reply key (customer -> provider, per linear and group) and the compact hint
# ------------------------------------------------------------------------------------------------

def ks_a(seed: bytes) -> np.ndarray:
    """Seeded public ``a``-parts ``k_l`` of the key-switching key, uniform *in the NTT domain* modulo each
    prime of ``q1 P`` (``(4, 16, 8192)``): each value is a 128-bit stream word reduced mod ``p``
    (statistical distance below ``2^-98`` per value)."""
    out = np.empty((len(QP), RANK, NR), dtype=_U64)
    sx = SeedExpander(seed)
    for i, p in enumerate(QP):
        raw = np.frombuffer(sx.random_bytes(RANK * NR * 16, KS_LABEL, i), dtype="<u8").reshape(-1, 2)
        pp = _U64(p)
        lo, hi = raw[:, 0] % pp, raw[:, 1] % pp
        out[i] = ((hi * _U64((1 << 64) % p) % pp + lo) % pp).reshape(RANK, NR)
    return out


@dataclass
class ReplyKey:
    """Key-switching key ``beta_l = k_l t' + e_l + P iota(t_l) mod q1 P`` (NTT domain, ``(4, 16, 8192)``)
    for one (linear, group), and the customer's fresh reply secret ``t'`` (never sent)."""

    beta_hat: np.ndarray
    t_reply: np.ndarray

    def wipe(self) -> None:
        self.t_reply.fill(0)


def make_reply_key(ct: UploadCiphertext, k_hat: np.ndarray, params: PKSParams, *, _seed: bytes | None = None
                   ) -> ReplyKey:
    """``t'`` from stream ``("pks-t-reply", 0)``, ``e`` (``16 x 8192``) from ``"pks-e-ksk"`` of ``_seed``."""
    sx = SeedExpander(new_seed() if _seed is None else _seed)
    tr = _ternary(sx, (NR,), "pks-t-reply").astype(np.int64)
    e = _cbd(sx, (RANK, NR), params.eta, "pks-e-ksk").astype(np.int64)
    emb = np.zeros((RANK, NR), dtype=np.int64)
    emb[:, ::PACK] = ct.t_components  # iota(t_l): X -> Z^32
    beta_hat = np.empty_like(k_hat)
    for i, p in enumerate(QP):
        nt = Ntt.of(p)
        pp = _U64(p)
        kt = nt.inv(k_hat[i] * nt.fwd(_residues(tr, p)) % pp)
        msg = (_residues(emb, p) * _U64(params.pmod % p)) % pp
        beta_hat[i] = nt.fwd((kt + _residues(e, p) + msg) % pp)
    return ReplyKey(beta_hat=beta_hat, t_reply=tr)


def _pack(rows_u: np.ndarray) -> np.ndarray:
    """``(m, 256)`` -> ``(m/32, 8192)`` with coefficient ``32 u + j`` = row ``32 b + j``, column ``u``."""
    m = rows_u.shape[0]
    return rows_u.reshape(m // PACK, PACK, D).transpose(0, 2, 1).reshape(m // PACK, NR).copy()


def _unpack(packed: np.ndarray) -> np.ndarray:
    nb = packed.shape[0]
    return packed.reshape(nb, D, PACK).transpose(0, 2, 1).reshape(nb * PACK, D)


def switch_hint(h: np.ndarray, params: PKSParams) -> np.ndarray:
    """``round(H q1 / 2^56) mod q1`` (``m x 4096`` uint64): the provider's per-linear key-switch operand."""
    return round_mul_shift(np.asarray(h, dtype=_U64) & _M56, params.q1, params.q) % _U64(params.q1)


def _lift_centered(x: np.ndarray, q: int) -> np.ndarray:
    xs = x.astype(np.int64)
    return np.where(xs >= (q + 1) // 2, xs - q, xs)


def _atilde_hat(hs: np.ndarray) -> np.ndarray:
    """NTTs of the centred packed ``A~`` modulo the four key-switching primes: ``(4, m/32, 16, 8192)``.
    ``A~_l[32 u + j] = hs[32 b + j, 256 l + u]``."""
    m = hs.shape[0]
    q1 = Q1[0] * Q1[1]
    comp = hs.reshape(m, RANK, D)
    at = np.empty((m // PACK, RANK, NR), dtype=np.int64)
    for li in range(RANK):
        at[:, li] = _lift_centered(_pack(comp[:, li, :]), q1)
    return np.stack([Ntt.of(p).fwd(_residues(at, p)) for p in QP])


def _sum_products(a_hat: np.ndarray, b_hat: np.ndarray, p: int) -> np.ndarray:
    """``sum_l a_l * b_l`` mod p in the NTT domain: ``a_hat (nb, 16, NR)``, ``b_hat (16, NR)``."""
    pp = _U64(p)
    acc = np.zeros((a_hat.shape[0], NR), dtype=_U64)
    for li in range(RANK):
        acc = (acc + a_hat[:, li] * b_hat[li] % pp) % pp
    return acc


def _mod_down(res: list[np.ndarray], params: PKSParams) -> list[np.ndarray]:
    """RNS ModDown: residues of ``C mod q1 P`` -> residues mod the ``Q1`` primes of ``round(C / P) mod q1``."""
    p3, p4 = PS
    r3, r4 = res[2], res[3]
    k = ((r4 + _U64(p4) - r3 % _U64(p4)) % _U64(p4)) * _U64(pow(p3, -1, p4)) % _U64(p4)
    cp = r3.astype(np.int64) + np.int64(p3) * k.astype(np.int64)  # [C]_P in [0, P)  (P < 2^61)
    cp = np.where(cp >= (params.pmod + 1) // 2, cp - params.pmod, cp)  # centred
    out = []
    for i, p in enumerate(Q1):
        pinv = _U64(pow(params.pmod, -1, p))
        out.append(((res[i] + _U64(p) - _residues(cp, p)) % _U64(p)) * pinv % _U64(p))
    return out


def _crt_q1(res: list[np.ndarray]) -> np.ndarray:
    p1, p2 = Q1
    k = ((res[1] + _U64(p2) - res[0] % _U64(p2)) % _U64(p2)) * _U64(pow(p1, -1, p2)) % _U64(p2)
    return res[0] + _U64(p1) * k  # [0, q1)


def hint_form(c1: list[np.ndarray]) -> np.ndarray:
    """Plain ``c1`` residues (per Q1 prime, ``(nb, 8192)``) -> the stored compact hint ``(nb, 2, 8192, 3)``
    uint8: forward NTT, times ``N^-1 2^32`` (the native kernel's Montgomery form), 24-bit little endian."""
    out = []
    for i, p in enumerate(Q1):
        f = Ntt.of(p).fwd(c1[i]) * _U64(pow(NR, -1, p) * (1 << 32) % p) % _U64(p)
        out.append(f)
    v = np.stack(out, axis=1).astype("<u4")  # (nb, 2, NR)
    return np.ascontiguousarray(v.view(np.uint8).reshape(*v.shape, 4)[..., :3])


def hint_residues(hint: np.ndarray) -> list[np.ndarray]:
    """Inverse of the storage step of :func:`hint_form`: NTT-domain residues of ``c1`` per Q1 prime."""
    h = np.asarray(hint, dtype=np.uint8)
    v = h[..., 0].astype(_U64) | (h[..., 1].astype(_U64) << _U64(8)) | (h[..., 2].astype(_U64) << _U64(16))
    out = []
    for i, p in enumerate(Q1):
        out.append(v[:, i] * _U64(NR * pow(1 << 32, -1, p) % p) % _U64(p))
    return out


def compact_hint(hs: np.ndarray, k_hat: np.ndarray, params: PKSParams) -> np.ndarray:
    """The customer's per-linear compact hint ``c1 = round(sum_l A~_l k_l / P) mod q1`` per 32-row block, in
    the stored form of :func:`hint_form` (``(m/32, 2, 8192, 3)`` uint8: 1.5 KB per output row).

    ``hs = switch_hint(W8 @ a_mlwe_rows(a) mod 2^56)``. Built by the customer from its own ``W8`` and seeds
    before any reply secret exists (the hint-integrity rule of the plan, section 7.4).
    """
    ah = _atilde_hat(hs)
    res = [Ntt.of(p).inv(_sum_products(ah[i], k_hat[i], p)) for i, p in enumerate(QP)]
    return hint_form(_mod_down(res, params))


def provider_reply(y: np.ndarray, hs: np.ndarray, beta_hat: np.ndarray, params: PKSParams) -> np.ndarray:
    """Provider side after ``y = W8 b mod 2^56`` (``m x 256``, one group): modulus switch to q1, pack 32
    rows, key switch to ``t'`` with ``beta_hat`` and round ``c0`` to ``yb`` bits. Returns ``(m/32, 8192)``
    uint64 values below ``2^yb``."""
    q1 = params.q1
    ys = round_mul_shift(np.asarray(y, dtype=_U64) & _M56, q1, params.q) % _U64(q1)
    body = _pack(ys)  # (nb, 8192) mod q1
    h_hat = _atilde_hat(hs)
    res = []
    for i, p in enumerate(QP):
        pp = _U64(p)
        acc = Ntt.of(p).inv(_sum_products(h_hat[i], beta_hat[i], p))
        pb = (body % pp) * _U64(params.pmod % p) % pp
        res.append((pb + pp - acc) % pp)
    c0 = _crt_q1(_mod_down(res, params))
    return round_pow2_div(c0, params.yb, q1) & _U64((1 << params.yb) - 1)


def decrypt_pks(c0r: np.ndarray, hint: np.ndarray, t_reply: np.ndarray, rows: int, params: PKSParams) -> np.ndarray:
    """Customer: ``W8 r mod 2^K`` (``rows x m`` int64) from the rounded reply (``(m/32, 8192)``), the stored
    compact hint and ``t'``. ``v = round(c0r q1 / 2^yb) + c1 t' mod q1``; mask ``= round(2^K v / q1) mod 2^K``."""
    q1 = params.q1
    c0 = round_mul_shift(np.asarray(c0r, dtype=_U64), q1, params.yb) % _U64(q1)
    hr = hint_residues(hint)
    res = []
    for i, p in enumerate(Q1):
        nt = Ntt.of(p)
        pp = _U64(p)
        res.append((c0 % pp + nt.inv(hr[i] * nt.fwd(_residues(t_reply, p)) % pp)) % pp)
    v = _crt_q1(res)
    msk = round_pow2_div(v, params.k, q1) & _U64((1 << params.k) - 1)
    return _unpack(msk)[:, :rows].T.astype(np.int64)
