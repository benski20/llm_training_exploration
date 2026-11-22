"""Research prototype (not wired, default off): hint-compact encrypted masks with packed key-switched replies.

Status: numpy reference of the scheme in ``research/plans/he-compact-hint-pks.md`` (scheme id ``pks-v3``).
Nothing in the factory, the service or the wire imports this module; it exists so the scheme can be
reviewed, tested for exactness at real shapes and benchmarked against ``pgw.hemask.lwe``.

Why: today's decryption needs the hint ``H = W8 A`` with ``dim = 4096`` coefficients per output row
(24.6 KB per row at q = 2^56: 34 GB for Qwen2.5-7B, ~169 GB for 72B) and ``rank`` NTT products per
output and slot. The provider, which holds ``W8`` and can compute ``H`` itself, instead

1. modulus-switches its reply ``y = W8 b mod 2^56`` and the public ``H`` to a smaller RNS modulus
   ``q1 = p1 p2`` (~2^48),
2. packs 16 output rows of one 256-slot group into one ring element of ``R = Z[Y]/(Y^4096 + 1)``
   (row ``j``, slot ``u`` -> coefficient ``16 u + j``; the degree-256 ring embeds as ``X = Y^16``), and
3. key-switches the packed ciphertext from the upload secret (16 components of degree 256) to a fresh
   ring secret ``t'`` of degree 4096, with a key-switching key the customer uploads per (linear, group)
   (standard hybrid key switching with a special modulus ``P``, one digit).

The packed ciphertext's second component ``c1`` depends only on public data (``W8``, ``A``, the seeded
key-switching ``a``-parts), so it is a fixed per-linear *compact hint*: 4096 coefficients per 16 rows,
i.e. 256 per output row (16x smaller than ``H``). Per reply the provider sends only ``c0`` rounded to
``yb`` bits (as v2 does). The customer decrypts with one length-4096 NTT product per 16 rows and group,
``v = c0 + c1 * t' mod q1``, and rounds: ``W8 r mod 2^K = round(2^K v / q1)``.

Upload (optional part of the scheme, provider-compatible): instead of 16 independent degree-256
module components per feature, the customer encrypts 16 features x 256 slots as ONE ring-LWE sample over
``R`` (``B_C = a_C t + e_C + Delta M_C``). Splitting ``B_C`` into its 16 residue classes mod 16 gives
exactly today's wire layout (``n x 256`` bodies), and in the provider's view feature ``16 C + k`` is a
module-LWE sample under the components ``t_l`` of ``t`` with public ``a``-part ``row k`` of the
multiplication matrix of ``a_C``. So the provider kernel (``W8 b mod 2^q``), the reply format and the old
hint decryption are unchanged (tested), while the customer's pad costs one length-4096 product per
feature block instead of 16 degree-256 products per feature.

All arithmetic here is exact integer numpy (uint64, primes < 2^31); it is a reference, not a fast path.
The fast customer kernels are in ``rust/pgw-pks``.
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
    "ROWS",
    "SCHEME_ID",
    "D",
    "Ntt",
    "PKSParams",
    "ReplyKey",
    "a_mlwe_rows",
    "compact_hint",
    "decrypt_pks",
    "encrypt_rlwe",
    "make_reply_key",
    "negacyclic_mul_mod2",
    "provider_reply",
    "public_ring_rows",
]

SCHEME_ID = "pks-v3"
NR = 4096  # reply ring degree (and upload ring degree)
D = 256  # module degree seen by the provider = slots per group
ROWS = NR // D  # 16 output rows per reply ciphertext (= upload module rank)
_U64 = np.uint64


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
def ntt_primes(bits: int, count: int, skip: int = 0) -> tuple[int, ...]:
    """The ``count`` largest primes below ``2^bits`` with ``p = 1 mod 2 NR`` (after skipping ``skip``)."""
    out: list[int] = []
    step = 2 * NR
    c = ((1 << bits) - 1) // step * step + 1
    while len(out) < count + skip:
        if c < (1 << bits) and _is_prime(c):
            out.append(c)
        c -= step
    return tuple(out[skip:])


#: Reply modulus q1 = Q1[0] * Q1[1] (~2^48). Customer-side decryption works modulo these two primes.
Q1 = ntt_primes(24, 2)
#: Special modulus P = PS[0] * PS[1] (~2^60) of the hybrid key switching (provider side only).
PS = ntt_primes(30, 2)
#: Primes for exact products of 56-bit and ternary polynomials (upload pad): range 2^92 > 2^69.
EP = ntt_primes(31, 3)


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
    """Negacyclic length-``NR`` NTT modulo one prime (Cooley-Tukey forward to bit-reversed evaluation
    order, Gentleman-Sande inverse). Vectorised over leading axes. Reference speed only."""

    _cache: ClassVar[dict[int, Ntt]] = {}

    def __init__(self, p: int):
        self.p = p
        lg = NR.bit_length() - 1
        psi = pow(_primitive_root(p), (p - 1) // (2 * NR), p)
        assert pow(psi, NR, p) == p - 1
        z = [pow(psi, _brv(k, lg), p) for k in range(NR)]
        self.z = np.array(z, dtype=_U64)
        self.zi = np.array([pow(v, p - 2, p) for v in z], dtype=_U64)
        self.ninv = _U64(pow(NR, p - 2, p))

    @classmethod
    def of(cls, p: int) -> Ntt:
        if p not in cls._cache:
            cls._cache[p] = cls(p)
        return cls._cache[p]

    def fwd(self, a: np.ndarray) -> np.ndarray:
        p = _U64(self.p)
        lead = a.shape[:-1]
        x = np.array(a, dtype=_U64, copy=True).reshape(-1, NR)
        length, k = NR // 2, 1
        while length >= 1:
            nb = NR // (2 * length)
            v = x.reshape(-1, nb, 2, length)
            t = (v[:, :, 1, :] * self.z[k:k + nb, None]) % p
            u = v[:, :, 0, :].copy()
            v[:, :, 0, :] = (u + t) % p
            v[:, :, 1, :] = (u + p - t) % p
            k += nb
            length //= 2
        return x.reshape(*lead, NR)

    def inv(self, a: np.ndarray) -> np.ndarray:
        p = _U64(self.p)
        lead = a.shape[:-1]
        x = np.array(a, dtype=_U64, copy=True).reshape(-1, NR)
        length = 1
        while length < NR:
            nb = NR // (2 * length)
            v = x.reshape(-1, nb, 2, length)
            u, w = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
            v[:, :, 0, :] = (u + w) % p
            v[:, :, 1, :] = ((u + p - w) % p * self.zi[nb:2 * nb, None]) % p
            length *= 2
        return ((x * self.ninv) % p).reshape(*lead, NR)


def _residues(x_signed: np.ndarray, p: int) -> np.ndarray:
    """Residues mod ``p`` of signed int64 values."""
    return np.mod(np.asarray(x_signed, dtype=np.int64), p).astype(_U64)


def negacyclic_mul_mod2(a: np.ndarray, t: np.ndarray, bits: int = 56) -> np.ndarray:
    """Exact ``a * t mod (Y^NR + 1, 2^bits)`` for ``a < 2^56`` (uint64, ``(..., NR)``) and ternary ``t``
    (``(NR,)`` int): three-prime NTT products and a centred Garner CRT (``|a t| < 2^69 << E/2``)."""
    res = []
    for p in EP:
        nt = Ntt.of(p)
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
# Parameters
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PKSParams:
    """``k``: mask ring bits (32). ``q``: upload modulus bits (56, unchanged wire). ``yb``: reply bits."""

    k: int = 32
    q: int = 56
    eta: int = 21
    yb: int = 36
    tail: float = 24.0

    def __post_init__(self) -> None:
        if self.k != 32 or self.q != 56:
            raise ConfigError("the pks-v3 prototype covers K32 / q = 2^56 only")

    @property
    def q1(self) -> int:
        return Q1[0] * Q1[1]

    @property
    def pmod(self) -> int:
        return PS[0] * PS[1]

    @property
    def qks_bits(self) -> float:
        """log2 of the key-switching modulus ``q1 P`` (RLWE N = 4096; HE standard 128-bit limit: 109)."""
        return math.log2(self.q1 * self.pmod)

    @property
    def sigma(self) -> float:
        return math.sqrt(self.eta / 2)

    def noise_terms(self, max_row_l2: float) -> dict[str, float]:
        """High-probability bounds (tail ``tail`` sigmas / Hoeffding at 24, as SPEC 7.9) of the reply's
        decryption error, in units of the q1 scale. Budget: ``q1 / 2^(K+1)``."""
        sc = self.q1 / 2.0 ** self.q
        ternary_l2 = math.sqrt(NR)  # ||s||_2 <= 64 for any ternary secret of 4096 coefficients
        return {
            "W8 e (upload noise)": self.tail * self.sigma * max_row_l2 * sc,
            "mod switch of y": 0.5,
            "mod switch of H times s (Hoeffding)": self.tail * 0.5 * ternary_l2,
            "key switch sum A~ e / P": self.tail * self.eta * (self.q1 / 2) * math.sqrt(ROWS * NR) / self.pmod + 1.0,
            "c1 rounding times t' (Hoeffding)": self.tail * 0.5 * ternary_l2,
            "reply rounding": self.q1 / 2.0 ** (self.yb + 1) + 0.5,
        }

    def noise_ok(self, max_row_l2: float, margin_bits: float = 1.0) -> bool:
        return sum(self.noise_terms(max_row_l2).values()) < self.q1 / 2.0 ** (self.k + 1 + margin_bits)


# ------------------------------------------------------------------------------------------------
# Upload: ring-LWE over R_4096, provider sees today's (n x 256) module-LWE bodies
# ------------------------------------------------------------------------------------------------

def public_ring_rows(seed: bytes, n: int) -> np.ndarray:
    """Public ring elements ``a_C`` (``n/16 x 4096``, uniform mod 2^56) for the feature blocks."""
    if n % ROWS:
        raise ConfigError(f"in_features {n} is not a multiple of {ROWS}")
    raw = SeedExpander(seed).random_bytes((n // ROWS) * NR * 8, "pgw/hemask/pks-v3/A")
    return np.frombuffer(raw, dtype="<u8").reshape(n // ROWS, NR) & _U64((1 << 56) - 1)


def _neg_shift(p: np.ndarray, bits: int = 56) -> np.ndarray:
    """``X * p`` in ``Z_2^bits[X]/(X^256 + 1)`` (last axis)."""
    out = np.roll(p, 1, axis=-1)
    out[..., 0] = (_U64(0) - out[..., 0]) & _U64((1 << bits) - 1)
    return out


def a_mlwe_rows(a_ring: np.ndarray) -> np.ndarray:
    """The provider-view module ``a``-parts (``n x 4096``: 16 components of degree 256 per feature).

    Feature ``16 C + k`` has component ``l`` equal to entry ``(k, l)`` of the multiplication matrix of
    ``a_C`` over ``R_256`` (basis ``1, Y, .., Y^15``, ``Y^16 = X``): ``a_{C,k-l}`` for ``l <= k`` and
    ``X a_{C,k-l+16}`` otherwise. ``H = W8 @ a_mlwe_rows(a)`` is the old-format hint for this upload.
    """
    nb = a_ring.shape[0]
    comps = a_ring.reshape(nb, D, ROWS).transpose(0, 2, 1)  # (nb, 16, 256): comps[:, k] = a_k
    xcomps = _neg_shift(comps)
    out = np.empty((nb, ROWS, ROWS, D), dtype=_U64)
    for k in range(ROWS):
        for li in range(ROWS):
            out[:, k, li] = comps[:, k - li] if li <= k else xcomps[:, k - li + ROWS]
    return out.reshape(nb * ROWS, ROWS * D)


@dataclass
class UploadCiphertext:
    b: np.ndarray  # (n, 256) uint64 mod 2^56: today's wire layout
    t: np.ndarray  # (4096,) int64 ternary upload secret (customer only)
    rows: int

    @property
    def t_components(self) -> np.ndarray:
        """``(16, 256)``: ``t_l[u] = t[16 u + l]`` (the module secret the provider's reply is under)."""
        return self.t.reshape(D, ROWS).T.copy()


def encrypt_rlwe(r: np.ndarray, a_ring: np.ndarray, params: PKSParams, *, _seed: bytes | None = None
                 ) -> UploadCiphertext:
    """Encrypt the masks ``r`` (``rows x n``, ``rows <= 256``) of one linear and group.

    ``B_C = a_C t + e_C + 2^(q-K) M_C`` over ``R_4096`` with ``M_C[16 u + k] = r[u, 16 C + k]``; returned as
    the ``(n, 256)`` bodies ``b[16 C + k, u] = B_C[16 u + k]``. Fresh ternary ``t`` and CBD noise per call.
    """
    rows, n = r.shape
    if rows > D or n % ROWS or a_ring.shape != (n // ROWS, NR):
        raise ConfigError("encrypt_rlwe: bad shapes")
    seed = new_seed() if _seed is None else _seed
    sx = SeedExpander(seed)
    t = _ternary(sx, (NR,), "pks-t").astype(np.int64)
    e = _cbd(sx, (n, D), params.eta, "pks-e").astype(np.int64)
    pad = negacyclic_mul_mod2(a_ring, t)  # (n/16, 4096)
    b = pad.reshape(n // ROWS, D, ROWS).transpose(0, 2, 1).reshape(n, D).copy()
    msg = np.zeros((n, D), dtype=_U64)
    msg[:, :rows] = np.asarray(r, dtype=np.int64).astype(_U64).T
    b = (b + e.astype(_U64) + (msg << _U64(params.q - params.k))) & _U64((1 << params.q) - 1)
    return UploadCiphertext(b=b, t=t, rows=rows)


# ------------------------------------------------------------------------------------------------
# Reply key (customer -> provider, per linear and group) and the compact hint
# ------------------------------------------------------------------------------------------------

QP = Q1 + PS  # RNS of the key-switching modulus q1 P


def _ks_a(seed: bytes) -> np.ndarray:
    """Seeded public ``a``-parts ``k_l`` of the key-switching key, sampled uniformly *in the NTT domain*
    modulo each prime of ``q1 P`` (uniform there = uniform polynomial mod q1 P). ``(4, 16, 4096)``."""
    out = np.empty((len(QP), ROWS, NR), dtype=_U64)
    sx = SeedExpander(seed)
    for i, p in enumerate(QP):
        raw = np.frombuffer(sx.random_bytes(ROWS * NR * 8, "pgw/hemask/pks-v3/ksk-a", i), dtype="<u8")
        out[i] = (raw % _U64(p)).reshape(ROWS, NR)  # bias < 2^-33: fine for a prototype
    return out


@dataclass
class ReplyKey:
    """Key-switching key ``beta_l = k_l t' + e_l + P iota(t_l) mod q1 P`` (NTT domain, ``(4, 16, 4096)``)
    for one (linear, group), and the customer's fresh reply secret ``t'`` (never sent)."""

    beta_hat: np.ndarray
    t_reply: np.ndarray

    def wipe(self) -> None:
        self.t_reply.fill(0)


def make_reply_key(ct: UploadCiphertext, ks_seed: bytes, params: PKSParams, *, _seed: bytes | None = None
                   ) -> ReplyKey:
    seed = new_seed() if _seed is None else _seed
    sx = SeedExpander(seed)
    tr = _ternary(sx, (NR,), "pks-t-reply").astype(np.int64)
    e = _cbd(sx, (ROWS, NR), params.eta, "pks-e-ksk").astype(np.int64)
    comps = ct.t_components  # (16, 256)
    emb = np.zeros((ROWS, NR), dtype=np.int64)
    emb[:, ::ROWS] = comps  # iota(t_l): X -> Y^16
    k_hat = _ks_a(ks_seed)
    beta_hat = np.empty_like(k_hat)
    for i, p in enumerate(QP):
        nt = Ntt.of(p)
        pp = _U64(p)
        kt = nt.inv(k_hat[i] * nt.fwd(_residues(tr, p)) % pp)
        msg = (_residues(emb, p) * _U64(params.pmod % p)) % pp
        beta = (kt + _residues(e, p) + msg) % pp
        beta_hat[i] = nt.fwd(beta)
    return ReplyKey(beta_hat=beta_hat, t_reply=tr)


def _pack(rows_u: np.ndarray) -> np.ndarray:
    """``(m, 256)`` -> ``(m/16, 4096)`` with coefficient ``16 u + j`` = row ``16 b + j``, column ``u``."""
    m = rows_u.shape[0]
    return rows_u.reshape(m // ROWS, ROWS, D).transpose(0, 2, 1).reshape(m // ROWS, NR).copy()


def _unpack(packed: np.ndarray) -> np.ndarray:
    nb = packed.shape[0]
    return packed.reshape(nb, D, ROWS).transpose(0, 2, 1).reshape(nb * ROWS, D)


def _switch_hint(h: np.ndarray, params: PKSParams) -> np.ndarray:
    """``A~`` per block: ``(m/16, 16 (l), 4096)`` values mod q1 of ``round(H q1 / 2^56)`` packed per component."""
    q1 = params.q1
    hs = round_mul_shift(h, q1, params.q) % _U64(q1)  # (m, 4096)
    m = hs.shape[0]
    comp = hs.reshape(m, ROWS, D)  # (m, l, u)
    out = np.empty((m // ROWS, ROWS, NR), dtype=_U64)
    for li in range(ROWS):
        out[:, li] = _pack(comp[:, li, :])
    return out


def _lift_centered(x: np.ndarray, q: int) -> np.ndarray:
    xs = x.astype(np.int64)
    return np.where(xs >= (q + 1) // 2, xs - q, xs)


def _atilde_hat(h: np.ndarray, params: PKSParams) -> np.ndarray:
    """Provider precompute: NTTs of the centred ``A~`` modulo the four key-switching primes."""
    at = _lift_centered(_switch_hint(h, params), params.q1)
    return np.stack([Ntt.of(p).fwd(_residues(at, p)) for p in QP])  # (4, m/16, 16, 4096)


def _sum_products(a_hat: np.ndarray, b_hat: np.ndarray, p: int) -> np.ndarray:
    """``sum_l a_l * b_l`` mod p in the NTT domain: ``a_hat (nb, 16, NR)``, ``b_hat (16, NR)``."""
    pp = _U64(p)
    acc = np.zeros((a_hat.shape[0], NR), dtype=_U64)
    for li in range(ROWS):
        acc = (acc + a_hat[:, li] * b_hat[li] % pp) % pp
    return acc


def _mod_down(res: list[np.ndarray], params: PKSParams) -> list[np.ndarray]:
    """RNS ModDown: residues of ``C mod q1 P`` -> residues mod ``Q1`` primes of ``round(C / P) mod q1``."""
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


def compact_hint(h: np.ndarray, ks_seed: bytes, params: PKSParams) -> np.ndarray:
    """The customer's per-linear compact hint: ``c1 = round(sum_l A~_l k_l / P) mod q1`` per 16-row block,
    in the NTT domain modulo the two q1 primes: ``(2, m/16, 4096)`` uint32-sized values (8 bytes per
    coefficient, 256 coefficients per output row = 2 KB per row; 1.5 KB in 48-bit coefficient form).

    ``h = W8 @ a_mlwe_rows(a) mod 2^56`` (``m x 4096``). Public: the provider can compute and ship it (the
    customer verifies it, see the plan), or the customer computes it from ``H`` blockwise.
    """
    ah = _atilde_hat(h, params)
    k_hat = _ks_a(ks_seed)
    res = [Ntt.of(p).inv(_sum_products(ah[i], k_hat[i], p)) for i, p in enumerate(QP)]
    c1 = _mod_down(res, params)
    return np.stack([Ntt.of(p).fwd(c1[i]) for i, p in enumerate(Q1)])


def provider_reply(y: np.ndarray, h_hat: np.ndarray, key: ReplyKey, params: PKSParams) -> np.ndarray:
    """Provider side after ``y = W8 b mod 2^56`` (``m x 256``): modulus switch to q1, pack 16 rows, key
    switch to ``t'`` and round ``c0`` to ``yb`` bits. ``h_hat = _atilde_hat(H)`` (public, per linear).
    Returns ``(m/16, 4096)`` uint64 values below ``2^yb``."""
    q1 = params.q1
    ys = round_mul_shift(y, q1, params.q) % _U64(q1)
    body = _pack(ys)  # (nb, 4096) mod q1
    res = []
    for i, p in enumerate(QP):
        pp = _U64(p)
        acc = Ntt.of(p).inv(_sum_products(h_hat[i], key.beta_hat[i], p))
        pb = (body % pp) * _U64(params.pmod % p) % pp
        res.append((pb + pp - acc) % pp)
    c0 = _crt_q1(_mod_down(res, params))
    return round_pow2_div(c0, params.yb, q1) & _U64((1 << params.yb) - 1)


def decrypt_pks(c0r: np.ndarray, hint: np.ndarray, key: ReplyKey, rows: int, params: PKSParams) -> np.ndarray:
    """Customer: ``W8 r mod 2^K`` (``rows x m`` int64) from the rounded reply and the compact hint.
    ``v = round(c0r q1 / 2^yb) + c1 t' mod q1``; mask ``= round(2^K v / q1) mod 2^K``. Wipes ``t'``."""
    try:
        q1 = params.q1
        c0 = round_mul_shift(c0r, q1, params.yb) % _U64(q1)
        res = []
        for i, p in enumerate(Q1):
            nt = Ntt.of(p)
            pp = _U64(p)
            res.append((c0 % pp + nt.inv(hint[i] * nt.fwd(_residues(key.t_reply, p)) % pp)) % pp)
        v = _crt_q1(res)
        msk = round_pow2_div(v, params.k, q1) & _U64((1 << params.k) - 1)
        return _unpack(msk)[:, :rows].T.astype(np.int64)
    finally:
        key.wipe()
