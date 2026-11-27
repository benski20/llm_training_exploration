"""``pks-v3-K56``: a direct K56 parameter set of the pks-v3 scheme (research reference; not wired, not reviewed).

Why (``research/reviews/mask-cost-consumer.md``, section 12): fused2's trips run at ring ``K56C53``, and today K56
masks come from pks-v3 (K32) by digit decomposition: 7 K32 products per mask (``pgw.factory.fused2_he``), or 3-4 with
the top-digit plan. This module is the same scheme with every modulus scaled so that one product gives
``W8 r mod 2^56`` directly, and every noise term keeps exactly its pks-v3 value relative to the budget:

==================  =======================  ===============================================
                    pks-v3 (K32)             pks-v3-K56 (this module)
==================  =======================  ===============================================
upload modulus      ``2^56``                 ``2^80`` (message scale ``2^24`` in both)
reply modulus q1    2 primes < 2^24 (~2^48)  3 primes < 2^24 (~2^72)
special P           2 primes < 2^30 (~2^60)  3 primes < 2^28 (~2^84): ``P / q1`` ~ ``2^12`` in both
key-switch RNS      4 primes (2^108)         6 primes (~2^156)
reply bits yb       36                       60 (``q1 / 2^yb`` ~ ``2^12`` in both)
pad primes          3 x 31 bits              4 x 31 bits (``|a t| < 2^92``)
budget              ``q1 / 2^33``            ``q1 / 2^57``
==================  =======================  ===============================================

Every term of :meth:`pgw.hemask.pks.PKSParams.noise_terms` depends on the moduli only through ``q1 / 2^q``,
``q1 / P`` and ``q1 / 2^yb``, which are the same up to the primes' distance from powers of two, so the worst-case
bound (all-127 rows at n = 29568) stays at ``2^13.4`` against the budget ``2^15`` (checked by
:meth:`K56Params.noise_terms`, tested). Rings, ranks, packing, the hint form and the hybrid key switch are pks-v3's.

Everything here is exact (Python integers wherever a value exceeds 64 bits; numpy for the NTTs, whose primes are
below ``2^31``). Speed is not a goal: this proves the parameter set is correct. Production kernels would be the
pks-v3 kernels with 10 instead of 7 byte planes, 6 instead of 4 key-switch primes and 3 instead of 2 decryption
primes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from pgw.core.prng import SeedExpander, new_seed
from pgw.hemask.lwe import _cbd, _ternary
from pgw.hemask.pks import NR, NU, PACK, RANK, D, Ntt, ntt_primes

__all__ = ["K56Params", "decrypt", "encrypt", "hint", "ks_a", "provider_reply", "public_ring_rows", "reply_key",
           "security", "switch_hint", "wr_mod"]

_U64 = np.uint64
Q1 = ntt_primes(24, 3)
PS = ntt_primes(28, 3)
QP = Q1 + PS
EP = ntt_primes(31, 4, 2 * NU)
A_LABEL = "pgw/hemask/pks-v3-k56/A"
KS_LABEL = "pgw/hemask/pks-v3-k56/ksk-a"


@dataclass(frozen=True)
class K56Params:
    k: int = 56
    q: int = 80
    eta: int = 21
    yb: int = 60
    tail: float = 24.0

    @property
    def q1(self) -> int:
        return math.prod(Q1)

    @property
    def pmod(self) -> int:
        return math.prod(PS)

    @property
    def sigma(self) -> float:
        return math.sqrt(self.eta / 2)

    def noise_terms(self, max_row_l2: float) -> dict[str, float]:
        """:meth:`pgw.hemask.pks.PKSParams.noise_terms` with this set's moduli (units of 1 mod q1)."""
        sc = self.q1 / 2.0 ** self.q
        t = self.tail
        return {
            "W8 e (upload noise)": t * self.sigma * max_row_l2 * sc,
            "mod switch of y": 0.5,
            "mod switch of H times s (Hoeffding)": t * 0.5 * math.sqrt(NU),
            "key switch sum A~ e / P": t * self.eta * (self.q1 / 2) * math.sqrt(RANK * NR) / self.pmod,
            "ModDown of c0": 0.5,
            "ModDown of c1 times t' (Hoeffding)": t * 0.5 * math.sqrt(NR),
            "reply rounding": self.q1 / 2.0 ** (self.yb + 1) + 0.5,
        }

    def noise_bound(self, max_row_l2: float) -> float:
        return sum(self.noise_terms(max_row_l2).values())

    def budget(self, margin_bits: float = 0.0) -> float:
        return self.q1 / 2.0 ** (self.k + 1 + margin_bits)


def security() -> dict[str, Any]:
    """Primal-uSVP core-SVP (the repo's model, :func:`pgw.hemask.mlwe_ring.core_svp_beta`) of both instances, next
    to pks-v3's, and the HomomorphicEncryption.org 128-bit limits (ternary secret, classical / quantum)."""
    from pgw.hemask.mlwe_ring import core_svp_beta

    s = math.sqrt(21 / 2)
    lqp = math.log2(math.prod(QP))
    out: dict[str, Any] = {}
    for name, n, lq, lim in (("v3 upload", NU, 56, (109, 101)), ("v3 reply key", NR, 107.98, (218, 202)),
                             ("K56 upload", NU, 80, (109, 101)), ("K56 reply key", NR, lqp, (218, 202)),
                             ("K56 upload at n=8192", 2 * NU, 80, (218, 202)),
                             ("K56 reply key at n=16384", 2 * NR, lqp, (438, 411))):
        b = core_svp_beta(n, lq, s)
        out[name] = {"n": n, "log2_q": round(lq, 2), "beta": b, "core_svp_classical": round(0.292 * b, 1),
                     "core_svp_quantum": round(0.265 * b, 1), "he_std_128_limit": lim}
    return out


# ---- exact integer helpers ---------------------------------------------------------------------------------

def _obj(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=object)


def _res(x: np.ndarray, p: int) -> np.ndarray:
    """Residues mod ``p`` (uint64) of an integer (object or int64) array."""
    return np.asarray(_obj(x) % p, dtype=_U64) if x.dtype == object else np.mod(x.astype(np.int64), p).astype(_U64)


def _crt(res: list[np.ndarray], primes: tuple[int, ...]) -> np.ndarray:
    """Residues -> the integer in ``[0, prod)`` (object array)."""
    m = math.prod(primes)
    acc = np.zeros(res[0].shape, dtype=object)
    for r, p in zip(res, primes, strict=True):
        mi = m // p
        acc = acc + _obj(r) * (mi * pow(mi, -1, p))
    return acc % m


def _centre(x: np.ndarray, m: int) -> np.ndarray:
    return np.where(x > m // 2, x - m, x)


def _round_div(x: np.ndarray, num: int, den_bits: int) -> np.ndarray:
    """``floor((x num + 2^(den_bits - 1)) / 2^den_bits)`` (object ints)."""
    return (x * num + (1 << (den_bits - 1))) >> den_bits


def wr_mod(w8: np.ndarray, x: np.ndarray, bits: int) -> np.ndarray:
    """``W8 x mod 2^bits`` exactly (``x`` object ints below ``2^bits``, any width): 20-bit limbs in int64 GEMMs."""
    w = w8.astype(np.int64)
    xo = _obj(x)
    acc = np.zeros((w.shape[0], xo.shape[1]), dtype=object)
    for i in range(-(-bits // 20)):
        limb = np.asarray((xo >> (20 * i)) & ((1 << 20) - 1), dtype=np.int64)
        acc = acc + _obj(w @ limb) * (1 << (20 * i))
    return acc % (1 << bits)


# ---- scheme --------------------------------------------------------------------------------------------------

def public_ring_rows(seed: bytes, n: int, linear: str) -> np.ndarray:
    """``a_C`` (``n/16 x 4096``) uniform mod ``2^80`` (16 stream bytes per value, the top 48 discarded)."""
    raw = SeedExpander(seed).random_bytes((n // RANK) * NU * 16, A_LABEL, linear)
    lo = np.frombuffer(raw, dtype="<u8")[0::2]
    hi = np.frombuffer(raw, dtype="<u8")[1::2]
    return (_obj(lo) + _obj(hi & _U64(0xFFFF)) * (1 << 64)).reshape(n // RANK, NU)


@cache
def _a_rows_cached(seed: bytes, n: int, linear: str) -> np.ndarray:
    """``A = a_mlwe_rows(a)`` over ``Z_{2^80}`` (``n x 4096`` object), as :func:`pgw.hemask.pks.a_mlwe_rows`."""
    a = public_ring_rows(seed, n, linear)
    nb = a.shape[0]
    comps = a.reshape(nb, D, RANK).transpose(0, 2, 1)  # (nb, 16, 256)
    xs = np.empty_like(comps)
    xs[..., 1:] = comps[..., :-1]
    xs[..., 0] = (-comps[..., -1]) % (1 << 80)  # X * comp in Z_{2^80}[X]/(X^256 + 1)
    out = np.empty((nb, RANK, RANK, D), dtype=object)
    for k in range(RANK):
        for li in range(RANK):
            out[:, k, li] = comps[:, k - li] if li <= k else xs[:, k - li + RANK]
    return out.reshape(nb * RANK, RANK * D)


def pad(a_ring: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``a_C t mod (Y^4096 + 1, 2^80)``: four 31-bit-prime NTTs, centred CRT (``|a t| < 2^92``)."""
    res = []
    for p in EP:
        nt = Ntt.of(p, NU)
        res.append(nt.inv(nt.fwd(_res(a_ring, p)) * nt.fwd(_res(t, p)) % _U64(p)))
    m = math.prod(EP)
    return _centre(_crt(res, EP), m) % (1 << 80)


@dataclass
class Upload:
    b: np.ndarray  # (n, 256) object below 2^80
    t: np.ndarray  # (4096,) ternary


def encrypt(r: np.ndarray, a_ring: np.ndarray, par: K56Params, *, seed: bytes | None = None) -> Upload:
    """``B_C = a_C t + e_C + 2^24 M_C mod 2^80``, ``M_C[16 u + k] = r[u, 16 C + k]`` (``r`` below ``2^56``)."""
    rows, n = r.shape
    sx = SeedExpander(new_seed() if seed is None else seed)
    t = _ternary(sx, (NU,), "pks-t").astype(np.int64)
    e = _cbd(sx, (n, D), par.eta, "pks-e").astype(np.int64)
    pd = pad(a_ring, t)
    b = pd.reshape(n // RANK, D, RANK).transpose(0, 2, 1).reshape(n, D)
    msg = np.zeros((n, D), dtype=object)
    msg[:, :rows] = _obj(np.asarray(r, dtype=np.int64)).T
    return Upload((b + _obj(e) + msg * (1 << (par.q - par.k))) % (1 << par.q), t)


def ks_a(seed: bytes) -> np.ndarray:
    """``k_l`` uniform in the NTT domain mod each key-switching prime: ``(6, 16, 8192)`` uint64."""
    out = np.empty((len(QP), RANK, NR), dtype=_U64)
    sx = SeedExpander(seed)
    for i, p in enumerate(QP):
        raw = np.frombuffer(sx.random_bytes(RANK * NR * 16, KS_LABEL, i), dtype="<u8").reshape(-1, 2)
        pp = _U64(p)
        out[i] = ((raw[:, 1] % pp) * _U64((1 << 64) % p) % pp + raw[:, 0] % pp).reshape(RANK, NR) % pp
    return out


def reply_key(up: Upload, k_hat: np.ndarray, par: K56Params, *, seed: bytes | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """``(beta_hat (6, 16, 8192), t')``: ``beta_l = k_l t' + e_l + P iota(t_l) mod q1 P``."""
    sx = SeedExpander(new_seed() if seed is None else seed)
    tr = _ternary(sx, (NR,), "pks-t-reply").astype(np.int64)
    e = _cbd(sx, (RANK, NR), par.eta, "pks-e-ksk").astype(np.int64)
    emb = np.zeros((RANK, NR), dtype=np.int64)
    emb[:, ::PACK] = up.t.reshape(D, RANK).T
    beta = np.empty_like(k_hat)
    for i, p in enumerate(QP):
        nt, pp = Ntt.of(p), _U64(p)
        kt = nt.inv(k_hat[i] * nt.fwd(_res(tr, p)) % pp)
        beta[i] = nt.fwd((kt + _res(e, p) + _res(emb, p) * _U64(par.pmod % p) % pp) % pp)
    return beta, tr


def switch_hint(h: np.ndarray, par: K56Params) -> np.ndarray:
    """``round(q1 H / 2^80) mod q1`` (object)."""
    return _round_div(_obj(h), par.q1, par.q) % par.q1


def _atilde_hat(hs: np.ndarray, par: K56Params) -> list[np.ndarray]:
    m = hs.shape[0]
    at = _centre(hs.reshape(m // PACK, PACK, RANK, D).transpose(0, 2, 3, 1).reshape(m // PACK, RANK, NR), par.q1)
    return [Ntt.of(p).fwd(_res(at, p)) for p in QP]


def _key_switch(ah: list[np.ndarray], keys: np.ndarray) -> list[np.ndarray]:
    out = []
    for i, p in enumerate(QP):
        pp = _U64(p)
        acc = np.zeros((ah[i].shape[0], NR), dtype=_U64)
        for li in range(RANK):
            acc = (acc + ah[i][:, li] * keys[i, li] % pp) % pp
        out.append(Ntt.of(p).inv(acc))
    return out


def _mod_down(res: list[np.ndarray], par: K56Params) -> np.ndarray:
    """``round(C / P) mod q1`` as an integer (object) from the residues of ``C`` mod ``q1 P``."""
    cp = _centre(_crt(res[len(Q1):], PS), par.pmod)  # [C]_P, centred
    q = [(_obj(res[i]) - cp) * pow(par.pmod, -1, p) % p for i, p in enumerate(Q1)]
    return _crt(q, Q1)


def hint(hs: np.ndarray, k_hat: np.ndarray, par: K56Params) -> list[np.ndarray]:
    """The customer's compact hint ``c1 = round(sum_l A~_l k_l / P) mod q1``, as NTT-domain residues per ``Q1`` prime
    (``(m/32, 8192)`` each)."""
    c1 = _mod_down(_key_switch(_atilde_hat(hs, par), k_hat), par)
    return [Ntt.of(p).fwd(_res(c1, p)) for p in Q1]


def provider_reply(y: np.ndarray, hs: np.ndarray, beta: np.ndarray, par: K56Params) -> np.ndarray:
    """``(m/32, 8192)`` reply values below ``2^60`` from ``y = W8 b mod 2^80`` of one group."""
    ys = _round_div(_obj(y), par.q1, par.q) % par.q1
    m = ys.shape[0]
    body = ys.reshape(m // PACK, PACK, D).transpose(0, 2, 1).reshape(m // PACK, NR)
    acc = _key_switch(_atilde_hat(hs, par), beta)
    res = [(_res(body, p) * _U64(par.pmod % p) % _U64(p) + _U64(p) - acc[i]) % _U64(p) for i, p in enumerate(QP)]
    c0 = _mod_down(res, par)
    return (c0 * (1 << par.yb) + par.q1 // 2) // par.q1 % (1 << par.yb)


def decrypt(c0r: np.ndarray, c1_hat: list[np.ndarray], t_reply: np.ndarray, rows: int, par: K56Params
            ) -> np.ndarray:
    """``W8 r mod 2^56`` (``rows x m`` object): ``v = round(c0r q1 / 2^60) + c1 t' mod q1``, ``round(2^56 v / q1)``."""
    c0 = _round_div(_obj(c0r), par.q1, par.yb) % par.q1
    res = []
    for i, p in enumerate(Q1):
        nt = Ntt.of(p)
        res.append((_res(c0, p) + nt.inv(c1_hat[i] * nt.fwd(_res(t_reply, p)) % _U64(p))) % _U64(p))
    v = _crt(res, Q1)
    msk = (v * (1 << par.k) + par.q1 // 2) // par.q1 % (1 << par.k)
    nb = msk.shape[0]
    return msk.reshape(nb, D, PACK).transpose(0, 2, 1).reshape(nb * PACK, D)[:, :rows].T


def decryption_error(c0r: np.ndarray, c1_hat: list[np.ndarray], t_reply: np.ndarray, w8: np.ndarray,
                     r: np.ndarray, par: K56Params) -> int:
    """``max |v - q1 W8 r / 2^56|`` in units of 1 mod q1 (the measured noise; the budget is ``q1 / 2^57``)."""
    c0 = _round_div(_obj(c0r), par.q1, par.yb) % par.q1
    res = []
    for i, p in enumerate(Q1):
        nt = Ntt.of(p)
        res.append((_res(c0, p) + nt.inv(c1_hat[i] * nt.fwd(_res(t_reply, p)) % _U64(p))) % _U64(p))
    v = _crt(res, Q1)
    nb = v.shape[0]
    v = v.reshape(nb, D, PACK).transpose(0, 2, 1).reshape(nb * PACK, D)[:, :r.shape[0]].T
    exact = wr_mod(w8, _obj(np.asarray(r, dtype=np.int64)).T, par.k).T  # (rows, m)
    err = _centre((v - (exact * par.q1 >> par.k)) % par.q1, par.q1)  # floor(q1 x / 2^56) vs v
    return int(np.max(np.abs(err.astype(object))))
