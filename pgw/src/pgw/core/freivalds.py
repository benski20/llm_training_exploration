"""Freivalds integrity checks over ``Z_{2^K}``.

For each linear the mask factory draws ``c`` secret check vectors ``u_j`` (uniform in ``Z_{2^K}^m``)
and precomputes ``u_j^T W8 mod 2^K``. The gateway accepts a service result ``y_hat`` for input
``x_hat`` only if, for every row ``t`` and every ``j``::

    u_j . y_hat[t]  ==  (u_j^T W8) . x_hat[t]   (mod 2^K)

Soundness: let ``e = y_hat - W8 x_hat != 0`` with 2-adic valuation ``v`` (largest power of two
dividing every entry). Then ``u . e`` is uniform over the multiples of ``2^v`` modulo ``2^K``, so one
check vector misses it with probability ``2^-(K - v)``. Errors with ``2^v`` above the genuine output
bound are caught deterministically by :func:`pgw.core.masking.check_output_range`. Combined, the
per-call false-accept probability is at most ``2^(-c * h)`` with ``h = K - ceil(log2(4 * bound))``
(see :func:`soundness_bits`). The check vectors must stay secret from the compute service.

Prime-field check (opt-in segment format ``check_field = "p31"``, SPEC 5.1)
--------------------------------------------------------------------------
With ``p = 2^31 - 1`` the factory draws ``c`` (default 1) secret vectors ``u_i`` uniform in ``F_p^m``
and stores ``u_i^T W8 mod p``. The gateway checks the *unmasked* integers it already has: the
quantized input ``xq`` (prepare) and the decoded output ``yq = centered(y_hat - W r)`` (finish)::

    u_i . yq[t]  ==  (u_i^T W8 mod p) . xq[t]   (mod p)     for every row t and every i

Soundness: after the range check ``|yq| <= B`` and with the genuine ``|W8 xq| <= B`` (the cap /
headroom bound), an accepted wrong row has ``e = yq - W8 xq != 0`` with ``|e_j| <= 2B < p``, so
``e mod p != 0`` and one uniform ``u_i`` misses it with probability exactly ``1/p``: at least
``c * (log2(p) - 2^-32)`` bits per call (:func:`prime_soundness_bits`), 31 bits for one vector,
against ``c * h`` (21 bits with 7 vectors at K32A12C27) for the ring check. The condition ``2B < p``
is checked per linear (0 bits otherwise, so such segments are refused).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError, ShapeError
from pgw.core.ring import RingParams
from pgw.core.ringmm import PreparedWeight

__all__ = [
    "CHECK_FIELD_P31",
    "P31",
    "PRIME_TAG",
    "check_soundness_bits",
    "checks_to_u64",
    "field_modulus",
    "freivalds_check",
    "freivalds_check_u64",
    "freivalds_precompute",
    "freivalds_verify",
    "is_prime_token",
    "prime_check",
    "prime_check_vectors",
    "prime_input_half",
    "prime_output_half",
    "prime_precompute",
    "prime_soundness_bits",
    "soundness_bits",
]


def freivalds_precompute(u: torch.Tensor, w8: torch.Tensor, params: RingParams) -> torch.Tensor:
    """Return ``u @ W8 mod 2^K`` as int64 ``(c, n)`` for check vectors ``u`` of shape ``(c, m)``."""
    if u.ndim != 2 or u.shape[1] != w8.shape[0]:
        raise ShapeError(f"check vectors must be (c, {w8.shape[0]}), got {tuple(u.shape)}")
    wt = PreparedWeight(w8.t().contiguous(), device=u.device, cache_float=False)
    return wt.matmul(u.to(torch.int64), params)


def _ring_dot(a: torch.Tensor, b: torch.Tensor, params: RingParams) -> np.ndarray:
    """``a @ b^T mod 2^K`` for ring matrices (uint64 arithmetic wraps mod 2^64, defined behaviour)."""
    an = (a.detach().cpu().numpy() & params.mask).astype(np.uint64)
    bn = (b.detach().cpu().numpy() & params.mask).astype(np.uint64)
    return (an @ bn.T) & np.uint64(params.mask)


def freivalds_check(
    x_hat: torch.Tensor,
    y_hat: torch.Tensor,
    u: torch.Tensor,
    uw: torch.Tensor,
    params: RingParams,
) -> torch.Tensor:
    """Per-row verdicts (bool ``(T,)``): True where every check vector agrees.

    Args:
        x_hat: Masked inputs sent to the service, int64 ``(T, n)``.
        y_hat: Service results, int64 ``(T, m)``.
        u: Secret check vectors, int64 ``(c, m)``.
        uw: Precomputed ``u @ W8 mod 2^K``, int64 ``(c, n)``.
        params: Ring parameters.
    """
    if x_hat.ndim != 2 or y_hat.ndim != 2 or x_hat.shape[0] != y_hat.shape[0]:
        raise ShapeError(f"bad shapes x_hat {tuple(x_hat.shape)} y_hat {tuple(y_hat.shape)}")
    if u.shape[1] != y_hat.shape[1] or uw.shape[1] != x_hat.shape[1] or u.shape[0] != uw.shape[0]:
        raise ShapeError("check vector shapes do not match the linear")
    if u.shape[0] < 1:
        # With no check vectors every row would "pass": refuse instead of silently disabling integrity.
        raise ConfigError("Freivalds check needs at least one check vector (checks=0)")
    lhs = _ring_dot(y_hat, u, params)  # (T, c)
    rhs = _ring_dot(x_hat, uw, params)  # (T, c)
    return torch.from_numpy((lhs == rhs).all(axis=1))


def checks_to_u64(u: torch.Tensor, uw: torch.Tensor, params: RingParams) -> tuple[np.ndarray, np.ndarray]:
    """Read-only uint64 copies of check vectors ``(u, uW)`` for :func:`freivalds_check_u64`."""
    u64 = (u.detach().cpu().numpy() & params.mask).astype(np.uint64)
    uw64 = (uw.detach().cpu().numpy() & params.mask).astype(np.uint64)
    u64.flags.writeable = False
    uw64.flags.writeable = False
    return u64, uw64


def freivalds_check_u64(
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    u64: np.ndarray,
    uw64: np.ndarray,
    params: RingParams,
) -> np.ndarray:
    """:func:`freivalds_check` on uint64 arrays already reduced mod ``2^K`` (gateway hot path).

    ``x_hat`` ``(T, n)``, ``y_hat`` ``(T, m)``, ``u64`` ``(c, m)``, ``uw64`` ``(c, n)``; returns bool ``(T,)``.
    uint64 products wrap mod ``2^64`` (defined behaviour), and ``2^K`` divides ``2^64``.
    """
    if x_hat.ndim != 2 or y_hat.ndim != 2 or x_hat.shape[0] != y_hat.shape[0]:
        raise ShapeError(f"bad shapes x_hat {x_hat.shape} y_hat {y_hat.shape}")
    if u64.shape[1] != y_hat.shape[1] or uw64.shape[1] != x_hat.shape[1] or u64.shape[0] != uw64.shape[0]:
        raise ShapeError("check vector shapes do not match the linear")
    if u64.shape[0] < 1:
        raise ConfigError("Freivalds check needs at least one check vector (checks=0)")
    m = np.uint64(params.mask)
    lhs = (y_hat @ u64.T) & m
    rhs = (x_hat @ uw64.T) & m
    out: np.ndarray = (lhs == rhs).all(axis=1)
    return out


def freivalds_verify(
    x_hat: torch.Tensor,
    y_hat: torch.Tensor,
    u: torch.Tensor,
    uw: torch.Tensor,
    params: RingParams,
    *,
    linear_id: str = "",
) -> None:
    """Raise :class:`IntegrityError` unless every row passes :func:`freivalds_check`."""
    ok = freivalds_check(x_hat, y_hat, u, uw, params)
    if not bool(ok.all()):
        bad = int((~ok).sum())
        raise IntegrityError(
            "compute service result failed the Freivalds check",
            details={"linear": linear_id, "bad_rows": bad, "rows": int(ok.numel())},
        )


def soundness_bits(params: RingParams, in_features: int, checks: int) -> float:
    """Lower bound on ``-log2(false accept probability)`` per call with range check + ``checks`` vectors."""
    h = params.k - math.ceil(math.log2(4 * max(1, params.output_bound(in_features))))
    return max(0, h) * checks


# ---- prime-field check (opt-in, segment header ``check_field = "p31"``) ------------------------------

#: The Mersenne prime ``2^31 - 1``: the only field of the ``"p31"`` check format.
P31 = (1 << 31) - 1
#: Header value of the prime-field check format (absent = today's ring check over ``Z_{2^K}``).
CHECK_FIELD_P31 = "p31"
#: Last element of the check token of a prime-field extent (``(segment_id, linear, PRIME_TAG)``), so
#: a consumer can tell the formats apart and never feeds prime vectors to a ring check.
PRIME_TAG = "p31"
_FIELDS = {CHECK_FIELD_P31: P31}
#: Upper bound on ``-log2`` of the per-element point-probability excess of ``u64 mod p`` sampling:
#: ``max_v Pr[u = v] <= ceil(2^64 / p) / 2^64 <= (1 + 2^-33) / p``.
_SAMPLING_SLACK_BITS = math.log2(1.0 + 2.0**-33)


def field_modulus(check_field: str | None) -> int | None:
    """The prime of a check format (``None``/``"ring"``: the ring check, returns ``None``)."""
    if check_field in (None, "", "ring"):
        return None
    p = _FIELDS.get(str(check_field))
    if p is None:
        raise ConfigError(f"unknown Freivalds check field {check_field!r} (known: ring, p31)")
    return p


def is_prime_token(token: object) -> bool:
    """True if a check token names prime-field vectors (see :data:`PRIME_TAG`)."""
    return isinstance(token, tuple) and len(token) == 3 and token[2] == PRIME_TAG


def prime_soundness_bits(params: RingParams, in_features: int, checks: int, p: int = P31) -> float:
    """``-log2(false accept probability)`` per call of the prime-field check with ``checks`` vectors.

    Requires the ring to be at most 32 bits wide (the stored ``u W mod p`` words) and ``2 B < p`` with
    ``B = params.output_bound(in_features)`` (an accepted wrong output differs from the genuine one by
    at most ``2 B`` per entry, so it is non-zero mod ``p``); otherwise 0 (refused by every floor).
    """
    if checks <= 0 or params.k > 32 or 2 * max(1, params.output_bound(in_features)) >= p:
        return 0.0
    return checks * (math.log2(p) - _SAMPLING_SLACK_BITS)


def check_soundness_bits(params: RingParams, in_features: int, checks: int, check_field: str | None = None) -> float:
    """:func:`soundness_bits` (ring check) or :func:`prime_soundness_bits` by segment check format."""
    p = field_modulus(check_field)
    if p is None:
        return float(soundness_bits(params, in_features, checks))
    return prime_soundness_bits(params, in_features, checks, p)


def prime_check_vectors(seed: bytes, checks: int, out_features: int, linear_key: str, p: int = P31) -> torch.Tensor:
    """The secret prime-field check vectors of a segment and linear, int64 ``(checks, m)`` in ``[0, p)``.

    Normative (SPEC 7.3): element ``i`` is ``u64_le(stream("u-p31", linear_id)[8i : 8i + 8]) mod p``.
    """
    from pgw.core.prng import SeedExpander

    if p != P31:
        raise ConfigError("only p = 2^31 - 1 is supported")
    raw = SeedExpander(seed).ring_uniform((checks, out_features), 64, "u-p31", linear_key)
    u = raw.numpy().view(np.uint64)
    out = torch.from_numpy((u % np.uint64(p)).astype(np.int64))
    u.fill(0)
    return out


def _mod_p(a: np.ndarray, p: int) -> np.ndarray:
    return np.mod(a, p).astype(np.int64)


def _matmul_mod_p(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    """``a @ b mod p`` exactly for int64 ``a (T, L)`` and ``b (L, c)`` with entries in ``[0, p)``, ``p < 2^31``.

    ``b`` is split into 16-bit limbs; partial products ``< 2^47`` are summed over chunks of 2^15
    (``< 2^62``) in int64 and reduced after each chunk."""
    lo = b & 0xFFFF
    hi = b >> 16
    acc_lo = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    acc_hi = np.zeros_like(acc_lo)
    step = 1 << 15
    for s0 in range(0, a.shape[1], step):
        aa = a[:, s0 : s0 + step]
        acc_lo = (acc_lo + aa @ lo[s0 : s0 + step]) % p
        acc_hi = (acc_hi + aa @ hi[s0 : s0 + step]) % p
    return (acc_lo + acc_hi * (1 << 16)) % p


def prime_precompute(u: torch.Tensor, w8: torch.Tensor, p: int = P31) -> torch.Tensor:
    """``u @ W8 mod p`` as int64 ``(c, n)`` in ``[0, p)`` for ``u (c, m)`` in ``[0, p)`` (exact).

    ``u`` is split into 16-bit limbs (each partial sum ``<= m * 2^16 * 128 < 2^53`` for ``m < 2^29``),
    so two float64 GEMMs per limb on ``W8``'s device are exact."""
    if u.ndim != 2 or u.shape[1] != w8.shape[0]:
        raise ShapeError(f"check vectors must be (c, {w8.shape[0]}), got {tuple(u.shape)}")
    if w8.shape[0] >= 1 << 29:
        raise ShapeError("out_features too large for the exact limb product")
    dev = w8.device
    wf = w8.to(dev, torch.float64)
    uu = u.to(dev, torch.int64)
    lo = (uu & 0xFFFF).to(torch.float64)
    hi = (uu >> 16).to(torch.float64)
    r_lo = torch.remainder((lo @ wf).to(torch.int64), p)
    r_hi = torch.remainder((hi @ wf).to(torch.int64), p)
    out = torch.remainder(r_lo + r_hi * (1 << 16), p).to("cpu")
    del lo, hi, uu
    return out


def prime_input_half(xq: np.ndarray, uw: np.ndarray, p: int = P31) -> np.ndarray:
    """``uw . xq mod p`` per row: int64 ``(T, c)`` for integer ``xq (T, n)`` and ``uw (c, n)`` in ``[0, p)``."""
    return _matmul_mod_p(_mod_p(np.asarray(xq, dtype=np.int64), p), np.asarray(uw, dtype=np.int64).T.copy(), p)


def prime_output_half(yq: np.ndarray, u: np.ndarray, p: int = P31) -> np.ndarray:
    """``u . yq mod p`` per row: int64 ``(T, c)`` for integer ``yq (T, m)`` and ``u (c, m)`` in ``[0, p)``."""
    return _matmul_mod_p(_mod_p(np.asarray(yq, dtype=np.int64), p), np.asarray(u, dtype=np.int64).T.copy(), p)


def prime_check(xq: np.ndarray, yq: np.ndarray, u: np.ndarray, uw: np.ndarray, p: int = P31) -> np.ndarray:
    """Per-row verdicts (bool ``(T,)``) of the prime-field check on the unmasked integers.

    Args:
        xq: Quantized inputs (gateway side), integers ``(T, n)``.
        yq: Decoded outputs ``centered(y_hat - W r)`` (after the range check), integers ``(T, m)``.
        u: Secret vectors ``(c, m)`` in ``[0, p)``.
        uw: ``u @ W8 mod p`` ``(c, n)``.
    """
    xq = np.asarray(xq)
    yq = np.asarray(yq)
    if xq.ndim != 2 or yq.ndim != 2 or xq.shape[0] != yq.shape[0]:
        raise ShapeError(f"bad shapes xq {xq.shape} yq {yq.shape}")
    if u.shape[1] != yq.shape[1] or uw.shape[1] != xq.shape[1] or u.shape[0] != uw.shape[0]:
        raise ShapeError("check vector shapes do not match the linear")
    if u.shape[0] < 1:
        raise ConfigError("Freivalds check needs at least one check vector (checks=0)")
    out: np.ndarray = (prime_output_half(yq, u, p) == prime_input_half(xq, uw, p)).all(axis=1)
    return out
