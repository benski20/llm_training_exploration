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
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pgw.core.errors import IntegrityError, ShapeError
from pgw.core.ring import RingParams
from pgw.core.ringmm import PreparedWeight

__all__ = ["freivalds_check", "freivalds_precompute", "freivalds_verify", "soundness_bits"]


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
    lhs = _ring_dot(y_hat, u, params)  # (T, c)
    rhs = _ring_dot(x_hat, uw, params)  # (T, c)
    return torch.from_numpy((lhs == rhs).all(axis=1))


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
