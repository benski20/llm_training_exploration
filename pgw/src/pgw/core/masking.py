"""One-time-mask math for a single masked linear call.

Gateway side, per linear op ``y = W x`` (``x`` already quantized to ``xq``)::

    x_hat = (xq + r) mod 2^K                    # mask_activations   -> sent to the compute service
    y_hat = W8 x_hat mod 2^K                    # computed by the service (ringmm)
    yq    = centered((y_hat - W r) mod 2^K)     # unmask_outputs     -> exact xq @ W8^T

``r`` is uniform in ``Z_{2^K}`` and used exactly once, so ``x_hat`` is uniform and independent of
``x`` (a one-time pad). ``W r`` is precomputed by the mask factory.
"""

from __future__ import annotations

import numpy as np
import torch

from pgw.core.errors import IntegrityError, RingOverflowError, ShapeError
from pgw.core.ring import RingParams, center

__all__ = ["check_output_range", "mask_activations", "unmask_outputs", "unmask_outputs_u64"]


def mask_activations(xq: torch.Tensor, r: torch.Tensor, params: RingParams) -> torch.Tensor:
    """Return ``x_hat = (xq + r) mod 2^K``.

    Args:
        xq: Signed quantized activations, int64 ``(T, n)``.
        r: One-time mask, int64 ``(T, n)`` uniform in ``[0, 2^K)``. Must never be reused.
        params: Ring parameters.
    """
    if xq.shape != r.shape:
        raise ShapeError(f"activation shape {tuple(xq.shape)} != mask shape {tuple(r.shape)}")
    if xq.dtype != torch.int64 or r.dtype != torch.int64:
        raise ShapeError("xq and r must be int64")
    if xq.numel():
        lo, hi = torch.aminmax(xq)
        if int(lo) < -params.qmax or int(hi) > params.qmax:
            # Outside the A-bit range the ring headroom (and so exact unmasking) is not guaranteed.
            raise RingOverflowError(f"quantized activations exceed |xq| <= qmax={params.qmax}")
    return (xq + r) & params.mask


def check_output_range(yq: torch.Tensor, params: RingParams, in_features: int) -> None:
    """Raise :class:`IntegrityError` if any decoded output exceeds the genuine bound.

    A correct product satisfies ``|yq| <= qmax * 127 * n``. A tampered result whose error has a high
    2-adic valuation (the case Freivalds checks catch least often) lands far outside that bound.
    """
    bound = params.output_bound(in_features)
    if yq.numel() and int(yq.abs().max()) > bound:
        raise IntegrityError(
            "decoded linear output outside the representable range (corrupt or tampered result)",
            details={"bound": bound, "ring": params.name},
        )


def unmask_outputs(
    y_hat: torch.Tensor,
    wr: torch.Tensor,
    params: RingParams,
    in_features: int | None = None,
) -> torch.Tensor:
    """Return the exact signed product ``yq = centered((y_hat - W r) mod 2^K)``.

    Args:
        y_hat: Service result ``W8 x_hat mod 2^K``, int64 ``(T, m)``.
        wr: Precomputed ``W8 r mod 2^K`` for the same mask rows, int64 ``(T, m)``.
        params: Ring parameters.
        in_features: If given, also run :func:`check_output_range`.
    """
    if y_hat.shape != wr.shape:
        raise ShapeError(f"result shape {tuple(y_hat.shape)} != W r shape {tuple(wr.shape)}")
    yq = center(y_hat - wr, params)
    if in_features is not None:
        check_output_range(yq, params, in_features)
    return yq


def unmask_outputs_u64(
    y_hat: np.ndarray,
    wr: np.ndarray,
    params: RingParams,
    in_features: int | None = None,
) -> np.ndarray:
    """:func:`unmask_outputs` on uint64/int64 arrays; returns the centered int64 product ``yq``.

    Same result and the same output range check as :func:`unmask_outputs` (gateway hot path).
    """
    if y_hat.shape != wr.shape:
        raise ShapeError(f"result shape {y_hat.shape} != W r shape {wr.shape}")
    mask = np.uint64(params.mask)
    d = (y_hat.view(np.uint64) - wr.view(np.uint64)) & mask
    yq = d.view(np.int64)
    yq -= (d >= np.uint64(params.half)).astype(np.int64) * np.int64(params.modulus)
    if in_features is not None and yq.size:
        bound = params.output_bound(in_features)
        if int(np.abs(yq).max()) > bound:
            raise IntegrityError(
                "decoded linear output outside the representable range (corrupt or tampered result)",
                details={"bound": bound, "ring": params.name},
            )
    return yq
