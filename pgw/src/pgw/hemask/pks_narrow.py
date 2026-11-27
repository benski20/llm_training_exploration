"""pks-v3 with a narrow message ring (research prototype; nothing uses it; ``consumer-bandwidth.md`` section 5).

With narrow wire rings (:mod:`pgw.core.narrow`) a mask is needed only modulo ``2^k`` with ``k = max_j w_j <= C + 1
= 28`` at K32A12C27. The customer then encodes its upload with the larger scale ``2^(q - k)`` (``q = 56`` unchanged)
and decrypts with ``round(2^k v / q1) mod 2^k``; nothing else changes on either side, except that the provider may
round its reply ``c0`` to ``yb < 36`` bits. The reference functions of :mod:`pgw.hemask.pks` are already written in
``params.k`` / ``params.yb``; this class only lifts the K32 / 36-bit guard of :class:`pgw.hemask.pks.PKSParams`.

Why it is the same scheme security-wise: the upload is the same RLWE instance (same ``n``, ``q``, ``sigma``, fresh
ternary secret), the reply key is unchanged, the provider's view is identically distributed for any message, and
the message scale is a customer-private choice. Noise: the budget is ``q1 / 2^(k+1)``, so lowering ``k`` from 32
to 28 gives 4 bits of room; the reply-rounding term ``q1 / 2^(yb+1)`` takes them back as ``yb`` drops.
:func:`smallest_reply_bits` picks the smallest ``yb`` whose bound keeps at least the production margin of v3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pgw.core.errors import ConfigError
from pgw.hemask.pks import PKSParams

__all__ = ["PKSNarrowParams", "smallest_reply_bits"]


@dataclass(frozen=True)
class PKSNarrowParams(PKSParams):
    """:class:`PKSParams` with a message ring of ``k`` bits (16..32) and replies of ``yb`` bits (k..36)."""

    k: int = 28
    yb: int = 30

    def __post_init__(self) -> None:
        if not (16 <= self.k <= 32) or self.q != 56 or not (self.k <= self.yb <= 36):
            raise ConfigError("narrow pks: 16 <= k <= 32, q = 2^56, k <= yb <= 36")


def margin_bits(params: PKSParams, max_row_l2: float) -> float:
    """``log2(budget / noise bound)``: the noise margin of a parameter set at that weight-row norm."""
    return math.log2(params.budget()) - math.log2(params.noise_bound(max_row_l2))


def smallest_reply_bits(k: int, max_row_l2: float, ref: PKSParams | None = None) -> int:
    """Smallest reply width ``yb`` at message ring ``k`` whose margin is at least the production v3 margin
    (K32, 36-bit replies) at the same weight-row norm."""
    want = margin_bits(ref or PKSParams(), max_row_l2)
    for yb in range(k, 37):
        if margin_bits(PKSNarrowParams(k=k, yb=yb), max_row_l2) >= want:
            return yb
    return 36
