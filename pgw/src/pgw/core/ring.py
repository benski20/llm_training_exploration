"""Ring parameters and fixed-point arithmetic over Z_{2^K}.

Everything the masked protocol computes lives in the ring Z_{2^K}. Values are held in
``torch.int64`` tensors in canonical form ``[0, 2^K)``; signed ("centered") values are held in
``[-2^(K-1), 2^(K-1))``.

Quantization conventions (normative, see SPEC.md section 5):

* Weights: int8, symmetric, per output row, ``sw[j] = max_i |W[j, i]| / 127`` (zero rows use 1.0).
  Weights never take the value -128.
* Activations: signed A-bit, symmetric, per token (row), ``sx[t] = max_i |x[t, i]| / qmax`` with
  ``qmax = 2^(A-1) - 1`` (zero rows use 1.0).
* Linear output: ``y[t, j] = yq[t, j] * sx[t] * sw[j] (+ bias[j])`` where ``yq = xq @ W8^T`` exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from pgw.core.errors import ConfigError, RingOverflowError, ShapeError

__all__ = [
    "DEFAULT_RING",
    "RING_K32_A8",
    "RING_K40_A12",
    "RING_K48_A20",
    "RingParams",
    "balanced_digits",
    "center",
    "dequantize_outputs",
    "quantize_activations",
    "quantize_weight_rows",
    "ring_add",
    "ring_sub",
    "to_ring",
]

#: Largest supported ring. Values stay in int64 and limb shifts may wrap mod 2^64, which is fine
#: for any K <= 64, but we keep a margin so ``a + b`` of two canonical values never overflows.
_MAX_K = 56
_MIN_K = 16


@dataclass(frozen=True, slots=True)
class RingParams:
    """Ring size ``K`` (bits) and activation precision ``A`` (bits, signed).

    Attributes:
        k: Ring bit width; values live in ``Z_{2^k}``. Must be a multiple of 8 in [16, 56].
        a: Signed activation bit width; quantized activations satisfy ``|xq| <= 2^(a-1) - 1``.
    """

    k: int
    a: int

    WEIGHT_QMAX: ClassVar[int] = 127

    def __post_init__(self) -> None:
        if not (_MIN_K <= self.k <= _MAX_K) or self.k % 8 != 0:
            raise ConfigError(f"ring K must be a multiple of 8 in [{_MIN_K}, {_MAX_K}], got {self.k}")
        if not (2 <= self.a <= self.k - 8):
            raise ConfigError(f"activation bits A must be in [2, K-8], got A={self.a} K={self.k}")

    # ---- derived constants -------------------------------------------------------------------
    @property
    def modulus(self) -> int:
        """``2^K``."""
        return 1 << self.k

    @property
    def mask(self) -> int:
        """``2^K - 1`` (bitmask for reduction)."""
        return (1 << self.k) - 1

    @property
    def half(self) -> int:
        """``2^(K-1)``: centered values lie in ``[-half, half)``."""
        return 1 << (self.k - 1)

    @property
    def qmax(self) -> int:
        """Largest quantized activation magnitude ``2^(A-1) - 1``."""
        return (1 << (self.a - 1)) - 1

    @property
    def limbs(self) -> int:
        """Number of int8 limb GEMMs per ring matmul (``K / 8``)."""
        return self.k // 8

    @property
    def bytes_per_element(self) -> int:
        """Bytes per ring element on the wire and in the mask store (``K / 8``)."""
        return self.k // 8

    @property
    def name(self) -> str:
        """Canonical name, e.g. ``"K40A12"`` (used on the wire and in mask store headers)."""
        return f"K{self.k}A{self.a}"

    def output_bound(self, in_features: int) -> int:
        """Largest possible ``|yq|`` for a genuine product with inner dimension ``in_features``."""
        return self.qmax * self.WEIGHT_QMAX * in_features

    def max_in_features(self) -> int:
        """Largest inner dimension for which decoding is unambiguous (``output_bound < 2^(K-1)``)."""
        return (self.half - 1) // (self.qmax * self.WEIGHT_QMAX)

    def check_in_features(self, in_features: int) -> None:
        """Raise :class:`RingOverflowError` if a linear with this inner dimension could wrap."""
        if in_features > self.max_in_features():
            raise RingOverflowError(
                f"{self.name}: inner dimension {in_features} exceeds the decodable maximum "
                f"{self.max_in_features()}; use a larger ring"
            )

    def headroom_bits(self, in_features: int) -> float:
        """Bits between the largest genuine output and ``2^(K-1)``; drives integrity-check soundness."""
        return (self.k - 1) - math.log2(max(1, self.output_bound(in_features)))

    @classmethod
    def from_name(cls, name: str) -> RingParams:
        """Parse ``"K40A12"`` (case-insensitive)."""
        s = name.strip().upper()
        if not s.startswith("K") or "A" not in s:
            raise ConfigError(f"bad ring name {name!r}; expected e.g. 'K40A12'")
        k_str, a_str = s[1:].split("A", 1)
        try:
            return cls(int(k_str), int(a_str))
        except ValueError as exc:
            raise ConfigError(f"bad ring name {name!r}") from exc

    def __str__(self) -> str:
        return self.name


RING_K32_A8 = RingParams(32, 8)
RING_K40_A12 = RingParams(40, 12)
RING_K48_A20 = RingParams(48, 20)
#: Default ring: 40-bit ring with 12-bit activations (A8 measured too lossy).
DEFAULT_RING = RING_K40_A12


# ---- quantization ------------------------------------------------------------------------------
def quantize_weight_rows(w: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float weight matrix ``(m, n)`` to int8 per output row.

    Returns:
        ``(w8, sw)`` where ``w8`` is ``int8 (m, n)`` with values in ``[-127, 127]`` and ``sw`` is
        ``float32 (m,)`` such that ``w ~= w8 * sw[:, None]``.
    """
    wt = torch.as_tensor(w).detach().to(torch.float64)
    if wt.ndim != 2:
        raise ShapeError(f"weight must be 2-D, got shape {tuple(wt.shape)}")
    amax = wt.abs().amax(dim=1)
    sw = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    w8 = torch.round(wt / sw[:, None]).clamp_(-127, 127).to(torch.int8)
    return w8, sw.to(torch.float32)


def quantize_activations(x: torch.Tensor, params: RingParams) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations ``(T, n)`` per token to signed A-bit integers.

    Returns:
        ``(xq, sx)``: ``xq`` int64 ``(T, n)`` with ``|xq| <= qmax`` and ``sx`` float64 ``(T,)``.
    """
    if x.ndim != 2:
        raise ShapeError(f"activations must be 2-D (tokens, features), got {tuple(x.shape)}")
    xf = x.detach().to(torch.float64)
    if not torch.isfinite(xf).all():
        raise RingOverflowError("non-finite activation values cannot be quantized")
    amax = xf.abs().amax(dim=1)
    sx = torch.where(amax > 0, amax / params.qmax, torch.ones_like(amax))
    xq = torch.round(xf / sx[:, None]).clamp_(-params.qmax, params.qmax).to(torch.int64)
    return xq, sx


def dequantize_outputs(
    yq: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rescale exact integer outputs ``(T, m)`` back to floats and add the bias."""
    y = yq.to(torch.float64) * sx.to(torch.float64)[:, None] * sw.to(torch.float64)[None, :]
    if bias is not None:
        y = y + bias.to(torch.float64)[None, :]
    return y.to(dtype)


# ---- ring helpers ------------------------------------------------------------------------------
def to_ring(v: torch.Tensor, params: RingParams) -> torch.Tensor:
    """Reduce signed int64 values to canonical ``[0, 2^K)``."""
    return v.to(torch.int64) & params.mask


def center(v: torch.Tensor, params: RingParams) -> torch.Tensor:
    """Map ring values to centered representatives ``[-2^(K-1), 2^(K-1))``."""
    v = v.to(torch.int64) & params.mask
    return torch.where(v >= params.half, v - params.modulus, v)


def ring_add(a: torch.Tensor, b: torch.Tensor, params: RingParams) -> torch.Tensor:
    """``(a + b) mod 2^K``."""
    return (a + b) & params.mask


def ring_sub(a: torch.Tensor, b: torch.Tensor, params: RingParams) -> torch.Tensor:
    """``(a - b) mod 2^K``."""
    return (a - b) & params.mask


def balanced_digits(v: torch.Tensor, k: int, bits: int = 8) -> list[torch.Tensor]:
    """Split canonical ring values into balanced base-``2^bits`` digits.

    Returns ``ceil(k / bits)`` int64 tensors ``d_i`` with ``-2^(bits-1) <= d_i < 2^(bits-1)`` and
    ``sum_i d_i * 2^(bits*i) == v (mod 2^k)``. For ``bits == 8`` the digits fit in int8 (callers cast).
    """
    if not 2 <= bits <= 62:
        raise ConfigError(f"digit width must be in [2, 62], got {bits}")
    n = -(-k // bits)
    base_mask = (1 << bits) - 1
    half = 1 << (bits - 1)
    full = 1 << bits
    out: list[torch.Tensor] = []
    cur = v.to(torch.int64)
    for _ in range(n):
        d = cur & base_mask
        d = torch.where(d >= half, d - full, d)
        out.append(d)
        cur = (cur - d) >> bits
    return out
