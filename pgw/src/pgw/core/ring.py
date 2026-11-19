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
        cap: Per-call output bound ``2^cap`` (0 = none). Without a cap the ring must hold the worst
            case ``qmax * 127 * n``. With a cap (preset ``K32A12C27``), the gateway quantizes each
            token with the most activation bits ``<= A`` for which the exact Cauchy-Schwarz bound
            ``max_j ||W8_j||_2 * ||xq||_2 <= 2^cap`` holds, so every genuine output fits ``2^cap``
            for any width ``n``. The range check and Freivalds soundness use ``2^cap``.
    """

    k: int
    a: int
    cap: int = 0

    WEIGHT_QMAX: ClassVar[int] = 127

    def __post_init__(self) -> None:
        if not (_MIN_K <= self.k <= _MAX_K) or self.k % 8 != 0:
            raise ConfigError(f"ring K must be a multiple of 8 in [{_MIN_K}, {_MAX_K}], got {self.k}")
        if not (2 <= self.a <= self.k - 8):
            raise ConfigError(f"activation bits A must be in [2, K-8], got A={self.a} K={self.k}")
        if self.cap and not (16 <= self.cap <= self.k - 3):
            raise ConfigError(f"output cap C must be 0 or in [16, K-3], got C={self.cap} K={self.k}")

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
        return f"K{self.k}A{self.a}" + (f"C{self.cap}" if self.cap else "")

    def output_bound(self, in_features: int) -> int:
        """Largest possible ``|yq|`` for a genuine product with inner dimension ``in_features``."""
        worst = self.qmax * self.WEIGHT_QMAX * in_features
        return min(worst, 1 << self.cap) if self.cap else worst

    def max_in_features(self) -> int:
        """Largest inner dimension for which decoding is unambiguous (``output_bound < 2^(K-1)``).

        Capped rings decode any width (the gateway enforces the cap per call)."""
        if self.cap:
            return 1 << 31
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
        """Parse ``"K40A12"`` or a capped ring such as ``"K32A12C27"`` (case-insensitive)."""
        import re

        m = re.fullmatch(r"K(\d+)A(\d+)(?:C(\d+))?", name.strip().upper())
        if m is None:
            raise ConfigError(f"bad ring name {name!r}; expected e.g. 'K40A12' or 'K32A12C27'")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))

    def __str__(self) -> str:
        return self.name


RING_K32_A8 = RingParams(32, 8)
RING_K40_A12 = RingParams(40, 12)
RING_K48_A20 = RingParams(48, 20)
#: 32-bit ring with 12-bit activations and a per-call output cap of 2^27 (adaptive bits; lossless on
#: Qwen2.5-0.5B, 1.5B and 7B; identical perplexity): 20% less traffic, service work and mask storage than
#: K40A12. The default since 2026-11-19.
RING_K32_A12_C27 = RingParams(32, 12, 27)
#: Default ring: 40-bit ring with 12-bit activations (A8 measured too lossy).
DEFAULT_RING = RING_K32_A12_C27


# ---- quantization ------------------------------------------------------------------------------

#: Up to this many elements the CPU quantize/dequantize helpers use single-threaded numpy.
_SMALL_NUMEL = 65_536


def quantize_weight_rows(w: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float weight matrix ``(m, n)`` to int8 per output row.

    Returns:
        ``(w8, sw)`` where ``w8`` is ``int8 (m, n)`` with values in ``[-127, 127]`` and ``sw`` is
        ``float32 (m,)`` such that ``w ~= w8 * sw[:, None]``.
    """
    wt = torch.as_tensor(w).detach().to(torch.float64)
    if wt.ndim != 2:
        raise ShapeError(f"weight must be 2-D, got shape {tuple(wt.shape)}")
    if not torch.isfinite(wt).all():
        raise RingOverflowError("weights contain NaN/Inf and cannot be quantized")
    amax = wt.abs().amax(dim=1)
    sw = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    w8 = torch.round(wt / sw[:, None]).clamp_(-127, 127).to(torch.int8)
    return w8, sw.to(torch.float32)


def quantize_activations(
    x: torch.Tensor, params: RingParams, wnorm: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations ``(T, n)`` per token to signed A-bit integers.

    For a capped ring (``params.cap``), ``wnorm`` is the largest Euclidean norm of the linear's int8
    weight rows. Tokens whose exact bound ``wnorm * ||xq||_2`` would exceed ``2^cap`` are quantized
    again with fewer bits (each token independently), so every genuine output satisfies
    ``|yq| <= 2^cap``. Without ``wnorm`` a capped ring requires the worst case to fit the cap.

    Returns:
        ``(xq, sx)``: ``xq`` int64 ``(T, n)`` with ``|xq| <= qmax`` and ``sx`` float64 ``(T,)``.
    """
    if x.ndim != 2:
        raise ShapeError(f"activations must be 2-D (tokens, features), got {tuple(x.shape)}")
    if params.cap:
        if wnorm is None:
            if params.qmax * params.WEIGHT_QMAX * int(x.shape[1]) > (1 << params.cap):
                raise ConfigError(f"{params.name}: the weight-row norm is needed to quantize width {x.shape[1]}")
        else:
            return _quantize_capped(x, params, float(wnorm))
    if x.device.type == "cpu" and x.numel() <= _SMALL_NUMEL:
        # Decode-sized inputs: single-threaded numpy avoids torch intra-op threading overhead.
        # Same IEEE operations in the same order (float64 division, round-half-even), so the
        # result is bit-identical to the torch path below.
        xf_np = x.detach().numpy().astype(np.float64)
        if not np.isfinite(xf_np).all():
            raise RingOverflowError("non-finite activation values cannot be quantized")
        amax_np = np.abs(xf_np).max(axis=1)
        sx_np = np.where(amax_np > 0, amax_np / params.qmax, 1.0)
        xq_np = np.rint(xf_np / sx_np[:, None])
        np.clip(xq_np, -params.qmax, params.qmax, out=xq_np)
        return torch.from_numpy(xq_np.astype(np.int64)), torch.from_numpy(sx_np)
    xf = x.detach().to(torch.float64)
    if not torch.isfinite(xf).all():
        raise RingOverflowError("non-finite activation values cannot be quantized")
    amax = xf.abs().amax(dim=1)
    sx = torch.where(amax > 0, amax / params.qmax, torch.ones_like(amax))
    xq = torch.round(xf / sx[:, None]).clamp_(-params.qmax, params.qmax).to(torch.int64)
    return xq, sx


#: Relative slack on the float64 norm product (the bound must hold exactly for integers).
_NORM_SLACK = 1.0 + 1e-9


def _quantize_capped(x: torch.Tensor, params: RingParams, wnorm: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token quantization with the most bits ``<= A`` such that ``wnorm * ||xq||_2 <= 2^cap``."""
    xf = x.detach().to("cpu", torch.float64).numpy()
    if not np.isfinite(xf).all():
        raise RingOverflowError("non-finite activation values cannot be quantized")
    limit = float(1 << params.cap)
    amax = np.abs(xf).max(axis=1)
    rows = xf.shape[0]
    xq = np.zeros(xf.shape, dtype=np.int64)
    sx = np.ones(rows, dtype=np.float64)
    todo = np.arange(rows)
    for bits in range(params.a, 1, -1):
        qmax = (1 << (bits - 1)) - 1
        s_b = np.where(amax[todo] > 0, amax[todo] / qmax, 1.0)
        q = np.clip(np.rint(xf[todo] / s_b[:, None]), -qmax, qmax)
        fits = np.sqrt((q * q).sum(axis=1)) * wnorm * _NORM_SLACK <= limit
        idx = todo[fits]
        xq[idx] = q[fits].astype(np.int64)
        sx[idx] = s_b[fits]
        todo = todo[~fits]
        if todo.size == 0:
            break
    if todo.size:  # pragma: no cover - only for absurd weight norms (1-bit activations still too big)
        raise RingOverflowError(f"{params.name}: activations cannot be bounded by 2^{params.cap} for this linear")
    return torch.from_numpy(xq), torch.from_numpy(sx)


def dequantize_outputs(
    yq: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rescale exact integer outputs ``(T, m)`` back to floats and add the bias."""
    if yq.device.type == "cpu" and yq.numel() <= _SMALL_NUMEL and sx.device.type == "cpu" and sw.device.type == "cpu":
        # Bit-identical numpy path for decode-sized outputs (same float64 ops in the same order).
        y_np = yq.numpy().astype(np.float64) * sx.numpy().astype(np.float64)[:, None]
        y_np = y_np * sw.numpy().astype(np.float64)[None, :]
        if bias is not None:
            y_np = y_np + bias.numpy().astype(np.float64)[None, :]
        return torch.from_numpy(y_np).to(dtype)
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
