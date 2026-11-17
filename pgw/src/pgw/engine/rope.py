"""Rotary position embeddings (plain, linear, Llama-3 and YaRN scaling), matching HF transformers.

Only the frequency computation differs between variants; application is the usual
``x * cos + rotate_half(x) * sin`` over the full head dimension (``cat(freqs, freqs)`` layout).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pgw.core.errors import ConfigError

__all__ = ["RotaryEmbedding", "apply_rope", "rope_inv_freq", "rotate_half"]

SUPPORTED_ROPE_TYPES = ("default", "linear", "llama3", "yarn")


def _base_inv_freq(theta: float, dim: int) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float32) / dim))


def rope_inv_freq(
    head_dim: int,
    theta: float,
    scaling: dict[str, Any] | None = None,
    max_position_embeddings: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Return ``(inv_freq float32 (head_dim/2,), attention_scaling)`` for a RoPE config.

    Raises:
        ConfigError: unsupported ``rope_type``.
    """
    if head_dim % 2:
        raise ConfigError(f"head_dim must be even for RoPE, got {head_dim}")
    scaling = dict(scaling or {})
    rtype = scaling.get("rope_type", scaling.get("type", "default")) or "default"
    if float(scaling.get("partial_rotary_factor", 1.0)) != 1.0:
        raise ConfigError("partial_rotary_factor != 1 is not supported")
    dim = head_dim
    inv = _base_inv_freq(theta, dim)
    if rtype == "default":
        return inv, 1.0
    if rtype == "linear":
        return inv / float(scaling["factor"]), 1.0
    if rtype == "llama3":
        factor = float(scaling["factor"])
        low = float(scaling["low_freq_factor"])
        high = float(scaling["high_freq_factor"])
        old_ctx = float(scaling["original_max_position_embeddings"])
        low_wl, high_wl = old_ctx / low, old_ctx / high
        wavelen = 2 * math.pi / inv
        inv_l = torch.where(wavelen > low_wl, inv / factor, inv)
        smooth = (old_ctx / wavelen - low) / (high - low)
        smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
        medium = ~(wavelen < high_wl) * ~(wavelen > low_wl)
        return torch.where(medium, smoothed, inv_l), 1.0
    if rtype == "yarn":
        orig = scaling.get("original_max_position_embeddings") or max_position_embeddings
        if orig is None:
            raise ConfigError("yarn rope scaling needs original_max_position_embeddings")
        factor = scaling.get("factor")
        if factor is None:
            if max_position_embeddings is None:
                raise ConfigError("yarn rope scaling needs factor")
            factor = max_position_embeddings / orig
        factor = float(factor)

        def mscale_of(scale: float, m: float = 1.0) -> float:
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

        attn = scaling.get("attention_factor")
        if attn is None:
            ms, msa = scaling.get("mscale"), scaling.get("mscale_all_dim")
            attn = mscale_of(factor, ms) / mscale_of(factor, msa) if ms and msa else mscale_of(factor)
        beta_fast = scaling.get("beta_fast") or 32
        beta_slow = scaling.get("beta_slow") or 1
        truncate = scaling.get("truncate", True)

        def corr_dim(rot: float) -> float:
            return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(theta))

        lo, hi = corr_dim(beta_fast), corr_dim(beta_slow)
        if truncate:
            lo, hi = math.floor(lo), math.ceil(hi)
        lo, hi = max(lo, 0), min(hi, dim - 1)
        if lo == hi:
            hi += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - lo) / (hi - lo), 0, 1)
        extra = 1 - ramp
        pos_freqs = theta ** (torch.arange(0, dim, 2).to(torch.float32) / dim)
        inv_ex = 1.0 / pos_freqs
        inv_in = 1.0 / (factor * pos_freqs)
        return inv_in * (1 - extra) + inv_ex * extra, float(attn)
    raise ConfigError(f"unsupported rope_type {rtype!r}; supported: {SUPPORTED_ROPE_TYPES}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``cat(-x2, x1)`` over the last dimension."""
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x (T, heads, hd)`` with ``cos/sin (T, hd)``."""
    c = cos[:, None, :]
    s = sin[:, None, :]
    return x * c + rotate_half(x) * s


class RotaryEmbedding:
    """Caches cos/sin tables and grows them on demand.

    Args:
        head_dim, theta, scaling, max_position_embeddings: RoPE configuration.
        device: Device of the tables.
    """

    def __init__(
        self,
        head_dim: int,
        theta: float,
        scaling: dict[str, Any] | None = None,
        max_position_embeddings: int | None = None,
        device: str | torch.device = "cpu",
    ):
        self.inv_freq, self.attention_scaling = rope_inv_freq(head_dim, theta, scaling, max_position_embeddings)
        self.inv_freq = self.inv_freq.to(device)
        self.device = torch.device(device)
        self._cos = torch.empty((0, head_dim), device=device)
        self._sin = torch.empty((0, head_dim), device=device)

    def _grow(self, length: int) -> None:
        cap = max(length, 2 * self._cos.shape[0], 256)
        pos = torch.arange(cap, device=self.device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos() * self.attention_scaling
        self._sin = emb.sin() * self.attention_scaling

    def cos_sin(self, start: int, count: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """Tables for positions ``[start, start + count)``."""
        end = start + count
        if end > self._cos.shape[0]:
            self._grow(end)
        return self._cos[start:end].to(dtype), self._sin[start:end].to(dtype)
