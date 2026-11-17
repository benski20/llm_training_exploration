"""Per-sequence KV cache (plaintext, gateway memory only; never leaves the customer side)."""

from __future__ import annotations

import torch

from pgw.core.errors import ConfigError, ErrorCode, PGWError

__all__ = ["ContextOverflowError", "KVCache"]


class ContextOverflowError(PGWError):
    """The sequence would exceed the cache capacity (context length)."""

    code = ErrorCode.BAD_SHAPE


class KVCache:
    """Key/value cache for one sequence.

    Shapes per layer: ``(seq_len, num_kv_heads, head_dim)`` (keys stored *after* RoPE).

    Writes for a forward pass go to positions ``[length, length + T)`` in every layer, but
    :attr:`length` only moves when :meth:`advance` commits them. A forward pass that fails halfway
    therefore leaves the cache logically unchanged; its partial writes are overwritten next time.

    Storage grows geometrically up to ``max_len`` so a large context limit does not preallocate
    memory that a short request never uses.

    Args:
        num_layers, num_kv_heads, head_dim: From :class:`pgw.modelpack.ArchConfig`.
        max_len: Capacity in positions (raises :class:`ContextOverflowError` beyond it).
        dtype, device: Storage dtype/device.
        initial_len: Initial allocation (defaults to ``min(max_len, 256)``).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        *,
        initial_len: int | None = None,
    ):
        if min(num_layers, num_kv_heads, head_dim, max_len) <= 0:
            raise ConfigError("KVCache dimensions and max_len must be positive")
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.dtype = dtype
        self.device = torch.device(device)
        cap = min(max_len, initial_len if initial_len is not None else 256)
        self._k = torch.zeros((num_layers, max(cap, 1), num_kv_heads, head_dim), dtype=dtype, device=self.device)
        self._v = torch.zeros_like(self._k)
        self._len = 0

    @property
    def length(self) -> int:
        """Number of committed positions."""
        return self._len

    @property
    def capacity(self) -> int:
        """Currently allocated positions."""
        return int(self._k.shape[1])

    @property
    def remaining(self) -> int:
        """Positions that can still be appended."""
        return self.max_len - self._len

    def reserve(self, total: int) -> None:
        """Ensure storage for ``total`` positions (``<= max_len``)."""
        if total > self.max_len:
            raise ContextOverflowError(
                f"sequence length {total} exceeds the context capacity {self.max_len}",
                details={"length": total, "max_len": self.max_len},
            )
        if total <= self.capacity:
            return
        new_cap = min(self.max_len, max(total, 2 * self.capacity))
        shape = (self.num_layers, new_cap, self.num_kv_heads, self.head_dim)
        k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        v = torch.zeros_like(k)
        k[:, : self._len] = self._k[:, : self._len]
        v[:, : self._len] = self._v[:, : self._len]
        self._k, self._v = k, v

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``(T, kvh, hd)`` keys/values for ``layer``; return full ``(length + T, kvh, hd)`` views.

        The position counter advances once per forward pass via :meth:`advance`, not per layer.
        """
        if not 0 <= layer < self.num_layers:
            raise ConfigError(f"layer {layer} out of range [0, {self.num_layers})")
        if k.shape != v.shape or k.ndim != 3 or k.shape[1:] != (self.num_kv_heads, self.head_dim):
            raise ConfigError(f"bad k/v shapes {tuple(k.shape)} {tuple(v.shape)}")
        t = int(k.shape[0])
        end = self._len + t
        self.reserve(end)
        self._k[layer, self._len : end] = k.to(self.dtype)
        self._v[layer, self._len : end] = v.to(self.dtype)
        return self._k[layer, :end], self._v[layer, :end]

    def advance(self, tokens: int) -> None:
        """Commit ``tokens`` new positions after all layers appended."""
        if tokens < 0 or self._len + tokens > self.max_len:
            raise ContextOverflowError(f"cannot advance by {tokens} from {self._len} (max {self.max_len})")
        if self._len + tokens > self.capacity:
            raise ConfigError("advance() beyond written positions")
        self._len += tokens

    def truncate(self, length: int) -> None:
        """Drop positions ``>= length`` (used for stop-sequence rollback / prefix reuse)."""
        if not 0 <= length <= self._len:
            raise ConfigError(f"truncate length {length} outside [0, {self._len}]")
        self._len = length

    def clear(self) -> None:
        """Zero the storage (plaintext hygiene) and reset the length."""
        self._k.zero_()
        self._v.zero_()
        self._len = 0

    def nbytes(self) -> int:
        """Memory allocated."""
        return int(self._k.numel() * self._k.element_size() * 2)
