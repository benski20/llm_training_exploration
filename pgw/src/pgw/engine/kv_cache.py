"""Per-sequence KV cache (plaintext, gateway memory only; never leaves the customer side)."""

from __future__ import annotations

from typing import Any

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
        self._hi = 0  # highest committed length since the last clear (positions still hold valid K/V)
        self._np: tuple[Any, Any, Any] | None = None  # (storage tensor, k numpy view, v numpy view)

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

    def numpy_views(self) -> tuple[Any, Any]:
        """``(k, v)`` numpy views ``(layers, capacity, kvh, hd)`` of the storage (CPU only; the native float
        path writes a pass's keys and values into them; a bfloat16 cache as ``uint16`` bit patterns). Valid
        until :meth:`reserve` grows the storage."""
        hit = self._np
        if hit is None or hit[0] is not self._k:
            if self.dtype == torch.bfloat16:  # numpy has no bfloat16: the bit patterns as uint16
                hit = self._np = (self._k, self._k.view(torch.uint16).numpy(), self._v.view(torch.uint16).numpy())
            else:
                hit = self._np = (self._k, self._k.numpy(), self._v.numpy())
        return hit[1], hit[2]

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

    def layer_storage(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(k, v)`` views ``(capacity, kvh, hd)`` of one layer's whole storage (tree attention writes
        ancestors and node rows into it directly). Valid until :meth:`reserve` grows the storage."""
        if not 0 <= layer < self.num_layers:
            raise ConfigError(f"layer {layer} out of range [0, {self.num_layers})")
        return self._k[layer], self._v[layer]

    def keep_path(self, base: int, nodes: list[int], copies: int = 0) -> None:
        """Token-tree speculation: after a tree pass stored node ``i`` at ``base + i``, move the accepted
        root-to-leaf path ``nodes`` (increasing node indices) to ``base, base + 1, ...`` in every layer.
        The caller then sets the length to ``base + len(nodes)``. ``copies``: make exactly this many
        position copies (at least ``len(nodes) - 1``; the rest copy a position onto itself), so the work does
        not depend on how long the accepted path is."""
        prev = -1
        top = max(self._hi, self._len)
        for e, n in enumerate(nodes):
            n = int(n)
            if n < e or n <= prev or base + n >= top:
                raise ConfigError(f"keep_path: bad node {n} at depth {e}")
            prev = n
        total = max(int(copies), len(nodes) - 1)
        for e in range(1, total + 1):
            n = int(nodes[e]) if e < len(nodes) else min(e, top - 1 - base)
            self._k[:, base + e] = self._k[:, base + n]
            self._v[:, base + e] = self._v[:, base + n]

    def advance(self, tokens: int) -> None:
        """Commit ``tokens`` new positions after all layers appended."""
        if tokens < 0 or self._len + tokens > self.max_len:
            raise ContextOverflowError(f"cannot advance by {tokens} from {self._len} (max {self.max_len})")
        if self._len + tokens > self.capacity:
            raise ConfigError("advance() beyond written positions")
        self._len += tokens
        self._hi = max(self._hi, self._len)

    def set_length(self, length: int) -> None:
        """Move the committed length within positions written since the last :meth:`clear`.

        Speculative decoding verifies several drafted positions in one pass and then walks the
        committed length forward position by position while it accepts them.
        """
        if not 0 <= length <= self._hi:
            raise ConfigError(f"set_length {length} outside [0, {self._hi}]")
        self._len = length

    def truncate(self, length: int) -> None:
        """Drop positions ``>= length`` (used for stop-sequence rollback / prefix reuse)."""
        if not 0 <= length <= self._len:
            raise ConfigError(f"truncate length {length} outside [0, {self._len}]")
        self._len = length
        self._hi = length

    def clear(self) -> None:
        """Zero the storage (plaintext hygiene) and reset the length."""
        self._k.zero_()
        self._v.zero_()
        self._len = 0
        self._hi = 0

    def nbytes(self) -> int:
        """Memory allocated."""
        return int(self._k.numel() * self._k.element_size() * 2)
