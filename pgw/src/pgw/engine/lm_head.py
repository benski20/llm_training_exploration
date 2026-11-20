"""Deterministic float32 LM head: ``logits = hn @ lm_head^T`` with a fixed summation order.

torch's float32 GEMV / GEMM picks its summation order by shape (a multi-row product differs from
the per-row one in the last bits) and CPU features, so batching rows would change logits. This
product fixes the order instead, so every output row is bit-identical whatever the number of rows
computed together, the thread count and the CPU:

1. 16 partial sums ``acc[l]`` starting at ``+0.0``;
2. for each full chunk of 16 columns, in order: ``acc[l] = acc[l] + h[16c+l] * w[16c+l]`` (product
   rounded to float32, then the sum: no FMA);
3. the last ``D % 16`` columns into ``acc[0 .. D % 16]`` the same way;
4. the tree ``acc[:8] + acc[8:]``, then ``[:4] + [4:]``, ``[:2] + [2:]``, ``[0] + [1]``.

:func:`lm_head_logits` uses the native kernel ``pgw_fast.lm_head`` (one pass over the weights for
all rows, multi-threaded) when it is built, else :func:`lm_head_reference`, the same order in torch
(slower; bit-identical, see ``tests/native/test_lm_head.py``).
"""

from __future__ import annotations

import torch

from pgw.core import native

__all__ = ["LANES", "lm_head_logits", "lm_head_reference", "native_lm_head"]

#: Partial sums per dot product (part of the fixed order; the native kernel uses the same).
LANES = 16


def native_lm_head() -> bool:
    """True when the native kernel is loaded (``PGW_NATIVE=0`` disables it)."""
    return native.native_available("lm_head")


def _tree(acc: torch.Tensor) -> torch.Tensor:
    while acc.shape[-1] > 1:
        half = acc.shape[-1] // 2
        acc = acc[..., :half] + acc[..., half:]
    return acc[..., 0]


def lm_head_reference(h: torch.Tensor, w: torch.Tensor, block: int = 4096) -> torch.Tensor:
    """``h (R, D) @ w (V, D)^T`` in the fixed order with plain torch element-wise ops (any device).

    Bit-identical to the native kernel. Processes ``block`` weight rows at a time.
    """
    if h.ndim != 2 or w.ndim != 2 or h.shape[1] != w.shape[1]:
        raise ValueError(f"lm_head shapes do not match: h {tuple(h.shape)}, w {tuple(w.shape)}")
    if h.dtype != torch.float32 or w.dtype != torch.float32:
        raise TypeError("lm_head needs float32 h and w")
    r, d = int(h.shape[0]), int(h.shape[1])
    v = int(w.shape[0])
    out = torch.empty((r, v), dtype=torch.float32, device=h.device)
    full = d // LANES * LANES
    tail = d - full
    hc = h[:, None, :]
    for v0 in range(0, v, block):
        wb = w[v0 : v0 + block][None]
        acc = torch.zeros((r, wb.shape[1], LANES), dtype=torch.float32, device=h.device)
        tmp = torch.empty_like(acc)
        for k in range(0, full, LANES):
            torch.mul(hc[..., k : k + LANES], wb[..., k : k + LANES], out=tmp)
            acc += tmp
        if tail:
            acc[..., :tail] += hc[..., full:] * wb[..., full:]
        out[:, v0 : v0 + block] = _tree(acc)
    return out


def lm_head_logits(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """float32 logits ``(R, V)`` of normalized rows ``h (R, D)`` against the head ``w (V, D)``.

    Row ``i`` of the result only depends on row ``i`` of ``h`` (bit for bit), so rows of different
    sequences and of a speculative verify pass can share one pass over ``w``.
    """
    if (native_lm_head() and h.device.type == "cpu" and w.device.type == "cpu"
            and h.dtype == torch.float32 and w.dtype == torch.float32):
        hn = h.detach().contiguous().numpy()
        wn = w.detach().contiguous().numpy()  # no copy for a contiguous head (the model keeps it so)
        return torch.from_numpy(native.fast.lm_head(hn, wn))
    return lm_head_reference(h.detach(), w.detach())
