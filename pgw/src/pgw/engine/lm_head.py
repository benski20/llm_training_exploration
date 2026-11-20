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

**Instruction set.** The native kernel picks the widest SIMD the CPU has (AVX-512F, AVX2, AVX, SSE2)
and register-blocks several rows of a step against each weight chunk; every choice performs the same
float32 operations per output, so the logits do not depend on it. ``PGW_LM_HEAD_ISA`` (``auto``
default, ``avx512``, ``avx2``, ``avx``, ``sse``, ``scalar``) forces one for A/B measurements; one the
CPU lacks falls back to the best available below it.

**bfloat16 heads.** Open checkpoints (Qwen, Llama, ...) ship their weights in bfloat16, so every
float32 value of the head is exactly a bfloat16 value. :func:`compact_head` then keeps the head in
bfloat16 (half the memory, half the bytes read per token) and both paths widen each weight back to
float32 on load, which is exact: the logits are bit-identical to those of the float32 head.
"""

from __future__ import annotations

import logging
import os

import torch

from pgw.core import native

__all__ = ["LANES", "compact_head", "lm_head_isa", "lm_head_logits", "lm_head_reference", "native_lm_head"]

log = logging.getLogger(__name__)

#: Partial sums per dot product (part of the fixed order; the native kernel uses the same).
LANES = 16

_ISAS = ("auto", "avx512", "avx2", "avx", "sse", "scalar")


def _isa_setting() -> str:
    """``PGW_LM_HEAD_ISA`` (validated; an unknown value means ``auto``)."""
    raw = os.environ.get("PGW_LM_HEAD_ISA", "auto").strip().lower() or "auto"
    if raw not in _ISAS:
        log.warning("PGW_LM_HEAD_ISA=%r is not one of %s; using auto", raw, ", ".join(_ISAS))
        return "auto"
    return raw


#: Requested instruction set of the native kernel (read once at import; tests may override).
ISA = _isa_setting()


def lm_head_isa() -> str | None:
    """The instruction set the native kernel uses here (``None`` without it or with an older build)."""
    fn = getattr(native.fast, "lm_head_isa", None) if native.fast is not None else None
    return str(fn(ISA)) if callable(fn) else None


def native_lm_head() -> bool:
    """True when the native kernel is loaded (``PGW_NATIVE=0`` disables it)."""
    return native.native_available("lm_head")


def compact_head(w: torch.Tensor, chunk_rows: int = 8192) -> torch.Tensor:
    """``w`` as bfloat16 if that is lossless (every value exactly representable), else ``w``.

    Only float32 CPU heads are compacted (the native kernel and the fallback read either dtype and
    give the same bits); the check runs in chunks so it needs no full-size temporary.
    """
    if w.dtype != torch.float32 or w.device.type != "cpu" or w.ndim != 2:
        return w
    bits = w.contiguous().view(torch.int32)
    for i in range(0, int(w.shape[0]), chunk_rows):
        if bool((bits[i : i + chunk_rows] & 0xFFFF).any()):
            return w
    out = w.to(torch.bfloat16)  # exact: the low 16 bits are zero, so this is a truncation
    return out


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
    if h.dtype != torch.float32 or w.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("lm_head needs float32 h and a float32 or bfloat16 w")
    r, d = int(h.shape[0]), int(h.shape[1])
    v = int(w.shape[0])
    out = torch.empty((r, v), dtype=torch.float32, device=h.device)
    full = d // LANES * LANES
    tail = d - full
    hc = h[:, None, :]
    for v0 in range(0, v, block):
        wb = w[v0 : v0 + block].to(torch.float32)[None]  # bfloat16 -> float32 is exact
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

    A CPU head with rows on another device (a GPU model keeps its head on the CPU for the native
    kernel) computes on the CPU and returns CPU logits: the same bits as the per-chunk reference on
    the GPU (element-wise float32 products and sums are correctly rounded on both).
    """
    if w.device.type == "cpu" and h.device.type != "cpu":
        h = h.detach().to("cpu")
    if native_lm_head() and h.device.type == "cpu" and w.device.type == "cpu" and h.dtype == torch.float32:
        hn = h.detach().contiguous().numpy()
        kw = {"isa": ISA} if ISA != "auto" else {}  # older builds take no isa (and pick AVX at best)
        if w.dtype == torch.float32:
            wn = w.detach().contiguous().numpy()  # no copy for a contiguous head (the model keeps it so)
            return torch.from_numpy(native.fast.lm_head(hn, wn, **kw))
        if w.dtype == torch.bfloat16 and native.native_available("lm_head_bf16"):
            wb = w.detach().contiguous().view(torch.int16).numpy().view("uint16")  # the raw bits, no copy
            return torch.from_numpy(native.fast.lm_head_bf16(hn, wb, **kw))
    return lm_head_reference(h.detach(), w.detach())
