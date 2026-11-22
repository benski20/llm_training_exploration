"""The deterministic LM head on a gateway-owned GPU (``lm_head_device = "cuda"``; opt-in).

:func:`pgw.engine.lm_head.lm_head_logits` fixes the summation order of ``hn @ lm_head^T`` (16 partial sums
over chunks of 16 columns, products rounded to float32 then added, no FMA, then a fixed tree). Every step
is a correctly rounded IEEE float32 multiply or add, so the same order on a GPU gives the same bits as the
native CPU kernel. :class:`GpuLmHead` is that order as one Triton kernel:

* the head lives on the device in bfloat16 (exact for bfloat16 checkpoints, as on the CPU) in a chunked
  layout ``(D / 16, V, 16)``, so each step of a program loads one contiguous ``(BLOCK_V, 16)`` tile;
* products and sums are PTX ``mul.rn.f32`` / ``add.rn.f32`` (inline assembly: no contraction into FMA,
  no flush of subnormals), the tail columns ``D % 16`` go into ``acc[0 .. D % 16]`` under a mask, then the
  tree ``acc[:8] + acc[8:]``, ``[:4] + [4:]``, ``[:2] + [2:]``, ``[0] + [1]``;
* each output row depends only on its own input row, so grouping rows changes no bit.

The normalized rows are copied to the device (``R x D`` float32), the logits back to pinned host memory and
returned as a CPU tensor, so everything after the head (sampling, logprobs) is unchanged. Before use the
gateway compares the GPU logits with the CPU kernel's bit for bit on public test rows (Gaussian, large,
subnormal products, zero, signed zeros, infinities) and keeps the CPU on any difference
(:meth:`GpuLmHead.self_check`). Under ``TRITON_INTERPRET=1`` the same kernel runs with numpy float32
operations (the tests' check of the order).

Privacy: the GPU is the gateway's own (as for ``local_linears_device``); nothing here reaches the provider.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

__all__ = ["GpuLmHead", "enable_gpu_lm_head", "gpu_lm_head_unavailable"]

LANES = 16


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


try:  # triton ships with CUDA builds of torch; without it the module imports but GpuLmHead cannot run
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only installs
    triton = None  # type: ignore[assignment]

if triton is not None:

    @triton.jit
    def _mul(a, b, ASM: tl.constexpr):  # type: ignore[no-untyped-def]
        if ASM:
            return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32,
                                             is_pure=True, pack=1)
        else:
            return a * b

    @triton.jit
    def _add(a, b, ASM: tl.constexpr):  # type: ignore[no-untyped-def]
        if ASM:
            return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32,
                                             is_pure=True, pack=1)
        else:
            return a + b

    @triton.jit
    def _half(acc, B: tl.constexpr, W: tl.constexpr, ASM: tl.constexpr):  # type: ignore[no-untyped-def]
        # (B, W) -> (B, W/2): acc[:, :W/2] + acc[:, W/2:]
        t = tl.reshape(acc, (B, 2, W // 2))
        t = tl.permute(t, (0, 2, 1))
        lo, hi = tl.split(t)
        return _add(lo, hi, ASM)

    @triton.jit
    def _lm_head_kernel(h_ptr, w_ptr, out_ptr, V, D, FULL: tl.constexpr, TAIL: tl.constexpr,  # type: ignore[no-untyped-def]
                        BLOCK_V: tl.constexpr, ASM: tl.constexpr):
        pid_v = tl.program_id(0)
        row = tl.program_id(1)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        lanes = tl.arange(0, 16)
        vmask = offs_v < V
        acc = tl.zeros((BLOCK_V, 16), dtype=tl.float32)
        h_row = h_ptr + row * D
        # chunked weights: chunk c is a (V, 16) block at w_ptr + c * V * 16
        for c in range(0, FULL // 16):
            hv = tl.load(h_row + c * 16 + lanes)
            wt = tl.load(w_ptr + c * V * 16 + offs_v[:, None] * 16 + lanes[None, :], mask=vmask[:, None], other=0.0)
            p = _mul(hv[None, :], wt.to(tl.float32), ASM)
            acc = _add(acc, p, ASM)
        if TAIL > 0:
            tmask = lanes < TAIL
            hv = tl.load(h_row + FULL + lanes, mask=tmask, other=0.0)
            wt = tl.load(w_ptr + (FULL // 16) * V * 16 + offs_v[:, None] * 16 + lanes[None, :],
                         mask=vmask[:, None] & tmask[None, :], other=0.0)
            p = _mul(hv[None, :], wt.to(tl.float32), ASM)
            acc = tl.where(tmask[None, :], _add(acc, p, ASM), acc)
        s8 = _half(acc, BLOCK_V, 16, ASM)
        s4 = _half(s8, BLOCK_V, 8, ASM)
        s2 = _half(s4, BLOCK_V, 4, ASM)
        a0, a1 = tl.split(s2)
        tl.store(out_ptr + row * V + offs_v, _add(a0, a1, ASM), mask=vmask)


def gpu_lm_head_unavailable(device: torch.device) -> str | None:
    """``None`` if :class:`GpuLmHead` can run on ``device``, else why not."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return "no CUDA device"
    if triton is None:
        return "triton unavailable"
    return None


def chunked_layout(w: torch.Tensor) -> torch.Tensor:
    """``w (V, D)`` as ``(ceil(D / 16), V, 16)`` (the tail chunk zero padded): chunk ``c`` holds columns
    ``16c .. 16c + 15`` of every row."""
    v, d = (int(s) for s in w.shape)
    chunks = -(-d // LANES)
    pad = chunks * LANES - d
    wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
    return wp.reshape(v, chunks, LANES).permute(1, 0, 2).contiguous()


class GpuLmHead:
    """The fixed-order LM head of one weight on one GPU (module docstring). Thread-safe."""

    def __init__(self, w: torch.Tensor, device: torch.device, *, block_v: int = 64, max_rows: int = 64):
        if w.ndim != 2 or w.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("GpuLmHead needs a 2-D float32 or bfloat16 head")
        self.device = device
        self.v, self.d = (int(s) for s in w.shape)
        self.full = self.d // LANES * LANES
        self.tail = self.d - self.full
        self.block_v = int(block_v)
        self.max_rows = int(max_rows)
        self.interpret = _interpret()
        self._lock = threading.Lock()
        src = w.detach()
        if src.dtype == torch.float32:
            bits = src.contiguous().view(torch.int32)
            if not bool((bits & 0xFFFF).eq(0).all()):  # not exactly bfloat16: keep float32 on the device
                self._w = chunked_layout(src.contiguous()).to(device)
            else:
                self._w = chunked_layout(src.to(torch.bfloat16)).to(device)
        else:
            self._w = chunked_layout(src.contiguous()).to(device)
        self._h = torch.zeros((self.max_rows, self.d), dtype=torch.float32, device=device)
        self._out = torch.zeros((self.max_rows, self.v), dtype=torch.float32, device=device)
        pin = device.type == "cuda" and not self.interpret
        self._h_host = torch.zeros((self.max_rows, self.d), dtype=torch.float32, pin_memory=pin)
        self._out_host = torch.zeros((self.max_rows, self.v), dtype=torch.float32, pin_memory=pin)
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None  # type: ignore[no-untyped-call]
        self.calls = 0

    def _launch(self, rows: int) -> None:
        kern = _lm_head_kernel
        grid = (triton.cdiv(self.v, self.block_v), rows)
        kern[grid](self._h, self._w, self._out, self.v, self.d, FULL=self.full, TAIL=self.tail,
                   BLOCK_V=self.block_v, ASM=not self.interpret, num_warps=4)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """float32 CPU logits ``(R, V)`` of float32 rows ``h (R, D)`` (any device)."""
        if h.ndim != 2 or int(h.shape[1]) != self.d:
            raise ValueError(f"lm_head shapes do not match: h {tuple(h.shape)}, w ({self.v}, {self.d})")
        hc = h.detach().to("cpu", torch.float32)
        r = int(hc.shape[0])
        if r == 0:
            return torch.empty((0, self.v), dtype=torch.float32)
        parts = []
        with self._lock:
            for i in range(0, r, self.max_rows):
                n = min(self.max_rows, r - i)
                self._h_host[:n].copy_(hc[i : i + n])
                if self._stream is None or self.interpret:
                    self._h[:n].copy_(self._h_host[:n])
                    self._launch(n)
                    parts.append(self._out[:n].to("cpu").clone())
                    continue
                with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                    self._stream.wait_stream(torch.cuda.current_stream(self.device))
                    self._h[:n].copy_(self._h_host[:n], non_blocking=True)
                    self._launch(n)
                    self._out_host[:n].copy_(self._out[:n], non_blocking=True)
                    self._stream.synchronize()
                parts.append(self._out_host[:n].clone())
        self.calls += 1
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def self_check(self, w_cpu: torch.Tensor, rows: int = 4) -> str | None:
        """``None`` if the GPU logits equal :func:`pgw.engine.lm_head.lm_head_logits` (the CPU kernel) bit for
        bit on public test rows, else what differed."""
        from pgw.engine.lm_head import lm_head_logits

        rng = np.random.default_rng(0x1EAD)
        d = self.d
        cases = [
            ("gaussian", rng.standard_normal((rows, d)).astype(np.float32)),
            ("large", (rng.standard_normal((1, d)) * 3e3).astype(np.float32)),
            ("subnormal products", (rng.standard_normal((2, d)) * 1e-36).astype(np.float32)),
            ("zero", np.zeros((1, d), dtype=np.float32)),
            ("signed zeros", np.where(rng.random((1, d)) < 0.5, np.float32(-0.0), np.float32(0.0)).astype(np.float32)),
        ]
        inf = rng.standard_normal((1, d)).astype(np.float32)
        inf[0, 3] = np.inf
        cases.append(("infinity", inf))
        for name, x in cases:
            xt = torch.from_numpy(x)
            ref = lm_head_logits(xt, w_cpu)
            got = self.logits(xt)
            if ref.shape != got.shape or ref.dtype != got.dtype:
                return f"{name}: shape/dtype differ"
            a = ref.contiguous().view(torch.int32)
            b = got.contiguous().view(torch.int32)
            nan_both = torch.isnan(ref) & torch.isnan(got)
            bad = (a != b) & ~nan_both
            if bool(bad.any()):
                return f"{name}: {int(bad.sum())} logits differ"
        return None


def enable_gpu_lm_head(model: Any, device: str = "cuda") -> dict[str, Any]:
    """Compute ``model``'s LM head on ``device`` if the GPU kernel's logits equal the CPU kernel's bit for bit
    on the self-check rows (then :func:`pgw.engine.lm_head.lm_head_logits` routes the head there); returns the
    policy (``device``, ``reason``, ``setup_s``). Any failure keeps the CPU."""
    import time

    from pgw.engine.lm_head import attach_device_head

    t0 = time.perf_counter()
    policy: dict[str, Any] = {"requested": device, "device": "cpu"}
    w = getattr(model, "_head", None)
    dev = torch.device(device)
    why = gpu_lm_head_unavailable(dev)
    if why is None and (w is None or w.device.type != "cpu" or getattr(model, "lm_head_fn", None) is not None):
        why = "the model has no CPU float LM head"
    if why is None:
        try:
            head = GpuLmHead(w, dev)
            why = head.self_check(w)
            if why is not None:
                why = f"logits differ from the CPU kernel ({why})"
        except Exception as exc:  # a compile or launch failure: the CPU is kept
            why = f"{type(exc).__name__}: {str(exc)[:200]}"
    policy["setup_s"] = round(time.perf_counter() - t0, 3)
    if why is not None:
        policy["reason"] = why
        log.warning("LM head: stays on the CPU (%s)", why)
        return policy
    attach_device_head(w, head)
    policy.update(device=str(dev), reason="identical logits")
    log.info("LM head: on %s (fixed-order kernel, identical logits on the self-check rows; %.2f s)", dev,
             policy["setup_s"])
    return policy
