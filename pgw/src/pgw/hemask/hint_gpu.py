"""The one-time encryption hint ``W8 A`` on the gateway's own GPU (opt-in, exact).

:meth:`pgw.hemask.backend.HintStore.build` computes, per linear and block of hint columns,
``to_planes(W8 @ A mod 2^64)`` on the CPU (int8 GEMMs on byte planes). For Qwen2.5-7B that is about 6.5e9
weights times 4096 columns: tens of minutes on a few cores. :class:`GpuHintPlanes` computes the same int8
planes on a CUDA device the gateway owns:

* ``A`` (uint64) is split into its 8 bytes; byte ``p`` is stored as ``byte - 128`` (int8), so one exact int8
  GEMM with int32 sums (``torch._int_mm``; ``|w| <= 128``, ``n <= 2^14`` keeps every sum far below ``2^31``)
  gives ``W8 @ (byte_p - 128)``; adding ``128 * rowsum(W8)`` gives ``W8 @ byte_p``.
* ``W8 @ A mod 2^64 = sum_p (W8 @ byte_p) * 2^(8p)`` in wrapping int64 (two's complement wrap is the
  reduction mod 2^64), then the result's bytes are written as ``byte ^ 0x80`` planes, exactly
  :func:`pgw.hemask.fastmm.to_planes`.

The bits are the CPU path's: integer arithmetic only, no rounding. Before use, :meth:`GpuHintPlanes.self_check`
compares it with the CPU path on public random data (shapes of the pack's linears, extreme weights included);
any difference or CUDA error keeps the CPU path. Privacy: ``A`` is public and chosen by the customer, the
weights are the gateway's own pack, and the device is the gateway's; nothing leaves the gateway.

Off by default (``PGW_HINT_DEVICE``, read by :class:`pgw.gateway.setup_jobs.HintBuildJob`).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

__all__ = ["GpuHintPlanes", "cpu_planes"]

log = logging.getLogger(__name__)


def cpu_planes(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The CPU reference: ``to_planes(matmul_mod64(w, a))`` (``(m, 8, cols)`` int8)."""
    from pgw.hemask.fastmm import to_planes
    from pgw.hemask.lwe import matmul_mod64

    out: np.ndarray = to_planes(matmul_mod64(w, a))
    return out


class GpuHintPlanes:
    """``(w, a) -> to_planes(w @ a mod 2^64)`` on ``device`` (see the module docstring).

    ``w``: ``(m, n)`` int8 weights; ``a``: ``(n, cols)`` uint64. Returns ``(m, 8, cols)`` int8 on the host.
    The last weight matrix stays on the device between calls (a linear's column blocks reuse it).
    """

    def __init__(self, device: str = "cuda"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device for the hint build")
        self._w_src: np.ndarray | None = None  # the cached weights' host array (held: no address reuse)
        self._w: Any = None
        self._wsum: Any = None
        self.calls = 0

    def _weights(self, w: np.ndarray) -> tuple[Any, Any]:
        torch = self.torch
        if w is not self._w_src:  # the same array object (HintStore.build passes one per linear)
            self._w = torch.from_numpy(np.ascontiguousarray(w, dtype=np.int8)).to(self.device)
            self._wsum = self._w.to(torch.int64).sum(dim=1, keepdim=True)  # (m, 1)
            self._w_src = w
        return self._w, self._wsum

    def __call__(self, w: np.ndarray, a: np.ndarray) -> np.ndarray:
        torch = self.torch
        m, n = w.shape
        if a.shape[0] != n:
            raise ValueError(f"shape mismatch: {w.shape} vs {a.shape}")
        cols = int(a.shape[1])
        wd, wsum = self._weights(w)
        # (cols, 8) bytes of each A entry -> 8 planes of (cols, n) int8 = byte - 128 (column-major mat2 for cuBLASLt)
        a_bytes = np.ascontiguousarray(np.asarray(a, dtype="<u8").T).view(np.uint8).reshape(cols, n, 8)
        ad = torch.from_numpy(a_bytes).to(self.device)
        acc = torch.zeros((m, cols), dtype=torch.int64, device=self.device)
        for p in range(8):
            bt = (ad[:, :, p].to(torch.int16) - 128).to(torch.int8).contiguous()  # (cols, n)
            part = torch._int_mm(wd, bt.t())  # (m, cols) int32, exact
            # W8 @ byte_p = W8 @ (byte_p - 128) + 128 * rowsum(W8); times 2^(8p), wrapping int64
            acc += (part.to(torch.int64) + 128 * wsum) << (8 * p)
        planes = torch.empty((m, 8, cols), dtype=torch.uint8, device=self.device)
        for d in range(8):
            planes[:, d, :] = ((acc >> (8 * d)) & 0xFF).to(torch.uint8) ^ 0x80
        self.calls += 1
        out: np.ndarray = planes.cpu().numpy().view(np.int8)
        return out

    def self_check(self, shapes: list[tuple[int, int]], cols: int = 512, seed: int = 7) -> bool:
        """Compare with :func:`cpu_planes` on public random data for each ``(m, n)`` (rows capped at 256);
        extreme weights (-128, 127, 0) and extreme ``A`` words (0, 2^64 - 1) included. False on any difference."""
        rng = np.random.default_rng(seed)
        try:
            for m, n in shapes:
                mm = min(int(m), 256)
                w = rng.integers(-128, 128, size=(mm, n), dtype=np.int8)
                w[0, :] = -128
                w[1 % mm, :] = 127
                w[2 % mm, :] = 0
                a = rng.integers(0, 2**63, size=(n, cols), dtype=np.uint64) * np.uint64(2) + rng.integers(
                    0, 2, size=(n, cols), dtype=np.uint64)
                a[:, 0] = np.uint64(2**64 - 1)
                a[:, 1 % cols] = 0
                if not np.array_equal(self(w, a), cpu_planes(w, a)):
                    log.warning("GPU hint build: self-check mismatch at shape %s; using the CPU", (m, n))
                    return False
        except Exception as exc:  # any CUDA / torch failure: keep the CPU path
            log.warning("GPU hint build unavailable (%s: %s); using the CPU", type(exc).__name__, exc)
            return False
        finally:
            self._w_src = self._w = self._wsum = None
        return True

    def release(self) -> None:
        self._w_src = self._w = self._wsum = None
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
