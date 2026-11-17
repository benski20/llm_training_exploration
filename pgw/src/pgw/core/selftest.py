"""End-to-end self test of the core math (used by ``pgw doctor`` and CI smoke tests)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from pgw.core.freivalds import freivalds_check, freivalds_precompute
from pgw.core.masking import mask_activations, unmask_outputs
from pgw.core.prng import SeedExpander, new_seed
from pgw.core.ring import RING_K32_A8, RING_K40_A12, RING_K48_A20, RingParams
from pgw.core.ringmm import PreparedWeight, cuda_int8_available

__all__ = ["SelfTestResult", "run_selftest"]


@dataclass
class SelfTestResult:
    """Outcome of :func:`run_selftest`."""

    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


def run_selftest(m: int = 96, n: int = 160, tokens: int = 5, include_gpu: bool = True) -> SelfTestResult:
    """Mask -> ringmm -> unmask -> Freivalds on random data for each ring preset and backend."""
    res = SelfTestResult(ok=True)
    sx = SeedExpander(new_seed())
    w8 = sx.uniform_int((m, n), -127, 128, "w").to(torch.int8)
    backends = ["cpu_f64", "cpu_f32"] + (["cuda_int8"] if include_gpu and cuda_int8_available() else [])
    rings: tuple[RingParams, ...] = (RING_K32_A8, RING_K40_A12, RING_K48_A20)
    for p in rings:
        xq = sx.uniform_int((tokens, n), -p.qmax, p.qmax + 1, "x", p.name)
        r = sx.ring_uniform((tokens, n), p.k, "r", p.name)
        u = sx.ring_uniform((2, m), p.k, "u", p.name)
        ref = xq.double() @ w8.double().t()
        for be in backends:
            dev = "cuda" if be == "cuda_int8" else "cpu"
            t0 = time.perf_counter()
            pw = PreparedWeight(w8, device=dev, backend=be)  # type: ignore[arg-type]
            x_hat = mask_activations(xq, r, p)
            y_hat = pw.matmul(x_hat.to(dev), p).cpu()
            wr = pw.matmul(r.to(dev), p).cpu()
            yq = unmask_outputs(y_hat, wr, p, in_features=n)
            exact = bool((yq.double() == ref).all())
            uw = freivalds_precompute(u, w8, p)
            fv = bool(freivalds_check(x_hat, y_hat, u, uw, p).all())
            bad = y_hat.clone()
            bad[0, 0] = (bad[0, 0] + 1) & p.mask
            caught = not bool(freivalds_check(x_hat, bad, u, uw, p).all())
            key = f"{p.name}/{be}"
            res.checks[f"{key}/exact"] = exact
            res.checks[f"{key}/freivalds_accepts_genuine"] = fv
            res.checks[f"{key}/freivalds_rejects_tampered"] = caught
            res.timings_ms[key] = round((time.perf_counter() - t0) * 1e3, 2)
    res.ok = all(res.checks.values())
    return res
