#!/usr/bin/env python3
"""Provider ring-GEMM throughput at large decode batches, for the scale plan (research/plans/scale-north-star.md).

Measures the exact ``x @ W8^T mod 2^K`` (K=32, :class:`pgw.core.ringmm.PreparedWeight`) on one GPU for the
linear shapes of Llama-3-70B (dense) and Mixtral-8x22B (MoE) at 1..1024 rows per call, and derives:

* per-model decode step time and tokens/s per GPU at batch ``B`` (all layers, one GPU doing all the work);
* the routing-oblivious MoE cost (every expert gets ``C = B`` rows) against an ideal, leaky dispatch
  (``B k / E`` rows per expert on average);
* the plain mask factory rate (``W r`` for 1024 slots in one GEMM) with weights resident;
* the gateway CPU cost of dummy rows (:func:`pgw.moe.oblivious.uniform_rows`).

Exactness is spot-checked against a CPU uint64 product. Usage::

    python benchmarks/scale_provider_bench.py --json out.json [--rows 1 16 64 256 1024] [--iters 5]
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgw.core.ring import DEFAULT_RING
from pgw.core.ringmm import PreparedWeight

#: (m, n) per group, plus layer count and MoE (experts, top-k) for the expert groups.
MODELS = {
    "llama-3-70b": {"layers": 80, "dense": {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192),
                                            "down": (8192, 28672)}, "experts": None},
    "mixtral-8x22b": {"layers": 56, "dense": {"qkv": (8192, 6144), "o": (6144, 6144)},
                      "experts": {"E": 8, "k": 2, "gate_up": (32768, 6144), "down": (6144, 16384)}},
}


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def time_shape(m: int, n: int, rows_list: list[int], dev: torch.device, iters: int) -> dict[int, float]:
    """Median seconds per exact ring matmul of ``rows x n`` by ``(m, n)`` int8 for each row count."""
    g = torch.Generator(device="cpu").manual_seed(m * 7 + n)
    w8 = torch.randint(-127, 128, (m, n), generator=g, dtype=torch.int8)
    pw = PreparedWeight(w8, device=dev, cache_float=False)
    out: dict[int, float] = {}
    for rows in rows_list:
        x = torch.randint(0, 1 << 32, (rows, n), generator=g, dtype=torch.int64).to(dev)
        try:
            pw.matmul(x, DEFAULT_RING)  # warm-up
            _sync(dev)
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                pw.matmul(x, DEFAULT_RING)
                _sync(dev)
                ts.append(time.perf_counter() - t0)
            out[rows] = statistics.median(ts)
        except torch.cuda.OutOfMemoryError:  # pragma: no cover - GPU only
            out[rows] = float("nan")
        del x
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    del pw
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def exactness_check(dev: torch.device) -> bool:
    """GPU result == CPU uint64 product (wraps mod 2^64; reduced mod 2^32) on a 512 x 3000 weight."""
    g = torch.Generator().manual_seed(1)
    w8 = torch.randint(-127, 128, (512, 3000), generator=g, dtype=torch.int8)
    x = torch.randint(0, 1 << 32, (37, 3000), generator=g, dtype=torch.int64)
    got = PreparedWeight(w8, device=dev, cache_float=False).matmul(x.to(dev), DEFAULT_RING).cpu().numpy()
    ref = (x.numpy().astype(np.uint64) @ w8.numpy().astype(np.int64).view(np.uint64).T) & np.uint64(0xFFFFFFFF)
    return bool(np.array_equal(got.astype(np.uint64), ref))


def interp(times: dict[int, float], rows: float) -> float:
    """Time at a (possibly fractional) row count: linear interpolation between measured points."""
    pts = sorted((r, t) for r, t in times.items() if t == t)
    if rows <= pts[0][0]:
        return pts[0][1]
    for (r0, t0), (r1, t1) in itertools.pairwise(pts):
        if rows <= r1:
            return t0 + (t1 - t0) * (rows - r0) / (r1 - r0)
    r_last, t_last = pts[-1]
    return t_last * rows / r_last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, nargs="*", default=[1, 16, 64, 128, 256, 512, 1024])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    res: dict = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor(),
                 "ring": DEFAULT_RING.name, "rows": a.rows, "exact": exactness_check(dev), "shapes": {}, "models": {}}
    print(f"device {res['device']}  ring {res['ring']}  exact {res['exact']}", flush=True)
    for name in a.models:
        spec = MODELS[name]
        shapes = dict(spec["dense"])
        if spec["experts"]:
            shapes.update({f"expert.{g}": spec["experts"][g] for g in ("gate_up", "down")})
        times = {}
        for g, (m, n) in shapes.items():
            t = time_shape(m, n, a.rows, dev, a.iters)
            times[g] = t
            res["shapes"][f"{name}.{g}"] = {"m": m, "n": n, "s": {str(r): v for r, v in t.items()},
                                            "ring_tops": {str(r): 2 * m * n * r / v / 1e12 for r, v in t.items()}}
            print(f"{name}.{g:<14} {m}x{n}  " + "  ".join(f"B={r}:{v * 1e3:.2f}ms" for r, v in t.items()), flush=True)
        L = spec["layers"]
        model_out = {}
        for b in a.rows:
            dense = sum(interp(times[g], b) for g in spec["dense"])
            if spec["experts"]:
                e, k = spec["experts"]["E"], spec["experts"]["k"]
                exp_obl = e * sum(interp(times[f"expert.{g}"], b) for g in ("gate_up", "down"))
                exp_leaky = e * sum(interp(times[f"expert.{g}"], b * k / e) for g in ("gate_up", "down"))
                step_obl = L * (dense + exp_obl)
                step_leaky = L * (dense + exp_leaky)
                model_out[str(b)] = {"step_s_oblivious": step_obl, "step_s_leaky_ideal": step_leaky,
                                     "tok_per_s_oblivious": b / step_obl, "tok_per_s_leaky_ideal": b / step_leaky,
                                     "oblivious_overhead": step_obl / step_leaky}
            else:
                step = L * dense
                model_out[str(b)] = {"step_s": step, "tok_per_s": b / step}
        # plain factory: W r for many slots in one GEMM (rows = slots), weights resident
        big = max(a.rows)
        if spec["experts"]:
            e, k = spec["experts"]["E"], spec["experts"]["k"]
            per_slot = L * (sum(times[g][big] for g in spec["dense"])
                            + k * sum(times[f"expert.{g}"][big] for g in ("gate_up", "down"))) / big
        else:
            per_slot = L * sum(times[g][big] for g in spec["dense"]) / big
        model_out["factory_plain_s_per_slot"] = per_slot
        res["models"][name] = model_out
        print(f"{name}: " + json.dumps(model_out), flush=True)
    # gateway CPU: dummy rows (ChaCha20, 4 bytes per element at K32)
    from pgw.moe.oblivious import uniform_rows

    t0 = time.perf_counter()
    arr = uniform_rows(2048, 6144, DEFAULT_RING)
    dt = time.perf_counter() - t0
    res["gateway_dummy_rows_elems_per_s"] = arr.size / dt
    print(f"gateway dummy rows: {arr.size / dt / 1e6:.0f} M elements/s (one core)", flush=True)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    return 0 if res["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
