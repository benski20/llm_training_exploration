#!/usr/bin/env python3
"""Micro-benchmark of the service's CUDA kernel: eager vs CUDA-graph replay, ms per call.

Times :meth:`pgw.service.ComputeKernel.run_compact` (what the service's request path calls: host
int64 rows in, host result out, including staging copies and the stream sync) on random int8
weights with the Qwen2.5-7B linear shapes, for 1, 5 and 8 rows. Every timed result is also
checked bit-for-bit between the two modes. Usage (on a GPU host)::

    PYTHONPATH=src python benchmarks/gpu_kernel_bench.py                  # K32A12C27, 7B shapes
    PYTHONPATH=src python benchmarks/gpu_kernel_bench.py --ring K40A12 --iters 500
    PYTHONPATH=src python benchmarks/gpu_kernel_bench.py --device cpu --scale 16 --iters 5  # smoke test

A 7B layer (all four linears, ~233 MB of int8 weights) at T4 bandwidth (~320 GB/s) needs about
0.73 ms per decode step, i.e. ~0.18 ms per call on average; the "layer" line sums the four linears.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch

from pgw.core import LinearId, LinearKind, LinearSpec, RingParams
from pgw.service.gpu_exec import device_executor, from_compact
from pgw.service.kernels import ComputeKernel

SHAPES_7B = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}
KINDS = {"qkv": LinearKind.QKV, "o": LinearKind.O, "gate_up": LinearKind.GATE_UP, "down": LinearKind.DOWN}


def _time(kernel: ComputeKernel, xs: list[torch.Tensor], ring: RingParams, iters: int) -> tuple[float, torch.Tensor]:
    """Median ms per call over ``iters`` calls (cycling through ``xs``), after 3 warm-up calls."""
    for x in xs[:3]:
        kernel.run_compact(x, ring)
    samples = []
    y = kernel.run_compact(xs[0], ring)
    for i in range(iters):
        t0 = time.perf_counter()
        kernel.run_compact(xs[i % len(xs)], ring)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3, y


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--rows", default="1,5,8")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scale", type=int, default=1, help="divide the 7B shapes by this (smoke tests)")
    ap.add_argument("--json", default=None, help="also write the results to this file")
    args = ap.parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available (use --device cpu for a smoke test)", file=sys.stderr)
        return 2
    ring = RingParams.from_name(args.ring)
    rows_list = [int(r) for r in args.rows.split(",")]
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    results: list[dict[str, object]] = []
    print(f"device={args.device} ring={ring.name} iters={args.iters} torch={torch.__version__}"
          + (f" gpu={torch.cuda.get_device_name()}" if args.device.startswith("cuda") else ""))
    print(f"{'linear':<8} {'m x n':>13} {'rows':>4} {'eager ms':>9} {'graph ms':>9} {'speedup':>8}")
    layer: dict[tuple[int, str], float] = {}
    for name, (m, n) in SHAPES_7B.items():
        m, n = max(1, m // args.scale), max(1, n // args.scale)
        w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
        spec = LinearSpec(LinearId(0, KINDS[name]), m, n)
        eager = ComputeKernel(spec, w, device=args.device, cuda_graphs=False)
        graph = ComputeKernel(spec, w, device=args.device, cuda_graphs=True)
        for rows in rows_list:
            xs = [torch.from_numpy(rng.integers(0, ring.modulus, size=(rows, n), dtype=np.int64)) for _ in range(4)]
            te, ye = _time(eager, xs, ring, args.iters)
            tg, yg = _time(graph, xs, ring, args.iters)
            if not torch.equal(from_compact(ye, ring.k), from_compact(yg, ring.k)):
                print(f"MISMATCH {name} rows={rows}: graph result differs from eager", file=sys.stderr)
                return 1
            layer[(rows, "eager")] = layer.get((rows, "eager"), 0.0) + te
            layer[(rows, "graph")] = layer.get((rows, "graph"), 0.0) + tg
            results.append({"linear": name, "m": m, "n": n, "rows": rows, "eager_ms": round(te, 4),
                            "graph_ms": round(tg, 4), "graphs": graph.cuda_graphs})
            print(f"{name:<8} {f'{m}x{n}':>13} {rows:>4} {te:>9.3f} {tg:>9.3f} {te / tg:>7.2f}x", flush=True)
        del eager, graph
    for rows in rows_list:
        te, tg = layer[(rows, "eager")], layer[(rows, "graph")]
        print(f"{'layer':<8} {'(4 linears)':>13} {rows:>4} {te:>9.3f} {tg:>9.3f} {te / tg:>7.2f}x "
              f"(avg/call eager {te / 4:.3f} graph {tg / 4:.3f})")
    if args.device.startswith("cuda"):
        print("executor stats:", device_executor(args.device if ":" in args.device else "cuda:0").stats)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": args.device, "ring": ring.name, "results": results}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
