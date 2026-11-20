#!/usr/bin/env python3
"""Fused Triton ring GEMV vs the digit-GEMM path on the 7B linear shapes (ms per call, exactness).

Three tables (Qwen2.5-7B shapes, ``K32A12C27`` by default):

1. **service call**: :meth:`pgw.service.ComputeKernel.run_compact` end to end (host int64 rows in,
   host result out: staging copies, kernels, stream sync), ``triton_gemv=False`` ("eager", today's
   path) vs ``triton_gemv=True``; the comparable number to ``gpu_kernel_bench.py``. Every timed
   result is compared bit for bit between the two.
2. **device only**: CUDA-event time of the kernels alone (operands already on the device):
   :func:`pgw.core.ringmm.offset_byte_core` vs :func:`pgw.core.triton_gemv.ring_gemv`, with the
   effective weight bandwidth.
3. **config sweep** (``--sweep``): device time of a few launch configurations per shape and row
   count, against the default heuristic (:func:`pgw.core.triton_gemv.gemv_config`).

Usage (GPU host)::

    PYTHONPATH=src python benchmarks/triton_gemv_bench.py --iters 300 --sweep
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
from pgw.core.ringmm import offset_byte_consts, offset_byte_core, offset_byte_rows
from pgw.core.triton_gemv import GemvConfig, gemv_config, low32, max_rows, ring_gemv, self_check
from pgw.service.gpu_exec import device_executor, from_compact
from pgw.service.kernels import ComputeKernel

SHAPES_7B = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}
KINDS = {"qkv": LinearKind.QKV, "o": LinearKind.O, "gate_up": LinearKind.GATE_UP, "down": LinearKind.DOWN}


def _host_ms(fn, xs: list[torch.Tensor], iters: int) -> float:  # type: ignore[no-untyped-def]
    for x in xs[:3]:
        fn(x)
    samples = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn(xs[i % len(xs)])
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def _dev_ms(fn, iters: int) -> float:  # type: ignore[no-untyped-def]
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def _sweep_configs(rows: int, dot: bool) -> list[GemvConfig]:
    """CUDA-core configs (rows padded to a power of two) and tensor-core configs (rows padded to 16)."""
    br = 1 << max(0, (rows - 1).bit_length())
    out = []
    for bn in (256, 512, 1024):
        for bm in (8, 16, 32, 64):
            if br * bm * bn > 32768:
                continue
            for split in (1, 2, 4, 8):
                for warps in (4, 8):
                    out.append(GemvConfig(br, bm, bn, split, warps, False))
    for bn in (32, 64, 128, 256) if dot else ():
        for bm in (16, 32, 64, 128):
            for split in (1, 2, 4):
                for warps in (4, 8):
                    out.append(GemvConfig(max(16, br), bm, bn, split, warps, True))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--rows", default="1,5,8,16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--sweep", action="store_true", help="also time a grid of launch configurations")
    ap.add_argument("--sweep-rows", default="1,8")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2
    dev = torch.device("cuda:0")
    ring = RingParams.from_name(args.ring)
    k = ring.k
    rows_list = [int(r) for r in args.rows.split(",")]
    t0 = time.perf_counter()
    ok = self_check(dev)
    print(f"self_check({dev}) = {ok} ({time.perf_counter() - t0:.1f} s incl. compiles); fused kernel serves up to "
          f"{max_rows(dev)} rows (more rows need the tensor-core variant)")
    if not ok:
        return 1
    props = torch.cuda.get_device_properties(dev)
    print(f"gpu={props.name} sms={props.multi_processor_count} torch={torch.__version__} ring={ring.name} "
          f"iters={args.iters}")
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    results: list[dict[str, object]] = []
    layer: dict[tuple[int, str], float] = {}
    lines_call, lines_dev, lines_sweep = [], [], []
    for name, (m, n) in SHAPES_7B.items():
        w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
        spec = LinearSpec(LinearId(0, KINDS[name]), m, n)
        eager = ComputeKernel(spec, w8, device=dev, cuda_graphs=False, triton_gemv=False)
        fused = ComputeKernel(spec, w8, device=dev, cuda_graphs=False, triton_gemv=True)
        assert fused.triton_gemv
        wpad = eager._pw._w_dev  # type: ignore[union-attr]
        rowsum = eager._pw._rowsum()  # type: ignore[union-attr]
        sh, pm = offset_byte_consts(k, dev)
        for rows in rows_list:
            xs = [torch.from_numpy(rng.integers(0, ring.modulus, size=(rows, n), dtype=np.int64)) for _ in range(4)]
            for x in xs:
                ye, yf = eager.run_compact(x, ring), fused.run_compact(x, ring)
                if not torch.equal(from_compact(ye, k), from_compact(yf, k)):
                    print(f"MISMATCH {name} rows={rows}", file=sys.stderr)
                    return 1
            te = _host_ms(lambda x: eager.run_compact(x, ring), xs, args.iters)
            tf = _host_ms(lambda x: fused.run_compact(x, ring), xs, args.iters)
            layer[(rows, "eager")] = layer.get((rows, "eager"), 0.0) + te
            layer[(rows, "fused")] = layer.get((rows, "fused"), 0.0) + tf
            lines_call.append(f"{name:<8} {f'{m}x{n}':>13} {rows:>4} {te:>9.3f} {tf:>9.3f} {te / tf:>7.2f}x")
            # device only
            xd = xs[0].to(dev)
            x32 = low32(xs[0]).to(dev)
            p = offset_byte_rows(k, rows)
            de = _dev_ms(lambda: offset_byte_core(xd, k, wpad, rowsum, sh, pm, p), args.iters)
            cfg = gemv_config(rows, m, int(wpad.shape[1]), dev)
            df = _dev_ms(lambda: ring_gemv(x32, wpad, k, n=n, m=m, config=cfg), args.iters)
            gbs = m * n / (df * 1e-3) / 1e9
            lines_dev.append(f"{name:<8} {f'{m}x{n}':>13} {rows:>4} {de:>9.3f} {df:>9.3f} {de / df:>7.2f}x "
                             f"{gbs:>7.0f} GB/s  {cfg}")
            rec: dict[str, object] = {"linear": name, "m": m, "n": n, "rows": rows, "call_eager_ms": round(te, 4),
                                      "call_fused_ms": round(tf, 4), "dev_eager_ms": round(de, 4),
                                      "dev_fused_ms": round(df, 4), "config": str(cfg)}
            if args.sweep and rows in [int(r) for r in args.sweep_rows.split(",")]:
                want = ring_gemv(x32, wpad, k, n=n, m=m, config=cfg).clone()
                best: tuple[float, GemvConfig] | None = None
                for c in _sweep_configs(rows, max_rows(dev) > 1):
                    try:
                        got = ring_gemv(x32, wpad, k, n=n, m=m, config=c)
                        if not torch.equal(got, want):
                            print(f"MISMATCH {name} rows={rows} {c}", file=sys.stderr)
                            return 1
                        t = _dev_ms(lambda c=c: ring_gemv(x32, wpad, k, n=n, m=m, config=c), max(20, args.iters // 5))
                    except Exception as exc:
                        lines_sweep.append(f"  {name} rows={rows} {c}: {type(exc).__name__}")
                        continue
                    if best is None or t < best[0]:
                        best = (t, c)
                if best is not None:
                    lines_sweep.append(f"{name:<8} rows={rows:>2} heuristic {df:.4f} ms  best {best[0]:.4f} ms "
                                       f"({m * n / (best[0] * 1e-3) / 1e9:.0f} GB/s) {best[1]}")
                    rec["sweep_best_ms"] = round(best[0], 4)
                    rec["sweep_best"] = str(best[1])
            results.append(rec)
            print(lines_call[-1], "|", lines_dev[-1], flush=True)
        torch.cuda.empty_cache()
    print()
    print("service call (ComputeKernel.run_compact, host in/out), median ms per call")
    print(f"{'linear':<8} {'m x n':>13} {'rows':>4} {'eager ms':>9} {'triton ms':>9} {'speedup':>8}")
    for line in lines_call:
        print(line)
    for rows in rows_list:
        te, tf = layer[(rows, "eager")], layer[(rows, "fused")]
        print(f"{'layer':<8} {'(4 linears)':>13} {rows:>4} {te:>9.3f} {tf:>9.3f} {te / tf:>7.2f}x")
    print()
    print("device only (CUDA events, operands on device), ms per call")
    print(f"{'linear':<8} {'m x n':>13} {'rows':>4} {'digit ms':>9} {'triton ms':>9} {'speedup':>8} {'W8 bw':>12}  config")
    for line in lines_dev:
        print(line)
    bytes_layer = sum(m * n for m, n in SHAPES_7B.values())
    print(f"layer int8 weight bytes {bytes_layer / 1e6:.1f} MB; at 300 GB/s: {bytes_layer / 300e9 * 1e3:.3f} ms")
    if lines_sweep:
        print()
        print("config sweep (device only)")
        for line in lines_sweep:
            print(line)
    print("executor stats:", device_executor(dev).stats)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ring": ring.name, "gpu": props.name, "results": results}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
