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

``--gemv`` instead times the fused Triton ring GEMV (:mod:`pgw.core.triton_gemv`) on the device
(CUDA events, operands already on the device) for 1-8 rows against the digit-GEMM path
(:func:`pgw.core.ringmm.offset_byte_core`): the configuration the service would use (device table
or heuristic, after :func:`pgw.core.triton_gemv.self_check`) and the memory-bound ``stream``
variant whether or not it is enabled. ``--gemv-sweep`` also times a grid of launch configurations
(every result compared bit for bit) and ``--emit-table FILE`` writes the winners as a device table
(JSON) that ``PGW_SERVICE_TRITON_GEMV_TABLE`` loads; every rule is self-checked again at load::

    PYTHONPATH=src python benchmarks/gpu_kernel_bench.py --gemv --shapes 7b,0.5b --rows 1,2,3,4,5,6,7,8
    PYTHONPATH=src python benchmarks/gpu_kernel_bench.py --gemv --gemv-sweep --emit-table t4_gemv.json
    TRITON_INTERPRET=1 PYTHONPATH=src python benchmarks/gpu_kernel_bench.py --gemv --device cpu --scale 64 \
        --iters 1 --rows 1,3                                                   # smoke test (interpreter)
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
SHAPES_05B = {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}
SHAPE_SETS = {"7b": SHAPES_7B, "0.5b": SHAPES_05B}
KINDS = {"qkv": LinearKind.QKV, "o": LinearKind.O, "gate_up": LinearKind.GATE_UP, "down": LinearKind.DOWN}


# ---- --gemv: fused Triton ring GEMV vs the digit-GEMM path, device time --------------------------
def _dev_ms(fn, iters: int, dev: torch.device) -> float:  # type: ignore[no-untyped-def]
    """Mean ms per call of ``fn`` (CUDA events on a GPU, wall clock on CPU), after warm-up."""
    for _ in range(min(3, iters)):
        fn()
    if dev.type != "cuda":
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1e3
    torch.cuda.synchronize(dev)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(dev)
    return float(start.elapsed_time(end)) / iters


def gemv_sweep_configs(rows: int, *, dot: bool) -> list:  # type: ignore[type-arg]
    """Launch configurations of the sweep: ``sum`` (one row) and ``stream`` tiles within a register
    budget (``rows * block_m * block_n / (32 * warps) <= 64`` accumulator words per thread), split-K
    1-8, 4 or 8 warps; tensor-core tiles too where the device has them."""
    from pgw.core.triton_gemv import MULTIROW_MAX_ROWS, GemvConfig

    out = []
    if rows == 1:
        for bn in (512, 1024):
            for bm in (8, 16, 32):
                for split in (1, 2, 4, 8):
                    out.append(GemvConfig(1, bm, bn, split, 4))
    if rows <= MULTIROW_MAX_ROWS:
        for bm in (8, 16, 32, 64):
            for bn in (32, 64, 128, 256):
                for warps in (4, 8):
                    if rows * bm * bn > 64 * 32 * warps or bm * bn < 32 * warps:
                        continue
                    for split in (1, 2, 4, 8):
                        out.append(GemvConfig(rows, bm, bn, split, warps, stream=True))
    if dot and rows > 1:
        for bn in (64, 128):
            for bm in (32, 64):
                for split in (1, 2, 4):
                    out.append(GemvConfig(16, bm, bn, split, 4, dot=True))
    return out


def _gemv_case(args: argparse.Namespace, dev: torch.device, k: int, x: torch.Tensor, wpad: torch.Tensor, m: int,
               n: int, *, sweep_dot: bool) -> tuple | None:  # type: ignore[type-arg]
    """One shape and row count: ``(digit ms, service ms, stream ms, (best ms, best config), service config)``,
    or None if any fused result differs from the digit-GEMM path."""
    from pgw.core.ringmm import offset_byte_consts, offset_byte_core, offset_byte_rows
    from pgw.core.triton_gemv import gemv_config, low32, ring_gemv, stream_config

    rows = int(x.shape[0])
    nw = int(wpad.shape[1])
    rowsum = wpad.to(torch.int64).sum(1)
    sh, pm = offset_byte_consts(k, dev)
    p = offset_byte_rows(k, rows)
    x32 = low32(x)

    def digit() -> torch.Tensor:
        return offset_byte_core(x, k, wpad, rowsum, sh, pm, p)

    want = (digit()[:, :m] & 0xFFFFFFFF).to(torch.int64)

    def fused(cfg) -> torch.Tensor:  # type: ignore[no-untyped-def]
        return ring_gemv(x32, wpad, k, n=n, m=m, config=cfg)

    def same(cfg) -> bool:  # type: ignore[no-untyped-def]
        return torch.equal(fused(cfg).to(torch.int64) & 0xFFFFFFFF, want)

    cfg_s = gemv_config(rows, m, nw, dev)
    cfg_st = stream_config(rows, m, nw, dev) if rows <= 8 else cfg_s
    if not (same(cfg_s) and same(cfg_st)):
        return None
    td = _dev_ms(digit, args.iters, dev)
    ts = _dev_ms(lambda: fused(cfg_s), args.iters, dev)
    tst = _dev_ms(lambda: fused(cfg_st), args.iters, dev)
    best = (ts, cfg_s)
    if args.gemv_sweep:
        for c in gemv_sweep_configs(rows, dot=sweep_dot):
            try:
                if not same(c):
                    return None
                t = _dev_ms(lambda c=c: fused(c), max(5, args.iters // 4), dev)
            except Exception as exc:  # a tile that does not compile on this GPU
                print(f"  skip {c}: {type(exc).__name__}", file=sys.stderr)
                continue
            if t < best[0]:
                best = (t, c)
    return td, ts, tst, best, cfg_s


def gemv_main(args: argparse.Namespace) -> int:
    """``--gemv``: device time of the digit-GEMM path, the service's fused config, the stream variant
    and (``--gemv-sweep``) the best swept config per shape and row count; all bit-compared."""
    from pgw.core.ringmm import pad_weight_cuda
    from pgw.core.triton_gemv import TableRule, device_name, max_rows, multirow_rows, self_check, table_to_json

    dev = torch.device(args.device if ":" in args.device or args.device == "cpu" else "cuda:0")
    ring = RingParams.from_name(args.ring)
    k = ring.k
    if k > 32:
        print("--gemv needs a ring with K <= 32", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    ok = self_check(dev)
    name = device_name(dev)
    print(f"device={dev} ({name}) ring={ring.name} iters={args.iters} torch={torch.__version__} "
          f"self_check={ok} ({time.perf_counter() - t0:.1f} s) max_rows={max_rows(dev)} "
          f"stream_rows={multirow_rows(dev)}")
    if not ok:
        return 1
    rows_list = [int(r) for r in args.rows.split(",")]
    gen = torch.Generator().manual_seed(0)
    results: list[dict[str, object]] = []
    best_rules: list[TableRule] = []
    layer: dict[tuple[str, int, str], float] = {}
    hdr = (f"{'shape':<12} {'m x n':>12} {'rows':>4} {'digit':>8} {'service':>8} {'stream':>8} "
           f"{'best':>8} {'GB/s':>5}  service config | best config   (device ms per call)")
    print(hdr)
    for set_name in args.shapes.split(","):
        for lin, (m, n) in SHAPE_SETS[set_name].items():
            m, n = max(1, m // args.scale), max(1, n // args.scale)
            w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=gen)
            wpad = pad_weight_cuda(w8, dev) if dev.type == "cuda" else w8.clone()
            nw = int(wpad.shape[1])
            for rows in rows_list:
                x = torch.randint(0, 1 << k, (rows, n), dtype=torch.int64, generator=gen).to(dev)
                res = _gemv_case(args, dev, k, x, wpad, m, n, sweep_dot=max_rows(dev) > 1 and rows > multirow_rows(dev))
                if res is None:
                    print(f"MISMATCH {set_name} {lin} rows={rows}", file=sys.stderr)
                    return 1
                td, ts, tst, (tb, cb), cfg_s = res
                if args.gemv_sweep and cb != cfg_s and tb * 1.02 < ts:
                    note = f"{set_name} {lin}: {ts:.4f} -> {tb:.4f} ms"
                    best_rules.append(TableRule(rows, cb, (m, m), (nw, nw), note))
                for col, t in (("digit", td), ("service", ts), ("stream", tst), ("best", tb)):
                    layer[(set_name, rows, col)] = layer.get((set_name, rows, col), 0.0) + t
                gbs = m * n / (tb * 1e-3) / 1e9
                print(f"{set_name + ' ' + lin:<12} {f'{m}x{n}':>12} {rows:>4} {td:>8.4f} {ts:>8.4f} {tst:>8.4f} "
                      f"{tb:>8.4f} {gbs:>5.0f}  {cfg_s} | {cb}", flush=True)
                results.append({"set": set_name, "linear": lin, "m": m, "n": n, "rows": rows, "digit_ms": round(td, 5),
                                "service_ms": round(ts, 5), "stream_ms": round(tst, 5), "best_ms": round(tb, 5),
                                "service_config": str(cfg_s), "best_config": str(cb)})
            del w8, wpad
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    print("\nper layer (4 linears), device ms per decode step")
    print(f"{'set':<6} {'rows':>4} {'digit':>8} {'service':>8} {'stream':>8} {'best':>8}")
    for set_name in args.shapes.split(","):
        for rows in rows_list:
            vals = [layer[(set_name, rows, c)] for c in ("digit", "service", "stream", "best")]
            print(f"{set_name:<6} {rows:>4} " + " ".join(f"{v:>8.4f}" for v in vals))
    if args.emit_table:
        with open(args.emit_table, "w") as f:
            json.dump(table_to_json({name: best_rules}), f, indent=1)
        print(f"wrote {len(best_rules)} rules for {name!r} to {args.emit_table}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": str(dev), "gpu": name, "ring": ring.name, "mode": "gemv", "results": results}, f,
                      indent=1)
    return 0


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
    ap.add_argument("--gemv", action="store_true", help="time the fused Triton ring GEMV vs the digit-GEMM path")
    ap.add_argument("--gemv-sweep", action="store_true", help="with --gemv: also sweep launch configurations")
    ap.add_argument("--emit-table", default=None, help="with --gemv-sweep: write the winners as a device table (JSON)")
    ap.add_argument("--shapes", default="7b", help="with --gemv: shape sets, comma separated (7b, 0.5b)")
    args = ap.parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available (use --device cpu for a smoke test)", file=sys.stderr)
        return 2
    if args.gemv:
        return gemv_main(args)
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
        # digit-GEMM path only (the fused Triton GEMV has its own benchmark: triton_gemv_bench.py)
        eager = ComputeKernel(spec, w, device=args.device, cuda_graphs=False, triton_gemv=False)
        graph = ComputeKernel(spec, w, device=args.device, cuda_graphs=True, triton_gemv=False)
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
