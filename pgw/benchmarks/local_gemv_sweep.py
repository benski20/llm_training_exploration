"""Launch-configuration sweep of the fused ring GEMV for the gateway's 1-row local linears (decode).

For each shape and configuration: the full local graph ``[H2D, ring_gemv, D2H]`` as
``pgw.engine.local_gpu`` captures it, replayed and synchronized (median wall us: what a decode call waits
for), and the kernel's device time (20 launches captured in one graph, CUDA events). Every configuration's
output is compared with the default configuration's (exact wrapping arithmetic: any configuration gives the
same integers; the gateway's start-up self-check repeats that against the CPU reference).

    PYTHONPATH=src python benchmarks/local_gemv_sweep.py [--shapes 896x1152,896x896,4864x896]

Prints JSON (per shape: the default, the best few configurations, and mismatches).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from typing import Any

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="896x1152,896x896,4864x896")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()
    from pgw.core.ringmm import pad_weight_cuda
    from pgw.core.triton_gemv import GemvConfig, gemv_config, ring_gemv

    dev = torch.device(args.device)
    stream = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call]
    res: dict[str, Any] = {"gpu": torch.cuda.get_device_name(dev)}
    for shape in args.shapes.split(","):
        n, m = (int(v) for v in shape.split("x"))
        w = pad_weight_cuda(torch.randint(-128, 128, (m, n), dtype=torch.int8), dev)
        nw = int(w.shape[1])
        h_in = torch.randint(-(1 << 11), 1 << 11, (1, n), dtype=torch.int32).pin_memory()
        h_out = torch.zeros((1, m), dtype=torch.int32, pin_memory=True)
        d_in = torch.zeros((1, n), dtype=torch.int32, device=dev)
        d_out = torch.zeros((1, m), dtype=torch.int32, device=dev)
        d_in.copy_(h_in)
        ref = ring_gemv(d_in, w, 32, n=n, m=m).cpu()
        default = gemv_config(1, m, nw, dev)
        cands = [default]
        for bm, bn, sk, nw_ in itertools.product((2, 4, 8, 16), (256, 512, 1024), (1, 2, 4, 8), (1, 2, 4)):
            if bn > 2 * nw or (sk > 1 and nw // sk < bn):
                continue
            cands.append(GemvConfig(1, bm, bn, sk, nw_))
        rows: list[dict[str, Any]] = []
        bad = 0
        for cfg in cands:
            try:
                def body(cfg: GemvConfig = cfg, d_in: torch.Tensor = d_in, h_in: torch.Tensor = h_in,
                         d_out: torch.Tensor = d_out, h_out: torch.Tensor = h_out, w: torch.Tensor = w,
                         n: int = n, m: int = m) -> None:
                    d_in.copy_(h_in, non_blocking=True)
                    ring_gemv(d_in, w, 32, n=n, m=m, out=d_out, config=cfg)
                    h_out.copy_(d_out, non_blocking=True)

                with torch.cuda.stream(stream):
                    body()
                stream.synchronize()
                if not torch.equal(h_out, ref):
                    bad += 1
                    continue
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    body()
                gk = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gk, stream=stream):
                    for _ in range(20):
                        ring_gemv(d_in, w, 32, n=n, m=m, out=d_out, config=cfg)
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    with torch.cuda.stream(stream):
                        g.replay()
                    stream.synchronize()
                    ts.append(time.perf_counter() - t0)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    gk.replay()
                    e0.record(stream)
                    for _ in range(5):
                        gk.replay()
                    e1.record(stream)
                stream.synchronize()
                rows.append({"cfg": [cfg.block_m, cfg.block_n, cfg.split_k, cfg.num_warps],
                             "call_us": round(1e6 * statistics.median(ts[30:]), 1),
                             "kernel_us": round(1e3 * e0.elapsed_time(e1) / 100, 2), "default": cfg == default})
            except Exception as exc:  # a config that does not compile or launch
                rows.append({"cfg": [cfg.block_m, cfg.block_n, cfg.split_k, cfg.num_warps],
                             "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
        ok = [r for r in rows if "call_us" in r]
        ok.sort(key=lambda r: (r["call_us"], r["kernel_us"]))
        res[shape] = {"default": next((r for r in ok if r["default"]), None), "best": ok[:6],
                      "best_kernel": sorted(ok, key=lambda r: r["kernel_us"])[:4], "mismatches": bad,
                      "configs": len(cands), "errors": sum(1 for r in rows if "error" in r)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
