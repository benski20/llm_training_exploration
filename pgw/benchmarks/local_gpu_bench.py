"""Decode sweep of the gateway's exact local linears on a gateway-owned GPU vs the native CPU kernel.

Per call (one row, K = 32 ring): the native quantizer with a zero mask writes the ring payload straight
into a pinned host buffer, a captured CUDA graph ``[H2D, ring_gemv, D2H]`` computes ``x_hat @ W8^T mod
2^32`` (the provider's exact lane kernel), and the native finish (zero ``W r``) range-checks and rescales.
That is the masked trip with ``r = 0`` and the product on the gateway's own device, so the output must be
bit-identical to ``pgw_fast.local_linear_into`` (checked on every call of the first sweep).

    PYTHONPATH=src python benchmarks/local_gpu_bench.py --layers 24 [--shapes 896x1152,896x896,4864x896]

Prints JSON: ms per sweep (all layers' shapes once, distinct weights) for the CPU kernel (threads 1 and
pool) and the GPU path per synchronisation method, plus per-call medians.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="896x1152,896x896,4864x896")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_num_threads(1)
    from pgw.core import native
    from pgw.core.ring import DEFAULT_RING
    from pgw.core.triton_gemv import ring_gemv

    ring = DEFAULT_RING
    assert ring.k == 32, ring
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    stream = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call]
    pool = torch.cuda.graph_pool_handle()
    layers: list[list[dict[str, Any]]] = []
    t_cap = time.perf_counter()
    for _ in range(args.layers):
        row = []
        for n, m in shapes:
            w8t = rng.integers(-128, 128, size=(n, m), dtype=np.int8)
            wnorm = float(np.sqrt((w8t.astype(np.float64) ** 2).sum(axis=0)).max())
            scale = (rng.random(m, dtype=np.float32) * np.float32(1e-3)).astype(np.float32)
            w_dev = torch.from_numpy(np.ascontiguousarray(w8t.T)).to(dev)  # (m, n) int8
            h_in = torch.zeros((1, n), dtype=torch.int32, pin_memory=True)
            h_out = torch.zeros((1, m), dtype=torch.int32, pin_memory=True)
            d_in = torch.zeros((1, n), dtype=torch.int32, device=dev)
            d_out = torch.zeros((1, m), dtype=torch.int32, device=dev)

            def body(h_in=h_in, h_out=h_out, d_in=d_in, d_out=d_out, w=w_dev, n=n, m=m) -> None:
                d_in.copy_(h_in, non_blocking=True)
                ring_gemv(d_in, w, 32, n=n, m=m, out=d_out)
                h_out.copy_(d_out, non_blocking=True)

            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                body()
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=stream):
                body()
            row.append({"n": n, "m": m, "w8t": w8t, "wnorm": wnorm, "scale": scale, "bound": int(ring.output_bound(n)),
                        "g": g, "h_in": h_in, "h_out": h_out, "pin_in": h_in.numpy().view(np.uint8).reshape(-1),
                        "pin_out": h_out.numpy().view(np.uint8).reshape(-1), "w_dev": w_dev,
                        "zr": np.zeros((1, n), dtype=np.int64), "zwr": np.zeros((1, m), dtype=np.int64),
                        "out": np.empty((1, m), np.float32), "ref": np.empty((1, m), np.float32)})
        layers.append(row)
    capture_s = time.perf_counter() - t_cap
    xs = {n: (rng.standard_normal((1, n)) * 2).astype(np.float32) for n, _ in shapes}
    sx = np.empty(1, dtype=np.float64)
    sx2 = np.empty(1, dtype=np.float64)
    ev = torch.cuda.Event()
    mismatches = 0

    def gpu_call(c: dict[str, Any], sync: str) -> None:
        native.fast.trip_prepare_into(xs[c["n"]], c["zr"], ring.k, ring.a, ring.cap, c["wnorm"], c["pin_in"], sx)
        with torch.cuda.stream(stream):
            c["g"].replay()
            if sync == "event_spin":
                ev.record(stream)
        if sync == "stream_sync":
            stream.synchronize()
        elif sync == "event_spin":
            while not ev.query():
                pass
        st, _ = native.fast.trip_finish_into(c["pin_out"], c["zwr"], sx, c["scale"], None, ring.k, c["bound"],
                                             c["out"])
        if st:
            raise RuntimeError(f"finish status {st}")

    def cpu_call(c: dict[str, Any], threads: int) -> None:
        native.fast.local_linear_into(xs[c["n"]], c["w8t"], c["scale"], None, ring.k, ring.a, ring.cap, c["wnorm"],
                                      c["bound"], sx2, c["ref"], threads)

    # exactness: every (layer, shape) once, GPU vs the native CPU kernel, plus perturbed inputs
    for trial in range(3):
        for n, _ in shapes:
            xs[n] = (rng.standard_normal((1, n)) * (3 if trial else 1)).astype(np.float32)
            if trial == 2:
                xs[n][0, int(rng.integers(0, n))] = 60.0
        for row in layers:
            for c in row:
                gpu_call(c, "stream_sync")
                cpu_call(c, 1)
                if not np.array_equal(c["out"].view(np.uint32), c["ref"].view(np.uint32)):
                    mismatches += 1
    res: dict[str, Any] = {"layers": args.layers, "shapes": args.shapes, "capture_s": round(capture_s, 2),
                           "gpu": torch.cuda.get_device_name(dev), "mismatches": mismatches,
                           "calls_per_sweep": args.layers * len(shapes),
                           "weights_mb": round(sum(n * m for n, m in shapes) * args.layers / 1e6, 1)}

    def sweep(fn: Any) -> float:
        ts = []
        for rep in range(args.reps + 1):
            t0 = time.perf_counter()
            for row in layers:
                for c in row:
                    fn(c)
            if rep:
                ts.append(time.perf_counter() - t0)
        return 1e3 * statistics.median(ts)

    for threads in (1, 0):
        res[f"cpu_threads{threads}_ms"] = round(sweep(lambda c, t=threads: cpu_call(c, t)), 3)
    for sync in ("stream_sync", "event_spin"):
        res[f"gpu_{sync}_ms"] = round(sweep(lambda c, s=sync: gpu_call(c, s)), 3)
    # per-call breakdown of the GPU path (layer 0 shapes)
    per: dict[str, Any] = {}
    for c in layers[0]:
        tp, tr, tf = [], [], []
        for _ in range(200):
            t0 = time.perf_counter()
            native.fast.trip_prepare_into(xs[c["n"]], c["zr"], ring.k, ring.a, ring.cap, c["wnorm"], c["pin_in"], sx)
            t1 = time.perf_counter()
            with torch.cuda.stream(stream):
                c["g"].replay()
            stream.synchronize()
            t2 = time.perf_counter()
            native.fast.trip_finish_into(c["pin_out"], c["zwr"], sx, c["scale"], None, ring.k, c["bound"], c["out"])
            t3 = time.perf_counter()
            tp.append(t1 - t0)
            tr.append(t2 - t1)
            tf.append(t3 - t2)
        per[f"{c['n']}x{c['m']}"] = {k: round(1e6 * statistics.median(v), 1)
                                     for k, v in (("prepare_us", tp), ("replay_sync_us", tr), ("finish_us", tf))}
    res["per_call"] = per
    # the gateway's class (pgw.engine.local_gpu.LocalGpuProducts): same products, checked against the CPU kernel
    from pgw.engine.local_gpu import LocalGpuProducts

    gp = LocalGpuProducts(dev, ring.k, ring.a, ring.cap, max_n=max(n for n, _ in shapes),
                          max_m=max(m for _, m in shapes))
    for li, row in enumerate(layers):
        for c in row:
            gp.add(f"{li}.{c['n']}x{c['m']}", torch.from_numpy(np.ascontiguousarray(c["w8t"].T)))
    bad = 0
    for li, row in enumerate(layers):
        for c in row:
            gp.forward(f"{li}.{c['n']}x{c['m']}", xs[c["n"]], c["wnorm"], c["scale"], None, c["bound"], c["out"], sx)
            cpu_call(c, 1)
            bad += int(not np.array_equal(c["out"].view(np.uint32), c["ref"].view(np.uint32)))
    res["class_mismatches"] = bad
    keyed = [(f"{li}.{c['n']}x{c['m']}", c) for li, row in enumerate(layers) for c in row]
    ts = []
    for rep in range(args.reps + 1):
        t0 = time.perf_counter()
        for key, c in keyed:
            gp.forward(key, xs[c["n"]], c["wnorm"], c["scale"], None, c["bound"], c["out"], sx)
        if rep:
            ts.append(time.perf_counter() - t0)
    res["class_gpu_ms"] = round(1e3 * statistics.median(ts), 3)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
