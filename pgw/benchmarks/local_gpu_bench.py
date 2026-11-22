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
    ap.add_argument("--breakdown", action="store_true", help="time graphs of copies only, the kernel only, "
                    "and back-to-back kernels per shape")
    ap.add_argument("--lm-head", default="", help="also time the GPU LM head of this VxD bfloat16 shape, "
                    "e.g. 151936x896")
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
    res["class_native"] = gp.native
    res["class_native_reason"] = gp.native_reason
    res["class_stats"] = dict(gp.stats)
    if gp.native:  # the same class with the Python replay (the path before the native kernel)
        gp.native = False
        bad = 0
        for key, c in keyed:
            gp.forward(key, xs[c["n"]], c["wnorm"], c["scale"], None, c["bound"], c["out"], sx)
            cpu_call(c, 1)
            bad += int(not np.array_equal(c["out"].view(np.uint32), c["ref"].view(np.uint32)))
        res["class_python_mismatches"] = bad
        ts = []
        for rep in range(args.reps + 1):
            t0 = time.perf_counter()
            for key, c in keyed:
                gp.forward(key, xs[c["n"]], c["wnorm"], c["scale"], None, c["bound"], c["out"], sx)
            if rep:
                ts.append(time.perf_counter() - t0)
        res["class_python_ms"] = round(1e3 * statistics.median(ts), 3)
        gp.native = True
    if args.lm_head:
        res["lm_head"] = lm_head_bench(dev, args.lm_head, args.reps)
    if args.breakdown:
        res["graph_breakdown_us"] = graph_breakdown(dev, shapes)
    print(json.dumps(res, indent=1))


def graph_breakdown(dev: torch.device, shapes: list[tuple[int, ...]]) -> dict[str, Any]:
    """Median replay + synchronize (us) of graphs with parts of a local call, per shape; and the kernel's own
    time from 200 back-to-back launches (CUDA events)."""
    from pgw.core.triton_gemv import ring_gemv

    out: dict[str, Any] = {}
    stream = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call]
    for n, m in shapes:
        w = torch.randint(-128, 128, (m, n), dtype=torch.int8, device=dev)
        h_in = torch.zeros((1, n), dtype=torch.int32, pin_memory=True)
        h_out = torch.zeros((1, m), dtype=torch.int32, pin_memory=True)
        d_in = torch.zeros((1, n), dtype=torch.int32, device=dev)
        d_out = torch.zeros((1, m), dtype=torch.int32, device=dev)
        parts = {
            "h2d": lambda: d_in.copy_(h_in, non_blocking=True),
            "d2h": lambda: h_out.copy_(d_out, non_blocking=True),
            "kernel": lambda: ring_gemv(d_in, w, 32, n=n, m=m, out=d_out),
            "h2d_d2h": lambda: (d_in.copy_(h_in, non_blocking=True), h_out.copy_(d_out, non_blocking=True)),
            "full": lambda: (d_in.copy_(h_in, non_blocking=True), ring_gemv(d_in, w, 32, n=n, m=m, out=d_out),
                             h_out.copy_(d_out, non_blocking=True)),
        }
        row: dict[str, float] = {}
        for name, body in parts.items():
            with torch.cuda.stream(stream):
                body()
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                body()
            ts = []
            for _ in range(400):
                t0 = time.perf_counter()
                with torch.cuda.stream(stream):
                    g.replay()
                stream.synchronize()
                ts.append(time.perf_counter() - t0)
            row[name] = round(1e6 * statistics.median(ts[50:]), 1)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            e0.record(stream)
            for _ in range(200):
                ring_gemv(d_in, w, 32, n=n, m=m, out=d_out)
            e1.record(stream)
        stream.synchronize()
        row["kernel_b2b"] = round(1e3 * e0.elapsed_time(e1) / 200, 1)
        out[f"{n}x{m}"] = row
    return out


def lm_head_bench(dev: torch.device, shape: str, reps: int) -> dict[str, Any]:
    """The fixed-order LM head on the GPU (:mod:`pgw.engine.lm_head_gpu`) vs the native CPU kernel: self-check
    (bit for bit) and median ms per 1-row call."""
    from pgw.engine.lm_head import lm_head_logits
    from pgw.engine.lm_head_gpu import GpuLmHead

    v, d = (int(t) for t in shape.split("x"))
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(v, d, generator=g) * 0.03).to(torch.bfloat16)
    t0 = time.perf_counter()
    head = GpuLmHead(w, dev)
    why = head.self_check(w)
    out: dict[str, Any] = {"shape": shape, "setup_s": round(time.perf_counter() - t0, 2), "self_check": why or "ok"}
    h = torch.randn(1, d, generator=g)
    bad = 0
    for i in range(20):
        hh = h * (i + 1) * 0.3
        bad += int(not torch.equal(head.logits(hh).view(torch.int32), lm_head_logits(hh, w).view(torch.int32)))
    out["mismatches_20_rows"] = bad
    for name, fn in (("gpu_ms", lambda: head.logits(h)), ("cpu_ms", lambda: lm_head_logits(h, w))):
        ts = []
        for _ in range(reps + 1):
            t1 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t1)
        out[name] = round(1e3 * statistics.median(ts[1:]), 3)
    for bv in (32, 128):
        alt = GpuLmHead(w, dev, block_v=bv)
        alt.logits(h)
        ts = []
        for _ in range(reps):
            t1 = time.perf_counter()
            alt.logits(h)
            ts.append(time.perf_counter() - t1)
        out[f"gpu_block{bv}_ms"] = round(1e3 * statistics.median(ts), 3)
        out[f"gpu_block{bv}_equal"] = bool(torch.equal(alt.logits(h), head.logits(h)))
        del alt
    return out


if __name__ == "__main__":
    main()
