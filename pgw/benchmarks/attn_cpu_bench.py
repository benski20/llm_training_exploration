"""Gateway attention over the private KV cache on CPU: microseconds per row vs context, GB/s and GFLOP/s.

Runs the native pinned attention call of the gateway (``pgw_fast.float_rope_attend[_bf16]``: RoPE, cache write
and attention of one layer; ``float_ops = "native"``) at the head shapes of Qwen2.5-0.5B/7B/32B/72B
(synthetic data), one decode pass at a time: every layer once, ``--rows`` rows at positions ``ctx - rows ..
ctx - 1``. The pass's layers live in distinct buffers (all ``L`` layers, capped at ``--max-gib`` of KV), so a
pass streams its real working set (beyond the last-level cache from 32B / 2k on) instead of re-reading one
layer from cache.

* ``block``: 0 = the v1 formulation (one block; parallel over KV heads and rows only), 256 = v2 "blocked"
  (parallel over KV heads, rows and 256-position blocks; same bits for any thread count).
* ``kv``: ``f32`` or ``bf16`` cache storage.
* Reports per pass (the best of ``--reps``, since the box may be shared; also the median and process CPU
  time): microseconds per layer call, per row per pass (all layers), KV bytes/s
  (``2 ctx kvh hd elem`` per layer per pass, read once by the whole pass) and FLOP/s (``4 nh hd ctx`` per
  layer per row: the q.k and e.v products).

Usage:
    PYTHONPATH=src python benchmarks/attn_cpu_bench.py [--models 0.5b,7b,32b,72b] [--ctx 512,2048,8192]
        [--threads 1,2,4] [--block 0,256] [--kv f32,bf16] [--rows 1] [--reps 5] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

#: (query heads, KV heads, head dim, layers)
SHAPES = {
    "0.5b": (14, 2, 64, 24),
    "7b": (28, 4, 128, 28),
    "32b": (40, 8, 128, 64),
    "72b": (64, 8, 128, 80),
}


def run(f: object, model: str, ctx: int, threads: int, block: int, kv: str, rows: int, reps: int,
        max_gib: float) -> dict:
    nh, kvh, hd, layers = SHAPES[model]
    elem = 4 if kv == "f32" else 2
    per_layer = 2 * ctx * kvh * hd * elem
    nl = max(1, min(layers, int(max_gib * 2**30 // per_layer)))
    rng = np.random.default_rng(0)
    dt = np.float32 if kv == "f32" else np.uint16
    base = rng.standard_normal((2, 1, ctx, kvh, hd), dtype=np.float32)
    if kv == "bf16":
        flat = np.empty(base.size, dtype=np.uint16)
        f.bf16_round(base.reshape(-1), flat)
        base = flat.reshape(base.shape)
    # distinct buffers with the same contents (the values do not change the speed)
    caches = [(np.array(base[0], dtype=dt), np.array(base[1], dtype=dt)) for _ in range(nl)]
    y = rng.standard_normal((rows, (nh + 2 * kvh) * hd), dtype=np.float32)
    cos = np.ones((rows, hd), dtype=np.float32)
    sin = np.zeros((rows, hd), dtype=np.float32)
    out = np.empty((rows, nh * hd), dtype=np.float32)
    call = f.float_rope_attend if kv == "f32" else f.float_rope_attend_bf16
    scale = hd ** -0.5
    start = ctx - rows
    walls, cpus = [], []
    for rep in range(reps + 1):
        t0, c0 = time.perf_counter(), time.process_time()
        for li in range(nl):
            kc, vc = caches[li]
            call(y, cos, sin, kc, vc, 0, start, nh, scale, out, threads=threads, block=block)
        dt_s, dc_s = time.perf_counter() - t0, time.process_time() - c0
        if rep:  # the first pass warms the pool and the scratch
            walls.append(dt_s / nl)
            cpus.append(dc_s / nl)
    best, med = min(walls), statistics.median(walls)
    flops = 4 * nh * hd * ctx * rows
    return {
        "model": model, "ctx": ctx, "threads": threads, "block": block, "kv": kv, "rows": rows,
        "layers_buffered": nl, "us_per_layer_min": round(best * 1e6, 1), "us_per_layer_median": round(med * 1e6, 1),
        "cpu_us_per_layer": round(min(cpus) * 1e6, 1), "ms_per_row_pass": round(best * layers * 1e3 / rows, 2),
        "gb_per_s": round(per_layer / best / 1e9, 2), "gflop_per_s": round(flops / best / 1e9, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="0.5b,7b,32b,72b")
    ap.add_argument("--ctx", default="512,2048,8192")
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--block", default="0,256")
    ap.add_argument("--kv", default="f32,bf16")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-gib", type=float, default=1.5)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    import pgw_fast as f

    print(f"isa level {f.attn_isa_level()} (2 = AVX-512, 1 = AVX2, 0 = baseline)")
    res = []
    for model in a.models.split(","):
        for ctx in (int(c) for c in a.ctx.split(",")):
            for kv in a.kv.split(","):
                for block in (int(b) for b in a.block.split(",")):
                    for threads in (int(t) for t in a.threads.split(",")):
                        r = run(f, model, ctx, threads, block, kv, a.rows, a.reps, a.max_gib)
                        res.append(r)
                        print(json.dumps(r), flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
