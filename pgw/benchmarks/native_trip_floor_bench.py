"""Native floor of the gateway's masked-trip work per decoded token at real shapes (no Python orchestration).

For every remote linear of one token (default Qwen2.5-7B with all four linears of every layer remote, plus
the masked LM head of ``pgw.engine.masked_head``) it runs exactly the native kernels the TripEngine runs, on
distinct per-linear buffers so the caches see a real token's working set (check vectors, ``W r`` rows):

* ``mask_rows_with_key``: the one-time pad ``r`` from the segment's ChaCha20 stream (SPEC 7.3),
* ``unpack_rows_u32``: the ``W r`` row out of a K-bit packed RAM buffer (the mask ring's in-place read),
* ``trip_prepare_into``: constant-time quantize + mask + pack, with the Freivalds input half ``uw . x_hat``,
* ``trip_finish_into``: unmask, range check, rescale (split check), the head with its full check inline
  (the last linear is always checked before its output is used),
* ``trip_freivalds_output``: the deferred output half ``u . y_hat`` of every other linear (the gateway runs it
  on its helper thread: a second core).

Reports microseconds per token for each stage (median of ``--tokens`` tokens), the main-core total and the
helper-core total. This is what a gateway whose per-trip orchestration were native would spend; the
measured Python path is in ``decode_cpu_bench.py``.

Usage:
    PYTHONPATH=src python benchmarks/native_trip_floor_bench.py [--preset 7b] [--layers 28] [--tokens 20] [--no-head]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

SHAPES = {
    "7b": {"hidden": 3584, "inter": 18944, "q": 3584, "kv": 512, "vocab": 152064, "layers": 28},
    "0.5b": {"hidden": 896, "inter": 4864, "q": 896, "kv": 128, "vocab": 151936, "layers": 24},
}


def main() -> None:
    import pgw_fast as f

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="7b", choices=sorted(SHAPES))
    ap.add_argument("--layers", type=int, default=0, help="0 = the preset's full layer count")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--checks", type=int, default=7)
    ap.add_argument("--wnorm-frac", type=float, default=0.3)
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--rows", type=int, default=1, help="rows per trip (strict speculation: R rows per pass)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    s = SHAPES[args.preset]
    layers = args.layers or s["layers"]
    k, a, cap, c = 32, 12, 27, args.checks
    h, i = s["hidden"], s["inter"]
    per_layer = [("qkv", h, s["q"] + 2 * s["kv"]), ("o", s["q"], h), ("gate_up", h, 2 * i), ("down", i, h)]
    lins = [(f"{li}.{kind}", n, m) for li in range(layers) for kind, n, m in per_layer]
    if not args.no_head:
        lins.append((f"{layers}.lm_head", h, s["vocab"]))
    rng = np.random.default_rng(0)
    R = max(1, args.rows)
    slots = (args.tokens + 2) * R
    st = []
    for key, n, m in lins:
        st.append({
            "key": key, "n": n, "m": m,
            "rkey": bytearray(rng.integers(0, 256, 32, dtype=np.uint8).tobytes()),
            "wr_packed": rng.integers(0, 256, slots * m * 4, dtype=np.uint8).tobytes(),
            "uw": rng.integers(0, 2**32, (c, n), dtype=np.uint64).astype(np.uint32),
            "u": rng.integers(0, 2**32, (c, m), dtype=np.uint64).astype(np.uint32),
            "y": rng.integers(0, 256, R * m * 4, dtype=np.uint8).tobytes(),
            "scale": np.full(m, 1e-3, np.float32),
            "wn": float(args.wnorm_frac * 127 * np.sqrt(n)),
            "x": rng.standard_normal((R, n)).astype(np.float32),
        })
    stages = ("r", "wr", "prepare", "finish", "freivalds_out")
    per_tok: dict[str, list[float]] = {x: [] for x in stages}
    for t in range(args.tokens):
        acc = dict.fromkeys(stages, 0.0)
        for e in st:
            n, m = e["n"], e["m"]
            last = e["key"].endswith("lm_head")
            t0 = time.perf_counter()
            r = f.mask_rows_with_key(e["rkey"], n, k, t * R, R)
            t1 = time.perf_counter()
            wr = f.unpack_rows_u32(e["wr_packed"], t * R, R, m, k)
            t2 = time.perf_counter()
            buf = np.empty(R * n * 4, np.uint8)
            sx = np.empty(R)
            rhs = np.empty((R, c), np.uint64)
            f.trip_prepare_into(e["x"], r, k, a, cap, e["wn"], buf, sx, None, None if last else e["uw"],
                                None if last else rhs, 1)
            r.fill(0)
            t3 = time.perf_counter()
            out = np.empty((R, m), np.float32)
            if last:  # synchronous full check (the logits are released right after)
                f.trip_finish_into(e["y"], wr, sx, e["scale"], None, k, 1 << 40, out, x_hat=buf.tobytes(),
                                   u=e["u"], uw=e["uw"], threads=1)
            else:
                f.trip_finish_into(e["y"], wr, sx, e["scale"], None, k, 1 << 40, out, threads=1)
            t4 = time.perf_counter()
            if not last:
                f.trip_freivalds_output(e["y"], e["u"], rhs, k)
            t5 = time.perf_counter()
            for name, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4), strict=True):
                acc[name] += dt
        for name in stages:
            per_tok[name].append(acc[name])
    med = {x: round(1e3 * statistics.median(v), 3) for x, v in per_tok.items()}
    main_core = round(med["r"] + med["wr"] + med["prepare"] + med["finish"], 3)
    out = {
        "preset": args.preset, "rows": R, "layers": layers, "head": not args.no_head, "trips_per_token": len(lins),
        "checks": c, "ms_per_token": med, "main_core_ms_per_token": main_core,
        "helper_core_ms_per_token": med["freivalds_out"],
        "check_vector_mb_per_token": round(sum(c * (e["n"] + e["m"]) * 4 for e in st) / 1e6, 2),
        "wr_mb_per_token": round(sum(e["m"] * 4 for e in st) / 1e6, 3),
        "r_keystream_mb_per_token": round(sum(e["n"] * 8 for e in st) / 1e6, 3),
    }
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
