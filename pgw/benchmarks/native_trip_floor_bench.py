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

``--prime``: the prime-field check format (``check_field = "p31"``, SPEC 5.1) instead: ``c = 1`` vector over
``F_p`` for every linear (head included), the input half ``uw . xq mod p`` fused in ``trip_prepare_into(...,
field=p)``, and the output half with the range check, ``trip_prime_output`` on ``y_hat`` and ``W r``, run
synchronously on the main core before ``trip_finish_into`` (no check there, nothing on the helper core).

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
    # Qwen2.5-32B: 40 heads / 8 KV heads of 128; Qwen2.5-72B: 64 / 8
    "32b": {"hidden": 5120, "inter": 27648, "q": 5120, "kv": 1024, "vocab": 152064, "layers": 64},
    "72b": {"hidden": 8192, "inter": 29568, "q": 8192, "kv": 1024, "vocab": 152064, "layers": 80},
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
    ap.add_argument("--prime", action="store_true", help="prime-field check format (one vector, p = 2^31 - 1)")
    ap.add_argument("--cores", type=int, default=1,
                    help="split each trip's prepare / finish / prime check over this many pool tasks (PGW_TRIP_SPLIT)")
    ap.add_argument("--rows", type=int, default=1, help="rows per trip (R)")
    ap.add_argument("--stat", choices=["median", "min"], default="median", help="per-token statistic (min: a busy box)")
    ap.add_argument("--r-offcore", action="store_true",
                    help="count the r keystream as helper-core time (derived ahead, PGW_TRIP_R_AHEAD)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    R, cores = args.rows, args.cores
    split = cores if cores >= 2 else 0
    s = SHAPES[args.preset]
    layers = args.layers or s["layers"]
    k, a, cap, c = 32, 12, 27, (1 if args.prime else args.checks)
    p31 = (1 << 31) - 1
    hi = p31 if args.prime else 2**32
    h, i = s["hidden"], s["inter"]
    per_layer = [("qkv", h, s["q"] + 2 * s["kv"]), ("o", s["q"], h), ("gate_up", h, 2 * i), ("down", i, h)]
    lins = [(f"{li}.{kind}", n, m) for li in range(layers) for kind, n, m in per_layer]
    if not args.no_head:
        lins.append((f"{layers}.lm_head", h, s["vocab"]))
    rng = np.random.default_rng(0)
    slots = (args.tokens + 2) * R
    st = []
    for key, n, m in lins:
        st.append({
            "key": key, "n": n, "m": m,
            "rkey": bytearray(rng.integers(0, 256, 32, dtype=np.uint8).tobytes()),
            "wr_packed": rng.integers(0, 256, slots * m * 4, dtype=np.uint8).tobytes(),
            "uw": rng.integers(0, hi, (c, n), dtype=np.uint64).astype(np.uint32),
            "u": rng.integers(0, hi, (c, m), dtype=np.uint64).astype(np.uint32),
            "y": rng.integers(0, 256, R * m * 4, dtype=np.uint8).tobytes(),
            "scale": np.full(m, 1e-3, np.float32),
            "wn": float(args.wnorm_frac * 127 * np.sqrt(n)),
            "x": rng.standard_normal((R, n)).astype(np.float32),
        })
    stages = ("r", "wr", "prepare", "prime_check", "finish", "freivalds_out")
    kt: dict[str, list[float]] = {}  # main-core seconds per linear kind, per token
    kn: dict[str, int] = {}  # trips per token per kind
    for e in st:
        kn[e["key"].split(".", 1)[1]] = kn.get(e["key"].split(".", 1)[1], 0) + 1
    per_tok: dict[str, list[float]] = {x: [] for x in stages}
    for t in range(args.tokens):
        acc = dict.fromkeys(stages, 0.0)
        acc_k: dict[str, float] = {}
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
            if args.prime:
                f.trip_prepare_into(e["x"], r, k, a, cap, e["wn"], buf, sx, None, e["uw"], rhs, 1, field=p31,
                                    **({"split": split} if split else {}))
            else:
                f.trip_prepare_into(e["x"], r, k, a, cap, e["wn"], buf, sx, None, None if last else e["uw"],
                                    None if last else rhs, 1, **({"split": split} if split else {}))
            r.fill(0)
            t3 = time.perf_counter()
            out = np.empty((R, m), np.float32)
            prime_dt = 0.0
            th = cores if cores >= 2 else 1
            if args.prime:  # synchronous: range + u . yq == rhs (mod p), then the finish without a check
                f.trip_prime_output(e["y"], wr, e["u"], rhs, k, 1 << cap, **({"split": split} if split else {}))
                t35 = time.perf_counter()
                prime_dt = t35 - t3
                f.trip_finish_into(e["y"], wr, sx, e["scale"], None, k, 1 << 40, out, threads=th)
                t4 = time.perf_counter()
                acc["prime_check"] += t35 - t3
                t3 = t35
            elif last:  # synchronous full check (the logits are released right after)
                f.trip_finish_into(e["y"], wr, sx, e["scale"], None, k, 1 << 40, out, x_hat=buf.tobytes(),
                                   u=e["u"], uw=e["uw"], threads=th)
            else:
                f.trip_finish_into(e["y"], wr, sx, e["scale"], None, k, 1 << 40, out, threads=th)
            t4 = time.perf_counter()
            if not last and not args.prime:
                f.trip_freivalds_output(e["y"], e["u"], rhs, k)
            t5 = time.perf_counter()
            for name, dt in (("r", t1 - t0), ("wr", t2 - t1), ("prepare", t3 - t2), ("finish", t4 - t3),
                             ("freivalds_out", t5 - t4)):
                acc[name] += dt
            kd = e["key"].split(".", 1)[1]
            acc_k[kd] = acc_k.get(kd, 0.0) + (t4 - t1) + (0.0 if args.r_offcore else t1 - t0)
        for name in stages:
            per_tok[name].append(acc[name])
        for kd, v in acc_k.items():
            kt.setdefault(kd, []).append(v)
    agg = min if args.stat == "min" else statistics.median
    med = {x: round(1e3 * agg(v), 3) for x, v in per_tok.items()}
    ktm = {kd: agg(v) for kd, v in kt.items()}
    on_main = ("wr", "prepare", "prime_check", "finish") + (() if args.r_offcore else ("r",))
    main_core = round(sum(med[x] for x in on_main), 3)
    trips = len(lins)
    full = s["layers"] / layers
    out = {
        "preset": args.preset, "layers": layers, "head": not args.no_head, "trips_per_token": len(lins),
        "checks": c, "check_format": "p31" if args.prime else "ring", "ms_per_token": med, "main_core_ms_per_token": main_core,
        "helper_core_ms_per_token": round(med["freivalds_out"] + (med["r"] if args.r_offcore else 0.0), 3),
        "rows_per_trip": R, "cores": cores, "r_offcore": args.r_offcore,
        "main_us_per_trip": round(1e3 * main_core / trips, 2),
        "main_us_per_trip_by_kind": {kd: round(1e6 * v / kn[kd], 2) for kd, v in ktm.items()},
        "main_core_ms_per_token_full_model": round(1e3 * sum(
            v * (1 if kd == "lm_head" else full) for kd, v in ktm.items()), 2),
        "check_vector_mb_per_token": round(sum(c * (e["n"] + e["m"]) * 4 for e in st) / 1e6, 2),
        "wr_mb_per_token": round(sum(e["m"] * 4 * R for e in st) / 1e6, 3),
        "r_keystream_mb_per_token": round(sum(e["n"] * 8 * R for e in st) / 1e6, 3),
    }
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
