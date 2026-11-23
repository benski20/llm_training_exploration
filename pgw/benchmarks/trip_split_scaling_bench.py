"""Per-phase scaling of one trip's gateway kernels over the native pool (``PGW_TRIP_SPLIT``) at real widths.

For each linear kind of a preset (Qwen2.5-7B / 32B / 72B: qkv, o, gate_up, down, the masked LM head) and each
core count ``c`` it times, back to back on warm pool workers, the kernels the native decode driver runs on the
critical path of a trip with the p31 check format:

* ``prepare``: ``trip_prepare_into(..., field=p31, split=c)`` (constant-time quantize + mask + pack + the prime
  input half; ``r`` is derived ahead on a helper in the gateway, so it is given here),
* ``check``: ``trip_prime_output(..., split=c)`` (range check + ``u . y_hat - u . W r == rhs`` mod p),
* ``finish``: ``trip_finish_into(..., threads=c)`` (unmask, rescale).

Reports the minimum and median microseconds per phase and kind, the speed-up over ``c = 1`` and a fit
``t(c) = serial + parallel / c`` per kind (least squares over the measured ``c``), which is what the
extrapolation to a many-core gateway uses (``--extrapolate 8,16``). Run it with the pool kept warm
(``PGW_POOL_SPIN_US`` >= 2000 or ``PGW_POOL_ACTIVE_SPIN_US``) and ``PGW_POOL_WORKERS`` >= max(c) - 1; on a
shared box, prefer ``min``.

Usage:
    PGW_POOL_SPIN_US=5000 PYTHONPATH=src python benchmarks/trip_split_scaling_bench.py --preset 32b --cores 1,2,3,4 \
        [--reps 60] [--rows 1] [--extrapolate 8,16] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

SHAPES = {
    "7b": {"hidden": 3584, "inter": 18944, "q": 3584, "kv": 512, "vocab": 152064, "layers": 28},
    "32b": {"hidden": 5120, "inter": 27648, "q": 5120, "kv": 1024, "vocab": 152064, "layers": 64},
    "72b": {"hidden": 8192, "inter": 29568, "q": 8192, "kv": 1024, "vocab": 152064, "layers": 80},
}


def main() -> None:
    from pgw.core.native import fast as f

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="32b", choices=sorted(SHAPES))
    ap.add_argument("--cores", default="1,2,3,4")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--wnorm-frac", type=float, default=0.3)
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--extrapolate", default="8,16")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    s = SHAPES[args.preset]
    h, i = s["hidden"], s["inter"]
    kinds = [("qkv", h, s["q"] + 2 * s["kv"]), ("o", s["q"], h), ("gate_up", h, 2 * i), ("down", i, h)]
    if not args.no_head:
        kinds.append(("lm_head", h, s["vocab"]))
    cores = [int(c) for c in args.cores.split(",")]
    R, k, a, cap = args.rows, 32, 12, 27
    p31 = (1 << 31) - 1
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for kind, n, m in kinds:
        x = rng.standard_normal((R, n)).astype(np.float32)
        r = rng.integers(-(2**31), 2**31, (R, n), dtype=np.int64)
        uw = rng.integers(0, p31, (1, n), dtype=np.uint64).astype(np.uint32)
        u = rng.integers(0, p31, (1, m), dtype=np.uint64).astype(np.uint32)
        y = rng.integers(0, 256, R * m * 4, dtype=np.uint8).tobytes()
        wr0 = rng.integers(0, 2**32, (R, m), dtype=np.uint64).astype(np.uint32)
        scale = np.full(m, 1e-3, np.float32)
        wn = float(args.wnorm_frac * 127 * np.sqrt(n))
        buf = np.empty(R * n * 4, np.uint8)
        sx = np.empty(R)
        rhs = np.empty((R, 1), np.uint64)
        res = np.empty((R, m), np.float32)
        per_c: dict[int, dict[str, float]] = {}
        for c in cores:
            sp = {"split": c} if c >= 2 else {}
            ts: dict[str, list[float]] = {"prepare": [], "check": [], "finish": []}
            for rep in range(args.reps + 5):
                rr = r.copy()
                wr = wr0.copy()
                t0 = time.perf_counter()
                f.trip_prepare_into(x, rr, k, a, cap, wn, buf, sx, None, uw, rhs, 1, field=p31, **sp)
                t1 = time.perf_counter()
                f.trip_prime_output(y, wr, u, rhs, k, 1 << cap, **sp)  # status irrelevant: timing only
                t2 = time.perf_counter()
                f.trip_finish_into(y, wr, sx, scale, None, k, 1 << 40, res, threads=c)
                t3 = time.perf_counter()
                if rep >= 5:
                    ts["prepare"].append(t1 - t0)
                    ts["check"].append(t2 - t1)
                    ts["finish"].append(t3 - t2)
            row = {}
            for ph, v in ts.items():
                row[f"{ph}_min_us"] = round(1e6 * min(v), 1)
                row[f"{ph}_med_us"] = round(1e6 * statistics.median(v), 1)
            row["total_min_us"] = round(sum(row[f"{p}_min_us"] for p in ts), 1)
            row["total_med_us"] = round(sum(row[f"{p}_med_us"] for p in ts), 1)
            per_c[c] = row
        # fit t(c) = serial + par / c on the minima (least squares in 1/c)
        xs = np.array([1.0 / c for c in cores])
        ys = np.array([per_c[c]["total_min_us"] for c in cores])
        if len(cores) >= 2:
            A = np.vstack([np.ones_like(xs), xs]).T
            (ser, par), *_ = np.linalg.lstsq(A, ys, rcond=None)
        else:
            ser, par = 0.0, float(ys[0])
        ext = {int(c): round(float(ser + par / int(c)), 1) for c in args.extrapolate.split(",") if c}
        out[kind] = {"n": n, "m": m, "by_cores": per_c, "fit_serial_us": round(float(ser), 1),
                     "fit_parallel_us": round(float(par), 1), "extrapolated_total_us": ext}
        line = "  ".join(f"c{c}: {per_c[c]['total_min_us']:6.1f} ({per_c[c]['prepare_min_us']:.0f}/"
                         f"{per_c[c]['check_min_us']:.0f}/{per_c[c]['finish_min_us']:.0f})" for c in cores)
        print(f"{kind:8s} n={n:6d} m={m:6d}  {line}  fit {ser:.1f} + {par:.1f}/c -> {ext}")
    L = s["layers"]
    summ = {}
    for c in cores + [int(x) for x in args.extrapolate.split(",") if x]:
        tot = 0.0
        for kind, _n, _m in kinds:
            o = out[kind]
            v = o["by_cores"][c]["total_min_us"] if c in o["by_cores"] else o["extrapolated_total_us"][c]
            tot += v * (1 if kind == "lm_head" else L)
        trips = 4 * L + (0 if args.no_head else 1)
        summ[c] = {"ms_per_token": round(tot / 1e3, 2), "us_per_trip": round(tot / trips, 1)}
    print("per token (kernels on the critical path, r ahead off-core):", json.dumps(summ))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"preset": args.preset, "rows": R, "kinds": out, "per_token": summ}, fh, indent=1)


if __name__ == "__main__":
    main()
