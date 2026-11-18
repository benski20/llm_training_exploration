"""Gateway cost of Freivalds checks (uint64 numpy, the PGW hot path freivalds_check_u64) for 7B/70B shapes."""
import json, os, time, sys
import numpy as np
SHAPES = {  # per layer: (m_out, n_in) for qkv, o, gate_up, down
    "qwen2.5-7b": (28, [(4608, 3584), (3584, 3584), (37888, 3584), (3584, 18944)]),
    "llama-3-70b": (80, [(10240, 8192), (8192, 8192), (57344, 8192), (8192, 28672)]),
}
rng = np.random.default_rng(0)
mask = np.uint64((1 << 48) - 1)
out = {}
for model, (L, shapes) in SHAPES.items():
    for T in (1, 16):
        for c in (1, 2, 4, 20):
            tot = 0.0
            for m, n in shapes:
                x = rng.integers(0, 1 << 48, (T, n), dtype=np.uint64)
                y = rng.integers(0, 1 << 48, (T, m), dtype=np.uint64)
                u = rng.integers(0, 1 << 48, (c, m), dtype=np.uint64)
                uw = rng.integers(0, 1 << 48, (c, n), dtype=np.uint64)
                reps = 5
                t0 = time.perf_counter()
                for _ in range(reps):
                    lhs = (y @ u.T) & mask
                    rhs = (x @ uw.T) & mask
                    ok = (lhs == rhs).all(axis=1)
                tot += (time.perf_counter() - t0) / reps
            per_tok_ms = tot * L / T * 1e3
            macs = sum(m + n for m, n in shapes) * L * c
            out[f"{model}/T{T}/c{c}"] = {"ms_per_token": round(per_tok_ms, 2), "macs_per_token": macs,
                                         "GMAC_s": round(macs / (per_tok_ms / 1e3) / 1e9, 2)}
            print(model, T, c, out[f"{model}/T{T}/c{c}"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
