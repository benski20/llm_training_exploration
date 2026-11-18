"""Summarize bound statistics at A=12: ring bits needed per group (CS bound, actual max, static worst case)."""
import json, math, sys
import numpy as np
npz, meta_json = sys.argv[1], sys.argv[2]
z = np.load(npz); meta = json.load(open(meta_json))["weights"]
out = {}
for g in ("qkv", "o", "gate_up", "down"):
    cs, act, fl, ly = (z[f"{k}__{g}"] for k in ("log2cs", "log2act", "flat", "layer"))
    kcs = np.floor(cs) + 2          # smallest K with CS bound < 2^(K-1)
    kact = np.floor(act) + 2        # smallest K with the actual max |yq| < 2^(K-1)
    n = next(v["n"] for k, v in meta.items() if k.endswith("/" + g))
    wn = np.array([v["wn_max"] for k, v in meta.items() if k.endswith("/" + g)])
    static = math.floor(math.log2(2047 * 127 * n)) + 2
    per_layer_max = [int(kcs[ly == l].max()) for l in np.unique(ly)]
    out[g] = {"n": n, "K_static_A12": static,
              "K_cs_A12_pct": {p: float(np.percentile(kcs, p)) for p in (50, 90, 99, 99.9, 100)},
              "K_actual_A12_pct": {p: float(np.percentile(kact, p)) for p in (50, 99, 100)},
              "cs_slack_bits_median": round(float(np.median(cs - act)), 2),
              "log2_wn_over_127sqrtn": {"min": round(float(np.log2(wn / (127 * math.sqrt(n))).min()), 2),
                                        "max": round(float(np.log2(wn / (127 * math.sqrt(n))).max()), 2)},
              "log2_flat_over_sqrtn_pct": {p: round(float(np.percentile(np.log2(fl / math.sqrt(n)), p)), 2) for p in (50, 99, 100)},
              "per_layer_K_cs_max": per_layer_max}
    print(g, json.dumps(out[g]))
json.dump(out, open(sys.argv[3], "w"), indent=1)
