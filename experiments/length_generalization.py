"""Prove one weight-tied model correct at every length up to the one it was trained at.

A weight-tied MixerNet's parameters (Toeplitz mixing kernel m, pooled and
last-position readout heads, per-position MLP) do not depend on L. Truncating
the kernel gives the same model at any shorter length, so we can certify it
at every L' <= L, not just the training length.

    python experiments/length_generalization.py results/cegis/L32_shared/model.npz --lengths 2:32:2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.artifact import load_artifact  # noqa: E402
from vt.global_verify import SPECS, verify_global  # noqa: E402


def truncate(P, L2):
    """The same weight-tied model at length L2 <= L."""
    L, d, C = P["L"], P["d"], P["C"]
    W = P["W_out"].reshape(C, L, d)
    pool = W[:, 0]
    last = W[:, -1] - pool
    if not (np.allclose(W[:, :-1], pool[:, None]) and
            all(np.allclose(np.diag(M, -k), np.diag(M, -k)[0]) for M in P["mix"] for k in range(L))):
        raise ValueError("artifact is not weight-tied")
    Q = dict(P, L=L2, mix=[M[:L2, :L2] for M in P["mix"]])
    Wn = np.tile(pool, (1, L2)).reshape(C, L2, d)
    Wn[:, -1] += last
    Q["W_out"] = Wn.reshape(C, L2 * d)
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--spec", default="dyck")
    ap.add_argument("--lengths", default="2:32:2", help="start:stop:step (stop inclusive)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = load_artifact(a.artifact)
    lo, hi, step = (int(v) for v in a.lengths.split(":"))
    spec = SPECS[a.spec]()
    rows = []
    for L2 in range(lo, hi + 1, step):
        t0 = time.perf_counter()
        r = verify_global(truncate(P, L2), spec, time_limit=a.time_limit, workers=a.workers)
        rows.append({"L": L2, "status": r["status"], "time": time.perf_counter() - t0,
                     "counterexamples": r["counterexamples"][:3]})
        print(f"L={L2:3d}  {r['status']:15s} {rows[-1]['time']:7.1f}s", flush=True)
    out = a.out or os.path.join(os.path.dirname(a.artifact), "length_generalization.json")
    with open(out, "w") as f:
        json.dump({"artifact": a.artifact, "spec": a.spec, "rows": rows}, f, indent=1)
    proved = [r["L"] for r in rows if r["status"] == "proved"]
    print(f"proved at {len(proved)}/{len(rows)} lengths")


if __name__ == "__main__":
    main()
