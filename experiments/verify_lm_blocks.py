"""Exact local verification of a trained GPT's MLP blocks.

For real validation tokens, take each layer's MLP input x0 (after RMSNorm)
and solve exactly, by MILP, the worst case of a scalar readout c . MLP(x)
over the ball ||x - x0||_inf <= eps. Pre-activation bounds are exact for a
single layer (z0 +- eps ||w||_1), so binaries = unstable neurons. Reports
solve time and binaries per (token, layer), comparing training variants.

    python experiments/verify_lm_blocks.py runs/lm_base runs/lm_margin --data data/tinyshakespeare.txt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.lm import GPT, GPTConfig  # noqa: E402
from vt.verify import _MILP  # noqa: E402


def mlp_worst_case(W1, b1, W2, x0, eps, c, time_limit):
    z0 = W1 @ x0 + b1
    rad = eps * np.abs(W1).sum(1)
    l, u = z0 - rad, z0 + rad
    m = _MILP()
    x = m.vars(x0 - eps, x0 + eps)
    z = m.vars(l, u)
    m.eq(z, [(sp.csr_matrix(W1), x)], b1)
    a = m.vars(np.zeros_like(l), np.maximum(u, 0))
    n_bin = 0
    for i in range(len(l)):
        if u[i] <= 0:
            m.ub[a[i]] = 0.0
        elif l[i] >= 0:
            m.row([(a[i], 1.0), (z[i], -1.0)], 0.0, 0.0)
        else:
            d = m.vars([0.0], [1.0], integer=True)[0]
            n_bin += 1
            m.row([(a[i], 1.0), (z[i], -1.0)], 0.0, np.inf)
            m.row([(a[i], 1.0), (d, -u[i])], -np.inf, 0.0)
            m.row([(a[i], 1.0), (z[i], -1.0), (d, -l[i])], -np.inf, -l[i])
    t0 = time.perf_counter()
    res = m.solve(a, -(c @ W2), time_limit)   # maximise c . W2 a
    return {"time": time.perf_counter() - t0, "binaries": n_bin,
            "status": int(res.status), "max": None if res.fun is None else -float(res.fun)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", default="results/lm_blocks.json")
    a = ap.parse_args()
    raw = np.memmap(a.data, dtype=np.uint8, mode="r")
    val = raw[int(0.9 * len(raw)):]
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(val) - 129, size=a.tokens)
    report = {}
    for run in a.runs:
        ck = torch.load(os.path.join(run, "ckpt.pt"), map_location="cpu")
        model = GPT(GPTConfig(**ck["config"]))
        model.load_state_dict(ck["model"])
        model.eval()
        captured = []
        hooks = [b.ln2.register_forward_hook(lambda mod, inp, out: captured.append(out[0, -1].double().numpy()))
                 for b in model.blocks]
        rows = []
        for s in starts:
            captured.clear()
            with torch.no_grad():
                model(torch.from_numpy(val[s:s + 128].astype(np.int64))[None])
            for li, (blk, x0) in enumerate(zip(model.blocks, captured)):
                W1 = blk.mlp.fc.qweight().detach().double().numpy()
                b1 = blk.mlp.fc.bias.detach().double().numpy()
                W2 = blk.mlp.out.qweight().detach().double().numpy()
                c = rng.standard_normal(W2.shape[0]); c /= np.linalg.norm(c)
                r = mlp_worst_case(W1, b1, W2, x0, a.eps, c, a.time_limit)
                r["layer"] = li
                rows.append(r)
        for h in hooks:
            h.remove()
        t = np.array([r["time"] for r in rows]); nb = np.array([r["binaries"] for r in rows])
        solved = np.mean([r["status"] == 0 for r in rows])
        report[run] = {"binaries_mean": float(nb.mean()), "time_mean": float(t.mean()),
                       "time_median": float(np.median(t)), "solved_frac": float(solved),
                       "n": len(rows), "rows": rows}
        print(f"{run:28s} binaries {nb.mean():6.1f}  time mean {t.mean():6.2f}s "
              f"median {np.median(t):6.2f}s  solved {solved:.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"args": vars(a), "report": report}, f, indent=1)


if __name__ == "__main__":
    main()
