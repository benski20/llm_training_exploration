"""Train each variant on several seeds, then verify N test points exactly.

    python experiments/run_sweep.py                      # full CPU run (~30 min, resumable)
    python experiments/run_sweep.py --seeds 1 --n-verify 10 --time-limit 5   # smoke test
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.tasks import make_splits  # noqa: E402
from vt.train import TrainConfig, evaluate, train  # noqa: E402
from vt.verify import verify  # noqa: E402

VARIANTS = {
    "baseline": dict(),
    "rs": dict(rs_weight=0.2),
    "ternary": dict(ternary=True),
    "ternary_rs": dict(ternary=True, rs_weight=0.2),
    "ibp": dict(ibp_weight=0.5),
}


def _verify_one(args):
    P, x, y, eps, tl = args
    return verify(P, x, y, eps, time_limit=tl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="dyck")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=40000)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-verify", type=int, default=60)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/dyck16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    tr, te, V, C = make_splits(a.task, a.n_samples, a.seq_len)
    print(f"train {len(tr[0])}  test {len(te[0])}  (class-balanced)")
    Xv, Yv = te[0][:a.n_verify].numpy(), te[1][:a.n_verify].numpy()

    rows = []
    for name in a.variants.split(","):
        for seed in range(a.seeds):
            ckpt = f"{a.out}/{name}_s{seed}.json"
            if os.path.exists(ckpt):  # resume: this run already finished
                with open(ckpt) as f:
                    rows.append(json.load(f))
                print(f"{name:11s} s{seed}  (loaded {ckpt})", flush=True)
                continue
            cfg = replace(TrainConfig(name=name, eps=a.eps, epochs=a.epochs, seed=seed),
                          **VARIANTS[name])
            t0 = time.perf_counter()
            model = train(cfg, tr, V, C)
            train_s = time.perf_counter() - t0
            ev = evaluate(model, te, a.eps)
            P = model.export()
            torch.save(model.state_dict(), f"{a.out}/{name}_s{seed}.pt")
            with Pool(a.workers) as pool:
                res = pool.map(_verify_one, [(P, Xv[i], int(Yv[i]), a.eps, a.time_limit)
                                             for i in range(len(Xv))])
            st = [r["status"] for r in res]
            times = np.array([r["time"] for r in res])
            row = {
                "variant": name, "seed": seed, "train_s": train_s, **ev,
                "verified": st.count("verified") / len(st),
                "falsified": st.count("falsified") / len(st),
                "unknown": st.count("unknown") / len(st),
                "verify_time_mean": float(times.mean()),
                "verify_time_median": float(np.median(times)),
                "binaries_mean": float(np.mean([r["binaries"] for r in res])),
                "config": asdict(cfg),
                "per_sample": res,
            }
            rows.append(row)
            with open(ckpt, "w") as f:
                json.dump(row, f)
            print(f"{name:11s} s{seed}  acc {ev['clean_acc']:.3f}  ibp {ev['ibp_verified_acc']:.3f}  "
                  f"unstable {ev['unstable_mean']:6.1f}  V/F/U {row['verified']:.2f}/"
                  f"{row['falsified']:.2f}/{row['unknown']:.2f}  t_mean {row['verify_time_mean']:.2f}s",
                  flush=True)

    with open(f"{a.out}/results.json", "w") as f:
        json.dump({"args": vars(a), "rows": rows}, f, indent=1)
    write_summary(rows, a, f"{a.out}/summary.md")


def write_summary(rows, a, path):
    keys = [("clean_acc", "clean acc", "{:.3f}"), ("ibp_verified_acc", "IBP-verified", "{:.3f}"),
            ("unstable_mean", "unstable ReLUs", "{:.1f}"), ("weight_zero_frac", "zero weights", "{:.2f}"),
            ("verified", "MILP verified", "{:.2f}"), ("falsified", "falsified", "{:.2f}"),
            ("unknown", "timeout", "{:.2f}"), ("verify_time_mean", "mean time (s)", "{:.2f}"),
            ("verify_time_median", "median time (s)", "{:.2f}")]
    lines = [f"# {a.task} L={a.seq_len}, eps={a.eps}, {a.seeds} seeds, "
             f"{a.n_verify} test points, {a.time_limit:.0f}s MILP limit\n",
             "Mean ± std over seeds. Verified/falsified/timeout are fractions of the "
             "verified test points; every point gets one of the three.\n",
             "| variant | " + " | ".join(k[1] for k in keys) + " |",
             "|---" * (len(keys) + 1) + "|"]
    for name in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == name]
        cells = []
        for k, _, fmt in keys:
            v = np.array([r[k] for r in rs])
            cells.append(fmt.format(v.mean()) + (f" ± {fmt.format(v.std())}" if len(v) > 1 else ""))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
