#!/usr/bin/env python3
"""tau and projected 7B tokens/s of strict chain speculation with a trained feature head.

Replays the held-out exact answers through the generator's acceptance rule (``tree_spec_sim.replay``, chain of
R - 1 drafts, masked filler rows, i.e. every pass sends R rows) using the head's per-depth ranks
(``ranks_*.json`` from ``feature_head_train.py``), then projects tokens/s on the CPU-only 7B gateway:

    pass = trips x (wait + gateway(R)) + (R - 1) x head step;   tokens/s = tau / pass

with the measured per-trip costs (``--trips 113``, ``--wait-ms 0.38``, ``--gw-ms 0.3`` at R = 1) and the
gateway's per-trip work scaled by ``--gw-scale`` for R = 2, 3, 4 (default 1.3 / 1.45 / 1.6).

    python benchmarks/feature_head_projection.py --ranks ranks_d384_v8192.json --step-us 1000 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tree_spec_sim import best_tree, chain, depth_rank_probs, replay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", nargs="+", required=True)
    ap.add_argument("--step-us", type=float, nargs="+", default=[1000.0],
                    help="head step time (us), one per ranks file or one for all")
    ap.add_argument("--trips", type=int, default=113)
    ap.add_argument("--wait-ms", type=float, default=0.38)
    ap.add_argument("--gw-ms", type=float, default=0.3)
    ap.add_argument("--gw-scale", default="1.3,1.45,1.6", help="gateway work factor at R = 2,3,4")
    ap.add_argument("--rs", default="2,3,4")
    ap.add_argument("--json")
    args = ap.parse_args()
    scale = dict(zip((2, 3, 4), (float(x) for x in args.gw_scale.split(",")), strict=True))
    scale[1] = 1.0
    plain_ms = args.trips * (args.wait_ms + args.gw_ms)
    out: dict = {"plain": {"pass_ms": plain_ms, "tok_s": 1e3 / plain_ms}}
    print(f"plain: pass {plain_ms:.1f} ms -> {1e3 / plain_ms:.1f} tok/s")
    for fi, f in enumerate(args.ranks):
        d = json.load(open(f))
        rk = d["ranks2d"]
        step_ms = (args.step_us[fi] if fi < len(args.step_us) else args.step_us[-1]) / 1e3
        p = depth_rank_probs(rk)
        res: dict = {"alpha_by_depth": [round(x[0], 4) for x in p[:6]], "step_ms": step_ms}
        print(f"{Path(f).stem}: P(rank 0 | path accepted) by depth {res['alpha_by_depth']}")
        for r in [int(x) for x in args.rs.split(",")]:
            for shape, nodes in (("chain", chain(r)), ("tree", best_tree(r, p))):
                tau = replay(rk, nodes, None, None, d["prompt_lens"])["tau"]
                pass_ms = args.trips * (args.wait_ms + args.gw_ms * scale.get(r, scale[4])) + (r - 1) * step_ms
                tps = tau / pass_ms * 1e3
                res[f"R={r} {shape}"] = {"tau": tau, "pass_ms": pass_ms, "tok_s": tps, "nodes": nodes}
                print(f"  R={r} {shape:5s} tau {tau:.3f}  pass {pass_ms:6.1f} ms  -> {tps:5.1f} tok/s"
                      + (f"  {nodes}" if shape == "tree" else ""))
        out[Path(f).stem] = res
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
