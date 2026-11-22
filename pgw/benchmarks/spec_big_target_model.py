#!/usr/bin/env python3
"""Tokens/s model of strict (tree) speculation for 32B / 72B targets on a CPU-only gateway.

pass time = trips x trip + gateway CPU(R) + draft critical path;  tokens/s = tau / pass time.

* trips per pass: 2L+1 (fused schedule, the floor) or 4L+1 (standard schedule: qkv, o, gate_up, down per
  layer + the masked LM head). L = 64 (Qwen2.5-32B), 80 (Qwen2.5-72B).
* gateway CPU per pass: scaled from the 7B measurement of ``strict-speculation-cpu-gateway.md`` section 5
  (main core, r keystream prefetched: 11.6 ms at R = 1, 32.3 ms at R = 4 for 113 trips, i.e. 4.7 + 6.9 R ms;
  a 16-core server core 1.5x faster) by the masked elements per row and layer (input + output widths of the
  four linears; 7B 79.4 k, 32B 115.7 k (64 layers), 72B 139.9 k (80 layers)), plus 0.035 ms of orchestration
  per trip. ``--row-overlap f`` puts a share f of the per-row work on spare cores (0: all on the main core).
* draft critical path: steps per pass x bytes per step / memory bandwidth (8 GB/s: the measured Modal
  gateway, 0.49 GB in ~60 ms per 0.5B step), as sequential drafting between passes (the conservative case;
  pipelining can hide it but costs acceptance).
* tau: measured on 0.5B->1.5B / 0.5B->3B and with the 1.5B feature head (``tree_spec_sim.py``), passed in.

Usage: ``python benchmarks/spec_big_target_model.py --taus taus.json`` (or edit the defaults below).
"""

from __future__ import annotations

import argparse
import json

ELEM_7B = 3584 * 3 + 18944 + (3584 + 2 * 512) + 3584 + 2 * 18944 + 3584
TARGETS = {
    "32B": {"layers": 64, "elem": 5120 * 3 + 27648 + (5120 + 2 * 1024) + 5120 + 2 * 27648 + 5120},
    "72B": {"layers": 80, "elem": 8192 * 3 + 29568 + (8192 + 2 * 1024) + 8192 + 2 * 29568 + 8192},
}


def gateway_cpu_ms(target: str, rows: int, overlap: float) -> float:
    t = TARGETS[target]
    scale = t["elem"] * t["layers"] / (ELEM_7B * 28)
    base, per_row = 4.7 * scale / 1.5, 6.9 * scale / 1.5
    return base + per_row * (1 + (rows - 1) * (1 - overlap))


def pass_ms(target: str, rows: int, trip_ms: float, fused: bool, overlap: float, draft_ms: float) -> float:
    L = TARGETS[target]["layers"]
    trips = (2 if fused else 4) * L + 1
    return trips * trip_ms + gateway_cpu_ms(target, rows, overlap) + trips * 0.035 + draft_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", help="json {config: {rows, tau, draft_steps, draft_mb_per_step}}")
    ap.add_argument("--bw", type=float, default=8.0, help="gateway memory bandwidth for the draft, GB/s")
    ap.add_argument("--json")
    args = ap.parse_args()
    cfgs = json.load(open(args.taus))
    out = []
    print(f"{'target':6s} {'trip':>5s} {'sched':6s} {'ovl':>4s} | " + " | ".join(f"{k}" for k in cfgs))
    for target in TARGETS:
        for trip in (0.3, 0.5, 1.0):
            for fused in (True, False):
                for overlap in (0.0, 0.7):
                    row = []
                    for name, c in cfgs.items():
                        c = {k: v for k, v in c.items() if k != "note"}
                        dms = c["draft_steps"] * c["draft_mb_per_step"] / args.bw
                        pm = pass_ms(target, c["rows"], trip, fused, overlap, dms)
                        tps = c["tau"] / pm * 1e3
                        row.append(tps)
                        out.append({"target": target, "trip_ms": trip, "fused": fused, "overlap": overlap,
                                    "config": name, "pass_ms": pm, "tok_s": tps, **c})
                    print(f"{target:6s} {trip:5.1f} {'2L+1' if fused else '4L+1':6s} {overlap:4.1f} | "
                          + " | ".join(f"{x:5.1f}" for x in row))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
