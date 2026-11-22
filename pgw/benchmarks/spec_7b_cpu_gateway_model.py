#!/usr/bin/env python3
"""Per-token model: Qwen2.5-7B target, CPU-only weight-light gateway, strict speculation with a 0.5B draft.

Pure arithmetic over measured inputs (research/plans/strict-speculation-cpu-gateway.md, section 5), so the
assumptions are explicit and can be re-run with other numbers:

* a decode pass = ``TRIPS`` sequential round trips (28 layers x 4 linears + the masked LM head = 113) of
  ``trip_ms`` each, plus the gateway's CPU for R rows per trip (native kernels + orchestration), plus the
  draft work on the critical path;
* sequential drafting (``BuiltinDrafter``): R - 1 draft steps between two passes;
* pipelined drafting (``PipelinedDrafter``, m foreground steps): m steps between two passes, plus any part of
  the background block (R steps) that does not fit under the pass;
* acceptance: per-token draft agreement ``alpha`` (i.i.d. model). ``alpha = 0.78`` reproduces the measured
  2.84 tokens per pass of the 1.5B target with the 0.5B draft at R = 4 (Kaggle run 28, the L3 plan); a 7B
  target is further from its 0.5B draft, so 0.70 is the central assumption here (no 7B acceptance was
  measured in this repository);
* tokens per pass, sequential: ``(1 - alpha^R) / (1 - alpha)``; pipelined: the stationary mean of a
  two-state chain (a pass carries R - 1 drafts after a hit of the background hypothesis, else m drafts;
  a pass with d drafts hits with probability ``alpha^(d + 1)``: all drafts and the bonus guess agree).

Usage: python benchmarks/spec_7b_cpu_gateway_model.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json

TRIPS = 113  # 28 x (qkv, o, gate_up, down) + the masked LM head (pgw.engine.masked_head)

# Draft (Qwen2.5-0.5B on NativeInt8DraftBackend, int8 head, native float ops), ms per 1-row draft step:
# measured 75 (1 core) and 46-54 (4 shared vCPUs, load 4-6) here (benchmarks/draft_cpu_bench.py); the step
# streams ~0.49 GB of int8 weights, so it is memory-bound: 16 server cores at ~50 GB/s give ~10 ms + ~2 ms
# of float ops and dispatch ("server16"); "server16_slow" = 20 ms (half that bandwidth, or a busy box).
DRAFT_MS = {"box4": 50.0, "server16": 12.0, "server16_slow": 20.0}


def cpu_pass_ms(gateway: str, rows: int) -> float:
    """Gateway CPU per pass of ``rows`` rows (ms).

    ``box``: this box, measured with ``native_trip_floor_bench.py --rows R`` (7B shapes, r keystream
    prefetched on another core: kernel floor 11.6 ms at R = 1, 32.3 at R = 4, i.e. +6.9 per extra row) plus
    7.2 ms of orchestration (the native per-trip path's ~0.2 ms per 1-row trip minus the 15.4 ms floor).
    ``server16``: r prefetched, a core 1.5x faster, the extra rows mostly overlapped on spare cores (30% of a
    row's cost each stays on the critical path), 4 ms of orchestration.
    """
    if gateway == "box":
        return 11.6 + 6.9 * (rows - 1) + 7.2
    return 4.0 + 11.6 / 1.5 * (1 + 0.3 * (rows - 1))


def tau_seq(alpha: float, rows: int) -> float:
    return (1 - alpha ** rows) / (1 - alpha)


def tau_pipe(alpha: float, rows: int, m: int) -> tuple[float, float]:
    """Stationary tokens per pass and share of full-draft passes of the pipelined drafter."""
    k = rows - 1
    e_full, h_full = tau_seq(alpha, k + 1), alpha ** (k + 1)
    e_miss, h_miss = tau_seq(alpha, m + 1), alpha ** (m + 1)
    if m >= k:
        return e_full, 1.0
    p_full = h_miss / ((1 - h_full) + h_miss)  # pi_F (1 - h_F) = pi_M h_M
    return p_full * e_full + (1 - p_full) * e_miss, p_full


def scenario(trip_ms: float, gateway: str, draft_ms: float, alpha: float, rows: int, mode: str,
             m: int = 0) -> dict[str, float]:
    if mode == "plain":
        pass_ms = TRIPS * trip_ms + cpu_pass_ms(gateway, 1)
        return {"pass_ms": pass_ms, "tau": 1.0, "tok_s": 1000.0 / pass_ms, "draft_crit_ms": 0.0}
    base = TRIPS * trip_ms + cpu_pass_ms(gateway, rows)
    if mode == "seq":
        crit = (rows - 1) * draft_ms
        tau = tau_seq(alpha, rows)
    else:
        bg = rows * draft_ms  # the background block (k + 1 steps) runs under the pass
        crit = m * draft_ms + max(0.0, bg - base)
        tau, _ = tau_pipe(alpha, rows, m)
    pass_ms = base + crit
    return {"pass_ms": pass_ms, "tau": tau, "tok_s": 1000.0 * tau / pass_ms, "draft_crit_ms": crit}


def smallest(trip: float, gw: str, dm: float, alpha: float, goal: float, mode: str) -> str:
    """The smallest R (then m) whose tokens/s reach ``goal``, as text ('-' if none up to R = 8)."""
    for r in range(2, 9):
        for m in ([0] if mode == "seq" else range(0, r)):
            s = scenario(trip, gw, dm, alpha, r, mode, m)
            if s["tok_s"] >= goal:
                return f"R={r}" + (f" m={m}" if mode == "pipe" else "") + f" {s['tok_s']:.1f}"
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = []
    print(f"{'trip':>5} {'gateway':>8} {'draft':>13} {'alpha':>5}  {'plain':>6}  {'seq (best R)':>15}  "
          f"{'seq R=4':>8}  {'pipe (best R, m)':>20}  {'pipe R=4 m=0':>12}")
    for trip in (0.3, 0.5):
        for gw, drafts in (("box", ("box4",)), ("server16", ("server16", "server16_slow"))):
            for dn in drafts:
                dm = DRAFT_MS[dn]
                for alpha in (0.70, 0.78):
                    plain = scenario(trip, gw, dm, alpha, 1, "plain")
                    seqs = {r: scenario(trip, gw, dm, alpha, r, "seq") for r in range(2, 9)}
                    pipes = {(r, m): scenario(trip, gw, dm, alpha, r, "pipe", m)
                             for r in range(2, 9) for m in range(0, r)}
                    bs = max(seqs, key=lambda r: seqs[r]["tok_s"])
                    bp = max(pipes, key=lambda x: pipes[x]["tok_s"])
                    rec = {"trip_ms": trip, "gateway": gw, "draft": dn, "draft_ms": dm, "alpha": alpha,
                           "plain_tok_s": round(plain["tok_s"], 1),
                           "seq_best": {"rows": bs, **{k: round(v, 2) for k, v in seqs[bs].items()}},
                           "seq_r4": {k: round(v, 2) for k, v in seqs[4].items()},
                           "pipe_best": {"rows": bp[0], "m": bp[1],
                                         **{k: round(v, 2) for k, v in pipes[bp].items()}},
                           "pipe_r4_m0": {k: round(v, 2) for k, v in pipes[(4, 0)].items()}}
                    out.append(rec)
                    print(f"{trip:>5} {gw:>8} {dn:>13} {alpha:>5}  {plain['tok_s']:>6.1f}  "
                          f"{seqs[bs]['tok_s']:>7.1f} (R={bs})  {seqs[4]['tok_s']:>8.1f}  "
                          f"{pipes[bp]['tok_s']:>7.1f} (R={bp[0]}, m={bp[1]})  {pipes[(4, 0)]['tok_s']:>12.1f}")
    print()
    print(f"smallest R (fewest rows and masks) reaching {TARGET} tok/s (and {MARGIN} with 10% margin):")
    print(f"{'trip':>5} {'gateway':>8} {'draft':>13} {'alpha':>5}  {'sequential':>22}  {'pipelined':>26}")
    for trip in (0.3, 0.4, 0.5):
        for gw, drafts in (("box", ("box4",)), ("server16", ("server16", "server16_slow"))):
            for dn in drafts:
                dm = DRAFT_MS[dn]
                for alpha in (0.70, 0.78):
                    a = (trip, gw, dm, alpha)
                    rec = {"trip_ms": trip, "gateway": gw, "draft": dn, "alpha": alpha,
                           "seq_25": smallest(*a, TARGET, "seq"), "seq_margin": smallest(*a, MARGIN, "seq"),
                           "pipe_25": smallest(*a, TARGET, "pipe"), "pipe_margin": smallest(*a, MARGIN, "pipe")}
                    out.append(rec)
                    print(f"{trip:>5} {gw:>8} {dn:>13} {alpha:>5}  {rec['seq_25']:>10} / {rec['seq_margin']:>10}"
                          f"  {rec['pipe_25']:>12} / {rec['pipe_margin']:>12}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)


TARGET = 25.0  # tokens/s streaming (~40 ms per generated token), the user's 7B goal
MARGIN = 27.5


if __name__ == "__main__":
    main()
