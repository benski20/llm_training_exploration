#!/usr/bin/env python3
"""Per-token latency model of the chat-speed plan (central, optimistic, pessimistic; all ms).

    token = trips x (remote per trip) + masking + float + head + other     (one pass, one row)
    spec  = (P1 + k x c + D) / tau                                        (exact chain speculation)

``P1`` is one pass, ``k`` drafts per pass, ``c`` the cost of one extra verify row
(``spec_cost_bench.py``), ``D`` the drafting time and ``tau`` the tokens per pass. The scenario inputs
are the plan's estimates (section 2). ``--validate RESULT.json`` replaces scenario (a)'s "today" terms
with a ``gpu_validate.py`` streaming result that has ``decode_unaccounted_split`` (WP-A0), so the model
can be re-based on a new Kaggle run: remote and masking from the trip counters, float and head from
the probe, and ``other`` = consume + step glue + loop.

Usage::

    python benchmarks/chat_speed_model.py [--validate benchmarks/results/kaggle/<run>/qwen7b.json] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

Triple = list[float]


def tok(trips: float, remote: float, mask: float, flt: float, head: float, other: float) -> float:
    return trips * remote + mask + flt + head + other


def spec(p1: float, k: float, c: float, d: float, tau: float) -> float:
    return (p1 + k * c + d) / tau


def trip_b(dev: float, rtt: float) -> float:
    """Remote ms per trip for (b): device + launch + RTT + framing + client."""
    return dev + 0.04 + rtt + 0.026 + 0.012


def today_a_from_validate(res: dict[str, Any]) -> dict[str, float] | None:
    """Scenario (a) 'today' terms from a gpu_validate result (None if it lacks the attribution)."""
    st = res.get("stream") or {}
    split = st.get("decode_unaccounted_split")
    if not split or "decode_round_trips_per_token" not in st:
        return None
    return {"trips": st["decode_round_trips_per_token"], "remote": st["decode_remote_ms_per_trip"],
            "mask": st["decode_local_s_per_token"] * 1e3, "float": split["float_ops_ms"],
            "head": split["head_ms"],
            "other": split["consume_ms"] + split["draft_ms"] + split["step_glue_ms"] + split["loop_ms"],
            "measured_s_per_token": st.get("decode_s_per_token") or 0.0}


def scenarios(today_a: dict[str, float] | None = None) -> dict[str, Triple]:
    s: dict[str, Triple] = {}
    # ---- (a) 7B, one T4, co-located (Kaggle: 2 physical cores, AVX2) ----
    if today_a is None:
        s["a today"] = [tok(112, 1.457, 88, 29, 44, 24)] * 3  # run 27: 349 ms measured
    else:
        t = today_a
        s["a today"] = [tok(t["trips"], t["remote"], t["mask"], t["float"], t["head"], t["other"])] * 3
    s["a A"] = [tok(112, 0.33, 12, 15, 44, 10), tok(112, 0.29, 7, 10, 40, 5), tok(112, 0.42, 20, 29, 44, 24)]
    s["a A+spec"] = [spec(s["a A"][0], 3, 38, 11, 2.7), spec(s["a A"][1], 3, 30, 9, 2.95),
                     spec(s["a A"][2], 2, 55, 10, 2.1)]
    s["a B"] = [tok(112, 0.33, 9, 4, 4.3, 5), tok(112, 0.29, 6, 3, 4, 3), tok(112, 0.42, 14, 8, 6, 12)]
    s["a B+spec"] = [spec(s["a B"][0], 3, 10, 11, 2.7), spec(s["a B"][1], 4, 8, 12, 3.3),
                     spec(s["a B"][2], 3, 16, 14, 2.4)]
    # ---- (b) 7B, H100-class provider, separate 16-vCPU AVX-512 gateway, same AZ (RTT 0.15; B+ 0.05) ----
    s["b A"] = [tok(112, trip_b(0.05, 0.15), 5, 7, 15, 5), tok(112, trip_b(0.03, 0.05), 4, 5, 12, 3),
                tok(112, trip_b(0.08, 0.25), 8, 10, 22, 10)]
    s["b A+spec"] = [spec(s["b A"][0], 4, 9, 9, 3.0), spec(s["b A"][1], 4, 7, 8, 3.3), spec(s["b A"][2], 3, 13, 8, 2.4)]
    s["b B"] = [tok(112, trip_b(0.05, 0.15), 4, 3, 2, 4), tok(112, trip_b(0.03, 0.05), 3, 2, 1.5, 3),
                tok(112, trip_b(0.08, 0.25), 6, 5, 3, 8)]
    s["b B+spec"] = [spec(s["b B"][0], 4, 6, 9, 3.0), spec(s["b B"][1], 4, 5, 8, 3.3), spec(s["b B"][2], 3, 9, 8, 2.4)]
    # ---- (c) Modal 0.5B, two parties ----
    s["c today"] = [tok(96, 6.8, 10, 5, 10, 5)] * 3
    # phase A on the public tunnel: exact local qkv/o (48 trips), lane, cpu_int8 MLP on the provider
    s["c A tunnel"] = [tok(48, 4.8 + 0.1 + 0.92, 6, 5, 10 + 4.4, 5), tok(48, 4.0 + 0.1 + 0.92, 4, 4, 9 + 3, 3),
                       tok(48, 5.5 + 0.2 + 1.3, 8, 8, 10 + 6, 8)]
    # phase A, i6pn + T4 provider (RTT unknown: 0.3 central, 0.1-1.0)
    s["c A i6pn"] = [tok(48, 0.3 + 0.05 + 0.03 + 0.02, 6, 5, 10 + 4.4, 5),
                     tok(48, 0.1 + 0.05 + 0.03 + 0.02, 4, 4, 9 + 3, 3),
                     tok(48, 1.0 + 0.1 + 0.05 + 0.02, 8, 8, 10 + 6, 8)]
    s["c A i6pn+spec"] = [spec(s["c A i6pn"][0], 3, 4, 12, 2.5), spec(s["c A i6pn"][1], 3, 3, 9, 2.8),
                          spec(s["c A i6pn"][2], 2, 6, 10, 2.0)]
    s["c B i6pn"] = [tok(48, 0.4, 5, 2, 2 + 4.4, 4), tok(48, 0.2, 3, 2, 1.5 + 3, 3), tok(48, 1.17, 7, 4, 3 + 6, 7)]
    s["c B i6pn+spec"] = [spec(s["c B i6pn"][0], 3, 3, 12, 2.5), spec(s["c B i6pn"][1], 3, 2, 9, 2.8),
                          spec(s["c B i6pn"][2], 2, 5, 10, 2.0)]
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--validate", default=None, help="gpu_validate result JSON for scenario (a) today")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    today = None
    if args.validate:
        today = today_a_from_validate(json.loads(Path(args.validate).read_text()))
        if today is None:
            print(f"{args.validate}: no decode_unaccounted_split; using the run-27 terms")
    out: dict[str, Any] = {}
    for k, (c, o, p) in scenarios(today).items():
        out[k] = {"central_ms": round(c, 1), "optimistic_ms": round(o, 1), "pessimistic_ms": round(p, 1)}
        print(f"{k:16s} central {c:6.0f} ms ({1000 / c:5.1f} tok/s)   range {o:5.0f}-{p:5.0f} ms "
              f"({1000 / p:4.1f}-{1000 / o:5.1f} tok/s)")
    if today is not None:
        out["a today inputs"] = today
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
