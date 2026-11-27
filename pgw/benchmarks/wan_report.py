#!/usr/bin/env python3
"""Markdown tables from ``wan_bench.py`` JSON results: ``python benchmarks/wan_report.py result.json [...]``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _tcp(res: dict[str, Any]) -> str:
    srv_ssai = res["ssai"] if res.get("server_ssai") is None else res["server_ssai"]
    return (f"cc {res['cc']}, slow start after idle {res['ssai']} (provider: {res.get('server_cc') or res['cc']}, "
            f"{srv_ssai}), rate {res.get('rate_mbit') or 'none'} Mbit/s down / "
            f"{res.get('up_mbit') or res.get('rate_mbit') or 'none'} up")


def lane(res: dict[str, Any]) -> None:
    print(f"\nlane bench: {_tcp(res)}")
    runs = res["runs"]
    print("\n| RTT ms | loss | TCP connect ms | lane connect (TCP+TLS+open) ms |")
    print("|---|---|---|---|")
    for r in runs:
        print(f"| {r['rtt_ms']:g} | {r['loss']:g} | {r['tcp_connect_ms']:.2f} | {r['lane_connect_ms']:.2f} |")
    print("\nPer trip, 1 row unless noted (p50 ms; excess = p50 - RTT; lane - raw = what the lane stack adds over bare "
          "TCP of the same bytes):")
    print("| RTT | loss | model | linear | rows | up KB | down KB | lane p50 | lane p99 | raw p50 | lane excess | "
          "raw excess | lane - raw |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        for p in r["payloads"]:
            print(f"| {r['rtt_ms']:g} | {r['loss']:g} | {p['preset']} | {p['linear']} | {p['rows']} | "
                  f"{p['up_bytes'] / 1024:.0f} | {p['down_bytes'] / 1024:.0f} | {p['lane_p50_ms']:.3f} | "
                  f"{p['lane_p99_ms']:.3f} | {p['raw_p50_ms']:.3f} | {p['lane_excess_ms']:.3f} | "
                  f"{p['raw_excess_ms']:.3f} | {p['lane_over_raw_ms']:.3f} |")
    print("\nOne decode pass (every trip of a token in order, 1 row, no compute):")
    print("| RTT | loss | model | trips | pass ms | ms per trip | excess per trip | MB per token |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        for p in r["pass"]:
            print(f"| {r['rtt_ms']:g} | {r['loss']:g} | {p['preset']} | {p['trips']} | {p['pass_p50_ms']:.1f} | "
                  f"{p['ms_per_trip']:.3f} | {p['excess_ms_per_trip']:.3f} | {p['bytes_per_token'] / 1e6:.2f} |")
    if any(r.get("idle") for r in runs):
        print("\nFirst 7B gate_up trip (14 KB up, 148 KB down) after an idle gap on a warm connection:")
        print("| RTT | gap ms | lane first trip ms | raw first trip ms | client cwnd before |")
        print("|---|---|---|---|---|")
        for r in runs:
            for p in r.get("idle", []):
                print(f"| {r['rtt_ms']:g} | {p['gap_ms']:g} | {p['lane_first_trip_p50_ms']:.2f} | "
                      f"{p['raw_first_trip_p50_ms']:.2f} | {p['client_cwnd_before']} |")


def sdk(res: dict[str, Any]) -> None:
    print(f"\nsdk bench: {_tcp(res)}; {res['tokens']} tokens, prompt {res['prompt']}, opts {res['opt']}")
    print("excess vs RTT 0 = ms/trip - ms/trip at RTT 0 - RTT; lane beyond compute = lane trip - provider compute "
          "- RTT (the network's share of a trip beyond one RTT, independent of the CPU provider's load)")
    print("| RTT ms | tok/s | TTFT ms | trips/token | ms per trip | provider compute ms | excess vs RTT 0 | "
          "lane beyond compute | lane floor ms | cold setup s | bit-identical |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["runs"]:
        lane_net = r["lane_trip_us"] / 1e3 - r["provider_compute_us"] / 1e3 - r["rtt_ms"]
        print(f"| {r['rtt_ms']:g} | {r['decode_tok_s']:.2f} | {r['ttft_ms']:.0f} | {r['trips_per_token']:.1f} | "
              f"{r['ms_per_trip_wall']:.3f} | {r['provider_compute_us'] / 1e3:.3f} | "
              f"{r.get('excess_ms_per_trip', float('nan')):.3f} | {lane_net:.3f} | {(r['lane_floor_ms'] or 0):.3f} | "
              f"{r['cold_setup_s']} | {r['bit_identical']} |")


def quic(res: dict[str, Any]) -> None:
    print(f"\nquic bench ({res['preset']} trip sizes; TCP: kernel {res['cc']} both ends; QUIC: quinn's controller "
          "on both ends): one decode pass, ms per trip (excess over RTT)")
    variants = list(dict.fromkeys(r["variant"] for r in res["runs"]))
    print("| RTT ms | loss | " + " | ".join(variants) + " |")
    print("|---|---|" + "---|" * len(variants))
    keys = list(dict.fromkeys((r["rtt_ms"], r["loss"]) for r in res["runs"]))
    for rtt, loss in keys:
        by = {r["variant"]: r for r in res["runs"] if r["rtt_ms"] == rtt and r["loss"] == loss}
        cells = [f"{by[v]['ms_per_trip']:.2f} (+{by[v]['excess_ms_per_trip']:.2f})" if v in by else "-"
                 for v in variants]
        print(f"| {rtt:g} | {loss:g} | " + " | ".join(cells) + " |")


def main() -> None:
    for f in sys.argv[1:]:
        res = json.loads(Path(f).read_text())
        {"lane": lane, "sdk": sdk, "quic": quic}[res["bench"]](res)


if __name__ == "__main__":
    main()
