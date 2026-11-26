"""Per-token decode latency model for research/plans/fewer-round-trips.md (R1). No I/O, no torch.

s/token = remote_trips * (RTT + o) + layers * local_linear_ms(placement) + F, divided by tau (tokens
per pass) for speculation. Inputs are measured values (see the memo for sources); `gw` scales local
linear cost: 1.0 = measured here (1 thread, shared 4-vCPU box, exact 2-digit int8 GEMV, ~3 GB/s of
weights), 0.53 = the same thread after the R1 critique's digit stacking in `exact_int_matmul` (both
activation digits in one int8 GEMM, one pass over the weight; measured 1.87-1.90x on the 0.5B/7B
shapes), 0.25 = a dedicated 4-8 core gateway on top of that (estimate, not measured). `--modal-o` uses
the per-trip overhead the Modal e2e implies (26 ms median trip at 17 ms RTT, so o ~ 9 ms) for 0.5B.

    python benchmarks/round_trip_model.py
    python benchmarks/round_trip_model.py --sdk     # the client library's path, 7B/32B/72B, same-data-center RTTs
"""

from __future__ import annotations

from typing import Any

# per layer, one row, ms, measured here (benchmarks: exact_int_matmul, torch 1 thread, median of 5)
LOCAL_MS = {
    "0.5B": {"qkv": 0.25, "o": 0.22, "gate_up": 2.52, "down": 1.24},
    "7B": {"qkv": 4.19, "o": 3.23, "gate_up": 51.5, "down": 23.5},
}
LAYERS = {"0.5B": 24, "7B": 28}
# per-trip cost beyond the network RTT: service + framing + gateway mask/unmask (Kaggle run 27, 7B:
# 1.47 ms remote at loopback + 0.71 ms masking per trip); 0.5B scaled down (estimate)
TRIP_O_MS = {"0.5B": 1.2, "7B": 2.2}
# fixed gateway work per token without linears: LM head, norms, attention, sampling (7B: run 27 "other"
# 94 ms; 0.5B: LM head ~12 ms + ~5 ms float ops, estimate)
FIXED_MS = {"0.5B": 17.0, "7B": 94.0}

PLACEMENTS = {
    "all remote (4/layer)": (),
    "attention local (2/layer, today's auto)": ("qkv", "o"),
    "attention_down (1/layer)": ("qkv", "o", "down"),
    "all local (0/layer)": ("qkv", "o", "gate_up", "down"),
}


def s_per_token(model: str, local: tuple[str, ...], rtt_ms: float, gw: float = 1.0, tau: float = 1.0) -> float:
    L = LAYERS[model]
    remote = 4 - len(local)
    local_ms = sum(LOCAL_MS[model][k] for k in local) * gw
    ms = L * (remote * (rtt_ms + TRIP_O_MS[model]) + local_ms) + FIXED_MS[model]
    return ms / 1e3 / tau


# ---- the client library's path (pgw.OpenAI): weight-free gateway, every linear remote, masked LM head ----------
# Per decode pass: trips x (RTT + lane + gateway work per trip) + device + wire + head + serving, divided by the
# tokens per pass (strict speculation). Components come from the co-located model of scaling_model.py, which
# reproduces the measured 7B run (35.7 ms per token at 27.97 tok/s); what is new here is marked.
SDK_SHAPES = {"7B": "qwen2.5-7b", "32B": "qwen2.5-32b", "72B": "qwen2.5-72b"}
#: gateway work per layer trip, 1 row, ms on the Modal gateway: 7B measured (10.8 ms / 113 trips); 32B the dev-box
#: 32B driver with lane-wait prefetch x the 7B calibration (STATUS 2026-11-23); 72B = 32B x its element ratio
#: (sum of n + m per layer, 139,904 / 115,712; estimate).
SDK_GW_TRIP_MS = {"7B": 10.8 / 113, "32B": 0.105, "72B": 0.105 * 139904 / 115712}
#: gateway work of the masked head trip (152k outputs): dev box 300 us x calibration 0.87 (estimate for 7B/72B).
SDK_GW_HEAD_MS = 0.26
#: measured here (sdk_trip_profile.py, Qwen2.5-0.5B, 4-vCPU box): gateway work per trip at 3 rows / 1 row = 2.65
#: (207-243 vs 70-98 us), i.e. 17.5% per-trip fixed and 82.5% per row.
SDK_ROW_FIXED = 0.175
#: lane transport + provider service per trip beyond RTT and device time: 0.09 ms measured over the shared-memory
#: lane in the 7B run; + 0.04 ms for TCP+TLS through a NIC (estimate: loopback TLS lane 55 us p50 vs ~13 us UDS/shm
#: on the dev box, lane_tls_bench.py).
SDK_LANE_MS = 0.09 + 0.04
#: serving outside the passes in the library (sampling over 152k logits with logprobs, the token hand-off to the
#: caller's thread): 1.2-1.4 ms measured at 0.5B (same vocabulary) on the dev box, rounded up (no HTTP / SSE; the
#: gateway's HTTP path measured 9.2 ms in the 7B run).
SDK_SERVE_MS = 2.0
#: tokens per verify pass (strict, chain): measured here on Qwen2.5-0.5B with prompt lookup (sdk_trip_profile.py,
#: 64 tokens): open chat 1.14 (R=2) / 1.15 (R=3), a code edit 1.66 (R=2) / 2.10-2.14 (R=3) / 2.52 (R=4); draft head:
#: alpha 0.46 is the 7B float head's measured top-1 (export bug pending), 0.8 an EAGLE-class head (not trained)
#: -> (1 - a^3) / (1 - a).
SDK_TAU = {"off (R=1)": (1, 1.0), "lookup, open chat (R=2)": (2, 1.14), "lookup, open chat (R=3)": (3, 1.15),
           "lookup, copy-heavy (R=2)": (2, 1.66), "lookup, copy-heavy (R=3)": (3, 2.1),
           "draft head a=0.46 (R=3)": (3, (1 - 0.46 ** 3) / (1 - 0.46)),
           "draft head a=0.8 (R=3)": (3, (1 - 0.8 ** 3) / (1 - 0.8))}


def _sm() -> Any:
    """The sibling module scaling_model.py (this directory on sys.path)."""
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import scaling_model

    return scaling_model


def sdk_ms_per_token(model: str, rtt_ms: float, rows: int = 1, tau: float = 1.0, nic_gbit: float = 25.0,
                     tp: int | None = None, lane_ms: float | None = None, gw_scale: float = 1.0) -> dict[str, float]:
    """ms per decoded token by component on the library's path (see the constants above)."""
    sm = _sm()

    shape = sm.SHAPES[SDK_SHAPES[model]]
    if tp is None:
        tp = sm.tp_degree(shape, sm.Scenario())
    trips = 4 * shape.layers + 1
    layer_mb = sum(n * m for _, n, m in shape.linears()) / sm.MB
    head_mb = shape.hidden * shape.vocab / sm.MB
    device = (shape.layers * layer_mb + head_mb) * sm.DEVICE_MS_PER_MB / tp + (trips * sm.TP2_GATHER_MS if tp > 1
                                                                              else 0.0)
    rowf = SDK_ROW_FIXED + (1 - SDK_ROW_FIXED) * rows
    lane = SDK_LANE_MS if lane_ms is None else lane_ms
    out = {"trips": float(trips), "network": trips * rtt_ms, "lane": trips * lane, "device": device,
           "wire": rows * (shape.sum_n() + shape.sum_m()) * 4 * 8 / (nic_gbit * 1e9) * 1e3,
           "gateway": (4 * shape.layers * SDK_GW_TRIP_MS[model] + SDK_GW_HEAD_MS) * rowf * gw_scale,
           "serve": SDK_SERVE_MS * (1 + 0.1 * (rows - 1))}
    per_pass = sum(v for k, v in out.items() if k != "trips")
    out["pass_ms"] = per_pass
    out["ms_per_token"] = per_pass / tau
    out["critical_trips_per_token"] = trips / tau
    return out


def sdk_main(nic_gbit: float = 25.0) -> None:
    rtts = (0.1, 0.2, 0.5)
    print(f"\nClient library path, ms per token (tok/s); NIC {nic_gbit:g} Gbit/s, H100 provider (72B: 2-way TP)")
    print("| model | mode | trips on the critical path per token | " +
          " | ".join(f"RTT {r} ms" for r in rtts) + " |")
    print("|---|---|---|" + "---|" * len(rtts))
    for model in SDK_SHAPES:
        for mode, (rows, tau) in SDK_TAU.items():
            cells = []
            for rtt in rtts:
                b = sdk_ms_per_token(model, rtt, rows, tau, nic_gbit)
                cells.append(f"{b['ms_per_token']:.1f} ({1e3 / b['ms_per_token']:.1f})")
            b = sdk_ms_per_token(model, rtts[0], rows, tau, nic_gbit)
            print(f"| {model} | {mode} | {b['critical_trips_per_token']:.0f} | " + " | ".join(cells) + " |")
    print("\nComponents per pass at RTT 0.2 ms, 1 row (ms): ")
    for model in SDK_SHAPES:
        b = sdk_ms_per_token(model, 0.2, 1, 1.0, nic_gbit)
        print(f"  {model}: " + ", ".join(f"{k} {v:.1f}" for k, v in b.items() if k not in ("ms_per_token",)))
    print("\nSensitivity, RTT 0.2 ms: per-trip overhead o = lane + gateway (ms per token, tok/s)")
    for model in SDK_SHAPES:
        cells = []
        for lane, gws, name in ((None, 1.0, "today"), (0.05, 1.0, "lane 0.05"), (0.05, 0.5, "lane 0.05 + gateway/2"),
                                (0.05, 0.5, "same + draft head a=0.8 R=3")):
            rows, tau = (3, (1 - 0.8 ** 3) / 0.2) if "draft" in name else (1, 1.0)
            b = sdk_ms_per_token(model, 0.2, rows, tau, nic_gbit, lane_ms=lane, gw_scale=gws)
            cells.append(f"{name}: {b['ms_per_token']:.0f} ({1e3 / b['ms_per_token']:.1f})")
        print(f"  {model}: " + "; ".join(cells))
    print("\nLocal linear on the client (factory-pack weights) vs one remote trip at RTT 0.2 ms, ms per layer, 1 row")
    print("(measured 1-thread exact GEMV with stacked digits at 0.5B/7B; 32B/72B scaled by weight count):")
    print("| model | qkv | o | gate_up | down | one remote trip |")
    print("|---|---|---|---|---|---|")
    sm = _sm()
    ref = sm.SHAPES["qwen2.5-7b"]
    ref_w = {k: n * m for k, n, m in ref.linears()}
    for model in ("0.5B", "7B", "32B", "72B"):
        if model in LOCAL_MS:
            loc = {k: v * 0.53 for k, v in LOCAL_MS[model].items()}
        else:
            w = {k: n * m for k, n, m in sm.SHAPES[SDK_SHAPES[model]].linears()}
            loc = {k: LOCAL_MS["7B"][k] * 0.53 * w[k] / ref_w[k] for k in w}
        trip = 0.2 + SDK_LANE_MS + SDK_GW_TRIP_MS.get(model, 0.07)
        print(f"| {model} | " + " | ".join(f"{loc[k]:.2f}" for k in ("qkv", "o", "gate_up", "down")) +
              f" | {trip:.2f} |")


def main() -> None:
    import sys

    if "--sdk" in sys.argv:
        nic = 25.0
        if "--nic" in sys.argv:
            nic = float(sys.argv[sys.argv.index("--nic") + 1])
        sdk_main(nic)
        return
    if "--modal-o" in sys.argv:
        TRIP_O_MS["0.5B"] = 9.0
    labels = {1.0: "measured, 1 thread, 2 digit GEMMs", 0.53: "measured, 1 thread, stacked digits",
              0.25: "dedicated gateway, est."}
    for model in ("0.5B", "7B"):
        for gw in (1.0, 0.53, 0.25):
            print(f"\n{model}, local-cost factor {gw} ({labels[gw]})")
            print("| placement | RTT 1 ms | RTT 5 ms | RTT 17 ms |")
            print("|---|---|---|---|")
            for name, local in PLACEMENTS.items():
                cells = []
                for rtt in (1.0, 5.0, 17.0):
                    s = s_per_token(model, local, rtt, gw)
                    cells.append(f"{s:.3f} s ({1 / s:.1f} tok/s)")
                print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
