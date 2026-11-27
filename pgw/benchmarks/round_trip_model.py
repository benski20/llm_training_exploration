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
    python benchmarks/round_trip_model.py --consumer [--tau 1,1.8,3 --rows 1,3,4] [--gw target] [--rows-free] [--tp 2]
        # consumer devices over the internet: fused schedules, RTT 0.5 / 5 / 20 / 50 ms (consumer-round-trips.md)
    python benchmarks/round_trip_model.py --twenty [--goal 20]   # the largest RTT per trip that still meets the goal
"""

from __future__ import annotations

import itertools
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
#: inline noise statistics of one sampled trip (``OutboundNoiseMonitor.inspect_planned``, before the trip's payload
#: leaves), us = fixed + per inspected value: measured here (one core, min of 30) at n = 896 / 4864 / 8192 / 29568
#: values: 93 / 160 / 225 / 746 us, fitted. The monitor samples ``SDK_NOISE_RATE`` of the trips (its default).
SDK_NOISE_US = (75.0, 0.0227)
SDK_NOISE_RATE = 0.05
#: tokens per verify pass (strict, chain) and whether the pass runs the draft head: prompt lookup measured here on
#: Qwen2.5-0.5B (sdk_trip_profile.py, 64 tokens): open chat 1.14 (R=2) / 1.15 (R=3), a code edit 1.66 (R=2) /
#: 2.10-2.14 (R=3) / 2.52 (R=4). Draft head (speculation="draft_head": the shipped head + prompt lookup), measured
#: 2026-11-26 (heads retrained without non-commercial data): 7B held-out open chat 1.45 / 1.64 / 1.72 at R = 2 / 3 / 4
#: (MBPP code 1.94 at R = 3), exact replay of the 7B's own greedy answers through the strict acceptance rule
#: (benchmarks/results/feature_head/kaggle_7b_r2; alpha 0.47 chat / 0.64 code at depth 1); 0.5B end to end through
#: the library 1.52 / 1.82 / 1.95 (sdk_spec_bench.py, kaggle_05b_r2).
#: 32B / 72B have no trained head: the 7B values are an assumption there. EAGLE-class (alpha 0.8, attention over
#: past features; not trained) for comparison: (1 - a^3) / (1 - a).
SDK_TAU = {"off (R=1)": (1, 1.0, False), "lookup, open chat (R=2)": (2, 1.14, False),
           "lookup, open chat (R=3)": (3, 1.15, False), "lookup, copy-heavy (R=2)": (2, 1.66, False),
           "lookup, copy-heavy (R=3)": (3, 2.1, False),
           "draft head, open chat (R=2)": (2, 1.45, True), "draft head, open chat (R=3)": (3, 1.64, True),
           "draft head, open chat (R=4)": (4, 1.72, True), "draft head, code (R=3)": (3, 1.94, True),
           "EAGLE-class a=0.8 (R=3, not trained)": (3, (1 - 0.8 ** 3) / (1 - 0.8), True)}
#: draft head CPU time per head step (R - 1 steps per pass), ms: 7B d = 384, V = 16,384 measured here (float32
#: kernel, 4 threads, feature_head_cpu_bench.py; 0.88 ms; the native int8 kernel 1.3 ms); 32B / 72B scaled by the
#: head's weights (estimate, H = 5120 / 8192).
SDK_HEAD_STEP_MS = {"7B": 0.88, "32B": 1.02, "72B": 1.30}


def _sm() -> Any:
    """The sibling module scaling_model.py (this directory on sys.path)."""
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import scaling_model

    return scaling_model


def _noise_us(values: int) -> float:
    return SDK_NOISE_US[0] + SDK_NOISE_US[1] * values


def sdk_ms_per_token(model: str, rtt_ms: float, rows: int = 1, tau: float = 1.0, nic_gbit: float = 25.0,
                     tp: int | None = None, lane_ms: float | None = None, gw_scale: float = 1.0,
                     head: bool = False) -> dict[str, float]:
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
           "serve": SDK_SERVE_MS * (1 + 0.1 * (rows - 1)),
           "noise": SDK_NOISE_RATE * (shape.layers * sum(_noise_us(rows * n) for _, n, _ in shape.linears())
                                      + _noise_us(rows * shape.hidden)) / 1e3,
           "draft": (rows - 1) * SDK_HEAD_STEP_MS[model] if head else 0.0}
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
        for mode, (rows, tau, head) in SDK_TAU.items():
            cells = []
            for rtt in rtts:
                b = sdk_ms_per_token(model, rtt, rows, tau, nic_gbit, head=head)
                cells.append(f"{b['ms_per_token']:.1f} ({1e3 / b['ms_per_token']:.1f})")
            b = sdk_ms_per_token(model, rtts[0], rows, tau, nic_gbit, head=head)
            print(f"| {model} | {mode} | {b['critical_trips_per_token']:.0f} | " + " | ".join(cells) + " |")
    print("\nComponents per pass at RTT 0.2 ms, 1 row (ms): ")
    for model in SDK_SHAPES:
        b = sdk_ms_per_token(model, 0.2, 1, 1.0, nic_gbit)
        print(f"  {model}: " + ", ".join(f"{k} {v:.1f}" for k, v in b.items() if k not in ("ms_per_token",)))
    print("\nSensitivity, RTT 0.2 ms: per-trip overhead o = lane + gateway (ms per token, tok/s)")
    for model in SDK_SHAPES:
        cells = []
        for lane, gws, name in ((None, 1.0, "today"), (0.05, 1.0, "lane 0.05"), (0.05, 0.5, "lane 0.05 + gateway/2"),
                                (0.05, 0.5, "same + shipped draft head R=3"),
                                (0.05, 0.5, "same + draft head a=0.8 R=3")):
            rows, tau = (1, 1.0)
            if "shipped" in name:
                rows, tau = SDK_TAU["draft head, open chat (R=3)"][:2]
            elif "draft" in name:
                rows, tau = (3, (1 - 0.8 ** 3) / 0.2)
            b = sdk_ms_per_token(model, 0.2, rows, tau, nic_gbit, lane_ms=lane, gw_scale=gws, head=rows > 1)
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


# ---- consumer devices over the internet: trips x RTT with fused schedules (research/reviews/consumer-round-trips.md)
# Shared model for the high-RTT work: other studies plug their tokens per pass (tau) and rows per pass in through
# ``consumer_pass_ms(...)`` / ``--tau`` / ``--rows``. Per pass:
#   trips x (RTT + o_wan + client work per trip(rows)) + device(rows) + serve + draft,   tok/s = tau / pass.
# Bytes on the link are a separate, optional term (``link_mbit``): WAN transport is studied elsewhere.

#: (layers, hidden, intermediate, qkv out features, vocab) of the Qwen2.5 instruct models.
C_SHAPES: dict[str, tuple[int, int, int, int, int]] = {
    "0.5B": (24, 896, 4864, 1152, 151936), "3B": (36, 2048, 11008, 2560, 151936),
    "7B": (28, 3584, 18944, 4608, 152064), "32B": (64, 5120, 27648, 7168, 152064),
    "72B": (80, 8192, 29568, 10240, 152064)}

#: schedule -> (trips per pass as a function of L, bytes per ring element of the layers, device factor vs K32 singles,
#: client work per layer trip-pair key). Trip counts are exact (engine tests); the head is on the provider in all of
#: them (a weight-free client cannot stream the float head).
#:  standard: qkv, o, gate_up, down per layer + the masked head trip (the client library today);
#:  fused2+head trip: 2L + 1 fused2 trips + the K32 masked head trip (what the gateway service could combine today);
#:  fused2 folded (this work): head folded into the last trip + layer 0's qkv local: 2L;
#:  parallel blocks (needs a parallel-block model): [qkv | gate_up] of layer 0, then one pair per layer, head folded.
C_SCHEDULES: dict[str, tuple[Any, int, float]] = {
    "standard (today, 4L+1)": (lambda L: 4 * L + 1, 4, 1.0),
    "fused2 + head trip (2L+2)": (lambda L: 2 * L + 2, 7, 1.22),
    "fused2 folded head, local qkv0 (2L)": (lambda L: 2 * L, 7, 1.22),
    "parallel blocks fused (L+1)": (lambda L: L + 1, 7, 1.22),
}
#: measured, H100: graphed K32 single trips 0.186 ms per 7B layer of 233 MB of int8 weights (1.25 TB/s), scaled by
#: weight bytes (``scaling_model.DEVICE_MS_PER_MB``); the fused2 ``planes`` pair op costs 9.27 vs 7.60 ms per 7B token
#: (1 row, x1.22; ``benchmarks/results/fused2/pair_op_planes_h100_2026-11-23.json``).
C_DEVICE_MS_PER_MB = 0.186 / 233.0
#: measured, H100, 7B fused2 pair trips, device time at R rows / 1 row (sum over the two pairs of a layer): today's
#: kernels are far from memory-bound in rows (gate_up dominates). ``rows_free`` replaces it with a target kernel that
#: stays memory-bound to 16 rows (+3 % per row; estimate, not built).
C_ROWS_DEVICE = {1: 1.0, 2: 1.62, 4: 2.83, 8: 5.37}
#: measured client work per trip, 32B widths, one core (STATUS 2026-11-23, calibrated to the Modal gateway): standard
#: decode driver 0.105 ms per trip, native fused2 driver 0.58 ms per pair trip; scaled to other models by the layer's
#: element count sum(n + m). "target" = 0.06 ms per pair trip at 32B (split over several cores; not measured).
C_GW = {"std": 0.105 / (115712 / 4), "f2": 0.58 / (115712 / 2), "f2_target": 0.06 / (115712 / 2)}
#: the masked head's client work (152k outputs), ms; per-row share of client work (SDK_ROW_FIXED).
C_GW_HEAD_MS = 0.26
#: per-trip transport overhead beyond the RTT on a WAN TLS lane (framing, syscalls; estimate).
C_WAN_O_MS = 0.05
#: client mask factory CPU per slot (one core, pks-v3, K32; round 3 measured 35 / 112 / 166 ms at 7B / 32B / 72B;
#: 0.5B / 3B scaled by elements). fused2 / parallel at K56 by pks-v3 digit decomposition (7 x 8-bit products per
#: mask, ``pgw.factory.fused2_he``): x ~6 (estimate: the HE parts are ~75 % of a slot and scale with the digits).
C_MASK_MS = {"0.5B": 9.0, "3B": 17.0, "7B": 35.0, "32B": 112.0, "72B": 166.0}
C_MASK_K56_FACTOR = 6.0


def c_layer_elems(model: str) -> tuple[int, int, int]:
    """(inputs, outputs, int8 weights) of one layer's four linears."""
    _L, d, inter, q, _v = C_SHAPES[model]
    ins = d + d + d + inter
    outs = q + d + 2 * inter + d
    w = d * q + d * d + d * 2 * inter + inter * d
    return ins, outs, w


def _rows_factor(rows: int, rows_free: bool) -> float:
    if rows_free:
        return 1.0 + 0.03 * (rows - 1)
    keys = sorted(C_ROWS_DEVICE)
    if rows >= keys[-1]:
        return C_ROWS_DEVICE[keys[-1]] * rows / keys[-1]
    for a, b in itertools.pairwise(keys):
        if a <= rows <= b:
            return C_ROWS_DEVICE[a] + (C_ROWS_DEVICE[b] - C_ROWS_DEVICE[a]) * (rows - a) / (b - a)
    return 1.0


def consumer_pass_ms(model: str, schedule: str, rtt_ms: float, rows: int = 1, *, tp: int = 1,
                     gw: str = "measured", rows_free: bool = False, draft_step_ms: float = 0.9,
                     link_mbit: tuple[float, float] | None = None, o_wan_ms: float = C_WAN_O_MS) -> dict[str, float]:
    """One decode pass of ``rows`` rows (ms by component). ``gw``: "measured" (one core) or "target"."""
    L, d, _i, _q, vocab = C_SHAPES[model]
    trips_fn, bpe, dev_f = C_SCHEDULES[schedule]
    trips = trips_fn(L)
    ins, outs, w = c_layer_elems(model)
    fused = schedule != "standard (today, 4L+1)"
    head_mb = d * vocab / 1e6
    device = (L * w / 1e6 * dev_f + head_mb) * C_DEVICE_MS_PER_MB * _rows_factor(rows, rows_free) / tp
    device += trips * (0.010 if tp > 1 else 0.0)
    rowf = SDK_ROW_FIXED + (1 - SDK_ROW_FIXED) * rows
    if fused:
        rate = C_GW["f2" if gw == "measured" else "f2_target"]
    else:
        rate = C_GW["std"] if gw == "measured" else C_GW["std"] * 0.6
    client = (L * (ins + outs) * rate + C_GW_HEAD_MS) * rowf
    if schedule.startswith("fused2 folded"):
        # local qkv0: exact int8 GEMV of the public folded matrix, memory-bound at few rows (~10 GB/s, one thread,
        # both activation digits stacked: one pass) or compute-bound (~50 G MAC/s, 2 digits) at many rows
        client += max(d * q_out(model) / 10e9, rows * 2 * d * q_out(model) / 50e9) * 1e3
    out = {"trips": float(trips), "network": trips * rtt_ms, "transport": trips * o_wan_ms, "device": device,
           "client": client, "serve": SDK_SERVE_MS * (1 + 0.1 * (rows - 1)),
           "draft": (rows - 1) * draft_step_ms}
    if link_mbit is not None:
        up = rows * L * ins * bpe * 8 / (link_mbit[0] * 1e6) * 1e3
        down = rows * (L * outs * bpe + vocab * 4) * 8 / (link_mbit[1] * 1e6) * 1e3
        out["link"] = up + down
    out["pass_ms"] = sum(v for k, v in out.items() if k != "trips")
    return out


def q_out(model: str) -> int:
    return C_SHAPES[model][3]


def consumer_tok_s(model: str, schedule: str, rtt_ms: float, tau: float = 1.0, rows: int = 1, **kw: Any) -> float:
    return 1e3 * tau / consumer_pass_ms(model, schedule, rtt_ms, rows, **kw)["pass_ms"]


def mask_cores(model: str, schedule: str, tok_s: float, rows: int, tau: float) -> float:
    """Client cores the mask factory needs at that decode rate (rows / tau slots per token)."""
    per = C_MASK_MS[model] * (1.0 if schedule.startswith("standard") else C_MASK_K56_FACTOR)
    return tok_s * rows / tau * per / 1e3


def consumer_main(taus: list[float], rows_list: list[int], gw: str, rows_free: bool, tp32: int) -> None:
    rtts = (0.5, 5.0, 20.0, 50.0)
    print("\nConsumer path: tok/s by RTT (one stream; H100 provider" + (f", 32B/72B on {tp32} GPUs (TP)" if tp32 > 1
          else "") + f"; client work {gw}" + ("; target multi-row kernel" if rows_free else "") + ")")
    for tau, rows in zip(taus, rows_list, strict=True):
        print(f"\ntokens per pass {tau:g}, rows per pass {rows}")
        print("| model | schedule | trips/pass | " + " | ".join(f"RTT {r:g} ms" for r in rtts) +
              " | provider device ms/pass | client work ms/pass | mask cores at RTT 20 |")
        print("|---|---|---|" + "---|" * len(rtts) + "---|---|---|")
        for model in C_SHAPES:
            tp = tp32 if model in ("32B", "72B") else 1
            for sched in C_SCHEDULES:
                cells = [f"{consumer_tok_s(model, sched, r, tau, rows, tp=tp, gw=gw, rows_free=rows_free):.2f}"
                         for r in rtts]
                b = consumer_pass_ms(model, sched, 20.0, rows, tp=tp, gw=gw, rows_free=rows_free)
                ts = consumer_tok_s(model, sched, 20.0, tau, rows, tp=tp, gw=gw, rows_free=rows_free)
                print(f"| {model} | {sched} | {b['trips']:.0f} | " + " | ".join(cells) +
                      f" | {b['device']:.1f} | {b['client']:.1f} | {mask_cores(model, sched, ts, rows, tau):.2f} |")


def twenty_main(goal: float = 20.0) -> None:
    """What it takes for ``goal`` tok/s: the largest RTT per trip that still meets it, by depth and tokens per pass.
    Per pass: trips x (RTT + o + device/trip + client/trip) + serve <= 1000 T / goal (rows = T: a lower bound on the
    device and client terms; real speculation needs rows > T). Device, client: H100, target client work, target
    multi-row kernel (optimistic on purpose: these are upper bounds on the allowed RTT)."""
    print(f"\nWhat {goal:g} tok/s needs: the largest RTT (ms) per trip, by model and tokens per pass T (rows = T; "
          "target client work and multi-row kernel; '-' = impossible even at RTT 0)")
    ts = (1, 2, 3, 4, 6, 8)
    print("| model (layers) | schedule | trips | " + " | ".join(f"T={t}" for t in ts) + " |")
    print("|---|---|---|" + "---|" * len(ts))
    for model in C_SHAPES:
        L = C_SHAPES[model][0]
        tp = 2 if model in ("32B", "72B") else 1
        for sched in C_SCHEDULES:
            cells = []
            for t in ts:
                b = consumer_pass_ms(model, sched, 0.0, t, tp=tp, gw="target", rows_free=True)
                budget = 1e3 * t / goal
                slack = (budget - b["pass_ms"]) / b["trips"]
                cells.append(f"{slack:.2f}" if slack > 0 else "-")
            print(f"| {model} ({L}) | {sched} | {C_SCHEDULES[sched][0](L)} | " + " | ".join(cells) + " |")
    print("\nPer-trip budget 1000 T / (goal x trips) in ms, before device and client work (any model of that depth):")
    print("| layers | 4L+1 | 2L | L+1 | (T = 1 / 4 / 8) |")
    print("|---|---|---|---|---|")
    for L in (24, 28, 36, 48, 64, 80):
        cells = []
        for f in (lambda x: 4 * x + 1, lambda x: 2 * x, lambda x: x + 1):
            cells.append(" / ".join(f"{1e3 * t / (goal * f(L)):.2f}" for t in (1, 4, 8)))
        print(f"| {L} | " + " | ".join(cells) + " | |")


def main() -> None:
    import sys

    if "--sdk" in sys.argv:
        nic = 25.0
        if "--nic" in sys.argv:
            nic = float(sys.argv[sys.argv.index("--nic") + 1])
        sdk_main(nic)
        return
    if "--consumer" in sys.argv or "--twenty" in sys.argv:
        def arg(name: str, default: str) -> str:
            return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

        if "--consumer" in sys.argv:
            taus = [float(x) for x in arg("--tau", "1,1.8,3").split(",")]
            rows = [int(x) for x in arg("--rows", "1,3,4").split(",")]
            consumer_main(taus, rows, arg("--gw", "measured"), "--rows-free" in sys.argv, int(arg("--tp", "1")))
        if "--twenty" in sys.argv:
            twenty_main(float(arg("--goal", "20")))
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
