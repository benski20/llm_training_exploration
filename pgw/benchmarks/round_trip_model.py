"""Per-token decode latency model for research/plans/fewer-round-trips.md (R1). No I/O, no torch.

s/token = remote_trips * (RTT + o) + layers * local_linear_ms(placement) + F, divided by tau (tokens
per pass) for speculation. Inputs are measured values (see the memo for sources); `gw` scales local
linear cost: 1.0 = measured here (1 thread, shared 4-vCPU box, exact 2-digit int8 GEMV, ~3 GB/s of
weights), 0.25 = a dedicated 4-8 core gateway or a 1-pass int16xint8 kernel (estimate, not measured).

    python benchmarks/round_trip_model.py
"""

from __future__ import annotations

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


def main() -> None:
    for model in ("0.5B", "7B"):
        for gw in (1.0, 0.25):
            print(f"\n{model}, local-cost factor {gw} ({'measured, 1 thread' if gw == 1 else 'dedicated gateway, est.'})")
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
