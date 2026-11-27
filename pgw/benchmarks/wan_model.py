#!/usr/bin/env python3
"""Decode tokens/s of a consumer client over a WAN: RTT, access-link bandwidth and the bytes every trip moves.

Per verify pass (``rows`` = speculation width R, ``tau`` = tokens accepted per pass)::

    pass = trips x (RTT + o) + device + up_bytes / up_bw + down_bytes / down_bw
    tok/s = tau / pass

* ``trips``: 4L + 1 today, 2L + 1 with fused2 (7B 113 / 57, 32B 257 / 129); every trip waits for the previous one;
* ``o``: per-trip cost beyond the network (lane, both ends' socket work, gateway work): measured over the emulated
  WAN by ``wan_bench.py lane`` (``--o-ms``; 0.15 ms default = the target);
* ``device``: H100 per-trip time from ``scaling_model.py`` (weights read once per trip, the 7B calibration);
* bytes: every linear's input goes up and its output comes down, ``bytes_per_element`` each (K32 = 4; fused2 needs
  K40 = 5 or K48 = 6), per row; the masked LM head's 152k logits come down every pass. The trip cannot start before
  the previous trip's bytes are in, so serialisation adds per trip (no overlap within a pass);
* ``maskgen``: with pks-v3 every generated token also consumes one mask slot whose provider reply (~4.5 bytes per
  output element) comes down the same link, unless the stock was filled while idle (``--stock prefilled``).

    python benchmarks/wan_model.py                    # the "who can get 20 tok/s" tables of docs/sdk/wan.md
    python benchmarks/wan_model.py --o-ms 0.3 --stock live
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scaling_model as sm

MODELS = {"7B": "qwen2.5-7b", "32B": "qwen2.5-32b"}


@dataclass(frozen=True)
class Access:
    name: str
    down_mbit: float
    up_mbit: float
    rtt_ms: float  # client to a GPU in the same metro, idle, wired unless noted


#: Typical consumer access (idle RTT to a well-peered GPU site in the same metro; see docs/sdk/wan.md for sources and
#: spread). Distance adds ~1 ms RTT per 100 km of fibre path (light in glass, ~200 km per ms one way) times the
#: route's detour factor (1.5-2 typical).
ACCESS = [
    Access("fibre 10G/10G, wired, same metro", 10_000, 10_000, 1.5),
    Access("fibre 1G/1G, wired, same metro", 1_000, 1_000, 2.0),
    Access("fibre 1G/1G over Wi-Fi 6, same metro", 1_000, 600, 4.0),
    Access("cable 1G/50 (DOCSIS 3.1), same metro", 1_000, 50, 12.0),
    Access("5G mobile / FWA 300/50", 300, 50, 25.0),
    Access("fibre 1G/1G, region ~1000 km away", 1_000, 1_000, 18.0),
    Access("co-located (same DC, 25 Gbit)", 25_000, 25_000, 0.2),
]


def pass_ms(model: str, rtt_ms: float, down_mbit: float, up_mbit: float, *, rows: int = 1, fused: bool = False,
            o_ms: float = 0.15, bpe: float | None = None, head: bool = True) -> dict[str, float]:
    shape = sm.SHAPES[MODELS[model]]
    trips = (2 if fused else 4) * shape.layers + (1 if head else 0)
    bpe = bpe if bpe is not None else (5.0 if fused else 4.0)
    layer_mb = sum(n * m for _, n, m in shape.linears()) / sm.MB
    device = shape.layers * layer_mb * sm.DEVICE_MS_PER_MB * (1.0 if not fused else 1.15)
    if head:
        device += shape.hidden * shape.vocab / sm.MB * sm.DEVICE_MS_PER_MB
    up = rows * (shape.sum_n(with_head=False) * bpe + (shape.hidden * 4 if head else 0))
    down = rows * (shape.sum_m(with_head=False) * bpe + (shape.vocab * 4 if head else 0))
    net = trips * rtt_ms
    ser = (up * 8 / (up_mbit * 1e6) + down * 8 / (down_mbit * 1e6)) * 1e3
    total = net + trips * o_ms + device + ser
    return {"trips": trips, "network_ms": net, "o_ms": trips * o_ms, "device_ms": device, "serialise_ms": ser,
            "up_mb": up / 1e6, "down_mb": down / 1e6, "pass_ms": total}


def maskgen_ms_per_token(model: str, down_mbit: float, up_mbit: float) -> float:
    """Link time of one pks-v3 slot (reply ~4.5 B per output element down, the upload ~1/8 of that: estimate)."""
    shape = sm.SHAPES[MODELS[model]]
    down = shape.sum_m(with_head=True) * 4.5
    up = shape.sum_n(with_head=True) * 4.5
    return (down * 8 / (down_mbit * 1e6) + up * 8 / (up_mbit * 1e6)) * 1e3


MODES = {"no speculation": (1, 1.0), "draft head R=3 (tau 1.64)": (3, 1.64), "4 tokens per pass (R=5, tau 4)": (5, 4.0)}


def tok_s(model: str, acc: Access, rows: int, tau: float, fused: bool, o_ms: float, stock: str) -> float:
    p = pass_ms(model, acc.rtt_ms, acc.down_mbit, acc.up_mbit, rows=rows, fused=fused, o_ms=o_ms)
    ms = p["pass_ms"] / tau
    if stock == "live":
        ms += maskgen_ms_per_token(model, acc.down_mbit, acc.up_mbit)
    return 1e3 / ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--o-ms", type=float, default=0.15)
    ap.add_argument("--stock", choices=["prefilled", "live"], default="prefilled")
    a = ap.parse_args()
    for model in MODELS:
        for fused in (False, True):
            sched = "fused2 (2L+1 trips, K40)" if fused else "today (4L+1 trips, K32)"
            print(f"\n{model}, {sched}, o = {a.o_ms} ms/trip, mask stock {a.stock}: decode tok/s")
            print("| access | RTT ms | " + " | ".join(MODES) + " |")
            print("|---|---|" + "---|" * len(MODES))
            for acc in ACCESS:
                cells = [f"{tok_s(model, acc, r, t, fused, a.o_ms, a.stock):.1f}" for r, t in MODES.values()]
                print(f"| {acc.name} | {acc.rtt_ms:g} | " + " | ".join(cells) + " |")
    print("\nPer pass at 1 row (MB up / down) and the split at fibre 1G/1G, RTT 2 ms:")
    for model in MODELS:
        for fused in (False, True):
            p = pass_ms(model, 2.0, 1000, 1000, fused=fused, o_ms=a.o_ms)
            print(f"  {model} {'fused2' if fused else 'today'}: {p['trips']} trips, {p['up_mb']:.1f} MB up / "
                  f"{p['down_mb']:.1f} MB down; network {p['network_ms']:.0f} ms, o {p['o_ms']:.0f} ms, device "
                  f"{p['device_ms']:.1f} ms, serialisation {p['serialise_ms']:.0f} ms -> {p['pass_ms']:.0f} ms")
    print("\nWhat 20 tok/s needs at 4 tokens per pass (R=5, tau 4: a 200 ms pass), fused2, o = 0.15 ms:")
    for model in MODELS:
        p0 = pass_ms(model, 0.0, 1e12, 1e12, rows=5, fused=True, o_ms=a.o_ms)
        budget = 4 / 20 * 1e3 - p0["pass_ms"]
        # all of the budget to the network: RTT with infinite bandwidth, bandwidth with RTT 0
        rtt_max = budget / p0["trips"]
        bw = (p0["up_mb"] + p0["down_mb"]) * 8 / (budget / 1e3)
        print(f"  {model}: {p0['trips']} trips, {p0['up_mb']:.0f} MB up + {p0['down_mb']:.0f} MB down per pass; "
              f"RTT <= {rtt_max:.2f} ms at infinite bandwidth, or >= {bw:.0f} Mbit/s (both ways summed) at RTT 0; "
              f"maskgen adds {maskgen_ms_per_token(model, 1000, 1000):.0f} ms per token at 1G/1G unless prefilled")


if __name__ == "__main__":
    main()
