#!/usr/bin/env python3
"""Numbers of ``consumer-bandwidth.md``: bytes per row and per mask slot, floors, reductions, link tables.

Run from ``pgw/``: ``PYTHONPATH=src python research/reviews/consumer_bandwidth_numbers.py [--json out.json]``.

Exact counts (shapes of the Qwen2.5 instruct models, the wire formats of ``pgw.core.packing`` /
``rust/pgw-lane/src/proto.rs`` / ``pgw.hemask.wire``) except where a line says "estimate". The per-link throughput
and TTFT use ``benchmarks/round_trip_model.py`` (the bandwidth term added for this note).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "benchmarks"))
import round_trip_model as rtm  # noqa: E402

MODELS = ("0.5B", "1.5B", "3B", "7B", "32B")
rtm.C_SHAPES.setdefault("1.5B", (28, 1536, 8960, 2048, 151936))
#: int8 weights the device holds for device-made masks / carry correction (linears + int8 head; tied heads count
#: once more because the head is its own int8 matrix in the factory pack), bytes.
LANE_REQ, LANE_REP, LANE_ITEM = 48, 40, 8  # lane header bytes per frame / reply / multi-frame item
TLS_REC = 22 / 16384  # TLS 1.3 record overhead per payload byte (5 B header + 1 B type + 16 B tag per 16 KiB)
#: measured here (one core, torch int8 GEMM, 256 / 64 columns): 112-247 G MAC/s; a mid value for mask GEMMs.
CPU_GMAC_S = 200.0
#: consumer GPU, RTX 4070 class: ~116 T int8 MAC/s dense (datasheet; estimate for real GEMM efficiency 60 %).
GPU_TMAC_S = 116.0 * 0.6


def elems(model: str) -> dict[str, int]:
    L, d, inter, q, vocab = rtm.C_SHAPES[model]
    per_layer = {"qkv": (d, q), "o": (d, d), "gate_up": (d, 2 * inter), "down": (inter, d)}
    return {"L": L, "d": d, "I": inter, "q": q, "vocab": vocab, "per_layer": per_layer,
            "sum_n": L * (3 * d + inter) + d, "sum_m": L * (q + 2 * d + 2 * inter) + vocab,
            "weights": L * (d * q + d * d + 2 * d * inter + inter * d) + d * vocab}


def today(model: str) -> dict[str, Any]:
    """Bytes per row on today's wire (K32: 4 B per element; fused K56: 7 B), schedules and trips."""
    e = elems(model)
    L = e["L"]
    out: dict[str, Any] = {}
    for sched, key in (("standard (today, 4L+1)", "standard"), ("fused2 folded head, local qkv0 (2L)", "fused2"),
                       ("parallel blocks fused (L+1)", "parallel")):
        up, down = rtm.row_bytes(model, sched, "k32")
        upp, downp = rtm.row_bytes(model, sched, "k32", logits=False)
        trips = rtm.C_SCHEDULES[sched][0](L)
        hdr = trips * (LANE_REQ + LANE_REP + LANE_ITEM)
        out[key] = {"trips": trips, "decode_up_MB": up / 1e6, "decode_down_MB": down / 1e6,
                    "decode_total_MB": (up + down) / 1e6, "prefill_row_MB": (upp + downp) / 1e6,
                    "headers_per_pass_B": hdr, "headers_pct_at_1_row": 100 * hdr / (up + down),
                    "tls_pct": 100 * TLS_REC}
    mu, md = rtm.mask_bytes_per_slot(model, "pks-v3")
    out["mask_slot"] = {"up_MB": mu / 1e6, "down_MB": md / 1e6, "store_MB": 4 * e["sum_m"] / 1e6,
                        "reply_keys_MB": (4 * L + 1) * rtm.KSK_PER_SLOT / 1e6}
    out["per_linear_7B_style"] = {k: {"up_B": 4 * n, "down_B": 4 * m} for k, (n, m) in e["per_layer"].items()}
    return out


def reductions(model: str) -> dict[str, Any]:
    """Bytes per decode row after each sound reduction (standard schedule) and per mask slot."""
    e = elems(model)
    base = sum(rtm.row_bytes(model, "standard (today, 4L+1)", "k32"))
    rows: dict[str, Any] = {}
    for wire in ("k32", "narrow", "carry16"):
        up, down = rtm.row_bytes(model, "standard (today, 4L+1)", wire)
        rows[wire] = {"up_MB": up / 1e6, "down_MB": down / 1e6, "saving_pct": 100 * (1 - (up + down) / base)}
    for m in (12, 20, 24):
        up, down = e["sum_n"] * m / 8, e["sum_m"] * 28 / 8
        rows[f"carry{m}"] = {"up_MB": up / 1e6, "down_MB": down / 1e6, "saving_pct": 100 * (1 - (up + down) / base)}
    # local LM head on a device that holds W8 (no head trip): minus vocab outputs and d inputs
    up, down = rtm.row_bytes(model, "standard (today, 4L+1)", "carry16")
    up -= e["d"] * 16 / 8
    down -= e["vocab"] * 28 / 8
    rows["carry16 + local head"] = {"up_MB": up / 1e6, "down_MB": down / 1e6,
                                    "saving_pct": 100 * (1 - (up + down) / base),
                                    "head_MAC_per_row": e["d"] * e["vocab"]}
    masks = {}
    for k in ("pks-v3", "pks-v3 narrow"):
        mu, md = rtm.mask_bytes_per_slot(model, k)
        masks[k] = {"up_MB": mu / 1e6, "down_MB": md / 1e6}
    # device-made masks: W8 r with r in [0, 2^m): ceil(m / 8) int8 byte planes of r
    dev = {}
    for label, m in (("narrow (m=28)", 28), ("carry16", 16), ("carry20", 20)):
        mac = math.ceil(m / 8) * e["weights"]
        dev[label] = {"MAC_per_slot": mac, "cpu_core_ms_per_slot": mac / (CPU_GMAC_S * 1e9) * 1e3,
                      "gpu_ms_per_slot_compute": mac / (GPU_TMAC_S * 1e12) * 1e3}
    return {"rows": rows, "masks_downloaded": masks, "masks_on_device": dev,
            "weights_GB": e["weights"] / 1e9}


def carry_cost(model: str, mean_abs: float = 100.0) -> dict[str, Any]:
    """Client carry work per decode row as a fraction of one full int8 product of the row (all linears + head):
    padded to the data-independent budget (worst case qmax = 2047), and the expected work at ``mean_abs`` (the
    0.5B measured mean |xq| is of this order; see the bench JSON)."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
    from pgw.core.narrow import carry_budget

    e = elems(model)
    full = e["weights"]
    out = {}
    for m in (14, 16, 18, 20, 22, 24):
        padded = 0
        typical = 0.0
        for _k, (n, mo) in e["per_layer"].items():
            padded += e["L"] * carry_budget(n, m, 2047) * mo
            typical += e["L"] * n * mean_abs / 2 ** m * mo
        padded += carry_budget(e["d"], m, 2047) * e["vocab"]
        typical += e["d"] * mean_abs / 2 ** m * e["vocab"]
        out[m] = {"padded_pct_of_product": 100 * padded / full, "typical_pct_of_product": 100 * typical / full,
                  "padded_ms_one_core_20GBps": padded / 20e9 * 1e3,
                  "up_bits": m}
    return out


def link_table(rtts: tuple[float, ...] = (5.0, 20.0)) -> dict[str, Any]:
    specs = {"off": (1, 1.0), "draft head R=2": (2, 1.45), "draft head R=3": (3, 1.64), "draft head R=4": (4, 1.72)}
    combos = [("standard (today, 4L+1)", "k32", "pks-v3"), ("standard (today, 4L+1)", "k32", "none"),
              ("standard (today, 4L+1)", "narrow", "pks-v3 narrow"), ("standard (today, 4L+1)", "narrow", "none"),
              ("standard (today, 4L+1)", "carry16", "none"), ("fused2 folded head, local qkv0 (2L)", "k32", "none"),
              ("fused2 folded head, local qkv0 (2L)", "carry16", "none"),
              ("parallel blocks fused (L+1)", "carry16", "none")]
    out: dict[str, Any] = {}
    for model in ("0.5B", "3B", "7B", "32B"):
        tp = 2 if model == "32B" else 1
        for sched, wire, masks in combos:
            key = f"{model} | {sched.split(' (')[0]} | {wire} | {masks}"
            row: dict[str, Any] = {}
            for lname, link in rtm.LINKS.items():
                cell = {"bw_only": 1e3 / rtm.link_ms(model, sched, 1, link, wire, masks)}
                for rtt in rtts:
                    best = 0.0
                    for rows, tau in specs.values():
                        b = rtm.consumer_pass_ms(model, sched, rtt, rows, tp=tp, gw="target", rows_free=True)
                        best = max(best, 1e3 * tau / (b["pass_ms"] + rtm.link_ms(model, sched, rows, link, wire,
                                                                                   masks)))
                    cell[f"rtt{rtt:g}"] = best
                row[lname] = cell
            out[key] = row
    return out


def ttft(prompt: int = 500, rtt: float = 20.0) -> dict[str, Any]:
    """Seconds to the first token for a ``prompt``-token prompt (no prefix-cache hit): every prompt row moves its
    masked values (no head outputs except the last row) plus one pass of trips."""
    out: dict[str, Any] = {}
    for model in ("0.5B", "7B", "32B"):
        L = rtm.C_SHAPES[model][0]
        for wire in ("k32", "narrow", "carry16"):
            up, down = rtm.row_bytes(model, "standard (today, 4L+1)", wire, logits=False)
            cells = {}
            for lname, (dn, upl) in rtm.LINKS.items():
                t = prompt * ((up + rtm.ACK_UP * down) / (upl * 1e6 * rtm.LINK_EFF / 8) + down / (dn * 1e6 *
                                                                                                  rtm.LINK_EFF / 8))
                cells[lname] = t + (4 * L + 1) * rtt / 1e3
            out[f"{model} {wire}"] = cells
    return out


def per_day(model: str) -> dict[str, Any]:
    """GB per day for a chat user: requests x (new prompt rows + generated rows x rows per token), every row its
    online bytes plus (pks-v3) its mask slot. Profiles: light 10, typical 40, heavy 150 requests per day, 600 new
    prompt tokens (after the prefix cache) and 400 generated tokens per request, 1.3 rows per generated token."""
    rows_per_req = 600 + 400 * 1.3
    out = {}
    for wire, masks in (("k32", "pks-v3"), ("k32", "none"), ("narrow", "pks-v3 narrow"), ("carry16", "none")):
        up, down = rtm.row_bytes(model, "standard (today, 4L+1)", wire)
        mu, md = rtm.mask_bytes_per_slot(model, masks)
        per_row = up + down + mu + md
        out[f"{wire} + masks {masks}"] = {p: n * rows_per_req * per_row / 1e9
                                          for p, n in (("light", 10), ("typical", 40), ("heavy", 150))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res: dict[str, Any] = {"today": {m: today(m) for m in MODELS}, "reductions": {m: reductions(m) for m in MODELS},
                           "carry_cost": {m: carry_cost(m) for m in ("0.5B", "7B", "32B")},
                           "links": link_table(), "ttft_500": ttft(), "per_day_GB": {m: per_day(m) for m in MODELS}}
    print("| model | trips | up MB/row | down MB/row | total | prefill row | mask slot up | mask slot down |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        t = res["today"][m]
        s = t["standard"]
        print(f"| {m} | {s['trips']} | {s['decode_up_MB']:.2f} | {s['decode_down_MB']:.2f} | "
              f"{s['decode_total_MB']:.2f} | {s['prefill_row_MB']:.2f} | {t['mask_slot']['up_MB']:.2f} | "
              f"{t['mask_slot']['down_MB']:.2f} |")
    print("\nfused2 (2L) / parallel (L+1) at K56, MB per decode row (up / down):")
    for m in MODELS:
        t = res["today"][m]
        print(f"  {m}: fused2 {t['fused2']['decode_up_MB']:.2f} / {t['fused2']['decode_down_MB']:.2f}; parallel "
              f"{t['parallel']['decode_up_MB']:.2f} / {t['parallel']['decode_down_MB']:.2f}; headers "
              f"{t['standard']['headers_pct_at_1_row']:.4f} % of a standard row")
    print("\nreductions, MB per decode row up / down (saving):")
    for m in MODELS:
        r = res["reductions"][m]["rows"]
        print(f"  {m}: " + "; ".join(f"{k} {v['up_MB']:.2f}/{v['down_MB']:.2f} ({v['saving_pct']:.0f} %)"
                                     for k, v in r.items()))
    print("\nmasks: downloaded (MB per slot up / down) and made on the device (ms per slot):")
    for m in MODELS:
        r = res["reductions"][m]
        md = r["masks_downloaded"]
        dv = r["masks_on_device"]
        print(f"  {m} (W8 {r['weights_GB']:.1f} GB): " + "; ".join(f"{k} {v['up_MB']:.2f}/{v['down_MB']:.2f}"
                                                                   for k, v in md.items()) + " | " +
              "; ".join(f"{k} CPU {v['cpu_core_ms_per_slot']:.0f} ms, GPU {v['gpu_ms_per_slot_compute']:.2f} ms"
                        for k, v in dv.items()))
    print("\ncarry work (fraction of one full product per row): padded / typical at mean |xq| 100")
    for m, cc in res["carry_cost"].items():
        print(f"  {m}: " + "; ".join(f"m={b} {v['padded_pct_of_product']:.2f}/{v['typical_pct_of_product']:.3f} %"
                                     for b, v in cc.items()))
    print("\nper-link tok/s: bandwidth only / RTT 5 ms / RTT 20 ms (best speculation, target client work)")
    print("| model | schedule | wire | masks while chatting | " + " | ".join(rtm.LINKS) + " |")
    print("|---|---|---|---|" + "---|" * len(rtm.LINKS))
    for k, row in res["links"].items():
        print(f"| {k} | " + " | ".join(f"{c['bw_only']:.2f} / {c['rtt5']:.2f} / {c['rtt20']:.2f}"
                                       for c in row.values()) + " |")
    print("\nTTFT, 500-token prompt, RTT 20 ms (s):")
    for k, row in res["ttft_500"].items():
        print(f"  {k}: " + ", ".join(f"{ln} {v:.1f}" for ln, v in row.items()))
    print("\nGB per day (light / typical / heavy):")
    for m in MODELS:
        print(f"  {m}: " + "; ".join(f"{k} {v['light']:.0f}/{v['typical']:.0f}/{v['heavy']:.0f}"
                                     for k, v in res["per_day_GB"][m].items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
