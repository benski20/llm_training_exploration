#!/usr/bin/env python3
"""Numbers of ``mask-cost-consumer.md``: mask cost per token at Qwen2.5-7B / 32B for consumer devices.

Run from ``pgw/``: ``PYTHONPATH=src python research/reviews/mask_cost_numbers.py [--json out.json]``.

Provider: the calibrated pks-v3 device-item model of :mod:`pgw.service.memplan` (``pks_item_seconds``, H100
datasheet rates, T4-calibrated kernels, the third round's kernels and fused NTTs). An item of ``G`` groups is
affine in ``G`` (``T(G) = c + g G``: ``c`` is the hint part, i.e. the ``W8 A`` GEMM, its switch, the ring rows and
the key switch's forward NTTs of ``A~``; ``g`` the per-group part: ``y = W8 b``, the key switch's products and
inverse NTTs, the uploads and replies). Coalescing ``S`` groups under one public seed costs ``c + g S``.

Client: the measured per-slot CPU of ``STATUS.md`` (2026-11-26, client mask CPU round 3, one core, all parts).
GPU offload: measured per-linear on a Kaggle T4 by ``benchmarks/pks_client_gpu_bench.py`` when its JSON is given
(``--client-gpu``), else the op-count estimate below.

Every device number is an ESTIMATE unless marked measured.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from typing import Any

from pgw.hemask.wire import V3_KSK_BYTES
from pgw.service.memplan import H100_RATES, MODELS, MemConfig, PksRates, pks_item_seconds

MODELS_USED = {"7B": "qwen2.5-7b", "32B": "qwen2.5-32b"}
# client CPU per slot, ms, one core (STATUS 2026-11-26 round 3, after): parts
CLIENT_MS = {
    "7B": {"r": 1.6, "encrypt+keys+wire": 17.9, "decrypt": 7.7, "self_check": 3.1, "uW8": 1.2, "writer": 3.0},
    "32B": {"r": 5.2, "encrypt+keys+wire": 55.4, "decrypt": 25.8, "self_check": 10.2, "uW8": 7.8, "writer": 8.9},
}
TOK_S = 20.0  # target tokens per second per user
SLOTS_PER_TOKEN = (1.3, 2.0, 2.6)  # strict speculation decode-only .. measured incl. prompts (STATUS 2026-11-23)
# other GPUs, relative to the H100 model (datasheet ratios: dense int8 GEMM, HBM bandwidth, 32-bit integer rate)
GPUS: dict[str, PksRates] = {
    "H100": H100_RATES,
    "H200": replace(H100_RATES, name="H200 (estimate)", hbm_bytes_s=H100_RATES.hbm_bytes_s * 4.8 / 3.35),
    "B200": replace(H100_RATES, name="B200 (estimate)", int8_ops_s=H100_RATES.int8_ops_s * 4.5 / 1.98,
                    hbm_bytes_s=H100_RATES.hbm_bytes_s * 8.0 / 3.35, int32_ops_s=H100_RATES.int32_ops_s * 1.25,
                    h2d_pinned_s=H100_RATES.h2d_pinned_s * 2, d2h_s=H100_RATES.d2h_s * 2),
    "MI300X": replace(H100_RATES, name="MI300X (estimate)", int8_ops_s=H100_RATES.int8_ops_s * 2.6 / 1.98,
                      hbm_bytes_s=H100_RATES.hbm_bytes_s * 5.3 / 3.35, int32_ops_s=H100_RATES.int32_ops_s * 1.25),
}


def linears(model: str) -> list[tuple[str, int, int]]:
    return MODELS[MODELS_USED[model]].linears(MemConfig().masked_head)


def item_s(m: int, n: int, groups: int, rates: PksRates, hide_upload: bool = False) -> float:
    t = pks_item_seconds(m, n, 256 * groups, "device", rates, ntt="fused", kernels="all")
    if hide_upload:  # the upload behind a second concurrent item (2 per device by default; not in memplan's model)
        t.pop("upload_h2d", None)
    return sum(t.values())


def affine(model: str, rates: PksRates = H100_RATES, hide_upload: bool = False) -> tuple[float, float]:
    """Seconds per full-model cycle: ``(c, g)`` with ``T(G) = c + g G`` (fit at G = 1 and 16; the model is affine
    up to its ``max`` terms, which only lower ``T`` at small ``G``)."""
    t1 = sum(item_s(m, n, 1, rates, hide_upload) for _, m, n in linears(model))
    t16 = sum(item_s(m, n, 16, rates, hide_upload) for _, m, n in linears(model))
    g = (t16 - t1) / 15
    return t1 - g, g


def slots_per_s(model: str, groups_per_hint: float, rates: PksRates = H100_RATES, hide_upload: bool = False
                ) -> float:
    c, g = affine(model, rates, hide_upload)
    return 256 * groups_per_hint / (c + g * groups_per_hint)


def stage_split(model: str, groups: int, rates: PksRates = H100_RATES) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, m, n in linears(model):
        for k, v in pks_item_seconds(m, n, 256 * groups, "device", rates, ntt="fused", kernels="all").items():
            out[k] = out.get(k, 0.0) + v
    return {k: round(1e3 * v / (256 * groups), 3) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def bytes_per_slot(model: str) -> dict[str, float]:
    lins = linears(model)
    sm, sn = sum(m for _, m, _ in lins), sum(n for _, _, n in lins)
    return {"sum_m": sm, "sum_n": sn,
            "online_up_MB": 4 * sn / 1e6, "online_down_MB": 4 * sm / 1e6,
            "mask_up_MB": (7 * sn + len(lins) * V3_KSK_BYTES / 256) / 1e6, "mask_down_MB": 4.5 * sm / 1e6,
            "mask_store_MB": 4 * sm / 1e6}


def ops_per_slot(model: str) -> dict[str, float]:
    """Operation counts per slot (the floor discussion of the note, section 2)."""
    lins = linears(model)
    sm, sn = sum(m for _, m, _ in lins), sum(n for _, _, n in lins)
    smn = sum(m * n for _, m, n in lins)
    return {
        "plain_Wr_int8_MAC": 4 * smn,  # W8 r mod 2^32 locally: 4 byte planes of r
        "y_gemm_int8_MAC": 7 * smn,  # W8 b mod 2^56: 7 byte planes of b (the floor of any b of 56 bits)
        "hint_gemm_int8_MAC_per_group": 7 * smn * 4096,  # W8 A, once per hint (divide by 256 S per slot)
        # key switch per output-slot: 16 x 4 pointwise MACs + 4 inverse NTTs (6.5 butterflies per value) + ModDown
        "ks_modmul_per_group_part": sm * (64 + 26 + 4),
        "ks_fwd_ntt_modmul_per_hint": sm * 16 * 4 * 8192 * 6.5 / 32,
        # client: pad (3 primes: pointwise + inverse NTT of 4096 = 3 x (1 + 6)), noise, decrypt (2 x (1 + 6.5))
        "client_pad_modmul": sn * 21, "client_decrypt_modmul": sm * 15,
    }


def users(per_gpu_slots: float, spt: float) -> float:
    return per_gpu_slots / (TOK_S * spt)


def coalesced_users(model: str, spt: float, rates: PksRates = H100_RATES, gpus: int = 1,
                    stock_groups: int = 1, hide_upload: bool = False) -> dict[str, float]:
    """Users served by ``gpus`` GPUs (linears split evenly) when every user has ``stock_groups`` groups of 256 slots
    in flight per sweep and the provider sweeps the linears once per period, serving every waiting group with one
    hint (common seed). ``U`` users: sweep ``T = (c + g U stock_groups) / gpus``; each user gets
    ``256 stock_groups`` slots per sweep and needs ``TOK_S spt`` per second, so ``T <= 256 stock_groups / (TOK_S
    spt)``. Largest ``U``."""
    c, g = affine(model, rates, hide_upload)
    need = TOK_S * spt
    tmax = 256 * stock_groups / need
    u = (tmax * gpus - c) / (g * stock_groups)
    return {"users": max(0.0, u), "users_per_gpu": max(0.0, u) / gpus, "sweep_s": tmax,
            "groups_per_hint": max(0.0, u) * stock_groups}


def client(model: str, gpu_ms: float | None) -> dict[str, Any]:
    parts = CLIENT_MS[model]
    total = sum(parts.values())
    # GPU offload: encryption (pad, noise, layout), the reply keys' NTTs, decryption, the self-check and u W8 move to
    # the device; the CPU keeps the request / reply bytes and the writer, plus a share of r (keystream) for safety
    cpu_left = parts["writer"] + parts["r"] + 0.15 * parts["encrypt+keys+wire"]  # 15 %: the wire framing on the CPU
    return {"cpu_ms_today": round(total, 1), "cpu_ms_gpu_offload": round(cpu_left, 1),
            "gpu_ms_per_slot": None if gpu_ms is None else round(gpu_ms, 2)}


# ---- K56 masks for fused2 (section 12 of the note) -------------------------------------------------------------
#: direct K56 parameter set (pgw.hemask.pks56) relative to one K32 product: provider 10/7 byte planes (GEMMs),
#: 6/4 key-switch primes (products, NTTs), 10/7 upload and 60/36 reply bytes -> ~1.45; client 4/3 pad primes,
#: 6/4 reply-key primes, 3/2 decryption primes -> ~1.4 (estimates from the operation counts; not benchmarked)
K56_DIRECT_PROVIDER = 1.45
K56_DIRECT_CLIENT = 1.4


def _lin_affine(m: int, n: int, rates: PksRates, hide_upload: bool) -> tuple[float, float]:
    t1, t16 = item_s(m, n, 1, rates, hide_upload), item_s(m, n, 16, rates, hide_upload)
    g = (t16 - t1) / 15
    return t1 - g, g


def k56_slots_per_s(model: str, option: str, groups_per_hint: float = 1.0, g_scale: float = 1.0,
                    rates: PksRates = H100_RATES, hide_upload: bool = True) -> float:
    """K56 mask slots per second per GPU for fused2 (the same linears' shapes; the folded head as the head).

    ``option``: ``digit7`` (today: 7 separate K32 products per mask), ``top`` (the top-digit plan, planes sent
    separately), ``top_stacked`` (planes in one item: the hint once), ``digit7_stacked``, ``direct`` (pks-v3-K56),
    ``k32`` (reference: a K32 slot). ``groups_per_hint``: groups of other customers sharing the hint (common seed).
    """
    from pgw.factory.fused2_he import digit_plan

    total = 0.0
    s = float(groups_per_hint)
    for _, m, n in linears(model):
        c, g = _lin_affine(m, n, rates, hide_upload)
        g *= g_scale
        planes = len(digit_plan(n, 56, 32))
        if option == "k32":
            total += (c + g * s) / s
        elif option == "digit7":
            total += 7 * (c + g * s) / s
        elif option == "top":
            total += planes * (c + g * s) / s
        elif option == "top_stacked":
            total += (c + planes * g * s) / s
        elif option == "digit7_stacked":
            total += (c + 7 * g * s) / s
        elif option == "direct":
            total += K56_DIRECT_PROVIDER * (c + g * s) / s
        else:
            raise ValueError(option)
    return 256 / total


def k56_client_planes(model: str) -> float:
    """Client work of one top-digit K56 slot in K32 slots: rows encrypted (by n) and decrypted (by m), weighted
    by the measured split of the K32 client parts (upload ~ 60 %, decryption + check ~ 40 %)."""
    from pgw.factory.fused2_he import digit_plan

    lins = linears(model)
    pn = sum(len(digit_plan(n, 56, 32)) * n for _, _m, n in lins) / sum(n for _, _m, n in lins)
    pm = sum(len(digit_plan(n, 56, 32)) * m for _, m, n in lins) / sum(m for _, m, _n in lins)
    return 0.6 * pn + 0.4 * pm


def k56_table() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in MODELS_USED:
        res: dict[str, Any] = {}
        for gs in (1.0, 4.0):
            for sh in (1, 64):
                key = f"g_scale={gs:g} S={sh}"
                res[key] = {o: round(k56_slots_per_s(model, o, sh, gs), 1)
                            for o in ("k32", "digit7", "top", "top_stacked", "digit7_stacked", "direct")}
        res["users_per_h100_spt2"] = {k: {o: round(v / (TOK_S * 2.0), 2) for o, v in d.items()}
                                      for k, d in res.items() if k.startswith("g_scale")}
        cp = k56_client_planes(model)
        k32ms = sum(CLIENT_MS[model].values())
        res["client_ms_per_slot_cpu"] = {"k32": round(k32ms, 1), "digit7": round(7 * k32ms, 1),
                                         "top_digit": round(cp * k32ms, 1),
                                         "direct (est.)": round(K56_DIRECT_CLIENT * k32ms, 1)}
        res["client_top_planes"] = round(cp, 2)
        out[model] = res
    return out


def table(client_gpu: dict[str, float] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in MODELS_USED:
        c, g = affine(model)
        res: dict[str, Any] = {"affine_s_per_cycle": {"hint_c": round(c, 3), "per_group_g": round(g, 3)},
                               "bytes": bytes_per_slot(model), "ops": ops_per_slot(model),
                               "stages_ms_per_slot_G1": stage_split(model, 1),
                               "stages_ms_per_slot_G4": stage_split(model, 4)}
        res["slots_per_s"] = {f"H100 S={s}": round(slots_per_s(model, s), 1) for s in (1, 4, 16, 64, 256)}
        res["slots_per_s"]["H100 S=inf (per-group floor of the model)"] = round(256 / g, 1)
        for sh in (1, 16, 64):
            res["slots_per_s"][f"H100 S={sh} upload hidden"] = round(slots_per_s(model, sh, hide_upload=True), 1)
        res["slots_per_s"]["H100 S=inf upload hidden"] = round(256 / affine(model, hide_upload=True)[1], 1)
        mb = res["bytes"]["mask_up_MB"] + res["bytes"]["mask_down_MB"]
        res["nic_slots_per_s"] = {f"{gb} Gbit/s": round(gb * 1e9 / 8 / (mb * 1e6), 1) for gb in (10, 25, 100, 400)}
        for name, rates in GPUS.items():
            res["slots_per_s"][f"{name} S=1"] = round(slots_per_s(model, 1, rates), 1)
            res["slots_per_s"][f"{name} S=64"] = round(slots_per_s(model, 64, rates), 1)
        res["users_per_h100"] = {}
        for spt in SLOTS_PER_TOKEN:
            res["users_per_h100"][f"spt={spt}"] = {
                "today_G1": round(users(slots_per_s(model, 1), spt), 2),
                "today_G4": round(users(slots_per_s(model, 4), spt), 2),
                "coalesced_1gpu": {k: round(v, 2) for k, v in coalesced_users(model, spt).items()},
                "coalesced_8gpu": {k: round(v, 2) for k, v in coalesced_users(model, spt, gpus=8).items()},
                "coalesced_8gpu_upload_hidden": {k: round(v, 2) for k, v in
                                                 coalesced_users(model, spt, gpus=8, hide_upload=True).items()},
                "upper_bound_S_inf_upload_hidden": round(users(256 / affine(model, hide_upload=True)[1], spt), 2),
            }
        gms = None if client_gpu is None else client_gpu.get(model)
        cl = client(model, gms)
        res["client"] = cl
        res["client_cores_per_user"] = {
            f"spt={spt}": {"today": round(TOK_S * spt * cl["cpu_ms_today"] / 1e3, 2),
                           "gpu_offload_cpu": round(TOK_S * spt * cl["cpu_ms_gpu_offload"] / 1e3, 2),
                           "gpu_busy_frac": None if gms is None else round(TOK_S * spt * gms / 1e3, 3)}
            for spt in SLOTS_PER_TOKEN}
        b = res["bytes"]
        res["bandwidth_per_user_Gbit_s"] = {
            f"spt={spt}": {"online": round(TOK_S * spt * (b["online_up_MB"] + b["online_down_MB"]) * 8e-3, 2),
                           "masks": round(TOK_S * spt * (b["mask_up_MB"] + b["mask_down_MB"]) * 8e-3, 2)}
            for spt in SLOTS_PER_TOKEN}
        out[model] = res
    out["k56_fused2"] = k56_table()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--client-gpu", default=None, help="JSON of benchmarks/pks_client_gpu_bench.py (ms per slot)")
    a = ap.parse_args()
    cg = None
    if a.client_gpu:
        with open(a.client_gpu) as f:
            cg = {k: float(v) for k, v in json.load(f)["ms_per_slot"].items()}
    t = table(cg)
    txt = json.dumps(t, indent=1, default=lambda x: round(x, 3) if isinstance(x, float) else x)
    print(txt)
    if a.json:
        with open(a.json, "w") as f:
            f.write(txt)
    _ = math


if __name__ == "__main__":
    main()
