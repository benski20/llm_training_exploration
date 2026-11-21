#!/usr/bin/env python3
"""Scale sizing for the north star: hard-to-serve models (70B+ dense, large MoE) at 1-5k tokens/s through PGW.

Extends ``docs/tools/pgw_sizing.py`` (one dense model, per-day volumes) to aggregate throughput targets,
mixture-of-experts models under routing-oblivious dispatch (:mod:`pgw.moe.oblivious`), provider GPU counts,
gateway counts (NIC-bound), mask-store bandwidth and capacity, the encrypted mask supply, and where the
bottleneck sits. Standard library only. Every figure carries a provenance tag:

* ``analytic``: exact arithmetic on the model shape and ring;
* ``derived``: arithmetic on a measured number (``docs/measurements.json`` or ``--gpu-tops``);
* ``estimate``: rests on an assumption (another GPU's speed, a scaling law, an RTT).

Examples::

    python benchmarks/scale_sizing.py --model llama-3-70b --tokens-per-s 1000
    python benchmarks/scale_sizing.py --model mixtral-8x22b --tokens-per-s 5000 --gpu h100 --encrypted
    python benchmarks/scale_sizing.py --all --tokens-per-s 1000 --markdown

MoE models are sized under the only private dispatch PGW offers: every expert receives one row per token in
the batch (dummies are uniform and need no mask), so wire and provider compute follow the *total* expert
parameters while masks, mask storage and the mask factory follow the *routed* (active) ones.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parents[1] / "docs" / "tools" / "pgw_sizing.py"


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("pgw_sizing_base", _TOOLS)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_base()
Ring = base.Ring
RINGS = base.RINGS
MB, GB, TB = 1e6, 1e9, 1e12

#: Effective *logical* ring-matmul throughput per GPU (``2 m n rows / t`` for one exact K32 ring matmul, i.e.
#: counting the int8 limbs as one GEMM). ``ring_tops = None``: derived from the measured Llama-3-70B T4 rate
#: at batch 32 (docs/measurements.json ``server_70b_t4_batch32``); ``--gpu-tops`` overrides it (e.g. with the
#: large-batch figure of benchmarks/scale_provider_bench.py). The other GPUs are the T4 figure scaled by the
#: ratio of dense int8 peak TOPS (estimate). ``mem_gb`` bounds how many int8 weights one GPU holds.
GPUS: dict[str, dict[str, float | None]] = {
    "t4": {"ring_tops": None, "mem_gb": 15.0, "peak_int8_tops": 130.0},
    "l4": {"ring_tops": None, "mem_gb": 23.0, "peak_int8_tops": 485.0},
    "a100": {"ring_tops": None, "mem_gb": 79.0, "peak_int8_tops": 624.0},
    "h100": {"ring_tops": None, "mem_gb": 79.0, "peak_int8_tops": 1979.0},
}
#: Customer-side encrypt + decrypt per 7B slot (s), by implementation.
#: ``dense_amx``: measured, 4 vCPU with AMX or a gateway T4 (docs/measurements.json he_gateway_7b_s_per_token).
#: ``ntt``: measured on the 4-vCPU development container with benchmarks/hemask_ntt_bench.py (public matrix and
#: hint held in NTT form, 2026-11-21: 24.2 ms/slot). The NTT-form hint takes 12 bytes per coefficient instead of 6.
HE_GATEWAY_7B = {"dense_amx": None, "ntt": 0.0242}

#: X3: the whole encrypted factory measured end to end at Qwen2.5-7B shapes (all 28 layers, synthetic int8
#: weights) on one Kaggle T4x2 VM with 4 vCPUs: the real compute service on GPU 0, the real factory
#: (MaskGenerator + HEMaskBackend + RemoteMulmod64 over loopback HTTP) on GPU 1, every row Freivalds-checked.
#: ``pipeline``: full-model slots/s of the generator as it is today (stages per linear run one after another);
#: ``stages``: each stage alone on layer 0, as full-model slots/s (``encrypt`` ntt_pre on the 4 vCPUs,
#: ``provider`` = wire + service + T4 GEMM, ``decrypt`` device passes on GPU 1). Source:
#: benchmarks/he_e2e_bench.py, Kaggle job pgw-x3-he-e2e-7b (benchmarks/results/kaggle/x3_he_e2e_7b).
#: The T4x2 run has not happened yet (Kaggle queue); until then the stage rates are, per stage:
#: ``encrypt`` 75.4 slots/s: ntt_pre alone on the 4-vCPU dev container (measured, he_e2e_bench --cpu, 7B
#: shapes, 2026-11-21); ``decrypt`` 24.3 slots/s: CPU int8 (AMX) decryption alone on the same box (measured;
#: device passes on a T4 not yet measured); ``provider`` 240 slots/s: Kaggle T4 maskgen at 7B shapes, fused
#: path, one GPU (benchmarks/results/kaggle/maskgen_provider_run2, measured, wire excluded). ``pipeline``:
#: 5.66 slots/s is the dev container's whole factory with a CPU provider (measured, not a T4 figure), so it
#: is not used for the pairs. ``hint_disk_gb_per_s``: O_DIRECT read of a hint file on the dev container.
HE_E2E_7B_T4X2: dict[str, Any] = {
    "config": "ntt_pre encrypt (CPU) + CPU decrypt, provider T4 (stage mix, see above)",
    "pipeline": None,
    "stages": {"encrypt": 75.4, "provider": 240.0, "decrypt": 24.3},
    "hint_disk_gb_per_s": 2.25,
}


def he_e2e_rates(shape: Shape, groups_per_pass: int = 6, disk_gb_per_s: float | None = None,
                 measured: dict[str, Any] | None = None) -> dict[str, Fig]:
    """End-to-end encrypted factory rate of one T4x2 "pair" (provider T4 + gateway T4, 4 vCPU) for ``shape``.

    The 7B stage rates are measured; another model's rate scales each stage by its work per slot: encryption
    by the inputs ``L n`` (one NTT row per input feature), decryption by the hint rows ``L m``, the provider by
    ``L m n``. ``overlapped`` (all stages concurrent, the target of the overlap work in HEMaskBackend) is the
    slowest stage; ``serial`` is today's structure (``1 / sum(1 / rate)``). The hint read bound is the disk
    rate over the hint bytes read per slot (the whole hint once per decryption pass of ``groups_per_pass``
    256-slot groups) and applies when the hint does not fit in RAM.
    """
    meas = HE_E2E_7B_T4X2 if measured is None else measured
    ref = MODELS["qwen2.5-7b"]
    a, r = shape.active(), ref.active()
    ratio = {"encrypt": ref.layers * r["n"] / (shape.layers * a["n"]),
             "decrypt": ref.layers * r["m"] / (shape.layers * a["m"]),
             "provider": ref.layers * r["mn"] / (shape.layers * a["mn"])}
    exact = shape.name == ref.name
    kind = "derived" if exact else "estimate"
    out: dict[str, Fig] = {}
    stages = meas.get("stages") or {}
    rates = {k: float(v) * ratio[k] for k, v in stages.items() if v}
    for k, v in rates.items():
        note = "stage alone, source per stage in HE_E2E_7B_T4X2" + ("" if exact else ", scaled from 7B")
        out[f"he_pair_{k}_slots_per_s"] = Fig(v, "slots/s", "measured" if exact else "estimate", note)
    if len(rates) == 3:
        out["he_pair_overlapped_slots_per_s"] = Fig(min(rates.values()), "slots/s", "estimate",
                                                    f"slowest stage ({min(rates, key=rates.get)}): stages overlapped")
        out["he_pair_serial_slots_per_s"] = Fig(1.0 / sum(1.0 / v for v in rates.values()), "slots/s", kind,
                                                "stages one after another")
    if meas.get("pipeline") and exact:
        out["he_pair_pipeline_slots_per_s"] = Fig(float(meas["pipeline"]), "slots/s", "measured",
                                                  f"whole factory, {meas.get('config')}")
    hint = shape.layers * shape.all_weights()["m"] * base.HE_RANK * base.HE_DEGREE * 6  # q=2^56: 6 byte planes
    per_slot = hint / (base.HE_DEGREE * groups_per_pass)
    out["he_hint_read_per_slot"] = Fig(per_slot / MB, "MB", "analytic",
                                       f"whole hint once per pass of {groups_per_pass} groups")
    disk = disk_gb_per_s if disk_gb_per_s is not None else meas.get("hint_disk_gb_per_s")
    if disk:
        out["he_hint_disk_bound_slots_per_s"] = Fig(float(disk) * GB / per_slot, "slots/s", "derived",
                                                    f"hint larger than RAM, read at {float(disk):.2f} GB/s")
    return out


@dataclass(frozen=True)
class Shape:
    """Decoder-only transformer, dense or MoE (SwiGLU experts; optional always-on shared expert)."""

    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int  # dense FFN width, or the per-expert width for MoE
    experts: int = 0  # 0 = dense
    top_k: int = 0
    shared_ffn: int = 0  # width of the always-on shared expert(s), 0 = none
    total_params_b: float = 0.0  # for display only

    @property
    def moe(self) -> bool:
        return self.experts > 0

    def attn(self) -> dict[str, tuple[int, int]]:
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        return {"qkv": (q + 2 * kv, self.hidden), "o": (self.hidden, q)}

    def ffn_groups(self, width: int) -> dict[str, tuple[int, int]]:
        return {"gate_up": (2 * width, self.hidden), "down": (self.hidden, width)}

    def per_layer(self, rows_per_expert: float) -> dict[str, float]:
        """Per layer and token: sum of input widths (``n``), output rows (``m``) and weights (``m n``) over
        every linear the token's rows touch, with each expert counted ``rows_per_expert`` times."""
        groups = list(self.attn().values())
        mult = [1.0, 1.0]
        if self.moe:
            if self.shared_ffn:
                groups += list(self.ffn_groups(self.shared_ffn).values())
                mult += [1.0, 1.0]
            groups += list(self.ffn_groups(self.ffn).values())
            mult += [rows_per_expert, rows_per_expert]
        else:
            groups += list(self.ffn_groups(self.ffn).values())
            mult += [1.0, 1.0]
        return {"n": sum(n * c for (_, n), c in zip(groups, mult, strict=True)),
                "m": sum(m * c for (m, _), c in zip(groups, mult, strict=True)),
                "mn": sum(m * n * c for (m, n), c in zip(groups, mult, strict=True))}

    def online(self) -> dict[str, float]:
        """What the provider computes and the wire carries per token (oblivious MoE: every expert)."""
        return self.per_layer(float(self.experts) if self.moe else 1.0)

    def active(self) -> dict[str, float]:
        """What needs masks per token (routed experts only)."""
        return self.per_layer(float(self.top_k) if self.moe else 1.0)

    def all_weights(self) -> dict[str, float]:
        """Every linear once (weights resident at the provider; the encrypted factory's hint)."""
        return self.per_layer(float(self.experts) if self.moe else 1.0)

    def trips_per_token(self) -> int:
        """Sequential round trips per decode step: qkv, o, gate_up, down per layer (MoE: shared and routed
        experts of a layer go in the same two trips)."""
        return 4 * self.layers


MODELS: dict[str, Shape] = {s.name: s for s in (
    Shape("qwen2.5-7b", 3584, 28, 28, 4, 128, 18944, total_params_b=7.6),
    Shape("llama-3-70b", 8192, 80, 64, 8, 128, 28672, total_params_b=70.6),
    Shape("llama-3.1-405b", 16384, 126, 128, 8, 128, 53248, total_params_b=405.0),
    Shape("mixtral-8x22b", 6144, 56, 48, 8, 128, 16384, experts=8, top_k=2, total_params_b=141.0),
    Shape("qwen3-235b-a22b", 4096, 94, 64, 4, 128, 1536, experts=128, top_k=8, total_params_b=235.0),
)}


@dataclass
class Fig:
    value: float
    unit: str
    kind: str
    note: str = ""


@dataclass
class Report:
    model: str
    tokens_per_s: float
    gpu: str
    encrypted: bool
    figures: dict[str, Fig] = field(default_factory=dict)
    bottlenecks: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "tokens_per_s": self.tokens_per_s, "gpu": self.gpu, "encrypted": self.encrypted,
                "figures": {k: asdict(v) for k, v in self.figures.items()},
                "bottlenecks": [{"resource": r, "units": u} for r, u in self.bottlenecks]}

    def format(self) -> str:
        w = max(len(k) for k in self.figures)
        out = [f"PGW scale sizing: {self.model} at {self.tokens_per_s:,.0f} tokens/s, provider GPU {self.gpu}, "
               f"mask supply {'encrypted' if self.encrypted else 'plain (trusted factory)'}", ""]
        for k, f in self.figures.items():
            note = f"  ({f.note})" if f.note else ""
            out.append(f"  {k:<{w}}  {f.value:>14,.4g} {f.unit:<14} [{f.kind}]{note}")
        out += ["", "  units needed (largest first): " + ", ".join(f"{r} {u:,.1f}" for r, u in self.bottlenecks)]
        return "\n".join(out)


def size(shape: Shape, *, tokens_per_s: float, ring: str = "K32A12C27", gpu: str = "t4", encrypted: bool = False,
         rtt_ms: float = 0.2, nic_gbit: float = 100.0, nic_efficiency: float = 0.7, buffer_s: float = 600.0,
         nvme_gb_per_s: float = 3.0, batch_per_gateway: int = 256, service_ms_per_trip: float = 0.5,
         he_gateway: str = "ntt", measurements: dict[str, dict[str, Any]] | None = None) -> Report:
    """Size one deployment. ``tokens_per_s`` counts prompt and generated tokens (one mask slot each)."""
    if not (tokens_per_s > 0 and math.isfinite(tokens_per_s)):
        raise ValueError("tokens_per_s must be a positive number")
    if gpu not in GPUS:
        raise ValueError(f"unknown gpu {gpu!r}; choose from {sorted(GPUS)}")
    r = RINGS[ring]
    meas = measurements if measurements is not None else base.load_measurements()
    b = r.bytes_per_element
    L = shape.layers
    on, act, allw = shape.online(), shape.active(), shape.all_weights()
    g = dict(GPUS[gpu])
    t4_ref = 2 * MODELS["llama-3-70b"].layers * MODELS["llama-3-70b"].all_weights()["mn"] / float(
        meas["server_70b_t4_batch32"]["value"]) / 1e12
    measured_gpu = g["ring_tops"] is not None or gpu == "t4"
    if g["ring_tops"] is None:
        g["ring_tops"] = t4_ref * float(g["peak_int8_tops"]) / float(GPUS["t4"]["peak_int8_tops"])
    rep = Report(shape.name, tokens_per_s, gpu, encrypted)
    f = rep.figures
    tps = tokens_per_s

    # --- per token (analytic) -----------------------------------------------------------------------------------
    up, down = L * on["n"] * b, L * on["m"] * b
    mask = L * act["m"] * b
    f["weights_int8_resident"] = Fig(L * allw["mn"] / GB, "GB", "analytic", "all masked linears, int8")
    f["online_wire_per_token"] = Fig((up + down) / MB, "MB", "analytic",
                                     f"up {up / MB:.1f} + down {down / MB:.1f}" + (", every expert one row"
                                                                                    if shape.moe else ""))
    f["mask_bytes_per_token"] = Fig(mask / MB, "MB", "analytic", "W r of routed rows; r is a seed")
    if shape.moe:
        leaky = L * shape.active()["n"] * b + mask
        f["moe_wire_overhead_vs_leaky"] = Fig((up + down) / leaky, "x", "analytic",
                                              "oblivious vs routed-rows-only dispatch (which would leak routing)")
    f["round_trips_per_step"] = Fig(float(shape.trips_per_token()), "trips", "analytic", "per decode step")

    # --- aggregate rates ----------------------------------------------------------------------------------------
    wire_bps = tps * (up + down)
    f["online_wire_aggregate"] = Fig(wire_bps * 8 / 1e9, "Gbit/s", "analytic", "both directions")
    f["mask_rate"] = Fig(tps * mask / GB, "GB/s", "analytic", "produced = consumed")
    he_up = he_down = 0.0
    if encrypted:
        q = r.he_q_bits
        he_up = L * act["n"] * (q - 3) / 8  # wire v2: q-3 bit upload, K+3 bit reply
        he_down = L * act["m"] * (r.k + 3) / 8
        f["he_supply_wire_per_token"] = Fig((he_up + he_down) / MB, "MB", "analytic", "maskgen wire v2")
        f["he_supply_wire_aggregate"] = Fig(tps * (he_up + he_down) * 8 / 1e9, "Gbit/s", "analytic")
        hint = L * allw["m"] * base.HE_RANK * base.HE_DEGREE * (q // 8 - 1)
        f["he_hint_size"] = Fig(hint / GB, "GB", "analytic", "W8 A for every linear (all experts), held by the gateway")

    # --- gateways (NIC-bound) -----------------------------------------------------------------------------------
    per_gw_bytes = nic_gbit * 1e9 / 8 * nic_efficiency
    gw_tps_nic = per_gw_bytes / (up + down + he_up + he_down)
    f["gateway_tok_s_nic_bound"] = Fig(gw_tps_nic, "tok/s", "estimate",
                                       f"{nic_gbit:g} Gbit/s NIC at {nic_efficiency:.0%}, online + supply")
    # decode step latency at batch B per gateway: trips x (RTT + service) (estimate)
    step_s = shape.trips_per_token() * (rtt_ms + service_ms_per_trip) / 1000.0
    gw_tps_batch = batch_per_gateway / step_s
    f["decode_step_latency"] = Fig(step_s, "s", "estimate",
                                   f"{shape.trips_per_token()} trips x ({rtt_ms:g} ms RTT + {service_ms_per_trip:g} ms "
                                   "service); one token per stream per step")
    f["gateway_tok_s_batch_bound"] = Fig(gw_tps_batch, "tok/s", "estimate", f"{batch_per_gateway} streams per gateway")
    gw_tps = min(gw_tps_nic, gw_tps_batch)
    gateways = tps / gw_tps
    f["gateways_needed"] = Fig(gateways, "gateways", "estimate", "min of NIC and batch bound")
    f["streams_needed"] = Fig(tps * step_s, "concurrent streams", "estimate", "tokens/s x step latency")

    # --- provider GPUs (online) ----------------------------------------------------------------------------------
    online_flops = 2 * L * on["mn"] * tps  # logical ring-matmul ops per second
    kind_gpu = "derived" if measured_gpu else "estimate"
    gpus_online = online_flops / (g["ring_tops"] * 1e12)
    gpus_mem = (L * allw["mn"]) / (g["mem_gb"] * GB * 0.85)
    f["provider_gpus_online"] = Fig(gpus_online, gpu, kind_gpu, f"{g['ring_tops']:.1f} ring-TOPS per GPU")
    f["provider_gpus_min_memory"] = Fig(gpus_mem, gpu, "analytic", "one replica's int8 weights at 85% of memory")

    # --- mask factory ------------------------------------------------------------------------------------------
    plain_flops = 2 * L * act["mn"] * tps  # W r for every routed row: one ring matmul per slot
    if encrypted:
        planes = r.he_q_bits // 8
        prov = plain_flops * planes / (r.k // 8)  # 7 byte planes at q=2^56 vs 4 limbs at K32
        f["provider_gpus_maskgen"] = Fig(prov / (g["ring_tops"] * 1e12), gpu, "estimate",
                                         f"{planes} ciphertext byte planes vs {r.k // 8} ring limbs")
        ref = MODELS["qwen2.5-7b"].active()
        ref_rows = MODELS["qwen2.5-7b"].layers * (ref["n"] + ref["m"])
        per7b = HE_GATEWAY_7B.get(he_gateway, 0.0)
        if he_gateway not in HE_GATEWAY_7B:
            raise ValueError(f"he_gateway must be one of {sorted(HE_GATEWAY_7B)}")
        if per7b is None:
            per7b = float(meas["he_gateway_7b_s_per_token"]["value"])
        gw_s = per7b * L * (act["n"] + act["m"]) / ref_rows
        f["gateway_he_s_per_slot"] = Fig(gw_s, "s/slot", "estimate",
                                         f"encrypt+decrypt ({he_gateway}), 7B measured, scaled by rows")
        f["gateway_he_devices"] = Fig(tps * gw_s, "4-vCPU units", "estimate", "slots/s x s/slot")
        f.update(he_e2e_rates(shape))
        for key, label in (("he_pair_overlapped_slots_per_s", "overlapped"),
                           ("he_pair_serial_slots_per_s", "as today")):
            if key in f:
                f[f"he_t4x2_pairs_{label.replace(' ', '_')}"] = Fig(
                    tps / f[key].value, "T4x2 VMs", "estimate", f"end-to-end encrypted factory, stages {label}")
    else:
        f["factory_gpus_plain"] = Fig(plain_flops / (g["ring_tops"] * 1e12), gpu, kind_gpu,
                                      "trusted factory holding the weights")

    # --- mask store ----------------------------------------------------------------------------------------------
    store_bytes = tps * mask * buffer_s
    f["mask_store_capacity"] = Fig(store_bytes / GB, "GB", "analytic", f"{buffer_s:g} s of supply buffered")
    f["mask_store_bandwidth"] = Fig(2 * tps * mask / GB, "GB/s", "analytic", "write + read")
    f["mask_store_nvme_drives"] = Fig(2 * tps * mask / (nvme_gb_per_s * GB), "drives", "estimate",
                                      f"{nvme_gb_per_s:g} GB/s each; or RAM ring buffer")

    units = {"gateways (NIC/batch)": gateways, "provider GPUs online": gpus_online,
             "provider GPUs for weights": gpus_mem, "mask-store NVMe": f["mask_store_nvme_drives"].value}
    if encrypted:
        units["provider GPUs maskgen"] = f["provider_gpus_maskgen"].value
        units["gateway HE units"] = f["gateway_he_devices"].value
    else:
        units["factory GPUs"] = f["factory_gpus_plain"].value
    rep.bottlenecks = sorted(units.items(), key=lambda kv: -kv[1])
    return rep


def markdown(reports: list[Report]) -> str:
    """One table, one column per report, for the plan document."""
    keys: list[str] = []
    for rp in reports:
        keys += [k for k in rp.figures if k not in keys]
    head = "| figure | " + " | ".join(f"{rp.model}{' (HE)' if rp.encrypted else ''}" for rp in reports) + " |"
    lines = [head, "|" + "---|" * (len(reports) + 1)]
    for k in keys:
        cells = []
        for rp in reports:
            fg = rp.figures.get(k)
            cells.append("" if fg is None else f"{fg.value:,.3g} {fg.unit}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=sorted(MODELS), default="llama-3-70b")
    p.add_argument("--all", action="store_true", help="every preset model")
    p.add_argument("--tokens-per-s", type=float, default=1000.0)
    p.add_argument("--ring", choices=sorted(RINGS), default="K32A12C27")
    p.add_argument("--gpu", choices=sorted(GPUS), default="t4")
    p.add_argument("--gpu-tops", type=float, default=None, help="override effective ring-TOPS per GPU")
    p.add_argument("--encrypted", action="store_true", help="encrypted mask supply (else trusted plain factory)")
    p.add_argument("--he-gateway", choices=sorted(HE_GATEWAY_7B), default="ntt",
                   help="customer-side HE implementation for --encrypted")
    p.add_argument("--rtt-ms", type=float, default=0.2)
    p.add_argument("--service-ms", type=float, default=0.5, help="service time per trip at the batch size")
    p.add_argument("--nic-gbit", type=float, default=100.0)
    p.add_argument("--batch", type=int, default=256, help="concurrent decode streams per gateway")
    p.add_argument("--buffer-s", type=float, default=600.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args(argv)
    if a.gpu_tops is not None:
        GPUS[a.gpu] = {**GPUS[a.gpu], "ring_tops": a.gpu_tops}
    names = sorted(MODELS) if a.all else [a.model]
    try:
        reps = [size(MODELS[n], tokens_per_s=a.tokens_per_s, ring=a.ring, gpu=a.gpu, encrypted=a.encrypted,
                     rtt_ms=a.rtt_ms, he_gateway=a.he_gateway, service_ms_per_trip=a.service_ms, nic_gbit=a.nic_gbit,
                     batch_per_gateway=a.batch, buffer_s=a.buffer_s) for n in names]
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps([rp.to_dict() for rp in reps], indent=1))
    elif a.markdown:
        print(markdown(reps))
    else:
        print("\n\n".join(rp.format() for rp in reps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
