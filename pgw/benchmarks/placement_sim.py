#!/usr/bin/env python3
"""Placement simulator: synthetic demand in several metros with diurnal curves, three placement policies, the
latency each user sees, tok/s per user, pods and $/hour (docs/operations/placement.md, "Simulation").

* demand: per metro, peak concurrent 7B streams (``METRO_PEAK``) on a weekday curve in local time (a sharp 09:00
  surge, business hours to 18:00 peaking at 14:00, an evening bump at 21:00, a 6 % night floor) with multiplicative
  noise; 48 h at 1-minute steps (the controller plans every minute);
* users: each metro's access mix (in-cloud servers, fibre, cable; pgw.placement.catalog) -> RTT to every data
  centre (distance model of docs/sdk/wan.md); a stream goes to the nearest data centre with a serving pod and spare
  capacity (what ``routing="signed"`` does), else the nearest serving one, overloaded;
* tok/s per stream: ``benchmarks/round_trip_model.sdk_ms_per_token`` (H100 provider, the client library's standard
  schedule, no speculation) at that RTT; a data centre above its capacity (``POD_STREAMS`` per pod) slows every
  stream there by capacity / load;
* cost: RunPod Secure Cloud price of an H100 (``deploy/runpod/launch.GPU_PRICE_H``) per pod-hour, a starting pod
  billed from its create (the cold start of ``pgw.placement.coldstart``, pre-staged pack) and serving only after it.

Policies: **single region** (US-KS-2, sized for the global peak), **static multi-region** (each metro's nearest data
centre, sized for that metro's own peak, always on), **dynamic** (``pgw.placement.policy.Controller``: one warm pod
per metro, gain per $ thresholds, idle timeout, anticipation one cold start ahead), and two ablations of it:
**dynamic-reactive** (no anticipation: sized for the demand now) and **dynamic-unstaged** (packs downloaded and
packed at boot: the longer cold start).

    python benchmarks/placement_sim.py [--hours 48] [--seed 1] [--json benchmarks/results/placement/sim.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "deploy" / "runpod"))
import round_trip_model as rtm  # noqa: E402

from pgw.placement.catalog import DATACENTERS, METROS, POD_STREAMS, rtt_ms  # noqa: E402
from pgw.placement.coldstart import estimate  # noqa: E402
from pgw.placement.forecast import Forecaster  # noqa: E402
from pgw.placement.policy import Controller, Ledger, PodState, PolicyConfig  # noqa: E402

MODEL = "7B"
GPU = "NVIDIA H100 80GB HBM3"
#: peak concurrent 7B streams per metro (synthetic)
METRO_PEAK = {"new-york": 40.0, "san-francisco": 30.0, "chicago": 12.0, "dallas": 10.0, "london": 35.0,
              "frankfurt": 25.0, "stockholm": 6.0, "tokyo": 20.0}
STEP_S = 60.0
#: logistic scale of the 09:00 surge (hours); 10 % -> 90 % takes 2 ln 9 x this (``--surge-min``)
SURGE_SCALE_H = 26.0 / 60.0 / (2 * math.log(9))
PACK = {"7B": "qwen2.5-7b-instruct", "32B": "qwen2.5-32b-instruct"}


def gpu_price() -> float:
    import launch

    return float(launch.GPU_PRICE_H[GPU][0])


def diurnal(local_h: float) -> float:
    """Fraction of the metro's peak at a local hour (weekday)."""
    start = 1.0 / (1.0 + math.exp(-(local_h - 9.0) / SURGE_SCALE_H))  # the 09:00 surge
    end = 1.0 / (1.0 + math.exp((local_h - 18.0) / 0.7))
    work = start * end * (0.8 + 0.2 * math.exp(-((local_h - 14.0) ** 2) / (2 * 2.0 ** 2)))
    evening = 0.35 * math.exp(-((local_h - 21.0) ** 2) / (2 * 1.2 ** 2))
    return min(1.0, 0.06 + 0.94 * max(work, evening))


def demand_at(t: float, rng: random.Random) -> dict[tuple[str, str], float]:
    out = {}
    for name, peak in METRO_PEAK.items():
        local = ((t / 3600.0) + METROS[name].utc_offset_h) % 24.0
        out[(name, MODEL)] = max(0.0, peak * diurnal(local) * rng.lognormvariate(0.0, 0.08))
    return out


class Tok:
    """Cached tok/s at an RTT (the benchmark model, rounded to 0.05 ms)."""

    def __init__(self) -> None:
        self.c: dict[float, float] = {}

    def __call__(self, rtt: float) -> float:
        k = round(rtt * 20) / 20
        if k not in self.c:
            self.c[k] = 1e3 / rtm.sdk_ms_per_token(MODEL, k)["ms_per_token"]
        return self.c[k]


def serve(demand: dict[tuple[str, str], float], serving: dict[str, int], tok: Tok) -> dict[str, Any]:
    """Route each metro's streams (per access class) to the nearest serving data centre with spare capacity, the
    overflow to the next; returns stream-weighted samples (rtt, tok/s, weight) and the load per data centre."""
    cap = {d: n * POD_STREAMS[MODEL] for d, n in serving.items() if n > 0}
    load: dict[str, float] = defaultdict(float)
    placed: list[tuple[str, float, str, float]] = []  # (metro/access, rtt, dc, streams)
    if not cap:
        return {"samples": [], "unserved": sum(demand.values()), "load": {}}
    for (metro, _m), streams in sorted(demand.items(), key=lambda kv: -kv[1]):
        m = METROS[metro]
        for access, share in m.access_mix.items():
            left = streams * share
            order = sorted(cap, key=lambda d: rtt_ms(m, DATACENTERS[d], access=access))
            for d in order:
                if left <= 1e-9:
                    break
                room = cap[d] - load[d]
                if room <= 0:
                    continue
                take = min(left, room)
                load[d] += take
                left -= take
                placed.append((f"{metro}/{access}", rtt_ms(m, DATACENTERS[d], access=access), d, take))
            if left > 1e-9:  # everything full: the nearest serving one, overloaded
                d = order[0]
                load[d] += left
                placed.append((f"{metro}/{access}", rtt_ms(m, DATACENTERS[d], access=access), d, left))
    samples = []
    for who, r, d, s in placed:
        slow = min(1.0, cap[d] / load[d]) if load[d] > 0 else 1.0
        metro, access = who.split("/")
        samples.append((r, tok(r) * slow, s, slow < 1.0, who, r - ideal_rtt(metro, access)))
    return {"samples": samples, "unserved": 0.0, "load": dict(load)}


_IDEAL: dict[tuple[str, str], float] = {}


def ideal_rtt(metro: str, access: str) -> float:
    """RTT to the metro's nearest data centre of the whole catalog (what a pod everywhere would give)."""
    k = (metro, access)
    if k not in _IDEAL:
        _IDEAL[k] = min(rtt_ms(METROS[metro], d, access=access) for d in DATACENTERS.values())
    return _IDEAL[k]


def wpct(vals: list[tuple[float, float]], q: float) -> float:
    vals = sorted(vals)
    tot = sum(w for _, w in vals)
    if tot <= 0:
        return float("nan")
    acc = 0.0
    for v, w in vals:
        acc += w
        if acc >= q * tot:
            return v
    return vals[-1][0]


def run(policy: str, hours: float, seed: int, cold_s: float, price: float) -> dict[str, Any]:
    rng = random.Random(seed)  # noqa: S311 - synthetic demand
    tok = Tok()
    steps = int(hours * 3600 / STEP_S)
    rtts: list[tuple[float, float]] = []
    excess: list[tuple[float, float]] = []  # RTT above the catalog's nearest data centre
    surge: list[tuple[float, float]] = []  # tok/s in each metro's first 15 minutes after 09:00 local
    surge_ex: list[tuple[float, float]] = []
    toks: list[tuple[float, float]] = []
    per_metro: dict[str, list[tuple[float, float]]] = defaultdict(list)
    pods_t: list[int] = []
    usd = 0.0
    overload_w = 0.0
    total_w = 0.0
    tokens = 0.0
    if policy == "single":
        need = math.ceil(sum(METRO_PEAK.values()) * 1.08 / (POD_STREAMS[MODEL] * 0.75))
        static = {"US-KS-2": need}
    elif policy == "static":
        static = defaultdict(int)
        for metro, peak in METRO_PEAK.items():
            m = METROS[metro]
            best = min(DATACENTERS, key=lambda d: sum(s * rtt_ms(m, DATACENTERS[d], access=a)
                                                      for a, s in m.access_mix.items()))
            static[best] += math.ceil(peak * 1.08 / (POD_STREAMS[MODEL] * 0.75))
        static = dict(static)
    else:
        reactive = policy == "dynamic-reactive"
        if policy == "dynamic-unstaged":
            cold_s = estimate(PACK[MODEL], prestaged=False, image_cached=False)["ready_s"]
        cfg = PolicyConfig(models=(MODEL,), warm={(m, MODEL): 1 for m in METRO_PEAK}, gain_up=2.0, gain_keep=1.0,
                           idle_timeout_s=1800, min_lifetime_s=900, max_usd_per_hour=200, max_usd_per_day=4000,
                           max_usd_total=1e6, cold_start_s=cold_s, max_pods_per_dc=16)
        ctl = Controller(cfg)
        fc = Forecaster()
        led = Ledger()
        fleet: list[PodState] = []
        n = 0
    # dynamic: a week of history for the profile (the controller would have it from the aggregates)
    if policy.startswith("dynamic"):
        hist = random.Random(seed + 99)  # noqa: S311 - synthetic demand
        for i in range(int(7 * 24 * 3600 / STEP_S)):
            t = -7 * 24 * 3600 + i * STEP_S
            if i % 60 == 0:
                fc.observe(t, demand_at(t, hist))
    for i in range(steps):
        t = i * STEP_S
        dem = demand_at(t, rng)
        if policy.startswith("dynamic"):
            if i % 60 == 0:  # the hourly-closed aggregates feed the profile
                fc.observe(t, dem)
            ant = dem if reactive else {k: max(dem[k], fc.anticipate(k, t, cold_s + STEP_S)) for k in dem}
            for p in fleet:
                if p.state == "starting" and t - p.started_at >= cold_s:
                    p.state = "serving"
            plan = ctl.plan(t, ant, fleet, ledger=led)
            for act, dc, _m, cnt in plan.actions:
                if act == "create":
                    for _ in range(cnt):
                        n += 1
                        fleet.append(PodState(f"p{n}", dc, MODEL, t, "starting"))
                else:
                    victims = sorted((p for p in fleet if p.dc == dc), key=lambda p: (p.state == "serving",
                                                                                         p.started_at))[:cnt]
                    fleet = [p for p in fleet if p not in victims]
            led.record(t, len(fleet) * price)
            serving: dict[str, int] = defaultdict(int)
            for p in fleet:
                if p.state == "serving":
                    serving[p.dc] += 1
            billed = len(fleet)
        else:
            serving = dict(static)
            billed = sum(static.values())
        res = serve(dem, dict(serving), tok)
        usd += billed * price * STEP_S / 3600.0
        pods_t.append(billed)
        for r, ts, w, over, who, ex in res["samples"]:
            rtts.append((r, w))
            excess.append((ex, w))
            if 9 <= ((t / 3600.0) + METROS[who.split("/")[0]].utc_offset_h) % 24.0 < 9.25:
                surge.append((ts, w))
                surge_ex.append((ex, w))
            toks.append((ts, w))
            per_metro[who.split("/")[0]].append((ts, w))
            total_w += w
            overload_w += w if over else 0.0
            tokens += ts * w * STEP_S
    tok_mean = sum(v * w for v, w in toks) / max(1e-9, sum(w for _, w in toks))
    return {"policy": policy, "hours": hours, "cold_start_s": cold_s, "pods_mean": sum(pods_t) / len(pods_t),
            "pods_max": max(pods_t),
            "usd_per_hour": usd / hours, "usd_total": usd,
            "rtt_ms": {"p50": wpct(rtts, 0.5), "p90": wpct(rtts, 0.9), "p99": wpct(rtts, 0.99)},
            "tok_s_per_user": {"mean": tok_mean, "p10": wpct(toks, 0.1), "p50": wpct(toks, 0.5)},
            "tok_s_by_metro": {m: round(sum(v * w for v, w in xs) / max(1e-9, sum(w for _, w in xs)), 2)
                               for m, xs in sorted(per_metro.items())},
            "overloaded_share": overload_w / max(1e-9, total_w),
            "excess_rtt_ms": {"mean": sum(v * w for v, w in excess) / max(1e-9, sum(w for _, w in excess)),
                              "p99": wpct(excess, 0.99)},
            "surge_tok_s": {"mean": sum(v * w for v, w in surge) / max(1e-9, sum(w for _, w in surge)),
                            "p10": wpct(surge, 0.1), "excess_rtt_ms_p90": wpct(surge_ex, 0.9)},
            "usd_per_million_tokens": usd / max(1e-9, tokens / 1e6), "stream_hours": total_w * STEP_S / 3600.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--model", choices=sorted(PACK), default="7B")
    ap.add_argument("--surge-min", type=float, default=26.0, help="minutes of the 09:00 surge from 10 %% to 90 %%")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every metro's peak streams")
    a = ap.parse_args()
    global MODEL, SURGE_SCALE_H, METRO_PEAK
    MODEL = a.model
    SURGE_SCALE_H = a.surge_min / 60.0 / (2 * math.log(9))
    METRO_PEAK = {k: v * a.scale for k, v in METRO_PEAK.items()}
    cs = estimate(PACK[MODEL], prestaged=True, image_cached=False)
    cold = cs["ready_s"]
    price = gpu_price()
    rows = [run(p, a.hours, a.seed, cold, price)
            for p in ("single", "static", "dynamic", "dynamic-reactive", "dynamic-unstaged")]
    print(f"{MODEL}, {a.hours:.0f} h, surge {a.surge_min:g} min, {len(METRO_PEAK)} metros, peak "
          f"{sum(METRO_PEAK.values()):.0f} streams (sum of metro peaks), H100 {price} $/h, cold start {cold:.0f} s "
          "(pre-staged pack, image not cached)\n")
    print("| policy | cold start s | pods mean / max | $/h | $ per 1M tokens | RTT p50 / p90 / p99 ms "
          "| tok/s per user mean / p10 | excess RTT mean / p99 ms | 09:00-09:15 tok/s mean / p10, excess RTT p90 "
          "| overloaded |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['policy']} | {r['cold_start_s']:.0f} | {r['pods_mean']:.1f} / {r['pods_max']} | "
              f"{r['usd_per_hour']:.1f} | "
              f"{r['usd_per_million_tokens']:.0f} | {r['rtt_ms']['p50']:.1f} / {r['rtt_ms']['p90']:.1f} / "
              f"{r['rtt_ms']['p99']:.1f} | {r['tok_s_per_user']['mean']:.2f} / {r['tok_s_per_user']['p10']:.2f} | "
              f"{r['excess_rtt_ms']['mean']:.2f} / {r['excess_rtt_ms']['p99']:.1f} | "
              f"{r['surge_tok_s']['mean']:.2f} / {r['surge_tok_s']['p10']:.2f}, "
              f"{r['surge_tok_s']['excess_rtt_ms_p90']:.1f} | {100 * r['overloaded_share']:.1f} % |")
    print("\ntok/s per user by metro:")
    for r in rows:
        print(f"  {r['policy']}: " + ", ".join(f"{m} {v}" for m, v in r["tok_s_by_metro"].items()))
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"model": MODEL, "surge_min": a.surge_min, "cold_start": cs, "price_h": price,
                       "metro_peak": METRO_PEAK, "results": rows}, f, indent=1)


if __name__ == "__main__":
    main()
