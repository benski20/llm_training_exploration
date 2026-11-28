"""The placement controller's policy: desired GPU pods per (data centre, model) from the anticipated demand.

One :meth:`Controller.plan` call (every few minutes):

1. **Candidates.** Data centres listing the GPU type (the live catalog's availability; a data centre whose last
   create failed for capacity is held out for ``unavailable_hold_s``). A data centre without capacity keeps the
   pods it has but gets no new ones; its metros fall back to the next-best data centre.
2. **Warm set.** Each configured metro keeps ``warm[(metro, model)]`` pods in its nearest available data centre
   (the floor: the first users of the day do not wait for a cold start).
3. **Gain per $.** Every user stream goes to the nearest data centre in the serving set. A data centre joins the
   set when the modelled gain it brings (sum over the streams that would move of their tok/s gain,
   :mod:`pgw.placement.predict`) per $/h of the pods it needs is at least ``gain_up``; one already serving stays
   while its gain per $ is at least ``gain_keep`` (< ``gain_up``: hysteresis, no flapping between two decisions).
4. **Capacity.** Pods per data centre = ceil(streams assigned / (``POD_STREAMS`` x ``target_util``)), at most
   ``max_pods_per_dc``; overflow goes to the next-nearest serving data centre (or opens one).
5. **Scale-down.** A count may fall only after it has been above what is needed for ``idle_timeout_s`` and every
   pod being removed is older than ``min_lifetime_s``. Removed pods are drained first (the directory marks them
   ``draining``; clients move between requests), then terminated.
6. **Spend caps.** The plan's hourly rate fits ``max_usd_per_hour``; with what has been spent (the ledger), one more
   hour at that rate fits ``max_usd_per_day`` and ``max_usd_total``. Pods are admitted in priority order: warm
   set, then pods already running, then capacity, then gain; what does not fit is left out (and logged).

Demand is anticipated one cold start ahead (:class:`pgw.placement.forecast.Forecaster`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pgw.placement import predict
from pgw.placement.catalog import (
    DATACENTERS,
    DEFAULT_GPU,
    GPU_PRICE_H,
    METROS,
    POD_STREAMS,
    DataCenter,
    Metro,
    rtt_ms,
)

__all__ = ["Controller", "Ledger", "Plan", "PodState", "PolicyConfig"]


@dataclass
class PolicyConfig:
    models: tuple[str, ...] = ("7B",)
    #: (metro, model) -> pods kept warm in the metro's nearest available data centre
    warm: Mapping[tuple[str, str], int] = field(default_factory=dict)
    gpu: str = DEFAULT_GPU
    streams_per_pod: Mapping[str, int] = field(default_factory=lambda: dict(POD_STREAMS))
    target_util: float = 0.75
    #: stream-tok/s gained per $/h to open a data centre / to keep one open
    gain_up: float = 2.0
    gain_keep: float = 1.0
    idle_timeout_s: float = 1800.0
    min_lifetime_s: float = 900.0
    max_pods_per_dc: int = 8
    max_usd_per_hour: float = 30.0
    max_usd_per_day: float = 400.0
    max_usd_total: float = 2000.0
    cold_start_s: float = 300.0
    unavailable_hold_s: float = 1800.0
    #: candidate data centres (default: the catalog)
    datacenters: tuple[str, ...] | None = None

    def price_h(self, prices: Mapping[str, float] | None = None) -> float:
        p = (prices or {}).get(self.gpu) or GPU_PRICE_H.get(self.gpu)
        if not p:
            raise ValueError(f"no price for {self.gpu!r}")
        return float(p)


@dataclass
class PodState:
    """A pod as the controller sees it."""

    id: str
    dc: str
    model: str
    started_at: float
    state: str = "starting"  # starting | serving | draining


class Ledger:
    """Spend so far: closed pod-hours plus the running pods' accrual (USD). ``record`` at each tick."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []  # (t, rate $/h from t on)
        self.total_before = 0.0

    def record(self, t: float, rate_h: float) -> None:
        self.samples.append((t, rate_h))

    def spent(self, t0: float, t1: float) -> float:
        """USD between ``t0`` and ``t1``."""
        out = 0.0
        for i, (t, r) in enumerate(self.samples):
            nxt = self.samples[i + 1][0] if i + 1 < len(self.samples) else t1
            a, b = max(t, t0), min(nxt, t1)
            if b > a:
                out += r * (b - a) / 3600.0
        return out

    def total(self, now: float) -> float:
        return self.total_before + (self.spent(self.samples[0][0], now) if self.samples else 0.0)


@dataclass
class Plan:
    desired: dict[tuple[str, str], int]
    rate_usd_h: float
    assignment: dict[tuple[str, str], str]  # (metro, model) -> dc
    predicted: dict[tuple[str, str], dict[str, float]]  # (metro, model) -> {rtt_ms, tok_s, streams}
    reasons: list[str]
    refused: list[str]
    actions: list[tuple[str, str, str, int]] = field(default_factory=list)  # (create | drain, dc, model, n)

    def public(self) -> dict[str, Any]:
        return {"desired": {f"{d}/{m}": n for (d, m), n in sorted(self.desired.items()) if n},
                "rate_usd_h": round(self.rate_usd_h, 3),
                "assignment": {f"{me}/{m}": d for (me, m), d in sorted(self.assignment.items())},
                "predicted": {f"{me}/{m}": {k: round(v, 3) for k, v in p.items()}
                              for (me, m), p in sorted(self.predicted.items())},
                "actions": [list(a) for a in self.actions], "reasons": self.reasons, "refused": self.refused}


class Controller:
    """See the module docstring. ``metros`` / ``datacenters``: the geography (defaults: the catalog)."""

    def __init__(self, cfg: PolicyConfig, *, metros: Mapping[str, Metro] | None = None,
                 datacenters: Mapping[str, DataCenter] | None = None, prices: Mapping[str, float] | None = None):
        self.cfg = cfg
        self.metros = dict(metros or METROS)
        dcs = dict(datacenters or DATACENTERS)
        if cfg.datacenters is not None:
            dcs = {k: v for k, v in dcs.items() if k in cfg.datacenters}
        self.dcs = dcs
        self.price = cfg.price_h(prices)
        self.unavailable: dict[str, float] = {}
        #: (dc, model) -> last time the pods there were all needed (scale-down idle timer)
        self.needed_at: dict[tuple[str, str], float] = {}
        self._rtt: dict[tuple[str, str, str], float] = {}

    # ---- model ------------------------------------------------------------------------------------------------
    def tok_s(self, metro: str, dc: str, model: str) -> float:
        """Mix-weighted modelled tok/s of one stream of ``metro`` served from ``dc``."""
        m = self.metros[metro]
        out = 0.0
        for access, share in m.access_mix.items():
            k = (metro, dc, access)
            if k not in self._rtt:
                self._rtt[k] = rtt_ms(m, self.dcs[dc], access=access)
            out += share * predict.tok_s(model, self._rtt[k])
        return out

    def rtt(self, metro: str, dc: str) -> float:
        m = self.metros[metro]
        return sum(share * rtt_ms(m, self.dcs[dc], access=a) for a, share in m.access_mix.items())

    def mark_unavailable(self, dc: str, now: float) -> None:
        """A create in ``dc`` failed for capacity: no new pods there for ``unavailable_hold_s``."""
        self.unavailable[dc] = now

    def _available(self, now: float, availability: Mapping[str, Iterable[str]] | None) -> set[str]:
        out = set()
        for d, dc in self.dcs.items():
            gpus = set(availability[d]) if availability is not None and d in availability else (
                set() if availability is not None else set(dc.gpus))
            if self.cfg.gpu in gpus and now - self.unavailable.get(d, -1e18) >= self.cfg.unavailable_hold_s:
                out.add(d)
        return out

    def _cap(self, model: str) -> float:
        per = self.cfg.streams_per_pod.get(model) or self.cfg.streams_per_pod.get(predict.model_key(model), 1)
        return max(1e-9, per * self.cfg.target_util)

    # ---- plan ---------------------------------------------------------------------------------------------------
    def plan(self, now: float, demand: Mapping[tuple[str, str], float], pods: Iterable[PodState] = (), *,
             ledger: Ledger | None = None, availability: Mapping[str, Iterable[str]] | None = None) -> Plan:
        """``demand``: (metro, model) -> anticipated peak concurrent streams. ``pods``: the fleet now."""
        cfg = self.cfg
        pods = list(pods)
        cur: dict[tuple[str, str], int] = defaultdict(int)
        for p in pods:
            if p.state != "draining":
                cur[(p.dc, p.model)] += 1
        avail = self._available(now, availability)
        reasons: list[str] = []
        raw: dict[tuple[str, str], int] = {}
        prio: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)  # per pod slot: (class, value)
        assignment: dict[tuple[str, str], str] = {}
        for model in cfg.models:
            self._plan_model(model, demand, cur, avail, reasons, raw, prio, assignment)
        # scale-down hysteresis: a count falls only after idle_timeout and min_lifetime
        desired: dict[tuple[str, str], int] = {}
        for key in set(raw) | set(cur):
            want, have = raw.get(key, 0), cur.get(key, 0)
            if want >= have:
                self.needed_at[key] = now
                desired[key] = want
                continue
            idle = now - self.needed_at.setdefault(key, now)
            young = sorted((p for p in pods if (p.dc, p.model) == key and p.state != "draining"),
                           key=lambda p: p.started_at)
            removable = [p for p in young if now - p.started_at >= cfg.min_lifetime_s]
            if idle < cfg.idle_timeout_s:
                desired[key] = have
                if want < have:
                    reasons.append(f"{key[0]}/{key[1]}: {have - want} pod(s) idle {idle:.0f}s < "
                                   f"{cfg.idle_timeout_s:.0f}s, kept")
                for _ in range(len(prio[key]), have):
                    prio[key].append((1, 0.0))
            else:
                desired[key] = max(want, have - len(removable))
                for _ in range(len(prio[key]), desired[key]):
                    prio[key].append((1, 0.0))
        # spend caps: admit pod slots in priority order
        led = ledger or Ledger()
        spent_day = led.spent(now - 86400.0, now)
        spent_total = led.total(now)
        hour_cap = min(cfg.max_usd_per_hour, max(0.0, cfg.max_usd_per_day - spent_day),
                       max(0.0, cfg.max_usd_total - spent_total))
        slots = sorted(((c, v, k, i) for k, lst in prio.items() for i, (c, v) in enumerate(lst[:desired.get(k, 0)])),
                       key=lambda t: (t[0], t[1], t[2], t[3]))
        final: dict[tuple[str, str], int] = defaultdict(int)
        refused: list[str] = []
        rate = 0.0
        for c, _v, k, _i in slots:
            if rate + self.price <= hour_cap + 1e-9:
                final[k] += 1
                rate += self.price
            else:
                refused.append(f"{k[0]}/{k[1]}: pod (class {c}) over the spend cap ({hour_cap:.2f} $/h left)")
        out = {k: final.get(k, 0) for k in set(desired) | set(cur)}
        actions = []
        for k in sorted(out):
            diff = out[k] - cur.get(k, 0)
            if diff > 0:
                actions.append(("create", k[0], k[1], diff))
            elif diff < 0:
                actions.append(("drain", k[0], k[1], -diff))
        predicted = {}
        for (me, model), d in assignment.items():
            if out.get((d, model), 0) > 0:
                predicted[(me, model)] = {"rtt_ms": self.rtt(me, d), "tok_s": self.tok_s(me, d, model),
                                          "streams": float(demand.get((me, model), 0.0))}
        return Plan(out, rate, assignment, predicted, reasons, refused, actions)


    def _plan_model(self, model: str, demand: Mapping[tuple[str, str], float], cur: Mapping[tuple[str, str], int],
                    avail: set[str], reasons: list[str], raw: dict[tuple[str, str], int],
                    prio: dict[tuple[str, str], list[tuple[int, float]]],
                    assignment: dict[tuple[str, str], str]) -> None:
        """Steps 1-4 of the module docstring for one model (fills ``raw``, ``prio``, ``assignment``)."""
        cfg = self.cfg
        metros = sorted({me for (me, mo) in demand if mo == model} | {me for (me, mo) in cfg.warm if mo == model})
        metros = [me for me in metros if me in self.metros]
        running = {d for (d, mo), n in cur.items() if mo == model and n > 0 and d in self.dcs}
        usable = avail | running  # a data centre without capacity keeps its pods
        if not usable:
            reasons.append(f"{model}: no data centre with {cfg.gpu} available")
            return
        serving: set[str] = set(running)
        warm_dc: dict[str, int] = defaultdict(int)
        for me in metros:
            w = int(cfg.warm.get((me, model), 0))
            if w <= 0:
                continue
            best = max(sorted(avail or usable), key=lambda d: self.tok_s(me, d, model))
            warm_dc[best] = max(warm_dc[best], w)
            serving.add(best)
        streams = {me: float(demand.get((me, model), 0.0)) for me in metros}

        def assign(sv: set[str]) -> dict[str, str]:
            return {me: max(sorted(sv), key=lambda d: self.tok_s(me, d, model)) for me in metros} if sv else {}

        def value(sv: set[str]) -> float:
            a = assign(sv)
            return sum(streams[me] * self.tok_s(me, a[me], model) for me in metros if me in a)

        def pods_for(sv: set[str]) -> int:
            a = assign(sv)
            load: dict[str, float] = defaultdict(float)
            for me, d in a.items():
                load[d] += streams[me]
            return sum(max(warm_dc.get(d, 0), math.ceil(load[d] / self._cap(model)) if load[d] > 0 else 0,
                           1 if d in sv else 0) for d in sv)

        if not serving and any(v > 0 for v in streams.values()):  # demand and nothing serving: the best one
            best = max(sorted(avail), key=lambda d: value({d})) if avail else None
            if best is not None:
                serving.add(best)
                reasons.append(f"{model}: opened {best} (demand, nothing serving)")
        # drop serving data centres whose gain per $ fell below gain_keep (never a warm one)
        changed = True
        while changed:
            changed = False
            for d in sorted(serving - set(warm_dc)):
                rest = serving - {d}
                if not rest:
                    continue
                gain = value(serving) - value(rest)
                dpods = max(1, pods_for(serving) - pods_for(rest))
                if gain / (dpods * self.price) < cfg.gain_keep:
                    serving = rest
                    changed = True
                    reasons.append(f"{model}: {d} below keep threshold ({gain / (dpods * self.price):.2f} "
                                   "stream-tok/s per $/h)")
                    break
        # add data centres while one brings at least gain_up per $
        while True:
            best_d, best_r = None, cfg.gain_up
            base_v, base_p = value(serving), pods_for(serving)
            for d in sorted(avail - serving):
                sv = serving | {d}
                r = (value(sv) - base_v) / (max(1, pods_for(sv) - base_p) * self.price)
                if r >= best_r:
                    best_d, best_r = d, r
            if best_d is None:
                break
            serving.add(best_d)
            reasons.append(f"{model}: open {best_d} ({best_r:.2f} stream-tok/s per $/h >= {cfg.gain_up})")
        a = assign(serving)
        load = defaultdict(float)
        for me, d in a.items():
            load[d] += streams[me]
            assignment[(me, model)] = d
        for d in sorted(serving):
            need = math.ceil(load[d] / self._cap(model)) if load[d] > 0 else 0
            n = max(warm_dc.get(d, 0), need, 1)
            if d not in avail:  # no new capacity there: keep what runs
                n = min(n, cur.get((d, model), 0))
            n = min(n, cfg.max_pods_per_dc)
            if need > n:
                reasons.append(f"{model}: {d} needs {need} pods, capped at {n} (availability / max_pods_per_dc)")
            raw[(d, model)] = n
            gain_each = load[d] * self.tok_s(next((me for me, dd in a.items() if dd == d), metros[0]), d, model) \
                if metros else 0.0
            for i in range(n):
                cls = 0 if i < warm_dc.get(d, 0) else (1 if i < cur.get((d, model), 0) else
                                                       (2 if i < need else 3))
                prio[(d, model)].append((cls, -gain_each))
