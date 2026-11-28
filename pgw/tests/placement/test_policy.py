"""The placement controller's policy: warm set, scale-up by gain per $, capacity, scale-down with idle timeout and
hysteresis, spend caps, GPU-availability fallback; the forecaster; the predictor against the benchmark model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pgw.placement import predict
from pgw.placement.catalog import DATACENTERS, GPU_PRICE_H, METROS, POD_STREAMS
from pgw.placement.forecast import Forecaster
from pgw.placement.policy import Controller, Ledger, PodState, PolicyConfig

ROOT = Path(__file__).resolve().parents[2]
H = 3600.0
PRICE = GPU_PRICE_H["NVIDIA H100 80GB HBM3"]


def _pods(desired: dict[tuple[str, str], int], t: float = 0.0) -> list[PodState]:
    out = []
    for (dc, m), n in desired.items():
        out += [PodState(f"{dc}-{m}-{i}", dc, m, t, "serving") for i in range(n)]
    return out


def _on(plan: object) -> dict[tuple[str, str], int]:
    return {k: n for k, n in plan.desired.items() if n}  # type: ignore[attr-defined]


def test_warm_set_in_nearest_dc() -> None:
    c = Controller(PolicyConfig(warm={("new-york", "7B"): 1, ("london", "7B"): 2}))
    p = c.plan(0.0, {})
    assert _on(p) == {("US-DE-1", "7B"): 1, ("EU-FR-1", "7B"): 2}
    assert ("create", "EU-FR-1", "7B", 2) in p.actions and p.rate_usd_h == pytest.approx(3 * PRICE)


def test_scale_up_by_gain_per_dollar_and_capacity() -> None:
    c = Controller(PolicyConfig(warm={("new-york", "7B"): 1}))
    # a little London demand: not worth a European pod (London rides the US pod)
    p = c.plan(0.0, {("new-york", "7B"): 4, ("london", "7B"): 0.5})
    assert _on(p) == {("US-DE-1", "7B"): 1} and p.assignment[("london", "7B")] == "US-DE-1"
    # enough of it: a European data centre opens
    p = c.plan(0.0, {("new-york", "7B"): 4, ("london", "7B"): 10})
    assert _on(p) == {("US-DE-1", "7B"): 1, ("EU-FR-1", "7B"): 1}
    assert p.predicted[("london", "7B")]["tok_s"] > 10 * c.tok_s("london", "US-DE-1", "7B")
    # capacity: 40 streams at 16 x 0.75 per pod -> 4 pods
    p = c.plan(0.0, {("new-york", "7B"): 40})
    assert _on(p) == {("US-DE-1", "7B"): 4}


def test_capacity_capped_per_dc() -> None:
    c = Controller(PolicyConfig(warm={("new-york", "7B"): 1}, max_pods_per_dc=3, max_usd_per_hour=100))
    p = c.plan(0.0, {("new-york", "7B"): 100})
    assert p.desired[("US-DE-1", "7B")] == 3 and any("capped" in r for r in p.reasons)


def test_scale_down_after_idle_timeout_and_min_lifetime() -> None:
    cfg = PolicyConfig(warm={("new-york", "7B"): 1}, idle_timeout_s=1800, min_lifetime_s=900)
    c = Controller(cfg)
    p = c.plan(0.0, {("new-york", "7B"): 40})
    pods = _pods(_on(p), t=0.0)
    assert len(pods) == 4
    # demand gone at t = 600: kept (idle < 30 min)
    p = c.plan(600.0, {("new-york", "7B"): 2}, pods)
    assert p.desired[("US-DE-1", "7B")] == 4 and not p.actions
    # at t = 600 + 1800: scaled down to the need (1, the warm pod)
    p = c.plan(600.0 + 1800.0, {("new-york", "7B"): 2}, pods)
    assert p.desired[("US-DE-1", "7B")] == 1 and ("drain", "US-DE-1", "7B", 3) in p.actions
    # young pods (min lifetime) are not removed even when idle long enough
    c2 = Controller(cfg)
    c2.plan(0.0, {("new-york", "7B"): 40})
    young = _pods({("US-DE-1", "7B"): 4}, t=2000.0)
    c2.needed_at[("US-DE-1", "7B")] = 0.0
    p = c2.plan(2400.0, {("new-york", "7B"): 2}, young)
    assert p.desired[("US-DE-1", "7B")] == 4


def test_hysteresis_keep_lower_than_open() -> None:
    """A data centre whose gain per $ sits between gain_keep and gain_up stays open if it runs, and is not opened
    if it does not."""
    cfg = PolicyConfig(warm={("new-york", "7B"): 1}, gain_up=2.0, gain_keep=1.0, idle_timeout_s=0, min_lifetime_s=0)
    c = Controller(cfg)
    gain_per_stream = c.tok_s("london", "EU-FR-1", "7B") - c.tok_s("london", "US-DE-1", "7B")
    s = 1.5 * PRICE / gain_per_stream  # 1.5 stream-tok/s per $/h: between keep (1) and up (2)
    p = c.plan(0.0, {("new-york", "7B"): 4, ("london", "7B"): s})
    assert ("EU-FR-1", "7B") not in _on(p)
    running = _pods({("US-DE-1", "7B"): 1, ("EU-FR-1", "7B"): 1})
    p = c.plan(10.0, {("new-york", "7B"): 4, ("london", "7B"): s}, running)
    assert _on(p) == {("US-DE-1", "7B"): 1, ("EU-FR-1", "7B"): 1}
    p = c.plan(20.0, {("new-york", "7B"): 4, ("london", "7B"): 0.5 * PRICE / gain_per_stream}, running)
    assert ("drain", "EU-FR-1", "7B", 1) in p.actions  # below keep: closes


def test_spend_caps_hour_day_total() -> None:
    warm = {("new-york", "7B"): 1, ("london", "7B"): 1}
    c = Controller(PolicyConfig(warm=warm, max_usd_per_hour=2 * PRICE + 0.01))
    p = c.plan(0.0, {("new-york", "7B"): 40, ("london", "7B"): 1})
    assert p.rate_usd_h <= 2 * PRICE + 0.01 and sum(p.desired.values()) == 2
    assert _on(p) == {("US-DE-1", "7B"): 1, ("EU-FR-1", "7B"): 1}  # the warm set first
    assert p.refused
    # day cap: 390 $ spent in the last 24 h of a 400 $ day -> 10 $ / h left -> 3 pods
    led = Ledger()
    led.record(0.0, 390.0 / 24.0)
    c = Controller(PolicyConfig(warm=warm, max_usd_per_day=400.0, max_usd_per_hour=100))
    p = c.plan(24 * H, {("new-york", "7B"): 40}, ledger=led)
    assert sum(p.desired.values()) == 3
    # total cap reached: nothing at all
    c = Controller(PolicyConfig(warm=warm, max_usd_total=100.0))
    led = Ledger()
    led.total_before = 100.0
    assert sum(c.plan(0.0, {("new-york", "7B"): 4}, ledger=led).desired.values()) == 0
    # the cap also stops running pods (drain), never exceeded
    c = Controller(PolicyConfig(warm=warm, max_usd_per_hour=PRICE + 0.01))
    p = c.plan(0.0, {}, _pods({("US-DE-1", "7B"): 1, ("EU-FR-1", "7B"): 1}))
    assert sum(p.desired.values()) == 1 and any(a[0] == "drain" for a in p.actions)


def test_gpu_availability_fallback() -> None:
    c = Controller(PolicyConfig(warm={("new-york", "7B"): 1}, unavailable_hold_s=600))
    avail = {d: ["NVIDIA H100 80GB HBM3"] for d in DATACENTERS if d != "US-DE-1"}
    p = c.plan(0.0, {("new-york", "7B"): 4}, availability=avail)
    assert _on(p) == {("CA-MTL-3", "7B"): 1}  # the next-best data centre for New York
    # a create failure holds a data centre out for unavailable_hold_s
    c.mark_unavailable("CA-MTL-3", 100.0)
    p = c.plan(200.0, {("new-york", "7B"): 4}, availability=avail)
    assert ("CA-MTL-3", "7B") not in _on(p) and len(_on(p)) == 1
    assert ("CA-MTL-3", "7B") in _on(c.plan(800.0, {("new-york", "7B"): 4}, availability=avail))
    # a data centre without capacity keeps the pods it runs, gets no new ones
    c = Controller(PolicyConfig(warm={("new-york", "7B"): 1}, idle_timeout_s=0))
    p = c.plan(0.0, {("new-york", "7B"): 40}, _pods({("US-DE-1", "7B"): 2}), availability=avail)
    assert p.desired[("US-DE-1", "7B")] == 2
    assert sum(n for (d, _m), n in p.desired.items() if d != "US-DE-1") >= 1  # overflow elsewhere


def test_no_gpu_anywhere() -> None:
    c = Controller(PolicyConfig(warm={("new-york", "7B"): 1}))
    p = c.plan(0.0, {("new-york", "7B"): 4}, availability={})
    assert not _on(p) and any("no data centre" in r for r in p.reasons)


def test_forecaster_anticipates_the_morning() -> None:
    f = Forecaster()
    key = ("london", "7B")
    for day in range(7):  # a week of a daily curve: 2 streams at night, 20 from 08:00
        for h in range(24):
            f.observe((day * 24 + h) * H, {key: 20.0 if 8 <= h < 18 else 2.0})
    t = 7 * 24 * H + 7 * H + 50 * 60  # 07:50 on day 8, cold start 10 min ahead of 08:00 -> 20
    f.observe(t, {key: 2.0})
    assert f.anticipate(key, t, 600) > 15
    assert f.anticipate(key, t - 3 * H, 600) < 5  # 04:50: nothing coming
    # trend: demand doubled -> the prediction follows
    for h in range(24):
        f.observe((8 * 24 + h) * H, {key: 40.0 if 8 <= h < 18 else 4.0})
    assert f.predict(key, 9 * 24 * H + 9 * H) > 25


def test_predictor_matches_benchmark_model() -> None:
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import round_trip_model as rtm

    for m in ("7B", "32B", "72B"):
        for rtt in (0.0, 0.2, 1.0, 5.0, 20.0):
            want = rtm.sdk_ms_per_token(m, rtt)["ms_per_token"]
            assert predict.ms_per_token(m, rtt) == pytest.approx(want, rel=0.002), (m, rtt)
    assert predict.model_key("qwen2.5-7b-instruct") == "7B" and predict.model_key("qwen2.5-32b-instruct") == "32B"
    # the link term (wan_model bytes per row)
    import wan_model

    w = wan_model.pass_ms("7B", 2.0, 1000, 1000, o_ms=0.0)
    assert predict.MODELS["7B"].up_mb == pytest.approx(w["up_mb"], rel=0.01)
    assert predict.MODELS["7B"].down_mb == pytest.approx(w["down_mb"], rel=0.01)
    assert predict.tok_s("7B", 2.0, link_mbit=(1000, 1000)) < predict.tok_s("7B", 2.0)


def test_catalog_prices_match_launcher() -> None:
    sys.path.insert(0, str(ROOT / "deploy" / "runpod"))
    import launch

    assert {k: v[0] for k, v in launch.GPU_PRICE_H.items()} == GPU_PRICE_H
    assert set(POD_STREAMS) == set(predict.MODELS)
    assert set(METROS) and all(d.id == k for k, d in DATACENTERS.items())
