"""Modal demo: client-reported trip latency and the ``--max-trip-ms`` gate (``deploy/modal/trip_gate.py``).

``trip_gate`` has no Modal imports; it is checked against the gateway's real Prometheus exposition."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

MODAL_DIR = Path(__file__).resolve().parents[2] / "deploy" / "modal"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"pgw_modal_{name}", MODAL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load("trip_gate")


def _scrape(metrics: Any) -> str:
    from prometheus_client import generate_latest

    return generate_latest(metrics.registry).decode()


@pytest.fixture
def gw_metrics() -> Any:
    from pgw.gateway.metrics import GatewayMetrics

    m = GatewayMetrics()
    if not m.enabled:
        pytest.skip("prometheus_client not installed")
    return m


def _observe(m: Any, seconds: list[float], kind: str = "qkv") -> None:
    h = m.get(tg.TRIP_METRIC).labels(model="qwen2.5-0.5b-instruct", linear_kind=kind)
    for s in seconds:
        h.observe(s)


# ------------------------------------------------------------------------------------- trip_gate


def test_metric_name_and_bucket_bounds_match_the_gateway() -> None:
    from pgw.gateway.metrics import HOP_BUCKETS, HOP_METRICS

    assert tg.TRIP_METRIC in HOP_METRICS
    assert 0.001 in HOP_BUCKETS  # "p50 above 1 ms" is decided on a bucket bound, without interpolation


def test_parse_real_exposition_sums_label_sets(gw_metrics: Any) -> None:
    _observe(gw_metrics, [0.0004, 0.0009, 0.0012], "qkv")
    _observe(gw_metrics, [0.004], "down")
    h = tg.parse_histogram(_scrape(gw_metrics))
    assert h["count"] == 4
    assert h["sum"] == pytest.approx(0.0004 + 0.0009 + 0.0012 + 0.004)
    b = h["buckets"]
    assert b[0.0005] == 1 and b[0.001] == 2 and b[0.0015] == 3 and b[0.005] == 4 and b[math.inf] == 4


def test_absent_metric_gives_an_empty_histogram() -> None:
    h = tg.parse_histogram("# HELP x y\nother_metric 3\n")
    assert h == {"buckets": {}, "count": 0.0, "sum": 0.0}
    assert tg.quantile(h, 0.5) is None and tg.quantile_bucket(h, 0.5) is None


def test_delta_counts_only_the_window(gw_metrics: Any) -> None:
    _observe(gw_metrics, [0.02] * 50)  # earlier traffic (wizard, first chat): slow
    before = _scrape(gw_metrics)
    _observe(gw_metrics, [0.0006] * 90 + [0.0030] * 10)
    after = _scrape(gw_metrics)
    lat = tg.trip_latency(before, after)
    assert lat["trips"] == 100
    assert lat["p50_bucket_ms"] == [0.5, 0.75]
    assert 0.5 < lat["p50_ms"] <= 0.75
    assert lat["mean_ms"] == pytest.approx((90 * 0.6 + 10 * 3.0) / 100, abs=1e-3)
    assert tg.gate(lat, 1.0)[0] is True


def test_gate_on_the_1ms_bound(gw_metrics: Any) -> None:
    before = _scrape(gw_metrics)
    _observe(gw_metrics, [0.0009] * 50 + [0.0011] * 50)  # exactly half at or below 1 ms
    lat = tg.trip_latency(before, _scrape(gw_metrics))
    assert lat["p50_bucket_ms"] == [0.75, 1.0]
    assert tg.gate(lat, 1.0)[0] is True
    _observe(gw_metrics, [0.0011])  # one more above: the median moves above 1 ms
    lat = tg.trip_latency(before, _scrape(gw_metrics))
    assert lat["p50_bucket_ms"] == [1.0, 1.5]
    passed, why = tg.gate(lat, 1.0)
    assert passed is False and "> 1.0 ms" in why


def test_gate_tunnel_like_latency_fails(gw_metrics: Any) -> None:
    before = _scrape(gw_metrics)
    _observe(gw_metrics, [0.0068] * 96)  # run on 2026-11-20: 6.8 ms per round trip over the tunnel
    lat = tg.trip_latency(before, _scrape(gw_metrics))
    assert lat["p50_bucket_ms"] == [5.0, 10.0]
    assert tg.gate(lat, 1.0)[0] is False
    assert tg.gate(lat, 10.0)[0] is True
    assert tg.gate(lat, 7.0)[0] == (lat["p50_ms"] <= 7.0)  # limit inside the bucket: interpolated


def test_gate_open_ended_bucket_and_no_data() -> None:
    h_text = ('pgw_gateway_round_trip_seconds_bucket{le="2.5",linear_kind="qkv",model="m"} 0.0\n'
              'pgw_gateway_round_trip_seconds_bucket{le="+Inf",linear_kind="qkv",model="m"} 3.0\n'
              'pgw_gateway_round_trip_seconds_count{linear_kind="qkv",model="m"} 3.0\n'
              'pgw_gateway_round_trip_seconds_sum{linear_kind="qkv",model="m"} 9.0\n')
    lat = tg.trip_latency("pgw_gateway_round_trip_seconds_count 0\n", h_text)
    assert lat["trips"] == 3 and lat["p50_bucket_ms"] == [2500.0, None]
    assert tg.gate(lat, 1.0)[0] is False
    assert tg.gate(lat, 5000.0)[0] is False  # unbounded median never passes
    assert tg.gate({"trips": 0}, 1.0) == (None, "no round trips measured")
    assert tg.gate(None, 1.0)[0] is None
    # a count without buckets (not producible by the real exposition): unmeasured, not a failed gate
    assert tg.gate({"trips": 3, "p50_ms": None}, 1.0)[0] is None
    nob = tg.trip_latency("pgw_gateway_round_trip_seconds_count 0\n",
                          "pgw_gateway_round_trip_seconds_count 4\npgw_gateway_round_trip_seconds_sum 0.01\n")
    assert nob["trips"] == 4 and "p50_bucket_ms" not in nob and tg.gate(nob, 1.0)[0] is None
    assert tg.trip_latency("", h_text)["trips"] == 0  # a failed scrape is never a cumulative "delta"


def test_client_timing_rows_1_split() -> None:
    t0 = {"1": {"trips": 10, "rows": 10, "multi_item": 0, "rtt_s": 0.01, "queue_s": 0.0, "compute_s": 0.004,
                "wire_s": 0.006}}
    t1 = {"1": {"trips": 110, "rows": 110, "multi_item": 0, "rtt_s": 0.09, "queue_s": 0.001, "compute_s": 0.034,
                "wire_s": 0.046},
          "9+": {"trips": 4, "rows": 160, "multi_item": 0, "rtt_s": 0.04, "queue_s": 0, "compute_s": 0.02,
                 "wire_s": 0.02}}
    lat = tg.trip_latency("x 1\n", "x 1\n", t0, t1)
    assert lat["trips"] == 0  # no histogram samples in these texts
    assert lat["rows_1_trips"] == 100
    assert lat["rows_1_mean_ms"] == pytest.approx(0.8)
    assert lat["rows_1_service_ms"] == pytest.approx(0.31)
    assert lat["rows_1_wire_ms"] == pytest.approx(0.4)


def test_quantile_matches_prometheus_interpolation() -> None:
    h = {"buckets": {0.001: 2.0, 0.002: 6.0, math.inf: 8.0}, "count": 8.0, "sum": 0.0}
    # rank 4 of 8 lies in (1 ms, 2 ms], 2 of its 4 observations in: 1 + 1 * 2/4 = 1.5 ms
    assert tg.quantile(h, 0.5) == pytest.approx(0.0015)
    assert tg.quantile_bucket(h, 0.5) == (0.001, 0.002)
    assert tg.quantile(h, 0.99) == pytest.approx(0.002)  # +Inf bucket: last finite bound
    assert tg.quantile_bucket(h, 0.99) == (0.002, math.inf)
