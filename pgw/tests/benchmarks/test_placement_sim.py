"""The placement simulator (``benchmarks/placement_sim.py``): sanity of its demand, routing and policy outcomes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH))

import placement_sim as ps  # noqa: E402


def test_diurnal_curve() -> None:
    assert ps.diurnal(3.0) == pytest.approx(0.06, abs=0.01)  # night floor
    assert ps.diurnal(14.0) > 0.95 and ps.diurnal(10.0) > 0.7  # business hours
    assert 0.06 < ps.diurnal(21.0) < 0.5  # evening bump


def test_serve_prefers_nearest_then_spills() -> None:
    tok = ps.Tok()
    dem = {("london", "7B"): 20.0}
    one = ps.serve(dem, {"EU-FR-1": 1, "US-DE-1": 4}, tok)  # 16 streams near, the rest across the Atlantic
    near = sum(w for r, _t, w, _o, _who, _ex in one["samples"] if r < 20)
    assert near == pytest.approx(16.0) and one["load"]["US-DE-1"] == pytest.approx(4.0)
    full = ps.serve(dem, {"EU-FR-1": 1}, tok)  # nowhere to spill: overloaded, slowed by capacity / load
    assert all(o for _r, _t, _w, o, _who, _ex in full["samples"])
    assert ps.serve(dem, {}, tok)["unserved"] == pytest.approx(20.0)


@pytest.mark.timeout(300)
def test_policies_rank_as_expected() -> None:
    price = ps.gpu_price()
    rows = {p: ps.run(p, 12.0, 3, 167.0, price) for p in ("single", "static", "dynamic")}
    assert rows["single"]["rtt_ms"]["p50"] > 5 * rows["static"]["rtt_ms"]["p50"]
    assert rows["static"]["tok_s_per_user"]["mean"] > 5 * rows["single"]["tok_s_per_user"]["mean"]
    assert rows["dynamic"]["tok_s_per_user"]["mean"] >= 0.97 * rows["static"]["tok_s_per_user"]["mean"]
    assert rows["dynamic"]["usd_per_hour"] < rows["static"]["usd_per_hour"]
    assert rows["dynamic"]["overloaded_share"] < 0.01
    assert ps.run("dynamic", 12.0, 3, 167.0, price) == rows["dynamic"]  # deterministic for a seed


def test_sim_uses_the_launcher_prices_and_known_dcs() -> None:
    sys.path.insert(0, str(BENCH.parent / "deploy" / "runpod"))
    import jobs

    from pgw.placement.catalog import DATACENTERS

    assert ps.gpu_price() == 2.99
    assert {c["dc"] for c in jobs.WAN_CLIENTS if c["dc"]} <= set(DATACENTERS)
