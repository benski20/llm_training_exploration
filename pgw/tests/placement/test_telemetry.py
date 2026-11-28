"""Content-free demand reports: strict schema, server-side coarse region, aggregation before storing, small-count
suppression, retention."""

from __future__ import annotations

import pytest

from pgw.placement.telemetry import Aggregator, RegionResolver, ReportError, demand_by_metro, validate_report

MODELS = ("qwen2.5-7b-instruct",)
OK = {"v": 1, "model": "qwen2.5-7b-instruct", "endpoint": "eu-nl", "rtt_bucket": "<=5", "peak_streams": 2}


@pytest.mark.parametrize("bad", [
    {**OK, "prompt": "hello"},
    {**OK, "client_ip": "1.2.3.4"},
    {**OK, "ts": 1.0},
    {**OK, "model": "secret-model"},
    {**OK, "rtt_bucket": "4.73"},
    {**OK, "endpoint": "EU NL / injected"},
    {**OK, "peak_streams": 10_000},
    {**OK, "peak_streams": True},
    {k: v for k, v in OK.items() if k != "rtt_bucket"},
    {**OK, "v": 2},
    "not an object",
])
def test_schema_is_closed(bad: object) -> None:
    with pytest.raises(ReportError):
        validate_report(bad, models=MODELS)


def test_resolver_is_coarse_and_unknown_safe() -> None:
    r = RegionResolver({"198.51.100.0/22": "london", "2001:db8::/32": "frankfurt", "198.51.100.0/24": "london-x"})
    assert r("198.51.100.7") == "london-x"  # most specific prefix first
    assert r("198.51.101.7") == "london" and r("2001:db8::1") == "frankfurt"
    assert r("203.0.113.1") == "unknown" and r(None) == "unknown" and r("not-an-ip") == "unknown"
    with pytest.raises(ValueError):
        RegionResolver({"10.0.0.0/8": "Bad Label!"})


def test_aggregation_counts_installations_once_and_stores_no_identifiers() -> None:
    t = [3600.0 * 1000 + 10]
    agg = Aggregator(models=MODELS, resolver=RegionResolver({"198.51.100.0/24": "london"}), k_min=3,
                     clock=lambda: t[0])
    for i in range(5):
        for _ in range(3):  # repeated reports of one installation count once
            agg.add(OK, tenant=f"sdk.{i:016x}", source_ip=f"198.51.100.{i + 1}")
    agg.add(OK, tenant="sdk.lonely", source_ip="203.0.113.5")  # one installation in an unknown region
    assert agg.export() == []  # the open hour is never exported
    t[0] += 3600
    rows = agg.export()
    assert rows == [{"hour": 3600 * 1000, "metro": "london", "model": "qwen2.5-7b-instruct", "endpoint": "eu-nl",
                     "rtt_bucket": "<=5", "installations": 5, "peak_streams": 10}]
    # nothing identifying is kept: no address, no tenant, no salt of a closed hour
    blob = repr(agg.__dict__)
    assert "198.51.100" not in blob and "sdk.0" not in blob and "203.0.113" not in blob
    assert demand_by_metro(rows) == {("london", "qwen2.5-7b-instruct"): 10.0}


def test_small_counts_folded_then_dropped_and_retention() -> None:
    t = [0.0]
    agg = Aggregator(models=MODELS, resolver=RegionResolver({"198.51.100.0/24": "london", "192.0.2.0/24": "paris"}),
                     k_min=3, retention_days=1, clock=lambda: t[0])
    agg.add(OK, tenant="a", source_ip="198.51.100.1")
    agg.add(OK, tenant="b", source_ip="192.0.2.1")
    t[0] = 3600.0
    assert agg.export() == []  # 1 + 1 installations: below k even folded into "other"
    for i in range(3):
        agg.add(OK, tenant=f"c{i}", source_ip="198.51.100.9")
    t[0] = 7200.0
    assert [r["installations"] for r in agg.export()] == [3]
    t[0] = 3 * 86400.0
    agg.flush()
    assert agg.export() == []  # past retention: deleted
