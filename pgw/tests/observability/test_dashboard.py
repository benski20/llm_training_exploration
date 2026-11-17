"""Tests for the self-contained ops dashboard (SPEC 11.4)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from starlette.applications import Starlette

from pgw.observability import DASHBOARD_CSP, PGWMetrics, dashboard_html, mount_dashboard
from pgw.observability.dashboard import core_js

NODE = shutil.which("node")


def test_page_is_self_contained() -> None:
    page = dashboard_html("PGW gateway")
    assert page.startswith("<!doctype html>")
    assert "<title>PGW gateway</title>" in page
    # No external assets: no <link>, no src= attributes, and the only absolute URL is the SVG namespace.
    assert "<link" not in page
    assert not re.search(r"\ssrc\s*=", page)
    urls = set(re.findall(r"https?://[^\s\"')]+", page))
    assert urls <= {"http://www.w3.org/2000/svg"}
    assert "@import" not in page
    for needle in (
        "Masks remaining",
        "Burn rate",
        "tokens/s",
        "Time to first token",
        "Freivalds",
        "Outbound-noise monitor",
        "prefers-color-scheme",
    ):
        assert needle in page
    # The admin token lives in sessionStorage only.
    assert "localStorage" not in page
    assert "innerHTML" not in page


def test_title_and_config_are_escaped() -> None:
    page = dashboard_html('<script>alert("x")</script>', metrics_url="m</script><b>", status_url=None)
    assert "<script>alert" not in page
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in page
    assert "m</script>" not in page
    assert "m\\u003c/script\\u003e\\u003cb\\u003e" in page
    assert '"statusUrl":null' in page


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/metrics",
        "//evil.example/m",
        "/\\evil.example/m",  # browsers treat backslash as slash: protocol-relative
        "\\\\evil.example/m",
        "/\t/evil.example/m",  # tabs/newlines are stripped by the URL parser
        "javascript:alert(1)",
        "data:text/plain,x",
        "",
    ],
)
def test_rejects_cross_origin_urls(url: str) -> None:
    with pytest.raises(ValueError):
        dashboard_html(metrics_url=url)
    with pytest.raises(ValueError):
        dashboard_html(status_url=url)


def test_rejects_bad_refresh() -> None:
    with pytest.raises(ValueError):
        dashboard_html(refresh_seconds=0)


def test_mount_on_fastapi() -> None:
    app = FastAPI()
    mount_dashboard(app, title="Ops")
    with TestClient(app) as client:
        r = client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["content-security-policy"] == DASHBOARD_CSP
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-frame-options"] == "DENY"
    assert "<title>Ops</title>" in r.text
    assert '"metricsUrl":"metrics"' in r.text
    assert "/dashboard" not in app.openapi()["paths"]


def test_mount_on_starlette_custom_path() -> None:
    app = Starlette()
    mount_dashboard(app, "/ops/dash", metrics_url="../metrics", status_url=None, refresh_seconds=2)
    with TestClient(app) as client:
        r = client.get("/ops/dash")
        assert r.status_code == 200
        assert '"metricsUrl":"../metrics"' in r.text and '"refreshSeconds":2.0' in r.text
        assert client.post("/ops/dash").status_code == 405
    with pytest.raises(ValueError):
        mount_dashboard(app, "dashboard")


def test_gateway_fallback_hook_uses_this_page() -> None:
    """The gateway app imports ``dashboard_html`` and falls back only on NotImplementedError."""
    from pgw.gateway import app as gw_app

    html = gw_app._dashboard_html()
    assert "PGWCore" in html and "PGW gateway" in html


# ---- JavaScript core (Node) ----------------------------------------------------------------------
def _scrape(fill: Any) -> str:
    reg = CollectorRegistry()
    pm = PGWMetrics(reg)
    fill(pm)
    return pm.render()[0].decode()


def _fill(scale: int) -> Any:
    def fill(pm: PGWMetrics) -> None:
        g = pm.get
        g("pgw_masks_available_slots").labels(fingerprint="abc", ring="K40A12").set(10_000 - 100 * scale)
        g("pgw_masks_available_slots").labels(fingerprint="def", ring="K40A12").set(5_000)
        g("pgw_masks_consumed_slots_total").labels(fingerprint="abc", ring="K40A12").inc(100 * scale)
        g("pgw_masks_low_water").labels(fingerprint="abc", ring="K40A12").set(0)
        g("pgw_gateway_tokens_total").labels(model="qwen", kind="completion").inc(50 * scale)
        g("pgw_gateway_tokens_total").labels(model="qwen", kind="prompt").inc(200 * scale)
        g("pgw_gateway_requests_total").labels(model="qwen", endpoint="chat", status="200").inc(10 * scale)
        g("pgw_gateway_requests_total").labels(model="qwen", endpoint="chat", status="502").inc(scale)
        g("pgw_gateway_active_sessions").labels(model="qwen").set(3)
        g("pgw_gateway_integrity_failures_total").labels(model="qwen", check="freivalds").inc(scale - 1)
        g("pgw_gateway_noise_alarm").labels(model="qwen").set(1 if scale > 1 else 0)
        g("pgw_gateway_noise_alarm").labels(model='we"ird\\name').set(0)
        ttft = g("pgw_gateway_ttft_seconds").labels(model="qwen")
        for _ in range(10 * scale):
            ttft.observe(0.3)
        tpot = g("pgw_gateway_tpot_seconds").labels(model="qwen")
        for i in range(100 * scale):
            tpot.observe(0.04 if i % 10 else 2.0)
        g("pgw_service_inflight").set(2)
        g("pgw_service_requests_total").labels(tenant="t", code="ok").inc(30 * scale)

    return fill


def _run_node(tmp_path: Path, body: str) -> Any:
    core = tmp_path / "core.js"
    core.write_text(core_js())
    script = tmp_path / "t.js"
    script.write_text(f"const C = require({json.dumps(str(core))});\n{body}\n")
    out = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=60, check=False)  # type: ignore[list-item]  # noqa: S603
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_parse_and_summarize(tmp_path: Path) -> None:
    prev, cur = _scrape(_fill(1)), _scrape(_fill(2))
    res = _run_node(
        tmp_path,
        f"""
const prev = C.parse({json.dumps(prev)});
const cur = C.parse({json.dumps(cur)});
const s = C.summarize(prev, cur, 10);
const first = C.summarize(null, prev, 0);
const lbl = cur.pgw_gateway_noise_alarm.map(e => e.labels.model);
console.log(JSON.stringify({{s, first, lbl}}));
""",
    )
    s, first = res["s"], res["first"]
    assert 'we"ird\\name' in res["lbl"]
    mk = s["masks"]
    assert mk["available"] == 10_000 - 200 + 5_000
    assert mk["burnRate"] == pytest.approx(10.0)  # 100 slots in 10 s
    assert mk["secondsLeft"] == pytest.approx(14_800 / 10.0)
    assert mk["lowWater"] is False
    gw = s["gateway"]
    assert gw["present"] is True
    assert gw["completionTps"] == pytest.approx(5.0)
    assert gw["promptTps"] == pytest.approx(20.0)
    assert gw["requestRate"] == pytest.approx(1.1)
    assert gw["errorsTotal"] == 2 and gw["requestsTotal"] == 22
    assert gw["activeSessions"] == 3
    assert mk["slotsPerToken"] == pytest.approx(2.0)
    # Interval histogram: 10 new TTFT samples of 0.3 s -> inside the (0.25, 0.5] bucket.
    assert gw["ttft"]["interval"] is True and gw["ttft"]["count"] == 10
    assert gw["ttft"]["mean"] == pytest.approx(0.3)
    assert 0.25 <= gw["ttft"]["p50"] <= 0.5
    # TPOT: 90% at 40 ms, 10% at 2 s -> p50 in (0.025, 0.05], p95 in (1, 2.5].
    assert 0.025 <= gw["tpot"]["p50"] <= 0.05
    assert 1.0 <= gw["tpot"]["p95"] <= 2.5
    assert gw["roundTrip"] is None
    assert s["integrity"]["total"] == 1 and s["integrity"]["recent"] == 1
    assert s["integrity"]["byCheck"] == {"freivalds": 1}
    assert s["noise"]["known"] is True and s["noise"]["alarm"] is True
    assert s["noise"]["byModel"]["qwen"] == 1
    assert s["service"]["present"] is True and s["service"]["inflight"] == 2
    assert s["service"]["requestRate"] == pytest.approx(3.0)
    assert s["factory"]["present"] is False
    # First scrape: no rates yet, cumulative histograms.
    assert first["masks"]["burnRate"] is None and first["gateway"]["completionTps"] is None
    assert first["gateway"]["ttft"]["interval"] is False and first["gateway"]["ttft"]["count"] == 10
    assert first["noise"]["alarm"] is False and first["integrity"]["recent"] is None


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_edge_cases(tmp_path: Path) -> None:
    res = _run_node(
        tmp_path,
        r"""
const text = [
  '# HELP x help', '# TYPE x counter', 'x_total 5', 'y{a="1",b="q\\"uo\\nte"} +Inf', 'z NaN',
  'h_bucket{le="0.1"} 0', 'h_bucket{le="1.0"} 4', 'h_bucket{le="+Inf"} 4', 'h_sum 2', 'h_count 4',
  'garbage line without value', 'bad-name 3', '', 'w{a="1"} 7 1700000000000',
].join("\n");
const m = C.parse(text);
const reset = C.rate(C.parse("x_total 9"), C.parse("x_total 2"), "x_total", 5);
const missing = C.rate(C.parse(""), C.parse("x_total 2"), "x_total", 5);
const empty = C.summarize(null, C.parse(""), 0);
console.log(JSON.stringify({
  x: C.sum(m, "x_total"), yl: m.y[0].labels, yinf: m.y[0].value === Infinity, z: isNaN(m.z[0].value),
  w: C.sum(m, "w"), bad: m["bad-name"] === undefined,
  q50: C.quantile(0.5, C.buckets(m, "h")), qinf: C.quantile(0.5, {"+Inf": 3}), qnone: C.quantile(0.5, {}),
  reset, missing, empty,
  f1: C.fmtSeconds(0.0004), f2: C.fmtSeconds(0.25), f3: C.fmtSeconds(259200), f4: C.fmtNum(1234567), f5: C.fmtNum(null),
}));
""",
    )
    assert res["x"] == 5
    assert res["yl"] == {"a": "1", "b": 'q"uo\nte'}
    assert res["yinf"] and res["z"]
    assert res["w"] == 7  # timestamp ignored
    assert res["bad"] is True
    assert res["q50"] == pytest.approx(0.1 + 0.9 * 0.5)
    assert res["qinf"] == 0 and res["qnone"] is None
    assert res["reset"] is None and res["missing"] is None
    e = res["empty"]
    assert e["masks"]["available"] is None and e["gateway"]["present"] is False
    assert e["noise"]["known"] is False and e["integrity"]["total"] is None
    assert res["f1"] == "400 µs" and res["f2"] == "250 ms" and res["f3"] == "3.0 d"
    assert res["f4"] == "1.23M" and res["f5"] == "\u2013"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_page_scripts_are_valid_javascript(tmp_path: Path) -> None:
    page = dashboard_html()
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    assert len(scripts) == 2
    for i, src in enumerate(scripts):
        f = tmp_path / f"s{i}.js"
        f.write_text(src)
        out = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True, timeout=60, check=False)  # type: ignore[list-item]  # noqa: S603
        assert out.returncode == 0, out.stderr
