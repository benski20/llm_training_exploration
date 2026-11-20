"""Alert rules (deploy/monitoring/pgw-alerts.yml and the charts' PrometheusRules) and Helm readiness values."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from pgw.observability.metrics import METRIC_NAMES

from ..cli.helm_render import RenderError, deep_merge, render_chart_objects

ROOT = Path(__file__).resolve().parents[2]
RULES_FILE = ROOT / "deploy" / "monitoring" / "pgw-alerts.yml"
HELM = ROOT / "deploy" / "helm"
CHARTS = ["pgw-gateway", "pgw-factory", "pgw-service"]
MINIMAL: dict[str, dict[str, Any]] = {
    c: yaml.safe_load((HELM / c / "ci" / "ci-values.yaml").read_text()) for c in CHARTS
}
#: How each chart turns its PrometheusRule on.
MONITOR_ON: dict[str, dict[str, Any]] = {
    "pgw-gateway": {"serviceMonitor": {"enabled": True}},
    "pgw-factory": {"metrics": {"enabled": True, "serviceMonitor": {"enabled": True}}},
    "pgw-service": {"metrics": {"serviceMonitor": {"enabled": True}}},
}
#: Which groups of the standalone file each chart ships.
CHART_GROUPS = {
    "pgw-gateway": {"pgw-mask-supply", "pgw-gateway"},
    "pgw-factory": {"pgw-factory"},
    "pgw-service": {"pgw-service"},
}
#: Labels Prometheus adds to every scraped series (allowed in ``by (...)`` clauses).
TARGET_LABELS = {"namespace", "job", "instance", "pod"}
_METRIC_RE = re.compile(r"\bpgw_[a-z0-9_]+\b")
_BY_RE = re.compile(r"\b(?:by|on)\s*\(([^)]*)\)")
_HIST_SUFFIXES = ("_bucket", "_sum", "_count")


def _file_rules() -> dict[str, tuple[str, dict[str, Any]]]:
    doc = yaml.safe_load(RULES_FILE.read_text())
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for g in doc["groups"]:
        for r in g["rules"]:
            assert r["alert"] not in out, f"duplicate alert {r['alert']}"
            out[r["alert"]] = (g["name"], r)
    return out


def _chart_rules(chart: str, values: dict[str, Any] | None = None) -> dict[str, tuple[str, dict[str, Any]]]:
    vals = deep_merge(deep_merge(MINIMAL[chart], MONITOR_ON[chart]), values or {})
    objs = [o for o in render_chart_objects(HELM / chart, vals, release="gw") if o["kind"] == "PrometheusRule"]
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for o in objs:
        assert o["apiVersion"] == "monitoring.coreos.com/v1"
        for g in o["spec"]["groups"]:
            for r in g["rules"]:
                out[r["alert"]] = (g["name"], r)
    return out


def _metrics_in(expr: str) -> set[str]:
    names = set()
    for m in _METRIC_RE.findall(expr):
        if m not in METRIC_NAMES:
            for suf in _HIST_SUFFIXES:
                base = m.removesuffix(suf)
                if base != m and METRIC_NAMES.get(base, ("",))[0] == "histogram":
                    m = base
                    break
        names.add(m)
    return names


def _norm(expr: str) -> str:
    return " ".join(str(expr).split())


def test_rule_file_metrics_exist() -> None:
    rules = _file_rules()
    assert rules
    for name, (_, r) in rules.items():
        used = _metrics_in(r["expr"])
        assert used, f"{name}: no pgw_* metric in {r['expr']!r}"
        unknown = used - set(METRIC_NAMES)
        assert not unknown, f"{name}: metrics not in METRIC_NAMES: {sorted(unknown)}"


def test_rule_file_aggregation_labels_exist() -> None:
    for name, (_, r) in _file_rules().items():
        expr = r["expr"]
        labels: set[str] = set()
        for name_ in _metrics_in(expr):
            labels |= set(METRIC_NAMES[name_][1])
        for clause in _BY_RE.findall(expr):
            for lab in (x.strip() for x in clause.split(",") if x.strip()):
                assert lab in labels | TARGET_LABELS, f"{name}: label {lab!r} not on {sorted(_metrics_in(expr))}"


def test_rule_file_covers_required_alerts() -> None:
    rules = _file_rules()
    required = {
        "PGWMaskRunwayLow": "warning", "PGWMaskStoreBelowLowWater": "warning", "PGWFactoryStalled": "warning",
        "PGWIntegrityCheckFailed": "page", "PGWNoiseAlarm": "page", "PGWModelNotReady": "warning",
        "PGWMaskBurnRatioHigh": "warning",
    }
    for alert, sev in required.items():
        assert alert in rules, alert
        assert rules[alert][1]["labels"]["severity"] == sev, alert
    assert "pgw_masks_burned_slots_total" in rules["PGWMaskBurnRatioHigh"][1]["expr"]
    for alert in ("PGWMaskRunwayLow", "PGWMaskStoreBelowLowWater", "PGWMaskBurnRatioHigh", "PGWFactoryNotRefilling"):
        assert "by (fingerprint, ring)" in rules[alert][1]["expr"], alert
    for r in (x for _, x in rules.values()):
        assert set(r) <= {"alert", "expr", "for", "labels", "annotations"}
        assert r["labels"]["severity"] in {"warning", "page"}
        assert r["annotations"]["summary"]


def test_promtool_check_rules(tmp_path: Path) -> None:
    promtool = shutil.which("promtool")
    if promtool is None:
        pytest.skip("promtool not installed")
    res = subprocess.run([promtool, "check", "rules", str(RULES_FILE)], capture_output=True, text=True,  # noqa: S603
                         check=False)
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.parametrize("chart", CHARTS)
def test_chart_rules_match_rule_file(chart: str) -> None:
    file_rules = _file_rules()
    rules = _chart_rules(chart)
    expected = {a for a, (g, _) in file_rules.items() if g in CHART_GROUPS[chart]}
    assert set(rules) == expected
    for alert, (group, r) in rules.items():
        fgroup, fr = file_rules[alert]
        assert group == fgroup
        assert _norm(r["expr"]) == _norm(fr["expr"]), alert
        assert r.get("for") == fr.get("for") and r["labels"] == fr["labels"], alert
        assert "{{" not in json.dumps(r["annotations"]), alert  # no unescaped Prometheus templates in Helm
        unknown = _metrics_in(r["expr"]) - set(METRIC_NAMES)
        assert not unknown, f"{chart}/{alert}: {sorted(unknown)}"


@pytest.mark.parametrize("chart", CHARTS)
def test_chart_rules_gated(chart: str) -> None:
    objs = render_chart_objects(HELM / chart, MINIMAL[chart], release="gw")
    assert not [o for o in objs if o["kind"] == "PrometheusRule"]  # ServiceMonitor off by default
    assert not _chart_rules(chart, {"alerts": {"enabled": False}})


def test_chart_thresholds_from_values() -> None:
    rules = _chart_rules("pgw-gateway", {"alerts": {"runwayMinSeconds": 900, "burnRatioMax": 0.25}})
    assert rules["PGWMaskRunwayLow"][1]["expr"].rstrip().endswith("< 900")
    assert rules["PGWMaskBurnRatioHigh"][1]["expr"].rstrip().endswith("> 0.25")
    rules = _chart_rules("pgw-factory", {"alerts": {"factoryStallSeconds": 600}})
    assert rules["PGWFactoryStalled"][1]["expr"].rstrip().endswith("> 600")
    rules = _chart_rules("pgw-service", {"alerts": {"errorRatioMax": 0.01}})
    assert rules["PGWServiceErrorRatioHigh"][1]["expr"].rstrip().endswith("> 0.01")
    labels = _chart_rules("pgw-gateway", {"alerts": {"labels": {"release": "kps"}}})
    assert labels


# ---- Helm readiness values -------------------------------------------------------------------------
def _pod(chart: str, values: dict[str, Any] | None = None) -> dict[str, Any]:
    objs = render_chart_objects(HELM / chart, deep_merge(MINIMAL[chart], values or {}), release="gw")
    dep = next(o for o in objs if o["kind"] == "Deployment")
    spec: dict[str, Any] = dep["spec"]["template"]["spec"]
    return spec


def _env(spec: dict[str, Any]) -> dict[str, Any]:
    return {e["name"]: e.get("value") for e in spec["containers"][0].get("env", [])}


def test_gateway_guaranteed_qos_by_default() -> None:
    res = _pod("pgw-gateway")["containers"][0]["resources"]
    assert res["requests"] == res["limits"]
    cpu = str(res["limits"]["cpu"])
    assert cpu.isdigit() and int(cpu) >= 1  # integer CPUs: exclusive cores with cpuManagerPolicy static
    assert _pod("pgw-gateway", {"priorityClassName": "pgw-gateway"})["priorityClassName"] == "pgw-gateway"


def test_factory_hard_cpu_limit_and_separation() -> None:
    res = _pod("pgw-factory")["containers"][0]["resources"]
    assert res["limits"]["cpu"] == res["requests"]["cpu"]
    export = {"output": "export", "secrets": {"existingSecret": ""},
              "export": {"existingClaim": "outbox", "recipientPublicKey": {"configMap": "c"}}}
    assert "affinity" not in _pod("pgw-factory", export)
    pref = _pod("pgw-factory", {**export, "separateFromGateway": "preferred"})["affinity"]["podAntiAffinity"]
    term = pref["preferredDuringSchedulingIgnoredDuringExecution"][0]["podAffinityTerm"]
    assert term["labelSelector"]["matchLabels"]["app.kubernetes.io/name"] == "pgw-gateway"
    req = _pod("pgw-factory", {**export, "separateFromGateway": "required"})["affinity"]["podAntiAffinity"]
    assert req["requiredDuringSchedulingIgnoredDuringExecution"][0]["topologyKey"] == "kubernetes.io/hostname"
    with pytest.raises(RenderError, match="separateFromGateway"):
        _pod("pgw-factory", {**export, "separateFromGateway": "yes"})
    # output=store keeps the required co-location with the gateway (the store volume is ReadWriteOnce)
    aff = _pod("pgw-factory", {"separateFromGateway": "required"})["affinity"]
    assert "podAffinity" in aff and "podAntiAffinity" not in aff


@pytest.mark.parametrize("chart", ["pgw-gateway", "pgw-service"])
def test_trusted_proxy_cidrs_env(chart: str) -> None:
    assert "PGW_TRUSTED_PROXY_CIDRS" not in _env(_pod(chart))  # default: trust no proxy
    env = _env(_pod(chart, {"trustedProxyCidrs": ["10.0.0.0/16", "192.168.1.0/24"]}))
    assert json.loads(env["PGW_TRUSTED_PROXY_CIDRS"]) == ["10.0.0.0/16", "192.168.1.0/24"]


def test_service_zone_spread_passthrough() -> None:
    tsc = [{"maxSkew": 1, "topologyKey": "topology.kubernetes.io/zone", "whenUnsatisfiable": "ScheduleAnyway",
            "labelSelector": {"matchLabels": {"app.kubernetes.io/name": "pgw-service"}}}]
    assert _pod("pgw-service", {"topologySpreadConstraints": tsc})["topologySpreadConstraints"] == tsc
    sel = {"topology.kubernetes.io/zone": "us-east-1a"}
    assert _pod("pgw-gateway", {"nodeSelector": sel})["nodeSelector"] == sel
