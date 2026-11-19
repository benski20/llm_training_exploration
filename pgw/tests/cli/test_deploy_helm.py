"""Helm charts (deploy/helm): structure, rendering, security posture, network policies, settings wiring.

Templates are rendered with :mod:`tests.cli.helm_render` (no ``helm`` binary needed). When ``helm`` is
installed, ``helm lint`` and ``helm template`` run as well.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from .conftest import DEPLOY
from .deploy_utils import construct_settings, unknown_env_names
from .helm_render import RenderError, deep_merge, render_chart, render_chart_objects

HELM = DEPLOY / "helm"
CHARTS = ["pgw-gateway", "pgw-factory", "pgw-service"]

#: chart-testing style CI values (``<chart>/ci/ci-values.yaml``), also used by the CI workflow.
MINIMAL: dict[str, dict[str, Any]] = {
    c: yaml.safe_load((HELM / c / "ci" / "ci-values.yaml").read_text()) for c in CHARTS
}

VARIANTS: dict[str, list[dict[str, Any]]] = {
    "pgw-gateway": [
        {},
        {"tls": {"enabled": True, "existingSecret": "gw-tls"}},
        {"tls": {"enabled": True, "existingSecret": "gw-tls", "requireClientCert": True}},
        {"serviceClientTls": {"existingSecret": "svc-client"}, "ingress": {"enabled": True, "className": "nginx"},
         "serviceMonitor": {"enabled": True}, "podDisruptionBudget": {"enabled": True}},
        {"modelPacks": {"existingClaim": "", "image": {"repository": "reg/packs", "tag": "1"}},
         "state": {"persistence": {"enabled": False}}},
        {"maskstore": {"keyProvider": "kms", "kmsKeyUri": "aws-kms://arn:aws:kms:eu-west-1:000000000000:key/x"},
         "secrets": {"kmsMacKey": "mac"}},
        {"networkPolicy": {"egress": {"computeService": {"cidrs": [], "podSelector": {"matchLabels": {"a": "b"}},
                                                          "namespaceSelector": {"matchLabels": {"ns": "c"}}}}}},
        {"networkPolicy": {"enabled": False}, "serviceAccount": {"create": False},
         "config": {"extraEnv": {"PGW_GATEWAY_CORS_ORIGINS": "[]"}, "integrityCooldownSeconds": 300}},
        {"console": {"viewerTokensKey": "viewer-tokens"},
         "consoleIngress": {"enabled": True, "className": "nginx-internal", "host": "pgw-console.example.internal",
                            "tls": [{"secretName": "console-tls", "hosts": ["pgw-console.example.internal"]}]}},
        {"console": {"enabled": False}},
    ],
    "pgw-factory": [
        {},
        {"gpu": {"enabled": True}, "image": {"repository": "reg/pgw-factory-cuda"}},
        {"output": "export", "secrets": {"existingSecret": ""},
         "export": {"existingClaim": "outbox", "recipientPublicKey": {"configMap": "customer-acme"}}},
        {"modelPacks": {"packDirs": ["a", "b", "c"]}, "config": {"rings": ["K40A12", "K48A20"]},
         "store": {"colocateWithGateway": {"gatewayRelease": "gw"}}},
        {"networkPolicy": {"egress": {"dns": {"enabled": True}}}, "store": {"keyProvider": "kms",
                                                                          "kmsKeyUri": "gcp-kms://x"}},
    ],
    "pgw-service": [
        {},
        {"tls": {"enabled": False}},
        {"tls": {"requireClientCert": False},
         "autoscaling": {"enabled": True, "customMetric": {"name": "pgw_service_inflight"}},
         "metrics": {"serviceMonitor": {"enabled": True}}},
        {"gpu": {"enabled": False}, "modelPacks": {"existingClaim": "", "image": {"repository": "reg/packs"}}},
        {"tenants": {"mtlsTenantsKey": "mtls-tenants"}, "config": {"tenantMaxInflight": 8,
                                                                   "tenantRowsPerSecond": 50000}},
    ],
}


def _cases() -> list[tuple[str, dict[str, Any]]]:
    return [(c, deep_merge(MINIMAL[c], v)) for c in CHARTS for v in VARIANTS[c]]


def _objects(chart: str, values: dict[str, Any]) -> list[dict[str, Any]]:
    return render_chart_objects(HELM / chart, values, release="gw")


def _pod_spec(objs: list[dict[str, Any]]) -> dict[str, Any]:
    deps = [o for o in objs if o["kind"] == "Deployment"]
    assert len(deps) == 1
    spec: dict[str, Any] = deps[0]["spec"]["template"]["spec"]
    return spec


def _env(container: dict[str, Any]) -> dict[str, Any]:
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in container.get("env", [])}


# ---- renderer sanity -------------------------------------------------------------------------------
def test_renderer_basics(tmp_path: Path) -> None:
    ch = tmp_path / "c"
    (ch / "templates").mkdir(parents=True)
    (ch / "Chart.yaml").write_text("apiVersion: v2\nname: c\nversion: 1.0.0\nappVersion: '2'\n")
    (ch / "values.yaml").write_text("a: {b: [1, 2]}\nflag: false\nname: x\n")
    (ch / "templates" / "_h.tpl").write_text('{{- define "c.n" -}}{{ .Values.name | upper }}{{- end -}}')
    (ch / "templates" / "t.yaml").write_text(
        "n: {{ include \"c.n\" . }}\n"
        "{{- if .Values.flag }}\nx: 1\n{{- else if .Values.a }}\nx: 2\n{{- end }}\n"
        "l:\n{{- range $i, $v := .Values.a.b }}\n  - {{ $i }}-{{ $v }}\n{{- end }}\n"
        "{{- $acc := list }}{{- range .Values.a.b }}{{- $acc = append $acc (mul2 .) }}{{- end }}\n"
    )
    with pytest.raises(RenderError, match="unknown function 'mul2'"):
        render_chart(ch)
    (ch / "templates" / "t.yaml").write_text(
        "n: {{ include \"c.n\" . }}\n"
        "{{- if .Values.flag }}\nx: 1\n{{- else if .Values.a }}\nx: 2\n{{- end }}\n"
        "l:\n{{- range $i, $v := .Values.a.b }}\n  - {{ $i }}-{{ $v }}\n{{- end }}\n"
        "j: {{ toJson (dict \"k\" (default \"d\" .Values.missing)) | quote }}\n"
    )
    out = yaml.safe_load(render_chart(ch)["c/templates/t.yaml"])
    assert out == {"n": "X", "x": 2, "l": ["0-1", "1-2"], "j": '{"k":"d"}'}
    (ch / "templates" / "t.yaml").write_text("{{ .Values.nope.deeper }}")
    with pytest.raises(RenderError, match="nil pointer"):
        render_chart(ch)
    (ch / "templates" / "t.yaml").write_text("{{ if .Values.flag }}x")
    with pytest.raises(RenderError, match="unclosed"):
        render_chart(ch)


# ---- chart metadata --------------------------------------------------------------------------------
@pytest.mark.parametrize("chart", CHARTS)
def test_chart_metadata(chart: str) -> None:
    meta = yaml.safe_load((HELM / chart / "Chart.yaml").read_text())
    assert meta["apiVersion"] == "v2" and meta["name"] == chart and meta["type"] == "application"
    assert re.fullmatch(r"\d+\.\d+\.\d+", meta["version"])
    from pgw import __version__

    assert meta["appVersion"] == __version__
    values = yaml.safe_load((HELM / chart / "values.yaml").read_text())
    assert isinstance(values, dict) and values["image"]["tag"] == ""
    assert (HELM / chart / "templates" / "NOTES.txt").is_file()
    assert (HELM / chart / "templates" / "_helpers.tpl").is_file()


@pytest.mark.parametrize("chart", CHARTS)
def test_values_hold_no_secrets(chart: str) -> None:
    text = (HELM / chart / "values.yaml").read_text()
    for pat in (r"BEGIN [A-Z ]*PRIVATE KEY", r"pgwt-[A-Za-z0-9_-]{20,}", r"pgw-[0-9a-f]{12}-", r"AKIA[0-9A-Z]{16}"):
        assert not re.search(pat, text), pat
    values = yaml.safe_load(text)
    flat = json.dumps(values)
    assert "existingSecret" in flat  # secrets come from pre-created Secrets


@pytest.mark.parametrize(("chart", "msg"), [
    ("pgw-gateway", "secrets.existingSecret is required"),
    ("pgw-factory", "is required"),
    ("pgw-service", "is required"),
])
def test_defaults_require_explicit_secrets_and_volumes(chart: str, msg: str) -> None:
    with pytest.raises(RenderError, match=msg):
        render_chart(HELM / chart)


# ---- every variant renders to valid, hardened Kubernetes objects -----------------------------------
@pytest.mark.parametrize(("chart", "values"), _cases(), ids=lambda x: x if isinstance(x, str) else "")
def test_render_valid_and_hardened(chart: str, values: dict[str, Any]) -> None:
    objs = _objects(chart, values)
    kinds = {o["kind"] for o in objs}
    assert "Deployment" in kinds
    for o in objs:
        assert o.get("apiVersion") and o.get("kind"), o
        name = o["metadata"]["name"]
        assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name) and len(name) <= 63
        labels = o["metadata"].get("labels", {})
        assert labels.get("app.kubernetes.io/part-of") == "pgw", o["kind"]
        assert labels.get("app.kubernetes.io/managed-by") == "Helm"
    spec = _pod_spec(objs)
    assert spec["automountServiceAccountToken"] is False
    psc = spec["securityContext"]
    assert psc["runAsNonRoot"] is True and psc["runAsUser"] == 10001
    assert psc["seccompProfile"]["type"] == "RuntimeDefault"
    containers = spec["containers"] + spec.get("initContainers", [])
    for c in containers:
        sc = c["securityContext"]
        assert sc["readOnlyRootFilesystem"] is True and sc["allowPrivilegeEscalation"] is False
        assert sc["capabilities"]["drop"] == ["ALL"]
        assert not c["image"].endswith(":latest") or c["name"] == "copy-packs"
    main = spec["containers"][0]
    assert main["args"][0] == "pgw"
    assert main["resources"]["requests"] and main["resources"]["limits"]["memory"]
    # volumes referenced by mounts exist, and ports referenced by probes exist
    vols = {v["name"] for v in spec["volumes"]}
    for c in containers:
        for m in c.get("volumeMounts", []):
            assert m["name"] in vols, m
    ports = {p["name"] for p in main.get("ports", [])}
    for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
        if chart == "pgw-factory" and probe == "readinessProbe":  # a batch worker: liveness only
            assert probe not in main
            continue
        p = main[probe]
        port = (p.get("httpGet") or p.get("tcpSocket"))["port"]
        assert port in ports
    # services select the pods and target named ports
    pod_labels = next(o for o in objs if o["kind"] == "Deployment")["spec"]["template"]["metadata"]["labels"]
    for svc in (o for o in objs if o["kind"] == "Service"):
        assert all(pod_labels.get(k) == v for k, v in svc["spec"]["selector"].items())
        for p in svc["spec"]["ports"]:
            assert p["targetPort"] in ports
    # env names are real settings fields and the settings parse
    env = _env(main)
    assert unknown_env_names(list(env)) == []


@pytest.mark.parametrize(("chart", "values"), _cases(), ids=lambda x: x if isinstance(x, str) else "")
def test_rendered_env_parses_into_settings(chart: str, values: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    import base64

    main = _pod_spec(_objects(chart, values))["containers"][0]
    env: dict[str, str] = {}
    for e in main["env"]:
        if "value" in e:
            env[e["name"]] = str(e["value"])
        else:  # secretKeyRef: plausible dummy value
            env[e["name"]] = base64.b64encode(b"\x01" * 32).decode() if e["name"].endswith(("KEK", "MAC_KEY")) \
                else "dummy-secret-value-" + "x" * 16
    built = construct_settings(env, monkeypatch)
    if chart == "pgw-service":
        s = built["PGW_SERVICE_"]
        assert s.port == 8443 and s.packs  # type: ignore[attr-defined]
        assert s.metrics_on_main_port is False  # type: ignore[attr-defined]
    if chart == "pgw-gateway":
        g = built["PGW_GATEWAY_"]
        assert g.require_service_tls is True  # type: ignore[attr-defined]
        ms = built["PGW_MASKSTORE_"]
        assert str(ms.root).startswith("/var/lib/pgw/state")  # type: ignore[attr-defined]
    if chart == "pgw-factory":
        f = built["PGW_FACTORY_"]
        assert f.output == values.get("output", "store")  # type: ignore[attr-defined]


# ---- gateway specifics -----------------------------------------------------------------------------
def test_gateway_models_template_is_valid_and_secret_free(tmp_path: Path) -> None:
    from pgw.gateway.config import load_models_file

    objs = _objects("pgw-gateway", deep_merge(MINIMAL["pgw-gateway"], {"serviceClientTls": {"existingSecret": "c"}}))
    cm = next(o for o in objs if o["kind"] == "ConfigMap")
    tmpl = cm["data"]["models.tmpl.json"]
    assert "${file:/etc/pgw/secrets/service-token}" in tmpl
    rendered = re.sub(r"\$\{file:[^}]+\}", "pgwt-dummy", tmpl)
    p = tmp_path / "models.json"
    p.write_text(rendered)
    (models,) = load_models_file(p)
    assert models.name == "llama3-8b" and models.mode == "masked" and models.ring == "K32A12C27"
    assert str(models.pack_path) == "/var/lib/pgw/packs/llama3-8b"
    assert models.fingerprint == "ab" * 32
    assert str(models.client_cert_file) == "/etc/pgw/service-tls/tls.crt"
    assert models.service_token is not None
    # the template must render through the real entrypoint helper too
    from pgw.cli.configtools import expand_placeholders

    tok = tmp_path / "service-token"
    tok.write_text("pgwt-abc\n")
    out = expand_placeholders(tmpl.replace("/etc/pgw/secrets/service-token", str(tok)))
    assert json.loads(out)[0]["service_token"] == "pgwt-abc"  # noqa: S105 - test fixture


def test_gateway_deployment_specifics() -> None:
    objs = _objects("pgw-gateway", MINIMAL["pgw-gateway"])
    dep = next(o for o in objs if o["kind"] == "Deployment")
    assert dep["spec"]["strategy"]["type"] == "Recreate" and dep["spec"]["replicas"] == 1
    main = dep["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e for e in main["env"]}
    for secret_env in ("PGW_GATEWAY_ADMIN_TOKEN", "PGW_MASKSTORE_KEK"):
        assert "secretKeyRef" in env[secret_env]["valueFrom"], secret_env
    assert env["PGW_RENDER"]["value"] == "/etc/pgw/models/models.tmpl.json:/run/pgw/models.json"
    vols = {v["name"]: v for v in dep["spec"]["template"]["spec"]["volumes"]}
    assert vols["run"]["emptyDir"]["medium"] == "Memory"
    assert vols["service-token"]["secret"]["items"] == [{"key": "service-token", "path": "service-token"}]
    assert "checksum/models" in dep["spec"]["template"]["metadata"]["annotations"]
    pvc = next(o for o in objs if o["kind"] == "PersistentVolumeClaim")
    assert pvc["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"


def test_gateway_network_policy_restricts_egress_to_compute_service() -> None:
    objs = _objects("pgw-gateway", MINIMAL["pgw-gateway"])
    np = next(o for o in objs if o["kind"] == "NetworkPolicy")
    assert set(np["spec"]["policyTypes"]) == {"Ingress", "Egress"}
    egress = np["spec"]["egress"]
    assert len(egress) == 2
    dns, compute = egress
    assert {p["port"] for p in dns["ports"]} == {53}
    assert dns["to"][0]["podSelector"]["matchLabels"] == {"k8s-app": "kube-dns"}
    assert compute["to"] == [{"ipBlock": {"cidr": "203.0.113.0/28"}}]
    assert {p["port"] for p in compute["ports"]} == {443, 8443}
    for rule in egress:
        for peer in rule["to"]:
            assert peer.get("ipBlock", {}).get("cidr") != "0.0.0.0/0"
    # without a configured destination there is no compute egress at all, and NOTES warns
    vals = deep_merge(MINIMAL["pgw-gateway"], {"networkPolicy": {"egress": {"computeService": {"cidrs": []}}}})
    np = next(o for o in _objects("pgw-gateway", vals) if o["kind"] == "NetworkPolicy")
    assert len(np["spec"]["egress"]) == 1
    notes = render_chart(HELM / "pgw-gateway", vals)["pgw-gateway/templates/NOTES.txt"]
    assert "cannot reach" in notes


def test_gateway_probes_use_internal_ops_listener_in_every_tls_mode() -> None:
    """With mTLS on the API port, readiness must still be checked (reliability review)."""
    for tls in ({"enabled": True, "existingSecret": "t", "requireClientCert": True},
                {"enabled": True, "existingSecret": "t"}, {"enabled": False}):
        vals = deep_merge(MINIMAL["pgw-gateway"], {"tls": tls})
        objs = _objects("pgw-gateway", vals)
        main = _pod_spec(objs)["containers"][0]
        for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
            assert main[probe]["httpGet"]["port"] == "ops" and "tcpSocket" not in main[probe]
        assert main["readinessProbe"]["httpGet"]["path"] == "/readyz"
        env = _env(main)
        assert env["PGW_GATEWAY_OPS_PORT"] == "9091"
        if tls.get("requireClientCert"):
            assert env["PGW_GATEWAY_TLS__REQUIRE_CLIENT_CERT"] == "true"


def test_gateway_graceful_shutdown_budget() -> None:
    main_spec = _pod_spec(_objects("pgw-gateway", MINIMAL["pgw-gateway"]))
    env = _env(main_spec["containers"][0])
    drain = int(env["PGW_GATEWAY_DRAIN_TIMEOUT_S"])
    pre = main_spec["containers"][0]["lifecycle"]["preStop"]["exec"]["command"]
    assert pre[0] == "sleep" and main_spec["terminationGracePeriodSeconds"] > drain + int(pre[1])
    with pytest.raises(RenderError, match="terminationGracePeriodSeconds"):
        _objects("pgw-gateway", deep_merge(MINIMAL["pgw-gateway"], {"terminationGracePeriodSeconds": 30}))


def test_gateway_volume_must_hold_target_slots() -> None:
    _objects("pgw-gateway", MINIMAL["pgw-gateway"])  # defaults are consistent
    with pytest.raises(RenderError, match="cannot hold"):
        _objects("pgw-gateway", deep_merge(MINIMAL["pgw-gateway"], {"maskstore": {"targetSlots": 65536}}))
    _objects("pgw-gateway", deep_merge(MINIMAL["pgw-gateway"], {"maskstore": {"targetSlots": 65536},
                                                               "state": {"persistence": {"size": "1Ti"}}}))


# ---- service specifics -----------------------------------------------------------------------------
def test_service_never_gets_mask_material() -> None:
    for v in VARIANTS["pgw-service"]:
        objs = _objects("pgw-service", deep_merge(MINIMAL["pgw-service"], v))
        spec = _pod_spec(objs)
        env = _env(spec["containers"][0])
        assert not [k for k in env if k.startswith(("PGW_MASKSTORE_", "PGW_FACTORY_", "PGW_GATEWAY_"))]
        names = {v["name"] for v in spec["volumes"]}
        assert not names & {"state", "masks", "kek", "store"}
        text = json.dumps(objs)
        assert "kek" not in text.lower()


def test_service_network_policy_denies_egress() -> None:
    objs = _objects("pgw-service", MINIMAL["pgw-service"])
    np = next(o for o in objs if o["kind"] == "NetworkPolicy")
    assert np["spec"]["egress"] == [] and "Egress" in np["spec"]["policyTypes"]
    ports = [{p["port"] for p in r["ports"]} for r in np["spec"]["ingress"]]
    assert ports == [{"compute"}, {"metrics"}]
    services = {o["metadata"]["name"]: o for o in objs if o["kind"] == "Service"}
    assert {p["name"] for p in services["gw-pgw-service"]["spec"]["ports"]} == {"compute"}
    assert services["gw-pgw-service-metrics"]["spec"]["type"] == "ClusterIP"


def test_service_gpu_and_scaling() -> None:
    objs = _objects("pgw-service", MINIMAL["pgw-service"])
    spec = _pod_spec(objs)
    assert spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"} in spec["tolerations"]
    env = _env(spec["containers"][0])
    assert json.loads(env["PGW_SERVICE_PACKS"]) == ["/var/lib/pgw/packs/llama3-8b"]
    assert spec["terminationGracePeriodSeconds"] > int(env["PGW_SERVICE_DRAIN_TIMEOUT_S"])
    dep = next(o for o in objs if o["kind"] == "Deployment")
    assert dep["spec"]["replicas"] == 2 and dep["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    # GPU deployments must not autoscale on CPU (it stays low while the GPU saturates)
    with pytest.raises(RenderError, match="customMetric"):
        _objects("pgw-service", deep_merge(MINIMAL["pgw-service"], {"autoscaling": {"enabled": True}}))
    hpa = next(o for o in _objects("pgw-service", deep_merge(MINIMAL["pgw-service"], {
        "autoscaling": {"enabled": True, "customMetric": {"name": "pgw_service_inflight"}}}))
        if o["kind"] == "HorizontalPodAutoscaler")
    assert hpa["spec"]["metrics"][0]["pods"]["metric"]["name"] == "pgw_service_inflight"
    objs = _objects("pgw-service", deep_merge(MINIMAL["pgw-service"], {"autoscaling": {"enabled": True},
                                                                       "gpu": {"enabled": False}}))
    dep = next(o for o in objs if o["kind"] == "Deployment")
    assert "replicas" not in dep["spec"]
    assert any(o["kind"] == "HorizontalPodAutoscaler" for o in objs)
    spec = dep["spec"]["template"]["spec"]
    assert "nvidia.com/gpu" not in spec["containers"][0]["resources"]["limits"]
    assert _env(spec["containers"][0])["PGW_SERVICE_DEVICE"] == "cpu"


# ---- factory specifics -----------------------------------------------------------------------------
def test_factory_has_no_network_and_colocates() -> None:
    objs = _objects("pgw-factory", MINIMAL["pgw-factory"])
    np = next(o for o in objs if o["kind"] == "NetworkPolicy")
    # only Prometheus may reach the metrics/health port; no egress at all
    assert [{p["port"] for p in r["ports"]} for r in np["spec"]["ingress"]] == [{"metrics"}]
    assert np["spec"]["egress"] == []
    spec = _pod_spec(objs)
    aff = spec["affinity"]["podAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
    assert aff["labelSelector"]["matchLabels"]["app.kubernetes.io/name"] == "pgw-gateway"
    assert [p["name"] for p in spec["containers"][0]["ports"]] == ["metrics"]
    assert spec["containers"][0]["livenessProbe"]["httpGet"]["path"] == "/healthz"
    env = _env(spec["containers"][0])
    assert env["PGW_MASKSTORE_ROOT"] == "/var/lib/pgw/state/masks"
    assert "secretKeyRef" in env["PGW_MASKSTORE_KEK"]


def test_factory_export_mode() -> None:
    vals = deep_merge(MINIMAL["pgw-factory"], VARIANTS["pgw-factory"][2])
    spec = _pod_spec(_objects("pgw-factory", vals))
    env = _env(spec["containers"][0])
    assert env["PGW_FACTORY_OUTPUT"] == "export" and env["PGW_FACTORY_EXPORT_DIR"] == "/var/lib/pgw/export"
    assert not [k for k in env if k.startswith("PGW_MASKSTORE_")]
    assert {v["name"] for v in spec["volumes"]} >= {"export", "recipient"}
    assert "affinity" not in spec


def test_factory_invalid_output_fails() -> None:
    with pytest.raises(RenderError, match="output must be"):
        _objects("pgw-factory", deep_merge(MINIMAL["pgw-factory"], {"output": "s3"}))


# ---- the real helm, when installed -------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
@pytest.mark.parametrize("chart", CHARTS)
def test_helm_lint_and_template(chart: str, tmp_path: Path) -> None:
    vals = HELM / chart / "ci" / "ci-values.yaml"
    subprocess.run(["helm", "lint", "--strict", str(HELM / chart), "-f", str(vals)], check=True)  # noqa: S603,S607
    out = subprocess.run(["helm", "template", "gw", str(HELM / chart), "-f", str(vals)],  # noqa: S603,S607
                         check=True, capture_output=True, text=True).stdout
    docs = [d for d in yaml.safe_load_all(out) if d]
    ours = sorted((o["kind"], o["metadata"]["name"]) for o in _objects(chart, MINIMAL[chart]))
    assert sorted((d["kind"], d["metadata"]["name"]) for d in docs) == ours


def test_gateway_console_wiring() -> None:
    """Console: on by default next to /admin, viewer tokens as a mounted Secret file, internal ingress."""
    spec = _pod_spec(_objects("pgw-gateway", MINIMAL["pgw-gateway"]))
    env = _env(spec["containers"][0])
    assert env["PGW_GATEWAY_CONSOLE_ENABLED"] == "true" and "PGW_GATEWAY_CONSOLE_VIEWER_TOKENS_FILE" not in env
    assert "console-viewers" not in {v["name"] for v in spec["volumes"]}

    vals = deep_merge(MINIMAL["pgw-gateway"], VARIANTS["pgw-gateway"][-2])
    objs = _objects("pgw-gateway", vals)
    spec = _pod_spec(objs)
    main = spec["containers"][0]
    env = _env(main)
    assert env["PGW_GATEWAY_CONSOLE_VIEWER_TOKENS_FILE"] == "/etc/pgw/console/viewer-tokens"
    vol = next(v for v in spec["volumes"] if v["name"] == "console-viewers")
    assert vol["secret"] == {"secretName": "gw-secrets", "defaultMode": 0o440,
                             "items": [{"key": "viewer-tokens", "path": "viewer-tokens"}]}
    mount = next(m for m in main["volumeMounts"] if m["name"] == "console-viewers")
    assert mount["mountPath"] == "/etc/pgw/console" and mount["readOnly"] is True and "subPath" not in mount
    ings = {o["metadata"]["name"]: o for o in objs if o["kind"] == "Ingress"}
    ing = ings["gw-pgw-gateway-console"]
    assert ing["spec"]["ingressClassName"] == "nginx-internal" and ing["spec"]["tls"]
    (rule,) = ing["spec"]["rules"]
    assert rule["host"] == "pgw-console.example.internal"
    assert {p["path"] for p in rule["http"]["paths"]} == {"/console", "/admin"}  # never /v1 or /
    notes = render_chart(HELM / "pgw-gateway", vals)["pgw-gateway/templates/NOTES.txt"]
    assert "https://pgw-console.example.internal/console/" in notes

    # fail closed: console ingress without TLS or host, or with the console disabled
    for override, msg in (({"consoleIngress": {"tls": []}}, "tls is required"),
                          ({"consoleIngress": {"host": ""}}, "host is required"),
                          ({"console": {"enabled": False}}, "needs console.enabled")):
        with pytest.raises(RenderError, match=msg):
            render_chart(HELM / "pgw-gateway", deep_merge(vals, override))

    off = deep_merge(MINIMAL["pgw-gateway"], {"console": {"enabled": False, "viewerTokensKey": "viewer-tokens"}})
    env = _env(_pod_spec(_objects("pgw-gateway", off))["containers"][0])
    assert env["PGW_GATEWAY_CONSOLE_ENABLED"] == "false" and "PGW_GATEWAY_CONSOLE_VIEWER_TOKENS_FILE" not in env
    # a catch-all API ingress is flagged in NOTES (it also publishes /admin and /console)
    notes = render_chart(HELM / "pgw-gateway", deep_merge(MINIMAL["pgw-gateway"], {"ingress": {"enabled": True}}))
    assert "route only /v1" in notes["pgw-gateway/templates/NOTES.txt"]


# ---- review hardening: fail closed on settings that would weaken isolation ------------------------
def test_gateway_refuses_more_than_one_replica() -> None:
    """Two gateways on one mask store could hand out the same single-use mask twice."""
    vals = deep_merge(MINIMAL["pgw-gateway"], {"replicaCount": 2})
    with pytest.raises(RenderError, match="replicaCount must be 1"):
        render_chart(HELM / "pgw-gateway", vals)


@pytest.mark.parametrize(("override", "msg"), [
    ({"ingressFrom": []}, "ingressFrom must list"),
    ({"egress": {"computeService": {"ports": []}}}, "ports must not be empty"),
])
def test_gateway_network_policy_empty_lists_fail_closed(override: dict[str, Any], msg: str) -> None:
    """An empty ``from`` / ``ports`` list means *everything* to Kubernetes, so it must not render."""
    vals = deep_merge(MINIMAL["pgw-gateway"], {"networkPolicy": override})
    with pytest.raises(RenderError, match=msg):
        render_chart(HELM / "pgw-gateway", vals)


def test_service_network_policy_empty_compute_from_fails_closed() -> None:
    vals = deep_merge(MINIMAL["pgw-service"], {"networkPolicy": {"computeFrom": []}})
    with pytest.raises(RenderError, match="computeFrom must list"):
        render_chart(HELM / "pgw-service", vals)


# ---- security review regressions -------------------------------------------------------------------
def test_gateway_ships_external_counter_on_separate_volume() -> None:
    """Rollback witness must not live on (and be restored with) the state volume."""
    objs = _objects("pgw-gateway", MINIMAL["pgw-gateway"])
    spec = _pod_spec(objs)
    env = _env(spec["containers"][0])
    assert env["PGW_MASKSTORE_COUNTER"] == "file"
    assert env["PGW_MASKSTORE_COUNTER_FILE"].startswith("/var/lib/pgw/counter/")
    assert "PGW_MASKSTORE_ALLOW_NO_EXTERNAL_COUNTER" not in env
    vols = {v["name"]: v for v in spec["volumes"]}
    assert vols["counter"]["persistentVolumeClaim"]["claimName"] == "gw-pgw-gateway-counter"
    assert vols["state"]["persistentVolumeClaim"]["claimName"] != vols["counter"]["persistentVolumeClaim"]["claimName"]
    pvcs = {o["metadata"]["name"]: o for o in objs if o["kind"] == "PersistentVolumeClaim"}
    assert pvcs["gw-pgw-gateway-counter"]["metadata"]["labels"]["velero.io/exclude-from-backup"] == "true"
    with pytest.raises(RenderError, match="allowNoExternalCounter"):
        _objects("pgw-gateway", deep_merge(MINIMAL["pgw-gateway"], {"maskstore": {"counter": "none"}}))
    dev = _env(_pod_spec(_objects("pgw-gateway", deep_merge(
        MINIMAL["pgw-gateway"], {"maskstore": {"counter": "none", "allowNoExternalCounter": True}})))["containers"][0])
    assert dev["PGW_MASKSTORE_ALLOW_NO_EXTERNAL_COUNTER"] == "true" and "PGW_MASKSTORE_COUNTER" not in dev


def test_gateway_requires_fingerprint_and_no_shared_store() -> None:
    vals = yaml.safe_load((HELM / "pgw-gateway" / "values.yaml").read_text())
    assert "ReadWriteMany" not in json.dumps(vals["state"]["persistence"]["accessModes"])
    bad = deep_merge(MINIMAL["pgw-gateway"], {})
    bad["models"] = [{**MINIMAL["pgw-gateway"]["models"][0], "fingerprint": ""}]
    with pytest.raises(RenderError, match="fingerprint"):
        _objects("pgw-gateway", bad)
    fac = deep_merge(MINIMAL["pgw-factory"], {"output": "store", "store": {"colocateWithGateway": {"enabled": False}}})
    with pytest.raises(RenderError, match="never be shared across nodes"):
        _objects("pgw-factory", fac)
