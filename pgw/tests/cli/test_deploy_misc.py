"""CI workflow, systemd units, VM/OVA build files and deployment docs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from .conftest import DEPLOY, PGW_ROOT
from .deploy_utils import unknown_env_names

WORKFLOW = PGW_ROOT / ".github" / "workflows" / "ci.yml"
SHELL_SCRIPTS = [
    DEPLOY / "docker" / "entrypoint.sh",
    DEPLOY / "compose" / "demo-init.sh",
    DEPLOY / "systemd" / "install.sh",
    DEPLOY / "vmware" / "pgw-ovf-config.sh",
    PGW_ROOT / "ci" / "check.sh",
]


# ---- CI workflow ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def workflow() -> dict[str, object]:
    data = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(data, dict)
    return data


def test_workflow_jobs(workflow: dict[str, object]) -> None:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert {"lint", "typecheck", "test", "deploy-lint", "audit", "images"} <= set(jobs)
    assert workflow["permissions"] == {"contents": "read"}
    steps = {name: " ".join(str(s.get("run", "")) for s in job["steps"]) for name, job in jobs.items()}
    assert "ruff check" in steps["lint"]
    assert "mypy src" in steps["typecheck"]
    assert "pytest" in steps["test"] and 'not gpu' in steps["test"]
    assert "helm lint --strict" in steps["deploy-lint"] and "docker compose config" in steps["deploy-lint"]
    assert set(jobs["test"]["strategy"]["matrix"]["python"]) == {"3.12", "3.13"}
    images = jobs["images"]["strategy"]["matrix"]["image"]
    assert sorted(images) == sorted(p.name.removesuffix(".Dockerfile") for p in (DEPLOY / "docker").glob("*.Dockerfile"))
    # on: is parsed by YAML 1.1 as True
    triggers: Any = workflow.get("on", workflow.get(True))  # type: ignore[call-overload]
    assert "pull_request" in triggers and "push" in triggers


def test_workflow_actions_are_pinned(workflow: dict[str, object]) -> None:
    for job in workflow["jobs"].values():  # type: ignore[attr-defined]
        for step in job["steps"]:
            if "uses" in step:
                # security review: actions pinned by commit SHA, not by a movable tag
                assert re.search(r"@[0-9a-f]{40}$", step["uses"]), step["uses"]


def test_workflow_audits_dependencies(workflow: dict[str, object]) -> None:
    jobs: Any = workflow["jobs"]
    run = " ".join(str(s.get("run", "")) for s in jobs["audit"]["steps"])
    assert "pip-audit" in run and "requirements/" in run and "cargo audit" in run
    assert "audit" in jobs["images"]["needs"]


def test_workflow_referenced_paths_exist() -> None:
    text = WORKFLOW.read_text()
    for rel in re.findall(r"\$\{PGW_DIR\}/([\w./*-]+)", text):
        if "*" in rel:
            assert list(PGW_ROOT.glob(rel)), rel
        else:
            assert (PGW_ROOT / rel).exists(), rel
    for chart in ("pgw-gateway", "pgw-factory", "pgw-service"):
        assert (DEPLOY / "helm" / chart / "ci" / "ci-values.yaml").is_file()


# ---- shell scripts -------------------------------------------------------------------------------
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts(script: Path) -> None:
    text = script.read_text()
    assert text.startswith("#!/bin/sh") and "set -eu" in text
    subprocess.run(["sh", "-n", str(script)], check=True)  # noqa: S603,S607
    assert script.stat().st_mode & 0o111, f"{script.name} is not executable"


# ---- systemd -------------------------------------------------------------------------------------
def _unit(path: Path) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    section = ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            out[section] = {}
            continue
        k, _, v = line.partition("=")
        out[section].setdefault(k, []).append(v)
    return out


@pytest.mark.parametrize("role", ["gateway", "factory"])
def test_systemd_units(role: str) -> None:
    u = _unit(DEPLOY / "systemd" / f"pgw-{role}.service")
    svc = u["Service"]
    assert svc["User"] == ["pgw"] and svc["NoNewPrivileges"] == ["yes"] and svc["ProtectSystem"] == ["strict"]
    assert svc["CapabilityBoundingSet"] == [""] and svc["UMask"] == ["0077"]
    assert "/opt/pgw/bin/pgw" in svc["ExecStart"][0]
    assert svc["EnvironmentFile"] == [f"/etc/pgw/{role}.env"]
    assert u["Install"]["WantedBy"] == ["multi-user.target"]
    creds = {c.split(":")[0] for c in svc.get("LoadCredential", [])}
    assert "kek" in creds
    env_names = [e.split("=")[0] for e in svc.get("Environment", [])]
    assert unknown_env_names(env_names) == []
    example = (DEPLOY / "systemd" / f"{role}.env.example").read_text()
    names = [line.split("=")[0].lstrip("# ").strip() for line in example.splitlines()
             if "=" in line and line.lstrip("# ").startswith("PGW_")]
    assert names and unknown_env_names(names) == []
    assert not re.search(r"(?i)(token|password|kek)=\S", example)
    if role == "factory":
        assert svc["PrivateNetwork"] == ["yes"]
    else:
        assert {"service-token", "admin-token"} <= creds
        assert any("pgw config render" in x for x in svc["ExecStartPre"])
        rendered = next(x.split()[-1] for x in svc["ExecStartPre"] if "config render" in x)
        assert f"PGW_GATEWAY_MODELS_FILE={rendered}" in svc["Environment"]


def test_models_template_example_parses(tmp_path: Path) -> None:
    from pgw.gateway.config import GatewayModelConfig

    text = (DEPLOY / "systemd" / "models.tmpl.yaml.example").read_text()
    assert "${file:/run/credentials/pgw-gateway.service/service-token}" in text
    data = yaml.safe_load(re.sub(r"\$\{file:[^}]+\}", "x", text).replace("<64-hex", "").replace(
        "fingerprint printed by pgw pack / pgw inspect>", ""))
    entry = dict(data[0])
    entry["fingerprint"] = "a" * 64
    m = GatewayModelConfig(**entry)
    assert m.mode == "masked" and m.service_url and m.service_url.startswith("https://")


# ---- VM / OVA --------------------------------------------------------------------------------------
def test_packer_template_references() -> None:
    text = (DEPLOY / "vmware" / "pgw-gateway.pkr.hcl").read_text()
    assert text.count("{") == text.count("}")
    assert 'output_format    = "ova"' in text
    assert "install.sh --role gateway" in text and "passwd -l pgw-build" in text
    assert "sensitive = true" in text
    assert (DEPLOY / "vmware" / "autoinstall" / "user-data.template").is_file()
    assert (DEPLOY / "vmware" / "pgw-ovf-config.sh").is_file()
    props = (DEPLOY / "vmware" / "ovf-properties.xml").read_text()
    ovf = (DEPLOY / "vmware" / "pgw-ovf-config.sh").read_text()
    for key in re.findall(r'ovf:key="([\w.]+)"', props):
        assert f"prop {key}" in ovf, key
    assert 'ovf:key="pgw.service_token" ovf:type="string" ovf:password="true"' in props


# ---- docs ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(("path", "needles"), [
    ("README.md", ["docker", "compose", "helm", "systemd", "vmware", "marketplace", "NetworkPolicy"]),
    ("compose/README.md", ["docker compose up", "demo API key", "refill"]),
    ("helm/README.md", ["pgw-gateway", "pgw-factory", "pgw-service", "existingSecret", "kubectl create secret"]),
    ("vmware/README.md", ["Packer", "OVF", "snapshot"]),
    ("marketplace/aws/README.md", ["PrivateLink", "AMI", "Metering"]),
    ("marketplace/azure/README.md", ["Private Link", "Kubernetes application", "Key Vault"]),
])
def test_docs(path: str, needles: list[str]) -> None:
    text = (DEPLOY / path).read_text()
    for n in needles:
        assert n in text, (path, n)
    for link in re.findall(r"\]\(((?!https?:)[^)#]+)\)", text):
        assert ((DEPLOY / path).parent / link).resolve().exists(), (path, link)


_OVF_NS = "http://schemas.dmtf.org/ovf/environment/1"


def _ovf_env(props: dict[str, str]) -> str:
    from xml.sax.saxutils import quoteattr

    items = "".join(f"<Property oe:key={quoteattr(k)} oe:value={quoteattr(v)}/>" for k, v in props.items())
    return (f'<Environment xmlns="{_OVF_NS}" xmlns:oe="{_OVF_NS}">'
            f'<PropertySection>{items}</PropertySection></Environment>')


_GOOD_OVF = {
    "pgw.service_url": "https://compute.example.net:8443",
    "pgw.service_cidr": "203.0.113.0/28",
    "pgw.model_name": "llama3-8b",
    "pgw.pack_dir": "llama3-8b",
    "pgw.fingerprint": "ab" * 32,
    "pgw.ring": "K40A12",
    "pgw.service_token": "pgwt-x",
    "pgw.admin_token": "a" * 32,
}


@pytest.mark.parametrize(("key", "bad"), [
    ("pgw.service_url", "http://plain.example.net"),
    ("pgw.service_url", "https://x\nservice_token: evil"),
    ("pgw.model_name", 'm"\n- name: injected'),
    ("pgw.pack_dir", "../../etc"),
    ("pgw.fingerprint", "not-hex"),
    ("pgw.ring", "K40A12\n  extra: 1"),
    ("pgw.service_cidr", "0.0.0.0/0 accept\n"),
])
def test_ovf_config_rejects_injection(tmp_path: Path, key: str, bad: str) -> None:
    """Bad OVF values stop the first-boot script before it writes anything."""
    if shutil.which("python3") is None:
        pytest.skip("python3 not on PATH")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (tmp_path / "ovf.xml").write_text(_ovf_env({**_GOOD_OVF, key: bad}))
    tool = bindir / "vmtoolsd"
    tool.write_text(f"#!/bin/sh\ncat '{tmp_path / 'ovf.xml'}'\n")
    tool.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    r = subprocess.run(["sh", str(DEPLOY / "vmware" / "pgw-ovf-config.sh")], env=env,  # noqa: S603,S607
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, r.stderr
    assert key in r.stderr
    assert "evil" not in r.stderr and "pgwt-x" not in r.stderr
