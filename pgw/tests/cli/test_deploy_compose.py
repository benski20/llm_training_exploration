"""docker-compose demo (deploy/compose): structure, trust boundaries, and the demo flow without Docker.

``test_demo_flow_without_docker`` runs the real ``demo-init.sh`` against a temporary root with a
tiny checkpoint, then starts the service and the gateway with the *environment from the compose
file* (paths remapped), through the image entrypoint, over TLS with the demo CA, and sends a chat
completion. It is the compose demo minus the containers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from .conftest import DEPLOY, PGW_ROOT, free_port
from .deploy_utils import unknown_env_names

COMPOSE = DEPLOY / "compose" / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(COMPOSE.read_text())
    return data


def _volumes(svc: dict[str, Any]) -> dict[str, str]:
    """``{volume or host path: container path[:ro]}``."""
    out = {}
    for v in svc.get("volumes", []):
        src, _, dst = v.partition(":")
        out[src] = dst
    return out


def test_services_and_order(compose: dict[str, Any]) -> None:
    svcs = compose["services"]
    assert set(svcs) == {"init", "factory", "factory-refill", "service", "gateway"}
    assert svcs["factory"]["depends_on"]["init"]["condition"] == "service_completed_successfully"
    assert svcs["service"]["depends_on"]["init"]["condition"] == "service_completed_successfully"
    assert svcs["gateway"]["depends_on"] == {
        "factory": {"condition": "service_completed_successfully"},
        "service": {"condition": "service_healthy"},
    }
    assert svcs["factory-refill"]["profiles"] == ["refill"]
    for name, svc in svcs.items():
        if "build" in svc:
            assert (PGW_ROOT / svc["build"]["dockerfile"]).is_file(), name
            assert (COMPOSE.parent / svc["build"]["context"]).resolve() == PGW_ROOT


def test_trust_boundaries(compose: dict[str, Any]) -> None:
    svcs = compose["services"]
    # the compute service: packs (ro) and its own token hashes only, internal network only
    sv = _volumes(svcs["service"])
    assert set(sv) == {"packs", "svc-secrets"}
    assert all(dst.endswith(":ro") for dst in sv.values())
    assert svcs["service"]["networks"] == ["compute"]
    assert compose["networks"]["compute"]["internal"] is True
    assert not [k for k in svcs["service"]["environment"] if k.startswith(("PGW_MASKSTORE", "PGW_GATEWAY"))]
    # the factory: no network at all
    for f in ("factory", "factory-refill"):
        assert svcs[f]["network_mode"] == "none"
        assert "svc-secrets" not in _volumes(svcs[f]) and "gw-secrets" not in _volumes(svcs[f])
    # the gateway: published on localhost only, secrets read-only
    gv = _volumes(svcs["gateway"])
    assert "svc-secrets" not in gv and gv["gw-secrets"].endswith(":ro") and gv["kek"].endswith(":ro")
    assert all(p.startswith("127.0.0.1:") for p in svcs["gateway"]["ports"])
    assert set(svcs["gateway"]["networks"]) == {"compute", "frontend"}
    # hardening on long-running containers
    for name in ("service", "gateway", "factory", "factory-refill"):
        s = svcs[name]
        assert s["read_only"] is True and s["cap_drop"] == ["ALL"], name
        assert "no-new-privileges:true" in s["security_opt"]
        assert "user" not in s  # image default: uid 10001


def test_environment_names_are_settings(compose: dict[str, Any]) -> None:
    for name, svc in compose["services"].items():
        env = svc.get("environment", {})
        assert isinstance(env, dict), name
        assert unknown_env_names([k for k in env if not k.startswith("PGW_DEMO_")]) == [], name


def test_no_secrets_in_compose_files() -> None:
    for p in (DEPLOY / "compose").iterdir():
        if p.is_file():
            text = p.read_text()
            assert "BEGIN PRIVATE KEY" not in text and "pgwt-" not in text.replace("pgwt-\" ", ""), p.name


def test_scripts_syntax() -> None:
    subprocess.run(["sh", "-n", str(DEPLOY / "compose" / "demo-init.sh")], check=True)  # noqa: S603,S607
    subprocess.run([sys.executable, "-m", "py_compile", str(DEPLOY / "compose" / "demo_certs.py")],  # noqa: S603
                   check=True)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_docker_compose_config_validates() -> None:
    r = subprocess.run(["docker", "compose", "-f", str(COMPOSE), "config", "-q"],  # noqa: S603,S607
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0 and "is not a docker command" in r.stderr:
        pytest.skip("docker compose plugin not installed")
    assert r.returncode == 0, r.stderr


def test_demo_certs(tmp_path: Path) -> None:
    from cryptography import x509

    r = subprocess.run([sys.executable, str(DEPLOY / "compose" / "demo_certs.py"), str(tmp_path), "service",  # noqa: S603
                        "127.0.0.1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ca = x509.load_pem_x509_certificate((tmp_path / "ca.crt").read_bytes())
    cert = x509.load_pem_x509_certificate((tmp_path / "tls.crt").read_bytes())
    cert.verify_directly_issued_by(ca)
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "service" in san.get_values_for_type(x509.DNSName)
    assert (tmp_path / "tls.key").stat().st_mode & 0o077 == 0
    before = (tmp_path / "tls.crt").read_bytes()
    subprocess.run([sys.executable, str(DEPLOY / "compose" / "demo_certs.py"), str(tmp_path), "x"],  # noqa: S603
                   check=True)
    assert (tmp_path / "tls.crt").read_bytes() == before  # idempotent


# ---- the whole demo, minus Docker ---------------------------------------------------------------
def _remap(value: str, root: Path) -> str:
    return (value.replace("/var/lib/pgw", str(root))
            .replace("/etc/pgw/tenants", str(root / "svc-secrets"))
            .replace("/run/pgw", str(root / "run")))


def _wait(url: str, proc: subprocess.Popen[bytes], verify: Any = True, timeout: float = 90.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise AssertionError(f"exited {proc.returncode}:\n{out[-4000:]}")
        try:
            if httpx.get(url, timeout=2.0, verify=verify).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise AssertionError(f"{url} not ready")


@pytest.mark.slow
def test_demo_flow_without_docker(compose: dict[str, Any], tiny_hf_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "pgw"
    bindir = str(Path(sys.executable).parent)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    base_env["PATH"] = bindir + os.pathsep + base_env.get("PATH", "")
    svc_port, gw_port = free_port(), free_port()
    init_env = {**base_env, "PGW_DEMO_ROOT": str(root), "PGW_DEMO_MODEL": str(tiny_hf_dir),
                "PGW_DEMO_SERVICE_URL": f"https://localhost:{svc_port}", "PGW_DEMO_SERVICE_HOST": "localhost",
                "PGW_DEMO_CHOWN": "0"}
    r = subprocess.run(["sh", str(DEPLOY / "compose" / "demo-init.sh")], env=init_env,  # noqa: S603,S607
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-4000:]
    assert "demo API key" in r.stderr
    api_key = json.loads((root / "secrets" / "demo_api_key.json").read_text())["key"]
    # idempotent second run
    r2 = subprocess.run(["sh", str(DEPLOY / "compose" / "demo-init.sh")], env=init_env,  # noqa: S603,S607
                        capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0 and "demo API key" not in r2.stderr and "demo admin token" not in r2.stderr
    assert "demo admin token" in r.stderr
    admin_file = root / "secrets" / "admin_token"
    assert admin_file.stat().st_mode & 0o077 == 0
    # factory step, exactly as in the compose file (fewer slots)
    fac = compose["services"]["factory"]["command"]
    fac = [_remap(str(a), root) for a in fac]
    fac[fac.index("${PGW_DEMO_SLOTS:-2048}")] = "64"
    fac[fac.index("${PGW_DEMO_RING:-K32A12C27}")] = "K32A12C27"
    r = subprocess.run([sys.executable, "-m", "pgw.cli.main", *fac[1:]], env=base_env,  # noqa: S603
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-4000:]

    def env_of(name: str, port_var: str, port: int) -> dict[str, str]:
        env = dict(base_env)
        for k, v in compose["services"][name]["environment"].items():
            v = str(v).replace("${PGW_LOG_LEVEL:-INFO}", "INFO").replace("${PGW_DEMO_RING:-K32A12C27}", "K32A12C27")
            env[k] = _remap(v, root)
        env[port_var] = str(port)
        return env

    svc_env = env_of("service", "PGW_SERVICE_PORT", svc_port)
    svc_env["PGW_SERVICE_HOST"] = "127.0.0.1"
    gw_env = env_of("gateway", "PGW_GATEWAY_PORT", gw_port)
    gw_env["PGW_GATEWAY_HOST"] = "127.0.0.1"
    ep = str(DEPLOY / "docker" / "entrypoint.sh")
    svc = subprocess.Popen(["sh", ep, sys.executable, "-m", "pgw.cli.main", "service", "serve"],  # noqa: S603,S607
                           env=svc_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    gw = None
    try:
        import ssl

        ctx = ssl.create_default_context(cafile=str(root / "secrets" / "ca.crt"))  # strict X.509 checks (3.13)
        _wait(f"https://localhost:{svc_port}/readyz", svc, verify=ctx)
        hc = subprocess.run([sys.executable, str(DEPLOY / "docker" / "healthcheck.py"), "service"],  # noqa: S603
                            env=svc_env, capture_output=True, text=True, timeout=30)
        assert hc.returncode == 0, hc.stderr
        gw = subprocess.Popen(["sh", ep, sys.executable, "-m", "pgw.cli.main", "gateway", "serve"],  # noqa: S603,S607
                              env=gw_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _wait(f"http://127.0.0.1:{gw_port}/readyz", gw)
        assert (root / "run" / "models.json").stat().st_mode & 0o077 == 0
        hc = subprocess.run([sys.executable, str(DEPLOY / "docker" / "healthcheck.py"), "gateway"],  # noqa: S603
                            env=gw_env, capture_output=True, text=True, timeout=30)
        assert hc.returncode == 0, hc.stderr
        resp = httpx.post(f"http://127.0.0.1:{gw_port}/v1/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}"},
                          json={"model": "demo", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
                          timeout=120)
        assert resp.status_code == 200, resp.text
        assert resp.json()["usage"]["completion_tokens"] >= 1
        # operator console: served by the gateway image, sign-in with the generated admin token file
        page = httpx.get(f"http://127.0.0.1:{gw_port}/console/", timeout=30)
        assert page.status_code == 200 and "/console/assets/" in page.text
        login = httpx.post(f"http://127.0.0.1:{gw_port}/console/api/login",
                           json={"token": admin_file.read_text().strip()}, timeout=30)
        assert login.status_code == 200 and login.json()["role"] == "admin", login.text
    finally:
        for p in (gw, svc):
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    p.kill()
