"""Dockerfiles, entrypoint and healthcheck (deploy/docker): static lint plus behaviour tests."""

from __future__ import annotations

import http.server
import os
import re
import shlex
import stat
import subprocess
import sys
import threading
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from .conftest import DEPLOY, PGW_ROOT, free_port
from .deploy_utils import unknown_env_names

DOCKER = DEPLOY / "docker"
IMAGES: dict[str, dict[str, Any]] = {
    "gateway": {"extras": {"gateway"}, "cmd": ["pgw", "gateway", "serve"], "health": True, "gpu": False},
    "factory": {"extras": {"factory"}, "cmd": ["pgw", "factory", "run"], "health": False, "gpu": False},
    "factory-cuda": {"extras": {"factory", "gpu"}, "cmd": ["pgw", "factory", "run"], "health": False, "gpu": True},
    "service-cuda": {"extras": {"service", "gpu"}, "cmd": ["pgw", "service", "serve"], "health": True, "gpu": True},
    "service-cpu": {"extras": {"service"}, "cmd": ["pgw", "service", "serve"], "health": True, "gpu": False},
}


def _instructions(path: Path) -> list[tuple[str, str]]:
    """``(INSTRUCTION, args)`` with line continuations joined and comments dropped."""
    out: list[tuple[str, str]] = []
    buf = ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        word, _, rest = buf.partition(" ")
        out.append((word.upper(), rest.strip()))
        buf = ""
    assert not buf, f"{path.name}: dangling line continuation"
    return out


def _stages(instrs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    stages: list[list[tuple[str, str]]] = []
    for ins in instrs:
        if ins[0] == "FROM":
            stages.append([])
        if stages:
            stages[-1].append(ins)
    return stages


def _env_names(args: str) -> list[str]:
    return re.findall(r"(?:^|\s)([A-Z_][A-Z0-9_]*)=", args)


def test_every_dockerfile_is_covered() -> None:
    found = {p.name.removesuffix(".Dockerfile") for p in DOCKER.glob("*.Dockerfile")}
    assert found == set(IMAGES)


@pytest.mark.parametrize("image", sorted(IMAGES))
def test_dockerfile_lint(image: str) -> None:
    spec = IMAGES[image]
    path = DOCKER / f"{image}.Dockerfile"
    text = path.read_text()
    assert text.startswith("# syntax=docker/dockerfile:1")
    instrs = _instructions(path)
    args = {a.split("=")[0]: a.split("=", 1)[1] for k, a in instrs if k == "ARG" and "=" in a}
    assert args["PYTHON_IMAGE"].startswith("python:3.12-") and ":latest" not in args["PYTHON_IMAGE"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", args["TORCH_VERSION"])
    stages = _stages(instrs)
    assert len(stages) == 2, "multi-stage build: build + runtime"
    build, runtime = stages
    # torch comes from the right index
    torch_install = next(a for k, a in build if k == "RUN" and "torch==" in a)
    assert ("TORCH_CUDA" in torch_install) == spec["gpu"]
    # the package is installed with the role's extras, which exist in pyproject
    pyproject = tomllib.loads((PGW_ROOT / "pyproject.toml").read_text())
    extras_available = set(pyproject["project"]["optional-dependencies"])
    install = next(a for k, a in build if k == "RUN" and 'pip install ".[' in a)
    extras = set(re.search(r'"\.\[([a-z,]+)\]"', install).group(1).split(","))  # type: ignore[union-attr]
    assert spec["extras"] <= extras and extras <= extras_available | {"pack"}
    # runtime: non-root, entrypoint, command, no build tools, no secrets
    users = [a for k, a in runtime if k == "USER"]
    assert users == ["10001:10001"]
    assert ("ENTRYPOINT", '["pgw-entrypoint"]') in runtime
    cmd = next(a for k, a in runtime if k == "CMD")
    assert shlex.split(cmd.replace("[", "").replace("]", "").replace(",", " ")) == spec["cmd"]
    hc = [a for k, a in runtime if k == "HEALTHCHECK"]
    assert hc and (("pgw-healthcheck" in hc[0]) == spec["health"])
    assert not re.search(r"(?i)(password|secret|token|kek)\s*=", " ".join(a for k, a in runtime if k == "ENV"))
    # COPY sources exist and are allowed by the dockerignore allowlist
    ignore = (DOCKER / f"{image}.Dockerfile.dockerignore").read_text().splitlines()
    allowed = [line[1:].rstrip("/") for line in ignore if line.startswith("!")]
    for k, a in instrs:
        if k != "COPY" or "--from=" in a:
            continue
        parts = [p for p in a.split() if not p.startswith("--")]
        for src in parts[:-1]:
            assert (PGW_ROOT / src).exists(), f"{image}: COPY source {src} missing"
            assert any(src.rstrip("/") == al or src.startswith(al + "/") for al in allowed), (image, src)
    # PGW_* env names are real settings
    names = [n for k, a in runtime if k == "ENV" for n in _env_names(a)]
    assert unknown_env_names(names) == []
    if spec["gpu"]:
        assert "NVIDIA_VISIBLE_DEVICES" in names


def test_unknown_env_detector() -> None:
    assert unknown_env_names(["PGW_SERVICE_PORTT", "PGW_GATEWAY_TLS__CERT_FILE", "PGW_GATEWAY_TLS__NOPE",
                              "PGW_LOG_LEVEL", "PGW_RENDER", "HOME"]) == ["PGW_SERVICE_PORTT", "PGW_GATEWAY_TLS__NOPE"]


def test_dockerignore_excludes_secrets_and_state() -> None:
    for p in DOCKER.glob("*.dockerignore"):
        lines = p.read_text().splitlines()
        assert "*" in lines, p.name  # allowlist style
        assert not any(line.startswith("!") and ("tests" in line or "masks" in line) for line in lines)


# ---- entrypoint ----------------------------------------------------------------------------------
def _pgw_bin_dir() -> str:
    return str(Path(sys.executable).parent)


def test_entrypoint_syntax_and_mode() -> None:
    ep = DOCKER / "entrypoint.sh"
    subprocess.run(["sh", "-n", str(ep)], check=True)  # noqa: S603,S607
    assert ep.read_text().startswith("#!/bin/sh")


def test_entrypoint_renders_and_execs(tmp_path: Path) -> None:
    (tmp_path / "token").write_text("pgwt-secret\n")
    tmpl = tmp_path / "models.tmpl.json"
    tmpl.write_text('[{"service_token": "${file:token}", "url": "${env:SVC_URL}"}]')
    out = tmp_path / "run" / "models.json"
    env = {**os.environ, "PATH": _pgw_bin_dir() + os.pathsep + os.environ.get("PATH", ""),
           "PGW_RENDER": f"{tmpl}:{out}", "SVC_URL": "https://svc:8443"}
    r = subprocess.run(["sh", str(DOCKER / "entrypoint.sh"), "sh", "-c", "echo started $0", "ok"],  # noqa: S603,S607
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "started ok"
    assert out.read_text() == '[{"service_token": "pgwt-secret", "url": "https://svc:8443"}]'
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert "pgwt-secret" not in r.stdout + r.stderr


def test_entrypoint_errors(tmp_path: Path) -> None:
    env = {**os.environ, "PATH": _pgw_bin_dir() + os.pathsep + os.environ.get("PATH", ""), "PGW_RENDER": "nocolon"}
    r = subprocess.run(["sh", str(DOCKER / "entrypoint.sh"), "true"], env=env,  # noqa: S603,S607
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "bad PGW_RENDER" in r.stderr
    tmpl = tmp_path / "t.json"
    tmpl.write_text("${env:DEFINITELY_UNSET_VAR_X}")
    env["PGW_RENDER"] = f"{tmpl}:{tmp_path / 'o.json'}"
    r = subprocess.run(["sh", str(DOCKER / "entrypoint.sh"), "echo", "never"], env=env,  # noqa: S603,S607
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "never" not in r.stdout and "DEFINITELY_UNSET_VAR_X" in r.stderr
    r = subprocess.run(["sh", str(DOCKER / "entrypoint.sh"), "echo", "plain"],  # noqa: S603,S607
                       env={k: v for k, v in env.items() if k != "PGW_RENDER"}, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "plain"


# ---- healthcheck ---------------------------------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:
        self.send_response(self.status if self.path == "/readyz" else 200)
        self.end_headers()
        self.wfile.write(b"x")

    def log_message(self, *a: object) -> None:
        pass


@pytest.fixture
def http_server() -> Iterator[tuple[int, type[_Handler]]]:
    port = free_port()

    class H(_Handler):
        pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port, H
    srv.shutdown()


def _health(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(DOCKER / "healthcheck.py"), *args],  # noqa: S603
                          env={**{k: v for k, v in os.environ.items() if not k.startswith("PGW_")}, **env},
                          capture_output=True, text=True, timeout=30)


def test_healthcheck(http_server: tuple[int, type[_Handler]]) -> None:
    port, handler = http_server
    env = {"PGW_GATEWAY_PORT": str(port)}
    assert _health(["gateway"], env).returncode == 0
    handler.status = 503
    assert _health(["gateway"], env).returncode == 1
    assert _health(["--live", "gateway"], env).returncode == 0
    assert _health(["service"], {"PGW_SERVICE_PORT": str(free_port())}).returncode == 1
    assert _health([], {}).returncode == 2
    assert _health(["nope"], {}).returncode == 2


def test_entrypoint_does_not_glob_render_paths(tmp_path: Path) -> None:
    """PGW_RENDER entries are split on commas only; a '*' must not expand to matching files."""
    (tmp_path / "a.tmpl").write_text("x")
    env = {**os.environ, "PATH": _pgw_bin_dir() + os.pathsep + os.environ.get("PATH", ""),
           "PGW_RENDER": f"{tmp_path}/*.tmpl:{tmp_path}/out"}
    r = subprocess.run(["sh", str(DOCKER / "entrypoint.sh"), "true"], env=env,  # noqa: S603,S607
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0  # the literal path '<dir>/*.tmpl' does not exist
    assert not (tmp_path / "out").exists()


def test_healthcheck_bad_port_is_unhealthy_not_a_crash() -> None:
    r = _health(["gateway"], {"PGW_GATEWAY_PORT": "eighty"})
    assert r.returncode == 1 and "invalid PGW_GATEWAY_PORT" in r.stderr and "Traceback" not in r.stderr
    r = _health(["gateway"], {"PGW_GATEWAY_PORT": "70000"})
    assert r.returncode == 1
