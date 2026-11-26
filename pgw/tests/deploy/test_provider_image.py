"""Static checks of the provider images (``deploy/docker/provider*.Dockerfile``): pinned bases and locks, builder
stage for the native extension, non-root service, health check, a build context that ships nothing else, and the
GHCR workflow. The images themselves are built by ``.github/workflows/provider-image.yml`` (and were built locally
with ``docker build``; see the report)."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

import pytest

PGW = Path(__file__).resolve().parents[2]
DOCKER = PGW / "deploy" / "docker"
IMAGES = {"provider.Dockerfile": "provider-cuda", "provider-cpu.Dockerfile": "provider-cpu"}


def _allowed(ignore: Path, rel: str) -> bool:
    """Docker's .dockerignore semantics for the patterns used here (last match wins)."""
    ok = True
    for raw in ignore.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        pat = line[1:] if neg else line
        pat = pat.rstrip("/")
        hit = (fnmatch.fnmatch(rel, pat) or rel.startswith(pat + "/") or fnmatch.fnmatch(rel, pat + "/*")
               or (pat.startswith("**/") and fnmatch.fnmatch(rel.split("/")[-1], pat[3:])))
        if hit:
            ok = neg
    return ok


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_provider_dockerfile_contract(name: str) -> None:
    text = (DOCKER / name).read_text()
    assert re.search(r"^ARG (CUDA|PYTHON)_IMAGE=\S+@sha256:[0-9a-f]{64}$", text, re.M), "base pinned by digest"
    assert "AS build" in text and "AS runtime" in text and "COPY --from=build /opt/pgw /opt/pgw" in text
    assert f"requirements/{IMAGES[name]}.txt" in text and "--require-hashes" in text
    assert (PGW / "requirements" / f"{IMAGES[name]}.txt").is_file()
    assert "ARG RUST_TOOLCHAIN=1." in text and "build_fast.py --site" in text and "cargo fetch --locked" in text
    assert "native.fast is not None" in text  # the build fails without the native extension
    # the Rust installer is verified (security review 2026-11, S14): pinned rustup-init, sha256 per architecture
    assert "sh.rustup.rs" not in text and "rustup-init.sh" not in text
    assert re.search(r"^ARG RUSTUP_VERSION=\d+\.\d+\.\d+$", text, re.M)
    for arch in ("X86_64", "AARCH64"):
        assert re.search(rf"^ARG RUSTUP_INIT_SHA256_{arch}=[0-9a-f]{{64}}$", text, re.M), arch
    assert 'sha256sum -c -' in text and text.index("sha256sum -c -") < text.index("rustup-init -y")
    assert "HEALTHCHECK" in text and 'ENTRYPOINT ["/usr/bin/tini", "--", "pgw-provider-entrypoint"]' in text
    runtime = text[text.index("AS runtime"):]
    assert "rustup" not in runtime and "cargo" not in runtime  # no toolchain in the runtime image
    ignore = DOCKER / f"{name}.dockerignore"
    for src in re.findall(r"^COPY (?:--\S+ )*(.+?) \S+$", text, re.M):
        for part in src.split():
            if part.startswith("${") or part.startswith("/"):
                continue
            assert _allowed(ignore, part.rstrip("/")), f"{part} is not in the build context allowlist"
    for secret in ("tests", "packs", "rust/target", "deploy/modal/launch.py", ".env", "state/accounts.db"):
        assert not _allowed(ignore, secret), f"{secret} must not reach the build context"


def test_provider_entrypoint_drops_root() -> None:
    sh = (DOCKER / "provider-entrypoint.sh").read_text()
    assert "setpriv --reuid=10001 --regid=10001 --clear-groups --inh-caps=-all --no-new-privs" in sh
    assert sh.count("exec ") >= 3
    if subprocess.run(["sh", "-n", str(DOCKER / "provider-entrypoint.sh")], check=False).returncode != 0:
        pytest.fail("provider-entrypoint.sh: shell syntax error")
    assert subprocess.run(["bash", "-n", str(DOCKER / "provider-smoke.sh")], check=False).returncode == 0


def test_ghcr_workflow() -> None:
    yml = (PGW / ".github" / "workflows" / "provider-image.yml").read_text()
    assert 'tags: ["pgw-provider-v*"]' in yml and "packages: write" in yml
    assert "provider-smoke.sh" in yml and "attest-build-provenance" in yml
    assert "secrets.GITHUB_TOKEN" in yml and "RUNPOD_API_KEY" not in yml
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(yml)
    assert set(doc["jobs"]) == {"image"}
