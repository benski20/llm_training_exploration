"""Packaging metadata: one version source, the native extensions the code imports, sdist contents, extras."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pgw

ROOT = Path(__file__).resolve().parents[2]
PY = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_version_single_source() -> None:
    assert PY["project"]["dynamic"] == ["version"]
    assert PY["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "pgw.__version__"}
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", pgw.__version__)
    # setuptools reads the attribute statically: it must stay a plain string literal
    tree = ast.parse((ROOT / "src/pgw/__init__.py").read_text())
    assigns = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
    assert len(assigns) == 1 and isinstance(assigns[0].value, ast.Constant)


def test_distribution_metadata() -> None:
    p = PY["project"]
    assert p["name"] == "pgw-sdk"  # `pgw` on PyPI is an unrelated project
    assert p["license"] == "LicenseRef-Proprietary"
    assert set(p["license-files"]) == {"LICENSE", "THIRD_PARTY_NOTICES.txt"}
    assert all((ROOT / f).is_file() for f in p["license-files"])
    assert p["requires-python"] == ">=3.11"
    assert (ROOT / "src/pgw/py.typed").is_file()
    assert p["scripts"]["pgw"] == "pgw.cli.main:main"


def test_client_install_is_the_base_install() -> None:
    deps = {re.split(r"[<>=\[;]", d)[0] for d in PY["project"]["dependencies"]}
    # what the in-process gateway imports (tokenizer, chat template, provider connection)
    assert {"torch", "numpy", "transformers", "tokenizers", "jinja2", "httpx"} <= deps
    # server-only packages stay in extras
    assert not deps & {"fastapi", "uvicorn", "hypercorn", "sse-starlette", "prometheus-client", "nvidia-ml-py"}
    extras = PY["project"]["optional-dependencies"]
    assert extras["client"] == []
    assert all("pgw[" not in d for ds in extras.values() for d in ds)  # self-references use pgw-sdk[...]


def test_native_extensions_match_imports_and_crates() -> None:
    setup = (ROOT / "setup.py").read_text()
    exts = dict(re.findall(r'\("(pgw_\w+)", "(rust/[\w-]+/Cargo\.toml)"\)', setup))
    assert set(exts) == {"pgw_fast", "pgw_ntt", "pgw_pks"}
    src = "\n".join(p.read_text() for p in (ROOT / "src/pgw").rglob("*.py"))
    for mod, cargo in exts.items():
        assert f"import {mod}" in src, mod
        toml = tomllib.loads((ROOT / cargo).read_text())
        assert toml["lib"]["name"] == mod and "cdylib" in toml["lib"]["crate-type"]
        pyo3 = toml["dependencies"]["pyo3"]
        assert "abi3-py311" in pyo3["features"], cargo  # one wheel per platform for 3.11+
    assert 'py_limited_api": "cp311"' in setup


def test_sdist_manifest_covers_the_rust_workspace() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text()
    members = tomllib.loads((ROOT / "rust/Cargo.toml").read_text())["workspace"]["members"]
    for m in members:
        assert f"recursive-include rust/{m} " in manifest, m
    assert "rust/Cargo.lock" in manifest and "prune rust/target" in manifest


def test_third_party_notices_cover_linked_crates() -> None:
    text = (ROOT / "THIRD_PARTY_NOTICES.txt").read_text()
    lock = (ROOT / "rust/Cargo.lock").read_text()
    for crate in ("pyo3", "numpy", "rustls", "ring", "aes-gcm", "chacha20", "zeroize"):
        ver = re.search(rf'name = "{re.escape(crate)}"\nversion = "([^"]+)"', lock)
        assert ver and f"{crate} {ver.group(1)} " in text, crate
