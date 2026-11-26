"""The committed console build is complete and ships with the package and the gateway image.

``src/pgw/console/static/`` is the output of ``npm run build`` in ``pgw/console`` (see
``console/README.md``). These checks fail when the build is missing, half-copied, references assets
that do not exist, or would be blocked by the console's own Content-Security-Policy.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import pytest

from pgw.console.routes import CONSOLE_SECURITY_HEADERS, DEFAULT_STATIC_DIR

PGW_ROOT = Path(__file__).resolve().parents[2]
STATIC = PGW_ROOT / "src" / "pgw" / "console" / "static"


class _Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.scripts: list[dict[str, str | None]] = []
        self.inline_script = False
        self.inline_style = False
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        for name in ("src", "href"):
            if a.get(name):
                self.refs.append(str(a[name]))
        if tag == "script":
            self.scripts.append(a)
            self._in_script = not a.get("src")
        if tag == "style" or a.get("style") is not None or any(k.startswith("on") for k in a):
            self.inline_style = True  # inline style or event handler: blocked by the CSP

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.inline_script = True


def test_default_static_dir_is_the_committed_build() -> None:
    assert STATIC.resolve() == DEFAULT_STATIC_DIR.resolve()


def test_index_exists_and_references_only_existing_assets() -> None:
    index = STATIC / "index.html"
    assert index.is_file(), "console not built: run `npm run build` in pgw/console (see console/README.md)"
    p = _Refs()
    p.feed(index.read_text(encoding="utf-8"))
    local = [r for r in p.refs if not re.match(r"^[a-z]+:", r)]
    assert local, "index.html references no assets"
    for ref in local:
        assert ref.startswith("/console/"), f"asset outside the /console/ base: {ref}"
        target = STATIC / ref.removeprefix("/console/").split("?")[0].split("#")[0]
        assert target.is_file(), f"index.html references a missing asset: {ref}"
    assert [s for s in p.scripts if s.get("src", "").startswith("/console/assets/")], "no application script"
    assert not p.inline_script and not p.inline_style, "inline script/style would violate the console CSP"
    assert "script-src 'self'" in CONSOLE_SECURITY_HEADERS["Content-Security-Policy"]


def test_assets_are_hashed_and_referenced() -> None:
    assets = sorted((STATIC / "assets").iterdir())
    assert assets
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    for a in assets:
        assert re.fullmatch(r"[\w.-]+-[\w-]{8}\.(js|css|svg|png|woff2?)", a.name), f"unhashed asset name {a.name}"
    referenced = {a.name for a in assets if a.name in index}
    # every stylesheet/entry script is referenced; chunks imported by the entry script are fine
    assert {a.name for a in assets if a.suffix == ".css"} <= referenced
    for a in assets:
        if a.suffix == ".js" and a.name not in referenced:
            assert any(a.name in b.read_text(encoding="utf-8") for b in assets if b.suffix == ".js" and b != a), \
                f"stale asset {a.name}: not referenced by index.html or another chunk (rebuild the console)"
    for a in (STATIC / "assets").glob("*.js"):
        text = a.read_text(encoding="utf-8")
        assert "sourceMappingURL" not in text, f"{a.name} references a source map that is not shipped"


def test_package_data_and_image_ship_the_console() -> None:
    meta = tomllib.loads((PGW_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pd = meta["tool"]["setuptools"]["package-data"]["pgw"]
    assert "console/static/**/*" in pd
    ignore = (PGW_ROOT / "deploy" / "docker" / "gateway.Dockerfile.dockerignore").read_text(encoding="utf-8")
    assert "!src/" in ignore.splitlines() and "static" not in ignore
    dockerfile = (PGW_ROOT / "deploy" / "docker" / "gateway.Dockerfile").read_text(encoding="utf-8")
    assert "COPY src ./src" in dockerfile and 'pip install --no-deps --no-build-isolation ".[gateway' in dockerfile


@pytest.mark.slow
def test_wheel_contains_the_console(tmp_path: Path) -> None:
    """Build the wheel the gateway image installs and check every static file is inside."""
    src = tmp_path / "src-tree"
    src.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(PGW_ROOT / name, src / name)
    shutil.copytree(PGW_ROOT / "src", src / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.so"))
    out = tmp_path / "dist"
    r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-q",  # noqa: S603
                        "-w", str(out), str(src)], capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and "No module named pip" in r.stderr:  # pragma: no cover
        pytest.skip("pip is not available")
    assert r.returncode == 0, r.stderr[-3000:]
    (wheel,) = out.glob("pgw_sdk-*.whl")  # the distribution is pgw-sdk (wheel names normalise "-" to "_")
    names = set(zipfile.ZipFile(wheel).namelist())
    for f in STATIC.rglob("*"):
        if f.is_file():
            assert f"pgw/console/static/{f.relative_to(STATIC).as_posix()}" in names, f"wheel misses {f.name}"
