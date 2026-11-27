"""The committed chat web app build is complete, CSP-clean and ships with the package."""

from __future__ import annotations

import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path

from pgw.chat.server import CSP, DEFAULT_STATIC_DIR

PGW_ROOT = Path(__file__).resolve().parents[2]
STATIC = PGW_ROOT / "src" / "pgw" / "chat" / "static"


class _Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.inline = False
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        for name in ("src", "href"):
            if a.get(name):
                self.refs.append(str(a[name]))
        if tag == "script":
            self._in_script = not a.get("src")
        if tag == "style" or a.get("style") is not None or any(k.startswith("on") for k in a):
            self.inline = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.inline = True


def test_static_dir_is_the_committed_build() -> None:
    assert STATIC.resolve() == DEFAULT_STATIC_DIR.resolve()


def test_index_references_only_local_existing_assets_and_nothing_inline() -> None:
    p = _Refs()
    p.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    assert p.refs
    for ref in p.refs:
        assert ref.startswith("/") and not ref.startswith("//"), f"external or relative reference {ref}"
        assert (STATIC / ref.lstrip("/")).is_file(), f"missing asset {ref}"
    assert not p.inline, "inline script/style would violate the CSP"
    assert "script-src 'self'" in CSP and "'unsafe-inline'" not in CSP and "'unsafe-eval'" not in CSP


def test_assets_are_hashed_and_free_of_mock_code_and_external_urls() -> None:
    assets = sorted((STATIC / "assets").iterdir())
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    for a in assets:
        assert re.fullmatch(r"[\w.-]+-[\w-]{8}\.(js|css|woff2)", a.name), f"unhashed asset {a.name}"
        if a.suffix in (".js", ".css"):
            text = a.read_text(encoding="utf-8")
            assert "sourceMappingURL" not in text
            assert "Masking keeps your words on this computer" not in text, "mock data in the production build"
            if a.suffix == ".css":
                assert a.name in index
                assert not re.search(r"url\(\s*['\"]?(https?:)?//", text), "a stylesheet loads an external URL"


def test_app_source_never_injects_html() -> None:
    """React's own bundle mentions innerHTML; our sources must never use it (answers render as elements only)."""
    for f in (PGW_ROOT / "chat" / "src").rglob("*.ts*"):
        if f.name.endswith(".test.ts") or f.name.endswith(".test.tsx"):
            continue
        text = f.read_text(encoding="utf-8")
        for bad in ("dangerouslySetInnerHTML", "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write",
                    "eval(", "new Function"):
            assert bad not in text, f"{f.name} uses {bad}"


def test_fonts_ship_with_their_licences() -> None:
    fonts = {a.name.split("-")[0] for a in (STATIC / "assets").glob("*.woff2")}
    assert {"ibm", "source"} <= fonts
    for lic in ("LICENSE-IBMPlexSans.txt", "LICENSE-SourceSerif4.txt", "LICENSE-IBMPlexMono.txt"):
        assert "SIL OPEN FONT LICENSE" in (STATIC / "licenses" / lic).read_text(encoding="utf-8")
        assert (PGW_ROOT / "src" / "pgw" / "design" / "fonts" / lic).is_file()


def test_package_data_ships_the_chat_app() -> None:
    meta = tomllib.loads((PGW_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "chat/static/**/*" in meta["tool"]["setuptools"]["package-data"]["pgw"]
