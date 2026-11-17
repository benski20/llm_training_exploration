"""Fixtures and Markdown helpers for the documentation tests (``pgw/docs``)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PGW_ROOT = Path(__file__).resolve().parents[2]
DOCS = PGW_ROOT / "docs"

#: Documents the docs component must ship (path relative to docs/).
REQUIRED_DOCS = (
    "README.md",
    "glossary.md",
    "security/whitepaper.md",
    "security/threat-model.md",
    "operations/install-on-prem.md",
    "operations/install-customer-cloud.md",
    "operations/air-gapped-factory.md",
    "operations/sizing.md",
    "api/gateway-api.md",
    "compliance/faq.md",
    "compliance/audit-readiness.md",
)


def all_markdown() -> dict[str, str]:
    """``{relative path: text}`` of every Markdown file under docs/."""
    return {p.relative_to(DOCS).as_posix(): p.read_text(encoding="utf-8") for p in sorted(DOCS.rglob("*.md"))}


def strip_code(text: str) -> str:
    """Text with fenced code blocks removed (inline code kept)."""
    return re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.S | re.M)


def code_blocks(text: str, langs: tuple[str, ...] | None = None) -> list[str]:
    """Contents of fenced code blocks, optionally only those whose info string is in ``langs``."""
    out = []
    for m in re.finditer(r"^```([^\n]*)\n(.*?)^```[ \t]*$", text, flags=re.S | re.M):
        lang = m.group(1).strip()
        if langs is None or lang in langs:
            out.append(m.group(2))
    return out


def github_slug(heading: str) -> str:
    """Anchor GitHub generates for a heading."""
    h = heading.strip().lower()
    h = re.sub(r"[`*_]", lambda m: "_" if m.group(0) == "_" else "", h)
    h = re.sub(r"[^\w\- ]", "", h, flags=re.UNICODE)
    return h.replace(" ", "-")


def anchors(text: str) -> set[str]:
    """All heading anchors of a document (duplicates get -1, -2 suffixes like GitHub)."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    for m in re.finditer(r"^#{1,6} +(.+?) *#*$", strip_code(text), flags=re.M):
        s = github_slug(m.group(1))
        if s in seen:
            seen[s] += 1
            s = f"{s}-{seen[s]}"
        else:
            seen[s] = 0
        out.add(s)
    return out


def section(text: str, heading_prefix: str) -> str:
    """Body of the first ``##`` section whose heading starts with ``heading_prefix``."""
    m = re.search(rf"^## +{re.escape(heading_prefix)}.*?$(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    assert m, f"section {heading_prefix!r} not found"
    return m.group(1)


def table_rows(text: str) -> list[list[str]]:
    """Cells of every Markdown table body row in ``text`` (header and separator rows skipped)."""
    rows: list[list[str]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[\s:|-]+\|$", line):
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if re.match(r"^\|[\s:|-]+\|$", nxt):  # header row
            continue
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def unticks(cell: str) -> list[str]:
    """Backticked fragments in a table cell."""
    return re.findall(r"`([^`]+)`", cell)


@pytest.fixture(scope="session")
def docs() -> dict[str, str]:
    """All Markdown documents."""
    return all_markdown()


@pytest.fixture(scope="session")
def measurements() -> dict[str, Any]:
    """Parsed ``docs/measurements.json``."""
    return json.loads((DOCS / "measurements.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sizing() -> ModuleType:
    """The sizing calculator module (``docs/tools/pgw_sizing.py``), imported by path."""
    path = DOCS / "tools" / "pgw_sizing.py"
    spec = importlib.util.spec_from_file_location("pgw_docs_sizing", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pgw_docs_sizing"] = mod
    spec.loader.exec_module(mod)
    return mod
