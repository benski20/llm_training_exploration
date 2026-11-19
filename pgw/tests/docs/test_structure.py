"""Structure, links and hygiene of the documentation set."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.docs.conftest import DOCS, PGW_ROOT, REQUIRED_DOCS, anchors, strip_code, table_rows


@pytest.mark.parametrize("rel", REQUIRED_DOCS)
def test_required_document_exists_with_title(rel: str, docs: dict[str, str]) -> None:
    assert rel in docs, f"missing docs/{rel}"
    text = docs[rel]
    first = text.lstrip().splitlines()[0]
    assert first.startswith("# "), f"{rel}: must start with a level-1 title"
    assert len(text) > 3000 or rel == "README.md", f"{rel}: suspiciously short"
    assert strip_code(text).count("\n# ") == 0, f"{rel}: exactly one level-1 heading expected"


def test_code_fences_balanced(docs: dict[str, str]) -> None:
    for rel, text in docs.items():
        fences = re.findall(r"^```", text, flags=re.M)
        assert len(fences) % 2 == 0, f"{rel}: unbalanced code fences"


def test_tables_have_consistent_columns(docs: dict[str, str]) -> None:
    for rel, text in docs.items():
        lines = strip_code(text).splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
                width = lines[i + 1].count("|")
                j = i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    # escaped pipes or pipes inside inline code are not used in our tables
                    assert lines[j].count("|") == width, f"{rel}: table row {j + 1} has wrong column count"
                    j += 1
                i = j
            else:
                i += 1


def _links(text: str) -> list[str]:
    return re.findall(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)", strip_code(text))


def test_relative_links_and_anchors_resolve(docs: dict[str, str]) -> None:
    for rel, text in docs.items():
        base = (DOCS / rel).parent
        for link in _links(text):
            if re.match(r"^[a-z]+://", link) or link.startswith("mailto:"):
                continue
            path, _, frag = link.partition("#")
            target = (base / path).resolve() if path else (DOCS / rel).resolve()
            assert target.exists(), f"{rel}: broken link {link}"
            assert PGW_ROOT in target.parents or target == PGW_ROOT, f"{rel}: link leaves the pgw tree: {link}"
            if frag:
                assert target.suffix == ".md", f"{rel}: anchor into non-Markdown file {link}"
                assert frag in anchors(target.read_text(encoding="utf-8")), f"{rel}: missing anchor {link}"


def test_readme_indexes_every_document(docs: dict[str, str]) -> None:
    linked = {Path(link.split("#")[0]).as_posix() for link in _links(docs["README.md"])}
    for rel in docs:
        if rel != "README.md":
            assert rel in linked, f"README.md does not link {rel}"
    assert "measurements.json" in linked
    assert "tools/pgw_sizing.py" in linked


def test_whitepaper_toc_matches_sections(docs: dict[str, str]) -> None:
    text = docs["security/whitepaper.md"]
    numbered = re.findall(r"^## (\d+)\. ", text, flags=re.M)
    assert numbered == [str(i) for i in range(1, len(numbered) + 1)]
    toc = re.findall(r"^\d+\. \[[^\]]+\]\(#([^)]+)\)", text, flags=re.M)
    assert len(toc) == len(numbered)


_SECRET_PATTERNS = [
    re.compile(r"pgw-[0-9a-f]{12}-[A-Za-z0-9_-]{43}"),  # a well-formed real API key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"\b[0-9a-f]{32}\b"),  # 32-hex tokens (API keys of external services)
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"(?i)kaggle_(username|key)"),
    re.compile(r"(?i)bensliv"),
]


def test_no_secrets_or_personal_data() -> None:
    for p in DOCS.rglob("*"):
        if p.is_file() and p.suffix in (".md", ".json", ".py"):
            text = p.read_text(encoding="utf-8")
            for pat in _SECRET_PATTERNS:
                assert not pat.search(text), f"{p.relative_to(DOCS)} matches secret pattern {pat.pattern}"


def test_example_hosts_use_reserved_or_internal_names(docs: dict[str, str]) -> None:
    allowed_suffixes = (".example.net", ".example.com", ".example.org", ".example.internal", ".internal")
    # public identity providers whose issuer host is fixed (console SSO examples)
    public_idps = ("login.microsoftonline.com",)
    for rel, text in docs.items():
        for host in re.findall(r"https?://([a-zA-Z0-9.-]+)", text):
            assert host.endswith(allowed_suffixes) or host in ("localhost", "127.0.0.1", *public_idps), (
                f"{rel}: example URL with a real-looking host {host}")


def test_planned_features_are_marked(docs: dict[str, str]) -> None:
    """Features that are not in v1.0 must never be described as available."""
    text = docs["security/whitepaper.md"] + docs["security/threat-model.md"] + docs["compliance/faq.md"]
    for feature in ("padding", "dummy"):
        for para in re.split(r"\n\s*\n|\n\|", strip_code(text)):
            if feature in para.lower() and "row" in para.lower():
                assert re.search(r"(?i)planned|roadmap|not in (the )?v1", para), f"unmarked planned feature: {para[:120]}"


def test_compliance_faq_marked_for_legal_review(docs: dict[str, str]) -> None:
    text = docs["compliance/faq.md"]
    head = text[:800]
    assert "REQUIRES LEGAL REVIEW" in head and "not legal advice" in head
    questions = re.findall(r"^### (Q\d+\..*)$", text, flags=re.M)
    assert len(questions) >= 12
    for q in questions:
        assert "[needs legal review]" in q, f"FAQ question not marked: {q}"
    for topic in ("GDPR", "HIPAA", "FERPA"):
        assert topic in text
    # no overclaiming
    for phrase in ("guarantees compliance", "is compliant", "fully compliant", "unbreakable"):
        assert phrase not in text.replace('"unbreakable"', "")


def test_threat_model_covers_stride_for_every_component(docs: dict[str, str]) -> None:
    text = docs["security/threat-model.md"]
    rows = [r for r in table_rows(text) if r and re.fullmatch(r"T-\d\d", r[0])]
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)) >= 35
    assert ids == sorted(ids), "threat ids must be in order"
    letters = {c for r in rows for c in r[1].replace(" ", "").split("/")}
    assert set("STRIDE") <= letters, f"missing STRIDE categories: {set('STRIDE') - letters}"
    for comp in ("Compute service", "Gateway", "Mask store", "Mask factory", "Network", "Model pack", "supply chain"):
        assert re.search(rf"^### .*{comp}", text, flags=re.M | re.I), f"no STRIDE section for {comp}"
    for r in rows:
        assert re.search(r"\*\*[HML]\*\*|\b[HML]\b|depends", r[-1]), f"{r[0]}: residual rating missing"
    # every threat id referenced elsewhere exists
    all_text = "\n".join((DOCS / p).read_text(encoding="utf-8") for p in ("security/whitepaper.md",
                                                                          "compliance/audit-readiness.md"))
    for ref in set(re.findall(r"\bT-\d\d\b", all_text + text)):
        assert ref in ids, f"reference to unknown threat {ref}"


def test_audit_checklist_ids_unique_and_referenced(docs: dict[str, str]) -> None:
    text = docs["compliance/audit-readiness.md"]
    ids = [r[0] for r in table_rows(text) if r and re.fullmatch(r"[GSMNOI]-\d\d", r[0])]
    assert len(ids) == len(set(ids)) >= 40
    for r in table_rows(text):
        if r and re.fullmatch(r"[GSMNOI]-\d\d", r[0]):
            assert len(r) == 5 and r[3] and set(r[3].replace(",", " ").split()) <= {"C", "V", "P"}, r
    refs = set()
    for t in docs.values():
        refs |= set(re.findall(r"(?:checklist|audit evidence)[^)]*?\b([GSMNOI]-\d\d)\b", t))
    for ref in refs:
        assert ref in ids, f"reference to unknown checklist item {ref}"


def test_glossary_sorted_and_covers_key_terms(docs: dict[str, str]) -> None:
    text = docs["glossary.md"]
    terms = re.findall(r"^\*\*(.+?)\*\*:", text, flags=re.M)
    assert len(terms) >= 40
    keys = [re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)", "", t).lower()) for t in terms]
    assert keys == sorted(keys), "glossary terms must be alphabetical"
    joined = " ".join(terms).lower()
    for term in ("freivalds", "mask reuse", "slot", "segment", "fingerprint", "collusion", "stride", "hpke",
                 "perfect secrecy", "noise monitor", "ring preset", "tenant token"):
        assert term in joined, f"glossary misses {term}"
