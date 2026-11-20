"""Every quoted number is traceable to measurements.json; analytic numbers recompute; estimates are labelled."""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.docs.conftest import DOCS, PGW_ROOT, section, strip_code, table_rows

KINDS = {"measured", "analytic", "derived", "estimate"}


def test_measurements_schema(measurements: dict[str, Any]) -> None:
    sources = measurements["sources"]
    ids = [e["id"] for e in measurements["entries"]]
    assert len(ids) == len(set(ids)) >= 35
    for e in measurements["entries"]:
        assert set(e) >= {"id", "value", "unit", "kind", "source", "display", "conditions"}, e["id"]
        assert e["kind"] in KINDS, e["id"]
        assert e["source"] in sources, e["id"]
        assert e["display"].strip() and e["conditions"].strip(), e["id"]
    kinds = {e["kind"] for e in measurements["entries"]}
    assert kinds == KINDS


def _blocks_containing(text: str, needle: str) -> list[str]:
    """Table rows (single line) or paragraphs that contain ``needle``."""
    out = []
    for para in re.split(r"\n\s*\n", strip_code(text)):
        if needle not in para:
            continue
        lines = para.splitlines()
        if all(ln.startswith("|") for ln in lines):
            out.extend(ln for ln in lines if needle in ln)
        else:
            out.append(para)
    return out


def test_every_measurement_is_quoted(measurements: dict[str, Any], docs: dict[str, str]) -> None:
    corpus = "\n".join(docs.values())
    missing = [e["id"] for e in measurements["entries"] if e["display"] not in corpus]
    assert not missing, f"entries never quoted in the docs: {missing}"


def test_estimates_are_labelled_wherever_quoted(measurements: dict[str, Any], docs: dict[str, str]) -> None:
    for e in measurements["entries"]:
        if e["kind"] != "estimate":
            continue
        n = 0
        for rel, text in docs.items():
            for block in _blocks_containing(text, e["display"]):
                n += 1
                assert re.search(r"(?i)estimate", block), f"{rel}: estimate {e['id']} quoted without label: {block!r}"
        assert n, e["id"]


def test_measured_numbers_are_called_measured_in_sizing(measurements: dict[str, Any], docs: dict[str, str]) -> None:
    rows = table_rows(docs["operations/sizing.md"])
    for e in measurements["entries"]:
        if e["kind"] == "measured" and e["source"] in ("otp-masks", "otp-cost"):
            for r in rows:
                if any(e["display"] in c for c in r):
                    assert r[-1].startswith(("measured", "derived")) or "measured" in " ".join(r), r


# Numbers quoted in the brief and in SPEC 14 that the docs must reproduce.
@pytest.mark.parametrize("phrase", [
    "0.45 s/token", "0.014 s/token", "8.3 MB", "0.0066 s/slot", "0.30 TOPS", "13.4 MB", "550k slots/h",
    "6.9 MB", "1.5 MB", "128 round trips", "max logit difference 0.0",
])
def test_headline_numbers_present(phrase: str, docs: dict[str, str]) -> None:
    assert phrase in "\n".join(docs.values())


# ------------------------------------------------------------------- consistency with the research logs

_RESEARCH = PGW_ROOT.parent / "results" / "kaggle"
_LOG_CHECKS = {
    "server_8b_t4_batch1": ("onprem-otp-masks", "'server_gpu_s_per_token_batch1': 0.452"),
    "server_8b_t4_batch32": ("onprem-otp-masks", "'server_gpu_s_per_token_amortized_batch32': 0.014"),
    "plain_fp16_8b_t4_batch1": ("onprem-otp-masks", "'plain_fp16_gpu_s_per_token_batch1': 0.0585"),
    "server_70b_t4_batch1": ("onprem-otp-masks", "'server_gpu_s_per_token_batch1': 3.094"),
    "server_70b_t4_batch32": ("onprem-otp-masks", "'server_gpu_s_per_token_amortized_batch32': 0.0927"),
    "mask_store_8b_k48": ("onprem-otp-masks", "'mask_storage_MB_per_token': 8.26"),
    "wire_up_8b_k48": ("onprem-otp-masks", "'upload_MB_per_token': 5.11"),
    "server_view_chi2": ("onprem-otp-masks", "'server_view_top_byte_chi2_255dof': 241.4"),
    "server_view_corr": ("onprem-otp-masks", "'server_view_corr_with_plaintext': -0.0006"),
    "exact_logit_diff_05b": ("onprem-otp-masks", "'masked_vs_quantized_max_logit_diff': 0.0"),
    "client_online_crypto_05b": ("onprem-otp-masks", "'client_online_crypto_s_total': 0.183"),
    "maskgen_8b_t4_k48": ("onprem-otp-cost", "'gpu_s_per_token': 0.00656"),
    "maskgen_70b_t4_k48": ("onprem-otp-cost", "'gpu_s_per_token': 0.04687"),
    "cpu_fbgemm_tops": ("onprem-otp-cost", "'fbgemm_quantized_linear_int8': 0.297"),
    "ppl_float": ("onprem-otp-cost", "T3 None {'ppl': 10.923"),
    "ppl_a12": ("onprem-otp-cost", "T3 12 {'ppl': 10.928"),
    "ppl_a8": ("onprem-otp-cost", "T3 8 {'ppl': 11.092"),
    "maskgen_cost_8b_k48_spot": ("onprem-otp-cost", "'usd_per_M_tokens_T4 spot (approx)': 0.36"),
}


@pytest.mark.skipif(not _RESEARCH.is_dir(), reason="research logs not present in this checkout")
@pytest.mark.parametrize("mid", sorted(_LOG_CHECKS))
def test_measured_values_match_research_logs(mid: str, measurements: dict[str, Any]) -> None:
    run, needle = _LOG_CHECKS[mid]
    log = (_RESEARCH / run / f"{run}.log").read_text(encoding="utf-8")
    assert needle in log, f"{mid}: {needle!r} not found in {run}.log"
    entry = next(e for e in measurements["entries"] if e["id"] == mid)
    logged = float(needle.rsplit(" ", 1)[1].rstrip("}"))
    assert abs(float(entry["value"]) - logged) <= max(0.02 * abs(logged), 0.001), (entry["value"], logged)


# --------------------------------------------------------------------------- analytic recomputation

_ANALYTIC = {
    # id -> (model, ring, figure key, scale)
    "mask_store_8b_k48": ("llama-3-8b", "K48A20", "wire_down_per_token"),
    "mask_store_8b_k40": ("llama-3-8b", "K40A12", "wire_down_per_token"),
    "mask_store_05b_k40": ("qwen2.5-0.5b", "K40A12", "wire_down_per_token"),
    "mask_store_70b_k48": ("llama-3-70b", "K48A20", "wire_down_per_token"),
    "wire_up_8b_k48": ("llama-3-8b", "K48A20", "wire_up_per_token"),
    "wire_down_8b_k48": ("llama-3-8b", "K48A20", "wire_down_per_token"),
    "wire_total_8b_k48": ("llama-3-8b", "K48A20", "wire_total_per_token"),
    "wire_total_8b_k40": ("llama-3-8b", "K40A12", "wire_total_per_token"),
    "rounds_8b": ("llama-3-8b", "K40A12", "rounds_per_token"),
    "rounds_70b": ("llama-3-70b", "K40A12", "rounds_per_token"),
    "slots_per_tb_8b_k48": ("llama-3-8b", "K48A20", "slots_per_tb"),
    "mask_store_7b_k32c27": ("qwen2.5-7b", "K32A12C27", "wire_down_per_token"),
}


@pytest.mark.parametrize("mid", sorted(_ANALYTIC))
def test_analytic_entries_recompute(mid: str, measurements: dict[str, Any], sizing: ModuleType) -> None:
    model, ring, key = _ANALYTIC[mid]
    rep = sizing.size(sizing.MODELS[model], sizing.RINGS[ring])
    got = rep.figures[key].value
    entry = next(e for e in measurements["entries"] if e["id"] == mid)
    assert entry["kind"] == "analytic"
    assert abs(got - float(entry["value"])) <= 0.006 * abs(got), (mid, got, entry["value"])


def _num(cell: str) -> float:
    m = re.search(r"([\d.]+)\s*(k)?", cell.replace(",", ""))
    assert m, cell
    return float(m.group(1)) * (1000 if m.group(2) else 1)


def test_sizing_reference_table_recomputes(docs: dict[str, str], sizing: ModuleType) -> None:
    names = {"Qwen2.5-0.5B": "qwen2.5-0.5b", "Qwen2.5-7B": "qwen2.5-7b", "Llama-3-8B": "llama-3-8b",
             "Llama-3-70B": "llama-3-70b"}
    sec = section(docs["operations/sizing.md"], "1. Per-token formulas")
    rows = [r for r in table_rows(sec) if r[0] in names]
    assert len(rows) == 10
    assert sum(r[1] == "K32A12C27" for r in rows) == 3  # the default ring has its own rows
    for model, ring, up, down, total, masks, per_tb, rounds in rows:
        f = sizing.size(sizing.MODELS[names[model]], sizing.RINGS[ring]).figures
        for cell, key, tol in ((up, "wire_up_per_token", 0.01), (down, "wire_down_per_token", 0.01),
                               (total, "wire_total_per_token", 0.01), (masks, "mask_storage_per_slot", 0.01),
                               (per_tb, "slots_per_tb", 0.01), (rounds, "rounds_per_token", 0)):
            want = f[key].value
            assert abs(_num(cell) - want) <= tol * want + 1e-9, (model, ring, key, cell, want)


def test_llama8b_shape_sums_in_sizing(docs: dict[str, str], sizing: ModuleType) -> None:
    s = sizing.MODELS["llama-3-8b"]
    text = docs["operations/sizing.md"]
    assert f"`Σm` = {s.sum_m}" in text and f"`Σn` = {s.sum_n}" in text


def test_worked_examples(docs: dict[str, str], sizing: ModuleType) -> None:
    text = docs["operations/sizing.md"]
    f = sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"], tokens_per_day=2e6, rtt_ms=1).figures
    assert f"{f['wire_per_day'].value:.1f} TB/day" in text  # 22.3
    assert f"{f['mask_storage_per_day'].value:.1f} TB/day" in text  # 13.8
    assert f"mean {f['mean_bandwidth'].value:.1f} Gbit/s" in text
    # 100 tok/s bandwidth example for 8B K48A20
    k48 = sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K48A20"]).figures
    assert f"{100 * k48['wire_total_per_token'].value * 8 / 1000:.1f} Gbit/s" in text  # 10.7
    # air-gapped example: 200k tokens/day, 10 days, K40A12
    ag = docs["operations/air-gapped-factory.md"]
    tb = 200_000 * 10 * sizing.size(sizing.MODELS["llama-3-8b"], sizing.RINGS["K40A12"]).figures[
        "wire_down_per_token"].value / 1e6
    assert f"about {tb:.1f} TB" in ag
    # factory GPUs example: 10M tokens/day at 0.0066 s/slot
    assert f"≈ {10e6 * 0.0066 / 86400:.2f} T4" in text


def test_spec_numbers_consistent(docs: dict[str, str]) -> None:
    """Numbers the docs share with SPEC.md (section 7.2 and 14) agree."""
    spec = (PGW_ROOT / "SPEC.md").read_text(encoding="utf-8")
    for phrase in ("1.5 MB", "6.9 MB", "8.3 MB", "0.0066 s", "0.45 s", "0.014 s"):
        assert phrase in spec
    assert "K40A12" in spec and "**default**" in spec


def test_docs_dir_has_no_stray_files() -> None:
    allowed = {".md", ".json", ".py"}
    for p in DOCS.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            assert p.suffix in allowed, f"unexpected file in docs/: {p.relative_to(DOCS)}"
    assert Path(DOCS / "tools" / "pgw_sizing.py").is_file()
