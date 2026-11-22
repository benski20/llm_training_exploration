"""benchmarks/scale_sizing.py: analytic figures agree with the base sizing tool and the MoE accounting."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pgw_test_scale_sizing", BENCH / "scale_sizing.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ss = _load()


@pytest.mark.parametrize("name", ["qwen2.5-7b", "llama-3-70b"])
def test_dense_figures_match_base_tool(name: str) -> None:
    rep = ss.size(ss.MODELS[name], tokens_per_s=100.0)
    b = ss.base.size(ss.base.MODELS[name], ss.base.RINGS["K32A12C27"])
    assert rep.figures["online_wire_per_token"].value == pytest.approx(b.figures["wire_total_per_token"].value)
    assert rep.figures["mask_bytes_per_token"].value == pytest.approx(b.figures["wire_down_per_token"].value)
    assert rep.figures["round_trips_per_step"].value == b.figures["rounds_per_token"].value


def test_qwen7b_slot_is_5_56_mb() -> None:
    assert ss.size(ss.MODELS["qwen2.5-7b"], tokens_per_s=1).figures["mask_bytes_per_token"].value == pytest.approx(
        5.56, abs=0.01)


def test_moe_masks_follow_routed_wire_follows_all_experts() -> None:
    m = ss.MODELS["mixtral-8x22b"]
    rep = ss.size(m, tokens_per_s=1000.0)
    L, b = m.layers, 4
    attn_m = sum(mm for mm, _ in m.attn().values())
    attn_n = sum(n for _, n in m.attn().values())
    routed_m = m.top_k * (2 * m.ffn + m.hidden)
    assert rep.figures["mask_bytes_per_token"].value == pytest.approx(L * (attn_m + routed_m) * b / 1e6)
    all_rows = m.experts * (2 * m.ffn + m.hidden + m.hidden + m.ffn)
    assert rep.figures["online_wire_per_token"].value == pytest.approx(L * (attn_m + attn_n + all_rows) * b / 1e6)
    assert rep.figures["moe_wire_overhead_vs_leaky"].value > 1.0


def test_scaling_and_validation() -> None:
    m = ss.MODELS["llama-3-70b"]
    r1 = ss.size(m, tokens_per_s=1000.0, gpu="h100", encrypted=True)
    r2 = ss.size(m, tokens_per_s=2000.0, gpu="h100", encrypted=True)
    for k in ("provider_gpus_online", "provider_gpus_maskgen", "gateway_he_devices", "mask_store_bandwidth"):
        assert r2.figures[k].value == pytest.approx(2 * r1.figures[k].value)
    assert r1.figures["he_hint_size"].value == pytest.approx(
        m.layers * m.all_weights()["m"] * 16 * 256 * 6 / 1e9)
    with pytest.raises(ValueError):
        ss.size(m, tokens_per_s=0.0)
    with pytest.raises(ValueError):
        ss.size(m, tokens_per_s=1.0, gpu="tpu")


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert ss.main(["--all", "--tokens-per-s", "500", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {r["model"] for r in out} == set(ss.MODELS)
    assert ss.main(["--model", "mixtral-8x22b", "--markdown", "--encrypted"]) == 0


def test_he_e2e_rates_scale_by_stage_work() -> None:
    meas = {"stages": {"encrypt": 100.0, "provider": 200.0, "decrypt": 50.0}, "pipeline": 30.0,
            "hint_disk_gb_per_s": 2.0}
    r7 = ss.he_e2e_rates(ss.MODELS["qwen2.5-7b"], measured=meas)
    assert r7["he_pair_overlapped_slots_per_s"].value == pytest.approx(50.0)
    assert r7["he_pair_serial_slots_per_s"].value == pytest.approx(1 / (1 / 100 + 1 / 200 + 1 / 50))
    assert r7["he_pair_pipeline_slots_per_s"].kind == "measured"
    # 7B hint 34.2 GB read once per 6-group pass: ~22 MB per slot
    assert r7["he_hint_read_per_slot"].value == pytest.approx(34.17e3 / (256 * 6), rel=0.01)
    assert r7["he_hint_disk_bound_slots_per_s"].value == pytest.approx(
        2e9 / (r7["he_hint_read_per_slot"].value * 1e6))
    r70 = ss.he_e2e_rates(ss.MODELS["llama-3-70b"], measured=meas)
    assert r70["he_pair_provider_slots_per_s"].kind == "estimate"
    assert r70["he_pair_provider_slots_per_s"].value < r7["he_pair_provider_slots_per_s"].value / 8
    assert "he_pair_pipeline_slots_per_s" not in r70
    rep = ss.size(ss.MODELS["llama-3-70b"], tokens_per_s=1000.0, encrypted=True)
    assert rep.figures["he_t4x2_pairs_overlapped"].value > 0
