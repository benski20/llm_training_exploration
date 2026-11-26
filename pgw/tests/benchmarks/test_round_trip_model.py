"""benchmarks/round_trip_model.py ``--sdk``: the client library's per-token model (trip counts, components, tau)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pgw_test_round_trip_model", BENCH / "round_trip_model.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rt = _load()


@pytest.mark.parametrize("model,trips", [("7B", 113), ("32B", 257), ("72B", 321)])
def test_trips_are_four_per_layer_plus_the_masked_head(model: str, trips: int) -> None:
    b = rt.sdk_ms_per_token(model, 0.2)
    assert b["trips"] == trips and b["critical_trips_per_token"] == trips
    parts = sum(v for k, v in b.items() if k in ("network", "lane", "device", "wire", "gateway", "serve", "noise"))
    assert b["pass_ms"] == pytest.approx(parts) and b["network"] == pytest.approx(0.2 * trips)


def test_speculation_divides_the_pass_by_tau_and_rows_cost_more() -> None:
    one = rt.sdk_ms_per_token("7B", 0.2)
    three = rt.sdk_ms_per_token("7B", 0.2, rows=3, tau=2.0)
    assert three["pass_ms"] > one["pass_ms"]  # wire, gateway work and serving grow with the rows
    assert three["ms_per_token"] == pytest.approx(three["pass_ms"] / 2.0)
    assert three["critical_trips_per_token"] == pytest.approx(113 / 2.0)
    # device time is memory-bound: the same at 1 and 3 rows
    assert three["device"] == one["device"] and three["network"] == one["network"]


def test_sdk_table_prints(capsys: pytest.CaptureFixture[str]) -> None:
    rt.sdk_main()
    out = capsys.readouterr().out
    assert "| 72B | off (R=1) | 321 |" in out and "Local linear on the client" in out


def test_draft_head_mode_uses_the_measured_tau_and_pays_the_head_steps() -> None:
    rows, tau, head = rt.SDK_TAU["draft head, open chat (R=3)"]
    assert head and rows == 3 and 1.5 < tau < 2.0  # measured (7B replay), not the 0.8-alpha guess
    with_head = rt.sdk_ms_per_token("7B", 0.2, rows, tau, head=True)
    without = rt.sdk_ms_per_token("7B", 0.2, rows, tau)
    assert with_head["draft"] == pytest.approx(2 * rt.SDK_HEAD_STEP_MS["7B"]) and without["draft"] == 0.0
    assert with_head["pass_ms"] == pytest.approx(without["pass_ms"] + with_head["draft"])
    # the shipped head beats speculation off on open chat at a same-data-center RTT
    assert with_head["ms_per_token"] < rt.sdk_ms_per_token("7B", 0.2)["ms_per_token"]
