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


# ---- --consumer / --twenty: fused schedules at internet RTTs (research/reviews/consumer-round-trips.md) ----------
@pytest.mark.parametrize("model,L", [("0.5B", 24), ("7B", 28), ("32B", 64), ("72B", 80)])
def test_consumer_trip_counts(model: str, L: int) -> None:
    want = {"standard (today, 4L+1)": 4 * L + 1, "fused2 + head trip (2L+2)": 2 * L + 2,
            "fused2 folded head, local qkv0 (2L)": 2 * L, "parallel blocks fused (L+1)": L + 1}
    for sched, trips in want.items():
        assert rt.consumer_pass_ms(model, sched, 20.0)["trips"] == trips


def test_consumer_fusion_wins_at_internet_rtt_and_tau_divides() -> None:
    for model in rt.C_SHAPES:
        std = rt.consumer_tok_s(model, "standard (today, 4L+1)", 20.0)
        f2 = rt.consumer_tok_s(model, "fused2 folded head, local qkv0 (2L)", 20.0)
        par = rt.consumer_tok_s(model, "parallel blocks fused (L+1)", 20.0)
        assert 1.6 < f2 / std < 2.2 and par > f2
    one = rt.consumer_pass_ms("7B", "fused2 folded head, local qkv0 (2L)", 20.0, rows=3)["pass_ms"]
    assert rt.consumer_tok_s("7B", "fused2 folded head, local qkv0 (2L)", 20.0, tau=2.0, rows=3) == pytest.approx(
        2e3 / one)
    # 20 tok/s at 32B: 128 trips at RTT 20 ms alone take 2.56 s per pass
    assert rt.consumer_tok_s("32B", "fused2 folded head, local qkv0 (2L)", 20.0, tau=8, rows=8, gw="target",
                             rows_free=True, tp=2) < 5


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


# ---- --links: the bandwidth term (research/reviews/consumer-bandwidth.md) ------------------------------------------
@pytest.mark.parametrize("model", ["0.5B", "7B", "32B"])
def test_row_bytes_match_the_wire_formats(model: str) -> None:
    """K32: 4 B per masked element; fused K56: 7 B; element counts as the engine's trip plans move them."""
    L, d, inter, q, vocab = rt.C_SHAPES[model]
    up, down = rt.row_bytes(model, "standard (today, 4L+1)", "k32")
    assert up == 4 * (L * (3 * d + inter) + d) and down == 4 * (L * (q + 2 * d + 2 * inter) + vocab)
    fu, fd = rt.row_bytes(model, "fused2 folded head, local qkv0 (2L)", "k32")
    assert fu == 7 * (L * (d + inter) + 2 * L * d)
    assert fd == 7 * (L * (2 * d + 2 * inter + q) - q + vocab)
    nu, nd = rt.row_bytes(model, "standard (today, 4L+1)", "narrow")
    assert nu == up * 28 / 32 and nd == down * 28 / 32
    cu, _ = rt.row_bytes(model, "standard (today, 4L+1)", "carry16")
    assert cu == up / 2
    assert rt.row_bytes(model, "standard (today, 4L+1)", "k32", logits=False)[1] == down - 4 * vocab


def test_link_term_adds_to_the_pass_and_masks_cost_bandwidth() -> None:
    link = rt.LINKS["cable 100/10"]
    s = "standard (today, 4L+1)"
    a = rt.link_ms("7B", s, 1, link, "k32")
    assert rt.link_ms("7B", s, 3, link, "k32") == pytest.approx(3 * a)
    assert rt.link_ms("7B", s, 1, link, "k32", "pks-v3") > 2 * a  # pks-v3 masks more than double the traffic
    assert rt.link_ms("7B", s, 1, link, "carry16") < rt.link_ms("7B", s, 1, link, "narrow") < a
    up, down = rt.row_bytes("7B", s, "k32")
    want = (up + rt.ACK_UP * down) / (10e6 * rt.LINK_EFF / 8) + down / (100e6 * rt.LINK_EFF / 8)
    assert a == pytest.approx(1e3 * want)


def test_links_table_prints(capsys: pytest.CaptureFixture[str]) -> None:
    rt.links_main(5.0)
    out = capsys.readouterr().out
    assert "fibre 1000/1000" in out and "| 32B | standard | k32 | pks-v3 |" in out


# ---- --layer-map / --mixed: mixed sequential / parallel layer maps (research/reviews/parallel-conversion.md) -------
def test_mixed_layer_map_trips_and_bytes() -> None:
    L = rt.C_SHAPES["7B"][0]
    full_p = rt.add_layer_map("100%")
    none_p = rt.add_layer_map("0%")
    half = rt.add_layer_map("50%")
    explicit = rt.add_layer_map("s" * 14 + "p" * 14)
    try:
        assert rt.consumer_pass_ms("7B", full_p, 20.0)["trips"] == L + 1
        assert rt.consumer_pass_ms("7B", none_p, 20.0)["trips"] == 2 * L + 1
        assert rt.consumer_pass_ms("7B", half, 20.0)["trips"] == 1 + 2 * 14 + 14
        assert rt.consumer_pass_ms("7B", explicit, 20.0)["trips"] == 1 + 2 * 14 + 14
        assert not rt.schedule_applies("32B", explicit) and rt.schedule_applies("7B", explicit)
        # bytes: the all-parallel map moves what the parallel schedule moves; fewer trips, fewer residual bytes
        assert rt.row_bytes("7B", full_p) == rt.row_bytes("7B", "parallel blocks fused (L+1)")
        assert rt.row_bytes("7B", full_p)[0] < rt.row_bytes("7B", half)[0] < rt.row_bytes("7B", none_p)[0]
        tok = [rt.consumer_tok_s("7B", s, 20.0) for s in (none_p, half, full_p)]
        assert tok[0] < tok[1] < tok[2]
    finally:
        for n in (full_p, none_p, half, explicit):
            rt.C_SCHEDULES.pop(n, None)
            rt.C_MAPS.pop(n, None)
    with pytest.raises(SystemExit):
        rt.add_layer_map("spx")


def test_mixed_table_prints_and_leaves_no_state(capsys: pytest.CaptureFixture[str]) -> None:
    before_s, before_m = dict(rt.C_SCHEDULES), dict(rt.C_SHAPES)
    rt.mixed_main([20.0])
    out = capsys.readouterr().out
    assert "Qwen3-14B (40) | trips | 81 | " in out and "| 41 |" in out
    assert rt.C_SCHEDULES == before_s and rt.C_SHAPES == before_m
