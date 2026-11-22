"""benchmarks/scaling_model.py: shape arithmetic, per-trip formulas, speculation algebra and the lever search."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pgw_test_scaling_model", BENCH / "scaling_model.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load()


@pytest.mark.parametrize("name,params_b", [("qwen2.5-7b", 7.62), ("qwen2.5-14b", 14.77), ("qwen2.5-32b", 32.76),
                                           ("qwen2.5-72b", 72.71), ("llama-3-70b", 70.55)])
def test_parameter_counts_match_the_checkpoints(name: str, params_b: float) -> None:
    # published sizes; norms and biases (< 0.01%) are not in the shape
    assert sm.SHAPES[name].params() / 1e9 == pytest.approx(params_b, rel=0.005)


def test_qwen7b_vectors_slot_and_hint_match_the_measured_layout() -> None:
    s = sm.SHAPES["qwen2.5-7b"]
    assert s.sum_n(with_head=False) == 831_488 and s.sum_m(with_head=False) == 1_390_592  # SPEC 7.9
    assert s.sum_n() == 835_072 and s.sum_m() == 1_542_656
    sc = sm.TODAY
    assert sm.slot_bytes(s, sc) / 1e6 == pytest.approx(6.17, abs=0.005)  # 5.56 layers + 0.61 head
    assert sm.slot_bytes(s, sc.with_(masked_head=False)) / 1e6 == pytest.approx(5.56, abs=0.005)
    assert s.sum_m(with_head=False) * sm.HE_HINT_BYTES_PER_ROW / 1e9 == pytest.approx(34.2, abs=0.05)
    assert s.kv_bytes_per_ctx_token(4) == 114_688  # "115 KB per token"


@pytest.mark.parametrize("name,trips4,trips2", [("qwen2.5-7b", 113, 57), ("qwen2.5-14b", 193, 97),
                                                ("qwen2.5-32b", 257, 129), ("qwen2.5-72b", 321, 161)])
def test_trips_per_pass(name: str, trips4: int, trips2: int) -> None:
    s = sm.SHAPES[name]
    assert sum(c for *_, c in sm.trip_groups(s, sm.TODAY)) == trips4
    assert sum(c for *_, c in sm.trip_groups(s, sm.TODAY.with_(trips_per_layer=2))) == trips2
    # fusing trips keeps the elements (and so the masks and the wire)
    for f in (lambda g: g[1] * g[3], lambda g: g[2] * g[3]):
        assert sum(map(f, sm.trip_groups(s, sm.TODAY))) == sum(map(f, sm.trip_groups(
            s, sm.TODAY.with_(trips_per_layer=2))))


def test_gateway_kernels_reproduce_the_measured_7b_floor() -> None:
    s = sm.SHAPES["qwen2.5-7b"]
    sc = sm.TODAY.with_(orch_ms=0.0)
    total = sum(c * sm.gateway_trip_ms(n, m, 1, sc) for _, n, m, c in sm.trip_groups(s, sc))
    assert total == pytest.approx(6.15 + 6.0 + 3.8, rel=1e-9)  # prepare + output side + r keystream, ms
    assert sum(c * sm.gateway_trip_ms(n, m, 1, sc.with_(r_offcore=True)) for _, n, m, c in
               sm.trip_groups(s, sc)) == pytest.approx(12.15, rel=1e-9)
    # rows scale linearly, cores divide the work and add the split sync
    one = sm.gateway_trip_ms(3584, 37888, 1, sc)
    assert sm.gateway_trip_ms(3584, 37888, 4, sc) == pytest.approx(4 * one)
    assert sm.gateway_trip_ms(3584, 37888, 4, sc.with_(trip_cores=4)) == pytest.approx(one + sc.split_sync_ms)


def test_gpu_trip_memory_and_compute_bound() -> None:
    sc = sm.TODAY.with_(gpu_kernel_ms=0.0)
    mem = 1e9 / (3.35e12 * sm.GPU_BW_EFF) * 1e3
    assert sm.gpu_trip_ms(1e9, 1, sc, 1) == pytest.approx(mem)
    assert sm.gpu_trip_ms(1e9, 1, sc, 4) == pytest.approx(mem / 4 + sc.tp_sync_ms)
    rows = 1000  # 4 limbs x 1000 rows: compute-bound
    cmp_ = 2e9 * 4 * rows / (1979e12 * sm.GPU_TOPS_EFF) * 1e3
    assert sm.gpu_trip_ms(1e9, rows, sc, 1) == pytest.approx(cmp_)


def test_tp_degree_fits_the_weights() -> None:
    assert sm.tp_degree(sm.SHAPES["qwen2.5-32b"], sm.TODAY) == 1
    assert sm.tp_degree(sm.SHAPES["qwen2.5-72b"], sm.TODAY) == 2  # 72.7 GB > 0.85 x 80 GB
    assert sm.tp_degree(sm.SHAPES["qwen2.5-72b"], sm.TODAY.with_(gpu="h200")) == 1
    assert sm.tp_degree(sm.SHAPES["qwen2.5-7b"], sm.TODAY.with_(tp_min=4)) == 4


def test_wire_per_trip() -> None:
    # 1 row of gate_up at 72B: (8192 + 59136) x 4 B at 25 Gbit/s
    assert sm.wire_trip_ms(8192, 59136, 1, sm.TODAY) == pytest.approx((8192 + 59136) * 32 / 25e9 * 1e3)


@pytest.mark.parametrize("alpha", [0.3, 0.49, 0.7, 0.9])
def test_speculation_algebra(alpha: float) -> None:
    assert sm.tau_chain(alpha, 1) == pytest.approx(1.0)
    assert sm.tau_chain(alpha, 4) == pytest.approx(1 + alpha + alpha ** 2 + alpha ** 3)
    # the pipelined drafter with m = k is sequential drafting
    assert sm.tau_pipe(alpha, 4, 3) == pytest.approx(sm.tau_chain(alpha, 4))
    assert 1.0 <= sm.tau_pipe(alpha, 4, 0) <= sm.tau_chain(alpha, 4)
    # a hypothesis tree without second-choice capture is the plain pipelined drafter
    assert sm.tau_pipe_hyp(alpha, 0.0, 3) == pytest.approx(sm.tau_pipe(alpha, 3, 0))
    assert sm.tau_pipe_hyp(alpha, 0.35, 3) > sm.tau_pipe(alpha, 3, 0)
    # a tree without second choices is the chain; with them it is never worse
    for rows in (2, 4, 8):
        t0, d0 = sm.tau_tree(alpha, 0.0, rows)
        assert t0 == pytest.approx(sm.tau_chain(alpha, rows)) and d0 == rows - 1
        assert sm.tau_tree(alpha, 0.35, rows)[0] >= t0 - 1e-12


def test_tree_counts_nodes() -> None:
    # alpha 0.5, gamma 0.5: ranks 0.5, 0.25, 0.125, ...; best 3 nodes: 0.5 (d1), 0.25 (d2), 0.25 (d1, 2nd)
    tau, depth = sm.tau_tree(0.5, 0.5, 4)
    assert tau == pytest.approx(1 + 0.5 + 0.25 + 0.25) and depth == 2


def test_measured_tau_gives_alpha_and_validation_lane() -> None:
    assert sm.tau_pipe(0.49, 2, 0) == pytest.approx(1.19, abs=0.005)
    assert sm.alpha_from_tau_pipe(1.19) == pytest.approx(0.49, abs=0.005)
    v = sm.validate_7b()
    assert v["model_lane_ms"] == pytest.approx(v["measured_lane_ms"], rel=0.05)
    assert v["draft_gb_s_from_step"] == pytest.approx(8.45, abs=0.05)


def test_mask_demand_and_factory_scale() -> None:
    s = sm.SHAPES["qwen2.5-72b"]
    r = sm.evaluate(s, sm.TODAY)
    md = sm.mask_demand(s, sm.TODAY, r)
    assert md["slots_per_token"] == pytest.approx(r.rows / r.tau)
    assert md["mask_gb_s"] == pytest.approx(md["slots_per_s"] * s.sum_m() * 4 / 1e9)
    f = sm.factory(s, sm.TODAY, 10.0)
    assert f["mlwe_hint"]["hint_gb"] == pytest.approx(s.sum_m() * 24576 / 1e9)
    assert f["plain_cloud_gpu"]["hint_gb"] == 0.0 and f["rlwe_hint_free"]["hint_gb"] == 0.0
    # customer HE work scales with the elements: 7B at the measured 96.8 core-ms per slot
    f7 = sm.factory(sm.SHAPES["qwen2.5-7b"], sm.TODAY, 1.0)
    assert f7["mlwe_hint"]["customer_core_ms_per_slot"] == pytest.approx(96.8)


def test_evaluate_and_levers() -> None:
    s = sm.SHAPES["qwen2.5-32b"]
    today, eng = sm.evaluate(s, sm.TODAY), sm.evaluate(s, sm.apply(sm.TODAY, sm.SCENARIOS["B engineered"]))
    assert eng.tok_s > today.tok_s
    assert today.tok_s == pytest.approx(1e3 * today.tau / today.pass_ms)
    assert sm.minimal_lever_sets(s, 0.1) == [((), pytest.approx(today.tok_s))]
    sets = sm.minimal_lever_sets(s, 25.0)
    assert sets and all(sm.evaluate(s, sm.apply(sm.TODAY, list(c))).tok_s >= 25.0 for c, _ in sets)
    assert len({len(c) for c, _ in sets}) == 1  # all of the smallest size
    assert sm.minimal_lever_sets(s, 1e6) == []


def test_report_runs() -> None:
    rep = sm.report(["qwen2.5-7b"], 2048, 0.7)
    assert len(rep["results"]) == len(sm.SCENARIOS)
    assert "| A today | qwen2.5-7b |" in sm._fmt_table(rep)
