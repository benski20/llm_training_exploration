"""Chat-speed measurement tools (WP-A0): the live-gateway timing probe, the decode attribution in
``gpu_validate.py`` and smoke runs of the new benches at toy sizes."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pgw.core import RING_K40_A12
from pgw.engine import ChatTokenizer, Generator, InMemoryPack, PlainBackend, SamplingParams, TransformerModel
from tests.engine.conftest import qwen_pack, tiny_tokenizer  # noqa: F401 - fixtures

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"pgw_test_bench_{name}", BENCH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_times_every_component_of_a_live_engine(qwen_pack: InMemoryPack,  # noqa: F811
                                                      tiny_tokenizer: ChatTokenizer) -> None:  # noqa: F811
    import pgw.engine.model as model_mod

    probe = load("gateway_probe")
    orig = (model_mod._attend, model_mod.rms_norm, model_mod.TransformerModel.__dict__["_logits"])
    gen = Generator(TransformerModel(qwen_pack, PlainBackend(qwen_pack, RING_K40_A12)), tiny_tokenizer)
    want = gen.generate([256, 72, 105], SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)).token_ids
    probe.install(None)
    try:
        got = asyncio.run(gen.agenerate([256, 72, 105], SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)))
        snap = probe.counters()
    finally:
        probe.uninstall()
    assert got.token_ids == want  # the probe only times, it never changes results
    assert snap["missing"] == []  # every target still exists (a rename must update the probe)
    c = snap["counters"]
    for name in ("attention", "norm", "rope", "kv", "head", "consume", "step"):
        assert c[name][1] > 0 and c[name][0] >= 0.0, name
    assert c["step"][1] == 6  # prefill step + 5 decode steps, one per token
    # exclusive timing: the final norm inside the head is not counted again under "norm"
    assert c["head"][1] == 6
    assert (model_mod._attend, model_mod.rms_norm, model_mod.TransformerModel.__dict__["_logits"]) == orig
    assert probe.counters()["counters"] == {}


def test_probe_snapshot_over_signal(tmp_path: Path) -> None:
    probe_file = tmp_path / "probe.json"
    code = (
        "import importlib.util, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('p', {str(BENCH / 'gateway_probe.py')!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"m.install(__import__('pathlib').Path({str(probe_file)!r}))\n"
        "import torch, pgw.engine.model as mm\n"
        "mm.rms_norm(torch.ones(1, 8), torch.ones(8), 1e-6)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(BENCH.parent / "src"), os.environ.get("PYTHONPATH", "")])}
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert p.stdout is not None and p.stdout.readline().strip() == "ready"
        probe = load("gateway_probe")
        s1 = probe.snapshot(p.pid, probe_file)
        s2 = probe.snapshot(p.pid, probe_file)
        assert s1 is not None and s2 is not None and s2["seq"] == s1["seq"] + 1
        assert s1["counters"]["norm"][1] == 1 and s1["missing"] == []
    finally:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=30)
    assert load("gateway_probe").snapshot(p.pid, tmp_path / "absent.json") is None


def test_attribution_parts_sum_to_the_unaccounted_time() -> None:
    gv = load("gpu_validate")
    n_dec, dec_s = 10, 0.350
    acc = {"decode_remote_s_per_token": 0.163, "decode_local_s_per_token": 0.088,
           "decode_unaccounted_s_per_token": round(dec_s - 0.163 - 0.088, 4)}

    def snap(scale: float) -> dict[str, Any]:
        per_tok = {"attention": 18.1, "norm": 3.9, "rope": 3.3, "kv": 1.3, "head": 44.0, "consume": 1.0,
                   "draft": 0.0, "step": 330.0}
        return {"counters": {k: [v * scale * n_dec / 1e3, int(scale * n_dec)] for k, v in per_tok.items()}}

    out = gv.attribute_unaccounted(snap(1.0), snap(2.0), n_dec, dec_s, acc)
    assert out["float_ops_ms"] == pytest.approx(26.6) and out["head_ms"] == pytest.approx(44.0)
    assert out["loop_ms"] == pytest.approx(20.0) and out["steps_per_token"] == 1.0
    parts = sum(out[k] for k in ("float_ops_ms", "head_ms", "consume_ms", "draft_ms", "step_glue_ms", "loop_ms"))
    assert parts == pytest.approx(out["unaccounted_ms"], abs=0.01)
    assert gv.attribute_unaccounted(None, snap(1.0), n_dec, dec_s, acc) == {}
    # chat_speed_model re-bases scenario (a) on such a result
    csm = load("chat_speed_model")
    res = {"stream": {"decode_round_trips_per_token": 112, "decode_remote_ms_per_trip": 1.457,
                      "decode_local_s_per_token": 0.088, "decode_s_per_token": dec_s,
                      "decode_unaccounted_split": out}}
    today = csm.today_a_from_validate(res)
    assert today is not None
    assert csm.scenarios(today)["a today"][0] == pytest.approx(dec_s * 1e3, abs=0.5)
    assert csm.scenarios()["a today"][0] == pytest.approx(349, abs=1)


def test_lane_bench_floor_transports() -> None:
    lb = load("lane_bench")
    args = argparse.Namespace(transports="tcp,uds", transport=None, rows="1,2", tokens=2, wait="both", layers=28)
    res = lb.run(args)
    labels = {(r["transport"], r["rows"]) for r in res["results"]}
    assert labels == {(f"{f}-{w}", r) for f in ("tcp", "uds") for w in ("block", "spin") for r in (1, 2)}
    assert all(r["trips"] == 8 and r["p50_us"] > 0 for r in res["results"])


def test_spec_cost_bench_smoke(tmp_path: Path) -> None:
    sc = load("spec_cost_bench")
    args = argparse.Namespace(preset="0.5b", layers=1, vocab=512, rows="2", context=4, steps=1, threads=0,
                              masking=False, ring="K32A12C27", service_url=None, token_file=None, fingerprint=None,
                              trips_ms=10.0, workdir=str(tmp_path))
    t = time.perf_counter()
    res = sc.run(args)
    assert time.perf_counter() - t < 120
    assert set(res["pass_ms"]) == {"1", "2"} and "2" in res["c_ms"]
    assert set(res["components_ms"]["2"]) == {"layers_ms", "head_ms", "sample_ms"}
    assert res["layers_full"] == 24 and res["layers_measured"] == 1
    assert "alpha_0.8" in res["projection"]["R2"]


def test_profile_prefill_smoke(tmp_path: Path) -> None:
    pp = load("profile_prefill")
    args = argparse.Namespace(preset="7b", layers=1, vocab=512, tokens="6", blocks="1,2", steps=1, threads=0,
                              ring="K32A12C27", masking=False, workdir=str(tmp_path))
    res = pp.run(args)
    row = res["prompts"]["6"]
    vol = row["volume"]
    # 7B at K=32: 8.9 MB of wire and 5.56 MB of masks per prompt token (chat-speed plan, section 1b)
    assert vol["up_bytes_per_token"] + vol["down_bytes_per_token"] == 8_888_320
    assert vol["mask_bytes_per_token"] == 5_562_368 and vol["round_trips_per_block"] == 112
    assert "2" in row["float"]["blocks"] and isinstance(row["float"]["blocks"]["2"]["last_logits_bit_identical"], bool)
