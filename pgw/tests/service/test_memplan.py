"""The provider memory plan (:mod:`pgw.service.memplan`): it reproduces the Kaggle 7B out-of-memory start-up from
shapes alone, its parts match what the code allocates, and the targets fit (7B on a 16 GB T4 and 24 GB cards,
32B on one 80 GB H100, 72B on one or two H100/H200)."""

from __future__ import annotations

import json

import pytest
import torch

from pgw.core.ringmm import offset_byte_rows, pad_weight_cuda, padded_shape
from pgw.service import memplan as mp
from pgw.service.memplan import GIB, GPUS, MODELS, MemConfig, plan_model


def test_run1_reproduces_the_kaggle_t4_oom() -> None:
    """results/kaggle/sdk-7b-validation/run1_7b: 'Tried to allocate 4.06 GiB ... 14.56 GiB total ... 13.23 GiB
    allocated by PyTorch' while warming up Qwen2.5-7B with the masked head on a T4."""
    p = plan_model("qwen2.5-7b", "t4", MemConfig(rev="run1"))
    d = p.devices[0]
    assert abs(d.steady["weights"] / GIB - 13.23) < 0.1  # both weight copies of every linear incl. the head
    assert abs(d.load["rowsum_int64_weight_copy"] / GIB - 4.06) < 0.01  # the head's int64 copy
    free_at_failure = GPUS["t4"].total_bytes - 13.53 * GIB
    assert d.load["rowsum_int64_weight_copy"] > free_at_failure
    assert not p.fits and d.peak_load > GPUS["t4"].total_bytes


def test_current_fits_the_targets() -> None:
    cfg = MemConfig()
    assert plan_model("qwen2.5-7b", "t4", cfg).fits
    for g in ("l4", "a10g", "rtx4090"):
        assert plan_model("qwen2.5-7b", g, cfg).fits, g
    assert plan_model("qwen2.5-32b", "h100-80", cfg).fits
    assert plan_model("qwen2.5-72b", "h100-80", cfg).fits
    assert plan_model("qwen2.5-72b", "h200", cfg).fits
    assert plan_model("qwen2.5-72b", "h100-80", MemConfig(tp=2)).fits
    assert plan_model("qwen2.5-72b", "h100-80", MemConfig(pipeline=2)).fits
    # and not where it cannot: 32B on one 24 GB card
    assert not plan_model("qwen2.5-32b", "l4", cfg).fits


def test_bfee514_still_fails_serving() -> None:
    """The first fix (one resident weight copy) still let one prefill-sized eager call take tens of GB: a
    4096-row call on 7B's gate_up alone needed ~15 GB, so the 7B plan does not fit a T4 or even an H100."""
    p = plan_model("qwen2.5-7b", "t4", MemConfig(rev="bfee514", masked_head=False))
    assert p.devices[0].serving["eager_call"] > 14 * GIB
    assert not p.fits
    assert not plan_model("qwen2.5-72b", "h100-80", MemConfig(rev="bfee514")).fits
    cur = plan_model("qwen2.5-7b", "t4", MemConfig(masked_head=False))
    assert cur.devices[0].serving["eager_call"] <= MemConfig().eager_work_bytes


def test_weights_match_the_padded_tensors() -> None:
    """The plan's weight bytes are the padded tensors the CUDA path allocates (pad_weight_cuda, run on CPU)."""
    shape = MODELS["qwen2.5-0.5b"]
    lins = shape.linears(True)
    sizes = {}
    for _, m, n in lins:
        if (m, n) not in sizes:
            sizes[(m, n)] = pad_weight_cuda(torch.zeros((m, n), dtype=torch.int8), "cpu").numel()
            assert sizes[(m, n)] == padded_shape(m, n)[0] * padded_shape(m, n)[1]
    want = sum(sizes[(m, n)] for _, m, n in lins)
    p = plan_model("qwen2.5-0.5b", "t4", MemConfig())
    assert p.devices[0].steady["weights"] == want
    # the published sizes: 7B's linears + head ~6.6 GiB int8, 72B's ~66.6 GiB
    assert abs(plan_model("qwen2.5-7b", "h100-80").devices[0].steady["weights"] / GIB - 6.58) < 0.02
    assert abs(plan_model("qwen2.5-72b", "h100-80").devices[0].steady["weights"] / GIB - 66.55) < 0.05


def test_tensor_parallel_splits_rows_and_keeps_the_head_on_the_first_device() -> None:
    one = plan_model("qwen2.5-72b", "h100-80", MemConfig(tp=1, masked_head=False)).devices[0].steady["weights"]
    two = plan_model("qwen2.5-72b", "h100-80", MemConfig(tp=2, masked_head=True))
    head = 152064 * 8192
    w0, w1 = (d.steady["weights"] for d in two.devices)
    assert abs(w1 - one / 2) < 0.01 * one and abs(w0 - w1 - head) < 0.01 * one


def test_legacy_core_model_matches_measured_allocations() -> None:
    """The model of the old combine (used for the bfee514/run1 plans) is checked like the current one."""
    from pgw.core.ringmm import cpu_int8_available
    from tests.core.alloc_tracker import AllocPeak
    from tests.core.test_ringmm_memory import _legacy_offset_byte_core

    if not cpu_int8_available():
        pytest.skip("CPU torch._int_mm unavailable or inexact")
    for k, b, m, n in [(32, 9, 160, 64), (40, 40, 96, 288), (56, 3, 64, 32)]:
        w = torch.randint(-128, 128, (m, n), dtype=torch.int8)
        x = torch.randint(0, 1 << k, (b, n))
        rs = w.to(torch.int64).sum(1)
        with AllocPeak(exclude=[x, w, rs]) as t:
            _legacy_offset_byte_core(x, k, w, rs, offset_byte_rows(k, b))
        model = mp._legacy_core_bytes(k, b, n, n, m)
        assert 0.8 * model <= t.peak <= 1.25 * model, (k, b, t.peak, model)


def test_cli_table_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert mp.main(["--model", "qwen2.5-7b", "--gpu", "t4", "--gpu", "h100-80", "--tp", "1,2"]) == 0
    out = capsys.readouterr().out
    assert "| qwen2.5-7b | t4 | 1 |" in out and "| qwen2.5-7b | h100-80 | 2 |" in out
    assert mp.main(["--model", "qwen2.5-72b", "--gpu", "h100-80", "--json", "--rev", "bfee514"]) == 0
    d = json.loads(capsys.readouterr().out.splitlines()[0])
    assert d["fits"] is False and d["rev"] == "bfee514" and "eager_call" in d["serving"]
