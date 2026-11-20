"""Few-row ``stream`` variant, device tables and the per-config self-check of :mod:`pgw.core.triton_gemv`.

Every exactness test compares bit for bit with the digit GEMM (:class:`pgw.core.ringmm.PreparedWeight`,
``cpu_f64`` on CPU, ``cuda_int8`` on a GPU: the service's existing path) and, for small operands, with the
Python-int reference. CPU tests run the kernels in the Triton interpreter (``TRITON_INTERPRET=1``, set by
``tests/conftest.py`` without CUDA); ``gpu`` tests compile them (run on Kaggle). The offline-compile tests
build the CUDA kernels for sm_75/80/90 without a GPU (Triton's bundled ptxas) and check registers, spills
and vectorized weight loads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator

import numpy as np
import pytest
import torch

import pgw.core.triton_gemv as tg
from pgw.core import LinearId, LinearKind, LinearSpec, RingParams, ShapeError
from pgw.core.ringmm import PreparedWeight, cpu_int8_available, int8_ring_matmul, pad_weight_cuda, ringmm_reference
from pgw.core.triton_gemv import (
    MULTIROW_MAX_ROWS,
    TRITON_GEMV_KS,
    GemvConfig,
    TableRule,
    gemv_config,
    low32,
    ring_gemv,
    stream_config,
    triton_available,
)

needs_triton = pytest.mark.skipif(not triton_available(), reason="triton not installed")
SHAPES_7B = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}
SHAPES_05B = {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}
META = torch.device("meta")  # a device key no real call uses


@pytest.fixture
def interp() -> None:
    if os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
        pytest.skip("Triton interpreter not enabled (CUDA host: see the gpu tests)")


@pytest.fixture
def compiled() -> None:
    if os.environ.get("TRITON_INTERPRET", "0") not in ("", "0"):
        pytest.skip("TRITON_INTERPRET is set: GPU kernels are not compiled")


@pytest.fixture
def fresh() -> Iterator[None]:
    """Isolate the module's per-device state (self-check verdicts, tables, few-row mode)."""
    saved = [(d, dict(d)) for d in (tg._SELF_CHECK, tg._RULES, tg._MULTIROW, tg._EXTRA_TABLES, tg._SETTINGS)]
    no_dot = set(tg._NO_DOT)
    yield
    for d, v in saved:
        d.clear()
        d.update(v)
    tg._NO_DOT.clear()
    tg._NO_DOT.update(no_dot)


def _digit(x: torch.Tensor, w8: torch.Tensor, k: int) -> np.ndarray:
    """The digit GEMM on CPU (float digits, exact) as int64 ``[0, 2^K)``."""
    out: np.ndarray = PreparedWeight(w8, device="cpu", backend="cpu_f64").matmul(x, k).numpy()
    return out


def _gemv64(x: torch.Tensor, w: torch.Tensor, k: int, **kw: object) -> np.ndarray:
    y = ring_gemv(low32(x).to(w.device), w, k, **kw)  # type: ignore[arg-type]
    assert y.dtype == torch.int32
    out: np.ndarray = (y.cpu().to(torch.int64) & ((1 << k) - 1)).numpy()
    return out


def _rand(b: int, m: int, n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 1 << k, (b, n), dtype=torch.int64, generator=g),
            torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g))


# ---- host side (no triton needed) ------------------------------------------------------------
@pytest.mark.parametrize("b", range(1, MULTIROW_MAX_ROWS + 1))
@pytest.mark.parametrize(("m", "n"), list(SHAPES_7B.values()) + list(SHAPES_05B.values()) + [(3, 5)])
def test_stream_config_heuristic(b: int, m: int, n: int) -> None:
    cfg = stream_config(b, m, n)
    assert cfg.stream and not cfg.dot and cfg.block_r == b and cfg.variant == "stream"
    assert 1 <= cfg.split_k <= min(8, max(1, n // 512))
    assert b * cfg.block_m * cfg.block_n <= 8 * 1024  # <= 64 accumulator words per thread at 4 warps
    cfg.validate()


def test_config_validation() -> None:
    with pytest.raises(ShapeError):
        stream_config(MULTIROW_MAX_ROWS + 1, 64, 64)
    for bad in (GemvConfig(9, 16, 64, stream=True), GemvConfig(16, 16, 64, dot=True, stream=True),
                GemvConfig(2, 24, 64, stream=True), GemvConfig(2, 16, 64, 0, stream=True),
                GemvConfig(3, 16, 64), GemvConfig(2, 16, 64, 1, 3, stream=True)):
        with pytest.raises(ShapeError):
            bad.validate()
        with pytest.raises(ShapeError):
            ring_gemv(torch.zeros((2, 8), dtype=torch.int32), torch.zeros((4, 8), dtype=torch.int8), 32, config=bad)
    GemvConfig(5, 16, 64, stream=True).validate()  # any row count up to 8 (launched with exactly b rows)


def test_gemv_config_dispatch(fresh: None) -> None:
    """Rules first, then one-row sum, stream for rows <= multirow_rows, then dot (or the NO_DOT sum kernel)."""
    key = str(META)
    base1 = gemv_config(1, 4608, 3584, META)
    assert base1 == gemv_config(1, 4608, 3584)  # nothing registered for the device: the plain heuristic
    assert gemv_config(5, 4608, 3584, META).dot and tg.max_rows(META) == tg.TRITON_GEMV_MAX_ROWS
    tg._MULTIROW[key] = 4
    for b in range(2, 5):
        assert gemv_config(b, 4608, 3584, META) == stream_config(b, 4608, 3584, META)
    assert gemv_config(5, 4608, 3584, META).dot  # the dot variant keeps rows the stream variant does not serve
    assert tg.max_rows(META) == tg.TRITON_GEMV_MAX_ROWS and tg.multirow_rows(META) == 4
    tg._NO_DOT.add(key)
    assert tg.max_rows(META) == 4  # without tensor cores the stream variant bounds the fused rows
    assert not gemv_config(5, 4608, 3584, META).dot and not gemv_config(5, 4608, 3584, META).stream
    tg._MULTIROW.pop(key)
    assert tg.max_rows(META) == 1  # previous behaviour on a T4
    # a checked rule wins for its exact rows and shape range only
    rule = TableRule(1, GemvConfig(1, 8, 512, 2), (4000, 5000), (3584, 3584), "test")
    srule = TableRule(3, GemvConfig(3, 32, 32, 2, stream=True), (1, 1 << 20), (1, 1 << 20), "stream test")
    tg._RULES[key] = (rule, srule)
    assert gemv_config(1, 4608, 3584, META) == rule.config
    assert gemv_config(1, 3584, 3584, META) == gemv_config(1, 3584, 3584)  # out of the rule's range
    assert gemv_config(2, 4608, 3584, META) != rule.config
    assert not gemv_config(3, 64, 64, META).stream  # stream rule unused while the stream variant is off
    tg._MULTIROW[key] = 3
    assert gemv_config(3, 64, 64, META) == srule.config


def test_self_check_flow(fresh: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The few-row mode, the table check and every fallback, with the device checks faked."""
    dev = "meta"
    monkeypatch.setattr(tg, "triton_available", lambda: True)
    monkeypatch.setattr(tg, "_can_time", lambda device: True)
    monkeypatch.setattr(tg, "_timing_gate", lambda device: 5)
    checked: list[list[tuple[int, GemvConfig, object]]] = []
    verdict = {"configs": []}  # type: dict[str, list[str]]

    def fake_check_configs(device: object, entries: object, *, quick: bool = False) -> list[str]:
        checked.append(list(entries))  # type: ignore[call-overload]
        return verdict["configs"]

    monkeypatch.setattr(tg, "check_configs", fake_check_configs)

    def run(results: list[tuple[bool, str]], mode: str) -> bool:
        calls = iter(results)
        monkeypatch.setattr(tg, "_run_check", lambda key, device: next(calls))
        tg._reset(dev)
        tg.configure(multirow=mode)
        checked.clear()
        return tg.self_check(dev)

    # T4-like (no tensor cores), auto: the stream variant is checked, then timed: 2..5 rows
    assert run([(False, "dot"), (True, "")], "auto")
    assert tg.multirow_rows(dev) == 5 and tg.max_rows(dev) == 5
    stream_entries = [e for batch in checked for e in batch if e[1].stream]
    assert {e[0] for e in stream_entries} == set(range(2, MULTIROW_MAX_ROWS + 1))  # every heuristic row count
    assert {e[1].split_k > 1 for e in stream_entries} == {False, True}  # with and without split-K atomics
    # auto on a device with tensor cores: unchanged (dot variant for several rows)
    assert run([(True, "")], "auto") and tg.multirow_rows(dev) == 0 and tg.max_rows(dev) == tg.TRITON_GEMV_MAX_ROWS
    # off: previous behaviour
    assert run([(False, "dot"), (True, "")], "off") and tg.max_rows(dev) == 1
    # on: no timing, every checked row count, also with tensor cores
    assert run([(True, "")], "on") and tg.multirow_rows(dev) == MULTIROW_MAX_ROWS
    # a failing stream check switches the variant off; the one-row kernel keeps serving
    verdict["configs"] = ["bad"]
    assert run([(False, "dot"), (True, "")], "auto") and tg.max_rows(dev) == 1 and tg.multirow_rows(dev) == 0
    verdict["configs"] = []
    # a failing timing gate does too
    monkeypatch.setattr(tg, "_timing_gate", lambda device: (_ for _ in ()).throw(RuntimeError("no events")))
    assert run([(False, "dot"), (True, "")], "auto") and tg.max_rows(dev) == 1
    # the base check failing twice disables the fused kernel entirely
    assert not run([(False, "a"), (False, "b")], "on") and tg.multirow_rows(dev) == 0
    with pytest.raises(ValueError):
        tg.configure(multirow="sometimes")


def test_table_rules_checked_and_dropped(fresh: None, monkeypatch: pytest.MonkeyPatch) -> None:
    good = TableRule(1, GemvConfig(1, 8, 512, 2), (100, 200), (64, 4096), "good")
    bad = TableRule(1, GemvConfig(1, 16, 512, 4), (300, 400), (64, 4096), "bad")
    dot = TableRule(4, GemvConfig(16, 16, 64, 1, dot=True), (1, 1 << 20), (1, 1 << 20), "dot")
    tg.configure(rules={"meta": [good, bad, dot]})
    assert tg.device_rules(META)[:3] == (good, bad, dot)
    seen: list[tuple[int, GemvConfig, object]] = []

    def fake(device: object, entries: object, *, quick: bool = False) -> list[str]:
        es = list(entries)  # type: ignore[call-overload]
        seen.extend(es)
        return ["mismatch"] if es[0][1] == bad.config else []

    monkeypatch.setattr(tg, "check_configs", fake)
    tg._NO_DOT.add("meta")
    rules = tg._checked_rules("meta", META)
    assert rules == (good,)  # failing rule dropped; dot rule refused without tensor cores
    assert (1, good.config, (100, 64)) in seen  # checked at a shape inside the rule's range too


def test_json_table_round_trip_and_env(fresh: None, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    rules = {"Tesla T4": tg.DEVICE_TABLES["Tesla T4"],
             "meta": (TableRule(5, GemvConfig(5, 16, 64, 3, 8, stream=True), (10, 20), (30, 40), "x"),)}
    data = tg.table_to_json(rules)
    back = tg.load_table(json.loads(json.dumps(data)))
    assert back == {k: tuple(v) for k, v in rules.items()}
    data["devices"]["meta"].append({"rows": 9, "config": {"block_r": 9, "block_m": 16, "block_n": 64}, "m": [1, 2],
                                    "nw": [1, 2]})  # 9 rows: no stream variant, block_r not a power of two
    data["devices"]["meta"].append({"rows": 1})  # malformed
    path = os.path.join(str(tmp_path), "t.json")
    with open(path, "w") as f:
        json.dump(data, f)
    monkeypatch.setenv(tg.ENV_TABLE, path)
    assert tg.device_rules(META) == rules["meta"]
    monkeypatch.setenv(tg.ENV_TABLE, os.path.join(str(tmp_path), "missing.json"))
    assert tg.device_rules(META) == ()
    monkeypatch.setenv(tg.ENV_MULTIROW, "bogus")
    assert tg._multirow_mode() == "auto"
    monkeypatch.setenv(tg.ENV_MULTIROW, "OFF")
    assert tg._multirow_mode() == "off"


@pytest.mark.parametrize("content", ["[1, 2]", '{"devices": {"Tesla T4": 5}}', '{"devices": [1, 2]}', '"x"',
                                     '{"devices": {"meta": "abc", "Tesla T4": {"rows": 1}}}', "{bad json"])
def test_malformed_table_never_raises(fresh: None, monkeypatch: pytest.MonkeyPatch, tmp_path: object,
                                      content: str) -> None:
    """A malformed table file (wrong top level, a device entry that is no list) loads as no rules and the
    self-check still runs: the service falls back to the heuristic instead of failing at startup."""
    path = os.path.join(str(tmp_path), "t.json")
    with open(path, "w") as f:
        f.write(content)
    monkeypatch.setenv(tg.ENV_TABLE, path)
    assert tg.load_table(path) == {} or all(v == () for v in tg.load_table(path).values())
    assert tg.device_rules(META) == ()
    assert tg._checked_rules("meta", META) == ()
    monkeypatch.setattr(tg, "_run_check", lambda key, device: (True, ""))
    tg._reset("cpu")
    assert tg.self_check("cpu") is True
    assert tg._SELF_CHECK["cpu"] is True and tg._RULES["cpu"] == ()


def test_checked_rules_survives_a_raising_table_lookup(fresh: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(device: object) -> tuple[TableRule, ...]:
        raise AttributeError("bad table")

    monkeypatch.setattr(tg, "device_rules", boom)
    assert tg._checked_rules("meta", META) == ()


def test_t4_table_matches_the_measured_shapes() -> None:
    """The built-in T4 rules cover the four 7B linears (one row) and are valid launch configs."""
    rules = tg.DEVICE_TABLES["Tesla T4"]
    for r in rules:
        r.validate()
    for m, n in SHAPES_7B.values():
        hit = [r for r in rules if r.matches(1, m, n)]
        assert hit, (m, n)
        assert not any(r.matches(2, m, n) for r in rules)
    assert next(r for r in rules if r.matches(1, 37888, 3584)).config == GemvConfig(1, 32, 1024, 8)
    assert next(r for r in rules if r.matches(1, 3584, 18944)).config == GemvConfig(1, 32, 1024, 8)


# ---- interpreter (CPU) exactness -----------------------------------------------------------
@needs_triton
@pytest.mark.usefixtures("interp")
@pytest.mark.parametrize("k", TRITON_GEMV_KS)
@pytest.mark.parametrize(("m", "n"), [(1, 1), (5, 7), (37, 129), (33, 700)])
def test_stream_random_exact_rows_1_to_8(k: int, m: int, n: int) -> None:
    for b in range(1, MULTIROW_MAX_ROWS + 1):
        x, w = _rand(b, m, n, k, seed=1000 * k + 10 * b + m)
        want = ringmm_reference(x, w, k)
        np.testing.assert_array_equal(_digit(x, w, k), want)
        for cfg in (stream_config(b, m, n), GemvConfig(b, 16, 32, 3, stream=True)):
            np.testing.assert_array_equal(_gemv64(x, w, k, config=cfg), want, err_msg=f"b={b} {cfg}")
        if cpu_int8_available() and b in (1, 8):
            np.testing.assert_array_equal(int8_ring_matmul(x, w, k).numpy(), want)


@needs_triton
@pytest.mark.usefixtures("interp")
@pytest.mark.parametrize("k", TRITON_GEMV_KS)
def test_stream_adversarial_operands_wrap_exactly(k: int) -> None:
    """x_hat in {0, 2^K - 1, 2^(K-1), 2^(K-1) - 1, random} against W in {-128, 127, mixed}: every
    partial sum wraps mod 2^32 many times (n = 2049: a partial last tile)."""
    top = (1 << k) - 1
    n, m = 2049, 9
    g = torch.Generator().manual_seed(k)
    rows = [torch.full((n,), v, dtype=torch.int64) for v in (0, top, 1 << (k - 1), (1 << (k - 1)) - 1)]
    rows += [torch.randint(0, 1 << k, (n,), dtype=torch.int64, generator=g) for _ in range(4)]
    x_all = torch.stack(rows)  # 8 adversarial rows
    w_mixed = torch.where(torch.arange(n) % 3 == 0, -128, 127).to(torch.int8).repeat(m, 1)
    for w in (torch.full((m, n), -128, dtype=torch.int8), torch.full((m, n), 127, dtype=torch.int8), w_mixed):
        for b in range(1, MULTIROW_MAX_ROWS + 1):
            x = x_all[:b]
            want = ringmm_reference(x, w, k)
            for cfg in (stream_config(b, m, n), GemvConfig(b, 8, 256, 4, stream=True)):
                np.testing.assert_array_equal(_gemv64(x, w, k, config=cfg), want, err_msg=f"b={b} {cfg}")


@needs_triton
@pytest.mark.usefixtures("interp")
def test_wraparound_exactness_long_inner_dimension() -> None:
    """The 7B down projection's inner length (18944): sum of 18944 products of (2^32 - 1) and -128 is
    ~2^50, far past 2^32; the wrapping uint32 accumulation must equal the exact integer mod 2^32."""
    n, m = 18944, 3
    for b in (1, 2, 8):
        x = torch.full((b, n), (1 << 32) - 1, dtype=torch.int64)
        x[-1, ::2] = 1 << 31
        w = torch.full((m, n), -128, dtype=torch.int8)
        w[1, 1::2] = 127
        want = np.array([[sum(int(a) * int(c) for a, c in zip(x[r].tolist(), w[j].tolist(), strict=True)) % (1 << 32)
                          for j in range(m)] for r in range(b)], dtype=np.int64)
        for cfg in (stream_config(b, m, n), GemvConfig(b, 4, 1024, 8, stream=True)):
            np.testing.assert_array_equal(_gemv64(x, w, 32, config=cfg), want, err_msg=f"b={b} {cfg}")


@needs_triton
@pytest.mark.usefixtures("interp")
@pytest.mark.parametrize(
    "cfg",
    [GemvConfig(1, 4, 64, 1, stream=True), GemvConfig(2, 8, 128, 3, stream=True), GemvConfig(3, 16, 64, 2, stream=True),
     GemvConfig(5, 32, 32, 5, 8, stream=True), GemvConfig(8, 16, 64, 7, stream=True),
     GemvConfig(8, 64, 256, 1, stream=True), GemvConfig(4, 4, 64, 2, 1, stream=True)],
)
def test_stream_tiles_and_split_any_order_is_exact(cfg: GemvConfig) -> None:
    """Every tile shape, warp count and split-K factor gives the same bits as the digit GEMM, also when the
    config's block_r exceeds the rows (the kernel is launched with exactly b accumulators)."""
    for b in sorted({1, cfg.block_r, max(1, cfg.block_r - 1)}):
        for k in TRITON_GEMV_KS:
            x, w = _rand(b, 41, 1001, k, seed=cfg.block_m + cfg.split_k + b)
            w[0] = -128
            x[0] = (1 << k) - 1
            np.testing.assert_array_equal(_gemv64(x, w, k, config=cfg), _digit(x, w, k), err_msg=f"b={b} K={k}")


@needs_triton
@pytest.mark.usefixtures("interp")
def test_stream_padded_weight_and_any_int64_input() -> None:
    """The service passes the 32-padded weight; padding columns/rows never matter; x is taken mod 2^K."""
    k = 24
    x, w8 = _rand(6, 45, 70, k, seed=5)
    wpad = pad_weight_cuda(w8, "cpu")
    wpad[:, 70:] = 77
    wpad[45:] = -3
    xw = x + (torch.randint(-(1 << 20), 1 << 20, x.shape, dtype=torch.int64) << 40)
    xw[1] -= 1 << 62
    for cfg in (stream_config(6, 45, int(wpad.shape[1])), GemvConfig(6, 16, 32, 2, stream=True)):
        np.testing.assert_array_equal(_gemv64(xw, wpad, k, n=70, m=45, config=cfg), ringmm_reference(x, w8, k))
    out = torch.full((6, 45), 123, dtype=torch.int32)
    y = ring_gemv(low32(xw), wpad, k, n=70, m=45, out=out, config=GemvConfig(6, 16, 32, 2, stream=True))
    assert y is out
    np.testing.assert_array_equal(y.to(torch.int64).numpy(), ringmm_reference(x, w8, k))


@needs_triton
@pytest.mark.slow
@pytest.mark.usefixtures("interp")
@pytest.mark.parametrize("name", list(SHAPES_05B))
def test_stream_qwen05b_shapes_vs_digit_gemm(name: str) -> None:
    """Qwen2.5-0.5B linear shapes (full size, interpreter), stream and one-row kernels vs the digit GEMM.
    Large tiles keep the interpreter fast; tile shape does not change the bits (tested above)."""
    m, n = SHAPES_05B[name]
    g = torch.Generator().manual_seed(len(name))
    w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    wpad = pad_weight_cuda(w8, "cpu")
    rows = range(1, MULTIROW_MAX_ROWS + 1) if m * n <= 1 << 21 else (1, 3, 8)
    for b in rows:
        x = torch.randint(0, 1 << 32, (b, n), dtype=torch.int64, generator=g)
        x[0, : n // 2] = (1 << 32) - 1
        want = _digit(x, w8, 32)
        cfg = GemvConfig(b, 64, 512, 2, stream=True)
        np.testing.assert_array_equal(_gemv64(x, wpad, 32, n=n, m=m, config=cfg), want, err_msg=f"b={b}")
        if b == 1:
            np.testing.assert_array_equal(_gemv64(x, wpad, 32, n=n, m=m, config=GemvConfig(1, 64, 1024, 2)), want)


@needs_triton
@pytest.mark.usefixtures("interp")
def test_t4_table_configs_equal_previous_heuristic_bitwise() -> None:
    """New path == old path: each built-in T4 rule's config gives the same bits as the previous
    heuristic config and the digit GEMM (split-K with partial tiles, every K)."""
    m, n = 70, 4100
    for rule in tg.DEVICE_TABLES["Tesla T4"]:
        for k in TRITON_GEMV_KS:
            x, w = _rand(1, m, n, k, seed=rule.config.block_m + k)
            x[0, ::7] = (1 << k) - 1
            old = _gemv64(x, w, k, config=gemv_config(1, m, n))  # previous heuristic (no device table)
            new = _gemv64(x, w, k, config=rule.config)
            np.testing.assert_array_equal(new, old, err_msg=f"{rule.note} K={k}")
            np.testing.assert_array_equal(new, _digit(x, w, k))


@needs_triton
def test_stream_helpers_are_jitted_per_mode() -> None:
    """The stream kernel's device helpers are jitted with the kernel (per mode), in a private copy of the
    kernel module's globals: the module keeps the plain functions, so another mode never inherits them."""
    from pgw.core import _triton_gemv_kernel as km

    k_a, k_b = tg._kernel(True, "stream"), tg._kernel(False, "stream")
    for kern in (k_a, k_b):
        g = kern.fn.__globals__
        assert g is not vars(km)
        for h in km.ROWS_HELPERS:
            assert g[h] is not getattr(km, h) and g[h].fn is getattr(km, h)
    assert k_a.fn.__globals__ is not k_b.fn.__globals__
    assert all(callable(getattr(km, h)) and not hasattr(getattr(km, h), "fn") for h in km.ROWS_HELPERS)


@needs_triton
@pytest.mark.usefixtures("interp")
def test_heuristic_stream_variants_pass_self_check_in_interpreter() -> None:
    """The self-check's stream entries (every heuristic row count, with and without split-K) pass."""
    entries = [(b, c, None) for b, c in tg.heuristic_variants(stream=True)]
    assert tg.check_configs("cpu", entries, quick=True) == []


@needs_triton
@pytest.mark.usefixtures("interp")
def test_self_check_mode_on_serves_stream_rows_on_cpu_executor(fresh: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """``on`` mode end to end in the interpreter (base check faked to keep it short): the CPU executor's
    ``run_gemv`` then launches the stream variant for 2-8 rows, bit-identical to the digit GEMM."""
    from pgw.service.gpu_exec import DeviceExecutor, from_compact

    monkeypatch.setattr(tg, "_run_check", lambda key, device: (True, ""))
    monkeypatch.setattr(tg, "check_configs", lambda device, entries, quick=False: tg._run_cases(
        torch.device(device), tg._config_cases(list(entries), True)))
    tg._reset("cpu")
    tg.configure(multirow="on")
    assert tg.self_check("cpu") and tg.multirow_rows("cpu") == MULTIROW_MAX_ROWS
    launched: list[str] = []
    orig = tg._kernel

    def spy(interpret: bool, variant: str) -> object:
        launched.append(variant)
        return orig(interpret, variant)

    monkeypatch.setattr(tg, "_kernel", spy)
    ex = DeviceExecutor("cpu")
    for k in TRITON_GEMV_KS:
        for b in (1, 2, 5, 8):
            x, w8 = _rand(b, 50, 99, k, seed=k + b)
            y = ex.run_gemv(x, k, pad_weight_cuda(w8, "cpu"), 99, 50)
            np.testing.assert_array_equal(from_compact(y, k).numpy(), _digit(x, w8, k))
            assert launched[-1] == ("sum" if b == 1 else "stream")
    tg._reset("cpu")


# ---- offline compile for CUDA archs (no GPU needed) -------------------------------------------
#: Runs in a child process: the kernels must be jitted for the compiler, not the interpreter this
#: process uses (``TRITON_INTERPRET`` is fixed at triton's first import).
_AOT_SCRIPT = r"""
import json, os, re, subprocess, sys, tempfile
import triton, triton.backends.nvidia as nv
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from pgw.core.triton_gemv import _kernel

cc, configs = int(sys.argv[1]), json.loads(sys.argv[2])
sig = {"x_ptr": "*i32", "w_ptr": "*i8", "y_ptr": "*i32", "B": "i32", "N": "i32", "M": "i32", "NW": "i32",
       "n_per_split": "i32", "kmask": "i32", "stride_x": "i32", "stride_w": "i32", "stride_y": "i32",
       "BLOCK_R": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "ATOMIC": "constexpr"}
names = list(sig)
# what the launcher specializes for the service's padded weights (multiples of 32) and 7B/0.5B dims
div16 = ("x_ptr", "w_ptr", "y_ptr", "N", "M", "NW", "n_per_split", "stride_x", "stride_w", "stride_y")
from triton.backends.nvidia.compiler import get_ptxas
ptxas = get_ptxas(cc).path  # the ptxas triton itself uses for this arch
out = []
for c in configs:
    consts = {"BLOCK_R": c["block_r"], "BLOCK_M": c["block_m"], "BLOCK_N": c["block_n"], "ATOMIC": c["split_k"] > 1}
    src = ASTSource(fn=_kernel(False, c["variant"]), signature=sig,  # the service's own jitted kernel
                    constexprs={(names.index(k),): v for k, v in consts.items()},
                    attrs={(names.index(n),): [["tt.divisibility", 16]] for n in div16})
    ptx = triton.compile(src, target=GPUTarget("cuda", cc, 32), options={"num_warps": c["num_warps"]}).asm["ptx"]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ptx")
        open(p, "w").write(ptx)
        arch = re.search(r"\.target\s+(sm_\w+)", ptx).group(1)  # sm_90a on Hopper
        info = subprocess.run([ptxas, "-arch=" + arch, "-v", p, "-o", os.devnull], capture_output=True,
                              text=True, timeout=300).stderr
    out.append({"regs": [int(v) for v in re.findall(r"Used (\d+) registers", info)],
                "spills": [int(v) for v in re.findall(r"(\d+) bytes spill stores", info)],
                "vector_loads": len(re.findall(r"ld\.global[^\n]*\.v[24]\.", ptx)),
                "atomics": len(re.findall(r"red\.global\.add\.u32|atom\.global", ptx)), "info": info[-400:]})
print(json.dumps(out))
"""


def _aot(configs: list[GemvConfig], cc: int) -> list[dict[str, object]]:
    """Compile the variants for ``sm_<cc>`` with Triton's bundled toolchain (child process, no GPU)."""
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(src), env.get("PYTHONPATH", "")])
    arg = json.dumps([{"variant": c.variant, "block_r": c.block_r, "block_m": c.block_m, "block_n": c.block_n,
                       "split_k": c.split_k, "num_warps": c.num_warps} for c in configs])
    cmd = [sys.executable, "-c", _AOT_SCRIPT, str(cc), arg]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)  # noqa: S603
    assert res.returncode == 0, res.stderr[-3000:]
    out: list[dict[str, object]] = json.loads(res.stdout.strip().splitlines()[-1])
    return out


def _aot_available() -> bool:
    if not triton_available() or os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
        return False  # on CUDA hosts the gpu tests compile for the real device
    try:
        import triton.backends.nvidia as nv

        return os.path.exists(os.path.join(os.path.dirname(nv.__file__), "bin", "ptxas"))
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _aot_available(), reason="no offline CUDA toolchain in triton (or a CUDA host)")
@pytest.mark.parametrize("cc", [75, 80, 90])
def test_stream_variant_compiles_for_cuda_without_spills(cc: int) -> None:
    """Every heuristic stream variant (2..8 rows, with and without split-K) and the T4 table's one-row
    configs compile for T4/A100/H100-class GPUs with no register spills; weight loads are vectorized."""
    entries = [c for _, c in tg.heuristic_variants(stream=True) if c.stream]
    if cc != 75:
        entries = [c for c in entries if c.split_k > 1 and c.block_r in (2, 5, 8)]
    else:
        entries += [r.config for r in tg.DEVICE_TABLES["Tesla T4"]]
    for cfg, r in zip(entries, _aot(entries, cc), strict=True):
        assert r["spills"] and max(r["spills"]) == 0, (cfg, r["info"])  # type: ignore[call-overload]
        assert r["regs"] and max(r["regs"]) <= 255, (cfg, r["info"])  # type: ignore[call-overload]
        assert r["vector_loads"], f"no vectorized loads for {cfg}"
        assert bool(r["atomics"]) == (cfg.split_k > 1), cfg  # wrapping unsigned atomics only with split-K


# ---- GPU (compiled) ----------------------------------------------------------------------------
def _cuda_ref(x: torch.Tensor, w8: torch.Tensor, k: int) -> np.ndarray:
    out: np.ndarray = PreparedWeight(w8, device="cuda").matmul(x.cuda(), k).cpu().numpy()
    return out


@pytest.mark.gpu
@pytest.mark.usefixtures("compiled")
@pytest.mark.parametrize("name", [f"7b.{k}" for k in SHAPES_7B] + [f"0.5b.{k}" for k in SHAPES_05B])
def test_gpu_stream_all_shapes_rows_1_to_8(name: str) -> None:  # pragma: no cover - GPU only
    """Stream variant (heuristic, split and tile variants) vs the cuda_int8 digit GEMM for rows 1..8, every
    K, every 7B and 0.5B shape, random and adversarial x_hat (0, 2^K - 1, 2^(K-1))."""
    set_name, lin = name.split(".")
    m, n = (SHAPES_7B if set_name == "7b" else SHAPES_05B)[lin]
    g = torch.Generator().manual_seed(len(name))
    w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    w8[0] = -128
    w8[1] = 127
    wpad = pad_weight_cuda(w8, "cuda")
    nw = int(wpad.shape[1])
    for k in TRITON_GEMV_KS:
        top = (1 << k) - 1
        for b in range(1, MULTIROW_MAX_ROWS + 1):
            x = torch.randint(0, 1 << k, (b, n), dtype=torch.int64, generator=g)
            x[0] = top
            if b > 1:
                x[1] = 0
            if b > 2:
                x[2] = 1 << (k - 1)
            want = _cuda_ref(x, w8, k)
            for cfg in (stream_config(b, m, nw, torch.device("cuda:0")), GemvConfig(b, 32, 64, 1, stream=True),
                        GemvConfig(b, 16, 128, 8, 8, stream=True)):
                np.testing.assert_array_equal(_gemv64(x, wpad, k, n=n, m=m, config=cfg), want, err_msg=f"{k} {b} {cfg}")


@pytest.mark.gpu
@pytest.mark.usefixtures("compiled")
def test_gpu_self_check_table_and_multirow(fresh: None) -> None:  # pragma: no cover - GPU only
    """self_check checks every table rule and stream variant; served row counts are exact end to end."""
    from pgw.service.gpu_exec import device_executor
    from pgw.service.kernels import ComputeKernel

    tg._reset("cuda:0")
    assert tg.self_check("cuda:0")
    name = tg.device_name("cuda:0")
    if name == "Tesla T4":
        assert len(tg._RULES["cuda:0"]) == len(tg.DEVICE_TABLES["Tesla T4"])  # every measured rule passed
    print(f"{name}: max_rows={tg.max_rows('cuda:0')} stream rows={tg.multirow_rows('cuda:0')} "
          f"rules={len(tg._RULES['cuda:0'])}")
    m, n = 300, 1000
    w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=torch.Generator().manual_seed(9))
    spec = LinearSpec(LinearId(0, LinearKind.O), m, n)
    kern = ComputeKernel(spec, w8, device="cuda", cuda_graphs=False)
    ex = device_executor("cuda:0")
    for k in TRITON_GEMV_KS:
        ring = RingParams(k, 12 if k > 16 else 4)
        for b in range(1, MULTIROW_MAX_ROWS + 2):
            x, _ = _rand(b, 1, n, k, seed=b + k)
            gemv0 = ex.stats.gemv
            np.testing.assert_array_equal(kern.run(x, ring).numpy(), ringmm_reference(x, w8, k))
            assert ex.stats.gemv - gemv0 == (2 if b <= tg.max_rows("cuda:0") else 0)  # first call cross-checked


@pytest.mark.gpu
@pytest.mark.usefixtures("compiled")
def test_gpu_timing_gate_and_mode_on(fresh: None) -> None:  # pragma: no cover - GPU only
    rows = tg._timing_gate(torch.device("cuda:0"))
    assert 0 <= rows <= MULTIROW_MAX_ROWS
    tg._reset("cuda:0")
    tg.configure(multirow="on")
    assert tg.self_check("cuda:0") and tg.multirow_rows("cuda:0") == MULTIROW_MAX_ROWS
    assert tg.gemv_config(5, 4608, 3584, torch.device("cuda:0")).stream
    tg._reset("cuda:0")
