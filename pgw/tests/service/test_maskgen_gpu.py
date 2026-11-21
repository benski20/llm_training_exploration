"""Device-side provider maskgen (:mod:`pgw.service.maskgen_gpu`): byte-identical to the reference path.

The module is device-agnostic torch code, so the CPU tests run it on CPU tensors (the Triton combine
in the interpreter, ``TRITON_INTERPRET=1`` from ``tests/conftest.py``) and compare it bit for bit with
the wire reference (:func:`pack_y_v2`, :func:`pack_y`, :func:`unpack_bits`) on top of
:func:`matmul_mod64`. The ``gpu`` tests compare the CUDA service path with the previous one.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import pgw.service.kernels as kernels_mod
from pgw.core.types import LinearId, LinearSpec
from pgw.hemask import DEGREE
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.wire import (
    MaskgenItem,
    pack_bits,
    pack_y,
    pack_y_v2,
    unpack_b,
    unpack_bits,
    y_round_offset,
    y_wire_bytes,
)
from pgw.service import maskgen_gpu as mg
from pgw.service.kernels import ComputeKernel, _combine_planes

BITS = [32, 35, 43, 48, 53, 56, 61, 64]


def _vals(count: int, bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 1 << 63, size=count, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, size=count,
                                                                                           dtype=np.uint64)
    if bits < 64:
        v &= np.uint64((1 << bits) - 1)
    v[:4] = np.array([0, 1, (1 << bits) - 1, (1 << bits) - 2], dtype=np.uint64)
    return v


def _weights(m: int, n: int, seed: int, extreme: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if extreme:
        return rng.choice(np.array([-128, -127, 127], dtype=np.int8), size=(m, n))
    w = rng.integers(-128, 128, size=(m, n)).astype(np.int8)
    w[0, :] = 127
    if m > 1:
        w[1, :] = -128
    return w


def _pad(w: np.ndarray) -> torch.Tensor:
    m, n = w.shape
    wp = torch.zeros((-(-m // 32) * 32, -(-n // 32) * 32), dtype=torch.int8)
    wp[:m, :n] = torch.from_numpy(w)
    return wp


@pytest.fixture
def cpu_int8() -> None:
    if not kernels_mod.cpu_int8_available():
        pytest.skip("CPU torch._int_mm unavailable or inexact")


# ---- v2 unpack / pack ------------------------------------------------------------------------------
@pytest.mark.parametrize("bits", BITS)
def test_v2_planes_source_is_unpack_bits(bits: int) -> None:
    n, cols = 7, 64
    v = _vals(n * cols, bits, bits)
    packed = torch.from_numpy(np.frombuffer(bytes(pack_bits(v, bits)), dtype=np.uint8).copy())
    srcs = mg.v2_planes_source(packed, n, cols, bits)
    assert len(srcs) == -(-bits // 8)
    ref = v.reshape(n, cols).view(np.uint8).reshape(n, cols, 8)
    for d, (src, j) in enumerate(srcs):
        assert np.array_equal(src[:, :, j].numpy(), ref[:, :, d]), d


def test_v2_planes_source_rejects_bad_size() -> None:
    with pytest.raises(ValueError, match="expected"):
        mg.v2_planes_source(torch.zeros(10, dtype=torch.uint8), 2, 8, 53)


@pytest.mark.parametrize("b_bits,y_bits", [(53, 35), (56, 56), (64, 64), (64, 35), (53, 53), (48, 40), (61, 43),
                                           (35, 8), (40, 33)])
def test_pack_v2_block_matches_pack_y_v2(b_bits: int, y_bits: int) -> None:
    m, cols = 5, 96
    y = _vals(m * cols, 64, b_bits * 100 + y_bits).reshape(m, cols)
    ref = bytes(pack_y_v2(y.copy(), b_bits, y_bits))
    out = torch.empty(m * cols * y_bits // 8, dtype=torch.uint8)
    yt = torch.from_numpy(y.view(np.int64).copy())
    for c0, cb in ((0, 32), (32, 8), (40, 56)):  # uneven blocks, multiples of 8
        mg.pack_v2_block(yt[:, c0:c0 + cb], b_bits, y_bits, out, cols, c0)
    assert out.numpy().tobytes() == ref


def test_pack_v2_block_rejects_unaligned_block() -> None:
    out = torch.empty(8 * 35, dtype=torch.uint8)
    with pytest.raises(ValueError, match="multiples of 8"):
        mg.pack_v2_block(torch.zeros((1, 4), dtype=torch.int64), 53, 35, out, 64, 4)


# ---- combine -------------------------------------------------------------------------------------
def _parts(mp: int, nb: int, cbp: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.randint(-(2**31), 2**31 - 1, (mp, nb, cbp), generator=g, dtype=torch.int32)
    p[0] = 2**31 - 1
    p[1] = -(2**31)
    return p


@pytest.mark.parametrize("nb", [1, 4, 5, 7, 8])
def test_triton_combine_matches_torch(nb: int) -> None:
    pytest.importorskip("triton")
    if os.environ.get("TRITON_INTERPRET", "0") in ("", "0") and not torch.cuda.is_available():
        pytest.skip("no CUDA and no Triton interpreter")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, cb = 37, 288
    part = _parts(m + 3, nb, cb + 32, nb).to(dev)
    const = torch.randint(-(2**63), 2**63 - 1, (m, 1), dtype=torch.int64).to(dev)
    const[0] = -(2**63)
    a = mg.combine(part, m, cb, const, fused=True)
    b = mg.combine(part, m, cb, const, fused=False)
    assert torch.equal(a, b)
    assert torch.equal(b, _combine_planes([part[:m, d, :cb] for d in range(nb)], const))


def test_triton_combine_self_check_on_cpu_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("triton")
    if os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
        pytest.skip("needs the Triton interpreter")
    monkeypatch.setattr(mg, "_TRITON_OK", {})
    assert mg.triton_combine_ok(torch.device("cpu")) is True
    assert mg._TRITON_OK == {"cpu": True}


def test_triton_combine_falls_back_when_self_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mg, "_TRITON_OK", {})
    monkeypatch.setattr(mg, "_triton_ready", lambda dev: True)
    monkeypatch.setattr(mg, "_self_check", lambda dev: False)
    assert mg.triton_combine_ok(torch.device("cpu")) is False
    part = _parts(34, 7, 64, 3)
    const = torch.zeros((32, 1), dtype=torch.int64)
    assert torch.equal(mg.combine(part, 32, 40, const), _combine_planes([part[:32, d, :40] for d in range(7)], const))


def test_combine_checks_each_kernel_specialisation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A (NB, BLOCK) specialisation that fails its own check falls back to torch; others stay fused."""
    monkeypatch.setattr(mg, "_TRITON_OK", {"cpu": True})
    monkeypatch.setattr(mg, "_SPEC_OK", {})
    seen: list[tuple[int, int, int]] = []

    def fake_case(dev: torch.device, nb: int, m: int, cb: int, g: torch.Generator) -> bool:
        seen.append((nb, m, cb))
        return nb != 7

    monkeypatch.setattr(mg, "_check_case", fake_case)
    fused_calls: list[int] = []
    monkeypatch.setattr(mg, "_combine_triton", lambda part, m, cb, const: fused_calls.append(cb) or
                        mg._combine_torch(part, m, cb, const))
    const = torch.zeros((32, 1), dtype=torch.int64)
    p7, p8 = _parts(34, 7, 288, 3), _parts(34, 8, 288, 3)
    assert torch.equal(mg.combine(p7, 32, 256, const), _combine_planes([p7[:32, d, :256] for d in range(7)], const))
    assert fused_calls == []  # NB=7 failed its check: torch reference
    mg.combine(p8, 32, 256, const)
    mg.combine(p8, 32, 64, const)
    mg.combine(p8, 32, 256, const)  # cached
    assert fused_calls == [256, 64, 256]
    assert mg._SPEC_OK == {("cpu", 7, 256): False, ("cpu", 8, 256): True, ("cpu", 8, 32): True}
    assert [s[0] for s in seen] == [7, 8, 8]


# ---- one item on CPU tensors ---------------------------------------------------------------------
def _const(w: np.ndarray, nb: int, rnd: int) -> torch.Tensor:
    kern = ComputeKernel(LinearSpec(LinearId(0, "qkv"), *w.shape), torch.from_numpy(w.copy()), device="cpu",
                         backend="cpu_f64")
    return kern._maskgen_const(nb, rnd)


@pytest.mark.parametrize("m,n,cols", [(37, 70, DEGREE), (96, 64, 2 * DEGREE)])
@pytest.mark.parametrize("q,drop", [(56, 16), (64, 0), (64, 16), (56, 0)])
@pytest.mark.parametrize("fused", [False, True])
def test_maskgen_device_v1_is_exact(cpu_int8: None, m: int, n: int, cols: int, q: int, drop: int,
                                    fused: bool) -> None:
    if fused and os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
        pytest.skip("needs the Triton interpreter")
    w = _weights(m, n, m + n, extreme=(m == 96))
    b = _vals(n * cols, q, q + drop).reshape(n, cols)
    data = np.ascontiguousarray(b, dtype="<u8").view(np.uint8).reshape(n, cols, 8)[:, :, :q // 8].tobytes()
    ylo, yhi = y_wire_bytes(q, drop)
    out = mg.maskgen_device(_pad(w), m, n, torch.from_numpy(np.frombuffer(data, np.uint8).copy()), cols,
                            _const(w, q // 8, y_round_offset(drop)), in_bytes=q // 8, ylo=ylo, yhi=yhi,
                            part_bytes=m * (q // 8) * 4 * 64, fused=fused)  # 64-column blocks
    ref = pack_y(matmul_mod64(w, unpack_b(data, (n, cols), q), fast=False), drop, q)
    assert out.numpy().tobytes() == ref


@pytest.mark.parametrize("b_bits,y_bits", [(53, 35), (56, 56), (64, 40), (45, 32)])
@pytest.mark.parametrize("fused", [False, True])
def test_maskgen_device_v2_is_exact(cpu_int8: None, b_bits: int, y_bits: int, fused: bool) -> None:
    if fused and os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
        pytest.skip("needs the Triton interpreter")
    m, n, cols = 45, 66, DEGREE
    w = _weights(m, n, b_bits)
    b = _vals(n * cols, b_bits, y_bits)
    data = bytes(pack_bits(b, b_bits))
    nb = -(-b_bits // 8)
    out = mg.maskgen_device(_pad(w), m, n, torch.from_numpy(np.frombuffer(data, np.uint8).copy()), cols,
                            _const(w, nb, 0), b_bits=b_bits, y_bits=y_bits, part_bytes=m * nb * 4 * 96, fused=fused)
    ref = bytes(pack_y_v2(matmul_mod64(w, unpack_bits(data, n * cols, b_bits).reshape(n, cols), fast=False),
                          b_bits, y_bits))
    assert out.numpy().tobytes() == ref


# ---- service wiring ------------------------------------------------------------------------------
def test_fused_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.service import handler as handler_mod

    monkeypatch.delenv(mg.ENV_FUSED, raising=False)
    assert mg.fused_enabled()
    cuda_kern = SimpleNamespace(device=torch.device("cuda", 0), _pw=object())
    assert handler_mod._maskgen_fused(cuda_kern)
    assert not handler_mod._maskgen_fused(SimpleNamespace(device=torch.device("cpu"), _pw=object()))
    assert not handler_mod._maskgen_fused(None)
    monkeypatch.setenv(mg.ENV_FUSED, "0")
    assert not handler_mod._maskgen_fused(cuda_kern)
    monkeypatch.setenv(mg.ENV_YIELD_MS, "2.5")
    assert mg.yield_ms() == 2.5
    monkeypatch.setenv(mg.ENV_YIELD_MS, "x")
    assert mg.yield_ms() == 0.0
    monkeypatch.delenv(mg.ENV_YIELD_MS)
    assert mg.yield_ms() == 0.0


async def test_maskgen_slots_are_per_cuda_device() -> None:
    from pgw.service import handler as handler_mod

    h = handler_mod.ServiceHandler.__new__(handler_mod.ServiceHandler)
    h.maskgen_concurrency = 2
    h._maskgen_slots = asyncio.Semaphore(2)
    h._maskgen_dev_slots = {}
    k0 = SimpleNamespace(device=torch.device("cuda", 0))
    k1 = SimpleNamespace(device=torch.device("cuda", 1))
    s0, s1 = h._maskgen_slots_for(k0), h._maskgen_slots_for(k1)
    assert s0 is not s1 and s0 is h._maskgen_slots_for(k0)
    assert h._maskgen_slots_for(SimpleNamespace(device=torch.device("cpu"))) is h._maskgen_slots
    assert h._maskgen_slots_for(None) is h._maskgen_slots


async def test_maskgen_request_items_run_concurrently_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The items of one request overlap (up to the slots) and the reply keeps request order."""
    import threading
    import time

    from pgw.hemask.wire import MaskgenRequest
    from pgw.service import handler as handler_mod

    active = peak = 0
    lock = threading.Lock()

    def item(kern: Any, it: Any) -> bytes:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.1 if it.linear.endswith("qkv") else 0.02)
        with lock:
            active -= 1
        return it.linear.encode()

    monkeypatch.setattr(handler_mod, "_maskgen_item", item)
    h = handler_mod.ServiceHandler.__new__(handler_mod.ServiceHandler)
    h.maskgen_concurrency = 3
    h._maskgen_slots = asyncio.Semaphore(3)
    h._maskgen_dev_slots = {}
    h.inflight = 0
    h._idle = asyncio.Event()
    h._admit = lambda tenant, rows: None
    dims = {f"0.{k}": (8, 8) for k in ("qkv", "o", "gate_up", "down")}
    h.registry = SimpleNamespace(get=lambda fp: SimpleNamespace(linear_dims=dims, kernel=lambda lid: None))
    items = tuple(MaskgenItem(lid, DEGREE, bytes(8 * DEGREE * 8)) for lid in dims)
    resp = await h.maskgen(MaskgenRequest("r", "fp", items), None)
    assert [r.y for r in resp.items] == [lid.encode() for lid in dims]
    assert peak == 3


# ---- GPU -----------------------------------------------------------------------------------------
def _gpu_kernel(w: np.ndarray, kind: str = "qkv") -> ComputeKernel:
    m, n = w.shape
    return ComputeKernel(LinearSpec(LinearId(0, kind), m, n), torch.from_numpy(w.copy()), device="cuda")


@pytest.mark.gpu
def test_cuda_triton_combine_self_check() -> None:  # pragma: no cover - GPU only
    pytest.importorskip("triton")
    if os.environ.get("TRITON_INTERPRET", "0") not in ("", "0"):
        pytest.skip("TRITON_INTERPRET is set: GPU kernels are not compiled")
    assert mg.triton_combine_ok(torch.device("cuda", 0))


@pytest.mark.gpu
@pytest.mark.parametrize("m,n,cols", [(96, 64, DEGREE), (37, 70, DEGREE), (4608, 3584, DEGREE), (64, 18944, 4 * DEGREE)])
@pytest.mark.parametrize("fmt", ["v1:56:16", "v1:64:0", "v2:53:35", "v2:56:56", "v2:45:32"])
def test_cuda_run_item_matches_previous_path(monkeypatch: pytest.MonkeyPatch, m: int, n: int, cols: int,
                                             fmt: str) -> None:  # pragma: no cover - GPU only
    from pgw.service import handler as handler_mod

    w = _weights(m, n, m + n, extreme=(n == 18944))
    kern = _gpu_kernel(w)
    ver, a, c = fmt.split(":")
    a_i, c_i = int(a), int(c)
    if ver == "v1":
        b = _vals(n * cols, a_i, m).reshape(n, cols)
        data = np.ascontiguousarray(b, dtype="<u8").view(np.uint8).reshape(n, cols, 8)[:, :, :a_i // 8].tobytes()
        it = MaskgenItem(kern.linear, cols, data, c_i, a_i)
    else:
        data = bytes(pack_bits(_vals(n * cols, a_i, m), a_i))
        it = MaskgenItem(kern.linear, cols, bytearray(data), b_bits=a_i, y_bits=c_i)
    new = mg.run_item(kern, it)
    monkeypatch.setenv(mg.ENV_FUSED, "0")
    old = bytes(handler_mod._maskgen_item(kern, it))
    assert new == old
    if m <= 96:  # and against the CPU reference
        if ver == "v1":
            ref = pack_y(matmul_mod64(w, unpack_b(data, (n, cols), a_i), fast=False), c_i, a_i)
        else:
            ref = bytes(pack_y_v2(matmul_mod64(w, unpack_bits(data, n * cols, a_i).reshape(n, cols), fast=False),
                                  a_i, c_i))
        assert new == ref


@pytest.mark.gpu
def test_cuda_run_item_yield_hook(monkeypatch: pytest.MonkeyPatch) -> None:  # pragma: no cover - GPU only
    monkeypatch.setenv(mg.ENV_YIELD_MS, "1")
    calls = 0

    def busy() -> bool:
        nonlocal calls
        calls += 1
        return True

    w = _weights(256, 512, 1)
    kern = _gpu_kernel(w)
    data = bytes(pack_bits(_vals(512 * DEGREE, 53, 2), 53))
    it = MaskgenItem(kern.linear, DEGREE, bytearray(data), b_bits=53, y_bits=35)
    assert mg.run_item(kern, it, decode_busy=busy) == mg.run_item(kern, it)
    assert calls > 0
