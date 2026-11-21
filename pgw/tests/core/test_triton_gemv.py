"""Exactness of the fused Triton ring GEMV (:mod:`pgw.core.triton_gemv`).

CPU tests run the kernel in the Triton interpreter (``TRITON_INTERPRET=1``, set by ``tests/conftest.py``
without CUDA) on small
shapes; ``gpu`` tests compile it for the device and compare with the digit-GEMM path at full 7B
shapes (run on Kaggle). Every comparison is bit for bit.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from pgw.core import LinearId, LinearKind, LinearSpec, RingParams, ShapeError
from pgw.core.ringmm import cpu_int8_available, int8_ring_matmul, pad_weight_cuda, ringmm_reference
from pgw.core.triton_gemv import (
    TRITON_GEMV_KS,
    TRITON_GEMV_MAX_ROWS,
    GemvConfig,
    check_against_reference,
    gemv_config,
    low32,
    max_rows,
    ring_gemv,
    self_check,
    triton_available,
)

needs_triton = pytest.mark.skipif(not triton_available(), reason="triton not installed")
RINGS = {16: RingParams(16, 4), 24: RingParams(24, 12), 32: RingParams.from_name("K32A12C27")}


@pytest.fixture
def interp() -> None:
    """Triton interpreter on CPU tensors: ``tests/conftest.py`` enables it before triton is imported on
    hosts without CUDA (the mode cannot be switched later). GPU hosts run the ``gpu`` tests instead."""
    if os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
        pytest.skip("Triton interpreter not enabled (CUDA host: see the gpu tests)")


@pytest.fixture
def compiled() -> None:
    """Compiled Triton kernels for the GPU."""
    if os.environ.get("TRITON_INTERPRET", "0") not in ("", "0"):
        pytest.skip("TRITON_INTERPRET is set: GPU kernels are not compiled")


def _gemv64(x: torch.Tensor, w: torch.Tensor, k: int, **kw: object) -> np.ndarray:
    """Fused kernel result as int64 ring elements (via the compact contract: low 32 bits)."""
    y = ring_gemv(low32(x).to(w.device), w, k, **kw)  # type: ignore[arg-type]
    assert y.dtype == torch.int32
    out: np.ndarray = (y.cpu().to(torch.int64) & ((1 << k) - 1)).numpy()
    return out


def _rand(b: int, m: int, n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 1 << k, (b, n), dtype=torch.int64, generator=g)
    w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    return x, w


# ---- host-side helpers (no triton needed) ----------------------------------------------------
def test_low32_matches_mod_2_32() -> None:
    vals = [0, 1, -1, (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32, (1 << 40) + 5, -(1 << 63), (1 << 63) - 1]
    x = torch.tensor([vals, vals[::-1]], dtype=torch.int64)
    for t in (x, x.t().contiguous().t()):  # contiguous (view) and strided (arithmetic) paths
        lo = low32(t)
        assert lo.dtype == torch.int32 and tuple(lo.shape) == (2, len(vals))
        got = (lo.to(torch.int64) & 0xFFFFFFFF).tolist()
        assert got == [[v % (1 << 32) for v in row] for row in t.tolist()]
    with pytest.raises(ShapeError):
        low32(torch.zeros(3, dtype=torch.int32))


@pytest.mark.parametrize("rows", list(range(1, TRITON_GEMV_MAX_ROWS + 1)))
@pytest.mark.parametrize(("m", "n"), [(4608, 3584), (37888, 3584), (3584, 18944), (3, 5)])
def test_gemv_config_shapes(rows: int, m: int, n: int) -> None:
    cfg = gemv_config(rows, m, n)
    assert cfg.block_r >= rows and cfg.block_r & (cfg.block_r - 1) == 0
    assert 1 <= cfg.split_k <= min(8, max(1, n // 512))
    if cfg.dot:
        assert cfg.block_r >= 16 and cfg.block_m >= 16 and cfg.block_n >= 32


def test_ring_gemv_rejects_bad_arguments() -> None:
    x = torch.zeros((2, 8), dtype=torch.int32)
    w = torch.zeros((4, 8), dtype=torch.int8)
    with pytest.raises(ShapeError, match="K <= 32"):
        ring_gemv(x, w, 40)
    with pytest.raises(ShapeError):
        ring_gemv(x.to(torch.int64), w, 32)
    with pytest.raises(ShapeError):
        ring_gemv(x, w.to(torch.int32), 32)
    with pytest.raises(ShapeError, match="exceed"):
        ring_gemv(x, w, 32, m=5)
    with pytest.raises(ShapeError, match="block_r"):
        ring_gemv(x, w, 32, config=GemvConfig(1, 4, 64))
    with pytest.raises(ShapeError, match="tensor-core"):
        ring_gemv(x, w, 32, config=GemvConfig(8, 16, 64, dot=True))
    with pytest.raises(ShapeError, match="bad out"):
        ring_gemv(x, w, 32, out=torch.zeros((2, 4), dtype=torch.int64))


# ---- interpreter (CPU) exactness -----------------------------------------------------------
@needs_triton
@pytest.mark.usefixtures("interp")
@pytest.mark.parametrize("k", TRITON_GEMV_KS)
@pytest.mark.parametrize(("m", "n"), [(1, 1), (5, 7), (37, 129), (33, 700)])
def test_random_exact_all_rows(k: int, m: int, n: int) -> None:
    for b in range(1, TRITON_GEMV_MAX_ROWS + 1):
        x, w = _rand(b, m, n, k, seed=1000 * k + 10 * b + m)
        want = ringmm_reference(x, w, k)
        np.testing.assert_array_equal(_gemv64(x, w, k), want, err_msg=f"b={b}")
        if cpu_int8_available() and b in (1, 7, 16):
            np.testing.assert_array_equal(int8_ring_matmul(x, w, k).numpy(), want)


@needs_triton
@pytest.mark.usefixtures("interp")
@pytest.mark.parametrize("k", TRITON_GEMV_KS)
@pytest.mark.parametrize("wv", [-128, 127])
def test_extreme_operands_wrap_exactly(k: int, wv: int) -> None:
    """All-extreme operands: every partial sum wraps mod 2^32 many times; the result is still exact."""
    top = (1 << k) - 1
    n, m = 2049, 9  # odd n: a partial last tile
    for xv in (0, 1, top, 1 << (k - 1), (1 << (k - 1)) - 1):
        x = torch.full((TRITON_GEMV_MAX_ROWS, n), xv, dtype=torch.int64)
        w = torch.full((m, n), wv, dtype=torch.int8)
        want = ringmm_reference(x, w, k)
        for cfg in (None, GemvConfig(16, 4, 128, 2), GemvConfig(16, 16, 64, 2, dot=True)):  # both kernels
            np.testing.assert_array_equal(_gemv64(x, w, k, config=cfg), want, err_msg=f"x={xv} {cfg}")
        np.testing.assert_array_equal(_gemv64(x[:1], w, k), want[:1])  # one row: CUDA-core kernel
    # mixed signs per row and column: rows at the bounds, alternating weight columns
    x = torch.tensor([[top, 0, 1 << (k - 1), (1 << (k - 1)) - 1][i % 4] for i in range(n)], dtype=torch.int64)
    x = torch.stack([x, x.roll(1), x.flip(0)])
    w = torch.where(torch.arange(n) % 3 == 0, -128, 127).to(torch.int8).repeat(m, 1)
    w[::2] = wv
    np.testing.assert_array_equal(_gemv64(x, w, k), ringmm_reference(x, w, k))
    np.testing.assert_array_equal(_gemv64(x, w, k, config=GemvConfig(4, 8, 256)), ringmm_reference(x, w, k))


@needs_triton
@pytest.mark.usefixtures("interp")
@pytest.mark.parametrize("k", TRITON_GEMV_KS)
@pytest.mark.parametrize(
    "cfg",
    [GemvConfig(1, 4, 64, 1), GemvConfig(2, 8, 128, 3), GemvConfig(4, 16, 64, 2), GemvConfig(8, 16, 64, 5),
     GemvConfig(16, 4, 32, 7), GemvConfig(16, 16, 32, 1, dot=True), GemvConfig(16, 32, 64, 3, dot=True),
     GemvConfig(32, 16, 128, 2, dot=True)],
)
def test_split_k_and_tiles_any_order_is_exact(k: int, cfg: GemvConfig) -> None:
    """Split-K (wrapping atomics) and every tile shape give the same bits; K < 32 is masked after."""
    b = min(cfg.block_r, 5)
    x, w = _rand(b, 41, 1001, k, seed=cfg.block_m + cfg.split_k)
    w[0] = -128
    x[0] = (1 << k) - 1
    np.testing.assert_array_equal(_gemv64(x, w, k, config=cfg), ringmm_reference(x, w, k))


@needs_triton
@pytest.mark.usefixtures("interp")
def test_padded_weight_and_any_int64_input() -> None:
    """The service passes the 32-padded weight; columns past n never matter; x is taken mod 2^K."""
    k = 24
    x, w8 = _rand(3, 45, 70, k, seed=5)
    wpad = pad_weight_cuda(w8, "cpu")
    wpad[:, 70:] = 77  # garbage in the padding columns must not leak into the result
    wpad[45:] = -3
    xw = x + (torch.randint(-(1 << 20), 1 << 20, x.shape, dtype=torch.int64) << 40)  # high bits: ignored
    xw[1] -= 1 << 62
    got = _gemv64(xw, wpad, k, n=70, m=45)
    np.testing.assert_array_equal(got, ringmm_reference(x, w8, k))
    out = torch.full((3, 45), 123, dtype=torch.int32)
    y = ring_gemv(low32(xw), wpad, k, n=70, m=45, out=out)
    assert y is out
    np.testing.assert_array_equal(y.to(torch.int64).numpy(), ringmm_reference(x, w8, k))


@needs_triton
@pytest.mark.usefixtures("interp")
def test_self_check_cases_pass_in_interpreter() -> None:
    assert check_against_reference("cpu", quick=True) == []


@needs_triton
@pytest.mark.usefixtures("interp")
@pytest.mark.parametrize("k", TRITON_GEMV_KS)
def test_executor_run_gemv_compact_contract(k: int) -> None:
    """DeviceExecutor.run_gemv (CPU executor): staged low words in, compact int32 out."""
    from pgw.service.gpu_exec import DeviceExecutor, from_compact

    ex = DeviceExecutor("cpu")
    x, w8 = _rand(6, 50, 99, k, seed=k)
    y = ex.run_gemv(x, k, pad_weight_cuda(w8, "cpu"), 99, 50)
    assert y.dtype == torch.int32 and tuple(y.shape) == (6, 50) and ex.stats.gemv == 1
    np.testing.assert_array_equal(from_compact(y, k).numpy(), ringmm_reference(x, w8, k))
    assert ex.gemv_eligible(TRITON_GEMV_MAX_ROWS + 1, k) is False
    assert ex.gemv_eligible(1, 40) is False


def test_settings_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.service import ServiceSettings

    assert ServiceSettings().triton_gemv is True
    monkeypatch.setenv("PGW_SERVICE_TRITON_GEMV", "false")
    assert ServiceSettings().triton_gemv is False


def test_cpu_kernels_never_use_triton() -> None:
    from pgw.service.kernels import ComputeKernel

    w = torch.randint(-128, 128, (8, 16), dtype=torch.int8)
    k = ComputeKernel(LinearSpec(LinearId(0, LinearKind.O), 8, 16), w, device="cpu", backend="cpu_f64")
    assert k.triton_gemv is False


# ---- GPU (compiled) ---------------------------------------------------------------------------
SHAPES_7B = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}


def _cuda_ref(x: torch.Tensor, w8: torch.Tensor, k: int) -> np.ndarray:
    from pgw.core.ringmm import PreparedWeight

    out: np.ndarray = PreparedWeight(w8, device="cuda").matmul(x.cuda(), k).cpu().numpy()
    return out


@pytest.mark.gpu
@pytest.mark.usefixtures("compiled")
def test_gpu_self_check() -> None:  # pragma: no cover - GPU only
    assert triton_available()
    assert self_check("cuda:0")  # also decides whether the tensor-core variant is usable (max_rows)
    assert check_against_reference("cuda:0") == []


@pytest.mark.gpu
@pytest.mark.usefixtures("compiled")
@pytest.mark.parametrize("name", list(SHAPES_7B))
def test_gpu_7b_shapes_all_rows(name: str) -> None:  # pragma: no cover - GPU only
    m, n = SHAPES_7B[name]
    g = torch.Generator().manual_seed(len(name))
    w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    wpad = pad_weight_cuda(w8, "cuda")
    assert self_check("cuda:0")  # without the tensor-core variant, several rows use the CUDA-core kernel here
    for k in TRITON_GEMV_KS:
        for b in range(1, TRITON_GEMV_MAX_ROWS + 1):
            x = torch.randint(0, 1 << k, (b, n), dtype=torch.int64, generator=g)
            np.testing.assert_array_equal(_gemv64(x, wpad, k, n=n, m=m), _cuda_ref(x, w8, k), err_msg=f"{k} {b}")
    # Python-int reference on a slice (independent of the digit-GEMM path)
    x = torch.randint(0, 1 << 32, (5, n), dtype=torch.int64, generator=g)
    np.testing.assert_array_equal(_gemv64(x, wpad, 32, n=n, m=m)[:, :64], ringmm_reference(x, w8[:64], 32))


@pytest.mark.gpu
@pytest.mark.usefixtures("compiled")
@pytest.mark.parametrize("k", TRITON_GEMV_KS)
def test_gpu_extreme_and_odd_shapes(k: int) -> None:  # pragma: no cover - GPU only
    top = (1 << k) - 1
    assert self_check("cuda:0")
    dot_ok = max_rows("cuda:0") > 1  # int8 tl.dot does not compile on every GPU/Triton (T4 + Triton 3.6)
    for m, n in [(1, 1), (7, 13), (33, 3585), (3584, 18943), (130, 18944)]:
        for wv in (-128, 127):
            w8 = torch.full((m, n), wv, dtype=torch.int8)
            for xv in (top, 1 << (k - 1), (1 << (k - 1)) - 1):
                x = torch.full((TRITON_GEMV_MAX_ROWS, n), xv, dtype=torch.int64)
                want = ringmm_reference(x[:2], w8[:3], k)
                got = _gemv64(x, pad_weight_cuda(w8, "cuda"), k, n=n, m=m)
                np.testing.assert_array_equal(got[:2, :3], want)
                assert (got == got[0, 0]).all()  # every entry is the same product
        for b in (1, 3, 8, 16):
            x, w8 = _rand(b, m, n, k, seed=m + n + b)
            np.testing.assert_array_equal(_gemv64(x, w8.cuda(), k), _cuda_ref(x, w8, k))
            for cfg in (GemvConfig(16, 4, 64, 1), GemvConfig(16, 8, 64, 4), GemvConfig(16, 32, 512, 3),
                        *((GemvConfig(16, 16, 32, 1, dot=True), GemvConfig(16, 64, 128, 3, dot=True),
                           GemvConfig(16, 128, 256, 2, 8, dot=True)) if dot_ok else ())):
                np.testing.assert_array_equal(_gemv64(x, w8.cuda(), k, config=cfg), _cuda_ref(x, w8, k))


@pytest.mark.gpu
@pytest.mark.usefixtures("compiled")
def test_gpu_compute_kernel_uses_gemv_and_falls_back() -> None:  # pragma: no cover - GPU only
    from pgw.service.gpu_exec import device_executor
    from pgw.service.kernels import ComputeKernel

    m, n = 300, 1000
    w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=torch.Generator().manual_seed(9))
    spec = LinearSpec(LinearId(0, LinearKind.O), m, n)
    kern = ComputeKernel(spec, w8, device="cuda", cuda_graphs=False)
    off = ComputeKernel(spec, w8, device="cuda", cuda_graphs=False, triton_gemv=False)
    assert kern.triton_gemv and not off.triton_gemv
    ex = device_executor("cuda:0")
    for k, ring in RINGS.items():
        for b in (1, 2, 5, 8, 16, 17):
            x, _ = _rand(b, 1, n, k, seed=b)
            gemv0 = ex.stats.gemv
            y = kern.run(x, ring)
            np.testing.assert_array_equal(y.numpy(), ringmm_reference(x, w8, k))
            assert torch.equal(kern.run_compact(x, ring), off.run_compact(x, ring))
            assert ex.stats.gemv - gemv0 == (2 if b <= max_rows("cuda:0") else 0)
    x, _ = _rand(3, 1, n, 40, seed=0)
    gemv0 = ex.stats.gemv
    np.testing.assert_array_equal(kern.run(x, RingParams(40, 12)).numpy(), ringmm_reference(x, w8, 40))
    assert ex.stats.gemv == gemv0  # K > 32: digit-GEMM path
    # a wrong fused result on the first call of a (K, rows) disables the fast path on the device
    import pgw.service.gpu_exec as gx

    orig = gx.ring_gemv
    try:
        gx.ring_gemv = lambda *a, **kw: orig(*a, **kw).add_(1)  # type: ignore[assignment]
        x, _ = _rand(1, 1, n, 32, seed=77)  # one row: eligible on every device
        kern2 = ComputeKernel(spec, w8, device="cuda", cuda_graphs=False)
        np.testing.assert_array_equal(kern2.run(x, RINGS[32]).numpy(), ringmm_reference(x, w8, 32))
        assert not ex.gemv_ready()
    finally:
        gx.ring_gemv = orig
        ex._gemv_ok = True


def test_self_check_falls_back_to_one_row_cuda_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tensor-core failure keeps the CUDA-core kernel for one-row calls; a second failure disables both."""
    import pgw.core.triton_gemv as tg

    dev = "meta"  # a key no real call uses
    monkeypatch.setattr(tg, "triton_available", lambda: True)
    for results, ok, rows in (([(False, "dot"), (True, "")], True, 1), ([(False, "a"), (False, "b")], False, 1),
                              ([(True, "")], True, TRITON_GEMV_MAX_ROWS)):
        calls = iter(results)
        monkeypatch.setattr(tg, "_run_check", lambda key, device, calls=calls: next(calls))
        tg._SELF_CHECK.pop(dev, None)
        tg._NO_DOT.discard(dev)
        try:
            assert tg.self_check(dev) is ok
            assert tg.max_rows(dev) == rows
            if rows == 1:
                assert not tg.gemv_config(4, 64, 1024, torch.device(dev)).dot
        finally:
            tg._SELF_CHECK.pop(dev, None)
            tg._NO_DOT.discard(dev)


@pytest.mark.parametrize(("cc", "dot"), [((7, 5), False), ((7, 0), False), ((8, 0), True), ((9, 0), True)])
def test_self_check_skips_tensor_core_variant_before_sm80(monkeypatch: pytest.MonkeyPatch, cc: tuple[int, int],
                                                           dot: bool) -> None:
    """Before sm_80 (T4 = sm_75) Triton has no int8 tensor-core dot: the variant is never compiled there (the
    choice is made from the capability, deterministically), and the CUDA-core kernels are checked once."""
    import pgw.core.triton_gemv as tg

    dev = "cuda:7"  # a key no real call uses on this host
    calls: list[str] = []
    monkeypatch.setattr(tg, "triton_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: cc)
    monkeypatch.setattr(tg, "_run_check", lambda key, device: (calls.append(key), (True, ""))[1])
    monkeypatch.setattr(tg, "_checked_rules", lambda key, device: ())
    monkeypatch.setattr(tg, "_enable_multirow", lambda key, device: 0)
    tg._reset(dev)
    try:
        assert tg.self_check(dev)
        assert calls == [dev]  # one check, no failing tensor-core attempt first
        assert (dev not in tg._NO_DOT) is dot
        assert tg.max_rows(dev) == (TRITON_GEMV_MAX_ROWS if dot else 1)
    finally:
        tg._reset(dev)
    assert tg._dot_unsupported("cpu") is None
