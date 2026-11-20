"""Provider maskgen fast path (WP5 / E4): byte-identical to the previous implementation.

The ``_old_*`` functions below are verbatim copies of the code this package replaced (wire helpers,
the service's ``unpack_b -> mulmod64 -> pack_y`` pipeline and the CUDA ``_mulmod64_cuda``). Every
test compares the new code with them bit for bit, on CPU here and on CUDA in the ``gpu`` tests.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import httpx
import numpy as np
import pytest
import torch

import pgw.service.kernels as kernels_mod
from pgw.core.types import LinearId, LinearSpec
from pgw.hemask import DEGREE
from pgw.hemask.fastmm import small_mm_big
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.wire import (
    MASKGEN_PATH,
    MaskgenItem,
    MaskgenRequest,
    decode_maskgen_response,
    encode_maskgen_request,
    pack_b,
    pack_y,
    unpack_b,
    unpack_y,
    y_round_offset,
    y_wire_bytes,
)
from pgw.service import ServiceSettings
from pgw.service.kernels import ComputeKernel, _combine_planes, _combine_planes_np, _maskgen_block_tn
from tests.service.conftest import TOKEN_A, running_app

U64_MAX = (1 << 64) - 1


# ---- previous implementation (reference) -------------------------------------------------------
def _old_to_bytes(v: np.ndarray, nb: int) -> bytes:
    a = np.ascontiguousarray(v, dtype="<u8")
    return np.ascontiguousarray(a.view(np.uint8).reshape(*a.shape, 8)[..., :nb]).tobytes()


def _old_from_bytes(data: bytes, shape: tuple[int, int], nb: int) -> np.ndarray:
    if nb == 8:
        return np.frombuffer(data, dtype="<u8").reshape(shape).astype(np.uint64)
    buf = np.zeros((*shape, 8), dtype=np.uint8)
    buf[..., :nb] = np.frombuffer(data, dtype=np.uint8).reshape(*shape, nb)
    return buf.view("<u8").reshape(shape).astype(np.uint64)


def _old_pack_b(b: np.ndarray, q_bits: int = 64) -> bytes:
    return _old_to_bytes(b, q_bits // 8)


def _old_unpack_b(data: bytes, shape: tuple[int, int], q_bits: int = 64) -> np.ndarray:
    return _old_from_bytes(data, shape, q_bits // 8)


def _old_pack_y(y: np.ndarray, drop_bits: int, q_bits: int = 64) -> bytes:
    yv = np.asarray(y, dtype=np.uint64)
    if q_bits < 64:
        yv = yv & np.uint64((1 << q_bits) - 1)
    if drop_bits:
        yv = (yv + np.uint64(1 << (drop_bits - 1))) >> np.uint64(drop_bits)
    return _old_to_bytes(yv, (q_bits - drop_bits) // 8)


def _old_unpack_y(data: bytes, shape: tuple[int, int], drop_bits: int, q_bits: int = 64) -> np.ndarray:
    v = _old_from_bytes(data, shape, (q_bits - drop_bits) // 8)
    return v << np.uint64(drop_bits) if drop_bits else v


def _old_mulmod64_cpu(w8: np.ndarray, b: np.ndarray, nbytes: int, f64: bool = False) -> np.ndarray:
    if f64:
        return matmul_mod64(w8, b, fast=False)
    return small_mm_big(w8, b, nbytes=nbytes)


def _old_provider(w8: np.ndarray, data: bytes, cols: int, q: int, drop: int, f64: bool = False) -> bytes:
    """The service's previous maskgen item: unpack_b, mulmod64, pack_y."""
    b = _old_unpack_b(data, (w8.shape[1], cols), q)
    return _old_pack_y(_old_mulmod64_cpu(w8, b, q // 8, f64), drop, q)


def _old_mulmod64_cuda(wpad: torch.Tensor, m: int, b: np.ndarray, nbytes: int = 8) -> np.ndarray:  # pragma: no cover
    n, cols = b.shape
    mp, np_ = wpad.shape
    colp = -(-cols // 32) * 32
    bt = torch.from_numpy(np.ascontiguousarray(b).view(np.int64)).to(wpad.device)
    rowsum = wpad.to(torch.int64).sum(1, keepdim=True)
    acc = torch.zeros((mp, colp), dtype=torch.int64, device=wpad.device)
    plane = torch.zeros((colp, np_), dtype=torch.int8, device=wpad.device)
    for d in range(nbytes):
        byte = (bt >> (8 * d)) & 0xFF
        plane[:cols, :n] = (byte - 128).to(torch.int8).t()
        part = torch._int_mm(wpad, plane.t()).to(torch.int64) + 128 * rowsum
        acc += part << (8 * d)
    return np.asarray(acc[:m, :cols].contiguous().cpu().numpy().view(np.uint64))


# ---- inputs --------------------------------------------------------------------------------------
def _weights(m: int, n: int, seed: int, kind: str = "random") -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "extreme":  # largest partial sums, -128 included
        return rng.choice(np.array([-128, -127, 127], dtype=np.int8), size=(m, n))
    w = rng.integers(-128, 128, size=(m, n)).astype(np.int8)
    w[0, :] = 127  # a row with the largest positive rowsum
    if m > 1:
        w[1, :] = -128
    return w


def _coeffs(n: int, cols: int, q: int, seed: int, kind: str = "random") -> np.ndarray:
    rng = np.random.default_rng(seed)
    top = (1 << q) - 1
    if kind == "ones":  # every byte 0xFF: largest planes, every wrap exercised
        return np.full((n, cols), top, dtype=np.uint64)
    b = rng.integers(0, 1 << 63, size=(n, cols), dtype=np.uint64) * np.uint64(2) \
        + rng.integers(0, 2, size=(n, cols), dtype=np.uint64)
    b &= np.uint64(top)
    b[0, : min(cols, 4)] = np.array([0, 1, top, top - 1], dtype=np.uint64)[: min(cols, 4)]
    return b


def _kernel(w8: np.ndarray, **kw: Any) -> ComputeKernel:
    m, n = w8.shape
    return ComputeKernel(LinearSpec(LinearId(0, "qkv"), m, n), torch.from_numpy(w8.copy()), device=kw.pop("device", "cpu"),
                         **kw)


@pytest.fixture
def int8_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the byte-plane int8 path deterministically (``cpu_int8_fast`` is a timing probe)."""
    if not kernels_mod.cpu_int8_available():
        pytest.skip("CPU torch._int_mm unavailable or inexact")
    monkeypatch.setenv("PGW_HEMASK_ENGINE", "cpu_int8")


# ---- wire helpers --------------------------------------------------------------------------------
@pytest.mark.parametrize("q,drop", [(64, 0), (64, 16), (56, 0), (56, 16), (64, 8), (48, 8), (60, 12), (61, 0)])
def test_wire_helpers_match_previous(q: int, drop: int) -> None:
    rng = np.random.default_rng(q * 100 + drop)
    y = rng.integers(0, 1 << 63, size=(37, 512), dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    edge = [0, 1, U64_MAX, U64_MAX - 1, (1 << q) - 1 if q < 64 else U64_MAX, (1 << 63), (1 << 15) - 1, 1 << 15,
            ((1 << q) - 1) & ~((1 << 15) - 1) if q < 64 else U64_MAX - (1 << 15) + 1]
    y[0, : len(edge)] = np.array(edge, dtype=np.uint64)  # rounding carries past bit q included
    assert pack_y(y, drop, q) == _old_pack_y(y, drop, q)
    assert pack_y(y.view(np.int64), drop, q) == _old_pack_y(y.view(np.int64), drop, q)  # signed input
    data = _old_pack_y(y, drop, q)
    if (q - drop) % 8 == 0:
        assert np.array_equal(unpack_y(data, y.shape, drop, q), _old_unpack_y(data, y.shape, drop, q))
    if q % 8 == 0:
        b = y & np.uint64((1 << q) - 1) if q < 64 else y
        assert pack_b(b, q) == _old_pack_b(b, q)
        pb = _old_pack_b(b, q)
        got = unpack_b(pb, b.shape, q)
        assert got.dtype == np.uint64 and got.flags.writeable and np.array_equal(got, _old_unpack_b(pb, b.shape, q))


@pytest.mark.parametrize("q,drop", [(64, 0), (64, 16), (56, 0), (56, 16)])
def test_y_wire_bytes_is_pack_y(q: int, drop: int) -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 1 << 63, size=(5, 256), dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    lo, hi = y_wire_bytes(q, drop)
    sl = (y + np.uint64(y_round_offset(drop))).view(np.uint8).reshape(5, 256, 8)[:, :, lo:hi].tobytes()
    assert sl == _old_pack_y(y, drop, q)
    with pytest.raises(ValueError):
        y_wire_bytes(60, 12)


# ---- kernel: wire to wire ------------------------------------------------------------------------
@pytest.mark.parametrize("m,n,cols", [(37, 70, DEGREE), (96, 64, 2 * DEGREE), (130, 260, DEGREE), (1, 8, DEGREE)])
@pytest.mark.parametrize("q,drop", [(56, 16), (64, 16), (56, 0), (64, 0)])
def test_mulmod64_wire_matches_previous(int8_engine: None, m: int, n: int, cols: int, q: int, drop: int) -> None:
    w8 = _weights(m, n, m + n)
    kern = _kernel(w8)
    assert kern._maskgen_planes_ok()
    for kind in ("random", "ones"):
        data = _old_pack_b(_coeffs(n, cols, q, m * n + q, kind), q)
        got = kern.mulmod64_wire(data, cols, q, drop)
        assert isinstance(got, bytes)
        assert got == _old_provider(w8, data, cols, q, drop)


@pytest.mark.parametrize("q,drop", [(56, 16), (64, 0)])
def test_mulmod64_wire_extreme_weights_and_blocks(int8_engine: None, monkeypatch: pytest.MonkeyPatch, q: int, drop: int) -> None:
    """Largest partial sums (|W| = 127/128, every byte 0xFF) and many column blocks (tiny budget)."""
    m, n, cols = 200, 1000, 3 * DEGREE
    w8 = _weights(m, n, 7, "extreme")
    kern = _kernel(w8)
    for kind in ("ones", "random"):
        data = _old_pack_b(_coeffs(n, cols, q, 3, kind), q)
        ref = _old_provider(w8, data, cols, q, drop)
        assert kern.mulmod64_wire(data, cols, q, drop) == ref
        monkeypatch.setattr(kernels_mod, "MASKGEN_PART_BYTES_CPU", m * (q // 8) * 4 * 100)  # 100-col blocks
        monkeypatch.setattr(kernels_mod, "_COMBINE_ROWS", 16)  # many row chunks and pool workers
        assert kern.mulmod64_wire(data, cols, q, drop) == ref
        monkeypatch.setattr(kernels_mod, "MASKGEN_PART_BYTES_CPU", 64 << 20)
        monkeypatch.setattr(kernels_mod, "_COMBINE_ROWS", 128)


@pytest.mark.parametrize("backend", ["cpu_f64", "cpu_f32"])
def test_mulmod64_wire_other_cpu_backends(int8_engine: None, backend: str) -> None:
    m, n, cols, q, drop = 50, 90, DEGREE, 56, 16
    w8 = _weights(m, n, 5)
    kern = _kernel(w8, backend=backend)
    data = _old_pack_b(_coeffs(n, cols, q, 9), q)
    assert kern.mulmod64_wire(data, cols, q, drop) == _old_provider(w8, data, cols, q, drop)


def test_mulmod64_wire_float64_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """PGW_HEMASK_ENGINE=cpu_f64 keeps the float64 bit-slice path (with the cached float weight)."""
    monkeypatch.setenv("PGW_HEMASK_ENGINE", "cpu_f64")
    m, n, cols = 40, 72, DEGREE
    w8 = _weights(m, n, 11)
    for backend in ("cpu_f64", "auto"):
        kern = _kernel(w8, backend=backend)
        assert not kern._maskgen_planes_ok()
        for q, drop in ((56, 16), (64, 0)):
            data = _old_pack_b(_coeffs(n, cols, q, 2), q)
            assert kern.mulmod64_wire(data, cols, q, drop) == _old_provider(w8, data, cols, q, drop, f64=True)
            b = _old_unpack_b(data, (n, cols), q)
            assert np.array_equal(kern.mulmod64(b, q // 8), _old_mulmod64_cpu(w8, b, q // 8, f64=True))


@pytest.mark.parametrize("nbytes", [8, 7, 3])
def test_mulmod64_numpy_api_matches_previous(int8_engine: None, nbytes: int) -> None:
    m, n, cols = 64, 96, DEGREE
    w8 = _weights(m, n, nbytes)
    kern = _kernel(w8)
    b = _coeffs(n, cols, 64, nbytes)
    if nbytes < 8:  # the caller promises b < 2^(8 nbytes); old and new both ignore the top bytes
        b &= np.uint64((1 << (8 * nbytes)) - 1)
    got = kern.mulmod64(b, nbytes)
    assert got.dtype == np.uint64 and got.shape == (m, cols)
    assert np.array_equal(got, _old_mulmod64_cpu(w8, b, nbytes))
    assert np.array_equal(got, matmul_mod64(w8.astype(np.int64), b, fast=False))
    assert kern.mulmod64(np.zeros((n, 0), dtype=np.uint64)).shape == (m, 0)


def test_rowsum_constant_is_cached(int8_engine: None) -> None:
    w8 = _weights(33, 40, 1)
    kern = _kernel(w8)
    data = _old_pack_b(_coeffs(40, DEGREE, 56, 1), 56)
    kern.mulmod64_wire(data, DEGREE, 56, 16)
    c = kern._mg_consts[(7, 1 << 15)]
    calls = 0
    orig = kern._maskgen_w_cpu

    def counting() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return orig()

    kern._maskgen_w_cpu = counting  # type: ignore[method-assign]
    assert kern.mulmod64_wire(data, DEGREE, 56, 16) == _old_provider(w8, data, DEGREE, 56, 16)
    assert kern._mg_consts[(7, 1 << 15)] is c
    assert calls == 1  # the GEMM operand only; no rowsum recomputation
    rs = w8.astype(np.int64).sum(1)
    want = (rs.astype(object) * 128 * sum(1 << (8 * d) for d in range(7)) + (1 << 15)) % (1 << 64)
    assert [int(v) for v in c.numpy().view(np.uint64).ravel()] == list(want)


def test_mulmod64_wire_rejects_bad_sizes() -> None:
    kern = _kernel(_weights(8, 16, 0))
    from pgw.core.errors import ShapeError

    with pytest.raises(ShapeError):
        kern.mulmod64_wire(bytes(7 * 16 * DEGREE - 1), DEGREE, 56, 16)
    with pytest.raises(ShapeError):
        kern.mulmod64_wire(bytes(7 * 16 * DEGREE), DEGREE, 56, 12)
    with pytest.raises(ShapeError):
        kern.mulmod64_wire(b"", 0, 56, 16)


def test_mulmod64_wire_threads_are_independent(int8_engine: None) -> None:
    """No shared mutable state between concurrent calls (the service runs them in parallel)."""
    m, n, cols = 120, 200, 2 * DEGREE
    w8 = _weights(m, n, 4)
    kern = _kernel(w8)
    datas = [_old_pack_b(_coeffs(n, cols, 56, s), 56) for s in range(4)]
    refs = [_old_provider(w8, d, cols, 56, 16) for d in datas]
    results: list[bytes | None] = [None] * 8

    def work(i: int) -> None:
        results[i] = kern.mulmod64_wire(datas[i % 4], cols, 56, 16)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert results == refs + refs


# ---- the CUDA path's block logic, run on CPU tensors -------------------------------------------------
@pytest.mark.parametrize("m,n,cols", [(37, 70, 40), (96, 64, DEGREE), (33, 300, 7)])
@pytest.mark.parametrize("q,drop", [(56, 16), (64, 0), (64, 16)])
def test_block_tn_matches_previous(int8_engine: None, m: int, n: int, cols: int, q: int, drop: int) -> None:
    """``_maskgen_block_tn`` (padded TN planes, torch combine, device byte slice) is device-agnostic."""
    w8 = _weights(m, n, m * cols, "extreme")
    mp, np_ = max(32, -(-m // 32) * 32), -(-n // 32) * 32
    wpad = torch.zeros((mp, np_), dtype=torch.int8)
    wpad[:m, :n] = torch.from_numpy(w8)
    kern = _kernel(w8)
    nb = q // 8
    lo, hi = y_wire_bytes(q, drop)
    const = kern._maskgen_const(nb, y_round_offset(drop))
    for kind in ("ones", "random"):
        data = _old_pack_b(_coeffs(n, cols, q, cols, kind), q)
        raw = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).reshape(n, cols, nb).copy())
        got = _maskgen_block_tn(wpad, raw, m, n, const, lo, hi)
        assert got.shape == (m, cols, hi - lo)
        assert got.contiguous().numpy().tobytes() == _old_provider(w8, data, cols, q, drop)


def test_combine_torch_and_numpy_agree() -> None:
    rng = np.random.default_rng(3)
    for npl in (1, 3, 4, 5, 7, 8):
        planes = [rng.integers(-(1 << 31), 1 << 31, size=(17, 33), dtype=np.int64).astype(np.int32) for _ in range(npl)]
        planes[0][0, :] = np.iinfo(np.int32).min
        planes[-1][1, :] = np.iinfo(np.int32).max
        const = rng.integers(0, 1 << 63, size=(17, 1), dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        want = np.zeros((17, 33), dtype=object)
        for d, p in enumerate(planes):
            want += p.astype(object) * (1 << (8 * d))
        want = (want + const.astype(object)) % (1 << 64)
        a = _combine_planes_np(planes, const)
        b = _combine_planes([torch.from_numpy(p) for p in planes], torch.from_numpy(const.view(np.int64)))
        assert [int(v) for v in a.ravel()] == list(want.ravel())
        assert np.array_equal(b.numpy().view(np.uint64), a)


# ---- service: bounded concurrency instead of one global lock ----------------------------------------
def _settings(pack_dir: Any, tokens_file: Any) -> ServiceSettings:
    return ServiceSettings(packs=[pack_dir], device="cpu", rings=["K40A12"], tokens_file=tokens_file)


@pytest.mark.parametrize("limit", [1, 2])
async def test_maskgen_runs_concurrently_up_to_the_limit(int8_engine: None, monkeypatch: pytest.MonkeyPatch, tiny_pack_dir: Any,
                                                         tokens_file: Any, pack_fp: str,
                                                         pack_weights: dict[str, torch.Tensor], limit: int) -> None:
    monkeypatch.setenv("PGW_SERVICE_MASKGEN_CONCURRENCY", str(limit))
    lid, w8 = next(iter(pack_weights.items()))
    m, n = (int(v) for v in w8.shape)
    active = peak = 0
    lock = threading.Lock()
    orig = ComputeKernel.mulmod64_wire

    def slow(self: ComputeKernel, *a: Any, **kw: Any) -> bytes:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.15)
            return orig(self, *a, **kw)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(ComputeKernel, "mulmod64_wire", slow)
    datas = [_old_pack_b(_coeffs(n, DEGREE, 56, s), 56) for s in range(3)]
    reqs = [MaskgenRequest(f"r{i}", pack_fp, (MaskgenItem(lid, DEGREE, d, 16, 56),)) for i, d in enumerate(datas)]
    hdr = {"content-type": "application/vnd.pgw+msgpack", "authorization": f"Bearer {TOKEN_A}"}
    async with running_app(_settings(tiny_pack_dir, tokens_file)) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            rs = await asyncio.gather(*(h.post(MASKGEN_PATH, content=encode_maskgen_request(r), headers=hdr)
                                        for r in reqs))
    assert [r.status_code for r in rs] == [200, 200, 200]
    assert peak == limit
    w = w8.numpy()
    for r, req, d in zip(rs, reqs, datas, strict=True):
        resp = decode_maskgen_response(r.content, req, {lid: m})
        assert resp.items[0].y == _old_provider(w, d, DEGREE, 56, 16)


# ---- GPU ---------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("m,n,cols", [(96, 64, DEGREE), (37, 70, DEGREE), (4608, 3584, DEGREE), (64, 18944, 4 * DEGREE)])
@pytest.mark.parametrize("q,drop", [(56, 16), (64, 0), (64, 16)])
def test_cuda_mulmod64_wire_matches_previous(m: int, n: int, cols: int, q: int, drop: int) -> None:  # pragma: no cover
    w8 = _weights(m, n, m + n, "extreme" if n == 18944 else "random")
    kern = _kernel(w8, device="cuda")
    for kind in ("random", "ones"):
        b = _coeffs(n, cols, q, m, kind)
        data = _old_pack_b(b, q)
        got = kern.mulmod64_wire(data, cols, q, drop)
        old_gpu = _old_pack_y(_old_mulmod64_cuda(kern._pw._w_dev, m, b, q // 8), drop, q)
        assert got == old_gpu
        assert got == _old_pack_y(matmul_mod64(w8.astype(np.int64), b), drop, q)
        assert np.array_equal(kern.mulmod64(b, q // 8), _old_mulmod64_cuda(kern._pw._w_dev, m, b, q // 8))


@pytest.mark.gpu
def test_cuda_mulmod64_wire_blocks_and_streams(monkeypatch: pytest.MonkeyPatch) -> None:  # pragma: no cover
    m, n, cols, q, drop = 300, 1000, 3 * DEGREE, 56, 16
    w8 = _weights(m, n, 2, "extreme")
    kern = _kernel(w8, device="cuda")
    datas = [_old_pack_b(_coeffs(n, cols, q, s), q) for s in range(4)]
    refs = [_old_provider(w8, d, cols, q, drop) for d in datas]
    monkeypatch.setattr(kernels_mod, "MASKGEN_PART_BYTES_CUDA", 320 * 7 * 4 * 64)  # 64-column blocks
    results: list[bytes | None] = [None] * 8

    def work(i: int) -> None:
        results[i] = kern.mulmod64_wire(datas[i % 4], cols, q, drop)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert results == refs + refs
