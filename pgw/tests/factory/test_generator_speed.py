"""Factory speed path (WP1): native r expansion, byte-gather digit split, in-place combine, output lag,
chunk sizing and per-stage stats. Everything here must leave the produced masks bit-identical."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A12_C27, RING_K40_A12, IntegrityError, RingParams, SeedExpander, native, ringmm_reference
from pgw.core.ring import balanced_digits
from pgw.core.ringmm import _combine_digit_rows
from pgw.factory import FactorySettings, MaskGenerator
from pgw.factory import generator as fg
from pgw.factory.bench import benchmark_preset, format_report
from pgw.factory.kernels import RingKernel, combine_digit_planes, cpu_int8_available, ring_digit_planes

from .conftest import FakeMaskStore, FakePack, reference_segment

SEED = bytes(range(32))
needs_int8 = pytest.mark.skipif(not cpu_int8_available(), reason="CPU torch._int_mm not exact here")
needs_native = pytest.mark.skipif(not native.native_available("mask_rows_with_key"),
                                  reason="pgw_fast extension not built")

#: sha256 of every (linear key, W r chunk, uW) the generator yields for FakePack(layers=2, seed=3), seed
#: bytes(range(32)), 37 slots, chunk 8, checks 2 -- computed with the factory as of a3cc580 (Python
#: ChaCha20, balanced_digits, int8_ring_matmul), before this speed work.
GOLDEN = {
    "K32A12C27": "2931e887f2d3d168dca5078f11f21dedc8fb03453674f514fc93cb9ead3affdd",
    "K40A12": "a76c935b9507529ce7362c4bb9c5b27c156e9d8fdd8c4be815d6e0e7e867686c",
    "K48A20": "502dab80bfe4f6dc7949a3f8d5f31db630eb47627f6a5f6668dbaa00fa6d107e",
}


def _digest(gen: MaskGenerator, slots: int = 37, chunk: int | None = 8) -> str:
    h = hashlib.sha256()
    for item in gen.generate(SEED, slots, checks=2, chunk_slots=chunk):
        h.update(item.id.key.encode())
        h.update(item.wr.numpy().tobytes())
        if item.uw is not None:
            h.update(item.uw.numpy().tobytes())
    return h.hexdigest()


# ---- digit split / combine ----------------------------------------------------------------------------
@pytest.mark.parametrize("k", [8, 16, 24, 32, 40, 48, 56])
def test_ring_digit_planes_equal_balanced_digits(k: int) -> None:
    g = torch.Generator().manual_seed(k)
    b, n = 7, 13
    r = torch.randint(0, 1 << min(k, 62), (b, n), dtype=torch.int64, generator=g) & ((1 << k) - 1)
    edge = [0, 1, 127, 128, 255, 256, (1 << k) - 1, 1 << (k - 1), (1 << (k - 1)) - 1, 0x80 * sum(256 ** i for i in range(k // 8)) % (1 << k)]
    r[0, : len(edge)] = torch.tensor(edge, dtype=torch.int64)
    n_l = k // 8
    want = balanced_digits(r, k, 8)
    # padded operand: P > B rows and NP > n columns; the padding must stay untouched
    p, np_ = 9, 16
    out = torch.full((n_l * p, np_), 0x5A, dtype=torch.int8)
    ring_digit_planes(r, k, out, p)
    exp = torch.full((n_l, p, np_), 0x5A, dtype=torch.int8)
    for i, d in enumerate(want):
        exp[i, :b, :n] = d.to(torch.int8)
    assert torch.equal(out.view(n_l, p, np_), exp)
    # unpadded, from a non-contiguous (transposed) r
    rt = r.t().contiguous().t()
    out2 = torch.empty((n_l * b, n), dtype=torch.int8)
    assert torch.equal(ring_digit_planes(rt, k, out2), torch.cat(want).to(torch.int8))


def test_ring_digit_planes_shape_checks() -> None:
    r = torch.zeros((4, 5), dtype=torch.int64)
    with pytest.raises(Exception, match="digit planes"):
        ring_digit_planes(r, 32, torch.zeros((4 * 3, 5), dtype=torch.int8))  # P < B
    with pytest.raises(Exception, match="digit planes"):
        ring_digit_planes(r, 32, torch.zeros((16, 4), dtype=torch.int8))  # NP < n
    with pytest.raises(Exception, match="int64"):
        ring_digit_planes(r.to(torch.int32), 32, torch.zeros((16, 5), dtype=torch.int8))


@pytest.mark.parametrize(("k", "inner"), [(32, 20), (32, 65536), (40, 100), (40, 1 << 16), (48, 20), (48, 4096),
                                                (56, 3584)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_combine_digit_planes_equals_core(k: int, inner: int, dtype: torch.dtype) -> None:
    """Both combine paths (single masked sum; reduce-then-shift) give the core's result, at extreme |y_i|."""
    g = torch.Generator().manual_seed(k + inner)
    n_l, p, mp, rows, m = k // 8, 6, 11, 5, 9
    lim = min(128 * 128 * inner, (1 << 31) - 1)
    y = torch.randint(-lim, lim + 1, (n_l * p, mp), dtype=torch.int64, generator=g)
    y[0] = lim
    y[1] = -lim
    y[-1] = -lim
    y = y.to(dtype)
    want = _combine_digit_rows(y.to(torch.int64), p, n_l, 8, k)[:rows, :m]
    got = combine_digit_planes(y, rows, p, n_l, k, m, inner)
    assert got.is_contiguous() and got.dtype == torch.int64
    assert torch.equal(got, want)
    assert int(got.min()) >= 0 and int(got.max()) < (1 << k)


def test_combine_digit_planes_row_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU row blocking (cache-sized blocks) does not change the result."""
    g = torch.Generator().manual_seed(1)
    y = torch.randint(-(1 << 30), 1 << 30, (4 * 50, 33), dtype=torch.int32, generator=g)
    whole = combine_digit_planes(y, 50, 50, 4, 32, 33, 4096)
    monkeypatch.setattr("pgw.factory.kernels._COMBINE_BLOCK_BYTES", 8 * 33 * 3)  # 16-row blocks (minimum)
    assert torch.equal(combine_digit_planes(y, 50, 50, 4, 32, 33, 4096), whole)


@needs_int8
@pytest.mark.parametrize("k", [16, 32, 40, 48, 56])
def test_ring_kernel_matches_reference_wide(k: int) -> None:
    g = torch.Generator().manual_seed(k)
    ring = RingParams(k, 8)
    w = torch.randint(-128, 128, (37, 300), dtype=torch.int8, generator=g)
    w[0] = -128
    w[1] = 127
    r = torch.randint(0, 1 << min(k, 62), (19, 300), dtype=torch.int64, generator=g) & ring.mask
    r[0] = 1 << (k - 1)  # every digit -128
    r[1] = ring.mask
    kern = RingKernel(w, "cpu_int8")
    got = kern.matmul(r, ring)
    assert got.tolist() == ringmm_reference(r, w, k).tolist()
    # the input is not modified (only the temporary offset copy and the digit planes are zeroed)
    assert r[1, 0].item() == ring.mask
    # non-canonical int64 (negative, above 2^K) = the same ring elements
    noisy = r - (1 << k) * torch.randint(-3, 4, r.shape, generator=g)
    assert torch.equal(kern.matmul(noisy, ring), got)


# ---- native r ------------------------------------------------------------------------------------------
@needs_native
@pytest.mark.parametrize(("cols", "k", "start", "rows"), [(20, 32, 0, 5), (26, 40, 13, 9), (3584, 32, 4095, 3),
                                                          (896, 48, 1, 64), (7, 56, (1 << 20), 2)])
def test_native_rows_bit_identical(cols: int, k: int, start: int, rows: int) -> None:
    exp = SeedExpander(SEED)
    want = exp.ring_uniform_rows(cols, k, start, rows, "r", "0.qkv")
    nat = fg._CpuRing(exp, True)
    py = fg._CpuRing(exp, False)
    try:
        assert torch.equal(nat.rows("0.qkv", cols, k, start, rows), want)
        assert torch.equal(py.rows("0.qkv", cols, k, start, rows), want)
    finally:
        nat.wipe()
        py.wipe()


def test_cpu_ring_wipes_finished_linear_keys() -> None:
    ring = fg._CpuRing(SeedExpander(SEED), False)
    ring.rows("0.qkv", 4, 32, 0, 1)
    k0 = ring._keys["0.qkv"]
    ring.rows("0.o", 4, 32, 0, 1)
    assert k0 == bytearray(32) and list(ring._keys) == ["0.o"]
    k1 = ring._keys["0.o"]
    ring.wipe()
    assert k1 == bytearray(32) and not ring._keys


def test_native_rng_requested_without_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fg, "_NATIVE_RNG", False)
    with pytest.raises(Exception, match="native_rng"):
        MaskGenerator(FakePack(layers=1), RING_K40_A12, native_rng=True)
    assert MaskGenerator(FakePack(layers=1), RING_K40_A12).native_rng is False


# ---- generator output: golden, native vs Python, lag, chunking -------------------------------------------
@needs_int8
@pytest.mark.parametrize("ring", sorted(GOLDEN))
@pytest.mark.parametrize("native_rng", [None, False])
@pytest.mark.parametrize("pipeline", [False, True])
def test_generator_output_matches_golden(ring: str, native_rng: bool | None, pipeline: bool) -> None:
    gen = MaskGenerator(FakePack(layers=2, seed=3), ring, backend="cpu_int8", pipeline=pipeline,
                        self_check_rows=-1, native_rng=native_rng)
    assert _digest(gen) == GOLDEN[ring]
    assert gen.last_stats.extra["native_rng"] is (fg._NATIVE_RNG if native_rng is None else False)


@pytest.mark.parametrize("backend", ["auto", "cpu_f64"])
@pytest.mark.parametrize("lag", [1, 2])
def test_output_lag_keeps_order_and_bytes(backend: str, lag: int) -> None:
    """Handing chunks on late (the CUDA copy-back overlap) yields the same items in the same order."""
    pack = FakePack(layers=2, seed=3)
    base = MaskGenerator(pack, RING_K40_A12, backend=backend, self_check_rows=-1)
    want = [(i.id.key, i.slot_start, i.wr.clone(), i.uw) for i in base.generate(SEED, 21, checks=2, chunk_slots=4)]
    gen = MaskGenerator(pack, RING_K40_A12, backend=backend, self_check_rows=-1)
    gen._output_lag = lag
    got = [(i.id.key, i.slot_start, i.wr.clone(), i.uw) for i in gen.generate(SEED, 21, checks=2, chunk_slots=4)]
    assert [(a, b) for a, b, _, _ in got] == [(a, b) for a, b, _, _ in want]
    for (_, _, w1, u1), (_, _, w2, u2) in zip(got, want, strict=True):
        assert torch.equal(w1, w2)
        assert (u1 is None) == (u2 is None)
        if u1 is not None and u2 is not None:
            assert torch.equal(u1, u2)
    assert gen.last_stats.self_checked_rows == 21 * len(pack.linear_specs())


def test_output_lag_failure_aborts_and_zeroes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-check failure with chunks still in flight aborts the segment; the queued chunks are zeroed."""
    pack = FakePack(layers=1, seed=3)
    seen: list[torch.Tensor] = []
    orig = RingKernel.matmul

    def bad(self: RingKernel, r: torch.Tensor, params: Any) -> torch.Tensor:
        out = orig(self, r, params)
        seen.append(out)
        if self.out_features == 52:  # gate_up
            out[:, 1] ^= 1 << 3
        return out

    monkeypatch.setattr(RingKernel, "matmul", bad)
    store = FakeMaskStore()
    gen = MaskGenerator(pack, RING_K40_A12, self_check_rows=-1)
    gen._output_lag = 2
    with pytest.raises(IntegrityError):
        gen.generate_into_store(store, 12, checks=2, chunk_slots=4)
    assert store.aborted == 1 and not store.segments
    gate = [t for t in seen if t.shape[1] == 52]
    assert gate and all(int(t.abs().sum()) == 0 for t in gate)  # failed and queued chunks zeroed


@needs_int8
def test_chunk_size_does_not_change_output() -> None:
    pack = FakePack(layers=1, seed=5)
    ref = reference_segment(pack, RING_K32_A12_C27, SEED, 19, 2)
    for chunk in (1, 3, 8, 19, 64, None):
        gen = MaskGenerator(pack, RING_K32_A12_C27, backend="cpu_int8")
        got: dict[str, list[torch.Tensor]] = {}
        for item in gen.generate(SEED, 19, checks=2, chunk_slots=chunk):
            got.setdefault(item.id.key, []).append(item.wr.clone())
        for key, (wr, _) in ref.items():
            assert torch.equal(torch.cat(got[key]), wr), (chunk, key)


def test_chunk_slots_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = MaskGenerator(FakePack(layers=1), RING_K40_A12)
    assert gen.chunk_slots_for(None) == fg.DEFAULT_CHUNK_SLOTS == 256  # CPU
    assert gen.chunk_slots_for(77) == 77
    # CUDA budget rule: 1024 when the per-row working set fits, halved down to 256 otherwise
    assert gen._chunk_for_budget(1e12) == fg.CUDA_CHUNK_SLOTS == 1024
    per_row = max(5 * s.in_features + 20 * s.out_features + 16 * s.out_features + 16 * s.in_features
                  for s in gen.specs)
    assert gen._chunk_for_budget(1024 * per_row) == 1024
    assert gen._chunk_for_budget(1024 * per_row - 1) == 512
    assert gen._chunk_for_budget(1.0) == 256
    with pytest.raises(Exception, match="chunk_slots"):
        list(gen.generate(SEED, 4, chunk_slots=0))
    # factory settings: unset = auto
    assert FactorySettings(pack_path=Path("/x")).chunk_slots is None
    assert FactorySettings(pack_path=Path("/x"), chunk_slots=512).chunk_slots == 512


def test_generate_into_store_auto_chunk_and_stage_log(caplog: pytest.LogCaptureFixture) -> None:
    pack = FakePack(layers=1, seed=2)
    store = FakeMaskStore()
    gen = MaskGenerator(pack, RING_K40_A12)
    with caplog.at_level(logging.INFO, logger="pgw.factory.generator"):
        seg = gen.generate_into_store(store, 9, checks=2, seed=SEED)
    ref = reference_segment(pack, RING_K40_A12, SEED, 9, 2)
    for key, (wr, uw) in ref.items():
        assert torch.equal(torch.cat(store.segments[seg].wr[key]), wr)
        assert torch.equal(store.segments[seg].uw[key], uw)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(f"segment {seg}"))
    for part in ("rng ", "gemm ", "check ", "write ", "throttle ", "chunk 256", "slots/s"):
        assert part in line, line
    assert ("[native]" if fg._NATIVE_RNG else "[python]") in line


class _FakeHE:
    """Stands in for pgw.hemask.HEMaskBackend: exact W r plus cumulative stats."""

    def __init__(self, pack: FakePack, ring: RingParams):
        self.pack, self.ring = pack, ring
        self.stats: dict[str, Any] = {"groups": 7, "encrypt_s": 1.0, "remote_s": 2.0, "decrypt_s": 3.0, "wall_s": 4.0}

    def matmul(self, linear: str, r: torch.Tensor) -> torch.Tensor:
        self.stats["groups"] += 1
        self.stats["encrypt_s"] += 0.25
        self.stats["remote_s"] += 0.5
        return torch.from_numpy(ringmm_reference(r, self.pack.weights[linear], self.ring.k))


def test_he_stats_logged_per_segment(caplog: pytest.LogCaptureFixture) -> None:
    pack = FakePack(layers=1, seed=4)
    he = _FakeHE(pack, RING_K40_A12)
    gen = MaskGenerator(pack, RING_K40_A12, remote=he)
    assert gen.chunk_slots_for(None) == 256
    store = FakeMaskStore()
    with caplog.at_level(logging.INFO, logger="pgw.factory.generator"):
        seg = gen.generate_into_store(store, 5, checks=2, chunk_slots=2, seed=SEED)
    calls = 3 * len(pack.linear_specs())  # 5 slots in chunks of 2
    d = gen.last_stats.extra["he"]
    assert d["groups"] == calls and d["encrypt_s"] == pytest.approx(0.25 * calls)
    assert d["remote_s"] == pytest.approx(0.5 * calls) and d["decrypt_s"] == 0
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(f"segment {seg}"))
    assert f"he: {calls} groups" in line and "provider" in line
    ref = reference_segment(pack, RING_K40_A12, SEED, 5, 2)
    for key, (wr, _) in ref.items():
        assert torch.equal(torch.cat(store.segments[seg].wr[key]), wr)


@needs_int8
def test_real_store_segment_round_trip_bit_identical(tmp_path: Path) -> None:
    """Stored segments decrypt to the reference masks (the at-rest bytes are W r packed as before)."""
    from pgw.maskstore import FileKeyProvider, MaskStore

    pack = FakePack(layers=1, seed=6)
    store = MaskStore.open(tmp_path / "store", FileKeyProvider(tmp_path / "kek", create=True))
    try:
        gen = MaskGenerator(pack, RING_K32_A12_C27, backend="cpu_int8")
        gen._output_lag = 1
        gen.generate_into_store(store, 11, checks=8, chunk_slots=4, seed=SEED)
        ref = reference_segment(pack, RING_K32_A12_C27, SEED, 11, 8)
        with store.reserve(pack.fingerprint, RING_K32_A12_C27, 11) as res:
            for spec in pack.linear_specs():
                r, wr = res.take(spec.id, 11)
                assert torch.equal(wr, ref[spec.id.key][0])
                assert torch.equal(r, SeedExpander(SEED).ring_uniform((11, spec.in_features), 32, "r", spec.id.key))
    finally:
        store.close()


def test_bench_reports_stages() -> None:
    res = benchmark_preset("qwen2.5-0.5b", device="cpu", slots=8, chunk_slots=None, warmup=False)
    assert res.chunk_slots == 256 and res.gemm_seconds > 0 and res.rng_seconds >= 0
    assert res.native_rng is fg._NATIVE_RNG and not res.gpu_prng
    rep = format_report(res)
    assert "stages" in rep and "chunk 256 slots" in rep
