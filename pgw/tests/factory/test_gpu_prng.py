"""GPU mask PRNG (pgw.factory.gpu_prng): ChaCha20 + digit split on the device, byte-identical to the CPU path.

CPU-runnable: the plain-torch emulation against ``cryptography``'s ChaCha20 and the CPU expander, the
Triton kernel in the Triton interpreter (``TRITON_INTERPRET=1``, set by ``tests/conftest.py`` without CUDA),
and the generator's device path (``cpu_int8`` + ``gpu_prng_impl``), its fallback and its self-check.
``gpu``-marked tests run the real kernel on CUDA.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from pgw.core import (
    RING_K32_A12_C27,
    RING_K40_A12,
    RING_K48_A20,
    ConfigError,
    IntegrityError,
    SeedExpander,
    ShapeError,
)
from pgw.core.ring import balanced_digits
from pgw.factory import MaskGenerator
from pgw.factory import gpu_prng as gp
from pgw.factory.kernels import RingKernel, cpu_int8_available

from .conftest import FakeMaskStore, FakePack, reference_segment

SEED = bytes(range(32))
KS = (32, 40, 48)
needs_int8 = pytest.mark.skipif(not cpu_int8_available(), reason="CPU torch._int_mm not exact here")


def _crypto_blocks(key: bytes, counter: int, nblocks: int) -> np.ndarray:
    nonce = counter.to_bytes(4, "little") + bytes(12)
    raw = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor().update(bytes(64 * nblocks))
    return np.frombuffer(raw, dtype="<u4").reshape(nblocks, 16).astype(np.int64)


def _triton_ok() -> bool:
    if not gp.triton_available():
        return False
    return torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


needs_triton = pytest.mark.skipif(not _triton_ok(), reason="triton (or its interpreter) unavailable")


@pytest.fixture(autouse=True)
def _fresh_self_test_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own self-test cache: a test that injects a fault must not poison later tests
    (the self-test does catch injected faults and would cache the failure process-wide)."""
    monkeypatch.setattr(gp, "_SELF_TEST", {})


# ---- the ChaCha20 block function (torch emulation) vs cryptography ----------------------------------
@pytest.mark.parametrize("trial", range(6))
def test_torch_chacha_block_matches_cryptography(trial: int) -> None:
    rng = random.Random(trial)
    key = rng.randbytes(32)
    counter = rng.choice([0, 1, rng.randrange(1 << 31), (1 << 31) - 1, 1 << 31, rng.randrange(1 << 31, 1 << 32)])
    n = min(5, (1 << 32) - counter)
    got = gp.chacha20_blocks_torch(key, counter + torch.arange(n, dtype=torch.int64))
    assert got.dtype == torch.int64 and tuple(got.shape) == (n, 16)
    assert np.array_equal(got.numpy(), _crypto_blocks(key, counter, n))


def test_torch_chacha_rfc8439_vector() -> None:
    """RFC 8439 2.3.2 test vector (its nonce is non-zero, so run the block function by hand on that state)."""
    key = bytes(range(32))
    # the emulation fixes nonce = 0; check the zero-nonce stream at counter 1 against cryptography and
    # the first keystream word of the all-zero key (RFC 8439 A.1 test vector 1: 0xade0b876)
    assert int(gp.chacha20_blocks_torch(bytes(32), torch.tensor([0]))[0, 0]) == 0xADE0B876
    assert np.array_equal(gp.chacha20_blocks_torch(key, torch.tensor([1])).numpy(), _crypto_blocks(key, 1, 1))


def test_torch_chacha_last_counter_and_range() -> None:
    key = os.urandom(32)
    top = (1 << 32) - 1
    assert np.array_equal(gp.chacha20_blocks_torch(key, torch.tensor([top])).numpy(), _crypto_blocks(key, top, 1))
    with pytest.raises(ConfigError):
        gp.chacha20_blocks_torch(key, torch.tensor([1 << 32]))
    with pytest.raises(ConfigError):
        gp.chacha20_blocks_torch(key, torch.tensor([-1]))


def test_key_words_little_endian() -> None:
    key = bytes(range(32))
    kw = gp.key_words(key)
    assert kw.dtype == torch.int32 and kw.tolist()[0] == 0x03020100
    with pytest.raises(ConfigError):
        gp.key_words(bytes(31))


# ---- ring rows and digit planes (emulation) vs SeedExpander -----------------------------------------
def _cases(seed: int, n: int) -> list[tuple[int, int, int, int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append((rng.choice(KS), rng.choice([1, 2, 3, 7, 8, 13, 64, 100, 129, 257]), rng.randrange(0, 50),
                    rng.randrange(0, 9)))
    return out


@pytest.mark.parametrize(("k", "cols", "start", "rows"), [*_cases(1, 40), (40, 8, (1 << 32) - 4, 3), (48, 1, 0, 0)])
def test_torch_rows_match_seed_expander(k: int, cols: int, start: int, rows: int) -> None:
    key = SeedExpander(SEED).stream_key("r", f"lin.{cols}")
    want = SeedExpander.ring_uniform_rows_with_key(key, cols, k, start, rows)
    assert torch.equal(gp.ring_rows_torch(key, cols, k, start, rows), want)
    # the planes, padded like the CUDA GEMM operand, equal balanced_digits of the CPU rows
    n_l = gp.limbs(k)
    p, np_ = rows + 2, cols + 3
    out = torch.full((n_l * p, np_), 7, dtype=torch.int8)
    gp.digit_planes_torch(key, cols, k, start, rows, out)
    exp = torch.full((n_l, p, np_), 7, dtype=torch.int8)
    for i, d in enumerate(balanced_digits(want, k, 8)):
        exp[i, :rows, :cols] = d.to(torch.int8)
    assert torch.equal(out.view(n_l, p, np_), exp)


def test_torch_rows_match_ring_uniform_whole_matrix() -> None:
    """Row slices at every offset reassemble ``ring_uniform`` (the normative derivation, SPEC 7.3)."""
    exp = SeedExpander(SEED)
    full = exp.ring_uniform((11, 9), 40, "r", "0.qkv")
    key = exp.stream_key("r", "0.qkv")
    parts = [gp.ring_rows_torch(key, 9, 40, s, min(3, 11 - s)) for s in range(0, 11, 3)]
    assert torch.equal(torch.cat(parts), full)


def test_stream_layout_and_range_checks() -> None:
    assert gp.stream_layout(3, 40, 5, 7) == (1, 7, 21, 4)  # byte offset 120: block 1, 56 bytes = 7 words in
    with pytest.raises(ConfigError):
        gp.stream_layout(8, 40, 1 << 32, 1)  # first block beyond the counter
    with pytest.raises(ConfigError):
        gp.stream_layout(8, 40, (1 << 32) - 1, 2)  # runs past the last block (stricter than the CPU path, which checks the start only)
    with pytest.raises(ConfigError):
        gp.stream_layout(0, 40, 0, 1)
    with pytest.raises(ConfigError):
        gp.stream_layout(4, 64, 0, 1)


def test_setting_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gp.GPU_PRNG_ENV, raising=False)
    assert gp.gpu_prng_setting() is None
    for raw, want in (("0", False), ("off", False), ("1", True), ("true", True), ("auto", None), ("", None)):
        monkeypatch.setenv(gp.GPU_PRNG_ENV, raw)
        assert gp.gpu_prng_setting() is want
    assert gp.gpu_prng_setting(False) is False
    monkeypatch.setenv(gp.GPU_PRNG_ENV, "maybe")
    with pytest.raises(ConfigError):
        gp.gpu_prng_setting()


def test_stream_wipes_device_key() -> None:
    key = os.urandom(32)
    st = gp.DeviceRingStream(key, 5, 40, "cpu", "torch")
    kt = st._key_t
    hk = st._key
    assert kt is not None and bool(kt.ne(0).any())
    assert hk is not None and bytes(hk) == key
    assert "redacted" in repr(st) and key.hex() not in repr(st)
    with st:
        st.rows(0, 2)
    assert int(kt.abs().sum()) == 0 and st._key_t is None
    assert hk == bytearray(32) and st._key is None  # host copy zeroed too
    with pytest.raises(ConfigError):
        st.rows(0, 1)
    with pytest.raises(ConfigError):
        st.cpu_rows(0, torch.tensor([0]))


def test_digit_plane_buffer_validation() -> None:
    st = gp.DeviceRingStream(os.urandom(32), 5, 40, "cpu", "torch")
    with pytest.raises(ShapeError):
        st.digit_planes(0, 3, torch.zeros((5 * 2, 5), dtype=torch.int8))  # P = 2 < rows
    with pytest.raises(ShapeError):
        st.digit_planes(0, 3, torch.zeros((4 * 3, 5), dtype=torch.int8))  # 4 planes for K=40
    st.wipe()


# ---- the Triton kernel (interpreter on CPU, compiled on CUDA) ----------------------------------------
@needs_triton
@pytest.mark.parametrize("k", KS)
def test_triton_kernel_matches_cpu(k: int) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    key = os.urandom(32)
    for cols, start, rows in [(1, 0, 1), (3, 5, 7), (37, 11, 9), (129, 1, 33), (8, (1 << 32) - 3, 2)]:
        want = SeedExpander.ring_uniform_rows_with_key(key, cols, k, start, rows)
        with gp.DeviceRingStream(key, cols, k, dev, "triton") as st:
            assert torch.equal(st.rows(start, rows).cpu(), want), (cols, start, rows)
            n_l = gp.limbs(k)
            p, np_ = rows + 1, cols + 4
            out = torch.full((n_l * p, np_), -3, dtype=torch.int8, device=dev)
            st.digit_planes(start, rows, out)
        exp = torch.full((n_l, p, np_), -3, dtype=torch.int8)
        for i, d in enumerate(balanced_digits(want, k, 8)):
            exp[i, :rows, :cols] = d.to(torch.int8)
        assert torch.equal(out.cpu().view(n_l, p, np_), exp), (cols, start, rows)


def test_self_test_torch_impl_passes_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert gp._SELF_TEST == {}  # fresh per test (autouse fixture)
    assert gp.self_test("cpu", "torch")
    calls = []
    monkeypatch.setattr(gp, "check_against_cpu", lambda *a, **kw: calls.append(1) or [])
    assert gp.self_test("cpu", "torch") and not calls  # cached


# ---- generator integration ---------------------------------------------------------------------------
@needs_int8
@pytest.mark.parametrize("ring", [RING_K32_A12_C27, RING_K40_A12, RING_K48_A20])
@pytest.mark.parametrize("pipeline", [True, False])
def test_generator_device_path_matches_reference(ring: object, pipeline: bool) -> None:
    pack = FakePack()
    gen = MaskGenerator(pack, ring, backend="cpu_int8", gpu_prng=True, gpu_prng_impl="torch", pipeline=pipeline,
                        self_check_rows=3)
    items = list(gen.generate(SEED, 7, checks=2, chunk_slots=3))  # chunks start at unaligned rows 3 and 6
    assert gen.last_stats.extra["gpu_prng"] is True
    ref = reference_segment(pack, ring, SEED, 7, 2)  # type: ignore[arg-type]
    for spec in pack.linear_specs():
        got = torch.cat([i.wr for i in items if i.id == spec.id])
        assert torch.equal(got, ref[spec.id.key][0])
        uw = next(i.uw for i in items if i.id == spec.id and i.uw is not None)
        assert torch.equal(uw, ref[spec.id.key][1])
    assert gen.last_stats.self_checked_rows == (3 + 3 + 1) * len(pack.linear_specs())


@needs_int8
@needs_triton
@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET", "0") in ("", "0"), reason="needs the Triton interpreter")
def test_generator_triton_interpreter_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real kernel inside the generator (interpreter on CPU; the self-test itself is covered above)."""
    monkeypatch.setattr(gp, "self_test", lambda *a, **kw: True)
    pack = FakePack(layers=1)
    gen = MaskGenerator(pack, RING_K40_A12, backend="cpu_int8", gpu_prng=True, gpu_prng_impl="triton")
    items = list(gen.generate(SEED, 5, checks=2, chunk_slots=2))
    assert gen.last_stats.extra["gpu_prng"] is True
    ref = reference_segment(pack, RING_K40_A12, SEED, 5, 2)
    for spec in pack.linear_specs():
        assert torch.equal(torch.cat([i.wr for i in items if i.id == spec.id]), ref[spec.id.key][0])


@needs_int8
def test_failed_self_test_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    assert gp._SELF_TEST == {}  # fresh per test (autouse fixture)
    monkeypatch.setattr(gp, "check_against_cpu", lambda *a, **kw: ["rows K=40 (simulated)"])
    pack = FakePack(layers=1)
    gen = MaskGenerator(pack, RING_K40_A12, backend="cpu_int8", gpu_prng=True, gpu_prng_impl="torch")
    with caplog.at_level(logging.ERROR, logger="pgw.factory.gpu_prng"):
        items = list(gen.generate(SEED, 4, checks=2, chunk_slots=4))
    assert "self-test FAILED" in caplog.text
    assert gen.device_prng() is None and gen.last_stats.extra["gpu_prng"] is False
    ref = reference_segment(pack, RING_K40_A12, SEED, 4, 2)
    for spec in pack.linear_specs():
        assert torch.equal(torch.cat([i.wr for i in items if i.id == spec.id]), ref[spec.id.key][0])


@needs_int8
def test_self_test_exception_counts_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    assert gp._SELF_TEST == {}  # fresh per test (autouse fixture)

    def boom(*a: object, **kw: object) -> list[str]:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(gp, "check_against_cpu", boom)
    assert gp.self_test("cpu", "torch") is False


@needs_int8
@pytest.mark.parametrize("mode", ["planes", "gemm"])
def test_device_path_fault_aborts_segment_and_disables(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """A wrong GPU keystream (or digit-plane GEMM) fails the CPU-rederived self-check; later segments use the CPU."""
    pack = FakePack(layers=1)
    store = FakeMaskStore()
    assert gp.self_test("cpu", "torch", (40,))  # passes before the fault is injected (cached for this test)
    if mode == "planes":
        orig = gp.DeviceRingStream.digit_planes

        def bad(self: gp.DeviceRingStream, row_start: int, row_count: int, out: torch.Tensor) -> torch.Tensor:
            orig(self, row_start, row_count, out)
            out[0, 0] ^= 1  # one wrong digit: r[0, 0] off by one
            return out

        monkeypatch.setattr(gp.DeviceRingStream, "digit_planes", bad)
    else:
        orig_mm = RingKernel.matmul_planes

        def bad_mm(self: RingKernel, d8: torch.Tensor, rows: int, params: object) -> torch.Tensor:
            out = orig_mm(self, d8, rows, params)  # type: ignore[arg-type]
            out[:, 0] ^= 1
            return out

        monkeypatch.setattr(RingKernel, "matmul_planes", bad_mm)
    gen = MaskGenerator(pack, RING_K40_A12, backend="cpu_int8", gpu_prng=True, gpu_prng_impl="torch",
                        self_check_rows=-1)
    with pytest.raises(IntegrityError):
        gen.generate_into_store(store, 4, checks=2, chunk_slots=4)
    assert store.aborted == 1 and not store.segments
    assert gen.device_prng() is None  # never emits masks from the failed GPU path again
    seg = gen.generate_into_store(store, 4, checks=2, chunk_slots=4, seed=SEED)
    assert seg in store.segments and gen.last_stats.extra["gpu_prng"] is False


@needs_int8
def test_device_prng_error_aborts_segment_and_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kernel launch/compile error (not only a failed check) switches the generator to the CPU path."""
    pack = FakePack(layers=1)
    store = FakeMaskStore()
    assert gp.self_test("cpu", "torch", (40,))

    def boom(self: gp.DeviceRingStream, row_start: int, row_count: int, out: torch.Tensor) -> torch.Tensor:
        out[0, 0] = 7  # partial write before the fault
        raise RuntimeError("launch failed")

    monkeypatch.setattr(gp.DeviceRingStream, "digit_planes", boom)
    gen = MaskGenerator(pack, RING_K40_A12, backend="cpu_int8", gpu_prng=True, gpu_prng_impl="torch")
    with pytest.raises(RuntimeError, match="launch failed"):
        gen.generate_into_store(store, 4, checks=2, chunk_slots=4)
    assert store.aborted == 1 and not store.segments
    assert gen.device_prng() is None
    seg = gen.generate_into_store(store, 4, checks=2, chunk_slots=4, seed=SEED)
    assert seg in store.segments and gen.last_stats.extra["gpu_prng"] is False


@needs_int8
def test_device_prng_needs_output_self_check() -> None:
    """Without the per-chunk self-check nothing would re-derive r on the CPU: the GPU path is refused."""
    pack = FakePack(layers=1)
    gen = MaskGenerator(pack, RING_K40_A12, backend="cpu_int8", gpu_prng=True, gpu_prng_impl="torch",
                        self_check_rows=0)
    assert gen.device_prng() is None
    items = list(gen.generate(SEED, 4, checks=2, chunk_slots=4))
    assert gen.last_stats.extra["gpu_prng"] is False
    ref = reference_segment(pack, RING_K40_A12, SEED, 4, 2)
    for spec in pack.linear_specs():
        assert torch.equal(torch.cat([i.wr for i in items if i.id == spec.id]), ref[spec.id.key][0])


@needs_int8
def test_env_switch_off_and_non_int8_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    pack = FakePack(layers=1)
    monkeypatch.setenv(gp.GPU_PRNG_ENV, "0")
    assert MaskGenerator(pack, RING_K40_A12, backend="cpu_int8", gpu_prng_impl="torch").device_prng() is None
    monkeypatch.setenv(gp.GPU_PRNG_ENV, "1")
    # CPU factories never use the device path unless a test implementation is named
    assert MaskGenerator(pack, RING_K40_A12, backend="cpu_int8").device_prng() is None
    assert MaskGenerator(pack, RING_K40_A12, backend="cpu_f64", gpu_prng_impl="torch").device_prng() is None
    assert MaskGenerator(pack, RING_K40_A12, backend="cpu_int8", gpu_prng_impl="torch").device_prng() == "torch"
    with pytest.raises(ConfigError):
        MaskGenerator(pack, RING_K40_A12, gpu_prng_impl="cuda")


def test_settings_wire_gpu_prng(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.factory import FactorySettings
    from pgw.factory.runtime import build_generators

    monkeypatch.delenv(gp.GPU_PRNG_ENV, raising=False)
    s = FactorySettings(pack_path=tmp_path, gpu_prng=False)  # type: ignore[arg-type]
    (g,) = build_generators(s, [FakePack(layers=1)]).values()
    assert g.gpu_prng_requested is False
    monkeypatch.setenv("PGW_FACTORY_GPU_PRNG", "0")
    assert FactorySettings(pack_path=tmp_path).gpu_prng is False  # type: ignore[arg-type]
    monkeypatch.setenv("PGW_FACTORY_GPU_PRNG", "auto")
    assert FactorySettings(pack_path=tmp_path).gpu_prng == "auto"  # type: ignore[arg-type]


@needs_int8
@pytest.mark.parametrize("k", KS)
def test_matmul_planes_equals_matmul_cpu(k: int) -> None:
    from pgw.core.ring import RingParams

    gen = torch.Generator().manual_seed(k)
    w = torch.randint(-128, 128, (13, 21), dtype=torch.int8, generator=gen)
    w[0, 0], w[-1, -1] = -128, 127
    kern = RingKernel(w, "cpu_int8")
    params = RingParams(k, 8)
    for rows in (0, 1, 5):
        r = torch.randint(0, 1 << k, (rows, 21), dtype=torch.int64, generator=gen)
        p, np_ = kern.plane_shape(rows)
        d8 = torch.zeros((gp.limbs(k) * p, np_), dtype=torch.int8)
        for i, d in enumerate(balanced_digits(r, k, 8)):
            d8[i * p : i * p + rows, :21] = d.to(torch.int8)
        assert torch.equal(kern.matmul_planes(d8, rows, params), kern.matmul(r, params))


# ---- CUDA ----------------------------------------------------------------------------------------------
def _gpu_cases() -> list[tuple[int, int, int, int]]:
    rng = random.Random(7)
    cases = []
    for k in KS:
        for cols in (1, 3, 7, 8, 64, 129, 896, 3584, 4096, 4864, 14336, 18944):
            for _ in range(3):
                cases.append((k, cols, rng.choice([0, 1, 3, rng.randrange(1, 5000)]), rng.choice([1, 2, 31, 33, 256])))
    cases += [(k, 8, (1 << 32) - 5, 5) for k in KS]
    return cases


@pytest.mark.gpu
def test_gpu_self_test_passes() -> None:  # pragma: no cover - GPU only
    gp._SELF_TEST.clear()
    assert gp.self_test("cuda", "triton", (32, 40, 48))
    assert gp.check_against_cpu("cuda", "triton", quick=False, keys=3) == []


@pytest.mark.gpu
def test_gpu_rows_and_planes_byte_identical() -> None:  # pragma: no cover - GPU only
    key = os.urandom(32)
    for k, cols, start, rows in _gpu_cases():
        want = SeedExpander.ring_uniform_rows_with_key(key, cols, k, start, rows)
        with gp.DeviceRingStream(key, cols, k, "cuda", "triton") as st:
            got = st.rows(start, rows)
            assert got.is_cuda and torch.equal(got.cpu(), want), (k, cols, start, rows)
            n_l = gp.limbs(k)
            p, np_ = max(32, -(-rows // 8) * 8), -(-cols // 32) * 32
            out = torch.zeros((n_l * p, np_), dtype=torch.int8, device="cuda")
            st.digit_planes(start, rows, out)
        exp = torch.zeros((n_l, p, np_), dtype=torch.int8)
        for i, d in enumerate(balanced_digits(want, k, 8)):
            exp[i, :rows, :cols] = d.to(torch.int8)
        assert torch.equal(out.cpu().view(n_l, p, np_), exp), (k, cols, start, rows)


@pytest.mark.gpu
@pytest.mark.parametrize("ring", [RING_K32_A12_C27, RING_K40_A12, RING_K48_A20])
def test_gpu_generator_identical_with_and_without_gpu_prng(ring: object) -> None:  # pragma: no cover - GPU only
    pack = FakePack()
    outs = {}
    for flag in (True, False):
        gen = MaskGenerator(pack, ring, device="cuda", gpu_prng=flag)
        items = list(gen.generate(SEED, 37, checks=2, chunk_slots=16))
        assert gen.last_stats.extra["gpu_prng"] is flag
        outs[flag] = {spec.id.key: torch.cat([i.wr for i in items if i.id == spec.id]) for spec in pack.linear_specs()}
    ref = reference_segment(pack, ring, SEED, 37, 2)  # type: ignore[arg-type]
    for key, (wr, _) in ref.items():
        assert torch.equal(outs[True][key], wr) and torch.equal(outs[False][key], wr)


@pytest.mark.gpu
def test_gpu_matmul_planes_equals_matmul() -> None:  # pragma: no cover - GPU only
    from pgw.core.ring import RingParams

    gen = torch.Generator().manual_seed(3)
    w = torch.randint(-128, 128, (100, 130), dtype=torch.int8, generator=gen).cuda()
    kern = RingKernel(w, "cuda_int8")
    key = os.urandom(32)
    for k in KS:
        params = RingParams(k, 8)
        for rows in (1, 7, 33, 256):
            with gp.DeviceRingStream(key, 130, k, "cuda", "triton") as st:
                p, np_ = kern.plane_shape(rows)
                d8 = torch.zeros((gp.limbs(k) * p, np_), dtype=torch.int8, device="cuda")
                st.digit_planes(5, rows, d8)
                r = st.rows(5, rows)
            assert torch.equal(kern.matmul_planes(d8, rows, params), kern.matmul(r, params))
