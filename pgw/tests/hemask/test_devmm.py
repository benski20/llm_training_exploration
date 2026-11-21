"""Fused device passes of the encrypted mask factory (pgw.hemask.devmm): bit-identical to the CPU engines.

The device code runs on CPU tensors here (same torch ops); the Kaggle job runs it on CUDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.ring import RingParams
from pgw.factory import MaskGenerator
from pgw.hemask import DEGREE, HEParams, devmm, encrypt_masks, fastmm, matmul_mod64, public_matrix
from pgw.hemask.backend import HEMaskBackend, HintStore
from pgw.hemask.fastmm import to_planes
from pgw.hemask.lwe import a_plane_count

from .test_factory_he import SEED, FakeProvider


@pytest.mark.parametrize("k", [32, 40])
def test_self_test_passes_on_cpu_tensors(k: int) -> None:
    devmm._TESTED.pop(f"cpu/k{k}/r16", None)
    assert devmm.self_test("cpu", k=k, rank=16)
    assert devmm.status()[f"cpu/k{k}/r16"] == {"ok": True}


def test_device_encrypt_equals_encrypt_masks() -> None:
    p = HEParams(k=32, rank=8)
    rng = np.random.default_rng(3)
    n, rows = 50, 2 * DEGREE + 3
    a = public_matrix(bytes(32), n, p)
    planes = to_planes(a)[:, :a_plane_count(p)]
    eng = devmm.DeviceHE(planes, p, "cpu", hint_budget=0)
    r = rng.integers(0, 1 << 32, size=(rows, n), dtype=np.int64)
    seeds = [bytes([i + 1]) * 32 for i in range(3)]
    cts, b = eng.encrypt(r, n, seeds=seeds)
    assert [ct.rows for ct in cts] == [DEGREE, DEGREE, 3]
    for gi, seed in enumerate(seeds):
        ref = encrypt_masks(r[gi * DEGREE:(gi + 1) * DEGREE], a, p, _seed=seed)
        assert np.array_equal(b[:, gi * DEGREE:(gi + 1) * DEGREE], ref.b)
        assert np.array_equal(cts[gi].secret, ref.secret)


def test_self_test_catches_a_wrong_combine(monkeypatch: pytest.MonkeyPatch) -> None:
    real = devmm.combine

    def off_by_one(part: torch.Tensor, offset: torch.Tensor, first_byte: int) -> torch.Tensor:
        out = real(part, offset, first_byte)
        out[0, 0] += 1
        return out

    monkeypatch.setattr(devmm, "combine", off_by_one)
    devmm._TESTED.pop("cpu/k32/r8", None)
    try:
        assert not devmm.self_test("cpu", k=32, rank=8)
        assert "error" in devmm.status()["cpu/k32/r8"]
    finally:
        devmm._TESTED.pop("cpu/k32/r8", None)
        devmm._STATUS.pop("cpu/k32/r8", None)


def test_failed_self_test_turns_the_cuda_engine_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fastmm, "engine", lambda: "cuda")
    monkeypatch.setattr(devmm, "self_test", lambda *a, **k: False)
    calls: list[str] = []
    monkeypatch.setattr(fastmm, "disable_cuda", calls.append)
    assert devmm.device_for(HEParams(k=32, rank=8)) is None
    assert calls


def test_disable_cuda_overrides_a_forced_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_HEMASK_ENGINE", "cuda")
    monkeypatch.setattr(fastmm, "_CUDA_DISABLED", None)
    assert fastmm.engine() == "cuda"
    fastmm.disable_cuda("test")
    assert fastmm.engine() in ("cpu_int8", "cpu_f64")
    assert fastmm.cuda_disabled() == "test"


@pytest.mark.parametrize("budget", [0, 1 << 30])
@pytest.mark.parametrize("ring_name", ["K32A12C27", "K40A12"])
def test_device_factory_matches_local_factory(tiny_pack: Any, tmp_path: Path, ring_name: str, budget: int,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamed and cached hints, several requests per chunk, a split pass: the same W8 r as local."""
    monkeypatch.setenv("PGW_HEMASK_GPU_HINT_MB", str(budget >> 20))
    ring = RingParams.from_name(ring_name)
    q = HEParams(k=ring.k, rank=8).q
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=q)
    remote = HEMaskBackend(tiny_pack, ring, hints, FakeProvider(tiny_pack), groups_per_call=1, groups_per_pass=2,
                           device="cpu")
    assert remote.engine == "device:cpu" and remote._dev is not None
    assert remote.preferred_chunk_slots() == 2 * DEGREE
    local = {(g.id.key, g.slot_start): g.wr.clone()
             for g in MaskGenerator(tiny_pack, ring, device="cpu").generate(SEED, 700, checks=2, chunk_slots=700)}
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    assert gen.chunk_slots_for(None) == 2 * DEGREE
    he = {(g.id.key, g.slot_start): g.wr.clone() for g in gen.generate(SEED, 700, checks=2, chunk_slots=700)}
    assert local.keys() == he.keys()
    for key in local:
        assert torch.equal(local[key], he[key]), key
    assert gen.last_stats.self_checked_rows == 700 * len(list(tiny_pack.linear_specs()))
    assert (remote._dev.stats["hint_uploads"] > 0) == (budget > 0)


def test_device_decrypt_rejects_a_tampered_reply(tiny_pack: Any, tmp_path: Path) -> None:
    from pgw.core.errors import IntegrityError

    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    remote = HEMaskBackend(tiny_pack, ring, hints, FakeProvider(tiny_pack, tamper=True), device="cpu")
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    with pytest.raises(IntegrityError):
        for _ in gen.generate(SEED, 600, checks=2, chunk_slots=300):
            pass


def test_combine_matches_fastmm_reference() -> None:
    rng = np.random.default_rng(5)
    rows, k, c = 33, 512, 40
    y = rng.integers(0, 2**64 - 1, size=(rows, k), dtype=np.uint64)
    s = rng.integers(-1, 2, size=(k, c)).astype(np.int8)
    ref = matmul_mod64(s, y, small_left=False, fast=False)
    pl = torch.from_numpy(to_planes(y))
    st = torch.from_numpy(np.ascontiguousarray(s.T))
    part = devmm.int_mm(pl.reshape(rows * 8, k), st).view(rows, 8, c)
    got = devmm.combine(part, st.sum(1, dtype=torch.int64) * 128, 0).numpy().view(np.uint64)
    assert np.array_equal(got, ref)


@pytest.mark.gpu
@pytest.mark.parametrize("budget", [0, 1 << 30])
def test_cuda_device_factory_matches_local_factory(tiny_pack: Any, tmp_path: Path, budget: int,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:  # pragma: no cover
    monkeypatch.setenv("PGW_HEMASK_GPU_HINT_MB", str(budget >> 20))
    for k in (32, 40):
        assert devmm.self_test("cuda", k=k, rank=16)
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    remote = HEMaskBackend(tiny_pack, ring, hints, FakeProvider(tiny_pack), groups_per_call=1, groups_per_pass=2,
                           device="cuda")
    local = {(g.id.key, g.slot_start): g.wr.clone()
             for g in MaskGenerator(tiny_pack, ring, device="cpu").generate(SEED, 700, checks=2, chunk_slots=700)}
    gen = MaskGenerator(tiny_pack, ring, device="cuda", remote=remote)
    he = {(g.id.key, g.slot_start): g.wr.clone() for g in gen.generate(SEED, 700, checks=2, chunk_slots=700)}
    assert local.keys() == he.keys()
    for key in local:
        assert torch.equal(local[key], he[key].cpu()), key


def test_hint_cache_falls_back_to_streaming_when_the_device_is_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device memory taken by another backend or process: the hint is streamed (same bits), not an error."""
    p = HEParams(k=32, rank=8)
    a = public_matrix(bytes(32), 8, p)
    eng = devmm.DeviceHE(to_planes(a)[:, :a_plane_count(p)], p, "cpu", hint_budget=1 << 30)
    hint = np.ones((16, 6, p.dim), dtype=np.int8)
    assert eng.hint_on_device("a", hint) is not None
    real_empty = torch.empty

    def full(*args: Any, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("device") is not None:
            raise torch.cuda.OutOfMemoryError("full")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(devmm.torch, "empty", full)
    assert eng.hint_on_device("b", hint) is None
    assert eng.cache.budget == eng.cache.used == hint.nbytes
    assert eng.hint_on_device("a", hint) is not None  # resident hints stay
    assert eng.stats["hint_oom"] == 1
