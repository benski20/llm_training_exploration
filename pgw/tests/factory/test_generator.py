"""MaskGenerator: normative derivation, exactness, chunking, pipelining, store writing and failure handling."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    ConfigError,
    RingOverflowError,
    RingParams,
    SeedExpander,
    center,
    freivalds_verify,
    mask_activations,
    quantize_activations,
    ringmm_reference,
)
from pgw.factory import FactoryStopped, GeneratedLinear, MaskGenerator, Throttle, WeightStreamer, slot_bytes
from pgw.factory.kernels import cpu_int8_available

from .conftest import FakeMaskStore, FakePack, reference_segment

SEED = bytes(range(32))


def _collect(gen: MaskGenerator, seed: bytes, slots: int, checks: int = 2, chunk: int = 4) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in gen.generate(seed, slots, checks=checks, chunk_slots=chunk):
        rec = out.setdefault(item.id.key, {"wr": [], "uw": None, "starts": []})
        rec["wr"].append(item.wr.clone())
        rec["starts"].append(item.slot_start)
        if item.uw is not None:
            assert rec["uw"] is None, "uw must only come with the first chunk"
            rec["uw"] = item.uw
    return out


def test_generate_matches_normative_derivation(fake_pack: FakePack, ring: RingParams) -> None:
    gen = MaskGenerator(fake_pack, ring, device="cpu")
    slots = 11
    got = _collect(gen, SEED, slots, checks=2, chunk=4)
    ref = reference_segment(fake_pack, ring, SEED, slots, 2)
    assert list(got) == [s.id.key for s in fake_pack.linear_specs()]  # canonical, linear-major
    for key, (wr_ref, uw_ref) in ref.items():
        assert got[key]["starts"] == [0, 4, 8]
        assert [t.shape[0] for t in got[key]["wr"]] == [4, 4, 3]
        assert torch.cat(got[key]["wr"]).tolist() == wr_ref.tolist()
        assert got[key]["uw"].tolist() == uw_ref.tolist()


@pytest.mark.parametrize("backend", ["cpu_f64", "cpu_f32"] + (["cpu_int8"] if cpu_int8_available() else []))
def test_backends_and_pipeline_agree(fake_pack: FakePack, backend: str) -> None:
    a = _collect(MaskGenerator(fake_pack, RING_K40_A12, backend=backend, pipeline=True), SEED, 9, chunk=5)
    b = _collect(MaskGenerator(fake_pack, RING_K40_A12, backend="cpu_f64", pipeline=False), SEED, 9, chunk=2)
    for key in a:
        assert torch.equal(torch.cat(a[key]["wr"]), torch.cat(b[key]["wr"]))
        assert torch.equal(a[key]["uw"], b[key]["uw"])


def test_masks_unmask_exactly_and_pass_freivalds(fake_pack: FakePack) -> None:
    """End to end with the core masking math: gateway subtracts W r and checks u."""
    ring = RING_K40_A12
    gen = MaskGenerator(fake_pack, ring)
    slots = 6
    got = _collect(gen, SEED, slots, chunk=6)
    exp = SeedExpander(SEED)
    for spec in fake_pack.linear_specs():
        key = spec.id.key
        w = fake_pack.weights[key]
        x = torch.randn(slots, spec.in_features)
        xq, _ = quantize_activations(x, ring)
        r = exp.ring_uniform_rows(spec.in_features, ring.k, 0, slots, "r", key)
        x_hat = mask_activations(xq, r, ring)
        y_hat = torch.from_numpy(ringmm_reference(x_hat, w, ring.k))  # honest service
        wr = got[key]["wr"][0]
        yq = center(y_hat - wr, ring)
        assert torch.equal(yq, xq @ w.to(torch.int64).t())
        u = exp.ring_uniform((2, spec.out_features), ring.k, "u", key)
        freivalds_verify(x_hat, y_hat, u, got[key]["uw"], ring, linear_id=key)
        bad = y_hat.clone()
        bad[0, 0] = (bad[0, 0] + 1) & ring.mask
        with pytest.raises(Exception, match="Freivalds"):
            freivalds_verify(x_hat, bad, u, got[key]["uw"], ring, linear_id=key)


def test_determinism_and_seed_separation(fake_pack: FakePack) -> None:
    gen = MaskGenerator(fake_pack, RING_K40_A12)
    a = _collect(gen, SEED, 3)
    b = _collect(gen, SEED, 3)
    c = _collect(gen, bytes(32), 3)
    key = "0.qkv"
    assert torch.equal(a[key]["wr"][0], b[key]["wr"][0])
    assert not torch.equal(a[key]["wr"][0], c[key]["wr"][0])


def test_checks_zero(fake_pack: FakePack) -> None:
    got = _collect(MaskGenerator(fake_pack, RING_K32_A8), SEED, 2, checks=0)
    assert got["1.down"]["uw"].shape == (0, 26)


def test_validation_errors(fake_pack: FakePack) -> None:
    gen = MaskGenerator(fake_pack, RING_K40_A12)
    for kwargs in ({"seed": b"short", "slots": 1}, {"seed": SEED, "slots": 0}, {"seed": SEED, "slots": 1, "checks": 9},
                   {"seed": SEED, "slots": 1, "chunk_slots": 0}):
        with pytest.raises(ConfigError):
            list(gen.generate(**kwargs))  # type: ignore[arg-type]


def test_ring_headroom_rejected() -> None:
    big = FakePack(dims={k: (8, 300_000) for k in ("qkv", "o", "gate_up", "down")}, layers=1)  # type: ignore[misc]
    with pytest.raises(RingOverflowError):
        MaskGenerator(big, RING_K32_A8)


def test_generate_into_store_commits_and_zeroizes(fake_pack: FakePack, fake_store: FakeMaskStore) -> None:
    gen = MaskGenerator(fake_pack, RING_K40_A12)
    seg_id = gen.generate_into_store(fake_store, 10, checks=2, chunk_slots=4, factory_id="t", seed=SEED)
    seg = fake_store.segments[seg_id]
    assert seg.seed == SEED and seg.factory_id == "t" and seg.slots == 10 and seg.chunk_slots == 4
    assert seg.fingerprint == fake_pack.fingerprint and seg.ring == RING_K40_A12
    ref = reference_segment(fake_pack, RING_K40_A12, SEED, 10, 2)
    for key, (wr_ref, uw_ref) in ref.items():
        assert torch.cat(seg.wr[key]).tolist() == wr_ref.tolist()
        assert seg.uw[key].tolist() == uw_ref.tolist()
    assert gen.tokens_per_second() > 0
    assert gen.last_stats.segment_id == seg_id
    assert gen.last_stats.wr_bytes == 10 * slot_bytes(fake_pack.linear_specs(), RING_K40_A12)


def test_generate_into_store_uses_fresh_seeds(fake_pack: FakePack, fake_store: FakeMaskStore) -> None:
    gen = MaskGenerator(fake_pack, RING_K40_A12)
    a = gen.generate_into_store(fake_store, 2)
    b = gen.generate_into_store(fake_store, 2)
    assert fake_store.segments[a].seed != fake_store.segments[b].seed
    assert len(fake_store.segments[a].seed) == 32


@pytest.mark.parametrize("pipeline", [True, False])
def test_writer_failure_aborts(fake_pack: FakePack, fake_store: FakeMaskStore, pipeline: bool) -> None:
    fake_store.fail_after = 3
    gen = MaskGenerator(fake_pack, RING_K40_A12, pipeline=pipeline)
    with pytest.raises(OSError, match="disk full"):
        gen.generate_into_store(fake_store, 8, chunk_slots=2)
    assert fake_store.aborted == 1 and not fake_store.segments
    assert not fake_store.writers[0].committed


def test_stop_aborts_segment(fake_pack: FakePack, fake_store: FakeMaskStore) -> None:
    stop = threading.Event()
    calls = {"n": 0}

    class StopAfter(Throttle):
        def account(self, work_seconds: float, slots: float) -> None:
            calls["n"] += 1
            if calls["n"] == 3:
                stop.set()
            super().account(work_seconds, slots)

    gen = MaskGenerator(fake_pack, RING_K40_A12, throttle=StopAfter(stop_event=stop))
    with pytest.raises(FactoryStopped):
        gen.generate_into_store(fake_store, 8, chunk_slots=2)
    assert fake_store.aborted == 1 and not fake_store.segments


def test_throttle_receives_slot_fractions(fake_pack: FakePack) -> None:
    seen: list[float] = []

    class Rec(Throttle):
        def account(self, work_seconds: float, slots: float) -> None:
            seen.append(slots)

    gen = MaskGenerator(fake_pack, RING_K40_A12, throttle=Rec())
    list(gen.generate(SEED, 6, chunk_slots=3))
    assert sum(seen) == pytest.approx(6.0)  # slot-equivalents add up to the segment size


def test_streamer_device_mismatch(fake_pack: FakePack) -> None:
    class S(WeightStreamer):
        pass

    s = S(fake_pack, "cpu", 0)
    s.device = torch.device("meta")
    with pytest.raises(ConfigError):
        MaskGenerator(fake_pack, RING_K40_A12, streamer=s)


def test_generated_linear_dataclass() -> None:
    from pgw.core import LinearId

    g = GeneratedLinear(LinearId.parse("0.o"), 0, torch.zeros((1, 2), dtype=torch.int64), None)
    assert g.uw is None and g.id.key == "0.o"


# ---- real model pack ---------------------------------------------------------------------------------
def test_real_tiny_pack(tiny_pack: Any, fake_store: FakeMaskStore) -> None:
    gen = MaskGenerator(tiny_pack, RING_K40_A12)
    assert gen.fingerprint == tiny_pack.fingerprint
    seg_id = gen.generate_into_store(fake_store, 5, chunk_slots=2, seed=SEED)
    seg = fake_store.segments[seg_id]
    ref = reference_segment(tiny_pack, RING_K40_A12, SEED, 5, 2)
    assert set(seg.wr) == set(ref)
    for key, (wr_ref, uw_ref) in ref.items():
        assert torch.equal(torch.cat(seg.wr[key]), wr_ref)
        assert torch.equal(seg.uw[key], uw_ref)


def test_real_tiny_pack_linears_only(tiny_pack_dir: Path) -> None:
    from pgw.modelpack import ModelPack

    pack = ModelPack.open(tiny_pack_dir, verify="full", components=("linears",))
    out = _collect(MaskGenerator(pack, "K48A20"), SEED, 3)
    assert len(out) == 4 * pack.num_layers
