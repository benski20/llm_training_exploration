"""Encrypted mask generation inside the real factory: bit-identical to local generation, tamper-safe."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import IntegrityError
from pgw.core.ring import RingParams
from pgw.factory import MaskGenerator
from pgw.hemask import matmul_mod64
from pgw.hemask.backend import HEMaskBackend, HintStore

SEED = bytes(range(1, 33))


class FakeProvider:
    """Stands in for the compute service: W8 b mod 2^64 (optionally corrupting one element)."""

    def __init__(self, pack: Any, tamper: bool = False):
        self.w = {lw.id.key: lw.w8.numpy() for lw in pack.iter_linears()}
        self.tamper = tamper
        self.calls = 0
        self.seen: list[np.ndarray] = []

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.seen.append(b.copy())
        y = matmul_mod64(self.w[linear], b)
        if self.tamper and self.calls == 3:
            y[1, 2] += np.uint64(1 << 30)
        return y


def _collect(gen: MaskGenerator, slots: int, chunk: int) -> dict[tuple[str, int], torch.Tensor]:
    return {(g.id.key, g.slot_start): g.wr.clone() for g in gen.generate(SEED, slots, checks=2, chunk_slots=chunk)}


@pytest.mark.parametrize("ring_name,q_bits", [("K40A12", 64), ("K32A12C27", 56)])
def test_he_factory_matches_local_factory(tiny_pack: Any, tmp_path: Path, ring_name: str, q_bits: int) -> None:
    ring = RingParams.from_name(ring_name)
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=q_bits)
    assert HintStore.open(tmp_path / "hints", tiny_pack.fingerprint, 8, q_bits) is not None  # reused, not rebuilt
    assert hints.hint(next(iter(hints.meta["row_norms"]))).shape[1] == q_bits // 8 - 1
    provider = FakeProvider(tiny_pack)
    remote = HEMaskBackend(tiny_pack, ring, hints, provider, groups_per_call=2)
    assert remote.params.q == q_bits
    local = _collect(MaskGenerator(tiny_pack, ring, device="cpu"), 300, 300)
    he_gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    assert he_gen.self_check_rows == -1  # every row is verified in encrypted mode
    he = _collect(he_gen, 300, 300)
    assert local.keys() == he.keys()
    for k in local:
        assert torch.equal(local[k], he[k]), k
    assert he_gen.last_stats.self_checked_rows == 300 * len(list(tiny_pack.linear_specs()))
    # the provider only ever saw uniform-looking q-bit words
    top = np.concatenate([(b >> np.uint64(q_bits - 1)).ravel() for b in provider.seen]).astype(np.float64)
    assert all(int(b.max()) < (1 << q_bits) for b in provider.seen)
    assert abs(top.mean() - 0.5) < 0.05


def test_he_factory_rejects_tampered_provider(tiny_pack: Any, tmp_path: Path) -> None:
    ring = RingParams.from_name("K40A12")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8)
    remote = HEMaskBackend(tiny_pack, ring, hints, FakeProvider(tiny_pack, tamper=True))
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    with pytest.raises(IntegrityError):
        _collect(gen, 256, 256)


def test_hint_seed_is_fresh_per_build(tiny_pack: Any, tmp_path: Path) -> None:
    a = HintStore.build(tmp_path / "a", tiny_pack, rank=8)
    b = HintStore.build(tmp_path / "b", tiny_pack, rank=8)
    assert a.a_seed != b.a_seed and len(a.a_seed) == 32


class SlowProvider(FakeProvider):
    """A provider with latency; records concurrency and the requests it saw (thread-safe)."""

    def __init__(self, pack: Any, delay_s: float = 0.02, fail_at: int = 0):
        super().__init__(pack)
        self.delay_s = delay_s
        self.fail_at = fail_at
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.order: list[tuple[str, tuple[int, ...]]] = []

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.order.append((linear, tuple(b.shape)))
            n = len(self.order)
        try:
            time.sleep(self.delay_s)
            if self.fail_at and n == self.fail_at:
                raise RuntimeError("provider down")
            return matmul_mod64(self.w[linear], b)
        finally:
            with self.lock:
                self.active -= 1


def _n_linears(pack: Any) -> int:
    return len(list(pack.linear_specs()))


@pytest.mark.parametrize("depth,inflight,chunk", [(2, 1, 256), (3, 2, 512), (4, 4, 768), (8, 2, 300)])
def test_he_pipeline_depth_is_bit_identical(tiny_pack: Any, tmp_path: Path, depth: int, inflight: int,
                                            chunk: int) -> None:
    """Overlapped stages (jobs across linears) give the same masks and the same requests (sizes, counts,
    order) as the serial path, and every row is still verified."""
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    serial_p = SlowProvider(tiny_pack, delay_s=0.0)
    serial = HEMaskBackend(tiny_pack, ring, hints, serial_p, groups_per_call=2, max_inflight=inflight)
    over_p = SlowProvider(tiny_pack, delay_s=0.01)
    over = HEMaskBackend(tiny_pack, ring, hints, over_p, groups_per_call=2, max_inflight=inflight,
                         pipeline_depth=depth)
    local = _collect(MaskGenerator(tiny_pack, ring, device="cpu"), 1000, chunk)
    gen_s = MaskGenerator(tiny_pack, ring, device="cpu", remote=serial)
    gen_o = MaskGenerator(tiny_pack, ring, device="cpu", remote=over)
    assert gen_s.he_pipeline_depth() == 0 and gen_o.he_pipeline_depth() == depth
    a = _collect(gen_s, 1000, chunk)
    b = _collect(gen_o, 1000, chunk)
    assert list(a) == list(b) == list(local)  # canonical hand-on order
    for k in local:
        assert torch.equal(local[k], a[k]) and torch.equal(local[k], b[k]), k
    nl = _n_linears(tiny_pack)
    assert gen_o.last_stats.self_checked_rows == 1000 * nl
    # the provider sees the same requests (in the same order with one request thread; with several, threads
    # dequeued in order may enter the provider in either order, in both modes)
    if inflight == 1:
        assert serial_p.order == over_p.order
    else:
        assert sorted(serial_p.order) == sorted(over_p.order)
    assert over_p.max_active <= inflight
    st = gen_o.last_stats.extra["he"]
    assert st["jobs"] == nl * -(-1000 // chunk)
    assert st["groups"] == nl * sum(-(-min(chunk, 1000 - s) // 256) for s in range(0, 1000, chunk))
    assert st["wall_s"] > 0


def test_he_pipeline_overlaps_stages(tiny_pack: Any, tmp_path: Path) -> None:
    """With a slow provider and depth >= 2, several linears' requests are with the provider at once (the
    serial path sends one request per linear here, so it never has two)."""
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    prov = SlowProvider(tiny_pack, delay_s=0.2)
    be = HEMaskBackend(tiny_pack, ring, hints, prov, groups_per_call=4, max_inflight=4, pipeline_depth=4)
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=be)
    _collect(gen, 256, 256)
    assert prov.max_active >= 2
    st = gen.last_stats.extra["he"]
    assert st["remote_s"] > st["wall_s"]  # request time summed over threads exceeds the busy wall time


@pytest.mark.parametrize("fail_at", [1, 3, 7])
def test_he_pipeline_failure_aborts_and_wipes(tiny_pack: Any, tmp_path: Path, fail_at: int) -> None:
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    prov = SlowProvider(tiny_pack, delay_s=0.005, fail_at=fail_at)
    be = HEMaskBackend(tiny_pack, ring, hints, prov, groups_per_call=1, max_inflight=2, pipeline_depth=3)
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=be)
    with pytest.raises(RuntimeError, match="provider down"):
        for _ in gen.generate(SEED, 512, checks=2, chunk_slots=512):
            pass
    assert be._busy == 0  # no job is left running in the stage threads
    prov.fail_at = 0  # the backend still works after a failed segment
    out = _collect(gen, 256, 256)
    local = _collect(MaskGenerator(tiny_pack, ring, device="cpu"), 256, 256)
    assert all(torch.equal(out[k], local[k]) for k in local)


def test_he_pipeline_tampered_provider_still_rejected(tiny_pack: Any, tmp_path: Path) -> None:
    ring = RingParams.from_name("K40A12")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8)
    remote = HEMaskBackend(tiny_pack, ring, hints, FakeProvider(tiny_pack, tamper=True), pipeline_depth=3)
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=remote)
    with pytest.raises(IntegrityError):
        _collect(gen, 256, 256)
    assert remote._busy == 0


def test_he_pipeline_abandoned_generator_cancels_jobs(tiny_pack: Any, tmp_path: Path) -> None:
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    prov = SlowProvider(tiny_pack, delay_s=0.02)
    be = HEMaskBackend(tiny_pack, ring, hints, prov, groups_per_call=1, max_inflight=2, pipeline_depth=4)
    gen = MaskGenerator(tiny_pack, ring, device="cpu", remote=be)
    it = gen.generate(SEED, 512, checks=2, chunk_slots=256)
    first = next(it)
    assert first.wr.shape[0] == 256
    it.close()  # abandoned with jobs outstanding: they are cancelled and waited for
    assert be._busy == 0


def test_he_pipeline_bytes_budget_limits_outstanding(tiny_pack: Any, tmp_path: Path) -> None:
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.open_or_build(tmp_path / "hints", tiny_pack, rank=8, q_bits=56)
    be = HEMaskBackend(tiny_pack, ring, hints, SlowProvider(tiny_pack, delay_s=0.0), pipeline_depth=8,
                       pipeline_bytes=1)  # below one job: one job at a time
    busy_at_submit: list[int] = []
    orig = be.submit

    def submit(linear: str, r: torch.Tensor) -> Any:
        busy_at_submit.append(be._busy)
        return orig(linear, r)

    be.submit = submit  # type: ignore[method-assign]
    out = _collect(MaskGenerator(tiny_pack, ring, device="cpu", remote=be), 256, 256)
    assert busy_at_submit and max(busy_at_submit) == 0
    local = _collect(MaskGenerator(tiny_pack, ring, device="cpu"), 256, 256)
    assert all(torch.equal(out[k], local[k]) for k in local)
