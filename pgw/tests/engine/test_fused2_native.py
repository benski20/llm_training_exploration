"""The native fused2 decode driver (``pgw_fast.Fused2Driver``, :mod:`pgw.engine.fused2_native`; opt-in).

Against the Python fused2 path on the same native float ops and the same lane (a real ``LaneManager`` serving the
Triton pair op on CPU): logits bit for bit (and == the plain reference), byte-identical provider-visible payloads
for the same mask seed, one mask slot per decoded token with segment masks (weight-free gateway), tampering of
``H``, a consistent ``H``, ``Z``, ``Z`` without the residual and skipped multipliers caught natively, the numpy
float64 roundings (pairwise sums, quantization) reproduced exactly, and a sampled trip ends the driver call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import IntegrityError
from pgw.engine import InMemoryPack, SequenceInput, TransformerModel
from pgw.engine.fused2 import (
    Fused2MaskedBackend,
    Fused2MaskSource,
    Fused2PlainBackend,
    Fused2Weights,
    enable_fused2,
    quantize_pair,
    quantize_single,
)
from pgw.engine.fused2_masks import Fused2MaskFactory, Fused2SegmentMasks, gateway_tables
from pgw.engine.fused2_native import Fused2NativeDecode, enable_fused2_native
from pgw.service.fused2_lane import Fused2LaneEntry
from tests.engine.test_fused2_lane import RING, _Lane, sock_dir  # noqa: F401  (fixture)

fast = pytest.importorskip("pgw_fast")
if not hasattr(fast, "Fused2Driver"):
    pytest.skip("pgw_fast without Fused2Driver (rebuild: rust/tools/build_fast.py)", allow_module_level=True)


@pytest.fixture(scope="module")
def weights(qwen_pack: InMemoryPack) -> Fused2Weights:
    from pgw.engine.fused2 import migration_multipliers

    return Fused2Weights(qwen_pack, multipliers=migration_multipliers(qwen_pack, 8))


@pytest.fixture(scope="module")
def entry(weights: Fused2Weights, tmp_path_factory: pytest.TempPathFactory) -> Fused2LaneEntry:
    p = tmp_path_factory.mktemp("f2n") / "prov.pt"
    weights.save(p, provider_only=True)
    return Fused2LaneEntry.from_weights(Fused2Weights.load(p), RING, device="cpu", max_rows=8)


def _model(pack: InMemoryPack, backend: Any, w: Fused2Weights, native: bool = False) -> TransformerModel:
    m = TransformerModel(pack, backend)
    m.set_float_ops("native")
    enable_fused2(m, w, RING)
    if native:
        enable_fused2_native(m)
    return m


def _gen(model: TransformerModel, steps: int = 4, prompt_len: int = 5, seed: int = 7) -> list[torch.Tensor]:
    """A prompt of ``prompt_len`` rows then ``steps`` greedy decode passes (one sequence)."""
    g = torch.Generator().manual_seed(seed)
    v = model.vocab_size
    cache = model.new_cache(64)
    out = model.forward_batch([SequenceInput(torch.randint(0, v, (prompt_len,), generator=g), cache,
                                             all_logits=True)])
    for _ in range(steps):
        tok = torch.tensor([int(out[-1].reshape(-1, v)[-1].argmax())])
        out += model.forward_batch([SequenceInput(tok, cache)])
    return out


def test_pairwise_sum_and_quantization_match_numpy(weights: Fused2Weights) -> None:
    rng = np.random.default_rng(0)
    for n in (1, 5, 8, 9, 64, 127, 128, 129, 300, 896, 3584, 18944):
        for _ in range(5):
            y = rng.standard_normal(n) * 10.0 ** float(rng.integers(-6, 15))
            y = y * y
            assert fast.f2_pairwise_sum(y) == y.sum()
    l1, l2 = weights.o[0], weights.gu[0]
    for _ in range(30):
        a = rng.standard_normal((1, l1.n)).astype(np.float32) * float(10.0 ** rng.integers(-2, 3))
        h = rng.standard_normal((1, l1.m)) * float(10.0 ** rng.integers(-2, 4))
        aq, rq, s, _bits = quantize_pair(torch.from_numpy(a), h, l1, l2, RING)
        got = fast.f2_quantize(a[0].astype(np.float64), h[0], np.ascontiguousarray(l1.grid.numpy()), float(l1.wnorm),
                               float(l1.sigma), float(l2.wnorm), RING.a, RING.cap)
        assert got is not None
        assert got[0] == s[0] and np.array_equal(got[1], aq[0].numpy()) and np.array_equal(got[2], rq[0].numpy())
        xq, s1 = quantize_single(torch.from_numpy(h[:, : weights.qkv[0].n]), weights.qkv[0], RING)
        g1 = fast.f2_quantize(h[0, : weights.qkv[0].n], None, None, float(weights.qkv[0].wnorm), 0.0, None, RING.a,
                              RING.cap)
        assert g1 is not None and g1[0] == s1[0] and np.array_equal(g1[1], xq[0].numpy())


@pytest.mark.parametrize("shm", [False, True], ids=["uds", "shm"])
def test_native_equals_python_fused2_bit_for_bit(qwen_pack: InMemoryPack, weights: Fused2Weights,
                                                 entry: Fused2LaneEntry, sock_dir: Path, shm: bool) -> None:  # noqa: F811
    ref = _gen(_model(qwen_pack, Fused2PlainBackend(RING), weights))
    outs, bodies = [], []
    for native in (False, True):
        d = sock_dir / ("n" if native else "p")
        d.mkdir()
        lane = _Lane(entry, d, shm=shm)
        try:
            prov = lane.provider(weights)
            seed = bytes(range(32))
            be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING, seed=seed))
            model = _model(qwen_pack, be, weights, native=native)
            outs.append(_gen(model))
            bodies.append(list(lane.bodies))  # (item, rows, body) of every lane call, prompt included
            if native:
                nd = model.schedule.native_decode
                assert isinstance(nd, Fused2NativeDecode)
                L = weights.num_layers
                # every pass native (the 5-row prompt too: <= the lane's 8 rows)
                assert nd.totals[0] == 5 and nd.totals[2] == 5 * (2 * L + 1)
                assert be.f2.checks == 5 * 4 * L  # both stages of each of the 2L - 1 pairs, one per single
            prov.close()
        finally:
            lane.close()
    for a, b, c in zip(ref, outs[0], outs[1], strict=True):
        assert torch.equal(a, b) and torch.equal(a, c)
    assert bodies[0] == bodies[1] and len(bodies[0]) == 5 * (2 * weights.num_layers + 1)


def test_weight_free_gateway_native_one_slot_per_token(qwen_pack: InMemoryPack, weights: Fused2Weights,
                                                       entry: Fused2LaneEntry, sock_dir: Path) -> None:  # noqa: F811
    ref_model = _model(qwen_pack, Fused2PlainBackend(RING), weights)
    gw_w = gateway_tables(weights)
    fp = entry.fingerprint
    fac = Fused2MaskFactory(weights, RING, lane_fingerprint=fp)
    masks = Fused2SegmentMasks(gw_w, RING, fac.check_blob(), lane_fingerprint=fp)
    masks.add_segment(fac.segment(16))
    lane = _Lane(entry, sock_dir, shm=True)
    try:
        prov = lane.provider(gw_w)
        be = Fused2MaskedBackend(RING, prov, masks)
        model = _model(qwen_pack, be, gw_w, native=True)
        v = model.vocab_size
        prompt = torch.randint(0, v, (6,), generator=torch.Generator().manual_seed(1))
        c_ref, c = ref_model.new_cache(32), model.new_cache(32)
        a = ref_model.forward_batch([SequenceInput(prompt, c_ref)])[0]
        b = model.forward_batch([SequenceInput(prompt, c)])[0]
        assert torch.equal(a, b)
        used = masks.slots_used()
        assert used == 6
        for _ in range(4):
            tok = torch.tensor([int(a.reshape(-1, v)[-1].argmax())])
            a = ref_model.forward_batch([SequenceInput(tok, c_ref)])[0]
            b = model.forward_batch([SequenceInput(tok, c)])[0]
            assert torch.equal(a, b)
            assert masks.slots_used() == used + 1  # one mask slot per token
            used += 1
        assert model.schedule.native_decode.totals[0] == 5  # the 6-row prompt too (<= the lane's 8 rows)
        prov.close()
    finally:
        lane.close()


@pytest.mark.parametrize("what", ["H", "H_consistent", "Z", "Z_skip_residual", "no_hmul"])
def test_native_tamper_detected(qwen_pack: InMemoryPack, weights: Fused2Weights, entry: Fused2LaneEntry,
                                sock_dir: Path, what: str) -> None:  # noqa: F811
    lane = _Lane(entry, sock_dir, shm=False, evil=what)
    try:
        prov = lane.provider(weights)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING))
        model = _model(qwen_pack, be, weights, native=True)
        with pytest.raises(IntegrityError):
            _gen(model, steps=1, prompt_len=3)
        assert model.schedule.native_decode.totals[1] >= 1  # it was the native driver that refused
        prov.close()
    finally:
        lane.close()


def test_sampled_trip_ends_the_call(qwen_pack: InMemoryPack, weights: Fused2Weights, entry: Fused2LaneEntry,
                                    sock_dir: Path) -> None:  # noqa: F811
    """A sampled trip's statistics run before the next payload is built (the call ends there)."""
    lane = _Lane(entry, sock_dir, shm=False)
    events: list[Any] = []

    class Monitor:
        n = 0

        def plan_call(self) -> bool:
            self.n += 1
            return self.n % 3 == 0

        def inspect_planned(self, xh: torch.Tensor, xq: torch.Tensor) -> None:
            events.append(("stats", len(lane.bodies), tuple(xh.shape)))
            assert int(xq.abs().max()) <= 1 << RING.cap

    try:
        prov = lane.provider(weights)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING))
        be.noise_monitor = Monitor()  # type: ignore[attr-defined]
        model = _model(qwen_pack, be, weights, native=True)
        _gen(model, steps=0, prompt_len=2)
        L = weights.num_layers
        trips = 2 * L + 1
        assert [e[1] for e in events] == [3 * (i + 1) for i in range(trips // 3)]  # right after the sampled trip
        assert model.schedule.native_decode.totals[1] == trips // 3 + (1 if trips % 3 else 0)
        prov.close()
    finally:
        lane.close()


def test_native_over_the_planes_pair_op(qwen_pack: InMemoryPack, weights: Fused2Weights, sock_dir: Path,  # noqa: F811
                                        tmp_path: Path) -> None:
    """The service on the digit-plane tensor-core op (``PGW_SERVICE_FUSED2_VARIANT=planes``): same logits."""
    p = tmp_path / "prov.pt"
    weights.save(p, provider_only=True)
    entry = Fused2LaneEntry.from_weights(Fused2Weights.load(p), RING, device="cpu", max_rows=8, variant="planes")
    assert all(op.config(1)[0].variant == "planes" for op in entry.ops)
    ref = _gen(_model(qwen_pack, Fused2PlainBackend(RING), weights), steps=2)
    lane = _Lane(entry, sock_dir, shm=True)
    try:
        prov = lane.provider(weights)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING))
        got = _gen(_model(qwen_pack, be, weights, native=True), steps=2)
        for a, b in zip(ref, got, strict=True):
            assert torch.equal(a, b)
        prov.close()
    finally:
        lane.close()
