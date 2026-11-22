"""fused2 (WP-B4 prototype): two provider round trips per layer, exact under its own fixed-point reference.

Masked == plain bit for bit, 2L + 1 trips per pass, the provider only receives masked ring elements, and
every provider output (and factory correction) that is used is checked: tampering raises.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import ConfigError, IntegrityError
from pgw.core.ring import RingParams
from pgw.engine import InMemoryPack, PlainBackend, SequenceInput, TransformerModel
from pgw.engine.backends import FloatBackend
from pgw.engine.fused2 import (
    P61,
    Fused2MaskedBackend,
    Fused2MaskSource,
    Fused2PlainBackend,
    Fused2Provider,
    Fused2Weights,
    dot_mod_p,
    enable_fused2,
    matmul_mod_p,
    soundness_bits,
)

RINGS = [RingParams(48, 12, 45), RingParams(40, 12, 37)]


@pytest.fixture(scope="module", params=["qwen", "llama"])
def pack(request: pytest.FixtureRequest, qwen_pack: InMemoryPack, llama_pack: InMemoryPack) -> InMemoryPack:
    return qwen_pack if request.param == "qwen" else llama_pack


@pytest.fixture(scope="module")
def weights(pack: InMemoryPack) -> Fused2Weights:
    return Fused2Weights(pack)


def _model(pack: InMemoryPack, backend: Any, w: Fused2Weights, ring: RingParams) -> TransformerModel:
    m = TransformerModel(pack, backend)
    enable_fused2(m, w, ring)
    return m


def _run(model: TransformerModel) -> list[torch.Tensor]:
    """Prefill two sequences in lockstep, then three decode steps each; all logits."""
    g = torch.Generator().manual_seed(7)
    v = model.vocab_size
    caches = [model.new_cache(64), model.new_cache(64)]
    prompts = [torch.randint(0, v, (5,), generator=g), torch.randint(0, v, (3,), generator=g)]
    out = model.forward_batch([SequenceInput(p, c, all_logits=True) for p, c in zip(prompts, caches, strict=True)])
    for _ in range(3):
        nxt = [torch.tensor([int(o.reshape(-1, v)[-1].argmax())]) for o in out[-2:]]
        out += model.forward_batch([SequenceInput(t, c) for t, c in zip(nxt, caches, strict=True)])
    return out


@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
def test_masked_equals_plain_bit_identical(pack: InMemoryPack, weights: Fused2Weights, ring: RingParams) -> None:
    ref = _run(_model(pack, Fused2PlainBackend(ring), weights, ring))
    provider = Fused2Provider(weights, ring.k)
    provider.record = True
    src = Fused2MaskSource(weights, ring)
    be = Fused2MaskedBackend(ring, provider, src)
    got = _run(_model(pack, be, weights, ring))
    assert len(got) == len(ref)
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    layers = pack.arch.num_layers
    # 1 prefill pass + 3 decode passes, two sequences each, 2L + 1 trips per pass per sequence
    assert be.f2.trips == 4 * 2 * (2 * layers + 1)
    assert be.f2.checks == 4 * 2 * (1 + 2 * (2 * layers - 1) + 1)  # pair trips check both stages
    # privacy: every array the provider saw is canonical ring elements, never the plaintext integers
    for _key, arr in provider.seen:
        assert int(arr.min()) >= 0 and int(arr.max()) < (1 << ring.k)
    big = [arr for _k, arr in provider.seen if arr.numel() >= 32]
    assert big and all(int((arr >> (ring.k - 4)).max()) > 0 for arr in big)  # high bits set: masked, not small


def test_trip_plan_and_reference_close(pack: InMemoryPack, weights: Fused2Weights) -> None:
    ring = RINGS[0]
    m = _model(pack, Fused2PlainBackend(ring), weights, ring)
    assert m._trip_plan() == (2 * pack.arch.num_layers + 1, False)
    std = TransformerModel(pack, PlainBackend(pack, RingParams(32, 12, 27)))
    assert std._trip_plan() == (4 * pack.arch.num_layers, False)
    flt = TransformerModel(pack, FloatBackend(pack, ring))
    ids = torch.arange(1, 9)
    a = m.forward(ids, m.new_cache(16), all_logits=True)
    b = flt.forward(ids, flt.new_cache(16), all_logits=True)
    c = std.forward(ids, std.new_cache(16), all_logits=True)
    # a different reference (folded weights are rounded again here): close to the float product, not equal
    err_f2, err_std = float((a - b).abs().max()), float((c - b).abs().max())
    assert 0 < err_f2 < 0.02 * float(b.abs().max())
    assert err_std < 0.02 * float(b.abs().max())
    sched = m.schedule
    assert sum(sched.bits_hist.values()) > 0  # type: ignore[union-attr]


def test_bounds_hold(pack: InMemoryPack, weights: Fused2Weights) -> None:
    ring = RingParams(40, 12, 37)
    be = Fused2PlainBackend(ring)
    _run(_model(pack, be, weights, ring))
    assert be.max_bits and max(be.max_bits.values()) <= ring.cap


def test_tight_cap_lowers_bits_and_stays_exact(pack: InMemoryPack, weights: Fused2Weights) -> None:
    from pgw.core.errors import RingOverflowError

    # the smallest cap the residual allows (at 2 activation bits the residual alone must fit): the gateway
    # then lowers activation bits per row; a cap below it is refused, never wrapped
    plain = ref = ring = None
    for cap in range(20, 30):
        ring = RingParams(32, 12, cap)
        plain = _model(pack, Fused2PlainBackend(ring), weights, ring)
        try:
            ref = _run(plain)
            break
        except RingOverflowError:
            continue
    assert ref is not None and ring is not None and plain is not None and ring.cap > 20
    assert min(plain.schedule.bits_hist) < 12  # type: ignore[union-attr]
    be = Fused2MaskedBackend(ring, Fused2Provider(weights, ring.k), Fused2MaskSource(weights, ring))
    got = _run(_model(pack, be, weights, ring))
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)


class _Evil(Fused2Provider):
    def __init__(self, weights: Fused2Weights, k: int, what: str, when: int = 0, delta: int = 1):
        super().__init__(weights, k)
        self.what, self.when, self.delta, self.n = what, when, delta, 0

    def _bump(self, t: torch.Tensor, idx: int = 0) -> torch.Tensor:
        t = t.clone()
        t.view(-1)[idx % t.numel()] += self.delta
        return t & ((1 << self.k) - 1)

    def single(self, k1: str, x_hat: torch.Tensor) -> torch.Tensor:
        y = super().single(k1, x_hat)
        self.n += 1
        return self._bump(y, 3) if self.what == "single" and self.n > self.when else y

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, z = super().pair(k1, k2, r_hat, x_hat)
        self.n += 1
        if self.n <= self.when:
            return h, z
        if self.what == "H":
            return self._bump(h, 5), z
        if self.what == "H_consistent":  # wrong H, Z recomputed from it: stage 1 must catch it
            h2 = self._bump(h, 5)
            return h2, self._w[k2].matmul(h2, self.k)
        if self.what == "Z":
            return h, self._bump(z, 11)
        if self.what == "Z_skip_residual":  # Z of W1 x only (residual dropped): a plausible shortcut
            return h, self._w[k2].matmul((h - r_hat) & ((1 << self.k) - 1), self.k)
        return h, z


@pytest.mark.parametrize("what", ["single", "H", "H_consistent", "Z", "Z_skip_residual"])
@pytest.mark.parametrize("delta", [1, 1 << 30, -(1 << 20)], ids=["one", "big", "neg"])
def test_tamper_detected(pack: InMemoryPack, weights: Fused2Weights, what: str, delta: int) -> None:
    ring = RINGS[0]
    be = Fused2MaskedBackend(ring, _Evil(weights, ring.k, what, when=2, delta=delta), Fused2MaskSource(weights, ring))
    m = _model(pack, be, weights, ring)
    with pytest.raises(IntegrityError):
        _run(m)


def test_bad_factory_correction_detected(pack: InMemoryPack, weights: Fused2Weights) -> None:
    ring = RINGS[0]

    class BadSource(Fused2MaskSource):
        def take(self, trip: Any) -> Any:
            m = super().take(trip)
            if m.wv is not None:
                m.wv.view(-1)[0] += 1  # a wrong W2' v row from the factory moves Z: caught on the unmasked Z
            return m

    be = Fused2MaskedBackend(ring, Fused2Provider(weights, ring.k), BadSource(weights, ring))
    with pytest.raises(IntegrityError):
        _run(_model(pack, be, weights, ring))


def test_masks_single_use(weights: Fused2Weights) -> None:
    ring = RINGS[0]
    src = Fused2MaskSource(weights, ring)
    from pgw.engine.fused2 import Fused2Trip

    trip = Fused2Trip("pair", weights.o[0], torch.zeros((2, weights.o[0].n), dtype=torch.int64), ring.cap,
                      weights.gu[0], torch.zeros((2, weights.o[0].m), dtype=torch.int64))
    a, b = src.take(trip), src.take(trip)
    assert not torch.equal(a.r_x, b.r_x) and not torch.equal(a.v, b.v)
    # corrections are two independent single-stage products (no chain): W1 r and W2' v
    from pgw.core.ringmm import ringmm

    assert torch.equal(a.wr_x, ringmm(a.r_x, weights.o[0].w8, ring.k))
    assert torch.equal(a.wv, ringmm(a.v, weights.gu[0].w8, ring.k))


def test_field_arithmetic_exact() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(-(1 << 53), 1 << 53, size=(3, 257), dtype=np.int64)
    u = rng.integers(0, P61, size=(2, 257), dtype=np.int64)
    exp = np.array([[sum(int(a) * int(b) for a, b in zip(yr, ur, strict=True)) % P61 for ur in u] for yr in y])
    assert np.array_equal(dot_mod_p(y, u), exp.astype(np.int64))
    w8 = torch.randint(-127, 128, (257, 19), dtype=torch.int8)
    exp2 = np.array([[sum(int(u[c, i]) * int(w8[i, j]) for i in range(257)) % P61 for j in range(19)] for c in range(2)])
    assert np.array_equal(matmul_mod_p(u, w8), exp2.astype(np.int64))
    assert soundness_bits(1) > 60.8 and math.isclose(soundness_bits(2), 2 * soundness_bits(1))


def test_defaults_off(pack: InMemoryPack) -> None:
    m = TransformerModel(pack, PlainBackend(pack, RingParams(32, 12, 27)))
    assert m.schedule is None
    with pytest.raises(ConfigError):
        enable_fused2(m, Fused2Weights(pack), RINGS[0])  # the backend does not take fused2 calls
    with pytest.raises(ConfigError):
        Fused2MaskedBackend(RingParams(48, 12), None, None)  # type: ignore[arg-type]  # uncapped ring refused
    from pgw.gateway.config import PENDING_SPEED_SETTINGS

    assert "schedule" in PENDING_SPEED_SETTINGS  # the gateway still refuses schedule="fused2"


def test_fold_from_float_weights(pack: InMemoryPack, tiny_hf_dir: Any) -> None:
    from safetensors.torch import load_file

    from pgw.engine.fused2 import hf_float_linears

    sd = {}
    for f in sorted(tiny_hf_dir.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    if "model.layers.0.self_attn.q_proj.bias" not in sd and pack.arch.arch == "qwen2":  # pragma: no cover
        pytest.skip("not the tiny qwen checkpoint")
    if pack.arch.arch != "qwen2":
        pytest.skip("the float fixture is the tiny qwen checkpoint")
    w = Fused2Weights(pack, hf_float_linears(sd, pack.arch))
    assert w.folded_from == "float"
    ring = RINGS[0]
    ref = _run(_model(pack, Fused2PlainBackend(ring), w, ring))
    got = _run(_model(pack, Fused2MaskedBackend(ring, Fused2Provider(w, ring.k), Fused2MaskSource(w, ring)), w, ring))
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
