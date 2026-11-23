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
    from pgw.gateway.config import GatewayModelConfig

    assert GatewayModelConfig.model_fields["schedule"].default == "standard"  # the gateway's fused2 is opt-in


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


# ---- split fold: gateway columns (the outlier columns of the folded weight stay off the provider) -----------
@pytest.fixture(scope="module")
def split_weights(pack: InMemoryPack) -> Fused2Weights:
    return Fused2Weights(pack, gateway_columns=3)


def test_split_fold_layout(pack: InMemoryPack, split_weights: Fused2Weights) -> None:
    w = split_weights
    for lin in w.qkv + w.gu:
        assert lin.gcols is not None and lin.gw is not None and lin.gscale is not None
        assert lin.gcols.numel() == 3 and torch.equal(lin.gcols, lin.gcols.sort().values)
        assert int(lin.w8[:, lin.gcols].abs().max()) == 0  # the provider's matrix skips those columns
        assert lin.gw.shape == (lin.m, 3) and int(lin.gw.abs().max()) <= (1 << 15) - 1
    for lin in w.o + w.down:
        assert lin.gcols is None  # only the folded linears are split
    cost = w.gateway_cost()
    per_layer = 3 * (w.qkv[0].m + w.gu[0].m)
    assert cost["columns"] == 3 * 2 * w.num_layers and cost["macs_per_token"] == per_layer * w.num_layers
    assert cost["weight_bytes"] == 2 * cost["macs_per_token"]  # int16 gateway weights
    # an explicit {key: columns} choice is taken as given
    w2 = Fused2Weights(pack, gateway_columns={"0.qkv/f2": [5, 1]})
    assert w2.qkv[0].gcols.tolist() == [1, 5]  # type: ignore[union-attr]
    assert w2.gu[0].gcols is None and w2.qkv[1].gcols is None


@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
def test_split_fold_masked_equals_plain(pack: InMemoryPack, split_weights: Fused2Weights, ring: RingParams) -> None:
    w = split_weights
    plain = _model(pack, Fused2PlainBackend(ring), w, ring)
    ref = _run(plain)
    provider = Fused2Provider(w, ring.k)
    provider.record = True
    be = Fused2MaskedBackend(ring, provider, Fused2MaskSource(w, ring))
    masked = _model(pack, be, w, ring)
    got = _run(masked)
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    layers = pack.arch.num_layers
    assert be.f2.trips == 4 * 2 * (2 * layers + 1)  # the same trips as the unsplit fold
    assert be.f2.checks == 4 * 2 * (1 + 2 * (2 * layers - 1) + 1)
    for _key, arr in provider.seen:
        assert int(arr.min()) >= 0 and int(arr.max()) < (1 << ring.k)
    big = [arr for _k, arr in provider.seen if arr.numel() >= 32]
    assert big and all(int((arr >> (ring.k - 4)).max()) > 0 for arr in big)
    assert masked.schedule.gateway_macs == plain.schedule.gateway_macs > 0  # type: ignore[union-attr]
    # the split reference is a different model from the unsplit one
    unsplit = _run(_model(pack, Fused2PlainBackend(ring), Fused2Weights(pack), ring))
    assert any(not torch.equal(a, b) for a, b in zip(ref, unsplit, strict=True))


def test_split_fold_same_bytes_and_factory_work(pack: InMemoryPack, weights: Fused2Weights,
                                                split_weights: Fused2Weights) -> None:
    ring = RINGS[1]
    stats = []
    for w in (weights, split_weights):
        be = Fused2MaskedBackend(ring, Fused2Provider(w, ring.k), Fused2MaskSource(w, ring))
        _run(_model(pack, be, w, ring))
        stats.append((be.f2.trips, be.f2.rows, be.f2.bytes_up, be.f2.bytes_down, be.masks.macs))
    assert stats[0] == stats[1]  # wire bytes per trip and factory products are unchanged by the split


class _EvilCol(Fused2Provider):
    """Moves ``H`` only on a gateway column (the gateway multiplies that column itself), with ``Z``
    recomputed consistently: stage 1 must catch it before the gateway uses it."""

    def __init__(self, weights: Fused2Weights, k: int, col: int, delta: int):
        super().__init__(weights, k)
        self.col, self.delta, self.n = col, delta, 0

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, z = super().pair(k1, k2, r_hat, x_hat)
        self.n += 1
        if self.n < 3:
            return h, z
        h = h.clone()
        h[:, self.col] += self.delta
        h &= (1 << self.k) - 1
        return h, self._w[k2].matmul(h, self.k)


@pytest.mark.parametrize("delta", [1, -(1 << 25)], ids=["one", "neg"])
def test_split_fold_tamper_on_gateway_column(pack: InMemoryPack, split_weights: Fused2Weights, delta: int) -> None:
    ring = RINGS[0]
    col = int(split_weights.gu[0].gcols[0])  # type: ignore[index]
    be = Fused2MaskedBackend(ring, _EvilCol(split_weights, ring.k, col, delta), Fused2MaskSource(split_weights, ring))
    with pytest.raises(IntegrityError):
        _run(_model(pack, be, split_weights, ring))


@pytest.mark.parametrize("what", ["single", "H", "H_consistent", "Z", "Z_skip_residual"])
def test_split_fold_tamper_detected(pack: InMemoryPack, split_weights: Fused2Weights, what: str) -> None:
    ring = RINGS[1]
    be = Fused2MaskedBackend(ring, _Evil(split_weights, ring.k, what, when=2, delta=1),
                             Fused2MaskSource(split_weights, ring))
    with pytest.raises(IntegrityError):
        _run(_model(pack, be, split_weights, ring))


def test_split_fold_int64_bound_refused(pack: InMemoryPack) -> None:
    w = Fused2Weights(pack, gateway_columns=3, gateway_bits=32)
    m = TransformerModel(pack, Fused2PlainBackend(RINGS[0]))
    with pytest.raises(ConfigError):
        enable_fused2(m, w, RINGS[0])  # 3 * (2^31 - 1) * 2^45 >= 2^63: refused, never wrapped
    with pytest.raises(ConfigError):
        Fused2Weights(pack, gateway_columns=1, gateway_bits=40)


# ---- migrated split fold: residual-grid multipliers + gateway columns (calibrate_fused2) -------------------
@pytest.fixture(scope="module")
def migrated(pack: InMemoryPack) -> Fused2Weights:
    from pgw.engine.fused2 import calibrate_fused2

    g = torch.Generator().manual_seed(3)
    calib = [torch.randint(0, pack.arch.vocab_size, (24,), generator=g) for _ in range(2)]
    cal = calibrate_fused2(pack, None, calib, RINGS[0], bits=4, columns=2, gateway_bits=12)
    return Fused2Weights(pack, gateway_columns=cal["gateway_columns"], multipliers=cal["multipliers"],
                         gateway_bits=cal["gateway_bits"])


def test_migrated_layout(pack: InMemoryPack, migrated: Fused2Weights) -> None:
    w = migrated
    assert w.migrated
    assert any(lin.hmul is not None and int(lin.hmul.max()) > 1 for lin in w.o + w.down)
    assert w.down[-1].hmul is None  # the last layer's down is a single trip
    for lin in w.o + w.down:
        if lin.hmul is not None:
            assert int(lin.hmul.min()) >= 1
            assert torch.allclose(lin.grid, lin.scale / lin.hmul.to(torch.float64))
    for lin in w.qkv + w.gu:
        assert lin.gcols is not None and lin.gcols.numel() >= 2
        assert int(lin.w8[:, lin.gcols].abs().max()) == 0
    with pytest.raises(ConfigError):
        Fused2Weights(pack, multipliers={f"{pack.arch.num_layers - 1}.down": torch.ones(pack.arch.hidden_size)})
    with pytest.raises(ConfigError):
        Fused2Weights(pack, multipliers={"0.o": torch.zeros(pack.arch.hidden_size)})  # multipliers are >= 1


@pytest.mark.parametrize("ring", RINGS, ids=lambda r: r.name)
def test_migrated_masked_equals_plain(pack: InMemoryPack, weights: Fused2Weights, migrated: Fused2Weights,
                                     ring: RingParams) -> None:
    w = migrated
    plain = Fused2PlainBackend(ring)
    ref = _run(_model(pack, plain, w, ring))
    assert plain.max_bits and max(plain.max_bits.values()) <= ring.cap
    provider = Fused2Provider(w, ring.k)
    provider.record = True
    be = Fused2MaskedBackend(ring, provider, Fused2MaskSource(w, ring))
    got = _run(_model(pack, be, w, ring))
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    layers = pack.arch.num_layers
    assert be.f2.trips == 4 * 2 * (2 * layers + 1)
    assert be.f2.checks == 4 * 2 * (1 + 2 * (2 * layers - 1) + 1)
    for _key, arr in provider.seen:
        assert int(arr.min()) >= 0 and int(arr.max()) < (1 << ring.k)
    big = [arr for _k, arr in provider.seen if arr.numel() >= 32]
    assert big and all(int((arr >> (ring.k - 4)).max()) > 0 for arr in big)
    # the same wire bytes and factory products as the unmigrated reference
    be0 = Fused2MaskedBackend(ring, Fused2Provider(weights, ring.k), Fused2MaskSource(weights, ring))
    _run(_model(pack, be0, weights, ring))
    assert (be.f2.trips, be.f2.bytes_up, be.f2.bytes_down, be.masks.macs) == (
        be0.f2.trips, be0.f2.bytes_up, be0.f2.bytes_down, be0.masks.macs)


@pytest.mark.parametrize("what", ["H", "H_consistent", "Z", "Z_skip_residual"])
@pytest.mark.parametrize("delta", [1, -(1 << 20)], ids=["one", "neg"])
def test_migrated_tamper_detected(pack: InMemoryPack, migrated: Fused2Weights, what: str, delta: int) -> None:
    ring = RINGS[0]
    be = Fused2MaskedBackend(ring, _Evil(migrated, ring.k, what, when=2, delta=delta),
                             Fused2MaskSource(migrated, ring))
    with pytest.raises(IntegrityError):
        _run(_model(pack, be, migrated, ring))


def test_remote_provider_over_socket_exact(pack: InMemoryPack, migrated: Fused2Weights, tmp_path: Any) -> None:
    """The e2e harness path: provider in its own server (public w8 + hmul from a provider-only file) on a Unix
    socket, masks prefilled; bit-identical to the plain reference, same trips, prefilled masks used."""
    import threading

    from pgw.engine.fused2_remote import Fused2Service, RemoteFused2Provider, serve

    ring = RINGS[0]
    ref = _run(_model(pack, Fused2PlainBackend(ring), migrated, ring))
    f_full, f_prov = tmp_path / "w.pt", tmp_path / "p.pt"
    migrated.save(f_full)
    migrated.save(f_prov, provider_only=True)
    w = Fused2Weights.load(f_full)
    prov_w = Fused2Weights.load(f_prov)
    assert all(a.scale.abs().max() == 0 for a in prov_w.linears())  # provider file: no scales, norms
    for a, b in zip(w.linears(), migrated.linears(), strict=True):
        assert a.key == b.key and torch.equal(a.w8, b.w8) and a.wnorm == b.wnorm and a.sigma_in == b.sigma_in
        assert (a.hmul is None) == (b.hmul is None) and (a.gcols is None) == (b.gcols is None)
    addr = f"unix:{tmp_path / 's.sock'}"
    ready = threading.Event()
    threading.Thread(target=serve, args=(Fused2Service(prov_w, ring.k, "cpu"), addr), kwargs={"ready": ready},
                     daemon=True).start()
    assert ready.wait(30)
    client = RemoteFused2Provider(addr, ring.k, {lin.key: (lin.m, lin.n) for lin in w.linears()})
    src = Fused2MaskSource(w, ring)
    src.prefill(4)
    src.prefill(20)  # two stock chunks: a take can span them
    assert src.stocked_rows() == 24
    be = Fused2MaskedBackend(ring, client, src)
    got = _run(_model(pack, be, w, ring))
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    assert src.stock_hits > 0 and len(client.trip_ms) == be.f2.trips
    client.close()


def test_pure_migration_k56(pack: InMemoryPack) -> None:
    """migration_multipliers (pack time, public weights only, no gateway columns) at K56C53: exact, checked."""
    from pgw.engine.fused2 import migration_multipliers

    ring = RingParams(56, 12, 53)
    mults = migration_multipliers(pack, 8)
    assert f"{pack.arch.num_layers - 1}.down" not in mults and int(max(int(v.max()) for v in mults.values())) > 1
    w = Fused2Weights(pack, multipliers=mults)
    plain = Fused2PlainBackend(ring)
    ref = _run(_model(pack, plain, w, ring))
    assert max(plain.max_bits.values()) <= ring.cap
    be = Fused2MaskedBackend(ring, Fused2Provider(w, ring.k), Fused2MaskSource(w, ring))
    got = _run(_model(pack, be, w, ring))
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    evil = Fused2MaskedBackend(ring, _Evil(w, ring.k, "H_consistent", when=2), Fused2MaskSource(w, ring))
    with pytest.raises(IntegrityError):
        _run(_model(pack, evil, w, ring))


def test_prefilled_masks_equal_lazy(weights: Fused2Weights) -> None:
    ring = RINGS[0]
    from pgw.engine.fused2 import Fused2Trip

    seed = bytes(range(32))
    a, b = Fused2MaskSource(weights, ring, seed=seed), Fused2MaskSource(weights, ring, seed=seed)
    b.prefill(5)
    for rows in (2, 3):
        t = Fused2Trip("pair", weights.o[0], torch.zeros((rows, weights.o[0].n), dtype=torch.int64), ring.cap,
                       weights.gu[0], torch.zeros((rows, weights.o[0].m), dtype=torch.int64))
        x, y = a.take(t), b.take(t)
        for f in ("r_x", "wr_x", "v", "wv"):
            assert torch.equal(getattr(x, f), getattr(y, f))
    assert b.stock_hits == 2 and a.stock_misses == 2


def test_migrated_provider_skipping_multiplier_detected(pack: InMemoryPack, migrated: Fused2Weights) -> None:
    """A provider that ignores the public multipliers (``H = R + W1 x``) is caught by stage 1."""
    ring = RINGS[0]

    class NoMul(Fused2Provider):
        def __init__(self, weights: Fused2Weights, k: int):
            super().__init__(weights, k)
            self._hmul = {}

    be = Fused2MaskedBackend(ring, NoMul(migrated, ring.k), Fused2MaskSource(migrated, ring))
    with pytest.raises(IntegrityError):
        _run(_model(pack, be, migrated, ring))
