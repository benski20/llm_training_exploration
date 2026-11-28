"""Mixed sequential / parallel layer maps on the fused schedule (``pgw.engine.fused_mixed``; opt-in, default off).

Proven on tiny 4-layer Qwen2 / Llama / Qwen3 packs: an all-sequential map gives fused2's tables and outputs bit for
bit, an all-parallel map the parallel prototype's; for mixed maps masked == plain bit for bit, ``1 + 2 S + P`` trips
per pass (head folded), every pair trip checks ``H`` and ``Z``, no masked payload row ever repeats, every provider
value is a uniform ring element; tampering with a sequential or a parallel trip raises; close to the float64
reference of the same map; a pack that declares its map (``arch.parallel_layers``, in the fingerprint) runs only
under that map; packs without the field keep their fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core.errors import ConfigError, IntegrityError
from pgw.core.ring import RingParams
from pgw.engine import InMemoryPack, PlainBackend, SequenceInput, TransformerModel
from pgw.engine.fused2 import (
    Fused2MaskedBackend,
    Fused2MaskSource,
    Fused2PlainBackend,
    Fused2Provider,
    Fused2Weights,
    enable_fused2,
    migration_multipliers,
)
from pgw.engine.fused_mixed import (
    MixedWeights,
    enable_fused_mixed,
    layer_map_from_parallel,
    mixed_float_reference,
    normalize_layer_map,
    trips_per_pass,
)
from pgw.engine.fused_parallel import ParallelWeights, enable_fused_parallel
from tests.engine.conftest import save_tiny_variant

RING = RingParams(56, 12, 53)
MAPS = ["spsp", "pssp", "sspp", "ppps", "spps"]


@pytest.fixture(scope="module", params=["qwen", "llama", "qwen3"])
def pack(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> InMemoryPack:
    if request.param == "qwen3":
        from tests.engine.test_qwen3 import _save_qwen3

        d = _save_qwen3(tmp_path_factory.mktemp("q3x4"), seed=21, num_hidden_layers=4)
    else:
        d = save_tiny_variant(tmp_path_factory.mktemp(f"{request.param}x4"), "qwen2" if request.param == "qwen"
                              else "llama", seed=17, num_hidden_layers=4)
    return InMemoryPack.from_hf(d, name=f"tiny-{request.param}-4")


def _model(pack: Any, be: Any, w: MixedWeights, head_fold: bool = True, local_qkv0: bool = False) -> TransformerModel:
    m = TransformerModel(pack, be)
    enable_fused_mixed(m, w, RING, head_fold=head_fold, local_qkv0=local_qkv0)
    return m


def _run(model: TransformerModel) -> list[Any]:
    g = torch.Generator().manual_seed(5)
    v = model.vocab_size
    caches = [model.new_cache(48), model.new_cache(48)]
    prompts = [torch.randint(0, v, (5,), generator=g), torch.randint(0, v, (2,), generator=g)]
    out: list[Any] = model.forward_batch([SequenceInput(p, c, all_logits=True)
                                          for p, c in zip(prompts, caches, strict=True)])
    last = [o[-1] for o in out]
    for _ in range(3):
        nxt = [torch.tensor([int(lg.argmax())]) for lg in last]
        last = model.forward_batch([SequenceInput(t, c) for t, c in zip(nxt, caches, strict=True)])
        out += last
    c3 = model.new_cache(16)
    out += model.forward_batch([SequenceInput(torch.randint(0, v, (4,), generator=g), c3)])  # last-row logits
    return out


def _same_linear(a: Any, b: Any) -> None:
    assert a.key == b.key
    assert torch.equal(a.w8, b.w8) and torch.equal(a.scale, b.scale)
    assert (a.bias is None) == (b.bias is None) and (a.bias is None or torch.equal(a.bias, b.bias))
    assert (a.hmul is None) == (b.hmul is None) and (a.hmul is None or torch.equal(a.hmul, b.hmul))
    assert a.wnorm == b.wnorm and a.sigma == b.sigma


def test_layer_map_helpers() -> None:
    assert layer_map_from_parallel(5, [1, 3]) == "spsps"
    assert normalize_layer_map("SPSP", 4) == "spsp"
    assert trips_per_pass("ssss") == 9 and trips_per_pass("pppp") == 5 and trips_per_pass("spsp") == 7
    assert trips_per_pass("spsp", local_qkv0=True) == 6 and trips_per_pass("spsp", head_trip=True) == 8
    for bad in ("sps", "spsx"):
        with pytest.raises(ConfigError):
            normalize_layer_map(bad, 4)
    with pytest.raises(ConfigError):
        layer_map_from_parallel(4, [4])


@pytest.mark.parametrize("head", [True, False])
def test_all_sequential_is_fused2_bit_for_bit(pack: InMemoryPack, head: bool) -> None:
    mw = MixedWeights(pack, "ssss", bits=8, head=head)
    fw = Fused2Weights(pack, None, multipliers=migration_multipliers(pack, 8, head=head), head=head)
    for i in range(4):
        _same_linear(mw.inp[i], fw.qkv[i])
        _same_linear(mw.o[i], fw.o[i])
        _same_linear(mw.gu[i], fw.gu[i])
        _same_linear(mw.end[i], fw.down[i])
    assert (mw.head is None) == (fw.head is None)
    if head:
        _same_linear(mw.head, fw.head)
    ref = TransformerModel(pack, Fused2PlainBackend(RING))
    enable_fused2(ref, fw, RING, head_fold=head)
    for a, b in zip(_run(ref), _run(_model(pack, Fused2PlainBackend(RING), mw, head)), strict=True):
        assert torch.equal(a, b)


def test_all_parallel_is_the_parallel_prototype_bit_for_bit(pack: InMemoryPack) -> None:
    mw = MixedWeights(pack, "pppp", bits=8, head=True)
    pw = ParallelWeights(pack, bits=8, head=True)
    for i in range(4):
        _same_linear(mw.inp[i], pw.second[i])
        _same_linear(mw.end[i], pw.first[i])
    _same_linear(mw.head, pw.head)
    ref = TransformerModel(pack, Fused2PlainBackend(RING))
    enable_fused_parallel(ref, pw, RING, head_fold=True)
    for a, b in zip(_run(ref), _run(_model(pack, Fused2PlainBackend(RING), mw)), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("lm", MAPS)
@pytest.mark.parametrize("head_fold", [True, False], ids=["head_fold", "gateway_head"])
def test_masked_equals_plain_mixed(pack: InMemoryPack, lm: str, head_fold: bool) -> None:
    mw = MixedWeights(pack, lm, bits=8, head=True)
    ref = _run(_model(pack, Fused2PlainBackend(RING), mw, head_fold))
    prov = Fused2Provider(mw, RING.k)  # type: ignore[arg-type]
    prov.record = True
    be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(mw, RING))  # type: ignore[arg-type]
    m = _model(pack, be, mw, head_fold)
    got = _run(m)
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    S, P = lm.count("s"), lm.count("p")
    assert m._trip_plan() == (1 + 2 * S + P, False) == (trips_per_pass(lm), False)
    rows: set[bytes] = set()
    for _k, arr in prov.seen:
        assert int(arr.min()) >= 0 and int(arr.max()) < (1 << RING.k)
        if arr.shape[-1] >= 32:
            assert int((arr >> (RING.k - 4)).max()) > 0
        for r in arr.reshape(-1, arr.shape[-1]):
            key = r.numpy().tobytes()
            assert key not in rows  # single-use masks: no payload row ever repeats
            rows.add(key)
    # one decode pass: 1 + 2S + P trips; every pair checks H and Z, every single its one output
    c = m.new_cache(8)
    m.forward(torch.tensor([3]), c)
    be.f2.trips = be.f2.checks = 0
    m.forward(torch.tensor([4]), c)
    trips = 1 + 2 * S + P
    assert be.f2.trips == trips
    pairs = trips - 1 - (0 if head_fold else 1)
    assert be.f2.checks == 2 * pairs + (trips - pairs)


def test_local_qkv0_and_prefilled_masks(pack: InMemoryPack) -> None:
    mw = MixedWeights(pack, "psps", bits=8, head=True)
    ref = _run(_model(pack, Fused2PlainBackend(RING), mw, True, local_qkv0=True))
    src = Fused2MaskSource(mw, RING)
    keys = [a.key + ("+" + b.key if b is not None else "") for a, b in src.trip_plan()]
    assert len(keys) == len(set(keys)) == 1 + 2 * 2 + 2 + 1  # input, 2 pairs + 1 pair per s, 1 per p, head
    src.prefill(16)
    be = Fused2MaskedBackend(RING, Fused2Provider(mw, RING.k), src)  # type: ignore[arg-type]
    m = _model(pack, be, mw, True, local_qkv0=True)
    for a, b in zip(ref, _run(m), strict=True):
        assert torch.equal(a, b)
    assert src.stock_hits > 0
    assert m._trip_plan() == (trips_per_pass("psps", local_qkv0=True), False)


class _Evil(Fused2Provider):
    """Tampers with the ``at``-th pair call (counted per pass from 1) in the way ``what`` says."""

    def __init__(self, w: Any, k: int, what: str, key_part: str):
        super().__init__(w, k)
        self.what, self.key_part, self.done = what, key_part, False

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, z = super().pair(k1, k2, r_hat, x_hat)
        if self.done or self.key_part not in k1:
            return h, z
        self.done = True
        mk = (1 << self.k) - 1
        if self.what == "H":
            h = h.clone()
            h.view(-1)[1] = (h.view(-1)[1] + 1) & mk
        elif self.what == "Z":
            z = z.clone()
            z.view(-1)[-1] = (z.view(-1)[-1] + (1 << 30)) & mk
        elif self.what == "H_consistent":
            h = h.clone()
            h.view(-1)[0] = (h.view(-1)[0] - 7) & mk
            z = self._w[k2].matmul(h, self.k)
        elif self.what == "drop_delta":
            h = r_hat.clone()
            z = self._w[k2].matmul(h, self.k)
        return h, z


@pytest.mark.parametrize("what", ["H", "Z", "H_consistent", "drop_delta"])
@pytest.mark.parametrize("key_part", ["o+down/p", ".down", ".o"], ids=["parallel_trip", "seq_down", "seq_o"])
def test_tamper_detected(pack: InMemoryPack, what: str, key_part: str) -> None:
    mw = MixedWeights(pack, "spps", bits=8, head=True)
    evil = _Evil(mw, RING.k, what, key_part)
    be = Fused2MaskedBackend(RING, evil, Fused2MaskSource(mw, RING))  # type: ignore[arg-type]
    m = _model(pack, be, mw)
    with pytest.raises(IntegrityError):
        m.forward(torch.tensor([1, 2, 3]), m.new_cache(8))
    assert evil.done


@pytest.mark.parametrize("lm", ["spsp", "ppss"])
def test_close_to_float_reference(pack: InMemoryPack, lm: str) -> None:
    ids = torch.arange(1, 9)
    m = _model(pack, Fused2PlainBackend(RING), MixedWeights(pack, lm, bits=8, head=True))
    a = m.forward(ids, m.new_cache(16), all_logits=True).double()
    b = mixed_float_reference(pack, ids, lm)
    assert a.shape == b.shape
    err = float((a - b).abs().max())
    assert 0 < err < 0.03 * float(b.abs().max()), err
    assert float((a.argmax(-1) == b.argmax(-1)).float().mean()) >= 0.75
    # the map matters: another map's reference is further away than the quantization error
    other = mixed_float_reference(pack, ids, "ssss" if lm != "ssss" else "pppp")
    assert float((other - b).abs().max()) > err


def test_float_reference_all_sequential_is_the_checkpoint(pack: InMemoryPack) -> None:
    ids = torch.arange(1, 9)
    m = TransformerModel(pack, PlainBackend(pack, RingParams(48, 12, 45)))
    a = m.forward(ids, m.new_cache(16), all_logits=True).double()
    b = mixed_float_reference(pack, ids, "ssss")
    assert float((a - b).abs().max()) < 0.03 * float(b.abs().max())


def _declared(pack: InMemoryPack, layers: tuple[int, ...]) -> InMemoryPack:
    arch = type(pack.arch)(**{**pack.arch.model_dump(), "parallel_layers": layers})
    return InMemoryPack(pack.name + "-par", arch, pack._linears, pack._plain)  # type: ignore[attr-defined]


def test_declared_layer_map_is_pinned(pack: InMemoryPack) -> None:
    assert "parallel_layers" not in pack.arch.model_dump(mode="json")  # unchanged fingerprints of existing packs
    dp = _declared(pack, (1, 2))
    assert dp.fingerprint != pack.fingerprint
    assert "parallel_layers" in dp.arch.model_dump(mode="json")
    # a declared map: every schedule that runs another map is refused (fail closed, never sequential blocks)
    std = TransformerModel(dp, PlainBackend(dp, RingParams(48, 12, 45)))
    with pytest.raises(ConfigError, match="parallel blocks"):
        std.forward(torch.tensor([1, 2]), std.new_cache(8))
    f2 = TransformerModel(dp, Fused2PlainBackend(RING))
    enable_fused2(f2, Fused2Weights(dp, None, multipliers=migration_multipliers(dp, 8)), RING)
    with pytest.raises(ConfigError, match="parallel blocks"):
        f2.forward(torch.tensor([1, 2]), f2.new_cache(8))
    with pytest.raises(ConfigError, match="declared map"):
        MixedWeights(dp, "spsp")
    mw = MixedWeights(dp, bits=8, head=True)  # None: the declared map
    assert mw.layer_map == "spps" and mw.pinned
    got = _run(_model(dp, Fused2PlainBackend(RING), mw))
    ref = _run(_model(pack, Fused2PlainBackend(RING), MixedWeights(pack, "spps", bits=8, head=True)))
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    assert not MixedWeights(pack, "spps").pinned and MixedWeights(pack).layer_map == "ssss"


def test_manifest_field_round_trip() -> None:
    from pgw.modelpack.manifest import ArchConfig

    base = {"arch": "qwen2", "hidden_size": 8, "intermediate_size": 16, "num_layers": 4, "num_heads": 2,
            "num_kv_heads": 1, "head_dim": 4, "vocab_size": 10, "rms_norm_eps": 1e-6, "rope_theta": 1e4,
            "max_position_embeddings": 64, "tie_word_embeddings": True}
    a = ArchConfig(**base)
    assert "parallel_layers" not in a.model_dump(mode="json")
    b = ArchConfig(**base, parallel_layers=[0, 3])
    assert ArchConfig.model_validate(b.model_dump(mode="json")) == b
    for bad in ([3, 0], [1, 1], [4]):
        with pytest.raises(ValueError):
            ArchConfig(**base, parallel_layers=bad)


def test_default_paths_unchanged(pack: InMemoryPack, tmp_path: Path) -> None:
    m = TransformerModel(pack, PlainBackend(pack, RingParams(32, 12, 27)))
    assert m.schedule is None and m.head_mode is None
    assert m._trip_plan() == (4 * pack.arch.num_layers, False)
