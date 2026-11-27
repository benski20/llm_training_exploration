"""fused2 completion for high-RTT clients (opt-in): the LM head folded into the last trip, and layer 0's qkv local.

Default off. With ``head_fold`` the head is the second stage of the last trip (last down + final-norm-folded int8
head) for the rows that need logits; with ``local_qkv0`` layer 0's qkv product is computed exactly on the gateway.
Together a pass costs 2L provider trips with the head on the provider (today's SDK path: 4L + 1).

Proven here (tiny Qwen2 and Llama): masked == plain bit for bit (prefill, decode, verify rows, lockstep batches);
the trip counts; every array the provider receives is uniformly masked ring elements, each mask row used once
(no two payload rows equal, mask cursors only move forward); every returned value is checked (range + F_p
Freivalds on both stages of every pair, the head's logits included) and tampering with any of them, or a wrong
factory correction of the head, raises before use; the default path is unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from pgw.core.errors import ConfigError, IntegrityError
from pgw.core.ring import RingParams
from pgw.engine import InMemoryPack, PlainBackend, SequenceInput, TransformerModel
from pgw.engine.fused2 import (
    Fused2FoldedHead,
    Fused2MaskedBackend,
    Fused2MaskSource,
    Fused2PlainBackend,
    Fused2Provider,
    Fused2Weights,
    enable_fused2,
    migration_multipliers,
)

RING = RingParams(56, 12, 53)
MODES = [(True, False), (True, True), (False, True)]
MODE_IDS = ["head_fold", "head_fold+local_qkv0", "local_qkv0"]


@pytest.fixture(scope="module", params=["qwen", "llama"])
def pack(request: pytest.FixtureRequest, qwen_pack: InMemoryPack, llama_pack: InMemoryPack) -> InMemoryPack:
    return qwen_pack if request.param == "qwen" else llama_pack


@pytest.fixture(scope="module")
def hw(pack: InMemoryPack) -> Fused2Weights:
    """Pure pack-time migration (8 bits) including the last down, and the folded head."""
    return Fused2Weights(pack, multipliers=migration_multipliers(pack, 8, head=True), head=True)


def _model(pack: InMemoryPack, backend: Any, w: Fused2Weights, head_fold: bool, local_qkv0: bool) -> TransformerModel:
    m = TransformerModel(pack, backend)
    enable_fused2(m, w, RING, head_fold=head_fold, local_qkv0=local_qkv0)
    return m


def _run(model: TransformerModel) -> list[Any]:
    """Two sequences: lockstep prefill (all logits), three decode steps, then one 3-row verify pass (last_logits)."""
    g = torch.Generator().manual_seed(11)
    v = model.vocab_size
    caches = [model.new_cache(64), model.new_cache(64)]
    prompts = [torch.randint(0, v, (5,), generator=g), torch.randint(0, v, (3,), generator=g)]
    out: list[Any] = model.forward_batch([SequenceInput(p, c, all_logits=True)
                                          for p, c in zip(prompts, caches, strict=True)])
    last = [o[-1] for o in out]
    for _ in range(3):
        nxt = [torch.tensor([int(lg.argmax())]) for lg in last]
        last = model.forward_batch([SequenceInput(t, c) for t, c in zip(nxt, caches, strict=True)])
        out += last
    ver = model.forward_batch([SequenceInput(torch.randint(0, v, (3,), generator=g), caches[0], last_logits=3),
                               SequenceInput(torch.randint(0, v, (4,), generator=g), caches[1])])
    out.append(torch.stack([ver[0][i] for i in range(3)]))
    out.append(ver[1])
    # a prefill whose last-row logits only (rows without logits take the single down trip)
    c3 = model.new_cache(16)
    out += model.forward_batch([SequenceInput(torch.randint(0, v, (6,), generator=g), c3)])
    return out


@pytest.mark.parametrize("head_fold,local_qkv0", MODES, ids=MODE_IDS)
def test_masked_equals_plain_and_trips(pack: InMemoryPack, hw: Fused2Weights, head_fold: bool,
                                       local_qkv0: bool) -> None:
    ref = _run(_model(pack, Fused2PlainBackend(RING), hw, head_fold, local_qkv0))
    prov = Fused2Provider(hw, RING.k)
    prov.record = True
    src = Fused2MaskSource(hw, RING)
    be = Fused2MaskedBackend(RING, prov, src)
    m = _model(pack, be, hw, head_fold, local_qkv0)
    got = _run(m)
    assert len(got) == len(ref)
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    L = pack.arch.num_layers
    trips = 2 * L + (0 if local_qkv0 else 1)
    assert m._trip_plan() == (trips, False)  # head_fold: no head trip of its own
    if head_fold:
        assert isinstance(m.head_mode, Fused2FoldedHead) and m.schedule.head_rows > 0  # type: ignore[union-attr]
    # provider view: canonical ring elements with high bits set (masked), and no payload row ever repeats
    # (one-time masks: a repeated row would mean a reused mask or an unmasked constant)
    rows: set[bytes] = set()
    for _key, arr in prov.seen:
        assert int(arr.min()) >= 0 and int(arr.max()) < (1 << RING.k)
        if arr.shape[-1] >= 32:
            assert int((arr >> (RING.k - 4)).max()) > 0
        for r in arr.reshape(-1, arr.shape[-1]):
            key = r.numpy().tobytes()
            assert key not in rows
            rows.add(key)
    if local_qkv0:
        assert all(not k.startswith("0.qkv/f2") for k, _ in prov.seen)  # layer 0's qkv never leaves


def test_every_returned_value_checked(pack: InMemoryPack, hw: Fused2Weights) -> None:
    """One decode pass: every trip's outputs pass a check: 2 per pair trip (H and Z, the head's Z included)."""
    be = Fused2MaskedBackend(RING, Fused2Provider(hw, RING.k), Fused2MaskSource(hw, RING))
    m = _model(pack, be, hw, True, True)
    c = m.new_cache(8)
    m.forward(torch.tensor([3]), c)
    be.f2.checks = be.f2.trips = 0
    m.forward(torch.tensor([5]), c)
    L = pack.arch.num_layers
    assert be.f2.trips == 2 * L  # every trip a pair: L x (o + gate_up), (L-1) x (down + qkv), down + head
    assert be.f2.checks == 2 * be.f2.trips
    # bytes: the head stage returns vocab values per logit row (as the separate masked head did)
    assert be.f2.bytes_down > m.vocab_size * RING.k // 8


class _Evil(Fused2Provider):
    def __init__(self, w: Fused2Weights, k: int, what: str, delta: int):
        super().__init__(w, k)
        self.what, self.delta, self.head_key = what, delta, w.head.key  # type: ignore[union-attr]

    def _bump(self, t: torch.Tensor, idx: int) -> torch.Tensor:
        t = t.clone()
        t.view(-1)[idx % t.numel()] += self.delta
        return t & ((1 << self.k) - 1)

    def single(self, k1: str, x_hat: torch.Tensor) -> torch.Tensor:
        y = super().single(k1, x_hat)
        return self._bump(y, 2) if self.what == "single_last" and k1.endswith(".down") else y

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, z = super().pair(k1, k2, r_hat, x_hat)
        if k2 != self.head_key:
            return h, z
        if self.what == "head_logit":
            return h, self._bump(z, 1234)
        if self.what == "head_H":
            return self._bump(h, 3), z
        if self.what == "head_H_consistent":
            h2 = self._bump(h, 3)
            return h2, self._w[k2].matmul(h2, self.k)
        if self.what == "head_skip_residual":
            return h, self._w[k2].matmul((h - r_hat) & ((1 << self.k) - 1), self.k)
        return h, z


@pytest.mark.parametrize("what", ["head_logit", "head_H", "head_H_consistent", "head_skip_residual", "single_last"])
@pytest.mark.parametrize("delta", [1, -(1 << 40)], ids=["one", "big"])
def test_tamper_on_folded_head_detected(pack: InMemoryPack, hw: Fused2Weights, what: str, delta: int) -> None:
    be = Fused2MaskedBackend(RING, _Evil(hw, RING.k, what, delta), Fused2MaskSource(hw, RING))
    m = _model(pack, be, hw, True, False)
    with pytest.raises(IntegrityError):
        # a 4-row prefill with last-row logits: rows 0-2 take the single down trip, row 3 the head pair
        m.forward(torch.tensor([1, 2, 3, 4]), m.new_cache(8))


def test_bad_head_correction_detected(pack: InMemoryPack, hw: Fused2Weights) -> None:
    class BadSource(Fused2MaskSource):
        def take(self, trip: Any) -> Any:
            mk = super().take(trip)
            if trip.l2 is not None and trip.l2.key == hw.head.key:  # type: ignore[union-attr]
                mk.wv.view(-1)[7] += 1  # a wrong W_head' v row moves one logit: caught on the unmasked Z
            return mk

    be = Fused2MaskedBackend(RING, Fused2Provider(hw, RING.k), BadSource(hw, RING))
    m = _model(pack, be, hw, True, False)
    with pytest.raises(IntegrityError):
        m.forward(torch.tensor([1, 2]), m.new_cache(8))


def test_close_to_current_pipeline(pack: InMemoryPack, hw: Fused2Weights) -> None:
    """A different pinned reference (the head is now int8 of W_head diag(g_final * grid)): close to today's
    int8 reference (standard schedule + gateway head), not equal."""
    ids = torch.arange(1, 9)
    f2 = _model(pack, Fused2PlainBackend(RING), hw, True, True)
    std = TransformerModel(pack, PlainBackend(pack, RingParams(32, 12, 27)))
    a = f2.forward(ids, f2.new_cache(16), all_logits=True)
    b = std.forward(ids, std.new_cache(16), all_logits=True)
    assert a.shape == b.shape
    assert 0 < float((a - b).abs().max()) < 0.05 * float(b.abs().max())


def test_defaults_unchanged(pack: InMemoryPack) -> None:
    w = Fused2Weights(pack)
    assert w.head is None and w.all_linears() == w.linears()
    m = TransformerModel(pack, Fused2PlainBackend(RING))
    s = enable_fused2(m, w, RING)
    assert not s.head_fold and not s.local_qkv0 and m.head_mode is None
    assert f"{pack.arch.num_layers - 1}.down" not in migration_multipliers(pack, 8)
    with pytest.raises(ConfigError):
        Fused2Weights(pack, multipliers=migration_multipliers(pack, 8, head=True))  # last down: head only
    with pytest.raises(ConfigError):
        enable_fused2(TransformerModel(pack, Fused2PlainBackend(RING)), w, RING, head_fold=True)  # no head
