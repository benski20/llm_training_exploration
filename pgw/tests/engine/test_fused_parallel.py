"""Parallel attention + MLP blocks with the RMSNorm fusion: one provider trip per layer (prototype, default off).

The tiny Qwen2 / Llama packs are read as parallel-block models (a different model; only exactness, trips, privacy and
integrity are meaningful here). Proven: masked == plain bit for bit; L + 1 trips per pass with the head folded into
the last trip (L + 1 with the gateway head); the provider sees only uniformly masked ring elements, every row once;
every returned value is checked (2 checks per pair trip) and tampering raises; close to the float64 parallel-block
reference; defaults unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from pgw.core.errors import IntegrityError
from pgw.core.ring import RingParams
from pgw.engine import InMemoryPack, SequenceInput, TransformerModel
from pgw.engine.fused2 import Fused2MaskedBackend, Fused2MaskSource, Fused2PlainBackend, Fused2Provider
from pgw.engine.fused_parallel import ParallelWeights, enable_fused_parallel, parallel_float_reference

RING = RingParams(56, 12, 53)


@pytest.fixture(scope="module", params=["qwen", "llama"])
def pack(request: pytest.FixtureRequest, qwen_pack: InMemoryPack, llama_pack: InMemoryPack) -> InMemoryPack:
    return qwen_pack if request.param == "qwen" else llama_pack


@pytest.fixture(scope="module")
def pw(pack: InMemoryPack) -> ParallelWeights:
    return ParallelWeights(pack, bits=8, head=True)


def _model(pack: InMemoryPack, be: Any, w: ParallelWeights, head_fold: bool) -> TransformerModel:
    m = TransformerModel(pack, be)
    enable_fused_parallel(m, w, RING, head_fold=head_fold)
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


@pytest.mark.parametrize("head_fold", [True, False], ids=["head_fold", "gateway_head"])
def test_masked_equals_plain_one_trip_per_layer(pack: InMemoryPack, pw: ParallelWeights, head_fold: bool) -> None:
    ref = _run(_model(pack, Fused2PlainBackend(RING), pw, head_fold))
    prov = Fused2Provider(pw, RING.k)  # type: ignore[arg-type]
    prov.record = True
    be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(pw, RING))  # type: ignore[arg-type]
    m = _model(pack, be, pw, head_fold)
    got = _run(m)
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    L = pack.arch.num_layers
    assert m._trip_plan() == (L + 1, False)
    rows: set[bytes] = set()
    for _k, arr in prov.seen:
        assert int(arr.min()) >= 0 and int(arr.max()) < (1 << RING.k)
        if arr.shape[-1] >= 32:
            assert int((arr >> (RING.k - 4)).max()) > 0
        for r in arr.reshape(-1, arr.shape[-1]):
            key = r.numpy().tobytes()
            assert key not in rows  # one-time masks: no payload row ever repeats
            rows.add(key)
    # one decode pass: L + 1 trips, every pair checks H and Z
    c = m.new_cache(8)
    m.forward(torch.tensor([3]), c)
    be.f2.trips = be.f2.checks = 0
    m.forward(torch.tensor([4]), c)
    assert be.f2.trips == L + 1
    pairs = L if head_fold else L - 1
    assert be.f2.checks == 2 * pairs + (L + 1 - pairs)


class _Evil(Fused2Provider):
    def __init__(self, w: Any, k: int, what: str):
        super().__init__(w, k)
        self.what, self.n = what, 0

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, z = super().pair(k1, k2, r_hat, x_hat)
        self.n += 1
        mk = (1 << self.k) - 1
        if self.n < 2:
            return h, z
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
        elif self.what == "drop_mlp_delta":  # skip the residual delta entirely (H = R): stage 1 catches it
            h = r_hat.clone()
            z = self._w[k2].matmul(h, self.k)
        return h, z


@pytest.mark.parametrize("what", ["H", "Z", "H_consistent", "drop_mlp_delta"])
def test_tamper_detected(pack: InMemoryPack, pw: ParallelWeights, what: str) -> None:
    be = Fused2MaskedBackend(RING, _Evil(pw, RING.k, what), Fused2MaskSource(pw, RING))  # type: ignore[arg-type]
    m = _model(pack, be, pw, True)
    with pytest.raises(IntegrityError):
        m.forward(torch.tensor([1, 2, 3]), m.new_cache(8))


def test_close_to_float_parallel_reference(pack: InMemoryPack, pw: ParallelWeights) -> None:
    ids = torch.arange(1, 9)
    m = _model(pack, Fused2PlainBackend(RING), pw, True)
    a = m.forward(ids, m.new_cache(16), all_logits=True).double()
    b = parallel_float_reference(pack, ids)
    assert a.shape == b.shape
    err = float((a - b).abs().max())
    assert 0 < err < 0.03 * float(b.abs().max()), err
    assert torch.equal(a.argmax(-1), b.argmax(-1)) or float((a.argmax(-1) == b.argmax(-1)).float().mean()) >= 0.75


def test_default_model_unchanged(pack: InMemoryPack) -> None:
    from pgw.engine import PlainBackend

    m = TransformerModel(pack, PlainBackend(pack, RingParams(32, 12, 27)))
    assert m.schedule is None and m.head_mode is None
    assert m._trip_plan() == (4 * pack.arch.num_layers, False)
