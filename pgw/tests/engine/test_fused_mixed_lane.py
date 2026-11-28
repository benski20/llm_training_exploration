"""Mixed sequential / parallel layer maps on the service trip lane (item table v4; opt-in): 1 + 2 S + P trips per pass.

The mixed schedule (:mod:`pgw.engine.fused_mixed`) runs over a real ``LaneManager`` serving its item table: masked ==
plain bit for bit, exactly ``1 + 2 S + P`` lane calls per decode pass, the ticket's item-table version 4, no masked
row sent twice, tampering caught; another map's table has another lane fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core.errors import IntegrityError
from pgw.engine import InMemoryPack, SequenceInput, TransformerModel
from pgw.engine.fused2 import Fused2MaskedBackend, Fused2MaskSource, Fused2PlainBackend
from pgw.engine.fused2_lane import LaneFused2Provider, items_of, single_multipliers
from pgw.engine.fused_mixed import MixedWeights, enable_fused_mixed, trips_per_pass
from pgw.service.fused2_lane import ITEMS_V4, Fused2LaneEntry, fused2_fingerprint, fused2_items
from tests.engine.conftest import save_tiny_variant
from tests.engine.test_fused2_lane import RING, _Lane, sock_dir  # noqa: F401  (fixture)

fast = pytest.importorskip("pgw_fast")
if not hasattr(fast, "TripChannel"):
    pytest.skip("pgw_fast without the lane client", allow_module_level=True)

MAP = "spps"


@pytest.fixture(scope="module")
def pack4(tmp_path_factory: pytest.TempPathFactory) -> InMemoryPack:
    d = save_tiny_variant(tmp_path_factory.mktemp("qwenx4lane"), "qwen2", seed=17, num_hidden_layers=4)
    return InMemoryPack.from_hf(d, name="tiny-qwen-4")


@pytest.fixture(scope="module")
def mw(pack4: InMemoryPack) -> MixedWeights:
    return MixedWeights(pack4, MAP, bits=8, head=True)


@pytest.fixture(scope="module")
def entry(mw: MixedWeights) -> Fused2LaneEntry:
    return Fused2LaneEntry.from_weights(mw, RING, device="cpu", max_rows=8)


def _model(pack: InMemoryPack, be: Any, w: MixedWeights) -> TransformerModel:
    m = TransformerModel(pack, be)
    enable_fused_mixed(m, w, RING, head_fold=True)
    return m


def _gen(model: TransformerModel, steps: int = 3) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    v = model.vocab_size
    cache = model.new_cache(32)
    out = model.forward_batch([SequenceInput(torch.randint(0, v, (5,), generator=g), cache, all_logits=True)])
    for _ in range(steps):
        out += model.forward_batch([SequenceInput(torch.tensor([int(out[-1].reshape(-1, v)[-1].argmax())]), cache)])
    return out


def _provider(lane: _Lane, w: Any) -> LaneFused2Provider:
    t = lane.mgr.open_ticket(None, lane.entry.fingerprint)
    assert t["fused2_items"] == ITEMS_V4
    ch = fast.TripChannel.connect_service_uds(t["socket"], bytes.fromhex(t["ticket"]), 30.0, 8, 0)
    return LaneFused2Provider(ch, t["linears"], items_of(w), RING.k, max_rows=8, hmul=single_multipliers(w))


def test_item_table_v4(pack4: InMemoryPack, mw: MixedWeights, entry: Fused2LaneEntry) -> None:
    items = fused2_items(mw)
    # input single; layer 0 (s): 2 pairs; layers 1, 2 (p): 1 pair each; layer 3 (s): 1 pair + the end single; head
    assert [it.name for it in items] == [
        "0.qkv/f2", "0.o+0.gate_up/f2", "0.down+1.qkv+gate_up/p", "1.o+down/p+2.qkv+gate_up/p",
        "2.o+down/p+3.qkv/f2", "3.o+3.gate_up/f2", "3.down", "3.down+4.lm_head/f2"]
    assert entry.items_version == ITEMS_V4
    other = MixedWeights(pack4, "sssp", bits=8, head=True)
    assert fused2_fingerprint(other, RING) != entry.fingerprint


def test_mixed_over_the_lane_equals_plain(pack4: InMemoryPack, mw: MixedWeights, entry: Fused2LaneEntry,
                                         sock_dir: Path) -> None:  # noqa: F811
    ref = _gen(_model(pack4, Fused2PlainBackend(RING), mw))
    lane = _Lane(entry, sock_dir, shm=False)
    try:
        prov = _provider(lane, mw)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(mw, RING))  # type: ignore[arg-type]
        got = _gen(_model(pack4, be, mw))
        assert prov.calls == 4 * trips_per_pass(MAP)  # 1 + 2 S + P trips per pass, the head folded into the last
        rows = [b[i:i + 7 * 16] for _, _, b in lane.bodies for i in range(0, len(b), 7 * 64)]
        assert len(rows) == len(set(rows))  # no masked row is ever sent twice
        prov.close()
    finally:
        lane.close()
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("what", ["H", "Z"])
def test_mixed_tamper_detected(pack4: InMemoryPack, mw: MixedWeights, entry: Fused2LaneEntry,
                               sock_dir: Path, what: str) -> None:  # noqa: F811
    lane = _Lane(entry, sock_dir, shm=False, evil=what)
    try:
        prov = _provider(lane, mw)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(mw, RING))  # type: ignore[arg-type]
        with pytest.raises(IntegrityError):
            _gen(_model(pack4, be, mw), steps=1)
        prov.close()
    finally:
        lane.close()
