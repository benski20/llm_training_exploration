"""Parallel-block fused items on the service trip lane (item table v3; opt-in prototype): L + 1 trips per pass.

The parallel-block schedule (:mod:`pgw.engine.fused_parallel`, a tiny Qwen pack read as parallel blocks) runs over a
real ``LaneManager`` serving its item table (one pair item per layer, the folded head paired with the last
``[o | down]``): masked == plain bit for bit, exactly L + 1 lane calls per decode pass, the ticket's item-table
version 3, the provider's view uniform with no repeated row, and tampering caught.
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
from pgw.engine.fused_parallel import ParallelWeights, enable_fused_parallel
from pgw.service.fused2_lane import ITEMS_V3, Fused2LaneEntry, fused2_items
from tests.engine.test_fused2_lane import RING, _Lane, sock_dir  # noqa: F401  (fixture)

fast = pytest.importorskip("pgw_fast")
if not hasattr(fast, "TripChannel"):
    pytest.skip("pgw_fast without the lane client", allow_module_level=True)


@pytest.fixture(scope="module")
def pw(qwen_pack: InMemoryPack) -> ParallelWeights:
    return ParallelWeights(qwen_pack, bits=8, head=True)


@pytest.fixture(scope="module")
def entry(pw: ParallelWeights) -> Fused2LaneEntry:
    return Fused2LaneEntry.from_weights(pw, RING, device="cpu", max_rows=8)


def _model(pack: InMemoryPack, be: Any, w: ParallelWeights) -> TransformerModel:
    m = TransformerModel(pack, be)
    enable_fused_parallel(m, w, RING, head_fold=True)
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
    assert t["fused2_items"] == ITEMS_V3
    ch = fast.TripChannel.connect_service_uds(t["socket"], bytes.fromhex(t["ticket"]), 30.0, 8, 0)
    return LaneFused2Provider(ch, t["linears"], items_of(w), RING.k, max_rows=8, hmul=single_multipliers(w))


def test_item_table_v3(pw: ParallelWeights, entry: Fused2LaneEntry) -> None:
    items = fused2_items(pw)
    L = pw.num_layers
    assert len(items) == L + 2 and entry.items_version == ITEMS_V3
    assert [it.k2 is None for it in items] == [True] + [False] * (L - 1) + [True, False]


def test_parallel_over_the_lane_equals_plain(qwen_pack: InMemoryPack, pw: ParallelWeights, entry: Fused2LaneEntry,
                                            sock_dir: Path) -> None:  # noqa: F811
    ref = _gen(_model(qwen_pack, Fused2PlainBackend(RING), pw))
    lane = _Lane(entry, sock_dir, shm=False)
    try:
        prov = _provider(lane, pw)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(pw, RING))  # type: ignore[arg-type]
        got = _gen(_model(qwen_pack, be, pw))
        L = pw.num_layers
        assert prov.calls == 4 * (L + 1)  # L + 1 trips per pass, the head folded into the last
        rows = [b[i:i + 7 * 16] for _, _, b in lane.bodies for i in range(0, len(b), 7 * 64)]
        assert len(rows) == len(set(rows))  # no masked row is ever sent twice
        prov.close()
    finally:
        lane.close()
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("what", ["H", "Z"])
def test_parallel_tamper_detected(qwen_pack: InMemoryPack, pw: ParallelWeights, entry: Fused2LaneEntry,
                                  sock_dir: Path, what: str) -> None:  # noqa: F811
    lane = _Lane(entry, sock_dir, shm=False, evil=what)
    try:
        prov = _provider(lane, pw)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(pw, RING))  # type: ignore[arg-type]
        with pytest.raises(IntegrityError):
            _gen(_model(qwen_pack, be, pw), steps=1)
        prov.close()
    finally:
        lane.close()
