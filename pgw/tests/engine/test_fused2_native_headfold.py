"""The native fused2 driver with the folded LM head and layer 0's qkv local (the client library's ``schedule =
"fused"``; opt-in): 2L trips per pass natively.

Against the Python fused2 path (same lane, same mask seed) and the plaintext reference: logits bit for bit, the same
provider-visible bodies, exactly 2L lane calls per pass (no trip for layer 0's qkv, no separate head trip), every
returned value checked (two checks per pair trip), and a dishonest provider caught natively on the head's logits,
on the last trip's ``H`` and on a consistently recomputed head. Also: the item table v2 (the head pair appended),
its ticket version, and the pair-safe lane multi frames (several pair items in one frame).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core.errors import IntegrityError, ProtocolError
from pgw.engine import InMemoryPack, SequenceInput, TransformerModel
from pgw.engine.fused2 import (
    Fused2MaskedBackend,
    Fused2MaskSource,
    Fused2PlainBackend,
    Fused2Weights,
    enable_fused2,
    migration_multipliers,
)
from pgw.engine.fused2_lane import LaneFused2Provider, items_of, single_multipliers
from pgw.engine.fused2_masks import gateway_tables
from pgw.service.fused2_lane import ITEMS_V2, Fused2LaneEntry, fused2_items
from tests.engine.test_fused2_lane import RING, _Lane, sock_dir  # noqa: F401  (fixture)

fast = pytest.importorskip("pgw_fast")
if not hasattr(fast, "Fused2Driver") or not hasattr(fast.Fused2Driver, "set_head_fold"):
    pytest.skip("pgw_fast without the head-fold driver (rebuild: rust/tools/build_fast.py)", allow_module_level=True)


@pytest.fixture(scope="module")
def weights(qwen_pack: InMemoryPack) -> Fused2Weights:
    return Fused2Weights(qwen_pack, multipliers=migration_multipliers(qwen_pack, 8, head=True), head=True)


@pytest.fixture(scope="module")
def entry(weights: Fused2Weights, tmp_path_factory: pytest.TempPathFactory) -> Fused2LaneEntry:
    p = tmp_path_factory.mktemp("f2hn") / "prov.pt"
    weights.save(p, provider_only=True)
    return Fused2LaneEntry.from_weights(Fused2Weights.load(p), RING, device="cpu", max_rows=8)


def _provider(lane: _Lane, w: Any, *, multi: bool = False) -> LaneFused2Provider:
    t = lane.mgr.open_ticket(None, lane.entry.fingerprint)
    assert t["fused2_items"] == ITEMS_V2
    ch = fast.TripChannel.connect_service_uds(t["socket"], bytes.fromhex(t["ticket"]), 30.0, 8, 0)
    mi = int(t.get("max_items", 0) or 0) if multi else 0
    return LaneFused2Provider(ch, t["linears"], items_of(w), RING.k, max_rows=8, hmul=single_multipliers(w),
                              max_items=mi, max_multi_rows=int(t.get("max_multi_rows", 0) or 0))


def _model(pack: InMemoryPack, backend: Any, w: Any, native: bool = False) -> TransformerModel:
    from pgw.engine.fused2_native import enable_fused2_native

    m = TransformerModel(pack, backend)
    m.set_float_ops("native")
    enable_fused2(m, w, RING, head_fold=True, local_qkv0=True)
    if native:
        enable_fused2_native(m)
    return m


def _gen(model: TransformerModel, steps: int = 4, prompt_len: int = 5, seed: int = 7, all_logits: bool = True
         ) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    v = model.vocab_size
    cache = model.new_cache(64)
    out = model.forward_batch([SequenceInput(torch.randint(0, v, (prompt_len,), generator=g), cache,
                                             all_logits=all_logits)])
    for _ in range(steps):
        tok = torch.tensor([int(out[-1].reshape(-1, v)[-1].argmax())])
        out += model.forward_batch([SequenceInput(tok, cache)])
    return out


def test_item_table_v2(weights: Fused2Weights, entry: Fused2LaneEntry) -> None:
    items = fused2_items(weights)
    L = weights.num_layers
    assert len(items) == 2 * L + 2 and entry.items_version == ITEMS_V2
    last = items[-1]
    assert last.k1 == f"{L - 1}.down" and last.k2 == f"{L}.lm_head/f2" and last.m2 == weights.head.m
    assert items[-2].k2 is None  # the single last down (rows without logits) stays
    assert single_multipliers(weights).keys() == {f"{L - 1}.down"}


def test_native_head_fold_equals_python_and_plain(qwen_pack: InMemoryPack, weights: Fused2Weights,
                                                  entry: Fused2LaneEntry, sock_dir: Path) -> None:  # noqa: F811
    ref = _gen(_model(qwen_pack, Fused2PlainBackend(RING), weights))
    gw = weights  # (the in-process mask source computes its products from the trip's linears: full weights)
    outs, bodies = [], []
    L = weights.num_layers
    for native in (False, True):
        d = sock_dir / ("n" if native else "p")
        d.mkdir()
        lane = _Lane(entry, d, shm=False)
        try:
            prov = _provider(lane, gw)
            be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING, seed=bytes(range(32))), weights=gw)
            model = _model(qwen_pack, be, gw, native=native)
            outs.append(_gen(model))
            bodies.append(list(lane.bodies))
            assert prov.calls == 5 * 2 * L  # 2L trips per pass (every pass's rows all get logits here)
            if native:
                nd = model.schedule.native_decode
                assert nd.totals[0] == 5 and nd.totals[2] == 5 * 2 * L
            assert be.f2.checks == 5 * 2 * 2 * L  # two checks per pair trip, every trip a pair
            prov.close()
        finally:
            lane.close()
    for a, b, c in zip(ref, outs[0], outs[1], strict=True):
        assert torch.equal(a, b) and torch.equal(a, c)
    assert bodies[0] == bodies[1]
    # no body is sent for layer 0's qkv (computed locally) and no payload row repeats (fresh masks)
    assert all(li != 0 for li, _, _ in bodies[1])
    rows = [b[i:i + 7] for _, _, b in bodies[1] for i in range(0, len(b), 7 * 64)]
    assert len(rows) == len(set(rows))


def test_prompt_rows_without_logits_take_the_single_item(qwen_pack: InMemoryPack, weights: Fused2Weights,
                                                         entry: Fused2LaneEntry, sock_dir: Path) -> None:  # noqa: F811
    """A prompt pass whose last row only needs logits: the other rows take the single last-down item (the client
    applies the public multipliers), both items in ONE pair-safe lane multi frame (one round trip)."""
    ref = _gen(_model(qwen_pack, Fused2PlainBackend(RING), weights), steps=2, prompt_len=6, all_logits=False)
    gw = weights
    lane = _Lane(entry, sock_dir, shm=False)
    try:
        prov = _provider(lane, gw, multi=True)
        assert prov.max_items >= 2
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING), weights=gw)
        model = _model(qwen_pack, be, gw, native=True)
        out = _gen(model, steps=2, prompt_len=6, all_logits=False)
        L = weights.num_layers
        assert prov.calls == 3 * 2 * L and prov.multi_calls >= 1
        names = [entry.items[li].name for li, _, _ in lane.bodies]
        assert f"{L - 1}.down" in names and f"{L - 1}.down+{L}.lm_head/f2" in names
        prov.close()
    finally:
        lane.close()
    for a, b in zip(ref, out, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("what", ["H", "H_consistent", "Z", "Z_skip_residual", "no_hmul"])
def test_native_head_fold_tamper_detected(qwen_pack: InMemoryPack, weights: Fused2Weights, entry: Fused2LaneEntry,
                                          sock_dir: Path, what: str) -> None:  # noqa: F811
    """Tampering on every pair item, the head's (the last) included: the native driver refuses before use."""
    gw = weights
    lane = _Lane(entry, sock_dir, shm=False, evil=what)
    try:
        prov = _provider(lane, gw)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING), weights=gw)
        model = _model(qwen_pack, be, gw, native=True)
        with pytest.raises(IntegrityError):
            _gen(model, steps=1, prompt_len=3)
        prov.close()
    finally:
        lane.close()


def test_head_only_tamper_detected(qwen_pack: InMemoryPack, weights: Fused2Weights, entry: Fused2LaneEntry,
                                   sock_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """A provider that is honest everywhere but changes one logit of the head item: caught (stage-2 check)."""
    from pgw.core.packing import pack_ring, unpack_ring

    gw = weights
    lane = _Lane(entry, sock_dir, shm=False)
    head_ix = len(entry.items) - 1
    real = entry.run

    def run(index: int, rows: int, body: Any) -> bytes:
        out = real(index, rows, body)
        if index != head_ix:
            return out
        it = entry.items[index]
        cut = rows * it.m1 * 7
        z = unpack_ring(out[cut:], (rows, it.m2), RING.k).clone()
        z[0, 5] = (z[0, 5] + 1) & ((1 << RING.k) - 1)
        return out[:cut] + pack_ring(z, RING.k)

    monkeypatch.setattr(entry, "run", run)
    try:
        prov = _provider(lane, gw)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING), weights=gw)
        model = _model(qwen_pack, be, gw, native=True)
        with pytest.raises(IntegrityError, match="lm_head"):
            _gen(model, steps=1, prompt_len=2)
        prov.close()
    finally:
        lane.close()


def test_native_refuses_a_missing_qkv0_matrix(qwen_pack: InMemoryPack, weights: Fused2Weights,
                                              entry: Fused2LaneEntry, sock_dir: Path) -> None:  # noqa: F811
    """local_qkv0 needs the public layer-0 matrix: weight-free tables without it are refused (fail closed)."""
    from pgw.core.errors import ConfigError

    gw = gateway_tables(weights)  # no keep_qkv0
    lane = _Lane(entry, sock_dir, shm=False)
    try:
        prov = _provider(lane, gw)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING), weights=gw)
        with pytest.raises(ConfigError, match="local_qkv0"):
            _model(qwen_pack, be, gw, native=True)
        prov.close()
    finally:
        lane.close()


def test_v1_gateway_refuses_a_v2_ticket_and_vice_versa(weights: Fused2Weights) -> None:
    import json

    from pgw.gateway.fused2 import _check_ticket
    from pgw.service.fused2_lane import fused2_fingerprint

    names = [it.name for it in fused2_items(weights)]
    fp = fused2_fingerprint(weights, RING)
    raw = {"version": 1, "transport": "uds", "ticket": "00" * 32, "max_rows": 8, "linears": names,
           "rings": [RING.name], "fingerprint": fp, "fused2_items": 2}
    assert _check_ticket(json.dumps(raw).encode(), fp, RING, names, ("uds",), 2)["linears"] == names
    with pytest.raises(ProtocolError, match="version"):
        _check_ticket(json.dumps({**raw, "fused2_items": 1}).encode(), fp, RING, names, ("uds",), 2)
    with pytest.raises(ProtocolError, match="version"):
        _check_ticket(json.dumps(raw).encode(), fp, RING, names, ("uds",), 1)
