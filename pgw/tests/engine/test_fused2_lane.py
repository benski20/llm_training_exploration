"""fused2 on the production lane: pair items over the native service lane (UDS and shared memory), the Triton
pair op behind it, and pair-mask segments from a customer-side factory (weight-free gateway).

Masked == plain bit for bit through the real ``LaneService`` + ``LaneManager`` executor, byte-identical
provider-visible bodies across transports, one mask slot per decoded token, tampering (H, consistent H, Z, Z
without the residual, a provider skipping the multipliers) detected, segment masks == in-process masks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pgw.core.errors import IntegrityError, ProtocolError
from pgw.core.ring import RingParams
from pgw.engine import InMemoryPack, SequenceInput, TransformerModel
from pgw.engine.fused2 import (
    Fused2MaskedBackend,
    Fused2MaskSource,
    Fused2PlainBackend,
    Fused2Trip,
    Fused2Weights,
    enable_fused2,
    migration_multipliers,
)
from pgw.engine.fused2_lane import LaneFused2Provider, items_of
from pgw.engine.fused2_masks import Fused2MaskFactory, Fused2SegmentMasks, gateway_tables
from pgw.service.fused2_lane import Fused2LaneEntry, fused2_fingerprint, fused2_items

fast = pytest.importorskip("pgw_fast")
RING = RingParams(56, 12, 53)


@pytest.fixture(scope="module")
def weights(qwen_pack: InMemoryPack) -> Fused2Weights:
    return Fused2Weights(qwen_pack, multipliers=migration_multipliers(qwen_pack, 8))


@pytest.fixture(scope="module")
def provider_weights(weights: Fused2Weights, tmp_path_factory: pytest.TempPathFactory) -> Fused2Weights:
    p = tmp_path_factory.mktemp("f2") / "prov.pt"
    weights.save(p, provider_only=True)
    return Fused2Weights.load(p)


@pytest.fixture(scope="module")
def entry(provider_weights: Fused2Weights) -> Fused2LaneEntry:
    return Fused2LaneEntry.from_weights(provider_weights, RING, device="cpu", max_rows=8)


def _model(pack: InMemoryPack, backend: Any, w: Fused2Weights) -> TransformerModel:
    m = TransformerModel(pack, backend)
    enable_fused2(m, w, RING)
    return m


def _run(model: TransformerModel, steps: int = 3) -> list[torch.Tensor]:
    """Prefill two sequences in lockstep (5 and 11 rows: a call above the lane's 8 rows is split), then decode."""
    g = torch.Generator().manual_seed(7)
    v = model.vocab_size
    caches = [model.new_cache(64), model.new_cache(64)]
    prompts = [torch.randint(0, v, (5,), generator=g), torch.randint(0, v, (11,), generator=g)]
    out = model.forward_batch([SequenceInput(p, c, all_logits=True) for p, c in zip(prompts, caches, strict=True)])
    for _ in range(steps):
        nxt = [torch.tensor([int(o.reshape(-1, v)[-1].argmax())]) for o in out[-2:]]
        out += model.forward_batch([SequenceInput(t, c) for t, c in zip(nxt, caches, strict=True)])
    return out


class _Lane:
    """A real ``LaneManager`` (native ``LaneService`` + its Python executor) serving one fused2 entry on CPU."""

    def __init__(self, entry: Fused2LaneEntry, sock_dir: Path, *, shm: bool, evil: str | None = None):
        from pgw.service import ModelRegistry, ServiceMetrics, ServiceSettings
        from pgw.service.lane import LaneManager

        self.entry = entry
        self.bodies: list[tuple[int, int, bytes]] = []
        state = SimpleNamespace(registry=ModelRegistry("cpu"), handler=SimpleNamespace(lane_admission=None,
                                draining=False), metrics=ServiceMetrics(enabled=False), fused2={entry.fingerprint: entry})
        s = ServiceSettings(device="cpu", allow_unauthenticated=True, lane="uds", lane_socket=sock_dir / "lane.sock")
        self.mgr = LaneManager(s, state)
        self._mp = pytest.MonkeyPatch()
        if shm:
            self._mp.setenv("PGW_SERVICE_LANE_SHM", "1")
        real = self.mgr._py_execute

        def spy(slot: int, linear: int, ring: int, rows: int, body: bytes, tenant: str, left: float) -> Any:
            self.bodies.append((linear, rows, bytes(body)))
            out = real(slot, linear, ring, rows, body, tenant, left)
            return _tamper(entry, evil, linear, rows, body, out) if evil and isinstance(out, bytes) else out

        self.mgr._py_execute = spy  # type: ignore[method-assign]  # the native service calls this executor
        loop = asyncio.new_event_loop()
        self._loop = loop
        assert self.mgr._start_sync(loop)
        self.shm = shm

    def provider(self, gw_weights: Fused2Weights) -> LaneFused2Provider:
        t = self.mgr.open_ticket(None, self.entry.fingerprint)
        assert t["linears"] == self.entry.names() and t["rings"] == [RING.name]
        ch = fast.TripChannel.connect_service_uds(t["socket"], bytes.fromhex(t["ticket"]), 30.0, 8, 0)
        if self.shm:
            assert ch.attach_shm() is True
            assert ch.transport == "shm"
        return LaneFused2Provider(ch, t["linears"], items_of(gw_weights), RING.k, max_rows=8)

    def close(self) -> None:
        self.mgr.svc.stop(2.0)
        self._mp.undo()
        self._loop.close()


def _tamper(entry: Fused2LaneEntry, what: str, li: int, rows: int, body: bytes, out: bytes) -> bytes:
    """A dishonest provider (after the honest product): returns a modified reply for pair items."""
    from pgw.core.packing import pack_ring, unpack_ring

    it, op = entry.items[li], entry.ops[li]
    if it.k2 is None or li < 2:
        return out
    k, bpe = RING.k, RING.k // 8
    cut = rows * it.m1 * bpe
    h = unpack_ring(out[:cut], (rows, it.m1), k)
    z = unpack_ring(out[cut:], (rows, it.m2), k)
    mask = (1 << k) - 1
    x = unpack_ring(body[: rows * it.n1 * bpe], (rows, it.n1), k)
    r = unpack_ring(body[rows * it.n1 * bpe:], (rows, it.m1), k)
    from pgw.core.ring64 import pair_reference

    if what == "H":
        h = h.clone()
        h[0, 1] = (h[0, 1] + 1) & mask
    elif what == "H_consistent":
        h = h.clone()
        h[0, 1] = (h[0, 1] + 5) & mask
        z = pair_reference(k, op.w2.cpu(), h)[0]  # type: ignore[union-attr]
    elif what == "Z":
        z = z.clone()
        z[0, 2] = (z[0, 2] - (1 << 20)) & mask
    elif what == "Z_skip_residual":
        z = pair_reference(k, op.w2.cpu(), (h - r) & mask)[0]  # type: ignore[union-attr]
    elif what == "no_hmul":
        h, z = pair_reference(k, op.w1.cpu(), x, op.w2.cpu(), r, None)  # type: ignore[assignment]
    return pack_ring(h, k) + pack_ring(z, k)


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    import shutil
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="pgwf2"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_item_table_and_fingerprint(weights: Fused2Weights, provider_weights: Fused2Weights,
                                    entry: Fused2LaneEntry) -> None:
    L = weights.num_layers
    items = fused2_items(weights)
    assert len(items) == 2 * L + 1
    assert [it.kind for it in items] == ["f2single"] + ["f2pair"] * (2 * L - 1) + ["f2single"]
    assert items[1].name == "0.o+0.gate_up/f2" and items[-1].name == f"{L - 1}.down"
    assert entry.table()[1] == (items[1].n1 + items[1].m1, items[1].m1 + items[1].m2, "f2pair")
    # the provider-only file gives the same lane fingerprint; any weight or multiplier change moves it
    assert fused2_fingerprint(weights, RING) == entry.fingerprint
    w2 = Fused2Weights.__new__(Fused2Weights)
    w2.__dict__.update(provider_weights.__dict__)
    import dataclasses

    w2.o = [dataclasses.replace(w2.o[0], hmul=w2.o[0].hmul + 1)] + w2.o[1:]  # type: ignore[operator]
    assert fused2_fingerprint(w2, RING) != entry.fingerprint
    assert fused2_fingerprint(weights, RingParams(48, 12, 45)) != entry.fingerprint


def test_entry_ops_match_reference(entry: Fused2LaneEntry) -> None:
    from pgw.core.ring64 import check_op

    for li in range(3):
        for rows in (1, 3):
            assert check_op(entry.ops[li], rows, seed=li + rows) is None
    assert entry.self_check() == []


@pytest.mark.parametrize("shm", [False, True], ids=["uds", "shm"])
def test_lane_pair_items_masked_equals_plain(qwen_pack: InMemoryPack, weights: Fused2Weights,
                                             entry: Fused2LaneEntry, sock_dir: Path, shm: bool) -> None:
    ref = _run(_model(qwen_pack, Fused2PlainBackend(RING), weights))
    lane = _Lane(entry, sock_dir, shm=shm)
    try:
        prov = lane.provider(weights)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING))
        got = _run(_model(qwen_pack, be, weights))
        for a, b in zip(ref, got, strict=True):
            assert torch.equal(a, b)
        L = weights.num_layers
        assert be.f2.trips == 4 * 2 * (2 * L + 1)
        # 11-row prefill items went as 8 + 3 rows: one more lane call per trip of that pass
        assert prov.calls == be.f2.trips + (2 * L + 1)
        assert entry.calls >= prov.calls
        prov.close()
    finally:
        lane.close()


def test_provider_visible_bytes_identical_across_transports(qwen_pack: InMemoryPack, weights: Fused2Weights,
                                                            entry: Fused2LaneEntry, sock_dir: Path) -> None:
    seen = []
    for shm in (False, True):
        d = sock_dir / ("s" if shm else "u")
        d.mkdir()
        lane = _Lane(entry, d, shm=shm)
        try:
            prov = lane.provider(weights)
            prov.record = []
            seed = bytes(range(32))
            be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING, seed=seed))
            _run(_model(qwen_pack, be, weights), steps=1)
            seen.append(prov.record)
            prov.close()
        finally:
            lane.close()
    assert seen[0] == seen[1] and len(seen[0]) > 0
    for name, rows, body in seen[0]:
        assert len(body) == rows * dict((it.name, it.n) for it in entry.items)[name] * RING.k // 8


@pytest.mark.parametrize("what", ["H", "H_consistent", "Z", "Z_skip_residual", "no_hmul"])
def test_lane_tamper_detected(qwen_pack: InMemoryPack, weights: Fused2Weights, entry: Fused2LaneEntry,
                              sock_dir: Path, what: str) -> None:
    """The service itself returns a wrong pair reply (from the third item on): the gateway refuses it."""
    lane = _Lane(entry, sock_dir, shm=False, evil=what)
    try:
        prov = lane.provider(weights)
        be = Fused2MaskedBackend(RING, prov, Fused2MaskSource(weights, RING))
        with pytest.raises(IntegrityError):
            _run(_model(qwen_pack, be, weights))
        prov.close()
    finally:
        lane.close()


def test_refusal_raises_and_unknown_item(weights: Fused2Weights, entry: Fused2LaneEntry, sock_dir: Path) -> None:
    lane = _Lane(entry, sock_dir, shm=False)
    try:
        prov = lane.provider(weights)
        with pytest.raises(Exception):  # noqa: B017 - a key not in the table
            prov.single("nope", torch.zeros((1, 4), dtype=torch.int64))
        prov.close()
    finally:
        lane.close()


def test_segments_equal_in_process_masks(weights: Fused2Weights) -> None:
    seed = bytes(range(32))
    fp = fused2_fingerprint(weights, RING)
    fac = Fused2MaskFactory(weights, RING, seed=seed, lane_fingerprint=fp)
    gw = Fused2SegmentMasks(gateway_tables(weights), RING, fac.check_blob(), lane_fingerprint=fp)
    ref = Fused2MaskSource(weights, RING, seed=seed)
    gw.add_segment(fac.segment(3))
    gw.add_segment(fac.segment(4))
    assert gw.stocked_rows() == 7
    w = weights
    for rows in (2, 4, 1):  # spans the two segments
        for l1, l2 in ((w.qkv[0], None), (w.o[0], w.gu[0])):
            t = Fused2Trip("single" if l2 is None else "pair", l1, torch.zeros((rows, l1.n), dtype=torch.int64),
                           RING.cap, l2, None if l2 is None else torch.zeros((rows, l1.m), dtype=torch.int64))
            a, b = ref.take(t), gw.take(t)
            for f in ("r_x", "wr_x", "v", "wv"):
                x, y = getattr(a, f), getattr(b, f)
                assert (x is None and y is None) or torch.equal(x, y)
    for lin in (w.o[0], w.gu[0]):
        assert (gw.check_vectors(lin).uw == ref.check_vectors(lin).uw).all()
    with pytest.raises(Exception):  # noqa: B017 - exhausted: no reuse
        gw.take(Fused2Trip("single", w.qkv[0], torch.zeros((1, w.qkv[0].n), dtype=torch.int64), RING.cap))
    bad = bytearray(fac.segment(1))
    bad[100] ^= 1
    with pytest.raises(IntegrityError):
        gw.add_segment(bytes(bad))
    with pytest.raises(ProtocolError):  # a skipped segment (rows out of order)
        fac.segment(1)
        gw.add_segment(fac.segment(1))


def test_weight_free_gateway_over_lane_one_slot_per_token(qwen_pack: InMemoryPack, weights: Fused2Weights,
                                                          entry: Fused2LaneEntry, sock_dir: Path) -> None:
    """Gateway with no int8 matrices (stride-0 zero views), masks from factory segments, provider on the lane:
    bit-identical to the plain reference, and exactly one slot per decoded token (one row of every trip key)."""
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
        model = _model(qwen_pack, be, gw_w)
        v = model.vocab_size
        g = torch.Generator().manual_seed(1)
        prompt = torch.randint(0, v, (6,), generator=g)
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
        prov.close()
    finally:
        lane.close()


def test_entry_derived_from_the_providers_pack(qwen_pack: InMemoryPack, entry: Fused2LaneEntry) -> None:
    """The service derives the same public weight set (and lane fingerprint) from its own pack as the pack-time
    file the gateway pins."""
    derived = Fused2LaneEntry.from_pack(qwen_pack, 8, RING, device="cpu", max_rows=2)
    assert derived.fingerprint == entry.fingerprint
    assert derived.names() == entry.names()
