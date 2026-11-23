"""fused2 pair masks from the encrypted mask store (opt-in): format, factory, gateway, single use.

* format: ``schedule = "fused2"`` / ``check_field = "p61"`` headers round-trip, standard headers keep their bytes,
  readers without the field refuse them, a standard reservation refuses fused2 segments;
* factory (:mod:`pgw.factory.fused2_gen`): exact products and p61 words vs independent references, start-up check,
  the per-chunk self-check aborts a segment on a wrong product;
* gateway: masked == plain bit for bit through store reservations (Python path and the native driver over the
  lane), passes that cross a segment extent, a growing incremental reservation, one slot per token, no weights;
* single use: tampering after a take, a crashed gateway process, burn on close.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ConfigDict, ValidationError, create_model

from pgw.core.errors import ConfigError, IntegrityError, MaskStoreError
from pgw.core.prng import SeedExpander
from pgw.core.ring import RingParams
from pgw.core.ringmm import ringmm_reference
from pgw.engine import InMemoryPack, SequenceInput, TransformerModel
from pgw.engine.fused2 import (
    P61,
    Fused2MaskedBackend,
    Fused2PlainBackend,
    Fused2Provider,
    Fused2Weights,
    enable_fused2,
    migration_multipliers,
)
from pgw.engine.fused2_masks import gateway_tables
from pgw.engine.fused2_store import Fused2StoreMasks
from pgw.factory.fused2_gen import Fused2MaskGenerator, fused2_specs
from pgw.maskstore import MaskStore, SegmentFile, SegmentHeader, StaticKeyProvider
from pgw.maskstore.format import encode_header
from pgw.maskstore.fused2 import p61_check_vectors
from pgw.service.fused2_lane import fused2_fingerprint

RING = RingParams(56, 12, 53)


@pytest.fixture(scope="module")
def weights(qwen_pack: InMemoryPack) -> Fused2Weights:
    return Fused2Weights(qwen_pack, multipliers=migration_multipliers(qwen_pack, 8))


@pytest.fixture(scope="module")
def gen(weights: Fused2Weights) -> Fused2MaskGenerator:
    return Fused2MaskGenerator(weights, RING, chunk_rows=3)


@pytest.fixture
def kek() -> bytes:
    return os.urandom(32)


@pytest.fixture
def store(tmp_path: Path, kek: bytes) -> Any:
    s = MaskStore.open(tmp_path / "store", StaticKeyProvider(kek), low_water_slots=0)
    yield s
    s.close()


def _model(pack: InMemoryPack, backend: Any, w: Fused2Weights, *, native_ops: bool = False) -> TransformerModel:
    m = TransformerModel(pack, backend)
    if native_ops:
        m.set_float_ops("native")
    enable_fused2(m, w, RING)
    return m


def _gen(model: TransformerModel, session: Any = None, steps: int = 3, prompt_len: int = 5,
         seed: int = 7) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    v = model.vocab_size
    cache = model.new_cache(64)
    out = model.forward_batch([SequenceInput(torch.randint(0, v, (prompt_len,), generator=g), cache,
                                             all_logits=True, session=session)])
    for _ in range(steps):
        tok = torch.tensor([int(out[-1].reshape(-1, v)[-1].argmax())])
        out += model.forward_batch([SequenceInput(tok, cache, session=session)])
    return out


# ---- format ---------------------------------------------------------------------------------------------------
def test_header_round_trip_and_old_readers_refuse(store: Any, gen: Fused2MaskGenerator) -> None:
    sid = gen.generate_into_store(store, 4, store_chunk_slots=2)
    path = next(store.root.rglob(f"{sid}.pgwm"))
    with SegmentFile(path) as seg:
        h = seg.header
        assert h.schedule == "fused2" and h.check_field == "p61" and h.ring == RING.name
        assert [lin.id for lin in h.linears] == [s.id for s in fused2_specs(gen.w)]
        assert h.fingerprint == fused2_fingerprint(gen.w, RING)
        raw = seg.header_raw
        assert SegmentHeader.model_validate(json.loads(raw)) == h and encode_header(h) == raw
        assert seg.layout.check_bpe == 8  # p61 words are 8 bytes, not K56 elements
    # a reader without the field (today's model minus `schedule`) refuses the header
    fields = {k: (f.annotation, f) for k, f in SegmentHeader.model_fields.items() if k != "schedule"}
    old = create_model("OldHeader", __config__=ConfigDict(extra="forbid"), **fields)  # type: ignore[call-overload]
    with pytest.raises(ValidationError):
        old.model_validate(json.loads(raw))
    # standard headers keep their exact bytes (no schedule key)
    std = h.model_copy(update={"schedule": None, "check_field": None, "ring": "K32A8C24",
                               "linears": (h.linears[1],)})
    assert b"schedule" not in encode_header(std)
    # inconsistent headers are refused
    bad = json.loads(raw)
    bad["schedule"] = None
    with pytest.raises(ValidationError):  # p61 / fused2 ids without the schedule
        SegmentHeader.model_validate(bad)
    bad = json.loads(raw)
    bad["check_field"] = "p31"
    with pytest.raises(ValidationError):
        SegmentHeader.model_validate(bad)


def test_writer_refusals(store: Any, gen: Fused2MaskGenerator) -> None:
    fp = gen.fingerprint
    with pytest.raises((ConfigError, ValueError)):  # fused2 ids without the schedule
        store.writer(fp, RING, gen.specs, slots=2, checks=1)
    with pytest.raises(ConfigError):
        store.writer(fp, RING, gen.specs, slots=2, checks=1, check_field="p61")
    with pytest.raises(ConfigError):
        store.writer(fp, RING, gen.specs, slots=2, checks=1, schedule="fused2")  # no p61
    with pytest.raises((ConfigError, ValueError)):  # no cap / unsound p61
        store.writer(fp, "K56A12", gen.specs, slots=2, checks=1, schedule="fused2", check_field="p61")


def test_standard_reservation_refuses_fused2_segments(store: Any, gen: Fused2MaskGenerator) -> None:
    gen.generate_into_store(store, 4)
    with pytest.raises(MaskStoreError):
        store.reserve(gen.fingerprint, RING, 2)  # schedule None
    # fail closed: the 2 slots stay consumed (never handed out)
    assert store.available_slots(gen.fingerprint, RING) == 2
    with store.reserve(gen.fingerprint, RING, 2, schedule="fused2") as res:
        assert res.schedule == "fused2" and res.extents[0].start == 2


# ---- factory ----------------------------------------------------------------------------------------------
def test_segment_content_is_exact(store: Any, gen: Fused2MaskGenerator, weights: Fused2Weights) -> None:
    """Rows, products and p61 words of a stored segment against independent references (Python integers)."""
    seed2 = os.urandom(32)
    gen.generate_into_store(store, 3, seed=seed2, store_chunk_slots=2)
    assert gen.last_stats.rows_checked > 0
    with store.reserve(gen.fingerprint, RING, 3, schedule="fused2") as res2:
        assert res2.extents[0].count == 3
        exp = SeedExpander(seed2)
        for lin in weights.linears():
            n, r, wr, (token, u, uw) = res2.take_checked(lin.key, 3, True)
            assert n == 3 and token[2] == "p61"
            assert torch.equal(r, exp.ring_uniform_rows(lin.n, RING.k, 0, 3, "r", lin.key))
            assert np.array_equal(wr.numpy(), ringmm_reference(r, lin.w8, RING.k))
            u_ref = p61_check_vectors(seed2, 1, lin.m, lin.key).numpy()
            assert np.array_equal(np.asarray(u, dtype=np.int64), u_ref)
            hm = [1] * lin.m if lin.hmul is None else [int(x) for x in lin.hmul.tolist()]
            w = lin.w8.numpy()
            for j in (0, lin.n // 2, lin.n - 1):
                want = sum(int(a) * h * int(c) for a, h, c in zip(u_ref[0].tolist(), hm, w[:, j].tolist(),
                                                                    strict=True)) % P61
                assert int(uw[0, j]) == want


def test_startup_check_and_self_check_abort(store: Any, weights: Fused2Weights,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    g = Fused2MaskGenerator(weights, RING, self_check_rows=-1)
    rep = g.startup_check(rows=1, linears=0, uw_columns=2)
    assert rep["ok"] and len(rep["linears_checked"]) == len(g.specs)
    orig = g.products

    def flip(key: str, r: torch.Tensor) -> torch.Tensor:
        y = orig(key, r)
        if key == g.specs[5].id:
            y[0, 3] ^= 1 << 40  # one wrong bit in one product
        return y

    monkeypatch.setattr(g, "products", flip)
    with pytest.raises(IntegrityError):
        g.startup_check(rows=1, linears=0, uw_columns=1)
    with pytest.raises(IntegrityError):
        g.generate_into_store(store, 2)
    assert store.available_slots(g.fingerprint, RING) == 0  # aborted, never committed
    assert not list(store.root.rglob("*.pgwm"))


# ---- gateway: masked == plain through the store -----------------------------------------------------------------
def test_store_masks_equal_plain_one_slot_per_token(qwen_pack: InMemoryPack, store: Any, gen: Fused2MaskGenerator,
                                                    weights: Fused2Weights) -> None:
    ref = _gen(_model(qwen_pack, Fused2PlainBackend(RING), weights))
    for _ in range(3):  # three segments of 3 slots: the 5-row prompt crosses an extent (trips run in pieces)
        gen.generate_into_store(store, 3, store_chunk_slots=2)
    gw = gateway_tables(weights)  # the gateway holds no int8 matrix
    prov = Fused2Provider(weights, RING.k)
    be = Fused2MaskedBackend(RING, prov, weights=gw)
    res = store.reserve(gen.fingerprint, RING, 9, schedule="fused2")
    model = _model(qwen_pack, be, gw)
    v = model.vocab_size
    g = torch.Generator().manual_seed(7)
    cache = model.new_cache(64)
    got = model.forward_batch([SequenceInput(torch.randint(0, v, (5,), generator=g), cache, all_logits=True,
                                             session=res)])
    keys = [lin.key for lin in weights.linears()]
    assert all(res.taken(k) == 5 for k in keys)
    src = be.masks_for(res)
    assert isinstance(src, Fused2StoreMasks) and src.split_takes > 0
    for i in range(3):
        tok = torch.tensor([int(got[-1].reshape(-1, v)[-1].argmax())])
        got += model.forward_batch([SequenceInput(tok, cache, session=res)])
        assert all(res.taken(k) == 6 + i for k in keys)  # one slot per token
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    res.close()
    be.release(res)
    assert not src._checks  # owned check copies wiped


def test_incremental_windowed_reservation(qwen_pack: InMemoryPack, store: Any, gen: Fused2MaskGenerator,
                                          weights: Fused2Weights) -> None:
    from pgw.engine.fused2_store import open_fused2_session

    ref = _gen(_model(qwen_pack, Fused2PlainBackend(RING), weights), steps=5)
    for _ in range(4):
        gen.generate_into_store(store, 4)
    gw = gateway_tables(weights)
    w = open_fused2_session(store, gen.fingerprint, RING, gw, 5, cap=5 + 6, chunk=2, ahead=False)
    be = Fused2MaskedBackend(RING, Fused2Provider(weights, RING.k), weights=gw)
    got = _gen(_model(qwen_pack, be, gw), session=w, steps=5)
    for a, b in zip(ref, got, strict=True):
        assert torch.equal(a, b)
    assert w.slots == 5 + 6 and w.extensions >= 2  # the prompt + one chunk, then whole chunks
    used = [p.taken(weights.linears()[0].key) for p in w.parts]
    assert sum(used) == 10  # 5 prompt rows + 5 decode tokens: one slot per token
    assert w.retired >= 1  # exhausted parts closed while generating
    w.close()
    assert store.available_slots(gen.fingerprint, RING) == 16 - 11  # unused rows burned, never returned


# ---- single use under faults, crash, burn ----------------------------------------------------------------------
class _Evil(Fused2Provider):
    def __init__(self, *a: Any, when: int) -> None:
        super().__init__(*a)
        self.n = 0
        self.when = when

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, z = super().pair(k1, k2, r_hat, x_hat)
        self.n += 1
        if self.n == self.when:
            z = z.clone()
            z[0, 0] = (z[0, 0] + 1) & ((1 << 56) - 1)
        return h, z


def test_tamper_caught_rows_never_reused(qwen_pack: InMemoryPack, store: Any, gen: Fused2MaskGenerator,
                                         weights: Fused2Weights) -> None:
    gen.generate_into_store(store, 8)
    gw = gateway_tables(weights)
    be = Fused2MaskedBackend(RING, _Evil(weights, RING.k, when=3), weights=gw)
    res = store.reserve(gen.fingerprint, RING, 8, schedule="fused2")
    model = _model(qwen_pack, be, gw)
    with pytest.raises(IntegrityError):
        _gen(model, session=res, steps=0, prompt_len=2)
    keys = [lin.key for lin in weights.linears()]
    taken = {k: res.taken(k) for k in keys}
    assert max(taken.values()) == 2  # the failed pass's rows stay consumed
    # a retry continues with fresh rows (the cursors only move forward)
    out = model.forward_batch([SequenceInput(torch.tensor([1, 2]), model.new_cache(16), session=res)])
    assert out and all(res.taken(k) == taken[k] + 2 for k in keys)
    res.close()


_CHILD = r"""
import hashlib, os, sys, torch
from pgw.maskstore import MaskStore, StaticKeyProvider
root, kek, fp, ring, key = sys.argv[1], bytes.fromhex(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
s = MaskStore.open(root, StaticKeyProvider(kek), low_water_slots=0)
res = s.reserve(fp, ring, 3, schedule="fused2")
n, r, wr, chk = res.take_checked(key, 2, True)
print(hashlib.sha256(r.numpy().tobytes()).hexdigest(), flush=True)
os._exit(9)   # crash: no close, no burn bookkeeping
"""


def test_crash_never_reuses_and_close_burns(tmp_path: Path, store: Any, kek: bytes, gen: Fused2MaskGenerator,
                                            weights: Fused2Weights) -> None:
    gen.generate_into_store(store, 6)
    key = weights.linears()[0].key
    import pgw

    env = dict(os.environ, PYTHONPATH=str(Path(pgw.__file__).resolve().parents[1]))
    p = subprocess.run([sys.executable, "-c", _CHILD, str(store.root), kek.hex(), gen.fingerprint, RING.name, key],  # noqa: S603
                       capture_output=True, text=True, timeout=120, check=False, env=env)
    assert p.returncode == 9, p.stderr
    child_hash = p.stdout.strip()
    assert store.available_slots(gen.fingerprint, RING) == 3  # the crashed reservation's slots are gone
    res = store.reserve(gen.fingerprint, RING, 2, schedule="fused2")
    assert res.extents[0].start == 3
    _, r, _, _ = res.take_checked(key, 2, True)
    assert hashlib.sha256(r.numpy().tobytes()).hexdigest() != child_hash
    res.close()  # nothing else taken: burned
    assert store.available_slots(gen.fingerprint, RING) == 1
    with store.reserve(gen.fingerprint, RING, 1, schedule="fused2") as last:
        assert last.extents[0].start == 5
    with pytest.raises(Exception):
        store.reserve(gen.fingerprint, RING, 1, schedule="fused2")


# ---- the native driver over the lane, masks from the store ------------------------------------------------------
def test_native_driver_from_store(qwen_pack: InMemoryPack, store: Any, gen: Fused2MaskGenerator,
                                  weights: Fused2Weights, tmp_path: Path) -> None:
    fast = pytest.importorskip("pgw_fast")
    if not hasattr(fast, "Fused2Driver"):
        pytest.skip("pgw_fast without Fused2Driver")
    from pgw.engine.fused2_native import enable_fused2_native
    from pgw.service.fused2_lane import Fused2LaneEntry
    from tests.engine.test_fused2_lane import _Lane

    p = tmp_path / "prov.pt"
    weights.save(p, provider_only=True)
    entry = Fused2LaneEntry.from_weights(Fused2Weights.load(p), RING, device="cpu", max_rows=8)
    assert entry.fingerprint == gen.fingerprint  # segments are bound to the lane's weight set
    ref = _gen(_model(qwen_pack, Fused2PlainBackend(RING), weights, native_ops=True), steps=6)
    for _ in range(3):  # 4-slot segments: the 5-row prompt crosses an extent (Python path in pieces)
        gen.generate_into_store(store, 4)
    gw = gateway_tables(weights)
    sock = tmp_path / "s"
    sock.mkdir()
    lane = _Lane(entry, sock, shm=True)
    try:
        prov = lane.provider(gw)
        be = Fused2MaskedBackend(RING, prov, weights=gw)
        model = _model(qwen_pack, be, gw, native_ops=True)
        nd = enable_fused2_native(model)
        res = store.reserve(gen.fingerprint, RING, 12, schedule="fused2")
        got = _gen(model, session=res, steps=6)
        for a, b in zip(ref, got, strict=True):
            assert torch.equal(a, b)
        keys = [lin.key for lin in weights.linears()]
        assert all(res.taken(k) == 11 for k in keys)  # 5 + 6: one slot per token
        assert nd.totals[0] == 6  # every decode pass native (1 row never crosses an extent); the prompt did not
        assert nd._tokens  # check vectors of the current segment installed
        res.close()
        with pytest.raises(Exception):  # a closed session hands out nothing
            model.forward_batch([SequenceInput(torch.tensor([1]), model.new_cache(8), session=res)])
        nd.wipe_checks()
        assert not nd._tokens
        prov.close()
    finally:
        lane.close()
