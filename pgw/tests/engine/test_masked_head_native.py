"""The masked LM head on real store reservations: the native per-trip path (``PGW_TRIP_NATIVE``) serves the head
trip from a lease like every linear, for the ring check and the prime-field check (``p31``), and the logits equal
the plaintext reference with the same int8 head bit for bit. Segments are made by the real factory for the derived
fingerprint and carry the head linear; the encrypted factory's hint for the head pack extends the base hint."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import native
from pgw.core.ring import DEFAULT_RING
from pgw.engine import InMemoryPack, LocalComputeClient, MaskedBackend, PlainBackend, TransformerModel
from pgw.engine.masked_head import MaskedHeadPack, enable_masked_head
from pgw.engine.model import SequenceInput
from pgw.gateway.masks import WindowedReservation

SLOTS = 400


@pytest.fixture(scope="module")
def head_pack(qwen_pack: InMemoryPack) -> MaskedHeadPack:
    return MaskedHeadPack(qwen_pack)


def _store(pack: Any, check_field: str | None) -> tuple[Any, Path]:
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter

    d = Path(tempfile.mkdtemp(prefix="pgw-mh-"))
    store = MaskStore.open(d / "store", FileKeyProvider(d / "kek", create=True), fsync=False,
                           counter=FileMonotonicCounter(d / "ctr", fsync=False))
    gen = MaskGenerator(pack, DEFAULT_RING, check_field=check_field)
    for _ in range(2):
        gen.generate_into_store(store, SLOTS // 2, chunk_slots=8, store_chunk_slots=8, factory_id="t")
    return store, d


def _drive(model: TransformerModel, session: Any) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    ids = [int(t) for t in torch.randint(0, 250, (21,), generator=g)]
    out: dict[str, torch.Tensor] = {}
    cache = model.new_cache(128)
    out["prefill"] = model.forward_batch([SequenceInput(ids, cache, session, all_logits=True)])[0]
    for i in range(6):
        out[f"d{i}"] = model.forward_batch([SequenceInput([10 + i], cache, session)])[0]
    v = model.forward_batch([SequenceInput([40, 41, 42], cache, session, last_logits=3, serial_tail=2)])[0]
    for i in range(3):
        out[f"v{i}"] = v[i]
    return out


def _reference(pack: MaskedHeadPack) -> dict[str, torch.Tensor]:
    m = TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))
    enable_masked_head(m)
    try:
        return _drive(m, None)
    finally:
        m.close()


@pytest.mark.skipif(not native.native_available("TripLease", "trip_prepare_into"),
                    reason="pgw_fast native trip lease not built")
@pytest.mark.parametrize("check_field", [None, "p31"], ids=["ring", "p31"])
def test_native_trip_path_exact(head_pack: MaskedHeadPack, monkeypatch: pytest.MonkeyPatch,
                                check_field: str | None) -> None:
    monkeypatch.setenv("PGW_TRIP_FAST", "1")
    monkeypatch.setenv("PGW_TRIP_NATIVE", "1")
    ref = _reference(head_pack)
    store, d = _store(head_pack, check_field)
    inv = store.inventory(head_pack.fingerprint, DEFAULT_RING)
    assert sum(i.slots_available for i in inv) == SLOTS  # segments are keyed by the derived fingerprint
    assert sum(i.slots_available for i in store.inventory(head_pack.base_fingerprint, DEFAULT_RING)) == 0
    client = LocalComputeClient(head_pack, record=True)
    be = MaskedBackend(head_pack, None, client, DEFAULT_RING, head_pack.fingerprint, trip_engine=True,
                       sync_trips=True, prime_check=check_field == "p31")
    be.defer_min_out = 0
    be.freivalds_rate = 1.0
    model = TransformerModel(head_pack, be)
    enable_masked_head(model)
    model.sync_decode = True
    served: list[str] = []
    real = be._prepare_trip_lease

    def spy(call: Any) -> Any:
        p = real(call)
        if p is not None:
            served.append(call.id.key)
        return p

    be._prepare_trip_lease = spy  # type: ignore[method-assign]

    def reserve(n: int) -> Any:
        return store.reserve(head_pack.fingerprint, DEFAULT_RING, n)

    res = WindowedReservation(reserve(8), reserve, cap=60, window=4)
    try:
        assert be.trip_native
        got = _drive(model, res)
        be.join_checks()
        assert be.stats.snapshot()["integrity_failures"] == 0
    finally:
        res.close()
        model.close()
        store.close()
        shutil.rmtree(d, ignore_errors=True)
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    assert head_pack.head_id.key in served  # the head trip took the native lease path


def test_segments_carry_the_head_and_refuse_the_base(head_pack: MaskedHeadPack, qwen_pack: InMemoryPack) -> None:
    from pgw.factory import MaskGenerator

    gen = MaskGenerator(head_pack, DEFAULT_RING)
    assert gen.fingerprint == head_pack.fingerprint
    keys = [s.id.key for s in gen.specs] if hasattr(gen, "specs") else []
    assert head_pack.head_id.key in keys
    assert MaskGenerator(qwen_pack, DEFAULT_RING).fingerprint == qwen_pack.fingerprint != gen.fingerprint


def test_factory_settings_wrap(tiny_pack_dir: Path) -> None:
    from pgw.factory.config import FactorySettings
    from pgw.factory.runtime import open_packs

    off = open_packs(FactorySettings(pack_path=tiny_pack_dir))
    assert not isinstance(off[0], MaskedHeadPack)
    on = open_packs(FactorySettings(pack_path=tiny_pack_dir, masked_head=True))
    assert isinstance(on[0], MaskedHeadPack) and on[0].base_fingerprint == off[0].fingerprint


def test_hint_extension_equals_a_full_build(tiny_pack_dir: Path, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from pgw.hemask import backend as hb
    from pgw.modelpack import ModelPack

    head_pack = MaskedHeadPack(ModelPack.open(tiny_pack_dir))

    base = hb.HintStore.build(tmp_path / "a", head_pack.base, rank=8)
    ext = hb.HintStore.open_or_build(tmp_path / "a", head_pack, rank=8)  # extends: base files linked
    assert ext.meta["extends"] == head_pack.base_fingerprint and ext.a_seed == base.a_seed
    monkeypatch.setattr(hb, "new_seed", lambda: base.a_seed)
    full = hb.HintStore.build(tmp_path / "b", head_pack, rank=8)
    assert set(ext.meta["row_norms"]) == set(full.meta["row_norms"])
    for s in head_pack.linear_specs():
        assert np.array_equal(ext.hint(s.id.key), full.hint(s.id.key)), s.id.key
        assert ext.meta["row_norms"][s.id.key] == full.meta["row_norms"][s.id.key]
    assert hb.HintStore.open(tmp_path / "a", head_pack.fingerprint, 8) is not None
