"""End to end: real factory -> real mask store -> engine MaskedBackend; and the admin helpers.

Masked generation over masks from the store must be token-for-token identical to the plain
(local integer) model, and logits bit-identical (SPEC 9.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, RING_K48_A20, MaskExhaustedError, RingParams
from pgw.maskstore import MaskStore, MaskStoreSettings, RecipientKeyPair, StaticKeyProvider
from pgw.maskstore.admin import gc_store, import_files, init_store, open_store, store_status

IDS = [256, 72, 105, 33, 10, 65, 66]


@pytest.fixture
def factory_store(tmp_path: Path, tiny_pack: Any) -> Any:
    import os

    store = MaskStore.open(tmp_path / "store", StaticKeyProvider(os.urandom(32)), low_water_slots=0)
    yield store
    store.close()


def _fill(store: MaskStore, pack: Any, ring: RingParams, sizes: list[int], chunk_slots: int = 4) -> list[str]:
    from pgw.factory import MaskGenerator

    gen = MaskGenerator(pack, ring, pipeline=False)
    return [gen.generate_into_store(store, n, checks=2, chunk_slots=chunk_slots, factory_id="it") for n in sizes]


@pytest.mark.parametrize("ring", [RING_K40_A12, RING_K48_A20], ids=lambda r: r.name)
def test_masked_generation_from_store_matches_plain(factory_store: MaskStore, tiny_pack: Any, ring: RingParams) -> None:
    from pgw.engine import (
        ChatTokenizer,
        Generator,
        LocalComputeClient,
        MaskedBackend,
        PlainBackend,
        SamplingParams,
        TransformerModel,
    )

    # segment sizes force reservations to span segment extents mid-prompt and mid-decode
    _fill(factory_store, tiny_pack, ring, [5, 3, 9, 20])
    tok = ChatTokenizer.from_pack(tiny_pack)
    client = LocalComputeClient(tiny_pack, record=False)
    backend = MaskedBackend(tiny_pack, None, client, ring, tiny_pack.fingerprint)
    masked = TransformerModel(tiny_pack, backend)
    plain = TransformerModel(tiny_pack, PlainBackend(tiny_pack, ring))
    p = SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)
    reservations: list[Any] = []

    def factory(n: int) -> Any:
        res = factory_store.reserve(tiny_pack.fingerprint, ring, n)
        reservations.append(res)
        return res

    rm = Generator(masked, tok, session_factory=factory).generate(IDS, p)
    rp = Generator(plain, tok).generate(IDS, p)
    assert rm.token_ids == rp.token_ids
    assert len(reservations) == 1 and len(reservations[0].extents) >= 2
    assert reservations[0].closed  # the generator burns the reservation at the end
    assert backend.stats.snapshot()["freivalds_checked"] > 0

    # logits of a prefill are bit-identical
    res = factory_store.reserve(tiny_pack.fingerprint, ring, len(IDS))
    a = plain.forward(torch.tensor(IDS), plain.new_cache(16))
    b = masked.forward(torch.tensor(IDS), masked.new_cache(16), session=res)
    assert torch.equal(a, b)
    res.close()
    used = len(IDS) + p.max_tokens + len(IDS)
    assert factory_store.available_slots(tiny_pack.fingerprint, ring) == 37 - used
    with pytest.raises(MaskExhaustedError):
        factory_store.reserve(tiny_pack.fingerprint, ring, 37)
    client.close()


def test_factory_scheduler_refills_on_low_water(tmp_path: Path, tiny_pack: Any) -> None:
    import os

    from pgw.factory import FactoryScheduler, FactorySettings, MaskGenerator

    store = MaskStore.open(tmp_path / "store", StaticKeyProvider(os.urandom(32)), low_water_slots=8, target_slots=16)
    gen = MaskGenerator(tiny_pack, RING_K40_A12, pipeline=False)
    fs = FactorySettings(pack_path=tmp_path, segment_slots=8, chunk_slots=4, low_water_slots=8, target_slots=16,
                         state_dir=tmp_path / "fstate")
    events: list[Any] = []
    store.add_low_water_listener(events.append)
    with FactoryScheduler(fs, store, {gen.key: gen}, metrics=False) as sch:
        sch.run_once()
        assert store.available_slots(tiny_pack.fingerprint, RING_K40_A12) >= 16
        store.reserve(tiny_pack.fingerprint, RING_K40_A12, 10).close()
        assert events and events[-1].slots_available < 8
        sch.run_once()
        assert store.available_slots(tiny_pack.fingerprint, RING_K40_A12) >= 16
    store.close()


def test_partner_export_sink_to_customer_import(tmp_path: Path, tiny_pack: Any) -> None:
    from pgw.factory import ExportSink, MaskGenerator
    from pgw.maskstore import PartnerSigningKey

    rec = RecipientKeyPair.generate(tmp_path / "recipient.key")
    signer = PartnerSigningKey.generate(tmp_path / "partner-sign.key")
    kf = tmp_path / "kek"
    store_id = init_store(tmp_path / "customer", kf, recipient_key_file=tmp_path / "recipient.key")["store_id"]
    sink = ExportSink(tmp_path / "partner", rec.public_bytes(), signing_key=signer, destination_store_id=store_id)
    gen = MaskGenerator(tiny_pack, RING_K40_A12, pipeline=False)
    gen.generate_into_store(sink, 6, chunk_slots=4)
    files = sorted((tmp_path / "partner" / "outbox").glob("*.pgwm"))
    assert len(files) == 1
    s = MaskStoreSettings(root=tmp_path / "customer", key_file=kf, recipient_key_file=tmp_path / "recipient.key",
                          low_water_slots=0, trusted_partner_keys=(signer.public_bytes().hex(),))
    out = import_files(files, s)  # source and signature are removed by default after a successful import
    assert out[0]["ok"] and not files[0].exists() and not files[0].with_name(files[0].name + ".sig").exists()
    again = import_files([tmp_path / "missing.pgwm"], s)
    assert not again[0]["ok"] and again[0]["code"] == "mask_store"
    st = store_status(s)
    assert st["inventory"][0]["slots_available"] == 6 and st["generation"] == 0
    with open_store(s) as store:
        store.reserve(tiny_pack.fingerprint, RING_K40_A12, 6).close()
    assert gc_store(s) == 0  # auto-shredded on close already
    assert store_status(s)["segments_by_status"] == {"shredded": 1}
