"""Partner export end to end with the real MaskStore and real HPKE (SPEC 7.7).

Partner: :class:`ExportSink` -> HPKE-sealed file in the outbox. Customer: ``import_segment`` with
its :class:`RecipientKeyPair`, reserve, unmask, Freivalds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, MaskReuseError, center, freivalds_verify, mask_activations, ringmm_reference
from pgw.factory import ExportSink, FactoryScheduler, FactorySettings, MaskGenerator

from .conftest import FakePack

maskstore = pytest.importorskip("pgw.maskstore")


@pytest.fixture
def recipient(tmp_path: Path) -> Any:
    return maskstore.RecipientKeyPair.generate(tmp_path / "customer" / "recipient.key")


def test_export_import_round_trip(tmp_path: Path, recipient: Any) -> None:
    pack = FakePack()
    kp = maskstore.FileKeyProvider(tmp_path / "customer" / "kek", create=True)
    signer = maskstore.PartnerSigningKey.generate(tmp_path / "partner-sign.key")
    store = maskstore.MaskStore.open(tmp_path / "customer" / "store", kp, recipient=recipient,
                                     trusted_partner_keys=[signer.public_bytes().hex()])
    sink = ExportSink(tmp_path / "partner", recipient.public_bytes(), signing_key=signer,
                      destination_store_id=store.store_id)
    assert sink.kek_id == recipient.kek_id
    gen = MaskGenerator(pack, RING_K40_A12)
    seg = gen.generate_into_store(sink, 6, checks=2, chunk_slots=4, factory_id="partner-1")
    files = sink.segments()
    assert [f.segment_id for f in files] == [seg]
    assert files[0].kek_id == recipient.kek_id
    assert sink.available_slots(pack.fingerprint, RING_K40_A12) == 6
    # nothing remains readable in staging
    assert not list(sink.staging_root.rglob("*.pgwm"))

    try:
        store.import_segment(files[0].path)
        with store.reserve(pack.fingerprint, RING_K40_A12, 6) as res:
            for spec in pack.linear_specs():
                w = pack.weights[spec.id.key]
                r, wr = res.take(spec.id, 6)
                assert wr.tolist() == ringmm_reference(r, w, 40).tolist()
                u, uw = res.checks(spec.id)
                xq = torch.randint(-100, 100, (6, spec.in_features), dtype=torch.int64)
                x_hat = mask_activations(xq, r, RING_K40_A12)
                y_hat = torch.from_numpy(ringmm_reference(x_hat, w, 40))
                freivalds_verify(x_hat, y_hat, u, uw, RING_K40_A12)
                assert torch.equal(center(y_hat - wr, RING_K40_A12), xq @ w.to(torch.int64).t())
        # importing the same file twice is mask reuse
        with pytest.raises(MaskReuseError):
            store.import_segment(files[0].path)
    finally:
        store.close()


def test_export_scheduler_counts_outbox(tmp_path: Path, recipient: Any) -> None:
    pack = FakePack()
    sink = ExportSink(tmp_path / "partner", recipient.public_bytes())
    g = MaskGenerator(pack, RING_K40_A12)
    s = FactorySettings(pack_path=tmp_path, output="export", export_dir=tmp_path / "partner", segment_slots=8,
                        chunk_slots=8, low_water_slots=10, target_slots=20)
    with FactoryScheduler(s, sink, {g.key: g}, metrics=False) as sch:
        assert len(sch.run_once()) == 3
        assert sch.plan() == []
    assert sink.available_slots(pack.fingerprint, RING_K40_A12) == 24


def test_recover_publishes_segment_stranded_in_staging(tmp_path: Path, recipient: Any) -> None:
    """Crash after the staging commit but before the move: the next start publishes it once."""
    pack = FakePack()
    sink = ExportSink(tmp_path / "partner", recipient.public_bytes())
    gen = MaskGenerator(pack, RING_K40_A12)
    seg = gen.generate_into_store(sink._staging, 4, chunk_slots=4)  # bypasses the publish step
    assert sink.segments() == []
    sink._staging.close()

    sink2 = ExportSink(tmp_path / "partner", recipient.public_bytes())
    s = FactorySettings(pack_path=tmp_path, output="export", export_dir=tmp_path / "partner", segment_slots=4,
                        chunk_slots=4, low_water_slots=1, target_slots=4)
    with FactoryScheduler(s, sink2, {gen.key: gen}, metrics=False) as sch:
        assert sch.run_once() == []  # recovered segment already covers the target
    assert [f.segment_id for f in sink2.segments()] == [seg]
    assert sink2.recover() == []  # idempotent: nothing left in staging


def test_small_order_recipient_key_rejected(tmp_path: Path) -> None:
    from pgw.core import ConfigError
    from pgw.factory.sinks import load_recipient_public_key

    for bad in (bytes(32), (1).to_bytes(32, "little")):
        with pytest.raises(ConfigError):
            ExportSink(tmp_path / "x", bad)
        f = tmp_path / "pub.hex"
        f.write_text(bad.hex())
        with pytest.raises(ConfigError):
            load_recipient_public_key(f)
    with pytest.raises(ConfigError):
        load_recipient_public_key(tmp_path / "missing")
