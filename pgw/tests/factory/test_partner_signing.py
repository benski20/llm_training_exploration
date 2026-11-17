"""Partner factory signs exported segments; the customer store accepts only trusted signers (SPEC 7.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from pgw.core import RING_K40_A12, PGWError
from pgw.factory import ExportSink, MaskGenerator

from .conftest import FakePack

maskstore = pytest.importorskip("pgw.maskstore")


def _customer_store(tmp_path: Path, recipient: Any, **overrides: Any) -> Any:
    kp = maskstore.FileKeyProvider(tmp_path / "customer" / "kek", create=True)
    return maskstore.MaskStore.open(tmp_path / "customer" / "store", kp, recipient=recipient, **overrides)


def test_signed_export_imports_with_trusted_key(tmp_path: Path) -> None:
    recipient = maskstore.RecipientKeyPair.generate(tmp_path / "customer" / "recipient.key")
    key = maskstore.PartnerSigningKey.generate(tmp_path / "partner-sign.key")
    pack = FakePack()
    store = _customer_store(tmp_path, recipient, trusted_partner_keys=[key.public_bytes().hex()])
    sink = ExportSink(tmp_path / "partner", recipient.public_bytes(), signing_key=key,
                      destination_store_id=store.store_id)
    seg = MaskGenerator(pack, RING_K40_A12).generate_into_store(sink, 4, checks=2, chunk_slots=4)
    [f] = sink.segments()
    assert f.segment_id == seg
    sig = f.path.with_name(f.path.name + ".sig")
    assert sig.is_file()
    assert not list(sink.staging_root.rglob("*.sig"))  # the sidecar moved with the segment
    try:
        store.import_segment(f.path)
        assert store.available_slots(pack.fingerprint, RING_K40_A12) == 4
    finally:
        store.close()


def test_unsigned_export_refused_when_signatures_required(tmp_path: Path) -> None:
    recipient = maskstore.RecipientKeyPair.generate(tmp_path / "customer" / "recipient.key")
    key = maskstore.PartnerSigningKey.generate(tmp_path / "partner-sign.key")
    sink = ExportSink(tmp_path / "partner", recipient.public_bytes())  # no signing key: warning, unsigned
    MaskGenerator(FakePack(), RING_K40_A12).generate_into_store(sink, 4, chunk_slots=4)
    [f] = sink.segments()
    assert not f.path.with_name(f.path.name + ".sig").exists()
    store = _customer_store(tmp_path, recipient, trusted_partner_keys=[key.public_bytes().hex()])
    try:
        with pytest.raises(PGWError):
            store.import_segment(f.path)
    finally:
        store.close()


def test_recover_signs_unsigned_outbox_segment(tmp_path: Path) -> None:
    recipient = maskstore.RecipientKeyPair.generate(tmp_path / "customer" / "recipient.key")
    sink = ExportSink(tmp_path / "partner", recipient.public_bytes())
    MaskGenerator(FakePack(), RING_K40_A12).generate_into_store(sink, 4, chunk_slots=4)
    sink._staging.close()
    key = maskstore.PartnerSigningKey.generate(tmp_path / "partner-sign.key")
    sink2 = ExportSink(tmp_path / "partner", recipient.public_bytes(), signing_key=key)
    sink2.recover()
    [f] = sink2.segments()
    assert f.path.with_name(f.path.name + ".sig").is_file()


def test_partner_key_cli(tmp_path: Path) -> None:
    import json

    from pgw.cli.main import cli

    r = CliRunner().invoke(cli, ["maskstore", "partner-key", "create", str(tmp_path / "p.key"),
                                 "--public-out", str(tmp_path / "p.pub")])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert (tmp_path / "p.pub").read_text().strip() == out["public_key_hex"]
    assert (tmp_path / "p.key").stat().st_mode & 0o077 == 0
    r2 = CliRunner().invoke(cli, ["maskstore", "partner-key", "show", str(tmp_path / "p.key")])
    assert json.loads(r2.stdout) == out
    r3 = CliRunner().invoke(cli, ["maskstore", "partner-key", "create", str(tmp_path / "p.key")])
    assert r3.exit_code == 1
    # factory option is wired (only valid in partner mode)
    from pgw.factory.cli import factory_group

    r4 = CliRunner().invoke(factory_group, ["generate", "--partner-key", str(tmp_path / "p.key")])
    assert r4.exit_code != 0 and "partner mode" in r4.output


def test_export_header_names_destination_and_auto_checks(tmp_path: Path) -> None:
    """Security review: exported segments are bound to one store; default checks meet 20 bits."""
    from pgw.core.freivalds import soundness_bits

    recipient = maskstore.RecipientKeyPair.generate(tmp_path / "customer" / "recipient.key")
    dest = "0123456789abcdef" * 2
    sink = ExportSink(tmp_path / "partner", recipient.public_bytes(), destination_store_id=dest)
    pack = FakePack()
    gen = MaskGenerator(pack, RING_K40_A12)
    gen.generate_into_store(sink, 4, chunk_slots=4)  # checks=None -> auto
    [f] = sink.segments()
    with maskstore.SegmentFile(f.path) as seg:
        h = seg.header
    assert h.destination == dest
    assert h.checks == gen.checks_for()
    assert all(soundness_bits(RING_K40_A12, s.in_features, h.checks) >= 20 for s in pack.linear_specs())
