"""Hardening review: partner signatures, cross-instance leases, import cleanup, ring checks."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, ConfigError, MaskExhaustedError, MaskStoreError
from pgw.maskstore import (
    KeyProvider,
    MaskStore,
    PartnerSigningKey,
    RecipientKeyPair,
    SegmentFile,
    TrustedPartners,
    WrappedKey,
    hpke_wrap_for_recipient,
    sign_segment,
)
from pgw.maskstore.partner import SIGNATURE_SUFFIX, read_signature
from pgw.maskstore.store import _HOST, _is_live

from .conftest import FP, ToyModel, write_segment


class _WrapOnly(KeyProvider):
    """Partner-side provider: seals DEKs to the customer, cannot unwrap."""

    def __init__(self, pub: bytes):
        self.pub = pub

    @property
    def kek_id(self) -> str:
        return hpke_wrap_for_recipient(b"\0" * 32, self.pub).kek_id

    def wrap(self, dek: bytes) -> WrappedKey:
        return hpke_wrap_for_recipient(dek, self.pub)

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        raise MaskStoreError("wrap-only")


def _partner_segments(tmp_path: Path, model: ToyModel, rec: RecipientKeyPair, n: int = 1, slots: int = 4) -> list[Path]:
    partner = MaskStore.open(tmp_path / "partner", _WrapOnly(rec.public_bytes()), low_water_slots=0)
    out = []
    for _ in range(n):
        sid = write_segment(partner, model, slots, chunk_slots=4)
        out.append(partner.export_segment(sid, tmp_path / "outbox"))
    partner.close()
    return out


@pytest.fixture
def rec(tmp_path: Path) -> RecipientKeyPair:
    return RecipientKeyPair.generate(tmp_path / "rec.key")


@pytest.fixture
def signer(tmp_path: Path) -> PartnerSigningKey:
    return PartnerSigningKey.generate(tmp_path / "partner-sign.key")


# ---- partner signatures (SPEC 7.7: only an independent, authenticated partner may deliver) -------
def test_signed_partner_segment_imports_with_trusted_key(
    tmp_path: Path, open_store: Any, model: ToyModel, rec: RecipientKeyPair, signer: PartnerSigningKey
) -> None:
    (seg,) = _partner_segments(tmp_path, model, rec)
    sig_path = sign_segment(seg, signer)
    assert sig_path.name.endswith(SIGNATURE_SUFFIX)
    parsed = read_signature(seg)
    assert parsed is not None and parsed.key_id == signer.key_id
    store = open_store(trusted_partner_keys=(signer.public_bytes().hex(),), recipient=rec)
    assert store.trusted_partners.key_ids == (signer.key_id,)
    sid = store.import_segment(seg)
    with store.reserve(FP, RING_K40_A12, 4) as res:
        r, wr = res.take("0.qkv", 4)
        assert torch.equal(wr, model.wr("0.qkv", r, RING_K40_A12))
    assert sid == parsed.segment_id


def test_unsigned_or_untrusted_partner_segments_are_refused(
    tmp_path: Path, open_store: Any, model: ToyModel, rec: RecipientKeyPair, signer: PartnerSigningKey
) -> None:
    a, b, c = _partner_segments(tmp_path, model, rec, n=3)
    store = open_store(trusted_partner_keys=(base64.b64encode(signer.public_bytes()).decode(),), recipient=rec)
    with pytest.raises(MaskStoreError, match="not signed"):
        store.import_segment(a)
    # signed by somebody else (e.g. whoever only knows the public recipient key)
    rogue = PartnerSigningKey.generate(tmp_path / "rogue.key")
    sign_segment(b, rogue)
    with pytest.raises(MaskStoreError, match="untrusted key"):
        store.import_segment(b)
    # a valid signature of segment c copied next to segment a
    sign_segment(c, signer)
    (tmp_path / "outbox" / (a.name + SIGNATURE_SUFFIX)).write_bytes(c.with_name(c.name + SIGNATURE_SUFFIX).read_bytes())
    with pytest.raises(MaskStoreError, match="another segment"):
        store.import_segment(a)
    # signature over a different header (segment id patched to match): invalid
    doc = json.loads(c.with_name(c.name + SIGNATURE_SUFFIX).read_bytes())
    with SegmentFile(a) as seg:
        doc["segment_id"] = seg.header.segment_id
    (tmp_path / "outbox" / (a.name + SIGNATURE_SUFFIX)).write_text(json.dumps(doc))
    with pytest.raises(MaskStoreError, match="invalid"):
        store.import_segment(a)
    # malformed sidecar
    (tmp_path / "outbox" / (a.name + SIGNATURE_SUFFIX)).write_text("{not json")
    with pytest.raises(MaskStoreError, match="malformed"):
        store.import_segment(a)
    assert store.inventory() == []
    assert not list(store.root.rglob("*.pgwm*"))
    # the properly signed one still goes in
    store.import_segment(c)
    assert store.available_slots(FP, RING_K40_A12) == 4


def test_require_signature_without_keys_is_a_config_error(
    tmp_path: Path, open_store: Any, model: ToyModel, rec: RecipientKeyPair
) -> None:
    (seg,) = _partner_segments(tmp_path, model, rec)
    store = open_store(require_partner_signature=True, recipient=rec)
    with pytest.raises(ConfigError, match="trusted_partner_keys"):
        store.import_segment(seg)


def test_unsigned_import_without_policy_warns(
    tmp_path: Path, store: MaskStore, model: ToyModel, rec: RecipientKeyPair, caplog: pytest.LogCaptureFixture
) -> None:
    (seg,) = _partner_segments(tmp_path, model, rec)
    with caplog.at_level(logging.WARNING, logger="pgw.maskstore.store"):
        store.import_segment(seg, recipient=rec)
    assert any("WITHOUT sender authentication" in r.getMessage() for r in caplog.records)


def test_trusted_partners_parsing() -> None:
    with pytest.raises(ConfigError):
        TrustedPartners(["zz"])
    with pytest.raises(ConfigError):
        TrustedPartners([b"\0" * 31])
    k = os.urandom(32)
    assert len(TrustedPartners([k, k.hex()])) == 1
    assert not TrustedPartners()


def test_local_kek_segments_need_no_partner_signature(
    tmp_path: Path, kp: Any, model: ToyModel, signer: PartnerSigningKey
) -> None:
    a = MaskStore.open(tmp_path / "a", kp, low_water_slots=0)
    sid = write_segment(a, model, 3)
    out = a.export_segment(sid, tmp_path / "move")
    b = MaskStore.open(tmp_path / "b", kp, low_water_slots=0, trusted_partner_keys=(signer.public_bytes().hex(),))
    assert b.import_segment(out) == sid
    a.close()
    b.close()


# ---- import cleanup ------------------------------------------------------------------------------
def test_failed_import_never_deletes_a_file_it_did_not_publish(
    tmp_path: Path, store: MaskStore, model: ToyModel, rec: RecipientKeyPair
) -> None:
    """A concurrent importer may own the destination file; a losing import must leave it alone."""
    (seg,) = _partner_segments(tmp_path, model, rec)
    with SegmentFile(seg) as sf:
        h = sf.header
    final = store.root / h.fingerprint / h.ring / "segments" / seg.name
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"owned by another importer")
    with pytest.raises(MaskStoreError, match="already exists"):
        store.import_segment(seg, recipient=rec)
    assert final.read_bytes() == b"owned by another importer"
    assert store.inventory() == []


# ---- leases across MaskStore instances in one process --------------------------------------------
def test_second_store_instance_does_not_break_live_reservation(open_store: Any, model: ToyModel) -> None:
    """verify/gc in another MaskStore object of the same process must not close or shred a live lease."""
    a = open_store()
    write_segment(a, model, 4)
    res = a.reserve(FP, RING_K40_A12, 4)  # consumes the whole segment -> status consumed
    assert _is_live(res.id)
    b = open_store()  # verify_integrity on open: must not treat res as stale (same pid)
    rows = b.conn.execute("SELECT closed_at, host FROM reservations WHERE id=?", (res.id,)).fetchone()
    assert rows[0] is None and rows[1] == _HOST
    assert b.gc() == 0  # still leased
    for s in model.specs:  # file intact: every row decrypts and is exact
        r, wr = res.take(s.id, 4)
        assert torch.equal(wr, model.wr(s.id.key, r, RING_K40_A12))
    res.close()
    assert not _is_live(res.id)
    assert not list(a.root.rglob("*.pgwm"))  # shredded once released


def test_failed_reserve_drops_in_process_lease(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 2)

    def boom(point: str) -> None:
        raise RuntimeError(point)

    store._fault_hook = boom
    with pytest.raises(RuntimeError):
        store.reserve(FP, RING_K40_A12, 2)
    store._fault_hook = None
    assert not store._open_reservations
    (open_rows,) = store.conn.execute("SELECT COUNT(*) FROM reservations WHERE closed_at IS NULL").fetchone()
    assert open_rows == 0
    with pytest.raises(MaskExhaustedError):
        store.reserve(FP, RING_K40_A12, 1)  # the 2 slots stay consumed (lost, never reused)
