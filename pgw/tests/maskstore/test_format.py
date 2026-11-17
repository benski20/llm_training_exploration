"""Segment file format: header validation, layout arithmetic, AEAD binding, partner import."""

from __future__ import annotations

import base64
import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, RING_K48_A20, MaskReuseError, MaskStoreError
from pgw.maskstore import (
    SEGMENT_MAGIC,
    KeyProvider,
    MaskStore,
    RecipientKeyPair,
    SegmentFile,
    SegmentHeader,
    StaticKeyProvider,
    WrappedKey,
    hpke_wrap_for_recipient,
)
from pgw.maskstore.format import SegmentLayout, encode_header, read_header

from .conftest import FP, ToyModel, write_segment


def _header(**kw: Any) -> dict[str, Any]:
    h = {
        "segment_id": "0" * 32, "fingerprint": FP, "ring": "K40A12", "slots": 10, "chunk_slots": 4, "checks": 2,
        "linears": [{"id": "0.qkv", "out_features": 24, "in_features": 16}],
        "kek_id": "k", "wrapped_dek": base64.b64encode(b"x").decode(), "nonce_prefix": "00112233",
        "factory_id": "f", "created_at": "2026-01-01T00:00:00Z",
    }
    h.update(kw)
    return h


@pytest.mark.parametrize(
    "bad",
    [
        {"version": 2}, {"segment_id": "xyz"}, {"fingerprint": "abc"}, {"ring": "K41A12"}, {"slots": 0},
        {"chunk_slots": 0}, {"checks": -1}, {"linears": []}, {"nonce_prefix": "0011"},
        {"linears": [{"id": "0.qkv", "out_features": 2, "in_features": 2}] * 2},
        {"linears": [{"id": "0.foo", "out_features": 2, "in_features": 2}]}, {"extra": 1},
    ],
)
def test_header_validation(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SegmentHeader.model_validate(_header(**bad))


def test_header_canonical_and_layout_sizes() -> None:
    h = SegmentHeader.model_validate(_header())
    raw = encode_header(h)
    assert raw == encode_header(SegmentHeader.model_validate(json.loads(raw)))
    lay = SegmentLayout(h, len(raw))
    bpe = 5
    assert lay.chunks_per_linear == 3 and [lay.chunk_rows(j) for j in range(3)] == [4, 4, 2]
    expected = 12 + len(raw) + (32 + 2 * 16 * bpe + 16) + (10 * 24 * bpe + 3 * 16)
    assert lay.total_size == expected
    c2 = lay.wr_chunk(0, 2)
    assert c2.index == 3 and c2.offset + c2.cipher_len == lay.total_size


def test_store_files_match_layout(store: MaskStore, model: ToyModel) -> None:
    sid = write_segment(store, model, 9, chunk_slots=4)
    path = next(store.root.rglob(f"{sid}.pgwm"))
    with path.open("rb") as f:
        h, raw = read_header(f)
    assert path.stat().st_size == SegmentLayout(h, len(raw)).total_size
    assert h.linears[0].id == "0.qkv" and [x.id for x in h.linears] == [s.id.key for s in model.specs]
    assert path.read_bytes()[:8] == SEGMENT_MAGIC


def test_corrupt_files_are_rejected(tmp_path: Path) -> None:
    p = tmp_path / "x.pgwm"
    p.write_bytes(b"NOTMAGIC" + b"\0" * 20)
    with pytest.raises(MaskStoreError, match="magic"):
        SegmentFile(p)
    p.write_bytes(SEGMENT_MAGIC + struct.pack("<I", 1 << 30))
    with pytest.raises(MaskStoreError, match="header length"):
        SegmentFile(p)
    p.write_bytes(SEGMENT_MAGIC + struct.pack("<I", 5) + b"{bad}")
    with pytest.raises(MaskStoreError, match="malformed"):
        SegmentFile(p)
    raw = json.dumps(_header()).encode()
    p.write_bytes(SEGMENT_MAGIC + struct.pack("<I", len(raw)) + raw + b"\0" * 10)
    with pytest.raises(MaskStoreError, match="size"):
        SegmentFile(p)
    with pytest.raises(MaskStoreError, match="cannot open"):
        SegmentFile(tmp_path / "missing.pgwm")


def test_header_is_authenticated(store: MaskStore, model: ToyModel, kp: StaticKeyProvider) -> None:
    """Editing the plaintext header (e.g. the factory id) breaks every chunk's AEAD."""
    sid = write_segment(store, model, 4)
    path = next(store.root.rglob(f"{sid}.pgwm"))
    data = path.read_bytes()
    edited = data.replace(b'"factory_id":"test"', b'"factory_id":"evil"')
    assert edited != data
    path.write_bytes(edited)
    with SegmentFile(path) as seg:
        seg.set_dek(kp.unwrap(WrappedKey(seg.header.kek_id, seg.header.wrapped_dek_bytes)))
        with pytest.raises(MaskStoreError, match="authentication"):
            seg.decrypt(seg.layout.secrets_chunk())


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


def _partner_segment(tmp_path: Path, model: ToyModel, recipient: RecipientKeyPair, slots: int = 6) -> Path:
    partner = MaskStore.open(tmp_path / "partner", _WrapOnly(recipient.public_bytes()), low_water_slots=0)
    sid = write_segment(partner, model, slots, chunk_slots=4)
    out = partner.export_segment(sid, tmp_path / "outbox")
    with pytest.raises(MaskReuseError):
        partner.export_segment(sid, tmp_path / "outbox")  # never twice
    partner.close()
    return out


def test_partner_import_hpke(tmp_path: Path, store: MaskStore, model: ToyModel) -> None:
    rec = RecipientKeyPair.generate(tmp_path / "rec.key")
    seg_path = _partner_segment(tmp_path, model, rec)
    with SegmentFile(seg_path) as seg:
        assert seg.header.kek_id == rec.kek_id
        nonce_before = seg.header.nonce_prefix
    with pytest.raises(MaskStoreError, match="no recipient"):
        store.import_segment(seg_path)
    sid = store.import_segment(seg_path, recipient=rec)
    local = next(store.root.rglob(f"{sid}.pgwm"))
    with SegmentFile(local) as seg:  # re-wrapped under the local KEK with a fresh DEK/nonce
        assert seg.header.kek_id == store.key_provider.kek_id and seg.header.nonce_prefix != nonce_before
    with pytest.raises(MaskReuseError):
        store.import_segment(seg_path, recipient=rec)  # second import = reuse
    assert store.available_slots(FP, RING_K40_A12) == 6
    with store.reserve(FP, RING_K40_A12, 6) as res:
        for s in model.specs:
            r, wr = res.take(s.id, 6)
            assert torch.equal(wr, model.wr(s.id.key, r, RING_K40_A12))


def test_import_with_store_recipient_and_copied_seed(tmp_path: Path, open_store: Any, model: ToyModel) -> None:
    rec = RecipientKeyPair.generate(tmp_path / "rec.key")
    store = open_store(recipient_key_file=tmp_path / "rec.key")
    seed = os.urandom(32)
    partner = MaskStore.open(tmp_path / "partner", _WrapOnly(rec.public_bytes()), low_water_slots=0)
    a = write_segment(partner, model, 3, seed=seed)
    pa = partner.export_segment(a, tmp_path / "outbox")
    store.import_segment(pa)
    # a different partner store that (wrongly) reuses the same seed: refused by seed hash
    partner2 = MaskStore.open(tmp_path / "partner2", _WrapOnly(rec.public_bytes()), low_water_slots=0)
    b = write_segment(partner2, model, 3, seed=seed)
    pb = partner2.export_segment(b, tmp_path / "outbox2")
    with pytest.raises(MaskReuseError, match="seed"):
        store.import_segment(pb)
    assert store.available_slots(FP, RING_K40_A12) == 3
    partner.close()
    partner2.close()


def test_import_rejects_tampered_and_leaves_nothing(tmp_path: Path, store: MaskStore, model: ToyModel) -> None:
    rec = RecipientKeyPair.generate(tmp_path / "rec.key")
    seg_path = _partner_segment(tmp_path, model, rec, slots=8)
    raw = bytearray(seg_path.read_bytes())
    raw[-5] ^= 0x80  # last chunk
    seg_path.write_bytes(bytes(raw))
    with pytest.raises(MaskStoreError, match="authentication"):
        store.import_segment(seg_path, recipient=rec)
    assert store.inventory() == []
    assert not list(store.root.rglob("*.pgwm*"))


def test_import_local_kek_segment_between_stores(tmp_path: Path, kp: StaticKeyProvider, model: ToyModel) -> None:
    """A segment from another store under the same KEK (e.g. a disaster-recovery move)."""
    a = MaskStore.open(tmp_path / "a", kp, low_water_slots=0)
    sid = write_segment(a, model, 5, ring=RING_K48_A20)
    out = a.export_segment(sid, tmp_path / "move")
    b = MaskStore.open(tmp_path / "b", kp, low_water_slots=0)
    assert b.import_segment(out) == sid
    with b.reserve(FP, RING_K48_A20, 5) as res:
        r, wr = res.take("1.o", 5)
        assert torch.equal(wr, model.wr("1.o", r, RING_K48_A20))
        u, uw = res.checks("1.o")
        assert u.shape == (2, 16) and uw.shape == (2, 16)
    a.close()
    b.close()
