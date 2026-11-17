"""Sinks: store adapter, partner export (HPKE-sealed outbox) and segment header parsing."""

from __future__ import annotations

import base64
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from pgw.core import RING_K40_A12, ConfigError, MaskStoreError
from pgw.factory import ExportSink, MaskGenerator, RecipientWrapProvider, SegmentSink, StoreSink
from pgw.factory.sinks import load_recipient_public_key, read_segment_header

from .conftest import FakeMaskStore, FakePack, write_fake_segment_file

PUB = bytes(range(1, 33))


@pytest.fixture
def fake_hpke(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Replace the maskstore HPKE seal with a recording fake (the real one is tested by maskstore)."""
    from pgw.maskstore import keys

    sealed: list[bytes] = []

    def wrap(dek: bytes, pub: bytes) -> Any:
        sealed.append(dek)
        return keys.WrappedKey(f"hpke:{hashlib.sha256(pub).hexdigest()[:16]}", b"sealed" + pub)

    monkeypatch.setattr(keys, "hpke_wrap_for_recipient", wrap)
    return sealed


class StagingStore:
    """Fake staging store: writes header-framed files where a MaskStore would put them."""

    def __init__(self, root: Path, kp: Any, kek_override: str | None = None):
        self.root = root
        self.kp = kp
        self.kek_override = kek_override

    def writer(self, fingerprint: str, ring: Any, specs: Any, *, slots: int, seed: bytes, checks: int = 2,
               chunk_slots: int = 256, factory_id: str = "local") -> Any:
        store = self

        class W:
            rows = 0

            def write_linear(self, lid: Any, wr: Any, uw: Any) -> None:
                W.rows += wr.shape[0]

            def commit(self) -> str:
                sid = uuid.uuid4().hex
                wk = store.kp.wrap(os.urandom(32))
                hdr = {"segment_id": sid, "fingerprint": fingerprint, "ring": ring.name, "slots": slots,
                       "kek_id": store.kek_override or wk.kek_id, "factory_id": factory_id}
                write_fake_segment_file(store.root / fingerprint / ring.name / "segments" / f"{sid}.pgwm", hdr)
                return sid

            def abort(self) -> None:
                pass

        return W()


def _export(tmp_path: Path, kek_override: str | None = None) -> ExportSink:
    return ExportSink(tmp_path / "export", PUB, store_factory=lambda root, kp: StagingStore(root, kp, kek_override))


def test_export_sink_publishes_sealed_segments(tmp_path: Path, fake_hpke: list[bytes]) -> None:
    pack = FakePack()
    sink = _export(tmp_path)
    assert isinstance(sink, SegmentSink)
    gen = MaskGenerator(pack, RING_K40_A12)
    assert sink.available_slots(pack.fingerprint, RING_K40_A12) == 0
    a = gen.generate_into_store(sink, 7, chunk_slots=3)
    b = gen.generate_into_store(sink, 5)
    files = sorted(p.name for p in sink.outbox.glob("*.pgwm"))
    assert files == sorted([f"{a}.pgwm", f"{b}.pgwm"])
    assert not list(sink.staging_root.rglob("*.pgwm"))  # moved out of staging
    assert sink.available_slots(pack.fingerprint, RING_K40_A12) == 12
    assert len(fake_hpke) == 2 and sink.kek_id == f"hpke:{hashlib.sha256(PUB).hexdigest()[:16]}"
    (sink.outbox / f"{a}.pgwm").unlink()  # delivered
    assert sink.available_slots(pack.fingerprint, RING_K40_A12) == 5
    assert sink.state_dir() == tmp_path / "export" and sink.default_levels() is None


def test_export_sink_rejects_unsealed(tmp_path: Path, fake_hpke: list[bytes]) -> None:
    sink = _export(tmp_path, kek_override="local-kek")
    gen = MaskGenerator(FakePack(), RING_K40_A12)
    with pytest.raises(MaskStoreError, match="not sealed"):
        gen.generate_into_store(sink, 2)
    assert not list(sink.outbox.glob("*.pgwm"))


def test_export_sink_skips_garbage_files(tmp_path: Path) -> None:
    sink = _export(tmp_path)
    (sink.outbox / "junk.pgwm").write_bytes(b"not a segment")
    write_fake_segment_file(sink.outbox / "ok.pgwm", {"segment_id": "s", "fingerprint": "f" * 64, "ring": "k40a12",
                                                      "slots": 9, "kek_id": "hpke:x"})
    segs = sink.segments()
    assert [s.slots for s in segs] == [9] and segs[0].ring == "K40A12"


def test_recipient_provider(fake_hpke: list[bytes]) -> None:
    p = RecipientWrapProvider(PUB)
    wk = p.wrap(b"k" * 32)
    assert wk.kek_id == p.kek_id and p.kek_id.startswith("hpke:")
    with pytest.raises(MaskStoreError):
        p.unwrap(wk)
    assert p.monotonic_counter() is None and p.increment_counter() is None
    with pytest.raises(ConfigError):
        RecipientWrapProvider(b"short")
    from pgw.maskstore import KeyProvider

    assert isinstance(p, KeyProvider)


def test_load_recipient_public_key(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.write_bytes(PUB)
    hexf = tmp_path / "hex"
    hexf.write_text(PUB.hex() + "\n")
    b64 = tmp_path / "b64"
    b64.write_text(base64.b64encode(PUB).decode())
    for f in (raw, hexf, b64):
        assert load_recipient_public_key(f) == PUB
    bad = tmp_path / "bad"
    bad.write_text("hello")
    with pytest.raises(ConfigError):
        load_recipient_public_key(bad)


def test_read_segment_header_errors(tmp_path: Path) -> None:
    p = write_fake_segment_file(tmp_path / "a.pgwm", {"segment_id": "x", "slots": 3})
    assert read_segment_header(p)["slots"] == 3
    (tmp_path / "b.pgwm").write_bytes(b"PGWMASK1" + (10).to_bytes(4, "little") + b"{bad")
    (tmp_path / "c.pgwm").write_bytes(b"XXXXXXXX")
    (tmp_path / "d.pgwm").write_bytes(b"PGWMASK1" + (0).to_bytes(4, "little"))
    (tmp_path / "e.pgwm").write_bytes(b"PGWMASK1\x01")
    for name in "bcde":
        with pytest.raises(MaskStoreError):
            read_segment_header(tmp_path / f"{name}.pgwm")
    with pytest.raises(MaskStoreError):
        read_segment_header(tmp_path / "missing.pgwm")


def test_store_sink(fake_store: FakeMaskStore, fake_pack: FakePack, tmp_path: Path) -> None:
    sink = StoreSink(fake_store)
    gen = MaskGenerator(fake_pack, RING_K40_A12)
    gen.generate_into_store(sink, 4)
    assert sink.available_slots(fake_pack.fingerprint, RING_K40_A12) == 4
    assert sink.available_slots("ab" * 32, RING_K40_A12) == 0
    assert sink.default_levels() == (100, 300)
    assert sink.state_dir() == tmp_path / "store"
    calls: list[Any] = []
    sink.add_low_water_listener(calls.append)
    assert fake_store.listeners
    assert StoreSink(object()).default_levels() is None and StoreSink(object()).state_dir() is None
    StoreSink(object()).add_low_water_listener(calls.append)  # no-op without support
