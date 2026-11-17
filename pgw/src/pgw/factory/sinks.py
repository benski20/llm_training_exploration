"""Where generated segments go: a local :class:`pgw.maskstore.MaskStore`, or export files for a customer.

``StoreSink``
    Customer-run factory writing straight into the customer's store. Inventory is the store's
    unconsumed slots, so consumption by the gateway drives refills.
``ExportSink``
    Independent partner factory (SPEC 7.7). Segment DEKs are HPKE-sealed to the customer's X25519
    public key (``kek_id = "hpke:<key id>"``), so the partner cannot decrypt its own output once the
    DEK is gone. Finished segments land in ``<export_dir>/outbox/``; the delivery channel (sftp,
    object storage sync, ...) removes delivered files. Inventory is the undelivered slots in the
    outbox, so the scheduler keeps a buffer of ready segments.

Both expose the same small protocol used by :class:`pgw.factory.FactoryScheduler`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pgw.core.errors import ConfigError, MaskStoreError
from pgw.core.ring import RingParams
from pgw.core.types import LinearSpec

__all__ = [
    "ExportSink",
    "RecipientWrapProvider",
    "SegmentFileInfo",
    "SegmentSink",
    "StoreSink",
    "load_recipient_public_key",
    "read_segment_header",
]

log = logging.getLogger(__name__)

_MAGIC = b"PGWMASK1"
_SUFFIX = ".pgwm"
_MAX_HEADER = 16 << 20


@runtime_checkable
class SegmentSink(Protocol):
    """Destination of generated segments (see module docstring)."""

    kind: str

    def available_slots(self, fingerprint: str, ring: RingParams) -> int:
        """Slots ready for use (store) or awaiting delivery (export) for ``(fingerprint, ring)``."""
        ...

    def writer(
        self,
        fingerprint: str,
        ring: RingParams,
        specs: Sequence[LinearSpec],
        *,
        slots: int,
        seed: bytes,
        checks: int = 2,
        chunk_slots: int = 256,
        factory_id: str = "local",
    ) -> Any:
        """A :class:`pgw.maskstore.SegmentWriter`-compatible writer for a new segment."""
        ...


# ---- segment header reading (public, unencrypted part of the format; SPEC 7.2) -------------------
@dataclass(frozen=True)
class SegmentFileInfo:
    """Header fields of a segment file relevant for inventory."""

    path: Path
    segment_id: str
    fingerprint: str
    ring: str
    slots: int
    kek_id: str
    size: int


def read_segment_header(path: str | Path) -> dict[str, Any]:
    """Parse the plaintext header JSON of a ``*.pgwm`` file (no decryption, no AEAD check).

    Raises:
        MaskStoreError: bad magic or malformed header.
    """
    p = Path(path)
    try:
        with p.open("rb") as f:
            magic = f.read(8)
            if magic != _MAGIC:
                raise MaskStoreError(f"{p.name}: not a PGW segment (bad magic)")
            raw = f.read(4)
            if len(raw) != 4:
                raise MaskStoreError(f"{p.name}: truncated header length")
            (hlen,) = struct.unpack("<I", raw)
            if not 0 < hlen <= _MAX_HEADER:
                raise MaskStoreError(f"{p.name}: implausible header length {hlen}")
            body = f.read(hlen)
            if len(body) != hlen:
                raise MaskStoreError(f"{p.name}: truncated header")
        hdr = json.loads(body)
        if not isinstance(hdr, dict):
            raise MaskStoreError(f"{p.name}: header is not an object")
        return hdr
    except OSError as exc:
        raise MaskStoreError(f"cannot read segment {p.name}: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise MaskStoreError(f"{p.name}: malformed header JSON") from exc


def _segment_info(path: Path) -> SegmentFileInfo:
    h = read_segment_header(path)
    try:
        return SegmentFileInfo(
            path, str(h["segment_id"]), str(h["fingerprint"]), str(h["ring"]).upper(), int(h["slots"]),
            str(h.get("kek_id", "")), path.stat().st_size,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MaskStoreError(f"{path.name}: header misses required fields") from exc


# ---- local store ---------------------------------------------------------------------------------
class StoreSink:
    """Adapter around a :class:`pgw.maskstore.MaskStore` (customer-run factory)."""

    kind = "store"

    def __init__(self, store: Any):
        self.store = store

    def available_slots(self, fingerprint: str, ring: RingParams) -> int:
        """Unconsumed slots in the store."""
        invs = self.store.inventory(fingerprint, ring)
        return sum(int(i.slots_available) for i in invs if i.fingerprint == fingerprint and i.ring.upper() == ring.name)

    def writer(self, fingerprint: str, ring: RingParams, specs: Sequence[LinearSpec], **kw: Any) -> Any:
        """Delegate to :meth:`MaskStore.writer`."""
        return self.store.writer(fingerprint, ring, specs, **kw)

    def add_low_water_listener(self, fn: Callable[[Any], None]) -> None:
        """Forward to the store if it supports listeners."""
        add = getattr(self.store, "add_low_water_listener", None)
        if add is None:
            return
        try:
            add(fn)
        except NotImplementedError:  # pragma: no cover - store without listeners
            log.debug("store does not support low-water listeners; polling only")

    def default_levels(self) -> tuple[int, int] | None:
        """``(low_water_slots, target_slots)`` from the store's settings, if exposed."""
        st = getattr(self.store, "settings", None)
        if st is None or not hasattr(st, "low_water_slots"):
            return None
        return int(st.low_water_slots), int(st.target_slots)

    def state_dir(self) -> Path | None:
        """The store root (factory state lives next to the store)."""
        st = getattr(self.store, "settings", None)
        root = getattr(st, "root", None) or getattr(self.store, "root", None)
        return Path(root) if root is not None else None


# ---- partner export ------------------------------------------------------------------------------
def load_recipient_public_key(path: str | Path) -> bytes:
    """Read a customer's X25519 public key: 32 raw bytes, or 64 hex chars, or base64 text."""
    data = Path(path).read_bytes()
    if len(data) == 32:
        return data
    text = data.decode("ascii", errors="strict").strip() if data.isascii() else ""
    for decode in (bytes.fromhex, base64.b64decode):
        try:
            key = decode(text)
        except (ValueError, binascii.Error):
            continue
        if len(key) == 32:
            return key
    raise ConfigError(f"{Path(path).name}: expected a 32-byte X25519 public key (raw, hex or base64)")


def _key_provider_base() -> Any:
    from pgw.maskstore.keys import KeyProvider

    return KeyProvider


class RecipientWrapProvider:
    """Wrap-only key provider: seals each segment DEK to a customer's X25519 key with HPKE.

    The partner holds no KEK, so after a segment is written the partner itself cannot decrypt it.
    Instances are also registered as :class:`pgw.maskstore.KeyProvider` subclasses.
    """

    def __init__(self, recipient_public: bytes):
        if len(recipient_public) != 32:
            raise ConfigError("recipient public key must be 32 bytes (X25519)")
        self._pub = bytes(recipient_public)
        self._key_id = hashlib.sha256(self._pub).hexdigest()[:16]

    @property
    def kek_id(self) -> str:
        """``hpke:<first 16 hex of SHA-256(public key)>``."""
        return f"hpke:{self._key_id}"

    def wrap(self, dek: bytes) -> Any:
        """HPKE-seal ``dek`` to the recipient."""
        from pgw.maskstore.keys import WrappedKey, hpke_wrap_for_recipient

        wk = hpke_wrap_for_recipient(dek, self._pub)
        return wk if wk.kek_id == self.kek_id else WrappedKey(self.kek_id, wk.wrapped)

    def unwrap(self, wrapped: Any) -> bytes:
        """Never possible on the partner side."""
        raise MaskStoreError("export provider cannot unwrap: segments are sealed to the customer")

    def monotonic_counter(self) -> int | None:
        """Unsupported."""
        return None

    def increment_counter(self) -> int | None:
        """Unsupported."""
        return None


def _register_provider() -> None:
    try:
        _key_provider_base().register(RecipientWrapProvider)
    except Exception:  # pragma: no cover - maskstore import problems are reported on use
        log.debug("could not register RecipientWrapProvider as KeyProvider")


_register_provider()


class _ExportWriter:
    """Wraps a staging-store writer; on commit moves the finished file into the outbox."""

    def __init__(self, sink: ExportSink, inner: Any):
        self._sink = sink
        self._inner = inner

    def write_linear(self, lid: Any, wr: Any, uw: Any) -> None:
        self._inner.write_linear(lid, wr, uw)

    def commit(self) -> str:
        seg_id = str(self._inner.commit())
        self._sink._publish(seg_id)
        return seg_id

    def abort(self) -> None:
        self._inner.abort()

    def __enter__(self) -> _ExportWriter:
        return self

    def __exit__(self, et: Any, ev: Any, tb: Any) -> None:
        if et is None:
            self.commit()
        else:
            self.abort()


class ExportSink:
    """Partner-factory output: HPKE-sealed segment files in ``<export_dir>/outbox``.

    Segments are produced by a private *staging* mask store under ``<export_dir>/.staging`` whose
    key provider is :class:`RecipientWrapProvider`, then moved atomically (same filesystem) into
    the outbox. Files in the outbox are complete and immutable; delivery removes them.

    Args:
        export_dir: Output directory.
        recipient_public: Customer X25519 public key (32 bytes).
        store_factory: ``(root, key_provider) -> MaskStore``; default ``MaskStore.open``.
    """

    kind = "export"

    def __init__(
        self,
        export_dir: str | Path,
        recipient_public: bytes,
        *,
        store_factory: Callable[[Path, Any], Any] | None = None,
    ):
        self.export_dir = Path(export_dir)
        self.outbox = self.export_dir / "outbox"
        self.staging_root = self.export_dir / ".staging"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.provider = RecipientWrapProvider(recipient_public)
        if store_factory is None:
            from pgw.maskstore import MaskStore

            def store_factory(root: Path, kp: Any) -> Any:
                return MaskStore.open(root, kp)

        self._staging = store_factory(self.staging_root, self.provider)

    @property
    def kek_id(self) -> str:
        """Recipient key id recorded in every exported header."""
        return self.provider.kek_id

    def segments(self) -> list[SegmentFileInfo]:
        """Undelivered segments in the outbox (unreadable files are skipped with a warning)."""
        out: list[SegmentFileInfo] = []
        for p in sorted(self.outbox.rglob(f"*{_SUFFIX}")):
            try:
                out.append(_segment_info(p))
            except MaskStoreError as exc:
                log.warning("ignoring outbox file %s: %s", p.name, exc.message)
        return out

    def available_slots(self, fingerprint: str, ring: RingParams) -> int:
        """Undelivered slots for ``(fingerprint, ring)``."""
        return sum(s.slots for s in self.segments() if s.fingerprint == fingerprint and s.ring == ring.name)

    def writer(self, fingerprint: str, ring: RingParams, specs: Sequence[LinearSpec], **kw: Any) -> _ExportWriter:
        """Writer whose commit publishes into the outbox."""
        return _ExportWriter(self, self._staging.writer(fingerprint, ring, specs, **kw))

    def _publish(self, seg_id: str) -> Path:
        exporter = getattr(self._staging, "export_segment", None)
        if exporter is not None:  # preferred if the store offers it (see INTERFACE_NOTES)
            dest = Path(exporter(seg_id, self.outbox))
        else:
            hits = list(self.staging_root.rglob(f"{seg_id}{_SUFFIX}"))
            if len(hits) != 1:
                raise MaskStoreError(f"exported segment {seg_id} not found in staging", details={"hits": len(hits)})
            dest = self.outbox / hits[0].name
            os.replace(hits[0], dest)
            _fsync_dir(self.outbox)
        info = _segment_info(dest)
        if info.kek_id != self.kek_id:
            dest.unlink(missing_ok=True)
            raise MaskStoreError(
                f"exported segment {seg_id} is not sealed to the recipient key",
                details={"kek_id": info.kek_id, "expected": self.kek_id},
            )
        log.info("exported segment %s (%d slots, %.1f MB) to outbox", seg_id, info.slots, info.size / 1e6)
        return dest

    def default_levels(self) -> tuple[int, int] | None:
        """No store-provided levels; the factory settings decide."""
        return None

    def state_dir(self) -> Path:
        """Factory state lives in the export directory."""
        return self.export_dir


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)
