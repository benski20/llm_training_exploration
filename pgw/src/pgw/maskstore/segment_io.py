"""Low-level segment file writing and secure deletion."""

from __future__ import annotations

import hashlib
import logging
import os
import struct
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pgw.core.errors import MaskStoreError
from pgw.maskstore.counters import fsync_dir
from pgw.maskstore.format import (
    SEGMENT_MAGIC,
    ChunkRef,
    SegmentHeader,
    SegmentLayout,
    chunk_aad,
    chunk_nonce,
    encode_header,
)

__all__ = ["RawSegmentWriter", "crypto_erase", "shred_file", "wipe_and_unlink"]

log = logging.getLogger(__name__)


class RawSegmentWriter:
    """Writes encrypted chunks of one segment into a temp file, then publishes it atomically.

    Chunks may be written in any order (offsets come from the header); every chunk must be written
    exactly once before :meth:`publish`.
    """

    def __init__(self, final_path: Path, header: SegmentHeader, dek: bytes, *, fsync: bool = True):
        self.final_path = final_path
        self.tmp_path = final_path.with_name(f".{final_path.name}.part")
        self.header = header
        self.header_raw = encode_header(header)
        self.layout = SegmentLayout(header, len(self.header_raw))
        self.fsync = fsync
        self._digest = hashlib.sha256(self.header_raw).digest()
        self._nonce_prefix = bytes.fromhex(header.nonce_prefix)
        self._aead: AESGCM | None = AESGCM(bytes(dek))
        self._written: set[int] = set()
        self._expected = 1 + len(header.linears) * self.layout.chunks_per_linear
        final_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd: int | None = os.open(self.tmp_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            raise MaskStoreError(f"cannot create segment file: {exc.strerror}") from exc
        try:
            pre = SEGMENT_MAGIC + struct.pack("<I", len(self.header_raw)) + self.header_raw
            os.pwrite(self._fd, pre, 0)
            os.ftruncate(self._fd, self.layout.total_size)
        except OSError as exc:
            self.abort()
            raise MaskStoreError(f"cannot write segment header: {exc.strerror}") from exc

    @property
    def size(self) -> int:
        """Final file size."""
        return self.layout.total_size

    def write_chunk(self, ref: ChunkRef, plaintext: bytes | bytearray | memoryview) -> None:
        """Encrypt and write one chunk."""
        if self._fd is None or self._aead is None:
            raise MaskStoreError("segment writer is closed")
        if len(plaintext) != ref.plain_len:
            raise MaskStoreError(f"chunk {ref.index}: plaintext length {len(plaintext)} != {ref.plain_len}")
        if ref.index in self._written:
            raise MaskStoreError(f"chunk {ref.index} written twice")
        ct = self._aead.encrypt(
            chunk_nonce(self._nonce_prefix, ref.index), bytes(plaintext), chunk_aad(self._digest, ref.index)
        )
        try:
            n = os.pwrite(self._fd, ct, ref.offset)
        except OSError as exc:
            raise MaskStoreError(f"segment write failed: {exc.strerror}") from exc
        if n != len(ct):
            raise MaskStoreError("short write to segment file")
        self._written.add(ref.index)

    def publish(self, *, replace: bool = False) -> int:
        """fsync, rename into place, fsync the directory; return the file size.

        Args:
            replace: Atomically replace an existing file at the final path (re-encryption during
                KEK rotation). By default an existing file is an error.
        """
        if self._fd is None:
            raise MaskStoreError("segment writer is closed")
        if len(self._written) != self._expected:
            raise MaskStoreError(
                f"segment incomplete: {len(self._written)} of {self._expected} chunks written",
                details={"segment_id": self.header.segment_id},
            )
        try:
            if self.fsync:
                os.fsync(self._fd)
            os.close(self._fd)
            self._fd = None
            if self.final_path.exists() and not replace:
                raise MaskStoreError(f"segment file {self.final_path.name} already exists")
            os.replace(self.tmp_path, self.final_path)
            if self.fsync:
                fsync_dir(self.final_path.parent)
        except OSError as exc:
            self.abort()
            raise MaskStoreError(f"cannot publish segment: {exc.strerror}") from exc
        except MaskStoreError:
            self.abort()
            raise
        self._aead = None
        return self.layout.total_size

    def abort(self) -> None:
        """Delete the temp file (idempotent)."""
        self._aead = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:  # pragma: no cover
                pass
            self._fd = None
        try:
            self.tmp_path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover
            log.warning("could not remove partial segment %s: %s", self.tmp_path.name, exc.strerror)


def crypto_erase(path: Path, *, header_bytes: int = 64 << 10, fsync: bool = True) -> int:
    """Cryptographically erase a segment file in place; return its size (0 if it is gone).

    The header region holds the wrapped DEK. Overwriting it with random bytes and syncing makes
    every chunk of the file undecryptable, so this is the security-relevant part of shredding and
    it is cheap (one 64 KiB write) regardless of the segment size.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    try:
        fd = os.open(path, os.O_WRONLY)
    except OSError as exc:
        raise MaskStoreError(f"cannot open {path.name} for shredding: {exc.strerror}") from exc
    try:
        head = min(size, header_bytes)
        os.pwrite(fd, os.urandom(head), 0)
        if fsync:
            os.fsync(fd)
    except OSError as exc:
        raise MaskStoreError(f"shredding {path.name} failed: {exc.strerror}") from exc
    finally:
        os.close(fd)
    return size


def wipe_and_unlink(path: Path, *, header_bytes: int = 64 << 10, zero_fill: bool = True, fsync: bool = True) -> int:
    """Bulk part of shredding (after :func:`crypto_erase`): zero the body, truncate, unlink.

    This can take minutes for multi-GB segments, so callers run it outside the store lock and off
    any latency-sensitive thread. ``zero_fill=False`` skips the overwrite (truncate + unlink only),
    which is all that is useful on SSDs and copy-on-write filesystems. Returns the bytes freed.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    try:
        fd = os.open(path, os.O_WRONLY)
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise MaskStoreError(f"cannot open {path.name} for wiping: {exc.strerror}") from exc
    block = 1 << 20
    try:
        if zero_fill:
            zeros = bytes(block)
            off = min(size, header_bytes)
            while off < size:
                n = min(block, size - off)
                os.pwrite(fd, zeros[:n], off)
                off += n
            if fsync:
                os.fsync(fd)
        os.ftruncate(fd, 0)
    except OSError as exc:
        raise MaskStoreError(f"wiping {path.name} failed: {exc.strerror}") from exc
    finally:
        os.close(fd)
    path.unlink(missing_ok=True)
    if fsync:
        fsync_dir(path.parent)
    return size


def shred_file(path: Path, *, header_bytes: int = 64 << 10, fsync: bool = True, zero_fill: bool = True) -> int:
    """Securely delete a segment file; return the bytes freed.

    The header region (which holds the wrapped DEK) is overwritten with random bytes first and
    synced, so the segment is cryptographically erased as soon as that write reaches the disk; the
    rest of the file is then overwritten with zeros (unless ``zero_fill=False``), the file
    truncated and unlinked. :class:`pgw.maskstore.MaskStore` runs the two halves separately
    (:func:`crypto_erase` under its lock, :func:`wipe_and_unlink` in a background thread).

    Note: on copy-on-write filesystems and SSDs with wear levelling an overwrite does not guarantee
    the old blocks are gone. Complete erasure there relies on retiring the KEK (key rotation) and on
    full-disk encryption; see the operations guide.
    """
    size = crypto_erase(path, header_bytes=header_bytes, fsync=fsync)
    if size == 0 and not path.exists():
        return 0
    wipe_and_unlink(path, header_bytes=header_bytes, zero_fill=zero_fill, fsync=fsync)
    return size
