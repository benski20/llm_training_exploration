"""Low-level segment file writing and secure deletion."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import logging
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pgw.core import native
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

__all__ = ["RawSegmentWriter", "crypto_erase", "partial_in_use", "shred_file", "wipe_and_unlink"]

log = logging.getLogger(__name__)

#: Native segment writer (``pgw_fast.seal_chunks`` / ``seal_ring_chunks``, Rust ``writer.rs``):
#: AES-256-GCM + ``pwrite`` with the GIL released, byte-identical to the Python path below
#: (``tests/native/test_writer.py``). ``PGW_NATIVE=0`` (or a build without it) uses Python.
NATIVE_WRITER = native.native_available("seal_chunks", "seal_ring_chunks")

#: Threads of the native writer per call (0 = automatic: about one per MiB of plaintext, at most
#: the CPU count and 8).
try:
    WRITER_THREADS = max(0, int(os.environ.get("PGW_WRITER_THREADS", "0") or 0))
except ValueError:  # pragma: no cover - bad environment value: automatic
    WRITER_THREADS = 0


class RawSegmentWriter:
    """Writes encrypted chunks of one segment into a temp file, then publishes it atomically.

    Chunks may be written in any order (offsets come from the header); every chunk must be written
    exactly once before :meth:`publish`.
    """

    def __init__(
        self, final_path: Path, header: SegmentHeader, dek: bytes, *, fsync: bool = True, use_native: bool | None = None
    ):
        self.final_path = final_path
        self.tmp_path = final_path.with_name(f".{final_path.name}.part")
        self.header = header
        self.header_raw = encode_header(header)
        self.layout = SegmentLayout(header, len(self.header_raw))
        self.fsync = fsync
        self._digest = hashlib.sha256(self.header_raw).digest()
        self._nonce_prefix = bytes.fromhex(header.nonce_prefix)
        #: Native fast path in use (``use_native=None``: whenever the extension provides it).
        self.native = NATIVE_WRITER if use_native is None else bool(use_native) and NATIVE_WRITER
        # Exactly one copy of the DEK per writer: the native path's bytearray (wiped on publish,
        # abort, a failed write or garbage collection) or the Python path's AESGCM object.
        self._key = bytearray(dek) if self.native else bytearray()
        self._aead: AESGCM | None = None if self.native else AESGCM(bytes(dek))
        self._closed = False
        self._written: set[int] = set()
        self._expected = 1 + len(header.linears) * self.layout.chunks_per_linear
        final_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd: int | None = os.open(self.tmp_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            raise MaskStoreError(f"cannot create segment file: {exc.strerror}") from exc
        # Liveness of this partial file for every other process: an exclusive flock held from creation until
        # the file is renamed into place (or aborted). Clean-up of partial files skips locked ones
        # (:func:`partial_in_use`), so a factory worker that restarts after a crash, or the gateway's gc,
        # never deletes the temp file of a concurrent writer (several factory workers per store).
        with contextlib.suppress(OSError):  # a fresh O_EXCL file: nobody else can hold it
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            pre = SEGMENT_MAGIC + struct.pack("<I", len(self.header_raw)) + self.header_raw
            os.pwrite(self._fd, pre, 0)
            os.ftruncate(self._fd, self.layout.total_size)
        except OSError as exc:
            self.abort()
            raise MaskStoreError(f"cannot write segment header: {exc.strerror}") from exc

    @property
    def closed(self) -> bool:
        """True after publish, abort or a failed write (no further writes are accepted)."""
        return self._closed

    @property
    def size(self) -> int:
        """Final file size."""
        return self.layout.total_size

    def write_chunk(self, ref: ChunkRef, plaintext: bytes | bytearray | memoryview) -> None:
        """Encrypt and write one chunk."""
        if self._closed or self._fd is None:
            raise MaskStoreError("segment writer is closed")
        if len(plaintext) != ref.plain_len:
            raise MaskStoreError(f"chunk {ref.index}: plaintext length {len(plaintext)} != {ref.plain_len}")
        if ref.index in self._written:
            raise MaskStoreError(f"chunk {ref.index} written twice")
        if self.native:
            mv = memoryview(plaintext)
            if mv.format != "B" or mv.ndim != 1:
                mv = mv.cast("B")
            try:
                native.fast.seal_chunks(
                    self._fd, self._key, self._nonce_prefix, self._digest, mv,
                    [(ref.index, ref.offset, 0, ref.plain_len)], WRITER_THREADS,
                )
            except OSError as exc:
                self._write_failed()
                raise MaskStoreError(f"segment write failed: {exc.strerror}") from exc
            finally:
                mv.release()
            self._written.add(ref.index)
            return
        assert self._aead is not None
        ct = self._aead.encrypt(
            chunk_nonce(self._nonce_prefix, ref.index), bytes(plaintext), chunk_aad(self._digest, ref.index)
        )
        try:
            n = os.pwrite(self._fd, ct, ref.offset)
        except OSError as exc:
            self._write_failed()
            raise MaskStoreError(f"segment write failed: {exc.strerror}") from exc
        if n != len(ct):
            self._write_failed()
            raise MaskStoreError("short write to segment file")
        self._written.add(ref.index)

    def write_ring_chunks(self, values: np.ndarray, k: int, jobs: list[tuple[ChunkRef, int, int]]) -> bool:
        """Range-check ``values``, then pack, encrypt and write row ranges of it (native path only).

        ``values`` is a C-contiguous 2-D ring matrix (int64 / uint64, or int32 / uint32 holding the
        low 32 bits for ``k <= 32``); each job ``(ref, row_start, rows)`` writes chunk ``ref`` with
        plaintext ``pack_ring(values[row_start:row_start + rows], k)``, exactly as
        :meth:`write_chunk` would. If any element of ``values`` (all rows, not only the jobs')
        lies outside ``[0, 2^k)`` nothing is written and the result is False.
        """
        if self._closed or self._fd is None:
            raise MaskStoreError("segment writer is closed")
        if not self.native:
            raise MaskStoreError("native segment writer not available")
        cols = int(values.shape[1])
        nb = k // 8
        raw: list[tuple[int, int, int, int]] = []
        for ref, row_start, rows in jobs:
            if rows * cols * nb != ref.plain_len:
                raise MaskStoreError(f"chunk {ref.index}: plaintext length {rows * cols * nb} != {ref.plain_len}")
            if ref.index in self._written:
                raise MaskStoreError(f"chunk {ref.index} written twice")
            raw.append((ref.index, ref.offset, row_start, rows))
        if len({j[0] for j in raw}) != len(raw):
            raise MaskStoreError("chunk listed twice")
        try:
            ok = native.fast.seal_ring_chunks(
                self._fd, self._key, self._nonce_prefix, self._digest, values, k, raw, WRITER_THREADS
            )
        except OSError as exc:
            self._write_failed()
            raise MaskStoreError(f"segment write failed: {exc.strerror}") from exc
        if ok:
            self._written.update(j[0] for j in raw)
        return bool(ok)

    def _wipe_key(self) -> None:
        if self._key:
            _zero_bytes(self._key)

    def _write_failed(self) -> None:
        """A write failed part-way: chunks already on disk are not tracked, so a retry with
        different plaintext would reuse (DEK, nonce) pairs. Close the writer for good."""
        self.abort()

    def __del__(self) -> None:
        # Garbage-collected without publish/abort: wipe the key, close the fd, drop the temp file.
        # Only a writer that created its temp file (fd still open) may delete it.
        with contextlib.suppress(Exception):  # interpreter shutdown: modules may be gone
            if getattr(self, "_fd", None) is not None:
                self.abort()
            elif getattr(self, "_key", None):
                self._wipe_key()

    def publish(self, *, replace: bool = False) -> int:
        """fsync, rename into place, fsync the directory; return the file size.

        Args:
            replace: Atomically replace an existing file at the final path (re-encryption during
                KEK rotation). By default an existing file is an error.
        """
        if self._closed or self._fd is None:
            raise MaskStoreError("segment writer is closed")
        if len(self._written) != self._expected:
            raise MaskStoreError(
                f"segment incomplete: {len(self._written)} of {self._expected} chunks written",
                details={"segment_id": self.header.segment_id},
            )
        try:
            if self.fsync:
                os.fsync(self._fd)
            if self.final_path.exists() and not replace:
                raise MaskStoreError(f"segment file {self.final_path.name} already exists")
            # Rename while the fd (and its flock) is still open: the partial file is never unlocked
            # before it stops being a partial file.
            os.replace(self.tmp_path, self.final_path)
            os.close(self._fd)
            self._fd = None
            if self.fsync:
                fsync_dir(self.final_path.parent)
        except OSError as exc:
            self.abort()
            raise MaskStoreError(f"cannot publish segment: {exc.strerror}") from exc
        except MaskStoreError:
            self.abort()
            raise
        self._closed = True
        self._aead = None
        self._wipe_key()
        return self.layout.total_size

    def abort(self) -> None:
        """Delete the temp file (idempotent)."""
        self._closed = True
        self._aead = None
        self._wipe_key()
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


def partial_in_use(path: Path) -> bool:
    """True while a live :class:`RawSegmentWriter` (any process) holds ``path``, a ``.part`` temp file.

    Probes the writer's exclusive ``flock`` with a shared non-blocking one. A missing file or a
    filesystem without ``flock`` reports False (the caller's age rule then applies, as before).
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as exc:
        return exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES)
    finally:
        os.close(fd)  # closing drops the probe lock too
    return False


def _zero_bytes(buf: bytearray) -> None:
    a: Any = np.frombuffer(buf, dtype=np.uint8)
    a.fill(0)
    del a


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
