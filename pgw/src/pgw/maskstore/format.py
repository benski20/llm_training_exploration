"""On-disk segment format (``*.pgwm``). Normative description: SPEC.md section 7.2.

Layout::

    magic   8 bytes  b"PGWMASK1"
    hlen    u32 LE   length of header JSON
    header  hlen     canonical JSON of SegmentHeader (authenticated as AAD of every chunk)
    chunks  ...      AES-256-GCM ciphertexts, each = ciphertext || 16-byte tag

Chunks, in order:
  1. ``secrets`` chunk: segment seed (32 bytes) || for each linear in header order: ``uW`` packed
     ``(checks, n)`` ring elements (with ``check_field = "p31"``: ``u W mod p`` for ``p = 2^31 - 1``,
     packed the same way; only K32 rings, so every word is 4 bytes and below ``2^31``).
  2. for each linear (header order), for each group of ``chunk_slots`` slots: ``W r`` rows packed
     ``(rows, m)`` ring elements (K/8 bytes little-endian each).

Each chunk's nonce is ``nonce_prefix (4 bytes) || u64_be(chunk_index)``; AAD is
``SHA-256(header JSON) || u32_be(chunk_index)``. The data key (DEK, 32 random bytes per segment)
is wrapped by the key provider's KEK; the wrapped DEK and KEK id are in the header.
Offsets of every chunk are derivable from the header, so a reader can decrypt only the rows it needs.

Mask derivation (shared with the factory and gateway)::

    r[slot, :]  = SeedExpander(seed).ring_uniform_rows(n, K, slot, 1, "r", linear_id)
    u           = SeedExpander(seed).ring_uniform((checks, m), K, "u", linear_id)
    u (p31)     = u64le(stream("u-p31", linear_id)) mod (2^31 - 1)      # check_field = "p31"
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pgw.core.errors import ConfigError, MaskStoreError
from pgw.core.fingerprint import canonical_json, validate_fingerprint
from pgw.core.prng import SEED_BYTES
from pgw.core.ring import RingParams
from pgw.core.types import LinearId

__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEK_BYTES",
    "GCM_TAG_BYTES",
    "MAX_CHUNK_BYTES",
    "MAX_HEADER_BYTES",
    "NONCE_PREFIX_BYTES",
    "SEGMENT_MAGIC",
    "SEGMENT_SUFFIX",
    "SEGMENT_VERSION",
    "ChunkRef",
    "SegmentFile",
    "SegmentHeader",
    "SegmentLayout",
    "SegmentLinear",
    "auto_chunk_slots",
    "chunk_aad",
    "chunk_cache_bytes",
    "chunk_nonce",
    "encode_header",
    "read_header",
    "seed_hash",
]

SEGMENT_MAGIC = b"PGWMASK1"
SEGMENT_VERSION = 1
SEGMENT_SUFFIX = ".pgwm"
GCM_TAG_BYTES = 16
DEK_BYTES = 32
NONCE_PREFIX_BYTES = 4
#: Default plaintext size of one encrypted ``W r`` chunk of the widest linear. A reader decrypts
#: whole chunks and keeps one per linear cached per reservation, so this bounds gateway memory per
#: session at about ``DEFAULT_CHUNK_BYTES x (sum of out_features / max out_features)``.
DEFAULT_CHUNK_BYTES = 1 << 20
#: Writers refuse chunks whose widest linear exceeds this many plaintext bytes.
MAX_CHUNK_BYTES = 16 << 20


def auto_chunk_slots(out_features: Sequence[int], ring: RingParams, target_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """Slots per encrypted chunk so that the widest linear's chunk is at most ``target_bytes``."""
    if target_bytes <= 0:
        raise ConfigError("target_bytes must be positive")
    widest = max((int(m) for m in out_features), default=1)
    return max(1, target_bytes // (widest * ring.bytes_per_element))


def chunk_cache_bytes(chunk_slots: int, out_features: Sequence[int], ring: RingParams) -> int:
    """Plaintext ``W r`` bytes a reservation caches when it holds one chunk of every linear."""
    return int(chunk_slots) * sum(int(m) for m in out_features) * ring.bytes_per_element
#: Upper bound on header size (a 200-layer model header is a few tens of KB).
MAX_HEADER_BYTES = 16 << 20
_PREAMBLE = len(SEGMENT_MAGIC) + 4
_SEG_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX8_RE = re.compile(r"^[0-9a-f]{8}$")


class SegmentLinear(BaseModel):
    """Per-linear entry of a segment header."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    out_features: int = Field(gt=0)
    in_features: int = Field(gt=0)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        try:
            return LinearId.parse(v).key
        except ConfigError as exc:
            raise ValueError(str(exc)) from exc


class SegmentHeader(BaseModel):
    """Plaintext (but authenticated) segment header."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: int = SEGMENT_VERSION
    segment_id: str = Field(description="UUIDv4 hex; unique forever")
    fingerprint: str
    ring: str = Field(description="e.g. 'K32A12C27'")
    slots: int = Field(gt=0, description="Token positions in this segment")
    chunk_slots: int = Field(gt=0, description="Slots per W r chunk")
    checks: int = Field(ge=0, description="Freivalds check vectors per linear")
    linears: tuple[SegmentLinear, ...]
    kek_id: str
    wrapped_dek: str = Field(description="base64 wrapped data key")
    nonce_prefix: str = Field(description="hex, 4 bytes")
    factory_id: str = Field(description="Identifier of the producing factory (no secrets)")
    created_at: str
    destination: str | None = Field(
        None,
        description="Store id of the only customer store allowed to import this segment (partner exports). "
        "Omitted from the JSON when unset, so older headers keep their exact bytes",
    )
    check_field: str | None = Field(
        None,
        description="Freivalds check format: unset = the ring check over Z_2^K (format 1); 'p31' = one or more "
        "vectors over F_p, p = 2^31 - 1, on the unmasked values (SPEC 5.1). Omitted from the JSON when unset, "
        "so older headers keep their exact bytes; readers that do not know the field refuse the header",
    )

    @field_validator("version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SEGMENT_VERSION:
            raise ValueError(f"unsupported segment version {v}")
        return v

    @field_validator("segment_id")
    @classmethod
    def _seg_id(cls, v: str) -> str:
        if not _SEG_ID_RE.match(v):
            raise ValueError("segment_id must be 32 lowercase hex chars")
        return v

    @field_validator("fingerprint")
    @classmethod
    def _fp(cls, v: str) -> str:
        try:
            return validate_fingerprint(v)
        except ConfigError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("ring")
    @classmethod
    def _ring(cls, v: str) -> str:
        try:
            return RingParams.from_name(v).name
        except ConfigError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("destination")
    @classmethod
    def _destination(cls, v: str | None) -> str | None:
        if v is not None and not _SEG_ID_RE.match(v):
            raise ValueError("destination must be a store id (32 lowercase hex chars)")
        return v

    @field_validator("check_field")
    @classmethod
    def _check_field(cls, v: str | None) -> str | None:
        if v is not None and v != "p31":
            raise ValueError(f"unknown check_field {v!r} (known: p31)")
        return v

    @model_validator(mode="after")
    def _p31_ring(self) -> SegmentHeader:
        if self.check_field is not None and RingParams.from_name(self.ring).k != 32:
            raise ValueError("check_field 'p31' needs a K32 ring")
        return self

    @field_validator("nonce_prefix")
    @classmethod
    def _nonce(cls, v: str) -> str:
        if not _HEX8_RE.match(v):
            raise ValueError("nonce_prefix must be 8 lowercase hex chars")
        return v

    @field_validator("linears")
    @classmethod
    def _linears(cls, v: tuple[SegmentLinear, ...]) -> tuple[SegmentLinear, ...]:
        if not v:
            raise ValueError("segment has no linears")
        ids = [x.id for x in v]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate linear ids")
        return v

    # ---- helpers ---------------------------------------------------------------------------------
    @property
    def ring_params(self) -> RingParams:
        """Parsed ring."""
        return RingParams.from_name(self.ring)

    @property
    def wrapped_dek_bytes(self) -> bytes:
        """Decoded wrapped DEK."""
        try:
            return base64.b64decode(self.wrapped_dek, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise MaskStoreError(f"segment {self.segment_id}: wrapped_dek is not base64") from exc

    def linear(self, lid: LinearId | str) -> tuple[int, SegmentLinear]:
        """``(index, entry)`` of linear ``lid``; raise :class:`ConfigError` if absent."""
        key = str(lid)
        for i, lin in enumerate(self.linears):
            if lin.id == key:
                return i, lin
        raise ConfigError(f"linear {key} is not part of segment {self.segment_id}")


def encode_header(header: SegmentHeader) -> bytes:
    """Canonical JSON bytes of a header (what is stored and authenticated)."""
    data = header.model_dump(mode="json")
    if data.get("destination") is None:
        data.pop("destination", None)
    if data.get("check_field") is None:
        data.pop("check_field", None)
    return canonical_json(data)


def chunk_nonce(nonce_prefix: bytes, index: int) -> bytes:
    """12-byte GCM nonce ``prefix(4) || u64be(index)``."""
    return nonce_prefix + struct.pack(">Q", index)


def chunk_aad(header_digest: bytes, index: int) -> bytes:
    """AAD ``SHA-256(header JSON) || u32be(index)``."""
    return header_digest + struct.pack(">I", index)


def seed_hash(seed: bytes) -> str:
    """Public identifier of a segment seed (used to refuse duplicate seeds forever)."""
    return hashlib.sha256(b"pgw/v1/seed-id\0" + bytes(seed)).hexdigest()


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """Position of one chunk in the file."""

    index: int
    offset: int  #: byte offset of the ciphertext
    plain_len: int  #: plaintext length (ciphertext is ``plain_len + 16``)

    @property
    def cipher_len(self) -> int:
        """Ciphertext length including the tag."""
        return self.plain_len + GCM_TAG_BYTES


class SegmentLayout:
    """Chunk geometry derived from a header (no I/O)."""

    def __init__(self, header: SegmentHeader, header_len: int):
        self.header = header
        self.ring = header.ring_params
        self.bpe = self.ring.bytes_per_element
        self.data_start = _PREAMBLE + header_len
        self.chunks_per_linear = -(-header.slots // header.chunk_slots)
        self.secrets_len = SEED_BYTES + sum(header.checks * lin.in_features * self.bpe for lin in header.linears)
        # cumulative ciphertext offsets per linear
        self._linear_base: list[int] = []
        off = self.data_start + self.secrets_len + GCM_TAG_BYTES
        full = header.chunk_slots
        last = header.slots - (self.chunks_per_linear - 1) * full
        for lin in header.linears:
            self._linear_base.append(off)
            row_bytes = lin.out_features * self.bpe
            off += (self.chunks_per_linear - 1) * (full * row_bytes + GCM_TAG_BYTES)
            off += last * row_bytes + GCM_TAG_BYTES
        self.total_size = off

    def secrets_chunk(self) -> ChunkRef:
        """Chunk 0 (seed and ``uW``)."""
        return ChunkRef(0, self.data_start, self.secrets_len)

    def chunk_rows(self, j: int) -> int:
        """Rows in W r chunk ``j`` of any linear."""
        return min(self.header.chunk_slots, self.header.slots - j * self.header.chunk_slots)

    def wr_chunk(self, linear_index: int, j: int) -> ChunkRef:
        """W r chunk ``j`` (slots ``[j*chunk_slots, ...)``) of linear number ``linear_index``."""
        if not 0 <= j < self.chunks_per_linear:
            raise ConfigError(f"chunk {j} out of range")
        lin = self.header.linears[linear_index]
        row_bytes = lin.out_features * self.bpe
        off = self._linear_base[linear_index] + j * (self.header.chunk_slots * row_bytes + GCM_TAG_BYTES)
        return ChunkRef(1 + linear_index * self.chunks_per_linear + j, off, self.chunk_rows(j) * row_bytes)

    def chunks(self) -> list[ChunkRef]:
        """All chunk references in file order (chunk index order)."""
        refs = [self.secrets_chunk()]
        for li in range(len(self.header.linears)):
            refs.extend(self.wr_chunk(li, j) for j in range(self.chunks_per_linear))
        return refs

    def secrets_split(self, plain: bytes | bytearray | memoryview) -> tuple[bytes, list[memoryview]]:
        """Split chunk-0 plaintext into ``(seed, [uW bytes per linear])``."""
        if len(plain) != self.secrets_len:
            raise MaskStoreError("secrets chunk has the wrong length")
        mv = memoryview(plain)
        seed = bytes(mv[:SEED_BYTES])
        out: list[memoryview] = []
        off = SEED_BYTES
        for lin in self.header.linears:
            n = self.header.checks * lin.in_features * self.bpe
            out.append(mv[off : off + n])
            off += n
        return seed, out


def read_header(f: BinaryIO, name: str = "segment") -> tuple[SegmentHeader, bytes]:
    """Read and validate the preamble and header; return ``(header, raw header bytes)``.

    Raises:
        MaskStoreError: bad magic, truncated file or malformed header.
    """
    pre = f.read(_PREAMBLE)
    if len(pre) != _PREAMBLE or pre[:8] != SEGMENT_MAGIC:
        raise MaskStoreError(f"{name}: not a PGW mask segment (bad magic)")
    (hlen,) = struct.unpack("<I", pre[8:])
    if not 0 < hlen <= MAX_HEADER_BYTES:
        raise MaskStoreError(f"{name}: implausible header length {hlen}")
    raw = f.read(hlen)
    if len(raw) != hlen:
        raise MaskStoreError(f"{name}: truncated header")
    try:
        obj = json.loads(raw)
        header = SegmentHeader.model_validate(obj)
    except (ValueError, ValidationError) as exc:
        raise MaskStoreError(f"{name}: malformed segment header: {_short(exc)}") from exc
    return header, raw


def _short(exc: Exception) -> str:
    s = str(exc).splitlines()
    return s[0] if s else type(exc).__name__


class SegmentFile:
    """Read-only access to an encrypted segment file.

    Args:
        path: File path.
        dek: Unwrapped data key; may be set later with :meth:`set_dek`.

    The file descriptor stays open for the lifetime of the object, so a segment that is unlinked
    while a reservation still reads from it remains readable (POSIX semantics).
    """

    def __init__(self, path: str | Path, dek: bytes | None = None):
        self.path = Path(path)
        try:
            self._f: BinaryIO | None = self.path.open("rb")
        except OSError as exc:
            raise MaskStoreError(f"cannot open segment {self.path.name}: {exc.strerror}") from exc
        try:
            self.header, self.header_raw = read_header(self._f, self.path.name)
            self.layout = SegmentLayout(self.header, len(self.header_raw))
            size = os.fstat(self._f.fileno()).st_size
            if size != self.layout.total_size:
                raise MaskStoreError(
                    f"{self.path.name}: file size {size} does not match header ({self.layout.total_size})",
                    details={"segment_id": self.header.segment_id},
                )
        except BaseException:
            self.close()
            raise
        self.header_digest = hashlib.sha256(self.header_raw).digest()
        self._nonce_prefix = bytes.fromhex(self.header.nonce_prefix)
        self._aead: AESGCM | None = None
        self._dek: bytearray | None = None
        if dek is not None:
            self.set_dek(dek)

    def set_dek(self, dek: bytes) -> None:
        """Install the unwrapped DEK."""
        if len(dek) != DEK_BYTES:
            raise MaskStoreError("data key must be 32 bytes")
        self._aead = AESGCM(bytes(dek))
        self._dek = bytearray(dek)

    def _read(self, ref: ChunkRef) -> bytes:
        if self._f is None:
            raise MaskStoreError(f"segment {self.header.segment_id} is closed")
        try:
            data = os.pread(self._f.fileno(), ref.cipher_len, ref.offset)
        except OSError as exc:
            raise MaskStoreError(f"read error in segment {self.header.segment_id}: {exc.strerror}") from exc
        if len(data) != ref.cipher_len:
            raise MaskStoreError(f"segment {self.header.segment_id} is truncated")
        return data

    def decrypt(self, ref: ChunkRef) -> bytes:
        """Read, authenticate and decrypt one chunk."""
        if self._aead is None:
            raise MaskStoreError("segment data key not available")
        ct = self._read(ref)
        try:
            return self._aead.decrypt(
                chunk_nonce(self._nonce_prefix, ref.index), ct, chunk_aad(self.header_digest, ref.index)
            )
        except InvalidTag as exc:
            raise MaskStoreError(
                f"segment {self.header.segment_id}: chunk {ref.index} failed authentication (corrupt or tampered)",
                details={"segment_id": self.header.segment_id, "chunk": ref.index},
            ) from exc

    def decrypt_into(self, ref: ChunkRef, out: bytearray | None = None) -> bytearray:
        """Like :meth:`decrypt`, but into a zeroable ``bytearray`` (no immutable plaintext copy, and
        about 7x faster for mask chunks). Zeroed and discarded if authentication fails.

        ``out``: a bytearray of exactly ``ref.plain_len`` bytes to decrypt into (a chunk buffer the
        caller is done with; it is overwritten). Reusing buffers saves the allocation and page
        faults of a fresh 1 MiB buffer per chunk. On failure ``out`` is zeroed.
        """
        if self._aead is None or self._dek is None:
            raise MaskStoreError("segment data key not available")
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        ct = self._read(ref)
        n = len(ct) - 16
        if n < 0:
            raise MaskStoreError(f"segment {self.header.segment_id} chunk {ref.index} is truncated")
        reuse = out is not None and len(out) == n
        if not reuse:
            out = bytearray(n + 15)  # update_into needs room for a partial block
        assert out is not None
        tail = bytearray(31) if reuse else None  # last <= 16 bytes of a reused (exact-size) buffer
        try:
            d = Cipher(algorithms.AES(self._dek), modes.GCM(chunk_nonce(self._nonce_prefix, ref.index),
                                                             ct[n:])).decryptor()
            d.authenticate_additional_data(chunk_aad(self.header_digest, ref.index))
            cv = memoryview(ct)
            if tail is None:
                got = d.update_into(cv[:n], out)
            else:
                head = max(0, n - 16)
                got = d.update_into(cv[:head], out) if head else 0
                got += d.update_into(cv[head:n], tail)
                out[head:n] = memoryview(tail)[: n - head]
            d.finalize()
            if got != n:  # pragma: no cover - GCM is a stream mode
                raise InvalidTag
        except InvalidTag as exc:
            out[:] = bytes(len(out))
            raise MaskStoreError(
                f"segment {self.header.segment_id}: chunk {ref.index} failed authentication (corrupt or tampered)",
                details={"segment_id": self.header.segment_id, "chunk": ref.index},
            ) from exc
        finally:
            if tail is not None:
                tail[:] = bytes(31)
        if not reuse:
            del out[n:]
        return out

    def iter_chunks(self) -> list[ChunkRef]:
        """All chunk references in file order."""
        return self.layout.chunks()

    def close(self) -> None:
        """Close the file and drop the key."""
        self._aead = None
        dek = getattr(self, "_dek", None)  # close() also runs when __init__ failed early
        if dek is not None:
            dek[:] = bytes(len(dek))
            self._dek = None
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None

    def __enter__(self) -> SegmentFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
