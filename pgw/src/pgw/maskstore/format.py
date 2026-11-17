"""On-disk segment format (``*.pgwm``). Normative description: SPEC.md section 7.2.

Layout::

    magic   8 bytes  b"PGWMASK1"
    hlen    u32 LE   length of header JSON
    header  hlen     canonical JSON of SegmentHeader (authenticated as AAD of every chunk)
    chunks  ...      AES-256-GCM ciphertexts, each = ciphertext || 16-byte tag

Chunks, in order:
  1. ``secrets`` chunk: segment seed (32 bytes) || for each linear in header order: ``uW`` packed
     ``(checks, n)`` ring elements.
  2. for each linear (header order), for each group of ``chunk_slots`` slots: ``W r`` rows packed
     ``(rows, m)`` ring elements (K/8 bytes little-endian each).

Each chunk's nonce is ``nonce_prefix (4 bytes) || u64_be(chunk_index)``; AAD is
``SHA-256(header JSON) || u32_be(chunk_index)``. The data key (DEK, 32 random bytes per segment)
is wrapped by the key provider's KEK; the wrapped DEK and KEK id are in the header.
Offsets of every chunk are derivable from the header, so a reader can decrypt only the rows it needs.

Mask derivation (shared with the factory and gateway)::

    r[slot, :]  = SeedExpander(seed).ring_uniform_rows(n, K, slot, 1, "r", linear_id)
    u           = SeedExpander(seed).ring_uniform((checks, m), K, "u", linear_id)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["GCM_TAG_BYTES", "SEGMENT_MAGIC", "SEGMENT_SUFFIX", "SEGMENT_VERSION", "SegmentHeader", "SegmentLinear"]

SEGMENT_MAGIC = b"PGWMASK1"
SEGMENT_VERSION = 1
SEGMENT_SUFFIX = ".pgwm"
GCM_TAG_BYTES = 16


class SegmentLinear(BaseModel):
    """Per-linear entry of a segment header."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    out_features: int
    in_features: int


class SegmentHeader(BaseModel):
    """Plaintext (but authenticated) segment header."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: int = SEGMENT_VERSION
    segment_id: str = Field(description="UUIDv4 hex; unique forever")
    fingerprint: str
    ring: str = Field(description="e.g. 'K40A12'")
    slots: int = Field(gt=0, description="Token positions in this segment")
    chunk_slots: int = Field(gt=0, description="Slots per W r chunk")
    checks: int = Field(ge=0, description="Freivalds check vectors per linear")
    linears: tuple[SegmentLinear, ...]
    kek_id: str
    wrapped_dek: str = Field(description="base64 wrapped data key")
    nonce_prefix: str = Field(description="hex, 4 bytes")
    factory_id: str = Field(description="Identifier of the producing factory (no secrets)")
    created_at: str
