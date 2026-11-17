"""Deterministic CSPRNG seed expansion (ChaCha20 keyed by HKDF-SHA256).

A 32-byte master seed plus a sequence of *labels* deterministically produces an independent
pseudorandom stream. This is how the mask factory and the gateway agree on mask vectors ``r``
without ever storing them: the store keeps the segment seed and the gateway regenerates ``r``.

Construction (normative, SPEC.md section 7.3)::

    info   = b"pgw/v1/prng" || for each label: u32_be(len(enc(label))) || enc(label)
    key    = HKDF-Expand(SHA-256, prk=seed, info=info, L=32)
    stream = ChaCha20(key, counter=0, nonce=0^12)   # 16-byte cryptography nonce = counter||nonce

Label encoding ``enc``: ``str`` -> UTF-8 prefixed by ``b"s"``, ``int`` -> ``b"i"`` + signed 8-byte big
endian, ``bytes`` -> ``b"b"`` + raw. Typed prefixes make ``("1",)`` and ``(1,)`` distinct.

Ring elements: 8 stream bytes per element, little-endian uint64, reduced mod ``2^K`` (exactly
uniform because ``2^K`` divides ``2^64``).
"""

from __future__ import annotations

import os
import struct
from collections.abc import Sequence

import numpy as np
import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

from pgw.core.errors import ConfigError

__all__ = ["MAX_STREAM_BYTES", "SEED_BYTES", "Label", "SeedExpander", "encode_labels", "max_stream_rows", "new_seed"]

SEED_BYTES = 32
Label = str | int | bytes
_DOMAIN = b"pgw/v1/prng"
_ZERO_NONCE = bytes(16)
#: Bytes addressable in one labelled stream (ChaCha20's 32-bit block counter x 64-byte blocks).
MAX_STREAM_BYTES = (1 << 32) * 64


def max_stream_rows(cols: int) -> int:
    """Largest ``row_start + row_count`` that :meth:`SeedExpander.ring_uniform_rows` supports for ``cols``.

    A segment's ``r`` stream of a linear with ``in_features = cols`` therefore holds at most this
    many slots (about 2.4M for an 8B down projection, 1.2M for a 70B one).
    """
    if cols <= 0:
        raise ConfigError("cols must be positive")
    return MAX_STREAM_BYTES // (8 * cols)


def new_seed() -> bytes:
    """Fresh 32-byte seed from the OS CSPRNG."""
    return os.urandom(SEED_BYTES)


def _enc(label: Label) -> bytes:
    if isinstance(label, bool):  # bool is an int subclass; reject to avoid silent ambiguity
        raise ConfigError("bool is not a valid PRNG label")
    if isinstance(label, int):
        return b"i" + struct.pack(">q", label)
    if isinstance(label, str):
        return b"s" + label.encode("utf-8")
    if isinstance(label, bytes | bytearray | memoryview):
        return b"b" + bytes(label)
    raise ConfigError(f"unsupported PRNG label type {type(label).__name__}")


def encode_labels(labels: Sequence[Label]) -> bytes:
    """Injective encoding of a label sequence (used as HKDF info)."""
    out = bytearray(_DOMAIN)
    for lab in labels:
        e = _enc(lab)
        out += struct.pack(">I", len(e)) + e
    return bytes(out)


class SeedExpander:
    """Expands a 32-byte seed into labelled, independent pseudorandom streams.

    Example:
        >>> sx = SeedExpander(bytes(32))
        >>> r = sx.ring_uniform((4, 8), 40, "mask", "3.qkv", 17)
        >>> child = sx.child("segment", 7)   # derive a sub-seed
    """

    __slots__ = ("_seed",)

    def __init__(self, seed: bytes):
        if not isinstance(seed, bytes | bytearray) or len(seed) != SEED_BYTES:
            raise ConfigError(f"seed must be exactly {SEED_BYTES} bytes")
        self._seed = bytes(seed)

    def __repr__(self) -> str:  # never print key material
        return "SeedExpander(<redacted>)"

    def _key(self, labels: Sequence[Label]) -> bytes:
        return bytes(HKDFExpand(hashes.SHA256(), 32, encode_labels(labels)).derive(self._seed))

    def child(self, *labels: Label) -> SeedExpander:
        """Derive an independent sub-expander (its seed is the HKDF output for ``labels``)."""
        return SeedExpander(self._key(("child", *labels)))

    def derive_seed(self, *labels: Label) -> bytes:
        """Derive a 32-byte sub-seed (e.g. to hand a segment seed to a store)."""
        return self._key(("child", *labels))

    def random_bytes(self, n: int, *labels: Label) -> bytes:
        """``n`` pseudorandom bytes of the stream identified by ``labels``."""
        if n < 0:
            raise ConfigError("n must be non-negative")
        enc = Cipher(algorithms.ChaCha20(self._key(labels), _ZERO_NONCE), mode=None).encryptor()
        return bytes(enc.update(bytes(n)))

    def ring_uniform(self, shape: tuple[int, ...] | int, k: int, *labels: Label) -> torch.Tensor:
        """Uniform ring elements in ``[0, 2^k)`` as an int64 CPU tensor of ``shape``."""
        shp = (shape,) if isinstance(shape, int) else tuple(shape)
        count = int(np.prod(shp, dtype=np.int64)) if shp else 1
        raw = np.frombuffer(self.random_bytes(8 * count, *labels), dtype="<u8")
        vals = (raw & np.uint64((1 << k) - 1)).astype(np.int64)
        return torch.from_numpy(vals.reshape(shp))

    def ring_uniform_rows(self, cols: int, k: int, row_start: int, row_count: int, *labels: Label) -> torch.Tensor:
        """Rows ``[row_start, row_start + row_count)`` of the ``(rows, cols)`` matrix that
        ``ring_uniform((rows, cols), k, *labels)`` would produce, without generating earlier rows.

        Uses ChaCha20 block-counter seeking, so regenerating a mask slice costs only its own size.
        """
        if cols <= 0 or row_start < 0 or row_count < 0:
            raise ConfigError("cols must be > 0 and row_start/row_count >= 0")
        offset = 8 * cols * row_start
        length = 8 * cols * row_count
        block, skip = divmod(offset, 64)
        if block >= 1 << 32:
            raise ConfigError("stream offset beyond the 2^32-block ChaCha20 counter range")
        nonce = block.to_bytes(4, "little") + bytes(12)
        enc = Cipher(algorithms.ChaCha20(self._key(labels), nonce), mode=None).encryptor()
        raw = np.frombuffer(enc.update(bytes(skip + length))[skip:], dtype="<u8")
        vals = (raw & np.uint64((1 << k) - 1)).astype(np.int64)
        return torch.from_numpy(vals.reshape(row_count, cols))

    def uniform_int(self, shape: tuple[int, ...] | int, low: int, high: int, *labels: Label) -> torch.Tensor:
        """Integers in ``[low, high)`` (rejection-free modulo; bias < 2^-32 for ranges < 2^32)."""
        if high <= low or high - low > (1 << 32):
            raise ConfigError("need low < high and high - low <= 2^32")
        shp = (shape,) if isinstance(shape, int) else tuple(shape)
        count = int(np.prod(shp, dtype=np.int64)) if shp else 1
        raw = np.frombuffer(self.random_bytes(8 * count, *labels), dtype="<u8")
        vals = (raw % np.uint64(high - low)).astype(np.int64) + low
        return torch.from_numpy(vals.reshape(shp))
