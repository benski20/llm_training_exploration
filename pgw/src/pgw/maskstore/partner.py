"""Partner segment authentication (SPEC.md section 7.7).

HPKE base mode (how a partner seals a segment DEK to the customer's X25519 public key) does not
authenticate the *sender*: anyone who knows the customer's public key can build a segment that
imports cleanly. If that someone also knows the seed (for example the compute-service operator
crafting segments and slipping them into the delivery channel), every mask is known to them and
the protocol leaks the plaintext. Freivalds checks do not help: such masks are perfectly valid.

Partner factories therefore sign each segment with an Ed25519 key, and the customer configures
the public keys of the partners it trusts (``PGW_MASKSTORE_TRUSTED_PARTNER_KEYS``). The signature
covers ``SHA-256(header JSON)``. The header contains the HPKE-sealed DEK and nonce prefix, and every
chunk is AEAD-authenticated with that header digest as AAD, so one signature over the header
authenticates the whole file.

Signature sidecar: ``<segment file>.sig`` holding canonical JSON
``{"version": 1, "key_id": "<16 hex>", "segment_id": "<32 hex>", "signature": "<base64>"}``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from pgw.core.errors import ConfigError, MaskStoreError
from pgw.maskstore.counters import atomic_write
from pgw.maskstore.format import read_header
from pgw.maskstore.keys import _read_key_file, write_key_file

__all__ = [
    "SIGNATURE_SUFFIX",
    "PartnerSigningKey",
    "SegmentSignature",
    "TrustedPartners",
    "partner_key_id",
    "read_signature",
    "sign_segment",
    "signature_message",
]

log = logging.getLogger(__name__)

SIGNATURE_SUFFIX = ".sig"
_SIG_DOMAIN = b"pgw/v1/segment-signature\0"
_SIG_VERSION = 1
_MAX_SIG_FILE = 4096


def partner_key_id(public_raw: bytes) -> str:
    """Public id of an Ed25519 partner key: first 16 hex chars of SHA-256 of the raw public key."""
    return hashlib.sha256(b"pgw-partner-key\0" + bytes(public_raw)).hexdigest()[:16]


def signature_message(header_raw: bytes) -> bytes:
    """Bytes signed by a partner: domain separator || SHA-256(header JSON)."""
    return _SIG_DOMAIN + hashlib.sha256(header_raw).digest()


@dataclass(frozen=True)
class SegmentSignature:
    """Parsed signature sidecar."""

    key_id: str
    segment_id: str
    signature: bytes


class PartnerSigningKey:
    """Ed25519 private key of a partner mask factory (file mode 0600, 32 raw bytes)."""

    def __init__(self, private_key_file: str | Path):
        self.path = Path(private_key_file)
        self._priv = Ed25519PrivateKey.from_private_bytes(_read_key_file(self.path))
        self._pub = self._priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def __repr__(self) -> str:
        return f"PartnerSigningKey(key_id={self.key_id!r})"

    @classmethod
    def generate(cls, private_key_file: str | Path) -> PartnerSigningKey:
        """Create a new signing key (refuses to overwrite an existing file)."""
        raw = Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
        )
        write_key_file(private_key_file, raw)
        return cls(private_key_file)

    @property
    def key_id(self) -> str:
        """Public key id (what the customer sees in logs and import results)."""
        return partner_key_id(self._pub)

    def public_bytes(self) -> bytes:
        """Raw 32-byte public key to hand to the customer (over an authenticated channel)."""
        return self._pub

    def sign_header(self, header_raw: bytes) -> bytes:
        """Ed25519 signature over :func:`signature_message`."""
        return self._priv.sign(signature_message(header_raw))


def sign_segment(path: str | Path, key: PartnerSigningKey, *, fsync: bool = True) -> Path:
    """Partner side: write ``<path>.sig`` for a finished segment file; return the sidecar path."""
    p = Path(path)
    try:
        with p.open("rb") as f:
            header, raw = read_header(f, p.name)
    except OSError as exc:
        raise MaskStoreError(f"cannot read segment {p.name}: {exc.strerror}") from exc
    doc = {
        "version": _SIG_VERSION,
        "key_id": key.key_id,
        "segment_id": header.segment_id,
        "signature": base64.b64encode(key.sign_header(raw)).decode(),
    }
    out = p.with_name(p.name + SIGNATURE_SUFFIX)
    atomic_write(out, json.dumps(doc, sort_keys=True, separators=(",", ":")).encode(), fsync=fsync, mode=0o644)
    return out


def read_signature(path: str | Path) -> SegmentSignature | None:
    """Parse the sidecar of segment ``path`` (``None`` if there is none)."""
    sp = Path(path)
    sp = sp.with_name(sp.name + SIGNATURE_SUFFIX)
    try:
        data = sp.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MaskStoreError(f"cannot read segment signature {sp.name}: {exc.strerror}") from exc
    if len(data) > _MAX_SIG_FILE:
        raise MaskStoreError(f"segment signature {sp.name} is implausibly large")
    try:
        obj = json.loads(data)
        if not isinstance(obj, dict) or obj.get("version") != _SIG_VERSION:
            raise ValueError("unsupported version")
        key_id, seg_id, sig_b64 = obj["key_id"], obj["segment_id"], obj["signature"]
        if not (isinstance(key_id, str) and isinstance(seg_id, str) and isinstance(sig_b64, str)):
            raise ValueError("bad field types")
        sig = base64.b64decode(sig_b64, validate=True)
    except (ValueError, KeyError, binascii.Error) as exc:
        raise MaskStoreError(f"segment signature {sp.name} is malformed") from exc
    if len(sig) != 64:
        raise MaskStoreError(f"segment signature {sp.name} has the wrong length")
    return SegmentSignature(key_id, seg_id, sig)


def _parse_public(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        raw = value
    else:
        v = value.strip()
        try:
            raw = bytes.fromhex(v) if len(v) == 64 else base64.b64decode(v, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ConfigError("trusted partner key must be 32 bytes as hex (64 chars) or base64") from exc
    if len(raw) != 32:
        raise ConfigError("trusted partner key must be a 32-byte Ed25519 public key")
    return raw


class TrustedPartners:
    """Set of Ed25519 public keys whose segment signatures the store accepts.

    Args:
        keys: Raw 32-byte public keys, or hex / base64 strings.
    """

    def __init__(self, keys: Iterable[str | bytes] = ()):
        self._keys: dict[str, Ed25519PublicKey] = {}
        for k in keys:
            raw = _parse_public(k)
            self._keys[partner_key_id(raw)] = Ed25519PublicKey.from_public_bytes(raw)

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def key_ids(self) -> tuple[str, ...]:
        """Ids of the trusted keys."""
        return tuple(self._keys)

    def verify(self, header_raw: bytes, segment_id: str, sig: SegmentSignature | None) -> str:
        """Check ``sig`` over ``header_raw``; return the signer's key id.

        Raises:
            MaskStoreError: no signature, unknown signer, wrong segment or invalid signature.
        """
        if sig is None:
            raise MaskStoreError(
                "partner segment is not signed; trusted partner keys are configured, so unsigned segments are refused",
                details={"segment_id": segment_id},
            )
        pub = self._keys.get(sig.key_id)
        if pub is None:
            raise MaskStoreError(
                f"partner segment is signed by untrusted key {sig.key_id}",
                details={"segment_id": segment_id, "key_id": sig.key_id},
            )
        if sig.segment_id != segment_id:
            raise MaskStoreError("segment signature belongs to another segment", details={"segment_id": segment_id})
        try:
            pub.verify(sig.signature, signature_message(header_raw))
        except InvalidSignature as exc:
            raise MaskStoreError(
                "partner segment signature is invalid (forged or modified file)",
                details={"segment_id": segment_id, "key_id": sig.key_id},
            ) from exc
        return sig.key_id
