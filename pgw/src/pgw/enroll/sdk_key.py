"""SDK keys: one reusable, revocable string that lets the in-process client library reach the provider.

Format (version 1)::

    pgw_<env>_<base32>

``env`` is ``[a-z][a-z0-9]{0,15}`` (``live``, ``test``, ...). The body is RFC 4648 base32, lowercase, no
padding, of::

    version (1 byte, = 1) | ca_sha256 (32 bytes) | key_id (8 bytes) | secret (24 bytes) | provider URL (UTF-8)
    | checksum (4 bytes)

``ca_sha256`` is the SHA-256 of the DER encoding of the provider CA certificate, pinned exactly as in an
enrollment code (:mod:`pgw.enroll.code`); 32 zero bytes = no pin, accepted for plain http to a loopback host
only. ``key_id`` is public (``sk_`` + 16 hex digits in listings, logs and the audit). ``secret`` is 192 random
bits; the provider stores only ``SHA-256("pgw-sdk-key-secret-v1" | key_id | secret)``. ``checksum`` is the first
4 bytes of ``SHA-256("pgw-sdk-key-v1" | env | 0x00 | everything before it)``: a damaged key is caught locally.

The key is presented to the provider as ``Authorization: Bearer <the whole key>``, only over TLS verified
against the pinned CA (``POST /pgw/v1/sdk/enroll``, ``GET /pgw/v1/sdk/usage``). Stdlib only: the client
library imports this module without the service's dependencies.
"""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
from dataclasses import dataclass, field

from pgw.core.errors import ConfigError

__all__ = [
    "KEY_ID_BYTES", "SDK_KEY_PREFIX", "SECRET_BYTES", "SDKKey", "SDKKeyError", "format_key_id", "parse_key_id",
    "secret_digest",
]

SDK_KEY_PREFIX = "pgw_"
KEY_ID_BYTES = 8
SECRET_BYTES = 24
_VERSION = 1
_FP_BYTES = 32
_CHECK_BYTES = 4
_CHECK_DOMAIN = b"pgw-sdk-key-v1"
_SECRET_DOMAIN = b"pgw-sdk-key-secret-v1"
_ENV_RE = re.compile(r"^[a-z][a-z0-9]{0,15}$")
_KEY_RE = re.compile(r"^pgw_([a-z][a-z0-9]{0,15})_([a-z2-7]+)$")
_KEY_ID_RE = re.compile(r"^sk_([0-9a-f]{16})$")
_MIN_BODY = 1 + _FP_BYTES + KEY_ID_BYTES + SECRET_BYTES + len("http://a")
_MAX_URL_BYTES = 200


class SDKKeyError(ConfigError):
    """The SDK key is malformed, damaged or of an unsupported version (message safe to show; never the key)."""


def format_key_id(raw: bytes) -> str:
    """Display form of a key id: ``sk_`` + 16 lowercase hex digits."""
    if len(raw) != KEY_ID_BYTES:
        raise ConfigError(f"key id must be {KEY_ID_BYTES} bytes")
    return "sk_" + raw.hex()


def parse_key_id(text: str) -> bytes:
    """Raw bytes of a displayed key id (``sk_<16 hex>``).

    Raises:
        ConfigError: not a key id.
    """
    m = _KEY_ID_RE.fullmatch(text.strip()) if isinstance(text, str) else None
    if not m:
        raise ConfigError("a key id looks like sk_ followed by 16 hex digits")
    return bytes.fromhex(m.group(1))


def secret_digest(key_id: bytes, secret: bytes) -> bytes:
    """What the provider stores for a key: ``SHA-256("pgw-sdk-key-secret-v1" | key_id | secret)``."""
    return hashlib.sha256(_SECRET_DOMAIN + key_id + secret).digest()


@dataclass(frozen=True)
class SDKKey:
    """Decoded SDK key.

    Attributes:
        env: Environment label (``live``, ``test``, ...).
        service_url: Base URL of the provider (https, or http to loopback).
        ca_sha256: SHA-256 of the provider CA (DER); 32 zero bytes = no pin (loopback http only).
        key_id: Public key id (8 bytes).
        secret: The credential (24 bytes).
    """

    env: str
    service_url: str
    ca_sha256: bytes
    key_id: bytes
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        from pgw.enroll.code import validate_service_url

        if not isinstance(self.env, str) or not _ENV_RE.fullmatch(self.env):
            raise ConfigError("SDK key env must match [a-z][a-z0-9]{0,15}")
        object.__setattr__(self, "service_url", validate_service_url(self.service_url))
        if len(self.service_url.encode("utf-8")) > _MAX_URL_BYTES:
            raise ConfigError(f"provider URL is longer than {_MAX_URL_BYTES} bytes")
        if len(self.ca_sha256) != _FP_BYTES:
            raise ConfigError("ca_sha256 must be 32 bytes")
        if len(self.key_id) != KEY_ID_BYTES:
            raise ConfigError(f"key id must be {KEY_ID_BYTES} bytes")
        if len(self.secret) != SECRET_BYTES:
            raise ConfigError(f"secret must be {SECRET_BYTES} bytes")
        if not self.pinned and not self.service_url.startswith("http://"):
            raise ConfigError("an https SDK key must pin the provider CA certificate")

    @classmethod
    def new(cls, env: str, service_url: str, ca_sha256: bytes | None) -> SDKKey:
        """Fresh key with a random id and secret."""
        return cls(env, service_url, ca_sha256 or bytes(_FP_BYTES), secrets.token_bytes(KEY_ID_BYTES),
                   secrets.token_bytes(SECRET_BYTES))

    @property
    def pinned(self) -> bool:
        """Whether a CA fingerprint is pinned."""
        return any(self.ca_sha256)

    @property
    def ca_fingerprint_hex(self) -> str:
        """The pinned CA fingerprint as lowercase hex."""
        return self.ca_sha256.hex()

    @property
    def key_id_str(self) -> str:
        """``sk_<16 hex>``."""
        return format_key_id(self.key_id)

    @property
    def key_id_hex(self) -> str:
        """The public key id as 16 lowercase hex digits (cache directory names, logs)."""
        return self.key_id.hex()

    @property
    def digest(self) -> bytes:
        """The stored form of the secret (:func:`secret_digest`)."""
        return secret_digest(self.key_id, self.secret)

    def encode(self) -> str:
        """The printable key."""
        body = (bytes([_VERSION]) + self.ca_sha256 + self.key_id + self.secret + self.service_url.encode("utf-8"))
        raw = body + _checksum(self.env, body)
        return f"{SDK_KEY_PREFIX}{self.env}_" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()

    def bearer(self) -> str:
        """The ``Authorization`` header value (``Bearer <the whole key>``); only for the two SDK routes."""
        return "Bearer " + self.encode()

    def __str__(self) -> str:  # never print the secret by accident
        return f"SDKKey(env={self.env!r}, key_id={self.key_id_str}, service_url={self.service_url!r})"

    __repr__ = __str__

    @classmethod
    def parse(cls, text: str) -> SDKKey:
        """Decode and verify a key (surrounding whitespace ignored; case-sensitive otherwise).

        Raises:
            SDKKeyError: wrong prefix, characters, checksum or version; the message never contains the key.
        """
        if not isinstance(text, str) or not text.strip():
            raise SDKKeyError("the SDK key is empty")
        m = _KEY_RE.fullmatch(text.strip())
        if not m:
            raise SDKKeyError("this is not a PGW SDK key (it should look like pgw_live_...)")
        env, b32 = m.group(1), m.group(2)
        try:
            raw = base64.b32decode(b32.upper() + "=" * (-len(b32) % 8))
        except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
            raise SDKKeyError("the SDK key is incomplete; copy the whole key again") from exc
        if len(raw) < _MIN_BODY + _CHECK_BYTES:
            raise SDKKeyError("the SDK key is too short; copy the whole key again")
        body, check = raw[:-_CHECK_BYTES], raw[-_CHECK_BYTES:]
        if not secrets.compare_digest(check, _checksum(env, body)):
            raise SDKKeyError("the SDK key is damaged (its checksum does not match); copy it again")
        if body[0] != _VERSION:
            raise SDKKeyError(f"SDK key version {body[0]} is not supported by this library; update it")
        o = 1
        fp = body[o:o + _FP_BYTES]
        o += _FP_BYTES
        kid = body[o:o + KEY_ID_BYTES]
        o += KEY_ID_BYTES
        secret = body[o:o + SECRET_BYTES]
        o += SECRET_BYTES
        try:
            return cls(env, body[o:].decode("utf-8"), fp, kid, secret)
        except (UnicodeDecodeError, ConfigError) as exc:
            msg = exc.message if isinstance(exc, ConfigError) else "invalid provider URL"
            raise SDKKeyError(f"the SDK key is not usable: {msg}") from exc


def _checksum(env: str, body: bytes) -> bytes:
    return hashlib.sha256(_CHECK_DOMAIN + env.encode("ascii") + b"\x00" + body).digest()[:_CHECK_BYTES]
