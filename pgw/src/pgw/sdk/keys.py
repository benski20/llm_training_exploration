"""SDK keys: one string that tells the client library where the provider is, which CA to trust and who it is.

Format (version 1)::

    pgw_<env>_<body>

``env`` is ``[a-z0-9]{1,16}`` (for example ``live`` or ``test``; informational, bound by the checksum).
``body`` is lowercase RFC 4648 base32 without padding of::

    version (1 byte, = 1) | ca_sha256 (32 bytes) | key_id (8 bytes) | secret (24 bytes) | provider URL (UTF-8)
    | checksum (5 bytes)

* ``ca_sha256``: SHA-256 of the DER of the CA certificate that issued the provider's TLS certificate (pinned on
  first contact exactly like an enrollment code, :mod:`pgw.enroll.code`); 32 zero bytes = no pin, accepted only
  for plain ``http://`` to a loopback host (tests, local sidecars).
* ``key_id``: public, stable identifier of the key (revocation, usage metering, logs). Never secret.
* ``secret``: 192 random bits. The provider stores only ``SHA-256(secret)`` (:meth:`SdkKey.secret_sha256`).
* ``checksum``: first 5 bytes of ``SHA-256("pgw-sdk-key-v1" | 0x00 | env | 0x00 | everything before it)``: a
  truncated or mistyped key is refused locally, before any network call.

The key is a reusable, revocable credential: it authenticates ``POST /pgw/v1/sdk/enroll``
(``Authorization: Bearer <the whole key>``), which returns the per-installation service token and client
certificate the library then uses for compute. :func:`parse_key` never echoes the key in an error message.
"""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
from dataclasses import dataclass, field

from pgw.core.errors import ConfigError, ErrorCode
from pgw.enroll.code import validate_service_url

__all__ = ["KEY_ID_BYTES", "SECRET_BYTES", "SdkKey", "SdkKeyError", "parse_key"]

KEY_PREFIX = "pgw"
_VERSION = 1
_FP_BYTES = 32
KEY_ID_BYTES = 8
SECRET_BYTES = 24
_CHECK_BYTES = 5
_CHECK_DOMAIN = b"pgw-sdk-key-v1"
_ENV_RE = re.compile(r"[a-z0-9]{1,16}")
_MAX_URL_BYTES = 200


class SdkKeyError(ConfigError):
    """The SDK key is missing, malformed, mistyped or of an unsupported version (never contains the key)."""

    code = ErrorCode.UNAUTHENTICATED


def _checksum(env: str, body: bytes) -> bytes:
    return hashlib.sha256(_CHECK_DOMAIN + b"\0" + env.encode("ascii") + b"\0" + body).digest()[:_CHECK_BYTES]


@dataclass(frozen=True)
class SdkKey:
    """A decoded SDK key.

    Attributes:
        env: Environment label (``live``, ``test``, ...).
        service_url: Base URL of the provider's compute service.
        ca_sha256: SHA-256 of the provider CA certificate (DER); 32 zero bytes = no pin (loopback http only).
        key_id: Public key identifier (8 bytes).
        secret: The secret part (24 bytes; never printed).
    """

    env: str
    service_url: str
    ca_sha256: bytes
    key_id: bytes
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not _ENV_RE.fullmatch(self.env):
            raise SdkKeyError("SDK key environment label must match [a-z0-9]{1,16}")
        object.__setattr__(self, "service_url", validate_service_url(self.service_url))
        if len(self.service_url.encode("utf-8")) > _MAX_URL_BYTES:
            raise SdkKeyError(f"provider URL is longer than {_MAX_URL_BYTES} bytes")
        if len(self.ca_sha256) != _FP_BYTES:
            raise SdkKeyError("ca_sha256 must be 32 bytes")
        if len(self.key_id) != KEY_ID_BYTES:
            raise SdkKeyError(f"key_id must be {KEY_ID_BYTES} bytes")
        if len(self.secret) != SECRET_BYTES:
            raise SdkKeyError(f"secret must be {SECRET_BYTES} bytes")
        if not self.pinned and not self.service_url.startswith("http://"):
            raise SdkKeyError("an https SDK key must pin the provider's CA certificate")

    # ---- construction (provider side) ------------------------------------------------------------
    @classmethod
    def new(cls, service_url: str, ca_sha256: bytes | None, *, env: str = "live") -> SdkKey:
        """A fresh key with a random id and secret (the provider issues keys with this)."""
        return cls(env, service_url, ca_sha256 or bytes(_FP_BYTES), secrets.token_bytes(KEY_ID_BYTES),
                   secrets.token_bytes(SECRET_BYTES))

    def encode(self) -> str:
        """The printable key ``pgw_<env>_<base32>`` (contains the secret: show it to its owner once)."""
        body = (bytes([_VERSION]) + self.ca_sha256 + self.key_id + self.secret + self.service_url.encode("utf-8"))
        raw = body + _checksum(self.env, body)
        return f"{KEY_PREFIX}_{self.env}_" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()

    # ---- views -----------------------------------------------------------------------------------
    @property
    def pinned(self) -> bool:
        """Whether a CA fingerprint is pinned."""
        return any(self.ca_sha256)

    @property
    def ca_fingerprint_hex(self) -> str:
        """The pinned CA fingerprint as lowercase hex."""
        return self.ca_sha256.hex()

    @property
    def key_id_hex(self) -> str:
        """The public key id as 16 lowercase hex characters (safe to log and display)."""
        return self.key_id.hex()

    def secret_sha256(self) -> bytes:
        """``SHA-256(secret)``: what the provider stores to check the key (constant-time compare)."""
        return hashlib.sha256(self.secret).digest()

    def bearer(self) -> str:
        """The ``Authorization`` value for ``POST /pgw/v1/sdk/enroll`` (``Bearer <key>``)."""
        return "Bearer " + self.encode()

    def __str__(self) -> str:  # never print the secret by accident
        return f"SdkKey(env={self.env!r}, key_id={self.key_id_hex}, service_url={self.service_url!r})"

    __repr__ = __str__


def parse_key(text: str | None) -> SdkKey:
    """Decode and verify an SDK key (``pgw_<env>_<base32>``).

    Whitespace around the key is ignored; case is not significant in the base32 body. Look-alike digits
    ``0``/``1``/``8`` are read as ``o``/``i``/``b`` (keys read aloud or retyped).

    Raises:
        SdkKeyError: missing, malformed, truncated or mistyped key (the message never contains the key).
    """
    if text is None or not isinstance(text, str) or not text.strip():
        raise SdkKeyError("no PGW API key: pass api_key=... or set PGW_API_KEY")
    s = re.sub(r"\s+", "", text)
    parts = s.split("_")
    if len(parts) != 3 or parts[0].lower() != KEY_PREFIX:
        raise SdkKeyError("this is not a PGW SDK key (it should look like pgw_live_...)")
    env, b32 = parts[1].lower(), parts[2].upper().translate(str.maketrans("018", "OIB"))
    if not _ENV_RE.fullmatch(env):
        raise SdkKeyError("the PGW SDK key has an invalid environment label")
    if not b32 or not re.fullmatch(r"[A-Z2-7]+", b32):
        raise SdkKeyError("the PGW SDK key contains characters that cannot appear in a key; copy it again")
    try:
        raw = base64.b32decode(b32 + "=" * (-len(b32) % 8))
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise SdkKeyError("the PGW SDK key is incomplete; copy the whole key again") from exc
    min_len = 1 + _FP_BYTES + KEY_ID_BYTES + SECRET_BYTES + len("http://a") + _CHECK_BYTES
    if len(raw) < min_len:
        raise SdkKeyError("the PGW SDK key is too short; copy the whole key again")
    body, check = raw[:-_CHECK_BYTES], raw[-_CHECK_BYTES:]
    if not secrets.compare_digest(check, _checksum(env, body)):
        raise SdkKeyError("the PGW SDK key has a typo (its checksum does not match); copy and paste it again")
    if body[0] != _VERSION:
        raise SdkKeyError(f"this PGW SDK key has version {body[0]}; update the pgw package")
    o = 1
    ca = body[o:o + _FP_BYTES]
    o += _FP_BYTES
    kid = body[o:o + KEY_ID_BYTES]
    o += KEY_ID_BYTES
    secret = body[o:o + SECRET_BYTES]
    o += SECRET_BYTES
    try:
        url = body[o:].decode("utf-8")
        return SdkKey(env, url, ca, kid, secret)
    except SdkKeyError:
        raise
    except (UnicodeDecodeError, ConfigError) as exc:
        msg = exc.message if isinstance(exc, ConfigError) else "invalid provider URL"
        raise SdkKeyError(f"the PGW SDK key is not usable: {msg}") from exc
