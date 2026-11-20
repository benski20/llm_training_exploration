"""Enrollment codes: one copy-paste string that lets a gateway join the compute service.

Format (version 1)::

    PGW1-XXXXX-XXXXX-...-XXXXX

The groups are RFC 4648 base32 (``A-Z``, ``2-7``; no ``0``, ``1``, ``8`` or ``9``, so they survive
being read out or retyped) of::

    version (1 byte, = 1) | ca_sha256 (32 bytes) | secret (16 bytes) | service URL (UTF-8) | checksum (5 bytes)

``ca_sha256`` is the SHA-256 of the DER encoding of the CA certificate that issued the service's TLS
certificate: the gateway pins it on first contact. ``secret`` is 128 random bits; the service stores
only its SHA-256. ``checksum`` is the first 5 bytes of ``SHA-256("pgw-enroll-v1" | everything before
it)``, so a typo is caught locally before any network call (miss probability 2^-40). Parsing ignores
case, spaces, line breaks and dashes, and reads the look-alikes ``0``/``1``/``8`` as ``O``/``I``/``B``.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import re
import secrets
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from pgw.core.errors import ConfigError

__all__ = ["CODE_PREFIX", "SECRET_BYTES", "EnrollmentCode", "EnrollmentCodeError", "validate_service_url"]

CODE_PREFIX = "PGW1"
SECRET_BYTES = 16
_VERSION = 1
_FP_BYTES = 32
_CHECK_BYTES = 5
_CHECK_DOMAIN = b"pgw-enroll-v1"
_GROUP = 5
_MAX_URL_BYTES = 200
_LOOKALIKES = str.maketrans({"0": "O", "1": "I", "8": "B"})


class EnrollmentCodeError(ConfigError):
    """The enrollment code is malformed, mistyped or of an unsupported version."""


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_service_url(url: str) -> str:
    """Normalized service base URL; https, or plain http to a loopback host only.

    Raises:
        ConfigError: anything else (query, fragment, credentials, other schemes, plain http elsewhere).
    """
    u = url.strip().rstrip("/")
    parts = urlsplit(u)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise ConfigError(f"service URL must be an absolute https URL, got {url!r}")
    if parts.query or parts.fragment or parts.username or parts.password:
        raise ConfigError("service URL must not contain credentials, a query or a fragment")
    try:
        _ = parts.port
    except ValueError as exc:
        raise ConfigError(f"invalid port in service URL {url!r}") from exc
    if parts.scheme == "http" and not _is_loopback(parts.hostname.lower()):
        raise ConfigError("service URL must use https (plain http is accepted for loopback hosts only)")
    if len(u.encode("utf-8")) > _MAX_URL_BYTES:
        raise ConfigError(f"service URL is longer than {_MAX_URL_BYTES} bytes")
    return u


@dataclass(frozen=True)
class EnrollmentCode:
    """Decoded enrollment code.

    Attributes:
        service_url: Base URL of the compute service.
        ca_sha256: SHA-256 (32 bytes) of the service CA certificate (DER); 32 zero bytes = no pin
            (only accepted for plain http to loopback, i.e. tests and local sidecars).
        secret: The single-use secret (16 bytes).
    """

    service_url: str
    ca_sha256: bytes
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_url", validate_service_url(self.service_url))
        if len(self.ca_sha256) != _FP_BYTES:
            raise ConfigError("ca_sha256 must be 32 bytes")
        if len(self.secret) != SECRET_BYTES:
            raise ConfigError(f"secret must be {SECRET_BYTES} bytes")
        if not self.pinned and not self.service_url.startswith("http://"):
            raise ConfigError("an https enrollment code must pin the service CA certificate")

    @classmethod
    def new(cls, service_url: str, ca_sha256: bytes | None) -> EnrollmentCode:
        """Fresh code with a random secret."""
        return cls(service_url, ca_sha256 or bytes(_FP_BYTES), secrets.token_bytes(SECRET_BYTES))

    @property
    def pinned(self) -> bool:
        """Whether a CA fingerprint is pinned."""
        return any(self.ca_sha256)

    @property
    def ca_fingerprint_hex(self) -> str:
        """The pinned CA fingerprint as lowercase hex."""
        return self.ca_sha256.hex()

    @property
    def secret_hex(self) -> str:
        """The secret as sent to ``POST /pgw/v1/enroll``."""
        return self.secret.hex()

    def encode(self) -> str:
        """The printable code (``PGW1-`` + dash-separated base32 groups)."""
        body = bytes([_VERSION]) + self.ca_sha256 + self.secret + self.service_url.encode("utf-8")
        raw = body + _checksum(body)
        b32 = base64.b32encode(raw).decode("ascii").rstrip("=")
        groups = [b32[i:i + _GROUP] for i in range(0, len(b32), _GROUP)]
        return CODE_PREFIX + "-" + "-".join(groups)

    def __str__(self) -> str:  # never print the secret by accident
        return f"EnrollmentCode(service_url={self.service_url!r}, ca={self.ca_fingerprint_hex[:16]}...)"

    @classmethod
    def parse(cls, text: str) -> EnrollmentCode:
        """Decode and verify a code typed or pasted by a person.

        Raises:
            EnrollmentCodeError: wrong prefix or version, invalid characters, or a checksum mismatch
                (a typo); the message is written for a non-technical reader.
        """
        if not isinstance(text, str) or not text.strip():
            raise EnrollmentCodeError("the enrollment code is empty")
        s = re.sub(r"[\s\-_]+", "", text).upper()
        m = re.match(r"^PGW(\d)", s)  # a single version number: the base32 body itself may start with 2-7
        if not m:
            raise EnrollmentCodeError("this is not a PGW enrollment code (it should start with PGW1-)")
        if m.group(1) != "1":
            raise EnrollmentCodeError(f"this enrollment code has version {m.group(1)}; this gateway understands "
                                      "version 1. Update the gateway software or ask for a new code")
        b32 = s[m.end():].translate(_LOOKALIKES)
        if not b32 or not re.fullmatch(r"[A-Z2-7]+", b32):
            raise EnrollmentCodeError("the enrollment code contains characters that cannot appear in a code; "
                                      "copy it again")
        try:
            raw = base64.b32decode(b32 + "=" * (-len(b32) % 8))
        except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
            raise EnrollmentCodeError("the enrollment code is incomplete; copy the whole code again") from exc
        if len(raw) < 1 + _FP_BYTES + SECRET_BYTES + len("http://a") + _CHECK_BYTES:
            raise EnrollmentCodeError("the enrollment code is too short; copy the whole code again")
        body, check = raw[:-_CHECK_BYTES], raw[-_CHECK_BYTES:]
        if not secrets.compare_digest(check, _checksum(body)):
            raise EnrollmentCodeError("the enrollment code has a typo (its checksum does not match); "
                                      "copy and paste it again instead of retyping it")
        if body[0] != _VERSION:
            raise EnrollmentCodeError("unsupported enrollment code version")
        fp = body[1:1 + _FP_BYTES]
        secret = body[1 + _FP_BYTES:1 + _FP_BYTES + SECRET_BYTES]
        try:
            url = body[1 + _FP_BYTES + SECRET_BYTES:].decode("utf-8")
            return cls(url, fp, secret)
        except (UnicodeDecodeError, ConfigError) as exc:
            msg = exc.message if isinstance(exc, ConfigError) else "invalid service URL"
            raise EnrollmentCodeError(f"the enrollment code is not usable: {msg}") from exc


def _checksum(body: bytes) -> bytes:
    return hashlib.sha256(_CHECK_DOMAIN + body).digest()[:_CHECK_BYTES]
