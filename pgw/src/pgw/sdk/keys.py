"""SDK keys on the client side: :func:`parse_key` over the provider's codec (:mod:`pgw.enroll.sdk_key`).

There is one key format and one codec (``docs/sdk/CONTRACT.md``, appendix A1)::

    pgw_<env>_<base32>
    env    = [a-z][a-z0-9]{0,15}
    base32 = RFC 4648, lowercase, no padding, of
             version (1) | ca_sha256 (32) | key_id (8) | secret (24) | provider URL (UTF-8, <= 200 B) | check (4)
    check  = SHA-256("pgw-sdk-key-v1" | env | 0x00 | everything before it)[:4]

The provider issues keys (:meth:`SDKKey.new`) and stores only ``SHA-256("pgw-sdk-key-secret-v1" | key_id |
secret)``; the library decodes them here. This module only adds the forgiving input handling a pasted key needs
(surrounding or embedded whitespace, upper-case body, ``0``/``1``/``8`` typed for ``o``/``i``/``b``: none of them
is a base32 character, so the mapping is unambiguous) and the library's error type. Errors never contain the key.
"""

from __future__ import annotations

import re

from pgw.core.errors import ConfigError, ErrorCode
from pgw.enroll.sdk_key import KEY_ID_BYTES, SECRET_BYTES, SDKKey, SDKKeyError

__all__ = ["KEY_ID_BYTES", "SECRET_BYTES", "SDKKey", "SdkKey", "SdkKeyError", "parse_key"]

#: The key class (alias of :class:`pgw.enroll.sdk_key.SDKKey`; one codec for the provider and the library).
SdkKey = SDKKey


class SdkKeyError(SDKKeyError):
    """The SDK key is missing, malformed, mistyped or of an unsupported version (never contains the key).

    Maps to :class:`pgw.sdk.AuthenticationError` (code ``unauthenticated``) at the API surface."""

    code = ErrorCode.UNAUTHENTICATED


_LOOKALIKE = str.maketrans("018", "oib")


def parse_key(text: str | None) -> SDKKey:
    """Decode and verify an SDK key (``pgw_<env>_<base32>``) with :meth:`SDKKey.parse`.

    Whitespace anywhere in the key is dropped, the body's case does not matter, and the look-alike digits
    ``0``/``1``/``8`` are read as ``o``/``i``/``b`` (keys read aloud or retyped); the checksum still has to match.

    Raises:
        SdkKeyError: missing, malformed, truncated or mistyped key (the message never contains the key).
    """
    if text is None or not isinstance(text, str) or not text.strip():
        raise SdkKeyError("no PGW API key: pass api_key=... or set PGW_API_KEY")
    s = re.sub(r"\s+", "", text)
    parts = s.split("_")
    if len(parts) == 3 and parts[0].lower() == "pgw":
        s = f"pgw_{parts[1].lower()}_{parts[2].lower().translate(_LOOKALIKE)}"
    try:
        return SDKKey.parse(s)
    except ConfigError as exc:
        raise SdkKeyError(exc.message) from None
