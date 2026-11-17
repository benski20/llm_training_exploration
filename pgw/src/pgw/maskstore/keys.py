"""Key providers: wrap/unwrap per-segment data keys (DEKs) with a key-encryption key (KEK).

Providers never expose the KEK itself through this interface. ``FileKeyProvider`` and
``EnvKeyProvider`` hold the KEK in process memory (acceptable for on-prem with disk encryption);
``Pkcs11KeyProvider`` (HSM) and ``KmsKeyProvider`` (cloud KMS) keep it in hardware.

Providers that support it also offer a hardware-backed monotonic counter, used to detect store
rollback (restoring an old state database would allow mask reuse).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "EnvKeyProvider", "FileKeyProvider", "KeyProvider", "KmsKeyProvider", "Pkcs11KeyProvider",
    "RecipientKeyPair", "WrappedKey", "hpke_wrap_for_recipient",
]


@dataclass(frozen=True)
class WrappedKey:
    """A DEK wrapped under KEK ``kek_id``."""

    kek_id: str
    wrapped: bytes


class KeyProvider(abc.ABC):
    """KEK operations."""

    @property
    @abc.abstractmethod
    def kek_id(self) -> str:
        """Id of the current KEK (new segments are wrapped under it)."""

    @abc.abstractmethod
    def wrap(self, dek: bytes) -> WrappedKey:
        """Wrap a 32-byte DEK (AES-256-GCM or AES-KW under the KEK)."""

    @abc.abstractmethod
    def unwrap(self, wrapped: WrappedKey) -> bytes:
        """Unwrap; raise :class:`pgw.core.MaskStoreError` on auth failure or unknown KEK."""

    def monotonic_counter(self) -> int | None:
        """Current hardware monotonic counter value, or ``None`` if unsupported."""
        return None

    def increment_counter(self) -> int | None:
        """Increment and return the hardware counter, or ``None`` if unsupported."""
        return None


class FileKeyProvider(KeyProvider):
    """KEK = 32 raw bytes in a file with mode 0600 (refuses looser permissions).

    ``kek_id`` = first 16 hex chars of SHA-256(b"pgw-kek-id" || KEK).
    """

    def __init__(self, path: str | Path, *, create: bool = False):
        raise NotImplementedError("maskstore team: FileKeyProvider")

    @property
    def kek_id(self) -> str:
        raise NotImplementedError

    def wrap(self, dek: bytes) -> WrappedKey:
        raise NotImplementedError

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        raise NotImplementedError


class EnvKeyProvider(KeyProvider):
    """KEK from a base64 environment variable (containers with secret injection)."""

    def __init__(self, var: str = "PGW_MASKSTORE_KEK"):
        raise NotImplementedError("maskstore team: EnvKeyProvider")

    @property
    def kek_id(self) -> str:
        raise NotImplementedError

    def wrap(self, dek: bytes) -> WrappedKey:
        raise NotImplementedError

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        raise NotImplementedError


class Pkcs11KeyProvider(KeyProvider):
    """HSM-held KEK via PKCS#11 (``CKM_AES_KEY_WRAP_KWP``). Stub: optional dependency ``python-pkcs11``."""

    def __init__(self, module: str | Path, token_label: str, key_label: str, pin: str | None = None):
        raise NotImplementedError("maskstore team: Pkcs11KeyProvider (stub)")

    @property
    def kek_id(self) -> str:
        raise NotImplementedError

    def wrap(self, dek: bytes) -> WrappedKey:
        raise NotImplementedError

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        raise NotImplementedError


class KmsKeyProvider(KeyProvider):
    """Cloud KMS envelope encryption (``aws-kms://``, ``gcp-kms://``, ``azure-kv://`` URIs). Stub."""

    def __init__(self, key_uri: str):
        raise NotImplementedError("maskstore team: KmsKeyProvider (stub)")

    @property
    def kek_id(self) -> str:
        raise NotImplementedError

    def wrap(self, dek: bytes) -> WrappedKey:
        raise NotImplementedError

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        raise NotImplementedError


class RecipientKeyPair:
    """X25519 key pair used to receive segments from a partner mask factory (SPEC.md section 7.7).

    The partner wraps each segment DEK with HPKE (RFC 9180: DHKEM(X25519, HKDF-SHA256),
    HKDF-SHA256, AES-256-GCM) to the customer's public key; ``kek_id`` is ``"hpke:<key id>"``.
    :meth:`MaskStore.import_segment` unwraps with the private key and re-wraps under the local KEK.
    """

    def __init__(self, private_key_file: str | Path):
        raise NotImplementedError("maskstore team: RecipientKeyPair")

    @classmethod
    def generate(cls, private_key_file: str | Path) -> RecipientKeyPair:
        """Create a new key pair (private key file mode 0600)."""
        raise NotImplementedError

    @property
    def key_id(self) -> str:
        """First 16 hex chars of SHA-256 of the raw public key."""
        raise NotImplementedError

    def public_bytes(self) -> bytes:
        """Raw 32-byte public key to hand to the partner."""
        raise NotImplementedError

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        """HPKE-open a partner-wrapped DEK."""
        raise NotImplementedError


def hpke_wrap_for_recipient(dek: bytes, recipient_public: bytes) -> WrappedKey:
    """Partner side: HPKE-seal ``dek`` to a customer's X25519 public key."""
    raise NotImplementedError("maskstore team: hpke_wrap_for_recipient")
