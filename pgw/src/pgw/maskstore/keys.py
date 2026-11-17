"""Key providers: wrap/unwrap per-segment data keys (DEKs) with a key-encryption key (KEK).

Providers never expose the KEK itself through this interface. ``FileKeyProvider`` and
``EnvKeyProvider`` hold the KEK in process memory (acceptable for on-prem with disk encryption);
``Pkcs11KeyProvider`` (HSM) and ``KmsKeyProvider`` (cloud KMS) keep it in hardware.

Providers that support it also offer:

* a keyed MAC (:meth:`KeyProvider.mac`), used by the store to authenticate its state rows and its
  HMAC-chained generation journal (rollback detection without hardware), and
* a hardware-backed monotonic counter, used to detect store rollback (restoring an old state
  database would allow mask reuse).

Wrapping formats (``WrappedKey.wrapped``):

* file/env: ``0x01 || nonce(12) || AES-256-GCM(KEK, nonce, DEK, aad="pgw/v1/dek-wrap" || kek_id)``.
* pkcs11: ``CKM_AES_KEY_WRAP_KWP`` output (RFC 5649).
* kms: the provider's ciphertext blob, with the KEK id as encryption context / AAD.
* hpke: ``enc(32) || AES-256-GCM ciphertext`` (RFC 9180 base mode, info = "pgw/v1/segment-dek" || key id).
"""

from __future__ import annotations

import abc
import base64
import binascii
import hashlib
import hmac
import logging
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from pgw.core.errors import ConfigError, MaskStoreError

__all__ = [
    "EnvKeyProvider", "FileKeyProvider", "KeyProvider", "KmsClient", "KmsKeyProvider", "Pkcs11KeyProvider",
    "RecipientKeyPair", "StaticKeyProvider", "WrappedKey", "hpke_wrap_for_recipient", "kek_id_for",
    "write_key_file",
]

log = logging.getLogger(__name__)

KEK_BYTES = 32
DEK_BYTES = 32
_WRAP_V1 = b"\x01"
_WRAP_AAD = b"pgw/v1/dek-wrap\0"
_MAC_INFO = b"pgw/v1/maskstore-mac"
_HPKE_INFO = b"pgw/v1/segment-dek\0"


@dataclass(frozen=True)
class WrappedKey:
    """A DEK wrapped under KEK ``kek_id``."""

    kek_id: str
    wrapped: bytes

    def __repr__(self) -> str:
        return f"WrappedKey(kek_id={self.kek_id!r}, wrapped=<{len(self.wrapped)} bytes>)"


def kek_id_for(kek: bytes) -> str:
    """Public id of a raw KEK: first 16 hex chars of SHA-256(b"pgw-kek-id" || KEK)."""
    return hashlib.sha256(b"pgw-kek-id" + bytes(kek)).hexdigest()[:16]


def _check_dek(dek: bytes) -> bytes:
    if not isinstance(dek, bytes | bytearray) or len(dek) != DEK_BYTES:
        raise MaskStoreError("data keys must be exactly 32 bytes")
    return bytes(dek)


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

    def mac(self, data: bytes) -> bytes | None:
        """HMAC-SHA256 of ``data`` under a key derived from the KEK, or ``None`` if unsupported.

        The store uses it to authenticate state rows and the generation journal. Providers that
        cannot MAC (pure HSM wrap without an HMAC key, wrap-only partner providers) return ``None``;
        the store then relies on the hardware monotonic counter alone and logs a warning.
        """
        return None

    def monotonic_counter(self) -> int | None:
        """Current hardware monotonic counter value, or ``None`` if unsupported."""
        return None

    def increment_counter(self) -> int | None:
        """Increment and return the hardware counter, or ``None`` if unsupported."""
        return None

    @property
    def can_unwrap(self) -> bool:
        """False for wrap-only providers (partner export)."""
        return True

    def describe(self) -> dict[str, Any]:
        """Non-secret description for status output."""
        return {"type": type(self).__name__, "kek_id": self.kek_id}


class _LocalAesProvider(KeyProvider):
    """Shared implementation for providers that hold raw KEK bytes in memory.

    Supports key rotation: ``previous`` KEKs can still unwrap (and verify MACs created under them is
    *not* supported: MACs always use the current KEK, the store re-MACs on rotation).
    """

    def __init__(self, kek: bytes, previous: Sequence[bytes] = ()):
        if len(kek) != KEK_BYTES:
            raise ConfigError(f"KEK must be exactly {KEK_BYTES} bytes, got {len(kek)}")
        self._kek = bytes(kek)
        self._kek_id = kek_id_for(self._kek)
        self._keys: dict[str, AESGCM] = {self._kek_id: AESGCM(self._kek)}
        for old in previous:
            if len(old) != KEK_BYTES:
                raise ConfigError("previous KEKs must be 32 bytes")
            self._keys.setdefault(kek_id_for(old), AESGCM(bytes(old)))
        self._mac_key = HKDF(hashes.SHA256(), 32, salt=None, info=_MAC_INFO).derive(self._kek)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kek_id={self._kek_id!r})"

    @property
    def kek_id(self) -> str:
        return self._kek_id

    def wrap(self, dek: bytes) -> WrappedKey:
        nonce = os.urandom(12)
        ct = self._keys[self._kek_id].encrypt(nonce, _check_dek(dek), _WRAP_AAD + self._kek_id.encode())
        return WrappedKey(self._kek_id, _WRAP_V1 + nonce + ct)

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        aead = self._keys.get(wrapped.kek_id)
        if aead is None:
            raise MaskStoreError(
                f"segment is wrapped under unknown KEK {wrapped.kek_id!r}", details={"kek_id": wrapped.kek_id}
            )
        blob = wrapped.wrapped
        if len(blob) != 1 + 12 + DEK_BYTES + 16 or blob[:1] != _WRAP_V1:
            raise MaskStoreError("wrapped data key has an unknown format")
        try:
            return aead.decrypt(blob[1:13], blob[13:], _WRAP_AAD + wrapped.kek_id.encode())
        except InvalidTag as exc:
            raise MaskStoreError("wrapped data key failed authentication", details={"kek_id": wrapped.kek_id}) from exc

    def mac(self, data: bytes) -> bytes:
        return hmac.new(self._mac_key, data, hashlib.sha256).digest()


def write_key_file(path: str | Path, kek: bytes | None = None) -> Path:
    """Create a new KEK file with mode 0600 (fails if it already exists)."""
    p = Path(path)
    key = kek if kek is not None else os.urandom(KEK_BYTES)
    if len(key) != KEK_BYTES:
        raise ConfigError("KEK must be 32 bytes")
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ConfigError(f"key file {p} already exists") from exc
    try:
        os.write(fd, key)
        os.fsync(fd)
    finally:
        os.close(fd)
    log.info("created KEK file %s (kek_id %s)", p, kek_id_for(key))
    return p


def _read_key_file(p: Path) -> bytes:
    try:
        st = os.stat(p)
    except FileNotFoundError as exc:
        raise ConfigError(f"key file {p} does not exist") from exc
    if not stat.S_ISREG(st.st_mode):
        raise ConfigError(f"key file {p} is not a regular file")
    if st.st_mode & 0o077:
        raise ConfigError(f"key file {p} has permissions {oct(st.st_mode & 0o777)}; require 0600 or stricter")
    data = p.read_bytes()
    if len(data) != KEK_BYTES:
        raise ConfigError(f"key file {p} must contain exactly {KEK_BYTES} raw bytes")
    return data


class FileKeyProvider(_LocalAesProvider):
    """KEK = 32 raw bytes in a file with mode 0600 (refuses looser permissions).

    ``kek_id`` = first 16 hex chars of SHA-256(b"pgw-kek-id" || KEK).

    Args:
        path: Key file.
        create: Create a new random key if the file does not exist.
        previous: Older key files (rotation): still accepted for unwrapping.
    """

    def __init__(self, path: str | Path, *, create: bool = False, previous: Sequence[str | Path] = ()):
        self.path = Path(path)
        if create and not self.path.exists():
            write_key_file(self.path)
        super().__init__(_read_key_file(self.path), [_read_key_file(Path(p)) for p in previous])


def _b64_key(value: str, what: str) -> bytes:
    try:
        key = base64.b64decode(value.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ConfigError(f"{what} is not valid base64") from exc
    if len(key) != KEK_BYTES:
        raise ConfigError(f"{what} must decode to exactly {KEK_BYTES} bytes")
    return key


class EnvKeyProvider(_LocalAesProvider):
    """KEK from a base64 environment variable (containers with secret injection).

    Args:
        var: Variable holding the current KEK.
        previous_var: Optional variable holding comma-separated base64 previous KEKs (rotation).
        environ: Mapping to read from (default ``os.environ``).
    """

    def __init__(
        self,
        var: str = "PGW_MASKSTORE_KEK",
        *,
        previous_var: str | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        env = os.environ if environ is None else environ
        value = env.get(var)
        if not value:
            raise ConfigError(f"environment variable {var} is not set (base64 32-byte KEK)")
        prev: list[bytes] = []
        if previous_var and env.get(previous_var):
            prev = [_b64_key(v, previous_var) for v in env[previous_var].split(",") if v.strip()]
        self.var = var
        super().__init__(_b64_key(value, var), prev)


class StaticKeyProvider(_LocalAesProvider):
    """In-memory KEK (tests, and tools that already hold the key). Never use with a constant key."""


# ---- PKCS#11 ---------------------------------------------------------------------------------------
class Pkcs11KeyProvider(KeyProvider):
    """HSM-held KEK via PKCS#11 (``CKM_AES_KEY_WRAP_KWP``). Optional dependency ``python-pkcs11``.

    The KEK never leaves the token. ``kek_id`` is ``"p11:<sha256(token label | key label)[:16]>"``.
    If the token holds an HMAC key labelled ``<key_label>-mac`` it is used for :meth:`mac`, and a
    counter object can be wired via :class:`pgw.maskstore.counters.MonotonicCounter`.

    Status: stub-quality. It is exercised against SoftHSM2 in the release pipeline, not in unit tests.
    """

    def __init__(
        self,
        module: str | Path,
        token_label: str,
        key_label: str,
        pin: str | None = None,
        *,
        lib: Any = None,
    ):
        try:
            import pkcs11
        except ImportError as exc:  # pragma: no cover - depends on optional package
            if lib is None:
                raise ConfigError(
                    "the pkcs11 key provider needs the optional 'python-pkcs11' package (pip install python-pkcs11)"
                ) from exc
            pkcs11 = None
        self._p11 = pkcs11
        self.token_label = token_label
        self.key_label = key_label
        self._kek_id = "p11:" + hashlib.sha256(f"{token_label}|{key_label}".encode()).hexdigest()[:16]
        try:  # pragma: no cover - needs an HSM
            library = lib if lib is not None else pkcs11.lib(str(module))
            token = library.get_token(token_label=token_label)
            self._session = token.open(user_pin=pin, rw=True)
            self._key = self._session.get_key(label=key_label)
            try:
                self._mac_key = self._session.get_key(label=f"{key_label}-mac")
            except Exception:
                self._mac_key = None
        except ConfigError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover - needs an HSM
            raise ConfigError(f"cannot open PKCS#11 token {token_label!r}: {type(exc).__name__}: {exc}") from exc

    @property
    def kek_id(self) -> str:
        return self._kek_id

    def wrap(self, dek: bytes) -> WrappedKey:  # pragma: no cover - needs an HSM
        mech = self._p11.Mechanism.AES_KEY_WRAP_PAD
        try:
            data = self._key.encrypt(_check_dek(dek), mechanism=mech)
        except Exception as exc:
            raise MaskStoreError(f"PKCS#11 wrap failed: {type(exc).__name__}") from exc
        return WrappedKey(self._kek_id, bytes(data))

    def unwrap(self, wrapped: WrappedKey) -> bytes:  # pragma: no cover - needs an HSM
        if wrapped.kek_id != self._kek_id:
            raise MaskStoreError(f"segment is wrapped under unknown KEK {wrapped.kek_id!r}")
        mech = self._p11.Mechanism.AES_KEY_WRAP_PAD
        try:
            return bytes(self._key.decrypt(wrapped.wrapped, mechanism=mech))
        except Exception as exc:
            raise MaskStoreError(f"PKCS#11 unwrap failed: {type(exc).__name__}") from exc

    def mac(self, data: bytes) -> bytes | None:  # pragma: no cover - needs an HSM
        if self._mac_key is None:
            return None
        try:
            return bytes(self._mac_key.sign(data, mechanism=self._p11.Mechanism.SHA256_HMAC))
        except Exception as exc:
            raise MaskStoreError(f"PKCS#11 HMAC failed: {type(exc).__name__}") from exc


# ---- Cloud KMS -------------------------------------------------------------------------------------
class KmsClient(Protocol):
    """Minimal envelope-encryption client (adapters for AWS, GCP and Azure below; fakes in tests)."""

    def encrypt(self, key_name: str, plaintext: bytes, aad: bytes) -> bytes:
        """Encrypt ``plaintext`` under the KMS key with additional authenticated data."""
        ...

    def decrypt(self, key_name: str, ciphertext: bytes, aad: bytes) -> bytes:
        """Inverse of :meth:`encrypt`; raise on authentication failure."""
        ...


class _AwsKms:  # pragma: no cover - needs cloud credentials
    def __init__(self) -> None:
        import boto3

        self._c = boto3.client("kms")

    def encrypt(self, key_name: str, plaintext: bytes, aad: bytes) -> bytes:
        ctx = {"pgw": aad.hex()}
        return bytes(self._c.encrypt(KeyId=key_name, Plaintext=plaintext, EncryptionContext=ctx)["CiphertextBlob"])

    def decrypt(self, key_name: str, ciphertext: bytes, aad: bytes) -> bytes:
        ctx = {"pgw": aad.hex()}
        return bytes(self._c.decrypt(KeyId=key_name, CiphertextBlob=ciphertext, EncryptionContext=ctx)["Plaintext"])


class _GcpKms:  # pragma: no cover - needs cloud credentials
    def __init__(self) -> None:
        from google.cloud import kms

        self._c = kms.KeyManagementServiceClient()

    def encrypt(self, key_name: str, plaintext: bytes, aad: bytes) -> bytes:
        r = self._c.encrypt(request={"name": key_name, "plaintext": plaintext, "additional_authenticated_data": aad})
        return bytes(r.ciphertext)

    def decrypt(self, key_name: str, ciphertext: bytes, aad: bytes) -> bytes:
        r = self._c.decrypt(request={"name": key_name, "ciphertext": ciphertext, "additional_authenticated_data": aad})
        return bytes(r.plaintext)


class _AzureKv:  # pragma: no cover - needs cloud credentials
    def __init__(self, key_name: str) -> None:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.keys.crypto import CryptographyClient, KeyWrapAlgorithm

        self._alg = KeyWrapAlgorithm.rsa_oaep_256
        self._c = CryptographyClient(f"https://{key_name}", DefaultAzureCredential())

    def encrypt(self, key_name: str, plaintext: bytes, aad: bytes) -> bytes:
        # Key Vault wrap has no AAD; bind it by wrapping (aad digest || dek).
        return bytes(self._c.wrap_key(self._alg, hashlib.sha256(aad).digest() + plaintext).encrypted_key)

    def decrypt(self, key_name: str, ciphertext: bytes, aad: bytes) -> bytes:
        out = bytes(self._c.unwrap_key(self._alg, ciphertext).key)
        if not hmac.compare_digest(out[:32], hashlib.sha256(aad).digest()):
            raise MaskStoreError("Key Vault unwrap: context mismatch")
        return out[32:]


_KMS_SCHEMES = ("aws-kms://", "gcp-kms://", "azure-kv://")


class KmsKeyProvider(KeyProvider):
    """Cloud KMS envelope encryption (``aws-kms://``, ``gcp-kms://``, ``azure-kv://`` URIs).

    The DEK is encrypted by the KMS with AAD/encryption context ``"pgw/v1/dek-wrap" || kek_id``.
    Cloud SDKs are optional dependencies, imported lazily; a ``client`` implementing
    :class:`KmsClient` can be injected (tests, custom KMS).

    Args:
        key_uri: e.g. ``aws-kms://arn:aws:kms:...``, ``gcp-kms://projects/p/locations/l/keyRings/r/cryptoKeys/k``,
            ``azure-kv://<vault>.vault.azure.net/keys/<name>/<version>``.
        client: Optional KMS client.
        mac_key_b64: Optional base64 HMAC key (e.g. injected from the cloud secret manager) enabling
            :meth:`mac`; without it the store relies on a monotonic counter for rollback detection.
    """

    def __init__(self, key_uri: str, *, client: KmsClient | None = None, mac_key_b64: str | None = None):
        scheme = next((s for s in _KMS_SCHEMES if key_uri.startswith(s)), None)
        if scheme is None or len(key_uri) <= len(scheme):
            raise ConfigError(f"unsupported KMS key URI; expected one of {', '.join(_KMS_SCHEMES)}")
        self.key_uri = key_uri
        self._key_name = key_uri[len(scheme) :]
        self._kek_id = "kms:" + hashlib.sha256(key_uri.encode()).hexdigest()[:16]
        if client is None:  # pragma: no cover - needs cloud SDKs
            try:
                if scheme == "aws-kms://":
                    client = _AwsKms()
                elif scheme == "gcp-kms://":
                    client = _GcpKms()
                else:
                    client = _AzureKv(self._key_name)
            except ImportError as exc:
                raise ConfigError(f"the {scheme} key provider needs its cloud SDK installed: {exc}") from exc
        self._client = client
        self._mac_key = _b64_key(mac_key_b64, "KMS MAC key") if mac_key_b64 else None

    def __repr__(self) -> str:
        return f"KmsKeyProvider(kek_id={self._kek_id!r})"

    @property
    def kek_id(self) -> str:
        return self._kek_id

    def _aad(self, kek_id: str) -> bytes:
        return _WRAP_AAD + kek_id.encode()

    def wrap(self, dek: bytes) -> WrappedKey:
        try:
            blob = self._client.encrypt(self._key_name, _check_dek(dek), self._aad(self._kek_id))
        except MaskStoreError:
            raise
        except Exception as exc:
            raise MaskStoreError(f"KMS encrypt failed: {type(exc).__name__}") from exc
        return WrappedKey(self._kek_id, bytes(blob))

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        if wrapped.kek_id != self._kek_id:
            raise MaskStoreError(f"segment is wrapped under unknown KEK {wrapped.kek_id!r}")
        try:
            out = self._client.decrypt(self._key_name, wrapped.wrapped, self._aad(wrapped.kek_id))
        except MaskStoreError:
            raise
        except Exception as exc:
            raise MaskStoreError(f"KMS decrypt failed: {type(exc).__name__}") from exc
        return _check_dek(out)

    def mac(self, data: bytes) -> bytes | None:
        if self._mac_key is None:
            return None
        return hmac.new(self._mac_key, data, hashlib.sha256).digest()


# ---- HPKE (partner delivery, SPEC 7.7) --------------------------------------------------------------
def _hpke_suite() -> Any:
    try:
        from cryptography.hazmat.primitives import hpke
    except ImportError as exc:  # pragma: no cover - old cryptography
        raise MaskStoreError("HPKE needs cryptography >= 46 (cryptography.hazmat.primitives.hpke)") from exc
    return hpke.Suite(hpke.KEM.X25519, hpke.KDF.HKDF_SHA256, hpke.AEAD.AES_256_GCM)


def _recipient_key_id(public_raw: bytes) -> str:
    return hashlib.sha256(public_raw).hexdigest()[:16]


def hpke_wrap_for_recipient(dek: bytes, recipient_public: bytes) -> WrappedKey:
    """Partner side: HPKE-seal ``dek`` to a customer's X25519 public key."""
    if len(recipient_public) != 32:
        raise ConfigError("recipient public key must be 32 raw bytes (X25519)")
    key_id = _recipient_key_id(bytes(recipient_public))
    pub = X25519PublicKey.from_public_bytes(bytes(recipient_public))
    ct = _hpke_suite().encrypt(_check_dek(dek), pub, info=_HPKE_INFO + key_id.encode())
    return WrappedKey(f"hpke:{key_id}", bytes(ct))


class RecipientKeyPair:
    """X25519 key pair used to receive segments from a partner mask factory (SPEC.md section 7.7).

    The partner wraps each segment DEK with HPKE (RFC 9180: DHKEM(X25519, HKDF-SHA256),
    HKDF-SHA256, AES-256-GCM) to the customer's public key; ``kek_id`` is ``"hpke:<key id>"``.
    :meth:`MaskStore.import_segment` unwraps with the private key and re-wraps under the local KEK.
    """

    def __init__(self, private_key_file: str | Path):
        p = Path(private_key_file)
        raw = _read_key_file(p)
        self.path = p
        self._priv = X25519PrivateKey.from_private_bytes(raw)
        self._pub = self._priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def __repr__(self) -> str:
        return f"RecipientKeyPair(key_id={self.key_id!r})"

    @classmethod
    def generate(cls, private_key_file: str | Path) -> RecipientKeyPair:
        """Create a new key pair (private key file mode 0600)."""
        priv = X25519PrivateKey.generate()
        raw = priv.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
        )
        write_key_file(private_key_file, raw)
        return cls(private_key_file)

    @property
    def key_id(self) -> str:
        """First 16 hex chars of SHA-256 of the raw public key."""
        return _recipient_key_id(self._pub)

    @property
    def kek_id(self) -> str:
        """``"hpke:<key id>"`` as it appears in partner segment headers."""
        return f"hpke:{self.key_id}"

    def public_bytes(self) -> bytes:
        """Raw 32-byte public key to hand to the partner."""
        return self._pub

    def unwrap(self, wrapped: WrappedKey) -> bytes:
        """HPKE-open a partner-wrapped DEK."""
        if wrapped.kek_id != self.kek_id:
            raise MaskStoreError(
                f"segment is sealed to recipient {wrapped.kek_id!r}, not to this key ({self.kek_id})",
                details={"kek_id": wrapped.kek_id},
            )
        try:
            dek = _hpke_suite().decrypt(wrapped.wrapped, self._priv, info=_HPKE_INFO + self.key_id.encode())
        except Exception as exc:
            raise MaskStoreError("HPKE open of the segment data key failed (corrupt or not for us)") from exc
        return _check_dek(dek)
