"""Key providers, DEK wrapping, HPKE partner delivery and settings-driven construction."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from pgw.core import ConfigError, MaskStoreError
from pgw.maskstore import (
    EnvKeyProvider,
    FileKeyProvider,
    KeyProvider,
    KmsKeyProvider,
    MaskStoreSettings,
    Pkcs11KeyProvider,
    RecipientKeyPair,
    StaticKeyProvider,
    WrappedKey,
    hpke_wrap_for_recipient,
    write_key_file,
)
from pgw.maskstore.admin import key_provider_from_settings
from pgw.maskstore.keys import kek_id_for


def test_file_provider_create_wrap_unwrap(tmp_path: Path) -> None:
    kf = tmp_path / "kek"
    p = FileKeyProvider(kf, create=True)
    assert oct(kf.stat().st_mode & 0o777) == "0o600"
    assert p.kek_id == kek_id_for(kf.read_bytes()) and len(p.kek_id) == 16
    dek = os.urandom(32)
    w = p.wrap(dek)
    assert w.kek_id == p.kek_id and dek not in w.wrapped
    assert p.wrap(dek).wrapped != w.wrapped  # randomised
    assert p.unwrap(w) == dek
    assert FileKeyProvider(kf).unwrap(w) == dek  # reload from file
    assert "kek_id" in repr(p) and kf.read_bytes().hex() not in repr(p)


def test_file_provider_refuses_loose_permissions_and_bad_files(tmp_path: Path) -> None:
    kf = tmp_path / "kek"
    kf.write_bytes(os.urandom(32))
    kf.chmod(0o644)
    with pytest.raises(ConfigError, match="permissions"):
        FileKeyProvider(kf)
    kf.chmod(0o600)
    FileKeyProvider(kf)
    short = tmp_path / "short"
    write_key_file(short, os.urandom(32))
    short.chmod(0o600)
    short.write_bytes(b"x" * 31)
    with pytest.raises(ConfigError, match="32"):
        FileKeyProvider(short)
    with pytest.raises(ConfigError, match="does not exist"):
        FileKeyProvider(tmp_path / "nope")
    with pytest.raises(ConfigError, match="exists"):
        write_key_file(kf)


def test_wrong_kek_and_tampering_fail(tmp_path: Path) -> None:
    a = StaticKeyProvider(os.urandom(32))
    b = StaticKeyProvider(os.urandom(32))
    w = a.wrap(os.urandom(32))
    with pytest.raises(MaskStoreError, match="unknown KEK"):
        b.unwrap(w)
    with pytest.raises(MaskStoreError, match="authentication"):
        a.unwrap(WrappedKey(w.kek_id, w.wrapped[:-1] + bytes([w.wrapped[-1] ^ 1])))
    with pytest.raises(MaskStoreError, match="format"):
        a.unwrap(WrappedKey(w.kek_id, b"\x02" + w.wrapped[1:]))
    with pytest.raises(MaskStoreError, match="32 bytes"):
        a.wrap(b"short")
    with pytest.raises(ConfigError):
        StaticKeyProvider(b"x" * 16)


def test_rotation_previous_keys_unwrap(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    p_old = FileKeyProvider(old, create=True)
    w = p_old.wrap(b"k" * 32)
    write_key_file(new)
    p_new = FileKeyProvider(new, previous=[old])
    assert p_new.kek_id != p_old.kek_id
    assert p_new.unwrap(w) == b"k" * 32
    assert p_new.wrap(b"k" * 32).kek_id == p_new.kek_id
    assert p_new.mac(b"x") != p_old.mac(b"x")


def test_env_provider() -> None:
    key = os.urandom(32)
    old = os.urandom(32)
    env = {"K": base64.b64encode(key).decode(), "OLD": base64.b64encode(old).decode()}
    p = EnvKeyProvider("K", previous_var="OLD", environ=env)
    assert p.kek_id == kek_id_for(key)
    assert p.unwrap(StaticKeyProvider(old).wrap(b"z" * 32)) == b"z" * 32
    assert p.mac(b"a") == StaticKeyProvider(key).mac(b"a")
    with pytest.raises(ConfigError, match="not set"):
        EnvKeyProvider("MISSING", environ={})
    with pytest.raises(ConfigError, match="base64"):
        EnvKeyProvider("K", environ={"K": "!!!"})
    with pytest.raises(ConfigError, match="32"):
        EnvKeyProvider("K", environ={"K": base64.b64encode(b"x" * 8).decode()})


class FakeKms:
    """Envelope encryption with AAD, like a cloud KMS."""

    def __init__(self) -> None:
        self.inner = StaticKeyProvider(os.urandom(32))
        self.calls = 0

    def encrypt(self, key_name: str, plaintext: bytes, aad: bytes) -> bytes:
        self.calls += 1
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = os.urandom(12)
        return nonce + AESGCM(self.inner._kek).encrypt(nonce, plaintext, key_name.encode() + aad)

    def decrypt(self, key_name: str, ciphertext: bytes, aad: bytes) -> bytes:
        self.calls += 1
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM(self.inner._kek).decrypt(ciphertext[:12], ciphertext[12:], key_name.encode() + aad)


def test_kms_provider_with_injected_client() -> None:
    fake = FakeKms()
    p = KmsKeyProvider("gcp-kms://projects/p/locations/l/keyRings/r/cryptoKeys/k", client=fake)
    assert p.kek_id.startswith("kms:")
    dek = os.urandom(32)
    w = p.wrap(dek)
    assert p.unwrap(w) == dek and fake.calls == 2
    assert p.mac(b"x") is None
    with pytest.raises(MaskStoreError, match="KMS decrypt failed"):
        p.unwrap(WrappedKey(w.kek_id, w.wrapped[:-1] + b"\0"))
    with pytest.raises(MaskStoreError, match="unknown KEK"):
        p.unwrap(WrappedKey("kms:other", w.wrapped))
    with pytest.raises(ConfigError, match="unsupported"):
        KmsKeyProvider("vault://x", client=fake)
    p2 = KmsKeyProvider("aws-kms://arn:aws:kms:x", client=fake, mac_key_b64=base64.b64encode(b"m" * 32).decode())
    assert p2.mac(b"x") is not None and p2.mac(b"x") == p2.mac(b"x")


def test_pkcs11_needs_optional_dependency() -> None:
    try:
        import pkcs11  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        with pytest.raises(ConfigError, match="python-pkcs11"):
            Pkcs11KeyProvider("/usr/lib/softhsm/libsofthsm2.so", "tok", "kek")
    else:  # pragma: no cover - environment dependent
        pytest.skip("python-pkcs11 installed; covered by the HSM pipeline")


def test_default_provider_capabilities() -> None:
    p = StaticKeyProvider(os.urandom(32))
    assert isinstance(p, KeyProvider)
    assert p.monotonic_counter() is None and p.increment_counter() is None
    assert p.can_unwrap and p.describe()["kek_id"] == p.kek_id


def test_hpke_round_trip_and_wrong_recipient(tmp_path: Path) -> None:
    kp = RecipientKeyPair.generate(tmp_path / "recipient.key")
    assert oct((tmp_path / "recipient.key").stat().st_mode & 0o777) == "0o600"
    assert len(kp.public_bytes()) == 32 and kp.kek_id == f"hpke:{kp.key_id}"
    dek = os.urandom(32)
    w = hpke_wrap_for_recipient(dek, kp.public_bytes())
    assert w.kek_id == kp.kek_id and len(w.wrapped) == 32 + 32 + 16
    assert RecipientKeyPair(tmp_path / "recipient.key").unwrap(w) == dek
    other = RecipientKeyPair.generate(tmp_path / "other.key")
    with pytest.raises(MaskStoreError, match="sealed to recipient"):
        other.unwrap(w)
    with pytest.raises(MaskStoreError, match="HPKE"):
        kp.unwrap(WrappedKey(w.kek_id, w.wrapped[:-1] + bytes([w.wrapped[-1] ^ 1])))
    with pytest.raises(ConfigError):
        hpke_wrap_for_recipient(dek, b"short")


def test_key_provider_from_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kf = tmp_path / "kek"
    write_key_file(kf)
    p = key_provider_from_settings(MaskStoreSettings(root=tmp_path, key_provider="file", key_file=kf))
    assert isinstance(p, FileKeyProvider)
    with pytest.raises(ConfigError, match="key_file"):
        key_provider_from_settings(MaskStoreSettings(root=tmp_path, key_provider="file"))
    monkeypatch.setenv("PGW_TEST_KEK", base64.b64encode(os.urandom(32)).decode())
    p = key_provider_from_settings(MaskStoreSettings(root=tmp_path, key_provider="env", key_env_var="PGW_TEST_KEK"))
    assert isinstance(p, EnvKeyProvider)
    with pytest.raises(ConfigError):
        key_provider_from_settings(MaskStoreSettings(root=tmp_path, key_provider="kms"))
    with pytest.raises(ConfigError):
        key_provider_from_settings(MaskStoreSettings(root=tmp_path, key_provider="pkcs11"))


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PGW_MASKSTORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PGW_MASKSTORE_LOW_WATER_SLOTS", "17")
    monkeypatch.setenv("PGW_MASKSTORE_COUNTER", "file")
    s = MaskStoreSettings()
    assert s.root == tmp_path and s.low_water_slots == 17 and s.counter == "file"
    assert MaskStoreSettings(root=tmp_path).fsync is True
