"""SDK key format 2 (the pinned endpoint-directory key) next to format 1, and the enrollment's directory key."""

from __future__ import annotations

import os

import pytest

from pgw.enroll.sdk_key import SDKKey, SDKKeyError
from pgw.sdk.keys import parse_key

CA = bytes(range(32))
DK = os.urandom(32)


def test_v1_keys_unchanged() -> None:
    k = SDKKey.new("live", "https://p.example:8443", CA)
    s = k.encode()
    assert k.directory_key is None and SDKKey.parse(s) == k
    raw_version = __import__("base64").b32decode(s.split("_")[2].upper() + "=" * (-len(s.split("_")[2]) % 8))[0]
    assert raw_version == 1  # a key without a directory pin keeps the version-1 string


def test_v2_roundtrip_and_damage() -> None:
    k = SDKKey.new("live", "https://p.example:8443", CA, DK)
    s = k.encode()
    got = parse_key(s)
    assert got.directory_key == DK and got.ca_sha256 == CA and got.secret == k.secret
    assert got.service_url == "https://p.example:8443"
    bad = s[:-3] + ("a" if s[-3] != "a" else "b") + s[-2:]
    with pytest.raises(SDKKeyError):
        SDKKey.parse(bad)
    with pytest.raises(Exception):  # noqa: B017 - a directory key must be 32 bytes
        SDKKey("live", "https://p.example:8443", CA, k.key_id, k.secret, b"short")


def test_session_keeps_only_a_valid_directory_key() -> None:
    from pgw.sdk.session import _directory_key

    assert _directory_key(DK.hex()) == DK.hex()
    assert _directory_key("zz") is None and _directory_key(None) is None and _directory_key(123) is None


def test_service_setting_validated_and_keys_pin_it() -> None:
    from pgw.service.accounts import directory_key_bytes
    from pgw.service.config import ServiceSettings

    s = ServiceSettings(endpoint_directory_key=DK.hex().upper(), allow_unauthenticated=True)
    assert s.endpoint_directory_key == DK.hex() and directory_key_bytes(s) == DK
    assert directory_key_bytes(ServiceSettings(allow_unauthenticated=True)) is None
    with pytest.raises(ValueError):
        ServiceSettings(endpoint_directory_key="00", allow_unauthenticated=True)
