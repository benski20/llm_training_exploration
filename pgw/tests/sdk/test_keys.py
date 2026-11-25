"""SDK keys on the client (``pgw.sdk.keys``): one codec with the provider (``pgw.enroll.sdk_key``), the published
test vector, forgiving input, checksum, pinning rules, no secret in messages or repr."""

from __future__ import annotations

import pytest

from pgw.enroll.sdk_key import SDKKey, secret_digest
from pgw.sdk.keys import KEY_ID_BYTES, SECRET_BYTES, SdkKey, SdkKeyError, parse_key

CA = bytes(range(32))

#: docs/sdk/CONTRACT.md, appendix A1
VECTOR = ("pgw_test_aeaacaqdaqcqmbyibefawdanbyhraeiscmkbkfqxdamrugy4dupb6aareizuivlgo5sgkzthnbuwu23mnvxg64drojzxi5lwo54h"
          "s6t3nb2hi4dthixs6mrqgmxdalrrgezs4nz2gizdinbtpqytwli")


def test_one_codec_and_the_published_vector() -> None:
    assert SdkKey is SDKKey
    k = SDKKey("test", "https://203.0.113.7:22443", CA, bytes.fromhex("0011223344556677"), bytes(range(100, 124)))
    assert k.encode() == VECTOR
    p = parse_key(VECTOR)
    assert p == k and p.key_id_hex == "0011223344556677" and p.key_id_str == "sk_0011223344556677"
    assert p.digest.hex() == "93825b55e599ec56935c715e19e9f1fe0dd0f1109564917224c183bbd5cb9ca3"
    assert p.digest == secret_digest(p.key_id, p.secret)


def test_round_trip_and_views() -> None:
    k = SdkKey.new("live", "https://gpu.example.com:8443/", CA)
    s = k.encode()
    assert s.startswith("pgw_live_") and s == s.strip() and s[9:].islower()
    p = parse_key(s)
    assert p == k and p.service_url == "https://gpu.example.com:8443"  # normalized
    assert len(p.key_id) == KEY_ID_BYTES and len(p.secret) == SECRET_BYTES
    assert p.pinned and p.ca_fingerprint_hex == CA.hex()
    assert len(p.key_id_hex) == 16 and p.bearer() == "Bearer " + s
    assert parse_key("  " + s + "\n") == k  # whitespace from copy and paste
    assert parse_key(s[:20] + "\n  " + s[20:]) == k  # a line break inside the key
    assert parse_key(s.upper()) == k  # case does not matter


def test_lookalike_digits() -> None:
    k = SdkKey.new("live", "https://gpu.example.com", CA)
    s = k.encode()
    body = s[len("pgw_live_"):]
    swapped = body.replace("o", "0").replace("i", "1").replace("b", "8")
    assert parse_key("pgw_live_" + swapped) == k


def test_secret_never_printed() -> None:
    k = SdkKey.new("live", "https://gpu.example.com", CA)
    s = k.encode()
    for text in (str(k), repr(k), f"{k}", repr([k])):
        assert k.secret.hex() not in text and s not in text and k.key_id_str in text


@pytest.mark.parametrize("bad", [None, "", "   ", "sk-abc", "pgw_live", "pgw_live_", "pgw__abc", "xyz_live_abc",
                                 "pgw_LIVE!_abc", "pgw_live_ab$cd", "pgw_1live_abc"])
def test_malformed(bad: str | None) -> None:
    with pytest.raises(SdkKeyError):
        parse_key(bad)


def test_typo_and_truncation_are_caught_locally() -> None:
    s = SdkKey.new("live", "https://gpu.example.com", CA).encode()
    i = 30
    typo = s[:i] + ("a" if s[i] != "a" else "c") + s[i + 1:]
    with pytest.raises(SdkKeyError, match="checksum") as ei:
        parse_key(typo)
    assert typo not in str(ei.value) and s not in str(ei.value)
    with pytest.raises(SdkKeyError):
        parse_key(s[:-8])
    with pytest.raises(SdkKeyError, match="checksum"):  # the env label is bound by the checksum
        parse_key(s.replace("pgw_live_", "pgw_test_"))


def test_pinning_rules() -> None:
    with pytest.raises(Exception, match="pin"):
        SdkKey.new("live", "https://gpu.example.com", None)
    with pytest.raises(Exception, match="https"):  # plain http only to loopback
        SdkKey.new("live", "http://gpu.example.com", None)
    k = SdkKey.new("dev", "http://127.0.0.1:9000", None)
    assert not k.pinned and parse_key(k.encode()) == k
    with pytest.raises(Exception, match="200 bytes"):
        SdkKey.new("live", "https://gpu.example.com/" + "a" * 200, CA)


def test_error_is_an_authentication_error_for_the_client() -> None:
    from pgw.sdk.errors import AuthenticationError, from_pgw_error

    for bad in ("nope", VECTOR[:-3]):
        with pytest.raises(SdkKeyError) as ei:
            parse_key(bad)
        assert isinstance(from_pgw_error(ei.value), AuthenticationError)
