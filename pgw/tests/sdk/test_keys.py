"""SDK key format (``pgw.sdk.keys``): round trip, checksum, pinning rules, no secret in messages or repr."""

from __future__ import annotations

import pytest

from pgw.sdk.keys import KEY_ID_BYTES, SECRET_BYTES, SdkKey, SdkKeyError, parse_key

CA = bytes(range(32))


def test_round_trip_and_views() -> None:
    k = SdkKey.new("https://gpu.example.com:8443/", CA, env="live")
    s = k.encode()
    assert s.startswith("pgw_live_") and s == s.strip() and s[9:].islower()
    p = parse_key(s)
    assert p == k and p.service_url == "https://gpu.example.com:8443"  # normalized
    assert len(p.key_id) == KEY_ID_BYTES and len(p.secret) == SECRET_BYTES
    assert p.pinned and p.ca_fingerprint_hex == CA.hex()
    assert len(p.key_id_hex) == 16 and p.bearer() == "Bearer " + s
    assert parse_key("  " + s + "\n") == k  # whitespace from copy and paste
    assert parse_key(s.upper().replace("PGW_LIVE_", "pgw_live_")) == k  # case of the body does not matter


def test_secret_never_printed() -> None:
    k = SdkKey.new("https://gpu.example.com", CA)
    s = k.encode()
    for text in (str(k), repr(k), f"{k}"):
        assert k.secret.hex() not in text and s not in text and k.key_id_hex in text


@pytest.mark.parametrize("bad", [None, "", "   ", "sk-abc", "pgw_live", "pgw_live_", "pgw__abc", "xyz_live_abc",
                                 "pgw_LIVE!_abc", "pgw_live_ab$cd"])
def test_malformed(bad: str | None) -> None:
    with pytest.raises(SdkKeyError):
        parse_key(bad)


def test_typo_and_truncation_are_caught_locally() -> None:
    s = SdkKey.new("https://gpu.example.com", CA).encode()
    i = 30
    typo = s[:i] + ("a" if s[i] != "a" else "b") + s[i + 1:]
    with pytest.raises(SdkKeyError, match="typo") as ei:
        parse_key(typo)
    assert typo not in str(ei.value) and s not in str(ei.value)
    with pytest.raises(SdkKeyError):
        parse_key(s[:-8])
    with pytest.raises(SdkKeyError, match="typo"):  # the env label is bound by the checksum
        parse_key(s.replace("pgw_live_", "pgw_test_"))


def test_pinning_rules() -> None:
    with pytest.raises(SdkKeyError, match="pin"):
        SdkKey.new("https://gpu.example.com", None)
    with pytest.raises(Exception, match="https"):  # plain http only to loopback
        SdkKey.new("http://gpu.example.com", None)
    k = SdkKey.new("http://127.0.0.1:9000", None, env="dev")
    assert not k.pinned and parse_key(k.encode()) == k


def test_secret_hash_for_the_provider() -> None:
    import hashlib

    k = SdkKey.new("https://gpu.example.com", CA)
    assert k.secret_sha256() == hashlib.sha256(k.secret).digest()


def test_error_is_an_authentication_error_for_the_client() -> None:
    from pgw.sdk.errors import AuthenticationError, from_pgw_error

    with pytest.raises(SdkKeyError) as ei:
        parse_key("nope")
    assert isinstance(from_pgw_error(ei.value), AuthenticationError)
