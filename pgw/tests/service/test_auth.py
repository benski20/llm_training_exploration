"""Tests for :mod:`pgw.service.auth`."""

from __future__ import annotations

from pathlib import Path

import pytest

from pgw.core import AuthError, ConfigError, ErrorCode, PGWError
from pgw.service.auth import TenantAuthenticator, TenantIdentity, hash_token, parse_subject_cn
from tests.service.conftest import TOKEN_A, TOKEN_B


def test_token_file(tokens_file: Path) -> None:
    a = TenantAuthenticator(tokens_file)
    assert a.authenticate(f"Bearer {TOKEN_A}") == TenantIdentity("tenant-a", "token")
    assert a.authenticate(f"bearer   {TOKEN_B} ") == TenantIdentity("tenant-b", "token")
    assert a.tenants == {"tenant-a", "tenant-b"}
    for bad in ("Bearer nope", "Basic abc", "Bearer", "", None, TOKEN_A):
        with pytest.raises(AuthError) as ei:
            a.authenticate(bad)
        assert ei.value.code == ErrorCode.UNAUTHENTICATED
        assert TOKEN_A not in ei.value.message


def test_hash_is_sha256() -> None:
    import hashlib

    assert hash_token("x") == hashlib.sha256(b"x").hexdigest()


@pytest.mark.parametrize(
    "content",
    ["tenant-a\n", "tenant-a:nothex\n", ":abc\n", "bad tenant:" + "0" * 64 + "\n", "t:" + "0" * 63 + "\n"],
)
def test_malformed_token_file(tmp_path: Path, content: str) -> None:
    p = tmp_path / "t.txt"
    p.write_text(content)
    with pytest.raises(ConfigError):
        TenantAuthenticator(p)


def test_missing_file_and_no_credentials(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        TenantAuthenticator(tmp_path / "missing.txt")
    with pytest.raises(ConfigError, match="no tenant credentials"):
        TenantAuthenticator(None)
    anon = TenantAuthenticator(None, allow_unauthenticated=True)
    assert anon.authenticate(None) == TenantIdentity("anonymous", "anonymous")
    with pytest.raises(AuthError):  # presented credentials must still be valid
        anon.authenticate("Bearer wrong")


def test_uppercase_hash_accepted(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text(f"t1:{hash_token('secret').upper()}\n")
    assert TenantAuthenticator(p).authenticate("Bearer secret").tenant_id == "t1"


def test_mtls_subjects(tmp_path: Path, tokens_file: Path) -> None:
    m = tmp_path / "mtls.txt"
    m.write_text("tenant-a:CN=gw-a,O=Acme\ntenant-c:gw-c\n")
    a = TenantAuthenticator(tokens_file, mtls_tenants_file=m)
    assert a.authenticate(None, "CN=gw-a,O=Acme") == TenantIdentity("tenant-a", "mtls")
    assert a.authenticate(None, "/O=Other/CN=gw-c") == TenantIdentity("tenant-c", "mtls")  # CN fallback
    assert a.authenticate(f"Bearer {TOKEN_A}", "CN=gw-a,O=Acme") == TenantIdentity("tenant-a", "token")
    with pytest.raises(PGWError) as ei:
        a.authenticate(f"Bearer {TOKEN_B}", "CN=gw-a,O=Acme")
    assert ei.value.code == ErrorCode.PERMISSION_DENIED
    with pytest.raises(PGWError) as ei:
        a.authenticate(None, "CN=intruder")
    assert ei.value.code == ErrorCode.PERMISSION_DENIED


def test_programmatic_maps() -> None:
    a = TenantAuthenticator(None, token_hashes={hash_token("k"): "t9"}, mtls_subjects={"gw9": "t9"})
    assert a.authenticate("Bearer k").tenant_id == "t9"
    assert a.authenticate(None, "CN=gw9").via == "mtls"
    with pytest.raises(ConfigError):
        TenantAuthenticator(None, token_hashes={"zz": "t"})


def test_parse_subject_cn() -> None:
    assert parse_subject_cn("CN=a,O=b") == "a"
    assert parse_subject_cn("/C=DE/CN=gw 1/O=x") == "gw 1"
    assert parse_subject_cn("O=b") is None
