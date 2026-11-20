"""Enrollment: code format, single-use records, the service endpoint and the operator CLI."""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from pgw.core.errors import ConfigError
from pgw.enroll.code import EnrollmentCode, EnrollmentCodeError, validate_service_url
from pgw.enroll.store import EnrollmentRejectedError, EnrollmentStore, parse_ttl
from pgw.service import ServiceSettings, hash_token
from tests.service.conftest import running_app

URL = "https://compute.example.net:8443"


# ---------------------------------------------------------------------------------------------- code
def _code() -> EnrollmentCode:
    return EnrollmentCode.new(URL, os.urandom(32))


def test_code_roundtrip_and_format() -> None:
    c = _code()
    s = c.encode()
    assert s.startswith("PGW1-")
    groups = s.split("-")[1:]
    assert all(len(g) == 5 for g in groups[:-1]) and 1 <= len(groups[-1]) <= 5
    assert set("".join(groups)) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
    assert EnrollmentCode.parse(s) == c
    assert len(c.secret) * 8 >= 128
    # copy-paste damage is tolerated: case, spaces, line breaks, missing dashes
    assert EnrollmentCode.parse(s.lower()) == c
    assert EnrollmentCode.parse(" " + s.replace("-", " ").replace("PGW1 ", "PGW1-\n") + "\n") == c
    assert EnrollmentCode.parse(s.replace("-", "")) == c
    assert c.secret.hex() not in repr(c) and c.secret.hex() not in str(c)


def test_code_lookalikes_are_read_as_letters() -> None:
    c = _code()
    s = c.encode()
    swapped = s[:5] + s[5:].replace("O", "0").replace("I", "1").replace("B", "8")
    assert EnrollmentCode.parse(swapped) == c


def test_every_single_character_typo_is_caught() -> None:
    c = _code()
    s = c.encode()
    rng = random.Random(0)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    positions = [i for i, ch in enumerate(s) if i >= 5 and ch != "-"]
    for i in positions:
        wrong = rng.choice([a for a in alphabet if a != s[i]])
        with pytest.raises(EnrollmentCodeError, match=r"typo|incomplete|not usable|too short"):
            EnrollmentCode.parse(s[:i] + wrong + s[i + 1:])
    # swapped neighbours and a dropped character
    i = positions[10]
    if s[i] != s[i + 1] and s[i + 1] != "-":
        with pytest.raises(EnrollmentCodeError):
            EnrollmentCode.parse(s[:i] + s[i + 1] + s[i] + s[i + 2:])
    with pytest.raises(EnrollmentCodeError):
        EnrollmentCode.parse(s[:-1])


@pytest.mark.parametrize("text, match", [
    ("", "empty"), ("hello world", "not a PGW"), ("PGW2-AAAAA-AAAAA", "version 2"),
    ("PGW1-!!!!!", "characters"), ("PGW1-AAAAA", "short"),
])
def test_bad_codes_have_plain_messages(text: str, match: str) -> None:
    with pytest.raises(EnrollmentCodeError, match=match):
        EnrollmentCode.parse(text)


def test_code_url_policy() -> None:
    assert validate_service_url("https://compute.example.net:8443/") == "https://compute.example.net:8443"
    assert validate_service_url("http://127.0.0.1:9000")
    for bad in ("http://compute.example.net", "ftp://x.example.net", "https://u:p@x.example.net",
                "https://x.example.net/?a=1"):
        with pytest.raises(ConfigError):
            validate_service_url(bad)
    with pytest.raises(ConfigError, match="pin"):
        EnrollmentCode(URL, bytes(32), os.urandom(16))  # https without a CA pin
    assert not EnrollmentCode("http://127.0.0.1:1", bytes(32), os.urandom(16)).pinned


# --------------------------------------------------------------------------------------------- store
def test_parse_ttl() -> None:
    assert parse_ttl("24h") == 86400 and parse_ttl("30m") == 1800 and parse_ttl("7d") == 7 * 86400
    assert parse_ttl("90") == 90
    for bad in ("0h", "31d", "soon", "-1h"):
        with pytest.raises(ConfigError):
            parse_ttl(bad)


def test_store_hashes_only_and_single_use(tmp_path: Path) -> None:
    st = EnrollmentStore(tmp_path / "enroll.json")
    rec, secret = st.create("acme", 3600, ["tiny-qwen2"])
    text = (tmp_path / "enroll.json").read_text()
    assert secret.hex() not in text and rec.secret_sha256 in text
    assert stat.S_IMODE((tmp_path / "enroll.json").stat().st_mode) == 0o600
    got, extra = st.consume(secret, "gw1", lambda r: r.tenant)
    assert got.id == rec.id and extra == "acme"
    with pytest.raises(EnrollmentRejectedError) as ei:
        st.consume(secret, "gw2", lambda r: None)
    assert ei.value.reason == "code_used"
    assert st.list()[0].public()["state"] == "used" and "secret_sha256" not in st.list()[0].public()


def test_store_concurrent_consume_has_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "enroll.json"
    _, secret = EnrollmentStore(path).create("acme", 3600)

    def attempt(i: int) -> bool:
        try:
            EnrollmentStore(path).consume(secret, f"gw{i}", lambda r: None)  # separate instances: file lock
            return True
        except EnrollmentRejectedError:
            return False

    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(attempt, range(16)))
    assert results.count(True) == 1


def test_store_expiry_revocation_and_failed_check(tmp_path: Path) -> None:
    now = [1000.0]
    st = EnrollmentStore(tmp_path / "e.json", clock=lambda: now[0])
    _, secret = st.create("acme", 60)
    with pytest.raises(RuntimeError):  # a failing check leaves the code unused
        st.consume(secret, "gw", lambda r: (_ for _ in ()).throw(RuntimeError("no model")))
    assert st.list()[0].state(now[0]) == "active"
    now[0] += 61
    with pytest.raises(EnrollmentRejectedError) as ei:
        st.consume(secret, "gw", lambda r: None)
    assert ei.value.reason == "code_expired"
    rec2, secret2 = st.create("acme", 600)
    st.revoke(rec2.id)
    with pytest.raises(EnrollmentRejectedError) as ei:
        st.consume(secret2, "gw", lambda r: None)
    assert ei.value.reason == "code_revoked"
    with pytest.raises(EnrollmentRejectedError) as ei:
        st.consume(os.urandom(16), "gw", lambda r: None)
    assert ei.value.reason == "unknown_code"


# ------------------------------------------------------------------------------------------ endpoint
def _settings(tmp_path: Path, pack_dir: Path, **kw: Any) -> ServiceSettings:
    tokens = tmp_path / "tenants.txt"
    tokens.write_text("")
    base: dict[str, Any] = dict(packs=[pack_dir], device="cpu", tokens_file=tokens, rings=["K32A12C27", "K40A12"],
                                enrollment_file=tmp_path / "enroll.json", public_url="http://127.0.0.1:9",
                                batch_window_ms=1.0)
    base.update(kw)
    return ServiceSettings(**base)


async def _post(app: Any, body: Any, client: str = "10.0.0.1") -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=(client, 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://svc") as h:
        return await h.post("/pgw/v1/enroll", json=body)


async def test_endpoint_issues_working_token_once(tmp_path: Path, tiny_pack_dir: Path, pack_fp: str) -> None:
    s = _settings(tmp_path, tiny_pack_dir)
    rec, secret = EnrollmentStore(s.enrollment_file).create("acme", 3600)  # type: ignore[arg-type]
    async with running_app(s) as app:
        r = await _post(app, {"secret": secret.hex(), "gateway_name": "gw-1"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["tenant"] == "acme" and data["service_url"] == "http://127.0.0.1:9"
        m = data["models"][0]
        assert m["fingerprint"] == pack_fp and m["ring"] == "K32A12C27" and m["hf_repo"]
        assert "client_cert_pem" not in data
        tok = data["service_token"]
        # the hash is in the tokens file and the running service accepts the token at once
        assert f"acme:{hash_token(tok)}" in s.tokens_file.read_text()  # type: ignore[union-attr]
        assert stat.S_IMODE(s.tokens_file.stat().st_mode) == 0o600  # type: ignore[union-attr]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
            r2 = await h.get("/pgw/v1/models", headers={"authorization": f"Bearer {tok}",
                                                         "x-pgw-protocol": "1"})
        assert r2.status_code == 200
        # single use
        r3 = await _post(app, {"secret": secret.hex(), "gateway_name": "gw-2"})
        assert r3.status_code == 401
        assert r3.json()["error"]["message"] == "invalid, used or expired enrollment code"
    audit = (tmp_path / "enroll.audit.jsonl").read_text()
    assert secret.hex() not in audit and tok not in audit and hash_token(tok) not in audit
    events = [json.loads(line) for line in audit.splitlines()]
    assert [e["event"] for e in events] == ["enroll.ok", "enroll.failed"]
    assert events[0]["tenant"] == "acme" and events[0]["gateway_name"] == "gw-1"
    assert events[1]["reason"] == "code_used" and events[1]["enrollment_id"] == rec.id


async def test_endpoint_refuses_expired_revoked_unknown_alike(tmp_path: Path, tiny_pack_dir: Path) -> None:
    s = _settings(tmp_path, tiny_pack_dir)
    st = EnrollmentStore(s.enrollment_file)  # type: ignore[arg-type]
    rec, secret = st.create("acme", 3600)
    st.revoke(rec.id)
    async with running_app(s) as app:
        bodies = [r.json() for r in (await _post(app, {"secret": secret.hex(), "gateway_name": "g"}),
                                     await _post(app, {"secret": os.urandom(16).hex(), "gateway_name": "g"}),
                                     await _post(app, {"secret": "zz", "gateway_name": "g"}))]
    assert all(b == bodies[0] for b in bodies)


async def test_endpoint_model_allowlist_and_unserved_models(tmp_path: Path, tiny_pack_dir: Path) -> None:
    s = _settings(tmp_path, tiny_pack_dir)
    st = EnrollmentStore(s.enrollment_file)  # type: ignore[arg-type]
    _, s_other = st.create("acme", 3600, ["qwen7b"])
    _, s_tiny = st.create("acme", 3600, ["tiny-qwen2"])
    async with running_app(s) as app:
        r = await _post(app, {"secret": s_other.hex(), "gateway_name": "g"})
        assert r.status_code == 503  # nothing to give: the code stays usable for later
        assert st.list()[0].state() == "active"
        r = await _post(app, {"secret": s_tiny.hex(), "gateway_name": "g"})
        assert r.status_code == 200 and [m["name"] for m in r.json()["models"]] == ["tiny-qwen2"]


async def test_endpoint_rate_limit_per_ip(tmp_path: Path, tiny_pack_dir: Path) -> None:
    s = _settings(tmp_path, tiny_pack_dir, enrollment_rate_limit_per_minute=3)
    async with running_app(s) as app:
        codes = [(await _post(app, {"secret": os.urandom(16).hex(), "gateway_name": "g"})).status_code
                 for _ in range(4)]
        assert codes == [401, 401, 401, 429]
        other = await _post(app, {"secret": os.urandom(16).hex(), "gateway_name": "g"}, client="10.0.0.2")
        assert other.status_code == 401


async def test_endpoint_signs_csr_for_local_key(tmp_path: Path, tiny_pack_dir: Path) -> None:
    from cryptography import x509
    from cryptography.x509.oid import ExtendedKeyUsageOID

    from pgw.enroll.client import _key_and_csr
    from tests.enroll_helpers import make_pki

    pki = make_pki(tmp_path / "pki")
    mt = tmp_path / "mtls.txt"
    s = _settings(tmp_path, tiny_pack_dir, client_issuing_ca_cert_file=pki["client_ca"],
                  client_issuing_ca_key_file=pki["client_ca_key"], trusted_client_cert_header="x-client-cert",
                  mtls_tenants_file=mt)
    mt.write_text("")
    _, secret = EnrollmentStore(s.enrollment_file).create("acme", 3600)  # type: ignore[arg-type]
    key_pem, csr = _key_and_csr("gw-1")
    async with running_app(s) as app:
        bad = await _post(app, {"secret": secret.hex(), "gateway_name": "gw-1", "csr_pem": "garbage"})
        assert bad.status_code == 400  # refused before the code is used
        r = await _post(app, {"secret": secret.hex(), "gateway_name": "gw-1", "csr_pem": csr})
        assert r.status_code == 200, r.text
    data = r.json()
    cert = x509.load_pem_x509_certificate(data["client_cert_pem"].encode())
    assert cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value == x509.ExtendedKeyUsage(
        [ExtendedKeyUsageOID.CLIENT_AUTH])
    assert not cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    days = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
    assert days <= 31
    assert key_pem.decode() not in r.text  # the private key never left the gateway
    assert mt.read_text().strip() == f"acme:{data['client_cert_cn']}"


def test_settings_require_tokens_url_and_ca(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tokens_file"):
        ServiceSettings(enrollment_file=tmp_path / "e.json", public_url=URL)
    with pytest.raises(ValueError, match="enrollment_ca_file"):
        ServiceSettings(enrollment_file=tmp_path / "e.json", public_url=URL, tokens_file=tmp_path / "t")
    with pytest.raises(ValueError, match="public_url"):
        ServiceSettings(public_url="http://compute.example.net")


# ------------------------------------------------------------------------------------------------ CLI
def test_cli_enroll_create_list_revoke(tmp_path: Path) -> None:
    from pgw.cli.main import cli
    from tests.enroll_helpers import make_pki

    pki = make_pki(tmp_path / "pki")
    ef = tmp_path / "enroll.json"
    runner = CliRunner()
    r = runner.invoke(cli, ["service", "enroll", "create", "--tenant", "acme", "--ttl", "2h", "--models",
                            "qwen7b,tiny", "--service-url", URL, "--ca-file", str(pki["ca"]),
                            "--enrollment-file", str(ef), "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    code = EnrollmentCode.parse(out["code"])
    from pgw.enroll.service import ca_fingerprint, load_ca_pem

    assert code.service_url == URL and code.ca_sha256 == ca_fingerprint(load_ca_pem(pki["ca"]))
    assert out["models"] == ["qwen7b", "tiny"] and code.secret.hex() not in ef.read_text()
    r = runner.invoke(cli, ["service", "enroll", "list", "--enrollment-file", str(ef), "--json"])
    recs = json.loads(r.stdout)
    assert recs[0]["state"] == "active" and "secret_sha256" not in recs[0]
    r = runner.invoke(cli, ["service", "enroll", "revoke", out["id"], "--enrollment-file", str(ef)])
    assert r.exit_code == 0 and json.loads(r.stdout)["state"] == "revoked"
    # https without a CA to pin is refused
    r = runner.invoke(cli, ["service", "enroll", "create", "--tenant", "acme", "--service-url", URL,
                            "--enrollment-file", str(ef)])
    assert r.exit_code == 1 and "pin" in r.output
