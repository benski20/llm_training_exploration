"""``pgw gateway enroll``: CA pinning against a real TLS service, credential files and permissions."""

from __future__ import annotations

import json
import ssl
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from pgw.core.errors import ConfigError
from pgw.enroll.client import EnrollmentError, enroll_gateway
from pgw.enroll.code import EnrollmentCodeError
from pgw.enroll.store import EnrollmentStore
from pgw.gateway.config import load_models_file
from tests.enroll_helpers import EnrollService, enroll_service

pytestmark = [pytest.mark.timeout(240)]


@pytest.fixture(scope="module")
def svc(tiny_pack_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[EnrollService]:
    with enroll_service(tiny_pack_dir, tmp_path_factory.mktemp("enroll-svc")) as s:
        yield s


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_enroll_writes_credentials_with_safe_permissions(svc: EnrollService, tmp_path: Path, tiny_pack: Any) -> None:
    pack_fp = tiny_pack.fingerprint
    code, _ = svc.create_code("acme")
    cfg = tmp_path / "etc"
    res = enroll_gateway(code, cfg, gateway_name="gw-test")
    assert res.tenant == "acme" and res.client_cert
    files = res.files
    assert _mode(files["token"]) == 0o600 and _mode(files["client_key"]) == 0o600 and _mode(files["models"]) == 0o600
    assert _mode(cfg / "secrets") & 0o077 == 0
    token = files["token"].read_text().strip()
    assert token.startswith("pgwt-")
    # models.json is a valid production entry: masked, pinned fingerprint, https + pinned CA, client cert
    (m,) = load_models_file(files["models"])
    assert m.mode == "masked" and m.fingerprint == pack_fp and m.service_url == svc.url
    assert m.service_ca_file == files["ca"] and m.client_cert_file == files["client_cert"]
    meta = json.loads(files["meta"].read_text())
    assert meta["tenant"] == "acme" and meta["models"][0]["hf_repo"] and token not in files["meta"].read_text()
    # the token and the client certificate work against the service right away
    ctx = ssl.create_default_context(cafile=str(files["ca"]))
    r = httpx.get(f"{svc.url}/pgw/v1/models", headers={"authorization": f"Bearer {token}", "PGW-Protocol": "1"},
                  verify=ctx)
    assert r.status_code == 200
    assert EnrollmentStore(svc.settings.enrollment_file).list()[-1].state() == "used"
    # a second run refuses to overwrite; --force replaces (with a new code)
    code2, _ = svc.create_code("acme")
    with pytest.raises(ConfigError, match="already enrolled"):
        enroll_gateway(code2, cfg)
    res2 = enroll_gateway(code2, cfg, force=True, request_client_cert=False)
    assert not res2.client_cert and not (cfg / "client.crt").exists() and not (cfg / "secrets/client.key").exists()
    assert files["token"].read_text().strip() != token


def test_wrong_ca_pin_is_refused_before_the_secret_is_sent(svc: EnrollService, tmp_path: Path) -> None:
    code, rec = svc.create_code("acme", ca_path=svc.dirs["rogue_ca"])
    with pytest.raises(EnrollmentError, match="does not match the enrollment code"):
        enroll_gateway(code, tmp_path / "etc")
    states = {r.id: r.state() for r in EnrollmentStore(svc.settings.enrollment_file).list()}
    assert states[rec.id] == "active"  # the secret never reached the service
    assert not (tmp_path / "etc" / "secrets" / "service_token").exists()


def test_typo_is_caught_before_any_network_call(svc: EnrollService, tmp_path: Path) -> None:
    code, _ = svc.create_code("acme")
    i = len(code) // 2
    typo = code[:i] + ("A" if code[i] != "A" else "B") + code[i + 1:]
    with pytest.raises(EnrollmentCodeError, match="typo"):
        enroll_gateway(typo, tmp_path / "etc", transport=_NoNetwork())


def test_used_code_has_a_plain_message(svc: EnrollService, tmp_path: Path) -> None:
    code, _ = svc.create_code("acme")
    enroll_gateway(code, tmp_path / "a")
    with pytest.raises(EnrollmentError, match="already used, revoked or expired"):
        enroll_gateway(code, tmp_path / "b")


def test_cli_enroll(svc: EnrollService, tmp_path: Path) -> None:
    from pgw.cli.main import cli

    code, _ = svc.create_code("acme")
    r = CliRunner().invoke(cli, ["gateway", "enroll", code, "--config-dir", str(tmp_path / "etc"),
                                 "--name", "gw-cli"])
    assert r.exit_code == 0, r.output
    assert "Enrolled as tenant acme" in r.output and "pgw pack" in r.output
    assert _mode(tmp_path / "etc" / "secrets" / "service_token") == 0o600
    r = CliRunner().invoke(cli, ["gateway", "enroll", "PGW1-AAAAA-BBBBB", "--config-dir", str(tmp_path / "x")])
    assert r.exit_code == 1 and "error [config]" in r.output


class _NoNetwork(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
        raise AssertionError("network used")
