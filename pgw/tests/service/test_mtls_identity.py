"""Real TLS handshakes: verified client certificates reach tenant auth under native hypercorn.

Regression for the security review finding that hypercorn does not publish the peer certificate
in the ASGI scope, which silently disabled the mTLS tenant map and the token/certificate
cross-check.
"""

from __future__ import annotations

import ssl
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization

pytest.importorskip("hypercorn")

from pgw.core import RING_K40_A12
from pgw.service import ServiceSettings, hash_token
from pgw.service.tls_peer import PeerCertMiddleware, subject_from_der
from pgw.wire import CONTENT_TYPE, encode_message
from tests.service.conftest import TOKEN_A, TOKEN_B, make_req, rand_x
from tests.service.test_server import _ca, _free_port, _issue, _key_pem, _ServerThread, _wait_ready


@pytest.fixture(scope="module")
def two_client_pki(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("mtls-id-pki")
    ca_key, ca = _ca("pgw-mtls-id-ca")
    out: dict[str, Path] = {"ca": d / "ca.pem"}
    out["ca"].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    for role, cn, server in (("server", "pgw-server", True), ("a", "gw-a", False), ("b", "gw-b", False)):
        k, c = _issue(ca_key, ca, cn, server=server)
        out[role] = d / f"{role}.pem"
        out[f"{role}_key"] = d / f"{role}.key"
        out[role].write_bytes(c.public_bytes(serialization.Encoding.PEM))
        out[f"{role}_key"].write_bytes(_key_pem(k))
    return out


@pytest.fixture(scope="module")
def id_server(tiny_pack_dir: Path, two_client_pki: dict[str, Path],
              tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    d = tmp_path_factory.mktemp("mtls-id")
    tok = d / "tenants.txt"
    tok.write_text(f"tenant-a:{hash_token(TOKEN_A)}\ntenant-b:{hash_token(TOKEN_B)}\n")
    mt = d / "mtls.txt"
    mt.write_text("tenant-a:gw-a\n")  # gw-b is a valid CA-issued cert but not mapped to any tenant
    port, mport = _free_port(), _free_port()
    pki = two_client_pki
    s = ServiceSettings(
        host="127.0.0.1", port=port, metrics_port=mport, packs=[tiny_pack_dir], device="cpu",
        tokens_file=tok, mtls_tenants_file=mt, batch_window_ms=1.0,
        tls={"cert_file": pki["server"], "key_file": pki["server_key"], "ca_file": pki["ca"],
             "require_client_cert": True},
    )
    srv = _ServerThread(s)
    srv.start()
    url = f"https://127.0.0.1:{port}"
    ctxs = {}
    for who in ("a", "b"):
        ctx = ssl.create_default_context(cafile=str(pki["ca"]))
        ctx.load_cert_chain(str(pki[who]), str(pki[f"{who}_key"]))
        ctxs[who] = ctx
    _wait_ready(url, verify=ctxs["a"])
    yield {"url": url, "ctx": ctxs}
    srv.stop()
    assert srv.error is None, srv.error


def _post(srv: dict[str, Any], who: str, pack_fp: str, token: str | None, http2: bool) -> int:
    ring = RING_K40_A12
    body = encode_message(make_req(pack_fp, ring, [("0.o", rand_x(1, 64, ring))]))
    headers = {"content-type": CONTENT_TYPE}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    with httpx.Client(verify=srv["ctx"][who], http2=http2, timeout=30) as h:
        return h.post(f"{srv['url']}/pgw/v1/compute", content=body, headers=headers).status_code


@pytest.mark.parametrize("http2", [False, True])
def test_cert_identity_reaches_tenant_auth(id_server: dict[str, Any], pack_fp: str, http2: bool) -> None:
    # mapped certificate alone authenticates (mtls_tenants_file-only deployments work)
    assert _post(id_server, "a", pack_fp, None, http2) == 200
    # matching token + certificate
    assert _post(id_server, "a", pack_fp, TOKEN_A, http2) == 200
    # leaked token of another tenant presented with tenant-a's certificate: cross-check refuses
    assert _post(id_server, "a", pack_fp, TOKEN_B, http2) == 403
    # CA-issued certificate that is not mapped to any tenant is refused even with a valid token
    assert _post(id_server, "b", pack_fp, TOKEN_B, http2) == 403


def test_settings_refuse_mtls_map_without_verified_certs(tmp_path: Path) -> None:
    m = tmp_path / "m.txt"
    m.write_text("t:cn\n")
    with pytest.raises(ValueError, match="verified client certificates"):
        ServiceSettings(mtls_tenants_file=m)
    ServiceSettings(mtls_tenants_file=m, trusted_client_cert_header="X-Client-Cert")  # proxy mode is fine


async def test_peer_cert_middleware_strips_injected_identity() -> None:
    seen: dict[str, Any] = {}

    async def inner(scope: dict[str, Any], receive: Any, send: Any) -> None:
        seen.update(scope)

    mw = PeerCertMiddleware(inner)
    await mw({"type": "http", "extensions": {"tls": {"client_cert_name": "CN=forged"}}}, None, None)
    assert "client_cert_name" not in seen["extensions"]["tls"]


def test_subject_from_der(two_client_pki: dict[str, Path]) -> None:
    pem = two_client_pki["a"].read_text()
    der = ssl.PEM_cert_to_DER_cert(pem)
    assert subject_from_der(der) == "CN=gw-a"
    assert subject_from_der(b"garbage") is None
