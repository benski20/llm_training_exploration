"""Session bootstrap (``pgw.sdk.session``) against a fake provider (httpx MockTransport): CA pin, payload,
files and permissions, reuse, concurrent first use, error mapping."""

from __future__ import annotations

import json
import stat
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from pgw.sdk import errors as E
from pgw.sdk.cache import CacheLayout
from pgw.sdk.keys import SdkKey, parse_key
from pgw.sdk.session import SDK_ENROLL_PATH, ensure_session, load_session

URL = "https://gpu.example.com:8443"
MODELS = [{"name": "qwen2.5-7b-instruct", "fingerprint": "a" * 64, "ring": "K32A12C27", "rings": ["K32A12C27"],
           "pack_path": "/provider/secret/path", "masked_head": True}]


class FakeProvider:
    def __init__(self, ca_pem: str, *, status: int = 200, cert: bool = True, lane: Any = None):
        self.ca_pem = ca_pem
        self.status = status
        self.cert = cert
        self.lane = lane
        self.enrolls: list[dict[str, Any]] = []
        self.auth: list[str] = []
        self._mu = threading.Lock()

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/pgw/v1/enroll/ca":
            return httpx.Response(200, text=self.ca_pem)
        if req.url.path == SDK_ENROLL_PATH:
            with self._mu:
                self.auth.append(req.headers.get("authorization", ""))
                body = json.loads(req.content)
                self.enrolls.append(body)
            if self.status != 200:
                return httpx.Response(self.status, json={"error": {"code": "x", "message": "refused"}})
            out: dict[str, Any] = {"tenant": "acme", "service_token": f"tok-{len(self.enrolls)}", "models": MODELS,
                                   "lane": self.lane}
            if self.cert:
                out["client_cert_pem"] = "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"
            return httpx.Response(200, json=out)
        return httpx.Response(404)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def ca(pki: dict[str, Any]) -> tuple[str, bytes]:
    return Path(pki["ca.pem"]).read_text(), pki["ca_sha256"]


def test_enroll_writes_private_files_and_is_reused(tmp_path: Path, ca: tuple[str, bytes]) -> None:
    pem, fp = ca
    key = SdkKey.new("live", URL, fp)
    prov = FakeProvider(pem, lane={"transport": "tcp", "port": 8444})
    lay = CacheLayout(tmp_path / "c", tmp_path / "s")
    s = ensure_session(key, lay, transport=prov.transport, client_name="host-1")
    assert prov.auth == [key.bearer()]
    body = prov.enrolls[0]
    assert body["client_name"] == "host-1" and "BEGIN CERTIFICATE REQUEST" in body["csr_pem"] and body["sdk_version"]
    assert s.tenant == "acme" and s.token().get_secret_value() == "tok-1" and s.lane == {"transport": "tcp",
                                                                                         "port": 8444}
    assert s.model_entry("QWEN2.5-7B-INSTRUCT")["fingerprint"] == "a" * 64
    assert "pack_path" not in s.models[0]  # provider-internal paths are not kept
    for f in (s.token_file, s.client_key_file):
        assert f is not None and stat.S_IMODE(f.stat().st_mode) == 0o600
    assert stat.S_IMODE(s.dir.stat().st_mode) == 0o700
    assert s.ca_file is not None and s.ca_file.read_text() == pem
    assert "tok-1" not in json.dumps(s.public()) and "tok-1" not in repr(s)
    again = ensure_session(key, lay, transport=prov.transport)
    assert len(prov.enrolls) == 1 and again.token_file == s.token_file  # reused, not enrolled again
    forced = ensure_session(key, lay, transport=prov.transport, force=True)
    assert len(prov.enrolls) == 2 and forced.token().get_secret_value() == "tok-2"
    other = SdkKey.new("live", URL, fp)
    assert load_session(other, lay) is None  # another key never reads this session


def test_concurrent_first_use_enrolls_once(tmp_path: Path, ca: tuple[str, bytes]) -> None:
    pem, fp = ca
    key = SdkKey.new("live", URL, fp)
    prov = FakeProvider(pem)
    lay = CacheLayout(tmp_path / "c", tmp_path / "s")
    out: list[Any] = []
    ts = [threading.Thread(target=lambda: out.append(ensure_session(key, lay, transport=prov.transport)))
          for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(out) == 4 and len(prov.enrolls) == 1


def test_ca_pin_mismatch_sends_nothing(tmp_path: Path, ca: tuple[str, bytes]) -> None:
    pem, _ = ca
    key = SdkKey.new("live", URL, b"\x01" * 32)
    prov = FakeProvider(pem)
    with pytest.raises(E.IntegrityError, match="does not match"):
        ensure_session(key, CacheLayout(tmp_path / "c", tmp_path / "s"), transport=prov.transport)
    assert prov.enrolls == [] and prov.auth == []  # the key was never sent


@pytest.mark.parametrize("status, cls", [(401, E.AuthenticationError), (403, E.PermissionDeniedError),
                                         (429, E.RateLimitError), (404, E.APIConnectionError),
                                         (500, E.InternalServerError), (400, E.BadRequestError)])
def test_refusals_map_to_typed_errors(tmp_path: Path, ca: tuple[str, bytes], status: int,
                                      cls: type[E.PGWAPIError]) -> None:
    pem, fp = ca
    key = SdkKey.new("live", URL, fp)
    with pytest.raises(cls) as ei:
        ensure_session(key, CacheLayout(tmp_path / "c", tmp_path / "s"), transport=FakeProvider(pem, status=status)
                       .transport)
    assert key.encode() not in str(ei.value)
    assert not list((tmp_path / "c").rglob("service_token"))


def test_no_client_cert_and_unpinned_loopback(tmp_path: Path) -> None:
    key = parse_key(SdkKey.new("dev", "http://127.0.0.1:9000", None).encode())
    prov = FakeProvider("", cert=False)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/pgw/v1/enroll/ca":
            return httpx.Response(404)
        return prov.handler(req)

    s = ensure_session(key, CacheLayout(tmp_path / "c", tmp_path / "s"), transport=httpx.MockTransport(handler))
    assert s.ca_file is None and s.client_cert_file is None and s.client_key_file is None
    assert not (s.dir / "secrets" / "client.key").exists()


def test_unreachable_provider(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    key = SdkKey.new("live", URL, b"\x02" * 32)
    with pytest.raises(E.APIConnectionError):
        ensure_session(key, CacheLayout(tmp_path / "c", tmp_path / "s"), transport=httpx.MockTransport(handler))
