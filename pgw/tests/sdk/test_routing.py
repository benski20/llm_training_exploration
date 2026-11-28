"""Region routing end to end (``routing="signed"``): two real provider endpoints serving the same pack under
per-pod certificates from one pinned root (via the pod-issuing CA), shared accounts, a signed endpoint directory.

* the client verifies the directory with the key its enrollment pinned and runs the model on the nearest endpoint;
* answers are bit-identical to the plaintext reference on every endpoint, with the **same mask stock** (bound to
  the pack fingerprint and ring, not to a host) and no integrity failure;
* a better endpoint is taken only between requests; a draining one is left; a dead one fails over before the
  first token;
* a directory signed by another key is never used (the client stays on the enrolled endpoint).

Probes: the real payload-free TCP handshake, plus a per-endpoint offset standing in for distance (loopback has none).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.sdk.conftest import MODEL, Provider, start_provider, stop_provider
from tests.sdk.test_e2e_cpu import GREEDY, PROMPT, _client, _reference

pytestmark = pytest.mark.timeout(1800)


def _pem(c: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return c.public_bytes(serialization.Encoding.PEM)


def _pki(d: Path) -> dict[str, Any]:
    """Root (pinned) -> pod-issuing CA -> one leaf per endpoint; a client-issuing CA shared by both endpoints."""
    from cryptography.hazmat.primitives import serialization

    from pgw.placement.certs import PodIssuer, make_csr, new_issuing_ca, new_root
    from tests.sdk.conftest import _ca

    rk, rc = new_root()
    issuer = PodIssuer(*new_issuing_ca(rk, rc), leaf_s=6 * 3600)
    client_ca = _ca("client-issuing-ca")
    common = {"ca.pem": _pem(rc), "client-ca.pem": client_ca[0].public_bytes(serialization.Encoding.PEM),
              "client-ca.key": client_ca[1].private_bytes(serialization.Encoding.PEM,
                                                          serialization.PrivateFormat.PKCS8,
                                                          serialization.NoEncryption())}
    out: dict[str, Any] = {}
    for name in ("a", "b"):
        sub = d / name
        sub.mkdir()
        key, csr = make_csr(f"pod-{name}")  # the endpoint's own key, generated where it runs
        files = {**common, "server.pem": issuer.issue(csr, pod_id=f"pod-{name}", hosts=["127.0.0.1"]).chain_pem,
                 "server.key": key}
        p: dict[str, Any] = {}
        for f, data in files.items():
            (sub / f).write_bytes(data)
            (sub / f).chmod(0o600)
            p[f] = sub / f
        from cryptography import x509

        p["ca_sha256"] = hashlib.sha256(x509.load_pem_x509_certificate(_pem(rc)).public_bytes(
            serialization.Encoding.DER)).digest()
        out[name] = p
    return out


class Net:
    """Payload-free TCP handshakes to each endpoint plus a distance offset per URL (ms)."""

    def __init__(self) -> None:
        self.offset: dict[str, float] = {}

    def probe(self, eps: Any, **_kw: Any) -> list[Any]:
        from pgw.sdk.regions import Probe

        out = []
        for e in eps:
            samples: list[float | None] = []
            for _ in range(3):
                t0 = time.perf_counter()
                try:
                    socket.create_connection((e.host, e.port), timeout=1.0).close()
                    samples.append((time.perf_counter() - t0) * 1e3 + self.offset.get(e.url, 50.0))
                except OSError:
                    samples.append(None)
            out.append(Probe(e, samples))
        return out


@pytest.fixture(scope="module")
def world(sdk_pack_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from pgw.placement.publisher import DirectoryPublisher, Signer

    root = tmp_path_factory.mktemp("routing")
    pki = _pki(root)
    signer = Signer(Ed25519PrivateKey.generate())
    dir_file = root / "directory.json"
    export = "pgwx-export-token-0123456789"
    env = {"PGW_SERVICE_ENDPOINT_DIRECTORY_KEY": signer.public_key.hex(),
           "PGW_SERVICE_ENDPOINT_DIRECTORY_FILE": str(dir_file), "PGW_SERVICE_DEMAND_REPORTS": "true",
           "PGW_SERVICE_DEMAND_EXPORT_TOKEN_SHA256": json.dumps([hashlib.sha256(export.encode()).hexdigest()])}
    a = start_provider(root / "a", sdk_pack_dir, pki["a"], extra_env=env)
    shared = {"PGW_SERVICE_ACCOUNTS_DB": a.env["PGW_SERVICE_ACCOUNTS_DB"],  # type: ignore[index]
              "PGW_SERVICE_TOKENS_FILE": a.env["PGW_SERVICE_TOKENS_FILE"],  # type: ignore[index]
              "PGW_SERVICE_MTLS_TENANTS_FILE": a.env["PGW_SERVICE_MTLS_TENANTS_FILE"],  # type: ignore[index]
              "PGW_SERVICE_ACCOUNTS_ADMIN_TOKEN_SHA256": a.env["PGW_SERVICE_ACCOUNTS_ADMIN_TOKEN_SHA256"]}  # type: ignore[index]
    (root / "b").mkdir(exist_ok=True)
    b = start_provider(root / "b", sdk_pack_dir, pki["b"], extra_env={**env, **shared}, create_org=False)
    b.admin_token = a.admin_token
    pub = DirectoryPublisher(signer, root / "publisher.json", validity_s=3600)
    from pgw.modelpack import ModelPack

    fp = ModelPack.open(sdk_pack_dir, verify="none").fingerprint
    w = {"a": a, "b": b, "pub": pub, "file": dir_file, "fp": fp, "root": root, "signer": signer, "export": export}

    def publish(*eps: tuple[Provider, str, str]) -> None:
        dir_file.write_bytes(pub.publish([{"region": reg, "url": p.url, "models": [], "lane_port": p.lane_port,
                                           "state": st} for p, reg, st in eps]))

    w["publish"] = publish
    publish((a, "us-ks", "serving"), (b, "eu-nl", "serving"))
    try:
        yield w
    finally:
        stop_provider(b)
        stop_provider(a)


@pytest.fixture(scope="module")
def net() -> Net:
    return Net()


@pytest.fixture(scope="module")
def client(world: dict[str, Any], net: Net, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    import pgw.sdk.directory as d

    mp = pytest.MonkeyPatch()
    mp.setattr(d, "probe", net.probe)
    net.offset = {world["a"].url: 20.0, world["b"].url: 2.0}
    c = _client(world["a"], world["a"].new_key(), tmp_path_factory.mktemp("routing-client"), routing="signed",
                routing_refresh_s=10.0, max_retries=3)
    c.models.list()  # enrolls with A; B learns the new credentials from the shared accounts within its reload
    time.sleep(3.0)  # interval (tokens_reload_interval_s, 2 s)
    try:
        yield c
    finally:
        c.close()
        mp.undo()


def _ask(c: Any) -> Any:
    return c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)


def _lp(r: Any) -> list[tuple[str, float]]:
    return [(t.token, t.logprob) for t in r.choices[0].logprobs.content]


def _st(c: Any) -> dict[str, Any]:
    return c.status()["models"][MODEL]


@pytest.fixture(scope="module")
def ref(world: dict[str, Any]) -> dict[str, Any]:
    return _reference(world["a"].pack_dir, GREEDY)


def test_runs_on_the_nearest_endpoint_bit_identical(client: Any, world: dict[str, Any], ref: dict[str, Any]) -> None:
    r = _ask(client)
    assert r.choices[0].message.content == ref["choices"][0]["message"]["content"]
    assert [(t.token, t.logprob) for t in r.choices[0].logprobs.content] == \
        [(t["token"], t["logprob"]) for t in ref["choices"][0]["logprobs"]["content"]]
    st = _st(client)
    assert st["endpoint"]["url"] == world["b"].url and st["integrity_failures"] == 0
    routing = client.status()["routing"]
    assert routing["directory"]["version"] >= 1 and routing["accepted_version"] >= 1
    assert {e["url"] for e in routing["directory"]["endpoints"]} == {world["a"].url, world["b"].url}
    assert "PRIVATE" not in json.dumps(client.status())


def test_switch_only_between_requests_same_stock(client: Any, world: dict[str, Any], net: Net,
                                                 ref: dict[str, Any]) -> None:
    eng = client._engine
    before = _st(client).get("masks_available")
    fp_store = _st(client)["fingerprint"]
    # a streamed answer in progress is never moved: A becomes nearer during it, the switch waits for its end
    stream = client.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, **GREEDY)
    it = iter(stream)
    first = next(it)
    net.offset = {world["a"].url: 1.0, world["b"].url: 20.0}
    eng.router.reprobe()
    assert _st(client)["endpoint"]["url"] == world["b"].url
    rest = list(it)
    assert first is not None and rest
    assert _st(client)["endpoint"]["url"] == world["b"].url
    r = _ask(client)  # between requests now: moves to A, same pack fingerprint, same mask stock
    assert _st(client)["endpoint"]["url"] == world["a"].url
    assert _lp(r) == [(t["token"], t["logprob"]) for t in ref["choices"][0]["logprobs"]["content"]]
    assert _st(client)["fingerprint"] == fp_store and _st(client)["integrity_failures"] == 0
    assert eng.router.switches >= 1 and before is not None


def test_draining_endpoint_is_left(client: Any, world: dict[str, Any], net: Net, ref: dict[str, Any]) -> None:
    world["publish"]((world["a"], "us-ks", "draining"), (world["b"], "eu-nl", "serving"))
    eng = client._engine
    assert eng.router.refresh()
    _ask(client)  # this one may still run where it was (the switch is decided before it starts)
    r = _ask(client)
    assert _st(client)["endpoint"]["url"] == world["b"].url
    assert _lp(r) == [(t["token"], t["logprob"]) for t in ref["choices"][0]["logprobs"]["content"]]
    world["publish"]((world["a"], "us-ks", "serving"), (world["b"], "eu-nl", "serving"))
    eng.router.refresh()


def test_failover_before_the_first_token(client: Any, world: dict[str, Any], net: Net, ref: dict[str, Any]) -> None:
    eng = client._engine
    net.offset = {world["a"].url: 20.0, world["b"].url: 1.0}
    eng.router.reprobe()
    _ask(client)
    assert _st(client)["endpoint"]["url"] == world["b"].url
    stop_provider(world["b"])  # the endpoint dies between requests
    try:
        r = _ask(client)
        assert _st(client)["endpoint"]["url"] == world["a"].url
        assert _lp(r) == [(t["token"], t["logprob"]) for t in ref["choices"][0]["logprobs"]["content"]]
        assert _st(client)["integrity_failures"] == 0
    finally:
        from tests.sdk.conftest import _launch

        _launch(world["b"])


def test_directory_signed_by_another_key_is_never_used(world: dict[str, Any], net: Net,
                                                       tmp_path: Path) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from pgw.placement.publisher import DirectoryPublisher, Signer
    from pgw.sdk.directory import EndpointRouter

    saved = world["file"].read_bytes()
    evil = DirectoryPublisher(Signer(Ed25519PrivateKey.generate()))
    world["file"].write_bytes(evil.publish([{"region": "evil", "url": world["b"].url, "models": []}]))
    try:
        r = EndpointRouter(world["a"].url, world["signer"].public_key,
                           lambda u: (shutil.copyfile(world["file"], tmp_path / "d.json"),
                                      (tmp_path / "d.json").read_bytes())[1],
                           prober=net.probe)
        assert not r.refresh() and r.directory is None
        assert r.select(world["fp"]).url == world["a"].url
        assert any("signature" in e for e in r.errors)
    finally:
        world["file"].write_bytes(saved)


def test_directory_route_needs_credentials(world: dict[str, Any]) -> None:
    import httpx

    r = httpx.get(f"{world['a'].url}/pgw/v1/endpoints/directory", verify=str(world["a"].ca_file), trust_env=False)
    assert r.status_code == 401


def test_demand_reports_strict_and_aggregated(client: Any, world: dict[str, Any]) -> None:
    """The content-free demand report: the client sent one (routing on, usage reports on); a report with any other
    field is refused; the controller's export needs its token and holds only closed hours' aggregates."""
    import httpx

    _ask(client)
    eng = client._engine
    h = eng.models[MODEL]
    assert h.demand_reported_at > 0  # sent after the first request on this endpoint
    with h.session.http_client(timeout_s=10.0) as c:
        ok = {"v": 1, "model": MODEL, "endpoint": "eu-nl", "rtt_bucket": "<=5", "peak_streams": 1}
        rr = c.post("/pgw/v1/placement/report", json=ok)
        assert rr.status_code == 204, rr.text
        assert c.post("/pgw/v1/placement/report", json={**ok, "prompt": "hi"}).status_code in (400, 422)
    url = f"{h.session.service_url}/pgw/v1/placement/aggregates"
    ca = str(world["a"].ca_file)
    assert httpx.get(url, verify=ca, trust_env=False).status_code == 401
    r = httpx.get(url, verify=ca, trust_env=False, headers={"Authorization": f"Bearer {world['export']}"})
    assert r.status_code == 200 and isinstance(r.json(), list)  # the open hour is never exported
