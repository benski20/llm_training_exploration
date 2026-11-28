"""Signed endpoint directory: signature, freshness, rollback, form (pgw.sdk.directory + pgw.placement.publisher),
and the client's router (refresh, failover, switch only between requests)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from pgw.placement.publisher import DirectoryPublisher, Signer, generate_key, load_signer
from pgw.sdk.directory import (
    MAX_VALIDITY_S,
    DirectoryError,
    EndpointRouter,
    VersionStore,
    directory_key_id,
    rtt_bucket,
    verify_directory,
)
from pgw.sdk.regions import Endpoint, Probe

ENROLLED = "https://main.example.net:8443"
EPS = [{"region": "us-ks", "url": "https://10.0.0.1:8443", "models": ["fp7"], "lane_port": 8444, "dc": "US-KS-2"},
       {"region": "eu-nl", "url": "https://10.0.0.2:8443", "models": ["fp7"], "dc": "EU-NL-1"}]


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _signer() -> Signer:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Signer(Ed25519PrivateKey.generate())


@pytest.fixture
def pub() -> tuple[DirectoryPublisher, Clock]:
    c = Clock()
    return DirectoryPublisher(_signer(), clock=c), c


def test_roundtrip(pub: tuple[DirectoryPublisher, Clock]) -> None:
    p, c = pub
    d = verify_directory(p.publish(EPS), p.signer.public_key, now=c(), enrolled_url=ENROLLED)
    assert d.version == 1 and d.key_id == directory_key_id(p.signer.public_key)
    assert [e.url for e in d.endpoints] == [ENROLLED, "https://10.0.0.1:8443", "https://10.0.0.2:8443"]
    assert d.lane_ports == {"https://10.0.0.1:8443": 8444} and d.dcs["https://10.0.0.2:8443"] == "EU-NL-1"
    assert p.publish(EPS) != p.publish(EPS)  # versions increase
    assert p.version == 3


def test_wrong_key_and_tamper(pub: tuple[DirectoryPublisher, Clock]) -> None:
    p, c = pub
    blob = p.publish(EPS)
    with pytest.raises(DirectoryError, match="signature"):
        verify_directory(blob, _signer().public_key, now=c())
    w = json.loads(blob)
    raw = base64.urlsafe_b64decode(w["payload"] + "==")
    evil = raw.replace(b"10.0.0.2", b"6.6.6.66")
    w["payload"] = base64.urlsafe_b64encode(evil).decode().rstrip("=")
    with pytest.raises(DirectoryError, match="signature"):
        verify_directory(json.dumps(w), p.signer.public_key, now=c())
    # a signature over the payload without the domain prefix does not verify
    w = json.loads(blob)
    w["sig"] = base64.urlsafe_b64encode(p.signer.sign(raw)).decode().rstrip("=")
    with pytest.raises(DirectoryError, match="signature"):
        verify_directory(json.dumps(w), p.signer.public_key, now=c())
    with pytest.raises(DirectoryError):
        verify_directory(b"{}", p.signer.public_key, now=c())
    with pytest.raises(DirectoryError):
        verify_directory(json.dumps({**json.loads(blob), "extra": 1}), p.signer.public_key, now=c())


def test_expiry_and_future(pub: tuple[DirectoryPublisher, Clock]) -> None:
    p, c = pub
    blob = p.publish(EPS)
    verify_directory(blob, p.signer.public_key, now=c() + 3600)  # expires_at, within skew
    with pytest.raises(DirectoryError, match="expired"):
        verify_directory(blob, p.signer.public_key, now=c() + 3600 + 301)
    with pytest.raises(DirectoryError, match="future"):
        verify_directory(blob, p.signer.public_key, now=c() - 1000)
    with pytest.raises(ValueError):
        DirectoryPublisher(p.signer, validity_s=MAX_VALIDITY_S + 1)


def test_rollback_and_same_version(pub: tuple[DirectoryPublisher, Clock], tmp_path: Path) -> None:
    p, c = pub
    v1, v2 = p.publish(EPS), p.publish(EPS[:1])
    store = VersionStore(tmp_path / "dir-version.json")
    d2 = verify_directory(v2, p.signer.public_key, now=c(), min_version=store.get())
    store.put(d2.version, d2.digest)
    assert (tmp_path / "dir-version.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(DirectoryError, match="rollback"):
        verify_directory(v1, p.signer.public_key, now=c(), min_version=VersionStore(tmp_path / "dir-version.json").get())
    verify_directory(v2, p.signer.public_key, now=c(), min_version=store.get())  # a re-fetch of the same bytes
    # the same version number with other content (a second controller, a forged replay): refused
    other = DirectoryPublisher(p.signer, clock=c)
    other._version = 1
    with pytest.raises(DirectoryError, match="different content"):
        verify_directory(other.publish(EPS), p.signer.public_key, now=c(), min_version=store.get())


def test_publisher_version_persisted(tmp_path: Path) -> None:
    key = tmp_path / "dir.key"
    s = generate_key(key)
    assert key.stat().st_mode & 0o777 == 0o600
    p = DirectoryPublisher(load_signer(key), tmp_path / "pub.json")
    p.publish(EPS)
    p.publish(EPS)
    assert DirectoryPublisher(s, tmp_path / "pub.json").version == 2  # a restart continues, never reuses
    key.chmod(0o644)
    with pytest.raises(PermissionError):
        load_signer(key)


@pytest.mark.parametrize("bad", [
    [{"region": "x", "url": "http://a.example:1"}],
    [{"region": "x", "url": "https://a.example", "lane_port": 0}],
    [{"region": "x", "url": "https://a.example", "state": "maybe"}],
    [{"region": "x", "url": "https://a.example", "secret": "y"}],
    [{"region": "x", "url": "https://a.example/p"}],
])
def test_bad_endpoints_refused(pub: tuple[DirectoryPublisher, Clock], bad: list) -> None:
    p, c = pub
    w = json.loads(p.publish([]))
    raw = json.loads(base64.urlsafe_b64decode(w["payload"] + "=="))
    raw["endpoints"] = bad
    pl = json.dumps(raw).encode()
    from pgw.sdk.directory import DOMAIN

    w = {"payload": base64.urlsafe_b64encode(pl).decode().rstrip("="),
         "sig": base64.urlsafe_b64encode(p.signer.sign(DOMAIN + pl)).decode().rstrip("=")}
    with pytest.raises(DirectoryError):
        verify_directory(json.dumps(w), p.signer.public_key, now=c())


def test_rtt_bucket() -> None:
    assert rtt_bucket(0.3) == "<=1" and rtt_bucket(4.9) == "<=5" and rtt_bucket(500) == ">160"


# ---- router ------------------------------------------------------------------------------------------------------
class Net:
    """Fake network: which endpoints answer the directory fetch and the probes, at what RTT."""

    def __init__(self, blob: bytes):
        self.blob = blob
        self.rtt = {ENROLLED: 30.0, "https://10.0.0.1:8443": 20.0, "https://10.0.0.2:8443": 5.0}
        self.down: set[str] = set()
        self.fetches: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.fetches.append(url)
        if url in self.down:
            raise ConnectionError("down")
        return self.blob

    def probe(self, eps: list[Endpoint]) -> list[Probe]:
        return [Probe(e, [None if e.url in self.down else self.rtt.get(e.url, 99.0)] * 3) for e in eps]


def _router(p: DirectoryPublisher, c: Clock, net: Net, tmp_path: Path) -> EndpointRouter:
    mono = Clock(0.0)
    r = EndpointRouter(ENROLLED, p.signer.public_key, net.fetch, versions=VersionStore(tmp_path / "v.json"),
                       prober=net.probe, clock=c, mono=mono, refresh_s=600, probe_s=600, failure_hold_s=120)
    r._mono_clock = mono  # type: ignore[attr-defined]
    return r


def test_router_selects_nearest_and_switches_with_hysteresis(pub: tuple[DirectoryPublisher, Clock],
                                                             tmp_path: Path) -> None:
    p, c = pub
    net = Net(p.publish(EPS))
    r = _router(p, c, net, tmp_path)
    assert r.select("fp7").url == ENROLLED  # no directory yet: the enrolled endpoint only
    assert r.refresh() and r.directory is not None and r.directory.version == 1
    r.reprobe()
    cur = r.select("fp7")
    assert cur.url == "https://10.0.0.2:8443" and r.lane_port(cur.url) is None
    assert r.select("other-fp").url == ENROLLED  # only the enrolled endpoint serves every model
    assert r.preferred_switch("fp7", cur) is None
    net.rtt["https://10.0.0.1:8443"] = 4.6  # 8 % better: stay
    r.reprobe()
    assert r.preferred_switch("fp7", cur) is None
    net.rtt["https://10.0.0.1:8443"] = 1.0
    r.reprobe()
    assert r.preferred_switch("fp7", cur).url == "https://10.0.0.1:8443"
    rec = r.demand_record("qwen2.5-7b-instruct", "https://10.0.0.1:8443")
    assert rec == {"v": 1, "model": "qwen2.5-7b-instruct", "endpoint": "us-ks", "rtt_bucket": "<=1"}


def test_router_failover_and_refresh_from_another_endpoint(pub: tuple[DirectoryPublisher, Clock],
                                                           tmp_path: Path) -> None:
    p, c = pub
    net = Net(p.publish(EPS))
    r = _router(p, c, net, tmp_path)
    r.refresh()
    cur = r.select("fp7")
    assert cur.url == "https://10.0.0.2:8443"
    net.down.add(cur.url)
    r.report_failure(cur.url)
    assert r.refresh_due() and r.probe_due()
    assert r.select("fp7").url == "https://10.0.0.1:8443"  # held out, next by RTT
    assert r.preferred_switch("fp7", cur).url == "https://10.0.0.1:8443"
    net.blob = p.publish(EPS[:1])  # the controller removed the dead pod
    net.fetches.clear()
    assert r.refresh(prefer=cur.url)
    assert net.fetches[0] == cur.url and len(net.fetches) == 2  # the dead one first, then the next that answers
    assert [e.url for e in r.known()] == [ENROLLED, "https://10.0.0.1:8443"]
    # the old list (still unexpired) is refused from now on, also after a restart
    r2 = _router(p, c, net, tmp_path)
    assert r2.versions.get()[0] == 2


def test_router_refuses_rollback_and_falls_back_on_expiry(pub: tuple[DirectoryPublisher, Clock],
                                                          tmp_path: Path) -> None:
    p, c = pub
    old = p.publish(EPS)
    net = Net(p.publish(EPS[1:]))
    r = _router(p, c, net, tmp_path)
    assert r.refresh()
    net.blob = old
    assert not r.refresh() and r.directory.version == 2 and any("rollback" in e for e in r.errors)
    c.t += 3600 + 301
    assert [e.url for e in r.known()] == [ENROLLED]  # expired: the enrolled endpoint only
    net.blob = b"garbage"
    assert not r.refresh()
    assert r.select("fp7").url == ENROLLED


def test_router_draining_endpoint_not_chosen(pub: tuple[DirectoryPublisher, Clock], tmp_path: Path) -> None:
    p, c = pub
    eps = [dict(EPS[0]), {**EPS[1], "state": "draining"}]
    net = Net(p.publish(eps))
    r = _router(p, c, net, tmp_path)
    r.refresh()
    assert r.select("fp7").url == "https://10.0.0.1:8443"
    cur = Endpoint("eu-nl", "https://10.0.0.2:8443", frozenset({"fp7"}))
    assert r.preferred_switch("fp7", cur).url == "https://10.0.0.1:8443"  # a draining one is left when idle
