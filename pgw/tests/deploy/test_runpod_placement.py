"""Placement on RunPod (``deploy/runpod/placement.py``, ``pod_boot.py placement``): pod payloads (data centre, pack
volume, one-time token), the actuator (create / drain / terminate, capacity fallback, caps), the CLI's dry runs, and
the pod registration flow against the controller's API. No RunPod call is made and nothing is spent."""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

PGW = Path(__file__).resolve().parents[2]
RUNPOD_DIR = PGW / "deploy" / "runpod"
if str(RUNPOD_DIR) not in sys.path:
    sys.path.insert(0, str(RUNPOD_DIR))

import placement as rp  # noqa: E402
import pod_boot  # noqa: E402

from pgw.placement.policy import Controller, Ledger, PolicyConfig  # noqa: E402

SECRET = "rpa_TESTKEY_placement_0123456789abcdef"  # noqa: S105 - fake RunPod API key
MODEL = "qwen2.5-7b-instruct"


class FakeRunPod:
    """Transport for ``RunPodAPI``: records calls; refuses GPU creates in ``full`` data centres."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.pods: dict[str, dict[str, Any]] = {}
        self.full: set[str] = set()

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        payload = json.loads(body) if body else None
        self.calls.append((method, url, payload))
        assert headers["Authorization"] == f"Bearer {SECRET}"
        path = url.split("/v1", 1)[1]
        if method == "POST" and path == "/pods":
            if payload["dataCenterIds"][0] in self.full:
                return 400, b'{"error": "There are no available GPUs of this type in this data center"}'
            pid = f"pod{len(self.pods) + 1}"
            self.pods[pid] = {"id": pid, "spec": payload, "publicIp": "203.0.113.10",
                              "portMappings": {"8443": 41000 + len(self.pods), "8444": 42000 + len(self.pods)}}
            return 200, json.dumps({"id": pid}).encode()
        if method == "GET" and path.startswith("/pods/"):
            p = self.pods.get(path.split("/")[2])
            return (200, json.dumps({k: v for k, v in p.items() if k != "spec"}).encode()) if p else (404, b"{}")
        if method == "DELETE" and path.startswith("/pods/"):
            return (200, b"") if self.pods.pop(path.split("/")[2], None) else (404, b"{}")
        if path == "/networkvolumes":
            return (200, b"[]") if method == "GET" else (200, json.dumps({"id": "nv-" + payload["dataCenterId"],
                                                                          **payload}).encode())
        return 404, b"{}"


class Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t


def _actuator(api: Any, c: Controller, clock: Clock, **kw: Any) -> rp.Actuator:
    vols = {d: f"nv-{d}" for d in c.dcs}
    return rp.Actuator(api, c, image="ghcr.io/x/pgw-provider:t", volumes=vols, controller_url="https://ctl:9443",
                       controller_ca_pem=b"-----BEGIN CERTIFICATE-----\nCTL\n-----END CERTIFICATE-----\n",
                       clock=clock, **kw)


def test_pod_spec_targets_dc_and_pack_volume() -> None:
    spec = rp.placement_pod_spec(name="pgw-place-eu-1", image="img", gpu="NVIDIA H100 80GB HBM3", dc="EU-FR-1",
                                 model=MODEL, volume_id="nv1", token="pgwreg_secret", controller_url="https://c",
                                 controller_ca_pem=b"CA", stop_at=2e9)
    assert spec["dataCenterIds"] == ["EU-FR-1"] and spec["networkVolumeId"] == "nv1"
    assert spec["cloudType"] == "SECURE" and spec["interruptible"] is False
    assert spec["dockerStartCmd"] == ["pgw-runpod-boot", "placement"]
    assert spec["env"]["PGW_RUNPOD_STOP_AT"] == "2000000000"  # the pod's own TTL backstop
    assert base64.b64decode(spec["env"]["PGW_PLACEMENT_CONTROLLER_CA_PEM_B64"]) == b"CA"
    assert "pgwreg_secret" not in json.dumps(rp.redacted(spec))
    assert not any("KEY" in k and "CA" in k for k in spec["env"])  # no CA private key ever goes to a pod


def test_actuator_dry_run_create_drain_terminate() -> None:
    clock = Clock()
    c = Controller(PolicyConfig(models=(MODEL,), warm={("new-york", MODEL): 1, ("london", MODEL): 1},
                                idle_timeout_s=0, min_lifetime_s=0))
    tokens: list[tuple[str, str, str]] = []
    act = _actuator(None, c, clock, expect=lambda n, d, m: tokens.append((n, d, m)) or "pgwreg_x", dry_run=True)
    out = act.apply(c.plan(clock(), {}, act.fleet()))
    created = [r["create_pod"] for r in out if "create_pod" in r]
    assert sorted(s["dataCenterIds"][0] for s in created) == ["EU-FR-1", "US-DE-1"]
    assert all(s["env"]["PGW_PLACEMENT_TOKEN"].startswith("<") for s in created)
    assert sorted(t[1] for t in tokens) == ["EU-FR-1", "US-DE-1"]
    for p in act.fleet():
        p.state = "serving"
    c.cfg.warm = {("new-york", MODEL): 1}  # London no longer warm
    drained: list[str] = []
    removed: list[str] = []
    act.drain, act.remove = drained.append, removed.append
    clock.t += 60
    out = act.apply(c.plan(clock(), {}, act.fleet()))
    assert [r for r in out if "drain" in r] and len(drained) == 1 and not removed
    clock.t += rp.DRAIN_GRACE_S - 1
    assert not [r for r in act.apply(c.plan(clock(), {}, act.fleet())) if "terminate" in r]
    clock.t += 2
    out = act.apply(c.plan(clock(), {}, act.fleet()))
    assert [r for r in out if "terminate" in r] and removed == drained  # out of the directory, then terminated
    assert [p.dc for p in act.fleet()] == ["US-DE-1"]


def test_actuator_live_calls_capacity_fallback_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRunPod()
    api = rp.RunPodAPI(SECRET, transport=fake)
    clock = Clock()
    c = Controller(PolicyConfig(models=(MODEL,), warm={("new-york", MODEL): 1}, max_usd_per_hour=7.0,
                                unavailable_hold_s=1800))
    fake.full.add("US-DE-1")
    act = _actuator(api, c, clock, dry_run=False)
    out = act.apply(c.plan(clock(), {}, act.fleet()))
    assert out[0]["create_failed"] == f"US-DE-1/{MODEL}" and "US-DE-1" in c.unavailable
    out = act.apply(c.plan(clock(), {}, act.fleet()))  # next tick: the next-best data centre
    assert out[0]["dc"] == "CA-MTL-3" and fake.pods["pod1"]["spec"]["dataCenterIds"] == ["CA-MTL-3"]
    assert rp.pod_address(api, "pod1") == ("203.0.113.10", 41000, 42000)
    # the hourly cap (7 $/h, 2.99 per pod) admits 2 pods whatever the demand asks for
    out = act.apply(c.plan(clock(), {("new-york", MODEL): 100}, act.fleet()))
    assert len(act.fleet()) == 2 and sum(1 for m, u, _ in fake.calls if m == "POST" and u.endswith("/pods")) == 3
    # max_pods
    act.max_pods = 2
    c.cfg.max_usd_per_hour = 100
    out = act.apply(c.plan(clock(), {("new-york", MODEL): 100}, act.fleet()))
    assert any("max-pods" in r.get("refused", "") for r in out)
    # a data centre without a staged pack volume is never used
    act.volumes = {}
    act.max_pods = 99
    assert any("no pre-staged" in r.get("refused", "") for r in act.apply(c.plan(clock(), {("new-york", MODEL): 200},
                                                                               act.fleet())))


def test_tick_anticipates_and_accounts() -> None:
    clock = Clock()
    c = Controller(PolicyConfig(models=(MODEL,), warm={("new-york", MODEL): 1}))
    act = _actuator(None, c, clock, dry_run=True)
    from pgw.placement.forecast import Forecaster

    f, led = Forecaster(), Ledger()
    plan = rp.tick(c, act, f, {("new-york", MODEL): 30.0}, clock(), led)
    assert plan.desired[("US-DE-1", MODEL)] == 3 and led.samples[-1][1] == pytest.approx(3 * c.price)


def test_cli_dry_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    dem = tmp_path / "demand.json"
    dem.write_text(json.dumps({"london": {MODEL: 12}, "new-york": {MODEL: 4}}))
    assert rp.main(["plan", "--demand", str(dem), "--warm", "new-york=1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["desired"] == {f"EU-FR-1/{MODEL}": 1, f"US-DE-1/{MODEL}": 1}

    def no_api() -> Any:
        raise AssertionError("a dry run must not build the API client")

    assert rp.main(["apply", "--demand", str(dem), "--warm", "new-york=1", "--dry-run"], api_factory=no_api) == 0
    text = capsys.readouterr().out
    assert '"dataCenterIds"' in text and "nothing spent" in text and "pgwreg_" not in text
    assert rp.main(["stage-packs", "--dcs", "US-DE-1,EU-FR-1", "--dry-run"], api_factory=no_api) == 0
    text = capsys.readouterr().out
    assert "pgw-packs-us-de-1" in text and '"stage-packs"' in text and "nothing spent" in text
    with pytest.raises(SystemExit, match="exceeds"):
        rp.main(["stage-packs", "--dcs", "US-DE-1,EU-FR-1", "--dry-run", "--max-usd", "0.1"], api_factory=no_api)
    assert rp.main(["coldstart", "--model", MODEL]) == 0
    assert "ready" in capsys.readouterr().out


def test_placement_service_env_has_no_ca_key_or_accounts(tmp_path: Path) -> None:
    pki = {"server_cert": tmp_path / "server.pem", "server_key": tmp_path / "k", "client_ca": tmp_path / "cc.pem"}
    env = pod_boot.placement_service_env(data=tmp_path, pki=pki, packs=[tmp_path / "p"], public_url="https://x:1",
                                         lane_public_port=2)
    assert not any(k.startswith(("PGW_SERVICE_ACCOUNTS", "PGW_SERVICE_CLIENT_ISSUING_CA", "PGW_SERVICE_ENROLLMENT"))
                   for k in env)
    assert env["PGW_SERVICE_VERIFY_PACKS"] == "full" and "RUNPOD_API_KEY" not in env
    with pytest.raises(pod_boot.BootError):
        pod_boot.placement_service_env(data=tmp_path, pki=pki, packs=[], public_url="https://x:1", lane_public_port=2,
                                       extra={"PGW_SERVICE_ACCOUNTS_DB": "/x"})


# ---- the pod's boot against the controller's API -------------------------------------------------------------
@pytest.fixture
def controller(tmp_path: Path) -> dict[str, Any]:
    from cryptography.hazmat.primitives import serialization
    from starlette.testclient import TestClient

    from pgw.placement.certs import PodIssuer, new_issuing_ca, new_root
    from pgw.placement.publisher import DirectoryPublisher, Signer
    from pgw.placement.registry import Registry
    from pgw.placement.server import create_controller_app

    rk, rc = new_root()
    issuer = PodIssuer(*new_issuing_ca(rk, rc))
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pub = DirectoryPublisher(Signer(Ed25519PrivateKey.generate()))
    published: list[bytes] = []
    st: dict[str, Any] = {"published": published, "pub": pub, "tmp": tmp_path}
    reg = Registry(issuer, {}, on_change=lambda eps: published.append(pub.publish(eps)))
    root_pem = rc.public_bytes(serialization.Encoding.PEM)

    def probe(url: str, cert: Any) -> dict[str, Any]:  # the served leaf = what the pod wrote from its registration
        served = (tmp_path / "data" / "pki" / "server.pem").read_bytes()
        from cryptography import x509

        leaf = x509.load_pem_x509_certificates(served)[0].public_bytes(serialization.Encoding.DER)
        import hashlib

        return {"ready": True, "leaf_sha256": hashlib.sha256(leaf).hexdigest(), "fingerprints": st["fps"]}

    tokens_txt = "sdk.0123456789abcdef:t1:" + "ab" * 32 + "\n"
    app = create_controller_app(reg, lookup=lambda pid: ("203.0.113.20", 41001, 42001), probe=probe,
                                root_pem=root_pem, client_ca_pem=b"CLIENT-CA-CERT",
                                tenants=lambda: {"tokens_file": tokens_txt, "mtls_file": "sdk.x:cn\n"})
    st.update(reg=reg, client=TestClient(app), root_pem=root_pem)
    return st


def _http(client: Any) -> tuple[Any, Any]:
    def post(url: str, body: dict[str, Any], ctx: Any, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
        r = client.post(url.split("ctl:9443", 1)[1], json=body, headers=headers or {})
        return r.status_code, r.content

    def get(url: str, ctx: Any, headers: dict[str, str]) -> tuple[int, bytes]:
        r = client.get(url.split("ctl:9443", 1)[1], headers=headers)
        return r.status_code, r.content

    return post, get


def _pod_env(tmp_path: Path, token: str, fp: str) -> dict[str, str]:
    return {"RUNPOD_POD_ID": "pod-7", "PGW_PLACEMENT_POD_NAME": "pgw-place-eu-1", "PGW_PLACEMENT_DC": "EU-FR-1",
            "PGW_RUNPOD_MODELS": "qwen2.5-0.5b-instruct", "PGW_PLACEMENT_TOKEN": token,
            "PGW_PLACEMENT_PACKS": str(tmp_path / "vol"), "PGW_PLACEMENT_KEY_DIR": str(tmp_path / "shm"),
            "PGW_RUNPOD_DATA": str(tmp_path / "data"), "PGW_PLACEMENT_CONTROLLER": "https://ctl:9443",
            "PGW_PLACEMENT_CONTROLLER_CA_PEM_B64": base64.b64encode(b"x").decode(),
            "PGW_PLACEMENT_EXPECT_FP": json.dumps({"qwen2.5-0.5b-instruct": fp})}


def test_pod_boot_registers_and_is_listed(controller: dict[str, Any], tiny_pack_dir: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.modelpack import ModelPack
    from pgw.modelpack.masked_head import head_fingerprint
    from pgw.placement.certs import verify_chain
    from pgw.placement.registry import PodBoot
    from pgw.sdk.directory import verify_directory

    tmp = controller["tmp"]
    shutil.copytree(tiny_pack_dir, tmp / "vol" / "qwen2.5-0.5b-instruct")
    fp = ModelPack.open(tiny_pack_dir, verify="none").fingerprint
    controller["reg"].pinned["qwen2.5-0.5b-instruct"] = fp
    controller["fps"] = [fp, head_fingerprint(fp)]
    token = controller["reg"].expect("pgw-place-eu-1", "EU-FR-1", "qwen2.5-0.5b-instruct")
    monkeypatch.setattr(pod_boot, "_controller", lambda env: ("https://ctl:9443", None))
    post, get = _http(controller["client"])
    started: list[dict[str, str]] = []
    ctx: dict[str, Any] = {}
    steps = pod_boot.placement_steps(_pod_env(tmp, token, fp), ctx, http_post=post, http_get=get,
                                     start_service=lambda env, data, uid: started.append(env) or "proc",
                                     check_device=lambda: True)
    boot = PodBoot(steps)
    boot.run(ctx)
    assert list(boot.timings) == ["verify_pack", "device", "keygen", "register", "tenants", "service_ready",
                                  "activate"]
    key = tmp / "shm" / "server.key"
    assert key.stat().st_mode & 0o777 == 0o600 and "PRIVATE KEY" in key.read_text()
    assert not list((tmp / "data").rglob("*.key"))  # the pod's key is on tmpfs only, never the volume
    verify_chain((tmp / "data" / "pki" / "server.pem").read_bytes(), controller["root_pem"], "203.0.113.20")
    senv = started[0]
    assert senv["PGW_SERVICE_TLS__KEY_FILE"] == str(key) and senv["PGW_SERVICE_PUBLIC_URL"] == "https://203.0.113.20:41001"
    assert (tmp / "data" / "state" / "tenants.txt").read_text().startswith("sdk.0123456789abcdef:t1:")
    d = verify_directory(controller["published"][-1], controller["pub"].signer.public_key)
    assert [e.url for e in d.endpoints] == ["https://203.0.113.20:41001"]
    assert d.lane_ports == {"https://203.0.113.20:41001": 42001}
    # the one-time token cannot register a second pod; an unregistered pod gets no tenant digests
    code, _ = post("https://ctl:9443/placement/v1/register", {**json.loads(json.dumps({
        "token": token, "pod_id": "pod-8", "pod_name": "pgw-place-eu-1", "dc": "EU-FR-1",
        "model": "qwen2.5-0.5b-instruct", "fingerprints": [fp], "selftests": {"x": True},
        "csr_pem": ctx["csr_pem"].decode()}))}, None)
    assert code == 403
    assert get("https://ctl:9443/placement/v1/tenants", None, {"PGW-Pod-Id": "pod-7",
                                                               "PGW-Pod-Secret": "guess"})[0] == 403


def test_pod_boot_refuses_a_tampered_volume(controller: dict[str, Any], tiny_pack_dir: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.modelpack import ModelPack
    from pgw.placement.registry import PodBoot

    tmp = controller["tmp"]
    dst = tmp / "vol" / "qwen2.5-0.5b-instruct"
    shutil.copytree(tiny_pack_dir, dst)
    fp = ModelPack.open(tiny_pack_dir, verify="none").fingerprint
    f = sorted(dst.glob("*.safetensors"))[0]
    b = bytearray(f.read_bytes())
    b[-1] ^= 1
    f.write_bytes(bytes(b))
    controller["reg"].pinned["qwen2.5-0.5b-instruct"] = fp
    token = controller["reg"].expect("pgw-place-eu-1", "EU-FR-1", "qwen2.5-0.5b-instruct")
    monkeypatch.setattr(pod_boot, "_controller", lambda env: ("https://ctl:9443", None))
    post, get = _http(controller["client"])
    calls: list[str] = []
    steps = pod_boot.placement_steps(_pod_env(tmp, token, fp), {}, http_post=lambda *a, **k: calls.append("post"),
                                     http_get=get, start_service=lambda *a: calls.append("svc"),
                                     check_device=lambda: True)
    with pytest.raises(Exception):  # noqa: B017 - FingerprintMismatchError or ConfigError: the boot stops
        PodBoot(steps).run({})
    assert calls == [] and not controller["published"] and not controller["reg"].pods
    assert os.path.exists(dst)
