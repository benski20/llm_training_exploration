"""RunPod provider tooling (``deploy/runpod``): API client, cost guard, pod payloads, watchdog, stop/status, the
in-pod boot (service env, PKI, self-stop) and the launcher's attested bootstrap against a real loopback service.

No RunPod call is made: the API transport is a fake."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

RUNPOD_DIR = Path(__file__).resolve().parents[2] / "deploy" / "runpod"
MODAL_DIR = Path(__file__).resolve().parents[2] / "deploy" / "modal"
SECRET = "rpa_TESTKEY_0123456789abcdefghijklmnopqrstuvwxyz"  # noqa: S105 - fake RunPod API key


def _load(name: str, path: Path) -> ModuleType:
    if str(RUNPOD_DIR) not in sys.path:
        sys.path.insert(0, str(RUNPOD_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pod_boot = _load("pod_boot", RUNPOD_DIR / "pod_boot.py")
runpod_api = _load("runpod_api", RUNPOD_DIR / "runpod_api.py")
launch = _load("pgw_runpod_launch", RUNPOD_DIR / "launch.py")
rtt_probe = _load("pgw_runpod_rtt", RUNPOD_DIR / "rtt_probe.py")


class FakeRunPod:
    """Transport for :class:`runpod_api.RunPodAPI`: records calls, answers like RunPod's REST/GraphQL API."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.pods: dict[str, dict[str, Any]] = {}
        self.prices = {"NVIDIA L40S": {"id": "NVIDIA L40S", "securePrice": 0.9, "communityPrice": 0.8}}
        self.headers: list[dict[str, str]] = []
        self.fail_with: tuple[int, str] | None = None

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        payload = json.loads(body) if body else None
        self.calls.append((method, url, payload))
        self.headers.append(headers)
        assert url.startswith(("https://rest.runpod.io/v1/", "https://api.runpod.io/graphql"))
        if self.fail_with:
            return self.fail_with[0], self.fail_with[1].encode()
        if url.endswith("/graphql"):
            return 200, json.dumps({"data": {"gpuTypes": list(self.prices.values())}}).encode()
        path = url.split("/v1", 1)[1]
        if method == "POST" and path == "/pods":
            pid = f"pod{len(self.pods) + 1}"
            self.pods[pid] = {"id": pid, "desiredStatus": "RUNNING", "publicIp": None, "portMappings": {},
                              "costPerHr": 0.9, "spec": payload}
            return 200, json.dumps({"id": pid}).encode()
        if path.startswith("/pods/"):
            pid = path.split("/")[2]
            if pid not in self.pods:
                return 404, b'{"error": "pod not found"}'
            if method == "GET":
                return 200, json.dumps({k: v for k, v in self.pods[pid].items() if k != "spec"}).encode()
            if method == "POST" and path.endswith("/stop"):
                self.pods[pid]["desiredStatus"] = "EXITED"
                return 200, b"{}"
            if method == "DELETE":
                self.pods.pop(pid)
                return 200, b""
        if path == "/networkvolumes":
            if method == "GET":
                return 200, b"[]"
            return 200, json.dumps({"id": "nv1", **(payload or {})}).encode()
        return 404, b"{}"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeRunPod:
    f = FakeRunPod()
    monkeypatch.setattr(launch, "STATE_DIR", tmp_path / "state")
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    return f


def _api(fake: FakeRunPod) -> Any:
    return runpod_api.RunPodAPI(transport=fake)


# ------------------------------------------------------------------------------------------- API client
def test_api_sends_key_in_header_only_and_redacts_it(fake: FakeRunPod) -> None:
    api = _api(fake)
    assert SECRET not in repr(api)
    pod = api.create_pod({"name": "x"})
    assert pod["id"] == "pod1" and fake.headers[0]["Authorization"] == f"Bearer {SECRET}"
    assert all(SECRET not in url for _, url, _ in fake.calls)
    fake.fail_with = (401, f'{{"error": "bad key {SECRET}"}}')  # a server echoing the key back
    with pytest.raises(runpod_api.RunPodError) as ei:
        api.get_pod("pod1")
    assert SECRET not in str(ei.value) and ei.value.status == 401 and "<RUNPOD_API_KEY>" in str(ei.value)
    with pytest.raises(runpod_api.RunPodError, match="RUNPOD_API_KEY is not set"):
        runpod_api.RunPodAPI(api_key="")


# -------------------------------------------------------------------------------------------- cost guard
def test_cost_estimate_and_hard_cap(fake: FakeRunPod) -> None:
    api = _api(fake)
    p, src = launch.gpu_price("NVIDIA L40S", community=False, gpu_count=1, api=api)
    assert (p, src) == (0.9, "RunPod API")
    p2, src2 = launch.gpu_price("NVIDIA H100 80GB HBM3", community=False, gpu_count=2, api=api)
    assert src2.startswith("table") and p2 == pytest.approx(2 * launch.GPU_PRICE_H["NVIDIA H100 80GB HBM3"][0])
    with pytest.raises(SystemExit, match="price-per-hour"):
        launch.gpu_price("Some Future GPU", community=False, gpu_count=1, api=None)
    est = launch.estimate(0.9, "x", 60, volume_gb=120, network_volume=False)
    assert est.hours == pytest.approx(1 + launch.WATCHDOG_GRACE_MIN / 60)  # the watchdog's grace is billed too
    assert est.total_usd == pytest.approx(est.gpu_usd + est.storage_usd) and est.storage_usd < 0.1
    launch.check_budget(est, 1.5, 60)
    with pytest.raises(SystemExit, match="exceeds --max-usd"):
        launch.check_budget(est, 0.5, 60)


def test_launch_refuses_over_budget_before_any_pod(fake: FakeRunPod, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="exceeds --max-usd"):
        launch.main(["--dc", "EU-RO-1", "--ttl-min", "600", "--max-usd", "1", "--image", "img:1"],
                    api_factory=lambda: _api(fake))
    assert not any(m == "POST" and u.endswith("/pods") for m, u, _ in fake.calls)
    with pytest.raises(SystemExit, match="max-usd is required"):
        launch.main(["--dc", "EU-RO-1"], api_factory=lambda: _api(fake))
    with pytest.raises(SystemExit, match="refused"):
        launch.main(["--dc", "EU-RO-1", "--gpu", "NVIDIA L4", "--max-usd", "9"], api_factory=lambda: _api(fake))


def test_dry_run_prints_payload_and_creates_nothing(fake: FakeRunPod, capsys: pytest.CaptureFixture[str],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY")

    def no_api() -> Any:
        raise AssertionError("no API in a dry run without a key")

    rc = launch.main(["--dc", "EU-RO-1", "--ttl-min", "30", "--max-usd", "2", "--dry-run", "--models",
                      "qwen2.5-0.5b-instruct,qwen2.5-7b-instruct", "--service-env",
                      "PGW_SERVICE_LANE_READ_SPIN_US=2000"], api_factory=no_api)
    assert rc == 0
    out = capsys.readouterr().out
    spec = json.loads(out[out.index("{"):out.rindex("}") + 1])["create_pod"]
    assert spec["cloudType"] == "SECURE" and spec["dataCenterIds"] == ["EU-RO-1"]
    assert spec["gpuTypeIds"] == ["NVIDIA L40S"] and spec["supportPublicIp"] is True
    assert spec["ports"] == ["8443/tcp", "8444/tcp"] and spec["volumeInGb"] == 120
    assert spec["dockerStartCmd"] == ["pgw-runpod-boot"] and spec["volumeMountPath"] == "/workspace"
    env = spec["env"]
    assert env["PGW_RUNPOD_ADMIN_TOKEN_SHA256"] == "<hidden>"  # noqa: S105
    assert env["PGW_RUNPOD_MODELS"] == "qwen2.5-0.5b-instruct,qwen2.5-7b-instruct"
    assert abs(int(env["PGW_RUNPOD_STOP_AT"]) - (time.time() + 1800)) < 60
    assert json.loads(env["PGW_RUNPOD_SERVICE_ENV"]) == {"PGW_SERVICE_LANE_READ_SPIN_US": "2000"}
    assert "nothing created" in out
    with pytest.raises(Exception, match="not allowed"):  # refused before paying, as the pod would refuse it
        launch.main(["--dc", "EU-RO-1", "--max-usd", "2", "--dry-run", "--service-env",
                     "PGW_SERVICE_TLS__CERT_FILE=/tmp/x"], api_factory=no_api)


# ----------------------------------------------------------------------------------------- launch flow
def test_launch_flow_with_mocked_api(fake: FakeRunPod, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    api = _api(fake)
    spawned: list[tuple[str, float, bool]] = []
    monkeypatch.setattr(launch, "spawn_watchdog", lambda pid, at, terminate: spawned.append((pid, at, terminate))
                        or 4242)

    def fake_wait(api_: Any, pod_id: str, **kw: Any) -> tuple[str, int, int]:
        fake.pods[pod_id].update(publicIp="203.0.113.7", portMappings={"8443": 41234, "8444": 41235})
        return launch.__dict__["_orig_wait"](api_, pod_id, sleep=lambda s: None, timeout_s=60)

    launch.__dict__["_orig_wait"] = launch.wait_for_address
    monkeypatch.setattr(launch, "wait_for_address", fake_wait)
    monkeypatch.setattr(launch, "wait_for_ready", lambda url, **kw: None)
    seen: dict[str, Any] = {}

    def fake_bootstrap(url: str, token: str, **kw: Any) -> tuple[str, str, dict[str, Any]]:
        seen.update(url=url, token=token, **kw)
        from tests.enroll_helpers import make_pki

        pem = make_pki(tmp_path / "pki")["ca"].read_text()
        return "pgw_test_SECRETKEYVALUE", pem, {"key_id": "sk_0011223344556677", "models": kw["models"]}

    monkeypatch.setattr(launch, "bootstrap_sdk_key", fake_bootstrap)
    rc = launch.main(["--dc", "EU-RO-1", "--ttl-min", "45", "--max-usd", "3", "--image", "ghcr.io/o/pgw-provider:1",
                      "--org", "acme"], api_factory=lambda: api)
    assert rc == 0
    out = capsys.readouterr().out
    assert "https://203.0.113.7:41234" in out and "sk_0011223344556677" in out
    assert "pgw_test_SECRETKEYVALUE" not in out and SECRET not in out  # the key is in a 0600 file only
    spec = fake.pods["pod1"]["spec"]
    assert spec["env"]["PGW_RUNPOD_ADMIN_TOKEN_SHA256"] == hashlib.sha256(seen["token"].encode()).hexdigest()
    assert seen["token"] not in json.dumps(spec)  # only the hash goes to RunPod
    assert spawned and spawned[0][0] == "pod1" and spawned[0][1] == pytest.approx(time.time() + 45 * 60, abs=60)
    st = launch.load_state()
    assert st["provider"]["url"] == "https://203.0.113.7:41234" and st["provider"]["lane_port"] == 41235
    for p in (launch.state_path(), Path(st["provider"]["sdk_key_file"]), Path(st["provider"]["admin_token_file"])):
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert Path(st["provider"]["sdk_key_file"]).read_text().strip() == "pgw_test_SECRETKEYVALUE"
    blob = "".join(p.read_text() for p in launch.STATE_DIR.iterdir() if p.is_file())
    assert SECRET not in blob
    # a second launch is refused while one is recorded
    with pytest.raises(SystemExit, match="recorded"):
        launch.main(["--dc", "EU-RO-1", "--max-usd", "3", "--image", "x"], api_factory=lambda: api)
    # the client pod lands in the same data center; the SDK key is not passed unless asked
    monkeypatch.setattr(launch, "spawn_watchdog", lambda pid, at, terminate: 4343)
    assert launch.main(["--client-pod", "--max-usd", "1", "--client-image", "ghcr.io/o/pgw-provider-cpu:1"],
                       api_factory=lambda: api) == 0
    cspec = fake.pods["pod2"]["spec"]
    assert cspec["computeType"] == "CPU" and cspec["dataCenterIds"] == ["EU-RO-1"]
    assert cspec["dockerStartCmd"] == ["pgw-runpod-boot", "client"] and "PGW_API_KEY" not in cspec["env"]
    assert cspec["env"]["PGW_PROVIDER_URL"] == "https://203.0.113.7:41234"
    # status and stop
    capsys.readouterr()
    assert launch.main(["--status"], api_factory=lambda: api) == 0
    assert "provider: pod pod1" in capsys.readouterr().out
    killed: list[int] = []
    monkeypatch.setattr(launch, "kill_watchdog", lambda pid: killed.append(pid))
    assert launch.main(["--stop"], api_factory=lambda: api) == 0
    assert fake.pods["pod1"]["desiredStatus"] == "EXITED" and fake.pods["pod2"]["desiredStatus"] == "EXITED"
    assert sorted(killed) == [4242, 4343]
    assert launch.main(["--stop", "--terminate"], api_factory=lambda: api) == 0
    assert "pod1" not in fake.pods


def test_watchdog_stops_at_deadline_plus_grace(fake: FakeRunPod) -> None:
    api = _api(fake)
    api.create_pod({"name": "p"})
    now = [1000.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    assert launch.run_watchdog(api, "pod1", 1600.0, sleep=sleep, clock=lambda: now[0]) == "stopped"
    assert now[0] >= 1600 + launch.WATCHDOG_GRACE_MIN * 60 and fake.pods["pod1"]["desiredStatus"] == "EXITED"
    assert launch.run_watchdog(api, "pod1", 0.0, sleep=sleep, clock=lambda: now[0]) == "already-stopped"
    assert launch.run_watchdog(api, "pod1", 0.0, terminate=True, sleep=sleep, clock=lambda: now[0]) == "terminated"
    assert launch.run_watchdog(api, "pod1", 0.0, sleep=sleep, clock=lambda: now[0]) == "gone"


def test_wait_for_address_reads_port_mapping(fake: FakeRunPod) -> None:
    api = _api(fake)
    api.create_pod({"name": "p"})
    n = [0]

    def sleep(_s: float) -> None:
        n[0] += 1
        if n[0] == 2:
            fake.pods["pod1"].update(publicIp="198.51.100.4", portMappings={"8443": 30001})

    assert launch.wait_for_address(api, "pod1", timeout_s=600, sleep=sleep) == ("198.51.100.4", 30001, None)
    fake.pods["pod1"]["desiredStatus"] = "TERMINATED"
    fake.pods["pod1"]["publicIp"] = None
    with pytest.raises(SystemExit, match="TERMINATED"):
        launch.wait_for_address(api, "pod1", timeout_s=600, sleep=lambda s: None)


# -------------------------------------------------------------------------------------------- pod boot
def test_models_table_matches_modal() -> None:
    spec = importlib.util.spec_from_file_location("pgw_modal_boot_rp", MODAL_DIR / "pgw_modal_boot.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert pod_boot.MODELS == mod.MODELS
    assert set(pod_boot.SDK_MODELS) <= set(pod_boot.MODELS)


def test_ensure_pack_rebuilds_older_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pack of pack format 1 on the volume (float32 embeddings) is rebuilt once as the current format, whose
    fingerprints the client library pins; a current-format pack is reused as is."""
    from pgw.modelpack import PACK_VERSION

    assert pod_boot.PACK_FORMAT_VERSION == PACK_VERSION
    runs: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> None:
        runs.append(cmd)
        out = next(Path(c) for c in cmd if c.endswith(".partial"))
        out.mkdir(parents=True)
        (out / "manifest.json").write_text(json.dumps({"format": "pgw-modelpack", "version": PACK_VERSION}))

    monkeypatch.setattr(pod_boot.subprocess, "run", fake_run)
    model = pod_boot.SDK_MODELS[0]
    packs = tmp_path / "packs"
    old = packs / model
    old.mkdir(parents=True)
    (old / "manifest.json").write_text(json.dumps({"format": "pgw-modelpack", "version": 1}))
    (old / "plain.safetensors").write_bytes(b"x")
    dst = pod_boot.ensure_pack(model, packs, tmp_path / "hf")
    assert dst == old and len(runs) == 1
    assert json.loads((dst / "manifest.json").read_text())["version"] == PACK_VERSION
    assert not (dst / "plain.safetensors").exists() and not (packs / f".{model}.old").exists()
    pod_boot.ensure_pack(model, packs, tmp_path / "hf")
    assert len(runs) == 1  # current format: reused
    assert runs[0][1] == "-c" and runs[0][2] == pod_boot.SHARDWISE_PACK  # shard-wise by default
    assert pod_boot.pack_command(model, packs / "x.partial", packs / "w", "whole")[1:4] == ["-m", "pgw.cli", "pack"]
    with pytest.raises(pod_boot.BootError):
        pod_boot.pack_command(model, packs / "x.partial", packs / "w", "other")


def test_shardwise_pack_command_equals_pgw_pack(tmp_path: Path) -> None:
    """The boot's shard-wise packing subprocess writes the pack ``pgw pack`` writes (same fingerprint and files):
    the client library's pins hold for either."""
    import subprocess

    from pgw.modelpack import ModelPack, pack_checkpoint
    from tests.conftest import _save_tiny

    src = _save_tiny(tmp_path / "hf", "qwen2", seed=0)
    model = pod_boot.SDK_MODELS[0]
    whole = tmp_path / "whole"
    pack_checkpoint(src, whole, name=model)
    cmd = pod_boot.pack_command(model, tmp_path / "shard.partial", tmp_path / "work", "shardwise")
    cmd[3] = str(src)  # a local checkpoint instead of the hub repo
    cmd[6] = ""  # no revision for a local directory
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)  # noqa: S603
    assert r.returncode == 0, r.stderr[-2000:]
    with ModelPack.open(whole) as a, ModelPack.open(tmp_path / "shard.partial", verify="full") as b:
        assert a.fingerprint == b.fingerprint
    assert sorted(p.name for p in whole.iterdir()) == sorted(p.name for p in (tmp_path / "shard.partial").iterdir())
    assert not (tmp_path / "work").exists()


def test_service_env_tuning_and_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.service import ServiceSettings

    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    pki = pod_boot.ensure_pki(tmp_path / "pki", ["203.0.113.7", "localhost", "127.0.0.1"])
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "tenants.txt").write_text("")
    (tmp_path / "state" / "mtls-tenants.txt").write_text("")
    env = pod_boot.service_env(data=tmp_path, pki=pki, packs=[tmp_path / "packs/m"], public_url="https://203.0.113.7:41234",
                               lane_public_port=41235, admin_token_sha256="ab" * 32)
    assert SECRET not in json.dumps(env) and "RUNPOD_API_KEY" not in env
    for k, v in pod_boot.PROVIDER_TUNING_ENV.items():
        assert env[k] == v
    assert env["PGW_SERVICE_LANE_ADVERTISED_PORT"] == "41235" and env["PGW_SERVICE_LANE"] == "tcp"
    assert not any("SHM" in k or "LANE_SOCKET" in k for k in env)  # same-host lanes do not apply
    assert json.loads(env["PGW_SERVICE_ACCOUNTS_ADMIN_TOKEN_SHA256"]) == ["ab" * 32]
    # the settings validate as the service would read them
    for k, v in env.items():
        if k.startswith("PGW_SERVICE_"):
            monkeypatch.setenv(k, v)
    s = ServiceSettings()
    assert s.lane == "tcp" and s.lane_advertised_port == 41235 and s.accounts_admin_api
    assert s.masked_head == "also" and s.public_url == "https://203.0.113.7:41234"
    no_lane = pod_boot.service_env(data=tmp_path, pki=pki, packs=[], public_url="https://h:1", lane_public_port=None,
                                   admin_token_sha256=None)
    assert no_lane["PGW_SERVICE_LANE"] == "off" and "PGW_SERVICE_ACCOUNTS_ADMIN_API" not in no_lane
    with pytest.raises(pod_boot.BootError):
        pod_boot.service_env(data=tmp_path, pki=pki, packs=[], public_url="https://h:1", lane_public_port=None,
                             admin_token_sha256=None, extra={"PGW_SERVICE_ACCOUNTS_DB": "x.db"})
    with pytest.raises(pod_boot.BootError):
        pod_boot.checked_service_env(json.dumps({"LD_PRELOAD": "x"}))


def test_pki_keeps_the_ca_and_reissues_the_server_cert(tmp_path: Path) -> None:
    from cryptography import x509

    a = pod_boot.ensure_pki(tmp_path / "pki", ["203.0.113.7", "localhost"])
    ca1 = pod_boot.ca_sha256(a["service_ca"])
    cert1 = a["server_cert"].read_bytes()
    b = pod_boot.ensure_pki(tmp_path / "pki", ["198.51.100.9", "localhost"])
    assert pod_boot.ca_sha256(b["service_ca"]) == ca1  # SDK keys pin it: stable across restarts
    assert b["server_cert"].read_bytes() != cert1
    leaf = x509.load_pem_x509_certificates(b["server_cert"].read_bytes())[0]
    sans = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "198.51.100.9" in {str(ip) for ip in sans.get_values_for_type(x509.IPAddress)}
    for k in ("service_ca_key", "server_key", "client_ca_key"):
        assert stat.S_IMODE(b[k].stat().st_mode) == 0o600


def test_public_endpoint_and_models() -> None:
    env = {"RUNPOD_PUBLIC_IP": "203.0.113.7", "RUNPOD_TCP_PORT_8443": "41234", "RUNPOD_TCP_PORT_8444": "41235"}
    assert pod_boot.public_endpoint(env) == ("203.0.113.7", 41234, 41235)
    assert pod_boot.public_endpoint({**env, "PGW_RUNPOD_PUBLIC_HOST": "gpu.example.net"})[0] == "gpu.example.net"
    with pytest.raises(pod_boot.BootError, match="public IP"):
        pod_boot.public_endpoint({"RUNPOD_PUBLIC_IP": "203.0.113.7"})
    assert pod_boot.url_for("2001:db8::1", 8443) == "https://[2001:db8::1]:8443"
    assert pod_boot.parse_models(None) == list(pod_boot.SDK_MODELS)
    with pytest.raises(pod_boot.BootError):
        pod_boot.parse_models("gpt-5")


def test_stop_timer_drains_then_stops_the_pod() -> None:
    now = [0.0]
    order: list[str] = []

    def stopper(pod: str, key: str) -> bool:
        order.append(f"stop:{pod}")
        assert key == SECRET
        return len(order) > 2  # first API attempt fails, the retry succeeds

    t = pod_boot.StopTimer(100.0, "podX", SECRET, lambda: order.append("drain"), clock=lambda: now[0],
                           sleep=lambda s: now.__setitem__(0, now[0] + s), stopper=stopper)
    t.run()
    assert order == ["drain", "stop:podX", "stop:podX"] and t.fired.is_set() and now[0] >= 100
    t2 = pod_boot.StopTimer(None, "podX", SECRET, lambda: order.append("x"))
    t2.run()
    assert "x" not in order


def test_stop_self_uses_pod_scoped_key_in_header() -> None:
    seen: dict[str, Any] = {}

    class Resp:
        status = 200

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    def opener(req: Any, timeout: float) -> Resp:
        seen.update(url=req.full_url, method=req.get_method(), auth=req.get_header("Authorization"))
        return Resp()

    assert pod_boot.stop_self("podX", SECRET, opener=opener)
    assert seen == {"url": "https://rest.runpod.io/v1/pods/podX/stop", "method": "POST", "auth": f"Bearer {SECRET}"}


# ---------------------------------------------------------- attested bootstrap against a real loopback service
@pytest.mark.timeout(240)
def test_bootstrap_sdk_key_against_real_tls_service(tiny_pack_dir: Path, tmp_path: Path) -> None:
    """The launcher's bootstrap: attestation (HMAC with the token's hash), CA pinned, org + key via the admin API;
    the key then enrolls an installation. A wrong token never gets past the attestation."""
    from pgw.enroll.client import enroll_with_sdk_key
    from pgw.enroll.service import ca_fingerprint, load_ca_pem
    from tests.enroll_helpers import enroll_service

    token = "pgwa-" + os.urandom(16).hex()
    with enroll_service(tiny_pack_dir, tmp_path / "svc", accounts_db=tmp_path / "svc" / "accounts.db",
                        accounts_admin_api=True,
                        accounts_admin_token_sha256=[hashlib.sha256(token.encode()).hexdigest()]) as svc:
        with pytest.raises(SystemExit, match="attestation failed"):
            launch.bootstrap_sdk_key(svc.url, "pgwa-wrong", org="acme", key_name="k", models=[])
        key, ca_pem, rec = launch.bootstrap_sdk_key(svc.url, token, org="acme", key_name="runpod-test",
                                                    models=["tiny-qwen2"])
        assert ca_fingerprint(ca_pem) == ca_fingerprint(load_ca_pem(svc.dirs["ca"]))
        assert rec["org_id"] == "acme" and rec["models"] == ["tiny-qwen2"]
        key2, _, _ = launch.bootstrap_sdk_key(svc.url, token, org="acme", key_name="second", models=[])
        assert key2 != key  # the organisation is reused
        res = enroll_with_sdk_key(key, tmp_path / "client", installation_name="client-pod")
        assert res.tenant.startswith("sdk.") and res.models[0]["name"] == "tiny-qwen2"
        out = rtt_probe.probe(svc.url, ca_fingerprint(ca_pem).hex(), 5)
        assert out["n"] == 5 and out["rtt_p50_ms"] > 0 and "warning" not in out
        with pytest.raises(SystemExit, match="pin mismatch"):
            rtt_probe.probe(svc.url, "00" * 32, 1)
