"""RunPod measurement sessions (``deploy/runpod/jobs.py``, ``pod_job.py``, ``sdk_remote_bench.py``): cost guards,
pod payloads, the in-pod control server (tokens, pinned bundle, peer channel), and a two-pod lane_nic session run
locally on CPU. No RunPod call is made and nothing is spent."""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

import pytest

PGW = Path(__file__).resolve().parents[2]
RUNPOD_DIR = PGW / "deploy" / "runpod"
if str(RUNPOD_DIR) not in sys.path:
    sys.path.insert(0, str(RUNPOD_DIR))

import jobs  # noqa: E402
import runpod_api  # noqa: E402


class FakeAPI:
    """Stands in for :class:`runpod_api.RunPodAPI`; records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.pods: dict[str, dict[str, Any]] = {}
        self.refuse_cpu = False

    def gpu_prices(self) -> dict[str, dict[str, Any]]:
        self.calls.append(("gpu_prices", None))
        return {jobs.H100: {"securePrice": 2.99, "communityPrice": 2.69}}

    def account(self) -> dict[str, Any]:
        self.calls.append(("account", None))
        return {"clientBalance": 100.0, "currentSpendPerHr": 0.0, "spendLimit": 80}

    def list_pods(self) -> list[dict[str, Any]]:
        self.calls.append(("list_pods", None))
        return list(self.pods.values())

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self.pods[pod_id]

    def create_pod(self, spec: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_pod", spec))
        if spec["computeType"] == "CPU" and self.refuse_cpu:
            raise runpod_api.RunPodError("no CPU capacity", 400)
        pid = f"pod{sum(1 for c in self.calls if c[0] == 'create_pod') - 1}"
        self.pods[pid] = {"id": pid, "name": spec["name"], "desiredStatus": "RUNNING", "costPerHr": 1.0}
        return {"id": pid}

    def stop_pod(self, pod_id: str) -> None:
        self.calls.append(("stop_pod", pod_id))
        self.pods[pod_id]["desiredStatus"] = "EXITED"

    def terminate_pod(self, pod_id: str) -> None:
        self.calls.append(("terminate_pod", pod_id))
        self.pods.pop(pod_id, None)


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "state"
    monkeypatch.setattr(jobs.launch, "STATE_DIR", d)
    monkeypatch.setattr(jobs, "STATE_FILE", d / "jobs.json")
    monkeypatch.setattr(jobs.launch, "spawn_watchdog", lambda pod_id, at, terminate: 4242)
    monkeypatch.setattr(jobs.launch, "kill_watchdog", lambda pid: None)
    return d


def test_plan_fits_the_default_budget(capsys: pytest.CaptureFixture[str]) -> None:
    assert jobs.main(["plan"]) == 0
    out = capsys.readouterr().out
    total = float(out.split("total (all sessions at their full TTL): $")[1].split(";")[0])
    assert 20 < total < 60
    assert list(jobs.SESSIONS) == ["perf", "sdk7b", "sdk32b", "72b"]  # the runbook's priority order


def test_boot_payload_is_pod_job_and_the_start_command_carries_no_secret() -> None:
    raw = zlib.decompress(jobs.base64.b64decode(jobs.boot_payload()))
    assert raw == (RUNPOD_DIR / "pod_job.py").read_bytes()
    compile(raw, "pod_job", "exec")
    assert len(jobs.boot_payload()) < 16_000
    assert "TOKEN" not in jobs.BOOT_CMD and "PGW_JOB_BOOT" in jobs.BOOT_CMD


def test_pod_specs_are_secure_ttl_bound_and_hide_tokens() -> None:
    cfg = jobs.SESSIONS["sdk7b"]
    env = {"PGW_JOB_BOOT": "x" * 100, "PGW_JOB_TOKEN_SHA256": "ab" * 32, "PGW_JOB_PAIR_TOKEN": "secret-pair"}
    g = jobs.gpu_spec("sdk7b", cfg, dc="US-KS-2", stop_at=1e9, env=env)
    assert g["cloudType"] == "SECURE" and g["interruptible"] is False and g["dataCenterIds"] == ["US-KS-2"]
    assert f"{jobs.CONTROL_PORT}/http" in g["ports"] and g["env"]["PGW_JOB_STOP_AT"] == "1000000000"
    assert g["imageName"].split("@sha256:")[1] and g["allowedCudaVersions"]
    c = jobs.cpu_spec("sdk7b", cfg, dc="US-KS-2", stop_at=1e9, env=env, flavor="cpu5c")
    assert c["computeType"] == "CPU" and c["vcpuCount"] == 32 and c["imageName"].split("@sha256:")[1]
    r = jobs.redacted(c)
    assert "secret-pair" not in json.dumps(r) and "x" * 100 not in json.dumps(r)


def test_dry_run_creates_and_spends_nothing(state: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def no_api() -> Any:
        raise AssertionError("a dry run must not build an API client")

    assert jobs.main(["launch", "perf", "--dc", "EU-RO-1", "--max-usd", "9", "--dry-run"], api_factory=no_api) == 0
    out = capsys.readouterr().out
    assert "nothing created, nothing spent" in out and "create_cpu_pod" in out
    assert not (state / "jobs.json").exists()


def test_cost_guards_refuse_before_any_pod(state: Path) -> None:
    api = FakeAPI()
    with pytest.raises(SystemExit, match="exceeds --max-usd"):
        jobs.main(["launch", "sdk7b", "--dc", "EU-RO-1", "--max-usd", "1"], api_factory=lambda: api)
    jobs.save_state({"sessions": {"old": {"est_usd": 55.0, "active": False, "pods": {}}}})
    with pytest.raises(SystemExit, match="exceeds --budget-usd"):
        jobs.main(["launch", "sdk7b", "--dc", "EU-RO-1", "--max-usd", "20"], api_factory=lambda: api)
    assert not [c for c in api.calls if c[0] == "create_pod"]


def test_launch_records_pods_uploads_and_stops(state: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeAPI()
    sent: list[tuple[str, str, str]] = []

    def control(pod_id: str, token: str, path: str, *, method: str = "GET", body: bytes | None = None,
                timeout: float = 90.0) -> tuple[int, bytes]:
        sent.append((pod_id, path, method))
        if path == "/status":
            return 200, json.dumps({"state": "waiting-bundle"}).encode()
        assert hashlib.sha256(body or b"").hexdigest() == spec_env(pod_id)["PGW_JOB_BUNDLE_SHA256"]
        assert hashlib.sha256(token.encode()).hexdigest() == spec_env(pod_id)["PGW_JOB_TOKEN_SHA256"]
        return 200, b'{"accepted": 1}'

    def spec_env(pod_id: str) -> dict[str, str]:
        specs = [c[1] for c in api.calls if c[0] == "create_pod"]
        return specs[int(pod_id[3:])]["env"]

    monkeypatch.setattr(jobs, "control", control)
    assert jobs.main(["launch", "sdk7b", "--dc", "EU-RO-1", "--max-usd", "10"], api_factory=lambda: api) == 0
    st = json.loads((state / "jobs.json").read_text())
    assert stat.S_IMODE((state / "jobs.json").stat().st_mode) == 0o600
    rec = st["sessions"]["sdk7b"]
    assert set(rec["pods"]) == {"gpu", "cpu"} and rec["active"]
    gpu_env, cpu_env = spec_env("pod0"), spec_env("pod1")
    assert cpu_env["PGW_JOB_PEER_POD"] == "pod0"
    assert hashlib.sha256(cpu_env["PGW_JOB_PAIR_TOKEN"].encode()).hexdigest() == gpu_env["PGW_JOB_PAIR_TOKEN_SHA256"]
    assert rec["job_token"] not in json.dumps([c[1] for c in api.calls])  # only its hash leaves this machine
    assert ("pod0", "/bundle", "PUT") in sent and ("pod1", "/bundle", "PUT") in sent
    with pytest.raises(SystemExit, match="recorded as active"):
        jobs.main(["launch", "sdk7b", "--dc", "EU-RO-1", "--max-usd", "10"], api_factory=lambda: api)
    assert jobs.main(["teardown-check"], api_factory=lambda: api) == 1
    assert jobs.main(["stop", "--all", "--terminate"], api_factory=lambda: api) == 0
    assert jobs.main(["teardown-check"], api_factory=lambda: api) == 0
    assert "clean: no pods" in capsys.readouterr().out
    # a rerun of a stopped session keeps the first run in the budget
    assert jobs.main(["launch", "sdk7b", "--dc", "EU-RO-1", "--max-usd", "10"], api_factory=lambda: api) == 0
    st = json.loads((state / "jobs.json").read_text())
    assert len(st["sessions"]) == 2 and sum(s["est_usd"] for s in st["sessions"].values()) > 17


def test_a_refused_client_pod_terminates_the_gpu_pod(state: Path) -> None:
    api = FakeAPI()
    api.refuse_cpu = True
    assert jobs.main(["launch", "perf", "--dc", "EU-RO-1", "--max-usd", "10"], api_factory=lambda: api) == 1
    assert ("terminate_pod", "pod0") in api.calls and not api.pods


def test_check_key_only_reads(state: Path, capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeAPI()
    assert jobs.main(["check-key"], api_factory=lambda: api) == 0
    assert {c[0] for c in api.calls} <= {"account", "list_pods", "gpu_prices"}
    assert "balance $100.0" in capsys.readouterr().out


def test_account_query_is_a_read_only_graphql_query() -> None:
    seen: list[Any] = []

    def transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        seen.append(json.loads(body or b"{}"))
        return 200, json.dumps({"data": {"myself": {"clientBalance": 12.5}}}).encode()

    api = runpod_api.RunPodAPI("rpa_fake", transport=transport)
    assert api.account() == {"clientBalance": 12.5}
    assert seen[0]["query"].startswith("query {") and "mutation" not in seen[0]["query"]


# ------------------------------------------------------------------------------------------ pod_job.py
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _bundle(tmp: Path) -> tuple[bytes, str]:
    path, sha = jobs.build_bundle(tmp / "bundle.tar.gz", [])
    return path.read_bytes(), sha


def _start_pod(root: Path, env: dict[str, str]) -> tuple[subprocess.Popen[bytes], int]:
    root.mkdir(parents=True)
    full = {**os.environ, "PGW_JOB_SETUP": "none", **env}
    full.pop("RUNPOD_POD_ID", None)
    p = subprocess.Popen([sys.executable, str(RUNPOD_DIR / "pod_job.py"), "--root", str(root), "--port", "0"],  # noqa: S603
                         env=full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        if (root / "port").exists() and (root / "port").read_text():
            return p, int((root / "port").read_text())
        time.sleep(0.05)
    p.kill()
    raise AssertionError("pod_job did not start")


def _req(port: int, path: str, *, method: str = "GET", body: bytes | None = None,
         headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - loopback
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError:  # the server answered (403/413) and closed before reading the whole body
        return -1, b""


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_pod_job_control_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_proxy(monkeypatch)
    data, sha = _bundle(tmp_path)
    tok, pair = "job-token-1", "pair-token-1"
    env = {"PGW_JOB_TOKEN_SHA256": hashlib.sha256(tok.encode()).hexdigest(), "PGW_JOB_BUNDLE_SHA256": sha,
           "PGW_JOB_PAIR_TOKEN_SHA256": hashlib.sha256(pair.encode()).hexdigest(), "PGW_JOB_ROLE": "gpu",
           "PGW_JOB_PLAN": "[]", "PGW_JOB_STOP_AT": str(int(time.time() + 600))}
    p, port = _start_pod(tmp_path / "pod", env)
    try:
        assert _req(port, "/status")[0] == 403
        assert _req(port, "/status", headers={"X-PGW-Job-Token": "wrong"})[0] == 403
        assert _req(port, "/peer/x", headers={"X-PGW-Job-Token": tok})[0] == 403  # the job token is not a pair token
        code, _ = _req(port, "/bundle", method="PUT", body=data + b"x", headers={"X-PGW-Job-Token": tok})
        assert code == 400  # not the pinned bundle
        assert _req(port, "/bundle", method="PUT", body=b"x", headers={"X-PGW-Job-Token": pair})[0] == 403
        code, body = _req(port, "/bundle", method="PUT", body=data, headers={"X-PGW-Job-Token": tok})
        assert code == 200, body
        assert _req(port, "/bundle", method="PUT", body=b"x", headers={"X-PGW-Job-Token": tok})[0] == 409
        for _ in range(300):
            st = json.loads(_req(port, "/status", headers={"X-PGW-Job-Token": tok})[1])
            if st["state"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert st["state"] == "done" and st["progress"]["step"] == "done"
        code, tgz = _req(port, "/results.tar.gz", headers={"X-PGW-Job-Token": tok})
        assert code == 200 and "results/job.log" in tarfile.open(fileobj=io.BytesIO(tgz)).getnames()
        hp = {"X-PGW-Pair-Token": pair}
        assert _req(port, "/peer/a.json", method="PUT", body=b"{}", headers=hp)[0] == 200
        assert _req(port, "/peer/a.json", headers=hp) == (200, b"{}")
        assert stat.S_IMODE((tmp_path / "pod" / "peer" / "a.json").stat().st_mode) == 0o600
        assert _req(port, "/peer/..%2Fport", headers=hp)[0] == 404
        assert _req(port, "/peer/.hidden", method="PUT", body=b"x", headers=hp)[0] == 400
        assert _req(port, "/peer/big", method="PUT", body=b"x" * (2 << 20), headers=hp)[0] in (413, -1)
        assert not (tmp_path / "pod" / "peer" / "big").exists()
    finally:
        p.kill()
        p.wait()


def test_pod_job_extraction_refuses_paths_out_of_the_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_proxy(monkeypatch)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    data = buf.getvalue()
    tok = "t"
    env = {"PGW_JOB_TOKEN_SHA256": hashlib.sha256(tok.encode()).hexdigest(),
           "PGW_JOB_BUNDLE_SHA256": hashlib.sha256(data).hexdigest(), "PGW_JOB_ROLE": "cpu", "PGW_JOB_PLAN": "[]"}
    p, port = _start_pod(tmp_path / "pod", env)
    try:
        code, _ = _req(port, "/bundle", method="PUT", body=data, headers={"X-PGW-Job-Token": tok})
        assert code != 200
        assert not (tmp_path / "pod" / "escape.txt").exists() and not (tmp_path / "escape.txt").exists()
    finally:
        p.kill()
        p.wait()


@pytest.mark.slow
def test_lane_nic_session_two_pods_on_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The perf session's lane_nic step end to end on loopback: GPU-role pod serves the lane bench (CPU executor,
    tiny widths) and publishes its info through the peer channel; the client-role pod fetches it, drives both
    listeners and marks nothing secret in its results."""
    _no_proxy(monkeypatch)
    data, sha = _bundle(tmp_path)
    tok, pair = "job-token-2", "pair-token-2"
    lane, stub = _free_port(), _free_port()
    plan = json.dumps([{"kind": "lane_nic", "presets": "tiny", "nic_minutes": 1.5}])
    common = {"PGW_JOB_TOKEN_SHA256": hashlib.sha256(tok.encode()).hexdigest(), "PGW_JOB_BUNDLE_SHA256": sha,
              "PGW_JOB_PLAN": plan, "PGW_JOB_STOP_AT": str(int(time.time() + 900)), "PGW_JOB_CPU_SMOKE": "1",
              "PGW_JOB_PROXY_URL_FMT": "http://{pod}", "PYTHONPATH": f"{PGW / 'src'}:{PGW}"}
    g, gport = _start_pod(tmp_path / "gpu", {
        **common, "PGW_JOB_ROLE": "gpu", "PGW_JOB_PAIR_TOKEN_SHA256": hashlib.sha256(pair.encode()).hexdigest(),
        "RUNPOD_PUBLIC_IP": "127.0.0.1", "PGW_JOB_NIC_PORT": str(lane), "PGW_JOB_NIC_STUB_PORT": str(stub),
        f"RUNPOD_TCP_PORT_{lane}": str(lane), f"RUNPOD_TCP_PORT_{stub}": str(stub)})
    c, cport = _start_pod(tmp_path / "cpu", {**common, "PGW_JOB_ROLE": "cpu", "PGW_JOB_PAIR_TOKEN": pair,
                                             "PGW_JOB_PEER_POD": f"127.0.0.1:{gport}"})
    try:
        for port in (gport, cport):
            assert _req(port, "/bundle", method="PUT", body=data, headers={"X-PGW-Job-Token": tok})[0] == 200
        states = {}
        for _ in range(1200):
            states = {r: json.loads(_req(p_, "/status", headers={"X-PGW-Job-Token": tok})[1])["state"]
                      for r, p_ in (("gpu", gport), ("cpu", cport))}
            if all(s in ("done", "failed") for s in states.values()):
                break
            time.sleep(0.5)
        log = (tmp_path / "cpu" / "results" / "job.log").read_text()
        assert states == {"gpu": "done", "cpu": "done"}, log[-3000:]
        res = json.loads((tmp_path / "cpu" / "results" / "lane_nic_tiny_client.json").read_text())
        assert res["preset"] == "tiny" and "derived_1conn" in res
        assert not (tmp_path / "cpu" / "results" / "lane_nic_tiny_info.json").exists()  # tickets deleted
        served = json.loads((tmp_path / "gpu" / "results" / "perf" / "lane_nic_tiny.json").read_text())
        assert "tickets" not in served and "stub_tickets" not in served
    finally:
        for p in (g, c):
            p.kill()
            p.wait()


@pytest.mark.slow
def test_sdk_remote_bench_cpu_smoke(tmp_path: Path) -> None:
    out = tmp_path / "sdkr"
    env = {**os.environ, "PYTHONPATH": f"{PGW / 'src'}:{PGW}"}
    r = subprocess.run([sys.executable, str(RUNPOD_DIR / "sdk_remote_bench.py"), "--smoke",  # noqa: S603
                        "--spec", "off,lookup,draft_head", "--stages", "setup,spec,conc,prefix",
                        "--conc", "1,2", "--tokens", "8", "--conc-tokens", "8", "--runs", "1", "--mask-store-gb", "1",
                        "--out", str(out)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads((out / "results.json").read_text())
    assert set(res["stages"]) == {"setup", "spec_off", "spec_lookup", "spec_draft_head", "conc_1", "conc_2", "prefix"}
    assert res["stages"]["spec_lookup"]["edit"]["bit_identical_to_off"]
    assert res["stages"]["spec_draft_head"]["chat"]["bit_identical_to_off"]
    assert res["stages"]["spec_draft_head"]["draft_head"]["source"] == "--draft-head"  # the smoke's tiny head
    assert res["stages"]["conc_2"]["bit_identical"] and res["stages"]["conc_2"]["errors"] == 0
    pre = res["stages"]["prefix"]
    assert pre["bit_identical"] and pre["masks_on"] < pre["masks_off"] and pre["prompt_rows_on"] < pre["prompt_rows_off"]


@pytest.mark.slow
def test_sdk_remote_bench_preview_smoke(tmp_path: Path) -> None:
    """``--preview`` with a pinned, unsupported model name (the tiny pack served under it): the run gets through the
    client step that used to raise ``UnsupportedModelError`` after the provider's download."""
    out = tmp_path / "sdkr"
    env = {k: v for k, v in os.environ.items() if k != "PGW_SDK_PREVIEW_MODELS"}
    env["PYTHONPATH"] = f"{PGW / 'src'}:{PGW}"
    r = subprocess.run([sys.executable, str(RUNPOD_DIR / "sdk_remote_bench.py"), "--smoke", "--preview",  # noqa: S603
                        "--model", "qwen2.5-32b-instruct", "--spec", "off", "--stages", "setup,conc", "--conc", "1",
                        "--tokens", "8", "--conc-tokens", "8", "--runs", "1", "--mask-store-gb", "1",
                        "--out", str(out)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads((out / "results.json").read_text())
    assert res["model"] == "qwen2.5-32b-instruct" and not res["errors"]
    assert res["stages"]["conc_1"]["bit_identical"] and any("preview" in n for n in res["notes"])


def test_sdk_steps_are_checked_before_any_pod() -> None:
    notes = jobs.check_sdk_steps(jobs.SESSIONS["sdk32b"]["steps"])
    assert any("preview" in n for n in notes)  # 32B: pinned, not supported
    assert any("shipped pinned head" in n for n in jobs.check_sdk_steps(jobs.SESSIONS["sdk7b"]["steps"]))
    with pytest.raises(SystemExit, match="no baked pin"):
        jobs.check_sdk_steps([{"kind": "sdk", "model": "qwen2.5-110b-instruct", "spec": "off"}])
    with pytest.raises(SystemExit, match="no shipped head"):
        jobs.check_sdk_steps([{"kind": "sdk", "model": "qwen2.5-32b-instruct", "spec": "off,draft_head"}])
    assert jobs.check_sdk_steps([{"kind": "sdk", "model": "qwen2.5-32b-instruct", "spec": "draft_head",
                                  "draft_head": "extras/h.pt"}])  # an --extra head overrides


def test_session_disks_fit_the_shardwise_pack() -> None:
    """Volumes sized for the shard-wise packer (the pack plus ~2 shards of scratch, not the whole checkpoint) and the
    client cache plus its mask stock (docs/sdk/scaling.md)."""
    pack_gb = {"qwen2.5-7b-instruct": 9, "qwen2.5-32b-instruct": 34}
    cache_gb = {"qwen2.5-7b-instruct": 13.3, "qwen2.5-32b-instruct": 44.8}
    for cfg in jobs.SESSIONS.values():
        for s in cfg["steps"]:
            if s["kind"] == "sdk":
                assert cfg["volume_gb"] >= pack_gb[s["model"]] + 9 + 10  # + scratch + bundle / native build / logs
                assert cfg["client_disk_gb"] >= cache_gb[s["model"]] + s["mask_store_gb"] + 10


def test_maskgen_progress_sampler_records_the_worst(tmp_path: Path) -> None:
    import http.server
    import threading

    views = iter([{"waiting": 1, "running": 2, "oldest_running_s": 0.4, "slowest_item_s": 0.3, "bound_s": 30.0,
                   "stalled": False, "done": 5, "failed": 0},
                  {"waiting": 0, "running": 1, "oldest_running_s": 1.5, "slowest_item_s": 0.9, "bound_s": 30.0,
                   "stalled": False, "done": 9, "failed": 0}])

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                body = json.dumps({"ready": True, "details": {"maskgen": "ok", "maskgen_progress": next(views)}})
            except StopIteration:
                body = json.dumps({"ready": True, "details": {}})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a: Any) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    stop = threading.Event()
    th = threading.Thread(target=jobs.maskgen_progress_sampler, args=(tmp_path / "mp", stop),
                          kwargs={"interval_s": 0.05, "url": f"http://127.0.0.1:{srv.server_address[1]}/readyz"})
    th.start()
    time.sleep(0.6)
    stop.set()
    th.join()
    srv.shutdown()
    worst = json.loads((tmp_path / "mp.json").read_text())
    assert worst["samples"] == 2 and worst["oldest_running_s"] == 1.5 and worst["slowest_item_s"] == 0.9
    assert worst["max_running"] == 2 and worst["done"] == 9 and worst["stalled_samples"] == 0
    assert len((tmp_path / "mp.jsonl").read_text().splitlines()) == 2


def test_factory_e2e_step_runs_on_and_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_JOB_STOP_AT", raising=False)
    res = tmp_path / "results"
    res.mkdir()
    rc = jobs.cpu_factory_e2e({"runs": [["0.5b", 64, "on"], ["0.5b", 64, "off"]]}, res, sys.executable)
    assert rc == 0, (res / "factory_e2e.log").read_text()[-3000:]
    lines = [json.loads(x) for x in (res / "factory_e2e.jsonl").read_text().splitlines()]
    runs = [x for x in lines if "layer_cpu_ms_per_slot" in x]
    assert len(runs) == 2 and runs[0]["wire_fused"] is True and runs[1]["wire_fused"] is False
    assert all(x["run"]["cpu_model"] for x in lines if "run" in x)
