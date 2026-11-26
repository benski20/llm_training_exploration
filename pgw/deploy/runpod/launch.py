"""Run the PGW provider (our GPU compute service) on a RunPod pod, with hard cost guards.

    export RUNPOD_API_KEY=...                       # never printed; only sent to RunPod's API
    export PGW_RUNPOD_IMAGE=ghcr.io/<owner>/pgw-provider:<tag>
    python deploy/runpod/launch.py --dc EU-RO-1 --gpu "NVIDIA L40S" --ttl-min 60 --max-usd 3 --dry-run
    python deploy/runpod/launch.py --dc EU-RO-1 --gpu "NVIDIA L40S" --ttl-min 60 --max-usd 3
    python deploy/runpod/launch.py --client-pod --ttl-min 60 --max-usd 1   # CPU pod in the SAME data center
    python deploy/runpod/launch.py --status
    python deploy/runpod/launch.py --stop            # stop both pods (GPU released); --terminate deletes them

A launch: estimate the cost (RunPod's GPU price, or the table below) and refuse above ``--max-usd``; create the
pod (Secure Cloud by default, the chosen GPU and data center, TCP ports 8443 (service) and 8444 (trip lane) on a
public IP, a volume for packs, the PKI and the accounts database); start a detached client-side watchdog that
stops the pod at its TTL; wait until the provider is ready; check that the endpoint holds the provider CA (an
HMAC attestation with the bootstrap admin token's hash, before the token is sent); create an organisation and an
SDK key through the admin API over TLS pinned to that CA; print the provider URL and the SDK key and keep them in
``~/.pgw/runpod/`` (0600).

Two independent TTL guards: the pod stops itself at ``PGW_RUNPOD_STOP_AT`` (``pod_boot.py``, with the pod-scoped
API key RunPod injects), and the watchdog started here stops it a few minutes later if it is still running.

The client library must run near the GPU: every token makes tens of sequential round trips, so put the client in
the same data center (``--client-pod``) and measure with ``rtt_probe.py`` first.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pod_boot  # noqa: E402  (stdlib only at import time)
from runpod_api import RunPodAPI, RunPodError  # noqa: E402

STATE_DIR = Path(os.environ.get("PGW_RUNPOD_STATE_DIR") or Path.home() / ".pgw" / "runpod")
STATE_FILE = "state.json"
#: GPU list prices, USD per GPU-hour (runpod.io/pricing, 2026-11; Secure / Community). Used only when the API does
#: not return a price; check before quoting. L4 is refused (as on Modal: a single-row masked GEMV is bandwidth
#: bound and an L4 is no faster than a T4).
GPU_PRICE_H: dict[str, tuple[float, float]] = {
    "NVIDIA H100 80GB HBM3": (2.99, 2.69),
    "NVIDIA H100 PCIe": (2.39, 1.99),
    "NVIDIA H100 NVL": (2.79, 2.59),
    "NVIDIA A100 80GB PCIe": (1.64, 1.19),
    "NVIDIA A100-SXM4-80GB": (1.74, 1.39),
    "NVIDIA L40S": (0.86, 0.79),
    "NVIDIA L40": (0.99, 0.69),
    "NVIDIA RTX A6000": (0.49, 0.33),
    "NVIDIA A40": (0.40, 0.35),
    "NVIDIA GeForce RTX 4090": (0.69, 0.34),
}
REFUSED_GPUS = ("NVIDIA L4",)
#: CPU pod price, USD per vCPU-hour (compute-optimised flavour; check before quoting).
CPU_VCPU_PRICE_H = 0.04
#: Volume storage, USD per GB-month: pod volume while running, network volume.
POD_VOLUME_GB_MONTH, NETWORK_VOLUME_GB_MONTH = 0.10, 0.07
#: Minutes the watchdog waits after the TTL before it stops the pod itself (the pod's own timer goes first).
WATCHDOG_GRACE_MIN = 3.0
DEFAULT_GPU = "NVIDIA L40S"
DEFAULT_CPU_FLAVOR = "cpu5c"
CONTAINER_DISK_GB = 30
CLIENT_CONTAINER_DISK_GB = 20


# ------------------------------------------------------------------------------------------------- state
def state_path() -> Path:
    return STATE_DIR / STATE_FILE


def load_state() -> dict[str, Any]:
    try:
        return dict(json.loads(state_path().read_text()))
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STATE_DIR, 0o700)
    tmp = state_path().with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, state_path())


def write_secret(name: str, text: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    p = STATE_DIR / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text + "\n")
    os.replace(tmp, p)
    return p


# -------------------------------------------------------------------------------------------------- cost
@dataclass
class Estimate:
    gpu_price_h: float
    price_source: str
    hours: float
    gpu_usd: float
    storage_usd: float
    total_usd: float

    def lines(self) -> list[str]:
        return [f"GPU {self.gpu_price_h:.2f} $/h ({self.price_source}) x {self.hours:.2f} h = ${self.gpu_usd:.2f}",
                f"storage ${self.storage_usd:.2f}", f"estimated total ${self.total_usd:.2f}"]


def gpu_price(gpu: str, *, community: bool, gpu_count: int, api: RunPodAPI | None,
              override: float | None = None) -> tuple[float, str]:
    """USD per hour for ``gpu_count`` GPUs and its source (override, API, table)."""
    if override is not None:
        return override * gpu_count, "--price-per-hour"
    if api is not None:
        try:
            row = api.gpu_prices().get(gpu)
            p = row and row.get("communityPrice" if community else "securePrice")
            if isinstance(p, int | float) and p > 0:
                return float(p) * gpu_count, "RunPod API"
        except RunPodError as exc:
            print(f"note: no price from the API ({exc}); using the table", file=sys.stderr)
    if gpu in GPU_PRICE_H:
        return GPU_PRICE_H[gpu][1 if community else 0] * gpu_count, "table (2026-11)"
    raise SystemExit(f"no known price for {gpu!r}: pass --price-per-hour (USD per GPU-hour) so the cost guard holds")


def estimate(price_h: float, source: str, ttl_min: float, *, volume_gb: int, network_volume: bool) -> Estimate:
    hours = ttl_min / 60.0 + WATCHDOG_GRACE_MIN / 60.0  # the worst case: the watchdog stops it, not the pod
    gb_month = NETWORK_VOLUME_GB_MONTH if network_volume else POD_VOLUME_GB_MONTH
    storage = (volume_gb + (0 if network_volume else CONTAINER_DISK_GB)) * gb_month * hours / 730.0
    gpu_usd = price_h * hours
    return Estimate(price_h, source, hours, gpu_usd, storage, gpu_usd + storage)


def check_budget(est: Estimate, max_usd: float, ttl_min: float) -> None:
    if est.total_usd > max_usd:
        afford = max(0.0, (max_usd / (est.total_usd / est.hours)) * 60.0 - WATCHDOG_GRACE_MIN)
        raise SystemExit(f"refused: estimated ${est.total_usd:.2f} for {ttl_min:.0f} min exceeds --max-usd "
                         f"{max_usd:.2f} (at most about {afford:.0f} min fit)")


# ------------------------------------------------------------------------------------------------- specs
def pod_spec(*, name: str, image: str, gpu: str, gpu_count: int, dc: str, community: bool, stop_at: float,
             models: list[str], admin_hash: str, volume_gb: int, network_volume_id: str | None,
             service_env: dict[str, str] | None, registry_auth_id: str | None, key_env: str,
             public_host: str | None = None) -> dict[str, Any]:
    """REST ``POST /pods`` body of the provider pod."""
    env = {"PGW_RUNPOD_MODELS": ",".join(models), "PGW_RUNPOD_STOP_AT": str(int(stop_at)),
           "PGW_RUNPOD_ADMIN_TOKEN_SHA256": admin_hash, "PGW_RUNPOD_DATA": "/workspace/pgw",
           "PGW_RUNPOD_KEY_ENV": key_env}
    if public_host:
        env["PGW_RUNPOD_PUBLIC_HOST"] = public_host
    if service_env:
        env["PGW_RUNPOD_SERVICE_ENV"] = json.dumps(service_env, sort_keys=True)
    spec: dict[str, Any] = {
        "name": name, "imageName": image, "computeType": "GPU", "gpuTypeIds": [gpu], "gpuCount": gpu_count,
        "cloudType": "COMMUNITY" if community else "SECURE", "dataCenterIds": [dc],
        "containerDiskInGb": CONTAINER_DISK_GB, "volumeMountPath": "/workspace",
        "ports": [f"{pod_boot.SERVICE_PORT}/tcp", f"{pod_boot.LANE_PORT}/tcp"], "supportPublicIp": True,
        "dockerStartCmd": ["pgw-runpod-boot"], "env": env, "interruptible": False,
    }
    if network_volume_id:
        spec["networkVolumeId"] = network_volume_id
    else:
        spec["volumeInGb"] = volume_gb
    if registry_auth_id:
        spec["containerRegistryAuthId"] = registry_auth_id
    return spec


def client_pod_spec(*, name: str, image: str, dc: str, flavor: str, vcpus: int, stop_at: float,
                    registry_auth_id: str | None, env_extra: dict[str, str] | None = None) -> dict[str, Any]:
    """REST ``POST /pods`` body of the CPU test pod in the provider's data center."""
    spec: dict[str, Any] = {
        "name": name, "imageName": image, "computeType": "CPU", "cpuFlavorIds": [flavor], "vcpuCount": vcpus,
        "dataCenterIds": [dc], "containerDiskInGb": CLIENT_CONTAINER_DISK_GB, "ports": ["22/tcp"],
        "supportPublicIp": True, "dockerStartCmd": ["pgw-runpod-boot", "client"],
        "env": {"PGW_RUNPOD_STOP_AT": str(int(stop_at)), **(env_extra or {})},
    }
    if registry_auth_id:
        spec["containerRegistryAuthId"] = registry_auth_id
    return spec


def redacted(spec: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(spec))
    for k in list(out.get("env", {})):
        if k in ("PGW_RUNPOD_ADMIN_TOKEN_SHA256", "PGW_API_KEY"):
            out["env"][k] = "<hidden>"
    return out


# ------------------------------------------------------------------------------------------------ waiting
def wait_for_address(api: RunPodAPI, pod_id: str, *, timeout_s: float, sleep: Callable[[float], None] = time.sleep,
                     clock: Callable[[], float] = time.time) -> tuple[str, int, int | None]:
    """``(public ip, public service port, public lane port)`` once RunPod reports the port mapping."""
    t0 = clock()
    while clock() - t0 < timeout_s:
        pod = api.get_pod(pod_id)
        status = str(pod.get("desiredStatus") or "")
        if status in ("EXITED", "TERMINATED"):
            raise SystemExit(f"pod {pod_id} is {status} before it got an address")
        ip = pod.get("publicIp")
        ports = pod.get("portMappings") or {}
        svc = ports.get(str(pod_boot.SERVICE_PORT))
        if ip and svc:
            lane = ports.get(str(pod_boot.LANE_PORT))
            return str(ip), int(svc), int(lane) if lane else None
        sleep(5.0)
    raise SystemExit(f"pod {pod_id} got no public address within {timeout_s / 60:.0f} min")


def wait_for_ready(url: str, *, timeout_s: float, http_get: Callable[[str], int] | None = None,
                   sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time) -> None:
    """Poll ``/readyz`` (unverified TLS: a liveness signal only; nothing is sent and nothing trusted from it)."""
    if http_get is None:
        import httpx

        def http_get(u: str) -> int:
            return httpx.get(u, verify=False, timeout=10.0, trust_env=False).status_code  # noqa: S501

    t0 = clock()
    last = ""
    while clock() - t0 < timeout_s:
        try:
            code = http_get(f"{url}/readyz")
            if code == 200:
                return
            last = f"HTTP {code}"
        except Exception as exc:
            last = type(exc).__name__
        sleep(10.0)
        mins = (clock() - t0) / 60
        print(f"  waiting for the provider ({last}, {mins:.1f} min; the first boot packs the models)", flush=True)
    raise SystemExit(f"the provider at {url} was not ready within {timeout_s / 60:.0f} min ({last})")


# ---------------------------------------------------------------------------------------------- bootstrap
def attested_ca(url: str, admin_token: str, *, client: Any = None) -> str:
    """The provider CA (PEM), accepted only if the endpoint proves knowledge of ``SHA-256(admin_token)`` for that
    CA (``POST /pgw/v1/admin/attest``; nothing secret is sent). Raises SystemExit on any mismatch."""
    import httpx

    from pgw.core.errors import ConfigError
    from pgw.enroll.service import pinned_ca
    from pgw.service.accounts import attest_mac

    h = client or httpx.Client(verify=False, timeout=30.0, trust_env=False)  # noqa: S501 - checked below
    try:
        nonce = secrets.token_bytes(32)
        r = h.post(f"{url}/pgw/v1/admin/attest", json={"nonce": nonce.hex()})
        if r.status_code != 200:
            raise SystemExit(f"attestation failed: HTTP {r.status_code} (is the admin API on?)")
        a = r.json()
        ca_hex = str(a.get("ca_sha256", ""))
        digest = hashlib.sha256(admin_token.encode()).digest()
        want = attest_mac(digest, nonce, bytes.fromhex(ca_hex)).hex()
        if not any(hmac.compare_digest(want, str(m)) for m in a.get("macs") or []):
            raise SystemExit("attestation failed: the endpoint does not know this launch's admin token (an "
                             "interception, or another pod). Nothing secret was sent; stop the pod and check")
        try:  # exactly one certificate, and only it is trusted (nothing an interceptor appends)
            fp, pem = pinned_ca(h.get(f"{url}/pgw/v1/enroll/ca").text)
        except ConfigError as exc:
            raise SystemExit(f"attestation failed: invalid CA reply ({exc.message})") from exc
        if not hmac.compare_digest(fp.hex(), ca_hex):
            raise SystemExit("attestation failed: the CA served does not match the attested fingerprint")
        return pem
    finally:
        if client is None:
            h.close()


def bootstrap_sdk_key(url: str, admin_token: str, *, org: str, key_name: str, models: list[str],
                      client_factory: Callable[[str], Any] | None = None, attest_client: Any = None,
                      ) -> tuple[str, str, dict[str, Any]]:
    """Attest the CA, then (TLS pinned to it) create ``org`` if missing and an SDK key. Returns
    ``(sdk key, CA PEM, key record)``."""
    import ssl

    import httpx

    ca_pem = attested_ca(url, admin_token, client=attest_client)
    if client_factory is not None:
        h = client_factory(ca_pem)
    else:
        h = httpx.Client(verify=ssl.create_default_context(cadata=ca_pem), timeout=30.0, trust_env=False)
    hdr = {"authorization": f"Bearer {admin_token}"}
    try:
        r = h.get(f"{url}/pgw/v1/admin/orgs/{org}", headers=hdr)
        if r.status_code == 404:
            r = h.post(f"{url}/pgw/v1/admin/orgs", json={"org_id": org}, headers=hdr)
        if r.status_code != 200:
            raise SystemExit(f"cannot create organisation {org!r}: HTTP {r.status_code} {r.text[:200]}")
        r = h.post(f"{url}/pgw/v1/admin/orgs/{org}/keys", json={"name": key_name, "models": models}, headers=hdr)
        if r.status_code != 200:
            raise SystemExit(f"cannot create an SDK key: HTTP {r.status_code} {r.text[:200]}")
        d = r.json()
        return str(d["key"]), ca_pem, dict(d["record"])
    finally:
        h.close()


# ----------------------------------------------------------------------------------------------- watchdog
def spawn_watchdog(pod_id: str, at: float, *, terminate: bool) -> int:
    """Detached ``launch.py --watchdog``: survives this shell; stops (or terminates) the pod at ``at`` + grace."""
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    logf = open(STATE_DIR / f"watchdog-{pod_id}.log", "ab")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--watchdog", pod_id, "--at", str(int(at))]
    if terminate:
        cmd.append("--terminate")
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=logf, start_new_session=True,  # noqa: S603
                         env={**os.environ})
    return int(p.pid)


def run_watchdog(api: RunPodAPI, pod_id: str, at: float, *, terminate: bool = False,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time) -> str:
    """Sleep until ``at`` + grace; then stop/terminate the pod if it is still up. Returns what it did."""
    deadline = at + WATCHDOG_GRACE_MIN * 60.0
    while (left := deadline - clock()) > 0:
        sleep(min(left, 60.0))
    for attempt in range(10):
        try:
            pod = api.get_pod(pod_id)
            status = str(pod.get("desiredStatus") or "")
            if status in ("EXITED", "TERMINATED") and not terminate:
                print(f"{time.ctime()}: pod {pod_id} already {status}", flush=True)
                return "already-stopped"
            (api.terminate_pod if terminate else api.stop_pod)(pod_id)
            print(f"{time.ctime()}: watchdog {'terminated' if terminate else 'stopped'} pod {pod_id}", flush=True)
            return "terminated" if terminate else "stopped"
        except RunPodError as exc:
            if exc.status == 404:
                return "gone"
            print(f"{time.ctime()}: watchdog attempt {attempt + 1} failed: {exc}", flush=True)
            sleep(min(300.0, 15.0 * 2 ** attempt))
    return "failed"


def kill_watchdog(pid: int | None) -> None:
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass


# --------------------------------------------------------------------------------------------- commands
def _print_cost(est: Estimate, max_usd: float) -> None:
    for line in est.lines():
        print(f"  {line}")
    print(f"  hard cap --max-usd {max_usd:.2f}: OK")


def cmd_launch(args: argparse.Namespace, api: RunPodAPI | None) -> int:
    if any(args.gpu.startswith(g) and args.gpu.strip() == g for g in REFUSED_GPUS):
        raise SystemExit(f"{args.gpu} is refused for the provider (bandwidth-bound decode; see the README)")
    models = pod_boot.parse_models(args.models)
    image = args.image or os.environ.get("PGW_RUNPOD_IMAGE") or ""
    if not image and not args.dry_run:
        raise SystemExit("no image: pass --image or set PGW_RUNPOD_IMAGE (ghcr.io/<owner>/pgw-provider:<tag>)")
    st = load_state()
    if st.get("provider", {}).get("pod_id") and not args.dry_run and not args.force:
        raise SystemExit(f"a provider pod is recorded ({st['provider']['pod_id']}): --status, --stop, or --force")
    price_h, src = gpu_price(args.gpu, community=args.community, gpu_count=args.gpu_count, api=api,
                             override=args.price_per_hour)
    est = estimate(price_h, src, args.ttl_min, volume_gb=args.volume_gb, network_volume=bool(args.network_volume))
    print(f"cost estimate for {args.gpu_count} x {args.gpu} ({'Community' if args.community else 'Secure'} Cloud, "
          f"{args.dc}), TTL {args.ttl_min:.0f} min:")
    check_budget(est, args.max_usd, args.ttl_min)
    _print_cost(est, args.max_usd)
    stop_at = time.time() + args.ttl_min * 60.0
    admin_token = "pgwa-" + secrets.token_urlsafe(32)
    admin_hash = hashlib.sha256(admin_token.encode()).hexdigest()
    name = args.name or f"pgw-provider-{time.strftime('%m%d-%H%M')}"
    service_env = {}
    for kv in args.service_env or []:
        k, sep, v = kv.partition("=")
        if not sep:
            raise SystemExit(f"--service-env {kv!r}: use NAME=VALUE")
        service_env[k] = v
    pod_boot.checked_service_env(json.dumps(service_env))  # refuse what the pod would refuse, before paying
    nv_id = None
    if args.network_volume and args.dry_run:
        nv_id = f"<network volume {args.network_volume}>"
    elif args.network_volume:
        assert api is not None
        nv_id = ensure_network_volume(api, args.network_volume, args.volume_gb, args.dc)
    spec = pod_spec(name=name, image=image or "<PGW_RUNPOD_IMAGE>", gpu=args.gpu, gpu_count=args.gpu_count,
                    dc=args.dc, community=args.community, stop_at=stop_at, models=models, admin_hash=admin_hash,
                    volume_gb=args.volume_gb, network_volume_id=nv_id, service_env=service_env,
                    registry_auth_id=args.registry_auth_id, key_env=args.key_env, public_host=args.public_host)
    if args.dry_run:
        print(json.dumps({"create_pod": redacted(spec), "watchdog_at": int(stop_at + WATCHDOG_GRACE_MIN * 60)},
                         indent=1))
        print("dry run: nothing created, nothing spent")
        return 0
    assert api is not None
    pod = api.create_pod(spec)
    pod_id = str(pod["id"])
    wd = spawn_watchdog(pod_id, stop_at, terminate=args.terminate_at_ttl)
    admin_file = write_secret(f"admin-token-{pod_id}", admin_token)
    st["provider"] = {"pod_id": pod_id, "name": name, "gpu": args.gpu, "gpu_count": args.gpu_count, "dc": args.dc,
                      "cloud": "COMMUNITY" if args.community else "SECURE", "price_h": price_h, "stop_at": stop_at,
                      "started_at": time.time(), "watchdog_pid": wd, "admin_token_file": str(admin_file),
                      "models": models, "max_usd": args.max_usd}
    save_state(st)
    print(f"pod {pod_id} created; client-side watchdog pid {wd} stops it at "
          f"{time.strftime('%H:%M:%S', time.localtime(stop_at + WATCHDOG_GRACE_MIN * 60))} if the pod has not")
    try:
        host, port, lane = wait_for_address(api, pod_id, timeout_s=args.start_timeout_min * 60)
        url = pod_boot.url_for(args.public_host or host, port)
        st["provider"].update(url=url, lane_port=lane)
        save_state(st)
        print(f"public address {url} (lane port {lane}); waiting for the provider", flush=True)
        wait_for_ready(url, timeout_s=args.ready_timeout_min * 60)
        key, ca_pem, rec = bootstrap_sdk_key(url, admin_token, org=args.org, key_name=args.key_name or name,
                                             models=models)
    except BaseException:
        print(f"launch did not finish; the pod keeps running until its TTL. Stop it now: "
              f"python {Path(__file__).name} --stop", file=sys.stderr)
        raise
    from pgw.enroll.service import ca_fingerprint

    key_file = write_secret(f"sdk-key-{pod_id}", key)
    (STATE_DIR / f"provider-ca-{pod_id}.pem").write_text(ca_pem)
    st["provider"].update(ca_sha256=ca_fingerprint(ca_pem).hex(), sdk_key_file=str(key_file),
                          sdk_key_id=rec["key_id"], org=args.org)
    save_state(st)
    print("\nprovider ready")
    print(f"  provider URL   {url}")
    print(f"  CA sha256      {st['provider']['ca_sha256']}")
    print(f"  SDK key id     {rec['key_id']} (org {args.org}, models {', '.join(rec['models']) or 'all'})")
    print(f"  SDK key file   {key_file} (0600)")
    if args.print_key:
        print(f"  SDK key        {key}")
    print(f"  stops at       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stop_at))} (pod) / +"
          f"{WATCHDOG_GRACE_MIN:.0f} min (watchdog)")
    print(f"next: python {Path(__file__).name} --client-pod --ttl-min {args.ttl_min:.0f} --max-usd 1   "
          "(CPU pod in the same data center)")
    return 0


def ensure_network_volume(api: RunPodAPI, name_or_id: str, size_gb: int, dc: str) -> str:
    for v in api.list_network_volumes():
        if name_or_id in (v.get("id"), v.get("name")):
            if v.get("dataCenterId") and v["dataCenterId"] != dc:
                raise SystemExit(f"network volume {name_or_id} is in {v['dataCenterId']}, not {dc}")
            return str(v["id"])
    v = api.create_network_volume(name_or_id, size_gb, dc)
    print(f"created network volume {v['id']} ({size_gb} GB, {dc})")
    return str(v["id"])


def cmd_client_pod(args: argparse.Namespace, api: RunPodAPI | None) -> int:
    st = load_state()
    prov = st.get("provider") or {}
    dc = args.dc or prov.get("dc")
    if not dc:
        raise SystemExit("no data center: launch the provider first or pass --dc")
    image = args.client_image or os.environ.get("PGW_RUNPOD_CLIENT_IMAGE") or ""
    if not image and not args.dry_run:
        raise SystemExit("no client image: pass --client-image or set PGW_RUNPOD_CLIENT_IMAGE "
                         "(ghcr.io/<owner>/pgw-provider-cpu:<tag>)")
    price_h = (args.price_per_hour or CPU_VCPU_PRICE_H * args.vcpus)
    est = estimate(price_h, "--price-per-hour" if args.price_per_hour else "table", args.ttl_min,
                   volume_gb=0, network_volume=False)
    print(f"client pod: {args.vcpus} vCPU {args.cpu_flavor} in {dc}, TTL {args.ttl_min:.0f} min")
    check_budget(est, args.max_usd, args.ttl_min)
    _print_cost(est, args.max_usd)
    stop_at = time.time() + args.ttl_min * 60.0
    extra: dict[str, str] = {}
    if prov.get("url"):
        extra["PGW_PROVIDER_URL"] = prov["url"]
        extra["PGW_PROVIDER_CA_SHA256"] = prov.get("ca_sha256", "")
    if args.pass_key and prov.get("sdk_key_file"):
        extra["PGW_API_KEY"] = Path(prov["sdk_key_file"]).read_text().strip()
    spec = client_pod_spec(name=f"pgw-client-{time.strftime('%m%d-%H%M')}", image=image or "<client image>", dc=dc,
                           flavor=args.cpu_flavor, vcpus=args.vcpus, stop_at=stop_at,
                           registry_auth_id=args.registry_auth_id, env_extra=extra)
    if args.dry_run:
        print(json.dumps({"create_pod": redacted(spec)}, indent=1))
        print("dry run: nothing created, nothing spent")
        return 0
    assert api is not None
    pod = api.create_pod(spec)
    pod_id = str(pod["id"])
    wd = spawn_watchdog(pod_id, stop_at, terminate=args.terminate_at_ttl)
    st["client"] = {"pod_id": pod_id, "dc": dc, "stop_at": stop_at, "watchdog_pid": wd, "price_h": price_h}
    save_state(st)
    print(f"client pod {pod_id} created in {dc} (watchdog pid {wd}). Connect with the RunPod web terminal or SSH, "
          "then:\n  python /opt/pgw-deploy/runpod/rtt_probe.py \"$PGW_PROVIDER_URL\" --ca-sha256 "
          "\"$PGW_PROVIDER_CA_SHA256\"")
    if not args.pass_key:
        print(f"  (the SDK key is not in the pod; paste it from {prov.get('sdk_key_file', '~/.pgw/runpod')}, or "
              "relaunch with --pass-key)")
    return 0


def cmd_status(api: RunPodAPI | None) -> int:
    st = load_state()
    if not st:
        print("no pods recorded")
        return 0
    for role in ("provider", "client"):
        p = st.get(role)
        if not p:
            continue
        line = f"{role}: pod {p['pod_id']} ({p.get('gpu') or 'CPU'}, {p.get('dc')})"
        live: dict[str, Any] = {}
        if api is not None:
            try:
                live = api.get_pod(p["pod_id"])
            except RunPodError as exc:
                line += f"  [API: {exc}]"
        hours = (time.time() - p.get("started_at", time.time())) / 3600
        cost = live.get("costPerHr") or p.get("price_h") or 0.0
        left = (p.get("stop_at", 0) - time.time()) / 60
        print(f"{line}  status {live.get('desiredStatus', '?')}  {cost} $/h  up {hours:.2f} h "
              f"(~${float(cost) * hours:.2f})  TTL {f'in {left:.0f} min' if left > 0 else 'reached'}")
        if role == "provider" and p.get("url"):
            print(f"  URL {p['url']}  CA sha256 {p.get('ca_sha256', '?')[:16]}...  key {p.get('sdk_key_id', '?')} "
                  f"in {p.get('sdk_key_file', '?')}")
    return 0


def cmd_stop(api: RunPodAPI, *, terminate: bool) -> int:
    st = load_state()
    rc = 0
    for role in ("client", "provider"):
        p = st.get(role)
        if not p:
            continue
        try:
            (api.terminate_pod if terminate else api.stop_pod)(p["pod_id"])
            print(f"{role} pod {p['pod_id']} {'terminated' if terminate else 'stopped'}")
        except RunPodError as exc:
            if exc.status == 404:
                print(f"{role} pod {p['pod_id']} is gone")
            else:
                print(f"could not stop {role} pod {p['pod_id']}: {exc}", file=sys.stderr)
                rc = 1
                continue
        kill_watchdog(p.get("watchdog_pid"))
        if terminate or role == "client":
            st.pop(role, None)
        else:
            p["stopped_at"] = time.time()
    save_state(st)
    return rc


# -------------------------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="show the recorded pods (live status and cost so far)")
    mode.add_argument("--stop", action="store_true", help="stop the recorded pods (GPU released; volume kept)")
    mode.add_argument("--client-pod", action="store_true", help="a CPU pod in the provider's data center")
    mode.add_argument("--watchdog", metavar="POD_ID", help=argparse.SUPPRESS)
    ap.add_argument("--at", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--terminate", action="store_true", help="with --stop: delete the pods (pod volumes too)")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and the exact API payload only")
    ap.add_argument("--gpu", default=DEFAULT_GPU, help=f"RunPod GPU type id (default {DEFAULT_GPU!r}; latency tier: "
                    "'NVIDIA H100 80GB HBM3')")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--dc", default=None, help="data center id, e.g. EU-RO-1, US-KS-2 (required for a launch)")
    ap.add_argument("--community", action="store_true", help="Community Cloud (default: Secure Cloud)")
    ap.add_argument("--ttl-min", type=float, default=60.0, help="lifetime in minutes (default 60)")
    ap.add_argument("--max-usd", type=float, default=None, help="hard cost cap of this launch (required)")
    ap.add_argument("--price-per-hour", type=float, default=None, help="USD per GPU-hour (or per client pod hour), "
                    "when RunPod's price is unknown")
    ap.add_argument("--models", default="qwen2.5-0.5b-instruct",
                    help=f"comma-separated targets to serve (packed on first boot); of {', '.join(pod_boot.MODELS)}")
    ap.add_argument("--image", default=None, help="provider image (default PGW_RUNPOD_IMAGE)")
    ap.add_argument("--registry-auth-id", default=None, help="RunPod container registry auth id (private GHCR)")
    ap.add_argument("--volume-gb", type=int, default=120, help="volume for packs, PKI, accounts (default 120)")
    ap.add_argument("--network-volume", default=None, metavar="NAME_OR_ID",
                    help="use (or create) a network volume in --dc instead of a pod volume (survives the pod)")
    ap.add_argument("--service-env", action="append", metavar="NAME=VALUE",
                    help="extra PGW_SERVICE_* setting for this run (repeatable), e.g. "
                    "PGW_SERVICE_LANE_READ_SPIN_US=2000")
    ap.add_argument("--org", default="pgw-test", help="organisation of the SDK key (created if missing)")
    ap.add_argument("--key-name", default=None, help="SDK key name (default: the pod name)")
    ap.add_argument("--key-env", default="test", help="SDK key env label (pgw_<env>_...; default test)")
    ap.add_argument("--print-key", action="store_true", help="also print the SDK key (default: file only)")
    ap.add_argument("--public-host", default=None, help="DNS name that points at the pod's public IP: used in the "
                    "server certificate and the SDK key instead of the IP")
    ap.add_argument("--name", default=None)
    ap.add_argument("--force", action="store_true", help="launch although a provider pod is recorded")
    ap.add_argument("--terminate-at-ttl", action="store_true", help="the watchdog terminates instead of stopping")
    ap.add_argument("--start-timeout-min", type=float, default=15.0)
    ap.add_argument("--ready-timeout-min", type=float, default=45.0)
    ap.add_argument("--client-image", default=None, help="client pod image (default PGW_RUNPOD_CLIENT_IMAGE)")
    ap.add_argument("--cpu-flavor", default=DEFAULT_CPU_FLAVOR)
    ap.add_argument("--vcpus", type=int, default=8)
    ap.add_argument("--pass-key", action="store_true", help="--client-pod: put the SDK key into the pod's env "
                    "(PGW_API_KEY; visible in the RunPod console of this account)")
    return ap


def main(argv: list[str] | None = None, *, api_factory: Callable[[], RunPodAPI] = RunPodAPI) -> int:
    args = build_parser().parse_args(argv)
    if args.watchdog:
        return 0 if run_watchdog(api_factory(), args.watchdog, float(args.at or 0), terminate=args.terminate) \
            in ("stopped", "terminated", "already-stopped", "gone") else 1
    if args.status:
        try:
            api: RunPodAPI | None = api_factory()
        except RunPodError:
            api = None
        return cmd_status(api)
    if args.stop:
        return cmd_stop(api_factory(), terminate=args.terminate)
    if args.max_usd is None:
        raise SystemExit("--max-usd is required: the hard cost cap of this launch")
    if args.ttl_min <= 0 or args.ttl_min > 24 * 60:
        raise SystemExit("--ttl-min must be in (0, 1440]")
    api = None
    if not args.dry_run:
        api = api_factory()
    elif os.environ.get("RUNPOD_API_KEY"):
        try:
            api = api_factory()  # prices only; a dry run creates nothing
        except RunPodError:
            api = None
    if args.client_pod:
        return cmd_client_pod(args, api)
    if not args.dc:
        raise SystemExit("--dc is required (the client pod, and a network volume, must be in the same data center)")
    return cmd_launch(args, api)


if __name__ == "__main__":
    sys.exit(main())
