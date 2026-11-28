#!/usr/bin/env python3
"""Dynamic GPU pods on RunPod in the data centres nearest the users (docs/operations/placement.md; RUNBOOK
section 3.9). The policy is :mod:`pgw.placement.policy`; this script turns its plans into RunPod API calls.

    python deploy/runpod/placement.py plan --demand demand.json --warm new-york=1,london=1     # no API call
    python deploy/runpod/placement.py stage-packs --dcs US-DE-1,EU-FR-1 --models qwen2.5-7b-instruct --dry-run
    python deploy/runpod/placement.py apply --demand demand.json --warm new-york=1 --dry-run   # exact payloads
    python deploy/runpod/placement.py coldstart --model qwen2.5-7b-instruct                     # estimate

``demand.json``: ``{"<metro>": {"<model>": peak concurrent streams}}`` or the controller's aggregate export
(``GET /pgw/v1/placement/aggregates``). Nothing here runs without ``--dry-run`` unless ``RUNPOD_API_KEY`` is set,
and every live path keeps the same guards as ``launch.py``: Secure Cloud, the policy's per-hour / per-day / total
caps (checked before every create), a TTL on every pod that the pod enforces itself (``PGW_RUNPOD_STOP_AT``), and
``--max-pods``.

Pod life cycle (Actuator): ``create`` (a one-time registration token in the pod's env, the data centre's pack
volume mounted, ``pgw-runpod-boot placement``) -> the pod verifies its pack and registers (pgw.placement.server)
-> listed in the signed directory -> ``drain`` (listed as draining; clients move between requests) -> after
``drain_grace_s`` (two client refresh intervals) ``terminate``. A create refused for capacity holds that data
centre out (the policy falls back to the next-best one).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch  # noqa: E402  (stdlib only at import time)
import pod_boot  # noqa: E402
from runpod_api import RunPodAPI, RunPodError  # noqa: E402

if (HERE.parents[1] / "src" / "pgw" / "placement").is_dir():  # this checkout's pgw first (an installed one may lag)
    sys.path.insert(0, str(HERE.parents[1] / "src"))

POD_PREFIX = "pgw-place-"
PACKER_PREFIX = "pgw-packer-"
#: Every placement pod stops itself after this many hours (the backstop if the controller dies).
POD_TTL_H = 12.0
#: Time a drained pod stays up before it is terminated: two client directory refreshes (routing_refresh_s 300).
DRAIN_GRACE_S = 600.0
PACKER_VCPUS = 8
PACKER_TTL_MIN = 90.0
#: RunPod error texts that mean "no capacity there right now" (the policy then holds the data centre out).
CAPACITY_HINTS = ("no available", "not enough", "insufficient", "capacity", "no gpu", "unavailable")


def placement_pod_spec(*, name: str, image: str, gpu: str, dc: str, model: str, volume_id: str, token: str,
                       controller_url: str, controller_ca_pem: bytes, stop_at: float,
                       registry_auth_id: str | None = None) -> dict[str, Any]:
    """REST ``POST /pods`` body of a placement pod (GPU, Secure Cloud, the data centre's pack volume)."""
    env = {"PGW_RUNPOD_MODELS": model, "PGW_RUNPOD_STOP_AT": str(int(stop_at)), "PGW_RUNPOD_DATA": "/pgw-data",
           "PGW_PLACEMENT_PACKS": "/workspace/packs", "PGW_PLACEMENT_TOKEN": token, "PGW_PLACEMENT_DC": dc,
           "PGW_PLACEMENT_POD_NAME": name, "PGW_PLACEMENT_CONTROLLER": controller_url,
           "PGW_PLACEMENT_CONTROLLER_CA_PEM_B64": base64.b64encode(controller_ca_pem).decode()}
    spec: dict[str, Any] = {
        "name": name, "imageName": image, "computeType": "GPU", "gpuTypeIds": [gpu], "gpuCount": 1,
        "cloudType": "SECURE", "dataCenterIds": [dc], "networkVolumeId": volume_id, "volumeMountPath": "/workspace",
        "containerDiskInGb": launch.CONTAINER_DISK_GB,
        "ports": [f"{pod_boot.SERVICE_PORT}/tcp", f"{pod_boot.LANE_PORT}/tcp"], "supportPublicIp": True,
        "dockerStartCmd": ["pgw-runpod-boot", "placement"], "env": env, "interruptible": False,
    }
    if registry_auth_id:
        spec["containerRegistryAuthId"] = registry_auth_id
    return spec


def packer_pod_spec(*, name: str, image: str, dc: str, volume_id: str, models: list[str], stop_at: float,
                    vcpus: int = PACKER_VCPUS, registry_auth_id: str | None = None) -> dict[str, Any]:
    """REST ``POST /pods`` body of the CPU packer pod that fills ``dc``'s pack volume (``stage-packs``)."""
    spec: dict[str, Any] = {
        "name": name, "imageName": image, "computeType": "CPU", "cpuFlavorIds": [launch.DEFAULT_CPU_FLAVOR],
        "vcpuCount": vcpus, "dataCenterIds": [dc], "networkVolumeId": volume_id, "volumeMountPath": "/workspace",
        "containerDiskInGb": launch.CLIENT_CONTAINER_DISK_GB, "supportPublicIp": False,
        "dockerStartCmd": ["pgw-runpod-boot", "stage-packs"],
        "env": {"PGW_RUNPOD_MODELS": ",".join(models), "PGW_RUNPOD_STOP_AT": str(int(stop_at)),
                "PGW_PLACEMENT_PACKS": "/workspace/packs"},
    }
    if registry_auth_id:
        spec["containerRegistryAuthId"] = registry_auth_id
    return spec


def redacted(spec: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(spec))
    if "PGW_PLACEMENT_TOKEN" in out.get("env", {}):
        out["env"]["PGW_PLACEMENT_TOKEN"] = "<one-time registration token>"  # noqa: S105 - a placeholder
    return out


def capacity_error(exc: RunPodError) -> bool:
    return any(h in str(exc).lower() for h in CAPACITY_HINTS)


@dataclass
class Actuator:
    """Applies plans (create / drain / terminate) through the RunPod API, or records the payloads (``dry_run``)."""

    api: Any
    controller: Any  # pgw.placement.policy.Controller
    image: str
    volumes: Mapping[str, str]  # dc -> network volume id (pre-staged packs)
    controller_url: str
    controller_ca_pem: bytes
    expect: Callable[[str, str, str], str] | None = None  # registry.expect(pod name, dc, model) -> token
    drain: Callable[[str], None] | None = None  # registry.drain(pod id)
    remove: Callable[[str], None] | None = None  # registry.remove(pod id)
    dry_run: bool = True
    max_pods: int = 16
    ttl_h: float = POD_TTL_H
    drain_grace_s: float = DRAIN_GRACE_S
    clock: Callable[[], float] = time.time
    pods: dict[str, Any] = field(default_factory=dict)  # pod id -> PodState
    draining_since: dict[str, float] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)
    _n: int = 0

    def fleet(self) -> list[Any]:
        return list(self.pods.values())

    def apply(self, plan: Any) -> list[dict[str, Any]]:
        from pgw.placement.policy import PodState

        now = self.clock()
        out: list[dict[str, Any]] = []
        gpu = self.controller.cfg.gpu
        for act, dc, model, n in plan.actions:
            if act == "create":
                for _ in range(n):
                    live = sum(1 for p in self.pods.values() if p.state != "draining")
                    if live >= self.max_pods:
                        out.append({"refused": f"{dc}/{model}: --max-pods {self.max_pods} reached"})
                        break
                    vol = self.volumes.get(dc)
                    if vol is None:
                        out.append({"refused": f"{dc}/{model}: no pre-staged pack volume in {dc} (stage-packs)"})
                        break
                    self._n += 1
                    name = f"{POD_PREFIX}{dc.lower()}-{int(now)}-{self._n}"
                    token = self.expect(name, dc, model) if self.expect else "<token>"
                    spec = placement_pod_spec(name=name, image=self.image, gpu=gpu, dc=dc, model=model,
                                              volume_id=vol, token=token, controller_url=self.controller_url,
                                              controller_ca_pem=self.controller_ca_pem,
                                              stop_at=now + self.ttl_h * 3600.0)
                    if self.dry_run:
                        out.append({"create_pod": redacted(spec)})
                        pid = f"dry-{self._n}"
                    else:
                        try:
                            pid = str(self.api.create_pod(spec)["id"])
                        except RunPodError as exc:
                            if capacity_error(exc):
                                self.controller.mark_unavailable(dc, now)
                            out.append({"create_failed": f"{dc}/{model}", "error": str(exc)[:200]})
                            break
                        out.append({"created": pid, "dc": dc, "model": model})
                    self.pods[pid] = PodState(pid, dc, model, now, "starting")
            elif act == "drain":
                cands = sorted((p for p in self.pods.values() if (p.dc, p.model) == (dc, model)
                                and p.state != "draining"), key=lambda p: (p.state == "serving", p.started_at))
                for p in cands[:n]:  # pods still starting first, then the oldest
                    p.state = "draining"
                    self.draining_since[p.id] = now
                    if self.drain:
                        self.drain(p.id)
                    out.append({"drain": p.id, "dc": dc, "model": model})
        for pid, t in list(self.draining_since.items()):
            if now - t >= self.drain_grace_s:
                if self.remove:
                    self.remove(pid)  # out of the directory first
                if not self.dry_run:
                    try:
                        self.api.terminate_pod(pid)
                    except RunPodError as exc:
                        if exc.status != 404:
                            out.append({"terminate_failed": pid, "error": str(exc)[:200]})
                            continue
                out.append({"terminate": pid})
                self.pods.pop(pid, None)
                del self.draining_since[pid]
        self.log += out
        return out


def tick(controller: Any, actuator: Actuator, forecaster: Any, demand_now: Mapping[tuple[str, str], float],
         now: float, ledger: Any, *, lead_s: float | None = None,
         availability: Mapping[str, Any] | None = None) -> Any:
    """One control step: observe the demand, anticipate one cold start ahead, plan, apply, account the spend."""
    forecaster.observe(now, demand_now)
    lead = controller.cfg.cold_start_s if lead_s is None else lead_s
    keys = set(demand_now) | set(forecaster.series)
    anticipated = {k: forecaster.anticipate(k, now, lead) for k in keys}
    plan = controller.plan(now, anticipated, actuator.fleet(), ledger=ledger, availability=availability)
    actuator.apply(plan)
    ledger.record(now, sum(1 for p in actuator.fleet()) * controller.price)
    return plan


def pod_address(api: Any, pod_id: str) -> tuple[str, int, int | None]:
    """RunPod's view of a pod's public address (the only address a leaf is issued for)."""
    pod = api.get_pod(pod_id)
    ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
    svc = ports.get(str(pod_boot.SERVICE_PORT))
    if not ip or not svc:
        raise KeyError(f"pod {pod_id} has no public address yet")
    lane = ports.get(str(pod_boot.LANE_PORT))
    return str(ip), int(svc), int(lane) if lane else None


# ------------------------------------------------------------------------------------------------ commands
def _warm(spec: str | None, models: list[str]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for item in [x for x in (spec or "").split(",") if x]:
        metro, _, n = item.partition("=")
        for m in models:
            out[(metro.strip(), m)] = int(n or 1)
    return out


def load_demand(path: str | None) -> dict[tuple[str, str], float]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):  # the aggregate export
        from pgw.placement.telemetry import demand_by_metro

        return demand_by_metro(raw)
    return {(metro, model): float(v) for metro, ms in raw.items() for model, v in ms.items()}


def build_controller(args: argparse.Namespace) -> Any:
    from pgw.placement.policy import Controller, PolicyConfig

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = PolicyConfig(models=tuple(models), warm=_warm(args.warm, models), gpu=args.gpu,
                       max_usd_per_hour=args.max_usd_hour, max_usd_per_day=args.max_usd_day,
                       max_usd_total=args.max_usd_total, gain_up=args.gain_up, gain_keep=args.gain_keep,
                       idle_timeout_s=args.idle_timeout_min * 60.0)
    return Controller(cfg)


def cmd_plan(args: argparse.Namespace) -> int:
    c = build_controller(args)
    plan = c.plan(time.time(), load_demand(args.demand))
    print(json.dumps(plan.public(), indent=1))
    return 0


def cmd_coldstart(args: argparse.Namespace) -> int:
    from pgw.placement.coldstart import estimate, measure_hash_mb_s

    h = measure_hash_mb_s(128)
    for pre in (True, False):
        for cached in (True, False):
            e = estimate(args.model, prestaged=pre, image_cached=cached, hash_mb_s=h)
            comp = ", ".join(f"{k} {v:.0f}" for k, v in e["components_s"].items())
            print(f"{args.model} prestaged={pre} image_cached={cached}: ready {e['ready_s']:.0f} s, in use "
                  f"{e['in_use_s']:.0f} s  ({comp})")
    print(f"(SHA-256 here: {h:.0f} MB/s per core; other figures are assumptions: pgw.placement.coldstart.DEFAULTS)")
    return 0


def cmd_stage_packs(args: argparse.Namespace, api: RunPodAPI | None) -> int:
    from pgw.placement.packs import stage_plan, summary

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    items = stage_plan([d.strip() for d in args.dcs.split(",") if d.strip()], models)
    print(json.dumps(summary(items), indent=1))
    est = launch.estimate(launch.CPU_VCPU_PRICE_H * PACKER_VCPUS, "table", PACKER_TTL_MIN, volume_gb=0,
                          network_volume=False)
    total = est.total_usd * len(items)
    print(f"packer pods: {len(items)} x {PACKER_VCPUS} vCPU x {PACKER_TTL_MIN:.0f} min = ${total:.2f} at most")
    if total > args.max_usd:
        raise SystemExit(f"refused: ${total:.2f} exceeds --max-usd {args.max_usd:.2f}")
    image = args.image or os.environ.get("PGW_RUNPOD_IMAGE") or "<PGW_RUNPOD_IMAGE>"
    stop_at = time.time() + PACKER_TTL_MIN * 60.0
    for it in items:
        if args.dry_run:
            vol = f"<network volume {it.volume} ({it.size_gb} GB) in {it.dc}>"
            print(json.dumps({"create_network_volume": {"name": it.volume, "size": it.size_gb,
                                                        "dataCenterId": it.dc}}))
        else:
            assert api is not None
            vol = launch.ensure_network_volume(api, it.volume, it.size_gb, it.dc)
        spec = packer_pod_spec(name=f"{PACKER_PREFIX}{it.dc.lower()}", image=image, dc=it.dc, volume_id=vol,
                               models=list(it.models), stop_at=stop_at)
        if args.dry_run:
            print(json.dumps({"create_pod": spec}, indent=1))
        else:
            assert api is not None
            pod = api.create_pod(spec)
            print(f"packer pod {pod['id']} in {it.dc} (stops itself when done, at most {PACKER_TTL_MIN:.0f} min)")
    if args.dry_run:
        print("dry run: nothing created, nothing spent")
    return 0


def cmd_apply(args: argparse.Namespace, api: RunPodAPI | None) -> int:
    c = build_controller(args)
    volumes = json.loads(args.volumes) if args.volumes else {}
    if args.dry_run and not volumes:
        from pgw.placement.packs import volume_name

        volumes = {d: f"<network volume {volume_name(d)}>" for d in c.dcs}
    ca = Path(args.controller_ca).read_bytes() if args.controller_ca else b"<controller CA PEM>"
    act = Actuator(api, c, image=args.image or os.environ.get("PGW_RUNPOD_IMAGE") or "<PGW_RUNPOD_IMAGE>",
                   volumes=volumes, controller_url=args.controller_url, controller_ca_pem=ca, dry_run=args.dry_run,
                   max_pods=args.max_pods, ttl_h=args.pod_ttl_h)
    plan = c.plan(time.time(), load_demand(args.demand), act.fleet())
    print(json.dumps(plan.public(), indent=1))
    for rec in act.apply(plan):
        print(json.dumps(rec, indent=1))
    if args.dry_run:
        print("dry run: nothing created, nothing spent")
    return 0


def build_service(args: argparse.Namespace, api: Any) -> dict[str, Any]:
    """The controller of a live trial (RUNBOOK section 3.9): registry with the pod-issuing CA, the directory publisher
    (writes the signed directory where the enrolled endpoint serves it), the pods' HTTP API and the control loop.
    Secrets read here: the pod-issuing CA key and the directory signing key (files 0600, refused otherwise)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    from pgw.placement.certs import PodIssuer
    from pgw.placement.forecast import Forecaster
    from pgw.placement.packs import pinned_fingerprint
    from pgw.placement.policy import Ledger
    from pgw.placement.publisher import DirectoryPublisher, load_signer
    from pgw.placement.registry import Registry
    from pgw.placement.server import create_controller_app, tls_probe

    d = Path(args.state_dir)
    for f in ("issuing-ca.key", "directory.key"):
        if (d / f).stat().st_mode & 0o077:
            raise SystemExit(f"{d / f} must be 0600")
    ikey = serialization.load_pem_private_key((d / "issuing-ca.key").read_bytes(), password=None)
    icert = x509.load_pem_x509_certificate((d / "issuing-ca.pem").read_bytes())
    root_pem, client_ca_pem = (d / "root.pem").read_bytes(), (d / "client-ca.pem").read_bytes()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    pub = DirectoryPublisher(load_signer(d / "directory.key"), d / "directory-version.json",
                             validity_s=args.directory_validity_s)
    out_file = Path(args.directory_file)
    enrolled = json.loads(args.enrolled) if args.enrolled else []

    def publish(eps: list[dict[str, Any]]) -> None:
        tmp = out_file.with_suffix(".tmp")
        tmp.write_bytes(pub.publish([*enrolled, *eps]))
        os.replace(tmp, out_file)

    reg = Registry(PodIssuer(ikey, icert), {m: pinned_fingerprint(m) for m in models}, on_change=publish)
    probe_token = (d / "probe-token").read_text().strip() if (d / "probe-token").exists() else None

    def tenants() -> dict[str, Any]:
        return {"tokens_file": Path(args.tokens_file).read_text(), "mtls_file": Path(args.mtls_file).read_text()}

    app = create_controller_app(reg, lookup=lambda pid: pod_address(api, pid),
                                probe=lambda url, cert: tls_probe(url, cert, root_pem=root_pem, token=probe_token),
                                root_pem=root_pem, client_ca_pem=client_ca_pem, tenants=tenants)
    c = build_controller(args)
    act = Actuator(api, c, image=args.image or os.environ.get("PGW_RUNPOD_IMAGE") or "", volumes=json.loads(
        args.volumes), controller_url=args.controller_url, controller_ca_pem=Path(args.controller_ca).read_bytes(),
        expect=reg.expect, drain=reg.drain, remove=reg.remove, dry_run=args.dry_run, max_pods=args.max_pods,
        ttl_h=args.pod_ttl_h)
    return {"app": app, "registry": reg, "publisher": pub, "controller": c, "actuator": act,
            "forecaster": Forecaster(), "ledger": Ledger(), "publish": publish}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def policy_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--demand", default=None, help="demand JSON (see the module docstring)")
        p.add_argument("--models", default="qwen2.5-7b-instruct")
        p.add_argument("--warm", default=None, help="metro=pods,... kept warm (e.g. new-york=1,london=1)")
        p.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
        p.add_argument("--max-usd-hour", type=float, default=10.0)
        p.add_argument("--max-usd-day", type=float, default=60.0)
        p.add_argument("--max-usd-total", type=float, default=60.0)
        p.add_argument("--gain-up", type=float, default=2.0)
        p.add_argument("--gain-keep", type=float, default=1.0)
        p.add_argument("--idle-timeout-min", type=float, default=30.0)

    p = sub.add_parser("plan", help="the policy's plan for a demand (no API call)")
    policy_args(p)
    p = sub.add_parser("apply", help="plan and create / drain pods (--dry-run prints the payloads)")
    policy_args(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--image", default=None)
    p.add_argument("--volumes", default=None, help='JSON {"<dc>": "<network volume id>"} of the staged packs')
    p.add_argument("--controller-url", default="https://<controller>:9443")
    p.add_argument("--controller-ca", default=None, help="the controller's CA certificate (PEM file)")
    p.add_argument("--max-pods", type=int, default=4)
    p.add_argument("--pod-ttl-h", type=float, default=POD_TTL_H, help="each pod stops itself after this")
    p = sub.add_parser("stage-packs", help="per-data-centre network volumes + a CPU packer pod each")
    p.add_argument("--dcs", required=True)
    p.add_argument("--models", default="qwen2.5-7b-instruct")
    p.add_argument("--image", default=None)
    p.add_argument("--max-usd", type=float, default=3.0)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("serve", help="the live trial's controller: pods' API + the control loop (RUNBOOK 3.9)")
    policy_args(p)
    p.add_argument("--dry-run", action="store_true", help="plan and log the payloads; create nothing")
    p.add_argument("--state-dir", required=True, help="root.pem, client-ca.pem, issuing-ca.{pem,key}, directory.key")
    p.add_argument("--directory-file", required=True, help="where the enrolled endpoint serves the directory from")
    p.add_argument("--enrolled", default=None, help="JSON list of fixed endpoint entries (the enrolled one)")
    p.add_argument("--tokens-file", required=True, help="the enrolled endpoint's tokens file (digests only)")
    p.add_argument("--mtls-file", required=True, help="the enrolled endpoint's mTLS tenants file")
    p.add_argument("--aggregates-url", default=None, help="GET /pgw/v1/placement/aggregates of the enrolled endpoint")
    p.add_argument("--image", default=None)
    p.add_argument("--volumes", required=True, help='JSON {"<dc>": "<network volume id>"}')
    p.add_argument("--controller-url", required=True)
    p.add_argument("--controller-ca", required=True)
    p.add_argument("--max-pods", type=int, default=4)
    p.add_argument("--pod-ttl-h", type=float, default=POD_TTL_H, help="each pod stops itself after this")
    p.add_argument("--directory-validity-s", type=int, default=3600)
    p.add_argument("--tick-s", type=float, default=120.0)
    p.add_argument("--listen", default="0.0.0.0:9443")
    p.add_argument("--tls-cert", required=True)
    p.add_argument("--tls-key", required=True)
    p = sub.add_parser("coldstart", help="cold-start time by component (estimate)")
    p.add_argument("--model", default="qwen2.5-7b-instruct")
    return ap


def main(argv: list[str] | None = None, *, api_factory: Callable[[], RunPodAPI] = RunPodAPI) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "coldstart":
        return cmd_coldstart(args)
    api = None if getattr(args, "dry_run", False) else api_factory()
    if args.cmd == "stage-packs":
        return cmd_stage_packs(args, api)
    if args.cmd == "serve":
        return cmd_serve(args, api)
    return cmd_apply(args, api)


def cmd_serve(args: argparse.Namespace, api: Any) -> int:  # pragma: no cover - the live trial's long-running loop
    import threading

    import uvicorn

    svc = build_service(args, api)
    svc["publish"]([])  # an initial directory: the enrolled endpoint(s) only
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            now = time.time()
            demand: dict[tuple[str, str], float] = {}
            if args.aggregates_url:
                import httpx

                from pgw.placement.telemetry import demand_by_metro

                tok = (Path(args.state_dir) / "export-token").read_text().strip()
                try:
                    r = httpx.get(args.aggregates_url, headers={"Authorization": f"Bearer {tok}"}, timeout=30,
                                  verify=str(Path(args.state_dir) / "root.pem"), trust_env=False)
                    demand = demand_by_metro(r.json()) if r.status_code == 200 else {}
                except httpx.HTTPError as exc:
                    print(f"aggregates not available: {type(exc).__name__}", flush=True)
            plan = tick(svc["controller"], svc["actuator"], svc["forecaster"], demand, now, svc["ledger"])
            print(json.dumps({"t": int(now), "plan": plan.public(), "actions": svc["actuator"].log[-8:]}), flush=True)
            stop.wait(args.tick_s)

    threading.Thread(target=loop, daemon=True).start()
    host, _, port = args.listen.rpartition(":")
    try:
        uvicorn.run(svc["app"], host=host, port=int(port), ssl_certfile=args.tls_cert, ssl_keyfile=args.tls_key,
                    log_level="warning")
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
