#!/usr/bin/env python3
"""RunPod measurement sessions driven only through RunPod's API and its HTTPS pod proxy (deploy/runpod/RUNBOOK.md).

A session is one GPU pod and, where the measurement needs the network, one CPU client pod in the same data center.
Both start from public base images (no registry of our own) with ``pod_job.py`` as their boot; this script uploads
the code bundle (``pgw/`` as tracked by git, SHA-256 pinned in the pods' environment), the pods set themselves up
(``job_setup.sh``) and run their steps (``jobs.py pod``), and this script polls their status and fetches the results
through ``https://<pod id>-8888.proxy.runpod.net`` (token-gated). Nothing here needs SSH or a raw TCP connection
from the session to a pod.

Session side::

    export RUNPOD_API_KEY=...     # or a placeholder when a proxy injects the real key; never printed
    python deploy/runpod/jobs.py check-key                 # read-only: balance, spend, pods, prices
    python deploy/runpod/jobs.py plan                      # the sessions, their minutes and $ (no API call)
    python deploy/runpod/jobs.py launch perf --dc US-KS-2 --max-usd 9 --dry-run
    python deploy/runpod/jobs.py launch perf --dc US-KS-2 --max-usd 9
    python deploy/runpod/jobs.py status [perf]            # pods, cost so far, each pod's step
    python deploy/runpod/jobs.py log perf gpu              # the pod's log tail
    python deploy/runpod/jobs.py fetch perf                # results -> benchmarks/results/runpod/<date>-perf/
    python deploy/runpod/jobs.py stop perf --terminate     # or: stop --all --terminate
    python deploy/runpod/jobs.py teardown-check            # exit 1 if any pod still runs

Cost guards (on top of RunPod's own spend limit): ``--max-usd`` per session (estimate for the TTL + watchdog grace,
refused above), ``--budget-usd`` over every session recorded in the state file (default 60), Secure Cloud only, a
TTL that the pods enforce themselves (``pod_job.py``), and a detached watchdog here (``launch.py --watchdog``).

Pod side (inside a job pod, started by ``pod_job.py``): ``python deploy/runpod/jobs.py pod`` runs the steps of
``PGW_JOB_PLAN`` for ``PGW_JOB_ROLE`` and writes everything under ``PGW_JOB_RESULTS``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import secrets
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PGW = HERE.parents[1]
sys.path.insert(0, str(HERE))
import launch  # noqa: E402  (stdlib only at import time)
import pod_boot  # noqa: E402
from runpod_api import RunPodAPI, RunPodError  # noqa: E402

STATE_FILE = launch.STATE_DIR / "jobs.json"
#: Public base images, pinned by digest: the provider Dockerfiles' bases (CUDA 13.2 / Python 3.12).
GPU_IMAGE = ("nvidia/cuda:13.2.1-base-ubuntu24.04@sha256:"
             "7be56e69d8ae7c3648b8fca009fa35980ecd9c6eefaafbff3ee8c224e0043eb5")
CPU_IMAGE = "python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df"
#: The pod's start command: Python from the base image (installed on the CUDA base), then pod_job.py from the env.
BOOT_CMD = ("command -v python3 >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install "
            "-y -qq --no-install-recommends python3 ca-certificates >/dev/null); exec python3 -c \"import base64,os,"
            "zlib;exec(compile(zlib.decompress(base64.b64decode(os.environ['PGW_JOB_BOOT'])),'pod_job','exec'))\"")
CONTROL_PORT = 8888
#: torch 2.15.0+cu132 needs a CUDA 13 driver on the host.
CUDA_VERSIONS = ["13.0", "13.1", "13.2"]
H100 = "NVIDIA H100 80GB HBM3"
H200 = "NVIDIA H200"
MODELS_SHORT = {"7b": "qwen2.5-7b", "32b": "qwen2.5-32b", "72b": "qwen2.5-72b"}

#: The sessions of the runbook, in priority order. ``minutes``: the planned wall time of the steps; the TTL adds
#: ``SETUP_MIN`` (pod start + job_setup.sh) and a margin. ``client_vcpus`` 0: no client pod.
#: The client CPU switches of the 2026-11-26 rounds (several-answer trips, sampling: on in the SDK's NATIVE_ENV by
#: setdefault, so an explicit 0 wins): the ``native_off`` rerun of ``conc_16`` measures what they bought.
CLIENT_ROUNDS_OFF = {"PGW_SYNC_MULTI": "0", "PGW_TRIP_NATIVE_MULTI": "0", "PGW_DECODE_NATIVE_MULTI": "0",
                     "PGW_TRIP_LEASE_AHEAD": "0", "PGW_SAMPLE_THREADS": "0", "PGW_NATIVE_MULTINOMIAL": "0"}
#: The client mask cost switches of round 3 (each restores the previous path): ``factory_e2e`` runs on and off.
MASK_ROUND3_OFF = {"PGW_FAST_CHACHA512": "0", "PGW_FACTORY_R_REUSE": "0", "PGW_PKS_WIRE_FUSED": "0",
                   "PGW_PKS_OUT_REUSE": "0"}
#: ``factory_e2e``: one-core client mask CPU per slot at real shapes (benchmarks/pks_factory_e2e.py) on the client pod
#: while the GPU pod packs and loads its model (the client is idle then): (model, slots, switches).
FACTORY_E2E_RUNS = [(m, 1024, "on") for m in ("7b", "32b", "72b")] + [(m, 1024, "off") for m in ("7b", "32b", "72b")] \
    + [(m, 256, "on") for m in ("32b", "72b")]  # 256: one group per request (the 8 GB default stock at 32B / 72B)

SESSIONS: dict[str, dict[str, Any]] = {
    "perf": {
        "what": "provider_perf_session on one H100 (memory, device time, GEMV table, local lane, maskgen incl. pks-v3 "
                "device path + fused kernels + wire forms + per-stage times for memplan) and lane_nic over the real "
                "NIC to a same-DC CPU pod",
        "gpu": H100, "gpu_count": 1, "client_vcpus": 16, "volume_gb": 20, "client_disk_gb": 30, "minutes": 110,
        "steps": [{"kind": "perf", "phases": "env,memory,device_time,gemv_table,lane_local,maskgen,pks_stages",
                   "models": "qwen2.5-7b,qwen2.5-32b,qwen2.5-72b", "budget_min": 88},
                  {"kind": "lane_nic", "presets": "7b,72b", "nic_minutes": 8}]},
    "sdk7b": {
        "what": "the client library end to end at 7B from a same-DC CPU pod: setup, TTFT, tok/s, trips/token, client "
                "CPU with speculation off / lookup / draft_head (the shipped head), N = 1 / 4 / 16 concurrent "
                "streams, the prefix cache off / on, N = 16 with the client CPU rounds off; client mask CPU per slot "
                "at 7B / 32B / 72B shapes while the provider packs",
        "gpu": H100, "gpu_count": 1, "client_vcpus": 32, "volume_gb": 40, "client_disk_gb": 70, "minutes": 105,
        "steps": [{"kind": "factory_e2e"},
                  {"kind": "sdk", "model": "qwen2.5-7b-instruct", "spec": "off,lookup,draft_head", "conc": "1,4,16",
                   "tokens": 128, "mask_store_gb": 24, "stages": "setup,spec,conc,prefix",
                   "reruns": [{"name": "native_off", "stages": "conc", "conc": "16", "env": CLIENT_ROUNDS_OFF}]}]},
    "sdk32b": {
        "what": "32B on one H100: provider memory / device time / maskgen at 32B, then the client library end to end "
                "at 32B (measurement-only preview of the pinned model; spec off / lookup, N = 1 / 4)",
        "gpu": H100, "gpu_count": 1, "client_vcpus": 32, "volume_gb": 80, "client_disk_gb": 140, "minutes": 140,
        "steps": [{"kind": "perf", "phases": "env,memory,device_time,maskgen", "models": "qwen2.5-32b",
                   "budget_min": 35},
                  {"kind": "factory_e2e"},
                  {"kind": "sdk", "model": "qwen2.5-32b-instruct", "spec": "off,lookup", "conc": "1,4",
                   "tokens": 64, "mask_store_gb": 60}]},
    "72b": {
        "what": "72B on two H100s: memory at TP 1 and per-trip device time at TP 1 and 2, maskgen at 72B "
                "(H200 alternative: --gpu 'NVIDIA H200' --gpu-count 1, TP 1 only)",
        "gpu": H100, "gpu_count": 2, "client_vcpus": 0, "volume_gb": 20, "client_disk_gb": 0, "minutes": 70,
        "steps": [{"kind": "perf", "phases": "env,memory,device_time,maskgen", "models": "qwen2.5-72b",
                   "budget_min": 60}]},
}
SETUP_MIN = 15
MARGIN_MIN = 10


# ============================================================================================ session side
def load_state() -> dict[str, Any]:
    try:
        return dict(json.loads(STATE_FILE.read_text()))
    except (OSError, ValueError):
        return {"sessions": {}}


def save_state(st: dict[str, Any]) -> None:
    launch.STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = STATE_FILE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_FILE)


def boot_payload() -> str:
    return base64.b64encode(zlib.compress((HERE / "pod_job.py").read_bytes(), 9)).decode()


def build_bundle(out: Path, extras: list[Path]) -> tuple[Path, str]:
    """``pgw/`` as git sees it (tracked and new files, working-tree content; ignored files such as built ``.so``,
    ``target/`` and caches left out), plus ``extras/<name>`` files: (path, sha256)."""
    files = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # noqa: S607
                           cwd=PGW, check=True, capture_output=True).stdout
    names = sorted(n for n in files.decode().split("\0") if n)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tf:
        for n in names:
            p = PGW / n
            if p.is_file() and not p.is_symlink():
                tf.add(p, arcname=n, recursive=False)
        for e in extras:
            tf.add(e, arcname=f"extras/{e.name}", recursive=False)
    data = buf.getvalue()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out, hashlib.sha256(data).hexdigest()


def session_minutes(name: str, cfg: dict[str, Any]) -> float:
    return float(cfg["minutes"]) + SETUP_MIN + MARGIN_MIN


def session_estimate(cfg: dict[str, Any], ttl_min: float, gpu_price_h: float) -> dict[str, float]:
    gpu = launch.estimate(gpu_price_h, "", ttl_min, volume_gb=cfg["volume_gb"], network_volume=False)
    cpu_usd = 0.0
    if cfg["client_vcpus"]:
        cpu = launch.estimate(launch.CPU_VCPU_PRICE_H * cfg["client_vcpus"], "", ttl_min,
                              volume_gb=cfg["client_disk_gb"], network_volume=False)
        cpu_usd = cpu.total_usd
    return {"gpu_usd": round(gpu.total_usd, 2), "cpu_usd": round(cpu_usd, 2),
            "total_usd": round(gpu.total_usd + cpu_usd, 2), "hours": round(gpu.hours, 2)}


def table_price(gpu: str, count: int) -> float:
    p = {**{k: v[0] for k, v in launch.GPU_PRICE_H.items()}, H200: 3.99}.get(gpu)
    if p is None:
        raise SystemExit(f"no table price for {gpu!r}: pass --price-per-hour")
    return p * count


def cmd_plan(args: argparse.Namespace) -> int:
    total = 0.0
    print("| # | session | GPU | client | wall (min) | TTL (min) | est. $ (table, Secure) | feeds |")
    print("|---|---|---|---|---|---|---|---|")
    for i, (name, cfg) in enumerate(SESSIONS.items(), 1):
        ttl = session_minutes(name, cfg)
        est = session_estimate(cfg, ttl, table_price(cfg["gpu"], cfg["gpu_count"]))
        total += est["total_usd"]
        client = f"{cfg['client_vcpus']} vCPU" if cfg["client_vcpus"] else "-"
        print(f"| {i} | {name} | {cfg['gpu_count']} x {cfg['gpu']} | {client} | {cfg['minutes']} | {ttl:.0f} | "
              f"{est['total_usd']:.2f} | {cfg['what'][:60]}... |")
    print(f"\ntotal (all sessions at their full TTL): ${total:.2f}; default --budget-usd 60")
    return 0


def gpu_spec(name: str, cfg: dict[str, Any], *, dc: str, stop_at: float, env: dict[str, str]) -> dict[str, Any]:
    return {"name": f"pgw-job-{name}-gpu", "imageName": GPU_IMAGE, "computeType": "GPU", "gpuTypeIds": [cfg["gpu"]],
            "gpuCount": cfg["gpu_count"], "cloudType": "SECURE", "dataCenterIds": [dc],
            "containerDiskInGb": 60, "volumeInGb": cfg["volume_gb"], "volumeMountPath": "/workspace",
            "ports": [f"{CONTROL_PORT}/http", f"{pod_boot.SERVICE_PORT}/tcp", f"{pod_boot.LANE_PORT}/tcp"],
            "supportPublicIp": True, "dockerStartCmd": ["bash", "-c", BOOT_CMD], "interruptible": False,
            "allowedCudaVersions": CUDA_VERSIONS, "env": {**env, "PGW_JOB_STOP_AT": str(int(stop_at))}}


def cpu_spec(name: str, cfg: dict[str, Any], *, dc: str, stop_at: float, env: dict[str, str],
             flavor: str) -> dict[str, Any]:
    return {"name": f"pgw-job-{name}-cpu", "imageName": CPU_IMAGE, "computeType": "CPU", "cpuFlavorIds": [flavor],
            "vcpuCount": cfg["client_vcpus"], "dataCenterIds": [dc], "containerDiskInGb": cfg["client_disk_gb"],
            "ports": [f"{CONTROL_PORT}/http"], "supportPublicIp": True, "dockerStartCmd": ["bash", "-c", BOOT_CMD],
            "env": {**env, "PGW_JOB_STOP_AT": str(int(stop_at))}}


def redacted(spec: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(spec))
    for k, v in list(out.get("env", {}).items()):
        if k == "PGW_JOB_BOOT":
            out["env"][k] = f"<pod_job.py, zlib+base64, {len(v)} chars>"
        elif k == "PGW_JOB_PAIR_TOKEN":
            out["env"][k] = "<hidden>"
    return out


def proxy_url(pod_id: str, path: str) -> str:
    """RunPod's HTTPS proxy URL of a pod's control port (``PGW_JOB_PROXY_URL_FMT`` overrides it in local tests)."""
    fmt = os.environ.get("PGW_JOB_PROXY_URL_FMT") or "https://{pod}-{port}.proxy.runpod.net"
    return fmt.format(pod=pod_id, port=CONTROL_PORT) + path


def control(pod_id: str, token: str, path: str, *, method: str = "GET", body: bytes | None = None,
            timeout: float = 90.0) -> tuple[int, bytes]:
    """A call to a pod's control port through RunPod's HTTPS proxy (the session's HTTPS proxy is honoured)."""
    req = urllib.request.Request(proxy_url(pod_id, path), data=body, method=method,  # noqa: S310 - https
                                 headers={"X-PGW-Job-Token": token, "User-Agent": "pgw-jobs"})
    if body is not None:
        req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return int(r.status), r.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read() or b""
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(exc).encode()


def cmd_check_key(args: argparse.Namespace, api: RunPodAPI) -> int:
    """Read-only: the key works, the balance and spend rate, the pods, the GPU prices this runbook uses."""
    try:
        acct = api.account()
        print(f"account: balance ${acct.get('clientBalance')}, spending ${acct.get('currentSpendPerHr')}/h now, "
              f"spend limit {acct.get('spendLimit')}")
    except RunPodError as exc:
        print(f"account: not available ({exc})")
    pods = api.list_pods()
    print(f"pods: {len(pods)}")
    for p in pods:
        print(f"  {p.get('id')} {p.get('name')} {p.get('desiredStatus')} {p.get('costPerHr')} $/h")
    try:
        prices = api.gpu_prices()
        for g in (H100, H200, "NVIDIA H100 PCIe", "NVIDIA H100 NVL"):
            r = prices.get(g) or {}
            print(f"price {g}: secure {r.get('securePrice')} / community {r.get('communityPrice')} $/h")
    except RunPodError as exc:
        print(f"prices: not available ({exc})")
    return 0


def check_sdk_steps(steps: list[dict[str, Any]]) -> list[str]:
    """Refuse, before any pod exists, an sdk step the client library would refuse only after the provider's download
    and pack: a model without a baked pin, an unpinned provider model, or a ``draft_head`` mode without a head. A
    pinned model outside ``SUPPORTED_MODELS`` runs as the measurement-only preview (``--preview``). Returns notes."""
    sdk = [s for s in steps if s["kind"] == "sdk"]
    if not sdk:
        return []
    sys.path.insert(0, str(PGW / "src"))
    from pgw.sdk.draft_heads import PINNED_DRAFT_HEADS
    from pgw.sdk.options import SUPPORTED_MODELS
    from pgw.sdk.provisioning import PINNED_MODELS

    notes = []
    for s in sdk:
        m = s["model"]
        pin = PINNED_MODELS.get(m)
        if pin is None or not pin.fingerprint:
            raise SystemExit(f"sdk step {m}: no baked pin in PINNED_MODELS; the client would refuse it")
        if m not in pod_boot.MODELS or pod_boot.MODELS[m][1] != pin.revision:
            raise SystemExit(f"sdk step {m}: the provider (pod_boot.MODELS) does not pack the pinned commit "
                             f"{pin.revision[:12]}")
        if m not in SUPPORTED_MODELS:
            notes.append(f"sdk {m}: pinned, not supported: measurement-only preview (PGW_SDK_PREVIEW_MODELS), "
                         f"verified against the baked pin {pin.fingerprint[:12]}...")
        if "draft_head" in s.get("spec", "") and not s.get("draft_head"):
            if m not in PINNED_DRAFT_HEADS:
                raise SystemExit(f"sdk step {m}: draft_head has no shipped head; pass --extra <head file>")
            notes.append(f"sdk {m}: draft_head uses the shipped pinned head {PINNED_DRAFT_HEADS[m].file}")
    return notes


def cmd_launch(args: argparse.Namespace, api: RunPodAPI | None) -> int:
    cfg = json.loads(json.dumps(SESSIONS[args.session]))
    if args.gpu:
        cfg["gpu"] = args.gpu
    if args.gpu_count:
        cfg["gpu_count"] = args.gpu_count
    if args.no_client:
        cfg["client_vcpus"] = 0
    if any(s["kind"] in ("lane_nic", "sdk", "factory_e2e") for s in cfg["steps"]) and not cfg["client_vcpus"]:
        raise SystemExit("this session needs its client pod (lane_nic / sdk / factory_e2e steps)")
    extras = [Path(e).resolve() for e in args.extra]
    for e in extras:
        if not e.is_file():
            raise SystemExit(f"--extra {e}: no such file")
    for s in cfg["steps"]:  # --extra overrides the shipped draft head
        if s["kind"] == "sdk" and extras and "draft_head" in s.get("spec", ""):
            s["draft_head"] = f"extras/{extras[0].name}"
    for line in check_sdk_steps(cfg["steps"]):
        print(f"  {line}")
    ttl =args.ttl_min or session_minutes(args.session, cfg)
    if api is not None and args.price_per_hour is None:
        price_h, src = launch.gpu_price(cfg["gpu"], community=False, gpu_count=cfg["gpu_count"], api=api)
    elif args.price_per_hour is not None:
        price_h, src = args.price_per_hour * cfg["gpu_count"], "--price-per-hour"
    else:
        price_h, src = table_price(cfg["gpu"], cfg["gpu_count"]), "table"
    est = session_estimate(cfg, ttl, price_h)
    st = load_state()
    spent_cap = sum(s.get("est_usd", 0.0) for s in st["sessions"].values())
    print(f"session {args.session}: {cfg['gpu_count']} x {cfg['gpu']} ({price_h:.2f} $/h, {src}), client "
          f"{cfg['client_vcpus']} vCPU, TTL {ttl:.0f} min, Secure Cloud, {args.dc}")
    print(f"  estimate ${est['total_usd']:.2f} (GPU {est['gpu_usd']:.2f} + client {est['cpu_usd']:.2f}); --max-usd "
          f"{args.max_usd:.2f}; sessions so far ${spent_cap:.2f} of --budget-usd {args.budget_usd:.2f}")
    if est["total_usd"] > args.max_usd:
        raise SystemExit(f"refused: estimate ${est['total_usd']:.2f} exceeds --max-usd {args.max_usd:.2f}")
    if spent_cap + est["total_usd"] > args.budget_usd:
        raise SystemExit(f"refused: ${spent_cap:.2f} already committed + ${est['total_usd']:.2f} exceeds "
                         f"--budget-usd {args.budget_usd:.2f}")
    if args.session in st["sessions"] and st["sessions"][args.session].get("active") and not args.dry_run:
        raise SystemExit(f"session {args.session} is recorded as active: stop it first (jobs.py stop {args.session})")
    if args.session in st["sessions"] and not args.dry_run:  # a rerun: the earlier run keeps counting in the budget
        old = st["sessions"].pop(args.session)
        st["sessions"][f"{args.session}@{int(old.get('launched_at', 0))}"] = old
    bundle, sha = build_bundle(launch.STATE_DIR / f"bundle-{args.session}.tar.gz", extras)
    job_token, pair_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    steps = cfg["steps"]
    stop_at = time.time() + ttl * 60.0
    common = {"PGW_JOB_BOOT": boot_payload(), "PGW_JOB_TOKEN_SHA256": hashlib.sha256(job_token.encode()).hexdigest(),
              "PGW_JOB_BUNDLE_SHA256": sha, "PGW_JOB_PLAN": json.dumps(steps), "PGW_JOB_SESSION": args.session}
    g = gpu_spec(args.session, cfg, dc=args.dc, stop_at=stop_at,
                 env={**common, "PGW_JOB_ROLE": "gpu",
                      "PGW_JOB_PAIR_TOKEN_SHA256": hashlib.sha256(pair_token.encode()).hexdigest()})
    print(f"  bundle {bundle} ({bundle.stat().st_size / 1e6:.1f} MB, sha256 {sha[:16]}...)")
    if args.dry_run:
        print(json.dumps({"create_gpu_pod": redacted(g)}, indent=1))
        if cfg["client_vcpus"]:
            c = cpu_spec(args.session, cfg, dc=args.dc, stop_at=stop_at, flavor=args.cpu_flavor,
                         env={**common, "PGW_JOB_ROLE": "cpu", "PGW_JOB_PAIR_TOKEN": pair_token,
                              "PGW_JOB_PEER_POD": "<gpu pod id>"})
            print(json.dumps({"create_cpu_pod": redacted(c)}, indent=1))
        print("dry run: nothing created, nothing spent")
        return 0
    assert api is not None
    rec: dict[str, Any] = {"active": True, "dc": args.dc, "est_usd": est["total_usd"], "max_usd": args.max_usd,
                           "price_h": price_h, "launched_at": time.time(), "stop_at": stop_at, "ttl_min": ttl,
                           "job_token": job_token, "bundle": str(bundle), "pods": {}}
    st["sessions"][args.session] = rec
    save_state(st)
    gp = api.create_pod(g)
    gid = str(gp["id"])
    rec["pods"]["gpu"] = {"id": gid, "watchdog_pid": launch.spawn_watchdog(gid, stop_at, terminate=True)}
    save_state(st)
    print(f"  GPU pod {gid} created (watchdog terminates it at TTL + {launch.WATCHDOG_GRACE_MIN:.0f} min)")
    if cfg["client_vcpus"]:
        c = cpu_spec(args.session, cfg, dc=args.dc, stop_at=stop_at, flavor=args.cpu_flavor,
                     env={**common, "PGW_JOB_ROLE": "cpu", "PGW_JOB_PAIR_TOKEN": pair_token, "PGW_JOB_PEER_POD": gid})
        try:
            cp = api.create_pod(c)
        except RunPodError as exc:
            print(f"client pod refused ({exc}); stopping the GPU pod", file=sys.stderr)
            api.terminate_pod(gid)
            rec["active"] = False
            save_state(st)
            return 1
        cid = str(cp["id"])
        rec["pods"]["cpu"] = {"id": cid, "watchdog_pid": launch.spawn_watchdog(cid, stop_at, terminate=True)}
        save_state(st)
        print(f"  client pod {cid} created")
    return upload(args.session, st, timeout_min=args.start_timeout_min)


def upload(session: str, st: dict[str, Any], *, timeout_min: float) -> int:
    rec = st["sessions"][session]
    data = Path(rec["bundle"]).read_bytes()
    for role, p in rec["pods"].items():
        t0 = time.time()
        while True:
            code, body = control(p["id"], rec["job_token"], "/status", timeout=30)
            if code == 200:
                state = json.loads(body).get("state")
                if state != "waiting-bundle":
                    print(f"  {role} pod {p['id']}: already {state}")
                    break
                code, body = control(p["id"], rec["job_token"], "/bundle", method="PUT", body=data, timeout=300)
                print(f"  {role} pod {p['id']}: bundle -> HTTP {code} {body[:200].decode(errors='replace')}")
                break
            if time.time() - t0 > timeout_min * 60:
                print(f"  {role} pod {p['id']}: control port not reachable after {timeout_min:.0f} min (last: {code} "
                      f"{body[:120]!r}); retry with: jobs.py upload {session}, or stop it", file=sys.stderr)
                return 1
            time.sleep(15)
    print(f"next: python deploy/runpod/jobs.py status {session}   (every few minutes)")
    return 0


def cmd_status(args: argparse.Namespace, api: RunPodAPI | None) -> int:
    st = load_state()
    names = [args.session] if args.session else list(st["sessions"])
    for name in names:
        rec = st["sessions"].get(name)
        if not rec:
            print(f"{name}: not recorded")
            continue
        left = (rec["stop_at"] - time.time()) / 60
        print(f"{name}: {'active' if rec.get('active') else 'stopped'}, est ${rec['est_usd']:.2f}, TTL "
              f"{f'in {left:.0f} min' if left > 0 else 'reached'}")
        for role, p in rec["pods"].items():
            live: dict[str, Any] = {}
            if api is not None:
                try:
                    live = api.get_pod(p["id"])
                except RunPodError as exc:
                    live = {"desiredStatus": f"API: {exc}"}
            up_h = (time.time() - rec["launched_at"]) / 3600
            cost = float(live.get("costPerHr") or 0.0)
            line = f"  {role} {p['id']}: {live.get('desiredStatus', '?')} {cost} $/h (~${cost * up_h:.2f} so far)"
            if rec.get("active"):
                code, body = control(p["id"], rec["job_token"], "/status", timeout=30)
                if code == 200:
                    s = json.loads(body)
                    line += f"  job {s.get('state')} rc={s.get('rc')} {json.dumps(s.get('progress', {}))[:200]}"
                else:
                    line += f"  control port: {code or 'unreachable'}"
            print(line)
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    rec = load_state()["sessions"][args.session]
    code, body = control(rec["pods"][args.role]["id"], rec["job_token"], "/log")
    sys.stdout.write(body.decode(errors="replace")[-args.chars:] if code == 200 else f"HTTP {code}\n")
    return 0 if code == 200 else 1


def cmd_fetch(args: argparse.Namespace) -> int:
    rec = load_state()["sessions"][args.session]
    out = Path(args.out or PGW / "benchmarks" / "results" / "runpod" /
               f"{time.strftime('%Y%m%d', time.gmtime(rec['launched_at']))}-{args.session}")
    rc = 0
    for role, p in rec["pods"].items():
        code, body = control(p["id"], rec["job_token"], "/results.tar.gz", timeout=300)
        if code != 200:
            print(f"{role}: HTTP {code} {body[:200]!r}", file=sys.stderr)
            rc = 1
            continue
        dst = out / role
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
            tf.extractall(dst, filter="data")
        print(f"{role}: {len(body) / 1e6:.2f} MB -> {dst}")
    return rc


def cmd_stop(args: argparse.Namespace, api: RunPodAPI) -> int:
    st = load_state()
    names = list(st["sessions"]) if args.all else [args.session]
    rc = 0
    for name in names:
        rec = st["sessions"].get(name)
        if not rec:
            continue
        for role, p in rec["pods"].items():
            try:
                (api.terminate_pod if args.terminate else api.stop_pod)(p["id"])
                print(f"{name} {role} {p['id']}: {'terminated' if args.terminate else 'stopped'}")
            except RunPodError as exc:
                if exc.status == 404:
                    print(f"{name} {role} {p['id']}: gone")
                else:
                    print(f"{name} {role} {p['id']}: {exc}", file=sys.stderr)
                    rc = 1
                    continue
            launch.kill_watchdog(p.get("watchdog_pid"))
        rec["active"] = False
        rec["stopped_at"] = time.time()
    save_state(st)
    return rc


def cmd_teardown_check(api: RunPodAPI) -> int:
    pods = api.list_pods()
    live = [p for p in pods if str(p.get("desiredStatus")) not in ("EXITED", "TERMINATED")]
    for p in pods:
        print(f"  {p.get('id')} {p.get('name')} {p.get('desiredStatus')} {p.get('costPerHr')} $/h")
    if live:
        print(f"NOT CLEAN: {len(live)} pod(s) still running: stop them (jobs.py stop --all --terminate)")
        return 1
    stopped = [p for p in pods if str(p.get("desiredStatus")) == "EXITED"]
    print(f"clean: no running pods ({len(stopped)} stopped pod(s) keep a volume and bill storage: terminate them)"
          if stopped else "clean: no pods")
    return 0


# ================================================================================================ pod side
def _progress(res: Path, **kw: Any) -> None:
    p = res / "progress.json"
    tmp = res / ".progress.tmp"
    tmp.write_text(json.dumps({**kw, "t": time.time()}))
    os.replace(tmp, p)


def _run(cmd: list[str], log: Path, env: dict[str, str] | None = None, timeout: float | None = None,
         stdout: Path | None = None, cpus: set[int] | None = None) -> int:
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    pin = (lambda: os.sched_setaffinity(0, cpus)) if cpus else None
    with log.open("ab") as f, (stdout.open("ab") if stdout is not None else f) as so:  # type: ignore[attr-defined]
        try:
            return subprocess.run(cmd, cwd=PGW, env=env, stdout=so, stderr=f,  # noqa: S603
                                  timeout=timeout, preexec_fn=pin).returncode
        except subprocess.TimeoutExpired:
            return -9


def _peer_url(name: str) -> str:
    return proxy_url(os.environ["PGW_JOB_PEER_POD"], f"/peer/{name}")


def peer_put_local(name: str, data: bytes) -> None:
    """GPU side: publish a file to the client pod (served by this pod's pod_job.py under /peer/)."""
    d = Path(os.environ.get("PGW_JOB_ROOT", "/workspace/job")) / "peer"
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = d / f".{name}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, d / name)


def peer_get(name: str, timeout_s: float) -> bytes | None:
    """CPU side: wait for the GPU pod's file ``name`` (pair token)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        req = urllib.request.Request(_peer_url(name),  # noqa: S310 - https RunPod proxy
                                     headers={"X-PGW-Pair-Token": os.environ["PGW_JOB_PAIR_TOKEN"]})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - https RunPod proxy
                return r.read()
        except (urllib.error.URLError, OSError):
            time.sleep(10)
    return None


def peer_put_remote(name: str, data: bytes) -> bool:
    """CPU side: tell the GPU pod something (a done marker)."""
    req = urllib.request.Request(_peer_url(name), data=data, method="PUT",  # noqa: S310
                                 headers={"X-PGW-Pair-Token": os.environ["PGW_JOB_PAIR_TOKEN"]})
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
                return 200 <= r.status < 300
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    return False


def peer_wait_local(name: str, timeout_s: float) -> bool:
    p = Path(os.environ.get("PGW_JOB_ROOT", "/workspace/job")) / "peer" / name
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.exists():
            return True
        time.sleep(5)
    return False


def minutes_left() -> float:
    return (float(os.environ.get("PGW_JOB_STOP_AT", time.time() + 86400)) - time.time()) / 60


def _smoke() -> list[str]:
    return ["--cpu"] if os.environ.get("PGW_JOB_CPU_SMOKE") == "1" else []


def nic_ports() -> tuple[int, int]:
    """(lane port, stub port) the GPU pod's lane bench listens on: RunPod maps 8444 and 8443 (the provider's ports;
    no provider runs during lane_nic)."""
    return (int(os.environ.get("PGW_JOB_NIC_PORT") or pod_boot.LANE_PORT),
            int(os.environ.get("PGW_JOB_NIC_STUB_PORT") or pod_boot.SERVICE_PORT))


def gpu_perf(step: dict[str, Any], res: Path, py: str) -> int:
    return _run([py, "deploy/runpod/provider_perf_session.py", "--out", str(res / "perf"), "--phases", step["phases"],
                 "--models", step["models"], "--budget-min", str(step.get("budget_min", 90)), *_smoke()],
                res / "perf.log")


def gpu_lane_nic(step: dict[str, Any], res: Path, py: str) -> int:
    out = res / "perf"
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def publish() -> None:  # the bench's info file (CA, throwaway tickets) + RunPod's public ports, to the client
        done: set[str] = set()
        while not stop.wait(3):
            for pre in step["presets"].split(","):
                f = out / f"lane_nic_{pre}.json"
                if pre not in done and f.exists():
                    info = json.loads(f.read_text())
                    info["public_port"] = int(os.environ.get(f"RUNPOD_TCP_PORT_{nic_ports()[0]}") or 0)
                    info["public_stub_port"] = int(os.environ.get(f"RUNPOD_TCP_PORT_{nic_ports()[1]}") or 0)
                    peer_put_local(f"lane_nic_{pre}.json", json.dumps(info).encode())
                    done.add(pre)

    th = threading.Thread(target=publish, daemon=True)
    th.start()
    rc = _run([py, "deploy/runpod/provider_perf_session.py", "--out", str(out), "--phases", "lane_nic",
               "--host", os.environ.get("RUNPOD_PUBLIC_IP", ""), "--nic-minutes", str(step["nic_minutes"]),
               "--nic-port", str(nic_ports()[0]), "--nic-stub-port", str(nic_ports()[1]),
               "--nic-presets", step["presets"], *_smoke()],
              res / "lane_nic.log")
    stop.set()
    for f in out.glob("lane_nic_*.json"):  # the results keep the bench's table, not its (expired) tickets
        try:
            info = json.loads(f.read_text())
        except ValueError:
            continue
        if isinstance(info, dict) and ("tickets" in info or "stub_tickets" in info):
            info.pop("tickets", None)
            info.pop("stub_tickets", None)
            f.write_text(json.dumps(info, indent=1))
    return rc


def cpu_lane_nic(step: dict[str, Any], res: Path, py: str) -> int:
    rc = 0
    for pre in step["presets"].split(","):
        raw = peer_get(f"lane_nic_{pre}.json", timeout_s=max(60.0, (minutes_left() - 5) * 60))
        if raw is None:
            print(f"lane_nic {pre}: no info from the GPU pod", flush=True)
            return 1
        info = res / f"lane_nic_{pre}_info.json"
        info.write_text(raw.decode())
        meta = json.loads(raw)
        rc |= _run([py, "benchmarks/lane_nic_bench.py", "client", "--info", str(info),
                    "--port", str(meta.get("public_port") or meta["port"]),
                    "--stub-port", str(meta.get("public_stub_port") or meta["stub_port"]),
                    "--conc", "1,4,16,64", "--iters", "400", "--json", str(res / f"lane_nic_{pre}_client.json")],
                   res / "lane_nic_client.log", timeout=step["nic_minutes"] * 60)
        info.unlink()  # the throwaway tickets
    return rc


def maskgen_progress_sampler(stem: Path, stop: threading.Event, *, interval_s: float = 2.0,
                             url: str | None = None) -> None:
    """The provider's maskgen progress (metrics port ``/readyz``, ``maskgen_progress``: operator totals, no tenant
    names) every ``interval_s`` while it serves: ``<stem>.jsonl`` (samples that changed) and ``<stem>.json`` (the
    worst seen: oldest running item, slowest item, bound, any stall), to check the maskgen watchdog's 30 s bound
    (``PGW_SERVICE_MASKGEN_STALL_S``, 4x the slowest recent item) against real item times."""
    url = url or f"http://127.0.0.1:{pod_boot.METRICS_PORT}/readyz"
    worst: dict[str, Any] = {"samples": 0, "stalled_samples": 0, "oldest_running_s": 0.0, "oldest_waiting_s": 0.0,
                             "slowest_item_s": 0.0, "bound_s": None, "max_running": 0, "max_waiting": 0,
                             "done": 0, "failed": 0}
    last = None
    while not stop.wait(interval_s):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 - loopback, plain metrics port
                body = json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):  # 503 while loading: the body still has the details
                try:
                    body = json.loads(exc.read())
                except ValueError:
                    continue
            else:
                continue
        details = body.get("details", body) if isinstance(body, dict) else {}
        prog = details.get("maskgen_progress") if isinstance(details, dict) else None
        if not isinstance(prog, dict):
            continue
        worst["samples"] += 1
        worst["stalled_samples"] += int(bool(prog.get("stalled")))
        for k in ("oldest_running_s", "oldest_waiting_s", "slowest_item_s"):
            if isinstance(prog.get(k), int | float):
                worst[k] = max(worst[k], float(prog[k]))
        worst["max_running"] = max(worst["max_running"], int(prog.get("running") or 0))
        worst["max_waiting"] = max(worst["max_waiting"], int(prog.get("waiting") or 0))
        worst["bound_s"] = prog.get("bound_s")
        worst["done"], worst["failed"] = int(prog.get("done") or 0), int(prog.get("failed") or 0)
        key = json.dumps(prog, sort_keys=True)
        if key != last:
            with stem.with_suffix(".jsonl").open("a") as f:
                f.write(json.dumps({"t": round(time.time(), 1), "maskgen": details.get("maskgen"), **prog}) + "\n")
            last = key
        stem.with_suffix(".json").write_text(json.dumps(worst, indent=1))


def gpu_sdk(step: dict[str, Any], res: Path, py: str) -> int:
    """The provider (pod_boot) serving one model; a test org + SDK key created here over loopback (attested, pinned);
    the client pod gets URL, CA fingerprint, key and pack fingerprint through the pair-token peer channel."""
    model = step["model"]
    admin = "pgwa-" + secrets.token_urlsafe(32)
    data = Path("/workspace/pgw")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_JOB_")}
    env.update(PGW_RUNPOD_MODELS=model, PGW_RUNPOD_DATA=str(data), PGW_RUNPOD_KEY_ENV="test",
               PGW_RUNPOD_ADMIN_TOKEN_SHA256=hashlib.sha256(admin.encode()).hexdigest(),
               PGW_RUNPOD_STOP_AT=os.environ.get("PGW_JOB_STOP_AT", ""),
               PGW_RUNPOD_SERVICE_ENV=json.dumps(step.get("service_env", {})),
               PGW_RUNPOD_PACKER=step.get("packer", "shardwise"))  # "whole": pgw pack (whole checkpoint on disk)
    logf = (res / f"provider-{model}.log").open("ab")
    prov = subprocess.Popen([py, "deploy/runpod/pod_boot.py"], cwd=PGW, env=env, stdout=logf,  # noqa: S603
                            stderr=subprocess.STDOUT)
    stop = threading.Event()
    sampler = threading.Thread(target=maskgen_progress_sampler, args=(res / f"maskgen_progress-{model}", stop),
                               daemon=True)
    sampler.start()
    try:
        url = "https://127.0.0.1:8443"
        launch.wait_for_ready(url, timeout_s=max(300.0, (minutes_left() - 20) * 60))
        key, _ca_pem, rec = launch.bootstrap_sdk_key(url, admin, org="pgw-bench", key_name=f"bench-{model}",
                                                    models=[model])
        boot = json.loads((data / "state" / "boot.json").read_text())
        from pgw.modelpack.loader import ModelPack

        fp = ModelPack.open(data / "packs" / model).fingerprint
        (res / f"provider-{model}.json").write_text(json.dumps({"boot": boot, "fingerprint": fp,
                                                                "key_id": rec.get("key_id")}, indent=1))
        peer_put_local(f"sdk-{model}.json", json.dumps({"url": boot["service_url"], "ca_sha256": boot["ca_sha256"],
                                                        "key": key, "fingerprint": fp}).encode())
        ok = peer_wait_local(f"sdk-{model}.done", timeout_s=max(60.0, (minutes_left() - 3) * 60))
        print(f"sdk {model}: client {'done' if ok else 'did not finish before the TTL'}", flush=True)
        return 0 if ok else 1
    finally:
        stop.set()
        sampler.join(timeout=10)
        prov.terminate()
        try:
            prov.wait(timeout=60)
        except subprocess.TimeoutExpired:
            prov.kill()
        for name in ("service.log",):
            p = data / "logs" / name
            if p.exists():
                (res / f"{model}-{name}").write_bytes(p.read_bytes()[-20_000_000:])


def cpu_sdk(step: dict[str, Any], res: Path, py: str) -> int:
    model = step["model"]
    raw = peer_get(f"sdk-{model}.json", timeout_s=max(60.0, (minutes_left() - 10) * 60))
    if raw is None:
        print(f"sdk {model}: the provider never published its key", flush=True)
        return 1
    info = json.loads(raw)
    env = dict(os.environ, PGW_API_KEY=info["key"])
    for k in [k for k in env if k.startswith("PGW_JOB_")]:
        env.pop(k)
    _run([py, "deploy/runpod/rtt_probe.py", info["url"], "--ca-sha256", info["ca_sha256"], "-n", "200"],
         res / f"rtt-{model}.log", env=env, timeout=300)
    from pgw.sdk.options import SUPPORTED_MODELS
    from pgw.sdk.provisioning import PINNED_MODELS

    pin = PINNED_MODELS[model].fingerprint  # checked at launch: every sdk model has a baked pin
    if info.get("fingerprint") != pin:  # the client refuses this provider (FingerprintMismatchError): a finding
        print(f"sdk {model}: WARNING the provider's pack {str(info.get('fingerprint'))[:16]}... is not the baked pin "
              f"{str(pin)[:16]}...; the client library will refuse it", flush=True)
    base = [py, "deploy/runpod/sdk_remote_bench.py", "--model", model, "--tokens", str(step.get("tokens", 128)),
            "--mask-store-gb", str(step.get("mask_store_gb", 24)), "--cache", "/workspace/pgw-client"]
    if step.get("draft_head"):  # an override (--extra); else the head the library ships pinned for the model
        base += ["--draft-head", step["draft_head"]]
    if model not in SUPPORTED_MODELS:  # pinned, not supported yet: the measurement-only preview (still pin-checked)
        base += ["--preview"]
    cmd = [*base, "--spec", step["spec"], "--conc", step["conc"], "--stages", step.get("stages", "setup,spec,conc"),
           "--out", str(res / f"sdk-{model}")]
    rc = _run(cmd, res / f"sdk-{model}.log", env=env, timeout=max(60.0, (minutes_left() - 3) * 60))
    for rr in step.get("reruns", []):  # the same cache and stock, another process with other process-wide switches
        if minutes_left() < 10:
            print(f"sdk {model} {rr['name']}: skipped ({minutes_left():.0f} min left)", flush=True)
            break
        cmd = [*base, "--spec", "off", "--conc", rr.get("conc", step["conc"]), "--stages", rr["stages"],
               "--out", str(res / f"sdk-{model}-{rr['name']}")]
        rc |= _run(cmd, res / f"sdk-{model}-{rr['name']}.log", env={**env, **rr.get("env", {})},
                   timeout=max(60.0, (minutes_left() - 3) * 60))
    peer_put_remote(f"sdk-{model}.done", str(rc).encode())
    return rc


def cpu_factory_e2e(step: dict[str, Any], res: Path, py: str) -> int:
    """Client mask CPU per slot on one core of this (server) CPU at 7B / 32B / 72B decoder-layer shapes
    (``benchmarks/pks_factory_e2e.py``), round-3 switches on and off, 1024- and 256-slot segments; runs while the GPU
    pod is still packing and loading, so it costs no GPU time. ``<res>/factory_e2e.jsonl``, one JSON line per run."""
    out = res / "factory_e2e.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_JOB_")}
    env["PYTHONPATH"] = f"{PGW / 'src'}:{PGW}"
    core = {max(os.sched_getaffinity(0))}
    rc = 0
    for model, slots, sw in step.get("runs") or FACTORY_E2E_RUNS:
        if minutes_left() < 15:
            break
        with out.open("a") as f:
            f.write(json.dumps({"run": {"model": model, "slots": slots, "switches": sw, "core": sorted(core),
                                        "cpu_model": _cpu_model()}}) + "\n")
        rc |= _run([py, "benchmarks/pks_factory_e2e.py", "--model", model, "--slots", str(slots), "--segments", "3",
                    "--prefault-gb", "2", "--tmpdir", str(res.parent)], res / "factory_e2e.log",
                   env={**env, **(MASK_ROUND3_OFF if sw == "off" else {})}, timeout=900, stdout=out, cpus=core)
    return rc


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


POD_STEPS = {("gpu", "perf"): gpu_perf, ("gpu", "lane_nic"): gpu_lane_nic, ("cpu", "lane_nic"): cpu_lane_nic,
             ("gpu", "sdk"): gpu_sdk, ("cpu", "sdk"): cpu_sdk, ("cpu", "factory_e2e"): cpu_factory_e2e}


def cmd_pod(args: argparse.Namespace) -> int:
    role = os.environ.get("PGW_JOB_ROLE", "cpu")
    res = Path(os.environ.get("PGW_JOB_RESULTS", "/workspace/job/results"))
    res.mkdir(parents=True, exist_ok=True)
    steps = json.loads(os.environ.get("PGW_JOB_PLAN", "[]"))
    py = sys.executable
    rcs: dict[str, int] = {}
    for i, step in enumerate(steps):
        fn = POD_STEPS.get((role, step["kind"]))
        name = f"{i}-{step['kind']}"
        if fn is None:
            continue
        if minutes_left() < 3:
            rcs[name] = -1
            break
        _progress(res, step=name, of=len(steps), rcs=rcs)
        t0 = time.time()
        try:
            rcs[name] = fn(step, res, py)
        except Exception as exc:  # recorded; the next step still runs
            print(f"step {name} failed: {type(exc).__name__}: {exc}", flush=True)
            rcs[name] = -2
        print(f"step {name}: rc {rcs[name]} in {(time.time() - t0) / 60:.1f} min", flush=True)
    _progress(res, step="done", of=len(steps), rcs=rcs)
    return 0 if all(v == 0 for v in rcs.values()) else 1


# ==================================================================================================== main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-key", help="read-only: balance, spend, pods, prices")
    sub.add_parser("plan", help="the sessions, minutes and $ (table prices; no API call)")
    p = sub.add_parser("launch", help="create a session's pods, upload the bundle")
    p.add_argument("session", choices=list(SESSIONS))
    p.add_argument("--dc", required=True, help="RunPod data center id (GPU and CPU pods both there)")
    p.add_argument("--max-usd", type=float, required=True, help="hard cap of this session (estimate at its TTL)")
    p.add_argument("--budget-usd", type=float, default=60.0, help="cap over every recorded session")
    p.add_argument("--ttl-min", type=float, default=None, help="default: the session's minutes + setup + margin")
    p.add_argument("--gpu", default=None)
    p.add_argument("--gpu-count", type=int, default=None)
    p.add_argument("--price-per-hour", type=float, default=None, help="USD per GPU-hour if the API has no price")
    p.add_argument("--cpu-flavor", default=launch.DEFAULT_CPU_FLAVOR)
    p.add_argument("--no-client", action="store_true")
    p.add_argument("--extra", action="append", default=[], help="file shipped as extras/<name> (the draft head)")
    p.add_argument("--start-timeout-min", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("upload", help="retry the bundle upload of a launched session")
    p.add_argument("session")
    p.add_argument("--start-timeout-min", type=float, default=20.0)
    p = sub.add_parser("status")
    p.add_argument("session", nargs="?")
    p = sub.add_parser("log")
    p.add_argument("session")
    p.add_argument("role", choices=["gpu", "cpu"])
    p.add_argument("--chars", type=int, default=6000)
    p = sub.add_parser("fetch")
    p.add_argument("session")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stop")
    p.add_argument("session", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--terminate", action="store_true")
    sub.add_parser("teardown-check", help="exit 1 while any pod runs")
    sub.add_parser("pod", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None, *, api_factory: Any = RunPodAPI) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "pod":
        return cmd_pod(args)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "log":
        return cmd_log(args)
    if args.cmd == "fetch":
        return cmd_fetch(args)
    if args.cmd == "upload":
        return upload(args.session, load_state(), timeout_min=args.start_timeout_min)
    if args.cmd == "launch" and args.dry_run:
        return cmd_launch(args, None)
    api = api_factory()
    if args.cmd == "check-key":
        return cmd_check_key(args, api)
    if args.cmd == "launch":
        return cmd_launch(args, api)
    if args.cmd == "status":
        return cmd_status(args, api)
    if args.cmd == "stop":
        if not (args.all or args.session):
            raise SystemExit("stop: name a session or --all")
        return cmd_stop(args, api)
    return cmd_teardown_check(api)


if __name__ == "__main__":
    sys.exit(main())
