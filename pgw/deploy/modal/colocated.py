"""Both parties in one Modal container (``pgw-colocated``): the low-latency placement of the demo.

``launch.py --placement colocated`` runs :func:`pair`: one container (the provider's GPU from
``PGW_MODAL_GPU``, or CPU) that starts

* the provider's compute service as Unix user ``pgwprov`` in ``/srv/pgw-provider`` (0700), listening
  on ``127.0.0.1`` only, with a fresh PKI for ``127.0.0.1``; the enrollment code points the gateway at
  ``https://127.0.0.1:8443`` and pins that CA;
* the customer's gateway as Unix user ``pgwcust`` in ``/srv/pgw`` (0700), in setup mode, with no GPU
  (``CUDA_VISIBLE_DEVICES=""``), its console on a public HTTPS URL (``modal.forward``) as in
  ``customer.py``.

Each round trip then crosses loopback TCP instead of Modal's public TCP tunnel (17 ms per round trip
measured on 2026-11-21) or i6pn (which did not schedule for 10-25 minutes twice). TLS stays end to end
(the gateway verifies the pinned service CA and presents the client certificate issued at enrollment),
and the two parties share no file: each has its own user, working directory, PKI, state, caches and
``HOME``; the only data between them is what the protocol sends over the socket.

Trust caveat (also in the README): the two parties share one kernel, one host and one container.
Process isolation (separate uids, no ptrace across uids, 0700 directories) keeps the provider's
process away from the gateway's files and memory, but it is not the separation of two organisations'
machines: a kernel or container escape on the provider side, or whoever controls the container (the
workspace owner, as for both apps of the two-container demo anyway), reaches both parties, and a
shared host offers cache and scheduling side channels (and, without ``hidepid``, each other's
``/proc/<pid>/stat`` CPU times). The provider's process can also reach the
gateway's listeners on loopback, with the same credentials as from the internet. The demo's ``pgw-live``
Dict is reachable from any process in a Modal container (the container's identity authenticates it),
exactly as from the two-container provider. Use this placement to measure what the protocol costs on
a fast link, not as a deployment model; the masked bytes the provider sees are the same either way.

The function writes both ``provider_boot`` and ``customer_boot`` first (one container: both parties
are scheduled together), publishes ``provider`` and ``customer`` in the Dict as the two-container
apps do, so ``launch.py`` drives it the same way, and returns the customer's result with the
service's log tail.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Inside the container this file is /root/<name>.py and the source tree is not there (nor needed).
PGW = HERE.parents[1] if modal.is_local() else Path("/opt/pgw-src")
# The served target, fixed at deploy time (launch.py --model, or PGW_MODAL_MODEL); default the 0.5B. Same table as
# provider.py / pgw_modal_boot.MODELS (a test checks it): revisions are pinned (the same bytes as provider.py).
MODELS = {
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen2.5-3b-instruct": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
}
MODEL_NAME = (os.environ.get("PGW_MODAL_MODEL") or "qwen2.5-0.5b-instruct").strip().lower()
if MODEL_NAME not in MODELS:
    raise ValueError(f"PGW_MODAL_MODEL={MODEL_NAME!r}: use one of {', '.join(MODELS)}")
MODEL_REPO, MODEL_REVISION = MODELS[MODEL_NAME]
PACK = "/opt/packs/" + MODEL_NAME
# Strict speculation with a draft model (launch.py --spec-draft; PGW_MODAL_DRAFT=1 at deploy time), as in
# customer.py: a pack of Qwen2.5-0.5B-Instruct the gateway runs in plaintext (here on the CPU: the gateway's
# process sees no GPU) to propose tokens. Default: no draft (prompt lookup only).
DRAFT = os.environ.get("PGW_MODAL_DRAFT", "0") == "1"
DRAFT_REPO, DRAFT_REVISION = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
DRAFT_PACK = "/opt/packs/draft"  # = pgw_modal_boot.DRAFT_PACK
MAX_LIFETIME_S = 88 * 60  # hard cap (function timeout): nothing outlives 90 minutes
DICT_NAME = "pgw-live"
REGION = os.environ.get("PGW_MODAL_REGION", "us-east") or None
GPU_TIERS = ("T4", "A10G", "L40S", "H100")
GPU = os.environ.get("PGW_MODAL_GPU") or None
if GPU is not None and GPU.split(":")[0].upper() == "L4":
    raise ValueError("PGW_MODAL_GPU=L4 is not supported for the provider (no faster than a T4 for the "
                     f"bandwidth-bound decode GEMV, and dearer); use one of {GPU_TIERS}")
#: Cores per party, as in the two-container apps: the provider 2 with a GPU (4 without), the gateway 4.
PROVIDER_CPUS = float(os.environ.get("PGW_MODAL_PROVIDER_CPUS") or (2.0 if GPU else 4.0))
GATEWAY_CPUS = 4.0
#: Memory: the provider's 6 GiB plus the gateway's (8 GiB; 16 GiB with a target larger than the 0.5B or the draft,
#: = customer.py's MEMORY_MIB / pgw_modal_boot.customer_memory_mib).
MEMORY_MIB = 6144 + (16384 if DRAFT or MODEL_NAME != "qwen2.5-0.5b-instruct" else 8192)
#: One Unix user per party (``pgw_modal_boot.PROVIDER_USER`` / ``CUSTOMER_USER``; created in the image).
PROVIDER_USER, CUSTOMER_USER = "pgwprov", "pgwcust"
#: Escape hatch, fixed at deploy time (``PGW_MODAL_COLOCATED_USERS=0``): run both parties as root (still
#: separate processes, working directories, PKI and state, but root's HOME and caches;
#: TLS end to end) if dropping to the party users fails in
#: Modal's container runtime (e.g. GPU device permissions). Weaker isolation; reported in the result.
SEPARATE_USERS = os.environ.get("PGW_MODAL_COLOCATED_USERS", "1") != "0"
#: Deploy-time choices the container needs at run time. Modal re-imports this file inside the container without
#: the deploy shell's environment, so MODEL_NAME / PACK would fall back to the 0.5B there and DRAFT to False (the
#: baked draft never used); the image's environment carries them instead (as RUNTIME_ENV in customer.py).
RUNTIME_ENV = {"PGW_MODAL_MODEL": MODEL_NAME, "PGW_MODAL_DRAFT": "1" if DRAFT else "0",
               "PGW_MODAL_COLOCATED_USERS": "1" if SEPARATE_USERS else "0"}
SERVICE_PORT, GATEWAY_PORT = 8443, 8080
#: The provider's TCP+TLS trip lane (``launch.py --lane tls``; 127.0.0.1 only, like the service port).
LANE_PORT = 8444
#: Both parties' logs, written by the (root) parent process: 0700 root, files 0600.
LOGS = Path("/var/log/pgw")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.15.0")  # CUDA build (PyPI default); the gateway's process sees no GPU
    .add_local_dir(PGW / "src", "/opt/pgw-src/src", copy=True, ignore=["**/__pycache__", "**/*.so"])
    .add_local_file(PGW / "pyproject.toml", "/opt/pgw-src/pyproject.toml", copy=True)
    .add_local_file(PGW / "README.md", "/opt/pgw-src/README.md", copy=True)
    .add_local_file(PGW / "deploy/hf-space/constraints.txt", "/opt/pgw-src/constraints.txt", copy=True)
    .run_commands("pip install --no-cache-dir -c /opt/pgw-src/constraints.txt "
                  "'/opt/pgw-src[service,pack,gateway,factory]'")
    .apt_install("curl", "build-essential")
    .add_local_dir(PGW / "rust", "/opt/pgw-src/rust", copy=True, ignore=["target/**", "**/target/**"])
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable",
        "cd /opt/pgw-src && . $HOME/.cargo/env && python rust/tools/build_fast.py --site && "
        "python rust/tools/build_ntt.py --site && rm -rf rust/target $HOME/.cargo/registry",
        "cd / && python -c 'from pgw.core import native; assert native.fast is not None; print(native.fast.version())'",
        "cd / && python -c 'from pgw.hemask import ntt; assert ntt.available(); print(\"pgw_ntt\", ntt.info())'",
    )
    .run_commands(f"python -m pgw.cli pack {MODEL_REPO} {PACK} --name {MODEL_NAME} --revision {MODEL_REVISION} -q"
                  " && rm -rf /root/.cache/huggingface && chmod -R a+rX /opt/packs")
)
if DRAFT:
    image = image.run_commands(
        f"python -m pgw.cli pack {DRAFT_REPO} {DRAFT_PACK} --name draft --revision {DRAFT_REVISION} -q"
        " && rm -rf /root/.cache/huggingface && chmod -R a+rX /opt/packs")
image = (
    image
    # one Unix user per party (own group, no login shell)
    .run_commands(f"useradd --system --user-group --shell /usr/sbin/nologin --no-create-home {PROVIDER_USER}",
                  f"useradd --system --user-group --shell /usr/sbin/nologin --no-create-home {CUSTOMER_USER}")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", **RUNTIME_ENV})
    .add_local_file(HERE / "pgw_modal_boot.py", "/opt/pgw-modal/pgw_modal_boot.py")
    .add_local_file(HERE / "wizard_client.py", "/opt/pgw-modal/wizard_client.py")
    .add_local_file(HERE / "trip_gate.py", "/opt/pgw-modal/trip_gate.py")
    .add_local_file(HERE / "e2e_driver.py", "/opt/pgw-modal/e2e_driver.py")
)

app = modal.App("pgw-colocated", image=image)


@app.function(gpu=GPU, cpu=PROVIDER_CPUS + GATEWAY_CPUS, memory=MEMORY_MIB, timeout=MAX_LIFETIME_S,
              max_containers=1, min_containers=0, retries=0, region=REGION)
def pair(run_id: str, ttl_s: int = 80 * 60, e2e: bool = False, local_linears: str = "auto",
         spec_rows: int = 0, lane: str = "stream") -> dict:
    """Provider service and customer gateway in this container, over loopback TLS (module docstring)."""
    import sys

    t_boot = time.time()
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    # First action once scheduled: both parties run now (launch.py's scheduling fallback watches these).
    live["provider_boot"] = {"run_id": run_id, "at": t_boot, "placement": "colocated"}
    live["customer_boot"] = {"run_id": run_id, "at": t_boot, "placement": "colocated"}
    sys.path.insert(0, "/opt/pgw-modal")
    import pgw_modal_boot as boot
    import torch

    deadline = t_boot + min(ttl_s, MAX_LIFETIME_S - 120)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    boot.log(f"co-located pair: torch {torch.__version__}, GPU {gpu}")
    result: dict = {"run_id": run_id, "placement": "colocated", "separate_users": SEPARATE_USERS, "lane": lane}
    puser, cuser = (PROVIDER_USER, CUSTOMER_USER) if SEPARATE_USERS else (None, None)
    if not SEPARATE_USERS:
        boot.log("PGW_MODAL_COLOCATED_USERS=0: both parties run as root (separate processes and directories)")
    procs: list = []
    reason = "ttl"
    info: dict = {}
    pinfo: dict = {}
    pwork = Path("/srv/pgw-provider")
    croot = Path("/srv/pgw")
    token = ""
    try:
        # ---- provider: own user, own directory, loopback listener only (pgw_modal_boot.start_provider_party)
        # launch.py --lane tls: the service also serves its native trip lane over TLS on LANE_PORT (opt-in)
        prov = boot.start_provider_party(pwork, Path(PACK), SERVICE_PORT, device="auto", user=puser,
                                         log_dir=LOGS, note=f"modal co-located demo {run_id}",
                                         lane_port=LANE_PORT if lane == "tls" else None)
        svc = prov["proc"]
        procs.append(svc)
        url, code = prov["url"], prov["code"]
        pinfo = {"run_id": run_id, "service_url": url, "code": code["code"], "code_id": code["id"],
                 "code_expires_at": code["expires_at"], "tenant": "customer-demo", "model": MODEL_NAME,
                 "ca_sha256": prov["ca_sha256"], "started_at": time.time(), "expires_at": deadline, "gpu": gpu,
                 "cpus": PROVIDER_CPUS, "network": "colocated", "network_path": boot.network_path(url)}
        live["provider"] = pinfo

        # ---- customer gateway: own user, own directory, no GPU (pgw_modal_boot.start_gateway_party)
        with modal.forward(GATEWAY_PORT) as tunnel:
            gurl = tunnel.url
            # launch.py --spec-rows: strict speculation (prompt lookup; plus the image's draft with --spec-draft,
            # on the CPU: the gateway's process sees no GPU)
            spec = boot.spec_options(spec_rows, DRAFT_PACK if DRAFT else None, None)
            gw, token = boot.start_gateway_party(croot, GATEWAY_PORT, user=cuser, log_dir=LOGS,
                                                 model_options={**boot.GATEWAY_MODEL_OPTIONS,
                                                                "local_linears": local_linears, **spec,
                                                                **boot.lane_options(lane)},
                                                 egress=boot.lane_egress(lane, LANE_PORT))
            procs.append(gw)
            info = {"run_id": run_id, "console_url": gurl.rstrip("/") + "/console/", "base_url": gurl,
                    "setup_token": token, "started_at": time.time(), "expires_at": deadline,
                    "network": "colocated"}
            live["customer"] = info
            boot.log(f"gateway in setup mode at {info['console_url']} (user {cuser or 'root'})")
            if e2e:
                import e2e_driver

                result["e2e"] = e2e_driver.run(live, run_id, gurl, token, deadline,
                                               f"http://127.0.0.1:{GATEWAY_PORT}", croot / "config",
                                               spec_rows=spec_rows)
                result["ok"] = (result["e2e"]["compare"]["logprobs_bit_identical"]
                                and result["e2e"]["compare"]["text_equal"])
                reason = "e2e finished"
            else:
                while time.time() < deadline:
                    if svc.poll() is not None:
                        reason = f"service exited ({svc.returncode})"
                        break
                    if gw.poll() is not None:
                        reason = f"gateway exited ({gw.returncode})"
                        break
                    if live.get("stop") in (run_id, "all"):
                        reason = "stop requested"
                        break
                    time.sleep(10)
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-3000:]
        reason = "error"
    boot.log(f"shutting down: {reason}")
    for p in reversed(procs):
        p.terminate()
        try:
            p.wait(40)
        except Exception:
            p.kill()
    glog = LOGS / "gateway.log"
    slog = LOGS / "service.log"
    gtail = glog.read_text(errors="replace") if glog.exists() else ""
    result["log_tail"] = (gtail.replace(token, "<setup-token>") if token else gtail)[-6000:]
    result["service_log_tail"] = slog.read_text(errors="replace")[-4000:] if slog.exists() else ""
    result["reason"] = reason
    if e2e:
        live["e2e"] = {k: v for k, v in result.items() if k not in ("log_tail", "service_log_tail")}
    if info and (live.get("customer") or {}).get("run_id") == run_id:
        live["customer"] = {**info, "setup_token": "(expired)", "stopped_at": time.time(), "stop_reason": reason}
    if pinfo and (live.get("provider") or {}).get("run_id") == run_id:
        live["provider"] = {**pinfo, "stopped_at": time.time(), "stop_reason": reason, "code": "(expired)"}
    return result
