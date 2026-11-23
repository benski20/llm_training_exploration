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

import json
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
    "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "qwen2.5-32b-instruct": ("Qwen/Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"),
}
MODEL_NAME = (os.environ.get("PGW_MODAL_MODEL") or "qwen2.5-0.5b-instruct").strip().lower()
if MODEL_NAME not in MODELS:
    raise ValueError(f"PGW_MODAL_MODEL={MODEL_NAME!r}: use one of {', '.join(MODELS)}")
MODEL_REPO, MODEL_REVISION = MODELS[MODEL_NAME]
#: Targets whose provider pack lives in a provider-side Volume instead of the image
#: (= pgw_modal_boot.VOLUME_PACK_MODELS, a test checks it): the 32B's 37.5 GB pack (65.5 GB download) is not baked
#: into the image (it would be rebuilt on every source change); ``prepare_provider_pack`` fills the Volume once and ``pair`` copies the pack into the
#: provider's own directory (its user's) before the service starts.
VOLUME_PACK_MODELS = ("qwen2.5-32b-instruct",)
VOLUME_PACK = MODEL_NAME in VOLUME_PACK_MODELS
PROVIDER_PACK_VOLUME = os.environ.get("PGW_MODAL_PROVIDER_PACK_VOLUME") or "pgw-provider-packs"
PROVIDER_PACK_ROOT = Path("/provider-packs")
provider_pack_volume = (modal.Volume.from_name(PROVIDER_PACK_VOLUME, create_if_missing=True) if VOLUME_PACK
                        else None)
PACK = "/srv/pgw-provider/pack" if VOLUME_PACK else "/opt/packs/" + MODEL_NAME
# Strict speculation with a draft model (launch.py --spec-draft; PGW_MODAL_DRAFT=1 at deploy time), as in
# customer.py: a pack of Qwen2.5-0.5B-Instruct the gateway runs in plaintext (here on the CPU: the gateway's
# process sees no GPU) to propose tokens. Default: no draft (prompt lookup only).
DRAFT = os.environ.get("PGW_MODAL_DRAFT", "0") == "1"
#: launch.py --draft-head PATH (deploy time): a trained target-feature draft head baked into the image
#: (public data only: trained on the public model's own outputs); read by the gateway's user.
DRAFT_HEAD_LOCAL = os.environ.get("PGW_MODAL_DRAFT_HEAD", "").strip()
DRAFT_HEAD = "/opt/pgw-modal/draft_head.pt" if DRAFT_HEAD_LOCAL else ""
DRAFT_REPO, DRAFT_REVISION = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
DRAFT_PACK = "/opt/packs/draft"  # = pgw_modal_boot.DRAFT_PACK
MAX_LIFETIME_S = 88 * 60  # hard cap (function timeout): nothing outlives 90 minutes
#: ``PGW_MODAL_APP_SUFFIX`` (deploy time, default empty): appended to every app name and to the Dict name, so
#: parallel experiments in one workspace do not replace each other's apps (the containers get it in their image env).
APP_SUFFIX = os.environ.get("PGW_MODAL_APP_SUFFIX", "").strip()
DICT_NAME = "pgw-live" + APP_SUFFIX
REGION = os.environ.get("PGW_MODAL_REGION", "us-east") or None
GPU_TIERS = ("T4", "A10G", "L40S", "H100")
GPU = os.environ.get("PGW_MODAL_GPU") or None
if GPU is not None and GPU.split(":")[0].upper() == "L4":
    raise ValueError("PGW_MODAL_GPU=L4 is not supported for the provider (no faster than a T4 for the "
                     f"bandwidth-bound decode GEMV, and dearer); use one of {GPU_TIERS}")
#: Cores per party, as in the two-container apps: the provider 2 with a GPU (4 without), the gateway 4.
PROVIDER_CPUS = float(os.environ.get("PGW_MODAL_PROVIDER_CPUS") or (2.0 if GPU else 4.0))
# the gateway runs the encrypted mask factory (encryption, decryption, every-row checks): it scales with cores
GATEWAY_CPUS = float(os.environ.get("PGW_MODAL_GATEWAY_CPUS") or 12.0)
#: Memory: the provider's 6 GiB plus the gateway's (8 GiB; 16 GiB with a target larger than the 0.5B or the draft,
#: = customer.py's MEMORY_MIB / pgw_modal_boot.customer_memory_mib).
#: Large targets, (provider GiB, gateway GiB) (= launch.py COLOCATED_MEM_GIB; a test checks it). A request, not a limit.
#: Qwen2.5-7B: the provider stages its int8 pack in host memory (16 GiB); the gateway holds its own pack (mapped), the
#: float32 embedding (2.2 GB), packs the checkpoint in the wizard and its factory reads a 34 GB encryption hint
#: through the page cache (48 GiB).
#: Qwen2.5-32B: the provider stages 31.2 GB of int8 linears (48 GiB). The gateway (160 GiB): the factory streams the
#: whole 114 GB hint once per hint pass (every <= 16 256-slot groups) and derives the Freivalds keys from the 37.5 GB
#: pack per segment, so both stay in the page cache (152 GB) instead of being re-read from disk by each worker; plus
#: the float32 embedding and LM head (6.2 GB), the factory workers' buffers and, during the e2e's plaintext
#: reference (root process), a resident int8 copy of the linears (31 GB; page cache gives way meanwhile).
COLOCATED_MEM_GIB = {"qwen2.5-7b-instruct": (16, 48), "qwen2.5-32b-instruct": (48, 160)}
_SMALL_GATEWAY_GIB = 16 if DRAFT or MODEL_NAME != "qwen2.5-0.5b-instruct" else 8
PROVIDER_GIB, GATEWAY_GIB = COLOCATED_MEM_GIB.get(MODEL_NAME, (6, _SMALL_GATEWAY_GIB))
#: ``launch.py --gateway-mem-gib`` / ``--provider-mem-gib`` (deploy time): explicit sizes (a request, not a limit), e.g.
#: the CPU-only 7B gateway with the draft, a large mask stock and several factory workers (hint pages, segments).
PROVIDER_GIB = int(os.environ.get("PGW_MODAL_PROVIDER_MEM_GIB") or PROVIDER_GIB)
GATEWAY_GIB = int(os.environ.get("PGW_MODAL_GATEWAY_MEM_GIB") or GATEWAY_GIB)
MEMORY_MIB = 1024 * (PROVIDER_GIB + GATEWAY_GIB)
#: ``launch.py --gateway-cache`` (deploy time, ``PGW_MODAL_GATEWAY_CACHE=1``; off by default): a Modal Volume of the
#: gateway side's public set-up artefacts (the model pack and the encrypted-mask hint ``H = W8 A``), mounted at
#: ``/cache`` (root only; the parties never see it): restored into the gateway's data directory before it starts and
#: filled after a run, so the 7B's ~20 min CPU hint build and its pack build happen once (``prepare_cache`` below
#: builds both on a cheap GPU container ahead of an expensive run). Contents and why they hold no customer secret:
#: deploy/modal/README.md, "Gateway cache".
GATEWAY_CACHE = os.environ.get("PGW_MODAL_GATEWAY_CACHE", "0") == "1"
CACHE_VOLUME = os.environ.get("PGW_MODAL_CACHE_VOLUME") or "pgw-gateway-cache"
CACHE_ROOT = Path("/cache")
cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True) if GATEWAY_CACHE else None
#: Disk: Modal's default ephemeral disk (at least 512 GiB, the smallest request it accepts) holds the 7B's
#: checkpoint (15 GB), two packs (10.9 GB each), the encryption hint (34 GB) and the mask store (5.6 MB per slot);
#: and the 32B's two pack copies (2 x 37.5 GB), its hint (114 GB) and a 1536-slot store (18.6 MB per slot, 29 GB):
#: about 220 GB (no checkpoint: both packs come from Volumes).
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
#: ``launch.py --gateway-gpu`` (``PGW_MODAL_GATEWAY_GPU=1`` at deploy time; needs a provider GPU): the container
#: gets two GPUs of the provider's tier. The provider's process sees only the first, the gateway's only the second
#: (``CUDA_VISIBLE_DEVICES`` by UUID; best effort, the device nodes are also given to their party's user), and the
#: gateway computes its exact local linears there (``local_linears_device = "cuda"``: the same integers, so the same
#: bits; self-checked at start-up, else the CPU). The gateway's mask factory stays on the CPU.
GATEWAY_GPU = os.environ.get("PGW_MODAL_GATEWAY_GPU", "0") == "1" and GPU is not None
GPU_SPEC = f"{GPU.split(':')[0]}:2" if GATEWAY_GPU and GPU else GPU
RUNTIME_ENV = {"PGW_MODAL_MODEL": MODEL_NAME, "PGW_MODAL_DRAFT": "1" if DRAFT else "0",
               "PGW_MODAL_DRAFT_HEAD": DRAFT_HEAD_LOCAL,
               "PGW_MODAL_SPEC_MASK": os.environ.get("PGW_MODAL_SPEC_MASK", ""),
               "PGW_MODAL_COLOCATED_USERS": "1" if SEPARATE_USERS else "0",
               "PGW_MODAL_GATEWAY_GPU": "1" if GATEWAY_GPU else "0",
               "PGW_MODAL_GPU_NODE_OWNERS": os.environ.get("PGW_MODAL_GPU_NODE_OWNERS", "1"),
               "PGW_MODAL_GATEWAY_CACHE": "1" if GATEWAY_CACHE else "0", "PGW_MODAL_CACHE_VOLUME": CACHE_VOLUME,
               "PGW_MODAL_PROVIDER_PACK_VOLUME": PROVIDER_PACK_VOLUME,
               **({"PGW_MODAL_APP_SUFFIX": APP_SUFFIX} if APP_SUFFIX else {})}
SERVICE_PORT, GATEWAY_PORT = 8443, 8080
#: The provider's TCP+TLS trip lane (``launch.py --lane tls``; 127.0.0.1 only, like the service port).
LANE_PORT = 8444
#: The provider's same-host UDS trip lane (``launch.py --lane uds``): its directory is the provider's, 0710 to the
#: gateway user's group, the socket 0660 to it; the service accepts the gateway user's uid only.
LANE_SOCKET = Path("/srv/pgw-lane/lane.sock")
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
)
if not VOLUME_PACK:  # the 32B's provider pack comes from its Volume (prepare_provider_pack), not the image
    image = image.run_commands(
        f"python -m pgw.cli pack {MODEL_REPO} {PACK} --name {MODEL_NAME} --revision {MODEL_REVISION} -q"
        " && rm -rf /root/.cache/huggingface && chmod -R a+rX /opt/packs")
if DRAFT:
    image = image.run_commands(
        f"python -m pgw.cli pack {DRAFT_REPO} {DRAFT_PACK} --name draft --revision {DRAFT_REVISION} -q"
        " && rm -rf /root/.cache/huggingface && chmod -R a+rX /opt/packs")
if DRAFT_HEAD_LOCAL and modal.is_local():  # before the users and the mounted files (a build step)
    image = image.add_local_file(DRAFT_HEAD_LOCAL, DRAFT_HEAD, copy=True).run_commands(f"chmod a+r {DRAFT_HEAD}")
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

app = modal.App("pgw-colocated" + APP_SUFFIX, image=image)


@app.function(gpu=GPU_SPEC, cpu=PROVIDER_CPUS + GATEWAY_CPUS, memory=MEMORY_MIB, timeout=MAX_LIFETIME_S,
              max_containers=1, min_containers=0, retries=0, region=REGION,
              volumes={**({str(CACHE_ROOT): cache_volume} if cache_volume is not None else {}),
                       **({str(PROVIDER_PACK_ROOT): provider_pack_volume} if provider_pack_volume is not None else {})})
def pair(run_id: str, ttl_s: int = 80 * 60, e2e: bool = False, local_linears: str = "auto",
         spec_rows: int = 0, lane: str = "stream", gateway_env: dict | None = None,
         provider_env: dict | None = None, sustain_s: float = 0.0) -> dict:
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
    result: dict = {"run_id": run_id, "placement": "colocated", "separate_users": SEPARATE_USERS, "lane": lane,
                    "gateway_env": dict(gateway_env or {}), "provider_env": dict(provider_env or {}),
                    "cpus": os.cpu_count(), "affinity_cpus": len(os.sched_getaffinity(0)),
                    "torch_threads": torch.get_num_threads()}  # the reference's float thread count
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
    # launch.py --float-threads N: the pinned float formulation. The gateway's process runs torch on N threads
    # (OMP_NUM_THREADS) and the e2e's plaintext reference on the same N, so thread-count-dependent float kernels
    # (CPU SDPA) compute the same way in both; the gateway's float_threads self-check then compares N with N.
    pin = int((gateway_env or {}).get("OMP_NUM_THREADS") or 0) or None
    result["float_threads_pin"] = pin
    # launch.py --float-ops native: the gateway's pinned native float path (PGW_FLOAT_OPS); the reference follows it
    float_ops = (gateway_env or {}).get("PGW_FLOAT_OPS") or None
    result["float_ops"] = float_ops or "torch"
    # launch.py --masked-head: the LM head as a masked linear (head_mode = "provider_int8"); the reference follows it
    masked_head = (gateway_env or {}).get("PGW_MODAL_MASKED_HEAD", "0") == "1"
    result["masked_head"] = masked_head
    # launch.py --schedule fused2: two provider trips per layer (pair items on the lane, fused2 pair masks from the
    # store); the reference follows it (the fused2 plaintext reference on the same public weights)
    f2 = (gateway_env or {}).get("PGW_MODAL_FUSED2", "").strip()
    fused2_bits = int(f2) if f2 else None
    result["schedule"] = "fused2" if fused2_bits is not None else "standard"
    # launch.py --gateway-gpu: one GPU per party (provider: the first, gateway: the second), else the gateway has none
    gsplit: dict = {}
    if os.environ.get("PGW_MODAL_GATEWAY_GPU", "0") == "1":
        gsplit = boot.split_gpus(puser, cuser)
        result["gpu_split"] = gsplit.get("report")
        boot.log(f"GPU split: {gsplit.get('report')}")
    genv = dict(gateway_env or {})
    if GATEWAY_CACHE:  # --gateway-cache: the pack and hint from the gateway side's cache (public artefacts)
        try:
            CACHE_ROOT.chmod(0o700)
        except OSError:
            pass
        t_c = time.time()
        try:
            result["cache_restore"] = boot.restore_cache(CACHE_ROOT, MODEL_NAME, croot / "data")
        except Exception as exc:  # a cache problem never stops the run: the wizard builds everything itself
            result["cache_restore"] = {"error": f"{type(exc).__name__}: {exc}"}
        result["cache_restore_s"] = round(time.time() - t_c, 1)
        boot.log(f"gateway cache restore: {result['cache_restore']}")
        genv.setdefault("PGW_SETUP_REUSE_PACK", "1")  # the wizard uses the restored pack iff it verifies
    try:
        # launch.py --mask-source: today's MLWE factory (default) or the GPU mask service (hook; refused until wired)
        mask_source = genv.get("PGW_MODAL_MASK_SOURCE") or "factory"
        genv.update(boot.mask_source_env(mask_source))
        # the 32B: never fall back to a download + CPU hint build inside the TTL on the expensive container
        boot.require_cached_setup(MODEL_NAME, result.get("cache_restore") or {}, mask_source)
        if VOLUME_PACK:  # the provider's pack from its Volume into its own directory (owned by its user below)
            src = boot.provider_pack_path(PROVIDER_PACK_ROOT, MODEL_NAME)
            if not (src / "manifest.json").is_file():
                raise RuntimeError(f"no provider pack in Volume {PROVIDER_PACK_VOLUME} ({src}); run launch.py ... "
                                   "--prepare-cache first")
            try:
                PROVIDER_PACK_ROOT.chmod(0o700)  # root only, as /cache: neither party's process reads the Volume
            except OSError:
                pass
            pwork.mkdir(parents=True, exist_ok=True, mode=0o700)
            result["provider_pack_copy"] = boot.copy_tree(src, Path(PACK), threads=16)
            boot.log(f"provider pack copied from its Volume: {result['provider_pack_copy']}")
        # ---- provider: own user, own directory, loopback listener only (pgw_modal_boot.start_provider_party)
        # launch.py --lane tls: the service also serves its native trip lane over TLS on LANE_PORT (opt-in)
        prov = boot.start_provider_party(pwork, Path(PACK), SERVICE_PORT, device="auto", user=puser,
                                         log_dir=LOGS, note=f"modal co-located demo {run_id}",
                                         lane_port=LANE_PORT if lane == "tls" else None,
                                         # launch.py --lane uds: the product's same-host lane, gateway uid only
                                         lane_socket=LANE_SOCKET if lane == "uds" else None, lane_client_user=cuser,
                                         extra_env={**boot.COLOCATED_PROVIDER_ENV, **boot.checked_env(provider_env),
                                                    **gsplit.get("provider", {})})
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
            # --draft-backend / --spec-pipeline-steps / --spec-mask (run time, carried as PGW_MODAL_* in gateway_env)
            pipe = genv.get("PGW_MODAL_SPEC_PIPELINE_STEPS", "")
            spec = boot.spec_options(spec_rows, DRAFT_PACK if DRAFT else None, None,
                                     draft_backend=genv.get("PGW_MODAL_DRAFT_BACKEND") or None,
                                     pipeline_steps=int(pipe) if pipe.strip() else None,
                                     spec_mask=genv.get("PGW_MODAL_SPEC_MASK"),
                                     draft_head=(DRAFT_HEAD if spec_rows and DRAFT_HEAD and Path(DRAFT_HEAD).is_file()
                                                 else None))
            result["spec_options"] = spec
            # with its own GPU (gsplit), the gateway's exact local linears run there (self-checked, else the CPU)
            # and its LM head too (the CPU kernel's fixed summation order as one GPU kernel: the same logits bit
            # for bit, self-checked at start-up, else the CPU; PGW_LM_HEAD_DEVICE=cpu via --gateway-env turns it off)
            gdev = {"local_linears_device": "cuda", "lm_head_device": "cuda"} if gsplit.get("gateway") else {}
            if masked_head:  # no float head on the gateway at all: the head is one more masked linear
                gdev = {k: v for k, v in gdev.items() if k != "lm_head_device"}
                gdev["head_mode"] = "provider_int8"
            if fused2_bits is not None:  # schedule = "fused2" (pgw_modal_boot.fused2_model_options)
                gdev.update(boot.fused2_model_options(fused2_bits, lane))
            gw, token = boot.start_gateway_party(croot, GATEWAY_PORT, user=cuser, log_dir=LOGS,
                                                 model_options={**boot.GATEWAY_MODEL_OPTIONS,
                                                                "local_linears": local_linears, **spec,
                                                                # --lane uds: the native UDS channel unless
                                                                # --lane-python (PGW_MODAL_LANE_NATIVE=0)
                                                                **boot.lane_options(lane, native=genv.get(
                                                                    "PGW_MODAL_LANE_NATIVE", "1") != "0"),
                                                                **gdev},
                                                 egress=boot.lane_egress(lane, LANE_PORT),
                                                 hide_gpu=not gsplit.get("gateway"),
                                                 extra_env={**boot.colocated_gateway_env(local_linears,
                                                                                        MODEL_NAME),
                                                            **gsplit.get("gateway", {}),
                                                            **boot.checked_env(genv)})
            procs.append(gw)
            info = {"run_id": run_id, "console_url": gurl.rstrip("/") + "/console/", "base_url": gurl,
                    "setup_token": token, "started_at": time.time(), "expires_at": deadline,
                    "network": "colocated"}
            live["customer"] = info
            boot.log(f"gateway in setup mode at {info['console_url']} (user {cuser or 'root'})")
            if e2e:
                import e2e_driver

                sampler = boot.CpuSampler()
                private: dict = {}  # the API key, admin token and first answer, for an in-run plain phase

                result["e2e"] = e2e_driver.run(live, run_id, gurl, token, deadline,
                                               f"http://127.0.0.1:{GATEWAY_PORT}", croot / "config",
                                               spec_rows=spec_rows, sustain_s=sustain_s,
                                               sampler_path=(gateway_env or {}).get("PGW_STACK_SAMPLER"),
                                               float_threads=pin, float_ops=float_ops, masked_head=masked_head,
                                               # --gateway-env PGW_MODAL_AB=base,pin,...: in-run A/B diagnostics
                                               ab=boot.ab_phases((gateway_env or {}).get("PGW_MODAL_AB", "")),
                                               ab_cleanup=boot.ab_cleanup,
                                               # --gateway-env PGW_MODAL_SUSTAIN_MIN_STOCK=N: the sustain window
                                               # starts once N mask slots are in stock (at most 240 s of waiting)
                                               sustain_min_stock=int((gateway_env or {}).get(
                                                   "PGW_MODAL_SUSTAIN_MIN_STOCK") or 0),
                                               private=private,
                                               # --gateway-env PGW_MODAL_REF_INT_THREADS=N: the plaintext
                                               # reference's exact integer products on N threads (same bits)
                                               ref_int_threads=int(genv.get("PGW_MODAL_REF_INT_THREADS") or 0)
                                               or None, fused2_bits=fused2_bits)
                if sustain_s and result["e2e"].get("sustain"):
                    sus = result["e2e"]["sustain"]
                    sus["factory"] = boot.factory_segments(LOGS / "gateway.log", sus["t0"], sus["t1"])
                    sus["keep_up"] = boot.keep_up(sus, sus["factory"])
                    sus["cores"] = sampler.cores(sus["t0"], sus["t1"])
                    boot.per_trip_cpu(sus)
                # --ab-plain: the same run without speculation (gateway restarted on the same store and keys)
                ab_plain = genv.get("PGW_MODAL_AB_PLAIN", "").strip()
                if ab_plain and spec_rows and private:
                    models_file = croot / "config" / "models.json"
                    holder = {"gw": gw}

                    def restart() -> dict:
                        rep = {"dropped": boot.drop_speculation(models_file)}
                        old = holder["gw"]
                        holder["gw"] = boot.restart_gateway(old, GATEWAY_PORT, LOGS)
                        procs[procs.index(old)] = holder["gw"]
                        return rep

                    try:
                        pp = e2e_driver.plain_phase(
                            f"http://127.0.0.1:{GATEWAY_PORT}", private["api_key"], private["admin"],
                            result["e2e"]["model"], private["ref_out"], restart, sustain_s=float(ab_plain),
                            sustain_min_stock=int(genv.get("PGW_MODAL_SUSTAIN_MIN_STOCK") or 0),
                            sampler_path=genv.get("PGW_STACK_SAMPLER"))
                        psus = pp.get("sustain")
                        if psus:
                            psus["factory"] = boot.factory_segments(LOGS / "gateway.log", psus["t0"], psus["t1"])
                            psus["keep_up"] = boot.keep_up(psus, psus["factory"])
                            psus["cores"] = sampler.cores(psus["t0"], psus["t1"])
                            boot.per_trip_cpu(psus)
                        result["e2e"]["plain"] = pp
                    except Exception as exc:
                        result["e2e"]["plain"] = {"error": f"{type(exc).__name__}: {exc}",
                                                  "traceback": traceback.format_exc()[-2000:]}
                    gw = holder["gw"]
                # every factory segment of the run (rates while alone, during the chats; yielded seconds)
                result["e2e"]["factory_all"] = boot.factory_segments(LOGS / "gateway.log", t_boot, time.time())
                w = result["e2e"].get("warm_window")
                if w:
                    result["e2e"]["warm_cores"] = sampler.cores(w[0] - 0.6, w[1] + 0.6)
                sampler.stop()
                spath = (gateway_env or {}).get("PGW_STACK_SAMPLER")
                if spath and Path(spath).exists():  # --gateway-env PGW_STACK_SAMPLER=...: the sampled stacks
                    result["stack_sample"] = json.loads(Path(spath).read_text())
                result["ok"] = (result["e2e"]["compare"]["logprobs_bit_identical"]
                                and result["e2e"]["compare"]["text_equal"]
                                and result["e2e"].get("stream_identical") is not False
                                and all(r.get("identical", True) for r in result["e2e"].get("ab") or [])
                                # --sustain-s: every back-to-back chat gave the same streamed answer
                                and (result["e2e"].get("sustain") or {}).get("identical") is not False
                                # --ab-plain: the plain phase gave the same answer (and its sustain chats too)
                                and (result["e2e"].get("plain") or {}).get("identical", True) is not False
                                and (((result["e2e"].get("plain") or {}).get("sustain") or {}).get("identical")
                                     is not False))
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
    if GATEWAY_CACHE and genv.get("PGW_MODAL_CACHE_SAVE", "1") != "0":
        # public artefacts the cache lacks (first run of a target): copied after the run, before the parties stop
        try:
            t_c = time.time()
            result["cache_save"] = boot.save_cache(CACHE_ROOT, MODEL_NAME, croot / "data")
            if result["cache_save"]:
                cache_volume.commit()
            result["cache_save_s"] = round(time.time() - t_c, 1)
        except Exception as exc:
            result["cache_save"] = {"error": f"{type(exc).__name__}: {exc}"}
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
    _stext = slog.read_text(errors="replace") if slog.exists() else ""
    result["service_log_tail"] = _stext[-4000:]
    # the root cause of a start-up failure is often far above the traceback tail: keep the error lines too
    result["service_log_errors"] = [ln[:400] for ln in _stext.splitlines()
                                    if any(k in ln for k in (" ERROR ", " CRITICAL ", "Error: ", "Exception: ",
                                                             "SystemExit", "refus"))][-40:]
    result["reason"] = reason
    if e2e:
        live["e2e"] = {k: v for k, v in result.items() if k not in ("log_tail", "service_log_tail",
                                                                   "service_log_errors")}
    if info and (live.get("customer") or {}).get("run_id") == run_id:
        live["customer"] = {**info, "setup_token": "(expired)", "stopped_at": time.time(), "stop_reason": reason}
    if pinfo and (live.get("provider") or {}).get("run_id") == run_id:
        live["provider"] = {**pinfo, "stopped_at": time.time(), "stop_reason": reason, "code": "(expired)"}
    return result


#: One-off set-up containers (``launch.py --prepare-cache``). The 32B: 65.5 GB download + 37.5 GB pack + 114 GB hint on
#: local disk (Modal's default ephemeral disk, >= 512 GiB), more cores for packing, and a 150 min timeout (estimated
#: 25-45 min: download 2-11, pack 5-10, hint on the T4 about 12, copies into the Volume 3-11; billed by the second).
_BIG_PREP = MODEL_NAME in VOLUME_PACK_MODELS
PREP_CPUS = 16.0 if _BIG_PREP else 8.0
PREP_MEMORY_MIB = 49152 if _BIG_PREP else 32768 if MODEL_NAME == "qwen2.5-7b-instruct" else 16384
PREP_TIMEOUT_S = (150 if _BIG_PREP else 60) * 60


def _pack_from_hub(out: Path) -> dict:
    """Download the pinned revision inside this container and pack it with the same command as the image build
    (``pgw.cli pack``, default shard size and float32 plain tensors: the same fingerprint as every other pack of it),
    then delete the checkpoint. Returns seconds and bytes."""
    import shutil
    import subprocess

    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "-m", "pgw.cli", "pack", MODEL_REPO, str(out), "--name", MODEL_NAME,  # noqa: S603, S607
                    "--revision", MODEL_REVISION, "-q"], check=True)
    shutil.rmtree(Path.home() / ".cache" / "huggingface", ignore_errors=True)
    return {"seconds": round(time.time() - t0, 1),
            "bytes": sum(f.stat().st_size for f in out.rglob("*") if f.is_file())}


@app.function(gpu="T4", cpu=PREP_CPUS, memory=PREP_MEMORY_MIB, timeout=PREP_TIMEOUT_S,
              max_containers=1, min_containers=0, retries=0, region=REGION,
              volumes={str(CACHE_ROOT): cache_volume} if cache_volume is not None else {})
def prepare_cache(force: bool = False) -> dict:
    """Fill the gateway side's cache (``launch.py --prepare-cache``; needs ``--gateway-cache``) on a cheap T4
    container, before an expensive run: the target's pack (public weights; the wizard reuses it only if it verifies
    in full to the enrolled fingerprint) and the encrypted-mask hint ``H = W8 A`` built with a fresh customer-side
    seed for ``A`` on the T4 (``PGW_HINT_DEVICE=cuda``: the CPU path's bits, self-checked first; 213 s for the 7B
    against about 20 min on 16 vCPUs). No provider process runs here and no provider Volume is mounted: a
    customer-side container. The pack: copied from this image, or for a :data:`VOLUME_PACK_MODELS` target (the
    32B) downloaded from the pinned revision and packed here, as the gateway's own wizard would. Resumable: a pack
    or hint already in the cache is kept (``force`` rebuilds both); the pack is committed before the hint starts."""
    import shutil
    import sys
    import tempfile

    sys.path.insert(0, "/opt/pgw-modal")
    import pgw_modal_boot as boot

    from pgw.gateway.setup_jobs import HintBuildJob

    if cache_volume is None:
        raise RuntimeError("deploy with --gateway-cache (PGW_MODAL_GATEWAY_CACHE=1)")
    t0 = time.time()
    paths = boot.cache_paths(CACHE_ROOT, MODEL_NAME)
    out: dict = {"model": MODEL_NAME}
    if force:
        shutil.rmtree(paths["pack"].parent, ignore_errors=True)
    src_pack = Path(PACK)
    if VOLUME_PACK:
        src_pack = Path(tempfile.mkdtemp(prefix="pgw-pack-")) / MODEL_NAME
    if not (paths["pack"] / "manifest.json").is_file():
        if VOLUME_PACK:
            out["pack_build"] = _pack_from_hub(src_pack)
        paths["pack"].parent.mkdir(parents=True, exist_ok=True)
        out["pack"] = boot.copy_tree(src_pack, paths["pack"], threads=16)
        cache_volume.commit()  # a later hint failure keeps the pack
    elif VOLUME_PACK:
        src_pack = paths["pack"]  # resume: the hint from the cached pack (read from the Volume)
    work = Path(tempfile.mkdtemp(prefix="pgw-hint-"))
    if not any(d.is_dir() and (d / "meta.json").is_file() for d in paths["hemask"].glob("*")):
        os.environ["PGW_HINT_DEVICE"] = "cuda"
        job = HintBuildJob(src_pack, boot.RING, work)
        t_h = time.time()
        job.run()
        st = job.status()
        out["hint_build"] = {"phase": st.get("phase"), "error": st.get("error"), "seconds": round(time.time() - t_h, 1)}
        if st.get("phase") != "done":
            raise RuntimeError(f"hint build failed: {st.get('error')}")
        for d in work.glob("*"):
            if d.is_dir() and (d / "meta.json").is_file():
                paths["hemask"].mkdir(parents=True, exist_ok=True)
                out[f"hint {d.name}"] = boot.copy_tree(d, paths["hemask"] / d.name, threads=16)
    cache_volume.commit()
    out["seconds"] = round(time.time() - t0, 1)
    out["cache"] = {str(p.relative_to(CACHE_ROOT)): sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    for p in (paths["pack"], paths["hemask"]) if p.exists()}
    return out


@app.function(cpu=PREP_CPUS, memory=PREP_MEMORY_MIB, timeout=PREP_TIMEOUT_S, max_containers=1, min_containers=0,
              retries=0, region=REGION,
              volumes={str(PROVIDER_PACK_ROOT): provider_pack_volume} if provider_pack_volume is not None else {})
def prepare_provider_pack(force: bool = False) -> dict:
    """The provider side's one-off set-up for a :data:`VOLUME_PACK_MODELS` target (``launch.py --prepare-cache`` runs
    it next to ``prepare_cache``): download the pinned revision inside Modal, pack it (the image build's command)
    and store it in the provider's pack Volume. CPU only, no GPU; mounts only the provider's Volume (never the
    gateway cache). Kept if already there unless ``force``."""
    import shutil
    import sys
    import tempfile

    sys.path.insert(0, "/opt/pgw-modal")
    import pgw_modal_boot as boot

    if provider_pack_volume is None:
        raise RuntimeError(f"{MODEL_NAME}: its provider pack is baked into the image (not a Volume target)")
    t0 = time.time()
    dst = boot.provider_pack_path(PROVIDER_PACK_ROOT, MODEL_NAME)
    out: dict = {"model": MODEL_NAME, "volume": PROVIDER_PACK_VOLUME}
    if force:
        shutil.rmtree(dst, ignore_errors=True)
    if not (dst / "manifest.json").is_file():
        local = Path(tempfile.mkdtemp(prefix="pgw-pack-")) / MODEL_NAME
        out["pack_build"] = _pack_from_hub(local)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out["pack"] = boot.copy_tree(local, dst, threads=16)
        provider_pack_volume.commit()
    man = json.loads((dst / "manifest.json").read_text())
    out.update(fingerprint=man.get("fingerprint"), seconds=round(time.time() - t0, 1),
               bytes=sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()))
    return out
