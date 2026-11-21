"""Party 1 on Modal: the compute provider (``pgw-provider``; CPU by default, a GPU with ``launch.py --gpu``).

The image installs pgw from this repository and packs the target once at build time (Qwen2.5-0.5B-Instruct by
default; ``launch.py --model`` / ``PGW_MODAL_MODEL`` picks a larger Qwen2.5 target for strict speculation).
:func:`serve` opens a raw TCP tunnel (``modal.forward(..., unencrypted=True)``: Modal forwards bytes and
does not terminate TLS, so TLS runs end to end between the customer's gateway and this service),
creates a fresh PKI for the tunnel's host name, starts ``pgw service serve`` with production settings,
issues a single-use enrollment code and publishes ``{service_url, code}`` in the ``pgw-live`` Dict.
It stops after ``ttl_s`` seconds (and in any case at the function timeout) or when asked via the Dict.

Use ``launch.py`` (one command for both parties) rather than calling this directly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Inside the container this file is /root/<name>.py and the source tree is not there (nor needed).
PGW = HERE.parents[1] if modal.is_local() else Path("/opt/pgw-src")
# The served target, fixed at deploy time (launch.py --model, or PGW_MODAL_MODEL); default the 0.5B. Same table as
# pgw_modal_boot.MODELS (a test checks it): revisions are pinned because both parties must pack the same bytes.
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
#: Deploy-time choices the container needs at run time. Modal re-imports this file inside the container without
#: the deploy shell's environment, so MODEL_NAME (hence PACK and the "model" published to the customer) would fall
#: back to the 0.5B there; the image's environment carries them instead.
RUNTIME_ENV = {"PGW_MODAL_MODEL": MODEL_NAME}
MAX_LIFETIME_S = 88 * 60  # hard cap (function timeout): nothing outlives 90 minutes
DICT_NAME = "pgw-live"
# Both parties in one region: every token makes ~100 sequential round trips between them, so a
# cross-region placement (about 85 ms RTT seen) makes decoding 10x slower. PGW_MODAL_REGION="" lifts it.
REGION = os.environ.get("PGW_MODAL_REGION", "us-east") or None
# Private IPv6 networking between the two containers (same workspace, same region): the gateway reaches
# the service directly instead of through the public TCP tunnel, which added about 5 ms to each of the
# ~100 round trips per token. TLS stays end to end. Both variants are deployed (``serve`` = tunnel,
# ``serve_i6pn`` = i6pn); launch.py uses the tunnel unless asked (--i6pn or PGW_MODAL_I6PN=1), because
# with i6pn Modal could not schedule a CPU container in us-east for 25 minutes on 2026-11-20, and it
# falls back to the tunnel when an i6pn pair is not scheduled in time.
# GPU type for the service, set at deploy time (launch.py --gpu, or PGW_MODAL_GPU). Default: none (CPU int8
# backend), because Modal only allows GPU functions on workspaces with a payment method. The image has
# CUDA torch either way. Tiers: T4 (cheapest), A10G or L40S (default choice for speed), H100 (latency
# tier). Never L4: its memory bandwidth (300 GB/s) is no better than a T4's (320 GB/s) at a higher price,
# and a single-row masked GEMV is bandwidth bound.
GPU_TIERS = ("T4", "A10G", "L40S", "H100")
GPU = os.environ.get("PGW_MODAL_GPU") or None
if GPU is not None and GPU.split(":")[0].upper() == "L4":
    raise ValueError("PGW_MODAL_GPU=L4 is not supported for the provider (no faster than a T4 for the "
                     f"bandwidth-bound decode GEMV, and dearer); use one of {GPU_TIERS}")
# With a GPU, 2 cores: one for the service's device worker and one for the network/event loop, so the
# trip path never waits for a core (Modal bills about $0.05/h per core).
CPUS = float(os.environ.get("PGW_MODAL_PROVIDER_CPUS") or (2.0 if GPU else 4.0))

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.15.0")  # CUDA build (PyPI default)
    .add_local_dir(PGW / "src", "/opt/pgw-src/src", copy=True, ignore=["**/__pycache__", "**/*.so"])
    .add_local_file(PGW / "pyproject.toml", "/opt/pgw-src/pyproject.toml", copy=True)
    .add_local_file(PGW / "README.md", "/opt/pgw-src/README.md", copy=True)
    .add_local_file(PGW / "deploy/hf-space/constraints.txt", "/opt/pgw-src/constraints.txt", copy=True)
    .run_commands("pip install --no-cache-dir -c /opt/pgw-src/constraints.txt '/opt/pgw-src[service,pack]'")
    # native gateway kernels (pgw_fast); without them pgw runs its much slower pure-Python path
    .apt_install("curl", "build-essential")
    .add_local_dir(PGW / "rust", "/opt/pgw-src/rust", copy=True, ignore=["target/**", "**/target/**"])
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable",
        "cd /opt/pgw-src && . $HOME/.cargo/env && python rust/tools/build_fast.py --site && "
        "rm -rf rust/target $HOME/.cargo/registry",
        "cd / && python -c 'from pgw.core import native; assert native.fast is not None; print(native.fast.version())'",
    )
    .run_commands(f"python -m pgw.cli pack {MODEL_REPO} {PACK} --name {MODEL_NAME} --revision {MODEL_REVISION} -q"
                  " && rm -rf /root/.cache/huggingface")
    .env({"PYTHONUNBUFFERED": "1", **RUNTIME_ENV})
    .add_local_file(HERE / "pgw_modal_boot.py", "/opt/pgw-modal/pgw_modal_boot.py")
)

app = modal.App("pgw-provider", image=image)


_FN: dict = dict(gpu=GPU, cpu=CPUS, memory=6144, timeout=MAX_LIFETIME_S, max_containers=1, min_containers=0,
                 retries=0, region=REGION)


@app.function(**_FN, i6pn=False)
def serve(run_id: str, ttl_s: int = 80 * 60, tenant: str = "customer-demo", code_ttl: str = "90m") -> dict:
    """The compute provider, reached through its public TCP tunnel."""
    return _serve(run_id, ttl_s, tenant, code_ttl, i6pn=False)


@app.function(**_FN, i6pn=True)
def serve_i6pn(run_id: str, ttl_s: int = 80 * 60, tenant: str = "customer-demo", code_ttl: str = "90m") -> dict:
    """The same provider on Modal's private IPv6 network: the enrollment code points at its i6pn address
    (pairs with the customer's ``gateway_i6pn``; the tunnel stays open as well)."""
    return _serve(run_id, ttl_s, tenant, code_ttl, i6pn=True)


def _serve(run_id: str, ttl_s: int, tenant: str, code_ttl: str, *, i6pn: bool) -> dict:
    import sys

    t_boot = time.time()
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    # First action once Modal has scheduled this container (launch.py's i6pn scheduling fallback watches it).
    live["provider_boot"] = {"run_id": run_id, "at": t_boot, "i6pn": i6pn}
    sys.path.insert(0, "/opt/pgw-modal")
    import pgw_modal_boot as boot
    import torch

    deadline = t_boot + min(ttl_s, MAX_LIFETIME_S - 120)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    boot.log(f"torch {torch.__version__}, GPU {gpu}, archs {torch.cuda.get_arch_list()}")
    work = Path("/srv/pgw-provider")
    port = 8443
    with modal.forward(port, unencrypted=True) as tunnel:
        host, tport = tunnel.tcp_socket
        url = f"https://{host}:{tport}"
        boot.log(f"public TCP tunnel {host}:{tport} -> :{port} (TLS end to end; Modal does not terminate it)")
        names = [host, "localhost", "127.0.0.1", "::1"]
        v6 = boot.i6pn_address() if i6pn else None
        if v6:
            names.append(v6)
            url = f"https://[{v6}]:{port}"
            boot.log(f"i6pn private address {v6}: the enrollment code points the gateway there (no tunnel hop)")
        pki = boot.make_pki(work / "pki", names)
        env = boot.service_env(work, Path(PACK), pki, url, port, device="auto")
        if v6:
            env["PGW_SERVICE_HOST"] = "::"  # dual stack: the tunnel (IPv4 loopback) and i6pn (IPv6)
        svc = boot.start_service(env, work / "service.log")
        t0 = time.time()
        boot.wait_ready(f"https://127.0.0.1:{port}/readyz", svc, verify=str(pki["service_ca"]), timeout_s=900)
        boot.log(f"service ready in {time.time() - t0:.0f}s")
        code = boot.create_enrollment_code(env, tenant, code_ttl, f"modal demo {run_id}")
        from pgw.enroll.service import ca_fingerprint, load_ca_pem

        info = {"run_id": run_id, "service_url": url, "code": code["code"], "code_id": code["id"],
                "code_expires_at": code["expires_at"], "tenant": tenant, "model": MODEL_NAME,
                "ca_sha256": ca_fingerprint(load_ca_pem(pki["service_ca"])).hex(),
                "started_at": time.time(), "expires_at": deadline, "gpu": gpu, "cpus": CPUS,
                "network": "i6pn" if v6 else "tunnel"}
        live["provider"] = info
        boot.log(f"published service URL and enrollment code (id {code['id']}) to Dict {DICT_NAME!r}")
        reason = "ttl"
        while time.time() < deadline:
            if svc.poll() is not None:
                reason = f"service exited ({svc.returncode})"
                break
            if live.get("stop") in (run_id, "all"):
                reason = "stop requested"
                break
            time.sleep(10)
        boot.log(f"shutting down: {reason}")
        svc.terminate()
        try:
            svc.wait(30)
        except Exception:
            svc.kill()
        tail = (work / "service.log").read_text(errors="replace")[-4000:]
    if live.get("provider", {}).get("run_id") == run_id:
        live["provider"] = {**info, "stopped_at": time.time(), "stop_reason": reason, "code": "(expired)"}
    return {"reason": reason, "log_tail": tail, "pid": os.getpid()}
