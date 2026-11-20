"""Party 1 on Modal: the compute provider (``pgw-provider``; CPU by default, a GPU with ``PGW_MODAL_GPU=T4``).

The image installs pgw from this repository and packs Qwen2.5-0.5B-Instruct once at build time.
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
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"  # pinned: both parties must pack the same bytes
MODEL_NAME = "qwen2.5-0.5b-instruct"
PACK = "/opt/packs/" + MODEL_NAME
MAX_LIFETIME_S = 88 * 60  # hard cap (function timeout): nothing outlives 90 minutes
DICT_NAME = "pgw-live"
# Both parties in one region: every token makes ~100 sequential round trips between them, so a
# cross-region placement (about 85 ms RTT seen) makes decoding 10x slower. PGW_MODAL_REGION="" lifts it.
REGION = os.environ.get("PGW_MODAL_REGION", "us-east") or None
# Private IPv6 networking between the two containers (same workspace, same region): the gateway reaches
# the service directly instead of through the public TCP tunnel, which added about 5 ms to each of the
# ~100 round trips per token. TLS stays end to end. PGW_MODAL_I6PN=0 uses the tunnel only.
I6PN = os.environ.get("PGW_MODAL_I6PN", "1") != "0"
# GPU type for the service, e.g. PGW_MODAL_GPU=T4 at deploy time. Default: none (CPU int8 backend), because
# Modal only allows GPU functions on workspaces with a payment method. The image has CUDA torch either way.
GPU = os.environ.get("PGW_MODAL_GPU") or None
CPUS = 1.0 if GPU else 4.0

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
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(HERE / "pgw_modal_boot.py", "/opt/pgw-modal/pgw_modal_boot.py")
)

app = modal.App("pgw-provider", image=image)


@app.function(gpu=GPU, cpu=CPUS, memory=6144, timeout=MAX_LIFETIME_S, max_containers=1, min_containers=0,
              retries=0, region=REGION, i6pn=I6PN)
def serve(run_id: str, ttl_s: int = 80 * 60, tenant: str = "customer-demo", code_ttl: str = "90m") -> dict:
    import sys

    sys.path.insert(0, "/opt/pgw-modal")
    import pgw_modal_boot as boot
    import torch

    deadline = time.time() + min(ttl_s, MAX_LIFETIME_S - 120)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    boot.log(f"torch {torch.__version__}, GPU {gpu}, archs {torch.cuda.get_arch_list()}")
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    work = Path("/srv/pgw-provider")
    port = 8443
    with modal.forward(port, unencrypted=True) as tunnel:
        host, tport = tunnel.tcp_socket
        url = f"https://{host}:{tport}"
        boot.log(f"public TCP tunnel {host}:{tport} -> :{port} (TLS end to end; Modal does not terminate it)")
        names = [host, "localhost", "127.0.0.1", "::1"]
        v6 = boot.i6pn_address() if I6PN else None
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
                "started_at": time.time(), "expires_at": deadline}
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
