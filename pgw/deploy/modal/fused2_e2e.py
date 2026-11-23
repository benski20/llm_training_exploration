"""One exact fused2 e2e on Modal (opt-in; 2 provider round trips per layer): ``pgw-fused2-e2e``.

One container, two processes: the provider's fused2 pair op on the GPU (``python -m pgw.engine.fused2_remote
serve``, public weights only, masked ring elements only, a 0600 Unix socket or loopback) and the gateway
(``benchmarks/fused2_e2e.py``: the fused2 reference built at start-up from the public checkpoint, masks from an
in-process plain factory on the gateway's CPU, every trip masked, range-checked and F_(2^61-1) Freivalds-checked on
both stages, then a plain replay that must match bit for bit). It does **not** use the production gateway
runtime, mask store, encrypted factory, trip lane or native decode driver (see the fused2 plan, section 7).

Run (the lead's one 7B measurement; the image build downloads the checkpoint and packs it once, CPU only):

    PGW_F2_MODEL=qwen2.5-7b-instruct PGW_F2_GPU=H100 modal run pgw/deploy/modal/fused2_e2e.py \\
        --tokens 32 --ring K56A12C53 --migrate-bits 8

Cost: H100 about $4/h + 24 vCPU + 96 GiB; the run takes about 10-15 min (start-up: folding, spectral norms and
the check vectors about 3-6 min on the CPU, masks for ~70 slots about 1-2 min, decode, then the plain replay
about 2-4 min), so roughly $1.2-1.8. The function's timeout caps it at 40 min.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PGW = HERE.parents[1] if modal.is_local() else Path("/opt/pgw-src")
MODELS = {
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
MODEL_NAME = (os.environ.get("PGW_F2_MODEL") or "qwen2.5-7b-instruct").strip().lower()
if MODEL_NAME not in MODELS:
    raise ValueError(f"PGW_F2_MODEL={MODEL_NAME!r}: use one of {', '.join(MODELS)}")
MODEL_REPO, MODEL_REVISION = MODELS[MODEL_NAME]
GPU = os.environ.get("PGW_F2_GPU") or "H100"
CPUS = float(os.environ.get("PGW_F2_CPUS") or 24)
MEM_GIB = int(os.environ.get("PGW_F2_MEM_GIB") or (96 if "7b" in MODEL_NAME else 32))
HF_DIR = "/opt/hf/" + MODEL_NAME
PACK = "/opt/packs/" + MODEL_NAME

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.15.0")
    .add_local_dir(PGW / "src", "/opt/pgw-src/src", copy=True, ignore=["**/__pycache__", "**/*.so"])
    .add_local_file(PGW / "pyproject.toml", "/opt/pgw-src/pyproject.toml", copy=True)
    .add_local_file(PGW / "README.md", "/opt/pgw-src/README.md", copy=True)
    .add_local_file(PGW / "deploy/hf-space/constraints.txt", "/opt/pgw-src/constraints.txt", copy=True)
    .run_commands("pip install --no-cache-dir -c /opt/pgw-src/constraints.txt '/opt/pgw-src[service,pack,gateway]'")
    # the native kernels (as colocated.py): the gateway's fixed-order LM head (pgw_fast.lm_head) is the one used
    # here; without it the bit-identical torch reference head costs ~1 s per token at 7B
    .apt_install("curl", "build-essential")
    .add_local_dir(PGW / "rust", "/opt/pgw-src/rust", copy=True, ignore=["target/**", "**/target/**"])
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable",
        "cd /opt/pgw-src && . $HOME/.cargo/env && python rust/tools/build_fast.py --site && rm -rf rust/target "
        "$HOME/.cargo/registry",
        "cd / && python -c 'from pgw.core import native; assert native.fast is not None; print(native.fast.version())'",
    )
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_REPO}', "
        f"revision='{MODEL_REVISION}', local_dir='{HF_DIR}', allow_patterns=['*.json', '*.safetensors', '*.txt', "
        f"'tokenizer*', 'merges.txt', 'vocab.json'])\"",
        f"python -m pgw.cli pack {HF_DIR} {PACK} --name {MODEL_NAME} -q && chmod -R a+rX /opt/packs /opt/hf",
    )
    .run_commands("useradd --system --user-group --shell /usr/sbin/nologin --no-create-home pgwprov")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PGW_F2_MODEL": MODEL_NAME,
          "PGW_F2_GPU": GPU})
    .add_local_file(PGW / "benchmarks/fused2_e2e.py", "/opt/pgw-modal/fused2_e2e.py", copy=True)
)
app = modal.App("pgw-fused2-e2e", image=image)


@app.function(gpu=GPU, cpu=CPUS, memory=1024 * MEM_GIB, timeout=40 * 60, max_containers=1, retries=0)
def run(tokens: int = 32, ring: str = "K56A12C53", migrate_bits: int = 8, columns: int = 0, gateway_bits: int = 12,
        threads: int = 0, provider_user: bool = False, replay: bool = True, replay_calls: int = 3) -> dict:
    """Start the provider (GPU) and the gateway (CPU); returns the e2e result (JSON)."""
    out = Path("/srv/f2_result.json")
    sock_dir = Path("/srv/f2prov")
    sock_dir.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if provider_user:  # the provider's process as its own Unix user (its socket directory is its own, 0700)
        subprocess.run(["chown", "pgwprov:pgwprov", str(sock_dir)], check=True)  # noqa: S603, S607
        prefix = "runuser -u pgwprov --"
    sock_dir.chmod(0o700)
    cmd = [sys.executable, "/opt/pgw-modal/fused2_e2e.py", "--hf-dir", HF_DIR, "--pack", PACK, "--ring", ring,
           "--migrate-bits", str(migrate_bits), "--columns", str(columns), "--gateway-bits", str(gateway_bits),
           "--tokens", str(tokens), "--provider-device", "cuda", "--address", f"unix:{sock_dir / 'f2.sock'}",
           "--provider-cmd-prefix", prefix, "--json", str(out)]
    if threads:
        cmd += ["--threads", str(threads)]
    if not replay:
        cmd.append("--no-replay")
    cmd += ["--replay-calls", str(replay_calls)]
    proc = subprocess.run(cmd, check=False)  # noqa: S603
    res = json.loads(out.read_text()) if out.is_file() else {}
    res["exit_code"] = proc.returncode
    res["gpu"] = GPU
    res["cpus"] = CPUS
    return res


@app.local_entrypoint()
def main(tokens: int = 32, ring: str = "K56A12C53", migrate_bits: int = 8, columns: int = 0, gateway_bits: int = 12,
         threads: int = 0, provider_user: bool = False, replay: bool = True, replay_calls: int = 3) -> None:
    res = run.remote(tokens, ring, migrate_bits, columns, gateway_bits, threads, provider_user, replay, replay_calls)
    print(json.dumps(res, indent=1))
    Path("fused2_e2e_result.json").write_text(json.dumps(res, indent=1))
