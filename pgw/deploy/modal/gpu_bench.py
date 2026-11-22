"""One-off GPU benchmarks on Modal (``pgw-gpu-bench``): run a benchmark command in a GPU container.

Used instead of Kaggle when its GPU queue is blocked. Runs ``benchmarks/*.py`` from this checkout in an
image with the native modules built, on a chosen GPU and CPU count, and returns stdout plus any JSON the
command wrote. Nothing is deployed: ``modal run`` starts the app, runs, and stops it (pay per second).

    modal run deploy/modal/gpu_bench.py --gpu T4 --cpus 16 \\
        --cmd "python benchmarks/he_factory_overlap_bench.py --preset qwen2.5-0.5b ... --json /tmp/out.json"

Every call has a hard timeout (``--timeout-min``, default 40) so a hung run cannot burn credit.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PGW = HERE.parents[1] if modal.is_local() else Path("/opt/pgw-src")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.15.0")
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
    )
    .add_local_dir(PGW / "benchmarks", "/opt/pgw-src/benchmarks", ignore=["**/__pycache__", "results/**", "kaggle/**"])
)
app = modal.App("pgw-gpu-bench", image=image)


def _run(cmd: str, timeout_min: float) -> dict:
    t = time.time()
    env = {**os.environ, "PYTHONPATH": "/opt/pgw-src/src", "PGW_OFFLINE": "1"}
    try:
        p = subprocess.run(["bash", "-lc", cmd], cwd="/opt/pgw-src", env=env, capture_output=True, text=True,
                           timeout=timeout_min * 60)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        rc, out, err = -9, (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""), "TIMEOUT"
    files = {}
    for tok in shlex.split(cmd):
        if tok.endswith(".json") and Path(tok).is_file():
            files[tok] = json.loads(Path(tok).read_text())
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"cmd": cmd, "rc": rc, "wall_s": round(time.time() - t, 1), "cpus": os.cpu_count(), "gpu": gpu,
            "stdout_tail": out[-6000:], "stderr_tail": err[-3000:], "json": files}


_KW = dict(memory=32768, timeout=45 * 60, region=os.environ.get("PGW_MODAL_REGION", "us-east") or None)


@app.function(gpu="T4", cpu=4.0, **_KW)
def t4_c4(cmd: str, timeout_min: float = 40) -> dict:
    return _run(cmd, timeout_min)


@app.function(gpu="T4", cpu=16.0, **_KW)
def t4_c16(cmd: str, timeout_min: float = 40) -> dict:
    return _run(cmd, timeout_min)


@app.function(gpu="H100", cpu=16.0, **_KW)
def h100_c16(cmd: str, timeout_min: float = 40) -> dict:
    return _run(cmd, timeout_min)


@app.local_entrypoint()
def main(cmd: str, target: str = "t4_c16", timeout_min: float = 40, out: str = "") -> None:
    fn = {"t4_c4": t4_c4, "t4_c16": t4_c16, "h100_c16": h100_c16}[target]
    res = fn.remote(cmd, timeout_min)
    text = json.dumps(res, indent=1)
    if out:
        Path(out).write_text(text)
    print(text[-8000:])
