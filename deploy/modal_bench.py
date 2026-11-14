"""The verifier's test bench, hosted on Modal: the page and API of `python -m vt.verify_service --engine lettuce
--demo` behind a public URL, on CPU only (no GPU), scaled to zero when idle.

    modal deploy deploy/modal_bench.py          # prints the URL; the deploy workflow's "bench" action runs this
    modal app stop verify-bench                 # take it down

Open (no API keys): it is a demo, and every visitor shares one calibration. Requests are rate-limited. LettuceDetect
is baked into the image at its pinned revision. Cost while up: 4 CPU cores and 6 GB memory (about $0.25 an hour on
Modal's list prices); nothing after SCALEDOWN idle seconds; a cold start loads the model in about a minute.
"""
import os
import subprocess

import modal

PORT = 8000
SCALEDOWN = int(os.environ.get("BENCH_SCALEDOWN", "300"))
LETTUCE_REV = "22296c700ef0ba4ab3e5c9afffa0185caaf61e52"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.22.0", "huggingface_hub==2.4.0", "numpy==2.5.3", "scipy==1.18.1",
                 "scikit-learn==1.9.1", "lettucedetect==0.2.3", "openai==3.50.0", "rag-fact-checker==0.1.5")
    .env({"HF_HOME": "/opt/hf", "PYTHONUNBUFFERED": "1", "MALLOC_ARENA_MAX": "2"})
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download("
                  f"'KRLabsOrg/lettucedect-large-modernbert-en-v1', revision='{LETTUCE_REV}')\"")
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .add_local_dir(os.path.join(ROOT, "vt"), "/app/vt")
    .add_local_dir(os.path.join(ROOT, "examples", "verify_smoke"), "/app/samples")
)
app = modal.App("verify-bench", image=image)


@app.function(cpu=4.0, memory=6144, timeout=24 * 3600, scaledown_window=SCALEDOWN, max_containers=1)
@modal.concurrent(max_inputs=16)
@modal.web_server(PORT, startup_timeout=600)
def serve():
    subprocess.Popen(["python", "-m", "vt.verify_service", "--engine", "lettuce", "--threads", "4", "--host", "0.0.0.0",
                      "--port", str(PORT), "--rate-per-minute", "60", "--demo", "/app/samples"],
                     env=dict(os.environ, PYTHONPATH="/app"))
