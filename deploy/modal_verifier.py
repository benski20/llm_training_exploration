"""The verification API hosted on Modal: one GPU, the reader and its probe, keys from a Modal secret.

    modal secret create verifier-keys VERIFY_API_KEYS="acme:<key>,globex:<key>"   # once, in the Modal dashboard
    modal deploy deploy/modal_verifier.py
    curl -H "Authorization: Bearer <key>" -d '{"reply": "...", "documents": ["..."]}' https://<app>.modal.run/v1/verify
    modal app stop verifier

VERIFIER_READER / VERIFIER_PROBE / VERIFIER_GPU choose the reader, its probe (a repo path) and the GPU at deploy
time; a 4B reader fits any 16 GB GPU. Idle containers stop after VERIFIER_SCALEDOWN seconds (billing).
"""
import os
import subprocess

import modal

READER = os.environ.get("VERIFIER_READER", "Qwen/Qwen3-4B")
PROBE = os.environ.get("VERIFIER_PROBE", "heads/grounding_probe_qwen3_4b.npz")
GPU = os.environ.get("VERIFIER_GPU", "L4")
SCALEDOWN = int(os.environ.get("VERIFIER_SCALEDOWN", "120"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate==1.1.1", "numpy==2.1.3", "scipy==1.14.1",
                 "scikit-learn==1.5.2", "fastapi==0.115.6")
    .env({"VERIFIER_READER": READER, "VERIFIER_PROBE": PROBE, "VERIFIER_GPU": GPU, "VERIFIER_SCALEDOWN": str(SCALEDOWN)})
    .add_local_dir(os.path.join(ROOT, "vt"), "/app/vt")
    .add_local_file(os.path.join(ROOT, PROBE), "/app/heads/probe.npz")
)
app = modal.App("verifier", image=image)
hf_cache = modal.Volume.from_name("verichat-hf-cache", create_if_missing=True)


@app.function(gpu=GPU, cpu=2.0, volumes={"/root/.cache/huggingface": hf_cache}, timeout=24 * 3600,
              scaledown_window=SCALEDOWN, max_containers=1, secrets=[modal.Secret.from_name("verifier-keys")])
@modal.concurrent(max_inputs=16)
@modal.web_server(PORT, startup_timeout=900)
def serve():
    subprocess.Popen(["python", "-m", "vt.verify_service", "--reader", READER, "--probe", "/app/heads/probe.npz",
                      "--host", "0.0.0.0", "--port", str(PORT)], env=dict(os.environ, PYTHONPATH="/app"))
