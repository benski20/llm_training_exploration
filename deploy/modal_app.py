"""VeriChat on Modal: the public demo (chat with every sentence and claim scored, /judge, /metrics).

    pip install modal && modal token new                  # once
    modal deploy deploy/modal_app.py                      # prints the public URL
    modal app stop verichat                               # stop serving

One GPU container serves everything: the chat model with its risk head (claim probe + learned
token/sentence risk), the 3B detector for /judge, and the toxicity filter. It scales to zero when idle
(SCALEDOWN seconds), so you pay only while it is up. A cold start downloads nothing after the first
deploy (the Hugging Face cache lives on a Modal volume), but loading the 32B model still takes about a
minute.

Choose the model with environment variables at deploy time:
    VERICHAT_MODEL  unsloth/Qwen3-32B-bnb-4bit (default; the head was trained on this checkpoint)
    VERICHAT_HEAD   results/kaggle/risk-32b2/risk/Qwen3-32B-bnb-4bit_risk_head.pt
    VERICHAT_GPU    L40S (default, 48 GB: the 4-bit 32B model, the 3B detector and the KV cache fit)
    VERICHAT_RISK_THRESHOLD  sentence risk that counts as doubtful (default: the head's, chosen for F1;
                    set it from kaggle/chateval's sweep to trade false alarms on everyday chat for recall)
"""
import os
import subprocess

import modal

MODEL = os.environ.get("VERICHAT_MODEL", "unsloth/Qwen3-32B-bnb-4bit")
HEAD = os.environ.get("VERICHAT_HEAD", "results/kaggle/risk-32b2/risk/Qwen3-32B-bnb-4bit_risk_head.pt")
DETECTOR = "results/kaggle/detector-3b/detector/Qwen2.5-3B-Instruct_detector.pt"
GPU = os.environ.get("VERICHAT_GPU", "L40S")
SCALEDOWN = int(os.environ.get("VERICHAT_SCALEDOWN", "300"))
RISK_THRESHOLD = os.environ.get("VERICHAT_RISK_THRESHOLD")         # default: the head's own
PORT = 8000
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate==1.1.1", "bitsandbytes==0.45.0",
                 "numpy==2.1.3", "scipy==1.14.1", "safetensors", "sentencepiece")
    .add_local_dir(os.path.join(ROOT, "vt"), "/app/vt")
    .add_local_file(os.path.join(ROOT, HEAD), "/app/heads/risk_head.pt")
    .add_local_file(os.path.join(ROOT, DETECTOR), "/app/heads/detector.pt")
)
app = modal.App("verichat", image=image)
hf_cache = modal.Volume.from_name("verichat-hf-cache", create_if_missing=True)


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache}, timeout=24 * 3600,
              scaledown_window=SCALEDOWN, max_containers=1)
@modal.concurrent(max_inputs=32)                          # the server batches concurrent requests itself
@modal.web_server(PORT, startup_timeout=1200)
def serve():
    env = dict(os.environ, PYTHONPATH="/app")
    subprocess.Popen(["python", "-m", "vt.verichat_server", "--model", MODEL, "--head", "/app/heads/risk_head.pt",
                      "--detector", "/app/heads/detector.pt", "--moderation", "--rate-per-minute", "10",
                      "--trust-proxy", "--deadline", "300", "--host", "0.0.0.0", "--port", str(PORT)]
                     + (["--risk-threshold", RISK_THRESHOLD] if RISK_THRESHOLD else []), env=env)
