"""VeriChat on Modal: the public demo (chat with every sentence and claim scored, /judge, /metrics).

    pip install modal && modal token new                  # once
    modal deploy deploy/modal_app.py                      # prints the public URL
    modal app stop verichat                               # stop serving

The head's measured serving settings (<head>.serving.json: claim calibration, doubt threshold) ship with it.
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
SCALEDOWN = int(os.environ.get("VERICHAT_SCALEDOWN", "120"))       # idle seconds before the GPU stops (billing)
RISK_THRESHOLD = os.environ.get("VERICHAT_RISK_THRESHOLD")         # default: the head's own
RATE = os.environ.get("VERICHAT_RATE", "10")                        # POSTs per client per minute; 0 for load tests
WIKIPEDIA = os.environ.get("VERICHAT_WIKIPEDIA", "1") == "1"        # look doubted answers up on Wikipedia
REVISION = os.environ.get("VERICHAT_REVISION")                      # pin the model to a Hugging Face commit
VERIFY = os.environ.get("VERICHAT_VERIFY", "1") == "1"            # check each reply's claims against sources
ASGI = os.environ.get("VERICHAT_ASGI", "1") == "1"                # serve in-process (0: web_server + local proxy)
RERANK = os.environ.get("VERICHAT_RERANK", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # reorders lookup passages; "" off
PORT = 8000
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate==1.1.1", "bitsandbytes==0.45.0",
                 "numpy==2.1.3", "scipy==1.14.1", "safetensors", "sentencepiece", "sentence-transformers==3.3.1",
                 "fastapi==0.115.6")                      # for Modal's web endpoints (asgi_app)
    # Modal re-imports this file inside the container: bake the deploy-time settings into its environment
    # so the server there sees the same values (otherwise every VERICHAT_* silently reverts to its default)
    .env({k: v for k, v in {"VERICHAT_MODEL": MODEL, "VERICHAT_HEAD": HEAD, "VERICHAT_GPU": GPU,
                             "VERICHAT_SCALEDOWN": str(SCALEDOWN), "VERICHAT_RISK_THRESHOLD": RISK_THRESHOLD,
                             "VERICHAT_RATE": RATE, "VERICHAT_WIKIPEDIA": "1" if WIKIPEDIA else "0",
                             "VERICHAT_REVISION": REVISION, "VERICHAT_RERANK": RERANK,
                             "VERICHAT_VERIFY": "1" if VERIFY else "0", "VERICHAT_ASGI": "1" if ASGI else "0"}.items() if v is not None})
    .add_local_dir(os.path.join(ROOT, "vt"), "/app/vt")
    .add_local_file(os.path.join(ROOT, HEAD), "/app/heads/risk_head.pt")
    .add_local_file(os.path.join(ROOT, os.path.splitext(HEAD)[0] + ".serving.json"), "/app/heads/risk_head.serving.json")
    .add_local_file(os.path.join(ROOT, DETECTOR), "/app/heads/detector.pt")
    # the claim-verification evaluation: the 32B judges evidence saved by a CPU run (no lookups)
    .add_local_file(os.path.join(ROOT, "experiments", "evidence_verify.py"), "/app/experiments/evidence_verify.py")
    .add_local_file(os.path.join(ROOT, "results", "evidence_verify_v3_judge1.5b.json"), "/app/eval/rows.json")
    .add_local_file(os.path.join(ROOT, "results", "kaggle", "chateval-32h", "chateval", "traces.json"),
                    "/app/eval/traces.json")
)
app = modal.App("verichat", image=image)
hf_cache = modal.Volume.from_name("verichat-hf-cache", create_if_missing=True)


@app.function(volumes={"/root/.cache/huggingface": hf_cache}, timeout=2 * 3600, cpu=2.0, memory=4096)
def download():
    """Fill the model cache on a cheap CPU container, so no GPU is billed while ~25 GB download:
    modal run deploy/modal_app.py::download (once, before the first deploy)."""
    from huggingface_hub import snapshot_download
    for repo, rev in ((MODEL, REVISION), ("Qwen/Qwen2.5-3B-Instruct", None), ("unitary/toxic-bert", None)) + \
            (((RERANK, None),) if RERANK else ()):
        print("downloading", repo, flush=True)
        snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"])
    hf_cache.commit()
    print("cache ready", flush=True)


@app.function(cpu=0.25, timeout=120)
def wiki_check():
    """Live Wikipedia lookups from Modal's network, on a tiny CPU container (no GPU):
    modal run deploy/modal_app.py::wiki_check"""
    import sys
    sys.path.insert(0, "/app")
    from vt.retrieval import Wikipedia
    w = Wikipedia(k=2, full_text=True)
    for q in ("Who designed the Eiffel Tower?", "What is the capital of Burkina Faso?"):
        got = w(q)
        print(q, "->", len(got), "passages:", [p[:90] for p in got], flush=True)
        if not got:
            raise SystemExit(f"no passages: Wikipedia unreachable or rate-limited from Modal ({dict(w.errors)})")


@app.function(gpu=GPU, cpu=2.0, volumes={"/root/.cache/huggingface": hf_cache}, timeout=30 * 60)
def judge_eval():
    """The 32B as the claim verifier on the 32B's labeled replies, from saved evidence (a few GPU minutes):
    modal run deploy/modal_app.py::judge_eval"""
    out = subprocess.run(["python", "/app/experiments/evidence_verify.py", "--rejudge", "/app/eval/rows.json",
                          "--traces", "/app/eval/traces.json", "--judge", MODEL, "--out", "/tmp/judge32.json"],
                         env=dict(os.environ, PYTHONPATH="/app"), capture_output=True, text=True)
    print(out.stdout[-60000:], out.stderr[-3000:] if out.returncode else "", flush=True)
    if out.returncode:
        raise SystemExit(out.returncode)


def server_args():
    return (["--model", MODEL, "--head", "/app/heads/risk_head.pt", "--detector", "/app/heads/detector.pt",
             "--moderation", "--rate-per-minute", RATE, "--trust-proxy", "--deadline", "300", "--log-requests"]
            + (["--risk-threshold", RISK_THRESHOLD] if RISK_THRESHOLD else [])
            + (["--wikipedia"] if WIKIPEDIA else [])
            + (["--rerank", RERANK] if WIKIPEDIA and RERANK else [])
            + (["--verify-claims"] if WIKIPEDIA and VERIFY else [])
            + (["--revision", REVISION] if REVISION else []))


SERVE = dict(gpu=GPU, cpu=2.0, volumes={"/root/.cache/huggingface": hf_cache}, timeout=24 * 3600,
             scaledown_window=SCALEDOWN, max_containers=1)
if ASGI:
    # In-process: Modal calls the handler directly. Through a web_server, Modal's proxy reads each reply
    # back over a local socket, and live it sometimes failed to ("Response payload is not completed"),
    # leaving the client to wait about 270 s for nothing (deploy/RUNBOOK.md, lost requests).
    @app.function(**SERVE)
    @modal.concurrent(max_inputs=32)                      # the server batches concurrent requests itself
    @modal.asgi_app()
    def serve():
        import sys
        sys.path.insert(0, "/app")
        from vt.asgi import asgi_from
        from vt.verichat_server import build
        return asgi_from(build(server_args(), port=0))
else:
    @app.function(**SERVE)
    @modal.concurrent(max_inputs=32)
    @modal.web_server(PORT, startup_timeout=1200)
    def serve():
        subprocess.Popen(["python", "-m", "vt.verichat_server"] + server_args()
                         + ["--host", "0.0.0.0", "--port", str(PORT)], env=dict(os.environ, PYTHONPATH="/app"))
