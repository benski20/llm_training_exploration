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
LEDGER = os.environ.get("VERICHAT_LEDGER", "0") == "1"            # the LLM-judge ledger (superseded; 1: on)
GROUNDING = os.environ.get("VERICHAT_GROUNDING", "heads/grounding_probe_qwen3_4b.npz")   # "" off
GROUNDING_MODEL = os.environ.get("VERICHAT_GROUNDING_MODEL", "Qwen/Qwen3-4B")          # the probe's reader
# web_server by default: through it a streamed reply arrives as written (deploy/stream_probe.py: a ping every 10 s);
# through asgi_app not even its status line came back within 90 s, so long replies looked dead to clients and were
# sent again (13 of 200 live requests). 1 serves in-process instead.
ASGI = os.environ.get("VERICHAT_ASGI", "0") == "1"
SEARCH = os.environ.get("VERICHAT_SEARCH", "")                    # web search too: "brave" or "tavily" ("" off)
SEARCH_KEY = os.environ.get("VERICHAT_SEARCH_KEY")                # its API key: never in the repo or the image
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
                             "VERICHAT_VERIFY": "1" if VERIFY else "0", "VERICHAT_ASGI": "1" if ASGI else "0",
                             "VERICHAT_SEARCH": SEARCH, "VERICHAT_LEDGER": "1" if LEDGER else "0",
                             "VERICHAT_GROUNDING": GROUNDING, "VERICHAT_GROUNDING_MODEL": GROUNDING_MODEL}.items()
               if v is not None})
    .add_local_dir(os.path.join(ROOT, "vt"), "/app/vt")
    .add_local_file(os.path.join(ROOT, HEAD), "/app/heads/risk_head.pt")
    .add_local_file(os.path.join(ROOT, os.path.splitext(HEAD)[0] + ".serving.json"), "/app/heads/risk_head.serving.json")
    .add_local_file(os.path.join(ROOT, DETECTOR), "/app/heads/detector.pt")
    # the claim-verification evaluation: the 32B judges evidence saved by a CPU run (no lookups)
    .add_local_file(os.path.join(ROOT, "experiments", "evidence_verify.py"), "/app/experiments/evidence_verify.py")
    .add_local_file(os.path.join(ROOT, "results", "evidence_verify_v3_judge1.5b.json"), "/app/eval/rows.json")
    .add_local_file(os.path.join(ROOT, "results", "kaggle", "chateval-32h", "chateval", "traces.json"),
                    "/app/eval/traces.json")
    # the judge-vs-probe study: the served model answers, its probe scores, it judges looked-up evidence
    .add_local_file(os.path.join(ROOT, "experiments", "study_32b.py"), "/app/experiments/study_32b.py")
    .add_local_file(os.path.join(ROOT, "results", "study_questions.json"), "/app/eval/study_questions.json")
    .add_local_file(os.path.join(ROOT, "results", "study_questions_2.json"), "/app/eval/study_questions_2.json")
)
if GROUNDING:                                                       # the grounding probe the reader serves
    image = image.add_local_file(os.path.join(ROOT, GROUNDING), "/app/heads/grounding_probe.npz")
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


@app.function(gpu=GPU, cpu=4.0, volumes={"/root/.cache/huggingface": hf_cache}, timeout=75 * 60)
def study(per_source: int = 300):
    """Judge vs probe on fresh TriviaQA and NQ answers of the served model (experiments/study_32b.py);
    returns the result JSON. About 25-30 GPU minutes for 300 + 300."""
    out = subprocess.run(["python", "/app/experiments/study_32b.py", "--model", MODEL, "--head", "/app/heads/risk_head.pt",
                          "--questions", "/app/eval/study_questions.json", "--limit", str(per_source),
                          "--out", "/tmp/study.json"],
                         env=dict(os.environ, PYTHONPATH="/app"), capture_output=True, text=True)
    print("\n".join(line for line in out.stdout.splitlines() if not line.startswith("ROWS"))[-8000:],
          out.stderr[-3000:] if out.returncode else "", flush=True)
    if out.returncode:
        raise SystemExit(out.returncode)
    return open("/tmp/study.json").read()


def _run_study(args, stdin_json):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        with open(src, "w") as f:
            f.write(stdin_json)
        r = subprocess.run(["python", "/app/experiments/study_32b.py", "--model", MODEL, "--head", "/app/heads/risk_head.pt",
                            "--questions", "/app/eval/study_questions.json", "--rows", src, "--out", out] + args,
                           env=dict(os.environ, PYTHONPATH="/app"), capture_output=True, text=True)
        print(r.stdout[-8000:], r.stderr[-3000:] if r.returncode else "", flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)
        return open(out).read()


@app.function(cpu=4.0, memory=8192, volumes={"/root/.cache/huggingface": hf_cache}, timeout=4 * 3600)
def study_gather(rows_json: str):
    """The lookups for saved answers, on CPU at a gentle bulk pace (Wikipedia throttles bursts): hours of CPU,
    cents; no GPU billed."""
    return _run_study(["--gather-only"], rows_json)


@app.function(gpu=GPU, cpu=2.0, volumes={"/root/.cache/huggingface": hf_cache}, timeout=60 * 60)
def study_answer(questions: str = "/app/eval/study_questions_2.json", per_source: int = 500):
    """The served model answers the questions closed-book (no lookups): about 1.4 s of GPU per answer."""
    r = subprocess.run(["python", "/app/experiments/study_32b.py", "--model", MODEL, "--head", "/app/heads/risk_head.pt",
                        "--questions", questions, "--limit", str(per_source), "--answers-only", "--out", "/tmp/answers.json"],
                       env=dict(os.environ, PYTHONPATH="/app"), capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-3000:] if r.returncode else "", flush=True)
    if r.returncode:
        raise SystemExit(r.returncode)
    return open("/tmp/answers.json").read()


@app.local_entrypoint()
def study_full(per_source: int = 500):
    """Fresh answers (GPU), their lookups (CPU, slow and cheap), the judging and grading (GPU): the GPU never
    waits on the network."""
    answers = study_answer.remote(per_source=per_source)
    with open("study_answers.json", "w") as f:
        f.write(answers)
    gathered = study_gather.remote(answers)
    with open("study.json", "w") as f:
        f.write(study_judge.remote(gathered))
    print("wrote study.json")


@app.function(gpu=GPU, cpu=2.0, volumes={"/root/.cache/huggingface": hf_cache}, timeout=30 * 60)
def study_judge(gathered_json: str):
    """The served model judges the gathered evidence (yes/no and three-way) and grades the answers against
    their reference answers: about 10 GPU minutes for 600."""
    return _run_study(["--judge-gathered"], gathered_json)


@app.local_entrypoint()
def study_rejudge(rows: str = "results/study_32b_v1.json"):
    gathered = study_gather.remote(open(rows).read())
    with open("study.json", "w") as f:
        f.write(study_judge.remote(gathered))
    print("wrote study.json")


@app.local_entrypoint()
def study_main(per_source: int = 300):
    with open("study.json", "w") as f:
        f.write(study.remote(per_source))
    print("wrote study.json")


def server_args():
    return (["--model", MODEL, "--head", "/app/heads/risk_head.pt", "--detector", "/app/heads/detector.pt",
             "--moderation", "--rate-per-minute", RATE, "--trust-proxy", "--deadline", "300", "--log-requests"]
            + (["--risk-threshold", RISK_THRESHOLD] if RISK_THRESHOLD else [])
            + (["--wikipedia"] if WIKIPEDIA else [])
            + (["--rerank", RERANK] if WIKIPEDIA and RERANK else [])
            + (["--web-search", SEARCH] if SEARCH else [])
            + (["--grounding", "/app/heads/grounding_probe.npz"]
               + (["--grounding-model", GROUNDING_MODEL] if GROUNDING_MODEL else []) if GROUNDING else [])
            + (["--ledger"] if LEDGER and not GROUNDING else
               ["--verify-claims"] if (WIKIPEDIA or SEARCH) and VERIFY else [])
            + (["--revision", REVISION] if REVISION else []))


# the search key reaches the container as a Modal secret: from this deploy's environment (a one-off workflow
# input), else the named secret "verichat-search" made in the Modal dashboard
SEARCH_SECRETS = ([modal.Secret.from_dict({"VERICHAT_SEARCH_KEY": SEARCH_KEY})] if SEARCH and SEARCH_KEY else
                  [modal.Secret.from_name("verichat-search")] if SEARCH else [])
SERVE = dict(gpu=GPU, cpu=2.0, volumes={"/root/.cache/huggingface": hf_cache}, timeout=24 * 3600,
             scaledown_window=SCALEDOWN, max_containers=1, secrets=SEARCH_SECRETS)
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
