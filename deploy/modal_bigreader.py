"""A bigger reader for the internals probe, on one Modal GPU, under a hard time cap (experiments/bigreader_halubench.py).

Reads the 300 HaluBench answers first, then as many RAGTruth training answers (up to --train) as the time left
allows, trains the probe on those and scores HaluBench. Shards go to a Modal volume, so a rerun resumes instead of
paying again. The model is downloaded on a CPU container first (no GPU billed while ~16 GB download).

    modal run deploy/modal_bigreader.py --train 1600        # BIGREADER=Qwen/Qwen3-8B (default), BIGREADER_GPU=L4

Cost bound: GPU_SECONDS of an L4 (about $0.80 an hour on Modal) plus 4 CPU cores and 24 GB memory for that long:
at the default 4,800 s, under $1.60 in all.
"""
import os
import subprocess
import sys
import time

import modal

READER = os.environ.get("BIGREADER", "Qwen/Qwen3-8B")
GPU = os.environ.get("BIGREADER_GPU", "L4")
GPU_SECONDS = int(os.environ.get("BIGREADER_SECONDS", "4800"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SLUG = READER.split("/")[-1].lower().replace(".", "_").replace("-", "_")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate==1.1.1", "numpy==2.1.3", "scipy==1.14.1",
                 "scikit-learn==1.5.2", "datasets==3.2.0", "safetensors", "huggingface_hub")
    .env({"BIGREADER": READER, "BIGREADER_GPU": GPU, "BIGREADER_SECONDS": str(GPU_SECONDS)})
    .add_local_dir(os.path.join(ROOT, "vt"), "/app/vt")
    .add_local_file(os.path.join(ROOT, "experiments", "ragtruth_sota.py"), "/app/experiments/ragtruth_sota.py")
    .add_local_file(os.path.join(ROOT, "experiments", "evidence_shift.py"), "/app/experiments/evidence_shift.py")
    .add_local_file(os.path.join(ROOT, "experiments", "halubench_compare.py"), "/app/experiments/halubench_compare.py")
    .add_local_file(os.path.join(ROOT, "experiments", "bigreader_halubench.py"),
                    "/app/experiments/bigreader_halubench.py")
    .add_local_file(os.path.join(ROOT, "results", "halubench_compare_qwen3_4b_seed1.json"), "/app/others.json")
)
app = modal.App("bigreader", image=image)
hf_cache = modal.Volume.from_name("verichat-hf-cache", create_if_missing=True)
shards = modal.Volume.from_name("bigreader-shards", create_if_missing=True)


@app.function(volumes={"/root/.cache/huggingface": hf_cache}, timeout=3600, cpu=2.0, memory=4096)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(READER, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    hf_cache.commit()


@app.function(gpu=GPU, cpu=4.0, memory=24576, timeout=GPU_SECONDS,
              volumes={"/root/.cache/huggingface": hf_cache, "/shards": shards})
def run(train: int = 1600, halu: int = 300) -> str:
    t0 = time.time()
    out = f"/shards/{SLUG}"
    env = dict(os.environ, PYTHONPATH="/app:/app/experiments")
    common = ["--reader", READER, "--project", "4096", "--every", "4", "--shard", "50", "--out", out]
    script = "/app/experiments/ragtruth_sota.py"

    def extract(split, n, budget):
        try:
            subprocess.run([sys.executable, script, "extract", "--split", split, "--n", str(n)] + common,
                           env=env, check=True, timeout=max(60, budget))
        except subprocess.TimeoutExpired:
            print(f"{split}: time cap reached; using the shards finished so far", flush=True)
        shards.commit()

    extract("halu", halu, GPU_SECONDS - 600)
    extract("train", train, GPU_SECONDS - (time.time() - t0) - 240)    # keep 4 minutes to fit and report
    report = f"/shards/{SLUG}_report.json"
    subprocess.run([sys.executable, "/app/experiments/bigreader_halubench.py", "--shards", out, "--train",
                    str(train), "--reader", READER, "--others", "/app/others.json", "--report", report],
                   env=env, check=True)
    shards.commit()
    with open(report) as f:
        return f.read()


@app.local_entrypoint()
def main(train: int = 1600, halu: int = 300):
    download.remote()
    text = run.remote(train, halu)
    path = os.path.join(ROOT, "results", f"bigreader_halubench_{SLUG}.json")
    with open(path, "w") as f:
        f.write(text)
    print(text)
