"""Run kaggle/omni/judge_internals.py (probe the judges' internals vs their Yes/No output) on Modal: 2x H100, one
judge per GPU, models from the demo's Hugging Face cache volume. The corpus is built beforehand (it is not in the
repository) and passed in as a local file; the script runs unchanged under the Kaggle directory layout.

    python experiments/omni_corpus.py build --out /tmp/omni/corpus.jsonl
    modal run deploy/modal_judge_internals.py                     # -> ./ji_out/ (judge_internals.json, ...)
"""
import os

import modal

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CORPUS = os.environ.get("JI_CORPUS", "/tmp/omni/corpus.jsonl")
SRC = "/kaggle/working/src"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.56.2", "accelerate==1.10.1", "bitsandbytes==0.45.0",
                 "numpy==2.1.3", "scipy==1.14.1", "safetensors", "sentencepiece")
    .add_local_dir(os.path.join(ROOT, "vt"), f"{SRC}/vt")
    .add_local_file(os.path.join(ROOT, "experiments", "omni_corpus.py"), f"{SRC}/experiments/omni_corpus.py")
    .add_local_file(os.path.join(ROOT, "experiments", "omni_eval.py"), f"{SRC}/experiments/omni_eval.py")
    .add_local_file(os.path.join(ROOT, "kaggle", "omni", "judge_internals.py"), f"{SRC}/kaggle/omni/judge_internals.py")
    .add_local_file(CORPUS, "/tmp/omni/corpus.jsonl")
)
app = modal.App("judge-internals")
hf_cache = modal.Volume.from_name("verichat-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="H100:2", cpu=8.0, memory=65536, timeout=3 * 3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def run(args: str = ""):
    import glob
    import subprocess
    import sys
    work = "/kaggle/working"
    env = dict(os.environ, PYTHONPATH=f"{SRC}:{SRC}/experiments")
    subprocess.run([sys.executable, f"{SRC}/kaggle/omni/judge_internals.py"] + args.split(), cwd=work, env=env)
    hf_cache.commit()
    out = {}
    for p in glob.glob(f"{work}/*"):
        if os.path.isfile(p) and os.path.getsize(p) < 200_000_000:
            out[os.path.basename(p)] = open(p, "rb").read()
    return out


@app.local_entrypoint()
def main(args: str = ""):
    out = run.remote(args)
    os.makedirs("ji_out", exist_ok=True)
    for k, v in out.items():
        open(os.path.join("ji_out", k), "wb").write(v)
    print("outputs:", sorted(out))
    if "judge_internals.json" in out:
        print(out["judge_internals.json"].decode()[:20000])
