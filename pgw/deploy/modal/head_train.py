"""One-off GPU job: train the gateway's target-feature draft head on Qwen2.5-7B-Instruct features (Modal).

Runs ``benchmarks/feature_head_train.py`` in one A10G container (lean image: torch + transformers, no native
build): the target in bf16 answers UltraChat prompts, a teacher-forced pass records (final hidden state, next
token, target's next-next token), the head is trained and exported (``FeatureHeadDrafter.from_file`` format)
and evaluated on held-out prompts. Outputs are committed to a Volume as they appear (``pgw-head-out`` +
``PGW_MODAL_APP_SUFFIX``) and the small ones are also returned. ``modal run`` stops the app when it returns;
the function has a hard timeout.

    PGW_MODAL_APP_SUFFIX=-head modal run deploy/modal/head_train.py --out /tmp/head_out

Cost (Modal list prices 2026-11: A10G $1.10/h + 4 cores x $0.0473 + 48 GiB x $0.008 = ~$1.67/h): the script's
wall budget is 38 min and the function timeout 50 min, so at most ~$1.40.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PGW = HERE.parents[1] if modal.is_local() else Path("/opt/pgw-src")
SUFFIX = os.environ.get("PGW_MODAL_APP_SUFFIX", "").strip()

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.15.0", "transformers==5.23.0", "huggingface_hub==2.4.0", "hf-xet==1.8.0",
                 "pyarrow==26.0.1", "safetensors==0.8.0", "jinja2==3.1.6", "numpy==2.5.3")
    .add_local_file(PGW / "benchmarks/feature_head_train.py", "/opt/pgw-src/benchmarks/feature_head_train.py")
)
app = modal.App(f"pgw-head-train{SUFFIX}", image=image)
vol = modal.Volume.from_name(f"pgw-head-out{SUFFIX}", create_if_missing=True)

DEFAULT_ARGS = ("--target Qwen/Qwen2.5-7B-Instruct --device cuda --n-gen 3000 --n-tf 1500 --n-eval 100 "
                "--max-new 320 --gen-batch 96 --configs 256:8192,384:8192,512:8192,512:16384 --epochs 4 "
                "--budget-min 38 --save-train-features")


@app.function(gpu="A10G", cpu=4.0, memory=49152, timeout=50 * 60, volumes={"/vol": vol},
              region=os.environ.get("PGW_MODAL_REGION", "us-east") or None)
def train(args: str, run: str) -> dict:
    import shlex
    import subprocess

    sys.path.insert(0, "/opt/pgw-src/benchmarks")
    import feature_head_train as fht

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out = Path("/vol") / run
    fht.COMMIT_HOOK[0] = vol.commit
    sys.argv = ["feature_head_train.py", *shlex.split(args), "--out", str(out)]
    t = time.time()
    err = None
    try:
        fht.main()
    except BaseException as exc:  # return what exists
        import traceback

        traceback.print_exc()
        err = repr(exc)
    vol.commit()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],  # noqa: S607
                         capture_output=True, text=True).stdout.strip()
    files = {p.name: p.read_bytes() for p in sorted(out.glob("*"))
             if p.suffix in (".json", ".pt") and p.stat().st_size < 64 << 20}
    return {"wall_s": round(time.time() - t, 1), "gpu": gpu, "error": err, "files": files}


@app.function(gpu="A10G", cpu=4.0, memory=32768, timeout=22 * 60, volumes={"/vol": vol},
              region=os.environ.get("PGW_MODAL_REGION", "us-east") or None)
def train_only(args: str, run: str) -> dict:
    """Train and evaluate again from an earlier run's saved features (``--from-features /vol/<run>``)."""
    return train.local(args, run)


@app.local_entrypoint()
def main(out: str, args: str = DEFAULT_ARGS, run: str = "", from_run: str = "") -> None:
    """``--from-run run-...``: train only, from that run's features on the Volume (no target model)."""
    run = run or time.strftime("run-%Y%m%d-%H%M%S")
    if from_run:
        res = train_only.remote(f"{args} --from-features /vol/{from_run}", run)
    else:
        res = train.remote(args, run)
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    for name, data in res["files"].items():
        (d / name).write_bytes(data)
    print({k: v for k, v in res.items() if k != "files"}, "files:", sorted(res["files"]), "volume run:", run)
