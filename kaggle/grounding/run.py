"""Kaggle kernel: the grounding probe on the served model itself (unsloth/Qwen3-32B-bnb-4bit over two T4s).

Trains the probe on constructed examples worded like answers (experiments/grounding_probe.py --phrasing: no
human labels), scores it on RAGTruth's human-labeled hallucinations next to the evidence shift and
disagreement, and exports the probe with each signal's natural-sentence scale for vt/grounding.py:

    /kaggle/working/grounding/grounding_probe_qwen3_32b.npz    the probe the server loads
    /kaggle/working/grounding/results.json                     RAGTruth AUROCs, per-sentence scores

    python run.py                                              # defaults: the 32B, 600 constructions, 200 answers
    python run.py --model Qwen/Qwen2.5-0.5B-Instruct --train 30 --test 6 --out /tmp/g   # smoke test
"""
import argparse
import os
import runpy
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "files"), os.path.join(HERE, "vtpkg"), os.path.join(HERE, "..", "..", "experiments"),
          os.path.join(HERE, "..", "..")):
    sys.path.insert(0, p)


def setup(model):
    """Qwen3 needs transformers >= 4.51; a pre-quantized model needs bitsandbytes."""
    from importlib.metadata import version
    tv = tuple(int(x) for x in version("transformers").split(".")[:2])
    if "qwen3" in model.lower() and tv < (4, 51):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51,<4.58"], check=False)
    if "bnb" in model.lower():
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
    for pkg in ("datasets", "scikit-learn", "scipy"):
        try:
            __import__({"scikit-learn": "sklearn"}.get(pkg, pkg))
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-32B-bnb-4bit")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--out", default="/kaggle/working/grounding")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    setup(a.model)
    name = a.model.split("/")[-1].lower().replace("-bnb-4bit", "")
    script = next(p for p in (os.path.join(HERE, "files", "grounding_probe.py"),
                              os.path.join(HERE, "..", "..", "experiments", "grounding_probe.py")) if os.path.exists(p))
    sys.argv = [script, "--model", a.model, "--train", str(a.train), "--test", str(a.test), "--phrasing",
                "--test-cache", os.path.join(a.out, "ragtruth_test_feats.npz"),
                "--export", os.path.join(a.out, f"grounding_probe_{name}.npz"),
                "--out", os.path.join(a.out, "results.json")]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
