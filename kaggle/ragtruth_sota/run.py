"""Kaggle kernel: the RAGTruth state-of-the-art attempt at scale (experiments/ragtruth_sota.py) on a free T4.

A frozen reader (default Qwen3-4B, float16) reads up to 8,000 RAGTruth training answers and the full 2,700-answer
test set; a logistic regression on its neurons (every 6th layer) and the evidence shift and disagreement is
trained on the training labels (streamed to the GPU) and scored like the published detectors: response-level
precision, recall and F1 overall and by task, with a bootstrap interval.

    /kaggle/working/ragtruth_sota/results.json

    python run.py                                              # defaults
    python run.py --train 40 --test 20 --out /tmp/rs            # smoke test
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "files"), os.path.join(HERE, "vtpkg"), os.path.join(HERE, "..", "..", "experiments"),
          os.path.join(HERE, "..", "..")):
    sys.path.insert(0, p)


def setup(reader):
    from importlib.metadata import version
    tv = tuple(int(x) for x in version("transformers").split(".")[:2])
    if "qwen3" in reader.lower() and tv < (4, 51):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51,<4.58"], check=False)
    for mod, pkg in (("datasets", "datasets"), ("sklearn", "scikit-learn")):
        try:
            __import__(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3-4B")
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=2700)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--out", default="/kaggle/working/ragtruth_sota")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    setup(a.reader)
    shards = "/kaggle/tmp/rt" if os.path.isdir("/kaggle") else os.path.join(a.out, "shards")
    script = next(p for p in (os.path.join(HERE, "files", "ragtruth_sota.py"),
                              os.path.join(HERE, "..", "..", "experiments", "ragtruth_sota.py")) if os.path.exists(p))
    common = ["--reader", a.reader, "--every", str(a.every), "--out", shards]
    for split, n in (("train", a.train), ("test", a.test)):
        subprocess.run([sys.executable, script, "extract", "--split", split, "--n", str(n)] + common, check=True,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    subprocess.run([sys.executable, script, "fit", "--fitter", "torch", "--report", os.path.join(a.out, "results.json")]
                   + common, check=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))


if __name__ == "__main__":
    main()
